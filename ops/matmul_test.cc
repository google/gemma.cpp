// Copyright 2023 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// End to end test of MatMul, comparing against a reference implementation.

#include "compression/types.h"
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

// matmul_static is not built as a test, hence does not define MatMulStatic for
// worse-than-baseline targets (to speed up builds), so we skip them here, too.
#ifndef HWY_SKIP_NON_BEST_BASELINE
#define HWY_SKIP_NON_BEST_BASELINE
#endif  // HWY_SKIP_NON_BEST_BASELINE

#include <stddef.h>
#include <stdio.h>

#include <atomic>

#include "ops/matmul.h"
#include "util/basics.h"
#include "util/mat.h"
#include "util/threading_context.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/nanobenchmark.h"  // Unpredictable1

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ops/matmul_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"
#include "compression/test_util-inl.h"
#include "ops/dot-inl.h"
#include "ops/matmul_static.h"  // also textual

HWY_BEFORE_NAMESPACE();
namespace gcpp {
// For running TestTiny only once. Defined within HWY_ONCE.
extern int64_t first_target;

namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// B is already transposed.
template <typename TA, typename TB, typename TC>
HWY_INLINE void MatMulSlow(const MatPtrT<TA> A, const MatPtrT<TB> B,
                           const float* HWY_RESTRICT add_row, MatMulEnv& env,
                           MatPtrT<TC>& C) {
  // TA can be any Packed except NuqStream because it uses pointer
  // arithmetic, because it is the second argument to Dot, which does not
  // support a v_ofs.
  static_assert(sizeof(TA) >= sizeof(BF16), "A matrix must be BF16/f32");
  const float scale = A.Scale() * B.Scale();

  const hn::ScalableTag<float> df;  // lane type is ignored
  const PackedSpan<const TB> b_span = B.Span();
  const IndexRange all_rows_c(0, A.Extents().rows);
  const IndexRange all_cols_c(0, C.Cols());

  NestedPools& pools = env.ctx.pools;
  hwy::ThreadPool& all_clusters = pools.AllClusters();
  const size_t multiple = env.ctx.allocator.QuantumBytes() / sizeof(TB);
  const IndexRangePartition get_col_c =
      StaticPartition(all_cols_c, all_clusters.NumWorkers(), multiple);
  ParallelForAcrossClusters(
      get_col_c.NumTasks(), env.ctx, env.ctx.pool_callers.Get(Callers::kTest),
      [&](size_t range_idx, size_t cluster_idx) HWY_ATTR {
        const IndexRange cols_c = get_col_c.Range(range_idx);
        for (size_t r : all_rows_c) {
          TC* HWY_RESTRICT C_row = C.Row(r);
          for (size_t c : cols_c) {
            const float add = add_row ? add_row[c] : 0.0f;
            const float dot =
                Dot(df, b_span, c * B.Stride(), A.Row(r), A.Cols());
            C_row[c] = hwy::ConvertScalarTo<TC>(add + scale * dot);
          }
        }
      });
}

template <typename TA, typename TB = TA, typename TC = float>
void TestMatMul(size_t rows_ac, size_t cols_a_rows_b, size_t cols_bc, bool add,
                MatMulEnv& env, int line) {
  fprintf(stderr, "TestMatMul %zu, K=%zu, %zu, add=%d, TA=%s, TB=%s, TC=%s\n",
          rows_ac, cols_a_rows_b, cols_bc, add, TypeName<TA>(), TypeName<TB>(),
          TypeName<TC>());

  env.print_config = false;  // Too verbose.
  env.print_best = true;

  const Extents2D A_extents(rows_ac, cols_a_rows_b);
  const Extents2D B_extents(cols_bc, cols_a_rows_b);  // already transposed
  const Extents2D C_extents(rows_ac, cols_bc);

  MatStorageT<TA> A(GenerateMat<TA>(A_extents, MatPadding::kOdd, env.ctx));
  // Must be packed because we call Span() on it.
  MatStorageT<TB> BT(
      GenerateTransposedMat<TB>(B_extents, MatPadding::kPacked, env.ctx));
  MatStorageT<TC> C_slow("C_slow", C_extents, env.ctx.allocator,
                         MatPadding::kOdd);
  MatStorageT<TC> C("C", C_extents, env.ctx.allocator, MatPadding::kOdd);
  MatStorageT<TC> C2("C", C_extents, env.ctx.allocator, MatPadding::kOdd);
  C.AllocateAndAttachRowPtrs(env.row_ptrs);
  C2.AllocateAndAttachRowPtrs(env.row_ptrs);

  MatStorageT<float> add_storage =
      add ? GenerateMat<float>(Extents2D(1, cols_bc), MatPadding::kPacked,
                               env.ctx)
          : MatStorageT<float>("add", Extents2D(), env.ctx.allocator,
                               MatPadding::kPacked);
  add_storage.SetScale(1.0f);
  const float* add_row = add ? add_storage.PackedScale1() : nullptr;

  MatMulSlow(A, BT, add_row, env, C_slow);
  // A few reps to get coverage of the various autotuned code paths.
  MMOptions options;
  for (size_t rep = 0; rep < 16; ++rep) {
    MMPerKey* per_key = MatMulStatic(A, BT, add_row, env, C, options);
    AssertClose(A, BT, C_slow, C, env.ctx.allocator, env.row_ptrs, line);
    // Check before TwoMatMulStatic(), which can invalidate per_key.
    const bool autotune_done = !!per_key->autotune.Best();

    // Ensure the tiled view returns the same result as C.
    if constexpr (IsBF16<TA>() && IsBF16<TC>()) {
      // The total view area should match the entire C matrix.
      std::atomic<size_t> total_view_area = 0;

      const auto fused = [&](RowPtrsBF C2_rows, IndexRange range_r,
                             IndexRange range_c, StridedViewBF C2_view,
                             size_t worker) {
        total_view_area.fetch_add(range_r.Num() * range_c.Num());
        HWY_ASSERT(range_c.Num() <= C2_view.Cols());
        HWY_ASSERT(worker < env.ctx.pools.MaxWorkers());
        for (size_t ir = 0; ir < range_r.Num(); ++ir) {
          const size_t r = range_r.begin() + ir;
          for (size_t ic = 0; ic < range_c.Num(); ++ic) {
            const size_t c = range_c.begin() + ic;
            const float expected =
                hwy::ConvertScalarTo<float>(C2_rows.Row(r)[c]);
            const float actual =
                hwy::ConvertScalarTo<float>(C2_view.Row(ir)[ic]);
            const float L1 = hwy::ScalarAbs(actual - expected);
            if (L1 > 1E-6f) {
              HWY_ABORT("%zu: ir %zu ic %zu L1 %f expected %f actual %f.",
                        worker, ir, ic, L1, expected, actual);
            }
          }
        }
      };
      options.SetFunc(fused);
      TwoMatMulStatic(A, BT, BT, env, C2, options);
      HWY_ASSERT_EQ(C.Extents().Area(), total_view_area.load());
      options.func = nullptr;  // reset for next call

      // TwoMatMulStatic() does not support adding a bias vector.
      if (!add) {
        AssertClose(A, BT, C, C2, env.ctx.allocator, env.row_ptrs, line);
      }
    }

    if (autotune_done) break;
  }
}

using F32 = float;
using SFP = SfpStream;

// Sweep all dimensions for a single input type and Highway target, to verify
// the remainder handling.
void TestTiny() {
  if (first_target == 0) first_target = HWY_TARGET;
  if (HWY_TARGET != first_target) return;

  ThreadingArgs threading_args;
  threading_args.bind = Tristate::kTrue;
  ThreadingContext ctx(threading_args);
  MatMulEnv env(ctx);
  NestedPools& pools = env.ctx.pools;

  fprintf(stderr, "TestTiny: %s %s\n", env.ctx.topology.TopologyString(),
          pools.PinString());

  pools.MaybeStartSpinning(threading_args.spin);

  for (size_t M = 1; M <= 12; ++M) {
    for (size_t K = 1; K <= 64; K *= 2) {
      for (size_t N = 4; N <= 64; N += 4) {
        TestMatMul<F32, F32, F32>(M, K, N, /*add=*/false, env, __LINE__);
        TestMatMul<BF16, F32, F32>(M, K, N, /*add=*/false, env, __LINE__);
        TestMatMul<F32, BF16, F32>(M, K, N, /*add=*/false, env, __LINE__);
        TestMatMul<BF16, BF16, F32>(M, K, N, /*add=*/false, env, __LINE__);
      }
    }
  }
  pools.MaybeStopSpinning(threading_args.spin);
}

void TestAllMatMul() {
  // Skip EMU128 (10x slower than SSE4 for SFP) and older x86.
  // Add Unpredictable1 to prevent erroneous "unreachable code" warning.
  if (hwy::Unpredictable1() == 1 &&
      (HWY_TARGET == HWY_EMU128 || HWY_TARGET == HWY_SSE4 ||
       HWY_TARGET == HWY_SSSE3 || HWY_TARGET == HWY_SSE2)) {
    return;
  }

  ThreadingArgs threading_args;
  threading_args.bind = Tristate::kTrue;

  ThreadingContext ctx(threading_args);
  MatMulEnv env(ctx);
  NestedPools& pools = env.ctx.pools;
  pools.MaybeStartSpinning(threading_args.spin);

  // Sizes seen in gemma_test 2B. Too slow for CI, enable on-demand.
  TestMatMul<F32>(1, 2048, 512, /*add=*/false, env, __LINE__);
  //  TestMatMul<F32>(1, 2048, 2048, /*add=*/false, env, __LINE__);
  //  TestMatMul<F32>(1, 2048, 16384, /*add=*/false, env, __LINE__);
  //  TestMatMul<F32>(1, 16384, 2048, /*add=*/false, env, __LINE__);
  //  TestMatMul<F32>(1, 2048, 256000, /*add=*/false, env, __LINE__);
  //  TestMatMul<F32>(5, 2048, 512, /*add=*/false, env, __LINE__);
  //  TestMatMul<F32>(5, 2048, 2048, /*add=*/false, env, __LINE__);
  //  TestMatMul<F32>(5, 2048, 16384, /*add=*/false, env, __LINE__);
  //  TestMatMul<F32>(5, 16384, 2048, /*add=*/false, env, __LINE__);

  // medium-sized square, f32 vs bf16 for A, B, C; plus add.
  TestMatMul<F32, F32, F32>(256, 256, 256, /*add=*/false, env, __LINE__);
  TestMatMul<F32, F32, BF16>(256, 256, 256, /*add=*/false, env, __LINE__);
  TestMatMul<F32, BF16, F32>(256, 256, 256, /*add=*/false, env, __LINE__);
  TestMatMul<F32, BF16, BF16>(256, 256, 256, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, F32, F32>(256, 256, 256, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, F32, BF16>(256, 256, 256, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, BF16, F32>(256, 256, 256, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, BF16, BF16>(256, 256, 256, /*add=*/false, env, __LINE__);
  TestMatMul<F32, F32, F32>(256, 256, 256, /*add=*/true, env, __LINE__);
  TestMatMul<F32, F32, BF16>(256, 256, 256, /*add=*/true, env, __LINE__);
  TestMatMul<F32, BF16, F32>(256, 256, 256, /*add=*/true, env, __LINE__);
  TestMatMul<F32, BF16, BF16>(256, 256, 256, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, F32, F32>(256, 256, 256, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, F32, BF16>(256, 256, 256, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, BF16, F32>(256, 256, 256, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, BF16, BF16>(256, 256, 256, /*add=*/true, env, __LINE__);

  TestMatMul<F32, SFP>(256, 256, 256, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, SFP>(256, 256, 256, /*add=*/true, env, __LINE__);
  TestMatMul<F32, Q4_0Stream>(256, 256, 256, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, Q4_0Stream>(256, 256, 256, /*add=*/true, env, __LINE__);
  TestMatMul<F32, MxFp4Stream>(256, 256, 256, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, MxFp4Stream>(256, 256, 256, /*add=*/true, env, __LINE__);

#if GEMMA_ENABLE_NUQ
  using NUQ = NuqStream;
  TestMatMul<F32, NUQ>(256, 256, 256, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, NUQ>(256, 256, 256, /*add=*/true, env, __LINE__);
  TestMatMul<F32, NUQ>(31, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, NUQ>(29, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, NUQ>(4, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, NUQ>(4, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<F32, NUQ>(3, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, NUQ>(3, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, NUQ>(2, 128, 64, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, NUQ>(2, 128, 64, /*add=*/false, env, __LINE__);
  TestMatMul<F32, NUQ>(1, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, NUQ>(1, 128, 32, /*add=*/true, env, __LINE__);
#endif

  // Non-vector-multiple K.
  TestMatMul<F32, BF16>(128, 258, 128, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, BF16>(128, 258, 128, /*add=*/true, env, __LINE__);

  // Multi-kc accumulation (K > kMaxKC) exercising MMSetC + MMAddC.
  TestMatMul<BF16, MxFp4Stream, BF16>(2, kMaxKC + 256, 32, /*add=*/false, env,
                                      __LINE__);
  TestMatMul<F32, MxFp4Stream, F32>(3, kMaxKC + 96, 32, /*add=*/true, env,
                                    __LINE__);

  // minimal non-square test. kColsARowsB must be at least 2 vectors.
  TestMatMul<F32>(35, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16>(34, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, BF16>(33, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, F32>(33, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, SFP>(31, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, SFP>(29, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, Q4_0Stream>(31, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, Q4_0Stream>(29, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, MxFp4Stream>(31, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, MxFp4Stream>(29, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, MxFp4Stream, BF16>(32, 128, 32, /*add=*/false, env,
                                      __LINE__);
  TestMatMul<F32>(4, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<BF16>(4, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<F32, BF16>(4, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, F32>(4, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<F32, SFP>(4, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, SFP>(4, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<F32, Q4_0Stream>(4, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, Q4_0Stream>(4, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<F32, MxFp4Stream>(4, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, MxFp4Stream>(4, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, MxFp4Stream, BF16>(4, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<F32>(3, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16>(3, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, BF16>(3, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, F32>(3, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, SFP>(3, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, SFP>(3, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, Q4_0Stream>(3, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, Q4_0Stream>(3, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, MxFp4Stream>(3, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, MxFp4Stream>(3, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, MxFp4Stream, BF16>(3, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32>(2, 128, 64, /*add=*/true, env, __LINE__);
  TestMatMul<BF16>(2, 128, 64, /*add=*/false, env, __LINE__);
  TestMatMul<F32, BF16>(2, 128, 64, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, F32>(2, 128, 64, /*add=*/false, env, __LINE__);
  TestMatMul<F32, SFP>(2, 128, 64, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, SFP>(2, 128, 64, /*add=*/false, env, __LINE__);
  TestMatMul<F32, Q4_0Stream>(2, 128, 64, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, Q4_0Stream>(2, 128, 64, /*add=*/false, env, __LINE__);
  TestMatMul<F32, MxFp4Stream>(2, 128, 64, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, MxFp4Stream>(2, 128, 64, /*add=*/false, env, __LINE__);
  TestMatMul<F32>(1, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16>(1, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, BF16>(1, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, F32>(1, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, SFP>(1, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, SFP>(1, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, Q4_0Stream>(1, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, Q4_0Stream>(1, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, MxFp4Stream>(1, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, MxFp4Stream>(1, 128, 32, /*add=*/true, env, __LINE__);

  pools.MaybeStopSpinning(threading_args.spin);
}

// Verifies the direct M <= 4 MxFp4Stream kernel across E8M0 edge scales
// {0, 1, 127, 253, 254, 255}, all 16 E2M1 nibble values, subnormal and huge
// BF16 activations, and partial tail blocks (kc % 32 != 0 with aligned stride).
void TestMxFp4EdgeCases() {
  ThreadingArgs threading_args;
  ThreadingContext ctx(threading_args);
  MatMulEnv env(ctx);
  env.print_config = false;
  env.print_best = false;

  const hn::ScalableTag<float> df;
  constexpr size_t kBlocks = 6;
  constexpr size_t K = kBlocks * MxFp4Stream::kBlockSize;  // 192
  constexpr size_t N = 8;

  MatStorageT<MxFp4Stream> BT("BT_edge", Extents2D(N, K), env.ctx.allocator,
                              MatPadding::kPacked);
  BT.SetScale(1.0f);
  MxFp4Stream* blocks = BT.Span().ptr;
  static constexpr uint8_t kScales[kBlocks] = {0, 1, 127, 253, 254, 255};
  for (size_t n = 0; n < N; ++n) {
    for (size_t b = 0; b < kBlocks; ++b) {
      MxFp4Stream& blk = blocks[n * kBlocks + b];
      blk.scale_e8m0 = kScales[b];
      for (size_t i = 0; i < 16; ++i) {
        if (b == 0 || b == 5) {
          // Non-zero +/-6.0 nibbles under zero/NaN scale codes must yield 0.0f.
          blk.bytes[i] = (b == 0) ? 0xF7 : 0xFF;
        } else if (b == 1) {
          // +/-0.5 * 2^-126 = +/-2^-127 (FP32 subnormal).
          blk.bytes[i] = static_cast<uint8_t>(((i + n) & 1) ? 0x19 : 0x91);
        } else if (b == 2) {
          // Exhaustively cover all 16 E2M1 nibbles (0..15).
          const uint8_t lo = static_cast<uint8_t>((i + n) & 0xF);
          const uint8_t hi = static_cast<uint8_t>((15 - i + n) & 0xF);
          blk.bytes[i] = static_cast<uint8_t>(lo | (hi << 4));
        } else {
          // Keep |E2M1| <= 1.5 so E2M1 * 2^127 stays finite in IEEE-754 float.
          const uint8_t lo = static_cast<uint8_t>(1 + ((i + n) % 3));
          const uint8_t hi = static_cast<uint8_t>(8 + 1 + ((i + n + 1) % 3));
          blk.bytes[i] = static_cast<uint8_t>(lo | (hi << 4));
        }
      }
    }
  }

  MatStorageT<BF16> A_full("A_edge", Extents2D(4, K), env.ctx.allocator,
                           MatPadding::kOdd);
  ZeroInit(A_full);
  // Row 0: 1e37f on zero/0xFF scale blocks (guards against Inf * 0 = NaN).
  for (size_t i = 0; i < 32; ++i) {
    A_full.Row(0)[0 + i] = hwy::BF16FromF32(1e37f);
    A_full.Row(0)[160 + i] = hwy::BF16FromF32(-1e37f);
  }
  // Row 1: isolate block 1 (e8m0 == 1 subnormal 2^-127) with period-3 signs
  // (16 % 3 != 0) so lo and hi nibbles do not cancel to 0.
  for (size_t i = 0; i < 32; ++i) {
    A_full.Row(1)[32 + i] = hwy::BF16FromF32(((i % 3) == 0) ? -0.75f : 1.0f);
  }
  // Row 2: min-normal BF16 (2^-126, tiny pre-scale path) on block 4
  // (e8m0 == 254, weight scale 2^127 -> finite O(10) dot product).
  for (size_t i = 0; i < 32; ++i) {
    A_full.Row(2)[128 + i] = hwy::BF16FromF32(
        ((i % 3) == 0) ? -8.0f * 1.17549435e-38f : 6.0f * 1.17549435e-38f);
  }
  // Row 3: all 16 E2M1 nibbles on block 2 (e8m0 == 127) and 2^-12f on blocks
  // 3..4 (e8m0 == 253, 254, guarding against premature dot_f * 2^127 overflow).
  for (size_t i = 0; i < 32; ++i) {
    A_full.Row(3)[64 + i] = hwy::BF16FromF32(1.0f / 4096.0f);
    A_full.Row(3)[96 + i] = hwy::BF16FromF32(1.0f / 4096.0f);
    A_full.Row(3)[128 + i] = hwy::BF16FromF32((i & 1) ? -1.0f / 4096.0f
                                                      : 1.0f / 2048.0f);
  }

  // Promote A_full and BT to float for exact per-(r, c) reference and L1 norm
  // so targets with BF16 FTZ do not flush Block 1's +/-2^-127 weights.
  MatStorageT<float> A_ref("A_ref", Extents2D(4, K), env.ctx.allocator,
                           MatPadding::kOdd);
  for (size_t r = 0; r < 4; ++r) {
    DecompressAndZeroPad(df, A_full.PaddedSpan(), r * A_full.Stride(),
                         A_ref.Row(r), K);
  }
  MatStorageT<float> B_ref("B_ref", Extents2D(N, K), env.ctx.allocator,
                           MatPadding::kOdd);
  for (size_t c = 0; c < N; ++c) {
    DecompressAndZeroPad(df, BT.PaddedSpan(), c * K, B_ref.Row(c), K);
  }

  MatStorageT<float> bias("bias", N, env.ctx.allocator);
  for (size_t c = 0; c < N; ++c) {
    bias.Row(0)[c] = 0.125f * static_cast<float>(c + 1);
  }

  for (size_t M = 1; M <= 4; ++M) {
    MatPtrT<BF16> A_sub("A_sub", Extents2D(M, K));
    A_sub.SetPtr(A_full.Row(0), A_full.Stride());
    for (bool use_add : {false, true}) {
      const float* add_ptr = use_add ? bias.PackedScale1() : nullptr;
      MatStorageT<float> C("C", Extents2D(M, N), env.ctx.allocator,
                           MatPadding::kOdd);
      C.AllocateAndAttachRowPtrs(env.row_ptrs);
      MMOptions options;
      for (size_t rep = 0; rep < 8; ++rep) {
        MatMulStatic(A_sub, BT, add_ptr, env, C, options);
        for (size_t r = 0; r < M; ++r) {
          for (size_t c = 0; c < N; ++c) {
            const float add_v = use_add ? add_ptr[c] : 0.0f;
            const float expected =
                add_v + Dot(df, MakeConst(BT.Span()), c * K, A_ref.Row(r), K);
            const float actual = C.Row(r)[c];
            HWY_ASSERT(hwy::ScalarIsFinite(expected));
            HWY_ASSERT(hwy::ScalarIsFinite(actual));
            if ((r == 1 || r == 2) && !use_add) {
              HWY_ASSERT(expected != 0.0f);
            }
            double abs_dot = 0.0;
            for (size_t k = 0; k < K; ++k) {
              abs_dot += hwy::ScalarAbs(static_cast<double>(A_ref.Row(r)[k]) *
                                        static_cast<double>(B_ref.Row(c)[k]));
            }
            const double tol = 0.01 * abs_dot +
                               (use_add ? 1e-6 * hwy::ScalarAbs(expected) : 0);
            if (hwy::ScalarAbs(static_cast<double>(actual) -
                               static_cast<double>(expected)) > tol) {
              HWY_ABORT("M=%zu add=%d r=%zu c=%zu expected=%g actual=%g", M,
                        use_add, r, c, expected, actual);
            }
          }
        }
      }
    }
  }

  // Partial tail block (K = 70, not a multiple of 32) with block-aligned
  // B.Stride() == 96 and non-zero trailing values in cols 70..95, covering
  // B3A2C0_4Bit (M = 1..4, with row 0 zeroed to cover max_abs == 0) and the
  // MMDecompress fallback (M = 8).
  constexpr size_t kFullK = 96;
  constexpr size_t kSubK = 70;
  MatStorageT<MxFp4Stream> BT_full = GenerateTransposedMat<MxFp4Stream>(
      Extents2D(N, kFullK), MatPadding::kPacked, env.ctx);
  MatPtrT<MxFp4Stream> BT_sub("BT_sub", Extents2D(N, kSubK));
  BT_sub.SetPtr(BT_full.Span().ptr, kFullK);
  BT_sub.SetScale(BT_full.Scale());
  static constexpr size_t kTailMs[] = {1, 2, 3, 4, 8};
  for (size_t tail_M : kTailMs) {
    MatStorageT<BF16> A_tail =
        GenerateMat<BF16>(Extents2D(tail_M, kSubK), MatPadding::kOdd, env.ctx);
    hwy::ZeroBytes(A_tail.Row(0), kSubK * sizeof(BF16));
    MatStorageT<float> C_tail("C_tail", Extents2D(tail_M, N), env.ctx.allocator,
                              MatPadding::kOdd);
    C_tail.AllocateAndAttachRowPtrs(env.row_ptrs);
    MMOptions options;
    MatMulStatic(A_tail, BT_sub, /*add=*/nullptr, env, C_tail, options);
    const float scale = A_tail.Scale() * BT_sub.Scale();
    for (size_t r = 0; r < tail_M; ++r) {
      for (size_t c = 0; c < N; ++c) {
        const float expected = scale * Dot(df, MakeConst(BT_full.Span()),
                                           c * kFullK, A_tail.Row(r), kSubK);
        const float actual = C_tail.Row(r)[c];
        HWY_ASSERT(hwy::ScalarAbs(actual - expected) <=
                   0.01f * hwy::ScalarAbs(expected) + 1e-4f);
      }
    }
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gcpp {
int64_t first_target = 0;  // none run yet
HWY_BEFORE_TEST(MatMulTest);
HWY_EXPORT_AND_TEST_P(MatMulTest, TestTiny);
HWY_EXPORT_AND_TEST_P(MatMulTest, TestMxFp4EdgeCases);
HWY_EXPORT_AND_TEST_P(MatMulTest, TestAllMatMul);
HWY_AFTER_TEST();

}  // namespace gcpp

#endif
