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

  // Non-vector-multiple K.
  TestMatMul<F32, BF16>(128, 258, 128, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, BF16>(128, 258, 128, /*add=*/true, env, __LINE__);

  // minimal non-square test. kColsARowsB must be at least 2 vectors.
  TestMatMul<F32>(35, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16>(34, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, BF16>(33, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, F32>(33, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, SFP>(31, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, SFP>(29, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32>(4, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<BF16>(4, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<F32, BF16>(4, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, F32>(4, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<F32, SFP>(4, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, SFP>(4, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<F32>(3, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16>(3, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, BF16>(3, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, F32>(3, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, SFP>(3, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, SFP>(3, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32>(2, 128, 64, /*add=*/true, env, __LINE__);
  TestMatMul<BF16>(2, 128, 64, /*add=*/false, env, __LINE__);
  TestMatMul<F32, BF16>(2, 128, 64, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, F32>(2, 128, 64, /*add=*/false, env, __LINE__);
  TestMatMul<F32, SFP>(2, 128, 64, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, SFP>(2, 128, 64, /*add=*/false, env, __LINE__);
  TestMatMul<F32>(1, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16>(1, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, BF16>(1, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, F32>(1, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, SFP>(1, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, SFP>(1, 128, 32, /*add=*/true, env, __LINE__);

  pools.MaybeStopSpinning(threading_args.spin);
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
HWY_EXPORT_AND_TEST_P(MatMulTest, TestAllMatMul);
HWY_AFTER_TEST();

}  // namespace gcpp

#endif
