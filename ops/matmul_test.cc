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
#include <cstring>
#include <set>
#include <vector>

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
  std::vector<TC> first_output;
  std::vector<TC> first_fused_output;
  for (size_t rep = 0; rep < 16; ++rep) {
    MMPerKey* per_key = MatMulStatic(A, BT, add_row, env, C, options);
    AssertClose(A, BT, C_slow, C, env.ctx.allocator, env.row_ptrs, line);
    // Check before TwoMatMulStatic(), which can invalidate per_key.
    const bool autotune_done = !!per_key->autotune.Best();
    if (env.schedule != MMSchedule::kAutoTune) {
      bool selected = autotune_done;
#if GEMMA_ONEDNN_BRGEMM
      selected |= per_key->brgemm_autotune.Best() != nullptr;
#endif
#if GEMMA_ONEDNN_MATMUL
      selected |= per_key->onednn_built;
#endif
      HWY_ASSERT(selected);
      if constexpr (!IsBF16<TA>()) {
        HWY_ASSERT(per_key->autotune_par_a.Best() != nullptr);
      }
      if (rep == 0) first_output.resize(C.Extents().Area());
      for (size_t row = 0; row < C.Rows(); ++row) {
        TC* first = first_output.data() + row * C.Cols();
        if (rep == 0) {
          hwy::CopyBytes(C.Row(row), first, C.Cols() * sizeof(TC));
        } else {
          HWY_ASSERT(std::memcmp(first, C.Row(row), C.Cols() * sizeof(TC)) ==
                     0);
        }
      }
    }

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
      if (env.schedule != MMSchedule::kAutoTune) {
        if (rep == 0) first_fused_output.resize(C2.Extents().Area());
        for (size_t row = 0; row < C2.Rows(); ++row) {
          TC* first = first_fused_output.data() + row * C2.Cols();
          if (rep == 0) {
            hwy::CopyBytes(C2.Row(row), first, C2.Cols() * sizeof(TC));
          } else {
            HWY_ASSERT(std::memcmp(first, C2.Row(row),
                                   C2.Cols() * sizeof(TC)) == 0);
          }
        }
      }
      options.func = nullptr;  // reset for next call

      // TwoMatMulStatic() does not support adding a bias vector.
      if (!add) {
        AssertClose(A, BT, C, C2, env.ctx.allocator, env.row_ptrs, line);
      }
    }

    if (autotune_done && env.schedule == MMSchedule::kAutoTune) break;
  }
}

using F32 = float;
using SFP = SfpStream;

// Fixed schedules exercise both activation conversion and the fused path.
// TestMatMul compares against the scalar reference and checks repeated results
// bit-for-bit, including the transition from the first call to the cached path.
void TestFixedSchedules() {
  ThreadingArgs threading_args;
  threading_args.max_threads = 2;
  threading_args.bind = Tristate::kFalse;
  ThreadingContext ctx(threading_args);
  for (MMSchedule schedule : {MMSchedule::kFixed, MMSchedule::kFixedMinK}) {
    MatMulEnv env(ctx, schedule);
    for (size_t M : {size_t{1}, size_t{3}, size_t{4}, size_t{17}, size_t{65}}) {
      TestMatMul<F32, BF16, F32>(M, 258, 32, true, env, __LINE__);
      TestMatMul<BF16, BF16, BF16>(M, 256, 32, false, env, __LINE__);
      TestMatMul<BF16, SFP, BF16>(M, 4096, 32, false, env, __LINE__);
    }
    // K above the one-block limit must still yield a legal fixed schedule.
    TestMatMul<F32, BF16, BF16>(3, 2 * kMaxKC, 32, true, env, __LINE__);
  }
}

void AssertSameConfig(const MMConfig& a, const MMConfig& b) {
  HWY_ASSERT_EQ(a.MR(), b.MR());
  HWY_ASSERT_EQ(a.MC(), b.MC());
  HWY_ASSERT_EQ(a.KC(), b.KC());
  HWY_ASSERT_EQ(a.NC(), b.NC());
  HWY_ASSERT_EQ(static_cast<int>(a.Order()), static_cast<int>(b.Order()));
  HWY_ASSERT_EQ(a.InnerTasks(), b.InnerTasks());
}

// The first M encountered in a shared bucket must not affect a fixed config.
void TestFixedBucketOrder() {
  ThreadingArgs threading_args;
  threading_args.max_threads = 2;
  threading_args.bind = Tristate::kFalse;
  ThreadingContext ctx(threading_args);
  auto A = GenerateMat<BF16>(Extents2D(7, 4096), MatPadding::kOdd, ctx);
  auto B = GenerateTransposedMat<SFP>(Extents2D(32, 4096),
                                      MatPadding::kPacked, ctx);
  MatStorageT<BF16> C1("C1", Extents2D(7, 32), ctx.allocator, MatPadding::kOdd);
  MatStorageT<BF16> C2("C2", Extents2D(7, 32), ctx.allocator, MatPadding::kOdd);
  for (MMSchedule schedule : {MMSchedule::kFixed, MMSchedule::kFixedMinK}) {
    MatMulEnv small_first(ctx, schedule);
    MatMulEnv large_first(ctx, schedule);
    A.OverrideRows(4);
    C1.OverrideRows(4);
    const MMConfig small_config = *MatMulStatic(
        A, B, nullptr, small_first, C1, MMOptions())->autotune.Best();
    A.OverrideRows(7);
    C2.OverrideRows(7);
    const MMConfig large_config = *MatMulStatic(
        A, B, nullptr, large_first, C2, MMOptions())->autotune.Best();
    AssertSameConfig(small_config, large_config);
    C1.OverrideRows(7);
    MatMulStatic(A, B, nullptr, small_first, C1, MMOptions());
    for (size_t row = 0; row < C1.Rows(); ++row) {
      HWY_ASSERT(std::memcmp(C1.Row(row), C2.Row(row),
                             C1.Cols() * sizeof(BF16)) == 0);
    }
  }
}

void TestScheduleCandidates() {
  ThreadingArgs threading_args;
  threading_args.max_threads = 2;
  threading_args.bind = Tristate::kFalse;
  ThreadingContext ctx(threading_args);
  for (size_t M : {size_t{1}, size_t{3}, size_t{17}, size_t{127}}) {
    for (size_t K : {size_t{1}, size_t{258}, size_t{4096}, 2 * kMaxKC}) {
      for (size_t num_B : {size_t{1}, size_t{2}}) {
        for (size_t sizeof_TC : {sizeof(BF16), sizeof(float)}) {
          const auto all =
              MMCandidates(ctx.cache_info, M, K, 128, num_B, sizeof_TC, false);
          for (MMSchedule schedule :
               {MMSchedule::kFixed, MMSchedule::kFixedMinK}) {
            const auto fixed = MMCandidates(ctx.cache_info, M, K, 128, num_B,
                                            sizeof_TC, false, schedule);
            HWY_ASSERT_EQ(size_t{1}, fixed.size());
            const auto again = MMCandidates(ctx.cache_info, M, K, 128, num_B,
                                            sizeof_TC, false, schedule);
            AssertSameConfig(fixed.front(), again.front());
            if (schedule == MMSchedule::kFixed) {
              AssertSameConfig(all.front(), fixed.front());
            } else {
              const size_t splits = fixed.front().RangesOfKC(K).NumTasks();
              for (const MMConfig& candidate : all) {
                HWY_ASSERT(splits <= candidate.RangesOfKC(K).NumTasks());
              }
              // First candidate with this split count wins ties.
              for (const MMConfig& candidate : all) {
                if (candidate.RangesOfKC(K).NumTasks() == splits) {
                  AssertSameConfig(candidate, fixed.front());
                  break;
                }
              }
            }
          }
        }
      }
    }
  }
}

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

  // minimal non-square test. kColsARowsB must be at least 2 vectors.
  TestMatMul<F32>(35, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16>(34, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, BF16>(33, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, F32>(33, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, SFP>(31, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, SFP>(29, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, Q4_0Stream>(31, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, Q4_0Stream>(29, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32>(4, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<BF16>(4, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<F32, BF16>(4, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, F32>(4, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<F32, SFP>(4, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, SFP>(4, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<F32, Q4_0Stream>(4, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, Q4_0Stream>(4, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<F32>(3, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16>(3, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, BF16>(3, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, F32>(3, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, SFP>(3, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, SFP>(3, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, Q4_0Stream>(3, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, Q4_0Stream>(3, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32>(2, 128, 64, /*add=*/true, env, __LINE__);
  TestMatMul<BF16>(2, 128, 64, /*add=*/false, env, __LINE__);
  TestMatMul<F32, BF16>(2, 128, 64, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, F32>(2, 128, 64, /*add=*/false, env, __LINE__);
  TestMatMul<F32, SFP>(2, 128, 64, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, SFP>(2, 128, 64, /*add=*/false, env, __LINE__);
  TestMatMul<F32, Q4_0Stream>(2, 128, 64, /*add=*/true, env, __LINE__);
  TestMatMul<BF16, Q4_0Stream>(2, 128, 64, /*add=*/false, env, __LINE__);
  TestMatMul<F32>(1, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16>(1, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, BF16>(1, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, F32>(1, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, SFP>(1, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, SFP>(1, 128, 32, /*add=*/true, env, __LINE__);
  TestMatMul<F32, Q4_0Stream>(1, 128, 32, /*add=*/false, env, __LINE__);
  TestMatMul<BF16, Q4_0Stream>(1, 128, 32, /*add=*/true, env, __LINE__);

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
HWY_EXPORT_AND_TEST_P(MatMulTest, TestFixedSchedules);
HWY_EXPORT_AND_TEST_P(MatMulTest, TestFixedBucketOrder);
HWY_EXPORT_AND_TEST_P(MatMulTest, TestScheduleCandidates);
HWY_EXPORT_AND_TEST_P(MatMulTest, TestAllMatMul);
HWY_AFTER_TEST();

TEST(MatMulSchedulingTest, SingleCandidate) {
  MMAutoTune<int> tuner;
  tuner.SetCandidates({42});
  ASSERT_EQ(tuner.NextConfig(), 42);
  tuner.NotifyTicks(123);
  ASSERT_NE(tuner.Best(), nullptr);
  EXPECT_EQ(*tuner.Best(), 42);
  EXPECT_EQ(tuner.BestTicks(), 123);
  EXPECT_EQ(tuner.FirstConfigTicks(), 123);
  EXPECT_EQ(tuner.WorstMinTicks(), 123);
}

TEST(MatMulSchedulingTest, MultipleCandidatesStillTune) {
  MMAutoTune<int> tuner;
  tuner.SetCandidates({0, 1});
  for (size_t round = 0; round < 4; ++round) {
    ASSERT_EQ(tuner.Best(), nullptr);
    ASSERT_EQ(tuner.NextConfig(), 0);
    tuner.NotifyTicks(110);
    ASSERT_EQ(tuner.Best(), nullptr);
    ASSERT_EQ(tuner.NextConfig(), 1);
    tuner.NotifyTicks(100);
  }
  ASSERT_NE(tuner.Best(), nullptr);
  EXPECT_EQ(*tuner.Best(), 1);
}

TEST(MatMulSchedulingTest, ActivationConversionCandidates) {
  for (size_t M : {size_t{1}, size_t{17}}) {
    EXPECT_EQ(MMParACandidates(M, MMSchedule::kAutoTune).size(), 4);
    for (MMSchedule schedule : {MMSchedule::kFixed, MMSchedule::kFixedMinK}) {
      const auto candidates = MMParACandidates(M, schedule);
      ASSERT_EQ(candidates.size(), 1);
      EXPECT_EQ(candidates.front(), MMParA::kK1);
    }
  }
}

#if GEMMA_ONEDNN_BRGEMM
TEST(MatMulSchedulingTest, BRGeMMCandidates) {
  for (size_t M : {size_t{32}, size_t{64}, size_t{127}}) {
    for (size_t K : {size_t{32}, size_t{1024}, size_t{16384}}) {
      const auto all = BRGeMMCandidates(M, K, 128);
      const auto fixed = BRGeMMCandidates(M, K, 128, false);
      ASSERT_EQ(fixed.size(), 1);
      EXPECT_EQ(fixed.front().M_blk, all.front().M_blk);
      EXPECT_EQ(fixed.front().batch_size, all.front().batch_size);
      const auto min_k = BRGeMMCandidates(M, K, 128, false, true);
      ASSERT_EQ(min_k.size(), 1);
      const size_t splits = hwy::DivCeil(K / 32, min_k.front().batch_size);
      for (const auto& candidate : all) {
        EXPECT_LE(splits, hwy::DivCeil(K / 32, candidate.batch_size));
      }
    }
  }
}
#endif

TEST(MatMulSchedulingTest, ActivationKeys) {
  std::set<MMKeys::Key> keys;
  for (size_t M : {size_t{1}, size_t{4}, size_t{64}, size_t{65535}}) {
    for (size_t K : {size_t{1}, size_t{1} << 19, (size_t{1} << 20) - 1}) {
      for (size_t N : {size_t{4}, size_t{1} << 19, (size_t{1} << 20) - 1}) {
        for (size_t num_B : {size_t{1}, size_t{2}}) {
          const auto original = static_cast<MMKeys::Key>(MMKeys::BucketM(M)) |
                                (static_cast<MMKeys::Key>(K) << 16) |
                                (static_cast<MMKeys::Key>(N) << 40) |
                                (static_cast<MMKeys::Key>(num_B) << 60);
          EXPECT_EQ(original, MMKeys::KeyFromDims(M, K, N, num_B));
          for (MMActivation activation :
               {MMActivation::kBF16, MMActivation::kI8,
                MMActivation::kI8Block}) {
            const auto key = MMKeys::KeyFromDims(M, K, N, num_B, activation);
            EXPECT_NE(key, MMKeys::kPadding);
            EXPECT_TRUE(keys.insert(key).second);
            EXPECT_EQ(key, MMKeys::KeyFromDims(MMKeys::BucketM(M), K, N, num_B,
                                               activation));
          }
        }
      }
    }
  }
}

}  // namespace gcpp

#endif
