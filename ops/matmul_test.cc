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

// Returns 1-norm, used for estimating tolerable numerical differences.
double MaxRowAbsSum(const MatStorageT<float>& a) {
  double max_row_abs_sum = 0.0;
  for (size_t r = 0; r < a.Rows(); r++) {
    const float* row = a.Row(r);
    double row_abs_sum = 0.0;
    for (size_t c = 0; c < a.Cols(); c++) {
      row_abs_sum += hwy::ScalarAbs(row[c]);
    }
    max_row_abs_sum = HWY_MAX(max_row_abs_sum, row_abs_sum);
  }
  return max_row_abs_sum;
}

// Returns the maximum absolute value of `a`.
float MaxAbs(const MatStorageT<float>& a) {
  float max_abs = 0.0f;
  for (size_t c = 0; c < a.Cols(); c++) {
    for (size_t r = 0; r < a.Rows(); r++) {
      const float* row = a.Row(r);
      max_abs = HWY_MAX(max_abs, hwy::ScalarAbs(row[c]));
    }
  }
  return max_abs;
}

// B is already transposed.
template <typename TA, typename TB, typename TC>
void AssertClose(const MatPtrT<TA>& A, const MatPtrT<TB>& B,
                 const MatPtrT<TC>& C_slow, const MatPtrT<TC>& C,
                 MatMulEnv& env, int line) {
  const hn::ScalableTag<float> df;
  const size_t cols = A.Cols();
  const size_t B_rows = B.Rows();
  // Round up for DecompressAndZeroPad.
  MatStorageT<float> a_batch("a_batch", A.Extents(), env.ctx.allocator,
                             MatPadding::kOdd);
  MatStorageT<float> b_trans_batch("b_trans_batch", B.Extents(),
                                   env.ctx.allocator, MatPadding::kOdd);
  MatStorageT<float> c_batch("c_batch", Extents2D(A.Rows(), B_rows),
                             env.ctx.allocator, MatPadding::kOdd);
  c_batch.AllocateAndAttachRowPtrs(env.row_ptrs);
  MatStorageT<float> c_slow_batch("c_slow_batch", Extents2D(A.Rows(), B_rows),
                                  env.ctx.allocator, MatPadding::kOdd);
  for (size_t m = 0; m < A.Rows(); ++m) {
    DecompressAndZeroPad(df, MakeSpan(A.Row(m), cols), 0, a_batch.Row(m), cols);
    DecompressAndZeroPad(df, MakeSpan(C.Row(m), B_rows), 0, c_batch.Row(m),
                         B_rows);
    DecompressAndZeroPad(df, MakeSpan(C_slow.Row(m), B_rows), 0,
                         c_slow_batch.Row(m), B_rows);
  }
  for (size_t n = 0; n < B_rows; ++n) {
    DecompressAndZeroPad(df, MakeSpan(B.Row(n), cols), 0, b_trans_batch.Row(n),
                         cols);
  }

  // MatMul rounds inputs to BF16, so error is proportional to the max input
  // magnitude, but also to f32 accumulation of rows in A and B.
  const double norm = MaxRowAbsSum(a_batch) * MaxRowAbsSum(b_trans_batch);
  const float max_abs = MaxAbs(a_batch) * MaxAbs(b_trans_batch);
  const double eps_bf16 = hwy::ConvertScalarTo<double>(hwy::Epsilon<BF16>());
  const double eps_f32 = hwy::ConvertScalarTo<double>(hwy::Epsilon<float>());
  double tolerance = 12 * norm * eps_f32;
  // Dot() also rounds F32,BF16 to BF16, but not with F32,F32, so increase the
  // tolerance there.
  if (IsF32<TA>() && IsF32<TB>()) {
    tolerance += 4 * max_abs * eps_bf16;
  }
  if (tolerance > 500.0) {
    HWY_WARN("high tolerance %f norm %f maxabs %f\n", tolerance, norm, max_abs);
  }
  const double max_rel = 1.0 + hwy::ConvertScalarTo<double>(hwy::Epsilon<TC>());

  for (size_t r = 0; r < A.Rows(); r++) {
    const float* expected_row = c_slow_batch.Row(r);
    const float* actual_row = c_batch.Row(r);
    for (size_t c = 0; c < B.Rows(); c++) {
      const double expected_value = static_cast<double>(expected_row[c]);
      const double actual_value = static_cast<double>(actual_row[c]);
      const bool in_range = expected_value - tolerance <= actual_value &&
                            actual_value <= expected_value + tolerance;

      if (!in_range) {
        const double max = HWY_MAX(expected_value, actual_value);
        const double min = HWY_MIN(expected_value, actual_value);
        const double rel = max / HWY_MAX(min, 1E-6);
        if (rel > max_rel) {
          hwy::Abort(__FILE__, line,
                     "(%zu,%zu): expected %f, actual %f, norm %f maxabs %f "
                     "tolerance %f rel %E max_rel %E\n",
                     r, c, expected_value, actual_value, norm, max_abs,
                     tolerance, rel, max_rel);
        }
      }
    }
  }
}

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
  hwy::ThreadPool& all_packages = pools.AllPackages();
  const IndexRangePartition get_row_c =
      StaticPartition(all_rows_c, all_packages.NumWorkers(), 1);
  ParallelizeOneRange(
      get_row_c, all_packages,
      [&](const IndexRange& rows_c, size_t package_idx) HWY_ATTR {
        hwy::ThreadPool& all_clusters = pools.AllClusters(package_idx);
        const size_t multiple = env.ctx.allocator.QuantumBytes() / sizeof(TB);
        const IndexRangePartition get_col_c =
            StaticPartition(all_cols_c, all_clusters.NumWorkers(), multiple);
        ParallelizeOneRange(
            get_col_c, all_clusters,
            [&](const IndexRange& cols_c, size_t cluster_idx) HWY_ATTR {
              for (size_t r : rows_c) {
                TC* HWY_RESTRICT C_row = C.Row(r);
                for (size_t c : cols_c) {
                  const float add = add_row ? add_row[c] : 0.0f;
                  C_row[c] = hwy::ConvertScalarTo<TC>(
                      add + scale * Dot(df, b_span, c * B.Stride(), A.Row(r),
                                        A.Cols()));
                }
              }
            });
      });
}

void PrintSpeed(const char* algo, const Extents2D& A_extents,
                const Extents2D& B_extents, double elapsed) {
  const size_t num_b = B_extents.Area();
  // 2x because of FMA.
  fprintf(stderr, "                     %10s: %f seconds, %.1f GFLOPS.\n", algo,
          elapsed, 2 * 1E-9 * A_extents.rows * num_b / elapsed);
}

template <typename TA, typename TB = TA, typename TC = float>
void TestMatMul(size_t rows_ac, size_t cols_a_rows_b, size_t cols_bc, bool add,
                MatMulEnv& env, int line) {
  hwy::ThreadPool& pool = env.ctx.pools.Pool();
  fprintf(stderr, "TestMatMul %zu, K=%zu, %zu, add=%d, TA=%s, TB=%s, TC=%s\n",
          rows_ac, cols_a_rows_b, cols_bc, add, TypeName<TA>(), TypeName<TB>(),
          TypeName<TC>());

  env.print_config = false;  // Too verbose.
  env.print_best = true;

  const Extents2D A_extents(rows_ac, cols_a_rows_b);
  const Extents2D B_extents(cols_bc, cols_a_rows_b);  // already transposed
  const Extents2D C_extents(rows_ac, cols_bc);

  MatStorageT<TA> A(
      GenerateMat<TA>(A_extents, env.ctx.allocator, MatPadding::kOdd, pool));
  // Must be packed because we call Span() on it.
  MatStorageT<TB> BT(GenerateTransposedMat<TB>(B_extents, env.ctx.allocator,
                                               MatPadding::kPacked, pool));
  MatStorageT<TC> C_slow("C_slow", C_extents, env.ctx.allocator,
                         MatPadding::kOdd);
  MatStorageT<TC> C("C", C_extents, env.ctx.allocator, MatPadding::kOdd);
  C.AllocateAndAttachRowPtrs(env.row_ptrs);

  MatStorageT<float> add_storage =
      add ? GenerateMat<float>(Extents2D(1, cols_bc), env.ctx.allocator,
                               MatPadding::kPacked, pool)
          : MatStorageT<float>("add", Extents2D(), env.ctx.allocator,
                               MatPadding::kPacked);
  add_storage.SetScale(1.0f);
  const float* add_row = add ? add_storage.PackedScale1() : nullptr;

  MatMulSlow(A, BT, add_row, env, C_slow);
  // A few reps to get coverage of the various autotuned code paths.
  for (size_t rep = 0; rep < 16; ++rep) {
    MMPerKey* per_key = MatMulStatic(A, BT, add_row, env, C);
    AssertClose(A, BT, C_slow, C, env, line);
    if (per_key->autotune.Best()) break;
  }
}

using F32 = float;
using SFP = SfpStream;

// Sweep all dimensions for a single input type and Highway target, to verify
// the remainder handling.
void TestTiny() {
  if (first_target == 0) first_target = HWY_TARGET;
  if (HWY_TARGET != first_target) return;

  for (size_t max_packages : {1, 2}) {
    ThreadingArgs threading_args;
    threading_args.bind = Tristate::kTrue;
    threading_args.max_packages = max_packages;
    ThreadingContext ctx(threading_args);
    MatMulEnv env(ctx);
    NestedPools& pools = env.ctx.pools;

    if constexpr (GEMMA_DISABLE_TOPOLOGY || kMaxPackages == 1) {
      if (max_packages == 2) break;  // we only have one package
    } else {
      // If less than the limit, we have already tested all num_packages.
      if (env.ctx.topology.FullTopology().packages.size() < max_packages) break;
    }
    fprintf(stderr, "TestTiny %zu: %s %s\n", max_packages,
            env.ctx.topology.TopologyString(), pools.PinString());

    pools.MaybeStartSpinning(threading_args.spin);

    for (size_t M = 1; M <= 12; ++M) {
      for (size_t K = 1; K <= 64; K *= 2) {
        for (size_t N = 4; N <= 64; N += max_packages * 4) {
          TestMatMul<F32, F32, F32>(M, K, N, /*add=*/false, env, __LINE__);
        }
      }
    }
    pools.MaybeStopSpinning(threading_args.spin);
  }
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
