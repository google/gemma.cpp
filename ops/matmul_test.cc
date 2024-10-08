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

#include "compression/shared.h"
#ifndef HWY_DISABLED_TARGETS
// Exclude HWY_SCALAR due to 2x bf16 -> f32.
#define HWY_DISABLED_TARGETS HWY_SCALAR
#endif

#include "ops/matmul.h"

#include <stddef.h>
#include <stdio.h>

#include <memory>

#include "compression/compress.h"
#include "util/threading.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/timer.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ops/matmul_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"
#include "ops/dot-inl.h"
#include "ops/matmul-inl.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

using FloatPtr = hwy::AlignedFreeUniquePtr<float[]>;

// Generates inputs: deterministic, within max SfpStream range.
template <typename MatT, size_t kRows, size_t kCols,
          size_t kNum = kRows * kCols,
          class MatPtr = std::unique_ptr<CompressedArray<MatT, kNum>>>
MatPtr GenerateMat(size_t offset, hwy::ThreadPool& pool) {
  gcpp::CompressWorkingSet ws;
  FloatPtr content = hwy::AllocateAligned<float>(kNum);
  HWY_ASSERT(content);
  const float scale = SfpStream::kMax / (kNum + offset);
  pool.Run(0, kRows, [&](const size_t i, size_t /*thread*/) {
    for (size_t j = 0; j < kCols; j++) {
      content[i * kCols + j] =
          static_cast<float>((i * kCols + j + offset) * scale);
    }
  });

  MatPtr mat = std::make_unique<CompressedArray<MatT, kNum>>();
  CompressScaled(content.get(), kNum, ws, *mat, pool);
  mat->set_scale(0.6f);  // Arbitrary value, different from 1.
  return mat;
}

template <typename MatT, size_t kRows, size_t kCols,
          size_t kNum = kRows * kCols,
          class MatPtr = std::unique_ptr<CompressedArray<MatT, kNum>>>
MatPtr GenerateTransposedMat(size_t offset, hwy::ThreadPool& pool) {
  gcpp::CompressWorkingSet ws;
  FloatPtr content = hwy::AllocateAligned<float>(kNum);
  const float scale = SfpStream::kMax / (kNum + offset);
  pool.Run(0, kRows, [&](const size_t i, size_t /*thread*/) {
    for (size_t j = 0; j < kCols; j++) {
      content[j * kRows + i] =
          static_cast<float>((i * kCols + j + offset) * scale);
    }
  });

  MatPtr mat = std::make_unique<CompressedArray<MatT, kNum>>();
  CompressScaled(content.get(), kNum, ws, *mat, pool);
  // Arbitrary value, different from 1, must match GenerateMatHeap.
  mat->set_scale(0.6f);
  return mat;
}

template <typename MatT, size_t kRows, size_t kCols,
          size_t kNum = kRows * kCols,
          class MatPtr = std::unique_ptr<CompressedArray<MatT, kNum>>>
MatPtr GenerateZeroMat(hwy::ThreadPool& pool) {
  gcpp::CompressWorkingSet ws;
  FloatPtr content = hwy::AllocateAligned<float>(kNum);
  HWY_ASSERT(content);

  pool.Run(0, kRows, [&](const size_t i, size_t thread) {
    hwy::ZeroBytes(&content[i * kCols], kCols * sizeof(content[0]));
  });

  MatPtr mat = std::make_unique<CompressedArray<MatT, kNum>>();
  CompressScaled(content.get(), kNum, ws, *mat, pool);
  mat->set_scale(1.2f);  // Arbitrary value, different from 1.
  return mat;
}

// Returns 1-norm, used for estimating tolerable numerical differences.
double MaxColAbsSum(const float* HWY_RESTRICT a, size_t rows, size_t cols) {
  double max_col_abs_sum = 0.0;
  for (size_t c = 0; c < cols; c++) {
    double col_abs_sum = 0.0;
    for (size_t r = 0; r < rows; r++) {
      col_abs_sum += hwy::ScalarAbs(a[r * cols + c]);
    }
    max_col_abs_sum = HWY_MAX(max_col_abs_sum, col_abs_sum);
  }
  return max_col_abs_sum;
}

template <typename MatTA, typename MatTB>
void AssertClose(size_t rows_ac, size_t cols_ab, size_t cols_c_rows_b,
                 const MatTA* HWY_RESTRICT pa,
                 const MatTB* HWY_RESTRICT pb_trans,
                 const float* HWY_RESTRICT expected_c,
                 const float* HWY_RESTRICT actual_c) {
  const hn::ScalableTag<float> df;
  const size_t num_a = rows_ac * cols_ab;
  const size_t num_b = cols_c_rows_b * cols_ab;
  HWY_ASSERT(num_a % hn::Lanes(df) == 0);  // for DecompressAndZeroPad
  HWY_ASSERT(num_b % hn::Lanes(df) == 0);  // for DecompressAndZeroPad
  const size_t num_c = rows_ac * cols_c_rows_b;
  FloatPtr a = hwy::AllocateAligned<float>(num_a);
  FloatPtr b_trans = hwy::AllocateAligned<float>(num_b);
  HWY_ASSERT(a && b_trans);
  DecompressAndZeroPad(df, MakeSpan(pa, num_a), 0, a.get(), num_a);
  DecompressAndZeroPad(df, MakeSpan(pb_trans, num_b), 0, b_trans.get(), num_b);

  const double norm = MaxColAbsSum(a.get(), rows_ac, cols_ab) *
                      MaxColAbsSum(b_trans.get(), cols_c_rows_b, cols_ab);
  // Dot(float,BF16) rounds both to BF16.
  using RefType = BF16;
  const double epsilon = hwy::ConvertScalarTo<double>(hwy::Epsilon<RefType>());
  const double tolerance = 200.0 * norm * epsilon;

  for (size_t idx = 0; idx < num_c; idx++) {
    const double expected_value = expected_c[idx];
    const double actual_value = actual_c[idx];

    if (!(expected_value - tolerance <= actual_value &&
          actual_value <= expected_value + tolerance)) {
      fprintf(
          stderr,
          "expected[%lu]: %f, actual[%lu]: %f, norm %f eps %E tolerance %f\n",
          idx, expected_value, idx, actual_value, norm, epsilon, tolerance);
      HWY_ASSERT(0);
    }
  }
}

template <typename MatTA, typename MatTB>
HWY_INLINE void MatMulSlow(size_t rows_ac, size_t cols_a_rows_b, size_t cols_bc,
                           const MatTA* HWY_RESTRICT a,
                           const MatTB* HWY_RESTRICT b_trans, const float scale,
                           const float* HWY_RESTRICT add, MatMulEnv& env,
                           float* HWY_RESTRICT out) {
  // MatTA can be any Packed except NuqStream because it uses pointer
  // arithmetic, because it is the second argument to Dot, which does not
  // support a v_ofs.
  static_assert(sizeof(MatTA) >= sizeof(BF16), "A matrix must be BF16/f32");

  const hn::ScalableTag<float> df;  // lane type is ignored
  const PackedSpan<const MatTB> b_span =
      MakeSpan(b_trans, cols_a_rows_b * cols_bc);

  env.Pools().Outer().Run(
      0, rows_ac, [&](const uint64_t i, size_t o_thread) HWY_ATTR {
        hwy::ThreadPool& inner = env.Pools().Inner(o_thread);
        if (add != nullptr) {
          inner.Run(0, cols_bc, [&](const uint64_t j, size_t i_thread) {
            out[i * cols_bc + j] =
                scale * Dot(df, b_span, j * cols_a_rows_b,
                            a + i * cols_a_rows_b, cols_a_rows_b) +
                add[j];
          });
        } else {
          inner.Run(0, cols_bc, [&](const uint64_t j, size_t i_thread) {
            out[i * cols_bc + j] =
                scale * Dot(df, b_span, j * cols_a_rows_b,
                            a + i * cols_a_rows_b, cols_a_rows_b);
          });
        }
      });
}

void PrintSpeed(const char* algo, size_t rows_ac, size_t cols_a_rows_b,
                size_t cols_bc, double elapsed) {
  const size_t num_b = cols_a_rows_b * cols_bc;
  // 2x because of FMA.
  fprintf(stderr, "                     %10s: %f seconds, %.1f GFLOPS.\n", algo,
          elapsed, 2 * 1E-9 * rows_ac * num_b / elapsed);
}

template <size_t kRowsAC, size_t kColsARowsB, size_t kColsBC, bool kAdd,
          typename MatTA, typename MatTB = MatTA>
void TestMatMul(MatMulEnv& env) {
  hwy::ThreadPool& pool = env.Pool();
  const bool want_bench = kColsBC > 2000;  // avoid spam for small matrices
  fprintf(stderr, "TestMatMul %lu, %lu, %lu, add=%d, MatTA=%s, MatTB=%s\n",
          kRowsAC, kColsARowsB, kColsBC, kAdd, TypeName<MatTA>(),
          TypeName<MatTB>());

  std::unique_ptr<CompressedArray<MatTA, kRowsAC * kColsARowsB>> a =
      GenerateMat<MatTA, kRowsAC, kColsARowsB>(0, pool);
  std::unique_ptr<CompressedArray<MatTB, kColsARowsB * kColsBC>> b_trans =
      GenerateTransposedMat<MatTB, kColsARowsB, kColsBC>(0, pool);
  FloatPtr c = hwy::AllocateAligned<float>(kRowsAC * kColsBC);
  HWY_ASSERT(c);

  const float scale = a->scale() * b_trans->scale();
  std::unique_ptr<CompressedArray<float, kColsBC>> add;
  if (kAdd) {
    add = GenerateMat<float, 1, kColsBC>(0, pool);
    add->set_scale(1.0f);
  }

  std::unique_ptr<CompressedArray<float, kRowsAC * kColsBC>> c_slow =
      GenerateZeroMat<float, kRowsAC, kColsBC>(pool);
  const double start_slow = hwy::platform::Now();
  // Compare the dnnl matmul results with the hwy matmul results.
  MatMul_hwy<kAdd>(kRowsAC, ConstMat(a->data(), kColsARowsB),
                 ConstMat(b_trans->data(), kColsARowsB), scale,
                 kAdd ? add->data_scale1() : nullptr, env,
                 MutableMat(c_slow->data(), kColsBC));
  // MatMulSlow(kRowsAC, kColsARowsB, kColsBC, a->data(), b_trans->data(), scale,
  //            kAdd ? add->data() : nullptr, c_slow->data());
  if (want_bench) {
    PrintSpeed("MatMulSlow", kRowsAC, kColsARowsB, kColsBC,
               hwy::platform::Now() - start_slow);
  }

  double min_elapsed = hwy::HighestValue<double>();
  for (int rep = 0; rep < (want_bench ? 3 : 1); ++rep) {
    const double start_tiled = hwy::platform::Now();
    MatMul<kAdd>(kRowsAC, ConstMat(a->data(), kColsARowsB),
                 ConstMat(b_trans->data(), kColsARowsB), scale,
                 kAdd ? add->data_scale1() : nullptr, env,
                 MutableMat(c.get(), kColsBC));
    min_elapsed = HWY_MIN(min_elapsed, hwy::platform::Now() - start_tiled);
  }
  if (want_bench) {
    PrintSpeed("MatMul", kRowsAC, kColsARowsB, kColsBC, min_elapsed);
  }

  AssertClose(kRowsAC, kColsARowsB, kColsBC, a->data(), b_trans->data(),
              c_slow->data(), c.get());
}

void TestAllMatMul() {
  tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, 128);
  // Skip EMU128 (10x slower than SSE4 for SFP) and older x86.
  if (HWY_TARGET != HWY_AVX3 && HWY_TARGET != HWY_AVX3_SPR) {
    return;
  }

  PerClusterPools pools(/*max_clusters=*/1, /*max_threads=*/4, /*pin=*/1);
  MatMulEnv env(pools);
  pools.StartSpinning();

  using F32 = float;
  using SFP = SfpStream;

  // large-scale test: batch_size=128 is better than 64 or 256 for SKX.
  TestMatMul<128, 24576, 3072, /*kAdd=*/false, F32, BF16>(env);
  TestMatMul<128, 3072, 24576, /*kAdd=*/false, BF16>(env);
  TestMatMul<1, 24576, 3072, /*kAdd=*/false, BF16>(env);
  TestMatMul<1, 3072, 24576, /*kAdd=*/false, F32, BF16>(env);

  // medium-sized square test - temporarily disabled for faster testing.
  if constexpr (false) {
    TestMatMul<512, 512, 512, /*kAdd=*/false, F32>(env);
    TestMatMul<512, 512, 512, /*kAdd=*/true, BF16>(env);
    TestMatMul<512, 512, 512, /*kAdd=*/false, F32, BF16>(env);
    TestMatMul<512, 512, 512, /*kAdd=*/true, BF16, F32>(env);
    TestMatMul<512, 512, 512, /*kAdd=*/false, F32, SFP>(env);
    TestMatMul<512, 512, 512, /*kAdd=*/true, BF16, SFP>(env);
  }

  // minimal non-square test. kColsARowsB must be at least 2 vectors.
  TestMatMul<35, 128, 32, /*kAdd=*/false, F32>(env);
  TestMatMul<34, 128, 32, /*kAdd=*/true, BF16>(env);
  TestMatMul<33, 128, 32, /*kAdd=*/false, F32, BF16>(env);
  TestMatMul<33, 128, 32, /*kAdd=*/true, BF16, F32>(env);
  TestMatMul<4, 128, 32, /*kAdd=*/true, F32>(env);
  TestMatMul<4, 128, 32, /*kAdd=*/false, BF16>(env);
  TestMatMul<4, 128, 32, /*kAdd=*/true, F32, BF16>(env);
  TestMatMul<4, 128, 32, /*kAdd=*/false, BF16, F32>(env);
  TestMatMul<3, 128, 32, /*kAdd=*/false, F32>(env);
  TestMatMul<3, 128, 32, /*kAdd=*/true, BF16>(env);
  TestMatMul<3, 128, 32, /*kAdd=*/false, F32, BF16>(env);
  TestMatMul<3, 128, 32, /*kAdd=*/true, BF16, F32>(env);
  TestMatMul<2, 128, 64, /*kAdd=*/true, F32>(env);
  TestMatMul<2, 128, 64, /*kAdd=*/false, BF16>(env);
  TestMatMul<2, 128, 64, /*kAdd=*/true, F32, BF16>(env);
  TestMatMul<2, 128, 64, /*kAdd=*/false, BF16, F32>(env);
  TestMatMul<1, 128, 32, /*kAdd=*/false, F32>(env);
  TestMatMul<1, 128, 32, /*kAdd=*/true, BF16>(env);
  TestMatMul<1, 128, 32, /*kAdd=*/false, F32, BF16>(env);
  TestMatMul<1, 128, 32, /*kAdd=*/true, BF16, F32>(env);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gcpp {
HWY_BEFORE_TEST(MatmulTest);
HWY_EXPORT_AND_TEST_P(MatmulTest, TestAllMatMul);
HWY_AFTER_TEST();

}  // namespace gcpp

#endif
