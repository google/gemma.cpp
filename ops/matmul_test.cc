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

#ifndef HWY_DISABLED_TARGETS
// Exclude HWY_SCALAR due to 2x bf16 -> f32.
#define HWY_DISABLED_TARGETS HWY_SCALAR
#endif

#include <stddef.h>
#include <stdio.h>

#include <memory>

#include "compression/compress.h"
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
#include "hwy/tests/test_util-inl.h"
// After highway.h
#include "ops/matmul-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Generates inputs: deterministic, within max SfpStream range.
template <typename MatT, size_t kRows, size_t kCols>
std::unique_ptr<CompressedArray<MatT, kRows * kCols>> GenerateMatHeap(
    size_t offset, hwy::ThreadPool& pool) {
  gcpp::CompressWorkingSet ws;
  hwy::AlignedFreeUniquePtr<float[]> content =
      hwy::AllocateAligned<float>(kRows * kCols);
  const float scale = 1.875f / (kCols * kRows + offset);
  pool.Run(0, kRows, [&](const size_t i, size_t /*thread*/) {
    for (size_t j = 0; j < kCols; j++) {
      content[i * kCols + j] =
          static_cast<float>((i * kCols + j + offset) * scale);
    }
  });

  std::unique_ptr<CompressedArray<MatT, kRows * kCols>> mat =
      std::make_unique<CompressedArray<MatT, kRows * kCols>>();
  Compress(content.get(), kRows * kCols, ws, kRows * kCols, mat->data(), 0,
           pool);
  mat->set_scale(0.6f);  // Arbitrary value, different from 1.
  return mat;
}

template <typename MatT, size_t kRows, size_t kCols>
std::unique_ptr<CompressedArray<MatT, kRows * kCols>> GenerateTransposeMatHeap(
    size_t offset, hwy::ThreadPool& pool) {
  gcpp::CompressWorkingSet ws;
  hwy::AlignedFreeUniquePtr<float[]> content =
      hwy::AllocateAligned<float>(kRows * kCols);
  const float scale = 1.875f / (kCols * kRows + offset);
  pool.Run(0, kRows, [&](const size_t i, size_t /*thread*/) {
    for (size_t j = 0; j < kCols; j++) {
      content[j * kRows + i] =
          static_cast<float>((i * kCols + j + offset) * scale);
    }
  });

  std::unique_ptr<CompressedArray<MatT, kRows * kCols>> mat =
      std::make_unique<CompressedArray<MatT, kRows * kCols>>();
  Compress(content.get(), kRows * kCols, ws, kRows * kCols, mat->data(), 0,
           pool);
  // Arbitrary value, different from 1, must match GenerateMatHeap.
  mat->set_scale(0.6f);
  return mat;
}

template <typename MatT, size_t kRows, size_t kCols>
std::unique_ptr<CompressedArray<MatT, kRows * kCols>> GenerateZeroMatHeap(
    hwy::ThreadPool& pool) {
  gcpp::CompressWorkingSet ws;
  hwy::AlignedFreeUniquePtr<float[]> content =
      hwy::AllocateAligned<float>(kRows * kCols);

  pool.Run(0, kRows, [&](const size_t i, size_t thread) {
    hwy::ZeroBytes(&content[i * kCols], kCols * sizeof(content[0]));
  });

  std::unique_ptr<CompressedArray<MatT, kRows * kCols>> mat =
      std::make_unique<CompressedArray<MatT, kRows * kCols>>();
  Compress(content.get(), kRows * kCols, ws, kRows * kCols, mat->data(), 0,
           pool);
  mat->set_scale(1.2f);  // Arbitrary value, different from 1.
  return mat;
}

template <typename MatT>
void Decompress(const MatT* compressed, size_t num, float* out) {
  const hn::ScalableTag<float> d;
  hwy::AlignedFreeUniquePtr<float[]> b = hwy::AllocateAligned<float>(num);
  CompressTraits<MatT>::Decompress(d, /*in_capacity=*/0, compressed, 0, out,
                                   num);
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
                 const MatTA* HWY_RESTRICT a_compr,
                 const MatTB* HWY_RESTRICT b_trans_compr,
                 const float* HWY_RESTRICT expected_c,
                 const float* HWY_RESTRICT actual_c) {
  const size_t num_a = rows_ac * cols_ab;
  const size_t num_b = cols_c_rows_b * cols_ab;
  const size_t num_c = rows_ac * cols_c_rows_b;
  hwy::AlignedFreeUniquePtr<float[]> a = hwy::AllocateAligned<float>(num_a);
  hwy::AlignedFreeUniquePtr<float[]> b_trans =
      hwy::AllocateAligned<float>(num_b);
  Decompress(a_compr, num_a, a.get());
  Decompress(b_trans_compr, num_b, b_trans.get());

  const double norm = MaxColAbsSum(a.get(), rows_ac, cols_ab) *
                      MaxColAbsSum(b_trans.get(), cols_c_rows_b, cols_ab);
  const double epsilon = hwy::ConvertScalarTo<double>(hwy::Epsilon<float>());
  const double tolerance = 50.0 * norm * epsilon;

  for (size_t idx = 0; idx < num_c; idx++) {
    const double expected_value = expected_c[idx];
    const double actual_value = actual_c[idx];

    if (!(expected_value - tolerance <= actual_value &&
          actual_value <= expected_value + tolerance)) {
      fprintf(stderr, "expected[%lu]: %f, actual[%lu]: %f\n", idx,
              expected_value, idx, actual_value);
      HWY_ASSERT(0);
    }
  }
}
// Largely unoptimized; reordered innermost loops nets ~5-10X speedup.
template <typename MatTA, typename MatTB, HWY_IF_T_SIZE_GT(MatTB, 1)>
HWY_INLINE void MatMulSlow(size_t rows_ac, size_t cols_a_rows_b, size_t cols_bc,
                           const MatTA* HWY_RESTRICT a,
                           const MatTB* HWY_RESTRICT b, const float scale,
                           const float* add, float* HWY_RESTRICT out) {
  for (size_t i = 0; i < rows_ac; ++i) {
    for (size_t k = 0; k < cols_a_rows_b; ++k) {
      for (size_t j = 0; j < cols_bc; ++j) {
        const float a1 = hwy::ConvertScalarTo<float>(a[i * cols_a_rows_b + k]);
        const float b1 = hwy::ConvertScalarTo<float>(b[k * cols_bc + j]);
        out[i * cols_bc + j] += scale * a1 * b1;
      }
    }
    if (add != nullptr) {
      for (size_t j = 0; j < cols_bc; ++j) {
        out[i * cols_bc + j] += add[j];
      }
    }
  }
}

// The above overload can handle combinations of f32 and bf16, but this one
// is required for MatTB = {SFP, NUQ}.
template <typename MatTA, typename MatTB, HWY_IF_T_SIZE(MatTB, 1)>
HWY_INLINE void MatMulSlow(size_t rows_ac, size_t cols_a_rows_b, size_t cols_bc,
                           const MatTA* HWY_RESTRICT a,
                           const MatTB* HWY_RESTRICT b_compr, const float scale,
                           const float* add, float* HWY_RESTRICT out) {
  const hn::ScalableTag<float> d;
  hwy::AlignedFreeUniquePtr<float[]> b =
      hwy::AllocateAligned<float>(cols_a_rows_b * cols_bc);
  CompressTraits<MatTB>::Decompress(d, /*in_capacity=*/0, b_compr, 0, b.get(),
                                    cols_a_rows_b * cols_bc);
  MatMulSlow(rows_ac, cols_a_rows_b, cols_bc, a, b.get(), scale, add, out);
}

void PrintSpeed(const char* algo, size_t rows_ac, size_t cols_a_rows_b,
                size_t cols_bc, double elapsed) {
  // 2 because of FMA.
  fprintf(stderr, "%s: %f seconds, %f GFLOPS.\n", algo, elapsed,
          2E-9 * rows_ac * cols_a_rows_b * cols_bc / elapsed);
}

template <size_t kRowsAC, size_t kColsARowsB, size_t kColsBC, bool kAdd,
          typename MatTA, typename MatTB = MatTA>
void TestMatMul(hwy::ThreadPool& pool) {
  using TraitsA = CompressTraits<MatTA>;
  using TraitsB = CompressTraits<MatTB>;
  const bool want_bench = kColsBC > 2000;  // avoid spam for small matrices
  fprintf(stderr, "TestMatMul %lu, %lu, %lu, add=%d, MatTA=%s, MatTB=%s\n",
          kRowsAC, kColsARowsB, kColsBC, kAdd, TraitsA::Name(),
          TraitsB::Name());

  std::unique_ptr<CompressedArray<MatTA, kRowsAC * kColsARowsB>> a =
      GenerateMatHeap<MatTA, kRowsAC, kColsARowsB>(0, pool);
  std::unique_ptr<CompressedArray<MatTB, kColsARowsB * kColsBC>> b_trans =
      GenerateTransposeMatHeap<MatTB, kColsARowsB, kColsBC>(0, pool);
  hwy::AlignedFreeUniquePtr<float[]> c =
      hwy::AllocateAligned<float>(kRowsAC * kColsBC);

  const float scale = a->scale() * b_trans->scale();
  std::unique_ptr<CompressedArray<float, kColsBC>> add;
  if (kAdd) {
    add = GenerateMatHeap<float, 1, kColsBC>(0, pool);
    add->set_scale(1.0f);
  }

  std::unique_ptr<CompressedArray<MatTB, kColsARowsB * kColsBC>> b =
      GenerateMatHeap<MatTB, kColsARowsB, kColsBC>(0, pool);
  HWY_ASSERT_EQ(scale, a->scale() * b->scale());
  std::unique_ptr<CompressedArray<float, kRowsAC * kColsBC>> c_slow =
      GenerateZeroMatHeap<float, kRowsAC, kColsBC>(pool);
  const double start_slow = hwy::platform::Now();
  MatMulSlow(kRowsAC, kColsARowsB, kColsBC, a->data(), b->data(), scale,
             kAdd ? add->data() : nullptr, c_slow->data());
  if (want_bench) {
    PrintSpeed("MatMulSlow", kRowsAC, kColsARowsB, kColsBC,
               hwy::platform::Now() - start_slow);
  }

  double min_elapsed = hwy::HighestValue<double>();
  for (int rep = 0; rep < (want_bench ? 3 : 1); ++rep) {
    const double start_tiled = hwy::platform::Now();
    MatMul_4x4<kAdd>(kRowsAC, MakeMat(a->data(), kColsARowsB),
                     MakeMat(b_trans->data(), kColsARowsB), scale,
                     kAdd ? add->data_scale1() : nullptr,
                     MakeMat(c.get(), kColsBC), pool);
    min_elapsed = HWY_MIN(min_elapsed, hwy::platform::Now() - start_tiled);
  }
  if (want_bench) {
    PrintSpeed("MatMul_4x4", kRowsAC, kColsARowsB, kColsBC, min_elapsed);
  }

  AssertClose(kRowsAC, kColsARowsB, kColsBC, a->data(), b_trans->data(),
              c_slow->data(), c.get());
}

void TestAllMatMul() {
  // Skip EMU128 (10x slower than SSE4 for SFP) and older x86.
  if (HWY_TARGET == HWY_EMU128 || HWY_TARGET == HWY_SSE4 ||
      HWY_TARGET == HWY_SSSE3 || HWY_TARGET == HWY_SSE2) {
    return;
  }

  hwy::ThreadPool pool(4);
  using F32 = float;
  using SFP = SfpStream;

  // large-scale test
  TestMatMul<64, 24576, 3072, /*kAdd=*/false, F32, SFP>(pool);
  TestMatMul<64, 3072, 24576, /*kAdd=*/false, F32, SFP>(pool);

  // medium-sized square test
  TestMatMul<512, 512, 512, /*kAdd=*/false, F32>(pool);
  TestMatMul<512, 512, 512, /*kAdd=*/true, BF16>(pool);
  TestMatMul<512, 512, 512, /*kAdd=*/false, F32, BF16>(pool);
  TestMatMul<512, 512, 512, /*kAdd=*/true, BF16, F32>(pool);
  TestMatMul<512, 512, 512, /*kAdd=*/false, F32, SFP>(pool);
  TestMatMul<512, 512, 512, /*kAdd=*/true, BF16, SFP>(pool);

  // minimal non-square test. kColsARowsB must be at least 2 vectors.
  TestMatMul<35, 128, 32, /*kAdd=*/false, F32>(pool);
  TestMatMul<34, 128, 32, /*kAdd=*/true, BF16>(pool);
  TestMatMul<33, 128, 32, /*kAdd=*/false, F32, BF16>(pool);
  TestMatMul<33, 128, 32, /*kAdd=*/true, BF16, F32>(pool);
  TestMatMul<31, 128, 32, /*kAdd=*/false, F32, SFP>(pool);
  TestMatMul<29, 128, 32, /*kAdd=*/true, BF16, SFP>(pool);
  TestMatMul<4, 128, 32, /*kAdd=*/true, F32>(pool);
  TestMatMul<4, 128, 32, /*kAdd=*/false, BF16>(pool);
  TestMatMul<4, 128, 32, /*kAdd=*/true, F32, BF16>(pool);
  TestMatMul<4, 128, 32, /*kAdd=*/false, BF16, F32>(pool);
  TestMatMul<4, 128, 32, /*kAdd=*/true, F32, SFP>(pool);
  TestMatMul<4, 128, 32, /*kAdd=*/false, BF16, SFP>(pool);
  TestMatMul<3, 128, 32, /*kAdd=*/false, F32>(pool);
  TestMatMul<3, 128, 32, /*kAdd=*/true, BF16>(pool);
  TestMatMul<3, 128, 32, /*kAdd=*/false, F32, BF16>(pool);
  TestMatMul<3, 128, 32, /*kAdd=*/true, BF16, F32>(pool);
  TestMatMul<3, 128, 32, /*kAdd=*/false, F32, SFP>(pool);
  TestMatMul<3, 128, 32, /*kAdd=*/true, BF16, SFP>(pool);
  TestMatMul<2, 128, 64, /*kAdd=*/true, F32>(pool);
  TestMatMul<2, 128, 64, /*kAdd=*/false, BF16>(pool);
  TestMatMul<2, 128, 64, /*kAdd=*/true, F32, BF16>(pool);
  TestMatMul<2, 128, 64, /*kAdd=*/false, BF16, F32>(pool);
  TestMatMul<2, 128, 64, /*kAdd=*/true, F32, SFP>(pool);
  TestMatMul<2, 128, 64, /*kAdd=*/false, BF16, SFP>(pool);
  TestMatMul<1, 128, 32, /*kAdd=*/false, F32>(pool);
  TestMatMul<1, 128, 32, /*kAdd=*/true, BF16>(pool);
  TestMatMul<1, 128, 32, /*kAdd=*/false, F32, BF16>(pool);
  TestMatMul<1, 128, 32, /*kAdd=*/true, BF16, F32>(pool);
  TestMatMul<1, 128, 32, /*kAdd=*/false, F32, SFP>(pool);
  TestMatMul<1, 128, 32, /*kAdd=*/true, BF16, SFP>(pool);
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
