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

#include <algorithm>
#include <array>
#include <cmath>
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
#include "ops/ops-inl.h"  // MulByConst

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <typename MatT, size_t kOuter, size_t kInner>
std::unique_ptr<CompressedArray<MatT, kOuter * kInner>> GenerateMatHeap(
    size_t offset, hwy::ThreadPool& pool) {
  gcpp::CompressWorkingSet ws;
  hwy::AlignedFreeUniquePtr<float[]> content =
      hwy::AllocateAligned<float>(kOuter * kInner);
  const float scale = 1.875f / (kInner * kOuter + offset);
  pool.Run(0, kOuter, [&](const size_t i, size_t /*thread*/) {
    for (size_t j = 0; j < kInner; j++) {
      content[i * kInner + j] =
          static_cast<float>((i * kInner + j + offset) * scale);
    }
  });

  std::unique_ptr<CompressedArray<MatT, kOuter * kInner>> mat =
      std::make_unique<CompressedArray<MatT, kOuter * kInner>>();
  Compress(content.get(), kOuter * kInner, ws, kOuter * kInner, mat->data(), 0,
           pool);
  mat->set_scale(0.6f);  // Arbitrary value, different from 1.
  return mat;
}

template <typename MatT, size_t kOuter, size_t kInner>
std::unique_ptr<CompressedArray<MatT, kOuter * kInner>>
GenerateTransposeMatHeap(size_t offset, hwy::ThreadPool& pool) {
  gcpp::CompressWorkingSet ws;
  hwy::AlignedFreeUniquePtr<float[]> content =
      hwy::AllocateAligned<float>(kOuter * kInner);
  const float scale = 1.875f / (kInner * kOuter + offset);
  pool.Run(0, kOuter, [&](const size_t i, size_t /*thread*/) {
    for (size_t j = 0; j < kInner; j++) {
      content[j * kOuter + i] =
          static_cast<float>((i * kInner + j + offset) * scale);
    }
  });

  std::unique_ptr<CompressedArray<MatT, kOuter * kInner>> mat =
      std::make_unique<CompressedArray<MatT, kOuter * kInner>>();
  Compress(content.get(), kOuter * kInner, ws, kOuter * kInner, mat->data(), 0,
           pool);
  // Arbitrary value, different from 1, must match GenerateMatHeap.
  mat->set_scale(0.6f);
  return mat;
}

template <typename MatT, size_t kOuter, size_t kInner>
std::unique_ptr<CompressedArray<MatT, kOuter * kInner>> GenerateZeroMatHeap(
    hwy::ThreadPool& pool) {
  gcpp::CompressWorkingSet ws;
  hwy::AlignedFreeUniquePtr<float[]> content =
      hwy::AllocateAligned<float>(kOuter * kInner);

  pool.Run(0, kOuter, [&](const size_t i, size_t thread) {
    hwy::ZeroBytes(&content[i * kInner], kInner * sizeof(content[0]));
  });

  std::unique_ptr<CompressedArray<MatT, kOuter * kInner>> mat =
      std::make_unique<CompressedArray<MatT, kOuter * kInner>>();
  Compress(content.get(), kOuter * kInner, ws, kOuter * kInner, mat->data(), 0,
           pool);
  mat->set_scale(1.2f);  // Arbitrary value, different from 1.
  return mat;
}

// A simple matrix multiplication. No optimization / tiling.
template <size_t kM, size_t kN, size_t kK>
hwy::AlignedFreeUniquePtr<float[]> SimpleMatMul(
    const hwy::AlignedFreeUniquePtr<float[]>& a,
    const hwy::AlignedFreeUniquePtr<float[]>& b) {
  hwy::AlignedFreeUniquePtr<float[]> out = hwy::AllocateAligned<float>(kM * kK);
  hwy::ZeroBytes(out.get(), kM * kK * sizeof(float));

  int i, j, k;
  for (i = 0; i < kM; ++i) {
    for (j = 0; j < kK; ++j) {
      for (k = 0; k < kN; ++k) {
        out[i * kK + j] += a[i * kN + k] * b[k * kK + j];
      }
    }
  }

  return out;
}

template <typename MatT>
void AssertClose(const MatT* HWY_RESTRICT expected,
                 const MatT* HWY_RESTRICT actual, size_t num) {
  for (size_t idx = 0; idx < num; idx++) {
    const double expected_value = hwy::ConvertScalarTo<double>(expected[idx]);
    const double actual_value = hwy::ConvertScalarTo<double>(actual[idx]);

    const double magnitude = std::abs(expected_value);

    const double tolerance =
        256.0 * hwy::ConvertScalarTo<double>(hwy::Epsilon<MatT>()) *
        HWY_MAX(magnitude, 1.0);

    if (!(expected_value - tolerance <= actual_value &&
          actual_value <= expected_value + tolerance)) {
      fprintf(stderr, "expected[%lu]: %f, actual[%lu]: %f\n", idx,
              expected_value, idx, actual_value);
      HWY_ASSERT(0);
    }
  }
}

template <size_t length>
void AssertClose(const hwy::AlignedFreeUniquePtr<float[]>& a,
                 const hwy::AlignedFreeUniquePtr<float[]>& b) {
  for (size_t idx = 0; idx < length; idx++) {
    const float rel_abs_delta = std::abs(a[idx] - b[idx]) /
                                std::max(std::abs(a[idx]), std::abs(b[idx]));
    EXPECT_LT(rel_abs_delta, 2e-6)
        << "a[" << idx << "]=" << a[idx] << ", b[" << idx << "]=" << b[idx];
  }
}

// Largely unoptimized; reordered innermost loops nets ~5-10X speedup.
template <size_t kN, size_t kK, typename MatTA, typename MatTB,
          HWY_IF_T_SIZE_GT(MatTB, 1)>
HWY_INLINE void MatMulSlowBatch(size_t batch_size, const MatTA* HWY_RESTRICT a,
                                const MatTB* HWY_RESTRICT b, const float scale,
                                const float* add, float* HWY_RESTRICT out) {
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t k = 0; k < kN; ++k) {
      for (size_t j = 0; j < kK; ++j) {
        const float a1 = hwy::ConvertScalarTo<float>(a[i * kN + k]);
        const float b1 = hwy::ConvertScalarTo<float>(b[k * kK + j]);
        out[i * kK + j] += scale * a1 * b1;
      }
    }
    if (add != nullptr) {
      for (size_t j = 0; j < kK; ++j) {
        out[i * kK + j] += add[j];
      }
    }
  }
}

// The above overload can handle combinations of f32 and bf16, but this one
// is required for MatTB = {SFP, NUQ}.
template <size_t kN, size_t kK, typename MatTA, typename MatTB,
          HWY_IF_T_SIZE(MatTB, 1)>
HWY_INLINE void MatMulSlowBatch(size_t batch_size, const MatTA* HWY_RESTRICT a,
                                const MatTB* HWY_RESTRICT b_compr,
                                const float scale, const float* add,
                                float* HWY_RESTRICT out) {
  const hn::ScalableTag<float> d;
  hwy::AlignedFreeUniquePtr<float[]> b = hwy::AllocateAligned<float>(kK * kN);
  CompressTraits<MatTB>::Decompress(d, /*in_capacity=*/0, b_compr, 0, b.get(),
                                    kK * kN);
  MatMulSlowBatch<kN, kK>(batch_size, a, b.get(), scale, add, out);
}

void PrintSpeed(const char* algo, size_t M, size_t N, size_t K,
                double elapsed) {
  // * 2 because of FMA.
  fprintf(stderr, "%s: %f seconds, %f GFLOPS.\n", algo, elapsed,
          2E-9 * M * N * K / elapsed);
}

template <size_t kM, size_t kN, size_t kK, bool kAdd, typename MatTA,
          typename MatTB = MatTA>
void TestMatMul(hwy::ThreadPool& pool) {
  using TraitsA = CompressTraits<MatTA>;
  using TraitsB = CompressTraits<MatTB>;
  fprintf(stderr, "TestMatMul %lu, %lu, %lu, add=%d, MatTA=%s, MatTB=%s\n", kM,
          kN, kK, kAdd, TraitsA::Name(), TraitsB::Name());

  std::unique_ptr<CompressedArray<MatTA, kM * kN>> a =
      GenerateMatHeap<MatTA, kM, kN>(0, pool);
  std::unique_ptr<CompressedArray<MatTB, kN * kK>> b_trans =
      GenerateTransposeMatHeap<MatTB, kN, kK>(0, pool);
  hwy::AlignedFreeUniquePtr<float[]> c = hwy::AllocateAligned<float>(kM * kK);

  const float scale = a->scale() * b_trans->scale();
  std::unique_ptr<CompressedArray<float, kK>> add;
  if (kAdd) {
    add = GenerateMatHeap<float, 1, kK>(0, pool);
    add->set_scale(1.0f);
  }

  std::unique_ptr<CompressedArray<float, kM * kK>> c_slow;
  const bool compare_slow = kN < 2048;
  if (compare_slow) {
    std::unique_ptr<CompressedArray<MatTB, kN * kK>> b =
        GenerateMatHeap<MatTB, kN, kK>(0, pool);
    HWY_ASSERT_EQ(scale, a->scale() * b->scale());
    c_slow = GenerateZeroMatHeap<float, kM, kK>(pool);
    const double start_slow = hwy::platform::Now();
    MatMulSlowBatch<kN, kK>(kM, a->data(), b->data(), scale,
                            kAdd ? add->data() : nullptr, c_slow->data());
    PrintSpeed("MatMulSlowBatch", kM, kN, kK,
               hwy::platform::Now() - start_slow);
  }

  double min_elapsed = hwy::HighestValue<double>();
  for (int rep = 0; rep < (compare_slow ? 1 : 3); ++rep) {
    const double start_tiled = hwy::platform::Now();
    MatMul_4x4<kAdd>(kM, MakeMat(a->data(), kN), MakeMat(b_trans->data(), kN),
                     scale, kAdd ? add->data_scale1() : nullptr,
                     MakeMat(c.get(), kK), pool);
    min_elapsed = HWY_MIN(min_elapsed, hwy::platform::Now() - start_tiled);
  }
  PrintSpeed("MatMul_4x4", kM, kN, kK, min_elapsed);

  if (compare_slow) {
    AssertClose(c_slow->data(), c.get(), kM * kK);
  }
}

void TestAllMatMul() {
  // Skip EMU128 (10x slower than SSE4 for SFP) and older x86.
  if (HWY_TARGET == HWY_EMU128 || HWY_TARGET == HWY_SSE4 ||
      HWY_TARGET == HWY_SSSE3 || HWY_TARGET == HWY_SSE2) {
    return;
  }

  hwy::ThreadPool pool(4);
  using BF16 = hwy::bfloat16_t;
  using F32 = float;
  using SFP = SfpStream;

  // large-scale test
  TestMatMul<64, 24576, 3072, /*kAdd=*/false, F32, SFP>(pool);

  // medium-sized square test
  TestMatMul<512, 512, 512, /*kAdd=*/false, F32>(pool);
  TestMatMul<512, 512, 512, /*kAdd=*/true, BF16>(pool);
  TestMatMul<512, 512, 512, /*kAdd=*/false, F32, BF16>(pool);
  TestMatMul<512, 512, 512, /*kAdd=*/true, BF16, F32>(pool);
  TestMatMul<512, 512, 512, /*kAdd=*/false, F32, SFP>(pool);
  TestMatMul<512, 512, 512, /*kAdd=*/true, BF16, SFP>(pool);

  // minimal non-square test. kK must be at least 2 vectors.
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

template <size_t kOuter, size_t kInner>
hwy::AlignedFreeUniquePtr<float[]> SimpleMatVecAdd(
    const CompressedArray<float, kOuter * kInner>& mat,
    const hwy::AlignedFreeUniquePtr<float[]>& vec,
    const hwy::AlignedFreeUniquePtr<float[]>& add) {
  hwy::AlignedFreeUniquePtr<float[]> uncompressed_mat =
      hwy::AllocateAligned<float>(kOuter * kInner);
  hwy::AlignedFreeUniquePtr<float[]> out = hwy::AllocateAligned<float>(kOuter);
  HWY_ASSERT(uncompressed_mat && out);
  Decompress(mat, 0, uncompressed_mat.get(), kOuter * kInner);
  MulByConst(mat.scale(), uncompressed_mat.get(), kOuter * kInner);
  for (size_t idx_row = 0; idx_row < kOuter; idx_row++) {
    out[idx_row] = add[idx_row];
    for (size_t idx_col = 0; idx_col < kInner; idx_col++) {
      out[idx_row] +=
          uncompressed_mat[kInner * idx_row + idx_col] * vec[idx_col];
    }
  }
  return out;
}

template <typename MatT, size_t kOuter, size_t kInner>
CompressedArray<MatT, kOuter * kInner> GenerateMat(size_t offset,
                                                   hwy::ThreadPool& pool) {
  gcpp::CompressWorkingSet ws;
  CompressedArray<MatT, kOuter * kInner> mat;
  std::array<float, kOuter * kInner> content;
  const float scale = 1.0f / kInner;
  pool.Run(0, kOuter, [&](const size_t i, size_t /*thread*/) {
    for (size_t j = 0; j < kInner; j++) {
      content[i * kInner + j] =
          static_cast<float>((i * kInner + j + offset) * scale);
    }
  });

  Compress(content, ws, mat, pool);
  mat.set_scale(1.9f);  // Arbitrary value, different from 1.
  return mat;
}

template <size_t length>
hwy::AlignedFreeUniquePtr<float[]> GenerateVec(size_t offset) {
  hwy::AlignedFreeUniquePtr<float[]> vec = hwy::AllocateAligned<float>(length);
  HWY_ASSERT(vec);
  for (size_t idx = 0; idx < length; idx++) {
    vec[idx] = static_cast<float>(idx + offset);
  }
  return vec;
}

void TestMatVecAdd() {
  hwy::ThreadPool pool(hwy::ThreadPool::MaxThreads());
  constexpr size_t kOuter = 128 * 3;
  constexpr size_t kInner = 128 * 5;
  CompressedArray<float, kOuter * kInner> mat =
      GenerateMat<float, kOuter, kInner>(0, pool);
  hwy::AlignedFreeUniquePtr<float[]> vec = GenerateVec<kInner>(0);
  hwy::AlignedFreeUniquePtr<float[]> add = GenerateVec<kOuter>(0);
  hwy::AlignedFreeUniquePtr<float[]> even_odd =
      hwy::AllocateAligned<float>(kInner * pool.NumWorkers());
  hwy::AlignedFreeUniquePtr<float[]> expected_out =
      SimpleMatVecAdd<kOuter, kInner>(mat, vec, add);
  hwy::AlignedFreeUniquePtr<float[]> actual_out =
      hwy::AllocateAligned<float>(kOuter);
  HWY_ASSERT(vec && add && even_odd && expected_out && actual_out);
  MatVecAdd<kOuter, kInner>(mat, 0, vec.get(), add.get(), even_odd.get(),
                            actual_out.get(), pool);
  AssertClose<kOuter>(actual_out, expected_out);
}

void TestTwoMatVecAdd() {
  hwy::ThreadPool pool(hwy::ThreadPool::MaxThreads());
  constexpr size_t kOuter = 128 * 3;
  constexpr size_t kInner = 128 * 5;
  CompressedArray<float, kOuter * kInner> mat0 =
      GenerateMat<float, kOuter, kInner>(0, pool);
  CompressedArray<float, kOuter * kInner> mat1 =
      GenerateMat<float, kOuter, kInner>(1, pool);
  hwy::AlignedFreeUniquePtr<float[]> vec = GenerateVec<kInner>(0);
  hwy::AlignedFreeUniquePtr<float[]> add0 = GenerateVec<kOuter>(0);
  hwy::AlignedFreeUniquePtr<float[]> add1 = GenerateVec<kOuter>(1);
  hwy::AlignedFreeUniquePtr<float[]> expected_out0 =
      SimpleMatVecAdd<kOuter, kInner>(mat0, vec, add0);
  hwy::AlignedFreeUniquePtr<float[]> expected_out1 =
      SimpleMatVecAdd<kOuter, kInner>(mat1, vec, add1);
  hwy::AlignedFreeUniquePtr<float[]> actual_out0 =
      hwy::AllocateAligned<float>(kOuter);
  hwy::AlignedFreeUniquePtr<float[]> actual_out1 =
      hwy::AllocateAligned<float>(kOuter);
  HWY_ASSERT(vec && add0 && add1 && expected_out0 && actual_out0 &&
             expected_out1 && actual_out1);
  TwoMatVecAdd<kOuter, kInner>(mat0, mat1, 0, vec.get(), add0.get(), add1.get(),
                               actual_out0.get(), actual_out1.get(), pool);
  AssertClose<kOuter>(actual_out0, expected_out0);
  AssertClose<kOuter>(actual_out1, expected_out1);
}

void TestTwoOfsMatVecAddLoop() {
  hwy::ThreadPool pool(hwy::ThreadPool::MaxThreads());
  constexpr size_t kOuter = 128 * 3;
  constexpr size_t kInner = 128 * 5;
  CompressedArray<float, kOuter * kInner> mat =
      GenerateMat<float, kOuter, kInner>(0, pool);
  hwy::AlignedFreeUniquePtr<float[]> vec = GenerateVec<kInner>(0);
  hwy::AlignedFreeUniquePtr<float[]> add0 = GenerateVec<kOuter>(0);
  hwy::AlignedFreeUniquePtr<float[]> add1 = GenerateVec<kOuter>(1);
  hwy::AlignedFreeUniquePtr<float[]> expected_out0 =
      SimpleMatVecAdd<kOuter, kInner>(mat, vec, add0);
  hwy::AlignedFreeUniquePtr<float[]> expected_out1 =
      SimpleMatVecAdd<kOuter, kInner>(mat, vec, add1);
  hwy::AlignedFreeUniquePtr<float[]> actual_out0 =
      hwy::AllocateAligned<float>(kOuter);
  hwy::AlignedFreeUniquePtr<float[]> actual_out1 =
      hwy::AllocateAligned<float>(kOuter);
  HWY_ASSERT(vec && add0 && add1 && expected_out0 && actual_out0 &&
             expected_out1 && actual_out1);
  TwoOfsMatVecAddLoop<kOuter, kInner>(mat, 0, 0, vec.get(), add0.get(),
                                      add1.get(), actual_out0.get(),
                                      actual_out1.get());
  AssertClose<kOuter>(actual_out0, expected_out0);
  AssertClose<kOuter>(actual_out1, expected_out1);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gcpp {
HWY_BEFORE_TEST(MatmulTest);
HWY_EXPORT_AND_TEST_P(MatmulTest, TestAllMatMul);
HWY_EXPORT_AND_TEST_P(MatmulTest, TestMatVecAdd);
HWY_EXPORT_AND_TEST_P(MatmulTest, TestTwoMatVecAdd);
HWY_EXPORT_AND_TEST_P(MatmulTest, TestTwoOfsMatVecAddLoop);
HWY_AFTER_TEST();

}  // namespace gcpp

#endif
