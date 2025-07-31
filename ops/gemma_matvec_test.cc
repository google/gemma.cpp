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

#include "compression/types.h"
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include <stddef.h>
#include <stdio.h>

#include <algorithm>  // std::max
#include <cmath>      // std::abs
#include <memory>

#include "util/mat.h"
#include "util/threading_context.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ops/gemma_matvec_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"
#include "ops/matvec-inl.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

using FloatPtr = hwy::AlignedFreeUniquePtr<float[]>;

FloatPtr SimpleMatVecAdd(const MatStorageT<float>& mat, const FloatPtr& vec,
                         const FloatPtr& add) {
  const size_t num = mat.Rows() * mat.Cols();
  FloatPtr raw_mat = hwy::AllocateAligned<float>(num);
  FloatPtr out = hwy::AllocateAligned<float>(mat.Rows());
  HWY_ASSERT(raw_mat && out);
  const hn::ScalableTag<float> df;
  DecompressAndZeroPad(df, mat.Span(), 0, raw_mat.get(), num);
  for (size_t idx_row = 0; idx_row < mat.Rows(); idx_row++) {
    out[idx_row] = 0.0f;
    for (size_t idx_col = 0; idx_col < mat.Cols(); idx_col++) {
      out[idx_row] += raw_mat[mat.Cols() * idx_row + idx_col] * vec[idx_col];
    }
    out[idx_row] *= mat.Scale();
    out[idx_row] += add[idx_row];
  }
  return out;
}

template <typename MatT, size_t kOuter, size_t kInner>
std::unique_ptr<MatStorageT<float>> GenerateMat(size_t offset,
                                                const Allocator& allocator,
                                                hwy::ThreadPool& pool) {
  gcpp::CompressWorkingSet ws;
  const Extents2D extents(kOuter, kInner);
  auto mat = std::make_unique<MatStorageT<float>>("TestMat", extents, allocator,
                                                  MatPadding::kPacked);
  FloatPtr raw_mat = hwy::AllocateAligned<float>(extents.Area());
  HWY_ASSERT(raw_mat);
  const float scale = 1.0f / kInner;
  pool.Run(0, kOuter, [&](const size_t i, size_t /*thread*/) {
    for (size_t j = 0; j < kInner; j++) {
      raw_mat[i * kInner + j] =
          static_cast<float>((i * kInner + j + offset) * scale);
    }
  });

  Compress(raw_mat.get(), extents.Area(), ws, mat->Span(), 0, pool);
  mat->SetScale(1.9f);  // Arbitrary value, different from 1.
  return mat;
}

template <size_t length>
FloatPtr GenerateVec(size_t offset) {
  FloatPtr vec = hwy::AllocateAligned<float>(length);
  HWY_ASSERT(vec);
  for (size_t idx = 0; idx < length; idx++) {
    vec[idx] = static_cast<float>(idx + offset);
  }
  return vec;
}

template <size_t length>
void AssertClose(const FloatPtr& a, const FloatPtr& b) {
  for (size_t idx = 0; idx < length; idx++) {
    const float rel_abs_delta = std::abs(a[idx] - b[idx]) /
                                std::max(std::abs(a[idx]), std::abs(b[idx]));
    EXPECT_LT(rel_abs_delta, 2e-6)
        << "a[" << idx << "]=" << a[idx] << ", b[" << idx << "]=" << b[idx];
  }
}

void TestMatVecAdd() {
  ThreadingArgs threading_args;
  ThreadingContext ctx(threading_args);
  hwy::ThreadPool& pool = ctx.pools.Pool();
  constexpr size_t kOuter = 128 * 3;
  constexpr size_t kInner = 128 * 5;
  auto mat = GenerateMat<float, kOuter, kInner>(0, ctx.allocator, pool);
  FloatPtr vec = GenerateVec<kInner>(0);
  FloatPtr add = GenerateVec<kOuter>(0);
  FloatPtr expected_out = SimpleMatVecAdd(*mat, vec, add);
  FloatPtr actual_out = hwy::AllocateAligned<float>(kOuter);
  HWY_ASSERT(vec && add && expected_out && actual_out);
  MatVecAdd(*mat, 0, kOuter, kInner, vec.get(), add.get(), actual_out.get(),
            pool);
  AssertClose<kOuter>(actual_out, expected_out);
}

void TestTwoMatVecAdd() {
  ThreadingArgs threading_args;
  ThreadingContext ctx(threading_args);
  hwy::ThreadPool& pool = ctx.pools.Pool();
  constexpr size_t kOuter = 128 * 3;
  constexpr size_t kInner = 128 * 5;
  auto mat0 = GenerateMat<float, kOuter, kInner>(0, ctx.allocator, pool);
  auto mat1 = GenerateMat<float, kOuter, kInner>(1, ctx.allocator, pool);
  FloatPtr vec = GenerateVec<kInner>(0);
  FloatPtr add0 = GenerateVec<kOuter>(0);
  FloatPtr add1 = GenerateVec<kOuter>(1);
  FloatPtr expected_out0 = SimpleMatVecAdd(*mat0, vec, add0);
  FloatPtr expected_out1 = SimpleMatVecAdd(*mat1, vec, add1);
  FloatPtr actual_out0 = hwy::AllocateAligned<float>(kOuter);
  FloatPtr actual_out1 = hwy::AllocateAligned<float>(kOuter);
  HWY_ASSERT(vec && add0 && add1 && expected_out0 && actual_out0 &&
             expected_out1 && actual_out1);
  TwoMatVecAdd(*mat0, *mat1, 0, kOuter, kInner, vec.get(), add0.get(),
               add1.get(), actual_out0.get(), actual_out1.get(), pool);
  AssertClose<kOuter>(actual_out0, expected_out0);
  AssertClose<kOuter>(actual_out1, expected_out1);
}

void TestTwoOfsMatVecAddLoop() {
  ThreadingArgs threading_args;
  ThreadingContext ctx(threading_args);
  hwy::ThreadPool& pool = ctx.pools.Pool();

  constexpr size_t kOuter = 128 * 3;
  constexpr size_t kInner = 128 * 5;
  auto mat = GenerateMat<float, kOuter, kInner>(0, ctx.allocator, pool);
  FloatPtr vec = GenerateVec<kInner>(0);
  FloatPtr add0 = GenerateVec<kOuter>(0);
  FloatPtr add1 = GenerateVec<kOuter>(1);
  FloatPtr expected_out0 = SimpleMatVecAdd(*mat, vec, add0);
  FloatPtr expected_out1 = SimpleMatVecAdd(*mat, vec, add1);
  FloatPtr actual_out0 = hwy::AllocateAligned<float>(kOuter);
  FloatPtr actual_out1 = hwy::AllocateAligned<float>(kOuter);
  HWY_ASSERT(vec && add0 && add1 && expected_out0 && actual_out0 &&
             expected_out1 && actual_out1);
  TwoOfsMatVecAddLoop(*mat, 0, 0, kOuter, kInner, vec.get(), add0.get(),
                      add1.get(), actual_out0.get(), actual_out1.get());
  AssertClose<kOuter>(actual_out0, expected_out0);
  AssertClose<kOuter>(actual_out1, expected_out1);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gcpp {
HWY_BEFORE_TEST(MatVecTest);
HWY_EXPORT_AND_TEST_P(MatVecTest, TestMatVecAdd);
HWY_EXPORT_AND_TEST_P(MatVecTest, TestTwoMatVecAdd);
HWY_EXPORT_AND_TEST_P(MatVecTest, TestTwoOfsMatVecAddLoop);
HWY_AFTER_TEST();

}  // namespace gcpp

#endif
