// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Include guard for headers.
#ifndef THIRD_PARTY_GEMMA_CPP_COMPRESSION_TEST_UTIL_INL_H_
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_TEST_UTIL_INL_H_

#include <stddef.h>

#include <vector>

// IWYU pragma: begin_exports
#include "compression/distortion.h"
#include "util/mat.h"
// IWYU pragma: end_exports

#include "compression/compress.h"
#include "util/threading_context.h"

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_TEST_UTIL_INL_H_

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_COMPRESS_TEST_UTIL_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef THIRD_PARTY_GEMMA_CPP_COMPRESS_TEST_UTIL_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_COMPRESS_TEST_UTIL_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_COMPRESS_TEST_UTIL_TOGGLE
#endif

#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"
#include "hwy/tests/test_util-inl.h"  // IWYU pragma: export

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// `Packed` is the type passed to `TestT`.
template <typename Packed, template <class> class TestT>
void ForeachRawType() {
  const hn::ForGEVectors<128, TestT<Packed>> test;
  // The argument selects the type to decode to: BF16 or float.
  test(BF16());
  test(float());
  // Do not include double because it is not supported as an input type - we
  // would also have to implement double -> Packed Compress().
}

template <template <class> class TestT>
void ForeachPackedAndRawType() {
  ForeachRawType<BF16, TestT>();
  ForeachRawType<float, TestT>();
  ForeachRawType<SfpStream, TestT>();
  if constexpr (GEMMA_ENABLE_NUQ) {
    ForeachRawType<NuqStream, TestT>();
  }
}

template <class Test, class D>
void ForeachActivationType1(D d) {
  Test test;
  test(float(), d);
  test(BF16(), d);
}

template <class Test, class D>
void ForeachActivationType2(D d) {
  Test test;
  test(float(), float(), d);
  test(float(), BF16(), d);
  test(BF16(), float(), d);
  test(BF16(), BF16(), d);
}

template <class Test, class D>
void ForeachActivationType3(D d) {
  Test test;
  test(float(), float(), float(), d);
  test(float(), float(), BF16(), d);
  test(float(), BF16(), float(), d);
  test(float(), BF16(), BF16(), d);
  test(BF16(), float(), float(), d);
  test(BF16(), float(), BF16(), d);
  test(BF16(), BF16(), float(), d);
  test(BF16(), BF16(), BF16(), d);
}

// Generates inputs: deterministic, within max SfpStream range.
template <typename MatT>
MatStorageT<MatT> GenerateMat(const Extents2D& extents, MatPadding padding,
                              ThreadingContext& ctx) {
  gcpp::CompressWorkingSet ws;
  ws.tls.resize(ctx.pools.MaxWorkers());
  MatStorageT<float> raw("raw", extents, ctx.allocator, MatPadding::kPacked);
  MatStorageT<MatT> compressed("mat", extents, ctx.allocator, padding);
  const float scale = SfpStream::kMax / extents.Area();
  ParallelFor(Parallelism::kFlat, extents.rows, ctx, /*cluster_idx=*/0,
              Callers::kTest, [&](size_t r, size_t thread) {
                float* HWY_RESTRICT row = raw.Row(r);
                for (size_t c = 0; c < extents.cols; c++) {
                  float f = static_cast<float>(r * extents.cols + c) * scale;
                  if ((r + c) & 1)
                    f = -f;  // Also generate some negative values.
                  row[c] = f;
                }
                Compress(raw.Row(r), raw.Cols(), ws.tls[thread],
                         MakeSpan(compressed.Row(r), extents.cols),
                         /*packed_ofs=*/0);
              });

  compressed.SetScale(0.6f);  // Arbitrary value, different from 1.
  return compressed;
}

// Same, but `extents` describes the transposed matrix and the computation of
// `f` swaps `r` and `c`.
template <typename MatT>
MatStorageT<MatT> GenerateTransposedMat(const Extents2D extents,
                                        MatPadding padding,
                                        ThreadingContext& ctx) {
  gcpp::CompressWorkingSet ws;
  ws.tls.resize(ctx.pools.MaxWorkers());
  MatStorageT<float> raw("raw", extents, ctx.allocator, MatPadding::kPacked);
  MatStorageT<MatT> compressed("trans", extents, ctx.allocator, padding);
  const float scale = SfpStream::kMax / extents.Area();
  ParallelFor(Parallelism::kFlat, extents.rows, ctx, /*cluster_idx=*/0,
              Callers::kTest, [&](size_t r, size_t thread) {
                float* HWY_RESTRICT row = raw.Row(r);
                for (size_t c = 0; c < extents.cols; c++) {
                  float f = static_cast<float>(c * extents.rows + r) * scale;
                  if ((r + c) & 1)
                    f = -f;  // Also generate some negative values.
                  row[c] = f;
                }
                Compress(raw.Row(r), raw.Cols(), ws.tls[thread],
                         MakeSpan(compressed.Row(r), extents.cols),
                         /*packed_ofs=*/0);
              });

  // Arbitrary value, different from 1, must match `GenerateMat`.
  compressed.SetScale(0.6f);
  return compressed;
}

// Returns 1-norm, used for estimating tolerable numerical differences.
inline double MaxRowAbsSum(const MatStorageT<float>& a) {
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
inline float MaxAbs(const MatStorageT<float>& a) {
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
                 const Allocator& allocator,
                 std::vector<hwy::AlignedFreeUniquePtr<uint8_t*[]>>& row_ptrs,
                 int line) {
  const hn::ScalableTag<float> df;
  const size_t cols = A.Cols();
  const size_t B_rows = B.Rows();
  // Round up for DecompressAndZeroPad.
  MatStorageT<float> a_batch("a_batch", A.Extents(), allocator,
                             MatPadding::kOdd);
  MatStorageT<float> b_trans_batch("b_trans_batch", B.Extents(), allocator,
                                   MatPadding::kOdd);
  MatStorageT<float> c_batch("c_batch", Extents2D(A.Rows(), B_rows), allocator,
                             MatPadding::kOdd);
  c_batch.AllocateAndAttachRowPtrs(row_ptrs);
  MatStorageT<float> c_slow_batch("c_slow_batch", Extents2D(A.Rows(), B_rows),
                                  allocator, MatPadding::kOdd);
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
  // Dot() uses double-precision summation.
  double tolerance = 20 * norm * eps_f32;
  // If either is F32, Dot() promotes F32 or even F64, but MatMul demotes the
  // F32 to BF16, so add extra tolerance.
  if (IsF32<TA>() || IsF32<TB>()) {
    tolerance += 2 * max_abs * eps_bf16;
  }

  if (tolerance > 500.0) {
    HWY_WARN("high tolerance %f norm %f maxabs %f\n", tolerance, norm, max_abs);
  }
  const double rel_tolerance =
      1.0 + hwy::ConvertScalarTo<double>(hwy::Epsilon<TC>());

  double max_rel = 0.0;
  size_t worst_r = 0;
  size_t worst_c = 0;
  double worst_actual = 0.0;
  double worst_expected = 0.0;
  size_t num_outside = 0;
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
          worst_expected = expected_value;
          worst_actual = actual_value;
          worst_r = r;
          worst_c = c;
          max_rel = rel;
          ++num_outside;
        }
      }
    }
  }

  if (max_rel > rel_tolerance) {
    hwy::Abort(__FILE__, line,
               "(%zu,%zu): expected %f, actual %f, norm %f maxabs %f "
               "tolerance %f rel %E max_rel %E num_outside %zu\n",
               worst_r, worst_c, worst_expected, worst_actual, norm, max_abs,
               tolerance, max_rel, rel_tolerance, num_outside);
  }
  HWY_ASSERT(hn::AllFalse(
      df, hn::IsEitherNaN(hn::Set(df, norm), hn::Set(df, max_abs))));
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
