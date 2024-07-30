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

// Include guard for non-SIMD code.
#ifndef THIRD_PARTY_GEMMA_CPP_OPS_MATVEC_INL_H_
#define THIRD_PARTY_GEMMA_CPP_OPS_MATVEC_INL_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "compression/compress.h"
#include "compression/sfp.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"

#endif  // THIRD_PARTY_GEMMA_CPP_OPS_MATVEC_INL_H_

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_MATVEC_TOGGLE) == defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_MATVEC_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_MATVEC_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_MATVEC_TOGGLE
#endif

#include "compression/compress-inl.h"
#include "hwy/contrib/dot/dot-inl.h"
#include "hwy/contrib/math/math-inl.h"
#include "hwy/contrib/matvec/matvec-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

HWY_INLINE void ToEvenOddF32(const hwy::bfloat16_t* HWY_RESTRICT vec_aligned,
                             const size_t size, float* HWY_RESTRICT out) {
  const hn::ScalableTag<float> df;
  const hn::Repartition<hwy::bfloat16_t, decltype(df)> dbf16;

  HWY_DASSERT(size % hn::Lanes(dbf16) == 0);
  HWY_DASSERT(hn::IsAligned(df, vec_aligned));

  for (size_t i = 0; i < size; i += hn::Lanes(dbf16)) {
    const auto interleaved = hn::LoadU(dbf16, vec_aligned + i);
    hn::Store(hn::PromoteEvenTo(df, interleaved), df, out + i);
    hn::Store(hn::PromoteOddTo(df, interleaved), df, out + i + hn::Lanes(df));
  }
}

HWY_INLINE void ToEvenOddF32(const float* HWY_RESTRICT vec_aligned,
                             const size_t size, float* HWY_RESTRICT out) {
  const hn::ScalableTag<float> df;
  using VF = hn::Vec<decltype(df)>;

  HWY_DASSERT(size % (hn::Lanes(df) * 2) == 0);
  HWY_DASSERT(hn::IsAligned(df, vec_aligned));

  VF vec0, vec1;
  for (size_t i = 0; i < size; i += hn::Lanes(df) * 2) {
    hn::LoadInterleaved2(df, vec_aligned + i, vec0, vec1);
    hn::Store(vec0, df, out + i);
    hn::Store(vec1, df, out + i + hn::Lanes(df));
  }
}

// Simple version without tiling nor threading, but two offsets/outputs and
// always with addition.
template <size_t kOuter, size_t kInner, typename ArrayT, typename VecT,
          typename AddT>
HWY_INLINE void TwoOfsMatVecAddLoop(const ArrayT& mat, const size_t mat_ofs0,
                                    const size_t mat_ofs1,
                                    const VecT* HWY_RESTRICT vec_aligned,
                                    const AddT* HWY_RESTRICT add0,
                                    const AddT* HWY_RESTRICT add1,
                                    float* HWY_RESTRICT out0,
                                    float* HWY_RESTRICT out1) {
  PROFILER_ZONE("TwoOfsMatVecAddLoop");
  constexpr bool kVecEO = false;
  const hn::ScalableTag<float> df;

  for (size_t idx_row = 0; idx_row < kOuter; ++idx_row) {
    const size_t row_ofs0 = mat_ofs0 + (idx_row)*kInner;
    const size_t row_ofs1 = mat_ofs1 + (idx_row)*kInner;
    out0[idx_row] = hwy::ConvertScalarTo<float>(add0[idx_row]) +
                    Dot<kVecEO>(df, mat, row_ofs0, vec_aligned, kInner);
    out1[idx_row] = hwy::ConvertScalarTo<float>(add1[idx_row]) +
                    Dot<kVecEO>(df, mat, row_ofs1, vec_aligned, kInner);
  }
}

HWY_INLINE constexpr size_t MaxCols() {
  // Vec + mat rows should fit into 32 KiB L1.
  return 2048;
}

template <size_t kOuter>
HWY_INLINE constexpr size_t RowsPerStrip() {
  // Aim for 128 work items to reduce pool overhead. Must be at least one
  // vector; prefer a power of two for faster division.
  constexpr size_t kLanes = hn::ScalableTag<float>().MaxLanes();
  constexpr size_t kRowsPerStrip =
      kOuter < 128 ? kLanes
                   : HWY_MAX(kLanes, 1ULL << hwy::FloorLog2(kOuter / 128));
  return kRowsPerStrip;
}

namespace detail {

// For each i = [0, num_rows), compute partial (length `num_cols`) dot product
// of row i with `vec_aligned` and add into `out[i]`. The upper-left
// coordinate of the tile is r0, c0.
template <bool kVecEO, class DF, typename ArrayT, typename VecT>
HWY_INLINE void AccumulatePartialDotProducts(
    DF df, const ArrayT& mat, size_t mat_ofs, size_t mat_stride, size_t r0,
    size_t c0, size_t num_rows, size_t num_cols,
    const VecT* HWY_RESTRICT vec_aligned, float* HWY_RESTRICT out) {
  for (size_t idx_row = 0; idx_row < num_rows; ++idx_row) {
    const size_t row_ofs = mat_ofs + (r0 + idx_row) * mat_stride;
    out[idx_row] +=
        Dot<kVecEO>(df, mat, row_ofs + c0, vec_aligned + c0, num_cols);
  }
}

// Same as AccumulatePartialDotProducts, but sets out[i] to the first partial
// dot product + init (if kInit), which avoids having to zero-initialize and
// accumulate.
template <bool kVecEO, bool kInit, class DF, typename ArrayT, typename VecT,
          typename InitT>
HWY_INLINE void SetFirstPartialDotProducts(DF df, const ArrayT& mat,
                                           size_t mat_ofs, size_t mat_stride,
                                           size_t r0, size_t c0,
                                           size_t num_rows, size_t num_cols,
                                           const VecT* HWY_RESTRICT vec_aligned,
                                           const InitT* HWY_RESTRICT init,
                                           float* HWY_RESTRICT out) {
  for (size_t idx_row = 0; idx_row < num_rows; ++idx_row) {
    const size_t row_ofs = mat_ofs + (r0 + idx_row) * mat_stride;
    if constexpr (kInit) {
      out[idx_row] =
          hwy::ConvertScalarTo<float>(init[idx_row + r0]) +
          Dot<kVecEO>(df, mat, row_ofs + c0, vec_aligned + c0, num_cols);
    } else {
      out[idx_row] =
          Dot<kVecEO>(df, mat, row_ofs + c0, vec_aligned + c0, num_cols);
    }
  }
}

// Adds together partial dot products for all tiles with the same r0 (a
// horizontal strip of the entire matrix); the result is the full dot product
// for rows r in [r0, r0 + num_rows) + optionally the add vector, which we
// store into in out[r - r0].
template <bool kVecEO, bool kAdd, class DF, typename ArrayT, typename VecT,
          typename AddT>
HWY_INLINE void FullDotProductsForStrip(DF df, const ArrayT& mat,
                                        size_t mat_ofs, size_t mat_stride,
                                        size_t r0, size_t num_rows,
                                        const VecT* HWY_RESTRICT vec_aligned,
                                        const AddT* HWY_RESTRICT add,
                                        float* HWY_RESTRICT out) {
  // Tall and skinny: set `out` to the single dot product.
  if (mat_stride < MaxCols()) {
    SetFirstPartialDotProducts<kVecEO, kAdd>(df, mat, mat_ofs, mat_stride, r0,
                                             0, num_rows, mat_stride,
                                             vec_aligned, add, out);
    return;
  }

  // We have at least MaxCols, so start by setting `out` to that:
  SetFirstPartialDotProducts<kVecEO, kAdd>(df, mat, mat_ofs, mat_stride, r0, 0,
                                           num_rows, MaxCols(), vec_aligned,
                                           add, out);
  // For further multiples of MaxCols, accumulate. Remainders handled below.
  size_t c0 = MaxCols();
  for (; c0 <= mat_stride - MaxCols(); c0 += MaxCols()) {
    AccumulatePartialDotProducts<kVecEO>(df, mat, mat_ofs, mat_stride, r0, c0,
                                         num_rows, MaxCols(), vec_aligned, out);
  }

  if (c0 < mat_stride) {  // Final cols
    AccumulatePartialDotProducts<kVecEO>(df, mat, mat_ofs, mat_stride, r0, c0,
                                         num_rows, mat_stride - c0, vec_aligned,
                                         out);
  }
}

template <bool kVecIsEvenOdd, bool kAdd, size_t kOuter, size_t kInner,
          typename ArrayT, typename VecT, typename AddT>
HWY_INLINE void MatVecAddInner(const ArrayT& mat, const size_t mat_ofs,
                               const VecT* HWY_RESTRICT const vec_aligned,
                               const AddT* HWY_RESTRICT const add,
                               float* HWY_RESTRICT out, hwy::ThreadPool& pool) {
  const hn::ScalableTag<float> df;
  constexpr size_t kRowsPerStrip = RowsPerStrip<kOuter>();
  constexpr size_t kNumStrips = kOuter / kRowsPerStrip;

  // For each entire strip.
  pool.Run(0, kNumStrips, [&](const uint64_t strip, size_t thread) HWY_ATTR {
    PROFILER_ZONE("MatVec.lambda");
    const size_t r0 = strip * kRowsPerStrip;
    detail::FullDotProductsForStrip<kVecIsEvenOdd, kAdd>(
        df, mat, mat_ofs, kInner, r0, kRowsPerStrip, vec_aligned, add,
        out + r0);
  });

  // Remaining rows
  const size_t r0 = kNumStrips * kRowsPerStrip;
  if (r0 < kOuter) {
    PROFILER_ZONE("MatVec remainder");
    const size_t num_rows = kOuter - r0;
    detail::FullDotProductsForStrip<kVecIsEvenOdd, kAdd>(
        df, mat, mat_ofs, kInner, r0, num_rows, vec_aligned, add, out + r0);
  }
}

}  // namespace detail

// Stores dot products of rows with `vec_aligned` + add the values from `add`
// (if kAdd), then stores them to `out`.
template <bool kAdd, size_t kOuter, size_t kInner, typename ArrayT,
          typename VecT, typename AddT>
HWY_INLINE void MatVecT(const ArrayT& mat, const size_t mat_ofs,
                        const VecT* HWY_RESTRICT const vec_aligned,
                        const AddT* HWY_RESTRICT const add,
                        float* HWY_RESTRICT even_odd, float* HWY_RESTRICT out,
                        hwy::ThreadPool& pool) {
  PROFILER_ZONE("MatVecAdd");

#if !defined(HWY_NATIVE_DOT_BF16) || !HWY_NATIVE_DOT_BF16
  using MatT = typename ArrayT::value_type;
  // Sfp -> float does not benefit enough to recoup the cost of ToEvenOddF32.
  if constexpr (CompressTraits<MatT>::kSupportsEvenOdd &&
                hwy::IsSameEither<VecT, float, hwy::bfloat16_t>() &&
                !(hwy::IsSame<MatT, SfpStream>() &&
                  hwy::IsSame<VecT, float>())) {
    ToEvenOddF32(vec_aligned, kInner, even_odd);
    detail::MatVecAddInner</*kVecIsEvenOdd=*/true, kAdd, kOuter, kInner>(
        mat, mat_ofs, even_odd, add, out, pool);
    return;
  }
#else
  (void)even_odd;
#endif

  detail::MatVecAddInner</*kVecIsEvenOdd=*/false, kAdd, kOuter, kInner>(
      mat, mat_ofs, vec_aligned, add, out, pool);
}

// With addition
template <size_t kOuter, size_t kInner, typename ArrayT, typename VecT,
          typename AddT>
HWY_INLINE void MatVecAdd(const ArrayT& mat, const size_t mat_ofs,
                          const VecT* HWY_RESTRICT const vec_aligned,
                          const AddT* HWY_RESTRICT const add,
                          float* HWY_RESTRICT even_odd, float* HWY_RESTRICT out,
                          hwy::ThreadPool& pool) {
  return MatVecT</*kAdd=*/true, kOuter, kInner>(mat, mat_ofs, vec_aligned, add,
                                                even_odd, out, pool);
}

// Without addition
template <size_t kOuter, size_t kInner, typename ArrayT, typename VecT>
HWY_INLINE void MatVec(const ArrayT& mat, const size_t mat_ofs,
                       const VecT* HWY_RESTRICT const vec_aligned,
                       float* HWY_RESTRICT even_odd, float* HWY_RESTRICT out,
                       hwy::ThreadPool& pool) {
  MatVecT</*kAdd=*/false, kOuter, kInner>(mat, mat_ofs, vec_aligned,
                                          /*add=*/static_cast<VecT*>(nullptr),
                                          even_odd, out, pool);
}

// Two matrices, same vector
template <bool kAdd, size_t kOuter, size_t kInner, typename ArrayT,
          typename VecT, typename AddT>
HWY_NOINLINE void TwoMatVecT(const ArrayT& mat0, const ArrayT& mat1,
                             const size_t mat_ofs,
                             const VecT* HWY_RESTRICT vec_aligned,
                             const AddT* HWY_RESTRICT add0,
                             const AddT* HWY_RESTRICT add1,
                             float* HWY_RESTRICT out0, float* HWY_RESTRICT out1,
                             hwy::ThreadPool& pool) {
  PROFILER_ZONE("TwoMatVecAdd");

  const hn::ScalableTag<float> df;
  constexpr size_t kRowsPerStrip = RowsPerStrip<kOuter>();
  constexpr size_t kNumStrips = kOuter / kRowsPerStrip;
  constexpr bool kVecIsEvenOdd = false;

  // For each entire strip.
  pool.Run(0, kNumStrips, [&](const uint64_t strip, size_t thread) HWY_ATTR {
    PROFILER_ZONE("TwoMatVec.lambda");
    const size_t r0 = strip * kRowsPerStrip;
    detail::FullDotProductsForStrip<kVecIsEvenOdd, kAdd>(
        df, mat0, mat_ofs, kInner, r0, kRowsPerStrip, vec_aligned, add0,
        out0 + r0);
    detail::FullDotProductsForStrip<kVecIsEvenOdd, kAdd>(
        df, mat1, mat_ofs, kInner, r0, kRowsPerStrip, vec_aligned, add1,
        out1 + r0);
  });

  // Remaining rows
  const size_t r0 = kNumStrips * kRowsPerStrip;
  if (r0 < kOuter) {
    PROFILER_ZONE("TwoMatVec remainder");
    const size_t num_rows = kOuter - r0;
    detail::FullDotProductsForStrip<kVecIsEvenOdd, kAdd>(
        df, mat0, mat_ofs, kInner, r0, num_rows, vec_aligned, add0, out0 + r0);
    detail::FullDotProductsForStrip<kVecIsEvenOdd, kAdd>(
        df, mat1, mat_ofs, kInner, r0, num_rows, vec_aligned, add1, out1 + r0);
  }
}

// With addition
template <size_t kOuter, size_t kInner, typename ArrayT, typename VecT,
          typename AddT>
HWY_NOINLINE void TwoMatVecAdd(
    const ArrayT& mat0, const ArrayT& mat1, const size_t mat_ofs,
    const VecT* HWY_RESTRICT vec_aligned, const AddT* HWY_RESTRICT add0,
    const AddT* HWY_RESTRICT add1, float* HWY_RESTRICT out0,
    float* HWY_RESTRICT out1, hwy::ThreadPool& pool) {
  return TwoMatVecT</*kAdd=*/true, kOuter, kInner>(
      mat0, mat1, mat_ofs, vec_aligned, add0, add1, out0, out1, pool);
}

// Without addition
template <size_t kOuter, size_t kInner, typename ArrayT, typename VecT>
HWY_NOINLINE void TwoMatVec(const ArrayT& mat0, const ArrayT& mat1,
                            const size_t mat_ofs,
                            const VecT* HWY_RESTRICT vec_aligned,
                            float* HWY_RESTRICT out0, float* HWY_RESTRICT out1,
                            hwy::ThreadPool& pool) {
  TwoMatVecT</*kAdd=*/false, kOuter, kInner, ArrayT, VecT, VecT>(
      mat0, mat1, mat_ofs, vec_aligned, /*add0=*/nullptr, /*add1=*/nullptr,
      out0, out1, pool);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
