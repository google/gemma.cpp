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
#include "ops/dot-inl.h"
#include "ops/matmul.h"
#include "util/mat.h"  // MatPtrT
#include "hwy/contrib/math/math-inl.h"
#include "hwy/contrib/matvec/matvec-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// For callers that pass `MatPtrT`, which is not necessarily packed - callers
// should use Stride() to compute `w_ofs`.
template <typename WT, typename VT>
HWY_INLINE float Dot(const MatPtrT<WT>& w, size_t w_ofs, const VT* vec_aligned,
                     size_t num) {
  const hn::ScalableTag<VT> d;
  return w.Scale() * Dot(d, w.PaddedSpan(), w_ofs, vec_aligned, num);
}

// ArrayT is MatPtrT.

// Simple version without tiling nor threading, but two offsets/outputs and
// always with addition.
template <typename ArrayT, typename VecT, typename AddT>
HWY_INLINE void TwoOfsMatVecAddLoop(const ArrayT& mat, const size_t mat_ofs0,
                                    const size_t mat_ofs1, const size_t outer,
                                    const size_t inner,
                                    const VecT* HWY_RESTRICT vec_aligned,
                                    const AddT* HWY_RESTRICT add0,
                                    const AddT* HWY_RESTRICT add1,
                                    float* HWY_RESTRICT out0,
                                    float* HWY_RESTRICT out1) {
  PROFILER_ZONE("TwoOfsMatVecAddLoop");

  for (size_t idx_row = 0; idx_row < outer; ++idx_row) {
    const size_t row_ofs0 = mat_ofs0 + idx_row * mat.Stride();
    const size_t row_ofs1 = mat_ofs1 + idx_row * mat.Stride();
    out0[idx_row] = hwy::ConvertScalarTo<float>(add0[idx_row]) +
                    Dot(mat, row_ofs0, vec_aligned, inner);
    out1[idx_row] = hwy::ConvertScalarTo<float>(add1[idx_row]) +
                    Dot(mat, row_ofs1, vec_aligned, inner);
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

HWY_INLINE size_t RowsPerStrip(const size_t outer) {
  // Aim for 128 work items to reduce pool overhead. Must be at least one
  // vector; prefer a power of two for faster division.
  constexpr size_t kLanes = hn::ScalableTag<float>().MaxLanes();
  return outer < 128 ? kLanes
                     : HWY_MAX(kLanes, 1ULL << hwy::FloorLog2(outer / 128));
}

namespace detail {

// For each i = [0, num_rows), compute partial (length `num_cols`) dot product
// of row i with `vec_aligned` and add into `out[i]`. The upper-left
// coordinate of the tile is r0, c0.
template <class DF, typename ArrayT, typename VecT>
HWY_INLINE void AccumulatePartialDotProducts(
    DF df, const ArrayT& mat, size_t mat_ofs, size_t r0, size_t c0,
    size_t num_rows, size_t num_cols, const VecT* HWY_RESTRICT vec_aligned,
    float* HWY_RESTRICT out) {
  for (size_t idx_row = 0; idx_row < num_rows; ++idx_row) {
    const size_t row_ofs = mat_ofs + (r0 + idx_row) * mat.Stride();
    out[idx_row] += Dot(mat, row_ofs + c0, vec_aligned + c0, num_cols);
  }
}

// Same as AccumulatePartialDotProducts, but sets out[i] to the first partial
// dot product + init (if kInit), which avoids having to zero-initialize and
// accumulate.
template <bool kInit, class DF, typename ArrayT, typename VecT, typename InitT>
HWY_INLINE void SetFirstPartialDotProducts(DF df, const ArrayT& mat,
                                           size_t mat_ofs, size_t r0, size_t c0,
                                           size_t num_rows, size_t num_cols,
                                           const VecT* HWY_RESTRICT vec_aligned,
                                           const InitT* HWY_RESTRICT init,
                                           float* HWY_RESTRICT out) {
  for (size_t idx_row = 0; idx_row < num_rows; ++idx_row) {
    const size_t row_ofs = mat_ofs + (r0 + idx_row) * mat.Stride();
    if constexpr (kInit) {
      out[idx_row] = hwy::ConvertScalarTo<float>(init[idx_row + r0]) +
                     Dot(mat, row_ofs + c0, vec_aligned + c0, num_cols);
    } else {
      out[idx_row] = Dot(mat, row_ofs + c0, vec_aligned + c0, num_cols);
    }
  }
}

// Adds together partial dot products for all tiles with the same r0 (a
// horizontal strip of the entire matrix); the result is the full dot product
// for rows r in [r0, r0 + num_rows) + optionally the add vector, which we
// store into in out[r - r0].
template <bool kAdd, class DF, typename ArrayT, typename VecT, typename AddT>
HWY_INLINE void FullDotProductsForStrip(DF df, const ArrayT& mat,
                                        size_t mat_ofs, size_t r0,
                                        size_t num_rows, size_t num_cols,
                                        const VecT* HWY_RESTRICT vec_aligned,
                                        const AddT* HWY_RESTRICT add,
                                        float* HWY_RESTRICT out) {
  HWY_DASSERT(num_cols <= mat.Cols());
  // Tall and skinny: set `out` to the single dot product.
  if (num_cols < MaxCols()) {
    SetFirstPartialDotProducts<kAdd>(df, mat, mat_ofs, r0, 0, num_rows,
                                     num_cols, vec_aligned, add, out);
    return;
  }

  // We have at least MaxCols, so start by setting `out` to that:
  SetFirstPartialDotProducts<kAdd>(df, mat, mat_ofs, r0, 0, num_rows, MaxCols(),
                                   vec_aligned, add, out);
  // For further multiples of MaxCols, accumulate. Remainders handled below.
  size_t c0 = MaxCols();
  for (; c0 <= num_cols - MaxCols(); c0 += MaxCols()) {
    AccumulatePartialDotProducts(df, mat, mat_ofs, r0, c0, num_rows, MaxCols(),
                                 vec_aligned, out);
  }

  if (c0 < num_cols) {  // Final cols
    AccumulatePartialDotProducts(df, mat, mat_ofs, r0, c0, num_rows,
                                 num_cols - c0, vec_aligned, out);
  }
}

}  // namespace detail

// Stores dot products of rows with `vec_aligned` + add the values from `add`
// (if kAdd), then stores them to `out`.
template <bool kAdd, typename ArrayT, typename VecT, typename AddT>
HWY_INLINE void MatVecT(const ArrayT& mat, const size_t mat_ofs,
                        const size_t outer, const size_t inner,
                        const VecT* HWY_RESTRICT const vec_aligned,
                        const AddT* HWY_RESTRICT const add,
                        float* HWY_RESTRICT out, hwy::ThreadPool& pool) {
  PROFILER_ZONE("MatVecAdd");

  const hn::ScalableTag<float> df;
  const size_t rows_per_strip = RowsPerStrip(outer);
  const size_t num_strips = outer / rows_per_strip;

  // For each entire strip.
  pool.Run(0, num_strips, [&](const uint64_t strip, size_t thread) HWY_ATTR {
    PROFILER_ZONE("MatVec.lambda");
    const size_t r0 = strip * rows_per_strip;
    detail::FullDotProductsForStrip<kAdd>(df, mat, mat_ofs, r0, rows_per_strip,
                                          inner, vec_aligned, add, out + r0);
  });

  // Remaining rows
  const size_t r0 = num_strips * rows_per_strip;
  if (r0 < outer) {
    PROFILER_ZONE("MatVec remainder");
    const size_t num_rows = outer - r0;
    detail::FullDotProductsForStrip<kAdd>(df, mat, mat_ofs, r0, num_rows, inner,
                                          vec_aligned, add, out + r0);
  }
}

// With addition
template <typename ArrayT, typename VecT, typename AddT>
HWY_INLINE void MatVecAdd(const ArrayT& mat, const size_t mat_ofs,
                          const size_t outer, const size_t inner,
                          const VecT* HWY_RESTRICT const vec_aligned,
                          const AddT* HWY_RESTRICT const add,
                          float* HWY_RESTRICT out, hwy::ThreadPool& pool) {
  return MatVecT</*kAdd=*/true>(mat, mat_ofs, outer, inner, vec_aligned, add,
                                out, pool);
}

// Without addition
template <typename ArrayT, typename VecT>
HWY_INLINE void MatVec(const ArrayT& mat, const size_t mat_ofs,
                       const size_t outer, const size_t inner,
                       const VecT* HWY_RESTRICT const vec_aligned,
                       float* HWY_RESTRICT out, hwy::ThreadPool& pool) {
  MatVecT</*kAdd=*/false>(mat, mat_ofs, outer, inner, vec_aligned,
                          /*add=*/static_cast<VecT*>(nullptr), out, pool);
}

// Two matrices, same vector
template <bool kAdd, typename ArrayT1, typename ArrayT2, typename VecT,
          typename AddT>
HWY_NOINLINE void TwoMatVecT(const ArrayT1& mat0, const ArrayT2& mat1,
                             const size_t mat_ofs, size_t outer, size_t inner,
                             const VecT* HWY_RESTRICT vec_aligned,
                             const AddT* HWY_RESTRICT add0,
                             const AddT* HWY_RESTRICT add1,
                             float* HWY_RESTRICT out0, float* HWY_RESTRICT out1,
                             hwy::ThreadPool& pool) {
  PROFILER_ZONE("TwoMatVecAdd");

  const hn::ScalableTag<float> df;
  const size_t rows_per_strip = RowsPerStrip(outer);
  const size_t num_strips = outer / rows_per_strip;

  // For each entire strip.
  pool.Run(0, num_strips, [&](const uint64_t strip, size_t thread) HWY_ATTR {
    PROFILER_ZONE("TwoMatVec.lambda");
    const size_t r0 = strip * rows_per_strip;
    detail::FullDotProductsForStrip<kAdd>(df, mat0, mat_ofs, r0, rows_per_strip,
                                          inner, vec_aligned, add0, out0 + r0);
    detail::FullDotProductsForStrip<kAdd>(df, mat1, mat_ofs, r0, rows_per_strip,
                                          inner, vec_aligned, add1, out1 + r0);
  });

  // Remaining rows
  const size_t r0 = num_strips * rows_per_strip;
  if (r0 < outer) {
    PROFILER_ZONE("TwoMatVec remainder");
    const size_t num_rows = outer - r0;
    detail::FullDotProductsForStrip<kAdd>(df, mat0, mat_ofs, r0, num_rows,
                                          inner, vec_aligned, add0, out0 + r0);
    detail::FullDotProductsForStrip<kAdd>(df, mat1, mat_ofs, r0, num_rows,
                                          inner, vec_aligned, add1, out1 + r0);
  }
}

// With addition
template <typename ArrayT1, typename ArrayT2, typename VecT, typename AddT>
HWY_NOINLINE void TwoMatVecAdd(
    const ArrayT1& mat0, const ArrayT2& mat1, const size_t mat_ofs,
    const size_t outer, const size_t inner,
    const VecT* HWY_RESTRICT vec_aligned, const AddT* HWY_RESTRICT add0,
    const AddT* HWY_RESTRICT add1, float* HWY_RESTRICT out0,
    float* HWY_RESTRICT out1, hwy::ThreadPool& pool) {
  return TwoMatVecT</*kAdd=*/true>(mat0, mat1, mat_ofs, outer, inner,
                                   vec_aligned, add0, add1, out0, out1, pool);
}

// Without addition
template <typename ArrayT1, typename ArrayT2, typename VecT>
HWY_NOINLINE void TwoMatVec(const ArrayT1& mat0, const ArrayT2& mat1,
                            const size_t mat_ofs, const size_t outer,
                            const size_t inner,
                            const VecT* HWY_RESTRICT vec_aligned,
                            float* HWY_RESTRICT out0, float* HWY_RESTRICT out1,
                            hwy::ThreadPool& pool) {
  TwoMatVecT</*kAdd=*/false, ArrayT1, ArrayT2, VecT, VecT>(
      mat0, mat1, mat_ofs, outer, inner, vec_aligned, /*add0=*/nullptr,
      /*add1=*/nullptr, out0, out1, pool);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
