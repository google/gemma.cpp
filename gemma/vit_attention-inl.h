// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_VIT_ATTENTION_INL_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_VIT_ATTENTION_INL_H_
#include <stddef.h>

#include "util/mat.h"
#endif

#if defined(GEMMA_VIT_ATTENTION_TOGGLE) == defined(HWY_TARGET_TOGGLE)
#ifdef GEMMA_VIT_ATTENTION_TOGGLE
#undef GEMMA_VIT_ATTENTION_TOGGLE
#else
#define GEMMA_VIT_ATTENTION_TOGGLE
#endif

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// Four query rows share each V load. Two vectors per row leave room for the
// weights and V in AVX2's 16 registers. Each output retains the original
// increasing-source-position FP32 MulAdd order; there is no BF16 conversion.
template <size_t kRows, bool kTwoVectors, bool kTail>
HWY_INLINE void VitValueTile(const MatPtrT<float>& probabilities,
                             const MatPtrT<float>& values,
                             MatPtrT<float>& output, size_t row, size_t col,
                             size_t output_offset) {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> d;
  const size_t lanes = hn::Lanes(d);
  const size_t count = kTail ? values.Cols() - col : lanes;
  auto a0 = hn::Zero(d), a1 = a0, a2 = a0, a3 = a0;
  auto b0 = a0, b1 = a0, b2 = a0, b3 = a0;
  const float* HWY_RESTRICT p0 = probabilities.Row(row);
  const float* HWY_RESTRICT p1 = kRows == 4 ? probabilities.Row(row + 1) : p0;
  const float* HWY_RESTRICT p2 = kRows == 4 ? probabilities.Row(row + 2) : p0;
  const float* HWY_RESTRICT p3 = kRows == 4 ? probabilities.Row(row + 3) : p0;
  for (size_t i = 0; i < values.Rows(); ++i) {
    const float* HWY_RESTRICT v = values.Row(i) + col;
    const auto v0 = kTail ? hn::LoadN(d, v, count) : hn::LoadU(d, v);
    const auto w0 = hn::Set(d, p0[i]);
    a0 = hn::MulAdd(w0, v0, a0);
    if constexpr (kRows == 4) {
      a1 = hn::MulAdd(hn::Set(d, p1[i]), v0, a1);
      a2 = hn::MulAdd(hn::Set(d, p2[i]), v0, a2);
      a3 = hn::MulAdd(hn::Set(d, p3[i]), v0, a3);
    }
    if constexpr (kTwoVectors) {
      const auto v1 = hn::LoadU(d, v + lanes);
      b0 = hn::MulAdd(w0, v1, b0);
      if constexpr (kRows == 4) {
        b1 = hn::MulAdd(hn::Set(d, p1[i]), v1, b1);
        b2 = hn::MulAdd(hn::Set(d, p2[i]), v1, b2);
        b3 = hn::MulAdd(hn::Set(d, p3[i]), v1, b3);
      }
    }
  }
  float* out0 = output.Row(row) + output_offset + col;
  if constexpr (kTail)
    hn::StoreN(a0, d, out0, count);
  else
    hn::StoreU(a0, d, out0);
  if constexpr (kTwoVectors) hn::StoreU(b0, d, out0 + lanes);
  if constexpr (kRows == 4) {
    float* out1 = output.Row(row + 1) + output_offset + col;
    float* out2 = output.Row(row + 2) + output_offset + col;
    float* out3 = output.Row(row + 3) + output_offset + col;
    if constexpr (kTail) {
      hn::StoreN(a1, d, out1, count);
      hn::StoreN(a2, d, out2, count);
      hn::StoreN(a3, d, out3, count);
    } else {
      hn::StoreU(a1, d, out1);
      hn::StoreU(a2, d, out2);
      hn::StoreU(a3, d, out3);
    }
    if constexpr (kTwoVectors) {
      hn::StoreU(b1, d, out1 + lanes);
      hn::StoreU(b2, d, out2 + lanes);
      hn::StoreU(b3, d, out3 + lanes);
    }
  }
}

template <size_t kRows>
HWY_INLINE void VitValueRows(const MatPtrT<float>& probabilities,
                             const MatPtrT<float>& values,
                             MatPtrT<float>& output, size_t row,
                             size_t output_offset) {
  const size_t lanes =
      hwy::HWY_NAMESPACE::Lanes(hwy::HWY_NAMESPACE::ScalableTag<float>());
  size_t col = 0;
  for (; col + 2 * lanes <= values.Cols(); col += 2 * lanes) {
    VitValueTile<kRows, true, false>(probabilities, values, output, row, col,
                                     output_offset);
  }
  if (col + lanes <= values.Cols()) {
    VitValueTile<kRows, false, false>(probabilities, values, output, row, col,
                                      output_offset);
    col += lanes;
  }
  if (col < values.Cols()) {
    VitValueTile<kRows, false, true>(probabilities, values, output, row, col,
                                     output_offset);
  }
}

// One task owns up to four output rows. Output can contain adjacent heads;
// only this head's columns are written, including for partial SIMD vectors.
HWY_INLINE void VitAttentionValueProduct(const MatPtrT<float>& probabilities,
                                         const MatPtrT<float>& values,
                                         MatPtrT<float>& output, size_t row,
                                         size_t output_offset) {
  HWY_DASSERT(probabilities.Cols() == values.Rows());
  HWY_DASSERT(probabilities.Rows() <= output.Rows());
  HWY_DASSERT(output_offset + values.Cols() <= output.Cols());
  if (row + 4 <= probabilities.Rows()) {
    VitValueRows<4>(probabilities, values, output, row, output_offset);
  } else {
    for (; row < probabilities.Rows(); ++row) {
      VitValueRows<1>(probabilities, values, output, row, output_offset);
    }
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();
#endif  // GEMMA_VIT_ATTENTION_TOGGLE
