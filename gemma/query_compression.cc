// Copyright 2026 Google LLC
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

#include <stddef.h>
#include <stdint.h>

#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#include "util/basics.h"
#include "hwy/base.h"

// Note: HWY_DISABLED_TARGETS needs to be defined the same everywhere.
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/query_compression.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "gemma/query_compression-inl.h"
#include "gemma/query_compression.h"  // includes highway.h

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// Note: q_ptr and out_ptr do not use HWY_RESTRICT because this function may be
// called for in-place compression.
template <typename OutT, class DF, class DOut>
static HWY_INLINE void CompressSingleQueryBF16orInt16(
    DF df, DOut d_out, const float* q_ptr, int qkv_dim, OutT* out_ptr,
    // scale_out is required if OutT is int16_t, and unused otherwise.
    float* scale_out = nullptr) {
  namespace hn = hwy::HWY_NAMESPACE;
  const size_t lanes = hn::Lanes(df);
  const hn::ScalableTag<OutT> d_out_full;
  float s = 1.0f;
  if constexpr (IsInt16<OutT>()) {
    HWY_DASSERT(scale_out != nullptr);
    float max_abs = AbsMaxOfSpan(hwy::Span<const float>(q_ptr, qkv_dim));
    s = max_abs == 0.0f ? 1.0f : 32767.0f / max_abs;
    *scale_out = 1.0f / s;
  }
  auto scale_vec = hn::Set(df, s);

  auto demote = [&](auto x0, auto x1) HWY_ATTR {
    if constexpr (IsInt16<OutT>()) {
      x0 = hn::Mul(x0, scale_vec);
      x1 = hn::Mul(x1, scale_vec);
      return hn::OrderedDemote2To(d_out_full, hn::NearestInt(x0),
                                  hn::NearestInt(x1));
    } else {
      return hn::OrderedDemote2To(d_out_full, x0, x1);
    }
  };

  const size_t dim = static_cast<size_t>(qkv_dim);
  size_t i = 0;
  for (; i + 2 * lanes <= dim; i += 2 * lanes) {
    const auto x0 = hn::LoadU(df, q_ptr + i);
    const auto x1 = hn::LoadU(df, q_ptr + i + lanes);
    hn::StoreU(demote(x0, x1), d_out_full, out_ptr + i);
  }
  // Masked tail so that `qkv_dim` need not be a multiple of `2 * lanes`.
  const size_t remaining = dim - i;
  if (HWY_UNLIKELY(remaining != 0)) {
    const auto x0 = hn::LoadN(df, q_ptr + i, HWY_MIN(remaining, lanes));
    const auto x1 = remaining > lanes
                        ? hn::LoadN(df, q_ptr + i + lanes, remaining - lanes)
                        : hn::Zero(df);
    hn::StoreN(demote(x0, x1), d_out_full, out_ptr + i, remaining);
  }
}

template <typename OutT>
static HWY_INLINE void CompressQueriesBF16orInt16(
    hwy::Span<const float* const> input, int qkv_dim, OutT* HWY_RESTRICT output,
    float* HWY_RESTRICT scale = nullptr) {
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  const DF df;
  auto d_out = hn::Rebind<OutT, decltype(df)>();
  const size_t num_queries = input.size();

  for (size_t q = 0; q < num_queries; ++q) {
    CompressSingleQueryBF16orInt16(df, d_out, input[q], qkv_dim,
                                   output + q * qkv_dim,
                                   scale == nullptr ? nullptr : scale + q);
  }
}

template <typename OutT>
static HWY_INLINE void CompressQueriesBF16orInt16Contiguous(
    const float* HWY_RESTRICT input, int qkv_dim, size_t num_queries,
    OutT* HWY_RESTRICT output, float* HWY_RESTRICT scale = nullptr) {
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  const DF df;
  auto d_out = hn::Rebind<OutT, decltype(df)>();

  for (size_t q = 0; q < num_queries; ++q) {
    CompressSingleQueryBF16orInt16(df, d_out, input + q * qkv_dim, qkv_dim,
                                   output + q * qkv_dim,
                                   scale == nullptr ? nullptr : scale + q);
  }
}

void CompressQueriesBF16(hwy::Span<const float* const> input, int qkv_dim,
                         BF16* HWY_RESTRICT output) {
  CompressQueriesBF16orInt16(input, qkv_dim, output);
}

void CompressQueriesBF16Contiguous(const float* HWY_RESTRICT input, int qkv_dim,
                                   size_t num_queries,
                                   BF16* HWY_RESTRICT output) {
  CompressQueriesBF16orInt16Contiguous(input, qkv_dim, num_queries, output);
}

void CompressQueriesInt16(hwy::Span<const float* const> input, int qkv_dim,
                          int16_t* HWY_RESTRICT output,
                          float* HWY_RESTRICT scale) {
  CompressQueriesBF16orInt16(input, qkv_dim, output, scale);
}

void CompressQueriesInt16Contiguous(const float* HWY_RESTRICT input,
                                    int qkv_dim, size_t num_queries,
                                    int16_t* HWY_RESTRICT output,
                                    float* HWY_RESTRICT scale) {
  CompressQueriesBF16orInt16Contiguous(input, qkv_dim, num_queries, output,
                                       scale);
}

// Quantizes one query row to int8 with a per-row scale. If `kBiased`, adds
// 128 (mod 256) so the bytes can be read as uint8 by VPDPBUSD (see k_sums);
// otherwise stores signed int8 for AMX TDPBSSD.
template <bool kBiased, class DF>
static HWY_INLINE void CompressSingleQueryInt8T(DF df, const float* q_ptr,
                                                int qkv_dim, int8_t* out_ptr,
                                                float* scale_out) {
  namespace hn = hwy::HWY_NAMESPACE;
  const size_t lanes = hn::Lanes(df);
  const hn::ScalableTag<int8_t> d_out_full;
  const hn::ScalableTag<int16_t> d16;

  HWY_DASSERT(scale_out != nullptr);
  float max_abs = AbsMaxOfSpan(hwy::Span<const float>(q_ptr, qkv_dim));
  float s = max_abs == 0.0f ? 1.0f : 127.0f / max_abs;
  *scale_out = 1.0f / s;
  const hn::Vec<DF> scale_vec = hn::Set(df, s);

  HWY_DASSERT(qkv_dim % (4 * lanes) == 0);

  for (size_t i = 0; i < qkv_dim; i += 4 * lanes) {
    hn::Vec<DF> x0 = hn::LoadU(df, q_ptr + i);
    hn::Vec<DF> x1 = hn::LoadU(df, q_ptr + i + lanes);
    hn::Vec<DF> x2 = hn::LoadU(df, q_ptr + i + 2 * lanes);
    hn::Vec<DF> x3 = hn::LoadU(df, q_ptr + i + 3 * lanes);

    x0 = hn::Mul(x0, scale_vec);
    x1 = hn::Mul(x1, scale_vec);
    x2 = hn::Mul(x2, scale_vec);
    x3 = hn::Mul(x3, scale_vec);

    const hn::Vec<decltype(d16)> demoted16_0 =
        hn::OrderedDemote2To(d16, hn::NearestInt(x0), hn::NearestInt(x1));
    const hn::Vec<decltype(d16)> demoted16_1 =
        hn::OrderedDemote2To(d16, hn::NearestInt(x2), hn::NearestInt(x3));
    hn::Vec<decltype(d_out_full)> demoted8 =
        hn::OrderedDemote2To(d_out_full, demoted16_0, demoted16_1);
    if constexpr (kBiased) {
      demoted8 =
          hn::Add(demoted8, hn::Set(d_out_full, static_cast<int8_t>(-128)));
    }
    hn::StoreU(demoted8, d_out_full, out_ptr + i);
  }
}

template <class DF>
static HWY_INLINE void CompressSingleQueryUint8(DF df, const float* q_ptr,
                                                int qkv_dim, int8_t* out_ptr,
                                                float* scale_out) {
  CompressSingleQueryInt8T</*kBiased=*/true>(df, q_ptr, qkv_dim, out_ptr,
                                             scale_out);
}

void CompressQueriesUint8(hwy::Span<const float* const> input, int qkv_dim,
                          int8_t* HWY_RESTRICT output,
                          float* HWY_RESTRICT scale) {
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  const DF df;
  const size_t num_queries = input.size();

  for (size_t q = 0; q < num_queries; ++q) {
    CompressSingleQueryUint8(df, input[q], qkv_dim, output + q * qkv_dim,
                             scale + q);
  }
}

void CompressQueriesUint8Contiguous(const float* HWY_RESTRICT input,
                                    int qkv_dim, size_t num_queries,
                                    int8_t* HWY_RESTRICT output,
                                    float* HWY_RESTRICT scale) {
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  const DF df;

  for (size_t q = 0; q < num_queries; ++q) {
    CompressSingleQueryUint8(df, input + q * qkv_dim, qkv_dim,
                             output + q * qkv_dim, scale + q);
  }
}

template <class DF>
static HWY_INLINE void CompressSingleQueryInt8(DF df, const float* q_ptr,
                                               int qkv_dim, int8_t* out_ptr,
                                               float* scale_out) {
  CompressSingleQueryInt8T</*kBiased=*/false>(df, q_ptr, qkv_dim, out_ptr,
                                              scale_out);
}

void CompressQueriesInt8(hwy::Span<const float* const> input, int qkv_dim,
                         int8_t* HWY_RESTRICT output,
                         float* HWY_RESTRICT scale) {
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  const DF df;
  const size_t num_queries = input.size();

  for (size_t q = 0; q < num_queries; ++q) {
    CompressSingleQueryInt8(df, input[q], qkv_dim, output + q * qkv_dim,
                            scale + q);
  }
}

void CompressQueriesInt8Contiguous(const float* HWY_RESTRICT input, int qkv_dim,
                                   size_t num_queries,
                                   int8_t* HWY_RESTRICT output,
                                   float* HWY_RESTRICT scale) {
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  const DF df;

  for (size_t q = 0; q < num_queries; ++q) {
    CompressSingleQueryInt8(df, input + q * qkv_dim, qkv_dim,
                            output + q * qkv_dim, scale + q);
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();
