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

// Per-target helpers shared by query_compression.cc and tiled_attention.cc.

// Include guard for non-SIMD code.
#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_QUERY_COMPRESSION_INL_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_QUERY_COMPRESSION_INL_H_

#include <stddef.h>

#include "compression/types.h"
#include "hwy/base.h"
#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_QUERY_COMPRESSION_INL_H_

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_GEMMA_QUERY_COMPRESSION_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_GEMMA_QUERY_COMPRESSION_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_GEMMA_QUERY_COMPRESSION_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_GEMMA_QUERY_COMPRESSION_TOGGLE
#endif

#include "compression/compress-inl.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// Returns the maximum absolute value in `span`.
template <typename T>
float AbsMaxOfSpan(hwy::Span<const T> span) {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> df;
  using VF = hn::Vec<decltype(df)>;
  VF max_vec = hn::Zero(df);
  HWY_LANES_CONSTEXPR size_t N = hn::Lanes(df);
  HWY_LANES_CONSTEXPR size_t step = 2 * N;
  const PackedSpan<const T> packed_span(span.data(), span.size());
  size_t i = 0;
  for (; i + step <= span.size(); i += step) {
    VF v0, v1;
    Decompress2(df, packed_span, i, v0, v1);
    max_vec = hn::Max(max_vec, hn::Max(hn::Abs(v0), hn::Abs(v1)));
  }
  float max_scalar = 0.0f;
  for (; i < span.size(); ++i) {
    max_scalar = HWY_MAX(max_scalar,
                         hwy::ScalarAbs(hwy::ConvertScalarTo<float>(span[i])));
  }
  return HWY_MAX(hn::ReduceMax(df, max_vec), max_scalar);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
