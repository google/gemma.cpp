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

#ifndef THIRD_PARTY_GEMMA_CPP_OPS_OPS_H_
#define THIRD_PARTY_GEMMA_CPP_OPS_OPS_H_

#include <stddef.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include "util/mat.h"
#include "hwy/aligned_allocator.h"  // Span
#include "hwy/base.h"

namespace gcpp {

// Applies vector logit mask to `logits`.
// `mask_words` points to an array of uint64_t words where bit i = 1 indicates
// token i is allowed, and bit i = 0 indicates token i is disallowed.
// Disallowed tokens are overwritten with `mask_value` (-infinity).
//
// Fast Path 1: word == ~0ULL -> all 64 tokens allowed; skip writes.
// Fast Path 2: word == 0ULL -> all 64 tokens disallowed; vector store
// mask_value (-inf). General Path: Highway SIMD LoadMaskBits, IfThenElse, and
// StoreU. Guardrail: Empty mask (0 set bits across entire vocabulary) preserves
// finite logits.
void ApplyLogitMaskKernel(
    hwy::Span<float> logits, const uint64_t* HWY_RESTRICT mask_words,
    size_t vocab_size,
    float mask_value = -std::numeric_limits<float>::infinity());

static inline HWY_MAYBE_UNUSED MatStorageT<float> CreateInvTimescale(
    const Allocator& allocator, size_t qkv_dim, bool half_rope,
    double base_frequency, float partial_rotary_factor = 1.0f) {
  const size_t rope_dim = half_rope ? qkv_dim / 2 : qkv_dim;
  const size_t total_entries = rope_dim / 2;
  const size_t active_entries =
      static_cast<size_t>(total_entries * partial_rotary_factor);
  MatStorageT<float> inv_timescale("inv_timescale", total_entries, allocator);
  float* data = inv_timescale.Row(0);
  for (size_t dim = 0; dim < active_entries; ++dim) {
    const double freq_exponents =
        static_cast<double>(2 * dim) / static_cast<double>(rope_dim);
    // Replacing with expf(ln(1E4) * freq_exponents) changes results
    // noticeably.
    data[dim] =
        static_cast<float>(1.0 / std::pow(base_frequency, freq_exponents));
  }
  // Zero out NoPE dimensions — cos(0)=1, sin(0)=0 gives identity.
  for (size_t dim = active_entries; dim < total_entries; ++dim) {
    data[dim] = 0.0f;
  }
  return inv_timescale;
}

struct SMOptions {
  float* HWY_RESTRICT max_out = nullptr;
  float* HWY_RESTRICT d_out = nullptr;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_OPS_OPS_H_
