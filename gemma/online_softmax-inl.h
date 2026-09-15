// Copyright 2025 Google LLC
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

// Online softmax state merge kernel shared across attention implementations.

#include <stddef.h>
#include <string.h>

#include <algorithm>
#include <cmath>

// Include guard (still compiled once per target)
#if defined(THIRD_PARTY_GEMMA_CPP_GEMMA_ONLINE_SOFTMAX_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_GEMMA_ONLINE_SOFTMAX_INL_H_
#undef THIRD_PARTY_GEMMA_CPP_GEMMA_ONLINE_SOFTMAX_INL_H_
#else
#define THIRD_PARTY_GEMMA_CPP_GEMMA_ONLINE_SOFTMAX_INL_H_
#endif

#include "hwy/highway.h"
// After highway.h
#include "ops/ops-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// Folds the online-softmax state `other` into `accumulator` in place, so that
// the result is as if both halves had been softmaxed together: the two
// attention outputs are rescaled by their renormalized denominators and
// summed. `*_att_out` hold `qkv_dim` floats each, `*_softmax_max` the running
// max logit and `*_softmax_d` the running sum of exponentials. A zero
// denominator marks an empty state, which is the identity of this operation.
HWY_INLINE void MergeOnlineSoftmax(const float* HWY_RESTRICT other_att_out,
                                   const float other_softmax_max,
                                   const float other_softmax_d, size_t qkv_dim,
                                   float* HWY_RESTRICT accumulator_att_out,
                                   float& accumulator_softmax_max,
                                   float& accumulator_softmax_d) {
  if (other_softmax_d == 0.0f) {
    return;
  }
  if (accumulator_softmax_d == 0.0f) {
    memcpy(accumulator_att_out, other_att_out,
           qkv_dim * sizeof(*accumulator_att_out));
    accumulator_softmax_max = other_softmax_max;
    accumulator_softmax_d = other_softmax_d;
    return;
  }
  const float m_new = std::max(accumulator_softmax_max, other_softmax_max);
  const float exp_l = std::exp(accumulator_softmax_max - m_new);
  const float exp_r = std::exp(other_softmax_max - m_new);
  const float d_new = accumulator_softmax_d * exp_l + other_softmax_d * exp_r;
  const float d_new_inv = 1.0f / d_new;
  const float c1 = accumulator_softmax_d * exp_l * d_new_inv;
  const float c2 = other_softmax_d * exp_r * d_new_inv;
  MulByConst(c1, accumulator_att_out, qkv_dim);
  MulByConstAndAdd(c2, other_att_out, accumulator_att_out, qkv_dim);
  accumulator_softmax_max = m_new;
  accumulator_softmax_d = d_new;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_ONLINE_SOFTMAX_INL_H_
