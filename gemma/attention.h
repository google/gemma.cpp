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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_ATTENTION_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_ATTENTION_H_

// Declares GemmaAttention for all SIMD targets.

#include <stddef.h>

#include "gemma/gemma.h"
#include "hwy/highway.h"

namespace gcpp {

// Passed to HWY_VISIT_TARGETS; declares for one target.
#define GEMMA_DECL_ATTENTION(TARGET, NAMESPACE)                                \
  namespace NAMESPACE {                                                        \
  void SingleDotSoftmaxWeightedSum(                                            \
      const size_t pos, const size_t start_pos, const size_t last_pos,         \
      float* HWY_RESTRICT q, const MatPtrT<float>& k, const MatPtrT<float>& v, \
      size_t layer_idx, const LayerWeightsPtrs& layer,                         \
      const Activations& activations, float* HWY_RESTRICT att,                 \
      float* HWY_RESTRICT att_out);                                            \
                                                                               \
  void DotSoftmaxWeightedSum(const size_t num_tokens,                          \
                             const QueriesPos& queries_pos,                    \
                             const QueriesPos& queries_prefix_end,             \
                             size_t layer_idx, const LayerWeightsPtrs& layer,  \
                             Activations& activations,                         \
                             const KVCaches& kv_caches, NestedPools& pools);   \
                                                                               \
  void GemmaAttention(size_t num_tokens, const QueriesPos& queries_pos,        \
                      const QueriesPos* queries_prefix_end,                    \
                      const size_t layer_idx, const LayerWeightsPtrs& layer,   \
                      Activations& activations, const KVCaches& kv_caches,     \
                      MatMulEnv& env, int flags);                              \
  /* NOLINTNEXTLINE(google-readability-namespace-comments) */                  \
  }  // namespace NAMESPACE

// Function declarations for each SIMD target. Allows direct call from the
// per-target namespace. We may later replace this with dynamic dispatch if
// the overhead is acceptable.
HWY_VISIT_TARGETS(GEMMA_DECL_ATTENTION)

#undef GEMMA_DECL_ATTENTION

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_ATTENTION_H_
