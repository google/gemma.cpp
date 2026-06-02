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

#include "gemma/activations.h"
#include "gemma/configs.h"   // AttentionImpl
#include "gemma/kv_cache.h"  // KV_t
#include "gemma/query.h"     // QBatch
#include "gemma/weights.h"   // LayerWeightsPtrs
#include "ops/matmul.h"
#include "hwy/highway.h"     // HWY_VISIT_TARGETS
#include "hwy/per_target.h"  // VectorBytes

namespace gcpp {

// Returns the number of floats per vector (aka NF).
inline size_t FloatsPerVector() { return hwy::VectorBytes() / sizeof(float); }

// The attention window usually starts at 0 unless `pos` is larger than
// the attention window size, then it is `pos` - window_size + 1.
inline size_t StartPos(size_t pos, const ModelConfig& config,
                       size_t layer_idx) {
  const size_t att_window_size = config.attention_window_sizes[layer_idx];
  return pos - HWY_MIN(att_window_size - 1, pos);
}

// The k-cache and v-cache are setup without knowing NF. So if it hasn't been
// done already, reshape it to take NF into account. Must be called before
// FlashAttention.
inline void MaybeReshapeCache(const size_t default_cols, MatPtrT<KV_t>& cache) {
  if (default_cols == cache.Cols()) {
    cache.ReshapePackedRowsToCols(2 * FloatsPerVector());
  }
}

// Passed to HWY_VISIT_TARGETS; declares for one target.
#define GEMMA_DECL_ATTENTION(TARGET, NAMESPACE)                               \
  namespace NAMESPACE {                                                       \
  void TransposeKVCacheRow(const KV_t* HWY_RESTRICT kv, KV_t* HWY_RESTRICT k, \
                           KV_t* HWY_RESTRICT v, size_t qkv_dim);             \
  void TransposeOOBKVCacheRow(KV_t* HWY_RESTRICT k, KV_t* HWY_RESTRICT v,     \
                              size_t qkv_dim);                                \
                                                                              \
  void PositionalEncodingQK(float* qk, size_t layer_idx,                      \
                            const AttentionActivationsPtrs& activations,      \
                            ThreadingContext& ctx, size_t worker, size_t pos, \
                            float mul);                                       \
                                                                              \
  void GemmaAttention(size_t num_tokens, const size_t layer_idx,              \
                      const LayerWeightsPtrs& layer,                          \
                      AttentionActivationsPtrs& activations, QBatch& qbatch,  \
                      MatMulEnv& env, AttentionImpl attention_impl,           \
                      int flags);                                             \
  /* NOLINTNEXTLINE(google-readability-namespace-comments) */                 \
  }  // namespace NAMESPACE

// Function declarations for each SIMD target. Allows direct call from the
// per-target namespace. We may later replace this with dynamic dispatch if
// the overhead is acceptable.
HWY_VISIT_TARGETS(GEMMA_DECL_ATTENTION)

#undef GEMMA_DECL_ATTENTION

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_ATTENTION_H_
