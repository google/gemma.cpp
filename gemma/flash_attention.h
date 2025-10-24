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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_ATTENTION_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_ATTENTION_H_

// Declares FlashAttention for all SIMD targets.

#include <stddef.h>

#include "gemma/gemma.h"
#include "hwy/highway.h"

namespace gcpp {

// Passed to HWY_VISIT_TARGETS; declares for one target.
#define GEMMA_DECL_FLASH_ATTENTION(TARGET, NAMESPACE)                        \
  namespace NAMESPACE {                                                      \
  void RMSNormAndPositionalEncoding(                                         \
      size_t num_tokens, const QBatch& qbatch, MatPtrT<float>& q,            \
      const MatPtrT<float>& query_norm_scale, size_t layer_idx,              \
      const AttentionActivationsPtrs& activations, ThreadingContext& ctx);   \
                                                                             \
  void SingleFlashAttention(size_t start_pos, size_t last_pos,               \
                            const float* HWY_RESTRICT q,                     \
                            const MatPtrT<KV_t>& k, const MatPtrT<KV_t>& v,  \
                            size_t layer_idx,                                \
                            const AttentionActivationsPtrs& activations,     \
                            float* HWY_RESTRICT att_out,                     \
                            ThreadingContext& ctx, size_t worker);           \
                                                                             \
  size_t GetVTileSize(size_t kNF, size_t num_head_groups, size_t num_tokens, \
                      size_t total_tasks, size_t target_parallelism);        \
                                                                             \
  void FlashAttention(size_t num_tokens, size_t target_parallelism,          \
                      size_t layer_idx,                                      \
                      const MatPtrT<float>& query_norm_scale,                \
                      AttentionActivationsPtrs& activations, QBatch& qbatch, \
                      ThreadingContext& ctx);                                \
  /* NOLINTNEXTLINE(google-readability-namespace-comments) */                \
  }  // namespace NAMESPACE

// Function declarations for each SIMD target. Allows direct call from the
// per-target namespace. We may later replace this with dynamic dispatch if
// the overhead is acceptable.
HWY_VISIT_TARGETS(GEMMA_DECL_FLASH_ATTENTION)

#undef GEMMA_DECL_FLASH_ATTENTION

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_ATTENTION_H_
