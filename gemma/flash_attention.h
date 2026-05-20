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

#include <cstdint>

#include "gemma/configs.h"
#include "gemma/flash_structs.h"
#include "gemma/kv_cache.h"
#include "gemma/query.h"
#include "util/basics.h"
#include "util/mat.h"
#include "util/threading_context.h"
#include "hwy/aligned_allocator.h"
#include "hwy/highway.h"

namespace gcpp {

// Passed to HWY_VISIT_TARGETS; declares for one target.
#define GEMMA_DECL_FLASH_ATTENTION(TARGET, NAMESPACE)                        \
  namespace NAMESPACE {                                                      \
  void RMSNormAndPositionalEncoding(                                         \
      size_t num_tokens, const QBatch& qbatch, MatPtrT<float>& q,            \
      const MatPtr& query_norm_scale, size_t layer_idx,                      \
      const AttentionActivationsPtrs& activations, ThreadingContext& ctx);   \
                                                                             \
  size_t GetVTileSize(size_t kNF, size_t num_head_groups, size_t num_tokens, \
                      size_t total_tasks, size_t target_parallelism);        \
                                                                             \
  void FlashAttention(size_t num_tokens, size_t target_parallelism,          \
                      size_t layer_idx, const MatPtr& query_norm_scale,      \
                      AttentionActivationsPtrs& activations, QBatch& qbatch, \
                      ThreadingContext& ctx, AttentionImpl attention_impl);  \
                                                                             \
  void DispatchTileFlashAttentionReturnExpSumsAndMaxLogits(                  \
      hwy::Span<const MatPtr> kvs, size_t q_count,                           \
      const float* HWY_RESTRICT q_base,                                      \
      hwy::Span<const size_t> start_pos_per_query,                           \
      hwy::Span<const size_t> last_pos_per_query, const float att_cap,       \
      MatPtrT<float>& att_out, float* HWY_RESTRICT exp_denominator_sums,     \
      float* HWY_RESTRICT max_logits);                                       \
                                                                             \
  void DispatchTileFlashAttentionReturnExpSumsAndMaxLogitsBF16(              \
      hwy::Span<const MatPtr> kvs, size_t q_count,                           \
      const BF16* HWY_RESTRICT q_base,                                       \
      hwy::Span<const size_t> start_pos_per_query,                           \
      hwy::Span<const size_t> last_pos_per_query, const float att_cap,       \
      MatPtrT<float>& att_out, float* HWY_RESTRICT exp_denominator_sums,     \
      float* HWY_RESTRICT max_logits);                                       \
                                                                             \
  void DispatchTileFlashAttentionReturnExpSumsAndMaxLogitsInt16(             \
      hwy::Span<const MatPtr> kvs, size_t q_count,                           \
      const int16_t* HWY_RESTRICT q_base, hwy::Span<const float> q_scales,   \
      hwy::Span<const size_t> start_pos_per_query,                           \
      hwy::Span<const size_t> last_pos_per_query, const float att_cap,       \
      MatPtrT<float>& att_out, float* HWY_RESTRICT exp_denominator_sums,     \
      float* HWY_RESTRICT max_logits);                                       \
                                                                             \
  /* NOLINTNEXTLINE(google-readability-namespace-comments) */                \
  }  // namespace NAMESPACE

// Function declarations for each SIMD target. Allows direct call from the
// per-target namespace. We may later replace this with dynamic dispatch if
// the overhead is acceptable.
HWY_VISIT_TARGETS(GEMMA_DECL_FLASH_ATTENTION)

#undef GEMMA_DECL_FLASH_ATTENTION

void DispatchDispatchTileFlashAttention148(
    Tile148Params& params, const MatPtrT<BF16>& q, const MatPtrT<KV_t>& k,
    const MatPtrT<KV_t>& v, const size_t layer_idx,
    const AttentionActivationsPtrs& activations, MatPtrT<float>& att_out,
    size_t qkv_dim, ThreadingContext& ctx, const size_t worker,
    AttentionImpl attention_impl);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_ATTENTION_H_
