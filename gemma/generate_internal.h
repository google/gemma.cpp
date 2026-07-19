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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_GENERATE_INTERNAL_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_GENERATE_INTERNAL_H_

// Per-target declarations of gemma.cc generation internals, for use by the
// DeepSeek speculative-decoding driver (deepseek/deepseek_spec.cc).

#include <stddef.h>

#include "gemma/gemma.h"
#include "util/mat.h"
#include "hwy/bit_set.h"
#include "hwy/highway.h"

namespace gcpp {

// Passed to HWY_VISIT_TARGETS; declares for one target.
#define GEMMA_DECL_GENERATE_INTERNAL(TARGET, NAMESPACE)                       \
  namespace NAMESPACE {                                                      \
  void TransformerLayer(const size_t num_tokens, const size_t layer_idx,     \
                        const LayerWeightsPtrs& layer,                       \
                        Activations& activations, QBatch& qbatch,            \
                        MatMulEnv& env);                                     \
  size_t EmbedMMToken(int token, size_t x_row, size_t pos,                   \
                      size_t pos_in_prompt, const ModelConfig& model_config, \
                      const WeightsPtrs& weights, MatStorageT<float>& x,     \
                      ThreadingContext& ctx,                                 \
                      const ImageTokens* image_tokens,                       \
                      size_t image_token_position);                          \
  void Transformer(const ModelConfig& config,                                \
                   const RuntimeConfig& runtime_config,                      \
                   const WeightsPtrs& weights, Activations& activations,     \
                   QBatch& qbatch, MatMulEnv& env);                          \
  /* Final norm x -> x_bf (DeepSeek plain-weight or gemma (1 + w)). */       \
  void FinalNormBatched(const ModelConfig& config,                           \
                        const WeightsPtrs& weights,                          \
                        Activations& activations, MatMulEnv& env);           \
  /* Logits from x_bf via the (untied if present) output head. */            \
  void FinalLogits(const WeightsPtrs& weights, Activations& activations,     \
                   MatMulEnv& env);                                          \
  size_t PrefillTBatchOrQBatch(                                              \
      const ModelConfig& config, const RuntimeConfig& runtime_config,        \
      const WeightsPtrs& weights, Activations& activations, QBatch& qbatch,  \
      MatMulEnv& env, TimingInfo& timing_info);                              \
  void StreamAndUpdateEOSAfterPrefill(const ModelConfig& config,             \
                                      const RuntimeConfig& runtime_config,   \
                                      QBatch& qbatch,                        \
                                      hwy::BitSet4096<>& non_eos, size_t qi);\
  /* NOLINTNEXTLINE(google-readability-namespace-comments) */                \
  }  // namespace NAMESPACE

// Function declarations for each SIMD target.
HWY_VISIT_TARGETS(GEMMA_DECL_GENERATE_INTERNAL)

#undef GEMMA_DECL_GENERATE_INTERNAL

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_GENERATE_INTERNAL_H_
