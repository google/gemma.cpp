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

#ifndef THIRD_PARTY_GEMMA_CPP_DEEPSEEK_DEEPSEEK_H_
#define THIRD_PARTY_GEMMA_CPP_DEEPSEEK_DEEPSEEK_H_

// Declares the DeepSeek V4 transformer layer (MLA + CSA/HCA + MoE + mHC) for
// all SIMD targets.

#include <stddef.h>

#include "gemma/gemma.h"
#include "hwy/highway.h"

namespace gcpp {

// Passed to HWY_VISIT_TARGETS; declares for one target.
#define GEMMA_DECL_DEEPSEEK(TARGET, NAMESPACE)                                 \
  namespace NAMESPACE {                                                        \
  void DeepSeekTransformerLayer(size_t num_tokens, size_t layer_idx,           \
                                const LayerWeightsPtrs& layer,                 \
                                Activations& activations, QBatch& qbatch,      \
                                MatMulEnv& env);                               \
  /* Initializes the mHC residual streams from activations.x (after            \
     embedding). No-op unless the model config has hc_mult > 1. */             \
  void DeepSeekMaybeInitHCStreams(Activations& activations, MatMulEnv& env);   \
  /* Collapses the mHC residual streams back into activations.x (before        \
     the final norm) using the model's hc_head weights (mean if absent).       \
     No-op unless the model config has hc_mult > 1. */                         \
  void DeepSeekMaybeFinalizeHCStreams(const WeightsPtrs& weights,              \
                                      Activations& activations,                \
                                      MatMulEnv& env);                         \
  /* Multi-token-prediction block for speculative decoding: consumes the       \
     main model's pre-collapse hc_streams rows plus the next tokens, runs      \
     the extra layer at positions qbatch.Pos(0)+r, optionally computing        \
     draft logits via the MTP head and the shared output head. */              \
  void DeepSeekMTPStep(size_t num_tokens, const int* next_tokens,              \
                       bool compute_logits, const WeightsPtrs& weights,        \
                       Activations& activations, QBatch& qbatch,               \
                       MatMulEnv& env);                                        \
  /* DSpark multi-layer speculative block: consumes dspark_main_hiddens plus   \
     next_tokens, runs all num_mtp_layers speculative layers, and optionally    \
     computes draft logits via final norm and output head. */                   \
  size_t DeepSeekDSparkStep(size_t num_draft_tokens, const int* next_tokens,    \
                            int* out_drafts, float* out_confidences,           \
                            float confidence_threshold,                        \
                            const WeightsPtrs& weights,                        \
                            Activations& activations, QBatch& qbatch,          \
                            MatMulEnv& env);                                   \
  /* Final norm (x -> x_bf) with plain weights; DeepSeek checkpoints store     \
     the true scale, unlike gemma's (1 + w) convention. */                     \
  void DeepSeekFinalNorm(const WeightsPtrs& weights, Activations& activations, \
                         MatMulEnv& env);                                      \
  void ReadRowF32(const MatPtr& w, size_t row, float* HWY_RESTRICT out,        \
                  size_t n);                                                   \
  void ReadRowBF16(const MatPtr& w, size_t row, BF16* HWY_RESTRICT out,        \
                   size_t n);                                                  \
  void GenerateSpecV4(const ModelConfig& config,                               \
                      const RuntimeConfig& runtime_config,                     \
                      const WeightsPtrs& weights, Activations& activations,    \
                      QBatch& qbatch, MatMulEnv& env,                          \
                      TimingInfo& timing_info);                                \
  /* DSpark EAGLE3 feature fusion hook: saves mean residual stream at target  \
     layers 40, 41, 42 into activations.dspark_main_hiddens. No-op unless     \
     config.num_mtp_layers > 1 and layer_idx is a target layer. */            \
  void DeepSeekMaybeSaveDSparkTarget(size_t layer_idx,                        \
                                     Activations& activations);               \
  /* Commits target features for num_tokens starting at pos_base into the      \
     draft layers' SWA KV cache. */                                            \
  void DeepSeekCommitDSparkKV(size_t num_tokens, size_t pos_base,              \
                              const WeightsPtrs& weights,                      \
                              Activations& activations, QBatch& qbatch,        \
                              MatMulEnv& env);                                 \
  /* DSpark EAGLE3 multi-layer speculative decoding driver                    \
     (deepseek_spec.cc); called by GenerateSpecV4 when                        \
     config.num_mtp_layers > 1. */                                            \
  void GenerateDSparkV4(const ModelConfig& config,                            \
                        const RuntimeConfig& runtime_config,                  \
                        const WeightsPtrs& weights, Activations& activations, \
                        QBatch& qbatch, MatMulEnv& env,                       \
                        TimingInfo& timing_info);                              \
  /* NOLINTNEXTLINE(google-readability-namespace-comments) */                  \
  }  // namespace NAMESPACE

// Function declarations for each SIMD target.
HWY_VISIT_TARGETS(GEMMA_DECL_DEEPSEEK)

#undef GEMMA_DECL_DEEPSEEK

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_DEEPSEEK_DEEPSEEK_H_
