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

#include "gemma/common.h"

#include <math.h>  // sqrtf
#include <stddef.h>

#include <string>
#include <vector>

#include "gemma/configs.h"
#include "util/basics.h"  // BF16
#include "hwy/base.h"       // ConvertScalarTo

namespace gcpp {

void Wrap(const ModelConfig& config, size_t pos, std::string& prompt) {

  // Instruction-tuned models are trained to expect control tokens.
  if (config.wrapping == PromptWrapping::GEMMA_IT) {
    // Prepend "<end_of_turn>" if this is a multi-turn dialogue continuation.
    const std::string start = (pos == 0)
                                  ? "<start_of_turn>user\n"
                                  : "<end_of_turn>\n<start_of_turn>user\n";
    prompt = start + prompt + "<end_of_turn>\n<start_of_turn>model\n";
  }
}

float EmbeddingScaling(size_t model_dim) {
  // Round to bf16 to match Gemma's Embedder, which casts before mul.
  return hwy::ConvertScalarTo<float>(
      hwy::ConvertScalarTo<BF16>(sqrtf(static_cast<float>(model_dim))));
}

float ChooseQueryScale(const ModelConfig& config) {
  if (config.query_scale == QueryScaleType::SqrtModelDimDivNumHeads)
    return 1.0f / sqrtf(static_cast<float>(config.model_dim /
                                           config.layer_configs[0].heads));
  // QueryScaleType::SqrtKeySize
  return 1.0f / sqrtf(static_cast<float>(config.layer_configs[0].qkv_dim));
}

void RangeChecks(const ModelConfig& weights_config,
                 size_t& max_generated_tokens, const size_t prompt_size) {
  if (!weights_config.use_local_attention) {
    if (max_generated_tokens > weights_config.seq_len) {
      HWY_WARN("max_generated_tokens %zu > kSeqLen %u, truncating.",
               max_generated_tokens, weights_config.seq_len);
      max_generated_tokens = weights_config.seq_len;
    }
  }
  HWY_ASSERT(prompt_size > 0);
}

}  // namespace gcpp
