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
#include <string.h>

#include <algorithm>  // std::transform
#include <cctype>
#include <string>
#include <vector>

#include "util/basics.h"  // BF16
// TODO: change include when PromptWrapping is moved.
#include "compression/shared.h"  // PromptWrapping
#include "hwy/base.h"

namespace gcpp {

constexpr const char* kModelFlags[] = {
    "2b-pt", "2b-it",                // Gemma 2B
    "7b-pt", "7b-it",                // Gemma 7B
    "gr2b-pt", "gr2b-it",            // RecurrentGemma
    "tiny",                          // Gemma Tiny (mostly for debugging)
    "gemma2-2b-pt", "gemma2-2b-it",  // Gemma2 2B
    "9b-pt", "9b-it",                // Gemma2 9B
    "27b-pt", "27b-it",              // Gemma2 27B
    "paligemma-224",                 // PaliGemma 224
    "paligemma-448",                 // PaliGemma 448
    "paligemma2-3b-224",             // PaliGemma2 3B 224
    "paligemma2-3b-448",             // PaliGemma2 3B 448
    "paligemma2-10b-224",            // PaliGemma2 10B 224
    "paligemma2-10b-448",            // PaliGemma2 10B 448
    "gemma3-4b",                     // Gemma3 4B
    "gemma3-1b",                     // Gemma3 1B
    "gemma3-12b",                    // Gemma3 12B
    "gemma3-27b",                    // Gemma3 27B
};
constexpr Model kModelTypes[] = {
    Model::GEMMA_2B, Model::GEMMA_2B,      // Gemma 2B
    Model::GEMMA_7B, Model::GEMMA_7B,      // Gemma 7B
    Model::GRIFFIN_2B, Model::GRIFFIN_2B,  // RecurrentGemma
    Model::GEMMA_TINY,                     // Gemma Tiny
    Model::GEMMA2_2B, Model::GEMMA2_2B,    // Gemma2 2B
    Model::GEMMA2_9B, Model::GEMMA2_9B,    // Gemma2 9B
    Model::GEMMA2_27B, Model::GEMMA2_27B,  // Gemma2 27B
    Model::PALIGEMMA_224,                  // PaliGemma 224
    Model::PALIGEMMA_448,                  // PaliGemma 448
    Model::PALIGEMMA2_3B_224,              // PaliGemma2 3B 224
    Model::PALIGEMMA2_3B_448,              // PaliGemma2 3B 448
    Model::PALIGEMMA2_10B_224,             // PaliGemma2 10B 224
    Model::PALIGEMMA2_10B_448,             // PaliGemma2 10B 448
    Model::GEMMA3_4B,                      // Gemma3 4B
    Model::GEMMA3_1B,                      // Gemma3 1B
    Model::GEMMA3_12B,                     // Gemma3 12B
    Model::GEMMA3_27B,                     // Gemma3 27B
};
constexpr PromptWrapping kPromptWrapping[] = {
    PromptWrapping::GEMMA_PT, PromptWrapping::GEMMA_IT,    // Gemma 2B
    PromptWrapping::GEMMA_PT, PromptWrapping::GEMMA_IT,    // Gemma 7B
    PromptWrapping::GEMMA_PT, PromptWrapping::GEMMA_IT,    // RecurrentGemma
    PromptWrapping::GEMMA_IT,                              // Gemma Tiny
    PromptWrapping::GEMMA_PT, PromptWrapping::GEMMA_IT,    // Gemma2 2B
    PromptWrapping::GEMMA_PT, PromptWrapping::GEMMA_IT,    // Gemma2 9B
    PromptWrapping::GEMMA_PT, PromptWrapping::GEMMA_IT,    // Gemma2 27B
    PromptWrapping::PALIGEMMA, PromptWrapping::PALIGEMMA,  // PaliGemma 224/448
    PromptWrapping::PALIGEMMA, PromptWrapping::PALIGEMMA,  // PG2 3B 224/448
    PromptWrapping::PALIGEMMA, PromptWrapping::PALIGEMMA,  // PG2 10B 224/448
    PromptWrapping::GEMMA_VLM,                             // Gemma3 4B
    PromptWrapping::GEMMA_IT,                              // Gemma3 1B
    PromptWrapping::GEMMA_VLM,                             // Gemma3 12B
    PromptWrapping::GEMMA_VLM,                             // Gemma3 27B
};

constexpr size_t kNumModelFlags = std::size(kModelFlags);
static_assert(kNumModelFlags == std::size(kModelTypes));
static_assert(kNumModelFlags == std::size(kPromptWrapping));

const char* ParseModelTypeAndWrapping(const std::string& model_flag,
                                      Model& model, PromptWrapping& wrapping) {
  static std::string kErrorMessageBuffer =
      "Invalid or missing model flag, need to specify one of ";
  for (size_t i = 0; i + 1 < kNumModelFlags; ++i) {
    kErrorMessageBuffer.append(kModelFlags[i]);
    kErrorMessageBuffer.append(", ");
  }
  kErrorMessageBuffer.append(kModelFlags[kNumModelFlags - 1]);
  kErrorMessageBuffer.append(".");
  std::string model_type_lc = model_flag;
  std::transform(model_type_lc.begin(), model_type_lc.end(),
                 model_type_lc.begin(), ::tolower);
  for (size_t i = 0; i < kNumModelFlags; ++i) {
    if (kModelFlags[i] == model_type_lc) {
      model = kModelTypes[i];
      wrapping = kPromptWrapping[i];
      HWY_ASSERT(std::string(ModelString(model, wrapping)) == model_type_lc);
      return nullptr;
    }
  }
  return kErrorMessageBuffer.c_str();
}

const char* ModelString(Model model, PromptWrapping wrapping) {
  for (size_t i = 0; i < kNumModelFlags; i++) {
    if (kModelTypes[i] == model && kPromptWrapping[i] == wrapping)
      return kModelFlags[i];
  }
  HWY_ABORT("Unknown model %d wrapping %d\n", static_cast<int>(model),
            static_cast<int>(wrapping));
}

const char* StringFromType(Type type) {
  return kTypeStrings[static_cast<size_t>(type)];
}

const char* ParseType(const std::string& type_string, Type& type) {
  constexpr size_t kNum = std::size(kTypeStrings);
  static std::string kErrorMessageBuffer =
      "Invalid or missing type, need to specify one of ";
  for (size_t i = 0; i + 1 < kNum; ++i) {
    kErrorMessageBuffer.append(kTypeStrings[i]);
    kErrorMessageBuffer.append(", ");
  }
  kErrorMessageBuffer.append(kTypeStrings[kNum - 1]);
  kErrorMessageBuffer.append(".");
  std::string type_lc = type_string;
  std::transform(type_lc.begin(), type_lc.end(), type_lc.begin(), ::tolower);
  for (size_t i = 0; i < kNum; ++i) {
    if (kTypeStrings[i] == type_lc) {
      type = static_cast<Type>(i);
      HWY_ASSERT(std::string(StringFromType(type)) == type_lc);
      return nullptr;
    }
  }
  return kErrorMessageBuffer.c_str();
}

void Wrap(const ModelInfo& info, size_t pos, std::string& prompt) {

  // Instruction-tuned models are trained to expect control tokens.
  if (info.wrapping == PromptWrapping::GEMMA_IT) {
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

}  // namespace gcpp
