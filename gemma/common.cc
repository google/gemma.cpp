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

#include "compression/shared.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

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
};
constexpr ModelTraining kModelTraining[] = {
    ModelTraining::GEMMA_PT, ModelTraining::GEMMA_IT,  // Gemma 2B
    ModelTraining::GEMMA_PT, ModelTraining::GEMMA_IT,  // Gemma 7B
    ModelTraining::GEMMA_PT, ModelTraining::GEMMA_IT,  // RecurrentGemma
    ModelTraining::GEMMA_IT,                           // Gemma Tiny
    ModelTraining::GEMMA_PT, ModelTraining::GEMMA_IT,  // Gemma2 2B
    ModelTraining::GEMMA_PT, ModelTraining::GEMMA_IT,  // Gemma2 9B
    ModelTraining::GEMMA_PT, ModelTraining::GEMMA_IT,  // Gemma2 27B
    ModelTraining::PALIGEMMA, ModelTraining::PALIGEMMA,  // PaliGemma 224 / 448
    ModelTraining::PALIGEMMA, ModelTraining::PALIGEMMA,  // PG2 3B 224 / 448
    ModelTraining::PALIGEMMA, ModelTraining::PALIGEMMA,  // PG2 10B 224 / 448
};

constexpr size_t kNumModelFlags = std::size(kModelFlags);
static_assert(kNumModelFlags == std::size(kModelTypes));
static_assert(kNumModelFlags == std::size(kModelTraining));

const char* ParseModelTypeAndTraining(const std::string& model_flag,
                                      Model& model, ModelTraining& training) {
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
      training = kModelTraining[i];
      HWY_ASSERT(std::string(ModelString(model, training)) == model_type_lc);
      return nullptr;
    }
  }
  return kErrorMessageBuffer.c_str();
}

const char* ModelString(Model model, ModelTraining training) {
  for (size_t i = 0; i < kNumModelFlags; i++) {
    if (kModelTypes[i] == model && kModelTraining[i] == training)
      return kModelFlags[i];
  }
  HWY_ABORT("Unknown model %d training %d\n", static_cast<int>(model),
            static_cast<int>(training));
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
  if (info.training == ModelTraining::GEMMA_IT) {
    // Prepend "<end_of_turn>" if this is a multi-turn dialogue continuation.
    const std::string start = (pos == 0)
                                  ? "<start_of_turn>user\n"
                                  : "<end_of_turn>\n<start_of_turn>user\n";
    prompt = start + prompt + "<end_of_turn>\n<start_of_turn>model\n";
  }
}

float EmbeddingScaling(size_t model_dim) {
  // Round to bf16 to match Gemma's Embedder, which casts before mul.
  return hwy::ConvertScalarTo<float>(hwy::ConvertScalarTo<hwy::bfloat16_t>(
      sqrtf(static_cast<float>(model_dim))));
}

float ChooseQueryScale(const ModelConfig& config) {
  if (config.query_scale == QueryScaleType::SqrtModelDimDivNumHeads)
    return 1.0f / sqrtf(static_cast<float>(config.model_dim /
                                           config.layer_configs[0].heads));
  // QueryScaleType::SqrtKeySize
  return 1.0f / sqrtf(static_cast<float>(config.layer_configs[0].qkv_dim));
}

}  // namespace gcpp
