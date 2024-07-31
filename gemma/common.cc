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

#include <stddef.h>
#include <string.h>

#include <algorithm>  // std::transform
#include <cctype>
#include <string>
#include <vector>

#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

constexpr const char* kModelFlags[] = {
    "2b-pt", "2b-it",                // Gemma 2B
    "7b-pt", "7b-it",                // Gemma 7B
    "9b-pt", "9b-it",                // Gemma 9B
    "27b-pt", "27b-it",              // Gemma 27B
    "gr2b-pt", "gr2b-it",            // RecurrentGemma
    "tiny",                          // Gemma Tiny (mostly for debugging)
    "gemma2-2b-pt", "gemma2-2b-it",  // Gemma2 2B
};
constexpr Model kModelTypes[] = {
    Model::GEMMA_2B, Model::GEMMA_2B,      // Gemma 2B
    Model::GEMMA_7B, Model::GEMMA_7B,      // Gemma 7B
    Model::GEMMA_9B, Model::GEMMA_9B,      // Gemma 9B
    Model::GEMMA_27B, Model::GEMMA_27B,    // Gemma 27B
    Model::GRIFFIN_2B, Model::GRIFFIN_2B,  // RecurrentGemma
    Model::GEMMA_TINY,                     // Gemma Tiny
    Model::GEMMA2_2B, Model::GEMMA2_2B,    // Gemma2 2B
};
constexpr ModelTraining kModelTraining[] = {
    ModelTraining::GEMMA_PT, ModelTraining::GEMMA_IT,  // Gemma 2B
    ModelTraining::GEMMA_PT, ModelTraining::GEMMA_IT,  // Gemma 7B
    ModelTraining::GEMMA_PT, ModelTraining::GEMMA_IT,  // Gemma 9B
    ModelTraining::GEMMA_PT, ModelTraining::GEMMA_IT,  // Gemma 27B
    ModelTraining::GEMMA_PT, ModelTraining::GEMMA_IT,  // RecurrentGemma
    ModelTraining::GEMMA_IT,                           // Gemma Tiny
    ModelTraining::GEMMA_PT, ModelTraining::GEMMA_IT,  // Gemma 2B2
};

constexpr size_t kNumModelFlags =
    std::end(kModelFlags) - std::begin(kModelFlags);
static_assert(kNumModelFlags ==
              std::end(kModelTypes) - std::begin(kModelTypes));
static_assert(kNumModelFlags ==
              std::end(kModelTraining) - std::begin(kModelTraining));

const char* ParseModelTypeAndTraining(const std::string& model_flag,
                                      Model& model, ModelTraining& training) {
  static char kErrorMessageBuffer[kNumModelFlags * 8 + 1024] =
      "Invalid or missing model flag, need to specify one of ";
  for (size_t i = 0; i + 1 < kNumModelFlags; i++) {
    strcat(kErrorMessageBuffer, kModelFlags[i]);  // NOLINT
    strcat(kErrorMessageBuffer, ", ");            // NOLINT
  }
  strcat(kErrorMessageBuffer, kModelFlags[kNumModelFlags - 1]);  // NOLINT
  strcat(kErrorMessageBuffer, ".");                    // NOLINT

  std::string model_type_lc = model_flag;
  std::transform(begin(model_type_lc), end(model_type_lc), begin(model_type_lc),
                 [](unsigned char c) { return std::tolower(c); });

  for (size_t i = 0; i < kNumModelFlags; i++) {
    if (kModelFlags[i] == model_type_lc) {
      model = kModelTypes[i];
      training = kModelTraining[i];
      HWY_ASSERT(std::string(ModelString(model, training)) == model_type_lc);
      return nullptr;
    }
  }
  return kErrorMessageBuffer;
}

const char* ModelString(Model model, ModelTraining training) {
  for (size_t i = 0; i < kNumModelFlags; i++) {
    if (kModelTypes[i] == model && kModelTraining[i] == training)
      return kModelFlags[i];
  }
  HWY_ABORT("Unknown model %d training %d\n", static_cast<int>(model),
            static_cast<int>(training));
}

constexpr const char* kTypeStrings[] = {"f32", "bf16", "sfp"};

const char* StringFromType(Type type) {
  return kTypeStrings[static_cast<size_t>(type)];
}

const char* ParseType(const std::string& type_string, Type& type) {
  constexpr size_t kNum = std::end(kTypeStrings) - std::begin(kTypeStrings);
  static char kErrorMessageBuffer[kNum * 8 + 100] =
      "Invalid or missing type, need to specify one of ";
  for (size_t i = 0; i + 1 < kNum; i++) {
    strcat(kErrorMessageBuffer, kTypeStrings[i]);  // NOLINT
    strcat(kErrorMessageBuffer, ", ");         // NOLINT
  }
  strcat(kErrorMessageBuffer, kTypeStrings[kNum - 1]);  // NOLINT
  strcat(kErrorMessageBuffer, ".");                 // NOLINT
  std::string type_lc = type_string;
  std::transform(begin(type_lc), end(type_lc), begin(type_lc),
                 [](unsigned char c) { return std::tolower(c); });
  for (size_t i = 0; i < kNum; i++) {
    if (kTypeStrings[i] == type_lc) {
      type = static_cast<Type>(i);
      HWY_ASSERT(std::string(StringFromType(type)) == type_lc);
      return nullptr;
    }
  }
  return kErrorMessageBuffer;
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
}  // namespace gcpp
