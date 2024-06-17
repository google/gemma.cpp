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

const char* ParseModelTypeAndTraining(const std::string& model_flag,
                                      Model& model, ModelTraining& training) {
  constexpr const char* kModelFlags[] = {
      "2b-pt", "7b-pt", "gr2b-pt", "2b-it", "7b-it", "gr2b-it", "tiny",
  };
  constexpr Model kModelTypes[] = {
      Model::GEMMA_2B, Model::GEMMA_7B, Model::GRIFFIN_2B, Model::GEMMA_2B,
      Model::GEMMA_7B, Model::GRIFFIN_2B, Model::GEMMA_TINY,
  };
  constexpr ModelTraining kModelTraining[] = {
      ModelTraining::GEMMA_PT, ModelTraining::GEMMA_PT, ModelTraining::GEMMA_PT,
      ModelTraining::GEMMA_IT, ModelTraining::GEMMA_IT, ModelTraining::GEMMA_IT,
      ModelTraining::GEMMA_IT,
  };

  constexpr size_t kNum = std::end(kModelFlags) - std::begin(kModelFlags);
  static char kErrorMessageBuffer[kNum * 8 + 1024] =
      "Invalid or missing model flag, need to specify one of ";
  for (size_t i = 0; i + 1 < kNum; i++) {
    strcat(kErrorMessageBuffer, kModelFlags[i]);  // NOLINT
    strcat(kErrorMessageBuffer, ", ");            // NOLINT
  }
  strcat(kErrorMessageBuffer, kModelFlags[kNum - 1]);  // NOLINT
  strcat(kErrorMessageBuffer, ".");                    // NOLINT

  std::string model_type_lc = model_flag;
  std::transform(begin(model_type_lc), end(model_type_lc), begin(model_type_lc),
                 [](unsigned char c) { return std::tolower(c); });

  for (size_t i = 0; i < kNum; i++) {
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
  if (model == Model::GEMMA_TINY) return "tiny";
  static_assert(static_cast<size_t>(ModelTraining::GEMMA_IT) == 0);
  constexpr const char* k2B[] = {"2b-it", "2b-pt"};
  constexpr const char* k7B[] = {"7b-it", "7b-pt"};
  constexpr const char* kGr2B[] = {"gr2b-it", "gr2b-pt"};
  if (model == Model::GEMMA_2B) return k2B[static_cast<size_t>(training)];
  if (model == Model::GEMMA_7B) return k7B[static_cast<size_t>(training)];
  if (model == Model::GRIFFIN_2B) return kGr2B[static_cast<size_t>(training)];
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

}  // namespace gcpp
