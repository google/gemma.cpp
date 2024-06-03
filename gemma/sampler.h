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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_SAMPLER_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_SAMPLER_H_

#include <vector>

#include "gemma/prompt.h"

namespace gcpp {

class PromptSampler {
 public:
  virtual Prompt Sample(std::mt19937& gen) = 0;

  std::vector<Prompt> SampleBatch(size_t batch_size, std::mt19937& gen) {
    std::vector<Prompt> batch;
    batch.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      batch.emplace_back(Sample(gen));
    }
    return batch;
  }
};

class ReverseSequenceSampler : public PromptSampler {
 public:
  explicit ReverseSequenceSampler(const std::vector<int>& length_histo)
      : token_dist_(0, 9) {
    for (int i = 0; i < length_histo.size(); ++i) {
      const int count = length_histo[i];
      for (int j = 0; j < count; ++j) {
        length_lut_.push_back(i + 1);
      }
    }
    length_dist_ = std::uniform_int_distribution<>(0, length_lut_.size() - 1);
  }

  static constexpr int kReverseToken = 10;
  static constexpr int kEndToken = 11;

  Prompt Sample(std::mt19937& gen) override {
    Prompt prompt;
    int len = length_lut_[length_dist_(gen)];
    prompt.tokens.resize(2 * len + 2);
    prompt.tokens[len] = kReverseToken;
    prompt.tokens[2 * len + 1] = kEndToken;
    for (size_t i = 0; i < len; ++i) {
      prompt.tokens[i] = prompt.tokens[2 * len - i] = token_dist_(gen);
    }
    prompt.context_size = len + 1;
    return prompt;
  }

  static void LogPrompt(const Prompt& prompt) {
    static const char* kVocab[] = {
      "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-->", "|",
    };
    for (int token : prompt.tokens) printf("%s", kVocab[token]);
    printf("  [context_size: %zu]\n", prompt.context_size);
  }

 private:
  std::uniform_int_distribution<> token_dist_;
  std::uniform_int_distribution<> length_dist_;
  std::vector<int> length_lut_;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_SAMPLER_H_
