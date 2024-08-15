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

#ifndef THIRD_PARTY_GEMMA_CPP_EVALS_BENCHMARK_HELPER_H_
#define THIRD_PARTY_GEMMA_CPP_EVALS_BENCHMARK_HELPER_H_

#include <stddef.h>

#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "gemma/gemma.h"
#include "util/app.h"
#include "util/threading.h"
#include "hwy/base.h"

namespace gcpp {

void InitGenerator(const InferenceArgs& inference, std::mt19937& gen);

// Convenience class to load a model and run inference.
class GemmaEnv {
 public:
  // Calls the other constructor with *Args arguments initialized from argv.
  GemmaEnv(int argc, char** argv);
  GemmaEnv(const LoaderArgs& loader, const InferenceArgs& inference,
           const AppArgs& app);

  size_t MaxTokens() const { return inference_args_.max_tokens; }
  // Sets the maximum number of output tokens to generate.
  void SetMaxGeneratedTokens(size_t max_tokens) {
    inference_args_.max_generated_tokens = max_tokens;
  }

  std::vector<int> Tokenize(const std::string& input) const {
    std::vector<int> tokens;
    HWY_ASSERT(model_->Tokenizer().Encode(input, &tokens));
    return tokens;
  }

  std::vector<int> TokenizeAndPrependBOS(const std::string& input) const {
    std::vector<int> tokens = Tokenize(input);
    tokens.insert(tokens.begin(), BOS_ID);
    return tokens;
  }

  std::string StringFromTokens(const std::vector<int>& tokens) const {
    std::string string;
    HWY_ASSERT(model_->Tokenizer().Decode(tokens, &string));
    return string;
  }

  // Runs inference on the given input and returns the top-1 result string and
  // the number of tokens that were generated.
  std::pair<std::string, size_t> QueryModel(const std::vector<int>& tokens);
  std::vector<std::pair<std::string, size_t>> BatchQueryModel2(
      const QueriesPromptTokens& queries_prompt);
  // Adds turn structure to input, tokenizes and calls the above overload.
  std::pair<std::string, size_t> QueryModel(std::string& input);
  std::vector<std::pair<std::string, size_t>> BatchQueryModel(
      const std::vector<std::string>& inputs);

  // Runs inference on the given input and returns the cross entropy, a measure
  // of how well the model predicts the correct output. It is the average
  // number of bits per token.
  float CrossEntropy(const std::string& input);

  // Returns nullptr if the model failed to load.
  Gemma* GetModel() const { return model_.get(); }

  int Verbosity() const { return app_.verbosity; }
  RuntimeConfig& MutableConfig() { return runtime_config_; }
  const ModelInfo& Info() const { return loader_.Info(); }
  InferenceArgs& MutableInferenceArgs() { return inference_args_; }
  std::mt19937& MutableGen() { return gen_; }
  KVCache& MutableKVCache() { return kv_caches_[0]; }

 private:
  // Arguments to the model loader: file locations, etc.
  LoaderArgs loader_;
  // Arguments to the inference function: max tokens, etc.
  InferenceArgs inference_args_;
  // Controls overall behavior of the app.
  AppArgs app_;
  // Thread pool for running inference.
  PerClusterPools pools_;
  // Random number generator.
  std::mt19937 gen_;
  // The model to run inference on.
  std::unique_ptr<Gemma> model_;
  // KV caches, same number as query batch.
  std::vector<KVCache> kv_caches_;
  RuntimeConfig runtime_config_;
};

// Logs the inference speed in tokens/sec.
void LogSpeedStats(double time_start, size_t total_tokens);

void ShowConfig(LoaderArgs& loader, InferenceArgs& inference, AppArgs& app,
                PerClusterPools& pools);
void ShowHelp(LoaderArgs& loader, InferenceArgs& inference, AppArgs& app);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_EVALS_BENCHMARK_HELPER_H_
