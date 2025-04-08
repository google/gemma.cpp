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
#include <vector>

#include "gemma/gemma.h"
#include "ops/matmul.h"
#include "util/app.h"
#include "util/threading.h"
#include "hwy/base.h"

namespace gcpp {

void InitGenerator(const InferenceArgs& inference, std::mt19937& gen);

// Return type for query model calls.
struct QueryResult {
  std::string response;
  size_t tokens_generated = 0;
  // The position in the response at which the generated tokens start.
  size_t response_start_pos = 0;
};

// Convenience class to load a model and run inference.
class GemmaEnv {
 public:
  // Calls the other constructor with *Args arguments initialized from argv.
  GemmaEnv(int argc, char** argv);
  GemmaEnv(const LoaderArgs& loader, const InferenceArgs& inference,
           const AppArgs& app);

  size_t MaxGeneratedTokens() const {
    return runtime_config_.max_generated_tokens;
  }
  void SetMaxGeneratedTokens(size_t max_generated_tokens) {
    runtime_config_.max_generated_tokens = max_generated_tokens;
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

  std::vector<int> WrapAndTokenize(std::string& input) const {
    return gcpp::WrapAndTokenize(model_->Tokenizer(), model_->ChatTemplate(),
                                 model_->Info(), 0, input);
  }

  std::string StringFromTokens(const std::vector<int>& tokens) const {
    std::string string;
    HWY_ASSERT(model_->Tokenizer().Decode(tokens, &string));
    return string;
  }

  // Runs inference on the given input and returns the top-1 result string and
  // the number of tokens that were generated.
  QueryResult QueryModel(const std::vector<int>& tokens);
  std::vector<QueryResult> BatchQueryModel(
      const QueriesPromptTokens& queries_prompt);
  // Adds turn structure to input, tokenizes and calls the above overload.
  QueryResult QueryModel(std::string& input);
  std::vector<QueryResult> BatchQueryModel(
      const std::vector<std::string>& inputs);

  // Runs inference on the given input and calls the callback for each token.
  void QueryModel(const std::vector<int>& tokens,
                  const StreamFunc& stream_token);

  // Runs inference on the given input and returns the cross entropy, a measure
  // of how well the model predicts the correct output. It is the average
  // number of bits per token.
  float CrossEntropy(const std::string& input);

  // Returns nullptr if the model failed to load.
  Gemma* GetModel() const { return model_.get(); }

  int Verbosity() const { return runtime_config_.verbosity; }
  RuntimeConfig& MutableConfig() { return runtime_config_; }
  std::mt19937& MutableGen() { return gen_; }
  KVCache& MutableKVCache() { return kv_caches_[0]; }

 private:
  BoundedTopology topology_;
  NestedPools pools_;  // Thread pool.
  MatMulEnv env_;
  std::mt19937 gen_;  // Random number generator.
  std::unique_ptr<Gemma> model_;
  std::vector<KVCache> kv_caches_;  // Same number as query batch.
  RuntimeConfig runtime_config_;
};

// Logs the inference speed in tokens/sec.
void LogSpeedStats(double time_start, size_t total_tokens);

void ShowConfig(LoaderArgs& loader, InferenceArgs& inference, AppArgs& app,
                const BoundedTopology& topology, NestedPools& pools);
void ShowHelp(LoaderArgs& loader, InferenceArgs& inference, AppArgs& app);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_EVALS_BENCHMARK_HELPER_H_
