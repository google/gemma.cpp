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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_BENCHMARK_HELPER_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_BENCHMARK_HELPER_H_

#include <memory>
#include <random>
#include <string>
#include <utility>

#include "gemma/gemma.h"
#include "util/app.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

// Convenience class to load a model and run inference.
class GemmaEnv {
 public:
  GemmaEnv(int argc, char** argv);

  // Sets the maximum number of output tokens to generate.
  void set_max_generated_tokens(int max_tokens) {
    inference_args_.max_generated_tokens = max_tokens;
  }

  // Runs inference on the given input and returns the top-1 result string and
  // the number of tokens that were generated.
  std::pair<std::string, int> QueryModel(const std::string& input);

 private:
  // Logs the inference speed in tokens/sec.
  void LogSpeedStats(double time_start, size_t total_tokens) const;

  // Arguments to the model loader: file locations, etc.
  LoaderArgs loader_;
  // Arguments to the inference function: max tokens, etc.
  InferenceArgs inference_args_;
  // Controls overall behavior of the app.
  AppArgs app_;
  // Thread pool for running inference.
  hwy::ThreadPool pool_;
  // Random number generator.
  std::mt19937 gen_;
  // The model to run inference on.
  std::unique_ptr<Gemma> model_;
  // The KV cache to use for inference.
  KVCache kv_cache_;
};

}  // namespace gcpp



#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_BENCHMARK_HELPER_H_
