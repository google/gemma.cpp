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

#include "gemma/benchmark_helper.h"
#include <cstdlib>  // EXIT_FAILURE
#include <iostream>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <utility>  // std::pair
#include <vector>

#include "gemma/common.h"
#include "gemma/gemma.h"
#include "util/app.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/highway.h"
#include "hwy/timer.h"

namespace gcpp {
  GemmaEnv::GemmaEnv(int argc, char** argv)
      : loader_(argc, argv), inference_args_(argc, argv), app_(argc, argv),
        pool_(app_.num_threads) {
    if (const char* error = loader_.Validate()) {
      HWY_ABORT("\nInvalid loader args: %s", error);
    }
    if (const char* error = inference_args_.Validate()) {
      HWY_ABORT("\nInvalid inference args: %s", error);
    }
    // For many-core, pinning workers to cores helps.
    if (app_.num_threads > 10) {
      gcpp::PinWorkersToCores(pool_);
    }
    model_ = AllocateGemma(loader_, pool_);
    kv_cache_ = KVCache::Create(loader_.ModelType());
    gen_.seed(42);
  }

std::pair<std::string, int> GemmaEnv::QueryModel(const std::string& input) {
  std::string prompt_string = input;
  if (loader_.ModelTrainingType() == ModelTraining::GEMMA_IT) {
    // For instruction-tuned models: add control tokens.
    prompt_string = "<start_of_turn>user\n" + input +
                    "<end_of_turn>\n<start_of_turn>model\n";
  }
  std::vector<int> prompt;
  HWY_ASSERT(model_->Tokenizer().Encode(input, &prompt));

  // For both pre-trained and instruction-tuned models: prepend "<bos>" token
  // if needed.
  prompt.insert(prompt.begin(), gcpp::BOS_ID);
  std::string res;
  size_t total_tokens = 0;
  auto accept_token = [](int) { return true; };
  std::mt19937 gen;
  gen.seed(42);

  const double time_start = hwy::platform::Now();
  auto stream_token = [&res, &total_tokens, &time_start, this](
                          int token, float) {
    ++total_tokens;
    std::string token_text;
    HWY_ASSERT(model_->Tokenizer().Decode(std::vector<int>{token},
                                          &token_text));
    res += token_text;
    if (app_.verbosity >= 1 && total_tokens % 100 == 0) {
      LogSpeedStats(time_start, total_tokens);
    }
    return true;
  };
  if (app_.verbosity >= 2) {
    std::cout << inference_args_.max_tokens << " "
              << inference_args_.max_generated_tokens << " "
              << inference_args_.temperature;
  }
  gcpp::TimingInfo timing_info;
  gcpp::RuntimeConfig runtime_config = {
      .max_tokens = inference_args_.max_tokens,
      .max_generated_tokens = inference_args_.max_generated_tokens,
      .temperature = inference_args_.temperature,
      .verbosity = app_.verbosity,
      .gen = &gen,
      .stream_token = stream_token,
      .accept_token = accept_token,
  };
  model_->Generate(runtime_config, prompt, /*start_pos=*/0, kv_cache_,
                  timing_info, /*layers_output=*/nullptr);
  if (app_.verbosity >= 1) {
    LogSpeedStats(time_start, total_tokens);
  }
  return {res, total_tokens};
}

void GemmaEnv::LogSpeedStats(double time_start, size_t total_tokens) const {
  const double time_end = hwy::platform::Now();
  const double time_elapsed = time_end - time_start;
  const double tok_sec = total_tokens / time_elapsed;
  std::cout << total_tokens << " tokens in " << time_elapsed << " seconds"
            << " [" << tok_sec << " tokens / sec" << "]\n";
}


}  // namespace gcpp

