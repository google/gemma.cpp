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

#include <stdio.h>
#include <time.h>

#include <iostream>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <thread>  // NOLINT
#include <utility>  // std::pair
#include <vector>

// Placeholder for internal header, do not modify.
#include "compression/compress.h"  // TypeName
#include "gemma/common.h"          // StringFromType
#include "gemma/cross_entropy.h"
#include "gemma/gemma.h"
#include "util/app.h"
#include "util/args.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/highway.h"
#include "hwy/per_target.h"
#include "hwy/timer.h"

namespace gcpp {

void InitGenerator(const InferenceArgs& inference, std::mt19937& gen) {
  if (inference.deterministic) {
    // Nothing up my sleeve number, at least some upper bits set.
    gen.seed(0x12345678);
  } else {
    // Depending on the library implementation, this may still be deterministic.
    std::random_device rd;
    gen.seed(rd());
  }
}

GemmaEnv::GemmaEnv(int argc, char** argv)
    : loader_(argc, argv),
      inference_args_(argc, argv),
      app_(argc, argv),
      pool_(app_.num_threads) {
  {
    // Placeholder for internal init, do not modify.
  }

  // For many-core, pinning workers to cores helps.
  if (app_.num_threads > 10) {
    gcpp::PinWorkersToCores(pool_);
  }

  AbortIfInvalidArgs(inference_args_);

  if (const char* err = loader_.Validate()) {
    loader_.Help();
    fprintf(stderr, "Skipping model load because: %s\n", err);
  } else {
    fprintf(stderr, "Loading model...\n");
    model_ = AllocateGemma(loader_, pool_);
    kv_cache_ = KVCache::Create(loader_.ModelType());
  }

  InitGenerator(inference_args_, gen_);

  runtime_config_ = {
      .max_tokens = inference_args_.max_tokens,
      .max_generated_tokens = inference_args_.max_generated_tokens,
      .temperature = inference_args_.temperature,
      .verbosity = app_.verbosity,
      .gen = &gen_,
  };
}

std::pair<std::string, size_t> GemmaEnv::QueryModel(
    const std::vector<int>& tokens) {
  std::string res;
  size_t total_tokens = 0;

  const double time_start = hwy::platform::Now();
  const StreamFunc stream_token = [&res, &total_tokens, &time_start, this](
                                      int token, float) {
    ++total_tokens;
    std::string token_text;
    HWY_ASSERT(
        model_->Tokenizer().Decode(std::vector<int>{token}, &token_text));
    res += token_text;
    if (app_.verbosity >= 1 && total_tokens % 128 == 0) {
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
  runtime_config_.stream_token = stream_token;
  model_->Generate(runtime_config_, tokens, /*start_pos=*/0, kv_cache_,
                   timing_info);
  if (app_.verbosity >= 1) {
    LogSpeedStats(time_start, total_tokens);
  }
  return {res, total_tokens};
}

std::pair<std::string, size_t> GemmaEnv::QueryModel(std::string& input) {
  const std::vector<int> prompt =
      WrapAndTokenize(model_->Tokenizer(), loader_.ModelTrainingType(),
                      /*pos=*/0, input);
  return QueryModel(prompt);
}

float GemmaEnv::CrossEntropy(const std::string& input) {
  std::vector<int> prompt = Tokenize(input);
  prompt.insert(prompt.begin(), BOS_ID);
  return ComputeCrossEntropy(*GetModel(), /*max_tokens=*/3072, prompt,
                             MutableKVCache(),
                             /*verbosity=*/0) /
         static_cast<int>(input.size());
}

void LogSpeedStats(double time_start, size_t total_tokens) {
  const double time_end = hwy::platform::Now();
  const double time_elapsed = time_end - time_start;
  const double tok_sec = total_tokens / time_elapsed;
  std::cout << total_tokens << " tokens in " << time_elapsed << " seconds"
            << " [" << tok_sec << " tokens / sec" << "]\n";
}

void ShowConfig(LoaderArgs& loader, InferenceArgs& inference, AppArgs& app) {
  loader.Print(app.verbosity);
  inference.Print(app.verbosity);
  app.Print(app.verbosity);

  if (app.verbosity >= 2) {
    time_t now = time(nullptr);
    char* dt = ctime(&now);  // NOLINT
    std::cout << "Date & Time                   : " << dt
              << "Prefill Token Batch Size      : " << gcpp::kPrefillBatchSize
              << "\n"
              << "Hardware concurrency          : "
              << std::thread::hardware_concurrency() << "\n"
              << "Instruction set               : "
              << hwy::TargetName(hwy::DispatchedTarget()) << " ("
              << hwy::VectorBytes() * 8 << " bits)" << "\n";
    char cpu100[100];
    if (hwy::platform::GetCpuString(cpu100)) {
      std::cout << "CPU                           : " << cpu100 << "\n";
    }
    std::cout << "Compiled config               : " << CompiledConfig() << "\n"
              << "Weight Type                   : "
              << gcpp::StringFromType(loader.WeightType()) << "\n"
              << "EmbedderInput Type            : "
              << gcpp::TypeName(gcpp::EmbedderInputT()) << "\n";
  }
}

void ShowHelp(gcpp::LoaderArgs& loader, gcpp::InferenceArgs& inference,
              gcpp::AppArgs& app) {
  std::cerr
      << "\n\ngemma.cpp : a lightweight, standalone C++ inference engine\n"
         "==========================================================\n\n"
         "To run gemma.cpp, you need to "
         "specify 3 required model loading arguments:\n"
         "    --tokenizer\n"
         "    --weights\n"
         "    --model.\n";
  std::cerr << "\n*Example Usage*\n\n./gemma --tokenizer tokenizer.spm "
               "--weights 2b-it-sfp.sbs --model 2b-it\n";
  std::cerr << "\n*Model Loading Arguments*\n\n";
  loader.Help();
  std::cerr << "\n*Inference Arguments*\n\n";
  inference.Help();
  std::cerr << "\n*Application Arguments*\n\n";
  app.Help();
  std::cerr << "\n";
}

}  // namespace gcpp
