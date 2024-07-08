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

#include "evals/benchmark_helper.h"

#include <stdio.h>
#include <time.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <thread>   // NOLINT
#include <utility>  // std::pair
#include <vector>

// Placeholder for internal header, do not modify.
#include "compression/compress.h"  // TypeName
#include "evals/cross_entropy.h"
#include "gemma/common.h"  // StringFromType
#include "gemma/gemma.h"
#include "util/app.h"
#include "util/args.h"
#include "hwy/aligned_allocator.h"
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

GemmaEnv::GemmaEnv(const LoaderArgs& loader, const InferenceArgs& inference,
                   const AppArgs& app)
    : loader_(loader),
      inference_args_(inference),
      app_(app),
      pool_(app_.num_threads) {
  // For many-core, pinning workers to cores helps.
  if (app_.num_threads > 10) {
    PinWorkersToCores(pool_);
  }

  AbortIfInvalidArgs(inference_args_);

  if (const char* err = loader_.Validate()) {
    loader_.Help();
    fprintf(stderr, "Skipping model load because: %s\n", err);
  } else {
    fprintf(stderr, "Loading model...\n");
    model_ = AllocateGemma(loader_, pool_);

    kv_caches_.reserve(16);
    for (int i = 0; i < 16; ++i) {
      kv_caches_.push_back(new KVCache(KVCache::Create(model_->Info().model)));
    }
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

// Internal init must run before the GemmaEnv ctor above, hence it cannot occur
// in the argv ctor below because its body runs *after* the delegating ctor.
// This helper function takes care of the init, and could be applied to any of
// the *Args classes, it does not matter which.
static AppArgs MakeAppArgs(int argc, char** argv) {
  {  // So that indentation matches expectations.
    // Placeholder for internal init, do not modify.
  }
  return AppArgs(argc, argv);
}

GemmaEnv::GemmaEnv(int argc, char** argv)
    : GemmaEnv(LoaderArgs(argc, argv), InferenceArgs(argc, argv),
               MakeAppArgs(argc, argv)) {}

std::pair<std::string, size_t> GemmaEnv::QueryModel(
    const std::vector<int>& tokens) {
  std::string res;
  size_t total_tokens = 0;

  const double time_start = hwy::platform::Now();
  const BatchStreamFunc batch_stream_token =
      [&res, &total_tokens, &time_start, this](
          size_t query_index, size_t pos, int token, float) {
    ++total_tokens;
    res += StringFromTokens(std::vector<int>{token});
    if (app_.verbosity >= 1 && total_tokens % 128 == 0) {
      LogSpeedStats(time_start, total_tokens);
    }
    return true;
  };
  if (app_.verbosity >= 2) {
    std::cout << "Max tokens: " << inference_args_.max_tokens
              << "\tmax generated tokens: "
              << inference_args_.max_generated_tokens
              << "\ttemperature: " << inference_args_.temperature << "\n";
  }
  gcpp::TimingInfo timing_info;
  runtime_config_.batch_stream_token = batch_stream_token;
  model_->Generate(runtime_config_, tokens, /*start_pos=*/0, *kv_caches_[0],
                   timing_info);
  if (app_.verbosity >= 1) {
    LogSpeedStats(time_start, total_tokens);
  }
  return {res, total_tokens};
}

std::vector<std::pair<std::string, size_t>> GemmaEnv::BatchQueryModel2(
    const hwy::Span<const hwy::Span<int>>& prompts) {
  std::vector<std::pair<std::string, size_t>> res(prompts.size());
  std::fill(res.begin(), res.end(), std::make_pair("", 0));
  size_t total_tokens = 0;

  const double time_start = hwy::platform::Now();
  const BatchStreamFunc batch_stream_token =
      [&res, &total_tokens, &time_start, this](
          size_t query_index, size_t pos, int token, float) {
    std::string token_text;
    HWY_ASSERT(
        model_->Tokenizer().Decode(std::vector<int>{token}, &token_text));
    res[query_index].first.append(token_text);
    res[query_index].second += 1;
    ++total_tokens;
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
  runtime_config_.batch_stream_token = batch_stream_token;
  model_->GenerateBatch(runtime_config_, prompts, /*start_pos=*/0, kv_caches_,
                        timing_info);
  if (app_.verbosity >= 1) {
    LogSpeedStats(time_start, total_tokens);
  }
  return res;
}

std::pair<std::string, size_t> GemmaEnv::QueryModel(std::string& input) {
  const std::vector<int> prompt = WrapAndTokenize(model_->Tokenizer(), Info(),
                                                  /*pos=*/0, input);
  return QueryModel(prompt);
}

std::vector<std::pair<std::string, size_t>> GemmaEnv::BatchQueryModel(
    const std::vector<std::string>& inputs) {
  std::vector<std::vector<int>> prompts;
  prompts.reserve(inputs.size());
  for (auto& input : inputs) {
    std::string mutable_prompt = input;
    prompts.push_back(WrapAndTokenize(model_->Tokenizer(), model_->Info(),
                                      /*pos=*/0, mutable_prompt));
  }
  std::vector<hwy::Span<int>> prompt_vector;
  prompt_vector.reserve(prompts.size());
  for (auto& prompt : prompts) {
    prompt_vector.push_back(hwy::Span<int>(prompt.data(), prompt.size()));
  }
  hwy::Span<const hwy::Span<int>> prompt_span = hwy::Span<const hwy::Span<int>>(
      prompt_vector.data(), prompt_vector.size());
  return BatchQueryModel2(prompt_span);
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
              << "Prefill Token Batch Size      : " << kPrefillBatchSize << "\n"
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
              << StringFromType(loader.Info().weight) << "\n"
              << "EmbedderInput Type            : "
              << TypeName(EmbedderInputT()) << "\n";
  }
}

void ShowHelp(LoaderArgs& loader, InferenceArgs& inference, AppArgs& app) {
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
