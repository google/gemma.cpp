// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by app_licable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stddef.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "third_party/gemma_cpp/gemma/gemma.h"
#include "third_party/gemma_cpp/gemma/tokenizer.h"
#include "third_party/gemma_cpp/util/app.h"  // LoaderArgs
#include "third_party/gemma_cpp/util/threading.h"
#include "third_party/highway/hwy/base.h"
#include "third_party/highway/hwy/contrib/thread_pool/thread_pool.h"

class SimplifiedGemma {
 public:
  SimplifiedGemma(const gcpp::LoaderArgs& loader,
                  const gcpp::InferenceArgs& inference = gcpp::InferenceArgs(),
                  const gcpp::AppArgs& app = gcpp::AppArgs())
      : loader_(loader),
        inference_(inference),
        app_(app),
        pools_(gcpp::CreatePools(app_)),
        model_(gcpp::CreateGemma(loader_, pools_)) {
    Init();
  }

  SimplifiedGemma(int argc, char** argv)
      : loader_(argc, argv, /*validate=*/true),
        inference_(argc, argv),
        app_(argc, argv),
        pools_(gcpp::CreatePools(app_)),
        model_(gcpp::CreateGemma(loader_, pools_)) {
    Init();
  }

  void Init() {
    gcpp::Allocator::Init(pools_.Topology());

    // Instantiate model and KV Cache
    kv_cache_ = gcpp::KVCache::Create(model_.GetModelConfig(),
                                      inference_.prefill_tbatch_size);

    // Initialize random number generator
    std::random_device rd;
    gen_.seed(rd());
  }

  void Generate(std::string& prompt, size_t max_generated_tokens = 1024,
                float temperature = 0.7,
                const std::set<int>& reject_tokens = {}) {
    size_t generated = 0;

    const std::vector<int> tokens = gcpp::WrapAndTokenize(
        model_.Tokenizer(), loader_.Info(), generated, prompt);
    const size_t prompt_size = tokens.size();

    // This callback function gets invoked every time a token is generated
    auto stream_token = [&generated, &prompt_size, this](int token, float) {
      ++generated;
      if (generated < prompt_size) {
        // print feedback
      } else if (token != gcpp::EOS_ID) {
        std::string token_text;
        HWY_ASSERT(this->model_.Tokenizer().Decode({token}, &token_text));
        std::cout << token_text << std::flush;
      }
      return true;
    };

    gcpp::TimingInfo timing_info;
    gcpp::RuntimeConfig runtime_config = {
        .max_generated_tokens = max_generated_tokens,
        .temperature = temperature,
        .gen = &gen_,
        .verbosity = 0,
        .stream_token = stream_token,
        .accept_token =
            [&](int token, float /* prob */) {
              return !reject_tokens.contains(token);
            },
    };
    model_.Generate(runtime_config, tokens, 0, kv_cache_, timing_info);
  }
  ~SimplifiedGemma() = default;

 private:
  gcpp::LoaderArgs loader_;
  gcpp::InferenceArgs inference_;
  gcpp::AppArgs app_;
  gcpp::NestedPools pools_;
  gcpp::Gemma model_;
  gcpp::KVCache kv_cache_;
  std::mt19937 gen_;
  std::string validation_error_;
};