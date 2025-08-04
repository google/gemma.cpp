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
#include "third_party/gemma_cpp/gemma/gemma_args.h"  // LoaderArgs
#include "third_party/gemma_cpp/gemma/tokenizer.h"
#include "third_party/gemma_cpp/ops/matmul.h"
#include "third_party/gemma_cpp/util/threading_context.h"
#include "third_party/highway/hwy/base.h"

class SimplifiedGemma {
 public:
  SimplifiedGemma(const gcpp::LoaderArgs& loader,
                  const gcpp::ThreadingArgs& threading = gcpp::ThreadingArgs(),
                  const gcpp::InferenceArgs& inference = gcpp::InferenceArgs())
      : ctx_(threading),
        env_(ctx_),
        gemma_(loader, inference, ctx_),
        kv_cache_(gemma_.Config(), inference, ctx_.allocator) {
    // Initialize random number generator
    std::random_device rd;
    gen_.seed(rd());
  }

  SimplifiedGemma(int argc, char** argv)
      : SimplifiedGemma(gcpp::LoaderArgs(argc, argv),
                        gcpp::ThreadingArgs(argc, argv),
                        gcpp::InferenceArgs(argc, argv)) {}

  void Generate(std::string& prompt, size_t max_generated_tokens = 1024,
                float temperature = 0.7,
                const std::set<int>& reject_tokens = {}) {
    size_t generated = 0;

    const std::vector<int> tokens = gcpp::WrapAndTokenize(
        gemma_.Tokenizer(), gemma_.ChatTemplate(),
        gemma_.Config().wrapping, generated, prompt);
    const size_t prompt_size = tokens.size();

    // This callback function gets invoked every time a token is generated
    auto stream_token = [&generated, &prompt_size, this](int token, float) {
      ++generated;
      if (generated < prompt_size) {
        // print feedback
      } else if (!gemma_.Config().IsEOS(token)) {
        std::string token_text;
        HWY_ASSERT(gemma_.Tokenizer().Decode({token}, &token_text));
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
    gemma_.Generate(runtime_config, tokens, 0, kv_cache_, env_, timing_info);
  }
  ~SimplifiedGemma() = default;

 private:
  gcpp::ThreadingContext ctx_;
  gcpp::MatMulEnv env_;
  gcpp::Gemma gemma_;
  gcpp::KVCache kv_cache_;
  std::mt19937 gen_;
  std::string validation_error_;
};
