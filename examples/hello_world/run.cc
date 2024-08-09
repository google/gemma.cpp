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

#include <stddef.h>

#include <iostream>
#include <random>
#include <string>
#include <vector>

// Placeholder for internal header, do not modify.
#include "gemma/common.h"
#include "gemma/gemma.h"
#include "gemma/tokenizer.h"
#include "util/app.h"  // LoaderArgs
#include "util/args.h"
#include "util/threading.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

int main(int argc, char** argv) {
  gcpp::LoaderArgs loader(argc, argv);
  gcpp::InferenceArgs inference(argc, argv);
  gcpp::AppArgs app(argc, argv);
  if (gcpp::HasHelp(argc, argv)) {
    loader.Help();
    return 0;
  } else if (const char* error = loader.Validate()) {
    loader.Help();
    HWY_ABORT("\nInvalid args: %s", error);
  }

  // Instantiate model and KV Cache
  gcpp::PerClusterPools pools(app.max_clusters, app.num_threads, app.pin);
  gcpp::Gemma model = gcpp::CreateGemma(loader, pools);
  gcpp::KVCache kv_cache =
      gcpp::KVCache::Create(loader.Info().model, inference.prefill_tbatch_size);
  size_t pos = 0;  // KV Cache position

  // Initialize random number generator
  std::mt19937 gen;
  std::random_device rd;
  gen.seed(rd());

  // Tokenize instructions.
  std::string prompt = "Write a greeting to the world.";
  const std::vector<int> tokens =
      gcpp::WrapAndTokenize(model.Tokenizer(), loader.Info(), pos, prompt);
  size_t ntokens = tokens.size();

  // This callback function gets invoked every time a token is generated
  auto stream_token = [&pos, &ntokens, &model](int token, float) {
    ++pos;
    if (pos < ntokens) {
      // print feedback
    } else if (token != gcpp::EOS_ID) {
      std::string token_text;
      HWY_ASSERT(model.Tokenizer().Decode({token}, &token_text));
      std::cout << token_text << std::flush;
    }
    return true;
  };

  gcpp::TimingInfo timing_info;
  gcpp::RuntimeConfig runtime_config = {
      .max_tokens = 1536,
      .max_generated_tokens = 1024,
      .temperature = 1.0,
      .verbosity = 0,
      .gen = &gen,
      .stream_token = stream_token,
  };
  model.Generate(runtime_config, tokens, 0, kv_cache, timing_info);
}
