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

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

// Placeholder for internal header, do not modify.
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

  // Demonstrate constrained decoding by never outputting certain tokens.
  std::set<int> reject_tokens;
  for (int arg = 0; arg < argc; ++arg) {
    // Find a --reject flag and consume everything after it.
    if (strcmp(argv[arg], "--reject") == 0) {
      while (++arg < argc) reject_tokens.insert(atoi(argv[arg]));
    }
  }

  // Instantiate model and KV Cache
  gcpp::PerClusterPools pools(app.max_clusters, app.max_threads, app.pin);
  gcpp::Gemma model = gcpp::CreateGemma(loader, pools);
  gcpp::KVCache kv_cache =
      gcpp::KVCache::Create(model.GetModelConfig(),
                            inference.prefill_tbatch_size);
  size_t generated = 0;

  // Initialize random number generator
  std::mt19937 gen;
  std::random_device rd;
  gen.seed(rd());

  // Tokenize instructions.
  std::string prompt = "Write a greeting to the world.";
  const std::vector<int> tokens = gcpp::WrapAndTokenize(
      model.Tokenizer(), loader.Info(), generated, prompt);
  const size_t prompt_size = tokens.size();

  // This callback function gets invoked every time a token is generated
  auto stream_token = [&generated, &prompt_size, &model](int token, float) {
    ++generated;
    if (generated < prompt_size) {
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
      .max_generated_tokens = 1024,
      .temperature = 1.0,
      .verbosity = 0,
      .gen = &gen,
      .stream_token = stream_token,
      .accept_token =
          [&](int token, float /* prob */) {
            return !reject_tokens.contains(token);
          },
  };
  model.Generate(runtime_config, tokens, 0, kv_cache, timing_info);
}
