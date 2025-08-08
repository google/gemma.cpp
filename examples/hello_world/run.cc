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

#include "gemma/gemma.h"
#include "gemma/gemma_args.h"  // LoaderArgs
#include "gemma/tokenizer.h"
#include "util/args.h"
#include "util/threading_context.h"
#include "hwy/base.h"

int main(int argc, char** argv) {
  gcpp::LoaderArgs loader(argc, argv);
  gcpp::ThreadingArgs threading(argc, argv);
  gcpp::InferenceArgs inference(argc, argv);
  if (gcpp::HasHelp(argc, argv)) {
    loader.Help();
    return 0;
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
  gcpp::ThreadingContext ctx(threading);
  gcpp::MatMulEnv env(ctx);
  gcpp::Gemma gemma(loader, inference, ctx);
  gcpp::KVCache kv_cache(gemma.Config(), inference, ctx.allocator);
  size_t generated = 0;

  // Initialize random number generator
  std::mt19937 gen;
  std::random_device rd;  // NOLINT
  gen.seed(rd());

  // Tokenize instructions.
  std::string prompt = "Write a greeting to the world.";
  const std::vector<int> tokens =
      gcpp::WrapAndTokenize(gemma.Tokenizer(), gemma.ChatTemplate(),
                            gemma.Config().wrapping, generated, prompt);
  const size_t prompt_size = tokens.size();

  // This callback function gets invoked every time a token is generated
  auto stream_token = [&generated, &prompt_size, &gemma](int token, float) {
    ++generated;
    if (generated < prompt_size) {
      // print feedback
    } else if (!gemma.Config().IsEOS(token)) {
      std::string token_text;
      HWY_ASSERT(gemma.Tokenizer().Decode({token}, &token_text));
      std::cout << token_text << std::flush;
    }
    return true;
  };

  gcpp::TimingInfo timing_info;
  gcpp::RuntimeConfig runtime_config = {
      .max_generated_tokens = 1024,
      .temperature = 1.0,
      .gen = &gen,
      .verbosity = 0,
      .stream_token = stream_token,
      .accept_token =
          std::function<bool(int, float)>([&](int token, float /* prob */) {
            return reject_tokens.find(token) == reject_tokens.end();
          }),
  };
  gemma.Generate(runtime_config, tokens, 0, kv_cache, env, timing_info);
}
