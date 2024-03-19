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

#include <iostream>

// copybara:import_next_line:gemma_cpp
#include "gemma.h"
// copybara:end
// copybara:import_next_line:gemma_cpp
#include "util/args.h"
// copybara:end
// copybara:import_next_line:gemma_cpp
#include "util/app.h" // LoaderArgs
// copybara:end
#include "hwy/contrib/thread_pool/thread_pool.h"

std::vector<int> tokenize(
    const std::string& prompt_string,
    const sentencepiece::SentencePieceProcessor* tokenizer) {
  std::string formatted = "<start_of_turn>user\n" + prompt_string +
                          "<end_of_turn>\n<start_of_turn>model\n";
  std::vector<int> tokens;
  HWY_ASSERT(tokenizer->Encode(formatted, &tokens).ok());
  tokens.insert(tokens.begin(), 2);  // BOS token
  return tokens;
}

int main(int argc, char** argv) {
  gcpp::LoaderArgs loader(argc, argv);

  // Rough heuristic for the number of threads to use
  size_t num_threads = static_cast<size_t>(std::clamp(
      static_cast<int>(std::thread::hardware_concurrency()) - 2, 1, 18));
  hwy::ThreadPool pool(num_threads);

  // Instantiate model and KV Cache
  gcpp::Gemma model(loader.tokenizer, loader.compressed_weights,
                    loader.ModelType(), pool);
  auto kv_cache = CreateKVCache(loader.ModelType());
  size_t pos = 0;  // KV Cache position

  // Initialize random number generator
  std::mt19937 gen;
  std::random_device rd;
  gen.seed(rd());

  // Tokenize instruction
  std::vector<int> tokens =
      tokenize("Write a greeting to the world.", model.Tokenizer());
  size_t ntokens = tokens.size();

  // This callback function gets invoked everytime a token is generated
  auto stream_token = [&pos, &gen, &ntokens, tokenizer = model.Tokenizer()](
                          int token, float) {
    ++pos;
    if (pos < ntokens) {
      // print feedback
    } else if (token != gcpp::EOS_ID) {
      std::string token_text;
      HWY_ASSERT(tokenizer->Decode(std::vector<int>{token}, &token_text).ok());
      std::cout << token_text << std::flush;
    }
    return true;
  };

  GenerateGemma(model,
                {.max_tokens = 2048,
                 .max_generated_tokens = 1024,
                 .temperature = 1.0,
                 .verbosity = 0},
                tokens, /*KV cache position = */ 0, kv_cache, pool,
                stream_token, gen);
  std::cout << std::endl;
}
