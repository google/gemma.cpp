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

#include <random>
#include <string>
#include <vector>

#include "third_party/gemma_cpp/gemma.h"
#include "util/app.h"  // LoaderArgs
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

std::vector<int> tokenize(const std::string& prompt_string,
                          const gcpp::GemmaTokenizer* tokenizer) {
  std::string formatted = "<start_of_turn>user\n" + prompt_string +
                          "<end_of_turn>\n<start_of_turn>model\n";
  std::vector<int> tokens;
  HWY_ASSERT(tokenizer->Encode(formatted, &tokens));
  tokens.insert(tokens.begin(), BOS_ID);
  return tokens;
}

int main(int argc, char** argv) {
  gcpp::LoaderArgs loader(argc, argv);
  gcpp::AppArgs app(argc, argv);

  // Instantiate model and KV Cache
  hwy::ThreadPool pool(app.num_threads);
  gcpp::Gemma model = gcpp::CreateGemma(loader, pool);
  gcpp::KVCache kv_cache = gcpp::KVCache::Create(loader.ModelType());
  size_t pos = 0;  // KV Cache position

  // Initialize random number generator
  std::mt19937 gen;
  std::random_device rd;
  gen.seed(rd());

  // Tokenize instruction
  std::vector<int> tokens =
      tokenize("Write a greeting to the world.", model.Tokenizer());
  size_t ntokens = tokens.size();

  // This callback function gets invoked every time a token is generated
  auto stream_token = [&pos, &ntokens, tokenizer = model.Tokenizer()](int token,
                                                                      float) {
    ++pos;
    if (pos < ntokens) {
      // print feedback
    } else if (token != gcpp::EOS_ID) {
      std::string token_text;
      HWY_ASSERT(tokenizer->Decode(std::vector<int>{token}, &token_text));
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
}
