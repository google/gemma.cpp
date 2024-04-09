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

#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "third_party/benchmark/include/benchmark/benchmark.h"
#include "gemma/gemma.h"
#include "util/app.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

gcpp::LoaderArgs* loader = nullptr;
gcpp::InferenceArgs* inference = nullptr;
gcpp::Gemma* model = nullptr;
hwy::ThreadPool* pool = nullptr;
hwy::ThreadPool* inner_pool = nullptr;

void run_gemma_prompt(const std::string& prompt_string,
                      benchmark::State& state) {
  std::mt19937 gen;
  std::vector<int> prompt;

  if (prompt_string.empty()) return;
  HWY_ASSERT(model->Tokenizer().Encode(prompt_string, &prompt).ok());

  int token_counter = 0;
  auto stream_token = [&token_counter](int, float) {
    token_counter++;
    return true;
  };

  for (auto s : state) {
    GenerateGemma(
        *model, *inference, prompt, /*start_token=*/0, *pool, *inner_pool,
        stream_token,
        /*accept=*/[](int) { return true; }, gen, /*verbosity=*/0);
  }

  state.SetItemsProcessed(token_counter);
}

static void BM_short_prompt(benchmark::State& state) {
  run_gemma_prompt("What is the capital of Spain?<ctrl23> ", state);
}

static void BM_factuality_prompt(benchmark::State& state) {
  run_gemma_prompt("How does an inkjet printer work?<ctrl23> ", state);
}

static void BM_creative_prompt(benchmark::State& state) {
  run_gemma_prompt(
      "Tell me a story about a magical bunny and their TRS-80.<ctrl23> ",
      state);
}

static void BM_coding_prompt(benchmark::State& state) {
  run_gemma_prompt(
      "Write a python program to generate a fibonacci sequence.<ctrl23> ",
      state);
}

static void BM_long_coding_prompt(benchmark::State& state) {
  std::ifstream t("benchmarks.cc", std::ios_base::in);
  std::stringstream buffer;
  buffer << t.rdbuf();
  std::string prompt_string = buffer.str();
  t.close();

  run_gemma_prompt("Make improvements to the following code:\n " +
                       prompt_string + "<ctrl23> ",
                   state);
}

int main(int argc, char** argv) {
  loader = new gcpp::LoaderArgs(argc, argv);
  inference = new gcpp::InferenceArgs(argc, argv);
  gcpp::AppArgs app(argc, argv);

  pool = new ::hwy::ThreadPool(app.num_threads);
  inner_pool = new ::hwy::ThreadPool(0);
  model = new gcpp::Gemma(*loader, *pool);

  inference->max_tokens = 128;
  BENCHMARK(BM_short_prompt)
      ->Iterations(3)
      ->Unit(benchmark::kMillisecond)
      ->UseRealTime();

  inference->max_tokens = 256;
  BENCHMARK(BM_factuality_prompt)
      ->Iterations(3)
      ->Unit(benchmark::kMillisecond)
      ->UseRealTime();

  BENCHMARK(BM_creative_prompt)
      ->Iterations(3)
      ->Unit(benchmark::kMillisecond)
      ->UseRealTime();

  BENCHMARK(BM_coding_prompt)
      ->Iterations(3)
      ->Unit(benchmark::kMillisecond)
      ->UseRealTime();

  inference->max_tokens = 1024;
  BENCHMARK(BM_long_coding_prompt)
      ->Iterations(3)
      ->Unit(benchmark::kMillisecond)
      ->UseRealTime();

  ::benchmark ::RunSpecifiedBenchmarks();
  ::benchmark ::Shutdown();

  delete loader;
  delete inference;
  delete model;
  delete pool;

  return 0;
}
