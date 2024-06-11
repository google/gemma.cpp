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
#include <ostream>
#include <random>
#include <sstream>
#include <string>

// Placeholder for internal header, do not modify.
#include "benchmark/benchmark.h"
#include "gemma/benchmark_helper.h"

void run_gemma_prompt(const std::string& prompt_string,
                      gcpp::GemmaEnv& env,
                      benchmark::State& state) {
  std::mt19937 gen;

  if (prompt_string.empty()) return;

  int token_counter = 0;
  for (auto s : state) {
    auto [response, n] = env.QueryModel(prompt_string);
    std::cout << "response: " << response << "\n";
    std::cout << "n: " << n << "\n";
    token_counter += n;
  }

  state.SetItemsProcessed(token_counter);
}

// Awkward global because benchmarks don't support additional state, so it is
// either this or cast to int64_t.
gcpp::GemmaEnv* global_env = nullptr;

static void BM_short_prompt(benchmark::State& state) {
  run_gemma_prompt("What is the capital of Spain?", *global_env,
                   state);
}

static void BM_factuality_prompt(benchmark::State& state) {
  run_gemma_prompt("How does an inkjet printer work?",
                   *global_env, state);
}

static void BM_creative_prompt(benchmark::State& state) {
  run_gemma_prompt(
      "Tell me a story about a magical bunny and their TRS-80.",
      *global_env, state);
}

static void BM_coding_prompt(benchmark::State& state) {
  run_gemma_prompt(
      "Write a python program to generate a fibonacci sequence.",
      *global_env, state);
}

static void BM_long_coding_prompt(benchmark::State& state) {
  std::ifstream t("benchmarks.cc", std::ios_base::in);
  std::stringstream buffer;
  buffer << t.rdbuf();
  std::string prompt_string = buffer.str();
  t.close();

  run_gemma_prompt("Make improvements to the following code:\n " +
                   prompt_string, *global_env, state);
}

int main(int argc, char** argv) {
  {
    // Placeholder for internal init, do not modify.
  }
  gcpp::GemmaEnv env(argc, argv);

  env.set_max_generated_tokens(128);
  global_env = &env;
  BENCHMARK(BM_short_prompt)
      ->Iterations(3)
      ->Unit(benchmark::kMillisecond)
      ->UseRealTime();

  env.set_max_generated_tokens(256);
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

  env.set_max_generated_tokens(1024);
  BENCHMARK(BM_long_coding_prompt)
      ->Iterations(3)
      ->Unit(benchmark::kMillisecond)
      ->UseRealTime();

  ::benchmark ::RunSpecifiedBenchmarks();
  ::benchmark ::Shutdown();
  return 0;
}
