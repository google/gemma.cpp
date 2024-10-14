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
#include <stdio.h>

#include <string>

#include "benchmark/benchmark.h"
#include "evals/benchmark_helper.h"
#include "evals/prompts.h"

namespace gcpp {

// Shared state for benchmarks - unfortunately the library does not allow
// passing context nor closures. Raw pointer because style guide forbids
// non-local static objects with dtors.t
GemmaEnv* s_env = nullptr;

void RunPrompt(const std::string& original_prompt, benchmark::State& state) {
  size_t total_tokens = 0;
  for (auto s : state) {
    std::string prompt = original_prompt;  // reset from original
    QueryResult result = s_env->QueryModel(prompt);
    if (s_env->Verbosity() != 0) {
      fprintf(stdout, "|%s|\n", result.response.c_str());
    }
    total_tokens += result.tokens_generated;
  }

  state.SetItemsProcessed(total_tokens);
}

}  // namespace gcpp

static void BM_short_prompt(benchmark::State& state) {
  gcpp::RunPrompt("What is the capital of Spain?", state);
}

static void BM_factuality_prompt(benchmark::State& state) {
  gcpp::RunPrompt("How does an inkjet printer work?", state);
}

static void BM_creative_prompt(benchmark::State& state) {
  gcpp::RunPrompt("Tell me a story about a magical bunny and their TRS-80.",
                  state);
}

static void BM_coding_prompt(benchmark::State& state) {
  gcpp::RunPrompt("Write a python program to generate a fibonacci sequence.",
                  state);
}

static void BM_diff_length_prompt(benchmark::State& state) {
  gcpp::RunPrompt(GetPrompt(state.range(0)), state);
}

BENCHMARK(BM_diff_length_prompt)
    ->Iterations(3)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(BM_short_prompt)
    ->Iterations(3)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

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

int main(int argc, char** argv) {
  gcpp::GemmaEnv env(argc, argv);
  env.SetMaxGeneratedTokens(256);
  gcpp::s_env = &env;

  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
