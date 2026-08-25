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

#include <stdio.h>

#include <algorithm>
#include <string>
#include <vector>

#include "evals/benchmark_helper.h"
#include "gemma/gemma.h"
#include "hwy/base.h"
#include "hwy/nanobenchmark.h"
#include "hwy/profiler.h"
#include "hwy/tests/hwy_gtest.h"

namespace gcpp {
namespace {

// Shared state. Requires argc/argv, so construct in main and use the same raw
// pointer approach as in benchmarks.cc. Note that the style guide forbids
// non-local static variables with dtors.
GemmaEnv* s_env = nullptr;

class GemmaBatchBench : public ::testing::Test {
 protected:
  QueryResultAndMetrics BatchGemmaReplyWithMetrics(
      const std::vector<std::string>& inputs) {
    s_env->MutableConfig().temperature = 0.0f;  // deterministic
    s_env->MutableConfig().verbosity = 2;
    return s_env->BatchQueryModelWithMetrics(inputs);
  }
};

std::vector<std::string> GenerateInputs() {
  std::vector<std::string> prompts = {
      {"Describe dynamic programming."},
      {"Explain how electric cars work."},
      {"Explain to me how to use Google Maps."},
      {"How does AI work?"},
      {"How would you describe a unicorn?"},
      {"Please share some good cooking tips."},
      {"Teach me about GPU programming."},
      {"Tell me a fact about World War 2."},
      {"Tell me about Google."},
      {"Tell me more about olympic sports."},
      {"Tell me something about space travel."},
      {"What is a horse?"},
      {"What is Michigan State?"},
      {"What's the history of Denmark?"},
      {"Write a poem about planet earth."},
      {"Write a story about Jupiter."},
      {"Write about the moon."},
      {"Write me a comedy story about Florida."},
      {"Write me a poem about France."},
  };
  const std::vector<std::string> start = {
      {"What is"}, {"When did"}, {"Where did"}, {"How did"}, {"Why did"}};
  const std::vector<std::string> concepts = {"Socrates",
                                             "Einstein",
                                             "Leonardo",
                                             "Cleopatra",
                                             "Adele",
                                             "Mars",
                                             "Turing",
                                             "Mozart",
                                             "democracy",
                                             "gravity",
                                             "AI",
                                             "evolution",
                                             "physics",
                                             "the internet",
                                             "steam engine",
                                             "inflation",
                                             "electricity",
                                             "the Sahara",
                                             "NASA",
                                             "Rome",
                                             "the UN",
                                             "Google",
                                             "the Renaissance",
                                             "Hamlet",
                                             "poetry",
                                             "Stoicism",
                                             "geometry",
                                             "DNA",
                                             "Star Wars",
                                             "1984"};
  const std::vector<std::string> end = {"exist?",   "work?",    "happen?",
                                        "lead to?", "believe?", "result in?"};
  for (const std::string& s : start) {
    for (const std::string& c : concepts) {
      for (const std::string& e : end) {
        prompts.push_back(s + " " + c + " " + e);
      }
    }
  }
  AesCtrEngine engine(true);
  std::shuffle(prompts.begin(), prompts.end(), RngStream(engine, 123));

  // Fills `inputs` by repeating from `prompts` until the desired batch size.
  std::vector<std::string> inputs;
  inputs.reserve(s_env->MutableConfig().decode_qbatch_size);
  size_t qpos = 0;
  for (size_t i = 0; i < inputs.capacity(); ++i) {
    inputs.push_back(prompts[qpos++]);
    if (qpos == prompts.size()) qpos = 0;
  }
  return inputs;
}

TEST_F(GemmaBatchBench, RandomQuestionsBatched) {
  s_env->SetMaxGeneratedTokens(12);
  const std::vector<std::string> inputs = GenerateInputs();
  constexpr size_t kNumReps = 7;

  std::vector<double> prefill_speeds;
  prefill_speeds.reserve(kNumReps);
  std::vector<double> generate_speeds;
  generate_speeds.reserve(kNumReps);

  size_t total_prefill_tokens = 0;
  double total_prefill_duration = 0.0;
  size_t total_generate_tokens = 0;
  double total_generate_duration = 0.0;

  size_t warm_prefill_tokens = 0;
  double warm_prefill_duration = 0.0;
  size_t warm_generate_tokens = 0;
  double warm_generate_duration = 0.0;

  for (size_t rep = 0; rep < kNumReps; ++rep) {
    QueryResultAndMetrics result = BatchGemmaReplyWithMetrics(inputs);
    const std::vector<QueryResult>& responses = result.query_results;
    const TimingInfo& timing = result.timing_info;

    const double prefill_tok_sec =
        timing.prefill_duration > 0.0
            ? static_cast<double>(timing.prefill_tokens) /
                  timing.prefill_duration
            : 0.0;
    const double gen_tok_sec =
        timing.generate_duration > 0.0
            ? static_cast<double>(timing.tokens_generated) /
                  timing.generate_duration
            : 0.0;

    prefill_speeds.push_back(prefill_tok_sec);
    generate_speeds.push_back(gen_tok_sec);

    total_prefill_tokens += timing.prefill_tokens;
    total_prefill_duration += timing.prefill_duration;
    total_generate_tokens += timing.tokens_generated;
    total_generate_duration += timing.generate_duration;

    if (rep > 0) {
      warm_prefill_tokens += timing.prefill_tokens;
      warm_prefill_duration += timing.prefill_duration;
      warm_generate_tokens += timing.tokens_generated;
      warm_generate_duration += timing.generate_duration;
    }

    for (size_t i = 0; i < HWY_MIN(hwy::Unpredictable1() * 3, responses.size());
         ++i) {
      fprintf(stderr, "Rep %zu batch answer %zu '%s'\n\n", rep, i,
              responses[i].response.c_str());
    }
    PROFILER_PRINT_RESULTS();
  }

  const double avg_prefill =
      total_prefill_duration > 0.0
          ? static_cast<double>(total_prefill_tokens) / total_prefill_duration
          : 0.0;
  const double avg_generate =
      total_generate_duration > 0.0
          ? static_cast<double>(total_generate_tokens) / total_generate_duration
          : 0.0;

  const double warm_avg_prefill =
      warm_prefill_duration > 0.0
          ? static_cast<double>(warm_prefill_tokens) / warm_prefill_duration
          : 0.0;
  const double warm_avg_generate =
      warm_generate_duration > 0.0
          ? static_cast<double>(warm_generate_tokens) / warm_generate_duration
          : 0.0;

  fprintf(stderr,
          "\n============================================================\n");
  fprintf(stderr,
          "[ Gemma Batch Benchmark Summary (%zu Repetitions) ]\n", kNumReps);
  for (size_t rep = 0; rep < kNumReps; ++rep) {
    fprintf(stderr,
            "  Rep %zu: Prefill = %7.2f tok/s | Generate = %7.2f tok/s%s\n",
            rep, prefill_speeds[rep], generate_speeds[rep],
            rep == 0 ? " (warmup / autotune)" : "");
  }
  fprintf(stderr,
          "------------------------------------------------------------\n");
  fprintf(stderr,
          "Overall Average: Prefill = %7.2f tok/s | Generate = %7.2f tok/s\n",
          avg_prefill, avg_generate);
  if (kNumReps > 1) {
    fprintf(stderr,
            "Warm Average   : Prefill = %7.2f tok/s | Generate = %7.2f tok/s\n",
            warm_avg_prefill, warm_avg_generate);
  }
  fprintf(stderr,
          "============================================================\n\n");
}

}  // namespace
}  // namespace gcpp

int main(int argc, char** argv) {
  fprintf(stderr, "GemmaEnv setup..\n");
  gcpp::ConsumedArgs consumed(argc, argv);
  gcpp::GemmaArgs args(argc, argv, consumed);
  consumed.AbortIfUnconsumed();

  gcpp::GemmaEnv env(args);
  gcpp::s_env = &env;

  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
