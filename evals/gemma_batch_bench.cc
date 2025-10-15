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
  std::vector<std::string> BatchGemmaReply(
      const std::vector<std::string>& inputs) {
    s_env->MutableConfig().temperature = 0.0f;  // deterministic
    s_env->MutableConfig().verbosity = 2;
    std::vector<std::string> replies;
    for (const QueryResult& result : s_env->BatchQueryModel(inputs)) {
      replies.push_back(result.response);
    }
    return replies;
  }
};

TEST_F(GemmaBatchBench, RandomQuestionsBatched) {
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
  s_env->SetMaxGeneratedTokens(24);
  std::vector<std::string> responses = BatchGemmaReply(inputs);
  for (size_t i = 0; i < HWY_MIN(hwy::Unpredictable1() * 3, responses.size());
       ++i) {
    fprintf(stderr, "Batch answer %zu '%s'\n\n", i, responses[i].c_str());
  }

  PROFILER_PRINT_RESULTS();

  // Run again: prefill will be faster due to autotuning. Fewer decode steps
  // because those are already fast.
  s_env->SetMaxGeneratedTokens(2);
  responses = BatchGemmaReply(inputs);

  PROFILER_PRINT_RESULTS();
}
}  // namespace
}  // namespace gcpp

int main(int argc, char** argv) {
  fprintf(stderr, "GemmaEnv setup..\n");
  gcpp::GemmaEnv env(argc, argv);
  gcpp::s_env = &env;

  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}


