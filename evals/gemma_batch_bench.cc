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
    s_env->SetMaxGeneratedTokens(24);
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
  const std::vector<std::string> questions = {
      {"Write me a poem about Australia?"},
      {"What's the history of Denmark?"},
      {"Write me a comedy story about the USA."},
      {"Teach me about GPU programming."},
      {"Write me a story about the moon."},
      {"Write me a story about the universe."},
      {"Write a poem about planet earth."},
      {"Tell me more about olympic sports."},
      {"How would you describe Washington State?"},
      {"Write me a story about Silicon Valley."},
      {"Write me about your best friend."},
      {"How would you describe a unicorn?"},
      {"Tell me about world war history."},
      {"Tell me about Google."},
      {"Explain to me how to use Google Maps."},
      {"Explain to me how AI works."},
      {"Write me a poem about France."},
      {"What's the history of Great Britain?"},
      {"Write me a comedy story about Florida."},
      {"Teach me about dynamic programming."},
      {"Write me a story about Jupiter."},
      {"Write me a story about space ships."},
      {"Write a poem about some random planet."},
      {"Tell me more about team sports."},
      {"How would you describe Michigan State?"},
      {"Write me a story about Europe."},
      {"Write me about your best colleague."},
      {"How would you describe a horse?"},
      {"Tell me about World War 2."},
      {"Please share some good cooking tips."},
      {"Tell me about space travel."},
      {"Explain to me how electric cars work."},
  };

  // Fills prompts round robin from `questions` until the desired batch size.
  std::vector<std::string> inputs;
  inputs.reserve(s_env->MutableConfig().decode_qbatch_size);
  size_t qpos = 0;
  for (size_t i = 0; i < inputs.capacity(); ++i) {
    inputs.push_back(questions[qpos++]);
    if (qpos == questions.size()) qpos = 0;
  }
  std::vector<std::string> responses = BatchGemmaReply(inputs);
  for (size_t i = 0; i < hwy::Unpredictable1() * 3; ++i) {
    fprintf(stderr, "Batch answer %zu '%s'\n\n", i, responses[i].c_str());
  }

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


