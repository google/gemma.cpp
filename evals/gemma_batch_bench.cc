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

#include "gemma/gemma.h"

#include <stdio.h>

#include <string>
#include <vector>

#include "evals/benchmark_helper.h"
#include "gemma/common.h"
#include "hwy/base.h"
#include "hwy/tests/hwy_gtest.h"

// This test can be run manually with the downloaded gemma weights.
// To run the test, pass the following flags:
// --model <model> --tokenizer <tokenizer_path> --weights <weights_path>
// It should pass for the following models:
// Gemma1: 2b-it (v1 and v1.1), 7b-it (v1 and v1.1), gr2b-it,
// Gemma2: gemma2-2b-it, 9b-it, 27b-it,

namespace gcpp {
namespace {

// Shared state. Requires argc/argv, so construct in main and use the same raw
// pointer approach as in benchmarks.cc. Note that the style guide forbids
// non-local static variables with dtors.
GemmaEnv* s_env = nullptr;

class GemmaTest : public ::testing::Test {
 protected:
  std::vector<std::string> BatchGemmaReply(
      const std::vector<std::string>& inputs) {
    s_env->SetMaxGeneratedTokens(64);
    s_env->MutableConfig().temperature = 0.0f;  // deterministic
    s_env->MutableConfig().verbosity = 5;
    std::vector<std::string> replies;
    // Using the turn structure worsens results sometimes.
    // However, some models need the turn structure to work.
    // It would be good to make these tests more consistent.
    if (s_env->GetModel()->Info().model == Model::GEMMA2_27B ||
        s_env->GetModel()->Info().model == Model::GRIFFIN_2B) {
      for (QueryResult result : s_env->BatchQueryModel(inputs)) {
        replies.push_back(result.response);
      }
      return replies;
    }
    // Otherwise, do not use turn structure.
    std::vector<std::vector<int>> prompts_vector;
    prompts_vector.reserve(inputs.size());
    for (const auto& input_string : inputs) {
      prompts_vector.push_back(s_env->TokenizeAndPrependBOS(input_string));
    }
    std::vector<PromptTokens> prompt_spans;
    for (const auto& prompt : prompts_vector) {
      prompt_spans.push_back(PromptTokens(prompt.data(), prompt.size()));
    }
    QueriesPromptTokens prompts(prompt_spans.data(), prompt_spans.size());
    for (const QueryResult& result : s_env->BatchQueryModel(prompts)) {
      replies.push_back(result.response);
    }
    return replies;
  }

  void GenerateTokens(std::vector<std::string> &kQA, size_t num_questions) {
    ASSERT_NE(s_env->GetModel(), nullptr);

    std::vector<std::string> inputs;
    for (size_t i = 0; i < num_questions; ++i) {
      inputs.push_back(kQA[i]);
    }
    std::vector<std::string> responses = BatchGemmaReply(inputs);
    for (size_t i = 0; i < num_questions; ++i) {
      std::string response = responses.at(i);
      fprintf(stderr, "Batch answer %zu '%s'\n\n", i + 1, response.c_str());
    }
  }
};

TEST_F(GemmaTest, RandomQuestionsBatched) {
  s_env->MutableConfig().decode_qbatch_size = 3;
  s_env->MutableConfig().verbosity = 5;

  static std::vector<std::string> kQA = {
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
  static const size_t kNum = kQA.size();
  GenerateTokens(kQA, kNum);
}
}  // namespace
}  // namespace gcpp

int main(int argc, char** argv) {
  gcpp::GemmaEnv env(argc, argv);
  gcpp::s_env = &env;

  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}


