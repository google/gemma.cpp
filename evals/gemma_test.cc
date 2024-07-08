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

#include <memory>
#include <string>
#include <vector>

#include "evals/benchmark_helper.h"
#include "gemma/common.h"
#include "hwy/aligned_allocator.h"
#include "hwy/tests/hwy_gtest.h"

// This test can be run manually with the downloaded gemma weights.
// To run the test, pass the following flags:
// --model <model> --tokenizer <tokenizer_path> --weights <weights_path>
// It should pass for the following models: 2b-it, 7b-it, 9b-it, 27b-it

namespace gcpp {
namespace {

// Shared state. Requires argc/argv, so construct in main and use the same raw
// pointer approach as in benchmarks.cc. Note that the style guide forbids
// non-local static variables with dtors.
GemmaEnv* s_env = nullptr;

class GemmaTest : public ::testing::Test {
 protected:
  std::string GemmaReply(const std::string& prompt) {
    s_env->SetMaxGeneratedTokens(2048);
    s_env->MutableConfig().temperature = 0.0f;  // deterministic
    s_env->MutableConfig().verbosity = 0;
    // Using the turn structure worsens results.
    const std::vector<int> tokens = s_env->TokenizeAndPrependBOS(prompt);
    auto [response, n] = s_env->QueryModel(tokens);
    return response;
  }

  std::vector<std::string> BatchGemmaReply(
      const std::vector<std::string>& inputs) {
    s_env->SetMaxGeneratedTokens(64);
    s_env->MutableConfig().temperature = 0.0f;  // deterministic
    s_env->MutableConfig().verbosity = 0;
    // Using the turn structure worsens results.
    std::vector<std::unique_ptr<std::vector<int>>> prompts;
    prompts.reserve(inputs.size());
    for (auto input_string : inputs) {
      std::string mutable_input_string = input_string;
      prompts.push_back(std::make_unique<std::vector<int>>(
          s_env->TokenizeAndPrependBOS(input_string)));
    }
    std::vector<hwy::Span<int>> prompt_vector;
    for (auto& prompt : prompts) {
      prompt_vector.push_back(hwy::Span<int>(prompt->data(), prompt->size()));
    }
    hwy::Span<const hwy::Span<int>> prompt_span =
        hwy::Span<const hwy::Span<int>>(prompt_vector.data(),
                                        prompt_vector.size());
    std::vector<std::string> replies;
    for (auto [response, n] : s_env->BatchQueryModel2(prompt_span)) {
      replies.push_back(response);
    }
    return replies;
  }

  void TestQuestions(const char* kQA[][2], size_t num_questions, bool batch) {
    if (!s_env->GetModel()) return;
    if (batch) {
      std::vector<std::string> inputs;
      for (size_t i = 0; i < num_questions; ++i) {
        fprintf(stderr, "Batch Question %zu\n\n", i + 1);
        inputs.push_back(kQA[i][0]);
      }
      std::vector<std::string> responses = BatchGemmaReply(inputs);
      for (size_t i = 0; i < num_questions; ++i) {
        std::string response = responses.at(i);
        fprintf(stderr, "Batch answer %zu '%s'\n\n", i + 1, response.c_str());
        EXPECT_TRUE(response.find(kQA[i][1]) != std::string::npos);  // NOLINT
      }
    } else {
      for (size_t i = 0; i < num_questions; ++i) {
        fprintf(stderr, "Question %zu\n\n", i + 1);
        std::string response = GemmaReply(kQA[i][0]);
        fprintf(stderr, "'%s'\n\n", response.c_str());
        EXPECT_TRUE(response.find(kQA[i][1]) != std::string::npos);  // NOLINT
      }
    }
  }
};

TEST_F(GemmaTest, Geography) {
  static const char* kQA[][2] = {
      {"What is the capital of Hungary?", "Budapest"},
      {"What is the capital of Australia?", "Canberra"},
      {"How many states does the US have?", "50"},
  };
  static const size_t kNum = sizeof(kQA) / sizeof(kQA[0]);
  TestQuestions(kQA, kNum, /*batch=*/false);
  static const char* kQA_single_question[][2] = {
      {"What is the capital of Australia?", "Canberra"},
  };
  TestQuestions(kQA_single_question, 1, /*batch=*/true);
  TestQuestions(kQA, kNum, /*batch=*/true);
}

TEST_F(GemmaTest, History) {
  static const char* kQA[][2] = {
      {"When was the battle of Hastings?", "1066"},
  };
  static const size_t kNum = sizeof(kQA) / sizeof(kQA[0]);
  TestQuestions(kQA, kNum, /*batch=*/false);
}

TEST_F(GemmaTest, Arithmetic) {
  static const char* kQA[][2] = {
      {"what is 13 + 14?", "27"},
      {"what is 7 * 8?", "56"},
  };
  static const size_t kNum = sizeof(kQA) / sizeof(kQA[0]);
  TestQuestions(kQA, kNum, /*batch=*/false);
}

static const char kJingleBells[] = R"(
Dashing through the snow
In a one-horse open sleigh
O'er the fields we go
Laughing all the way
Bells on bobtails ring
Making spirits bright
What fun it is to ride and sing
A sleighing song tonight
)";

// The "Hay Draft" of the Gettysburg Address.
static const char kGettysburg[] = {
    "Four score and seven years ago our fathers brought forth, upon this "
    "continent, a new nation, conceived in Liberty, and dedicated to the "
    "proposition that all men are created equal.\n\nNow we are engaged in a "
    "great civil war, testing whether that nation, or any nation, so "
    "conceived, and so dedicated, can long endure. We are met here on a great "
    "battlefield of that war. We have come to dedicate a portion of it as a "
    "final resting place for those who here gave their lives that that nation "
    "might live. It is altogether fitting and proper that we should do "
    "this.\n\nBut in a larger sense we can not dedicate -- we can not "
    "consecrate -- we can not hallow this ground. The brave men, living and "
    "dead, who struggled, here, have consecrated it far above our poor power "
    "to add or detract. The world will little note, nor long remember, what we "
    "say here, but can never forget what they did here. It is for us, the "
    "living, rather to be dedicated here to the unfinished work which they "
    "have, thus far, so nobly carried on. It is rather for us to be here "
    "dedicated to the great task remaining before us -- that from these "
    "honored dead we take increased devotion to that cause for which they here "
    "gave the last full measure of devotion -- that we here highly resolve "
    "that these dead shall not have died in vain; that this nation shall have "
    "a new birth of freedom; and that this government of the people, by the "
    "people, for the people, shall not perish from the earth.\n"};

TEST_F(GemmaTest, CrossEntropySmall) {
  if (!s_env->GetModel()) return;
  static const char kSmall[] =
      "The capital of Hungary is Budapest which is located in Europe.";
  float entropy = s_env->CrossEntropy(kSmall);
  fprintf(stderr, "per-token entropy: %f\n", entropy);
  float expected_entropy;
  switch (s_env->GetModel()->Info().model) {
    case gcpp::Model::GEMMA_2B:
      expected_entropy = 2.56f;
      break;
    case gcpp::Model::GEMMA_7B:
      expected_entropy = 2.91f;
      break;
    case gcpp::Model::GEMMA_9B:
      expected_entropy = 1.28f;
      break;
    case gcpp::Model::GEMMA_27B:
      expected_entropy = 1.30f;
      break;
    default:
      FAIL() << "no entropy expectation for this model";
      break;
  }
  EXPECT_NEAR(entropy, expected_entropy, 0.02f);
}

TEST_F(GemmaTest, CrossEntropyJingleBells) {
  if (!s_env->GetModel()) return;
  float entropy = s_env->CrossEntropy(kJingleBells);
  fprintf(stderr, "per-token entropy: %f\n", entropy);
  float expected_entropy;
  switch (s_env->GetModel()->Info().model) {
    case gcpp::Model::GEMMA_2B:
      expected_entropy = 1.85f;
      break;
    case gcpp::Model::GEMMA_7B:
      expected_entropy = 1.06f;
      break;
    case gcpp::Model::GEMMA_9B:
      expected_entropy = 0.37f;
      break;
    case gcpp::Model::GEMMA_27B:
      expected_entropy = 0.33f;
      break;
    default:
      FAIL() << "no entropy expectation for this model";
      break;
  }
  EXPECT_NEAR(entropy, expected_entropy, 0.02f);
}

TEST_F(GemmaTest, CrossEntropyGettysburg) {
  if (!s_env->GetModel()) return;
  float entropy = s_env->CrossEntropy(kGettysburg);
  fprintf(stderr, "per-token entropy: %f\n", entropy);
  float expected_entropy;
  switch (s_env->GetModel()->Info().model) {
    case gcpp::Model::GEMMA_2B:
      expected_entropy = 1.05f;
      break;
    case gcpp::Model::GEMMA_7B:
      expected_entropy = 0.83f;
      break;
    case gcpp::Model::GEMMA_9B:
      expected_entropy = 0.15f;
      break;
    case gcpp::Model::GEMMA_27B:
      expected_entropy = 0.14f;
      break;
    default:
      FAIL() << "no entropy expectation for this model";
      break;
  }
  EXPECT_NEAR(entropy, expected_entropy, 0.02f);
}

}  // namespace
}  // namespace gcpp

int main(int argc, char** argv) {
  gcpp::GemmaEnv env(argc, argv);
  gcpp::s_env = &env;

  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
