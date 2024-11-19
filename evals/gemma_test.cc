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
  std::string GemmaReply(const std::string& prompt) {
    s_env->SetMaxGeneratedTokens(2048);
    s_env->MutableConfig().temperature = 0.0f;  // deterministic
    s_env->MutableConfig().verbosity = 0;
    // Using the turn structure worsens results sometimes.
    // However, some models need the turn structure to work.
    // It would be good to make these tests more consistent.
    if (s_env->GetModel()->Info().model == Model::GEMMA2_27B ||
        s_env->GetModel()->Info().model == Model::GRIFFIN_2B) {
      std::string mutable_prompt = prompt;
      QueryResult result = s_env->QueryModel(mutable_prompt);  // Uses turns.
      return result.response;
    }
    // Otherwise, do not use turn structure.
    const std::vector<int> tokens = s_env->TokenizeAndPrependBOS(prompt);
    QueryResult result = s_env->QueryModel(tokens);
    return result.response;
  }

  std::vector<std::string> BatchGemmaReply(
      const std::vector<std::string>& inputs) {
    s_env->SetMaxGeneratedTokens(64);
    s_env->MutableConfig().temperature = 0.0f;  // deterministic
    s_env->MutableConfig().verbosity = 0;
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

  void TestQuestions(const char* kQA[][2], size_t num_questions, bool batch) {
    ASSERT_NE(s_env->GetModel(), nullptr);
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

TEST_F(GemmaTest, GeographyBatched) {
  s_env->MutableConfig().decode_qbatch_size = 3;
  // 6 are enough to test batching and the loop.
  static const char* kQA[][2] = {
      {"What is the capital of Australia?", "Canberra"},
      {"What is the capital of Denmark?", "Copenhagen"},
      {"Ljubljana is the capital of which country?", "Slovenia"},
      {"Is Chicago a country?", "city"},
      {"How many states does the US have?", "50"},
      {"What is the Pacific?", "ocean"},
  };
  static const size_t kNum = sizeof(kQA) / sizeof(kQA[0]);
  TestQuestions(kQA, HWY_MIN(kNum, 3), /*batch=*/false);
  TestQuestions(kQA, 1, /*batch=*/true);
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

TEST_F(GemmaTest, Multiturn) {
  Gemma* model = s_env->GetModel();
  ASSERT_NE(model, nullptr);
  size_t abs_pos = 0;
  std::string dialog;
  auto stream_token = [&](int token, float) {
    ++abs_pos;
    std::string token_text;
    EXPECT_TRUE(
        model->Tokenizer().Decode(std::vector<int>{token}, &token_text));
    dialog += token_text;
    return true;
  };
  RuntimeConfig runtime_config{
      .max_generated_tokens = 64,
      .temperature = 0.0f,
      .gen = &s_env->MutableGen(),
      .verbosity = 2,
      .stream_token = stream_token,
  };
  TimingInfo timing_info{.verbosity = 0};
  // First "say" something slightly unusual.
  std::string mutable_prompt = "I have a car and its color is turquoise.";
  std::vector<int> tokens = WrapAndTokenize(model->Tokenizer(), model->Info(),
                                            abs_pos, mutable_prompt);
  model->Generate(runtime_config, tokens, abs_pos, s_env->MutableKVCache(),
                  timing_info);
  mutable_prompt = "Please repeat all prior statements.";
  tokens = WrapAndTokenize(model->Tokenizer(), model->Info(), abs_pos,
                           mutable_prompt);
  // Reset the `dialog` string here, then check that the model actually has
  // access to the previous turn by asking to reproduce.
  dialog.clear();
  model->Generate(runtime_config, tokens, abs_pos, s_env->MutableKVCache(),
                  timing_info);
  fprintf(stderr, "decoded: %s\n", dialog.c_str());
  bool remembered_turquoise =
      dialog.find("turquoise") != std::string::npos;              // NOLINT
  bool remembered_car = dialog.find("car") != std::string::npos;  // NOLINT
  EXPECT_TRUE(remembered_turquoise || remembered_car);
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
  ASSERT_NE(s_env->GetModel(), nullptr);
  static const char kSmall[] =
      "The capital of Hungary is Budapest which is located in Europe.";
  float entropy = s_env->CrossEntropy(kSmall);
  fprintf(stderr, "per-token entropy: %f\n", entropy);
  switch (s_env->GetModel()->Info().model) {
    case gcpp::Model::GEMMA_2B:
      // 2B v.1 and v.1.1 produce slightly different results.
      EXPECT_NEAR(entropy, 2.6f, 0.2f);
      break;
    case gcpp::Model::GEMMA_7B:
      // 7B v.1 and v.1.1 produce slightly different results.
      EXPECT_NEAR(entropy, 2.8f, 0.2f);
      break;
    case gcpp::Model::GRIFFIN_2B:
      EXPECT_NEAR(entropy, 2.61f, 0.02f);
      break;
    case gcpp::Model::GEMMA2_2B:
      EXPECT_NEAR(entropy, 1.14f, 0.02f);
      break;
    case gcpp::Model::GEMMA2_9B:
      EXPECT_NEAR(entropy, 1.28f, 0.02f);
      break;
    case gcpp::Model::GEMMA2_27B:
      EXPECT_NEAR(entropy, 1.30f, 0.02f);
      break;
    default:
      FAIL() << "no entropy expectation for this model";
      break;
  }
}

TEST_F(GemmaTest, CrossEntropyJingleBells) {
  ASSERT_NE(s_env->GetModel(), nullptr);
  float entropy = s_env->CrossEntropy(kJingleBells);
  fprintf(stderr, "per-token entropy: %f\n", entropy);
  switch (s_env->GetModel()->Info().model) {
    case gcpp::Model::GEMMA_2B:
      // 2B v.1 and v.1.1 produce slightly different results.
      EXPECT_NEAR(entropy, 1.9f, 0.2f);
      break;
    case gcpp::Model::GEMMA_7B:
      // 7B v.1 and v.1.1 produce slightly different results.
      EXPECT_NEAR(entropy, 1.07f, 0.05f);
      break;
    case gcpp::Model::GRIFFIN_2B:
      EXPECT_NEAR(entropy, 1.62f, 0.02f);
      break;
    case gcpp::Model::GEMMA2_2B:
      EXPECT_NEAR(entropy, 0.49f, 0.02f);
      break;
    case gcpp::Model::GEMMA2_9B:
      EXPECT_NEAR(entropy, 0.37f, 0.02f);
      break;
    case gcpp::Model::GEMMA2_27B:
      EXPECT_NEAR(entropy, 0.33f, 0.02f);
      break;
    default:
      FAIL() << "no entropy expectation for this model";
      break;
  }
}

TEST_F(GemmaTest, CrossEntropyGettysburg) {
  ASSERT_NE(s_env->GetModel(), nullptr);
  float entropy = s_env->CrossEntropy(kGettysburg);
  fprintf(stderr, "per-token entropy: %f\n", entropy);
  switch (s_env->GetModel()->Info().model) {
    case gcpp::Model::GEMMA_2B:
      // 2B v.1 and v.1.1 produce slightly different results.
      EXPECT_NEAR(entropy, 1.1f, 0.1f);
      break;
    case gcpp::Model::GEMMA_7B:
      // 7B v.1 and v.1.1 produce slightly different results.
      EXPECT_NEAR(entropy, 0.75f, 0.1f);
      break;
    case gcpp::Model::GRIFFIN_2B:
      EXPECT_NEAR(entropy, 0.71f, 0.02f);
      break;
    case gcpp::Model::GEMMA2_2B:
      EXPECT_NEAR(entropy, 0.20f, 0.02f);
      break;
    case gcpp::Model::GEMMA2_9B:
      EXPECT_NEAR(entropy, 0.15f, 0.02f);
      break;
    case gcpp::Model::GEMMA2_27B:
      EXPECT_NEAR(entropy, 0.14f, 0.02f);
      break;
    default:
      FAIL() << "no entropy expectation for this model";
      break;
  }
}

}  // namespace
}  // namespace gcpp

int main(int argc, char** argv) {
  gcpp::GemmaEnv env(argc, argv);
  gcpp::s_env = &env;

  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
