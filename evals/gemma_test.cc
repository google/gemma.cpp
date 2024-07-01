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

#include "gemma/benchmark_helper.h"
#include "gemma/common.h"
#include "hwy/tests/hwy_gtest.h"

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

  void TestQuestions(const char* kQA[][2], size_t num_questions) {
    if (!s_env->GetModel()) return;
    for (size_t i = 0; i < num_questions; ++i) {
      fprintf(stderr, "Question %zu\n\n", i + 1);
      std::string response = GemmaReply(kQA[i][0]);
      fprintf(stderr, "'%s'\n\n", response.c_str());
      EXPECT_TRUE(response.find(kQA[i][1]) != std::string::npos);  // NOLINT
    }
  }
};

TEST_F(GemmaTest, Geography) {
  static const char* kQA[][2] = {
      {"What is the capital of Hungary?", "Budapest"},
      {"How many states does the US have?", "50"},
  };
  static const size_t kNum = sizeof(kQA) / sizeof(kQA[0]);
  TestQuestions(kQA, kNum);
}

TEST_F(GemmaTest, History) {
  static const char* kQA[][2] = {
      {"When was the battle of Hastings?", "1066"},
  };
  static const size_t kNum = sizeof(kQA) / sizeof(kQA[0]);
  TestQuestions(kQA, kNum);
}

TEST_F(GemmaTest, Arithmetic) {
  static const char* kQA[][2] = {
      {"what is 13 + 14?", "27"},
      {"what is 7 * 8?", "56"},
  };
  static const size_t kNum = sizeof(kQA) / sizeof(kQA[0]);
  TestQuestions(kQA, kNum);
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
  fprintf(stderr, "per-byte entropy: %f\n", entropy);
  EXPECT_LT(entropy,
            (s_env->ModelType() == gcpp::Model::GEMMA_7B) ? 2.1f : 2.0f);
}

TEST_F(GemmaTest, CrossEntropyJingleBells) {
  if (!s_env->GetModel()) return;
  float entropy = s_env->CrossEntropy(kJingleBells);
  fprintf(stderr, "per-byte entropy: %f\n", entropy);
  EXPECT_LT(entropy,
            (s_env->ModelType() == gcpp::Model::GEMMA_7B) ? 0.9f : 1.8f);
}

TEST_F(GemmaTest, CrossEntropyGettysburg) {
  if (!s_env->GetModel()) return;
  float entropy = s_env->CrossEntropy(kGettysburg);
  fprintf(stderr, "per-byte entropy: %f\n", entropy);
  EXPECT_LT(entropy,
            (s_env->ModelType() == gcpp::Model::GEMMA_7B) ? 0.8f : 1.2f);
}

}  // namespace
}  // namespace gcpp

int main(int argc, char** argv) {
  gcpp::GemmaEnv env(argc, argv);
  gcpp::s_env = &env;

  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}