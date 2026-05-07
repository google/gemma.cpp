// Copyright 2026 Google LLC
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

#include <cstring>
#include <string>
#include <vector>

#include "evals/benchmark_helper.h"
#include "gemma/configs.h"
#include "gemma/gemma.h"
#include "io/io.h"
#include "hwy/base.h"
#include "hwy/tests/hwy_gtest.h"

// This test can be run manually with the downloaded gemma weights.
// To run the test, pass the following flags:
// --tokenizer <tokenizer_path> --weights <weights_path>
// or just use the single-file weights file with --weights <weights_path>.
// It should pass for the following models:
// Gemma2: gemma2-2b-it

namespace gcpp {
namespace {

static const char* kQuestions =
    "From the above information, please answer the following questions: "
    "What did Marcia find in the sand? "
    "What is Albert's preferred holiday activity? "
    "How long did it take to dig out the object from the sand? "
    "What is Marcia's preferred holiday activity? "
    "What made the castle turrets look like daleks? "
    "Which people first proposed the quark model of hadrons, and when?";

// All phrases in kAnswers must appear in the response in the order given for
// the test to pass. Multiple acceptable answers can be provided for each
// expected phrase.
static const std::vector<std::vector<const char*>> kAnswers = {
    {"rusty metal", "ship's anchor"},
    {"dark forest"},
    {"an hour"},
    {"sand"},
    {"castle"},
    {"limpet shells"},
    {"Murray Gell-Mann"},
    {"George Zweig"},
    {"1964"}};

std::string LoadPromptFile(const std::string& filename) {
  // If the filename is empty, return an empty string.
  if (filename.empty()) {
    return "";
  }
  std::string path = testing::SrcDir() +
                     "evals/testdata/"
                     + filename;
  return ReadFileToString(Path(path));
}

std::string BuildPrompt(const std::vector<std::string>& files,
                        const std::string& suffix) {
  std::string prompt;
  for (const std::string& file : files) {
    prompt += LoadPromptFile(file);
  }
  prompt += suffix;
  return prompt;
}

class GemmaTest : public ::testing::Test {
 public:
  // Requires argc/argv, hence do not use `SetUpTestSuite`.
  static void InitEnv(int argc, char** argv) {
    HWY_ASSERT(s_env == nullptr);  // Should only be called once.
    ConsumedArgs consumed(argc, argv);
    GemmaArgs args(argc, argv, consumed);
    consumed.AbortIfUnconsumed();

    s_env = new GemmaEnv(args);
    const gcpp::ModelConfig& config = s_env->GetGemma()->Config();
    fprintf(stderr, "Using %s\n", config.Specifier().c_str());
  }

  static void DeleteEnv() { delete s_env; }

 protected:
  std::string GemmaReply(const std::string& input,
                         AttentionImpl attention_mode) {
    HWY_ASSERT(s_env);  // must have called InitEnv()
    s_env->SetMaxGeneratedTokens(256);
    s_env->MutableConfig().attention_impl = attention_mode;
    s_env->MutableConfig().temperature = 0.0f;  // deterministic
    s_env->MutableConfig().verbosity = 1;
    // Always use turn structure (WrapAndTokenize).
    auto response = s_env->QueryModel(input);
    return response.response.substr(response.response_start_pos);
  }

  // Checks that the response contains the expected answer substrings int the
  // expected order. Testing against a few keywords is more robust than checking
  // the whole string.
  void TestExpectations(const std::string& response) {
    fprintf(stderr, "Response: '%s'\n", response.c_str());
    size_t pos = 0;
    for (const auto& answer_group : kAnswers) {
      size_t earliest_pos = std::string::npos;
      const char* matched_answer = nullptr;
      for (const char* answer : answer_group) {
        auto found = response.find(answer, pos);
        if (found != std::string::npos &&
            (earliest_pos == std::string::npos || found < earliest_pos)) {
          earliest_pos = found;
          matched_answer = answer;
        }
      }
      EXPECT_NE(earliest_pos, std::string::npos)
          << "Response does not contain acceptable answers, e.g., "
          << answer_group[0];
      if (earliest_pos != std::string::npos) {
        pos = earliest_pos + strlen(matched_answer);
      }
    }
    s_env->PrintProfileResults();
  }

  // Shared state. Requires argc/argv, so construct in main via InitEnv.
  // Note that the style guide forbids non-local static variables with dtors.
  static GemmaEnv* s_env;
};

GemmaEnv* GemmaTest::s_env = nullptr;

// Tests whether Gemma can find the right answer in varying levels of
// background information, ranging from the bare facts to outright distraction.
TEST_F(GemmaTest, WheatFromChaff) {
  const AttentionImpl modes[] = {AttentionImpl::kFlash};
  fprintf(stderr, "Warmup, mode %s\n", GetAttentionImplName(modes[0]).c_str());
  auto prompt = BuildPrompt({"quark_1.txt", "holiday_story.txt"}, kQuestions);
  auto response = GemmaReply(prompt, modes[0]);
  TestExpectations(response);
  for (const AttentionImpl mode : modes) {
    const std::string mode_name = GetAttentionImplName(mode);
    fprintf(stderr, "\nTesting quark_1 prompt, mode %s\n", mode_name.c_str());
    prompt = BuildPrompt({"holiday_story.txt", "quark_1.txt"}, kQuestions);
    response = GemmaReply(prompt, mode);
    TestExpectations(response);
    fprintf(stderr, "\nTesting quark_2 prompt, mode %s\n", mode_name.c_str());
    prompt = BuildPrompt({"holiday_story.txt", "quark_2.txt"}, kQuestions);
    response = GemmaReply(prompt, mode);
    TestExpectations(response);
    fprintf(stderr, "\nTesting standard_model prompt, mode %s\n",
            mode_name.c_str());
    prompt = BuildPrompt(
        {"holiday_story.txt", "quark_2.txt", "standard_model.txt"}, kQuestions);
    response = GemmaReply(prompt, mode);
    TestExpectations(response);
    if (s_env->MutableKVCache().SeqLen() > 38000) {
      fprintf(stderr, "\nTesting special_relativity, mode %s\n",
              mode_name.c_str());
      prompt = BuildPrompt(
          {"holiday_story.txt", "quark_2.txt", "special_relativity.txt"},
          kQuestions);
    } else {
      fprintf(stderr, "\nSkipping special_relativity, mode %s\n",
              mode_name.c_str());
      prompt = BuildPrompt({"quark_1.txt", "holiday_story.txt"}, kQuestions);
    }
    response = GemmaReply(prompt, mode);
    TestExpectations(response);
  }
}

}  // namespace
}  // namespace gcpp

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  gcpp::GemmaTest::InitEnv(argc, argv);
  int ret = RUN_ALL_TESTS();
  gcpp::GemmaTest::DeleteEnv();
  return ret;
}
