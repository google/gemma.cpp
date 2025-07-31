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
#include "gemma/configs.h"
#include "io/io.h"
#include "hwy/base.h"
#include "hwy/tests/hwy_gtest.h"

// This test can be run manually with the downloaded gemma weights.
// To run the test, pass the following flags:
// --model <model> --tokenizer <tokenizer_path> --weights <weights_path>
// or just use the single-file weights file with --weights <weights_path>.
// It should pass for the following models:
// Gemma1: 2b-it (v1 and v1.1), 7b-it (v1 and v1.1), gr2b-it,
// Gemma2: gemma2-2b-it, 9b-it, 27b-it,

namespace gcpp {
namespace {

class GemmaTest : public ::testing::Test {
 public:
  // Requires argc/argv, hence do not use `SetUpTestSuite`.
  static void InitEnv(int argc, char** argv) {
    HWY_ASSERT(s_env == nullptr);  // Should only be called once.
    s_env = new GemmaEnv(argc, argv);
    const gcpp::ModelConfig& config = s_env->GetGemma()->Config();
    fprintf(stderr, "Using %s\n", config.Specifier().c_str());
  }

  static void DeleteEnv() { delete s_env; }

 protected:
  std::vector<std::string> BatchGemmaReply(
      const std::vector<std::string>& inputs) {
    HWY_ASSERT(s_env);  // must have called InitEnv()
    s_env->SetMaxGeneratedTokens(64);
    s_env->MutableConfig().temperature = 0.0f;  // deterministic
    s_env->MutableConfig().verbosity = 0;
    // Always use turn structure (WrapAndTokenize).
    std::vector<std::string> replies;
    for (QueryResult result : s_env->BatchQueryModel(inputs)) {
      replies.push_back(result.response);
    }
    return replies;
  }

  // Shared state. Requires argc/argv, so construct in main via InitEnv.
  // Note that the style guide forbids non-local static variables with dtors.
  static GemmaEnv* s_env;
};

GemmaEnv* GemmaTest::s_env = nullptr;

TEST_F(GemmaTest, Batched) {
  // Test remainder handling in MatMul (four rows per tile), but avoid a
  // second batch in debug builds to speed up the test.
  s_env->MutableConfig().decode_qbatch_size = HWY_IS_DEBUG_BUILD ? 6 : 3;
  static const char* kQA[][2] = {
      {"What is the capital of Australia?", "Canberra"},
      {"How many states does the US have?", "50"},
      {"What is the Pacific?", "ocean"},
      {"When was the battle of Hastings?", "1066"},
      {"what is 13 + 14?", "27"},
      {"what is 7 * 8?", "56"},
  };
  const size_t kNum = sizeof(kQA) / sizeof(kQA[0]);
  std::vector<std::string> inputs;
  for (size_t i = 0; i < kNum; ++i) {
    inputs.push_back(kQA[i][0]);
  }
  std::vector<std::string> responses = BatchGemmaReply(inputs);
  HWY_ASSERT(responses.size() == kNum);
  for (size_t i = 0; i < kNum; ++i) {
    fprintf(stderr, "#%zu: '%s'\n\n", i, responses[i].c_str());
    EXPECT_TRUE(responses[i].find(kQA[i][1]) != std::string::npos);  // NOLINT
  }
}

TEST_F(GemmaTest, Multiturn) {
  const Gemma* model = s_env->GetGemma();
  const ModelConfig& config = model->Config();
  size_t abs_pos = 0;
  std::string response;
  auto stream_token = [&](size_t query_idx, size_t pos, int token, float) {
    HWY_ASSERT(query_idx == 0);
    HWY_ASSERT(pos == abs_pos);
    ++abs_pos;
    if (config.IsEOS(token)) return true;
    std::string token_text;
    EXPECT_TRUE(
        model->Tokenizer().Decode(std::vector<int>{token}, &token_text));
    response += token_text;
    return true;
  };
  RuntimeConfig runtime_config{
      .max_generated_tokens = 64,
      .temperature = 0.0f,
      .gen = &s_env->MutableGen(),
      .verbosity = 2,
      .batch_stream_token = stream_token,
  };
  TimingInfo timing_info{.verbosity = 0};
  // First "say" something slightly unusual.
  std::string mutable_prompt = "I have a car and its color is turquoise.";
  std::vector<int> tokens =
      WrapAndTokenize(model->Tokenizer(), model->ChatTemplate(),
                      config.wrapping, abs_pos, mutable_prompt);

  model->Generate(runtime_config, tokens, abs_pos, s_env->MutableKVCache(),
                  s_env->MutableEnv(), timing_info);
  // Note: we do not rewind any <end_of_turn> tokens here. If the model
  // produced one and WrapAndTokenize() inserts another one, it will just be
  // duplicated.
  mutable_prompt = "Please repeat all prior statements.";
  tokens = WrapAndTokenize(model->Tokenizer(), model->ChatTemplate(),
                           config.wrapping, abs_pos, mutable_prompt);

  // Reset the `response` string here, then check that the model actually has
  // access to the previous turn by asking to reproduce.
  response.clear();
  model->Generate(runtime_config, tokens, abs_pos, s_env->MutableKVCache(),
                  s_env->MutableEnv(), timing_info);
  fprintf(stderr, "decoded: '%s'\n", response.c_str());
  bool remembered_turquoise =
      response.find("turquoise") != std::string::npos;              // NOLINT
  bool remembered_car = response.find("car") != std::string::npos;  // NOLINT
  EXPECT_TRUE(remembered_turquoise || remembered_car);
}

TEST_F(GemmaTest, CrossEntropySmall) {
  HWY_ASSERT(s_env->GetGemma() != nullptr);
  const ModelConfig& config = s_env->GetGemma()->Config();
  static const char kSmall[] =
      "The capital of Hungary is Budapest which is located in Europe.";
  float entropy = s_env->CrossEntropy(kSmall);
  fprintf(stderr, "per-token entropy: %f\n", entropy);
  switch (config.model) {
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

}  // namespace
}  // namespace gcpp

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  gcpp::InternalInit();
  gcpp::GemmaTest::InitEnv(argc, argv);
  int ret = RUN_ALL_TESTS();
  gcpp::GemmaTest::DeleteEnv();
  return ret;
}
