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

#include <cstdio>
#include <string>
#include <vector>

#include "compression/types.h"
#include "evals/benchmark_helper.h"
#include "gemma/configs.h"
#include "gemma/gemma.h"
#include "util/allocator.h"
#include "hwy/base.h"
#include "hwy/tests/hwy_gtest.h"

// This test can be run manually with the downloaded PaliGemma weights.
// It should pass for `paligemma-3b-mix-224` and `paligemma2-3b-pt-448`.

namespace gcpp {
namespace {

// Shared state. Requires argc/argv, so construct in main and use the same raw
// pointer approach as in benchmarks.cc. Note that the style guide forbids
// non-local static variables with dtors.
GemmaEnv* s_env = nullptr;

class PaliGemmaTest : public ::testing::Test {
 protected:
  void InitVit(const std::string& path);
  std::string GemmaReply(const std::string& prompt_text) const;
  void TestQuestions(const char* kQA[][2], size_t num_questions);

  ImageTokens image_tokens_;
};

void PaliGemmaTest::InitVit(const std::string& path) {
  ASSERT_NE(s_env->GetGemma(), nullptr);
  const Allocator& allocator = s_env->Env().ctx.allocator;
  const Gemma& gemma = *(s_env->GetGemma());
  image_tokens_ = ImageTokens(
      allocator, Extents2D(gemma.GetModelConfig().vit_config.seq_len,
                           gemma.GetModelConfig().model_dim));
  Image image;
  HWY_ASSERT(gemma.GetModelConfig().wrapping == PromptWrapping::PALIGEMMA);
  HWY_ASSERT(image.ReadPPM(path));
  const size_t image_size = gemma.GetModelConfig().vit_config.image_size;
  image.Resize(image_size, image_size);
  RuntimeConfig runtime_config = {.gen = &s_env->MutableGen(), .verbosity = 0};
  gemma.GenerateImageTokens(runtime_config, image, image_tokens_);
}

std::string PaliGemmaTest::GemmaReply(const std::string& prompt_text) const{
  const Gemma& model = *(s_env->GetGemma());
  s_env->MutableGen().seed(0x12345678);
  RuntimeConfig runtime_config = {.max_generated_tokens = 512,
                                  .gen = &s_env->MutableGen(),
                                  .verbosity = 0};
  runtime_config.image_tokens = &image_tokens_;
  size_t abs_pos = 0;
  std::string mutable_prompt = prompt_text;
  std::vector<int> tokens = s_env->WrapAndTokenize(mutable_prompt);
  std::string response;
  auto stream_token = [&](int token, float) {
    std::string token_text;
    HWY_ASSERT(model.Tokenizer().Decode(std::vector<int>{token}, &token_text));
    response += token_text;
    return true;
  };
  runtime_config.stream_token = stream_token,
  tokens.insert(tokens.begin(), image_tokens_.BatchSize(), 0);
  size_t num_tokens = tokens.size();
  size_t prefix_end = num_tokens;
  runtime_config.prefill_tbatch_size = num_tokens;
  TimingInfo timing_info = {.verbosity = 0};
  model.Generate(runtime_config, tokens, abs_pos, prefix_end,
                 s_env->MutableKVCache(), timing_info);
  return response;
}

void PaliGemmaTest::TestQuestions(const char* kQA[][2], size_t num_questions) {
  ASSERT_NE(s_env->GetGemma(), nullptr);
  std::string path = "paligemma/testdata/image.ppm";
  InitVit(path);
  for (size_t i = 0; i < num_questions; ++i) {
    fprintf(stderr, "Question %zu\n\n", i + 1);
    std::string response = GemmaReply(kQA[i][0]);
    fprintf(stderr, "'%s'\n\n", response.c_str());
    EXPECT_TRUE(response.find(kQA[i][1]) != std::string::npos);  // NOLINT
  }
}

TEST_F(PaliGemmaTest, General) {
  ASSERT_NE(s_env->GetGemma(), nullptr);
  static const char* kQA_2_3B_pt_448[][2] = {
      {"describe this image", "The Grossmünster in Zürich"},
      {"describe image briefly", "The Grossmünster"},
      {"answer en What objects are in the image?", "Building, Tower"},
      {"segment water", "<loc1023> water"},
  };
  const char* (*qa)[2];
  size_t num;
  switch (s_env->GetGemma()->GetModelConfig().model) {
    case Model::PALIGEMMA2_3B_448:
      qa = kQA_2_3B_pt_448;
      num = sizeof(kQA_2_3B_pt_448) / sizeof(kQA_2_3B_pt_448[0]);
      break;
    default:
      FAIL() << "Unsupported model: "
             << s_env->GetGemma()->GetModelConfig().display_name;
      break;
  }
  TestQuestions(qa, num);
}

}  // namespace
}  // namespace gcpp

int main(int argc, char** argv) {
  gcpp::GemmaEnv env(argc, argv);
  gcpp::s_env = &env;

  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
