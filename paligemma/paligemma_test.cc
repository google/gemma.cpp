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
#include <memory>
#include <string>
#include <vector>

#include "compression/types.h"
#include "evals/benchmark_helper.h"
#include "gemma/configs.h"
#include "gemma/gemma.h"
#include "io/io.h"
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
  void TestQuestion(const char* question, const char* expected_substring);

  std::unique_ptr<ImageTokens> image_tokens_;
};

void PaliGemmaTest::InitVit(const std::string& path) {
  ASSERT_NE(s_env->GetGemma(), nullptr);
  const Gemma& gemma = *(s_env->GetGemma());
  const ModelConfig& config = gemma.GetModelConfig();
  image_tokens_ = std::make_unique<ImageTokens>(
      "image", Extents2D(config.vit_config.seq_len, config.model_dim),
      MatPadding::kPacked);
  image_tokens_->AllocateAndAttachRowPtrs(s_env->Env().row_ptrs);
  Image image;
  HWY_ASSERT(config.wrapping == PromptWrapping::PALIGEMMA);
  HWY_ASSERT(image.ReadPPM(path));
  const size_t image_size = config.vit_config.image_size;
  image.Resize(image_size, image_size);
  RuntimeConfig runtime_config = {.gen = &s_env->MutableGen(), .verbosity = 0};
  gemma.GenerateImageTokens(runtime_config, image, *image_tokens_);
}

std::string PaliGemmaTest::GemmaReply(const std::string& prompt_text) const{
  const Gemma& model = *(s_env->GetGemma());
  s_env->MutableGen().seed(0x12345678);
  RuntimeConfig runtime_config = {.max_generated_tokens = 512,
                                  .gen = &s_env->MutableGen(),
                                  .verbosity = 0};
  runtime_config.image_tokens = image_tokens_.get();
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
  tokens.insert(tokens.begin(), image_tokens_->Rows(), 0);
  size_t num_tokens = tokens.size();
  size_t prefix_end = num_tokens;
  runtime_config.prefill_tbatch_size = num_tokens;
  TimingInfo timing_info = {.verbosity = 0};
  model.Generate(runtime_config, tokens, abs_pos, prefix_end,
                 s_env->MutableKVCache(), timing_info);
  return response;
}

void PaliGemmaTest::TestQuestion(const char* question,
                                 const char* expected_substring) {
  ASSERT_NE(s_env->GetGemma(), nullptr);
  std::string path = "paligemma/testdata/image.ppm";
  InitVit(path);
  const std::string reply = GemmaReply(question);
  fprintf(stderr, "'%s'\n\n", reply.c_str());
  EXPECT_TRUE(reply.find(expected_substring) != std::string::npos);  // NOLINT
}

TEST_F(PaliGemmaTest, QueryObjects) {
  ASSERT_NE(s_env->GetGemma(), nullptr);
  const char* question = "answer en What objects are in the image?";
  const char* expected_substring = "Building, Tower";  // 3B PT 224, 10B Mix 224
  const Model model = s_env->GetGemma()->GetModelConfig().model;
  if (model == Model::PALIGEMMA2_3B_448) {
    expected_substring = "Lake.";
  } else if (model == Model::PALIGEMMA2_3B_224) {
    expected_substring = "Cloud, Water.";
  } else if (model == Model::PALIGEMMA2_10B_224) {
    expected_substring = "Building.";
  }
  TestQuestion(question, expected_substring);
}

}  // namespace
}  // namespace gcpp

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  gcpp::InternalInit();

  gcpp::GemmaEnv env(argc, argv);
  gcpp::s_env = &env;

  return RUN_ALL_TESTS();
}
