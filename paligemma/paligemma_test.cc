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

#include <memory>
#include <string>

#include "evals/benchmark_helper.h"
#include "gemma/configs.h"
#include "gemma/gemma.h"
#include "io/io.h"
#include "util/allocator.h"
#include "hwy/tests/hwy_gtest.h"
#include "paligemma/paligemma_helper.h"

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
  void TestQuestion(const char* question, const char* expected_substring) {
    ASSERT_NE(s_env->GetGemma(), nullptr);
    std::string path = "paligemma/testdata/image.ppm";

    PaliGemmaHelper paligemma_helper(s_env);
    paligemma_helper.InitVit(path);
    const std::string reply = paligemma_helper.GemmaReply(question);
    fprintf(stderr, "'%s'\n\n", reply.c_str());
    EXPECT_TRUE(reply.find(expected_substring) != std::string::npos);  // NOLINT
  }

  std::unique_ptr<ImageTokens> image_tokens_;
};

TEST_F(PaliGemmaTest, QueryObjects) {
  ASSERT_NE(s_env->GetGemma(), nullptr);
  const char* question = "answer en What objects are in the image?";
  // 3B PT/Mix 224, 10B Mix 224
  const char* expected_substring = "Building, Tower";
  const Model model = s_env->GetGemma()->Config().model;
  if (model == Model::PALIGEMMA2_3B_448) {
    expected_substring = "Lake.";
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
