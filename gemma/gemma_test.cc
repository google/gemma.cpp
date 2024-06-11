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
#include <random>
#include <string>
#include <vector>

// Placeholder for internal header, do not modify.
#include "gemma/cross_entropy.h"
#include "gemma/ops.h"
#include "util/app.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/tests/test_util-inl.h"

namespace gcpp {
namespace {

int s_argc = 0;
char** s_argv = nullptr;

class GemmaTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    gcpp::LoaderArgs loader(s_argc, s_argv);
    gcpp::AppArgs app(s_argc, s_argv);
    if (const char* err = loader.Validate()) {
      fprintf(stderr, "Insufficient LoaderArgs, skipping e2e tests.\n");
    } else {
      s_pool = std::make_unique<hwy::ThreadPool>(app.num_threads);
      s_model = AllocateGemma(loader, *s_pool);
      s_kv_cache = KVCache::Create(loader.ModelType());
    }
  }

  static void TearDownTestSuite() {
    s_pool.reset();
    s_model.reset();
  }

  std::string GemmaReply(const std::string& prompt_string) {
    std::mt19937 gen;
    gen.seed(42);

    std::vector<int> prompt;
    HWY_ASSERT(s_model->Tokenizer().Encode(prompt_string, &prompt));
    // For both pre-trained and instruction-tuned models: prepend "<bos>" token
    // if needed.
    prompt.insert(prompt.begin(), BOS_ID);

    std::vector<int> response;
    auto stream_token = [&response](int token, float) {
      response.push_back(token);
      return true;
    };
    gcpp::RuntimeConfig runtime_config = {
        .max_tokens = 3072,
        .max_generated_tokens = 2048,
        .temperature = 1.0,
        .verbosity = 0,
        .gen = &gen,
        .stream_token = stream_token,
    };
    gcpp::TimingInfo timing_info;
    s_model->Generate(runtime_config, prompt, /*start_pos=*/0, s_kv_cache,
                      timing_info, /*layers_output=*/nullptr);
    std::string response_text;
    HWY_ASSERT(s_model->Tokenizer().Decode(response, &response_text));
    return response_text;
  }

  float GemmaCrossEntropy(const std::string& prompt_string) {
    std::vector<int> prompt;
    HWY_ASSERT(s_model->Tokenizer().Encode(prompt_string, &prompt));
    prompt.insert(prompt.begin(), BOS_ID);
    return ComputeCrossEntropy(*s_model, /*max_tokens=*/3072, prompt,
                               s_kv_cache,
                               /*verbosity=*/0) /
           prompt_string.size();
  }

  void TestQuestions(const char* kQA[][2], size_t num_questions) {
    if (!s_model) return;
    for (size_t i = 0; i < num_questions; ++i) {
      fprintf(stderr, "Question %zu\n\n", i + 1);
      std::string response = GemmaReply(kQA[i][0]);
      fprintf(stderr, "'%s'\n\n", response.c_str());
      EXPECT_TRUE(response.find(kQA[i][1]) != std::string::npos);  // NOLINT
    }
  }

  static std::unique_ptr<hwy::ThreadPool> s_pool;
  static std::unique_ptr<gcpp::Gemma> s_model;
  static gcpp::KVCache s_kv_cache;
};

/*static*/ std::unique_ptr<hwy::ThreadPool> GemmaTest::s_pool;
/*static*/ std::unique_ptr<gcpp::Gemma> GemmaTest::s_model;
/*static*/ gcpp::KVCache GemmaTest::s_kv_cache;

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
      {"When was the Battle of Hastings?", "1066"},
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
  if (!s_model) return;
  static const char kSmall[] =
      "The capital of Hungary is Budapest which is located in Europe.";
  float entropy = GemmaCrossEntropy(kSmall);
  fprintf(stderr, "per-byte entropy: %f\n", entropy);
  // Note that entropy is 3x higher for the 7b-it model.
  EXPECT_LT(entropy, 1.7f);
}

TEST_F(GemmaTest, CrossEntropyJingleBells) {
  if (!s_model) return;
  float entropy = GemmaCrossEntropy(kJingleBells);
  fprintf(stderr, "per-byte entropy: %f\n", entropy);
  EXPECT_LT(entropy, 1.7f);
}

TEST_F(GemmaTest, CrossEntropyGettysburg) {
  if (!s_model) return;
  float entropy = GemmaCrossEntropy(kGettysburg);
  fprintf(stderr, "per-byte entropy: %f\n", entropy);
  EXPECT_LT(entropy, 1.2f);
}

}  // namespace
}  // namespace gcpp

int main(int argc, char** argv) {
  {
    // Placeholder for internal init, do not modify.
  }

  // For later use by SetUp.
  gcpp::s_argc = argc;
  gcpp::s_argv = argv;

  // Probably should be called before SetUpTestSuite.
  testing::InitGoogleTest(&gcpp::s_argc, argv);

  return RUN_ALL_TESTS();
}