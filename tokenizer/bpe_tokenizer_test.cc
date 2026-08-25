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

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "gemma/tokenizer.h"
#include "tokenizer/bpe_tokenizer.h"
#include "io/io.h"

namespace gcpp {
namespace {

// Tokenizer fixtures live next to the downloaded Gemma checkpoint. Tests run
// with the working directory set to the repository root.
constexpr const char* kTokenizerJson =
    "tokenizer/testdata/tokenizer.json";
constexpr const char* kTokenizerPacked =
    "tokenizer/testdata/tokenizer.packed";
constexpr const char* kTokenizerModel =
    "tokenizer/testdata/tokenizer.model";
constexpr const char* kCorpus =
    "testdata/frankenstein.txt";

std::string ReadFileOrAbort(const std::string& path) {
  return ReadFileToString(Path(path));
}

std::vector<std::string> ReadLines(const std::string& path) {
  std::string content = ReadFileToString(Path(path));
  std::vector<std::string> lines;
  std::stringstream ss(content);
  std::string line;
  while (std::getline(ss, line)) {
    lines.push_back(line);
  }
  return lines;
}

// Shares the tokenizer construction across test cases.
class BpeTokenizerTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    sp_ = new GemmaTokenizer(ReadFileOrAbort(kTokenizerModel));
    bpe_ = new GemmaTokenizer(
        CreateBpeTokenizer(ReadFileOrAbort(kTokenizerPacked)));
  }
  static void TearDownTestSuite() {
    delete sp_;
    delete bpe_;
    sp_ = nullptr;
    bpe_ = nullptr;
  }

  // Asserts the BPE encoding of `text` matches SentencePiece exactly.
  void ExpectSameIds(const std::string& text) {
    std::vector<int> sp_ids, bpe_ids;
    ASSERT_TRUE(sp_->Encode(text, &sp_ids));
    ASSERT_TRUE(bpe_->Encode(text, &bpe_ids));
    EXPECT_EQ(bpe_ids, sp_ids) << "Mismatch on: [" << text << "]";
  }

  static inline GemmaTokenizer* sp_ = nullptr;
  static inline GemmaTokenizer* bpe_ = nullptr;
};

TEST_F(BpeTokenizerTest, MatchesSentencePieceOnCorpus) {
  const std::vector<std::string> lines = ReadLines(kCorpus);
  ASSERT_FALSE(lines.empty());
  size_t total_tokens = 0;
  for (const std::string& line : lines) {
    std::vector<int> sp_ids, bpe_ids;
    ASSERT_TRUE(sp_->Encode(line, &sp_ids));
    ASSERT_TRUE(bpe_->Encode(line, &bpe_ids));
    ASSERT_EQ(bpe_ids, sp_ids) << "Mismatch on line: [" << line << "]";
    total_tokens += sp_ids.size();
  }
  EXPECT_GT(total_tokens, 0u);
}

TEST_F(BpeTokenizerTest, MatchesOnWholeFile) {
  ExpectSameIds(ReadFileOrAbort(kCorpus));
}

TEST_F(BpeTokenizerTest, EdgeCases) {
  ExpectSameIds("");
  ExpectSameIds(" ");
  ExpectSameIds("  ");
  ExpectSameIds("   leading spaces");
  ExpectSameIds("trailing spaces   ");
  ExpectSameIds("\n");
  ExpectSameIds("\t\ttabs");
  ExpectSameIds("Hello, world!");
  ExpectSameIds("multiple   internal    spaces");
  ExpectSameIds("CamelCaseAndnumbers12345");
  ExpectSameIds("punctuation?!.,;:'\"()[]{}");
}

TEST_F(BpeTokenizerTest, Unicode) {
  ExpectSameIds("café déjà vu naïve");
  ExpectSameIds("Ελληνικά κείμενο");
  ExpectSameIds("日本語のテキスト");
  ExpectSameIds("emoji 😀🎉🚀 fall back to bytes");
  ExpectSameIds("mixed 中文 and English 123");
}

TEST_F(BpeTokenizerTest, SpecialTokens) {
  ExpectSameIds("<start_of_turn>user\n");
  ExpectSameIds("<start_of_turn>model\n");
  ExpectSameIds("<end_of_turn>\n");
  ExpectSameIds("hello<end_of_turn>world");
  ExpectSameIds("\n\n<start_of_image>");
  ExpectSameIds("<end_of_image>\n\n");
}

TEST_F(BpeTokenizerTest, DecodeRoundTrip) {
  const std::vector<std::string> lines = ReadLines(kCorpus);
  ASSERT_FALSE(lines.empty());
  for (const std::string& line : lines) {
    std::vector<int> ids;
    ASSERT_TRUE(sp_->Encode(line, &ids));
    std::string bpe_text, sp_text;
    ASSERT_TRUE(bpe_->Decode(ids, &bpe_text));
    ASSERT_TRUE(sp_->Decode(ids, &sp_text));
    EXPECT_EQ(bpe_text, sp_text) << "Decode mismatch on line: [" << line << "]";
    EXPECT_EQ(bpe_text, line)
        << "Round-trip mismatch on line: [" << line << "]";
  }
}

// A tokenizer rebuilt from the compact binary should behave identically to
// the one parsed from tokenizer.json.
TEST_F(BpeTokenizerTest, CompactRoundTrip) {
  const std::string compact = bpe_->Serialize();
  EXPECT_LT(compact.size(), ReadFileOrAbort(kTokenizerJson).size());

  const GemmaTokenizer compact_bpe(CreateBpeTokenizer(compact));

  const std::vector<std::string> samples = {
      "",
      " ",
      "Hello, world!",
      "   leading spaces",
      "trailing spaces   ",
      "café déjà vu naïve",
      "日本語のテキスト",
      "emoji 😀🎉🚀 fall back to bytes",
      "<start_of_turn>user\n",
      "hello<end_of_turn>world",
      "CamelCaseAndnumbers12345",
      "punctuation?!.,;:'\"()[]{}",
  };
  for (const std::string& s : samples) {
    std::vector<int> json_ids, compact_ids;
    ASSERT_TRUE(bpe_->Encode(s, &json_ids));
    ASSERT_TRUE(compact_bpe.Encode(s, &compact_ids));
    EXPECT_EQ(compact_ids, json_ids) << "Mismatch on: [" << s << "]";
  }

  const std::vector<std::string> lines = ReadLines(kCorpus);
  ASSERT_FALSE(lines.empty());
  for (const std::string& line : lines) {
    std::vector<int> json_ids, compact_ids;
    ASSERT_TRUE(bpe_->Encode(line, &json_ids));
    ASSERT_TRUE(compact_bpe.Encode(line, &compact_ids));
    ASSERT_EQ(compact_ids, json_ids)
        << "Mismatch on corpus line: [" << line << "]";
  }

  std::vector<int> ids;
  ASSERT_TRUE(bpe_->Encode("Hello world", &ids));
  std::string json_text, compact_text;
  ASSERT_TRUE(bpe_->Decode(ids, &json_text));
  ASSERT_TRUE(compact_bpe.Decode(ids, &compact_text));
  EXPECT_EQ(compact_text, json_text);
}

}  // namespace
}  // namespace gcpp
