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

#include "tokenizer/sentencepiece_tokenizer.h"

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "io/io.h"
#include "tokenizer/tokenizer.h"
#include "hwy/aligned_allocator.h"  // hwy::Span
// copybara:import_next_line:sentencepiece
#include "lite/sentencepiece_model_converters.h"
// copybara:import_next_line:sentencepiece
#include "sentencepiece_model.pb.h"

namespace gcpp {
namespace {

constexpr const char* kTokenizerModel =
    "tokenizer/testdata/tokenizer.model";

constexpr char kText[] = "Hello world! This is a test.";

std::string ReadModelProto() { return ReadFileToString(Path(kTokenizerModel)); }

// Converts a serialized ModelProto into a serialized FlatBuffer (.spm.fb).
std::string ToFlatbufferOrDie(const std::string& serialized_proto) {
  sentencepiece::ModelProto model_proto;
  EXPECT_TRUE(model_proto.ParseFromString(serialized_proto));
  auto fb_model_or = sentencepiece::lite::ToFlatbuffer(model_proto);
  EXPECT_TRUE(fb_model_or.ok());
  return fb_model_or.ok() ? *fb_model_or : std::string();
}

TEST(SentencePieceTokenizerTest, LoadsModelProto) {
  const std::string proto = ReadModelProto();
  const std::unique_ptr<Tokenizer> tokenizer =
      CreateSentencePieceTokenizer(proto);

  std::vector<int> ids;
  ASSERT_TRUE(tokenizer->Encode(kText, ids));
  EXPECT_FALSE(ids.empty());

  std::string decoded;
  ASSERT_TRUE(
      tokenizer->Decode(hwy::Span<const int>(ids.data(), ids.size()), decoded));
  EXPECT_EQ(decoded, kText);

  EXPECT_EQ(tokenizer->Serialize(), proto);
}

TEST(SentencePieceTokenizerTest, LoadsFlatbuffer) {
  const std::string proto = ReadModelProto();
  const std::string flatbuffer = ToFlatbufferOrDie(proto);
  ASSERT_FALSE(flatbuffer.empty());

  const std::unique_ptr<Tokenizer> from_proto =
      CreateSentencePieceTokenizer(proto);
  const std::unique_ptr<Tokenizer> from_flatbuffer =
      CreateSentencePieceTokenizer(flatbuffer);

  std::vector<int> proto_ids, flatbuffer_ids;
  ASSERT_TRUE(from_proto->Encode(kText, proto_ids));
  ASSERT_TRUE(from_flatbuffer->Encode(kText, flatbuffer_ids));
  EXPECT_EQ(flatbuffer_ids, proto_ids);

  std::string decoded;
  ASSERT_TRUE(from_flatbuffer->Decode(
      hwy::Span<const int>(flatbuffer_ids.data(), flatbuffer_ids.size()),
      decoded));
  EXPECT_EQ(decoded, kText);

  EXPECT_EQ(from_flatbuffer->Serialize(), flatbuffer);
}

TEST(SentencePieceTokenizerTest, CanBeReloaded) {
  const std::string proto = ReadModelProto();
  const std::string flatbuffer = ToFlatbufferOrDie(proto);
  ASSERT_FALSE(flatbuffer.empty());

  const std::unique_ptr<Tokenizer> tokenizer =
      CreateSentencePieceTokenizer(flatbuffer);
  std::vector<int> expected_ids;
  ASSERT_TRUE(tokenizer->Encode(kText, expected_ids));

  // FlatBuffer -> FlatBuffer and FlatBuffer -> ModelProto reloads must not
  // leave the processor pointing at the previous buffer.
  ASSERT_TRUE(tokenizer->Deserialize(flatbuffer));
  std::vector<int> ids;
  ASSERT_TRUE(tokenizer->Encode(kText, ids));
  EXPECT_EQ(ids, expected_ids);

  ASSERT_TRUE(tokenizer->Deserialize(proto));
  ids.clear();
  ASSERT_TRUE(tokenizer->Encode(kText, ids));
  EXPECT_EQ(ids, expected_ids);
}

TEST(SentencePieceTokenizerTest, FailedDeserializeUnloadsTokenizer) {
  const std::string flatbuffer = ToFlatbufferOrDie(ReadModelProto());
  ASSERT_FALSE(flatbuffer.empty());
  const std::unique_ptr<Tokenizer> tokenizer =
      CreateSentencePieceTokenizer(flatbuffer);

  EXPECT_FALSE(tokenizer->Deserialize("neither a ModelProto nor a FlatBuffer"));

  // The rejected bytes must not be kept either.
  EXPECT_TRUE(tokenizer->Serialize().empty());

  std::vector<int> ids;
  EXPECT_FALSE(tokenizer->Encode(kText, ids));
  std::string decoded;
  const int kIds[] = {1, 2, 3};
  EXPECT_FALSE(tokenizer->Decode(hwy::Span<const int>(kIds, 3), decoded));
}

}  // namespace
}  // namespace gcpp
