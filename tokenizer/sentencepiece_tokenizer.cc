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
#include <string_view>
#include <utility>
#include <vector>

#include "tokenizer/tokenizer.h"
#include "hwy/aligned_allocator.h"  // hwy::Span
#include "hwy/base.h"               // HWY_ABORT
#include "hwy/profiler.h"
// copybara:import_next_line:sentencepiece
#include "lite/sentencepiece_lite.h"
// copybara:import_next_line:sentencepiece
#include "lite/sentencepiece_model_converters.h"
// copybara:import_next_line:sentencepiece
#include "sentencepiece_model.pb.h"

namespace gcpp {

namespace {

class SentencePieceTokenizer : public Tokenizer {
 public:
  SentencePieceTokenizer() = default;

  std::string Serialize() const override { return serialized_model_proto_; }

  bool Deserialize(std::string_view data) override {
    PROFILER_ZONE("Startup.tokenizer");
    // `spp_` may hold a view into `serialized_model_proto_` (FlatBuffer mode),
    // so it must be released before the buffer is overwritten.
    spp_.reset();
    serialized_model_proto_.assign(data.data(), data.size());
    spp_ = CreateProcessor(serialized_model_proto_);
    if (!spp_) {
      // Leave the tokenizer fully unloaded, so that Serialize() does not return
      // the rejected bytes.
      serialized_model_proto_.clear();
      return false;
    }
    return true;
  }

  bool Encode(std::string_view input, std::vector<int>& ids) const override {
    if (!spp_) return false;
    return spp_->Encode(input, &ids) == sentencepiece::lite::StatusCode::kOk;
  }

  bool Decode(hwy::Span<const int> ids,
              std::string& detokenized) const override {
    if (!spp_) return false;
    return spp_->Decode(
               sentencepiece::lite::Span<const int>(ids.data(), ids.size()),
               &detokenized) == sentencepiece::lite::StatusCode::kOk;
  }

 private:
  // Creates a processor from `data`, which is either a serialized ModelProto
  // (tried first) or a FlatBuffer (.spm.fb). In the FlatBuffer case the
  // processor keeps a view into `data`, which must outlive it. Returns nullptr
  // on failure.
  static std::unique_ptr<sentencepiece::lite::SentencePieceLiteProcessor>
  CreateProcessor(std::string_view data) {
    std::unique_ptr<sentencepiece::lite::SentencePieceLiteProcessor> spp;
    sentencepiece::ModelProto model_proto;
    if (model_proto.ParseFromString(data)) {
      auto fb_model_or = sentencepiece::lite::ToFlatbuffer(model_proto);
      if (!fb_model_or.ok()) {
        return nullptr;
      }
      spp = std::make_unique<sentencepiece::lite::SentencePieceLiteProcessor>(
          std::make_shared<std::string>(std::move(*fb_model_or)));
    } else {
      spp = std::make_unique<sentencepiece::lite::SentencePieceLiteProcessor>(
          data);
    }
    if (spp->status() != sentencepiece::lite::StatusCode::kOk) {
      return nullptr;
    }
    return spp;
  }

  std::string serialized_model_proto_;
  std::unique_ptr<sentencepiece::lite::SentencePieceLiteProcessor> spp_;
};

}  // namespace

std::unique_ptr<Tokenizer> CreateSentencePieceTokenizer(
    const std::string& tokenizer_proto) {
  auto tokenizer = std::make_unique<SentencePieceTokenizer>();
  if (!tokenizer->Deserialize(tokenizer_proto)) {
    HWY_ABORT("Failed to load tokenizer from %zu byte serialized proto.",
              tokenizer_proto.size());
  }
  return tokenizer;
}

}  // namespace gcpp
