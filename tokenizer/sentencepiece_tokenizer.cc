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

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "tokenizer/tokenizer.h"
#include "hwy/aligned_allocator.h"  // hwy::Span
#include "hwy/base.h"  // HWY_ABORT
#include "hwy/profiler.h"
// copybara:import_next_line:sentencepiece
#include "src/sentencepiece_processor.h"
#include "tokenizer/sentencepiece_tokenizer.h"

namespace gcpp {

namespace {

class SentencePieceTokenizer : public Tokenizer {
 public:
  SentencePieceTokenizer() = default;

  std::string Serialize() const override {
    return spp_.serialized_model_proto();
  }

  bool Deserialize(std::string_view data) override {
    PROFILER_ZONE("Startup.tokenizer");
    return spp_.LoadFromSerializedProto(data).ok();
  }

  bool Encode(std::string_view input, std::vector<int>& ids) const override {
    return spp_.Encode(input, &ids).ok();
  }

  bool Decode(hwy::Span<const int> ids,
              std::string& detokenized) const override {
    return spp_.Decode(std::vector<int>(ids.begin(), ids.end()),
                       &detokenized).ok();
  }

 private:
  sentencepiece::SentencePieceProcessor spp_;
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
