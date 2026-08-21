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

#ifndef THIRD_PARTY_GEMMA_CPP_TOKENIZER_TOKENIZER_H_
#define THIRD_PARTY_GEMMA_CPP_TOKENIZER_TOKENIZER_H_

#include <string>
#include <string_view>
#include <vector>

#include "hwy/aligned_allocator.h"  // hwy::Span

namespace gcpp {

// Interface for tokenizer backends.
// Implementations:
// 1. the SentencePiece wrapper (tokenizer/sentencepiece_tokenizer.h)
// 2. the from-scratch HuggingFace BPE engine (tokenizer/bpe_tokenizer.h).
// `GemmaTokenizer` holds a pointer to this interface and delegates all
// operations to it.
class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  // Returns the bytes to embed in the `.sbs` `tokenizer` blob.
  virtual std::string Serialize() const = 0;

  // Deserializes the tokenizer from `tokenizer` blob. Returns false on failure.
  virtual bool Deserialize(std::string_view data) = 0;

  // Encodes `input` into token ids. No BOS is added. Returns false on failure.
  virtual bool Encode(std::string_view input,
                      std::vector<int>& ids) const = 0;

  // Decodes `ids` back into text. Returns false on failure.
  virtual bool Decode(hwy::Span<const int> ids,
                      std::string& detokenized) const = 0;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_TOKENIZER_TOKENIZER_H_
