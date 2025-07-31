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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_TOKENIZER_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_TOKENIZER_H_

#include <stddef.h>

#include <memory>
#include <string>
#include <vector>

#include "gemma/configs.h"  // PromptWrapping

namespace gcpp {

constexpr int BOS_ID = 2;  // beginning of sequence

// To avoid the complexity of storing the tokenizer into testdata/ or
// downloading from gs://, while still always writing a blob for the tokenizer,
// but also avoiding empty blobs, we store this placeholder string.
constexpr const char* kMockTokenizer = "unavailable";

class GemmaTokenizer {
  // These must be defined after the definition of `Impl`.
 public:
  // If unavailable, pass `kMockTokenizer`.
  explicit GemmaTokenizer(const std::string& tokenizer_proto);
  ~GemmaTokenizer();
  GemmaTokenizer(GemmaTokenizer&& other);
  GemmaTokenizer& operator=(GemmaTokenizer&& other);

  // Returns `kMockTokenizer` if unavailable.
  std::string Serialize() const;

  // Returns false on failure or if unavailable.
  bool Encode(const std::string& input, std::vector<std::string>* pieces) const;
  bool Encode(const std::string& input, std::vector<int>* ids) const;
  bool Decode(const std::vector<int>& ids, std::string* detokenized) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class GemmaChatTemplate {
 public:
  // No effect if `tokenizer` is unavailable, but any other method may abort.
  GemmaChatTemplate(const GemmaTokenizer& tokenizer, Model model);

  // Given prompt tokens, this returns the wrapped prompt including BOS and
  // any "start_of_turn" structure required by the model.
  std::vector<int> Apply(size_t pos, const std::vector<int>& ids) const;
  std::vector<int> WrapPali(const std::vector<int>& text_part,
                            size_t image_batch_size) const;
  std::vector<int> WrapVLM(const std::vector<int>& text_part,
                           size_t image_batch_size) const;

 private:
  std::vector<int> sot_user_;
  std::vector<int> sot_model_;
  std::vector<int> eot_;
  std::vector<int> pali_sep_;
  std::vector<int> vlm_soi_;
  std::vector<int> vlm_eoi_;
};

std::vector<int> WrapAndTokenize(const GemmaTokenizer& tokenizer,
                                 const GemmaChatTemplate& chat_template,
                                 PromptWrapping wrapping, size_t pos,
                                 const std::string& prompt);

std::vector<int> WrapAndTokenize(const GemmaTokenizer& tokenizer,
                                 const GemmaChatTemplate& chat_template,
                                 PromptWrapping wrapping, size_t pos,
                                 const std::string& prompt,
                                 size_t image_batch_size);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_TOKENIZER_H_
