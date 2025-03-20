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

#include "compression/io.h"  // Path
#include "gemma/common.h"    // ModelInfo

namespace gcpp {

// The tokenizer's end of sentence and beginning of sentence token ids.
constexpr int EOS_ID = 1;
constexpr int SECONDARY_EOS_ID = 106;  // for Gemma 3
constexpr int BOS_ID = 2;

class GemmaTokenizer {
 public:
  GemmaTokenizer();
  explicit GemmaTokenizer(const Path& tokenizer_path);

  // must come after definition of Impl
  ~GemmaTokenizer();
  GemmaTokenizer(GemmaTokenizer&& other);
  GemmaTokenizer& operator=(GemmaTokenizer&& other);

  std::string Serialize() const;
  void Deserialize(const std::string& tokenizer_proto);

  bool Encode(const std::string& input, std::vector<std::string>* pieces) const;
  bool Encode(const std::string& input, std::vector<int>* ids) const;
  bool Decode(const std::vector<int>& ids, std::string* detokenized) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

std::vector<int> WrapAndTokenize(const GemmaTokenizer& tokenizer,
                                 const ModelInfo& info, size_t pos,
                                 std::string& prompt);

std::vector<int> WrapVLM(const GemmaTokenizer& tokenizer, const ModelInfo& info,
                         size_t pos, std::vector<int>& tokens,
                         size_t image_batch_size, size_t max_image_batch_size);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_TOKENIZER_H_
