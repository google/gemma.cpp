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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_PROMPT_UTILS_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_PROMPT_UTILS_H_

#include <stddef.h>

#include <string>
#include <vector>

#include "gemma/common.h"     // ModelInfo
#include "gemma/tokenizer.h"  // GemmaTokenizer, GemmaChatTemplate

namespace gcpp {

// Adds BOS token and possibly 'turn' annotations, which depend on `info`
// and `pos`, the number of tokens decoded so far; returns the corresponding
// tokens. Asserts that tokenization is successful.
// Handles text-only prompts.
std::vector<int> WrapAndTokenize(const GemmaTokenizer& tokenizer,
                                 const GemmaChatTemplate& chat_template,
                                 const ModelInfo& info, size_t pos,
                                 const std::string& prompt);

// Adds BOS token and possibly 'turn' annotations, which depend on `info`
// and `pos`, the number of tokens decoded so far; returns the corresponding
// tokens. Asserts that tokenization is successful.
// Handles multimodal prompts (text + image). `image_batch_size` indicates
// the number of tokens representing the image.
std::vector<int> WrapAndTokenize(const GemmaTokenizer& tokenizer,
                                 const GemmaChatTemplate& chat_template,
                                 const ModelInfo& info, size_t pos,
                                 const std::string& prompt,
                                 size_t image_batch_size);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_PROMPT_UTILS_H_
