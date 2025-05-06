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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_COMMON_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_COMMON_H_

#include <stddef.h>

#include <string>

#include "gemma/configs.h"  // IWYU pragma: export

namespace gcpp {

// Wraps the given prompt using the expected control tokens for IT models.
// DEPRECATED, use WrapAndTokenize instead if a tokenized return value is fine.
void Wrap(const ModelConfig& config, size_t pos, std::string& prompt);

// Returns the scale value to use for the embedding (basically sqrt model_dim).
// Also used by backprop/.
float EmbeddingScaling(size_t model_dim);

// Returns the scale value to use for the query in the attention computation.
float ChooseQueryScale(const ModelConfig& config);

void RangeChecks(const ModelConfig& weights_config,
                 size_t& max_generated_tokens, size_t prompt_size);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_COMMON_H_
