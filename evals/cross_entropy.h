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

#ifndef THIRD_PARTY_GEMMA_CPP_EVALS_CROSS_ENTROPY_H_
#define THIRD_PARTY_GEMMA_CPP_EVALS_CROSS_ENTROPY_H_

#include <stddef.h>

#include <vector>

#include "gemma/gemma.h"

namespace gcpp {

float ComputeCrossEntropy(const Gemma& gemma, size_t max_generated_tokens,
                          const std::vector<int>& prompt, KVCache& kv_cache,
                          MatMulEnv& env, int verbosity);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_EVALS_CROSS_ENTROPY_H_
