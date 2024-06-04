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

#include "hwy/aligned_allocator.h"

namespace gcpp {

using ByteStorageT = hwy::AlignedFreeUniquePtr<uint8_t[]>;

// Model variants: see configs.h for details.
enum class Model { GEMMA_2B, GEMMA_7B, GRIFFIN_2B, GEMMA_TINY };

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_COMMON_H_
