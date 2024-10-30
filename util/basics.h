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

#ifndef THIRD_PARTY_GEMMA_CPP_UTIL_BASICS_H_
#define THIRD_PARTY_GEMMA_CPP_UTIL_BASICS_H_

// IWYU pragma: begin_exports
#include <stddef.h>
#include <stdint.h>

#include "hwy/base.h"
// IWYU pragma: end_exports

#if HWY_IS_MSAN
#include <sanitizer/msan_interface.h>
#endif

namespace gcpp {

using BF16 = hwy::bfloat16_t;

static inline void MaybeCheckInitialized(const void* ptr, size_t size) {
#if HWY_IS_MSAN
  __msan_check_mem_is_initialized(ptr, size);
#else
  (void)ptr;
  (void)size;
#endif
}

// Shared between gemma.h and ops-inl.h.
struct TokenAndProb {
  int token;
  float prob;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_BASICS_H_
