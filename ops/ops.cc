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

#include "ops/ops.h"

#include <stddef.h>
#include <stdint.h>

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ops/ops.cc"  // NOLINT
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

// After highway.h
#include "ops/ops-inl.h"
// clang-format on

#if HWY_ONCE
namespace gcpp {
HWY_EXPORT(ApplyLogitMaskKernel);

void ApplyLogitMaskKernel(hwy::Span<float> logits,
                          const uint64_t* HWY_RESTRICT mask_words,
                          size_t vocab_size, float mask_value) {
  HWY_DYNAMIC_DISPATCH(ApplyLogitMaskKernel)(logits, mask_words, vocab_size,
                                             mask_value);
}
}  // namespace gcpp
#endif  // HWY_ONCE
