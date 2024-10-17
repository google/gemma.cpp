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

#include "backprop/forward.h"

#include "backprop/activations.h"
#include "backprop/prompt.h"
#include "gemma/common.h"
#include "gemma/configs.h"
#include "util/allocator.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

// Compiles this file for multiple architectures via "foreach_target.h", to
// which we pass the filename via macro 'argument'.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "backprop/forward.cc"  // NOLINT
#include "hwy/foreach_target.h"        // IWYU pragma: keep

#include "hwy/highway.h"
// After highway.h
#include "backprop/forward-inl.h"
#include "gemma/weights.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

float CrossEntropyLossForwardPassT(const Prompt& prompt,
                                   const ModelWeightsPtrs<float>& weights,
                                   ForwardPass<float>& forward,
                                   RowVectorBatch<float>& inv_timescale,
                                   hwy::ThreadPool& pool) {
  return CrossEntropyLossForwardPass(prompt.tokens, prompt.context_size,
                                     weights, forward, inv_timescale, pool);
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {

HWY_EXPORT(CrossEntropyLossForwardPassT);

float CrossEntropyLossForwardPass(const Prompt& prompt,
                                  const ModelWeightsPtrs<float>& weights,
                                  ForwardPass<float>& forward,
                                  RowVectorBatch<float>& inv_timescale,
                                  hwy::ThreadPool& pool) {
  return HWY_DYNAMIC_DISPATCH(CrossEntropyLossForwardPassT)(
      prompt, weights, forward, inv_timescale, pool);
}

}  // namespace gcpp
#endif  // HWY_ONCE
