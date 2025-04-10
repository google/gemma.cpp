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

#include "backprop/backward.h"

#include "backprop/activations.h"
#include "backprop/prompt.h"
#include "gemma/weights.h"
#include "util/mat.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

// Compiles this file for multiple architectures via "foreach_target.h", to
// which we pass the filename via macro 'argument'.
// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "backprop/backward.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"        // IWYU pragma: keep

#include "hwy/highway.h"
// After highway.h
#include "backprop/backward-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

void CrossEntropyLossBackwardPassT(const Prompt& prompt,
                                   const ModelWeightsPtrs<float>& weights,
                                   const ForwardPass<float>& forward,
                                   ModelWeightsPtrs<float>& grad,
                                   ForwardPass<float>& backward,
                                   RowVectorBatch<float>& inv_timescale,
                                   hwy::ThreadPool& pool) {
  CrossEntropyLossBackwardPassInl(prompt, weights, forward, grad, backward,
                                  inv_timescale, pool);
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {

HWY_EXPORT(CrossEntropyLossBackwardPassT);

void CrossEntropyLossBackwardPass(const Prompt& prompt,
                                  const ModelWeightsPtrs<float>& weights,
                                  const ForwardPass<float>& forward,
                                  ModelWeightsPtrs<float>& grad,
                                  ForwardPass<float>& backward,
                                  RowVectorBatch<float>& inv_timescale,
                                  hwy::ThreadPool& pool) {
  return HWY_DYNAMIC_DISPATCH(CrossEntropyLossBackwardPassT)(
      prompt, weights, forward, grad, backward, inv_timescale, pool);
}

}  // namespace gcpp
#endif  // HWY_ONCE
