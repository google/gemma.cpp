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

#include "gemma/backward.h"

// Compiles this file for multiple architectures via "foreach_target.h", to
// which we pass the filename via macro 'argument'.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/backward.cc"  // NOLINT
#include "hwy/foreach_target.h"        // IWYU pragma: keep

#include "gemma/backward-inl.h"
#include "gemma/weights.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <typename TConfig>
void CrossEntropyLossBackwardPass(const Prompt& prompt,
                                  const ByteStorageT& weights_u8,
                                  const ByteStorageT& forward_u8,
                                  ByteStorageT& grad_u8,
                                  ByteStorageT& backward_u8,
                                  hwy::ThreadPool& pool) {
  using TWeights = WeightsF<TConfig>;
  const auto& weights = *reinterpret_cast<const TWeights*>(weights_u8.get());
  auto& grad = *reinterpret_cast<TWeights*>(grad_u8.get());
  using TAct = ForwardPass<float, TConfig>;
  const auto& forward = *reinterpret_cast<const TAct*>(forward_u8.get());
  auto& backward = *reinterpret_cast<TAct*>(backward_u8.get());
  CrossEntropyLossBackwardPass(prompt, weights, forward, grad, backward, pool);
}

void CrossEntropyLossBackwardPassT(Model model,
                                   const Prompt& prompt,
                                   const ByteStorageT& weights,
                                   const ByteStorageT& forward,
                                   ByteStorageT& grad,
                                   ByteStorageT& backward,
                                   hwy::ThreadPool& pool) {
  switch (model) {
    case Model::GEMMA_2B:
      CrossEntropyLossBackwardPass<ConfigGemma2B>(
          prompt, weights, forward, grad, backward, pool);
      break;
    case Model::GEMMA_TINY:
      CrossEntropyLossBackwardPass<ConfigGemmaTiny>(
          prompt, weights, forward, grad, backward, pool);
      break;
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {

HWY_EXPORT(CrossEntropyLossBackwardPassT);

void CrossEntropyLossBackwardPass(
    const Model& model, const Prompt& prompt,
    const ByteStorageT& weights, const ByteStorageT& forward,
    ByteStorageT& grad, ByteStorageT& backward, hwy::ThreadPool& pool) {
  return HWY_DYNAMIC_DISPATCH(CrossEntropyLossBackwardPassT)(
      model, prompt, weights, forward, grad, backward, pool);
}

}  // namespace gcpp
#endif  // HWY_ONCE
