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

#include "gemma/forward.h"

// Compiles this file for multiple architectures via "foreach_target.h", to
// which we pass the filename via macro 'argument'.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/forward.cc"  // NOLINT
#include "hwy/foreach_target.h"        // IWYU pragma: keep

#include "gemma/forward-inl.h"
#include "gemma/weights.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <typename TConfig>
float CrossEntropyLossForwardPass(const Prompt& prompt,
                                  const ByteStorageT& weights_u8,
                                  ByteStorageT& forward_u8,
                                  hwy::ThreadPool& pool) {
  const auto& weights =
      *reinterpret_cast<WeightsF<TConfig>*>(weights_u8.get());
  auto& forward =
      *reinterpret_cast<ForwardPass<float, TConfig>*>(forward_u8.get());
  return CrossEntropyLossForwardPass<TConfig, WeightsF, LayerF>(
      prompt.tokens, prompt.context_size, weights, forward, pool);
}

float CrossEntropyLossForwardPassT(Model model, const Prompt& prompt,
                                   const ByteStorageT& weights,
                                   ByteStorageT& forward,
                                   hwy::ThreadPool& pool) {
  switch (model) {
    case Model::GEMMA_2B:
      return CrossEntropyLossForwardPass<ConfigGemma2B>(
          prompt, weights, forward, pool);
    case Model::GEMMA_TINY:
      return CrossEntropyLossForwardPass<ConfigGemmaTiny>(
          prompt, weights, forward, pool);
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {

HWY_EXPORT(CrossEntropyLossForwardPassT);

float CrossEntropyLossForwardPass(
    const Model& model, const Prompt& prompt, const ByteStorageT& weights,
    ByteStorageT& forward, hwy::ThreadPool& pool) {
  return HWY_DYNAMIC_DISPATCH(CrossEntropyLossForwardPassT)(
      model, prompt, weights, forward, pool);
}

}  // namespace gcpp
#endif  // HWY_ONCE
