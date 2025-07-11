// Copyright 2025 Google LLC
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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_VIT_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_VIT_H_

// Declares vision transformer FFW/Prefill for all SIMD targets.

#include <stddef.h>

#include "gemma/gemma.h"
#include "hwy/highway.h"

namespace gcpp {

// Passed to HWY_VISIT_TARGETS; declares for one target.
#define GEMMA_DECL_VIT(TARGET, NAMESPACE)                                      \
  namespace NAMESPACE {                                                        \
  void FFWVit(const LayerWeightsPtrs& layer, Activations& activations,         \
              MatMulEnv& env);                                                 \
                                                                               \
  void PrefillVit(const ModelConfig& model_config, const WeightsPtrs& weights, \
                  const RuntimeConfig& runtime_config, const Image& image,     \
                  ImageTokens& image_tokens, Activations& activations,         \
                  MatMulEnv& env);                                             \
  /* NOLINTNEXTLINE(google-readability-namespace-comments) */                  \
  }  // namespace NAMESPACE

// Function declarations for each SIMD target. Allows direct call from the
// per-target namespace. We may later replace this with dynamic dispatch if
// the overhead is acceptable.
HWY_VISIT_TARGETS(GEMMA_DECL_VIT)

#undef GEMMA_DECL_VIT

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_VIT_H_
