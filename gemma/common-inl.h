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

// Include guard for non-SIMD code.
#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_COMMON_INL_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_COMMON_INL_H_

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <cmath>

#include "gemma/activations.h"

#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {
}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_COMMON_INL_H_

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_COMMON_TOGGLE) == defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_COMMON_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_COMMON_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_COMMON_TOGGLE
#endif

#include "gemma/ops.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// `EmbeddingScaling` can be constexpr only if `Sqrt` and `hwy::ConvertScalarTo`
// are both constexpr
#if HWY_COMPILER_GCC_ACTUAL
#define GEMMA_CONSTEXPR_EMBSCALING HWY_BF16_CONSTEXPR
#else
#define GEMMA_CONSTEXPR_EMBSCALING
#endif

template <typename TConfig>
GEMMA_CONSTEXPR_EMBSCALING float EmbeddingScaling() {
  // Round to bf16 to match Gemma's Embedder, which casts before mul.
  return hwy::ConvertScalarTo<float>(hwy::ConvertScalarTo<hwy::bfloat16_t>(
      Sqrt(static_cast<float>(TConfig::kModelDim))));
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
