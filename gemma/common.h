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

#include <math.h>  // sqrtf
#include <stdint.h>

#include <string>

#include "gemma/configs.h"  // IWYU pragma: export
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"  // ConvertScalarTo

namespace gcpp {

using ByteStorageT = hwy::AlignedFreeUniquePtr<uint8_t[]>;

template <typename T>
ByteStorageT AllocateSizeof() {
  return hwy::AllocateAligned<uint8_t>(sizeof(T));
}

// Model variants: see configs.h for details.
enum class Model { GEMMA_2B, GEMMA_7B, GRIFFIN_2B, GEMMA_TINY };

enum class ModelTraining { GEMMA_IT, GEMMA_PT };

// Returns the return value of Func<T>().operator() called with `args`, where
// `T` is selected based on `model`.
//
// This is used to implement type-erased functions such as
// LoadCompressedWeights, which can be called from other .cc files, by calling a
// functor LoadCompressedWeightsT, which has a template argument. `Func` must
// be a functor because function templates cannot be passed as a template
// template argument, and we prefer to avoid the overhead of std::function.
//
// This function avoids having to update all call sites when we extend `Model`.
template <template <typename Config> class Func, typename... Args>
decltype(auto) CallFunctorForModel(Model model, Args&&... args) {
  switch (model) {
    case Model::GEMMA_TINY:
      return Func<ConfigGemmaTiny>()(std::forward<Args>(args)...);
    case Model::GEMMA_2B:
      return Func<ConfigGemma2B>()(std::forward<Args>(args)...);
    case Model::GEMMA_7B:
      return Func<ConfigGemma7B>()(std::forward<Args>(args)...);
    case Model::GRIFFIN_2B:
      return Func<ConfigGriffin2B>()(std::forward<Args>(args)...);
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

// Like CallFunctorForModel, but for SIMD function templates. This is a macro
// because it boils down to N_SSE4::FUNC, which would not work if FUNC was a
// normal function argument.
#define GEMMA_EXPORT_AND_DISPATCH_MODEL(MODEL, FUNC, ARGS)          \
  switch (MODEL) {                                                  \
    case Model::GEMMA_TINY: {                                       \
      HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(FUNC<ConfigGemmaTiny>)      \
      ARGS;                                                         \
      break;                                                        \
    }                                                               \
    case Model::GEMMA_2B: {                                         \
      HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(FUNC<ConfigGemma2B>)        \
      ARGS;                                                         \
      break;                                                        \
    }                                                               \
    case Model::GEMMA_7B: {                                         \
      HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(FUNC<ConfigGemma7B>)        \
      ARGS;                                                         \
      break;                                                        \
    }                                                               \
    case Model::GRIFFIN_2B: {                                       \
      HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(FUNC<ConfigGriffin2B>)      \
      ARGS;                                                         \
      break;                                                        \
    }                                                               \
    default:                                                        \
      HWY_ABORT("Model type %d unknown.", static_cast<int>(MODEL)); \
  }

// Returns error string or nullptr if OK.
// Thread-hostile.
const char* ParseModelTypeAndTraining(const std::string& model_flag,
                                      Model& model, ModelTraining& training);

// __builtin_sqrt is not constexpr as of Clang 17.
#if HWY_COMPILER_GCC_ACTUAL
#define GEMMA_CONSTEXPR_SQRT constexpr
static GEMMA_CONSTEXPR_SQRT HWY_INLINE float Sqrt(float x) {
  return __builtin_sqrt(x);
}
#else
#define GEMMA_CONSTEXPR_SQRT
static GEMMA_CONSTEXPR_SQRT HWY_INLINE float Sqrt(float x) { return sqrtf(x); }
#endif

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

static HWY_INLINE GEMMA_CONSTEXPR_EMBSCALING float EmbeddingScaling(
    size_t model_dim) {
  // Round to bf16 to match Gemma's Embedder, which casts before mul.
  return hwy::ConvertScalarTo<float>(hwy::ConvertScalarTo<hwy::bfloat16_t>(
      Sqrt(static_cast<float>(model_dim))));
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_COMMON_H_
