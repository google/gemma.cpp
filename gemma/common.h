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
#include <stddef.h>

#include <string>

#include "compression/compress.h"
#include "gemma/configs.h"  // IWYU pragma: export
#include "hwy/base.h"  // ConvertScalarTo

namespace gcpp {

// Model variants: see configs.h for details. When adding a new one, also
// update GEMMA_FOREACH* and Call* below, and add instantiations/*.cc.
enum class Model {
  GEMMA_2B,
  GEMMA_7B,
  GEMMA2_9B,
  GEMMA2_27B,
  GRIFFIN_2B,
  GEMMA_TINY,
  GEMMA2_2B,
  PALIGEMMA_224,
};

// Instruction-tuned models require extra 'turn structure' tokens in prompts.
enum class ModelTraining { GEMMA_IT, GEMMA_PT, PALIGEMMA };

// Tensor types for loading weights. When adding a new one, also
// update GEMMA_FOREACH* and Call* below, and add instantiations/*.cc.
enum class Type { kF32, kBF16, kSFP };

// TODO(janwas): merge with functions below.
struct ModelInfo {
  Model model;
  ModelTraining training;
  Type weight;
};

// Returns error string or nullptr if OK.
// Thread-hostile.
const char* ParseModelTypeAndTraining(const std::string& model_flag,
                                      Model& model, ModelTraining& training);
const char* ParseType(const std::string& type_string, Type& type);

// Inverse of ParseModelTypeAndTraining.
const char* ModelString(Model model, ModelTraining training);
const char* StringFromType(Type type);

void Wrap(const ModelInfo& info, size_t pos, std::string& prompt);

// Returns the return value of FuncT<Config*<TWeight>>().operator()(args), where
// Config* is selected via `model`. Typically called by CallForModelAndWeight,
// but can also be called directly when FuncT does not actually use TWeight.
//
// Note that a T prefix indicates a concrete type template argument, whereas a
// T suffix indicates the argument is itself a template.
//
// `FuncT` must be a functor because function templates cannot be passed as a
// template template argument, and we prefer to avoid the overhead of
// std::function.
template <typename TWeight, template <typename TConfig> class FuncT,
          typename... TArgs>
decltype(auto) CallForModel(Model model, TArgs&&... args) {
  switch (model) {
    case Model::GEMMA_TINY:
      return FuncT<ConfigGemmaTiny<TWeight>>()(std::forward<TArgs>(args)...);
    case Model::GEMMA_2B:
      return FuncT<ConfigGemma2B<TWeight>>()(std::forward<TArgs>(args)...);
    case Model::GEMMA_7B:
      return FuncT<ConfigGemma7B<TWeight>>()(std::forward<TArgs>(args)...);
    case Model::GEMMA2_9B:
      return FuncT<ConfigGemma2_9B<TWeight>>()(std::forward<TArgs>(args)...);
    case Model::GEMMA2_27B:
      return FuncT<ConfigGemma2_27B<TWeight>>()(std::forward<TArgs>(args)...);
    case Model::GRIFFIN_2B:
      return FuncT<ConfigGriffin2B<TWeight>>()(std::forward<TArgs>(args)...);
    case Model::GEMMA2_2B:
      return FuncT<ConfigGemma2_2B<TWeight>>()(std::forward<TArgs>(args)...);
    case Model::PALIGEMMA_224:
      return FuncT<ConfigPaliGemma_224<TWeight>>()(
          std::forward<TArgs>(args)...);
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

// Returns the return value of FuncT<TConfig>().operator()(args),
// where `TConfig` is selected based on `model` and `weight`.

// This makes it easy to extend `Model` or `Type` without updating callers.
//
// Usage example: LoadWeights is type-erased so that it can be called from other
// .cc files. It uses this function to call the appropriate instantiation of a
// template functor LoadCompressedWeightsT<TConfig>.
template <template <typename TConfig> class FuncT, typename... TArgs>
decltype(auto) CallForModelAndWeight(Model model, Type weight,
                                     TArgs&&... args) {
  switch (weight) {
    case Type::kF32:
      return CallForModel<float, FuncT, TArgs...>(  //
          model, std::forward<TArgs>(args)...);
    case Type::kBF16:
      return CallForModel<hwy::bfloat16_t, FuncT, TArgs...>(
          model, std::forward<TArgs>(args)...);
    case Type::kSFP:
      return CallForModel<SfpStream, FuncT, TArgs...>(
          model, std::forward<TArgs>(args)...);
    default:
      HWY_ABORT("Weight type %d unknown.", static_cast<int>(weight));
  }
}

#define GEMMA_FOREACH_WEIGHT(X, CONFIGT) \
  X(CONFIGT, float)                      \
  X(CONFIGT, hwy::bfloat16_t)            \
  X(CONFIGT, SfpStream)

#define GEMMA_FOREACH_CONFIG_AND_WEIGHT(X)              \
  GEMMA_FOREACH_WEIGHT(X, ConfigGemmaTiny)              \
  GEMMA_FOREACH_WEIGHT(X, ConfigGemma2B)                \
  GEMMA_FOREACH_WEIGHT(X, ConfigGemma7B)                \
  GEMMA_FOREACH_WEIGHT(X, ConfigGriffin2B)              \
  GEMMA_FOREACH_WEIGHT(X, ConfigGemma2_2B)              \
  GEMMA_FOREACH_WEIGHT(X, ConfigGemma2_9B)              \
  GEMMA_FOREACH_WEIGHT(X, ConfigGemma2_27B)             \
  GEMMA_FOREACH_WEIGHT(X, ConfigPaliGemma_224)          \
  static_assert(true, "Allow trailing ;")

// Used by GEMMA_EXPORT_AND_DISPATCH. For a given TWEIGHT (e.g. float),
// calls FUNC<ConfigT<TWEIGHT>> where ConfigT is chosen via MODEL enum.
#define GEMMA_DISPATCH_MODEL(MODEL, TWEIGHT, FUNC, ARGS)                       \
  switch (MODEL) {                                                             \
    case Model::GEMMA_TINY: {                                                  \
      using CP = ConfigPair<ConfigGemmaTiny<TWEIGHT>, ConfigGemmaTiny<float>>; \
      HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(FUNC<CP>) ARGS;                        \
      break;                                                                   \
    }                                                                          \
    case Model::GEMMA_2B: {                                                    \
      using CP = ConfigPair<ConfigGemma2B<TWEIGHT>, ConfigGemma2B<float>>;     \
      HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(FUNC<CP>) ARGS;                        \
      break;                                                                   \
    }                                                                          \
    case Model::GEMMA_7B: {                                                    \
      using CP = ConfigPair<ConfigGemma7B<TWEIGHT>, ConfigGemma7B<float>>;     \
      HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(FUNC<CP>) ARGS;                        \
      break;                                                                   \
    }                                                                          \
    case Model::GRIFFIN_2B: {                                                  \
      using CP = ConfigPair<ConfigGriffin2B<TWEIGHT>, ConfigGriffin2B<float>>; \
      HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(FUNC<CP>) ARGS;                        \
      break;                                                                   \
    }                                                                          \
    case Model::GEMMA2_2B: {                                                   \
      using CP = ConfigPair<ConfigGemma2_2B<TWEIGHT>, ConfigGemma2_2B<float>>; \
      HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(FUNC<CP>) ARGS;                        \
      break;                                                                   \
    }                                                                          \
    case Model::GEMMA2_9B: {                                                   \
      using CP = ConfigPair<ConfigGemma2_9B<TWEIGHT>, ConfigGemma2_9B<float>>; \
      HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(FUNC<CP>) ARGS;                        \
      break;                                                                   \
    }                                                                          \
    case Model::GEMMA2_27B: {                                                  \
      using CP =                                                               \
          ConfigPair<ConfigGemma2_27B<TWEIGHT>, ConfigGemma2_27B<float>>;      \
      HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(FUNC<CP>) ARGS;                        \
      break;                                                                   \
    }                                                                          \
    case Model::PALIGEMMA_224: {                                               \
      using CP = ConfigPair<ConfigPaliGemma_224<TWEIGHT>,                      \
                            ConfigPaliGemma_224<float>>;                       \
      HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(FUNC<CP>) ARGS;                        \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      HWY_ABORT("Model type %d unknown.", static_cast<int>(MODEL));            \
  }

// Like CallForModelAndWeight, but for SIMD function templates. This is a macro
// because it boils down to N_SSE4::FUNC, which would not work if FUNC was a
// normal function argument. MODEL and WEIGHT are enums.
// For gemma.cc, we use overloaded extern functions for faster builds. However,
// this is still used in compress_weights because its compile time is OK.
#define GEMMA_EXPORT_AND_DISPATCH(MODEL, WEIGHT, FUNC, ARGS)          \
  switch (WEIGHT) {                                                   \
    case Type::kF32:                                                  \
      GEMMA_DISPATCH_MODEL(MODEL, float, FUNC, ARGS);                 \
      break;                                                          \
    case Type::kBF16:                                                 \
      GEMMA_DISPATCH_MODEL(MODEL, hwy::bfloat16_t, FUNC, ARGS);       \
      break;                                                          \
    case Type::kSFP:                                                  \
      GEMMA_DISPATCH_MODEL(MODEL, SfpStream, FUNC, ARGS);             \
      break;                                                          \
    default:                                                          \
      HWY_ABORT("Weight type %d unknown.", static_cast<int>(WEIGHT)); \
  }

// ----------------------------------------------------------------------------
//

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

template <class TConfig>
GEMMA_CONSTEXPR_SQRT float ChooseQueryScale() {
  if (TConfig::kQueryScale == QueryScaleType::SqrtModelDimDivNumHeads)
    return 1.0f /
           Sqrt(static_cast<float>(TConfig::kModelDim / TConfig::kHeads));
  // QueryScaleType::SqrtKeySize
  return 1.0f / Sqrt(static_cast<float>(TConfig::kQKVDim));
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_COMMON_H_
