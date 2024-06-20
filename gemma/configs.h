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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_CONFIGS_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_CONFIGS_H_

// Model configurations

#include <stddef.h>

#include <array>

#include "hwy/base.h"                // hwy::bfloat16_t

namespace gcpp {

// Allow changing pre-allocated kv cache size as a compiler flag
#ifndef GEMMA_MAX_SEQLEN
#define GEMMA_MAX_SEQLEN 4096
#endif  // !GEMMA_MAX_SEQLEN

// Allow changing k parameter of `SampleTopK` as a compiler flag
#ifndef GEMMA_TOPK
#define GEMMA_TOPK 1
#endif  // !GEMMA_TOPK

// Allow changing upper bound on threads as a compiler flag
#ifndef GEMMA_MAX_THREADS
#define GEMMA_MAX_THREADS 128
#endif  // !GEMMA_MAX_THREADS

static constexpr size_t kSeqLen = GEMMA_MAX_SEQLEN;
static constexpr size_t kTopK = GEMMA_TOPK;
static constexpr size_t kMaxThreads = GEMMA_MAX_THREADS;

using EmbedderInputT = hwy::bfloat16_t;

enum class LayerAttentionType {
  kGemma,
  kGriffinRecurrentBlock,
};

template <size_t kNum>
constexpr std::array<LayerAttentionType, kNum> FixedLayerConfig(
    LayerAttentionType type) {
  std::array<LayerAttentionType, kNum> config = {};
  for (LayerAttentionType& l : config) {
    l = type;
  }
  return config;
}

template <size_t kNumLayers>
constexpr size_t NumLayersOfTypeBefore(
    const std::array<LayerAttentionType, kNumLayers>& layers,
    LayerAttentionType type, size_t num) {
  size_t count = 0;
  for (size_t i = 0; i < num; i++) {
    if (layers[i] == type) count++;
  }
  return count;
}

template <class TConfig, typename = void>
struct CacheLayerSize {
  constexpr size_t operator()() const {
    return TConfig::kKVHeads * TConfig::kQKVDim * 2;
  }
};

template <class TConfig, typename = void>
struct CachePosSize {
  constexpr size_t operator()() const {
    return TConfig::kGemmaLayers * CacheLayerSize<TConfig>()();
  }
};

struct ConfigNoSSM {
  static constexpr int kGriffinLayers = 0;

  static constexpr int kConv1dWidth = 0;
  static constexpr bool kFFBiases = false;
  static constexpr bool kSoftmaxAttnOutputBiases = false;
  static constexpr bool kUseHalfRope = false;
  static constexpr bool kUseLocalAttention = false;
  static constexpr bool kInterleaveQKV = true;
  static constexpr int kNumTensorScales = 0;
};

template <typename TWeight>
struct ConfigGemma7B : public ConfigNoSSM {
  using Weight = TWeight;  // make accessible where we only have a TConfig

  static constexpr int kSeqLen = gcpp::kSeqLen;
  static constexpr int kVocabSize = 256000;
  static constexpr std::array<LayerAttentionType, 28> kLayerConfig =
      FixedLayerConfig<28>(LayerAttentionType::kGemma);
  static constexpr int kLayers = kLayerConfig.size();
  static constexpr int kGemmaLayers = kLayers;
  static constexpr int kModelDim = 3072;
  static constexpr int kFFHiddenDim = 16 * 3072 / 2;  // = 24576
  static constexpr int kHeads = 16;
  static constexpr int kKVHeads = 16;  // standard MHA
  static constexpr int kQKVDim = 256;  // query size == key size == value size
  static constexpr int kTopK = gcpp::kTopK;
  static constexpr bool kAbsolutePE = false;
  static constexpr bool kPostNormScale = false;
};

template <typename TWeight>
struct ConfigGemma2B : public ConfigNoSSM {
  using Weight = TWeight;  // make accessible where we only have a TConfig

  static constexpr int kSeqLen = gcpp::kSeqLen;
  static constexpr int kVocabSize = 256000;
  static constexpr std::array<LayerAttentionType, 18> kLayerConfig =
      FixedLayerConfig<18>(LayerAttentionType::kGemma);
  static constexpr int kLayers = kLayerConfig.size();
  static constexpr int kGemmaLayers = kLayers;
  static constexpr int kModelDim = 2048;
  static constexpr int kFFHiddenDim = 16 * 2048 / 2;  // = 16384
  static constexpr int kHeads = 8;
  static constexpr int kKVHeads = 1;
  static constexpr int kQKVDim = 256;  // query size == key size == value size
  static constexpr int kTopK = gcpp::kTopK;
  static constexpr bool kAbsolutePE = false;
  static constexpr bool kPostNormScale = false;
};

template <typename TWeight>
struct ConfigGemmaTiny : public ConfigNoSSM {
  using Weight = TWeight;  // make accessible where we only have a TConfig

  static constexpr int kSeqLen = 32;
  static constexpr int kVocabSize = 64;
  static constexpr std::array<LayerAttentionType, 3> kLayerConfig =
      FixedLayerConfig<3>(LayerAttentionType::kGemma);
  static constexpr int kLayers = kLayerConfig.size();
  static constexpr int kGemmaLayers = kLayers;
  static constexpr int kModelDim = 128;
  static constexpr int kFFHiddenDim = 256;
  static constexpr int kHeads = 4;
  static constexpr int kKVHeads = 1;
  static constexpr int kQKVDim = 16;  // query size == key size == value size
  static constexpr int kTopK = gcpp::kTopK;
  static constexpr bool kAbsolutePE = false;
  static constexpr bool kPostNormScale = false;
};

template <typename TWeight>
struct ConfigGriffin2B {
  using Weight = TWeight;  // make accessible where we only have a TConfig

  // Griffin uses local attention, so kSeqLen is actually the local attention
  // window.
  static constexpr int kSeqLen = 2048;
  static constexpr int kVocabSize = 256000;
  static constexpr std::array<LayerAttentionType, 26> kLayerConfig = {
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGemma,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGemma,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGemma,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGemma,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGemma,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGemma,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGemma,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGemma,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGriffinRecurrentBlock,
  };
  static constexpr int kLayers = kLayerConfig.size();
  static constexpr int kGemmaLayers =
      NumLayersOfTypeBefore(kLayerConfig, LayerAttentionType::kGemma, kLayers);
  static constexpr int kGriffinLayers =
      NumLayersOfTypeBefore(kLayerConfig,
                            LayerAttentionType::kGriffinRecurrentBlock,
                            kLayers);
  static constexpr int kModelDim = 2560;
  static constexpr int kFFHiddenDim = 7680;
  static constexpr int kHeads = 10;
  static constexpr int kKVHeads = 1;
  static constexpr int kQKVDim = 256;  // query size == key size == value size
  static constexpr int kTopK = gcpp::kTopK;
  static constexpr bool kAbsolutePE = false;
  static constexpr bool kPostNormScale = false;

  // SSM config.
  static constexpr int kConv1dWidth = 4;
  static constexpr bool kFFBiases = true;
  static constexpr bool kSoftmaxAttnOutputBiases = true;
  static constexpr bool kUseHalfRope = true;
  static constexpr bool kUseLocalAttention = true;
  static constexpr bool kInterleaveQKV = false;
  static constexpr int kNumTensorScales = 140;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_CONFIGS_H_
