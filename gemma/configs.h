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
#include <string>
#include <unordered_set>
#include <vector>

#include "compression/shared.h"  // BF16

namespace gcpp {

// Allow changing pre-allocated kv cache size as a compiler flag
#ifndef GEMMA_MAX_SEQLEN
#define GEMMA_MAX_SEQLEN 4096
#endif  // !GEMMA_MAX_SEQLEN

// Allow changing k parameter of `SampleTopK` as a compiler flag
#ifndef GEMMA_TOPK
#define GEMMA_TOPK 1
#endif  // !GEMMA_TOPK

static constexpr size_t kSeqLen = GEMMA_MAX_SEQLEN;
static constexpr size_t kTopK = GEMMA_TOPK;
static constexpr size_t kVocabSize = 256000;

using EmbedderInputT = BF16;

enum class LayerAttentionType {
  kGemma,
  kGriffinRecurrentBlock,
  kVit,
};

// Post attention and ffw normalization type.
enum class PostNormType {
  None,
  Scale,
};

// Post qk projection operation type.
enum class PostQKType {
  Rope,
  HalfRope,
};

// FFW activation function.
enum class ActivationType {
  Gelu,
};

// Attention query scale.
enum class QueryScaleType {
  SqrtKeySize,
  SqrtModelDimDivNumHeads,
};

// Residual connection type.
enum class ResidualType {
  Add,
};

template <size_t kNum>
std::vector<LayerAttentionType> FixedLayerConfig(LayerAttentionType type) {
  return std::vector<LayerAttentionType>(kNum, type);
}

template <size_t kNum>
std::vector<size_t> FixedAttentionWindowSizes(size_t window_size) {
  return std::vector<size_t>(kNum, window_size);
}

// Repeat window_size_pattern for kNum / kPatternSize times.
template <size_t kNum, size_t kPatternSize>
std::vector<size_t> RepeatedAttentionWindowSizes(
    const std::array<size_t, kPatternSize>& window_size_pattern) {
  static_assert(kNum % kPatternSize == 0,
                "kNum must be a multiple of kPatternSize");
  std::vector<size_t> window_size_configs(kNum);
  for (size_t i = 0; i < kNum; ++i) {
    window_size_configs[i] = window_size_pattern[i % kPatternSize];
  }
  return window_size_configs;
}

// Model variants: see configs.cc for details.
enum class Model {
  UNKNOWN,
  GEMMA_2B,
  GEMMA_7B,
  GEMMA2_9B,
  GEMMA2_27B,
  GRIFFIN_2B,
  GEMMA_TINY,
  GEMMA2_2B,
  PALIGEMMA_224,
};

struct LayerConfig {
  size_t CacheLayerSize() const { return kv_heads * qkv_dim * 2; }

  size_t model_dim = 0;
  size_t griffin_dim = 0;
  size_t ff_hidden_dim = 0;
  size_t heads = 0;
  size_t kv_heads = 0;
  size_t qkv_dim = 0;
  size_t conv1d_width = 0;
  bool ff_biases = false;
  bool softmax_attn_output_biases = false;
  PostNormType post_norm = PostNormType::None;
  LayerAttentionType type = LayerAttentionType::kGemma;
  ActivationType activation = ActivationType::Gelu;
  PostQKType post_qk = PostQKType::Rope;
};

struct ModelConfig {
  size_t CachePosSize() const {
    size_t num_layers = layer_configs.size();
    return num_layers * layer_configs[0].CacheLayerSize();
  }

  size_t NumLayersOfTypeBefore(LayerAttentionType type, size_t num) const {
    size_t count = 0;
    for (size_t i = 0; i < num; i++) {
      if (layer_configs[i].type == type) ++count;
    }
    return count;
  }

  size_t NumLayersOfType(LayerAttentionType type) const {
    return NumLayersOfTypeBefore(type, layer_configs.size());
  }

  size_t NumHeads() const {
    size_t num_heads = 0;
    for (const auto& layer_config : layer_configs) {
      num_heads = std::max(num_heads, layer_config.heads);
    }
    return num_heads;
  }

  std::string model_name;
  Model model;
  ModelTraining training;
  Type weight;
  size_t model_dim = 0;
  size_t vit_model_dim = 0;
  size_t vocab_size = 0;
  size_t seq_len = 0;
  size_t vit_seq_len = 0;
  size_t num_tensor_scales = 0;
  size_t num_vit_scales = 0;
  size_t top_k = kTopK;
  float att_cap = 0.0f;
  float final_cap = 0.0f;
  bool absolute_pe = false;
  bool use_local_attention = false;
  QueryScaleType query_scale = QueryScaleType::SqrtKeySize;
  std::vector<LayerConfig> layer_configs;
  std::vector<size_t> attention_window_sizes;
  std::vector<LayerConfig> vit_layer_configs;
  std::unordered_set<std::string> scale_names;
  int norm_num_groups = 1;
  int model_family_version = 1;
  // Dimensions related to image processing.
  int patch_width = 14;
  int image_size = 224;
};

// Returns the config for the given model.
ModelConfig ConfigFromModel(Model model);

// Returns the sub-config for the ViT model of the PaliGemma model.
ModelConfig VitConfig(const ModelConfig& config);
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

  static constexpr PostQKType kPostQK = PostQKType::Rope;
  static constexpr ActivationType kActivation = ActivationType::Gelu;
  static constexpr ResidualType kResidual = ResidualType::Add;

  // Self-extend parameters with defaul values
  static constexpr bool kSelfExtend = false;
  static constexpr size_t kSelfExtendNgbSize = 0;
  static constexpr size_t kSelfExtendGrpSize = 1;
};

struct ConfigBaseGemmaV1 : ConfigNoSSM {
  static constexpr float kAttCap = 0.0f;
  static constexpr float kFinalCap = 0.0f;
  static constexpr PostNormType kPostNorm = PostNormType::None;
  static constexpr QueryScaleType kQueryScale = QueryScaleType::SqrtKeySize;
};

struct ConfigBaseGemmaV2 : ConfigNoSSM {
  static constexpr float kAttCap = 50.0f;
  static constexpr float kFinalCap = 30.0f;
  static constexpr PostNormType kPostNorm = PostNormType::Scale;
};

template <typename TWeight>
struct ConfigGemma27B : public ConfigBaseGemmaV2 {
  using Weight = TWeight;  // make accessible where we only have a TConfig

  static constexpr int kSeqLen = 8192;
  static constexpr int kVocabSize = 256000;
  static constexpr std::array<LayerAttentionType, 46> kLayerConfig =
      FixedLayerConfig<46>(LayerAttentionType::kGemma);
  static constexpr std::array<size_t, 46> kAttentionWindowSizes =
      RepeatedAttentionWindowSizes<46, 2>({4096, kSeqLen});
  static constexpr int kLayers = kLayerConfig.size();
  static constexpr int kGemmaLayers = kLayers;
  static constexpr int kModelDim = 4608;
  static constexpr int kFFHiddenDim = 16 * 4608 / 2;  // = 36864
  static constexpr int kHeads = 32;
  static constexpr int kKVHeads = 16;
  static constexpr int kQKVDim = 128;  // query size == key size == value size
  static constexpr int kTopK = gcpp::kTopK;
  static constexpr bool kAbsolutePE = false;
  static constexpr QueryScaleType kQueryScale =
      QueryScaleType::SqrtModelDimDivNumHeads;
};

template <typename TWeight>
struct ConfigGemma9B : public ConfigBaseGemmaV2 {
  using Weight = TWeight;  // make accessible where we only have a TConfig

  static constexpr int kSeqLen = 8192;
  static constexpr int kVocabSize = 256000;
  static constexpr std::array<LayerAttentionType, 42> kLayerConfig =
      FixedLayerConfig<42>(LayerAttentionType::kGemma);
  static constexpr std::array<size_t, 42> kAttentionWindowSizes =
      RepeatedAttentionWindowSizes<42, 2>({4096, kSeqLen});
  static constexpr int kLayers = kLayerConfig.size();
  static constexpr int kGemmaLayers = kLayers;
  static constexpr int kModelDim = 3584;
  static constexpr int kFFHiddenDim = 8 * 3584 / 2;  // = 14336
  static constexpr int kHeads = 16;
  static constexpr int kKVHeads = 8;
  static constexpr int kQKVDim = 256;  // query size == key size == value size
  static constexpr int kTopK = gcpp::kTopK;
  static constexpr bool kAbsolutePE = false;
  static constexpr QueryScaleType kQueryScale = QueryScaleType::SqrtKeySize;
};

template <typename TWeight>
struct ConfigGemma7B : public ConfigBaseGemmaV1 {
  using Weight = TWeight;  // make accessible where we only have a TConfig

  static constexpr int kSeqLen = gcpp::kSeqLen;
  static constexpr int kVocabSize = 256000;
  static constexpr std::array<LayerAttentionType, 28> kLayerConfig =
      FixedLayerConfig<28>(LayerAttentionType::kGemma);
  static constexpr std::array<size_t, 28> kAttentionWindowSizes =
      FixedAttentionWindowSizes<28>(kSeqLen);
  static constexpr int kLayers = kLayerConfig.size();
  static constexpr int kGemmaLayers = kLayers;
  static constexpr int kModelDim = 3072;
  static constexpr int kFFHiddenDim = 16 * 3072 / 2;  // = 24576
  static constexpr int kHeads = 16;
  static constexpr int kKVHeads = 16;  // standard MHA
  static constexpr int kQKVDim = 256;  // query size == key size == value size
  static constexpr int kTopK = gcpp::kTopK;
  static constexpr bool kAbsolutePE = false;
};

template <typename TWeight>
struct ConfigGemma2B : public ConfigBaseGemmaV1 {
  using Weight = TWeight;  // make accessible where we only have a TConfig

  static constexpr int kSeqLen = gcpp::kSeqLen;
  static constexpr int kVocabSize = 256000;
  static constexpr std::array<LayerAttentionType, 18> kLayerConfig =
      FixedLayerConfig<18>(LayerAttentionType::kGemma);
  static constexpr std::array<size_t, 18> kAttentionWindowSizes =
      FixedAttentionWindowSizes<18>(kSeqLen);
  static constexpr int kLayers = kLayerConfig.size();
  static constexpr int kGemmaLayers = kLayers;
  static constexpr int kModelDim = 2048;
  static constexpr int kFFHiddenDim = 16 * 2048 / 2;  // = 16384
  static constexpr int kHeads = 8;
  static constexpr int kKVHeads = 1;
  static constexpr int kQKVDim = 256;  // query size == key size == value size
  static constexpr int kTopK = gcpp::kTopK;
  static constexpr bool kAbsolutePE = false;
};

template <typename TWeight>
struct ConfigGemma2_2B : public ConfigBaseGemmaV2 {
  using Weight = TWeight;  // make accessible where we only have a TConfig

  static constexpr int kSeqLen = 8192;
  static constexpr int kVocabSize = 256000;
  static constexpr std::array<LayerAttentionType, 26> kLayerConfig =
      FixedLayerConfig<26>(LayerAttentionType::kGemma);
  static constexpr std::array<size_t, 26> kAttentionWindowSizes =
      RepeatedAttentionWindowSizes<26, 2>({4096, kSeqLen});
  static constexpr int kLayers = kLayerConfig.size();
  static constexpr int kGemmaLayers = kLayers;
  static constexpr int kModelDim = 2304;
  static constexpr int kFFHiddenDim = 8 * 2304 / 2;  // = 9216
  static constexpr int kHeads = 8;
  static constexpr int kKVHeads = 4;
  static constexpr int kQKVDim = 256;  // query size == key size == value size
  static constexpr int kTopK = gcpp::kTopK;
  static constexpr bool kAbsolutePE = false;
  static constexpr QueryScaleType kQueryScale = QueryScaleType::SqrtKeySize;
};

template <typename TWeight>
struct ConfigGemmaTiny : public ConfigNoSSM {
  using Weight = TWeight;  // make accessible where we only have a TConfig

  static constexpr int kSeqLen = 32;
  static constexpr int kVocabSize = 64;
  static constexpr std::array<LayerAttentionType, 3> kLayerConfig =
      FixedLayerConfig<3>(LayerAttentionType::kGemma);
  static constexpr std::array<size_t, 3> kAttentionWindowSizes =
      FixedAttentionWindowSizes<3>(kSeqLen);
  static constexpr int kLayers = kLayerConfig.size();
  static constexpr int kGemmaLayers = kLayers;
  static constexpr int kModelDim = 128;
  static constexpr int kFFHiddenDim = 256;
  static constexpr int kHeads = 4;
  static constexpr int kKVHeads = 1;
  static constexpr int kQKVDim = 16;  // query size == key size == value size
  static constexpr int kTopK = gcpp::kTopK;
  static constexpr bool kAbsolutePE = false;
  static constexpr PostNormType kPostNorm = PostNormType::None;
  static constexpr QueryScaleType kQueryScale = QueryScaleType::SqrtKeySize;

  static constexpr float kAttCap = 0.0f;
  // This is required for optimize_test to pass.
  static constexpr float kFinalCap = 30.0f;
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
  static constexpr std::array<size_t, 26> kAttentionWindowSizes =
      FixedAttentionWindowSizes<26>(kSeqLen);
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
  static constexpr PostNormType kPostNorm = PostNormType::None;

  // No SoftCap.
  static constexpr float kAttCap = 0.0f;
  static constexpr float kFinalCap = 0.0f;

  // SSM config.
  static constexpr int kConv1dWidth = 4;
  static constexpr bool kFFBiases = true;
  static constexpr bool kSoftmaxAttnOutputBiases = true;
  static constexpr bool kUseHalfRope = true;
  static constexpr bool kUseLocalAttention = true;
  static constexpr bool kInterleaveQKV = false;
  static constexpr int kNumTensorScales = 140;
  static constexpr PostQKType kPostQK = PostQKType::Rope;
  static constexpr ActivationType kActivation = ActivationType::Gelu;
  static constexpr QueryScaleType kQueryScale = QueryScaleType::SqrtKeySize;
  static constexpr ResidualType kResidual = ResidualType::Add;

  // Self-extend parameters with defaul values
  static constexpr bool kSelfExtend = false;
  static constexpr size_t kSelfExtendNgbSize = 0;
  static constexpr size_t kSelfExtendGrpSize = 1;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_CONFIGS_H_
