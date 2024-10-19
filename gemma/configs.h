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
  bool self_extend = false;
  size_t ngb_size = 0;
  size_t grp_size = 1;

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

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_CONFIGS_H_
