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
#include <stdint.h>

#include <array>
#include <string>
#include <unordered_set>
#include <vector>

#include "compression/fields.h"  // IFieldsVisitor
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
static constexpr size_t kMaxConv1DWidth = 4;

using EmbedderInputT = BF16;

// Instruction-tuned models require extra 'turn structure' tokens in prompts.
enum class PromptWrapping {
  GEMMA_IT,
  GEMMA_PT,
  GEMMA_VLM,  // for >1B Gemma3
  PALIGEMMA,
  kSentinel  // must be last
};

// Defined as the suffix for use with `ModelString`.
static inline const char* ToString(PromptWrapping wrapping) {
  switch (wrapping) {
    case PromptWrapping::GEMMA_IT:
      return "-it";
    case PromptWrapping::GEMMA_PT:
      return "-pt";
    case PromptWrapping::GEMMA_VLM:
      return "-vlm";
    case PromptWrapping::PALIGEMMA:
      return "-pg";
    default:
      return "-?";
  }
}

static inline bool EnumValid(PromptWrapping wrapping) {
  return static_cast<size_t>(wrapping) <
         static_cast<size_t>(PromptWrapping::kSentinel);
}

enum class LayerAttentionType {
  kGemma,
  kGriffinRecurrentBlock,
  kVit,
};

static inline bool EnumValid(LayerAttentionType type) {
  return type == LayerAttentionType::kGemma ||
         type == LayerAttentionType::kGriffinRecurrentBlock ||
         type == LayerAttentionType::kVit;
}

// Post attention and ffw normalization type.
enum class PostNormType {
  None,
  Scale,
  kSentinel  // must be last
};

static inline bool EnumValid(PostNormType type) {
  return static_cast<size_t>(type) <
         static_cast<size_t>(PostNormType::kSentinel);
}

// Post qk projection operation type.
enum class PostQKType {
  Rope,
  HalfRope,
  kSentinel  // must be last
};

static inline bool EnumValid(PostQKType type) {
  return static_cast<size_t>(type) <
         static_cast<size_t>(PostNormType::kSentinel);
}

// FFW activation function.
enum class ActivationType {
  Gelu,
  kSentinel  // must be last
};

static inline bool EnumValid(ActivationType type) {
  return static_cast<size_t>(type) <
         static_cast<size_t>(ActivationType::kSentinel);
}

// Attention query scale.
enum class QueryScaleType {
  SqrtKeySize,
  SqrtModelDimDivNumHeads,
  kSentinel  // must be last
};

static inline bool EnumValid(QueryScaleType type) {
  return static_cast<size_t>(type) <
         static_cast<size_t>(QueryScaleType::kSentinel);
}

// Residual connection type.
enum class ResidualType {
  Add,
  kSentinel  // must be last
};

static inline bool EnumValid(ResidualType type) {
  return static_cast<size_t>(type) <
         static_cast<size_t>(ResidualType::kSentinel);
}

template <size_t kNum>
std::vector<LayerAttentionType> FixedLayerConfig(LayerAttentionType type) {
  return std::vector<LayerAttentionType>(kNum, type);
}

template <uint32_t kNum>
std::vector<uint32_t> FixedAttentionWindowSizes(uint32_t window_size) {
  return std::vector<uint32_t>(kNum, window_size);
}

// Repeat window_size_pattern for kNum / kPatternSize times.
template <uint32_t kNum, uint32_t kPatternSize>
std::vector<uint32_t> RepeatedAttentionWindowSizes(
    const std::array<uint32_t, kPatternSize>& window_size_pattern) {
  std::vector<uint32_t> window_size_configs(kNum);
  for (uint32_t i = 0; i < kNum; ++i) {
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
  PALIGEMMA_448,
  PALIGEMMA2_3B_224,
  PALIGEMMA2_3B_448,
  PALIGEMMA2_10B_224,
  PALIGEMMA2_10B_448,
  GEMMA3_4B,
  GEMMA3_1B,
  GEMMA3_12B,
  GEMMA3_27B,
  kSentinel,
};

// Allows the Model enum to be iterated over.
static constexpr Model kAllModels[] = {
    Model::GEMMA_2B, Model::GEMMA_7B, Model::GEMMA2_9B, Model::GEMMA2_27B,
    Model::GRIFFIN_2B, Model::GEMMA_TINY, Model::GEMMA2_2B,
    Model::PALIGEMMA_224, Model::PALIGEMMA_448, Model::PALIGEMMA2_3B_224,
    Model::PALIGEMMA2_3B_448, Model::PALIGEMMA2_10B_224,
    Model::PALIGEMMA2_10B_448, Model::GEMMA3_4B, Model::GEMMA3_1B,
    Model::GEMMA3_12B, Model::GEMMA3_27B,
};

template <class Func>
void ForEachModel(const Func& func) {
  for (size_t i = static_cast<size_t>(Model::UNKNOWN) + 1;
       i < static_cast<size_t>(Model::kSentinel); ++i) {
    func(static_cast<Model>(i));
  }
}

static inline bool EnumValid(Model model) {
  const size_t i = static_cast<size_t>(model);
  if (i < static_cast<size_t>(Model::kSentinel)) {
    return true;
  }
  return false;
}

struct LayerConfig : public IFields {
  // Returns true if *this and other are equal.
  // If partial is true, then we don't check for items that are only set after
  // the tensors are loaded from the checkpoint.
  // If debug is true, then we output the mismatched fields to stderr.
  bool TestEqual(const LayerConfig& other, bool partial, bool debug) const;

  size_t CacheLayerSize() const { return kv_heads * qkv_dim * 2; }

  // Multi-Head Attention?
  bool IsMHA() const { return heads == kv_heads; }

  // Stride between subsequent queries. Each of Q, K, V are of length kQKVDim,
  // but for MHA we store them as Q,K,V, Q,K,V, .. instead of Q..Q, K..K, V..V.
  size_t QStride() const { return qkv_dim * (IsMHA() ? 3 : 1); }

  const char* Name() const override { return "LayerConfig"; }

  void VisitFields(IFieldsVisitor& visitor) override {
    visitor(model_dim);
    visitor(griffin_dim);
    visitor(ff_hidden_dim);
    visitor(heads);
    visitor(kv_heads);
    visitor(qkv_dim);
    visitor(conv1d_width);
    visitor(ff_biases);
    visitor(softmax_attn_output_biases);
    visitor(optimized_gating);
    visitor(post_norm);
    visitor(type);
    visitor(activation);
    visitor(post_qk);
    visitor(use_qk_norm);
  }

  uint32_t model_dim = 0;
  uint32_t griffin_dim = 0;
  uint32_t ff_hidden_dim = 0;
  uint32_t heads = 0;
  uint32_t kv_heads = 0;
  uint32_t qkv_dim = 0;
  uint32_t conv1d_width = 0;  // griffin only
  bool ff_biases = false;
  bool softmax_attn_output_biases = false;
  bool optimized_gating = true;
  PostNormType post_norm = PostNormType::None;
  LayerAttentionType type = LayerAttentionType::kGemma;
  ActivationType activation = ActivationType::Gelu;
  PostQKType post_qk = PostQKType::Rope;
  bool use_qk_norm = false;
};

// Dimensions related to image processing.
struct VitConfig : public IFields {
  // Returns true if *this and other are equal.
  // If partial is true, then we don't check for items that are only set after
  // the tensors are loaded from the checkpoint.
  // If debug is true, then we output the mismatched fields to stderr.
  bool TestEqual(const VitConfig& other, bool partial, bool debug) const;

  const char* Name() const override { return "VitConfig"; }

  void VisitFields(IFieldsVisitor& visitor) override {
    visitor(model_dim);
    visitor(seq_len);
    visitor(num_scales);
    visitor(patch_width);
    visitor(image_size);
    visitor(layer_configs);
    visitor(pool_dim);
  }

  uint32_t model_dim = 0;
  uint32_t seq_len = 0;
  uint32_t num_scales = 0;
  uint32_t patch_width = 14;
  uint32_t image_size = 224;
  uint32_t pool_dim = 1;
  std::vector<LayerConfig> layer_configs;
};

struct ModelConfig : public IFields {
  // Returns true if *this and other are equal.
  // If partial is true, then we don't check for items that are only set after
  // the tensors are loaded from the checkpoint.
  // If debug is true, then we output the mismatched fields to stderr.
  bool TestEqual(const ModelConfig& other, bool partial, bool debug) const;

  void AddLayerConfig(const LayerConfig& layer_config) {
    layer_configs.push_back(layer_config);
  }

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
    uint32_t num_heads = 0;
    for (const auto& layer_config : layer_configs) {
      num_heads = HWY_MAX(num_heads, layer_config.heads);
    }
    return num_heads;
  }

  const char* Name() const override { return "ModelConfig"; }

  bool IsEOS(int id) const { return (id == eos_id || id == secondary_eos_id); }

  void VisitFields(IFieldsVisitor& visitor) override {
    visitor(model_family_version);
    visitor(model_name);
    visitor(model);
    visitor(wrapping);
    visitor(weight);
    visitor(num_layers);
    visitor(model_dim);
    visitor(vocab_size);
    visitor(seq_len);
    visitor(num_tensor_scales);
    visitor(att_cap);
    visitor(final_cap);
    visitor(absolute_pe);
    visitor(use_local_attention);
    visitor(query_scale);
    visitor(layer_configs);
    visitor(attention_window_sizes);
    visitor(norm_num_groups);
    visitor(vit_config);
    visitor(pool_dim);
    visitor(eos_id);
    visitor(secondary_eos_id);
  }

  // Major version of the model family. It is used as a fallback to distinguish
  // between model types when there is no explicit information in the config.
  uint32_t model_family_version = 1;
  std::string model_name;
  Model model = Model::UNKNOWN;
  PromptWrapping wrapping = PromptWrapping::GEMMA_PT;
  Type weight = Type::kUnknown;
  uint32_t num_layers = 0;
  uint32_t model_dim = 0;
  uint32_t vocab_size = 0;
  uint32_t seq_len = 0;
  uint32_t num_tensor_scales = 0;
  float att_cap = 0.0f;
  float final_cap = 0.0f;
  bool absolute_pe = false;
  bool use_local_attention = false;  // griffin only
  QueryScaleType query_scale = QueryScaleType::SqrtKeySize;
  std::vector<LayerConfig> layer_configs;
  std::vector<uint32_t> attention_window_sizes;
  std::unordered_set<std::string> scale_names;
  uint32_t norm_num_groups = 1;
  // Dimensions related to image processing.
  VitConfig vit_config;
  uint32_t pool_dim = 1;  // used only for VitConfig copy
  int eos_id = 1;
  int secondary_eos_id = 1;
};

// Returns the config for the given model.
ModelConfig ConfigFromModel(Model model);

// Returns the model for the given config, if it matches any standard model.
Model ModelFromConfig(const ModelConfig& config);

// Returns the sub-config for the ViT model of the PaliGemma model.
ModelConfig GetVitConfig(const ModelConfig& config);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_CONFIGS_H_
