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
#include <vector>

#include "compression/types.h"  // Type
#include "io/fields.h"           // IFieldsVisitor
#include "io/io.h"               // Path
#include "util/basics.h"

namespace gcpp {

static constexpr size_t kMaxConv1DWidth = 4;
static constexpr size_t kMaxQKVDim = 1024;

// Instruction-tuned models require extra 'turn structure' tokens in prompts.
enum class PromptWrapping {
  GEMMA_IT,
  GEMMA_PT,
  GEMMA_VLM,  // for >1B Gemma3
  PALIGEMMA,
  kSentinel  // must be last
};

// This is used in `ModelConfig.Specifier`, so the strings will not change,
// though new ones may be added.
static inline const char* WrappingSuffix(PromptWrapping wrapping) {
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
  UNKNOWN = 0,
  // 1 and 2 are obsolete.
  GEMMA2_9B = 3,
  GEMMA2_27B,
  GRIFFIN_2B,
  GEMMA_TINY,  // for testing only
  GEMMA2_2B,
  // 8 and 9 are obsolete.
  PALIGEMMA2_3B_224 = 10,
  PALIGEMMA2_3B_448,
  PALIGEMMA2_10B_224,
  PALIGEMMA2_10B_448,
  GEMMA3_4B,
  GEMMA3_1B,
  GEMMA3_12B,
  GEMMA3_27B,
  GEMMA3_270M,
  kSentinel,
};

// Returns canonical model name without the PromptWrapping suffix. This is used
// in Specifier and thus does not change.
const char* ModelPrefix(Model model);

// Gemma3 is multimodal and has a different prompt wrapping than PaliGemma.
// This is used for deducing the PromptWrapping for pre-2025 BlobStore.
static inline bool IsVLM(Model model) {
  return model == Model::GEMMA3_4B || model == Model::GEMMA3_1B ||
         model == Model::GEMMA3_12B || model == Model::GEMMA3_27B;
}

static inline bool IsPaliGemma(Model model) {
  if (model == Model::PALIGEMMA2_3B_224 || model == Model::PALIGEMMA2_3B_448 ||
      model == Model::PALIGEMMA2_10B_224 ||
      model == Model::PALIGEMMA2_10B_448) {
    return true;
  }
  return false;
}

// Visits every valid model enum, skipping `UNKNOWN` and `kSentinel`.
template <class Func>
void ForEachModel(const Func& func) {
  for (size_t i = static_cast<size_t>(Model::GEMMA2_9B);
       i < static_cast<size_t>(Model::kSentinel); ++i) {
    if (i == 8 || i == 9) continue;
    func(static_cast<Model>(i));
  }
}

static inline bool EnumValid(Model model) {
  // Valid for purposes of serialization, even if unknown.
  if (model == Model::UNKNOWN) return true;
  const size_t i = static_cast<size_t>(model);
  if (i >= static_cast<size_t>(Model::GEMMA2_9B) &&
      i < static_cast<size_t>(Model::kSentinel) && i != 8 && i != 9) {
    return true;
  }
  return false;
}

struct InternalLayerConfig : public IFields {
  const char* Name() const override { return "InternalLayerConfig"; }

  // Source of truth for field ordering.
  void VisitFields(IFieldsVisitor& visitor) override {
    // Append new fields here, then update `python/configs.cc`.
  }
};

// Per-layer configuration.
struct LayerConfig : public IFields {
  const char* Name() const override { return "LayerConfig"; }

  // Source of truth for field ordering.
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
    internal.VisitFields(visitor);
    // Append new fields here, then update `python/configs.cc`.
  }

  // Returns whether all fields match.
  bool TestEqual(const LayerConfig& other, bool print) const;

  size_t CacheLayerSize() const { return kv_heads * qkv_dim * 2; }

  // Multi-Head Attention?
  bool IsMHA() const { return heads == kv_heads; }

  uint32_t model_dim = 0;
  uint32_t griffin_dim = 0;
  uint32_t ff_hidden_dim = 0;
  uint32_t heads = 0;
  uint32_t kv_heads = 0;
  uint32_t qkv_dim = 0;       // length of Q, K, V vectors (contiguous).
  uint32_t conv1d_width = 0;  // Griffin only
  bool ff_biases = false;
  bool softmax_attn_output_biases = false;  // for Griffin
  bool optimized_gating = true;             // for Gemma3
  PostNormType post_norm = PostNormType::None;
  LayerAttentionType type = LayerAttentionType::kGemma;
  ActivationType activation = ActivationType::Gelu;
  PostQKType post_qk = PostQKType::Rope;
  bool use_qk_norm = false;
  InternalLayerConfig internal;
};

// Dimensions related to image processing.
struct VitConfig : public IFields {
  const char* Name() const override { return "VitConfig"; }

  // Source of truth for field ordering.
  void VisitFields(IFieldsVisitor& visitor) override {
    visitor(model_dim);
    visitor(seq_len);
    visitor(num_scales);
    visitor(patch_width);
    visitor(image_size);
    visitor(layer_configs);
    visitor(pool_dim);
    // Append new fields here, then update `python/configs.cc`.
  }

  // Returns whether all fields match.
  bool TestEqual(const VitConfig& other, bool print) const;

  uint32_t model_dim = 0;
  uint32_t seq_len = 0;
  uint32_t num_scales = 0;
  uint32_t patch_width = 14;
  uint32_t image_size = 224;
  uint32_t pool_dim = 1;
  std::vector<LayerConfig> layer_configs;
};

// Returns a valid `PromptWrapping` for the given `model`, for passing to the
// `ModelConfig` ctor when the caller does not care about the wrapping. The
// wrapping mode is either determined by the model (for PaliGemma and Gemma3),
// or defaults to IT, subject to user override for PT.
PromptWrapping ChooseWrapping(Model model,
                              Tristate wrapping = Tristate::kDefault);

struct InternalModelConfig : public IFields {
  const char* Name() const override { return "InternalModelConfig"; }

  // Source of truth for field ordering.
  void VisitFields(IFieldsVisitor& visitor) override {
    // Append new fields here, then update `python/configs.cc`.
  }
};

struct ModelConfig : public IFields {
  // Preferred usage (single-file format): default-construct, then deserialize
  // from a blob. Also used by `config_converter.py`, which sets sufficient
  // fields for `TestEqual` and then calls `OverwriteWithCanonical()`.
  ModelConfig() = default;
  // For use by `model_store.cc` for pre-2025 format after deducing the model
  // from tensors plus a user-specified `wrapping` override (`ChooseWrapping`).
  ModelConfig(Model model, Type weight, PromptWrapping wrapping);
  // Parses a string returned by `Specifier()`. Used by the exporter to select
  // the model from command line arguments. Do not use this elsewhere - the
  // second ctor is preferred because it is type-checked.
  ModelConfig(const std::string& specifier);

  const char* Name() const override { return "ModelConfig"; }

  // Source of truth for field ordering.
  void VisitFields(IFieldsVisitor& visitor) override {
    visitor(model_family_version);
    visitor(display_name);
    visitor(model);
    visitor(wrapping);
    visitor(weight);

    visitor(num_layers);
    visitor(model_dim);
    visitor(vocab_size);
    visitor(max_seq_len);

    visitor(unused_num_tensor_scales);

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

    visitor(scale_base_names);

    internal.VisitFields(visitor);

    // Append new fields here, then update `python/configs.cc`.
  }

  // Returns whether all fields match except `model` and `display_name`, and
  // some others that are not yet set by config_converter.py. This is for
  // internal use by `OverwriteWithCanonical`, but potentially useful elsewhere.
  bool TestEqual(const ModelConfig& other, bool print) const;

  // For each model, constructs its canonical `ModelConfig` and if `TestEqual`
  // returns true, overwrites `*this` with that. Otherwise, returns false to
  // indicate this is not a known model. Called by `config_converter.py`.
  bool OverwriteWithCanonical();

  // Returns a string encoding of the model family, size, weight, and
  // `PromptWrapping`. Stable/unchanging; can be used as the model file name.
  // The third ctor also expects a string returned by this.
  std::string Specifier() const;

  void AddLayerConfig(const LayerConfig& layer_config) {
    layer_configs.push_back(layer_config);
    HWY_ASSERT(layer_configs.size() <= num_layers);
  }

  bool IsGlobalLayer(size_t layer_idx) const {
    return attention_window_sizes[layer_idx] == max_seq_len;
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

  size_t KVCacheCols() const {
    size_t num_layers = layer_configs.size();
    return num_layers * layer_configs[0].CacheLayerSize();
  }

  bool IsEOS(int id) const { return (id == eos_id || id == secondary_eos_id); }

  // Major version of the model family, reflecting architecture changes. This is
  // more convenient to compare than `Model` because that also includes the
  // model size.
  uint32_t model_family_version = 1;
  // For display only, may change. Use `Specifier()` for setting the
  // file name. Not checked by `TestEqual` because `config_converter.py` does
  // not set this.
  std::string display_name;
  Model model = Model::UNKNOWN;  // Not checked by `TestEqual`, see above.
  PromptWrapping wrapping = PromptWrapping::GEMMA_PT;
  Type weight = Type::kUnknown;

  uint32_t num_layers = 0;
  uint32_t model_dim = 0;
  uint32_t vocab_size = 0;
  uint32_t max_seq_len = 0;

  // We no longer set nor use this: config_converter is not able to set this,
  // and only pre-2025 format stores scales, and we do not require advance
  // knowledge of how many there will be. Any scales present will just be
  // assigned in order to the tensors matching `scale_base_names`.
  uint32_t unused_num_tensor_scales = 0;

  float att_cap = 0.0f;
  float final_cap = 0.0f;

  bool absolute_pe = false;
  bool use_local_attention = false;  // Griffin only
  QueryScaleType query_scale = QueryScaleType::SqrtKeySize;
  std::vector<LayerConfig> layer_configs;
  std::vector<uint32_t> attention_window_sizes;
  uint32_t norm_num_groups = 1;

  // Dimensions related to image processing.
  VitConfig vit_config;
  uint32_t pool_dim = 1;  // used only for VitConfig copy

  int eos_id = 1;
  int secondary_eos_id = 1;

  // Tensor base names without a layer suffix, used by `ModelStore` only for
  // pre-2025 format.
  std::vector<std::string> scale_base_names;

  InternalModelConfig internal;
};

// Returns the sub-config for the ViT model of the PaliGemma model.
ModelConfig GetVitConfig(const ModelConfig& config);

enum DeducedLayerTypes {
  kDeducedGriffin = 1,
  kDeducedViT = 2,
  kDeduced448 = 4,   // For ViT, 448x448 resolution instead of 224x224.
};

// layer_types is one or more of `DeducedLayerTypes`.
Model DeduceModel(const Path& blob_path, size_t layers, int layer_types);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_CONFIGS_H_
