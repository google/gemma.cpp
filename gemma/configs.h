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
#include "io/fields.h"          // IFieldsVisitor
#include "io/io.h"              // Path
#include "util/basics.h"
#include "hwy/detect_compiler_arch.h"

namespace gcpp {

constexpr size_t kMaxBF16PerVector = HWY_ARCH_MAX_BYTES / sizeof(BF16);

HWY_INLINE_VAR constexpr int kSkipKV = 1;

HWY_INLINE_VAR constexpr size_t kMaxQKVDim = 1024;

#ifndef GEMMA_FUSED_FFN
#define GEMMA_FUSED_FFN 1
#endif  // !GEMMA_FUSED_FFN

// Instruction-tuned models require extra 'turn structure' tokens in prompts.
enum class PromptWrapping {
  GEMMA_IT,
  GEMMA_PT,
  GEMMA_VLM,  // for >1B Gemma3
  PALIGEMMA,
  kSentinel  // must be last
};

// Which tokenizer engine the `.sbs` tokenizer blob targets.
enum class TokenizerKind {
  kSentencePiece = 0,  // blob is a SentencePiece proto
  kHfBpe = 1,  // blob is a compact BPE format built from HF `tokenizer.json`
  kSentinel    // must be last
};

static inline bool EnumValid(TokenizerKind kind) {
  return static_cast<size_t>(kind) <
         static_cast<size_t>(TokenizerKind::kSentinel);
}

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
  kVit,
  // DeepSeek-style Multi-head Latent Attention (V3/V4 family): a per-token
  // compressed KV latent plus a decoupled RoPE key, optionally with
  // sequence-axis compression (CSA/HCA) selected via `AttentionVariant`.
  kDeepSeekMLA,
};

static inline bool EnumValid(LayerAttentionType type) {
  return type == LayerAttentionType::kGemma ||
         type == LayerAttentionType::kVit ||
         type == LayerAttentionType::kDeepSeekMLA;
}

// Sequence-axis KV compression variant for `kDeepSeekMLA` layers.
enum class AttentionVariant {
  kDense = 0,  // attend to every cached latent (no pooling, no indexer)
  kCSA,        // Compressed Sparse Attention: pooled KV + top-k indexer
  kHCA,        // Heavily Compressed Attention: heavy pooling, dense attention
  kSentinel    // must be last
};

static inline bool EnumValid(AttentionVariant variant) {
  return static_cast<size_t>(variant) <
         static_cast<size_t>(AttentionVariant::kSentinel);
}

// Values stated explicitly to allow for semantic reordering
enum class KVEncoding {
  kUnspecified = 0,
  kF32 = 1,
  kBF16 = 2,
  kF32TwoTranspositions = 3,
  kBF16TwoTranspositions = 4,
  kInt8 = 5,
  kInt8TwoTranspositions = 6,
  kBF16MatrixAccumulation = 7,
  kInt8MatrixAccumulation = 8,
};

// Returns a string representation of the KVEncoding.
// This representation will not change and can be used for
// serialization or logging.
// Note that no reverse function exists to convert a string back to a
// KVEncoding.
std::string KVEncodingToString(KVEncoding encoding);

enum class AttentionImpl {
  kFlash = 0,  // Flash Attention (default)
  kFlashTransposedQs,
  kFlashTransposedQsBF16,
  kFlashTransposedQsInt16,
  kFlashMatrixAccumulation,
  kInt8MatrixAccumulation,
  kSentinel,
};

std::string GetAttentionImplName(AttentionImpl impl);
AttentionImpl GetAttentionImpl(const std::string& impl);

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
  NormLocalRope = 8,  // Norm without scale, and rope for local attention layers
  kSentinel           // must be last
};

static inline bool EnumValid(PostQKType type) {
  return static_cast<size_t>(type) <
         static_cast<size_t>(PostNormType::kSentinel);
}

// FFW activation function.
enum class ActivationType {
  Gelu,
  Silu,         // used by DeepSeek models
};

// MoE router scoring function. `kSigmoidGatingCompat` (the default) defers to
// the legacy `LayerConfig::sigmoid_gating` bool so that older serialized
// configs keep their behavior.
enum class RouterScoreFunc {
  kSigmoidGatingCompat = 0,
  kSoftmax,
  kSigmoid,
  kSqrtSoftplus,  // sqrt(softplus(x)), used by DeepSeek V4
  kSentinel       // must be last
};

static inline bool EnumValid(RouterScoreFunc func) {
  return static_cast<size_t>(func) <
         static_cast<size_t>(RouterScoreFunc::kSentinel);
}

static inline bool EnumValid(ActivationType type) {
  return (type == ActivationType::Gelu || type == ActivationType::Silu);
}

// Attention query scale.
enum class QueryScaleType {
  SqrtKeySize,
  SqrtModelDimDivNumHeads,
  One,
};

static inline bool EnumValid(QueryScaleType type) {
  return type == QueryScaleType::SqrtKeySize ||
         type == QueryScaleType::SqrtModelDimDivNumHeads ||
         type == QueryScaleType::One;
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
  // 5 and 6 are obsolete.
  GEMMA2_2B = 7,
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
  CUSTOM,
  // Text-only variants of Gemma 3, distinguished by absence of a vision tower
  // (e.g. TranslateGemma). Added after CUSTOM to preserve serialized enum
  // values for existing weight files.
  GEMMA3_4B_LM,
  GEMMA3_12B_LM,
  GEMMA3_27B_LM,
  GEMMA4_26B_MOE,
  GEMMA4_2B,
  DEEPSEEK4_FLASH,
  // T5Gemma family - starting with S/S.
  T5GEMMA_S_S,
  kSentinel,
};

// Returns canonical model name without the PromptWrapping suffix. This is used
// in Specifier and thus does not change.
const char* ModelPrefix(Model model);

static inline bool IsPaliGemma(Model model) {
  if (model == Model::PALIGEMMA2_3B_224 || model == Model::PALIGEMMA2_3B_448 ||
      model == Model::PALIGEMMA2_10B_224 ||
      model == Model::PALIGEMMA2_10B_448) {
    return true;
  }
  return false;
}

static inline bool IsObsolete(Model model) {
  const size_t i = static_cast<size_t>(model);
  if (i == 5 || i == 6 || i == 8 || i == 9) return true;
  return false;
}

// Visits every valid model enum, skipping `UNKNOWN`, `kSentinel` and `CUSTOM`.
template <class Func>
void ForEachModel(const Func& func) {
  for (size_t i = static_cast<size_t>(Model::GEMMA2_9B);
       i < static_cast<size_t>(Model::kSentinel); ++i) {
    const Model model = static_cast<Model>(i);
    if (!IsObsolete(model) && model != Model::CUSTOM) func(model);
  }
}

static inline bool EnumValid(Model model) {
  // Valid for purposes of serialization, even if unknown.
  if (model == Model::UNKNOWN) return true;
  const size_t i = static_cast<size_t>(model);
  if (i >= static_cast<size_t>(Model::GEMMA2_9B) &&
      i < static_cast<size_t>(Model::kSentinel) && !IsObsolete(model)) {
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
    // Formerly used for Griffin.
    uint32_t unused_griffin_dim = 0;
    uint32_t unused_conv1d_width = 0;
    bool unused_softmax_attn_output_biases = false;

    visitor(model_dim);
    visitor(unused_griffin_dim);
    visitor(ff_hidden_dim);
    visitor(heads);
    visitor(kv_heads);
    visitor(qkv_dim);
    visitor(unused_conv1d_width);
    visitor(ff_biases);
    visitor(unused_softmax_attn_output_biases);
    visitor(optimized_gating);
    visitor(post_norm);
    visitor(type);
    visitor(activation);
    visitor(post_qk);
    visitor(use_qk_norm);
    internal.VisitFields(visitor);
    visitor(norm_v);
    visitor(num_experts);
    visitor(num_experts_per_datapoint);
    visitor(ple_dim);
    visitor(kv_share_layer_idx);
    visitor(kv_lora_rank);
    visitor(q_lora_rank);
    visitor(rope_head_dim);
    visitor(v_head_dim);
    visitor(attention_variant);
    visitor(kv_compression_rate);
    visitor(indexer_heads);
    visitor(indexer_head_dim);
    visitor(indexer_top_k);
    visitor(num_shared_experts);
    visitor(sigmoid_gating);
    visitor(use_routing_bias);
    visitor(o_lora_rank);
    visitor(o_groups);
    visitor(swiglu_limit);
    visitor(route_scale);
    visitor(router_score);
    visitor(hash_routing);
    // Append new fields here, then update `python/configs.cc`.
  }

  uint32_t NumExperts() const {
    return num_experts;
  }
  uint32_t NumExpertsPerDatapoint() const {
    return num_experts_per_datapoint;
  }

  bool IsMoE() const { return NumExperts() > 0; }

  // DeepSeek-style Multi-head Latent Attention?
  bool IsMLA() const { return kv_lora_rank > 0; }

  // DeepSeek V4-style MLA: single wide shared K=V latent, grouped low-rank
  // output projection (wo_a/wo_b), learned gated compressor, attention sink.
  bool IsV4MLA() const { return IsMLA() && o_lora_rank > 0; }

  // Query head dim without the decoupled RoPE part (MLA only).
  size_t NopeHeadDim() const {
    HWY_DASSERT(qkv_dim >= rope_head_dim);
    return qkv_dim - rope_head_dim;
  }

  // Width of one cached latent row (c_kv + decoupled RoPE key).
  size_t KVLatentDim() const { return kv_lora_rank + rope_head_dim; }

  // V4 compressor: ratio-4 layers use overlapping windows (doubled dims).
  size_t CompressorCoff() const { return kv_compression_rate == 4 ? 2 : 1; }

  bool HasCompressor() const { return IsV4MLA() && kv_compression_rate > 1; }
  bool HasIndexer() const { return indexer_heads > 0 && indexer_top_k > 0; }

  // Plain sliding-window attention: no sequence compression, no indexer
  // (DeepSeek V4 layers 0-1 and the MTP block). Keep in sync with
  // HasCompressor/HasIndexer and CacheLayerSize.
  void SetDenseAttention() {
    attention_variant = AttentionVariant::kDense;
    kv_compression_rate = 1;
    indexer_heads = 0;
    indexer_head_dim = 0;
    indexer_top_k = 0;
  }

  // Returns whether all fields match.
  bool TestEqual(const LayerConfig& other, bool print) const;

  size_t CacheLayerSize() const {
    if (IsMLA()) {
      // MLA caches a single latent (c_kv + RoPE key) per token, shared across
      // all heads.
      size_t size = KVLatentDim();
      if (IsV4MLA()) {
        // V4 additionally caches, at the last row of each sealed block, the
        // compressor output (same width as the latent) and, on CSA layers,
        // the indexer's own compressed entry.
        if (HasCompressor()) size += KVLatentDim();
        if (HasIndexer()) size += indexer_head_dim;
      }
      return size;
    }
    return kv_heads * qkv_dim * 2;
  }

  // f32 elements of per-query incremental compressor state for this layer:
  // (kv_state + score_state) for the attention compressor and, on CSA layers,
  // for the indexer's compressor. See `Compressor` in the DeepSeek V4
  // reference implementation.
  size_t DSStateSize() const {
    if (!HasCompressor()) return 0;
    const size_t coff = CompressorCoff();
    size_t size = 2 * (coff * kv_compression_rate) * (coff * KVLatentDim());
    if (HasIndexer()) {
      size += 2 * (coff * kv_compression_rate) * (coff * indexer_head_dim);
    }
    return size;
  }

  // Multi-Head Attention?
  bool IsMHA() const { return heads == kv_heads; }

  uint32_t model_dim = 0;
  uint32_t ff_hidden_dim = 0;
  uint32_t heads = 0;
  uint32_t kv_heads = 0;
  uint32_t qkv_dim = 0;  // length of Q, K, V vectors (contiguous).
  bool ff_biases = false;
  bool optimized_gating = true;  // for Gemma3
  PostNormType post_norm = PostNormType::None;
  LayerAttentionType type = LayerAttentionType::kGemma;
  ActivationType activation = ActivationType::Gelu;
  PostQKType post_qk = PostQKType::Rope;
  bool use_qk_norm = false;
  bool norm_v = false;  // Normalize V projections before caching.
  uint32_t num_experts = 0;
  uint32_t num_experts_per_datapoint = 0;
  uint32_t ple_dim = 0;  // Per-Layer Embedding dimension (0 = disabled).
  int kv_share_layer_idx = -1;

  // DeepSeek MLA (all zero for other models). `qkv_dim` is the per-head query
  // dim (nope + rope); `rope_head_dim` is the decoupled RoPE sub-dim.
  uint32_t kv_lora_rank = 0;   // latent KV compression rank; 0 = no MLA
  uint32_t q_lora_rank = 0;    // query compression rank; 0 = direct q proj
  uint32_t rope_head_dim = 0;  // decoupled RoPE key/query dim
  uint32_t v_head_dim = 0;     // per-head value dim after up-projection
  AttentionVariant attention_variant = AttentionVariant::kDense;
  uint32_t kv_compression_rate = 1;  // sequence pooling rate (CSA=4, HCA=128)
  uint32_t indexer_heads = 0;        // lightning indexer (CSA); 0 = disabled
  uint32_t indexer_head_dim = 0;
  uint32_t indexer_top_k = 0;  // compressed entries selected per query

  // DeepSeek MoE extensions.
  uint32_t num_shared_experts = 0;  // uses the layer's dense FFN tensors
  bool sigmoid_gating = false;      // sigmoid routing instead of softmax
  bool use_routing_bias = false;    // aux-loss-free bias added for selection

  // DeepSeek V4 grouped low-rank output projection: heads are split into
  // `o_groups` groups; each group's concatenated outputs go through a
  // per-group [o_lora_rank, heads/o_groups * qkv_dim] wo_a, then the
  // concatenated group outputs through wo_b. 0 = V3-style direct o-proj.
  uint32_t o_lora_rank = 0;
  uint32_t o_groups = 1;

  // DeepSeek V4 MoE: SwiGLU clamp (0 = disabled), routed expert weight scale,
  // router scoring function, and hash-based routing (expert indices read from
  // a per-token-id table instead of top-k over scores).
  float swiglu_limit = 0.0f;
  float route_scale = 1.0f;
  RouterScoreFunc router_score = RouterScoreFunc::kSigmoidGatingCompat;
  bool hash_routing = false;

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
    bool unused_use_local_attention = false;  // formerly used for Griffin
    visitor(unused_use_local_attention);
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

    visitor(use_global_timescale);
    visitor(partial_rotary_factor);
    visitor(ple_dim);
    visitor(hc_mult);

    visitor(rope_theta);
    visitor(compress_rope_theta);
    visitor(yarn_orig_seq_len);
    visitor(yarn_factor);
    visitor(yarn_beta_fast);
    visitor(yarn_beta_slow);
    visitor(hc_sinkhorn_iters);
    visitor(hc_eps);
    visitor(num_mtp_layers);

    visitor(tokenizer_kind);
    visitor(is_encoder_decoder);
    if (is_encoder_decoder) {
      visitor(encoder_num_layers);
      visitor(encoder_layer_configs);
      visitor(encoder_attention_window_sizes);
      visitor(decoder_num_layers);
      visitor(decoder_layer_configs);
      visitor(decoder_attention_window_sizes);
    }

    // Append new fields here, then update `python/configs.cc`.
  }

  // Any layer using DeepSeek-style Multi-head Latent Attention?
  bool HasMLA() const {
    for (const auto& lc : layer_configs) {
      if (lc.IsMLA()) return true;
    }
    return false;
  }

  // Synthesized config for the multi-token-prediction block (DeepSeek V4):
  // a full extra layer (dense MLA + MoE) used for speculative decoding, not
  // part of the main stack. Only valid when `num_mtp_layers > 0`.
  LayerConfig MTPLayerConfig() const;

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

  // Overwrites `max_seq_len` with `new_max_seq_len` and updates all global
  // layers' attention window sizes to `new_max_seq_len`. This function must be
  // called before instantiating the KVCache object.
  void SetMaxSeqLen(size_t new_max_seq_len) {
    for (size_t i = 0; i < attention_window_sizes.size(); ++i) {
      if (attention_window_sizes[i] == max_seq_len) {
        attention_window_sizes[i] = new_max_seq_len;
      }
    }
    for (size_t i = 0; i < encoder_attention_window_sizes.size(); ++i) {
      if (encoder_attention_window_sizes[i] == max_seq_len) {
        encoder_attention_window_sizes[i] = new_max_seq_len;
      }
    }
    for (size_t i = 0; i < decoder_attention_window_sizes.size(); ++i) {
      if (decoder_attention_window_sizes[i] == max_seq_len) {
        decoder_attention_window_sizes[i] = new_max_seq_len;
      }
    }
    max_seq_len = new_max_seq_len;
  }

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
    for (const auto& layer_config : encoder_layer_configs) {
      num_heads = HWY_MAX(num_heads, layer_config.heads);
    }
    for (const auto& layer_config : decoder_layer_configs) {
      num_heads = HWY_MAX(num_heads, layer_config.heads);
    }
    return num_heads;
  }

  size_t KVCacheCols() const {
    if (is_encoder_decoder) {
      size_t cols = 0;
      for (const auto& lc : decoder_layer_configs) {
        cols += lc.CacheLayerSize();
      }
      return cols;
    }
    size_t cols = 0;
    for (const auto& lc : layer_configs) {
      cols += lc.CacheLayerSize();
    }
    // The MTP block caches its latents in an extra trailing segment.
    if (num_mtp_layers > 0) cols += MTPLayerConfig().CacheLayerSize();
    return cols;
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
  bool use_global_timescale = false;  // for Gemma 3
  float partial_rotary_factor =
      1.0f;              // Fraction of dims with RoPE (0.25 for Gemma4 MoE).
  uint32_t ple_dim = 0;  // Per-Layer Embedding dimension (0 = disabled).
  // Manifold-constrained hyper-connections (DeepSeek V4): number of parallel
  // residual streams. 0 or 1 = conventional residual connections.
  uint32_t hc_mult = 0;

  // RoPE base frequency for the raw/sliding-window attention path.
  float rope_theta = 10000.0f;
  // DeepSeek V4: separate base frequency for compressed-attention layers
  // (query, window keys and compressed keys all use this on those layers).
  // 0 = same as rope_theta.
  float compress_rope_theta = 0.0f;
  // YaRN frequency interpolation, applied only to the compressed-path
  // timescales. 0 = disabled.
  uint32_t yarn_orig_seq_len = 0;
  float yarn_factor = 1.0f;
  float yarn_beta_fast = 32.0f;
  float yarn_beta_slow = 1.0f;
  // mHC dynamic Sinkhorn-Knopp projection parameters (DeepSeek V4).
  uint32_t hc_sinkhorn_iters = 20;
  float hc_eps = 1e-6f;
  // Number of multi-token-prediction blocks (DeepSeek V4: 1). The MTP block
  // is an extra transformer layer used for speculative decoding; it is not
  // counted in `num_layers` and never runs as part of the main stack.
  uint32_t num_mtp_layers = 0;

  TokenizerKind tokenizer_kind = TokenizerKind::kSentencePiece;

  // Text encoder-decoder models such as T5Gemma have separate encoder and
  // decoder stacks. Existing decoder-only models leave these fields at their
  // defaults and continue to use `layer_configs`.
  bool is_encoder_decoder = false;
  uint32_t encoder_num_layers = 0;
  std::vector<LayerConfig> encoder_layer_configs;
  std::vector<uint32_t> encoder_attention_window_sizes;
  uint32_t decoder_num_layers = 0;
  std::vector<LayerConfig> decoder_layer_configs;
  std::vector<uint32_t> decoder_attention_window_sizes;
};

// Returns the sub-config for the ViT model of the PaliGemma model.
ModelConfig GetVitConfig(const ModelConfig& config);

enum DeducedLayerTypes {
  kDeducedViT = 2,
  kDeduced448 = 4,  // For ViT, 448x448 resolution instead of 224x224.
  kDeducedKqNorm = 8,
  kDeducedT5Gemma = 16,
};

// layer_types is one or more of `DeducedLayerTypes`.
Model DeduceModel(const Path& blob_path, size_t layers, int layer_types);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_CONFIGS_H_
