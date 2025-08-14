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

#include "gemma/configs.h"

#include <stddef.h>
#include <stdio.h>

#include <string>
#include <vector>

#include "compression/types.h"  // Type
#include "io/fields.h"           // IFields
#include "io/io.h"               // Path
#include "hwy/base.h"

namespace gcpp {

static constexpr size_t kVocabSize = 256000;

static constexpr size_t kGemmaV3VocabSize = 262144;

static ModelConfig ConfigNoSSM() {
  ModelConfig config;
  config.scale_base_names = {"att_ein",    "qkv_ein",      "gr_lin_x_w",
                             "gr_lin_y_w", "gr_lin_out_w", "gr_gate_w",
                             "gating_ein", "linear_w"};
  return config;
}

static ModelConfig ConfigBaseGemmaV2() {
  ModelConfig config = ConfigNoSSM();
  config.att_cap = 50.0f;
  config.final_cap = 30.0f;
  config.eos_id = 1;
  config.secondary_eos_id = 107;
  return config;
}

static LayerConfig LayerConfigGemma2_27B(size_t model_dim) {
  LayerConfig config;
  config.model_dim = model_dim;
  config.ff_hidden_dim = 16 * 4608 / 2;  // = 36864
  config.heads = 32;
  config.kv_heads = 16;
  config.qkv_dim = 128;
  config.optimized_gating = false;
  config.post_norm = PostNormType::Scale;
  return config;
}

static ModelConfig ConfigGemma2_27B() {
  ModelConfig config = ConfigBaseGemmaV2();
  config.display_name = "Gemma2_27B";
  config.model = Model::GEMMA2_27B;
  config.model_dim = 4608;
  config.vocab_size = kVocabSize;
  config.max_seq_len = 8192;
  LayerConfig layer_config = LayerConfigGemma2_27B(config.model_dim);
  config.num_layers = 46;
  config.layer_configs = {config.num_layers, layer_config};
  config.query_scale = QueryScaleType::SqrtModelDimDivNumHeads;
  config.attention_window_sizes =
      RepeatedAttentionWindowSizes<46, 2>({4096, config.max_seq_len});
  return config;
}

static LayerConfig LayerConfigGemma2_9B(size_t model_dim) {
  LayerConfig config;
  config.model_dim = model_dim;
  config.ff_hidden_dim = 8 * 3584 / 2;  // = 14336
  config.heads = 16;
  config.kv_heads = 8;
  config.qkv_dim = 256;
  config.optimized_gating = false;
  config.post_norm = PostNormType::Scale;
  return config;
}

static ModelConfig ConfigGemma2_9B() {
  ModelConfig config = ConfigBaseGemmaV2();
  config.display_name = "Gemma2_9B";
  config.model = Model::GEMMA2_9B;
  config.model_dim = 3584;
  config.vocab_size = kVocabSize;
  config.max_seq_len = 8192;
  LayerConfig layer_config = LayerConfigGemma2_9B(config.model_dim);
  config.num_layers = 42;
  config.layer_configs = {config.num_layers, layer_config};
  config.query_scale = QueryScaleType::SqrtKeySize;
  config.attention_window_sizes =
      RepeatedAttentionWindowSizes<42, 2>({4096, config.max_seq_len});
  return config;
}

static LayerConfig LayerConfigGemma2_2B(size_t model_dim) {
  LayerConfig config;
  config.model_dim = model_dim;
  config.ff_hidden_dim = 8 * 2304 / 2;  // = 9216
  config.heads = 8;
  config.kv_heads = 4;
  config.qkv_dim = 256;
  config.optimized_gating = false;
  config.post_norm = PostNormType::Scale;
  return config;
}

static ModelConfig ConfigGemma2_2B() {
  ModelConfig config = ConfigBaseGemmaV2();
  config.display_name = "Gemma2_2B";
  config.model = Model::GEMMA2_2B;
  config.model_dim = 2304;
  config.vocab_size = kVocabSize;
  config.max_seq_len = 8192;
  LayerConfig layer_config = LayerConfigGemma2_2B(config.model_dim);
  config.num_layers = 26;
  config.layer_configs = {config.num_layers, layer_config};
  config.query_scale = QueryScaleType::SqrtKeySize;
  config.attention_window_sizes =
      RepeatedAttentionWindowSizes<26, 2>({4096, config.max_seq_len});
  return config;
}

static LayerConfig LayerConfigGemmaTiny(size_t model_dim) {
  LayerConfig config;
  config.model_dim = model_dim;
  config.ff_hidden_dim = 256;
  config.heads = 4;
  config.kv_heads = 1;
  config.qkv_dim = 16;
  return config;
}

static ModelConfig ConfigGemmaTiny() {
  ModelConfig config = ConfigNoSSM();
  config.display_name = "GemmaTiny";
  config.model = Model::GEMMA_TINY;
  config.wrapping = PromptWrapping::GEMMA_IT;
  config.model_dim = 32;
  config.vocab_size = 32;  // at least two f32 vectors
  config.max_seq_len = 32;
  LayerConfig layer_config = LayerConfigGemmaTiny(config.model_dim);
  config.num_layers = 2;
  config.layer_configs = {config.num_layers, layer_config};
  config.query_scale = QueryScaleType::SqrtKeySize;
  config.attention_window_sizes = FixedAttentionWindowSizes<2>(32);
  config.att_cap = 50.0f;
  config.final_cap = 30.0f;
  config.eos_id = 11;
  config.secondary_eos_id = 11;
  return config;
}

static LayerConfig LayerConfigGriffin2B(size_t model_dim) {
  LayerConfig config;
  config.model_dim = model_dim;
  config.griffin_dim = model_dim;
  config.ff_hidden_dim = 7680;
  config.heads = 10;
  config.kv_heads = 1;
  config.qkv_dim = 256;
  config.conv1d_width = 4;
  HWY_DASSERT(config.conv1d_width <= kMaxConv1DWidth);
  config.ff_biases = true;
  config.softmax_attn_output_biases = true;
  config.optimized_gating = false;
  config.type = LayerAttentionType::kGriffinRecurrentBlock;
  config.activation = ActivationType::Gelu;
  config.post_qk = PostQKType::HalfRope;
  return config;
}

static ModelConfig ConfigGriffin2B() {
  ModelConfig config = ConfigNoSSM();
  config.display_name = "Griffin2B";
  config.model = Model::GRIFFIN_2B;
  // Griffin uses local attention, so max_seq_len is actually the local
  // attention window.
  config.model_dim = 2560;
  config.vocab_size = kVocabSize;
  config.max_seq_len = 2048;
  LayerConfig layer_config = LayerConfigGriffin2B(config.model_dim);
  config.num_layers = 26;
  config.layer_configs = {config.num_layers, layer_config};
  for (size_t i = 2; i < config.num_layers; i += 3) {
    config.layer_configs[i].type = LayerAttentionType::kGemma;
    config.layer_configs[i].griffin_dim = 0;
  }
  config.attention_window_sizes =
      FixedAttentionWindowSizes<26>(config.max_seq_len);
  config.use_local_attention = true;
  config.final_cap = 0.0f;
  return config;
}

static LayerConfig LayerConfigVit(size_t model_dim) {
  LayerConfig config;
  config.model_dim = model_dim;
  config.ff_hidden_dim = 4304;
  config.heads = 16;
  config.kv_heads = 16;
  config.qkv_dim = 72;
  config.ff_biases = true;
  config.type = LayerAttentionType::kVit;
  return config;
}

// Adds a ViT config (SigLIP SoViT ViT, used in PaliGemma) to the model config.
static void AddVitConfig(ModelConfig& config, size_t image_size = 224) {
  config.vit_config.model_dim = 1152;
  config.vocab_size = 256000 + 1024 + 128;  // = 257152
  config.vit_config.image_size = image_size;
  config.vit_config.patch_width = 14;
  const size_t num_patches =
      config.vit_config.image_size / config.vit_config.patch_width;
  config.vit_config.seq_len = num_patches * num_patches;
  for (auto& layer_config : config.layer_configs) {
    layer_config.optimized_gating = false;
  }
  LayerConfig vit_layer_config = LayerConfigVit(config.vit_config.model_dim);
  config.vit_config.layer_configs = {27, vit_layer_config};
  config.vit_config.num_scales = 4 * config.vit_config.layer_configs.size();
}

ModelConfig GetVitConfig(const ModelConfig& config) {
  ModelConfig vit_config = ConfigNoSSM();
  vit_config.model_dim = config.vit_config.model_dim;
  vit_config.max_seq_len = config.vit_config.seq_len;
  vit_config.layer_configs = config.vit_config.layer_configs;
  vit_config.pool_dim = config.vit_config.pool_dim;
  vit_config.wrapping = config.wrapping;
  // The Vit part does not have a vocabulary, the image patches are embedded.
  vit_config.vocab_size = 0;
  return vit_config;
}

static ModelConfig ConfigPaliGemma2_3B_224() {
  ModelConfig config = ConfigGemma2_2B();
  config.display_name = "PaliGemma2_3B_224";
  config.model = Model::PALIGEMMA2_3B_224;
  config.wrapping = PromptWrapping::PALIGEMMA;
  AddVitConfig(config);
  return config;
}

static ModelConfig ConfigPaliGemma2_3B_448() {
  ModelConfig config = ConfigGemma2_2B();
  config.display_name = "PaliGemma2_3B_448";
  config.model = Model::PALIGEMMA2_3B_448;
  config.wrapping = PromptWrapping::PALIGEMMA;
  AddVitConfig(config, /*image_size=*/448);
  return config;
}

static ModelConfig ConfigPaliGemma2_10B_224() {
  ModelConfig config = ConfigGemma2_9B();
  config.display_name = "PaliGemma2_10B_224";
  config.model = Model::PALIGEMMA2_10B_224;
  config.wrapping = PromptWrapping::PALIGEMMA;
  AddVitConfig(config);
  return config;
}

static ModelConfig ConfigPaliGemma2_10B_448() {
  ModelConfig config = ConfigGemma2_9B();
  config.display_name = "PaliGemma2_10B_448";
  config.model = Model::PALIGEMMA2_10B_448;
  config.wrapping = PromptWrapping::PALIGEMMA;
  AddVitConfig(config, /*image_size=*/448);
  return config;
}

static ModelConfig ConfigBaseGemmaV3() {
  ModelConfig config = ConfigNoSSM();
  config.att_cap = 0.0f;
  config.final_cap = 0.0f;
  config.eos_id = 1;
  config.secondary_eos_id = 106;
  return config;
}

// 1B does not include a vision encoder.
static LayerConfig LayerConfigGemma3_1B_LM(size_t model_dim) {
  LayerConfig config;
  config.model_dim = model_dim;
  config.ff_hidden_dim = 6912;
  config.heads = 4;
  config.kv_heads = 1;
  config.qkv_dim = 256;
  config.optimized_gating = true;
  config.post_norm = PostNormType::Scale;
  config.use_qk_norm = true;
  return config;
}

static ModelConfig ConfigGemma3_1B() {
  ModelConfig config = ConfigBaseGemmaV3();
  config.display_name = "Gemma3_1B";
  config.model = Model::GEMMA3_1B;
  config.wrapping = PromptWrapping::GEMMA_VLM;
  config.model_dim = 1152;
  config.vocab_size = kGemmaV3VocabSize;  // new vocab size / tokenizer
  config.max_seq_len = 32 * 1024;
  LayerConfig layer_config = LayerConfigGemma3_1B_LM(config.model_dim);
  config.num_layers = 26;
  config.layer_configs = {config.num_layers, layer_config};
  config.query_scale = QueryScaleType::SqrtKeySize;
  // interleaved local / global attention
  config.attention_window_sizes = RepeatedAttentionWindowSizes<26, 6>(
      {512, 512, 512, 512, 512, config.max_seq_len});
  return config;
}

static LayerConfig LayerConfigGemma3_4B_LM(size_t model_dim) {
  LayerConfig config;
  config.model_dim = model_dim;
  config.ff_hidden_dim = 8 * 2560 / 2;  // = 10240
  config.heads = 8;
  config.kv_heads = 4;
  config.qkv_dim = 256;
  config.optimized_gating = true;
  config.post_norm = PostNormType::Scale;
  config.use_qk_norm = true;
  return config;
}

// Until we have the SigLIP checkpoints included, we use the LM config directly.
static ModelConfig ConfigGemma3_4B_LM() {
  ModelConfig config = ConfigBaseGemmaV3();
  config.display_name = "Gemma3_4B";
  config.model = Model::GEMMA3_4B;
  config.wrapping = PromptWrapping::GEMMA_VLM;
  config.model_dim = 2560;
  config.vocab_size = kGemmaV3VocabSize;  // new vocab size / tokenizer
  config.max_seq_len = 32 * 1024;
  LayerConfig layer_config = LayerConfigGemma3_4B_LM(config.model_dim);
  config.num_layers = 34;
  config.layer_configs = {config.num_layers, layer_config};
  config.query_scale = QueryScaleType::SqrtKeySize;
  // interleaved local / global attention
  config.attention_window_sizes = RepeatedAttentionWindowSizes<34, 6>(
      {1024, 1024, 1024, 1024, 1024, config.max_seq_len});
  return config;
}

static ModelConfig ConfigGemma3_4B() {
  ModelConfig config = ConfigGemma3_4B_LM();
  config.display_name = "Gemma3_4B";
  config.model = Model::GEMMA3_4B;
  config.wrapping = PromptWrapping::GEMMA_VLM;
  AddVitConfig(config, /*image_size=*/896);
  config.vocab_size = kGemmaV3VocabSize;
  config.vit_config.pool_dim = 4;
  const size_t num_patches =
      config.vit_config.image_size / config.vit_config.patch_width;
  config.vit_config.seq_len = (num_patches * num_patches);
  // The above resets optimized gating to false; for Gemma 3 it should be true.
  for (auto& layer_config : config.layer_configs) {
    layer_config.optimized_gating = true;
  }
  return config;
}

static LayerConfig LayerConfigGemma3_12B_LM(size_t model_dim) {
  LayerConfig config;
  config.model_dim = model_dim;
  config.ff_hidden_dim = 15360;
  config.heads = 16;
  config.kv_heads = 8;
  config.qkv_dim = 256;
  config.optimized_gating = true;
  config.post_norm = PostNormType::Scale;
  config.use_qk_norm = true;
  return config;
}

static ModelConfig ConfigGemma3_12B_LM() {
  ModelConfig config = ConfigBaseGemmaV3();
  config.display_name = "Gemma3_12B";
  config.model = Model::GEMMA3_12B;
  config.wrapping = PromptWrapping::GEMMA_VLM;
  config.model_dim = 3840;
  config.vocab_size = kGemmaV3VocabSize;  // new vocab size / tokenizer
  config.max_seq_len = 32 * 1024;
  LayerConfig layer_config = LayerConfigGemma3_12B_LM(config.model_dim);
  config.num_layers = 48;
  config.layer_configs = {config.num_layers, layer_config};
  config.query_scale = QueryScaleType::SqrtKeySize;
  // interleaved local / global attention
  config.attention_window_sizes = RepeatedAttentionWindowSizes<48, 6>(
      {1024, 1024, 1024, 1024, 1024, config.max_seq_len});
  return config;
}

static ModelConfig ConfigGemma3_12B() {
  ModelConfig config = ConfigGemma3_12B_LM();
  config.display_name = "Gemma3_12B";
  config.model = Model::GEMMA3_12B;
  config.wrapping = PromptWrapping::GEMMA_VLM;
  AddVitConfig(config, /*image_size=*/896);
  config.vocab_size = kGemmaV3VocabSize;
  config.vit_config.pool_dim = 4;
  const size_t num_patches =
      config.vit_config.image_size / config.vit_config.patch_width;
  config.vit_config.seq_len = (num_patches * num_patches);
  // The above resets optimized gating to false; for Gemma 3 it should be true.
  for (auto& layer_config : config.layer_configs) {
    layer_config.optimized_gating = true;
  }
  return config;
}

static LayerConfig LayerConfigGemma3_27B_LM(size_t model_dim) {
  LayerConfig config;
  config.model_dim = model_dim;
  config.ff_hidden_dim = 21504;
  config.heads = 32;
  config.kv_heads = 16;
  config.qkv_dim = 128;
  config.optimized_gating = true;
  config.post_norm = PostNormType::Scale;
  config.use_qk_norm = true;
  return config;
}

static ModelConfig ConfigGemma3_27B_LM() {
  ModelConfig config = ConfigBaseGemmaV3();
  config.display_name = "Gemma3_27B";
  config.model = Model::GEMMA3_27B;
  config.wrapping = PromptWrapping::GEMMA_VLM;
  config.model_dim = 5376;
  config.vocab_size = kGemmaV3VocabSize;  // new vocab size / tokenizer
  config.max_seq_len = 32 * 1024;
  LayerConfig layer_config = LayerConfigGemma3_27B_LM(config.model_dim);
  config.num_layers = 62;
  config.layer_configs = {config.num_layers, layer_config};
  config.query_scale = QueryScaleType::SqrtKeySize;
  // interleaved local / global attention
  config.attention_window_sizes = RepeatedAttentionWindowSizes<62, 6>(
      {1024, 1024, 1024, 1024, 1024, config.max_seq_len});
  return config;
}

static ModelConfig ConfigGemma3_27B() {
  ModelConfig config = ConfigGemma3_27B_LM();
  config.display_name = "Gemma3_27B";
  config.model = Model::GEMMA3_27B;
  config.wrapping = PromptWrapping::GEMMA_VLM;
  AddVitConfig(config, /*image_size=*/896);
  config.vocab_size = kGemmaV3VocabSize;
  config.vit_config.pool_dim = 4;
  const size_t num_patches =
      config.vit_config.image_size / config.vit_config.patch_width;
  config.vit_config.seq_len = (num_patches * num_patches);
  // The above resets optimized gating to false; for Gemma 3 it should be true.
  for (auto& layer_config : config.layer_configs) {
    layer_config.optimized_gating = true;
  }
  return config;
}

static LayerConfig LayerConfigGemma3_270M_LM(size_t model_dim) {
  LayerConfig config;
  config.model_dim = model_dim;
  config.ff_hidden_dim = 2048;
  config.heads = 4;
  config.kv_heads = 1;
  config.qkv_dim = 256;
  config.optimized_gating = true;
  config.post_norm = PostNormType::Scale;
  config.use_qk_norm = true;
  return config;
}

static ModelConfig ConfigGemma3_270M() {
  ModelConfig config = ConfigBaseGemmaV3();
  config.display_name = "Gemma3_270M";
  config.model = Model::GEMMA3_270M;
  config.wrapping = PromptWrapping::GEMMA_IT;
  config.model_dim = 640;
  config.vocab_size = kGemmaV3VocabSize;  // new vocab size / tokenizer
  config.max_seq_len = 32 * 1024;
  LayerConfig layer_config = LayerConfigGemma3_270M_LM(config.model_dim);
  config.num_layers = 18;
  config.layer_configs = {config.num_layers, layer_config};
  config.query_scale = QueryScaleType::SqrtKeySize;
  // interleaved local / global attention
  config.attention_window_sizes = RepeatedAttentionWindowSizes<18, 6>(
      {512, 512, 512, 512, 512, config.max_seq_len});
  return config;
}

static ModelConfig ConfigFromModel(Model model) {
  switch (model) {
    case Model::GEMMA2_2B:
      return ConfigGemma2_2B();
    case Model::GEMMA2_9B:
      return ConfigGemma2_9B();
    case Model::GEMMA2_27B:
      return ConfigGemma2_27B();
    case Model::GRIFFIN_2B:
      return ConfigGriffin2B();
    case Model::GEMMA_TINY:
      return ConfigGemmaTiny();
    case Model::PALIGEMMA2_3B_224:
      return ConfigPaliGemma2_3B_224();
    case Model::PALIGEMMA2_3B_448:
      return ConfigPaliGemma2_3B_448();
    case Model::PALIGEMMA2_10B_224:
      return ConfigPaliGemma2_10B_224();
    case Model::PALIGEMMA2_10B_448:
      return ConfigPaliGemma2_10B_448();
    case Model::GEMMA3_4B:
      return ConfigGemma3_4B();
    case Model::GEMMA3_1B:
      return ConfigGemma3_1B();
    case Model::GEMMA3_12B:
      return ConfigGemma3_12B();
    case Model::GEMMA3_27B:
      return ConfigGemma3_27B();
    case Model::GEMMA3_270M:
      return ConfigGemma3_270M();
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

const char* ModelPrefix(Model model) {
  switch (model) {
    case Model::UNKNOWN:
      return "unknown";
    case Model::GEMMA2_2B:
      return "gemma2-2b";
    case Model::GEMMA2_9B:
      return "9b";
    case Model::GEMMA2_27B:
      return "27b";
    case Model::GRIFFIN_2B:
      return "gr2b";
    case Model::GEMMA_TINY:
      return "tiny";
    case Model::PALIGEMMA2_3B_224:
      return "paligemma2-3b-224";
    case Model::PALIGEMMA2_3B_448:
      return "paligemma2-3b-448";
    case Model::PALIGEMMA2_10B_224:
      return "paligemma2-10b-224";
    case Model::PALIGEMMA2_10B_448:
      return "paligemma2-10b-448";
    case Model::GEMMA3_4B:
      return "gemma3-4b";
    case Model::GEMMA3_1B:
      return "gemma3-1b";
    case Model::GEMMA3_12B:
      return "gemma3-12b";
    case Model::GEMMA3_27B:
      return "gemma3-27b";
    case Model::GEMMA3_270M:
      return "gemma3-270m";
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

PromptWrapping ChooseWrapping(const Model model, Tristate wrapping) {
  if (IsPaliGemma(model)) {
    if (wrapping != Tristate::kDefault) {
      HWY_WARN("Ignoring unnecessary --wrapping for PaliGemma models.");
    }
    return PromptWrapping::PALIGEMMA;
  }
  if (IsVLM(model)) {
    if (wrapping != Tristate::kDefault) {
      HWY_WARN("Ignoring unnecessary --wrapping for VLM models.");
    }
    return PromptWrapping::GEMMA_VLM;
  }
  // Default to IT unless --wrapping=0.
  return wrapping == Tristate::kFalse ? PromptWrapping::GEMMA_PT
                                      : PromptWrapping::GEMMA_IT;
}

ModelConfig::ModelConfig(const Model model, Type weight,
                         PromptWrapping wrapping) {
  HWY_ASSERT(weight != Type::kUnknown);
  HWY_ASSERT(wrapping != PromptWrapping::kSentinel);
  this->model = model;
  if (model != Model::UNKNOWN) *this = ConfigFromModel(model);
  HWY_ASSERT(this->model == model);
  this->weight = weight;
  this->wrapping = wrapping;
}

static Model FindModel(const std::string& specifier) {
  Model found_model = Model::UNKNOWN;
  ForEachModel([&](Model model) {
    // Some model names are prefixes of other model names
    const std::string prefix = std::string(ModelPrefix(model)) + "-";
    if (specifier.rfind(prefix, 0) == 0) {  // Starts with prefix.
      // We only expect one match.
      HWY_ASSERT_M(found_model == Model::UNKNOWN, specifier.c_str());
      found_model = model;
    }
  });
  HWY_ASSERT_M(found_model != Model::UNKNOWN, specifier.c_str());
  return found_model;
}

static Type FindType(const std::string& specifier) {
  Type found_type = Type::kUnknown;
  for (size_t i = 1; i < kNumTypes; ++i) {
    const Type type = static_cast<Type>(i);
    if (specifier.find(TypeName(type)) != std::string::npos) {  // NOLINT
      // We only expect one match.
      HWY_ASSERT_M(found_type == Type::kUnknown, specifier.c_str());
      found_type = type;
    }
  }
  HWY_ASSERT_M(found_type != Type::kUnknown, specifier.c_str());
  return found_type;
}

static PromptWrapping FindWrapping(const std::string& specifier) {
  PromptWrapping found_wrapping = PromptWrapping::kSentinel;
  for (size_t i = 0; i < static_cast<size_t>(PromptWrapping::kSentinel); ++i) {
    const PromptWrapping w = static_cast<PromptWrapping>(i);
    if (specifier.find(WrappingSuffix(w)) != std::string::npos) {  // NOLINT
      // We expect zero or one match.
      HWY_ASSERT_M(found_wrapping == PromptWrapping::kSentinel,
                   specifier.c_str());
      found_wrapping = w;
    }
  }
  if (found_wrapping == PromptWrapping::kSentinel) {
    return ChooseWrapping(FindModel(specifier));
  }
  return found_wrapping;
}

// Obtains model/weight/wrapping by finding prefix and suffix strings.
ModelConfig::ModelConfig(const std::string& specifier)
    : ModelConfig(FindModel(specifier), FindType(specifier),
                  FindWrapping(specifier)) {}

std::string ModelConfig::Specifier() const {
  HWY_ASSERT(model != Model::UNKNOWN);
  HWY_ASSERT(weight != Type::kUnknown);
  HWY_ASSERT(wrapping != PromptWrapping::kSentinel);

  std::string base_name = ModelPrefix(model);

  base_name += '-';
  base_name += TypeName(weight);

  if (wrapping != PromptWrapping::GEMMA_VLM &&
      wrapping != PromptWrapping::PALIGEMMA) {
    base_name += WrappingSuffix(wrapping);
  }

  return base_name;
}

// Returns whether all fields match.
static bool AllEqual(const IFields& a, const IFields& b, bool print) {
  const std::vector<uint32_t> serialized_a = a.Write();
  const std::vector<uint32_t> serialized_b = b.Write();
  if (serialized_a != serialized_b) {
    if (print) {
      fprintf(stderr, "%s differs. Recommend generating a diff:\n", a.Name());
      a.Print();
      b.Print();
    }
    return false;
  }
  return true;
}

bool LayerConfig::TestEqual(const LayerConfig& other, bool print) const {
  return AllEqual(*this, other, print);
}

bool VitConfig::TestEqual(const VitConfig& other, bool print) const {
  return AllEqual(*this, other, print);
}

bool ModelConfig::TestEqual(const ModelConfig& other, bool print) const {
  // Early out to guard the loop below; a differing number of layers will anyway
  // cause a mismatch.
  if (layer_configs.size() != other.layer_configs.size()) {
    if (print) {
      HWY_WARN("Layer configs size mismatch %zu vs %zu", layer_configs.size(),
               other.layer_configs.size());
    }
    return false;
  }

  // Copy so we can 'ignore' fields by setting them to the same value.
  ModelConfig a = *this;
  ModelConfig b = other;
  // Called by `OverwriteWithCanonical`, so ignore the fields it will set.
  a.display_name = b.display_name;
  a.model = b.model;

  // The following are not yet set by config_converter.py, so we here ignore
  // them for purposes of comparison, and there overwrite the converter's config
  // with the canonical ModelConfig constructed via (deduced) enum, so that
  // these fields will be set.
  // `vit_config` is also not yet set, but we must not ignore it because
  // otherwise PaliGemma models will be indistinguishable for `configs_test`.
  a.pool_dim = b.pool_dim;  // ViT
  a.eos_id = b.eos_id;
  a.secondary_eos_id = b.secondary_eos_id;
  a.scale_base_names = b.scale_base_names;
  for (size_t i = 0; i < a.layer_configs.size(); ++i) {
    a.layer_configs[i].optimized_gating = b.layer_configs[i].optimized_gating;
  }

  return AllEqual(a, b, print);
}

// Constructs the canonical ModelConfig for each model. If there is one for
// which TestEqual returns true, overwrites `*this` with that and returns true.
bool ModelConfig::OverwriteWithCanonical() {
  bool found = false;
  const bool print = false;
  ForEachModel([&](Model model) {
    const ModelConfig config(model, weight, wrapping);
    if (config.TestEqual(*this, print)) {
      HWY_ASSERT(!found);  // Should only find one.
      found = true;
      *this = config;
    }
  });
  return found;
}

Model DeduceModel(const Path& blob_path, size_t layers, int layer_types) {
  switch (layers) {
    case 2:
      return Model::GEMMA_TINY;
    case 18:
      return Model::GEMMA3_270M;

    case 26:
      if (layer_types & kDeducedGriffin) return Model::GRIFFIN_2B;
      if (layer_types & kDeducedViT) return Model::GEMMA3_1B;
      return Model::GEMMA2_2B;
    case 27:
      return (layer_types & kDeduced448) ? Model::PALIGEMMA2_3B_448
                                         : Model::PALIGEMMA2_3B_224;
    case 34:
      return Model::GEMMA3_4B;
    case 42:
      if (layer_types & kDeducedViT) {
        return (layer_types & kDeduced448) ? Model::PALIGEMMA2_10B_448
                                           : Model::PALIGEMMA2_10B_224;
      }
      return Model::GEMMA2_9B;
    case 46:
      return Model::GEMMA2_27B;
    case 48:
      return Model::GEMMA3_12B;
    case 62:
      return Model::GEMMA3_27B;

    // TODO: detect these.
    /*
    return Model::GEMMA2_772M;
    return Model::PALIGEMMA2_772M_224;
    */
    default:
      HWY_WARN("Failed to deduce model type from %s, layer count %zu types %x.",
               blob_path.path.c_str(), layers, layer_types);
      return Model::UNKNOWN;
  }
}

}  // namespace gcpp
