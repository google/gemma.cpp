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
#include <string_view>
#include <utility>
#include <vector>

#include "compression/types.h"  // Type
#include "io/fields.h"          // IFields
#include "io/io.h"              // Path
#include "hwy/base.h"

namespace gcpp {

static constexpr size_t kVocabSize = 256000;

static constexpr size_t kGemmaV3VocabSize = 262144;

static ModelConfig ConfigNoSSM() {
  ModelConfig config;
  config.scale_base_names = {"att_ein",    "qkv_ein",      "gr_lin_x_w",
                             "gr_lin_y_w", "gr_lin_out_w", "gr_gate_w",
                             "gating_ein", "linear_w"};
  config.rope_theta = 10000.0f;
  config.global_rope_theta = 1000000.0f;
  return config;
}

static ModelConfig ConfigBaseGemmaV2() {
  ModelConfig config = ConfigNoSSM();
  config.att_cap = 50.0f;
  config.final_cap = 30.0f;
  config.eos_id = 1;
  config.secondary_eos_id = 107;
  config.rope_theta = 10000.0f;
  config.global_rope_theta = 10000.0f;
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
  config.rope_theta = 10000.0f;
  config.global_rope_theta = 1000000.0f;
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
  config.use_global_timescale = true;
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

// Shared LM-only config for Gemma3 4B: used directly for text-only checkpoints
// (e.g. TranslateGemma) and as the base for the VLM build.
static ModelConfig ConfigGemma3_4B_LM() {
  ModelConfig config = ConfigBaseGemmaV3();
  config.display_name = "Gemma3_4B_LM";
  config.model = Model::GEMMA3_4B_LM;
  config.wrapping = PromptWrapping::GEMMA_IT;
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
  config.use_global_timescale = true;
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
  config.display_name = "Gemma3_12B_LM";
  config.model = Model::GEMMA3_12B_LM;
  config.wrapping = PromptWrapping::GEMMA_IT;
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
  config.use_global_timescale = true;
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
  config.display_name = "Gemma3_27B_LM";
  config.model = Model::GEMMA3_27B_LM;
  config.wrapping = PromptWrapping::GEMMA_IT;
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
  config.use_global_timescale = true;
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
  config.global_rope_theta = 10000.0f;
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

static ModelConfig ConfigBaseGemmaV4() {
  ModelConfig config = ConfigNoSSM();
  config.model_family_version = 4;
  config.att_cap = 0.0f;
  config.final_cap = 0.0f;
  config.eos_id = 1;
  config.secondary_eos_id = 106;
  config.vocab_size = 262208;
  config.rope_theta = 10000.0f;
  config.global_rope_theta = 1000000.0f;
  return config;
}


static LayerConfig LayerConfigGemma4_26B_MoE_LM(size_t model_dim) {
  LayerConfig config;
  config.model_dim = model_dim;
  config.ff_hidden_dim = 2112;
  config.heads = 16;
  config.kv_heads = 8;
  config.qkv_dim = 256;
  config.optimized_gating = true;
  config.post_norm = PostNormType::Scale;
  config.activation = ActivationType::Gelu;
  config.post_qk = PostQKType::NormLocalRope;
  config.use_qk_norm = true;
  config.norm_v = true;

  config.num_experts = 128;
  config.num_experts_per_datapoint = 8;

  return config;
}

static ModelConfig ConfigGemma4_26B_MoE() {
  ModelConfig config = ConfigBaseGemmaV4();
  config.display_name = "Gemma4_26B_MoE";
  config.final_cap = 0.0f;
  config.att_cap = 0.0f;
  config.model = Model::GEMMA4_26B_MOE;
  config.wrapping = PromptWrapping::GEMMA_IT;
  config.use_global_timescale = true;
  config.partial_rotary_factor = 0.25f;
  config.model_dim = 2816;
  config.vocab_size = kGemmaV3VocabSize;
  config.max_seq_len = 32 * 1024;
  config.num_layers = 30;
  LayerConfig layer_config = LayerConfigGemma4_26B_MoE_LM(config.model_dim);
  config.layer_configs = {config.num_layers, layer_config};
  for (size_t i = 0; i < config.num_layers; ++i) {
    if (i % 6 == 5) {
      config.layer_configs[i].qkv_dim = 512;
      config.layer_configs[i].kv_heads = 2;
    }
  }
  config.secondary_eos_id = 106;  // <turn|> is the EOT for Gemma4 MoE
  config.query_scale = QueryScaleType::One;
  config.attention_window_sizes = RepeatedAttentionWindowSizes<30, 6>(
      {1024, 1024, 1024, 1024, 1024, config.max_seq_len});
  return config;
}

static LayerConfig LayerConfigGemma4_2B_Local(size_t model_dim) {
  LayerConfig config;
  config.model_dim = model_dim;
  config.ff_hidden_dim = 6144;
  config.heads = 8;
  config.kv_heads = 1;
  config.qkv_dim = 256;
  config.optimized_gating = true;
  config.post_norm = PostNormType::Scale;
  config.activation = ActivationType::Gelu;
  config.post_qk = PostQKType::NormLocalRope;
  config.use_qk_norm = true;
  config.norm_v = true;
  config.ple_dim = 256;
  return config;
}

static LayerConfig LayerConfigGemma4_2B_Global(size_t model_dim) {
  LayerConfig config = LayerConfigGemma4_2B_Local(model_dim);
  config.qkv_dim = 512;
  return config;
}

// Until we have the audio checkpoints included, we use the LM config directly.
static ModelConfig ConfigGemma4_2B_LM() {
  ModelConfig config = ConfigBaseGemmaV4();
  config.display_name = "Gemma4_2B_LM";
  config.model = Model::GEMMA4_2B_LM;
  config.wrapping = PromptWrapping::GEMMA_IT;
  config.model_dim = 1536;
  config.vocab_size = kGemmaV3VocabSize;  // 262144
  config.max_seq_len = 128 * 1024;
  config.final_cap = 0.0f;
  config.ple_dim = 256;
  config.num_layers = 35;
  config.use_global_timescale = true;
  config.partial_rotary_factor = 0.25f;
  config.query_scale = QueryScaleType::One;
  LayerConfig local_config = LayerConfigGemma4_2B_Local(config.model_dim);
  config.layer_configs = {config.num_layers, local_config};
  // Global attention layers: [4, 9, 14, 19, 24, 29, 34] (stride 5)
  for (size_t i = 0; i < config.num_layers; ++i) {
    if (i % 5 == 4) {
      config.layer_configs[i] = LayerConfigGemma4_2B_Global(config.model_dim);
    }
  }
  // Double-wide MLP for last 20 layers (KV-shared layers 15-34)
  for (size_t i = 15; i < config.num_layers; ++i) {
    config.layer_configs[i].ff_hidden_dim = 12288;
    config.layer_configs[i].kv_share_layer_idx = (i % 5 == 4) ? 14 : 13;
  }
  config.attention_window_sizes = RepeatedAttentionWindowSizes<35, 5>(
      {512, 512, 512, 512, config.max_seq_len});
  return config;
}

static ModelConfig ConfigGemma4_2B() {
  ModelConfig config = ConfigGemma4_2B_LM();
  config.display_name = "Gemma4_2B";
  config.model = Model::GEMMA4_2B;
  config.wrapping = PromptWrapping::GEMMA_IT;
  config.use_global_timescale = true;

  config.vit_config.model_dim = 768;
  config.vit_config.patch_width = 16;
  config.vit_config.seq_len = 2520;
  config.vit_config.pool_dim = 3;
  config.vit_config.image_size = 896;

  LayerConfig vit_layer;
  vit_layer.model_dim = 768;
  vit_layer.ff_hidden_dim = 3072;
  vit_layer.heads = 12;
  vit_layer.kv_heads = 12;
  vit_layer.qkv_dim = 64;
  vit_layer.type = LayerAttentionType::kVitGemma4;
  vit_layer.use_qk_norm = true;
  vit_layer.post_norm = PostNormType::Scale;

  config.vit_config.layer_configs = {16, vit_layer};
  return config;
}

static LayerConfig LayerConfigGemma4_E4B_Local(size_t model_dim) {
  LayerConfig config;
  config.model_dim = model_dim;
  config.ff_hidden_dim = 10240;
  config.heads = 8;
  config.kv_heads = 2;
  config.qkv_dim = 256;
  config.optimized_gating = true;
  config.post_norm = PostNormType::Scale;
  config.activation = ActivationType::Gelu;
  config.post_qk = PostQKType::NormLocalRope;
  config.use_qk_norm = true;
  config.norm_v = true;
  config.ple_dim = 256;
  return config;
}

static LayerConfig LayerConfigGemma4_E4B_Global(size_t model_dim) {
  LayerConfig config = LayerConfigGemma4_E4B_Local(model_dim);
  config.qkv_dim = 512;
  return config;
}

static ModelConfig ConfigGemma4_E4B_LM() {
  ModelConfig config = ConfigBaseGemmaV4();
  config.display_name = "Gemma4_E4B_LM";
  config.model = Model::GEMMA4_E4B_LM;
  config.wrapping = PromptWrapping::GEMMA_IT;
  config.model_dim = 2560;
  config.vocab_size = kGemmaV3VocabSize;  // 262144
  config.max_seq_len = 128 * 1024;
  config.final_cap = 0.0f;
  config.ple_dim = 256;
  config.num_layers = 42;
  config.use_global_timescale = true;
  config.partial_rotary_factor = 0.25f;
  config.query_scale = QueryScaleType::One;
  LayerConfig local_config = LayerConfigGemma4_E4B_Local(config.model_dim);
  config.layer_configs = {config.num_layers, local_config};
  // Global attention layers: [5, 11, 17, 23, 29, 35, 41] (stride 6)
  for (size_t i = 0; i < config.num_layers; ++i) {
    if (i % 6 == 5) {
      config.layer_configs[i] = LayerConfigGemma4_E4B_Global(config.model_dim);
    }
  }
  // KV-shared layers 24-41
  for (size_t i = 24; i < config.num_layers; ++i) {
    config.layer_configs[i].kv_share_layer_idx = (i % 6 == 5) ? 23 : 22;
  }
  config.attention_window_sizes = RepeatedAttentionWindowSizes<42, 6>(
      {512, 512, 512, 512, 512, config.max_seq_len});
  return config;
}

static ModelConfig ConfigGemma4_E4B() {
  ModelConfig config = ConfigGemma4_E4B_LM();
  config.display_name = "Gemma4_E4B";
  config.model = Model::GEMMA4_E4B;
  config.wrapping = PromptWrapping::GEMMA_IT;
  config.use_global_timescale = true;

  config.vit_config.model_dim = 768;
  config.vit_config.patch_width = 16;
  config.vit_config.seq_len = 2520;
  config.vit_config.pool_dim = 3;
  config.vit_config.image_size = 896;

  LayerConfig vit_layer;
  vit_layer.model_dim = 768;
  vit_layer.ff_hidden_dim = 3072;
  vit_layer.heads = 12;
  vit_layer.kv_heads = 12;
  vit_layer.qkv_dim = 64;
  vit_layer.type = LayerAttentionType::kVitGemma4;
  vit_layer.use_qk_norm = true;
  vit_layer.post_norm = PostNormType::Scale;

  config.vit_config.layer_configs = {16, vit_layer};
  return config;
}

static LayerConfig LayerConfigDeepSeek4_Flash(size_t model_dim) {
  LayerConfig config;
  config.model_dim = model_dim;
  config.ff_hidden_dim = 2048;  // moe_inter_dim: shared + routed expert width
  config.heads = 64;
  config.kv_heads = 1;  // unused by MLA; kept consistent for legacy fields
  // Per-head query dim = head_dim = nope (448) + decoupled rope (64).
  config.qkv_dim = 512;
  config.optimized_gating = false;
  config.post_norm = PostNormType::None;
  config.type = LayerAttentionType::kDeepSeekMLA;
  config.activation = ActivationType::Silu;
  config.use_qk_norm = false;

  config.kv_lora_rank = 448;  // head_dim (512) - rope_head_dim (64)
  config.q_lora_rank = 1024;
  config.rope_head_dim = 64;
  config.v_head_dim = 512;  // V4: value = the full 512-wide latent
  config.o_lora_rank = 1024;
  config.o_groups = 8;
  config.attention_variant = AttentionVariant::kCSA;  // per-layer, see below
  config.kv_compression_rate = 4;
  config.indexer_heads = 64;
  config.indexer_head_dim = 128;
  config.indexer_top_k = 512;

  config.num_experts = 256;
  config.num_experts_per_datapoint = 6;
  config.num_shared_experts = 1;
  config.sigmoid_gating = false;
  config.router_score = RouterScoreFunc::kSqrtSoftplus;
  config.route_scale = 1.5f;
  config.swiglu_limit = 10.0f;
  config.use_routing_bias = true;
  return config;
}

static ModelConfig ConfigDeepSeek4_Flash() {
  ModelConfig config;
  config.model_family_version = 4;
  config.display_name = "DeepSeek4_Flash";
  config.model = Model::DEEPSEEK4_FLASH;
  config.wrapping = PromptWrapping::GEMMA_IT;
  config.model_dim = 4096;
  config.vocab_size = 129280;
  // The released model supports 1,048,576 tokens (max_position_embeddings, via
  // YaRN). Capped here pending sliding-window / paged KV caches.
  config.max_seq_len = 131072;
  config.num_layers = 43;
  config.hc_mult = 4;
  config.eos_id = 1;
  config.secondary_eos_id = 1;
  LayerConfig layer_config = LayerConfigDeepSeek4_Flash(config.model_dim);
  config.layer_configs = {config.num_layers, layer_config};

  // Per-layer attention from the released `compress_ratios`
  // [0, 0, 4, 128, 4, 128, ..., 4, 0] (index 43 is the unmodeled MTP layer):
  //   0   -> pure sliding-window attention (no compressed entries),
  //   4   -> CSA with the lightning indexer,
  //   128 -> HCA (heavy pooling, no indexer).
  // Layers 0-1 use ratio 0; from layer 2 on, even layers are CSA and odd
  // layers HCA. All layers are MoE; the first 3 use hash routing (per-token-id
  // expert table, no routing bias).
  for (size_t i = 0; i < config.num_layers; ++i) {
    LayerConfig& lc = config.layer_configs[i];
    if (i < 2) {
      lc.SetDenseAttention();
    } else if ((i % 2) == 0) {
      lc.attention_variant = AttentionVariant::kCSA;
      lc.kv_compression_rate = 4;
      lc.indexer_heads = 64;
      lc.indexer_head_dim = 128;
      lc.indexer_top_k = 512;
    } else {
      lc.attention_variant = AttentionVariant::kHCA;
      lc.kv_compression_rate = 128;
      lc.indexer_heads = 0;
      lc.indexer_head_dim = 0;
      lc.indexer_top_k = 0;
    }
    if (i < 3) {
      lc.hash_routing = true;
      lc.use_routing_bias = false;
    }
  }
  config.query_scale = QueryScaleType::SqrtKeySize;
  // Sliding window of raw latents; compressed entries cover the rest of the
  // history on CSA/HCA layers.
  config.attention_window_sizes = FixedAttentionWindowSizes<43>(128);

  config.rope_theta = 10000.0f;
  config.global_rope_theta = 10000.0f;
  config.compress_rope_theta = 160000.0f;
  config.yarn_orig_seq_len = 65536;
  config.yarn_factor = 16.0f;
  config.yarn_beta_fast = 32.0f;
  config.yarn_beta_slow = 1.0f;
  config.hc_sinkhorn_iters = 20;
  config.hc_eps = 1e-6f;
  // The `mtp.0.*` multi-token-prediction block: one extra transformer layer
  // (dense MLA + MoE) used only for speculative decoding.
  config.num_mtp_layers = 1;
  return config;
}

// The MTP block has the attention shape of the dense layers (no compressor,
// no indexer) and the learned-gate MoE of layers >= 3 (no hash routing).
LayerConfig ModelConfig::MTPLayerConfig() const {
  HWY_ASSERT(num_mtp_layers > 0 && !layer_configs.empty());
  LayerConfig lc = layer_configs[0];
  lc.SetDenseAttention();
  lc.hash_routing = false;
  lc.use_routing_bias = true;
  return lc;
}

static ModelConfig ConfigBaseT5Gemma() {
  ModelConfig config = ConfigNoSSM();
  config.att_cap = 50.0f;
  config.final_cap = 30.0f;
  config.eos_id = 1;
  config.secondary_eos_id = 107;
  config.rope_theta = 10000.0f;
  config.global_rope_theta = 10000.0f;
  return config;
}

static LayerConfig LayerConfigT5GemmaS(size_t model_dim) {
  LayerConfig config;
  config.model_dim = model_dim;
  config.ff_hidden_dim = 1024;
  config.heads = 8;
  config.kv_heads = 8;
  config.qkv_dim = 64;
  config.optimized_gating = false;
  config.post_norm = PostNormType::Scale;
  return config;
}

static ModelConfig ConfigT5Gemma_S_S() {
  ModelConfig config = ConfigBaseT5Gemma();
  config.display_name = "T5Gemma_S_S";
  config.model = Model::T5GEMMA_S_S;
  config.wrapping = PromptWrapping::GEMMA_PT;
  config.model_dim = 512;
  config.vocab_size = kVocabSize;
  config.max_seq_len = 8192;
  LayerConfig layer_config = LayerConfigT5GemmaS(config.model_dim);
  config.is_encoder_decoder = true;
  config.encoder_num_layers = 8;
  config.encoder_layer_configs = {config.encoder_num_layers, layer_config};
  config.encoder_attention_window_sizes =
      RepeatedAttentionWindowSizes<8, 2>({4096, config.max_seq_len});
  config.decoder_num_layers = 8;
  config.decoder_layer_configs = {config.decoder_num_layers, layer_config};
  config.decoder_attention_window_sizes =
      RepeatedAttentionWindowSizes<8, 2>({4096, config.max_seq_len});

  // TODO: Update users of `layer_configs` to route encoder-decoder models
  // through the explicit encoder/decoder stacks above.
  config.num_layers = 8;
  config.layer_configs = {config.num_layers, layer_config};
  config.query_scale = QueryScaleType::SqrtKeySize;
  config.attention_window_sizes =
      RepeatedAttentionWindowSizes<8, 2>({4096, config.max_seq_len});
  return config;
}

static ModelConfig ConfigBaseQwen3() {
  ModelConfig config = ConfigNoSSM();
  config.vocab_size = 151936;
  config.max_seq_len = 32768;
  config.eos_id = 151645;
  config.secondary_eos_id = 151643;
  return config;
}

static LayerConfig LayerConfigQwen3(size_t model_dim, size_t ff_hidden_dim,
                                    size_t heads, size_t kv_heads,
                                    size_t qkv_dim) {
  LayerConfig config;
  config.model_dim = model_dim;
  config.ff_hidden_dim = ff_hidden_dim;
  config.heads = heads;
  config.kv_heads = kv_heads;
  config.qkv_dim = qkv_dim;
  config.optimized_gating = false;
  config.post_norm = PostNormType::None;
  config.activation = ActivationType::Silu;
  config.use_qk_norm = true;
  return config;
}

static ModelConfig ConfigQwen3_600M() {
  ModelConfig config = ConfigBaseQwen3();
  config.display_name = "Qwen3_0.6B";
  config.model = Model::QWEN3_600M;
  config.wrapping = PromptWrapping::GEMMA_IT;
  config.rope_theta = 1000000.0f;
  config.global_rope_theta = 1000000.0f;
  config.model_dim = 1024;

  LayerConfig layer_config =
      LayerConfigQwen3(config.model_dim, 3072, 16, 8, 128);
  config.num_layers = 28;
  config.layer_configs = {config.num_layers, layer_config};
  config.query_scale = QueryScaleType::SqrtKeySize;
  config.use_global_timescale = true;
  config.attention_window_sizes =
      FixedAttentionWindowSizes<28>(config.max_seq_len);
  return config;
}

static ModelConfig ConfigQwen3_2B() {
  ModelConfig config = ConfigBaseQwen3();
  config.display_name = "Qwen3_1.7B";
  config.model = Model::QWEN3_2B;
  config.wrapping = PromptWrapping::GEMMA_IT;
  config.rope_theta = 1000000.0f;
  config.global_rope_theta = 1000000.0f;
  config.model_dim = 2048;

  LayerConfig layer_config =
      LayerConfigQwen3(config.model_dim, 6144, 16, 8, 128);
  config.num_layers = 28;
  config.layer_configs = {config.num_layers, layer_config};
  config.query_scale = QueryScaleType::SqrtKeySize;
  config.use_global_timescale = true;
  config.attention_window_sizes =
      FixedAttentionWindowSizes<28>(config.max_seq_len);
  return config;
}

static ModelConfig ConfigQwen3_4B() {
  ModelConfig config = ConfigBaseQwen3();
  config.display_name = "Qwen3_4B";
  config.model = Model::QWEN3_4B;
  config.wrapping = PromptWrapping::GEMMA_IT;
  config.rope_theta = 1000000.0f;
  config.global_rope_theta = 1000000.0f;
  config.model_dim = 2560;

  LayerConfig layer_config =
      LayerConfigQwen3(config.model_dim, 9728, 32, 8, 128);
  config.num_layers = 36;
  config.layer_configs = {config.num_layers, layer_config};
  config.query_scale = QueryScaleType::SqrtKeySize;
  config.use_global_timescale = true;
  config.attention_window_sizes =
      FixedAttentionWindowSizes<36>(config.max_seq_len);
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
    case Model::GEMMA3_4B_LM:
      return ConfigGemma3_4B_LM();
    case Model::GEMMA3_12B_LM:
      return ConfigGemma3_12B_LM();
    case Model::GEMMA3_27B_LM:
      return ConfigGemma3_27B_LM();
    case Model::GEMMA4_26B_MOE:
      return ConfigGemma4_26B_MoE();
    case Model::GEMMA4_2B:
      return ConfigGemma4_2B();
    case Model::DEEPSEEK4_FLASH:
      return ConfigDeepSeek4_Flash();
    case Model::T5GEMMA_S_S:
      return ConfigT5Gemma_S_S();
    case Model::QWEN3_600M:
      return ConfigQwen3_600M();
    case Model::QWEN3_2B:
      return ConfigQwen3_2B();
    case Model::QWEN3_4B:
      return ConfigQwen3_4B();
    case Model::GEMMA4_2B_LM:
      return ConfigGemma4_2B_LM();
    case Model::GEMMA4_E4B:
      return ConfigGemma4_E4B();
    case Model::GEMMA4_E4B_LM:
      return ConfigGemma4_E4B_LM();
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
    case Model::GEMMA3_4B_LM:
      return "gemma3-4b-lm";
    case Model::GEMMA3_12B_LM:
      return "gemma3-12b-lm";
    case Model::GEMMA3_27B_LM:
      return "gemma3-27b-lm";
    case Model::GEMMA4_26B_MOE:
      return "gemma4-26b-moe";
    case Model::GEMMA4_2B:
      return "gemma4-2b";
    case Model::DEEPSEEK4_FLASH:
      return "deepseek4-flash";
    case Model::T5GEMMA_S_S:
      return "t5gemma-s-s";
    case Model::QWEN3_600M:
      return "qwen3-0_6b";
    case Model::QWEN3_2B:
      return "qwen3-2b";
    case Model::QWEN3_4B:
      return "qwen3-4b";
    case Model::GEMMA4_2B_LM:
      return "gemma4-2b-lm";
    case Model::GEMMA4_E4B:
      return "gemma4-e4b";
    case Model::GEMMA4_E4B_LM:
      return "gemma4-e4b-lm";
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

PromptWrapping ChooseWrapping(const Model model, Tristate wrapping) {
  const PromptWrapping config_wrapping = ConfigFromModel(model).wrapping;

  // For models with a fixed wrapping mode, ignore user override.
  if (IsVlmWrapping(config_wrapping)) {
    if (wrapping != Tristate::kDefault) {
      HWY_WARN("Ignoring unnecessary --wrapping for model %s.",
               ModelPrefix(model));
    }
    return config_wrapping;
  }

  // For other models, default to IT unless --wrapping=0 is passed.
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
  if (!IsVlmWrapping(this->wrapping)) {
    this->wrapping = wrapping;
  }
}

static Model FindModel(const std::string& specifier) {
  // Some model prefixes are prefixes of other prefixes (e.g. `gemma3-4b-` is a
  // prefix of `gemma3-4b-lm-`). Pick the longest matching prefix so the more
  // specific model wins.
  Model found_model = Model::UNKNOWN;
  size_t longest_match = 0;
  ForEachModel([&](Model model) {
    const std::string prefix = std::string(ModelPrefix(model)) + "-";
    if (specifier.rfind(prefix, 0) == 0 && prefix.size() > longest_match) {
      found_model = model;
      longest_match = prefix.size();
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

  if (!IsVlmWrapping(wrapping)) {
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
  // Order matters: overwrite `b` with `a` because that is the known-good config
  // when called by `OverwriteWithCanonical`.
  b.display_name = a.display_name;
  b.model = a.model;

  // The following are not yet set by config_converter.py, so we here ignore
  // them for purposes of comparison, and there overwrite the converter's config
  // with the canonical ModelConfig constructed via (deduced) enum, so that
  // these fields will be set.
  // `vit_config` is also not yet set, but we must not ignore it because
  // otherwise PaliGemma models will be indistinguishable for `configs_test`.
  b.pool_dim = a.pool_dim;  // ViT
  b.eos_id = a.eos_id;
  b.secondary_eos_id = a.secondary_eos_id;
  b.scale_base_names = a.scale_base_names;
  b.rope_theta = a.rope_theta;
  b.global_rope_theta = a.global_rope_theta;
  for (size_t i = 0; i < b.layer_configs.size(); ++i) {
    b.layer_configs[i].optimized_gating = a.layer_configs[i].optimized_gating;
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

static inline bool StrContains(std::string_view str, std::string_view substr) {
  return str.find(substr) != std::string_view::npos;  // NOLINT
}

Model DeduceModel(const Path& blob_path, size_t layers, int layer_types) {
  switch (layers) {
    case 8:
      if (layer_types & kDeducedT5Gemma) {
        return Model::T5GEMMA_S_S;
      }
      // Unknown 8-layer model.
      break;

    case 18:
      return Model::GEMMA3_270M;

    case 26:
      if (layer_types & (kDeducedViT | kDeducedKqNorm)) {
        return Model::GEMMA3_1B;
      }
      return Model::GEMMA2_2B;
    case 27:
      return (layer_types & kDeduced448) ? Model::PALIGEMMA2_3B_448
                                         : Model::PALIGEMMA2_3B_224;
    case 28:
      if (StrContains(blob_path.path, "qwen3-2b") ||
          StrContains(blob_path.path, "qwen3-1_7b")) {
        return Model::QWEN3_2B;
      }
      return Model::QWEN3_600M;

    case 30:
      return Model::GEMMA4_26B_MOE;

    case 34:
      return (layer_types & kDeducedViT) ? Model::GEMMA3_4B
                                         : Model::GEMMA3_4B_LM;
    case 35:
      return (layer_types & kDeducedViT) ? Model::GEMMA4_2B
                                         : Model::GEMMA4_2B_LM;
    case 36:
      return Model::QWEN3_4B;
    case 42:
      if (layer_types & kDeducedViT) {
        if (StrContains(blob_path.path, "gemma4") ||
            StrContains(blob_path.path, "e4b")) {
          return Model::GEMMA4_E4B;
        }
        return (layer_types & kDeduced448) ? Model::PALIGEMMA2_10B_448
                                           : Model::PALIGEMMA2_10B_224;
      }
      if (StrContains(blob_path.path, "gemma4") ||
          StrContains(blob_path.path, "e4b")) {
        return Model::GEMMA4_E4B_LM;
      }
      return Model::GEMMA2_9B;
    case 46:
      return Model::GEMMA2_27B;
    case 48:
      return (layer_types & kDeducedViT) ? Model::GEMMA3_12B
                                         : Model::GEMMA3_12B_LM;
    case 62:
      return (layer_types & kDeducedViT) ? Model::GEMMA3_27B
                                         : Model::GEMMA3_27B_LM;
    // TODO: detect these.
    /*
    return Model::GEMMA2_772M;
    return Model::PALIGEMMA2_772M_224;
    */
    default:
      break;
  }
  HWY_WARN("Failed to deduce model type from %s, layer count %zu types %x.",
           blob_path.path.c_str(), layers, layer_types);
  return Model::UNKNOWN;
}

// NOTE: keep the `--attention_impl` help text in `gemma_args.h` synced
constexpr std::pair<const char*, AttentionImpl> kAttentionImplNameToEnum[] = {
    {"flash", AttentionImpl::kFlash},
    {"flash_transposed_qs", AttentionImpl::kFlashTransposedQs},
    {"flash_transposed_qs_bf16", AttentionImpl::kFlashTransposedQsBF16},
    {"flash_transposed_qs_int16", AttentionImpl::kFlashTransposedQsInt16},
    {"flash_transposed_qs_int8", AttentionImpl::kFlashTransposedQsInt8},
    {"flash_matrix_accumulation", AttentionImpl::kFlashMatrixAccumulation},
    {"int8_matrix_accumulation", AttentionImpl::kInt8MatrixAccumulation},
};

std::string GetAttentionImplName(AttentionImpl impl) {
  for (const auto& [name, attention_impl] : kAttentionImplNameToEnum) {
    if (attention_impl == impl) return std::string(name);
  }
  return "unknown";
}

AttentionImpl GetAttentionImpl(const std::string& impl_name) {
  for (const auto& [name, attention_impl] : kAttentionImplNameToEnum) {
    if (name == impl_name) return attention_impl;
  }
  std::string valid;
  for (const auto& [name, attention_impl] : kAttentionImplNameToEnum) {
    if (!valid.empty()) {
      valid += ", ";
    }
    valid += name;
  }
  HWY_WARN("Unknown attention implementation: %s. Valid: %s. Using kFlash.\n",
           impl_name.c_str(), valid.c_str());
  return AttentionImpl::kFlash;
}

std::string KVEncodingToString(KVEncoding encoding) {
  switch (encoding) {
    case KVEncoding::kF32:
      return "F32";
    case KVEncoding::kBF16:
      return "BF16";
    case KVEncoding::kF32TwoTranspositions:
      return "F32TwoTranspositions";
    case KVEncoding::kBF16TwoTranspositions:
      return "BF16TwoTranspositions";
    case KVEncoding::kInt8:
      return "Int8";
    case KVEncoding::kInt8TwoTranspositions:
      return "Int8TwoTranspositions";
    case KVEncoding::kInt8VNNITwoTranspositions:
      return "Int8VNNITwoTranspositions";
    case KVEncoding::kBF16MatrixAccumulation:
      return "BF16MatrixAccumulation";
    case KVEncoding::kInt8MatrixAccumulation:
      return "Int8MatrixAccumulation";
    default:
      return "Unknown";
  }
}

}  // namespace gcpp
