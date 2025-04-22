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

#include <cstddef>
#include <iostream>

#include "hwy/base.h"

namespace gcpp {

static ModelConfig ConfigNoSSM() {
  ModelConfig config;
  config.scale_names = {"att_ein",      "qkv_ein",   "gr_lin_x_w", "gr_lin_y_w",
                        "gr_lin_out_w", "gr_gate_w", "gating_ein", "linear_w"};
  return config;
}

static ModelConfig ConfigBaseGemmaV1() { return ConfigNoSSM(); }

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
  config.model_name = "Gemma2_27B";
  config.model = Model::GEMMA2_27B;
  config.model_dim = 4608;
  config.vocab_size = kVocabSize;
  config.seq_len = 8192;
  LayerConfig layer_config = LayerConfigGemma2_27B(config.model_dim);
  config.layer_configs = {46, layer_config};
  config.num_tensor_scales = 4 * config.layer_configs.size();
  config.query_scale = QueryScaleType::SqrtModelDimDivNumHeads;
  config.attention_window_sizes =
      RepeatedAttentionWindowSizes<46, 2>({4096, 8192});
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
  config.model_name = "Gemma2_9B";
  config.model = Model::GEMMA2_9B;
  config.model_dim = 3584;
  config.vocab_size = kVocabSize;
  config.seq_len = 8192;
  LayerConfig layer_config = LayerConfigGemma2_9B(config.model_dim);
  config.layer_configs = {42, layer_config};
  config.num_tensor_scales = 4 * config.layer_configs.size();
  config.query_scale = QueryScaleType::SqrtKeySize;
  config.attention_window_sizes =
      RepeatedAttentionWindowSizes<42, 2>({4096, 8192});
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
  config.model_name = "Gemma2_2B";
  config.model = Model::GEMMA2_2B;
  config.model_dim = 2304;
  config.vocab_size = kVocabSize;
  config.seq_len = 8192;
  LayerConfig layer_config = LayerConfigGemma2_2B(config.model_dim);
  config.layer_configs = {26, layer_config};
  config.num_tensor_scales = 4 * config.layer_configs.size();
  config.query_scale = QueryScaleType::SqrtKeySize;
  config.attention_window_sizes =
      RepeatedAttentionWindowSizes<26, 2>({4096, 8192});
  return config;
}

static LayerConfig LayerConfigGemma7B(size_t model_dim) {
  LayerConfig config;
  config.model_dim = model_dim;
  config.ff_hidden_dim = 16 * 3072 / 2;  // = 24576
  config.heads = 16;
  config.kv_heads = 16;
  config.qkv_dim = 256;
  return config;
}

static ModelConfig ConfigGemma7B() {
  ModelConfig config = ConfigBaseGemmaV1();
  config.model_name = "Gemma7B";
  config.model = Model::GEMMA_7B;
  config.model_dim = 3072;
  config.vocab_size = kVocabSize;
  config.seq_len = kSeqLen;
  LayerConfig layer_config = LayerConfigGemma7B(config.model_dim);
  config.layer_configs = {28, layer_config};
  config.num_tensor_scales = 4 * config.layer_configs.size();
  config.query_scale = QueryScaleType::SqrtKeySize;
  config.attention_window_sizes = FixedAttentionWindowSizes<28>(kSeqLen);
  return config;
}

static LayerConfig LayerConfigGemma2B(size_t model_dim) {
  LayerConfig config;
  config.model_dim = model_dim;
  config.ff_hidden_dim = 16 * 2048 / 2;  // = 16384
  config.heads = 8;
  config.kv_heads = 1;
  config.qkv_dim = 256;
  return config;
}

static ModelConfig ConfigGemma2B() {
  ModelConfig config = ConfigBaseGemmaV1();
  config.model_name = "Gemma2B";
  config.model = Model::GEMMA_2B;
  config.model_dim = 2048;
  config.vocab_size = kVocabSize;
  config.seq_len = kSeqLen;
  LayerConfig layer_config = LayerConfigGemma2B(config.model_dim);
  config.layer_configs = {18, layer_config};
  config.num_tensor_scales = 4 * config.layer_configs.size();
  config.attention_window_sizes = FixedAttentionWindowSizes<18>(kSeqLen);
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
  config.model_name = "GemmaTiny";
  config.model = Model::GEMMA_TINY;
  config.wrapping = PromptWrapping::GEMMA_IT;
  config.model_dim = 128;
  config.vocab_size = 64;
  config.seq_len = 32;
  LayerConfig layer_config = LayerConfigGemmaTiny(config.model_dim);
  config.layer_configs = {3, layer_config};
  config.num_tensor_scales = 4 * config.layer_configs.size();
  config.query_scale = QueryScaleType::SqrtKeySize;
  config.attention_window_sizes = FixedAttentionWindowSizes<3>(32);
  // This is required for optimize_test to pass.
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
  config.model_name = "Griffin2B";
  config.model = Model::GRIFFIN_2B;
  // Griffin uses local attention, so kSeqLen is actually the local attention
  // window.
  config.model_dim = 2560;
  config.vocab_size = kVocabSize;
  config.seq_len = 2048;
  LayerConfig layer_config = LayerConfigGriffin2B(config.model_dim);
  config.layer_configs = {26, layer_config};
  for (size_t i = 2; i < config.layer_configs.size(); i += 3) {
    config.layer_configs[i].type = LayerAttentionType::kGemma;
    config.layer_configs[i].griffin_dim = 0;
  }
  config.num_tensor_scales = 140;
  config.attention_window_sizes = FixedAttentionWindowSizes<26>(config.seq_len);
  config.use_local_attention = true;
  // This is required for optimize_test to pass.
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

static ModelConfig ConfigPaliGemma_224() {
  ModelConfig config = ConfigGemma2B();
  config.model_name = "PaliGemma_224";
  config.model = Model::PALIGEMMA_224;
  config.wrapping = PromptWrapping::PALIGEMMA;
  AddVitConfig(config);
  return config;
}

static ModelConfig ConfigPaliGemma_448() {
  ModelConfig config = ConfigGemma2B();
  config.model_name = "PaliGemma_448";
  config.model = Model::PALIGEMMA_448;
  config.wrapping = PromptWrapping::PALIGEMMA;
  AddVitConfig(config, /*image_size=*/448);
  return config;
}

ModelConfig GetVitConfig(const ModelConfig& config) {
  ModelConfig vit_config = ConfigNoSSM();
  vit_config.model_dim = config.vit_config.model_dim;
  vit_config.seq_len = config.vit_config.seq_len;
  vit_config.layer_configs = config.vit_config.layer_configs;
  vit_config.pool_dim = config.vit_config.pool_dim;
  vit_config.wrapping = config.wrapping;
  // The Vit part does not have a vocabulary, the image patches are embedded.
  vit_config.vocab_size = 0;
  return vit_config;
}

static ModelConfig ConfigPaliGemma2_3B_224() {
  ModelConfig config = ConfigGemma2_2B();
  config.model_name = "PaliGemma2_3B_224";
  config.model = Model::PALIGEMMA2_3B_224;
  config.wrapping = PromptWrapping::PALIGEMMA;
  AddVitConfig(config);
  return config;
}

static ModelConfig ConfigPaliGemma2_3B_448() {
  ModelConfig config = ConfigGemma2_2B();
  config.model_name = "PaliGemma2_3B_448";
  config.model = Model::PALIGEMMA2_3B_448;
  config.wrapping = PromptWrapping::PALIGEMMA;
  AddVitConfig(config, /*image_size=*/448);
  return config;
}

static ModelConfig ConfigPaliGemma2_10B_224() {
  ModelConfig config = ConfigGemma2_9B();
  config.model_name = "PaliGemma2_10B_224";
  config.model = Model::PALIGEMMA2_10B_224;
  config.wrapping = PromptWrapping::PALIGEMMA;
  AddVitConfig(config);
  return config;
}

static ModelConfig ConfigPaliGemma2_10B_448() {
  ModelConfig config = ConfigGemma2_9B();
  config.model_name = "PaliGemma2_10B_448";
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
  config.model_name = "Gemma3_1B";
  config.model = Model::GEMMA3_1B;
  config.wrapping = PromptWrapping::GEMMA_VLM;
  config.model_dim = 1152;
  config.vocab_size = 262144;  // new vocab size / tokenizer
  config.seq_len = 32 * 1024;
  LayerConfig layer_config = LayerConfigGemma3_1B_LM(config.model_dim);
  config.layer_configs = {26, layer_config};
  config.num_tensor_scales = 4 * config.layer_configs.size();
  config.query_scale = QueryScaleType::SqrtKeySize;
  // interleaved local / global attention
  config.attention_window_sizes = RepeatedAttentionWindowSizes<26, 6>(
      {512, 512, 512, 512, 512, config.seq_len});
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
  config.model_name = "Gemma3_4B";
  config.model = Model::GEMMA3_4B;
  config.wrapping = PromptWrapping::GEMMA_VLM;
  config.model_dim = 2560;
  config.vocab_size = 262144;  // new vocab size / tokenizer
  config.seq_len = 32 * 1024;
  LayerConfig layer_config = LayerConfigGemma3_4B_LM(config.model_dim);
  config.layer_configs = {34, layer_config};
  config.num_tensor_scales = 4 * config.layer_configs.size();
  config.query_scale = QueryScaleType::SqrtKeySize;
  // interleaved local / global attention
  config.attention_window_sizes = RepeatedAttentionWindowSizes<34, 6>(
      {1024, 1024, 1024, 1024, 1024, config.seq_len});
  return config;
}

static ModelConfig ConfigGemma3_4B() {
  ModelConfig config = ConfigGemma3_4B_LM();
  config.model_name = "Gemma3_4B";
  config.model = Model::GEMMA3_4B;
  config.wrapping = PromptWrapping::GEMMA_VLM;
  AddVitConfig(config, /*image_size=*/896);
  config.vocab_size = 262144;
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
  config.model_name = "Gemma3_12B";
  config.model = Model::GEMMA3_12B;
  config.wrapping = PromptWrapping::GEMMA_VLM;
  config.model_dim = 3840;
  config.vocab_size = 262144;  // new vocab size / tokenizer
  config.seq_len = 32 * 1024;
  LayerConfig layer_config = LayerConfigGemma3_12B_LM(config.model_dim);
  config.layer_configs = {48, layer_config};
  config.num_tensor_scales = 4 * config.layer_configs.size();
  config.query_scale = QueryScaleType::SqrtKeySize;
  // interleaved local / global attention
  config.attention_window_sizes = RepeatedAttentionWindowSizes<48, 6>(
      {1024, 1024, 1024, 1024, 1024, config.seq_len});
  return config;
}

static ModelConfig ConfigGemma3_12B() {
  ModelConfig config = ConfigGemma3_12B_LM();
  config.model_name = "Gemma3_12B";
  config.model = Model::GEMMA3_12B;
  config.wrapping = PromptWrapping::GEMMA_VLM;
  AddVitConfig(config, /*image_size=*/896);
  config.vocab_size = 262144;
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
  config.model_name = "Gemma3_27B";
  config.model = Model::GEMMA3_27B;
  config.wrapping = PromptWrapping::GEMMA_VLM;
  config.model_dim = 5376;
  config.vocab_size = 262144;  // new vocab size / tokenizer
  config.seq_len = 32 * 1024;
  LayerConfig layer_config = LayerConfigGemma3_27B_LM(config.model_dim);
  config.layer_configs = {62, layer_config};
  config.num_tensor_scales = 4 * config.layer_configs.size();
  config.query_scale = QueryScaleType::SqrtKeySize;
  // interleaved local / global attention
  config.attention_window_sizes = RepeatedAttentionWindowSizes<62, 6>(
      {1024, 1024, 1024, 1024, 1024, config.seq_len});
  return config;
}

static ModelConfig ConfigGemma3_27B() {
  ModelConfig config = ConfigGemma3_27B_LM();
  config.model_name = "Gemma3_27B";
  config.model = Model::GEMMA3_27B;
  config.wrapping = PromptWrapping::GEMMA_VLM;
  AddVitConfig(config, /*image_size=*/896);
  config.vocab_size = 262144;
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

ModelConfig ConfigFromModel(Model model) {
  switch (model) {
    case Model::GEMMA_2B:
      return ConfigGemma2B();
    case Model::GEMMA_7B:
      return ConfigGemma7B();
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
    case Model::PALIGEMMA_224:
      return ConfigPaliGemma_224();
    case Model::PALIGEMMA_448:
      return ConfigPaliGemma_448();
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
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

#define TEST_EQUAL(a, b)                                               \
  if (a != b) {                                                        \
    if (debug)                                                         \
      std::cerr << #a << "=" << a << " != " << #b << "=" << b << "\n"; \
    result = false;                                                    \
  }

#define RETURN_IF_NOT_EQUAL(a, b)                                      \
  if (a != b) {                                                        \
    if (debug)                                                         \
      std::cerr << #a << "=" << a << " != " << #b << "=" << b << "\n"; \
    return false;                                                      \
  }

#define WARN_IF_NOT_EQUAL(a, b)                                        \
  if (a != b) {                                                        \
    std::cerr << #a << "=" << a << " != " << #b << "=" << b << "\n";   \
  }

bool LayerConfig::TestEqual(const LayerConfig& other, bool partial,
                            bool debug) const {
  bool result = true;
  // Optimized gating may not be set correctly in the c++ configs.
  if (debug) {
    WARN_IF_NOT_EQUAL(optimized_gating, other.optimized_gating)
  }
  TEST_EQUAL(model_dim, other.model_dim);
  TEST_EQUAL(griffin_dim, other.griffin_dim);
  TEST_EQUAL(ff_hidden_dim, other.ff_hidden_dim);
  TEST_EQUAL(heads, other.heads);
  TEST_EQUAL(kv_heads, other.kv_heads);
  TEST_EQUAL(qkv_dim, other.qkv_dim);
  TEST_EQUAL(conv1d_width, other.conv1d_width);
  if (!partial) {
    TEST_EQUAL(ff_biases, other.ff_biases);
    TEST_EQUAL(softmax_attn_output_biases, other.softmax_attn_output_biases);
  }
  TEST_EQUAL(static_cast<int>(post_norm), static_cast<int>(other.post_norm));
  TEST_EQUAL(static_cast<int>(type), static_cast<int>(other.type));
  TEST_EQUAL(static_cast<int>(activation), static_cast<int>(other.activation));
  TEST_EQUAL(static_cast<int>(post_qk), static_cast<int>(other.post_qk));
  return result;
}

bool VitConfig::TestEqual(const VitConfig& other, bool partial,
                          bool debug) const {
  bool result = true;
  TEST_EQUAL(model_dim, other.model_dim);
  TEST_EQUAL(seq_len, other.seq_len);
  if (!partial) {
    TEST_EQUAL(num_scales, other.num_scales);
  }
  TEST_EQUAL(patch_width, other.patch_width);
  TEST_EQUAL(image_size, other.image_size);
  RETURN_IF_NOT_EQUAL(layer_configs.size(), other.layer_configs.size());
  for (size_t i = 0; i < layer_configs.size(); ++i) {
    result &=
        layer_configs[i].TestEqual(other.layer_configs[i], partial, debug);
  }
  return result;
}

bool ModelConfig::TestEqual(const ModelConfig& other, bool partial,
                            bool debug) const {
  bool result = true;
  TEST_EQUAL(model_family_version, other.model_family_version);
  // We don't care about model_name, model, wrapping, or weight being different,
  // but will output in debug mode if they are.
  if (debug) {
    WARN_IF_NOT_EQUAL(model_name, other.model_name);
    WARN_IF_NOT_EQUAL(static_cast<int>(model), static_cast<int>(other.model));
    WARN_IF_NOT_EQUAL(static_cast<int>(wrapping),
                      static_cast<int>(other.wrapping));
    WARN_IF_NOT_EQUAL(static_cast<int>(weight), static_cast<int>(other.weight));
  }
  TEST_EQUAL(model_dim, other.model_dim);
  TEST_EQUAL(vocab_size, other.vocab_size);
  TEST_EQUAL(seq_len, other.seq_len);
  if (!partial) {
    TEST_EQUAL(num_tensor_scales, other.num_tensor_scales);
  }
  TEST_EQUAL(att_cap, other.att_cap);
  TEST_EQUAL(final_cap, other.final_cap);
  TEST_EQUAL(absolute_pe, other.absolute_pe);
  TEST_EQUAL(use_local_attention, other.use_local_attention);
  TEST_EQUAL(static_cast<int>(query_scale),
             static_cast<int>(other.query_scale));
  RETURN_IF_NOT_EQUAL(layer_configs.size(), other.layer_configs.size());
  for (size_t i = 0; i < layer_configs.size(); ++i) {
    result &=
        layer_configs[i].TestEqual(other.layer_configs[i], partial, debug);
  }
  RETURN_IF_NOT_EQUAL(attention_window_sizes.size(),
                     other.attention_window_sizes.size());
  for (size_t i = 0; i < attention_window_sizes.size(); ++i) {
    TEST_EQUAL(attention_window_sizes[i], other.attention_window_sizes[i]);
  }
  if (!partial) {
    if (scale_names != other.scale_names) {
      result = false;
      if (debug) {
        std::cerr << "scale_names mismatch\n";
      }
    }
  }
  TEST_EQUAL(norm_num_groups, other.norm_num_groups);
  result &= vit_config.TestEqual(other.vit_config, partial, debug);
  return result;
}

Model ModelFromConfig(const ModelConfig& config) {
  for (Model model : kAllModels) {
    ModelConfig model_config = ConfigFromModel(model);
    if (config.TestEqual(model_config, /*partial=*/true, /*debug=*/false)) {
      return model;
    }
  }
  return Model::UNKNOWN;
}

}  // namespace gcpp
