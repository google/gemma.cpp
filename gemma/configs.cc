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

#include <iostream>

#include "hwy/base.h"

namespace gcpp {

static ModelConfig ConfigNoSSM() {
  ModelConfig config = {.scale_names = {"att_ein", "qkv_ein", "gr_lin_x_w",
                                        "gr_lin_y_w", "gr_lin_out_w",
                                        "gr_gate_w", "gating_ein", "linear_w"}};
  return config;
}

static ModelConfig ConfigBaseGemmaV1() { return ConfigNoSSM(); }

static ModelConfig ConfigBaseGemmaV2() {
  ModelConfig config = ConfigNoSSM();
  config.att_cap = 50.0f;
  config.final_cap = 30.0f;
  return config;
}

static ModelConfig ConfigGemma2_27B() {
  ModelConfig config = ConfigBaseGemmaV2();
  config.model_name = "Gemma2_27B";
  config.model = Model::GEMMA2_27B;
  config.model_dim = 4608;
  config.vocab_size = kVocabSize;
  config.seq_len = 8192;
  LayerConfig layer_config = {.model_dim = config.model_dim,
                              .ff_hidden_dim = 16 * 4608 / 2,  // = 36864
                              .heads = 32,
                              .kv_heads = 16,
                              .qkv_dim = 128,
                              .optimized_gating = false,
                              .post_norm = PostNormType::Scale};
  config.layer_configs = {46, layer_config};
  config.num_tensor_scales = 4 * config.layer_configs.size();
  config.query_scale = QueryScaleType::SqrtModelDimDivNumHeads;
  config.attention_window_sizes =
      RepeatedAttentionWindowSizes<46, 2>({4096, 8192});
  return config;
}

static ModelConfig ConfigGemma2_9B() {
  ModelConfig config = ConfigBaseGemmaV2();
  config.model_name = "Gemma2_9B";
  config.model = Model::GEMMA2_9B;
  config.model_dim = 3584;
  config.vocab_size = kVocabSize;
  config.seq_len = 8192;
  LayerConfig layer_config = {.model_dim = config.model_dim,
                              .ff_hidden_dim = 8 * 3584 / 2,  // = 14336
                              .heads = 16,
                              .kv_heads = 8,
                              .qkv_dim = 256,
                              .optimized_gating = false,
                              .post_norm = PostNormType::Scale};
  config.layer_configs = {42, layer_config};
  config.num_tensor_scales = 4 * config.layer_configs.size();
  config.query_scale = QueryScaleType::SqrtKeySize;
  config.attention_window_sizes =
      RepeatedAttentionWindowSizes<42, 2>({4096, 8192});
  return config;
}

static ModelConfig ConfigGemma2_2B() {
  ModelConfig config = ConfigBaseGemmaV2();
  config.model_name = "Gemma2_2B";
  config.model = Model::GEMMA2_2B;
  config.model_dim = 2304;
  config.vocab_size = kVocabSize;
  config.seq_len = 8192;
  LayerConfig layer_config = {.model_dim = config.model_dim,
                              .ff_hidden_dim = 8 * 2304 / 2,  // = 9216
                              .heads = 8,
                              .kv_heads = 4,
                              .qkv_dim = 256,
                              .optimized_gating = false,
                              .post_norm = PostNormType::Scale};
  config.layer_configs = {26, layer_config};
  config.num_tensor_scales = 4 * config.layer_configs.size();
  config.query_scale = QueryScaleType::SqrtKeySize;
  config.attention_window_sizes =
      RepeatedAttentionWindowSizes<26, 2>({4096, 8192});
  return config;
}

static ModelConfig ConfigGemma7B() {
  ModelConfig config = ConfigBaseGemmaV1();
  config.model_name = "Gemma7B";
  config.model = Model::GEMMA_7B;
  config.model_dim = 3072;
  config.vocab_size = kVocabSize;
  config.seq_len = kSeqLen;
  LayerConfig layer_config = {
      .model_dim = config.model_dim,
      .ff_hidden_dim = 16 * 3072 / 2,  // = 24576
      .heads = 16,
      .kv_heads = 16,
      .qkv_dim = 256,
  };
  config.layer_configs = {28, layer_config};
  config.num_tensor_scales = 4 * config.layer_configs.size();
  config.query_scale = QueryScaleType::SqrtKeySize;
  config.attention_window_sizes = FixedAttentionWindowSizes<28>(kSeqLen);
  return config;
}

static ModelConfig ConfigGemma2B() {
  ModelConfig config = ConfigBaseGemmaV1();
  config.model_name = "Gemma2B";
  config.model = Model::GEMMA_2B;
  config.model_dim = 2048;
  config.vocab_size = kVocabSize;
  config.seq_len = kSeqLen;
  LayerConfig layer_config = {
      .model_dim = config.model_dim,
      .ff_hidden_dim = 16 * 2048 / 2,  // = 16384
      .heads = 8,
      .kv_heads = 1,
      .qkv_dim = 256,
  };
  config.layer_configs = {18, layer_config};
  config.num_tensor_scales = 4 * config.layer_configs.size();
  config.attention_window_sizes = FixedAttentionWindowSizes<18>(kSeqLen);
  return config;
}

static ModelConfig ConfigGemmaTiny() {
  ModelConfig config = ConfigNoSSM();
  config.model_name = "GemmaTiny";
  config.model = Model::GEMMA_TINY;
  config.model_dim = 128;
  config.vocab_size = 64;
  config.seq_len = 32;
  LayerConfig layer_config = {
      .model_dim = config.model_dim,
      .ff_hidden_dim = 256,
      .heads = 4,
      .kv_heads = 1,
      .qkv_dim = 16,
  };
  config.layer_configs = {3, layer_config};
  config.num_tensor_scales = 4 * config.layer_configs.size();
  config.query_scale = QueryScaleType::SqrtKeySize;
  config.attention_window_sizes = FixedAttentionWindowSizes<3>(32);
  // This is required for optimize_test to pass.
  config.final_cap = 30.0f;
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
  LayerConfig layer_config = {
      .model_dim = config.model_dim,
      .griffin_dim = config.model_dim,
      .ff_hidden_dim = 7680,
      .heads = 10,
      .kv_heads = 1,
      .qkv_dim = 256,
      .conv1d_width = 4,
      .ff_biases = true,
      .softmax_attn_output_biases = true,
      .optimized_gating = false,
      .type = LayerAttentionType::kGriffinRecurrentBlock,
      .activation = ActivationType::Gelu,
      .post_qk = PostQKType::HalfRope,
  };
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

// Adds a ViT config (SigLIP SoViT ViT, used in PaliGemma) to the model config.
static void AddVitConfig(ModelConfig& config, size_t image_size = 224) {
  config.vit_model_dim = 1152;
  config.vocab_size = 256000 + 1024 + 128;  // = 257152
  config.image_size = image_size;
  config.patch_width = 14;
  for (auto& layer_config : config.layer_configs) {
    layer_config.optimized_gating = false;
  }
  const size_t num_patches = config.image_size / config.patch_width;
  config.vit_seq_len = num_patches * num_patches;
  LayerConfig vit_layer_config = {
      .model_dim = config.vit_model_dim,
      .ff_hidden_dim = 4304,
      .heads = 16,
      .kv_heads = 16,
      .qkv_dim = 72,
      .ff_biases = true,
      .type = LayerAttentionType::kVit,
  };
  config.vit_layer_configs = {27, vit_layer_config};
  config.num_vit_scales = 4 * config.vit_layer_configs.size();
}

static ModelConfig ConfigPaliGemma_224() {
  ModelConfig config = ConfigGemma2B();
  config.model_name = "PaliGemma_224";
  config.model = Model::PALIGEMMA_224;
  AddVitConfig(config);
  return config;
}

static ModelConfig ConfigPaliGemma_448() {
  ModelConfig config = ConfigGemma2B();
  config.model_name = "PaliGemma_448";
  config.model = Model::PALIGEMMA_448;
  AddVitConfig(config, /*image_size=*/448);
  return config;
}

ModelConfig VitConfig(const ModelConfig& config) {
  ModelConfig vit_config = ConfigNoSSM();
  vit_config.model_dim = config.vit_model_dim;
  vit_config.seq_len = config.vit_seq_len;
  vit_config.layer_configs = config.vit_layer_configs;
  // The Vit part does not have a vocabulary, the image patches are embedded.
  vit_config.vocab_size = 0;
  return vit_config;
}

static ModelConfig ConfigPaliGemma2_3B_224() {
  ModelConfig config = ConfigGemma2_2B();
  config.model_name = "PaliGemma2_3B_224";
  config.model = Model::PALIGEMMA2_3B_224;
  AddVitConfig(config);
  return config;
}

static ModelConfig ConfigPaliGemma2_3B_448() {
  ModelConfig config = ConfigGemma2_2B();
  config.model_name = "PaliGemma2_3B_448";
  config.model = Model::PALIGEMMA2_3B_448;
  AddVitConfig(config, /*image_size=*/448);
  return config;
}

static ModelConfig ConfigPaliGemma2_10B_224() {
  ModelConfig config = ConfigGemma2_9B();
  config.model_name = "PaliGemma2_10B_224";
  config.model = Model::PALIGEMMA2_10B_224;
  AddVitConfig(config);
  return config;
}

static ModelConfig ConfigPaliGemma2_10B_448() {
  ModelConfig config = ConfigGemma2_9B();
  config.model_name = "PaliGemma2_10B_448";
  config.model = Model::PALIGEMMA2_10B_448;
  AddVitConfig(config, /*image_size=*/448);
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

bool ModelConfig::TestEqual(const ModelConfig& other, bool partial,
                            bool debug) const {
  bool result = true;
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
  TEST_EQUAL(vit_model_dim, other.vit_model_dim);
  TEST_EQUAL(vocab_size, other.vocab_size);
  TEST_EQUAL(seq_len, other.seq_len);
  TEST_EQUAL(vit_seq_len, other.vit_seq_len);
  if (!partial) {
    TEST_EQUAL(num_tensor_scales, other.num_tensor_scales);
    TEST_EQUAL(num_vit_scales, other.num_vit_scales);
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
  RETURN_IF_NOT_EQUAL(vit_layer_configs.size(), other.vit_layer_configs.size());
  for (size_t i = 0; i < vit_layer_configs.size(); ++i) {
    result &= vit_layer_configs[i].TestEqual(other.vit_layer_configs[i],
                                             partial, debug);
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
  TEST_EQUAL(model_family_version, other.model_family_version);
  TEST_EQUAL(patch_width, other.patch_width);
  TEST_EQUAL(image_size, other.image_size);
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
