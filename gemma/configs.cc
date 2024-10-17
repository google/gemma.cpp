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
      .type = LayerAttentionType::kGriffinRecurrentBlock,
      .activation = ActivationType::Gelu,
      .post_qk = PostQKType::Rope,
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

static ModelConfig ConfigPaliGemma_224() {
  ModelConfig config = ConfigGemma2B();
  config.model_name = "PaliGemma_224";
  config.model = Model::PALIGEMMA_224;
  config.vit_model_dim = 1152;
  config.vocab_size = 256000 + 1024 + 128;  // = 257152
  config.image_size = 224;
  config.patch_width = 14;
  const size_t num_patches = config.image_size / config.patch_width;
  config.vit_seq_len = num_patches * num_patches;
  LayerConfig layer_config = {
      .model_dim = config.vit_model_dim,
      .ff_hidden_dim = 4304,
      .heads = 16,
      .kv_heads = 16,
      .qkv_dim = 72,
      .ff_biases = true,
      .type = LayerAttentionType::kVit,
  };
  config.vit_layer_configs = {27, layer_config};
  config.num_vit_scales = 4 * config.vit_layer_configs.size();
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
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

}  // namespace gcpp
