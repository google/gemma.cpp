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

#ifndef THIRD_PARTY_GEMMA_CPP_BACKPROP_ACTIVATIONS_H_
#define THIRD_PARTY_GEMMA_CPP_BACKPROP_ACTIVATIONS_H_

#include <stddef.h>

#include <vector>

#include "gemma/configs.h"  // ModelConfig
#include "util/mat.h"       // MatStorageT

namespace gcpp {

template <typename T>
struct ForwardLayer {
  ForwardLayer(const LayerConfig& config, size_t seq_len)
      : input(MakePacked<T>("input", seq_len, config.model_dim)),
        pre_att_rms_out(
            MakePacked<T>("pre_att_rms_out", seq_len, config.model_dim)),
        qkv(MakePacked<T>("qkv", seq_len * (config.heads + 2), config.qkv_dim)),
        att(MakePacked<T>("att", seq_len * config.heads, seq_len)),
        att_out(
            MakePacked<T>("att_out", seq_len * config.heads, config.qkv_dim)),
        att_post1(MakePacked<T>("att_post1", seq_len, config.model_dim)),
        attention_out(
            MakePacked<T>("attention_out", seq_len, config.model_dim)),
        bf_pre_ffw_rms_out(
            MakePacked<T>("bf_preFF_rms_out", seq_len, config.model_dim)),
        ffw_hidden(
            MakePacked<T>("ffw_hidden", seq_len, config.ff_hidden_dim * 2)),
        ffw_hidden_gated(
            MakePacked<T>("ffw_hidden_gated", seq_len, config.ff_hidden_dim)),
        layer_config(config) {}

  MatStorageT<T> input;
  MatStorageT<T> pre_att_rms_out;
  MatStorageT<T> qkv;
  MatStorageT<T> att;
  MatStorageT<T> att_out;
  MatStorageT<T> att_post1;
  MatStorageT<T> attention_out;
  MatStorageT<T> bf_pre_ffw_rms_out;
  MatStorageT<T> ffw_hidden;
  MatStorageT<T> ffw_hidden_gated;
  const LayerConfig& layer_config;
};

template <typename T>
struct ForwardPass {
  ForwardPass(const ModelConfig& config)
      : final_layer_output(
            MakePacked<T>("fin_layer_out", config.seq_len, config.model_dim)),
        final_norm_output(
            MakePacked<T>("fin_norm_out", config.seq_len, config.model_dim)),
        logits(MakePacked<T>("logits", config.seq_len, config.vocab_size)),
        probs(MakePacked<T>("probs", config.seq_len, config.vocab_size)),
        weights_config(config) {
    for (const auto& layer_config : config.layer_configs) {
      layers.emplace_back(layer_config, config.seq_len);
    }
  }

  std::vector<ForwardLayer<T>> layers;
  MatStorageT<T> final_layer_output;
  MatStorageT<T> final_norm_output;
  MatStorageT<T> logits;
  MatStorageT<T> probs;
  const ModelConfig& weights_config;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_BACKPROP_ACTIVATIONS_H_
