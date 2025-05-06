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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_ACTIVATIONS_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_ACTIVATIONS_H_

#include <stddef.h>

#include "gemma/configs.h"   // ModelConfig
#include "ops/matmul.h"      // MatMulEnv
#include "ops/ops.h"         // CreateInvTimescale
#include "util/allocator.h"  // Allocator
#include "util/basics.h"     // BF16
#include "util/mat.h"        // RowVectorBatch

namespace gcpp {

struct Activations {
  explicit Activations(const ModelConfig& config)
      : weights_config(config),
        layer_config(config.layer_configs[0]),
        seq_len(config.seq_len),
        cache_pos_size(config.CachePosSize()) {}

  RowVectorBatch<float> x;  // input
  RowVectorBatch<float> q;  // query, also KV if MHA.
  RowVectorBatch<float> logits;

  // Attention
  RowVectorBatch<float> pre_att_rms_out;
  RowVectorBatch<float> att;      // attention vector
  RowVectorBatch<float> att_out;  // attention output
  // Accumulation of attention outputs over heads
  RowVectorBatch<float> att_sums;

  // Gated FFW
  RowVectorBatch<BF16> bf_pre_ffw_rms_out;
  RowVectorBatch<float> C1;
  RowVectorBatch<float> C2;
  RowVectorBatch<float> ffw_out;

  // Griffin
  RowVectorBatch<float> griffin_x;
  RowVectorBatch<float> griffin_y;
  RowVectorBatch<float> griffin_gate_x;
  RowVectorBatch<float> griffin_multiplier;

  // Rope
  RowVectorBatch<float> inv_timescale;
  RowVectorBatch<float> inv_timescale_global;

  // Dynamic because no default ctor and only initialized in `Allocate`.
  MatMulEnv* env;

  PostQKType post_qk = PostQKType::Rope;
  // And the config.
  const ModelConfig& weights_config;
  const LayerConfig& layer_config;
  size_t seq_len;
  size_t cache_pos_size = 0;

  void Allocate(size_t batch_size, MatMulEnv* env) {
    const Allocator& allocator = env->ctx.allocator;

    post_qk = layer_config.post_qk;
    const size_t model_dim = weights_config.model_dim;
    const size_t ff_hidden_dim = layer_config.ff_hidden_dim;
    const size_t vocab_size = weights_config.vocab_size;
    const size_t qkv_dim = layer_config.qkv_dim;
    const size_t heads = layer_config.heads;

    x = RowVectorBatch<float>(allocator, Extents2D(batch_size, model_dim));
    q = RowVectorBatch<float>(
        allocator, Extents2D(batch_size, heads * layer_config.QStride()));
    if (vocab_size > 0) {
      logits =
          RowVectorBatch<float>(allocator, Extents2D(batch_size, vocab_size));
    }

    pre_att_rms_out =
        RowVectorBatch<float>(allocator, Extents2D(batch_size, model_dim));
    att = RowVectorBatch<float>(
        allocator, Extents2D(batch_size, heads * weights_config.seq_len));
    att_out = RowVectorBatch<float>(allocator,
                                    Extents2D(batch_size, heads * qkv_dim));
    att_sums =
        RowVectorBatch<float>(allocator, Extents2D(batch_size, model_dim));

    bf_pre_ffw_rms_out =
        RowVectorBatch<BF16>(allocator, Extents2D(batch_size, model_dim));
    C1 = RowVectorBatch<float>(allocator, Extents2D(batch_size, ff_hidden_dim));
    C2 = RowVectorBatch<float>(allocator, Extents2D(batch_size, ff_hidden_dim));
    ffw_out =
        RowVectorBatch<float>(allocator, Extents2D(batch_size, model_dim));

    if (layer_config.type == LayerAttentionType::kGriffinRecurrentBlock) {
      griffin_x =
          RowVectorBatch<float>(allocator, Extents2D(batch_size, model_dim));
      griffin_y =
          RowVectorBatch<float>(allocator, Extents2D(batch_size, model_dim));
      griffin_gate_x =
          RowVectorBatch<float>(allocator, Extents2D(batch_size, model_dim));
      griffin_multiplier =
          RowVectorBatch<float>(allocator, Extents2D(batch_size, model_dim));
    }

    inv_timescale = CreateInvTimescale(allocator, layer_config.qkv_dim,
                                       post_qk == PostQKType::HalfRope);
    inv_timescale_global = CreateInvTimescale(
        allocator, qkv_dim, post_qk == PostQKType::HalfRope, 1000000.0);

    this->env = env;
  }
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_ACTIVATIONS_H_
