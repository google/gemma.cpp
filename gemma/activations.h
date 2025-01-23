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

#include <cmath>
#include <memory>  // std::unique_ptr

#include "compression/shared.h"  // BF16
#include "gemma/configs.h"
#include "ops/matmul.h"          // MatMulEnv
#include "util/allocator.h"      // RowVectorBatch
#include "util/threading.h"
#include "hwy/base.h"  // HWY_DASSERT
#include "hwy/contrib/thread_pool/thread_pool.h"

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

  // Dynamic because no default ctor and only initialized in `Allocate`.
  std::unique_ptr<MatMulEnv> env;

  PostQKType post_qk = PostQKType::Rope;
  // And the config.
  const ModelConfig& weights_config;
  const LayerConfig& layer_config;
  size_t seq_len;
  size_t cache_pos_size = 0;

  static RowVectorBatch<float> CreateInvTimescale(
      size_t qkv_dim, PostQKType post_qk, double base_frequency = 10000.0) {
    const size_t rope_dim =
        post_qk == PostQKType::HalfRope ? qkv_dim / 2 : qkv_dim;
    RowVectorBatch<float> inv_timescale(Extents2D(1, rope_dim / 2));
    for (size_t dim = 0; dim < rope_dim / 2; ++dim) {
      const double freq_exponents =
          static_cast<double>(2 * dim) / static_cast<double>(rope_dim);
      // Replacing with expf(ln(1E4) * freq_exponents) changes results
      // noticeably.
      inv_timescale.Batch(0)[dim] =
          static_cast<float>(1.0 / std::pow(base_frequency, freq_exponents));
    }
    return inv_timescale;
  }

  void Allocate(size_t batch_size, NestedPools& pools) {
    post_qk = layer_config.post_qk;
    const size_t model_dim = weights_config.model_dim;
    const size_t ff_hidden_dim = layer_config.ff_hidden_dim;
    const size_t vocab_size = weights_config.vocab_size;
    const size_t qkv_dim = layer_config.qkv_dim;
    const size_t heads = layer_config.heads;

    x = RowVectorBatch<float>(Extents2D(batch_size, model_dim));
    q = RowVectorBatch<float>(
        Extents2D(batch_size, heads * layer_config.QStride()));
    if (vocab_size > 0) {
      logits = RowVectorBatch<float>(Extents2D(batch_size, vocab_size));
    }

    pre_att_rms_out = RowVectorBatch<float>(Extents2D(batch_size, model_dim));
    att = RowVectorBatch<float>(
        Extents2D(batch_size, heads * weights_config.seq_len));
    att_out = RowVectorBatch<float>(Extents2D(batch_size, heads * qkv_dim));
    att_sums = RowVectorBatch<float>(Extents2D(batch_size, model_dim));

    bf_pre_ffw_rms_out = RowVectorBatch<BF16>(Extents2D(batch_size, model_dim));
    C1 = RowVectorBatch<float>(Extents2D(batch_size, ff_hidden_dim));
    C2 = RowVectorBatch<float>(Extents2D(batch_size, ff_hidden_dim));
    ffw_out = RowVectorBatch<float>(Extents2D(batch_size, model_dim));

    if (layer_config.type == LayerAttentionType::kGriffinRecurrentBlock) {
      griffin_x = RowVectorBatch<float>(Extents2D(batch_size, model_dim));
      griffin_y = RowVectorBatch<float>(Extents2D(batch_size, model_dim));
      griffin_gate_x = RowVectorBatch<float>(Extents2D(batch_size, model_dim));
      griffin_multiplier =
          RowVectorBatch<float>(Extents2D(batch_size, model_dim));
    }

    inv_timescale = CreateInvTimescale(qkv_dim, post_qk);

    env = std::make_unique<MatMulEnv>(pools);
  }
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_ACTIVATIONS_H_
