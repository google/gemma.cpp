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

#include <math.h>  // sqrtf
#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <vector>

#include "gemma/configs.h"   // ModelConfig
#include "ops/matmul.h"      // MatMulEnv
#include "ops/ops.h"         // CreateInvTimescale
#include "util/allocator.h"  // Allocator
#include "util/basics.h"     // BF16
#include "util/mat.h"        // MatStorageT

namespace gcpp {

struct GriffinActivations {
  GriffinActivations(const ModelConfig& config, size_t batch_size,
                     const Allocator& allocator)
      : griffin_x(
            MatFactory("griffin_x", batch_size, config.model_dim, allocator)),
        griffin_y(
            MatFactory("griffin_y", batch_size, config.model_dim, allocator)),
        griffin_gate_x(MatFactory("griffin_gate_x", batch_size,
                                  config.model_dim, allocator)),
        griffin_multiplier(MatFactory("griffin_mul", batch_size,
                                      config.model_dim, allocator)) {}

  void SetBatchSize(size_t batch_size) {
    if (griffin_x.Rows() == 0) return;
    griffin_x.OverrideRows(batch_size);
    griffin_y.OverrideRows(batch_size);
    griffin_gate_x.OverrideRows(batch_size);
    griffin_multiplier.OverrideRows(batch_size);
  }

  MatStorageT<float> griffin_x;
  MatStorageT<float> griffin_y;
  MatStorageT<float> griffin_gate_x;
  MatStorageT<float> griffin_multiplier;
};

struct AttentionActivations {
  // Returns the scale value to use for the query in the attention computation.
  // Also called by ops_test.
  static inline float ChooseQueryScale(const ModelConfig& config) {
    if (config.query_scale == QueryScaleType::SqrtModelDimDivNumHeads)
      return 1.0f / sqrtf(static_cast<float>(config.model_dim /
                                             config.layer_configs[0].heads));
    // QueryScaleType::SqrtKeySize
    return 1.0f / sqrtf(static_cast<float>(config.layer_configs[0].qkv_dim));
  }

  AttentionActivations(
      const ModelConfig& config, const LayerConfig& layer_config,
      size_t batch_size, size_t seq_len, const Allocator& allocator,
      std::vector<hwy::AlignedFreeUniquePtr<uint8_t*[]>>& row_ptrs)
      : config(config),

        // `vocab_size == 0` means it is for Vit part, VitAttention is still MHA
        // and does not use an external KV cache.
        q(MatFactory("q", batch_size,
                     config.vocab_size == 0
                         ? layer_config.heads * 3 * layer_config.qkv_dim
                         : layer_config.heads * layer_config.qkv_dim,
                     allocator)),

        pre_att_rms_out(MatFactory("pre_att_rms_out", batch_size,
                                   config.model_dim, allocator)),
        att(MatFactory("att", batch_size, layer_config.heads * seq_len,
                       allocator)),
        att_out(MatFactory("att_out", batch_size,
                           layer_config.heads * layer_config.qkv_dim,
                           allocator)),
        att_sums(
            MatFactory("att_sums", batch_size, config.model_dim, allocator)),

        inv_timescale(
            CreateInvTimescale(allocator, layer_config.qkv_dim,
                               layer_config.post_qk == PostQKType::HalfRope)),
        inv_timescale_global(CreateInvTimescale(
            allocator, layer_config.qkv_dim,
            layer_config.post_qk == PostQKType::HalfRope, 1000000.0)),

        div_seq_len(static_cast<uint32_t>(seq_len)),
        div_heads(static_cast<uint32_t>(layer_config.heads)),
        query_scale(ChooseQueryScale(config)) {
    // Batch size can be 0 in experimental code so do not assert.
    if (batch_size == 0) {
      static std::atomic_flag warned = ATOMIC_FLAG_INIT;
      if (!warned.test_and_set()) {
        HWY_WARN("Creating mostly empty activations with a batch_size of 0.");
      }
      return;
    }

    // For MatMul outputs, precompute their row pointers.
    // If we forget any MatMul outputs here, debug builds print a warning but
    // fill them in each MatMul call.
    q.AllocateAndAttachRowPtrs(row_ptrs);
    att_sums.AllocateAndAttachRowPtrs(row_ptrs);
  }

  void SetBatchSize(size_t batch_size) {
    q.OverrideRows(batch_size);

    pre_att_rms_out.OverrideRows(batch_size);
    att.OverrideRows(batch_size);
    att_out.OverrideRows(batch_size);
    att_sums.OverrideRows(batch_size);
  }

  const ModelConfig& config;

  MatStorageT<float> q;  // query

  MatStorageT<float> pre_att_rms_out;
  MatStorageT<float> att;      // attention vector
  MatStorageT<float> att_out;  // attention output
  // Accumulation of attention outputs over heads
  MatStorageT<BF16> att_sums;

  // Rope
  MatStorageT<float> inv_timescale;
  MatStorageT<float> inv_timescale_global;

  hwy::Divisor div_seq_len;
  // Unfortunately, some models (Griffin) have non-power-of-two heads.
  hwy::Divisor div_heads;
  float query_scale;
};

struct Activations {
  Activations(const ModelConfig& config, size_t batch_size, size_t seq_len,
              const Allocator& allocator,
              std::vector<hwy::AlignedFreeUniquePtr<uint8_t*[]>>& row_ptrs)
      : layer_config(config.layer_configs[0]),

        x(MatFactory("x", batch_size, config.model_dim, allocator)),
        logits(MatFactory("logits", batch_size, config.vocab_size, allocator)),

        pre_ffw_rms_out(MatFactory("pre_ffw_rms_out", batch_size,
                                   config.model_dim, allocator)),
        C1(MatFactory("C1", batch_size, layer_config.ff_hidden_dim, allocator)),
        C2(MatFactory("C2", batch_size, layer_config.ff_hidden_dim, allocator)),
        ffw_out(MatFactory("ffw_out", batch_size, config.model_dim, allocator)),

        attention(config, layer_config, batch_size, seq_len, allocator,
                  row_ptrs),
        griffin(config, config.model == Model::GRIFFIN_2B ? batch_size : 0,
                allocator) {
    HWY_ASSERT(batch_size != 0);

    // For MatMul outputs, precompute their row pointers.
    // If we forget any MatMul outputs here, debug builds print a warning but
    // fill them in each MatMul call.
    x.AllocateAndAttachRowPtrs(row_ptrs);
    logits.AllocateAndAttachRowPtrs(row_ptrs);
    C1.AllocateAndAttachRowPtrs(row_ptrs);
    C2.AllocateAndAttachRowPtrs(row_ptrs);
    ffw_out.AllocateAndAttachRowPtrs(row_ptrs);

    // Note that BindC on any MatMul output considerably slows down Prefill.
  }

  // Negligible CPU time.
  void SetBatchSize(size_t batch_size) {
    x.OverrideRows(batch_size);
    logits.OverrideRows(batch_size);

    pre_ffw_rms_out.OverrideRows(batch_size);
    C1.OverrideRows(batch_size);
    C2.OverrideRows(batch_size);
    ffw_out.OverrideRows(batch_size);

    attention.SetBatchSize(batch_size);
    griffin.SetBatchSize(batch_size);
  }

  const LayerConfig& layer_config;

  MatStorageT<float> x;  // input
  MatStorageT<float> logits;

  // Gated FFW
  MatStorageT<BF16> pre_ffw_rms_out;
  // Norm may be large, so prefer to keep as f32.
  MatStorageT<float> C1;
  MatStorageT<float> C2;
  MatStorageT<BF16> ffw_out;

  AttentionActivations attention;
  GriffinActivations griffin;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_ACTIVATIONS_H_
