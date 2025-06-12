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

#include <vector>

#include "gemma/configs.h"   // ModelConfig
#include "ops/matmul.h"      // MatMulEnv
#include "ops/ops.h"         // CreateInvTimescale
#include "util/allocator.h"  // Allocator
#include "util/basics.h"     // BF16
#include "util/mat.h"        // MatStorageT
#include "hwy/profiler.h"

namespace gcpp {

// Returns the scale value to use for the query in the attention computation.
// Also called by ops_test.
static inline float ChooseQueryScale(const ModelConfig& config) {
  if (config.query_scale == QueryScaleType::SqrtModelDimDivNumHeads)
    return 1.0f / sqrtf(static_cast<float>(config.model_dim /
                                           config.layer_configs[0].heads));
  // QueryScaleType::SqrtKeySize
  return 1.0f / sqrtf(static_cast<float>(config.layer_configs[0].qkv_dim));
}

struct Activations {
  Activations(const ModelConfig& config, size_t batch_size, size_t seq_len,
              std::vector<hwy::AlignedFreeUniquePtr<uint8_t*[]>>& row_ptrs)
      : weights_config(config),
        layer_config(config.layer_configs[0]),
        div_seq_len(static_cast<uint32_t>(seq_len)),
        is_griffin(config.model == Model::GRIFFIN_2B),

        x("x", Extents2D(batch_size, config.model_dim), pad_),
        // `vocab_size == 0` means it is for Vit part, VitAttention is still MHA
        // and does not use an external KV cache.
        q("q",
          Extents2D(batch_size,
                    config.vocab_size == 0
                        ? layer_config.heads * 3 * layer_config.qkv_dim
                        : layer_config.heads * layer_config.qkv_dim),
          pad_),
        logits("logits", Extents2D(batch_size, config.vocab_size), pad_),

        pre_att_rms_out("pre_att_rms_out",
                        Extents2D(batch_size, config.model_dim), pad_),
        att("att", Extents2D(batch_size, layer_config.heads * seq_len), pad_),
        att_out(
            "att_out",
            Extents2D(batch_size, layer_config.heads * layer_config.qkv_dim),
            pad_),
        att_sums("att_sums", Extents2D(batch_size, config.model_dim), pad_),

        pre_ffw_rms_out("pre_ffw_rms_out",
                        Extents2D(batch_size, config.model_dim), pad_),
        C1("C1", Extents2D(batch_size, layer_config.ff_hidden_dim), pad_),
        C2("C2", Extents2D(batch_size, layer_config.ff_hidden_dim), pad_),
        ffw_out("ffw_out", Extents2D(batch_size, config.model_dim), pad_),

        griffin_x("griffin_x",
                  is_griffin ? Extents2D(batch_size, config.model_dim) : none_,
                  pad_),
        griffin_y("griffin_y",
                  is_griffin ? Extents2D(batch_size, config.model_dim) : none_,
                  pad_),
        griffin_gate_x(
            "griffin_gate_x",
            is_griffin ? Extents2D(batch_size, config.model_dim) : none_, pad_),
        griffin_multiplier(
            "griffin_mul",
            is_griffin ? Extents2D(batch_size, config.model_dim) : none_, pad_),

        inv_timescale(
            CreateInvTimescale(layer_config.qkv_dim,
                               layer_config.post_qk == PostQKType::HalfRope)),
        inv_timescale_global(CreateInvTimescale(
            layer_config.qkv_dim, layer_config.post_qk == PostQKType::HalfRope,
            1000000.0)),

        query_scale(ChooseQueryScale(config)) {
    HWY_ASSERT(batch_size != 0);

    // For MatMul outputs, precompute their row pointers.
    // If we forget any MatMul outputs here, debug builds print a warning but
    // fill them in each MatMul call.
    x.AllocateAndAttachRowPtrs(row_ptrs);
    q.AllocateAndAttachRowPtrs(row_ptrs);
    logits.AllocateAndAttachRowPtrs(row_ptrs);
    att_sums.AllocateAndAttachRowPtrs(row_ptrs);
    C1.AllocateAndAttachRowPtrs(row_ptrs);
    C2.AllocateAndAttachRowPtrs(row_ptrs);
    ffw_out.AllocateAndAttachRowPtrs(row_ptrs);

    // Note that BindC on any MatMul output considerably slows down Prefill.
  }

  void SetBatchSize(size_t batch_size) {
    PROFILER_ZONE("SetBatchSize");
    x.OverrideRows(batch_size);
    q.OverrideRows(batch_size);
    logits.OverrideRows(batch_size);

    pre_att_rms_out.OverrideRows(batch_size);
    att.OverrideRows(batch_size);
    att_out.OverrideRows(batch_size);
    att_sums.OverrideRows(batch_size);

    pre_ffw_rms_out.OverrideRows(batch_size);
    C1.OverrideRows(batch_size);
    C2.OverrideRows(batch_size);
    ffw_out.OverrideRows(batch_size);

    if (is_griffin) {
      griffin_x.OverrideRows(batch_size);
      griffin_y.OverrideRows(batch_size);
      griffin_gate_x.OverrideRows(batch_size);
      griffin_multiplier.OverrideRows(batch_size);
    }
  }

  bool IsGlobalLayer(size_t layer_idx) const {
    return weights_config.attention_window_sizes[layer_idx] ==
           div_seq_len.GetDivisor();
  }

  const ModelConfig& weights_config;
  const LayerConfig& layer_config;
  hwy::Divisor div_seq_len;
  bool is_griffin;
  const Extents2D none_ = Extents2D();
  const MatPadding pad_ = MatPadding::kOdd;

  MatStorageT<float> x;  // input
  MatStorageT<float> q;  // query
  MatStorageT<float> logits;

  // Attention
  MatStorageT<float> pre_att_rms_out;
  MatStorageT<float> att;      // attention vector
  MatStorageT<float> att_out;  // attention output
  // Accumulation of attention outputs over heads
  MatStorageT<BF16> att_sums;

  // Gated FFW
  MatStorageT<BF16> pre_ffw_rms_out;
  MatStorageT<float> C1;  // TODO: BF16 after Activation() supports it
  MatStorageT<float> C2;
  MatStorageT<BF16> ffw_out;

  // Griffin
  MatStorageT<float> griffin_x;
  MatStorageT<float> griffin_y;
  MatStorageT<float> griffin_gate_x;
  MatStorageT<float> griffin_multiplier;

  // Rope
  MatStorageT<float> inv_timescale;
  MatStorageT<float> inv_timescale_global;

  float query_scale;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_ACTIVATIONS_H_
