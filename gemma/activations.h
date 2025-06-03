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

#include <vector>

#include "gemma/configs.h"   // ModelConfig
#include "ops/matmul.h"      // MatMulEnv
#include "ops/ops.h"         // CreateInvTimescale
#include "util/allocator.h"  // Allocator
#include "util/basics.h"     // BF16
#include "util/mat.h"        // MatStorageT

namespace gcpp {

struct Activations {
  Activations(const ModelConfig& config, size_t batch_size, MatMulEnv* env)
      : weights_config(config),
        layer_config(config.layer_configs[0]),
        seq_len(config.seq_len),
        cache_pos_size(config.CachePosSize()),
        is_griffin(config.model == Model::GRIFFIN_2B),

        x("x", Extents2D(batch_size, config.model_dim), pad_),
        // `vocab_size == 0` means it is for Vit part, VitAttention is still MHA
        // and does not use an external KV cache.
        q("q", Extents2D(batch_size, config.vocab_size == 0 ?
          layer_config.heads * 3 * layer_config.qkv_dim :
          layer_config.heads * layer_config.qkv_dim), pad_),
        logits("logits", Extents2D(batch_size, config.vocab_size), pad_),

        pre_att_rms_out("pre_att_rms_out",
                        Extents2D(batch_size, config.model_dim), pad_),
        att("att", Extents2D(batch_size, layer_config.heads * config.seq_len),
            pad_),
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

        inv_timescale(CreateInvTimescale(
            ThreadingContext::Get().allocator, layer_config.qkv_dim,
            layer_config.post_qk == PostQKType::HalfRope)),
        inv_timescale_global(CreateInvTimescale(
            ThreadingContext::Get().allocator, layer_config.qkv_dim,
            layer_config.post_qk == PostQKType::HalfRope, 1000000.0)),

        env(env) {
    HWY_ASSERT(batch_size != 0);

    // For MatMul outputs, precompute their row pointers.
    const auto init_row_ptrs = [&](MatPtrT<float>& mat) {
      if (!mat.HasPtr()) return;
      row_ptrs.push_back(hwy::AllocateAligned<uint8_t*>(mat.Rows()));
      uint8_t** ptrs = row_ptrs.back().get();
      for (size_t r = 0; r < mat.Rows(); ++r) {
        ptrs[r] = mat.RowBytes(r);
      }
      mat.AttachRowPtrs(ptrs);
    };
    // If we forget any MatMul outputs here, debug builds print a warning but
    // fill them in each MatMul call.
    init_row_ptrs(q);
    init_row_ptrs(logits);
    init_row_ptrs(att_sums);
    init_row_ptrs(C1);
    init_row_ptrs(C2);
    init_row_ptrs(ffw_out);
    // TODO: also init rows for image_tokens.

    // Note that BindC on any MatMul output considerably slows down Prefill.
  }

  void SetBatchSize(size_t batch_size) {
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

  const ModelConfig& weights_config;
  const LayerConfig& layer_config;
  size_t seq_len;
  size_t cache_pos_size = 0;  // TODO: after moving KVCache to MatStorageT.
  bool is_griffin = false;
  const Extents2D none_ = Extents2D();
  const MatPadding pad_ = MatPadding::kOdd;

  MatStorageT<float> x;  // input
  MatStorageT<float> q;  // query, also KV if MHA.
  MatStorageT<float> logits;

  // Attention
  MatStorageT<float> pre_att_rms_out;
  MatStorageT<float> att;      // attention vector
  MatStorageT<float> att_out;  // attention output
  // Accumulation of attention outputs over heads
  MatStorageT<float> att_sums;

  // Gated FFW
  MatStorageT<BF16> pre_ffw_rms_out;
  MatStorageT<float> C1;
  MatStorageT<float> C2;
  MatStorageT<float> ffw_out;

  // Griffin
  MatStorageT<float> griffin_x;
  MatStorageT<float> griffin_y;
  MatStorageT<float> griffin_gate_x;
  MatStorageT<float> griffin_multiplier;

  // Rope
  MatStorageT<float> inv_timescale;
  MatStorageT<float> inv_timescale_global;

  MatMulEnv* env;
  // Per-tensor allocations to make it likelier that asan detects bugs such as
  // use after free, overrun, and dangling references.
  std::vector<hwy::AlignedFreeUniquePtr<uint8_t*[]>> row_ptrs;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_ACTIVATIONS_H_
