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

#include "gemma/configs.h"     // ModelConfig
#include "gemma/gemma_args.h"  // AttentionImpl
#include "gemma/kv_cache.h"
#include "gemma/tensor_stats.h"
#include "ops/ops.h"      // CreateInvTimescale
#include "util/basics.h"  // BF16
#include "util/mat.h"     // MatStorageT
#include "util/threading_context.h"

namespace gcpp {

typedef std::vector<float, hwy::AlignedAllocator<float>> AlignedFloatVector;
typedef std::vector<BF16, hwy::AlignedAllocator<BF16>> AlignedBF16Vector;

// Returns the scale value to use for the query in the attention computation.
// Also called by ops_test.
static inline float ChooseQueryScale(const ModelConfig& config) {
  const LayerConfig& layer_config = config.layer_configs[0];
  if (config.query_scale == QueryScaleType::SqrtModelDimDivNumHeads)
    return 1.0f /
           sqrtf(static_cast<float>(config.model_dim / layer_config.heads));
  // QueryScaleType::SqrtKeySize
  return 1.0f / sqrtf(static_cast<float>(layer_config.qkv_dim));
}

struct AttentionActivations {
  AttentionActivations(
      const ModelConfig& config, const LayerConfig& layer_config,
      size_t batch_size, size_t seq_len, const RuntimeConfig& runtime_config,
      const Allocator& allocator,
      std::vector<hwy::AlignedFreeUniquePtr<uint8_t*[]>>& row_ptrs)
      :  // `vocab_size == 0` means it is for Vit part, VitAttention is still
         // MHA and does not use an external KV cache.
        q(MatFactory("q", batch_size,
                     config.vocab_size == 0
                         ? layer_config.heads * 3 * layer_config.qkv_dim
                         : layer_config.heads * layer_config.qkv_dim,
                     allocator)),
        q_bf(MatFactory("q_bf", batch_size,
                        config.vocab_size == 0
                            ? layer_config.heads * 3 * layer_config.qkv_dim
                            : layer_config.heads * layer_config.qkv_dim,
                        allocator)),
        q_T(MatFactory("q_T", layer_config.qkv_dim,
                       config.vocab_size == 0
                           ? batch_size * layer_config.heads * 3
                           : batch_size * layer_config.heads,
                       allocator)),
        vit_Q(MatFactory("Q2", batch_size, layer_config.qkv_dim, allocator)),
        vit_K(MatFactory("K2", seq_len, layer_config.qkv_dim, allocator)),
        vit_C(MatFactory("C2", batch_size, seq_len, allocator)),
        pre_att_rms_out(MatFactory("pre_att_rms_out", batch_size,
                                   config.model_dim, allocator)),
        att(MatFactory("att", batch_size, layer_config.heads * seq_len,
                       allocator)),
        att_out(MatFactory("att_out", batch_size,
                           layer_config.heads * layer_config.qkv_dim,
                           allocator)),
        softmax_max(MatFactory("softmax_max", batch_size, layer_config.heads,
                               allocator)),
        softmax_d(
            MatFactory("softmax_d", batch_size, layer_config.heads, allocator)),
        att_sums(
            MatFactory("att_sums", batch_size, config.model_dim, allocator)),

        inv_timescale(
            CreateInvTimescale(allocator, layer_config.qkv_dim,
                               layer_config.post_qk == PostQKType::HalfRope)),
        inv_timescale_global(CreateInvTimescale(
            allocator, layer_config.qkv_dim,
            layer_config.post_qk == PostQKType::HalfRope, 1000000.0)) {
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
    q_bf.AllocateAndAttachRowPtrs(row_ptrs);
    q_T.AllocateAndAttachRowPtrs(row_ptrs);
    vit_C.AllocateAndAttachRowPtrs(row_ptrs);
    att_sums.AllocateAndAttachRowPtrs(row_ptrs);
  }

  void SetBatchSize(size_t batch_size) {
    q.OverrideRows(batch_size);
    q_bf.OverrideRows(batch_size);
    // q_T rows are always qkv_dim!

    vit_Q.OverrideRows(batch_size);
    // vit_K stays seq_len!
    vit_C.OverrideRows(batch_size);

    pre_att_rms_out.OverrideRows(batch_size);
    att.OverrideRows(batch_size);
    att_out.OverrideRows(batch_size);
    softmax_max.OverrideRows(batch_size);
    softmax_d.OverrideRows(batch_size);
    att_sums.OverrideRows(batch_size);

    // `inv_timescale*` are not batched.
  }

  MatStorageT<float> q;  // query
  MatStorageT<BF16> q_bf;
  MatStorageT<BF16> q_T;  // Transposed to maximize attention speed.

  MatStorageT<float> vit_Q;
  MatStorageT<float> vit_K;
  MatStorageT<float> vit_C;

  MatStorageT<float> pre_att_rms_out;
  MatStorageT<float> att;          // attention vector
  MatStorageT<float> att_out;      // attention output
  MatStorageT<float> softmax_max;  // see OnlineSoftmaxState
  MatStorageT<float> softmax_d;    // see OnlineSoftmaxState
  // Accumulation of attention outputs over heads
  MatStorageT<BF16> att_sums;

  // Rope
  MatStorageT<float> inv_timescale;
  MatStorageT<float> inv_timescale_global;
};

// A non-owning view of AttentionActivations.
struct AttentionActivationsPtrs {
  AttentionActivationsPtrs(const ModelConfig& config, size_t seq_len)
      : config(config),
        div_seq_len(static_cast<uint32_t>(seq_len)),
        div_heads(static_cast<uint32_t>(config.layer_configs[0].heads)),
        query_scale(ChooseQueryScale(config)) {}

  AttentionActivationsPtrs(const ModelConfig& config, size_t seq_len,
                           const AttentionActivations& activations)
      : AttentionActivationsPtrs(config, seq_len) {
    q = activations.q;
    q_bf = activations.q_bf;
    q_T = activations.q_T;
    vit_Q = activations.vit_Q;
    vit_K = activations.vit_K;
    vit_C = activations.vit_C;
    pre_att_rms_out = activations.pre_att_rms_out;
    att = activations.att;
    att_out = activations.att_out;
    softmax_max = activations.softmax_max;
    softmax_d = activations.softmax_d;
    att_sums = activations.att_sums;
    inv_timescale = activations.inv_timescale;
    inv_timescale_global = activations.inv_timescale_global;
  }

  void SetBatchSize(size_t batch_size) {
    q.OverrideRows(batch_size);
    q_bf.OverrideRows(batch_size);
    // q_T rows are always qkv_dim!

    vit_Q.OverrideRows(batch_size);
    // vit_K stays seq_len!
    vit_C.OverrideRows(batch_size);

    pre_att_rms_out.OverrideRows(batch_size);
    att.OverrideRows(batch_size);
    att_out.OverrideRows(batch_size);
    softmax_max.OverrideRows(batch_size);
    softmax_d.OverrideRows(batch_size);
    att_sums.OverrideRows(batch_size);
    // `inv_timescale*` are not batched.
  }

  size_t SeqLen() const {
    return static_cast<size_t>(div_seq_len.GetDivisor());
  }

  const ModelConfig& config;

  // For the matrices below, the batch_size dimension is really qbatch.Size() *
  // token_batch_size, but in all known uses, one of those is 1.  Specifically,
  // during PrefillTBatch, it is prompt length (up to some max batch size)
  // and otherwise it's qbatch.Size().

  // Query matrix of size batch_size x (q_heads * qkv_dim).
  MatPtrT<float> q;
  // Query matrix of size batch_size x (q_heads * qkv_dim).
  MatPtrT<BF16> q_bf;
  // Transposed query matrix for faster Q*K^T.
  MatPtrT<BF16> q_T;

  MatPtrT<float> vit_Q;
  MatPtrT<float> vit_K;
  MatPtrT<float> vit_C;

  // Output of RMSNorm before attention, size batch_size x model_dim.
  MatPtrT<float> pre_att_rms_out;
  // Attention scores computed from Q*K^T, size batch_size x (q_heads *
  // seq_len).
  MatPtrT<float> att;
  // Attention output computed from att * V, size batch_size x (q_heads *
  // qkv_dim).
  MatPtrT<float> att_out;
  // The maximum logit value encountered when computing att_out from att,
  // size batch_size x q_heads . See OnlineSoftmaxState for details.
  // WARNING: Only filled in for AttentionImpl::kOld.
  MatPtrT<float> softmax_max;
  // The sum of scaled exponentials when computing att_out from att,
  // size batch_size x q_heads . See OnlineSoftmaxState for details.
  // WARNING: Only filled in for AttentionImpl::kOld.
  MatPtrT<float> softmax_d;
  // Accumulation of attention outputs over heads, size batch_size x
  // model_dim.
  MatPtrT<BF16> att_sums;
  // Inverse timescales for RoPE computation.
  MatPtrT<float> inv_timescale;
  // Inverse timescales for global RoPE computation.
  MatPtrT<float> inv_timescale_global;
  // Divisor for faster division by sequence length.
  hwy::Divisor div_seq_len;
  // Divisor for faster division by number of heads.
  hwy::Divisor div_heads;
  // Query scaling factor for attention computation.
  float query_scale;
};

struct Activations {
  Activations(const RuntimeConfig& runtime_config, const ModelConfig& config,
              size_t batch_size, size_t seq_len, ThreadingContext& ctx,
              std::vector<hwy::AlignedFreeUniquePtr<uint8_t*[]>>& row_ptrs)
      : layer_config(config.layer_configs[0]),

        x(MatFactory("x", batch_size, config.model_dim, ctx.allocator)),
        x_bf(MatFactory("x_bf", batch_size, config.model_dim, ctx.allocator)),
        logits(
            MatFactory("logits", batch_size, config.vocab_size, ctx.allocator)),
        sampled(MatFactory("sampled", batch_size, 3, ctx.allocator)),

        pre_ffw_rms_out(MatFactory("pre_ffw_rms_out", batch_size,
                                   config.model_dim, ctx.allocator)),
        C1(MatFactory("C1", batch_size, layer_config.ff_hidden_dim,
                      ctx.allocator)),
        C2(MatFactory("C2", batch_size, layer_config.ff_hidden_dim,
                      ctx.allocator)),
        ffw_out(
            MatFactory("ffw_out", batch_size, config.model_dim, ctx.allocator)),

        max_workers(ctx.pools.MaxWorkers()),
        s_ffw_in(config.num_layers, max_workers),
        s_ffw_hidden(config.num_layers, max_workers),
        s_ffw_out(config.num_layers, max_workers),

        s_w_gating_einsum_w1(config.num_layers, max_workers),
        s_w_gating_einsum_w2(config.num_layers, max_workers),
        s_w_linear_w(config.num_layers, max_workers),
        attention_impl(runtime_config.attention_impl),
        attention_storage(config, layer_config, batch_size, seq_len,
                          runtime_config, ctx.allocator, row_ptrs),
        attention(config, seq_len, attention_storage) {
    HWY_ASSERT(batch_size != 0);

    // For MatMul outputs, precompute their row pointers.
    // If we forget any MatMul outputs here, debug builds print a warning but
    // fill them in each MatMul call.
    x.AllocateAndAttachRowPtrs(row_ptrs);
    x_bf.AllocateAndAttachRowPtrs(row_ptrs);
    logits.AllocateAndAttachRowPtrs(row_ptrs);
    C1.AllocateAndAttachRowPtrs(row_ptrs);
    C2.AllocateAndAttachRowPtrs(row_ptrs);
    ffw_out.AllocateAndAttachRowPtrs(row_ptrs);

    // Note that BindC on any MatMul output considerably slows down Prefill.
  }

  ~Activations() {
    s_ffw_in.ReduceAndPrint("ffw_in");
    s_ffw_hidden.ReduceAndPrint("ffw_hidden");
    s_ffw_out.ReduceAndPrint("ffw_out");
  }

  // Negligible CPU time.
  void SetBatchSize(size_t batch_size) {
    x.OverrideRows(batch_size);
    x_bf.OverrideRows(batch_size);
    logits.OverrideRows(batch_size);
    sampled.OverrideRows(batch_size);

    pre_ffw_rms_out.OverrideRows(batch_size);
    C1.OverrideRows(batch_size);
    C2.OverrideRows(batch_size);
    ffw_out.OverrideRows(batch_size);

    attention_storage.SetBatchSize(batch_size);
    // `AttentionActivationsPtrs` holds `MatPtrT` which also require updating;
    // their row override is not updated when the underlying storage changes.
    attention.SetBatchSize(batch_size);
  }

  const LayerConfig& layer_config;

  MatStorageT<float> x;    // input
  MatStorageT<BF16> x_bf;  // output of final RMSNorm, input to EmbeddingMatmul
  MatStorageT<float> logits;      // TODO: BF16 after Softmax supports that.
  MatStorageT<uint32_t> sampled;  // batch_size x 3 (padded)

  // Gated FFW
  MatStorageT<BF16> pre_ffw_rms_out;
  MatStorageT<BF16> C1;
  MatStorageT<BF16> C2;
  MatStorageT<float> ffw_out;

  const size_t max_workers;
  TensorStats s_ffw_in;
  TensorStats s_ffw_hidden;  // after Activation+gating
  TensorStats s_ffw_out;

  TensorStats s_w_gating_einsum_w1;
  TensorStats s_w_gating_einsum_w2;
  TensorStats s_w_linear_w;

  AttentionImpl attention_impl;

  AttentionActivations attention_storage;
  AttentionActivationsPtrs attention;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_ACTIVATIONS_H_
