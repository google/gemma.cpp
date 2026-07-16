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
#include "gemma/flash_structs.h"
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
static inline size_t MaxQkvDim(const ModelConfig& config) {
  size_t max_dim = 0;
  for (const auto& lc : config.layer_configs) {
    max_dim = HWY_MAX(max_dim, lc.qkv_dim);
  }
  return max_dim;
}
static inline size_t MaxFFHiddenDim(const ModelConfig& config) {
  size_t max_dim = 0;
  for (const auto& lc : config.layer_configs) {
    max_dim = HWY_MAX(max_dim, static_cast<size_t>(lc.ff_hidden_dim));
  }
  return max_dim;
}


static inline float ChooseQueryScale(const ModelConfig& config) {
  const LayerConfig& layer_config = config.layer_configs[0];
  if (config.query_scale == QueryScaleType::SqrtModelDimDivNumHeads)
    return 1.0f /
           sqrtf(static_cast<float>(config.model_dim / layer_config.heads));
  if (config.query_scale == QueryScaleType::One)
    return 1.0f;
  // QueryScaleType::SqrtKeySize
  return 1.0f / sqrtf(static_cast<float>(layer_config.qkv_dim));
}

struct AttentionActivations {
  AttentionActivations(
      const ModelConfig& config, const LayerConfig& layer_config,
      size_t batch_size, size_t seq_len, const RuntimeConfig& runtime_config,
      size_t max_workers, const Allocator& allocator,
      std::vector<hwy::AlignedFreeUniquePtr<uint8_t*[]>>& row_ptrs)
      : heads(layer_config.heads),
        qkv_dim(layer_config.qkv_dim),
        max_qkv_dim(MaxQkvDim(config)),
        rep_factor(max_workers *
                   AttentionActivations::kThreadReplicationFactor /
                   layer_config.heads),
        // `vocab_size == 0` means it is for Vit part, VitAttention
        // is still MHA and does not use an external KV cache.
        q(MatFactory("q", batch_size,
                     config.vocab_size == 0
                         ? layer_config.heads * 3 * max_qkv_dim
                         : layer_config.heads * max_qkv_dim,
                     allocator)),
        q_bf(MatFactory("q_bf", batch_size,
                        config.vocab_size == 0
                            ? layer_config.heads * 3 * max_qkv_dim
                            : layer_config.heads * max_qkv_dim,
                        allocator)),

        vit_Q(MatFactory("Q2", batch_size, max_qkv_dim, allocator)),
        vit_K_T(MatFactory(
            "K2_T", hwy::RoundUpTo(seq_len, kMaxBF16PerVector),
            layer_config.heads *
                hwy::RoundUpTo(max_qkv_dim, kMaxBF16PerVector),
            allocator, MatPadding::kPacked)),
        vit_V_T(MatFactory(
            "V2_T", hwy::RoundUpTo(seq_len, kMaxBF16PerVector),
            layer_config.heads *
                hwy::RoundUpTo(max_qkv_dim, kMaxBF16PerVector),
            allocator, MatPadding::kPacked)),
        pre_att_rms_out(MatFactory("pre_att_rms_out", batch_size,
                                   config.model_dim, allocator)),
        att_out(MatFactory("att_out", batch_size,
                           layer_config.heads * max_qkv_dim,
                           allocator)),
        att_out_reps(MatFactory("att_out", batch_size * rep_factor,
                                layer_config.heads * max_qkv_dim,
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
            allocator,
            max_qkv_dim,
            layer_config.post_qk == PostQKType::HalfRope, 1000000.0,
            config.partial_rotary_factor)) {
    // Batch size can be 0 in experimental code so do not assert.
    if (batch_size == 0) {
      static std::atomic_flag warned = ATOMIC_FLAG_INIT;
      if (!warned.test_and_set()) {
        HWY_WARN("Creating mostly empty activations with a batch_size of 0.");
      }
      return;
    }
    // This is a guess at the maximum number of params we might need to avoid
    // reallocations. The actual number of params is determined by the number of
    // query tiles, which is not known here.
    flash_params.reserve(batch_size * layer_config.heads);
    split_flash_params.reserve(batch_size * layer_config.heads);

    // For MatMul outputs, precompute their row pointers.
    // If we forget any MatMul outputs here, debug builds print a warning but
    // fill them in each MatMul call.
    q.AllocateAndAttachRowPtrs(row_ptrs);
    q_bf.AllocateAndAttachRowPtrs(row_ptrs);
    att_sums.AllocateAndAttachRowPtrs(row_ptrs);
  }

  void SetBatchSize(size_t batch_size) {
    q.OverrideRows(batch_size);
    q_bf.OverrideRows(batch_size);

    vit_Q.OverrideRows(batch_size);
    // vit_K_T and vit_V_T stay seq_len!

    pre_att_rms_out.OverrideRows(batch_size);
    att_out.OverrideRows(batch_size);
    att_out_reps.OverrideRows(batch_size * rep_factor);
    // There is no override for [split_]flash_params, because we reserved an
    // upper bound, and flash attention controls the actual size when it
    // calculates the size and number of tiles.
    softmax_max.OverrideRows(batch_size);
    softmax_d.OverrideRows(batch_size);
    att_sums.OverrideRows(batch_size);

    // `inv_timescale*` are not batched.
  }

  size_t heads;
  size_t qkv_dim;
  size_t max_qkv_dim;
  AlignedBF16Vector bf16_queries;
  std::vector<int16_t, hwy::AlignedAllocator<int16_t>> int16_queries;
  hwy::AlignedVector<int8_t> int8_queries;
  AlignedFloatVector float_queries;
  AlignedFloatVector q_scales;

  // Maximum factor by which we might scale-up work to maximize parallelism.
  size_t rep_factor = 1;
  // Parameters for flash attention. The size of the vector is somewhere between
  // the number of query rows and 1/8th of that.
  std::vector<Tile148Params> flash_params;
  // Parameters for flash attention, split by k-position. May be significantly
  // larger than flash_params in decode mode, when the number of query rows is
  // small.
  std::vector<Tile148Params> split_flash_params;
  MatStorageT<float> q;  // query
  MatStorageT<BF16> q_bf;

  MatStorageT<float> vit_Q;
  MatStorageT<KV_t> vit_K_T;
  MatStorageT<KV_t> vit_V_T;

  MatStorageT<float> pre_att_rms_out;
  MatStorageT<float> att_out;      // attention output
  MatStorageT<float> att_out_reps;  // attention output for each thread.
  MatStorageT<float> softmax_max;  // see OnlineSoftmaxState
  MatStorageT<float> softmax_d;    // see OnlineSoftmaxState
  // Accumulation of attention outputs over heads
  MatStorageT<BF16> att_sums;

  MatStorageT<float> k_tile_vec;
  MatStorageT<float> v_tile_vec;
  std::vector<MatStorageT<float>> sub_task_att_out;
  std::vector<AlignedFloatVector>
      sub_task_exp_denominator_sums;
  std::vector<AlignedFloatVector>
      sub_task_max_logits;

  // Rope
  MatStorageT<float> inv_timescale;
  MatStorageT<float> inv_timescale_global;
  // Replication factor to help evenly share work over threads.
  static constexpr size_t kThreadReplicationFactor = 4;
};

// A non-owning view of AttentionActivations.
struct AttentionActivationsPtrs {
  AttentionActivationsPtrs(const ModelConfig& config, size_t seq_len,
                           std::vector<Tile148Params>& flash_params,
                           std::vector<Tile148Params>& split_flash_params)
      : config(config),
        flash_params(flash_params),
        split_flash_params(split_flash_params),
        bf16_queries(nullptr),
        int16_queries(nullptr),
        int8_queries(nullptr),
        float_queries(nullptr),
        q_scales(nullptr),
        div_seq_len(static_cast<uint32_t>(seq_len)),
        div_heads(static_cast<uint32_t>(config.layer_configs[0].heads)),
        query_scale(ChooseQueryScale(config)) {}

  AttentionActivationsPtrs(const ModelConfig& config, size_t seq_len,
                           AttentionActivations& activations)
      : AttentionActivationsPtrs(config, seq_len, activations.flash_params,
                                 activations.split_flash_params) {
    q = activations.q;
    q_bf = activations.q_bf;
    vit_Q = activations.vit_Q;
    vit_K_T = activations.vit_K_T;
    vit_V_T = activations.vit_V_T;
    pre_att_rms_out = activations.pre_att_rms_out;
    att_out = activations.att_out;
    att_out_reps = activations.att_out_reps;
    softmax_max = activations.softmax_max;
    softmax_d = activations.softmax_d;
    att_sums = activations.att_sums;
    inv_timescale = activations.inv_timescale;
    inv_timescale_global = activations.inv_timescale_global;
    bf16_queries = &activations.bf16_queries;
    int16_queries = &activations.int16_queries;
    int8_queries = &activations.int8_queries;
    float_queries = &activations.float_queries;
    q_scales = &activations.q_scales;
  }

  void SetBatchSize(size_t batch_size) {
    q.OverrideRows(batch_size);
    q_bf.OverrideRows(batch_size);

    vit_Q.OverrideRows(batch_size);
    // vit_K_T and vit_V_T stay seq_len!

    pre_att_rms_out.OverrideRows(batch_size);
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
  // Parameters for flash attention.
  std::vector<Tile148Params>& flash_params;
  std::vector<Tile148Params>& split_flash_params;

  // For the matrices below, the batch_size dimension is really qbatch.Size() *
  // token_batch_size, but in all known uses, one of those is 1.  Specifically,
  // during PrefillTBatch, it is prompt length (up to some max batch size)
  // and otherwise it's qbatch.Size().

  // Query matrix of size batch_size x (q_heads * qkv_dim).
  MatPtrT<float> q;
  // Query matrix of size batch_size x (q_heads * qkv_dim).
  MatPtrT<BF16> q_bf;

  MatPtrT<float> vit_Q;
  MatPtrT<KV_t> vit_K_T;
  MatPtrT<KV_t> vit_V_T;

  // Output of RMSNorm before attention, size batch_size x model_dim.
  MatPtrT<float> pre_att_rms_out;
  // Attention output computed from att * V, size batch_size x (q_heads *
  // qkv_dim).
  MatPtrT<float> att_out;
  MatPtrT<float> att_out_reps;
  // The maximum logit value encountered when computing att_out, shape
  // batch_size x q_heads . See OnlineSoftmaxState for details.
  MatPtrT<float> softmax_max;
  // The sum of scaled exponentials when computing att_out, shape
  // batch_size x q_heads . See OnlineSoftmaxState for details.
  MatPtrT<float> softmax_d;
  // Accumulation of attention outputs over heads, size batch_size x
  // model_dim.
  MatPtrT<BF16> att_sums;
  // Stores intermediate results of computing QKV,
  // [qbatch * kv_heads , k_tile_size * qkv_dim]
  MatPtrT<float> k_tile_vec;
  MatPtrT<float> v_tile_vec;
  // Used by TiledFlashAttention to store intermediate results.
  std::vector<MatStorageT<float>>* sub_task_att_out;
  std::vector<AlignedFloatVector>*
      sub_task_exp_denominator_sums;
  std::vector<AlignedFloatVector>*
      sub_task_max_logits;
  AlignedBF16Vector* bf16_queries;
  std::vector<int16_t, hwy::AlignedAllocator<int16_t>>* int16_queries;
  hwy::AlignedVector<int8_t>* int8_queries;
  AlignedFloatVector* float_queries;
  AlignedFloatVector* q_scales;
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

static inline size_t MoEBatchSize(const LayerConfig& layer_config,
                                  size_t batch_size) {
  return layer_config.IsMoE() ? batch_size : 0;
}

struct PerToken {
  float weight;
  uint16_t expert_idx;
  uint16_t row_idx;
};
static_assert(sizeof(PerToken) == 8);
// Multiple to round up experts_per_token to avoid false sharing.
HWY_INLINE_VAR constexpr size_t kPerTokenPerLine =
    HWY_ALIGNMENT / sizeof(PerToken);

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
        C1(MatFactory("C1", batch_size, MaxFFHiddenDim(config),
                      ctx.allocator)),
        C2(MatFactory("C2", batch_size, MaxFFHiddenDim(config),
                      ctx.allocator)),
        ffw_out(
            MatFactory("ffw_out", batch_size, config.model_dim, ctx.allocator)),
        ple_embeds(
            MatFactory("ple_embeds", batch_size,
                       config.num_layers * config.ple_dim, ctx.allocator)),
        gate_out(
            MatFactory("gate_out", batch_size, config.ple_dim, ctx.allocator)),
        ple_token_emb(config.num_layers * config.ple_dim),

        max_workers(ctx.pools.MaxWorkers()),
        s_ffw_in(config.num_layers, max_workers),
        s_ffw_hidden(config.num_layers, max_workers),
        s_ffw_out(config.num_layers, max_workers),
        router_in(MatFactory("router_in",
                             MoEBatchSize(layer_config, batch_size),
                             config.model_dim, ctx.allocator)),
        router_logits(
            MatFactory("router_logits", MoEBatchSize(layer_config, batch_size),
                       layer_config.NumExperts(), ctx.allocator)),
        s_router_in(config.num_layers, max_workers),
        s_router_logits(config.num_layers, max_workers),
        s_expert_in(config.num_layers, max_workers),
        s_expert_hidden(config.num_layers, max_workers),
        s_expert_out(config.num_layers, max_workers),
        s_w_expert_in1(config.num_layers, max_workers),
        s_w_expert_in2(config.num_layers, max_workers),
        s_w_expert_hidden(config.num_layers, max_workers),
        s_w_gating_einsum_w1(config.num_layers, max_workers),
        s_w_gating_einsum_w2(config.num_layers, max_workers),
        s_w_linear_w(config.num_layers, max_workers),
        attention_impl(runtime_config.attention_impl),
        attention_storage(config, layer_config, batch_size, seq_len,
                          runtime_config, ctx.pools.MaxWorkers(), ctx.allocator,
                          row_ptrs),
        attention(config, seq_len, attention_storage) {
    HWY_ASSERT(batch_size != 0);

    // For MatMul outputs, precompute their row pointers.
    // If we forget any MatMul outputs here, debug builds print a warning but
    // fill them in each MatMul call.
    x.AllocateAndAttachRowPtrs(row_ptrs);
    x_bf.AllocateAndAttachRowPtrs(row_ptrs);
    logits.AllocateAndAttachRowPtrs(row_ptrs);
    if (config.ple_dim > 0) {
      ple_embeds.AllocateAndAttachRowPtrs(row_ptrs);
      gate_out.AllocateAndAttachRowPtrs(row_ptrs);
    }
    C1.AllocateAndAttachRowPtrs(row_ptrs);
    C2.AllocateAndAttachRowPtrs(row_ptrs);
    ffw_out.AllocateAndAttachRowPtrs(row_ptrs);

    if (layer_config.IsMoE()) {
      router_logits.AllocateAndAttachRowPtrs(row_ptrs);

      const size_t experts_per_token =
          layer_config.NumExpertsPerDatapoint();
      per_token_stride = hwy::RoundUpTo(
          layer_config.NumExpertsPerDatapoint(), kPerTokenPerLine);
      per_token = ctx.allocator.Alloc<PerToken>(batch_size * per_token_stride);
      expert_tokens =
          ctx.allocator.Alloc<uint16_t>(batch_size * experts_per_token);

      const size_t num_clusters = ctx.pools.NumClusters();
      per_cluster.reserve(num_clusters);
      for (size_t cluster_idx = 0; cluster_idx < num_clusters; ++cluster_idx) {
        per_cluster.emplace_back(config, layer_config, batch_size,
                                 ctx.allocator, row_ptrs);
      }

      const size_t num_experts = layer_config.NumExperts();
      ffw_expert_out.reserve(num_experts);
      for (size_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        ffw_expert_out.emplace_back(MatFactory(
            "ffw_partial_out", MoEBatchSize(layer_config, batch_size),
            config.model_dim, ctx.allocator));
        ffw_expert_out.back().AllocateAndAttachRowPtrs(row_ptrs);
      }
    }

    // Note that BindC on any MatMul output considerably slows down Prefill.
  }

  ~Activations() {
    s_ffw_in.ReduceAndPrint("ffw_in");
    s_ffw_hidden.ReduceAndPrint("ffw_hidden");
    s_ffw_out.ReduceAndPrint("ffw_out");
    s_router_in.ReduceAndPrint("router_in");
    s_router_logits.ReduceAndPrint("router_logits");
    s_expert_in.ReduceAndPrint("expert_in");
    s_expert_hidden.ReduceAndPrint("expert_hidden");
    s_expert_out.ReduceAndPrint("expert_out");
    s_w_expert_in1.ReduceAndPrint("w_expert_in1");
    s_w_expert_in2.ReduceAndPrint("w_expert_in2");
    s_w_expert_hidden.ReduceAndPrint("w_expert_hidden");
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
    if (layer_config.ple_dim > 0) {
      ple_embeds.OverrideRows(batch_size);
      gate_out.OverrideRows(batch_size);
    }

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
  MatStorageT<float> ple_embeds;
  MatStorageT<BF16> gate_out;
  std::vector<float> ple_token_emb;

  const size_t max_workers;
  TensorStats s_ffw_in;
  TensorStats s_ffw_hidden;  // after Activation+gating
  TensorStats s_ffw_out;

  // For MoE layers. These are used outside the expert-parallel loop:
  MatStorageT<BF16> router_in;
  MatStorageT<float> router_logits;  // batch_size x num_experts

  size_t per_token_stride;           // padded experts_per_token
  AlignedPtr<PerToken[]> per_token;  // batch_size x per_token_stride

  PerToken* GetPerToken(size_t token_idx) {
    return &per_token[token_idx * per_token_stride];
  }

  AlignedPtr<uint16_t[]> expert_tokens;  // ragged array

  struct PerCluster {
    PerCluster(const ModelConfig& config, const LayerConfig& layer_config,
               size_t batch_size, Allocator& allocator,
               std::vector<hwy::AlignedFreeUniquePtr<uint8_t*[]>>& row_ptrs)
        : moe_C1(MatFactory("C1", batch_size, layer_config.ff_hidden_dim,
                            allocator)),
          moe_C2(MatFactory("C2", batch_size, layer_config.ff_hidden_dim,
                            allocator)),
          ffw_expert_in(MatFactory("ffw_partial_in",
                                   MoEBatchSize(layer_config, batch_size),
                                   config.model_dim, allocator)) {
      moe_C1.AllocateAndAttachRowPtrs(row_ptrs);
      moe_C2.AllocateAndAttachRowPtrs(row_ptrs);
      ffw_expert_in.AllocateAndAttachRowPtrs(row_ptrs);
    }
    MatStorageT<BF16> moe_C1;
    MatStorageT<BF16> moe_C2;
    MatStorageT<BF16> ffw_expert_in;
  };
  std::vector<PerCluster> per_cluster;
  std::vector<MatStorageT<BF16>> ffw_expert_out;

  TensorStats s_router_in;
  TensorStats s_router_logits;
  TensorStats s_expert_in;
  TensorStats s_expert_hidden;  // after Activation+gating
  TensorStats s_expert_out;
  TensorStats s_w_expert_in1;
  TensorStats s_w_expert_in2;
  TensorStats s_w_expert_hidden;  // after Activation+gating
  TensorStats s_w_gating_einsum_w1;
  TensorStats s_w_gating_einsum_w2;
  TensorStats s_w_linear_w;

  AttentionImpl attention_impl;

  AttentionActivations attention_storage;
  AttentionActivationsPtrs attention;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_ACTIVATIONS_H_
