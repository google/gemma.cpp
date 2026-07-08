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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_WEIGHTS_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_WEIGHTS_H_

#include <math.h>  // isnan
#include <stddef.h>
#include <stdint.h>

#include <limits>
#include <string>
#include <vector>

#include "compression/types.h"
#include "gemma/configs.h"      // ModelConfig
#include "gemma/gemma_args.h"   // InferenceArgs
#include "gemma/model_store.h"  // ModelStore
#include "gemma/tensor_info.h"  // TensorInfoRegistry
#include "io/blob_store.h"      // BlobWriter
#include "util/mat.h"           // MatPtr
#include "util/threading_context.h"

namespace gcpp {

// Argument passed to the `ForEachTensor` callback.
struct TensorArgs {
  // `other_mat1` and `other_mat2` can be nullptr, or tensor(s) of the same
  // name/type from another `LayerWeightsPtrs` for iterating over tensor pairs
  // (for copying) or triples (for `AdamUpdateMV`). Set by `TENSOR_ARGS`.
  // `flags` is a combination of zero or more `Flags`.
  TensorArgs(MatPtr& mat, MatPtr* other_mat1, MatPtr* other_mat2, int flags)
      : mat(mat),
        other_mat1(other_mat1),
        other_mat2(other_mat2),
        flags(flags) {}

  MatPtr& mat;
  MatPtr* other_mat1;  // either/both can be nullptr.
  MatPtr* other_mat2;

  enum Flags {
    // Default: Read the tensor from the file and abort if it is not found.
    kMustRead = 0,

    // Not an error if the tensor is not present in the file. For example,
    // the _w1/_w2 tensors are not always present.
    kMaybeRead = 1,

    // Avoid padding tensor rows when reading.
    kPacked = 2,
  };
  const int flags;
};

// Shorthand for creating the argument to the `ForEachTensor` callback. A macro
// seems less bad than member pointer syntax.
#define TENSOR_ARGS(mat, flag)                     \
  TensorArgs(mat, other1 ? &other1->mat : nullptr, \
             other2 ? &other2->mat : nullptr, TensorArgs::flag)

// Finds tensors by name in `TensorInfoRegistry` (constructed from
// `ModelConfig`) and constructs `MatPtr` metadata with those shapes.
class MatFinder {
 public:
  MatFinder(const std::string& suffix, const TensorInfoRegistry& tensors)
      : suffix_(suffix), tensors_(tensors) {}

  // Retrieves shape by name via `TensorInfo` from `TensorInfoRegistry`.
  MatPtr operator()(const std::string& base_name) const {
    const std::string name = std::string(base_name) + suffix_;
    return MatPtr(name.c_str(), Type::kUnknown,
                  ExtentsFromInfo(tensors_.Find(name)));
  }

 private:
  const std::string suffix_;
  const TensorInfoRegistry& tensors_;
};

// Stores pre-converted float min/max for clamping activations. The BF16->float
// conversion happens once at load time in Fixup(), not at every inference call.
// Note: with a one-sided clamp (if min is inactive) IsActive() will
// still return true. This is fine for the way it is used currently, but it's
// worth noting if this is ever used in a context where the difference matters.
struct ClampRange {
  float min = -std::numeric_limits<float>::infinity();
  float max = std::numeric_limits<float>::infinity();
  bool IsActive() const {
    return max != std::numeric_limits<float>::infinity();
  }
};

// Per-layer weight metadata and pointers. The tensor data is owned by
// `MatOwner`.
struct LayerWeightsPtrs {
  // Initializes tensor metadata without allocating.
  // NOTE: do not store layer_idx, TransformerLayer and Attention may use
  // other values for purposes of the KV cache.
  LayerWeightsPtrs(size_t layer_idx, const LayerConfig& config,
                   const TensorInfoRegistry& tensors)
      : layer_idx(layer_idx),
        finder_(LayerSuffix(layer_idx), tensors),
        qkv_einsum_w(finder_("qkv_ein")),
        qkv_einsum_w1(finder_("qkv1_w")),
        qkv_einsum_w2(finder_("qkv2_w")),
        attention_output_biases(finder_("attn_ob")),
        // MultiHeadDotProductAttention.
        vit({.attn_out_w = finder_("attn_out_w"),
             .attn_out_b = finder_("attn_out_b"),
             .qkv_einsum_w = finder_("qkv_ein_w"),
             .qkv_einsum_b = finder_("qkv_ein_b"),
             .linear_0_w = finder_("linear_0_w"),
             .linear_0_b = finder_("linear_0_b"),
             .linear_1_w = finder_("linear_1_w"),
             .linear_1_b = finder_("linear_1_b"),
             .layer_norm_0_bias = finder_("ln_0_bias"),
             .layer_norm_0_scale = finder_("ln_0_scale"),
             .layer_norm_1_bias = finder_("ln_1_bias"),
             .layer_norm_1_scale = finder_("ln_1_scale")}),
        gating_einsum_w(finder_("gating_ein")),
        gating_einsum_w1(finder_("gating1_w")),
        gating_einsum_w2(finder_("gating2_w")),
        linear_w(finder_("linear_w")),
        pre_attention_norm_scale(finder_("pre_att_ns")),
        pre_ffw_norm_scale(finder_("pre_ff_ns")),
        post_attention_norm_scale(finder_("post_att_ns")),
        post_ffw_norm_scale(finder_("post_ff_ns")),
        skip_scale(finder_("skip_scale")),
        ffw_gating_biases(finder_("ffw_gat_b")),
        ffw_output_biases(finder_("ffw_out_b")),

        attn_vec_einsum_w(finder_("att_ein")),
        att_weights(finder_("att_w")),

        key_norm_scale(finder_("key_norm")),
        query_norm_scale(finder_("query_norm")),

        router_scale(finder_("router_scale")),
        p_expert_sc(finder_("p_expert_sc")),
        post_ffw1_ns(finder_("post_ffw1_ns")),
        post_ffw2_ns(finder_("post_ffw2_ns")),
        pre_ffw2_ns(finder_("pre_ffw2_ns")),
        moe_router(finder_("moe_router")),
        moe_router_bias(finder_("moe_r_bias")),

        mla_q_a(finder_("mla_q_a")),
        mla_q_a_norm(finder_("mla_q_a_ns")),
        mla_q_b(finder_("mla_q_b")),
        mla_kv_a(finder_("mla_kv_a")),
        mla_kv_a_norm(finder_("mla_kv_a_ns")),
        mla_kv_b(finder_("mla_kv_b")),
        mla_o_a(finder_("mla_o_a")),
        mla_o_b(finder_("mla_o_b")),
        attn_sink(finder_("attn_sink")),
        mla_pool_w(finder_("mla_pool_w")),
        mla_pool_b(finder_("mla_pool_b")),
        comp_wkv(finder_("comp_wkv")),
        comp_wgate(finder_("comp_wgate")),
        comp_ape(finder_("comp_ape")),
        comp_norm(finder_("comp_ns")),
        idx_q_w(finder_("idx_q_w")),
        idx_k_w(finder_("idx_k_w")),
        idx_w_proj(finder_("idx_w_proj")),
        idxc_wkv(finder_("idxc_wkv")),
        idxc_wgate(finder_("idxc_wgate")),
        idxc_ape(finder_("idxc_ape")),
        idxc_norm(finder_("idxc_ns")),
        hash_tid2eid(finder_("hash_tid2eid")),

        hc_att_fn(finder_("hc_att_fn")),
        hc_att_base(finder_("hc_att_base")),
        hc_att_scale(finder_("hc_att_scale")),
        hc_ffw_fn(finder_("hc_ffw_fn")),
        hc_ffw_base(finder_("hc_ffw_base")),
        hc_ffw_scale(finder_("hc_ffw_scale")),

        ple_gate(finder_("ple_gate")),
        ple_proj(finder_("ple_proj")),
        post_ple_ns(finder_("post_ple_ns")),

        layer_config(config) {
    if (layer_config.IsMoE()) {
      for (uint32_t i = 0; i < layer_config.NumExperts(); ++i) {
        const std::string moe_suffix = MoESuffix(layer_idx, i);
        MatFinder moe_finder(moe_suffix, tensors);
        moe_gating_einsum_w1.emplace_back(moe_finder("gating1_w"));
        moe_gating_einsum_w2.emplace_back(moe_finder("gating2_w"));
        moe_linear_w.emplace_back(moe_finder("linear_w"));
      }
    }
  }
  ~LayerWeightsPtrs() = default;

  const size_t layer_idx;
  const MatFinder finder_;

  // Files either have qkv_einsum_w with 2 stacked matrices or separate
  // w1/w2 tensors. Fixup ensures w1/w2 are ready for use by gemma-inl.h.
  MatPtr qkv_einsum_w;
  MatPtr qkv_einsum_w1;
  MatPtr qkv_einsum_w2;
  MatPtrT<float> attention_output_biases;

  struct {
    // MultiHeadDotProductAttention.
    MatPtr attn_out_w;  // at least BF16.
    MatPtrT<float> attn_out_b;
    MatPtr qkv_einsum_w;  // at least BF16.
    MatPtrT<float> qkv_einsum_b;
    // MlpBlock.
    MatPtr linear_0_w;  // at least BF16.
    MatPtrT<float> linear_0_b;
    MatPtr linear_1_w;  // at least BF16.
    MatPtrT<float> linear_1_b;
    // LayerNorm.
    MatPtr layer_norm_0_bias;   // at least BF16.
    MatPtr layer_norm_0_scale;  // at least BF16.
    MatPtr layer_norm_1_bias;   // at least BF16.
    MatPtr layer_norm_1_scale;  // at least BF16.
  } vit;

  // Files either have gating_einsum_w with 2 stacked matrices or separate
  // w1/w2 tensors. `Fixup` ensures w1/w2 are ready for use by gemma-inl.h.
  MatPtr gating_einsum_w;
  MatPtr gating_einsum_w1;
  MatPtr gating_einsum_w2;
  MatPtr linear_w;
  MatPtr pre_attention_norm_scale;   // at least BF16.
  MatPtr pre_ffw_norm_scale;         // at least BF16.
  MatPtr post_attention_norm_scale;  // at least BF16.
  MatPtr post_ffw_norm_scale;        // at least BF16.
  MatPtr skip_scale;                 // at least BF16.

  MatPtrT<float> ffw_gating_biases;
  MatPtrT<float> ffw_output_biases;

  MatPtr attn_vec_einsum_w;  // Use att_weights instead of this.
  MatPtr att_weights;        // Use this instead of attn_vec_einsum_w.

  MatPtr key_norm_scale;    // at least BF16.
  MatPtr query_norm_scale;  // at least BF16.
  
  MatPtr router_scale;
  MatPtr p_expert_sc;
  MatPtr post_ffw1_ns;
  MatPtr post_ffw2_ns;
  MatPtr pre_ffw2_ns;
  MatPtr moe_router;
  MatPtr moe_router_bias;  // DeepSeek aux-loss-free routing bias.
  std::vector<MatPtr> moe_gating_einsum_w1;
  std::vector<MatPtr> moe_gating_einsum_w2;
  std::vector<MatPtr> moe_linear_w;

  // DeepSeek Multi-head Latent Attention.
  MatPtr mla_q_a;        // [q_lora_rank, model_dim]
  MatPtr mla_q_a_norm;   // [q_lora_rank]
  MatPtr mla_q_b;        // [heads * qkv_dim, q_lora_rank or model_dim]
  MatPtr mla_kv_a;       // [KVLatentDim, model_dim]
  MatPtr mla_kv_a_norm;  // V4: [KVLatentDim]; V3: [kv_lora_rank]
  MatPtr mla_kv_b;       // V3 only: [heads * (nope + v_head_dim), kv_lora_rank]
  // V4 grouped low-rank output projection and attention sink.
  MatPtr mla_o_a;     // [o_groups * o_lora_rank, heads/o_groups * qkv_dim]
  MatPtr mla_o_b;     // [model_dim, o_groups * o_lora_rank]
  MatPtr attn_sink;   // [heads] f32
  MatPtr mla_pool_w;  // V3 only: [KVLatentDim]
  MatPtr mla_pool_b;  // V3 only: [kv_compression_rate]
  // V4 learned gated compressor (rate > 1).
  MatPtr comp_wkv;    // [coff * KVLatentDim, model_dim]
  MatPtr comp_wgate;  // [coff * KVLatentDim, model_dim]
  MatPtr comp_ape;    // [rate, coff * KVLatentDim] f32
  MatPtr comp_norm;   // [KVLatentDim]
  // Lightning indexer. V4: idx_q_w is [heads*head_dim, q_lora_rank] over qr;
  // V3: [heads*head_dim, model_dim] over x with idx_k_w keys.
  MatPtr idx_q_w;
  MatPtr idx_k_w;     // V3 only
  MatPtr idx_w_proj;  // V4: [indexer_heads, model_dim]
  // V4 indexer's own compressor.
  MatPtr idxc_wkv;    // [coff * indexer_head_dim, model_dim]
  MatPtr idxc_wgate;  // [coff * indexer_head_dim, model_dim]
  MatPtr idxc_ape;    // [rate, coff * indexer_head_dim] f32
  MatPtr idxc_norm;   // [indexer_head_dim]
  // Hash routing (V4 first n_hash_layers): per-token-id expert indices.
  MatPtr hash_tid2eid;  // [vocab_size, experts_per_token] f32

  // Manifold-constrained hyper-connections (empty unless hc_mult > 1).
  // Per-token dynamic weights: mixes = hc_fn @ flatten(streams) * rsqrt(ms),
  // then split into read/write/mixing weights via sigmoid / Sinkhorn-Knopp.
  MatPtr hc_att_fn;     // [(2 + hc_mult) * hc_mult, hc_mult * model_dim] f32
  MatPtr hc_att_base;   // [(2 + hc_mult) * hc_mult] f32
  MatPtr hc_att_scale;  // [3] f32
  MatPtr hc_ffw_fn;
  MatPtr hc_ffw_base;
  MatPtr hc_ffw_scale;

  MatPtr ple_gate;
  MatPtr ple_proj;
  MatPtr post_ple_ns;

  const LayerConfig& layer_config;

  // Calls `func(TensorArgs)` for each tensor which is in use for the
  // current `layer_config`. `other1` and `other2` are optional arguments so we
  // can also iterate over pairs or triples of tensors for `AdamUpdateMV`.
  // Public because also called by `WeightsPtrs`.
  template <class Func>
  void ForEachTensor(LayerWeightsPtrs* other1, LayerWeightsPtrs* other2,
                     Func func) {
    if (layer_config.type == LayerAttentionType::kVit) {
      // MHA.
      func(TENSOR_ARGS(vit.attn_out_w, kMustRead));
      func(TENSOR_ARGS(vit.attn_out_b, kMustRead));
      func(TENSOR_ARGS(vit.qkv_einsum_w, kMustRead));
      // Used as 1D MatMul bias, but has `heads + 2 * kv_heads` rows, hence
      // must not be padded.
      func(TENSOR_ARGS(vit.qkv_einsum_b, kMustRead | TensorArgs::kPacked));
      // MlpBlock.
      func(TENSOR_ARGS(vit.linear_0_w, kMustRead));
      func(TENSOR_ARGS(vit.linear_0_b, kMustRead));
      func(TENSOR_ARGS(vit.linear_1_w, kMustRead));
      func(TENSOR_ARGS(vit.linear_1_b, kMustRead));
      // LayerNorm.
      func(TENSOR_ARGS(vit.layer_norm_0_bias, kMustRead));
      func(TENSOR_ARGS(vit.layer_norm_0_scale, kMustRead));
      func(TENSOR_ARGS(vit.layer_norm_1_bias, kMustRead));
      func(TENSOR_ARGS(vit.layer_norm_1_scale, kMustRead));
      return;
    }
    if (layer_config.type == LayerAttentionType::kGemma) {
      // Either read from file, or allocated during Fixup().
      func(TENSOR_ARGS(att_weights, kMaybeRead));
      func(TENSOR_ARGS(attn_vec_einsum_w, kMaybeRead));
      func(TENSOR_ARGS(qkv_einsum_w, kMaybeRead));
      func(TENSOR_ARGS(qkv_einsum_w1, kMaybeRead));
      func(TENSOR_ARGS(qkv_einsum_w2, kMaybeRead));
    }
    if (layer_config.type == LayerAttentionType::kDeepSeekMLA) {
      if (layer_config.q_lora_rank > 0) {
        func(TENSOR_ARGS(mla_q_a, kMustRead));
        func(TENSOR_ARGS(mla_q_a_norm, kMustRead));
      }
      func(TENSOR_ARGS(mla_q_b, kMustRead));
      func(TENSOR_ARGS(mla_kv_a, kMustRead));
      func(TENSOR_ARGS(mla_kv_a_norm, kMustRead));
      if (layer_config.IsV4MLA()) {
        func(TENSOR_ARGS(mla_o_a, kMustRead));
        func(TENSOR_ARGS(mla_o_b, kMustRead));
        func(TENSOR_ARGS(attn_sink, kMustRead));
        if (layer_config.HasCompressor()) {
          func(TENSOR_ARGS(comp_wkv, kMustRead));
          func(TENSOR_ARGS(comp_wgate, kMustRead));
          func(TENSOR_ARGS(comp_ape, kMustRead));
          func(TENSOR_ARGS(comp_norm, kMustRead));
        }
        if (layer_config.HasIndexer()) {
          func(TENSOR_ARGS(idx_q_w, kMustRead));
          func(TENSOR_ARGS(idx_w_proj, kMustRead));
          func(TENSOR_ARGS(idxc_wkv, kMustRead));
          func(TENSOR_ARGS(idxc_wgate, kMustRead));
          func(TENSOR_ARGS(idxc_ape, kMustRead));
          func(TENSOR_ARGS(idxc_norm, kMustRead));
        }
      } else {
        func(TENSOR_ARGS(att_weights, kMustRead));  // o_proj, stored directly
        func(TENSOR_ARGS(mla_kv_b, kMustRead));
        if (layer_config.kv_compression_rate > 1) {
          func(TENSOR_ARGS(mla_pool_w, kMustRead));
          func(TENSOR_ARGS(mla_pool_b, kMustRead));
        }
        if (layer_config.indexer_heads > 0) {
          func(TENSOR_ARGS(idx_q_w, kMustRead));
          func(TENSOR_ARGS(idx_k_w, kMustRead));
        }
      }
      // Registered only when the model config has hc_mult > 1.
      if (hc_att_fn.Rows() > 0) {
        func(TENSOR_ARGS(hc_att_fn, kMustRead));
        func(TENSOR_ARGS(hc_att_base, kMustRead));
        func(TENSOR_ARGS(hc_att_scale, kMustRead));
        func(TENSOR_ARGS(hc_ffw_fn, kMustRead));
        func(TENSOR_ARGS(hc_ffw_base, kMustRead));
        func(TENSOR_ARGS(hc_ffw_scale, kMustRead));
      }
    }
    {
      func(TENSOR_ARGS(gating_einsum_w, kMaybeRead));
      func(TENSOR_ARGS(gating_einsum_w1, kMaybeRead));
      func(TENSOR_ARGS(gating_einsum_w2, kMaybeRead));
      func(TENSOR_ARGS(linear_w, kMaybeRead));
      func(TENSOR_ARGS(pre_attention_norm_scale, kMustRead));
      func(TENSOR_ARGS(pre_ffw_norm_scale, kMustRead));
      func(TENSOR_ARGS(skip_scale, kMaybeRead));
    }

    if (layer_config.post_norm == PostNormType::Scale) {
      func(TENSOR_ARGS(post_attention_norm_scale, kMustRead));
      func(TENSOR_ARGS(post_ffw_norm_scale, kMustRead));
    }
    if (layer_config.use_qk_norm) {
      func(TENSOR_ARGS(key_norm_scale, kMustRead));
      func(TENSOR_ARGS(query_norm_scale, kMustRead));
    }
    if (layer_config.IsMoE()) {
      func(TENSOR_ARGS(moe_router, kMustRead));
      if (layer_config.type == LayerAttentionType::kDeepSeekMLA) {
        // DeepSeek MoE: no router scale / per-expert scale / extra norms.
        if (layer_config.use_routing_bias) {
          func(TENSOR_ARGS(moe_router_bias, kMustRead));
        }
        if (layer_config.hash_routing) {
          func(TENSOR_ARGS(hash_tid2eid, kMustRead));
        }
      } else {
        func(TENSOR_ARGS(router_scale, kMustRead));
        func(TENSOR_ARGS(p_expert_sc, kMustRead));
        func(TENSOR_ARGS(post_ffw1_ns, kMustRead));
        func(TENSOR_ARGS(post_ffw2_ns, kMustRead));
        func(TENSOR_ARGS(pre_ffw2_ns, kMustRead));
      }
      for (uint32_t i = 0; i < layer_config.NumExperts(); ++i) {
        func(TENSOR_ARGS(moe_gating_einsum_w1[i], kMustRead));
        func(TENSOR_ARGS(moe_gating_einsum_w2[i], kMustRead));
        func(TENSOR_ARGS(moe_linear_w[i], kMustRead));
      }
    }

    if (layer_config.ple_dim > 0) {
      func(TENSOR_ARGS(ple_gate, kMustRead));
      func(TENSOR_ARGS(ple_proj, kMustRead));
      func(TENSOR_ARGS(post_ple_ns, kMustRead));
    }

    if (layer_config.ff_biases) {
      func(TENSOR_ARGS(ffw_gating_biases, kMustRead));
      func(TENSOR_ARGS(ffw_output_biases, kMustRead));
    }
  }  // `ForEachTensor`

  // Zero-initializes all allocated tensors in the layer.
  void ZeroInit() {
    ForEachTensor(nullptr, nullptr, [](const TensorArgs& t) {
      if (!t.mat.HasPtr()) return;
      gcpp::ZeroInit(t.mat);
    });
  }

  // Must be called after reading weights via `ForEachTensor`.
  // TODO: exporters should bake this into the weights already.
  // WARNING: called from multiple threads; `mat_owners` requires a lock.
  void Fixup(Model model, std::vector<MatOwner>& mat_owners,
             ThreadingContext& ctx);

 private:
  // Copies att_weights from `attn_vec_einsum_w`.
  void InitAttWeights(std::vector<MatOwner>& mat_owners,
                      const Allocator& allocator);

  // For FFN. Fast, only updates pointers.
  void SplitW1();

  // For attention, which might not have a w2. Fast, only updates pointers.
  void SplitAttW1();
};

struct T5GemmaEncoderLayerWeightsPtrs {
  T5GemmaEncoderLayerWeightsPtrs(size_t layer_idx, const LayerConfig& config,
                                 const TensorInfoRegistry& tensors)
      : layer_idx(layer_idx),
        finder_(LayerSuffix(layer_idx), tensors),
        qkv_einsum_w(finder_("e_qkv")),
        qkv_einsum_w1(finder_("e_qkv1")),
        qkv_einsum_w2(finder_("e_qkv2")),
        attn_vec_einsum_w(finder_("e_att")),
        att_weights(finder_("e_att_w")),
        gating_einsum_w(finder_("e_gate")),
        gating_einsum_w1(finder_("e_gate1")),
        gating_einsum_w2(finder_("e_gate2")),
        linear_w(finder_("e_lin")),
        pre_attention_norm_scale(finder_("e_pre_att")),
        post_attention_norm_scale(finder_("e_post_att")),
        pre_ffw_norm_scale(finder_("e_pre_ff")),
        post_ffw_norm_scale(finder_("e_post_ff")),
        layer_config(config) {}

  const size_t layer_idx;
  const MatFinder finder_;

  MatPtr qkv_einsum_w;
  MatPtr qkv_einsum_w1;
  MatPtr qkv_einsum_w2;
  MatPtr attn_vec_einsum_w;
  MatPtr att_weights;
  MatPtr gating_einsum_w;
  MatPtr gating_einsum_w1;
  MatPtr gating_einsum_w2;
  MatPtr linear_w;
  MatPtr pre_attention_norm_scale;   // at least BF16.
  MatPtr post_attention_norm_scale;  // at least BF16.
  MatPtr pre_ffw_norm_scale;         // at least BF16.
  MatPtr post_ffw_norm_scale;        // at least BF16.

  const LayerConfig& layer_config;

  template <class Func>
  void ForEachTensor(T5GemmaEncoderLayerWeightsPtrs* other1,
                     T5GemmaEncoderLayerWeightsPtrs* other2, Func func) {
    func(TENSOR_ARGS(qkv_einsum_w, kMustRead));
    func(TENSOR_ARGS(qkv_einsum_w1, kMaybeRead));
    func(TENSOR_ARGS(qkv_einsum_w2, kMaybeRead));
    func(TENSOR_ARGS(attn_vec_einsum_w, kMustRead));
    func(TENSOR_ARGS(att_weights, kMaybeRead));
    func(TENSOR_ARGS(gating_einsum_w, kMustRead));
    func(TENSOR_ARGS(gating_einsum_w1, kMaybeRead));
    func(TENSOR_ARGS(gating_einsum_w2, kMaybeRead));
    func(TENSOR_ARGS(linear_w, kMustRead));
    func(TENSOR_ARGS(pre_attention_norm_scale, kMustRead));
    func(TENSOR_ARGS(post_attention_norm_scale, kMustRead));
    func(TENSOR_ARGS(pre_ffw_norm_scale, kMustRead));
    func(TENSOR_ARGS(post_ffw_norm_scale, kMustRead));
  }

  void Fixup(std::vector<MatOwner>& mat_owners, ThreadingContext& ctx);
};

struct T5GemmaDecoderLayerWeightsPtrs {
  T5GemmaDecoderLayerWeightsPtrs(size_t layer_idx, const LayerConfig& config,
                                 const TensorInfoRegistry& tensors)
      : layer_idx(layer_idx),
        finder_(LayerSuffix(layer_idx), tensors),
        self_qkv_einsum_w(finder_("d_qkv")),
        self_qkv_einsum_w1(finder_("d_qkv1")),
        self_qkv_einsum_w2(finder_("d_qkv2")),
        self_attn_vec_einsum_w(finder_("d_att")),
        self_att_weights(finder_("d_att_w")),
        cross_q_einsum_w(finder_("dc_q")),
        cross_k_einsum_w(finder_("dc_k")),
        cross_v_einsum_w(finder_("dc_v")),
        cross_attn_vec_einsum_w(finder_("dc_att")),
        cross_att_weights(finder_("dc_att_w")),
        gating_einsum_w(finder_("d_gate")),
        gating_einsum_w1(finder_("d_gate1")),
        gating_einsum_w2(finder_("d_gate2")),
        linear_w(finder_("d_lin")),
        pre_self_attention_norm_scale(finder_("d_pre_sa")),
        post_self_attention_norm_scale(finder_("d_post_sa")),
        pre_cross_attention_norm_scale(finder_("d_pre_ca")),
        post_cross_attention_norm_scale(finder_("d_post_ca")),
        pre_ffw_norm_scale(finder_("d_pre_ff")),
        post_ffw_norm_scale(finder_("d_post_ff")),
        layer_config(config) {}

  const size_t layer_idx;
  const MatFinder finder_;

  MatPtr self_qkv_einsum_w;
  MatPtr self_qkv_einsum_w1;
  MatPtr self_qkv_einsum_w2;
  MatPtr self_attn_vec_einsum_w;
  MatPtr self_att_weights;
  MatPtr cross_q_einsum_w;
  MatPtr cross_k_einsum_w;
  MatPtr cross_v_einsum_w;
  MatPtr cross_attn_vec_einsum_w;
  MatPtr cross_att_weights;
  MatPtr gating_einsum_w;
  MatPtr gating_einsum_w1;
  MatPtr gating_einsum_w2;
  MatPtr linear_w;
  MatPtr pre_self_attention_norm_scale;    // at least BF16.
  MatPtr post_self_attention_norm_scale;   // at least BF16.
  MatPtr pre_cross_attention_norm_scale;   // at least BF16.
  MatPtr post_cross_attention_norm_scale;  // at least BF16.
  MatPtr pre_ffw_norm_scale;               // at least BF16.
  MatPtr post_ffw_norm_scale;              // at least BF16.

  const LayerConfig& layer_config;

  template <class Func>
  void ForEachTensor(T5GemmaDecoderLayerWeightsPtrs* other1,
                     T5GemmaDecoderLayerWeightsPtrs* other2, Func func) {
    func(TENSOR_ARGS(self_qkv_einsum_w, kMustRead));
    func(TENSOR_ARGS(self_qkv_einsum_w1, kMaybeRead));
    func(TENSOR_ARGS(self_qkv_einsum_w2, kMaybeRead));
    func(TENSOR_ARGS(self_attn_vec_einsum_w, kMustRead));
    func(TENSOR_ARGS(self_att_weights, kMaybeRead));
    func(TENSOR_ARGS(cross_q_einsum_w, kMustRead));
    func(TENSOR_ARGS(cross_k_einsum_w, kMustRead));
    func(TENSOR_ARGS(cross_v_einsum_w, kMustRead));
    func(TENSOR_ARGS(cross_attn_vec_einsum_w, kMustRead));
    func(TENSOR_ARGS(cross_att_weights, kMaybeRead));
    func(TENSOR_ARGS(gating_einsum_w, kMustRead));
    func(TENSOR_ARGS(gating_einsum_w1, kMaybeRead));
    func(TENSOR_ARGS(gating_einsum_w2, kMaybeRead));
    func(TENSOR_ARGS(linear_w, kMustRead));
    func(TENSOR_ARGS(pre_self_attention_norm_scale, kMustRead));
    func(TENSOR_ARGS(post_self_attention_norm_scale, kMustRead));
    func(TENSOR_ARGS(pre_cross_attention_norm_scale, kMustRead));
    func(TENSOR_ARGS(post_cross_attention_norm_scale, kMustRead));
    func(TENSOR_ARGS(pre_ffw_norm_scale, kMustRead));
    func(TENSOR_ARGS(post_ffw_norm_scale, kMustRead));
  }

  void Fixup(std::vector<MatOwner>& mat_owners, ThreadingContext& ctx);
};

// Holds layer-independent weight metadata and pointers plus per-layer
// `LayerWeightsPtrs`. The tensor data is owned by `MatOwner`.
struct WeightsPtrs {
  explicit WeightsPtrs(const ModelConfig& config)
      : config_(config),
        tensors_(config_),
        finder_("", tensors_),  // no suffix because these are per-model.
        embedder_input_embedding(finder_("c_embedding")),
        final_norm_scale(finder_("c_final_norm")),
        lm_head(finder_("lm_head")),
        hc_head_fn(finder_("hc_head_fn")),
        hc_head_base(finder_("hc_head_base")),
        hc_head_scale(finder_("hc_head_scale")),
        mtp_e_proj(finder_("mtp_e_proj")),
        mtp_h_proj(finder_("mtp_h_proj")),
        mtp_enorm(finder_("mtp_enorm")),
        mtp_hnorm(finder_("mtp_hnorm")),
        mtp_norm(finder_("mtp_norm")),
        mtp_hc_fn(finder_("mtp_hc_fn")),
        mtp_hc_base(finder_("mtp_hc_base")),
        mtp_hc_scale(finder_("mtp_hc_scale")),
        t5gemma_encoder_embedding(finder_("enc_embedding")),
        t5gemma_decoder_embedding(finder_("dec_embedding")),
        t5gemma_encoder_final_norm_scale(finder_("enc_final_norm")),
        t5gemma_decoder_final_norm_scale(finder_("dec_final_norm")),
        vit_encoder_norm_bias(finder_("enc_norm_bias")),
        vit_encoder_norm_scale(finder_("enc_norm_scale")),
        vit_img_embedding_bias(finder_("img_emb_bias")),
        vit_img_embedding_kernel(finder_("img_emb_kernel")),
        vit_img_pos_embedding(finder_("img_pos_emb")),
        vit_img_head_bias(finder_("img_head_bias")),
        vit_img_head_kernel(finder_("img_head_kernel")),
        mm_embed_norm(finder_("mm_embed_norm")),
        ple_embeddings(finder_("ple_embeddings")),
        ple_model_proj(finder_("ple_model_proj")),
        ple_proj_norm(finder_("ple_proj_norm")),
        c_layers() {
    if (config_.is_encoder_decoder) {
      t5gemma_encoder_layers.reserve(config_.encoder_layer_configs.size());
      for (size_t idx = 0; idx < config_.encoder_layer_configs.size(); ++idx) {
        const LayerConfig& layer_config = config_.encoder_layer_configs[idx];
        t5gemma_encoder_layers.emplace_back(idx, layer_config, tensors_);
      }
      t5gemma_decoder_layers.reserve(config_.decoder_layer_configs.size());
      for (size_t idx = 0; idx < config_.decoder_layer_configs.size(); ++idx) {
        const LayerConfig& layer_config = config_.decoder_layer_configs[idx];
        t5gemma_decoder_layers.emplace_back(idx, layer_config, tensors_);
      }
    } else {
      c_layers.reserve(config_.layer_configs.size());
      for (size_t idx = 0; idx < config_.layer_configs.size(); ++idx) {
        const LayerConfig& layer_config = config_.layer_configs[idx];
        c_layers.emplace_back(idx, layer_config, tensors_);
      }
    }
    for (size_t idx = 0; idx < config_.vit_config.layer_configs.size(); ++idx) {
      const LayerConfig& layer_config = config_.vit_config.layer_configs[idx];
      vit_layers.emplace_back(idx, layer_config, tensors_);
    }
    if (config_.num_mtp_layers > 0) {
      // The multi-token-prediction block: an extra layer at index
      // `num_layers`, outside the main stack (see `MTPLayerConfig`).
      mtp_layer_config = config_.MTPLayerConfig();
      mtp_layers.emplace_back(config_.layer_configs.size(), mtp_layer_config,
                              tensors_);
    }
  }

  ~WeightsPtrs() = default;

  const ModelConfig& config_;
  // Passed to finder_, hence must be initialized first.
  const TensorInfoRegistry tensors_;
  const MatFinder finder_;

  // TODO: switch to SFP?
  MatPtr embedder_input_embedding;
  MatPtr final_norm_scale;  // at least BF16.
  MatPtr lm_head;           // untied output head (DeepSeek); empty if tied.
  // mHC head collapse (DeepSeek V4; empty unless hc_mult > 1).
  MatPtr hc_head_fn;     // [hc_mult, hc_mult * model_dim] f32
  MatPtr hc_head_base;   // [hc_mult] f32
  MatPtr hc_head_scale;  // [1] f32

  // Multi-token-prediction extras (DeepSeek V4; empty unless
  // num_mtp_layers > 0). The MTP block itself is in `mtp_layers`.
  MatPtr mtp_e_proj;    // [model_dim, model_dim]
  MatPtr mtp_h_proj;    // [model_dim, model_dim]
  MatPtr mtp_enorm;     // [model_dim]
  MatPtr mtp_hnorm;     // [model_dim]
  MatPtr mtp_norm;      // [model_dim] final norm before the shared head
  MatPtr mtp_hc_fn;     // [hc_mult, hc_mult * model_dim] f32
  MatPtr mtp_hc_base;   // [hc_mult] f32
  MatPtr mtp_hc_scale;  // [1] f32

  // T5Gemma text encoder-decoder parts.
  MatPtr t5gemma_encoder_embedding;          // at least BF16.
  MatPtr t5gemma_decoder_embedding;          // at least BF16.
  MatPtr t5gemma_encoder_final_norm_scale;   // at least BF16.
  MatPtr t5gemma_decoder_final_norm_scale;   // at least BF16.

  // Vit parts.
  MatPtr vit_encoder_norm_bias;   // at least BF16.
  MatPtr vit_encoder_norm_scale;  // at least BF16.
  MatPtrT<float> vit_img_embedding_bias;
  MatPtr vit_img_embedding_kernel;  // at least BF16.
  MatPtr vit_img_pos_embedding;     // F32?
  // The head maps from VitConfig::model_dim (Vit final layer) to
  // model_dim (LLM input).
  MatPtrT<float> vit_img_head_bias;
  MatPtr vit_img_head_kernel;  // at least BF16.

  MatPtr mm_embed_norm;  // at least BF16.

  MatPtr ple_embeddings;
  MatPtr ple_model_proj;
  MatPtr ple_proj_norm;

  std::vector<LayerWeightsPtrs> c_layers;
  std::vector<LayerWeightsPtrs> vit_layers;
  // Multi-token-prediction block (0 or 1 entries). `mtp_layer_config` must
  // outlive `mtp_layers`, whose elements hold a reference to it.
  LayerConfig mtp_layer_config;
  std::vector<LayerWeightsPtrs> mtp_layers;
  std::vector<T5GemmaEncoderLayerWeightsPtrs> t5gemma_encoder_layers;
  std::vector<T5GemmaDecoderLayerWeightsPtrs> t5gemma_decoder_layers;

  const LayerWeightsPtrs* GetLayer(size_t layer) const {
    return &c_layers[layer];
  }
  LayerWeightsPtrs* GetLayer(size_t layer) { return &c_layers[layer]; }
  const LayerWeightsPtrs* VitLayer(size_t layer) const {
    return &vit_layers[layer];
  }
  LayerWeightsPtrs* VitLayer(size_t layer) { return &vit_layers[layer]; }

  // Called via `CallT`. `other1` and `other2` are usually null, but can be
  // used to copy from another set of weights. Public because called by tests
  // and `WeightsOwner`.
  template <class Func>
  void ForEachTensor(WeightsPtrs* other1, WeightsPtrs* other2, Func func) {
    LayerWeightsPtrs* other_layer1 = nullptr;
    LayerWeightsPtrs* other_layer2 = nullptr;
    if (config_.is_encoder_decoder) {
      func(TENSOR_ARGS(t5gemma_encoder_embedding, kMustRead));
      func(TENSOR_ARGS(t5gemma_decoder_embedding, kMustRead));
      func(TENSOR_ARGS(t5gemma_encoder_final_norm_scale, kMustRead));
      func(TENSOR_ARGS(t5gemma_decoder_final_norm_scale, kMustRead));

      for (size_t layer_idx = 0; layer_idx < t5gemma_encoder_layers.size();
           ++layer_idx) {
        auto* other_t5_layer1 =
            other1 ? &other1->t5gemma_encoder_layers[layer_idx] : nullptr;
        auto* other_t5_layer2 =
            other2 ? &other2->t5gemma_encoder_layers[layer_idx] : nullptr;
        t5gemma_encoder_layers[layer_idx].ForEachTensor(
            other_t5_layer1, other_t5_layer2, func);
      }

      for (size_t layer_idx = 0; layer_idx < t5gemma_decoder_layers.size();
           ++layer_idx) {
        auto* other_t5_layer1 =
            other1 ? &other1->t5gemma_decoder_layers[layer_idx] : nullptr;
        auto* other_t5_layer2 =
            other2 ? &other2->t5gemma_decoder_layers[layer_idx] : nullptr;
        t5gemma_decoder_layers[layer_idx].ForEachTensor(
            other_t5_layer1, other_t5_layer2, func);
      }
      return;
    }

    func(TENSOR_ARGS(embedder_input_embedding, kMustRead));
    func(TENSOR_ARGS(final_norm_scale, kMustRead));
    if (config_.HasMLA()) {
      func(TENSOR_ARGS(lm_head, kMustRead));
    }
    if (config_.hc_mult > 1) {
      func(TENSOR_ARGS(hc_head_fn, kMustRead));
      func(TENSOR_ARGS(hc_head_base, kMustRead));
      func(TENSOR_ARGS(hc_head_scale, kMustRead));
    }
    if (config_.num_mtp_layers > 0) {
      func(TENSOR_ARGS(mtp_e_proj, kMustRead));
      func(TENSOR_ARGS(mtp_h_proj, kMustRead));
      func(TENSOR_ARGS(mtp_enorm, kMustRead));
      func(TENSOR_ARGS(mtp_hnorm, kMustRead));
      func(TENSOR_ARGS(mtp_norm, kMustRead));
      if (config_.hc_mult > 1) {
        func(TENSOR_ARGS(mtp_hc_fn, kMustRead));
        func(TENSOR_ARGS(mtp_hc_base, kMustRead));
        func(TENSOR_ARGS(mtp_hc_scale, kMustRead));
      }
    }

    if (config_.ple_dim > 0) {
      func(TENSOR_ARGS(ple_embeddings, kMustRead));
      func(TENSOR_ARGS(ple_model_proj, kMustRead));
      func(TENSOR_ARGS(ple_proj_norm, kMustRead));
    }

    if (!config_.vit_config.layer_configs.empty()) {  // Vit parts.
      func(TENSOR_ARGS(vit_encoder_norm_bias, kMustRead));
      func(TENSOR_ARGS(vit_encoder_norm_scale, kMustRead));
      func(TENSOR_ARGS(vit_img_embedding_bias, kMustRead));
      func(TENSOR_ARGS(vit_img_embedding_kernel, kMustRead));
      func(TENSOR_ARGS(vit_img_pos_embedding, kMustRead));
      func(TENSOR_ARGS(vit_img_head_bias, kMustRead));
      func(TENSOR_ARGS(vit_img_head_kernel, kMustRead));

      if (config_.wrapping == PromptWrapping::GEMMA_VLM) {
        func(TENSOR_ARGS(mm_embed_norm, kMustRead));
      }
    }

    for (size_t layer_idx = 0; layer_idx < c_layers.size(); ++layer_idx) {
      if (other1) other_layer1 = other1->GetLayer(layer_idx);
      if (other2) other_layer2 = other2->GetLayer(layer_idx);
      GetLayer(layer_idx)->ForEachTensor(other_layer1, other_layer2, func);
    }

    for (size_t idx = 0; idx < mtp_layers.size(); ++idx) {
      other_layer1 = other1 ? &other1->mtp_layers[idx] : nullptr;
      other_layer2 = other2 ? &other2->mtp_layers[idx] : nullptr;
      mtp_layers[idx].ForEachTensor(other_layer1, other_layer2, func);
    }

    HWY_ASSERT(config_.vit_config.layer_configs.empty() == vit_layers.empty());
    for (size_t layer_idx = 0; layer_idx < vit_layers.size(); ++layer_idx) {
      HWY_ASSERT(vit_layers[layer_idx].layer_config.type ==
                 LayerAttentionType::kVit);
      other_layer1 = other1 ? other1->VitLayer(layer_idx) : nullptr;
      other_layer2 = other2 ? other2->VitLayer(layer_idx) : nullptr;
      VitLayer(layer_idx)->ForEachTensor(other_layer1, other_layer2, func);
    }
  }  // `ForEachTensor`

  // Zero-initializes only the allocated tensors in `*this`.
  void ZeroInit();
  // Copies only the allocated tensors in `*this` from tensors in `other`.
  void CopyFrom(const WeightsPtrs& other);

  enum class Mode {
    // Parallel I/O, decompress to BF16. Best for large batch sizes.
    kReadBF16,
    // Parallel I/O, insert row-wise padding. Safe default.
    kRead,
    // Best for large weights relative to available memory, especially for
    // frequent invocations of small batches and short sequences. Adds noise to
    // performance measurements due to I/O variability.
    kMap
  };

  static const char* ToString(Mode mode) {
    switch (mode) {
      case Mode::kReadBF16:
        return "ReadBF16";
      case Mode::kRead:
        return "Read";
      case Mode::kMap:
        return "Map";
      default:
        HWY_DASSERT(false);
        return "?";
    }
  }

  // Reads tensor data from `BlobStore` or aborts on error. `map` is a user
  // override for whether to map blobs or read them. Returns the mode used.
  Mode ReadFromBlobs(const ModelStore& model, BlobReader& reader,
                     const LoaderArgs& loader, const InferenceArgs& inference,
                     std::vector<MatOwner>& mat_owners, ThreadingContext& ctx);

  // Adds one blob for each tensor's data and returns all serialized MatPtr.
  std::vector<uint32_t> AddTensorDataToWriter(BlobWriter& writer) const;

 private:
  // For reshaping file tensors to the shape expected by the code. This would
  // ideally already happen in the importer. Called by ReadFromBlobs.
  void Fixup(std::vector<MatOwner>& mat_owners, ThreadingContext& ctx);

  MapPtr mapped_;
};  // `WeightsPtrs`
#undef TENSOR_ARGS

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_WEIGHTS_H_
