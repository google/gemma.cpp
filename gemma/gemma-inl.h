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

// SIMD functions for Gemma/Griffin transformers.

// Include guard (still compiled once per target)
#if defined(THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_INL_H_
#undef THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_INL_H_
#else
#define THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_INL_H_
#endif

#include <stddef.h>
#include <stdio.h>

#include <algorithm>  // std::min
#include <memory>     // std::unique_ptr
#include <string>
#include <type_traits>
#include <vector>

#include "gemma/activations.h"
#include "gemma/common.h"
#include "gemma/configs.h"
#include "gemma/gemma.h"
#include "gemma/ops.h"
#include "gemma/weights.h"
// Placeholder for internal test4, do not remove
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/bit_set.h"
#include "hwy/contrib/matvec/matvec-inl.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/contrib/thread_pool/topology.h"
#include "hwy/highway.h"
#include "hwy/profiler.h"
#include "hwy/timer.h"

#ifndef GEMMA_CONFIG
#if HWY_IDE
// Provide a definition so the IDE does not complain.
#define GEMMA_CONFIG ConfigGemmaTiny<float>
#else
#error "Only include from instantiations/*.cc, which must define GEMMA_CONFIG"
#endif  // HWY_IDE
#endif  // GEMMA_CONFIG

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

template <class TConfig>
HWY_NOINLINE void GriffinRecurrent(
    size_t batch_start, size_t num_tokens, size_t num_queries, size_t layer,
    Activations& activations, const CompressedLayer<TConfig>* layer_weights,
    const std::vector<KVCache*>& kv_caches, hwy::ThreadPool& pool) {
  PROFILER_ZONE("Gen.Griffin");
  HWY_ASSERT(num_queries == 1);  // TODO: add batch query support for Griffin.
  KVCache& kv_cache = *kv_caches[0];
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kConv1dWidth = TConfig::kConv1dWidth;
  static constexpr size_t kHeads = TConfig::kHeads;

  // X / Y linear layers.
  for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
    float* HWY_RESTRICT y = activations.griffin_y.Batch(batch_idx);
    float* HWY_RESTRICT x = activations.griffin_x.Batch(batch_idx);
    TwoMatVecAdd<kModelDim, kModelDim>(
        layer_weights->griffin.linear_x_w, layer_weights->griffin.linear_y_w, 0,
        activations.pre_att_rms_out.Batch(batch_idx),
        /*add0=*/layer_weights->griffin.linear_x_biases.data_scale1(),
        /*add1=*/layer_weights->griffin.linear_y_biases.data_scale1(),
        /*out0=*/x, /*out1=*/y, pool);
    Gelu(y, kModelDim);
  }

  // Conv1D.
  for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
    const size_t pos = batch_start + batch_idx;
    float* HWY_RESTRICT x = activations.griffin_x.Batch(batch_idx);
    HWY_FULL(float) df;
    HWY_DASSERT(kModelDim % hn::Lanes(df) == 0);
    const size_t layer_offset = layer * kModelDim * (kConv1dWidth - 1);

    // cache[i] = input at time t-i.
    float* HWY_RESTRICT cache[HWY_MAX(kConv1dWidth, 1)];
    cache[0] = x;
    for (size_t i = 1; i < kConv1dWidth; i++) {
      cache[i] =
          kv_cache.conv1d_cache.get() + layer_offset +
          ((pos + kConv1dWidth - 1 - i) % (kConv1dWidth - 1)) * kModelDim;
    }
    for (size_t i = 0; i < kModelDim; i += hn::Lanes(df)) {
      auto xv = hn::Load(df, x + i);
      auto accum0 =
          hn::Load(df, layer_weights->griffin.conv_biases.data_scale1() + i);
      auto accum1 = hn::Zero(df);
      static_assert(kConv1dWidth % 2 == 0, "Conv width must be even");
      for (size_t l = 0; 2 * l < kConv1dWidth; l++) {
        auto wv0 = hn::Load(df, layer_weights->griffin.conv_w.data_scale1() +
                                (kConv1dWidth - 1 - 2 * l) * kModelDim + i);
        auto wv1 = hn::Load(df, layer_weights->griffin.conv_w.data_scale1() +
                                (kConv1dWidth - 2 - 2 * l) * kModelDim + i);
        accum0 = hn::MulAdd(wv0, hn::Load(df, cache[l * 2] + i), accum0);
        accum1 = hn::MulAdd(wv1, hn::Load(df, cache[l * 2 + 1] + i), accum1);
      }
      hn::Store(hn::Add(accum0, accum1), df, x + i);
      hn::Store(xv, df, cache[HWY_MAX(kConv1dWidth, 1) - 1] + i);
    }
  }

  // RGLRU
  for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
    const size_t pos = batch_start + batch_idx;
    float* HWY_RESTRICT y = activations.griffin_y.Batch(batch_idx);
    float* HWY_RESTRICT x = activations.griffin_x.Batch(batch_idx);
    float* HWY_RESTRICT gate_x = activations.griffin_gate_x.Batch(batch_idx);
    float* HWY_RESTRICT a = activations.griffin_multiplier.Batch(batch_idx);
    float* HWY_RESTRICT rnn_state =
        kv_cache.rglru_cache.get() + layer * kModelDim;

    pool.Run(0, kHeads, [&](const uint64_t head, size_t /*thread*/) HWY_ATTR {
      constexpr size_t kHeadDim = kModelDim / kHeads;
      constexpr size_t kMatrixSize = kHeadDim * kHeadDim;
      size_t head_offset = head * kHeadDim;
      TwoOfsMatVecAddLoop<kHeadDim, kHeadDim>(
          layer_weights->griffin.gate_w, kMatrixSize * head,
          kMatrixSize * (kHeads + head), x + head_offset,
          /*add0=*/layer_weights->griffin.gate_biases.data_scale1() +
              head_offset,
          /*add1=*/layer_weights->griffin.gate_biases.data_scale1() +
              kModelDim + head_offset,
          /*out0=*/gate_x + head_offset, /*out1=*/a + head_offset);
      Sigmoid(gate_x + head_offset, kHeadDim);
      Sigmoid(a + head_offset, kHeadDim);
      const auto fn_mul = [](D d, hn::Vec<D> x, hn::Vec<D> gate_x)
                          HWY_ATTR { return hn::Mul(x, gate_x); };
      hn::Transform1(D(), a + head_offset, kHeadDim,
                     layer_weights->griffin.a.data_scale1() + head_offset,
                     fn_mul);
      hn::Transform1(D(), x + head_offset, kHeadDim, gate_x + head_offset,
                     fn_mul);
      // RNN scan
      HWY_FULL(float) df;
      HWY_DASSERT(kHeadDim % hn::Lanes(df) == 0);
      for (size_t i = 0; i < kHeadDim; i += hn::Lanes(df)) {
        auto log_a = hn::Load(df, a + head_offset + i);
        auto gated_x = hn::Load(df, x + head_offset + i);
        auto rnn = hn::Load(df, rnn_state + head_offset + i);
        auto a = hn::Exp(df, log_a);
        auto x_multiplier = hn::Sqrt(hn::NegMulAdd(a, a, hn::Set(df, 1.0f)));
        if (pos == 0) {
          x_multiplier = hn::Set(df, 1.0f);
        }
        auto new_x = hn::MulAdd(x_multiplier, gated_x, hn::Mul(a, rnn));
        hn::Store(new_x, df, rnn_state + head_offset + i);

        // Join branches.
        auto yv = hn::Load(df, y + head_offset + i);
        auto pre_out = hn::Mul(yv, new_x);
        hn::Store(pre_out, df, x + head_offset + i);
      }
    });
  }

  // Final linear layer.
  for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
    float* HWY_RESTRICT x = activations.griffin_x.Batch(batch_idx);
    float* out_ptr = activations.att_post2.Batch(batch_idx);
    MatVecAdd<kModelDim, kModelDim>(
        layer_weights->griffin.linear_out_w, 0, x,
        layer_weights->griffin.linear_out_biases.data_scale1(),
        activations.even_odd.All(), out_ptr, pool);
  }
}

template <class TConfig, typename T>
HWY_NOINLINE void PostQK(T* HWY_RESTRICT t, size_t pos, size_t layer) {
  constexpr size_t kQKVDim = TConfig::kQKVDim;
  // PostQKType::Rope
  Rope(t, TConfig::kUseHalfRope ? kQKVDim / 2 : kQKVDim, pos);
}

template <class TConfig>
HWY_NOINLINE void Attention(size_t batch_and_query_start, size_t num_tokens,
                            size_t num_queries, size_t layer,
                            Activations& activations,
                            const CompressedLayer<TConfig>* layer_weights,
                            const std::vector<KVCache*>& kv_caches,
                            hwy::ThreadPool& pool) {
  PROFILER_ZONE("Gen.Attention");
  HWY_DASSERT(batch_and_query_start % num_queries == 0);
  constexpr size_t kQKVDim = TConfig::kQKVDim;
  constexpr size_t kQStride = Activations::QStride<TConfig>();
  constexpr size_t kCachePosSize = CachePosSize<TConfig>()();
  constexpr size_t kCacheLayerSize = CacheLayerSize<TConfig>()();
  constexpr size_t kModelDim = TConfig::kModelDim;
  constexpr size_t kHeads = TConfig::kHeads;
  constexpr size_t kKVHeads = TConfig::kKVHeads;
  constexpr size_t kSeqLen = TConfig::kSeqLen;
  GEMMA_CONSTEXPR_SQRT float kQueryScale = ChooseQueryScale<TConfig>();
  // Multi-Head Attention a.k.a. "use_qkv_einsum".
  constexpr bool kIsMHA = Activations::IsMHA<TConfig>();
  static_assert(!kIsMHA || TConfig::kInterleaveQKV);  // MHA => interleaved
  const size_t batch_start = batch_and_query_start / num_queries;
  const size_t num_tokens_and_queries = num_tokens * num_queries;

  // For the computation of Q, K, and V, it is useful to remember that
  // qkv_einsum_w has shape [(kHeads + kKVHeads * 2), kKQVDim, kModelDim]
  // and kQStride = kQKVDim * (kIsMHA ? 3 : 1);
  //
  // Compute Q only or QKV (if MHA).
  // If MHA, this also computes KV, which we copy to the KV cache below.
  const float scale = layer_weights->qkv_einsum_w.scale();
  MatMul_4x4_Batch<kModelDim, kHeads * kQStride>(
      num_tokens_and_queries, activations.pre_att_rms_out.All(),
      layer_weights->qkv_einsum_w.data(), scale, activations.q.All(), pool);

  // Compute KV if not MHA.
  if constexpr (!kIsMHA) {
    for (size_t batch_and_query_idx = 0;
         batch_and_query_idx < num_tokens_and_queries; ++batch_and_query_idx) {
      const float* x = activations.pre_att_rms_out.Batch(batch_and_query_idx);
      const size_t query_idx = batch_and_query_idx % num_queries;
      const size_t batch_idx = batch_and_query_idx / num_queries;
      KVCache& kv_cache = *kv_caches[query_idx];
      const size_t pos = batch_start + batch_idx;
      const size_t cache_pos = pos % (kSeqLen + kPrefillBatchSize);
      const size_t kv_offset =
          cache_pos * kCachePosSize + layer * kCacheLayerSize;
      float* HWY_RESTRICT kv = kv_cache.kv_cache.get() + kv_offset;
      // KV structure is [k, v, k, v, ....] = kKVHeads pairs of (k, v).
      // TODO: requires MatMul support for offsets.
      MatVec<kKVHeads * 2 * kQKVDim, kModelDim>(
          layer_weights->qkv_einsum_w, kHeads * kQKVDim * kModelDim, x,
          activations.even_odd.All(), kv, pool);
    }
  }

  // Apply positional encodings for K (and copy KV to cache if MHA).
  pool.Run(
      0, kKVHeads * num_tokens_and_queries,
      [&](uint64_t task, size_t thread) HWY_ATTR {
        const size_t head = task % kKVHeads;
        const size_t batch_and_query_idx = task / kKVHeads;
        const size_t query_idx = batch_and_query_idx % num_queries;
        const size_t batch_idx = batch_and_query_idx / num_queries;
        const size_t pos = batch_start + batch_idx;
        const size_t cache_pos = pos % (kSeqLen + kPrefillBatchSize);
        const size_t kv_offset = cache_pos * kCachePosSize +
                                 layer * kCacheLayerSize + head * kQKVDim * 2;
        KVCache& kv_cache = *kv_caches[query_idx];
        float* HWY_RESTRICT kv = kv_cache.kv_cache.get() + kv_offset;
        if constexpr (kIsMHA) {
          // For MHA, copy KV into the KV cache from scratch space (see above).
          const float* HWY_RESTRICT q =
              activations.q.Batch(batch_and_query_idx) + head * kQStride;
          // Skip past the Q part of `q`, and copy KV to `kv`.
          hwy::CopyBytes(q + kQKVDim, kv, 2 * kQKVDim * sizeof(float));
        }
        PostQK<TConfig>(kv, pos, layer);
      });

  static_assert((kHeads % kKVHeads) == 0,
                "query heads must be a multiple of key-value heads");
  constexpr size_t kGroupHeads = kHeads / kKVHeads;
  // For each head (token, query), compute Q.K, softmax, and weighted V.
  pool.Run(0, kHeads * num_tokens_and_queries,
           [&](uint64_t task, size_t thread) HWY_ATTR {
    const size_t head = task % kHeads;
    const size_t batch_and_query_idx = task / kHeads;
    const size_t query_idx = batch_and_query_idx % num_queries;
    const size_t batch_idx = batch_and_query_idx / num_queries;
    const size_t head_offset = (head / kGroupHeads) * kQKVDim * 2;
    KVCache& kv_cache = *kv_caches[query_idx];
    float* HWY_RESTRICT q =
        activations.q.Batch(batch_and_query_idx) + head * kQStride;

    // Apply rope and scaling to Q.
    const size_t pos = batch_start + batch_idx;
    PostQK<TConfig>(q, pos, layer);
    MulByConst(kQueryScale, q, kQKVDim);

    // Compute Q.K scores, yielding "logits" (or scores) in head_att.
    float* HWY_RESTRICT head_att =
        activations.att.Batch(batch_and_query_idx) + head * kSeqLen;
    const size_t start_pos =
        pos - std::min(TConfig::kAttentionWindowSizes[layer] - 1, pos);
    for (size_t pos2 = start_pos; pos2 <= pos; ++pos2) {
      const size_t cache_pos = pos2 % (kSeqLen + kPrefillBatchSize);
      const size_t kv_offset =
          cache_pos * kCachePosSize + layer * kCacheLayerSize + head_offset;
      const float* HWY_RESTRICT k2 = kv_cache.kv_cache.get() + kv_offset;
      const float score = Dot(q, k2, kQKVDim);
      head_att[pos2 % kSeqLen] = score;
    }

    // SoftMax. May be preceded by SoftCap. Yields "probabilities" in head_att.
    const size_t head_att_len = std::min(pos + 1, kSeqLen);
    if constexpr (TConfig::kAttCap > 0.0f) {
      LogitsSoftCap(TConfig::kAttCap, head_att, head_att_len);
    }
    Softmax(head_att, head_att_len);

    // Summation of v (kv_cache) weighted by probs (head_att)
    // into "encoded" (att_out). Compare gemma/modules.py:
    // encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)
    float* HWY_RESTRICT att_out =
        activations.att_out.Batch(batch_and_query_idx) + head * kQKVDim;
    hwy::ZeroBytes(att_out, kQKVDim * sizeof(*att_out));
    for (size_t pos2 = start_pos; pos2 <= pos; ++pos2) {
      const size_t cache_pos = pos2 % (kSeqLen + kPrefillBatchSize);
      const size_t kv_offset =
          cache_pos * kCachePosSize + layer * kCacheLayerSize + head_offset;
      float* HWY_RESTRICT v2 = kv_cache.kv_cache.get() + kv_offset + kQKVDim;
      MulByConstAndAdd(head_att[pos2 % kSeqLen], v2, att_out, kQKVDim);
    }
  });

  // Sum encoded (att_out) over num_heads and head_dim (kQKVDim)
  // into output (layer_out). Compare gemma/modules.py:
  // attn_output = self.attn_vec_einsum('BTNH,NHD->BTD', encoded)
  for (size_t batch_and_query_idx = 0;
       batch_and_query_idx < num_tokens_and_queries; ++batch_and_query_idx) {
    // TODO(szabadka) Use a single MatVecAdd like in GriffinRecurrent() after
    // rearranging the weights.
    float* HWY_RESTRICT att_out =
        activations.att_out.Batch(batch_and_query_idx);
    float* HWY_RESTRICT layer_out =
        activations.att_post2.Batch(batch_and_query_idx);
    // Head 0 (and potentially biases) -> layer_out.
    // attn_vec_einsum_w has shape [kHeads, kQKVDim, kModelDim].
    MatVecT</*kAdd=*/TConfig::kSoftmaxAttnOutputBiases, kModelDim, kQKVDim>(
        layer_weights->attn_vec_einsum_w, 0, att_out,
        layer_weights->attention_output_biases.data_scale1(),
        activations.even_odd.All(), layer_out, pool);
    // Head 1 and following are added to layer_out.
    for (size_t head = 1; head < kHeads; ++head) {
      // NOTE: this is a single kModelDim temp output. If parallelized or using
      // MatMul, add per-thread storage.
      float* HWY_RESTRICT head_out = activations.att_post1.All();
      // TODO: requires MatMul support for offsets.
      MatVec<kModelDim, kQKVDim>(
          layer_weights->attn_vec_einsum_w, head * kModelDim * kQKVDim,
          att_out + head * kQKVDim, activations.even_odd.All(), head_out, pool);
      AddFrom(head_out, layer_out, kModelDim);
    }
  }
}

template <class TConfig, typename T>
HWY_NOINLINE void Activation(T* HWY_RESTRICT c1, T* HWY_RESTRICT c2,
                             size_t count) {
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<T>;
  using VF = hn::Vec<DF>;
  // ActivationType::Gelu
  hn::Transform1(DF(), c1, count, c2, [](DF df, VF v, VF mul) HWY_ATTR {
    return hn::Mul(mul, Gelu(df, v));
  });
}

template <class TConfig>
HWY_NOINLINE void FFW(Activations& activations, size_t num_tokens,
                      const CompressedLayer<TConfig>* layer_weights,
                      hwy::ThreadPool& pool) {
  PROFILER_ZONE("Gen.FFW");
  constexpr size_t kModelDim = TConfig::kModelDim;
  constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;

  // MatMul expects col-major B, which is what we have: kModelDim consecutive
  // elements in memory, repeated kFFHiddenDim times.
  constexpr size_t kColsA = kModelDim;
  constexpr size_t kColsB = kFFHiddenDim;
  HWY_DASSERT(num_tokens <= activations.bf_pre_ffw_rms_out.BatchSize());
  const auto A = activations.bf_pre_ffw_rms_out.All();
  const float scale = layer_weights->gating_einsum_w.scale();
  const auto B1 = layer_weights->gating_einsum_w.data();
  const auto B2 = B1 + kColsA * kColsB;
  auto C1 = activations.C1.All();
  auto C2 = activations.C2.All();
  constexpr bool kAddBias = TConfig::kFFBiases;
  const auto bias1 = layer_weights->ffw_gating_biases.data_scale1();
  const auto bias2 = bias1 + kFFHiddenDim;

  // Will go through GELU.
  MatMul_4x4_Batch_Add<kColsA, kColsB, kAddBias>(num_tokens, A, B1, scale, C1,
                                                 bias1, pool);
  // What to multiply by.
  MatMul_4x4_Batch_Add<kColsA, kColsB, kAddBias>(num_tokens, A, B2, scale, C2,
                                                 bias2, pool);

  // Activation (Gelu) and multiply by gate. Store activations in C1.
  Activation<TConfig>(C1, C2, kFFHiddenDim * num_tokens);

  // Hidden layer -> output layer.
  MatMul_4x4_Batch_Add<kFFHiddenDim, kModelDim, kAddBias>(
      num_tokens, C1, layer_weights->linear_w.data(),
      layer_weights->linear_w.scale(), activations.ffw_out.All(),
      layer_weights->ffw_output_biases.data_scale1(), pool);
}

// TODO: pass Activations.x instead of Activations.
// `pos` is for the entire batch and does not include `batch_idx`.
template <class TConfig>
HWY_NOINLINE void EmbedToken(int token, size_t batch_idx, size_t pos,
                             const CompressedWeights<TConfig>& weights,
                             Activations& activations) {
  constexpr size_t kModelDim = TConfig::kModelDim;
  GEMMA_CONSTEXPR_EMBSCALING const float kEmbScaling =
      EmbeddingScaling<TConfig>();

  HWY_DASSERT(token >= 0);
  HWY_DASSERT(token < TConfig::kVocabSize);

  Decompress(weights.embedder_input_embedding, token * kModelDim,
             activations.x.Batch(batch_idx), kModelDim);
  MulByConst(kEmbScaling, activations.x.Batch(batch_idx), kModelDim);
  if constexpr (TConfig::kAbsolutePE) {
    AddAbsolutePositionalEmbeddings(activations.x.Batch(batch_idx), kModelDim,
                                    pos + batch_idx);
  };
}

template <class TConfig, typename T>
HWY_NOINLINE void ResidualConnection(
    size_t num_tokens_and_queries, T* HWY_RESTRICT other, T* HWY_RESTRICT x,
    const CompressedLayer<TConfig>* layer_weights, bool is_attention) {
  constexpr size_t kModelDim = TConfig::kModelDim;
  // ResidualType::Add
  AddFromBatched(num_tokens_and_queries, other, x, kModelDim);
}

template <class TConfig>
HWY_NOINLINE void TransformerLayer(
    size_t num_tokens, size_t num_queries, size_t pos, size_t layer,
    const CompressedLayer<TConfig>* layer_weights, Activations& activations,
    const std::vector<KVCache*>& kv_caches, hwy::ThreadPool& pool) {
  constexpr size_t kModelDim = TConfig::kModelDim;
  const size_t num_tokens_and_queries = num_tokens * num_queries;
  auto type = TConfig::kLayerConfig[layer];
  size_t layer_of_type =
      NumLayersOfTypeBefore(TConfig::kLayerConfig, type, layer);
  RMSNormBatched(num_tokens_and_queries, activations.x.All(),
                 layer_weights->pre_attention_norm_scale.data_scale1(),
                 activations.pre_att_rms_out.All(), kModelDim);
  if (type == LayerAttentionType::kGemma) {
    Attention<TConfig>(pos, num_tokens, num_queries, layer_of_type, activations,
                       layer_weights, kv_caches, pool);
  } else {
    // Only reached if the model is Griffin. `if constexpr` prevents generating
    // this code for non-Griffin models.
    if constexpr (TConfig::kGriffinLayers > 0) {
      HWY_ASSERT(num_queries == 1);
      GriffinRecurrent<TConfig>(pos, num_tokens, num_queries, layer_of_type,
                                activations, layer_weights, kv_caches, pool);
    }
  }

  if (TConfig::kPostNorm == PostNormType::Scale) {
    RMSNormInplaceBatched(
        num_tokens_and_queries,
        layer_weights->post_attention_norm_scale.data_scale1(),
        activations.att_post2.All(), kModelDim);
  }

  ResidualConnection<TConfig>(num_tokens_and_queries,
                              activations.att_post2.All(), activations.x.All(),
                              layer_weights, /*is_attention=*/true);
  RMSNormBatched(num_tokens_and_queries, activations.x.All(),
                 layer_weights->pre_ffw_norm_scale.data_scale1(),
                 activations.bf_pre_ffw_rms_out.All(), kModelDim);
  FFW<TConfig>(activations, num_tokens_and_queries, layer_weights, pool);
  if (TConfig::kPostNorm == PostNormType::Scale) {
    RMSNormInplaceBatched(num_tokens_and_queries,
                          layer_weights->post_ffw_norm_scale.data_scale1(),
                          activations.ffw_out.All(), kModelDim);
  }
  ResidualConnection<TConfig>(num_tokens_and_queries, activations.ffw_out.All(),
                              activations.x.All(), layer_weights,
                              /*is_attention=*/false);
}

// For prefill, we have two-level parallelism:
// - Outer: input tokens are split into batches, each of which is one task
//   processed by a worker in `outer_pool_`, which includes the main thread
//   because it is the one that calls `Prefill`.
// - Inner: each `outer` worker passes `inner_pools_[outer]` to
//   `TransformerLayer` for tensor-level parallelism.
//
// This class holds the thread pools and activations, recreated for each query.
//
// It is safe to parallelize batches because we write to KVCache at
// `pos % kSeqLen`, which is far greater than the number of workers.
// Note however that this currently leads to nondeterministic results because
// the RNG is invoked in different order.
class PrefillState {
 public:
  explicit PrefillState(hwy::ThreadPool& main_pool) : main_pool_(&main_pool) {}

  ~PrefillState() { DeleteInnerPools(); }

  // Called before each query. Recreates thread pools, which has the advantage
  // of tailoring the parallelism to the prompt length.
  template <class TConfig>
  void Init(size_t prefill_size) {
    // Would be zero for single-token prompts (prefill_size == num_tokens - 1).
    num_batches_ =
        HWY_MAX(size_t{1}, hwy::DivCeil(prefill_size, kPrefillBatchSize));
    // More than num_batches_ would waste workers on idling in the outer Run;
    // more than NumWorkers() would exceed the global --num_threads.
    const size_t outer_workers =
        HWY_MIN(num_batches_, main_pool_->NumWorkers());
    HWY_ASSERT(outer_workers != 0);  // Otherwise activations_ is empty.

    // One activation per outer worker. Allocating in parallel saves 30 ms.
    activations_.resize(outer_workers);
    main_pool_->Run(0, outer_workers, [this](uint64_t task, size_t /*thread*/) {
      activations_[task].Allocate<TConfig>(kPrefillBatchSize);
    });

    DeleteInnerPools();

    // If we'd create just one inner pool with all the workers, skip the cost of
    // thread creation and pinning (about 60 ms) by reusing the main pool.
    if (outer_workers <= 1) {
      // Still allocate a dummy pool to simplify Prefill().
      outer_pool_ = std::make_unique<hwy::ThreadPool>(1);
      inner_pools_.push_back(main_pool_);
      return;
    }

    // Before creating new threads, stop the old ones from spinning. Caller is
    // responsible for undoing this by calling `ResumeMainSpinning`.
    main_pool_->SetWaitMode(hwy::PoolWaitMode::kBlock);
    outer_pool_ = std::make_unique<hwy::ThreadPool>(outer_workers);
    outer_pool_->SetWaitMode(hwy::PoolWaitMode::kSpin);

    // Assign up to `max_workers` to inner pools. Each inner pool creates
    // `workers_per_outer - 1` threads in addition to its 'main' thread, which
    // is the one calling `inner_pools[outer]->Run`, i.e., `outer`. In total,
    // `outer_workers * (max_workers / outer_workers)` workers are used.
    const size_t workers_per_outer = main_pool_->NumWorkers() / outer_workers;
    for (size_t outer = 0; outer < outer_workers; ++outer) {
      inner_pools_.push_back(new hwy::ThreadPool(workers_per_outer));
      inner_pools_.back()->SetWaitMode(hwy::PoolWaitMode::kSpin);
    }

    PinThreads(outer_workers, workers_per_outer);
  }

  // `tokens` are from interleaved queries. (See InterleaveQueries() below.)
  template <class TConfig>
  HWY_NOINLINE void Prefill(hwy::Span<const int> tokens, size_t num_queries,
                            size_t pos,
                            const CompressedWeights<TConfig>& weights,
                            const RuntimeConfig& runtime_config,
                            const std::vector<KVCache*>& kv_caches) {
    PROFILER_ZONE("Gen.Prefill");

    HWY_ASSERT(activations_.size() == outer_pool_->NumWorkers());
    HWY_ASSERT(inner_pools_.size() == outer_pool_->NumWorkers());

    outer_pool_->Run(
        0, num_batches_, [&](const uint64_t batch_num, size_t thread) HWY_ATTR {
          const size_t batch_start = batch_num * kPrefillBatchSize;
          const size_t batch_size =
              HWY_MIN(kPrefillBatchSize, tokens.size() - batch_start);
          HWY_DASSERT(batch_start % num_queries == 0);
          HWY_DASSERT(batch_size % num_queries == 0);
          const size_t pos_per_query = pos + batch_start / num_queries;
          const size_t num_tokens = batch_size / num_queries;

          // Negligible time compared to TransformerLayer.
          for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            EmbedToken<TConfig>(tokens[batch_start + batch_idx], batch_idx,
                                pos_per_query, weights, activations_[thread]);
          }

          for (size_t layer = 0; layer < TConfig::kLayers; ++layer) {
            const auto* layer_weights = weights.GetLayer(layer);
            TransformerLayer<TConfig>(
                num_tokens, num_queries, pos_per_query, layer, layer_weights,
                activations_[thread], kv_caches, *inner_pools_[thread]);
          }

          // NOTE: we unconditionally call StreamToken, even if EOS.
          for (size_t i = 0; i < batch_size; ++i) {
            const size_t query_idx = i % num_queries;
            const size_t batch_idx = i / num_queries;
            runtime_config.StreamToken(query_idx, pos_per_query + batch_idx,
                                       tokens[i], 0.0f);
          }
        });
  }

  // Stops spinning in our pools and resume spinning in main_pool_.
  void ResumeMainSpinning() {
    // If we didn't create a new inner pool, we didn't stop spinning on the
    // main pool, so nothing to do here.
    if (inner_pools_[0] == main_pool_) return;

    for (hwy::ThreadPool* p : inner_pools_) {
      p->SetWaitMode(hwy::PoolWaitMode::kBlock);
    }
    outer_pool_->SetWaitMode(hwy::PoolWaitMode::kBlock);
    main_pool_->SetWaitMode(hwy::PoolWaitMode::kSpin);
  }

 private:
  // Pins each outer thread after their inner threads so they are likely to
  // run on the same socket.
  void PinThreads(size_t outer_workers, size_t workers_per_outer) {
    outer_pool_->Run(
        0, outer_workers,
        [this, workers_per_outer](uint64_t outer, size_t outer_thread) {
          HWY_ASSERT(outer == outer_thread);
          // Pins inner *and* `outer` - the latter is the calling thread.
          inner_pools_[outer]->Run(
              0, workers_per_outer,
              [outer, workers_per_outer](uint64_t task, size_t thread) {
                HWY_ASSERT(task == thread);  // each worker has one task
                const size_t lp = outer * workers_per_outer + task;
                hwy::PinThreadToLogicalProcessor(lp);
              });
        });
  }

  void DeleteInnerPools() {
    for (hwy::ThreadPool* p : inner_pools_) {
      if (p != main_pool_) delete p;
    }
    inner_pools_.clear();
  }

  hwy::ThreadPool* main_pool_;
  std::unique_ptr<hwy::ThreadPool> outer_pool_;  // always allocated
  std::vector<Activations> activations_;  // size == outer_pool->NumWorkers()
  // Either there is a single pointer equal to main_pool, or newly created pools
  // that we own. The former case avoids thread creation overhead for prompts
  // that fit in a single batch.
  std::vector<hwy::ThreadPool*> inner_pools_;
  size_t num_batches_ = 0;
};

// `tokens` is length `num_tokens * num_queries`. In autoregressive decode,
// `num_tokens == 1`.
template <class TConfig>
HWY_NOINLINE void Transformer(const int* tokens, size_t num_tokens,
                              size_t num_queries, size_t pos,
                              const CompressedWeights<TConfig>& weights,
                              Activations& activations,
                              const std::vector<KVCache*>& kv_caches,
                              hwy::ThreadPool& pool,
                              const LayersOutputFunc& layers_output) {
  const size_t num_tokens_and_queries = num_tokens * num_queries;
  if (layers_output) {
    for (size_t token_idx = 0; token_idx < num_tokens_and_queries;
         ++token_idx) {
      float token_f = tokens[token_idx];
      layers_output(pos + token_idx, "Tokens", &token_f, 1);
    }
  }
  constexpr size_t kModelDim = TConfig::kModelDim;
  for (size_t token_idx = 0; token_idx < num_tokens_and_queries; ++token_idx) {
    EmbedToken<TConfig>(tokens[token_idx], token_idx, pos, weights,
                        activations);
  }

  for (size_t layer = 0; layer < TConfig::kLayers; ++layer) {
    const CompressedLayer<TConfig>* layer_weights = weights.GetLayer(layer);
    TransformerLayer<TConfig>(num_tokens, num_queries, pos, layer,
                              layer_weights, activations, kv_caches, pool);

    if (layers_output) {
      const std::string block_name = "blocks." + std::to_string(layer);
      for (size_t token_idx = 0; token_idx < num_tokens_and_queries;
           ++token_idx) {
        layers_output(pos + token_idx, block_name,
                      activations.x.Batch(token_idx), kModelDim);
      }
    }
  }

  RMSNormInplaceBatched(num_tokens_and_queries,
                        weights.final_norm_scale.data_scale1(),
                        activations.x.All(), kModelDim);
  if (layers_output) {
    for (size_t token_idx = 0; token_idx < num_tokens_and_queries;
         ++token_idx) {
      layers_output(pos + token_idx, "final_norm",
                    activations.x.Batch(token_idx), kModelDim);
    }
  }
}

template <class TConfig>
void RangeChecks(size_t& max_tokens, size_t& max_generated_tokens,
                 size_t& prompt_size) {
  if (!TConfig::kUseLocalAttention) {
    if (max_tokens > TConfig::kSeqLen) {
      fprintf(stderr, "WARNING: max_tokens %zu > kSeqLen %d, truncating.\n",
              max_tokens, TConfig::kSeqLen);
      max_tokens = static_cast<size_t>(TConfig::kSeqLen);
    }
  }

  if (max_generated_tokens > max_tokens) {
    fprintf(stderr,
            "WARNING: max_generated_tokens %zu > max_tokens %zu, truncating.\n",
            max_generated_tokens, max_tokens);
    max_generated_tokens = max_tokens - 1;
  }

  if (!TConfig::kUseLocalAttention) {
    if (prompt_size + max_generated_tokens > max_tokens) {
      fprintf(stderr,
              "WARNING: prompt_size %zu + max_generated_tokens %zu > "
              "max_tokens %zu, truncating to ",
              prompt_size, max_generated_tokens, max_tokens);
      prompt_size = std::min(prompt_size, max_tokens - max_generated_tokens);
      fprintf(stderr, "%zu\n", prompt_size);
    }
  }

  HWY_ASSERT(prompt_size > 0);
}

// Placeholder for internal test3, do not remove

// Returns interleaved tokens: one from each query, followed by the second from
// all queries, with EOS padding.
static std::vector<int> InterleaveQueries(
    const hwy::Span<const hwy::Span<int>>& queries,
    const RuntimeConfig& runtime_config, size_t& min_prompt_size,
    size_t& max_prompt_size) {
  const size_t num_queries = queries.size();
  min_prompt_size = hwy::LimitsMax<size_t>();
  max_prompt_size = 0;
  for (size_t i = 0; i < num_queries; ++i) {
    min_prompt_size = std::min(min_prompt_size, queries[i].size());
    max_prompt_size = std::max(max_prompt_size, queries[i].size());
  }

  std::vector<int> prompt;
  prompt.reserve(max_prompt_size * num_queries);
  for (size_t pos = 0; pos < max_prompt_size; ++pos) {
    for (size_t q = 0; q < num_queries; ++q) {
      if (pos < queries[q].size()) {
        prompt.push_back(queries[q][pos]);
      } else {
        prompt.push_back(runtime_config.eos_id);
      }
    }
  }
  return prompt;
}

// Holds "is at end of stream" state for each query.
class TokenStreamer {
 public:
  explicit TokenStreamer(const RuntimeConfig& runtime_config)
      : runtime_config_(runtime_config) {}

  // Returns whether the query was already at, or has just reached, the end of
  // the stream: either via token == eos_id, or StreamToken returning false.
  bool operator()(size_t query_idx, size_t pos, int token, float prob) {
    if (HWY_UNLIKELY(is_eos_.Get(query_idx))) return true;

    if (!runtime_config_.StreamToken(query_idx, pos, token, prob) ||
        token == runtime_config_.eos_id) {
      is_eos_.Set(query_idx);
      return true;
    }

    return false;
  }

 private:
  const RuntimeConfig& runtime_config_;
  // BitSet4096 divides the arg by 64, so ensure it is at least 64.
  hwy::BitSet4096<HWY_MAX(64, kBatchedQueryBatchSize)> is_eos_;
};

// Generates one token per query in the batch.
//
// pos indexes the KV cache. In the first turn of a chat, pos = 0, and it
// continues to increase by one for each prefilled/generated token per query.
// query_idx_start is the first query index in the batch.
template <class TConfig, size_t kQueryBatchSize>
void GenerateT(const ByteStorageT& weights_u8, Activations& activations,
               const RuntimeConfig& runtime_config,
               const hwy::Span<const hwy::Span<int>>& prompts, const size_t pos,
               const size_t query_idx_start,
               const std::vector<KVCache*>& kv_caches, hwy::ThreadPool& pool,
               TimingInfo& timing_info) {
  constexpr size_t kVocabSize = TConfig::kVocabSize;
  const CompressedWeights<TConfig>& weights =
      *reinterpret_cast<const CompressedWeights<TConfig>*>(weights_u8.get());

  const size_t num_queries = prompts.size();
  HWY_DASSERT(num_queries <= kQueryBatchSize);
  size_t min_prompt_size, max_prompt_size;
  const std::vector<int> prompt = InterleaveQueries(
      prompts, runtime_config, min_prompt_size, max_prompt_size);

  size_t max_tokens = runtime_config.max_tokens;
  size_t max_generated_tokens = runtime_config.max_generated_tokens;
  RangeChecks<TConfig>(max_tokens, max_generated_tokens, max_prompt_size);
  if (pos >= max_tokens) {
    fprintf(stderr, "Warning: pos %zu >= max_tokens %zu, aborting.\n", pos,
            max_tokens);
    return;
  }

  // If no sample_func is provided, we use top-k sampling.
  const SampleFunc sample_token =
      runtime_config.sample_func
          ? runtime_config.sample_func
          : [&](const float* logits, size_t vocab_size) -> int {
    return SampleTopK<TConfig::kTopK>(logits, vocab_size, *runtime_config.gen,
                                      runtime_config.temperature,
                                      runtime_config.accept_token);
  };

  // Prefill stops before min_prompt_size - 1 because the last prompt token is
  // the first input token for generation.
  const size_t prefill_per_query = min_prompt_size - 1;
  const hwy::Span<const int> prefill_tokens(prompt.data(),
                                            prefill_per_query * num_queries);
  PrefillState prefill(pool);
  prefill.Init<TConfig>(prefill_tokens.size());
  const double prefill_start = hwy::platform::Now();
  size_t interleaved_pos = pos * num_queries;
  prefill.Prefill<TConfig>(prefill_tokens, num_queries, interleaved_pos,
                           weights, runtime_config, kv_caches);
  interleaved_pos += prefill_tokens.size();
  timing_info.NotifyPrefill(prefill_tokens.size(), prefill_start);

  prefill.ResumeMainSpinning();

  // Storage for the last generated token from each query, passed to the next
  // Transformer() call.
  std::vector<int> gen_tokens(num_queries);

  // Stream the last prompt token from each query and fill gen_tokens.
  hwy::CopyBytes(&prompt[prefill_tokens.size()], gen_tokens.data(),
                 num_queries * sizeof(prompt[0]));
  TokenStreamer token_streamer(runtime_config);
  for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
    (void)token_streamer(query_idx_start + query_idx, prefill_per_query,
                         gen_tokens[query_idx], 0.0f);
  }

  const double gen_start = hwy::platform::Now();
  for (size_t gen_per_query = 0;
       gen_per_query < HWY_MIN(max_tokens, max_generated_tokens);
       ++gen_per_query) {
    // Decode: generate one token for each query.
    Transformer<TConfig>(gen_tokens.data(), /*num_tokens=*/1, num_queries,
                         interleaved_pos, weights, activations, kv_caches, pool,
                         runtime_config.layers_output);
    interleaved_pos += num_queries;

    bool all_queries_eos = true;
    PROFILER_ZONE("Gen.Embedding");
    for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
      float* HWY_RESTRICT logits = activations.logits.Batch(query_idx);
      // Compute logits from last layer activations. TODO: MatMul
      MatVec<kVocabSize, TConfig::kModelDim>(
          weights.embedder_input_embedding, 0, activations.x.Batch(query_idx),
          activations.even_odd.All(), logits, pool);
      if constexpr (TConfig::kFinalCap > 0.0f) {
        LogitsSoftCap(TConfig::kFinalCap, logits, kVocabSize);
      }
      Softmax(logits, kVocabSize);
      const int token = sample_token(logits, kVocabSize);
      timing_info.NotifyGenerated(prefill_start);

      const bool is_eos = token_streamer(query_idx_start + query_idx,
                                         prefill_per_query + 1 + gen_per_query,
                                         token, logits[token]);
      all_queries_eos &= is_eos;
      gen_tokens[query_idx] = is_eos ? runtime_config.eos_id : token;
    }
    if (all_queries_eos) break;
  }  // foreach token to generate

  timing_info.NotifyGenerateDone(gen_start);
}

// TODO: prompt should also be span, not a vector.
template <class TConfig>
void GenerateSingleT(const ByteStorageT& weights_u8, Activations& activations,
                     const RuntimeConfig& runtime_config,
                     const std::vector<int>& prompt, size_t pos,
                     KVCache& kv_cache, hwy::ThreadPool& pool,
                     TimingInfo& timing_info) {
  const hwy::Span<int> prompt_span(const_cast<int*>(prompt.data()),
                                   prompt.size());
  const hwy::Span<const hwy::Span<int>> prompts(&prompt_span, 1);
  // TODO: also span of kv_cache, or batching inside KVCache?
  std::vector<KVCache*> kv_caches = {&kv_cache};
  const size_t query_idx_start = 0;
  GenerateT<TConfig, /*kQueryBatchSize=*/1>(
      weights_u8, activations, runtime_config, prompts, pos, query_idx_start,
      kv_caches, pool, timing_info);
}

template <class TConfig>
void GenerateBatchT(const ByteStorageT& weights_u8, Activations& activations,
                    const RuntimeConfig& runtime_config,
                    const hwy::Span<const hwy::Span<int>>& prompts, size_t pos,
                    const std::vector<KVCache*>& kv_caches,
                    hwy::ThreadPool& pool, TimingInfo& timing_info) {
  // Disable query batching for Griffin models.
  constexpr size_t kQueryBatchSize =
      (TConfig::kGriffinLayers > 0) ? 1 : kBatchedQueryBatchSize;
  for (size_t query_idx_start = 0; query_idx_start < prompts.size();
       query_idx_start += kQueryBatchSize) {
    const size_t num_queries =
        std::min(prompts.size() - query_idx_start, kQueryBatchSize);
    const hwy::Span<const hwy::Span<int>> query_batch(
        prompts.data() + query_idx_start, num_queries);
    GenerateT<TConfig, kQueryBatchSize>(weights_u8, activations, runtime_config,
                                        query_batch, pos, query_idx_start,
                                        kv_caches, pool, timing_info);
  }
}

}  // namespace HWY_NAMESPACE

#if HWY_ONCE

// These are extern functions defined by instantiations/*.cc, which include this
// 'header' after defining GEMMA_CONFIG, which is for function overloading.
void GenerateSingle(  // NOLINT(misc-definitions-in-headers)
    GEMMA_CONFIG, const ByteStorageT& weights_u8, Activations& activations,
    const RuntimeConfig& runtime_config, const std::vector<int>& prompt,
    size_t pos, KVCache& kv_cache, hwy::ThreadPool& pool,
    TimingInfo& timing_info) {
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(GenerateSingleT<GEMMA_CONFIG>)
  (weights_u8, activations, runtime_config, prompt, pos, kv_cache, pool,
   timing_info);
}

void GenerateBatch(  // NOLINT(misc-definitions-in-headers)
    GEMMA_CONFIG, const ByteStorageT& weights_u8, Activations& activations,
    const RuntimeConfig& runtime_config,
    const hwy::Span<const hwy::Span<int>>& prompts, size_t pos,
    const std::vector<KVCache*>& kv_caches, hwy::ThreadPool& pool,
    TimingInfo& timing_info) {
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(GenerateBatchT<GEMMA_CONFIG>)
  (weights_u8, activations, runtime_config, prompts, pos, kv_caches, pool,
   timing_info);
}

#endif  // HWY_ONCE

}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_INL_H_
