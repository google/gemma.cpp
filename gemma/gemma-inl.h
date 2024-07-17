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
#include <string.h>  // memcpy

#include <algorithm>
#include <string>
#include <type_traits>
#include <vector>

#include "gemma/activations.h"
#include "gemma/common.h"
#include "gemma/gemma.h"
#include "gemma/ops.h"
#include "gemma/weights.h"
// Placeholder for internal test4, do not remove
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/matvec/matvec-inl.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
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
        /*add0=*/layer_weights->griffin.linear_x_biases.data(),
        /*add1=*/layer_weights->griffin.linear_y_biases.data(), /*out0=*/x,
        /*out1=*/y, pool);
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
          hn::Load(df, layer_weights->griffin.conv_biases.data() + i);
      auto accum1 = hn::Zero(df);
      static_assert(kConv1dWidth % 2 == 0, "Conv width must be even");
      for (size_t l = 0; 2 * l < kConv1dWidth; l++) {
        auto wv0 = hn::Load(df, layer_weights->griffin.conv_w.data() +
                                (kConv1dWidth - 1 - 2 * l) * kModelDim + i);
        auto wv1 = hn::Load(df, layer_weights->griffin.conv_w.data() +
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
          /*add0=*/layer_weights->griffin.gate_biases.data() + head_offset,
          /*add1=*/layer_weights->griffin.gate_biases.data() + kModelDim +
              head_offset,
          /*out0=*/gate_x + head_offset, /*out1=*/a + head_offset);
      Sigmoid(gate_x + head_offset, kHeadDim);
      Sigmoid(a + head_offset, kHeadDim);
      const auto fn_mul = [](D d, hn::Vec<D> x, hn::Vec<D> gate_x)
                          HWY_ATTR { return hn::Mul(x, gate_x); };
      hn::Transform1(D(), a + head_offset, kHeadDim,
                     layer_weights->griffin.a.data() + head_offset, fn_mul);
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
        layer_weights->griffin.linear_out_biases.data(),
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
  MatMul_4x4_Batch<kModelDim, kHeads * kQStride>(
      num_tokens_and_queries, activations.pre_att_rms_out.All(),
      layer_weights->qkv_einsum_w.data(), activations.q.All(), pool);

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
          memcpy(kv, q + kQKVDim, 2 * kQKVDim * sizeof(float));
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
        layer_weights->attention_output_biases.data(),
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
  const auto B1 = layer_weights->gating_einsum_w.data();
  const auto B2 = B1 + kColsA * kColsB;
  auto C1 = activations.C1.All();
  auto C2 = activations.C2.All();
  constexpr bool kAddBias = TConfig::kFFBiases;
  const auto bias = layer_weights->ffw_gating_biases.data();

  // Will go through GELU.
  MatMul_4x4_Batch_Add<kColsA, kColsB, kAddBias>(num_tokens, A, B1, C1,
                                                 bias, pool);
  // What to multiply by.
  MatMul_4x4_Batch_Add<kColsA, kColsB, kAddBias>(num_tokens, A, B2, C2,
                                                 bias + kFFHiddenDim, pool);

  // Activation (Gelu) and multiply by gate. Store activations in C1.
  Activation<TConfig>(C1, C2, kFFHiddenDim * num_tokens);

  // linear_w may have a scale value different from 1, apply that here.
  // We multiply all activations by the scale value to compensate for the
  // missing scale value in the weights.
  if (layer_weights->linear_w.scale() != 1.0f) {
    MulByConst(layer_weights->linear_w.scale(), C1, kFFHiddenDim * num_tokens);
  }

  // Hidden layer -> output layer.
  MatMul_4x4_Batch_Add<kFFHiddenDim, kModelDim, kAddBias>(
      num_tokens, C1, layer_weights->linear_w.data(), activations.ffw_out.All(),
      layer_weights->ffw_output_biases.data(), pool);
}

// TODO: pass Activations.x instead of Activations.
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

template <class TConfig, size_t kQueryBatchSize>
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
                 layer_weights->pre_attention_norm_scale.data(),
                 activations.pre_att_rms_out.All(), kModelDim);
  if (type == LayerAttentionType::kGemma) {
    Attention<TConfig>(pos, num_tokens, num_queries, layer_of_type, activations,
                       layer_weights, kv_caches, pool);
  } else {
    // This Griffin layers should never exist unless the model is a Griffin
    // model. This conditional prevents the compiler from generating code for
    // this branch when building a non-Griffin model, since we have static
    // asserts about the query batch size for Griffin layers.
    if constexpr (TConfig::kGriffinLayers > 0) {
      static_assert(kQueryBatchSize == 1,
                    "Griffin does not support batched queries.");
      GriffinRecurrent<TConfig>(pos, num_tokens, num_queries, layer_of_type,
                                activations, layer_weights, kv_caches, pool);
    }
  }

  if (TConfig::kPostNorm == PostNormType::Scale) {
    RMSNormInplaceBatched(num_tokens_and_queries,
                          layer_weights->post_attention_norm_scale.data(),
                          activations.att_post2.All(), kModelDim);
  }

  ResidualConnection<TConfig>(num_tokens_and_queries,
                              activations.att_post2.All(), activations.x.All(),
                              layer_weights, /*is_attention=*/true);
  RMSNormBatched(num_tokens_and_queries, activations.x.All(),
                 layer_weights->pre_ffw_norm_scale.data(),
                 activations.bf_pre_ffw_rms_out.All(), kModelDim);
  FFW<TConfig>(activations, num_tokens_and_queries, layer_weights, pool);
  if (TConfig::kPostNorm == PostNormType::Scale) {
    RMSNormInplaceBatched(num_tokens_and_queries,
                          layer_weights->post_ffw_norm_scale.data(),
                          activations.ffw_out.All(), kModelDim);
  }
  ResidualConnection<TConfig>(num_tokens_and_queries, activations.ffw_out.All(),
                              activations.x.All(), layer_weights,
                              /*is_attention=*/false);
}

template <class TConfig, size_t kBatchSize, size_t kQueryBatchSize>
HWY_NOINLINE void Prefill(const int* tokens, size_t num_tokens,
                          size_t num_queries, size_t pos,
                          const CompressedWeights<TConfig>& weights,
                          Activations& activations,
                          const std::vector<KVCache*>& kv_caches,
                          hwy::ThreadPool& pool) {
  PROFILER_ZONE("Gen.Prefill");
  HWY_DASSERT(num_queries <= kQueryBatchSize);
  const size_t minibatch_size = std::min(num_tokens, kBatchSize);
  // TODO: hoist pool.Run out of the loop, change the unit of work to batches.
  for (size_t i = 0; i < num_tokens; i += minibatch_size) {
    const size_t offset = i * num_queries;
    const size_t current_token_count = std::min(
        minibatch_size, num_tokens - i);
    pool.Run(0, current_token_count * num_queries,
             [&](const uint64_t token_idx, size_t /*thread*/) HWY_ATTR {
               EmbedToken<TConfig>(tokens[token_idx + offset], token_idx,
                                   pos + offset, weights, activations);
             });

    for (size_t layer = 0; layer < TConfig::kLayers; ++layer) {
      const auto* layer_weights = weights.GetLayer(layer);
      TransformerLayer<TConfig, kQueryBatchSize>(
          current_token_count, num_queries, pos + offset, layer, layer_weights,
          activations, kv_caches, pool);
    }
  }
}

// Compute the transformer for a batch of input tokens. During generation,
// we usually have num_tokens == 1 (and also kBatchSize == 1).
template <class TConfig, size_t kQueryBatchSize>
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
    TransformerLayer<TConfig, kQueryBatchSize>(num_tokens, num_queries, pos,
                                               layer, layer_weights,
                                               activations, kv_caches, pool);

    if (layers_output) {
      const std::string block_name = "blocks." + std::to_string(layer);
      for (size_t token_idx = 0; token_idx < num_tokens_and_queries;
           ++token_idx) {
        layers_output(pos + token_idx, block_name,
                      activations.x.Batch(token_idx), kModelDim);
      }
    }
  }

  RMSNormInplaceBatched(num_tokens_and_queries, weights.final_norm_scale.data(),
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

template <class TConfig, size_t kQueryBatchSize>
void GenerateT(const ByteStorageT& weights_u8, Activations& prefill,
               Activations& activations, const RuntimeConfig& runtime_config,
               const hwy::Span<const hwy::Span<int>>& prompts, size_t pos,
               const size_t query_index_offset,
               const std::vector<KVCache*>& kv_caches, hwy::ThreadPool& pool,
               TimingInfo& timing_info) {
  constexpr size_t kAdjustedPrefillBatchSize =
      std::max((size_t)1, kPrefillBatchSize / kQueryBatchSize);
  static_assert(kAdjustedPrefillBatchSize >= kMinAdjustedPrefillBatchSize);
  const size_t num_queries = prompts.size();
  HWY_DASSERT(num_queries <= kQueryBatchSize);
  pos *= num_queries;  // position in (num_queries) interleaved token sequence.
  const CompressedWeights<TConfig>& weights =
      *reinterpret_cast<const CompressedWeights<TConfig>*>(weights_u8.get());

  size_t min_prompt_size =  (size_t)-1;
  size_t max_prompt_size = 0;
  for (int i=0; i < prompts.size(); ++i) {
    min_prompt_size = std::min(min_prompt_size, prompts[i].size());
    max_prompt_size = std::max(max_prompt_size, prompts[i].size());
  }

  std::vector<int> prompt;
  prompt.reserve(max_prompt_size * prompts.size());
  for (int i = 0; i < max_prompt_size; ++i) {
    for (int j=0; j < prompts.size(); ++j) {
      if (i < prompts[j].size()) {
        prompt.push_back(prompts[j][i]);
      } else {
        prompt.push_back(0);
      }
    }
  }

  constexpr size_t kVocabSize = TConfig::kVocabSize;

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

  std::vector<bool> reached_eos(num_queries);
  std::fill(reached_eos.begin(), reached_eos.end(), false);

  // pos indexes the KV cache. In the first turn of a chat, pos = 0.
  //
  // After the first turn, pos gets passed in with > 0 corresponding to the
  // current token position in the KV cache.
  //
  // pos_offset keeps track of the relative position within the turn, starting
  // at 0 each turn. During prefill, pos_offset corresponds to the index into
  // the prompt vector.
  //
  // In single-turn (non-chat) usage, pos and pos_offset start at 0 and are
  // always equal.
  size_t pos_offset = 0;  // offset relative to pos
  // Used to keep track of how many tokens are processed per prompt,
  // so that we know when to start generating tokens.
  size_t single_prompt_pos_offset = 0;
  const double prefill_start = hwy::platform::Now();

  // Prefill stops before prompt_size - 1 since the last prompt token is the
  // first input token for generation.
  while (single_prompt_pos_offset < min_prompt_size - 1) {
    const size_t batch_size = std::min(
        kPrefillBatchSize, min_prompt_size - 1 - single_prompt_pos_offset);
    const size_t batch_and_query_size = batch_size * num_queries;
    HWY_DASSERT(batch_size <= kPrefillBatchSize);
    HWY_DASSERT(single_prompt_pos_offset + batch_size <= min_prompt_size - 1);
    HWY_DASSERT(pos_offset + batch_size <= (min_prompt_size - 1) * num_queries);
    const int* batch_tokens = prompt.data() + pos_offset;
    Prefill<TConfig, kAdjustedPrefillBatchSize, kQueryBatchSize>(
        batch_tokens, batch_size, num_queries, pos, weights, prefill, kv_caches,
        pool);
    for (size_t idx = 0; idx < batch_size; ++idx) {
      bool all_tokens_eos = true;
      for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
        if (reached_eos[query_idx]) continue;
        if (runtime_config.StreamToken(
                query_idx + query_index_offset, single_prompt_pos_offset,
                batch_tokens[idx * num_queries + query_idx], 0.0f)) {
          all_tokens_eos = false;
        } else {
          reached_eos[query_idx] = true;
        }
      }
      if (all_tokens_eos) {
        return;
      }
    }
    pos += batch_and_query_size;
    pos_offset += batch_and_query_size;
    single_prompt_pos_offset += batch_size;
  }

  timing_info.prefill_tok_sec =
      static_cast<double>(pos_offset) / (hwy::platform::Now() - prefill_start);

  // Start generation.
  const double gen_start = hwy::platform::Now();
  HWY_DASSERT(single_prompt_pos_offset == min_prompt_size - 1);
  size_t pos_gen_start = pos_offset;
  int token = prompt.at(pos_offset);
  std::vector<int>::const_iterator first = prompt.begin() + pos_offset;
  std::vector<int>::const_iterator last = first + num_queries;
  std::vector<int> gen_tokens(first, last);
  // The loop below is not yet prepared for decode batch size > 1.
  HWY_ASSERT(kDecodeBatchSize == 1);
  bool all_tokens_eos = true;
  for (size_t i=0; i < num_queries; ++i) {
    if (reached_eos[i]) continue;
    if (runtime_config.StreamToken(i + query_index_offset,
                                   single_prompt_pos_offset, gen_tokens[i],
                                   0.0f)) {
      all_tokens_eos = false;
    } else {
      reached_eos[i] = true;
    }
  }
  if (all_tokens_eos) {
    return;
  }
  for (size_t generate_pos = 0;
       generate_pos < max_tokens && generate_pos < max_generated_tokens;
       ++single_prompt_pos_offset, ++generate_pos) {
    Transformer<TConfig, kQueryBatchSize>(
        gen_tokens.data(), kDecodeBatchSize, num_queries, pos, weights,
        activations, kv_caches, pool, runtime_config.layers_output);
    float token_logit = 0.0f;
    // The condition below is always true if we are doing Prefill above.
    // We keep it here for clarity so that the code is correct even if Prefill
    // is disabled.
    bool all_tokens_eos = true;
    for (size_t i = 0; i < num_queries; ++i, ++pos, ++pos_offset) {
      const float* HWY_RESTRICT x = activations.x.Batch(i);
      float* HWY_RESTRICT logits = activations.logits.Batch(i);
      const size_t prompt_size = prompts[i].size();
      const bool is_generating_phase =
          (single_prompt_pos_offset >= prompt_size - 1);
      if (is_generating_phase) {
        PROFILER_ZONE("Gen.Embedding");
        // Compute logits from last layer activations.
        MatVec<kVocabSize, TConfig::kModelDim>(weights.embedder_input_embedding,
                                               0, x, activations.even_odd.All(),
                                               logits, pool);
        if constexpr (TConfig::kFinalCap > 0.0f) {
          LogitsSoftCap(TConfig::kFinalCap, logits, kVocabSize);
        }
        // Barrier: must have all logits so we can subtract max.
        Softmax(logits, kVocabSize);
        token = sample_token(logits, kVocabSize);
        token_logit = logits[token];
        if (generate_pos == 0) {
          timing_info.time_to_first_token = hwy::platform::Now() - gen_start;
        }
      } else {
        // We would take this branch if we were not doing Prefill but would
        // process the tokens of the prompt one at a time.
        token = prompt.at(pos_offset);
        token_logit = 0.0f;
      }

      if (!reached_eos[i]) {
        if (!runtime_config.StreamToken(i + query_index_offset,
                                        single_prompt_pos_offset + 1, token,
                                        token_logit)) {
          token = runtime_config.eos_id;
        }
        if (token != runtime_config.eos_id) {
          all_tokens_eos = false;
        } else {
          reached_eos[i] = true;
        }
      }
      gen_tokens[i] = token;
    }
    if (all_tokens_eos) {
      break;
    }
  }
  timing_info.gen_tok_sec = static_cast<double>(pos_offset - pos_gen_start) /
                            (hwy::platform::Now() - gen_start);
}

template <class TConfig>
void GenerateSingleT(const ByteStorageT& weights_u8, Activations& prefill,
                     Activations& activations,
                     const RuntimeConfig& runtime_config,
                     const std::vector<int>& prompt, size_t pos,
                     KVCache& kv_cache, hwy::ThreadPool& pool,
                     TimingInfo& timing_info) {
  // TODO: the input should also be span, not a vector.
  const hwy::Span<int> prompt_span(const_cast<int*>(prompt.data()),
                                   prompt.size());
  const hwy::Span<const hwy::Span<int>> prompts(&prompt_span, 1);
  // TODO: also span of kv_cache.
  std::vector<KVCache*> kv_caches = {&kv_cache};
  const size_t query_index_offset = 0;
  GenerateT<TConfig, /*kQueryBatchSize=*/1>(
      weights_u8, prefill, activations, runtime_config, prompts, pos,
      query_index_offset, kv_caches, pool, timing_info);
}

template <class TConfig>
void GenerateBatchT(const ByteStorageT& weights_u8, Activations& prefill,
                    Activations& activations,
                    const RuntimeConfig& runtime_config,
                    const hwy::Span<const hwy::Span<int>>& prompts, size_t pos,
                    const std::vector<KVCache*>& kv_caches,
                    hwy::ThreadPool& pool, TimingInfo& timing_info) {
  // Disable query batching for Griffin models.
  constexpr size_t kQueryBatchSize =
      (TConfig::kGriffinLayers > 0) ? 1 : kBatchedQueryBatchSize;
  for (size_t i = 0; i < prompts.size(); i += kQueryBatchSize) {
    const size_t num_queries = std::min(prompts.size() - i, kQueryBatchSize);
    const hwy::Span<const hwy::Span<int>> current_prompts(
      prompts.data() + i, num_queries);
    GenerateT<TConfig, kQueryBatchSize>(weights_u8, prefill, activations,
                                        runtime_config, current_prompts, pos, i,
                                        kv_caches, pool, timing_info);
  }
}

}  // namespace HWY_NAMESPACE

#if HWY_ONCE

// These are extern functions defined by instantiations/*.cc, which include this
// 'header' after defining GEMMA_CONFIG, which is for function overloading.
void GenerateSingle(  // NOLINT(misc-definitions-in-headers)
    GEMMA_CONFIG, const ByteStorageT& weights_u8, Activations& prefill,
    Activations& activations, const RuntimeConfig& runtime_config,
    const std::vector<int>& prompt, size_t pos, KVCache& kv_cache,
    hwy::ThreadPool& pool, TimingInfo& timing_info) {
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(GenerateSingleT<GEMMA_CONFIG>)
  (weights_u8, prefill, activations, runtime_config, prompt, pos, kv_cache,
   pool, timing_info);
}

void GenerateBatch(  // NOLINT(misc-definitions-in-headers)
    GEMMA_CONFIG, const ByteStorageT& weights_u8, Activations& prefill,
    Activations& activations, const RuntimeConfig& runtime_config,
    const hwy::Span<const hwy::Span<int>>& prompts, size_t pos,
    const std::vector<KVCache*>& kv_caches, hwy::ThreadPool& pool,
    TimingInfo& timing_info) {
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(GenerateBatchT<GEMMA_CONFIG>)
  (weights_u8, prefill, activations, runtime_config, prompts, pos, kv_caches,
   pool, timing_info);
}

#endif  // HWY_ONCE

}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_INL_H_
