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
#include "gemma/weights.h"
// Placeholder for internal test4, do not remove
#include "ops/matmul-inl.h"
#include "ops/ops-inl.h"
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

// Different functions use different naming conventions for the number of
// tokens. Functions that are query-independent, such as RMSNorm*, call the
// count `num_interleaved`. Functions that are query-dependent, such as
// `Attention`, use separate `num_tokens` and `num_queries`.

template <class TConfig>
HWY_NOINLINE void GriffinRecurrent(
    size_t batch_start, size_t num_tokens, size_t num_queries, size_t layer,
    Activations& activations, const CompressedLayer<TConfig>* layer_weights,
    const KVCaches& kv_caches, hwy::ThreadPool& pool) {
  PROFILER_ZONE("Gen.Griffin");
  HWY_ASSERT(num_queries == 1);  // TODO: add batch query support for Griffin.
  KVCache& kv_cache = kv_caches[0];
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
HWY_NOINLINE void PostQK(T* HWY_RESTRICT inout, size_t pos, size_t layer) {
  constexpr size_t kQKVDim = TConfig::kQKVDim;
  // PostQKType::Rope
  Rope(inout, TConfig::kUseHalfRope ? kQKVDim / 2 : kQKVDim, pos);
}

template <class TConfig>
HWY_NOINLINE void GemmaAttention(size_t interleaved_start, size_t num_tokens,
                                 size_t num_queries, size_t layer,
                                 Activations& activations,
                                 const CompressedLayer<TConfig>* layer_weights,
                                 const KVCaches& kv_caches,
                                 hwy::ThreadPool& pool) {
  PROFILER_ZONE("Gen.Attention");
  HWY_DASSERT(interleaved_start % num_queries == 0);
  constexpr size_t kQKVDim = TConfig::kQKVDim;
  constexpr size_t kQStride = Activations::QStride<TConfig>();
  constexpr size_t kCachePosSize = CachePosSize<TConfig>()();
  constexpr size_t kCacheLayerSize = CacheLayerSize<TConfig>()();
  constexpr size_t kModelDim = TConfig::kModelDim;
  constexpr size_t kHeads = TConfig::kHeads;
  constexpr size_t kKVHeads = TConfig::kKVHeads;
  constexpr size_t kSeqLen = TConfig::kSeqLen;
  GEMMA_CONSTEXPR_SQRT float kQueryScale = ChooseQueryScale<TConfig>();

  HWY_ASSERT(num_queries <= kv_caches.size());
  const hwy::Divisor div_seq_len(static_cast<uint32_t>(kv_caches[0].seq_len));

  // Multi-Head Attention a.k.a. "use_qkv_einsum".
  constexpr bool kIsMHA = Activations::IsMHA<TConfig>();
  static_assert(!kIsMHA || TConfig::kInterleaveQKV);  // MHA => interleaved
  const size_t batch_start = interleaved_start / num_queries;
  const size_t num_interleaved = num_tokens * num_queries;

  // For the computation of Q, K, and V, it is useful to remember that
  // qkv_einsum_w has shape [(kHeads + kKVHeads * 2), kKQVDim, kModelDim]
  // and kQStride = kQKVDim * (kIsMHA ? 3 : 1);
  //
  // Compute Q only or QKV (if MHA).
  // If MHA, this also computes KV, which we copy to the KV cache below.
  MatMul_4x4</*kAdd=*/false>(
      num_interleaved, MakeMat(activations.pre_att_rms_out.All(), kModelDim),
      MakeMat(layer_weights->qkv_einsum_w.data(), kModelDim),
      layer_weights->qkv_einsum_w.scale(), /*add=*/nullptr,
      MakeMat(activations.q.All(), kHeads * kQStride), pool);

  // Compute KV if not MHA.
  if constexpr (!kIsMHA) {
    // Single query and no wraparound means we can use a matmul and write
    // directly into the KV cache with a stride of kCachePosSize.
    if (num_queries == 1 &&
        batch_start + num_tokens <= div_seq_len.GetDivisor()) {
      const size_t kv_ofs =
          batch_start * kCachePosSize + layer * kCacheLayerSize;
      // KV structure is [k, v, k, v, ....] = kKVHeads pairs of (k, v).
      float* HWY_RESTRICT kv = kv_caches[0].kv_cache.get() + kv_ofs;
      MatMul_4x4</*kAdd=*/false>(
          num_tokens, MakeMat(activations.pre_att_rms_out.All(), kModelDim),
          MakeMat(layer_weights->qkv_einsum_w.data(), kModelDim, kModelDim,
                  kHeads * kQKVDim * kModelDim),
          layer_weights->qkv_einsum_w.scale(), /*add=*/nullptr,
          MakeMat(kv, kKVHeads * 2 * kQKVDim, kCachePosSize), pool);
    } else {
      for (size_t interleaved_idx = 0; interleaved_idx < num_interleaved;
           ++interleaved_idx) {
        const float* x = activations.pre_att_rms_out.Batch(interleaved_idx);
        const size_t query_idx = interleaved_idx % num_queries;
        const size_t batch_idx = interleaved_idx / num_queries;
        KVCache& kv_cache = kv_caches[query_idx];
        const size_t cache_pos = div_seq_len.Remainder(batch_start + batch_idx);
        const size_t kv_offset =
            cache_pos * kCachePosSize + layer * kCacheLayerSize;
        float* HWY_RESTRICT kv = kv_cache.kv_cache.get() + kv_offset;
        // KV structure is [k, v, k, v, ....] = kKVHeads pairs of (k, v).
        MatVec<kKVHeads * 2 * kQKVDim, kModelDim>(
            layer_weights->qkv_einsum_w, kHeads * kQKVDim * kModelDim, x,
            activations.even_odd.All(), kv, pool);
      }
    }
  }

  // Apply positional encodings for K (and copy KV to cache if MHA).
  pool.Run(
      0, kKVHeads * num_interleaved,
      [&](uint64_t task, size_t /*thread*/) HWY_ATTR {
        const size_t head = task % kKVHeads;
        const size_t interleaved_idx = task / kKVHeads;
        const size_t query_idx = interleaved_idx % num_queries;
        const size_t batch_idx = interleaved_idx / num_queries;
        const size_t pos = batch_start + batch_idx;
        const size_t cache_pos = div_seq_len.Remainder(pos);
        const size_t kv_offset = cache_pos * kCachePosSize +
                                 layer * kCacheLayerSize + head * kQKVDim * 2;
        KVCache& kv_cache = kv_caches[query_idx];
        float* HWY_RESTRICT kv = kv_cache.kv_cache.get() + kv_offset;
        if constexpr (kIsMHA) {
          // For MHA, copy KV into the KV cache from scratch space (see above).
          const float* HWY_RESTRICT q =
              activations.q.Batch(interleaved_idx) + head * kQStride;
          // Skip past the Q part of `q`, and copy KV to `kv`.
          hwy::CopyBytes(q + kQKVDim, kv, 2 * kQKVDim * sizeof(float));
        }
        PostQK<TConfig>(kv, pos, layer);
      });

  // A "head group" in the context of GQA refers to a collection of query heads
  // that share the same key and value heads.
  static_assert((kHeads % kKVHeads) == 0,
                "query heads must be a multiple of key-value heads");
  constexpr size_t kHeadGroups = kHeads / kKVHeads;
  // For each head (token, query), compute Q.K, softmax, and weighted V.
  pool.Run(
      0, kHeads * num_interleaved,
      [&](uint64_t task, size_t /*thread*/) HWY_ATTR {
        const size_t head = task % kHeads;
        const size_t interleaved_idx = task / kHeads;
        const size_t query_idx = interleaved_idx % num_queries;
        const size_t batch_idx = interleaved_idx / num_queries;
        const size_t head_offset = (head / kHeadGroups) * kQKVDim * 2;
        KVCache& kv_cache = kv_caches[query_idx];
        float* HWY_RESTRICT q =
            activations.q.Batch(interleaved_idx) + head * kQStride;

        // Apply rope and scaling to Q.
        const size_t pos = batch_start + batch_idx;
        PostQK<TConfig>(q, pos, layer);
        MulByConst(kQueryScale, q, kQKVDim);

        // Compute Q.K scores, yielding "logits" (or scores) in head_att.
        float* HWY_RESTRICT head_att =
            activations.att.Batch(interleaved_idx) + head * kSeqLen;
        // Usually start_pos is 0, unless pos is larger than the attention
        // window size, then it is pos - window_size + 1.
        const size_t start_pos =
            pos - std::min(TConfig::kAttentionWindowSizes[layer] - 1, pos);
        for (size_t pos2 = start_pos; pos2 <= pos; ++pos2) {
          const size_t cache_pos = div_seq_len.Remainder(pos2);
          const size_t kv_offset =
              cache_pos * kCachePosSize + layer * kCacheLayerSize + head_offset;
          const float* HWY_RESTRICT k = &kv_cache.kv_cache[kv_offset];
          const float score = Dot(q, k, kQKVDim);
          head_att[pos2 % kSeqLen] = score;
        }

        // SoftMax. May be preceded by SoftCap. Yields "probabilities" in
        // head_att.
        const size_t head_att_len = std::min(pos + 1, kSeqLen);
        if constexpr (TConfig::kAttCap > 0.0f) {
          LogitsSoftCap(TConfig::kAttCap, head_att, head_att_len);
        }
        Softmax(head_att, head_att_len);

        // Summation of v (kv_cache) weighted by probs (head_att)
        // into "encoded" (att_out). Compare gemma/modules.py:
        // encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)
        float* HWY_RESTRICT att_out =
            activations.att_out.Batch(interleaved_idx) + head * kQKVDim;
        hwy::ZeroBytes(att_out, kQKVDim * sizeof(*att_out));
        for (size_t pos2 = start_pos; pos2 <= pos; ++pos2) {
          const size_t cache_pos = div_seq_len.Remainder(pos2);
          const size_t kv_offset =
              cache_pos * kCachePosSize + layer * kCacheLayerSize + head_offset;
          float* HWY_RESTRICT v =
              kv_cache.kv_cache.get() + kv_offset + kQKVDim;
          MulByConstAndAdd(head_att[pos2 % kSeqLen], v, att_out, kQKVDim);
        }
      });

  // Sum encoded (att_out) over num_heads and head_dim (kQKVDim)
  // into output (layer_out). Compare gemma/modules.py:
  // attn_output = self.attn_vec_einsum('BTNH,NHD->BTD', encoded)
  for (size_t interleaved_idx = 0; interleaved_idx < num_interleaved;
       ++interleaved_idx) {
    // TODO(szabadka) Use a single MatVecAdd like in GriffinRecurrent() after
    // rearranging the weights.
    float* HWY_RESTRICT att_out = activations.att_out.Batch(interleaved_idx);
    float* HWY_RESTRICT layer_out =
        activations.att_post2.Batch(interleaved_idx);
    // Head 0 (and potentially biases) -> layer_out.
    // attn_vec_einsum_w has shape [kHeads, kQKVDim, kModelDim].
    constexpr bool kAdd = TConfig::kSoftmaxAttnOutputBiases;
    const float* bias =
        kAdd ? layer_weights->attention_output_biases.data_scale1() : nullptr;
    MatVecT<kAdd, kModelDim, kQKVDim>(
        layer_weights->attn_vec_einsum_w, 0, att_out, bias,
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

template <class TConfig>
HWY_NOINLINE void Attention(LayerAttentionType type, size_t interleaved_start,
                            size_t num_tokens, size_t num_queries, size_t layer,
                            Activations& activations,
                            const CompressedLayer<TConfig>* layer_weights,
                            const KVCaches& kv_caches, hwy::ThreadPool& pool) {
  if (type == LayerAttentionType::kGemma) {
    GemmaAttention<TConfig>(interleaved_start, num_tokens, num_queries, layer,
                            activations, layer_weights, kv_caches, pool);
  } else {
    // Only reached if the model is Griffin. `if constexpr` prevents generating
    // this code for non-Griffin models.
    if constexpr (TConfig::kGriffinLayers > 0) {
      HWY_ASSERT(num_queries == 1);
      GriffinRecurrent<TConfig>(interleaved_start, num_tokens, num_queries,
                                layer, activations, layer_weights, kv_caches,
                                pool);
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
HWY_NOINLINE void FFW(Activations& activations, size_t num_interleaved,
                      const CompressedLayer<TConfig>* layer_weights,
                      hwy::ThreadPool& pool) {
  PROFILER_ZONE("Gen.FFW");
  constexpr size_t kModelDim = TConfig::kModelDim;
  constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;

  // MatMul expects col-major B, which is what we have: kModelDim consecutive
  // elements in memory, repeated kFFHiddenDim times.
  HWY_DASSERT(num_interleaved <= activations.bf_pre_ffw_rms_out.BatchSize());
  const auto A = MakeMat(activations.bf_pre_ffw_rms_out.All(), kModelDim);
  const auto B1 = MakeMat(layer_weights->gating_einsum_w.data(), kModelDim);
  const auto B2 = MakeMat(layer_weights->gating_einsum_w.data(), kModelDim,
                          kModelDim, kModelDim * kFFHiddenDim);
  const float scale = layer_weights->gating_einsum_w.scale();
  constexpr bool kAddBias = TConfig::kFFBiases;
  const float* bias1 = nullptr;
  const float* bias2 = nullptr;
  const float* output_bias = nullptr;
  if constexpr (kAddBias) {
    bias1 = layer_weights->ffw_gating_biases.data_scale1();
    bias2 = bias1 + kFFHiddenDim;
    output_bias = layer_weights->ffw_output_biases.data_scale1();
  }
  auto C1 = MakeMat(activations.C1.All(), kFFHiddenDim);
  auto C2 = MakeMat(activations.C2.All(), kFFHiddenDim);

  // Will go through GELU.
  MatMul_4x4<kAddBias>(num_interleaved, A, B1, scale, bias1, C1, pool);
  // What to multiply by.
  MatMul_4x4<kAddBias>(num_interleaved, A, B2, scale, bias2, C2, pool);

  // Activation (Gelu) and multiply by gate. Store activations in C1.
  Activation<TConfig>(C1.ptr, C2.ptr, kFFHiddenDim * num_interleaved);

  // Hidden layer -> output layer.
  MatMul_4x4<kAddBias>(num_interleaved, C1,
                       MakeMat(layer_weights->linear_w.data(), kFFHiddenDim),
                       layer_weights->linear_w.scale(), output_bias,
                       MakeMat(activations.ffw_out.All(), kModelDim), pool);
}

// `batch_idx` indicates which row of `x` to write to.
// `pos` is the *token*'s position, not the start of the batch, because this is
// called for batches of tokens in prefill, but batches of queries in decode.
template <class TConfig>
HWY_NOINLINE void EmbedToken(int token, size_t batch_idx, size_t pos,
                             const CompressedWeights<TConfig>& weights,
                             RowVectorBatch<float>& x) {
  constexpr size_t kModelDim = TConfig::kModelDim;
  GEMMA_CONSTEXPR_EMBSCALING const float kEmbScaling =
      EmbeddingScaling<TConfig>();

  HWY_DASSERT(token >= 0);
  HWY_DASSERT(token < TConfig::kVocabSize);

  Decompress(weights.embedder_input_embedding, token * kModelDim,
             x.Batch(batch_idx), kModelDim);
  MulByConst(kEmbScaling, x.Batch(batch_idx), kModelDim);
  if constexpr (TConfig::kAbsolutePE) {
    AddAbsolutePositionalEmbeddings(x.Batch(batch_idx), kModelDim, pos);
  };
}

template <class TConfig, typename T>
HWY_NOINLINE void ResidualConnection(
    size_t num_interleaved, T* HWY_RESTRICT other, T* HWY_RESTRICT x,
    const CompressedLayer<TConfig>* layer_weights, bool is_attention) {
  constexpr size_t kModelDim = TConfig::kModelDim;
  // ResidualType::Add
  AddFromBatched(num_interleaved, other, x, kModelDim);
}

template <class TConfig, typename WeightT, typename InOutT>
void PostNorm(size_t num_interleaved, const WeightT& weights, InOutT* inout) {
  if (TConfig::kPostNorm == PostNormType::Scale) {
    RMSNormInplaceBatched(num_interleaved, weights.data_scale1(), inout,
                          TConfig::kModelDim);
  }
}

template <class TConfig>
HWY_NOINLINE void TransformerLayer(
    size_t num_tokens, size_t num_queries, size_t pos, size_t layer,
    const CompressedLayer<TConfig>* layer_weights, Activations& activations,
    const KVCaches& kv_caches, hwy::ThreadPool& pool) {
  constexpr size_t kModelDim = TConfig::kModelDim;
  const size_t num_interleaved = num_tokens * num_queries;
  auto type = TConfig::kLayerConfig[layer];
  size_t layer_of_type =
      NumLayersOfTypeBefore(TConfig::kLayerConfig, type, layer);

  RMSNormBatched(num_interleaved, activations.x.All(),
                 layer_weights->pre_attention_norm_scale.data_scale1(),
                 activations.pre_att_rms_out.All(), kModelDim);

  Attention<TConfig>(type, pos, num_tokens, num_queries, layer_of_type,
                     activations, layer_weights, kv_caches, pool);

  PostNorm<TConfig>(num_interleaved, layer_weights->post_attention_norm_scale,
                    activations.att_post2.All());

  ResidualConnection<TConfig>(num_interleaved, activations.att_post2.All(),
                              activations.x.All(), layer_weights,
                              /*is_attention=*/true);

  RMSNormBatched(num_interleaved, activations.x.All(),
                 layer_weights->pre_ffw_norm_scale.data_scale1(),
                 activations.bf_pre_ffw_rms_out.All(), kModelDim);

  FFW<TConfig>(activations, num_interleaved, layer_weights, pool);

  PostNorm<TConfig>(num_interleaved, layer_weights->post_ffw_norm_scale,
                    activations.ffw_out.All());

  ResidualConnection<TConfig>(num_interleaved, activations.ffw_out.All(),
                              activations.x.All(), layer_weights,
                              /*is_attention=*/false);
}

// Batches are important for amortizing loading weights over multiple tokens.
// This is possible in prefill because we know all tokens beforehand, whereas
// decode depends on the previous output token. However, each prefill batch of a
// query requires that preceding batches already wrote to the KV cache, hence we
// sequentially loop over token batches. We can reduce the number of iterations
// by increasing the batch size, but this also increases arithmetic intensity,
// and so we are eventually compute-limited. The tensor parallelism (number of
// threads collaborating on MatMul) is also limited by the CPU topology:
// fork/join barriers are slow(er) when some threads reside in a different NUMA
// node. To allow more threads to help, we also support parallelizing over
// queries in case GenerateBatch was called.
//
// Thus we have two-level parallelism:
// - Outer: handles one 'qbatch' of entire queries. The set of outer workers
//   includes the main thread because it is the one that calls `Prefill`, and is
//   determined by the number of 'clusters' (shared L3 caches or sockets).
// - Inner: each `outer` worker passes `inner_pools_[outer]` to
//   `TransformerLayer` for tensor-level parallelism, and processes
//   `tbatch_size` tokens from a single query at a time.
//
// This class holds the thread pools and one activation per outer worker. It is
// NOT reused across calls to GenerateSingle/GenerateBatch so that we can adapt
// to their num_queries.
class PrefillState {
  // TODO: move helper functions, also those in app.h, to a threading header
  using LPS = hwy::LogicalProcessorSet;
  LPS Intersection(const LPS& big_set, const LPS& small_set) {
    LPS both_set;
    // Reduce expected work by iterating over the smaller set.
    small_set.Foreach([&big_set, &both_set](size_t idx) {
      if (big_set.Get(idx)) both_set.Set(idx);
    });
    return both_set;
  }

  std::vector<size_t> CoresInLPS(const LPS& cluster) {
    std::vector<size_t> cores;
    cores.reserve(cluster.Count());
    cluster.Foreach([&cores](size_t idx) { cores.push_back(idx); });
    return cores;
  }

  // For each cluster (shared L3 cache), a bitset of cores.
  using CoresPerCluster = std::vector<LPS>;

  // Returns empty if detection failed.
  CoresPerCluster DetectClusters() {
    CoresPerCluster clusters;
    // Which processors are not disabled via OS, taskset, or numactl.
    LPS enabled;
    // If we don't know, better to use just a single inner pool rather than risk
    // oversubscribing to enabled cores.
    if (!GetThreadAffinity(enabled)) return clusters;

    hwy::Topology topology;
    if (topology.packages.empty()) return clusters;

    // For each cluster = outer, the cores that will be used for an inner pool.
    CoresPerCluster inner_lps;
    for (const hwy::Topology::Package& package : topology.packages) {
      for (const hwy::Topology::Cluster& cluster : package.clusters) {
        // Only use enabled cores, and only add if not empty.
        const LPS lps = Intersection(enabled, cluster.lps);
        if (lps.Any()) clusters.push_back(lps);
      }
    }

    // Sort by descending number of enabled cores, so that we preferentially
    // use the largest clusters.
    std::sort(clusters.begin(), clusters.end(),
              [](const LPS& a, const LPS& b) { return a.Count() > b.Count(); });

    return clusters;
  }

  // Returns false if the main pool should be reused instead.
  bool AssignInnerPoolsToClusters(const size_t num_queries) {
    HWY_ASSERT(num_queries != 0);

    CoresPerCluster inner_lps = DetectClusters();
    // If we have more outer workers than queries, discard the excess.
    if (inner_lps.size() > num_queries) inner_lps.resize(num_queries);
    // If we're not going to create multiple pools, avoid the overhead of
    // re-pinning (60 ms) and reuse the main pool.
    if (inner_lps.size() <= 1) return false;

    // Before creating new threads, stop the old ones from spinning. Caller is
    // responsible for undoing this by calling `ResumeMainSpinning`.
    main_pool_->SetWaitMode(hwy::PoolWaitMode::kBlock);

    outer_pool_ = std::make_unique<hwy::ThreadPool>(inner_lps.size());
    outer_pool_->SetWaitMode(hwy::PoolWaitMode::kSpin);

    HWY_ASSERT(inner_pools_.empty());
    for (const LPS& inner : inner_lps) {
      inner_pools_.push_back(new hwy::ThreadPool(inner.Count()));
      inner_pools_.back()->SetWaitMode(hwy::PoolWaitMode::kSpin);
    }

    // For each inner pool, pin their threads AND the associated outer thread
    // to the enabled cores in the cluster.
    outer_pool_->Run(
        0, inner_lps.size(),
        [this, &inner_lps](uint64_t outer, size_t outer_thread) {
          HWY_ASSERT(outer == outer_thread);  // each outer has one task
          const std::vector<size_t> cores = CoresInLPS(inner_lps[outer]);

          inner_pools_[outer]->Run(
              0, cores.size(), [&cores](uint64_t task, size_t thread) {
                HWY_ASSERT(task == thread);  // each inner has one task
                hwy::PinThreadToLogicalProcessor(cores[task]);
              });
        });

    return true;
  }

  void ReuseMainPoolAsInner() {
    // Still allocate an empty pool to simplify Prefill().
    outer_pool_ = std::make_unique<hwy::ThreadPool>(1);

    HWY_ASSERT(inner_pools_.empty());
    inner_pools_.push_back(main_pool_);
  }

 public:
  // Creates pools. AllocateActivations must still be called separately; it has
  // a template argument.
  PrefillState(hwy::ThreadPool& main_pool, size_t num_queries)
      : main_pool_(&main_pool) {
    PROFILER_ZONE("Init.Prefill.Ctor");
    if (!AssignInnerPoolsToClusters(num_queries)) {
      ReuseMainPoolAsInner();
    }
  }

  ~PrefillState() {
    for (hwy::ThreadPool* p : inner_pools_) {
      if (p != main_pool_) delete p;
    }
  }

  // `tbatch_size` is the number of tokens from one query to prefill at a time.
  template <class TConfig>
  void AllocateActivations(size_t num_queries, size_t tbatch_size) {
    PROFILER_ZONE("Init.Prefill.AllocateActivations");

    const size_t outer_workers = outer_pool_->NumWorkers();
    HWY_ASSERT(outer_workers != 0);  // Otherwise activations_ is empty.

    HWY_ASSERT(activations_.empty());  // only call once.
    activations_.resize(outer_workers);

    if (outer_workers == 1) {
      activations_[0].Allocate<TConfig>(tbatch_size);
    } else {
      // Allocating in parallel can save 30 ms.
      main_pool_->Run(0, outer_workers,
                      [this, tbatch_size](uint64_t task, size_t /*thread*/) {
                        activations_[task].Allocate<TConfig>(tbatch_size);
                      });
    }
  }

  template <class TConfig>
  HWY_NOINLINE void Prefill(const MultiplePromptsTokens& prompts,
                            const size_t prefill_per_query, const size_t pos,
                            const size_t query_idx_start,
                            const CompressedWeights<TConfig>& weights,
                            const RuntimeConfig& runtime_config,
                            const KVCaches& kv_caches) {
    PROFILER_ZONE("Gen.Prefill");
    const size_t num_queries = prompts.size();
    HWY_ASSERT(kv_caches.size() == num_queries);
    const size_t max_tbatch_size = activations_[0].x.BatchSize();

    // For each query (parallel): an outer worker processes all its tokens.
    // `qi` is relative to the batch, not the global query index.
    outer_pool_->Run(
        0, num_queries, [&](const uint64_t qi, size_t qthread) HWY_ATTR {
          Activations& activations = activations_[qthread];
          hwy::ThreadPool& inner_pool = *inner_pools_[qthread];

          // Single query at a time, so pass a slice of the KV cache because
          // GemmaAttention will only access the first.
          const size_t kPrefillQueries = 1;
          KVCaches prefill_kv_caches(&kv_caches[qi], kPrefillQueries);

          // For each batch of tokens in the query:
          for (size_t tbatch_start = 0; tbatch_start < prefill_per_query;
               tbatch_start += max_tbatch_size) {
            // Fill activations.x (much faster than TransformerLayer).
            const size_t tbatch_size =
                HWY_MIN(max_tbatch_size, prefill_per_query - tbatch_start);
            for (size_t ti = 0; ti < tbatch_size; ++ti) {
              const int token = prompts[qi][tbatch_start + ti];
              EmbedToken<TConfig>(token, ti, pos + ti, weights, activations.x);
            }

            // Transformer with one batch of tokens from a single query.
            for (size_t layer = 0; layer < TConfig::kLayers; ++layer) {
              const auto* layer_weights = weights.GetLayer(layer);
              TransformerLayer<TConfig>(
                  tbatch_size, kPrefillQueries, pos + tbatch_start, layer,
                  layer_weights, activations, prefill_kv_caches, inner_pool);
            }

            // NOTE: we unconditionally call StreamToken, even if EOS.
            for (size_t ti = 0; ti < tbatch_size; ++ti) {
              const int token = prompts[qi][tbatch_start + ti];
              runtime_config.StreamToken(query_idx_start + qi,
                                         pos + tbatch_start + ti, token, 0.0f);
            }
          }  // for tbatch_start
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
  hwy::ThreadPool* main_pool_;
  std::unique_ptr<hwy::ThreadPool> outer_pool_;  // always allocated
  // Holds a single pointer equal to main_pool_, or new allocations; in either
  // case, size() is equal to outer_pool_->NumWorkers(). The first case avoids
  // allocation overhead for the common case of a single query.
  std::vector<hwy::ThreadPool*> inner_pools_;

  // size() == outer_pool_->NumWorkers(); filled by AllocateActivations.
  std::vector<Activations> activations_;
};

// `tokens` is length `num_tokens * num_queries`. In autoregressive decode,
// `num_tokens == 1`.
template <class TConfig>
HWY_NOINLINE void Transformer(const int* tokens, size_t num_tokens,
                              size_t num_queries, size_t pos,
                              const CompressedWeights<TConfig>& weights,
                              Activations& activations,
                              const KVCaches& kv_caches, hwy::ThreadPool& pool,
                              const LayersOutputFunc& layers_output) {
  const size_t num_interleaved = num_tokens * num_queries;
  if (layers_output) {
    for (size_t token_idx = 0; token_idx < num_interleaved; ++token_idx) {
      const size_t query_idx = token_idx % num_queries;
      const size_t logical_pos = (pos + token_idx) / num_queries;
      const float token_f = tokens[token_idx];
      layers_output(query_idx, logical_pos, "tokens", -1, &token_f, 1);
    }
  }
  constexpr size_t kModelDim = TConfig::kModelDim;
  for (size_t token_idx = 0; token_idx < num_interleaved; ++token_idx) {
    EmbedToken<TConfig>(tokens[token_idx], token_idx, pos, weights,
                        activations.x);
  }

  for (size_t layer = 0; layer < TConfig::kLayers; ++layer) {
    const CompressedLayer<TConfig>* layer_weights = weights.GetLayer(layer);
    TransformerLayer<TConfig>(num_tokens, num_queries, pos, layer,
                              layer_weights, activations, kv_caches, pool);

    if (layers_output) {
      for (size_t token_idx = 0; token_idx < num_interleaved; ++token_idx) {
        const size_t logical_pos = (pos + token_idx) / num_queries;
        layers_output(token_idx % num_queries, logical_pos, "blocks", layer,
                      activations.x.Batch(token_idx), kModelDim);
      }
    }
  }

  RMSNormInplaceBatched(num_interleaved, weights.final_norm_scale.data_scale1(),
                        activations.x.All(), kModelDim);
  if (layers_output) {
    for (size_t token_idx = 0; token_idx < num_interleaved; ++token_idx) {
      const size_t query_idx = token_idx % num_queries;
      const size_t logical_pos = (pos + token_idx) / num_queries;
      layers_output(query_idx, logical_pos, "final_norm", -1,
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
static std::vector<int> InterleaveQueries(const MultiplePromptsTokens& queries,
                                          const RuntimeConfig& runtime_config,
                                          size_t& min_prompt_size,
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
  hwy::BitSet4096<> is_eos_;
};

// Generates one token for each query in `prompts`, which is one qbatch whose
// size is at most the `batch_size` passed to `activations.Allocate`.
//
// `pos` indexes the KV cache. In the first turn of a chat, pos = 0, and it
// continues to increase by one for each prefilled/generated token per query.
//
// `query_idx_start` is the query_idx of the first query in the batch, so that
// `StreamFunc` gets the global query index, not relative to the batch.
//
// `kv_caches` is for the batch, size must match `prompts`.
template <class TConfig>
void GenerateT(const ByteStorageT& weights_u8, Activations& activations,
               const RuntimeConfig& runtime_config,
               const MultiplePromptsTokens& prompts, const size_t pos,
               const size_t query_idx_start, const KVCaches& kv_caches,
               hwy::ThreadPool& pool, TimingInfo& timing_info) {
  constexpr size_t kModelDim = TConfig::kModelDim;
  constexpr size_t kVocabSize = TConfig::kVocabSize;
  const CompressedWeights<TConfig>& weights =
      *reinterpret_cast<const CompressedWeights<TConfig>*>(weights_u8.get());

  const size_t num_queries = prompts.size();
  HWY_ASSERT(num_queries <= 4096);  // TokenStreamer uses BitSet4096.
  HWY_ASSERT(num_queries <= activations.x.BatchSize());
  HWY_ASSERT(kv_caches.size() == num_queries);

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
  double prefill_start;
  {
    PrefillState prefill(pool, num_queries);
    prefill.AllocateActivations<TConfig>(num_queries,
                                         runtime_config.prefill_tbatch_size);
    prefill_start = hwy::platform::Now();
    prefill.Prefill<TConfig>(prompts, prefill_per_query, pos, query_idx_start,
                             weights, runtime_config, kv_caches);
    timing_info.NotifyPrefill(prefill_per_query * num_queries, prefill_start);
    prefill.ResumeMainSpinning();
  }

  size_t interleaved_pos = (pos + prefill_per_query) * num_queries;

  // Storage for the last generated token from each query, passed to the next
  // Transformer() call.
  std::vector<int> gen_tokens(num_queries);

  // Stream the last prompt token from each query and fill gen_tokens.
  TokenStreamer token_streamer(runtime_config);
  for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
    gen_tokens[query_idx] = prompts[query_idx][prefill_per_query];
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
    // Compute logits from last layer activations.
    MatMul_4x4</*kAdd=*/false>(
        num_queries, MakeMat(activations.x.All(), kModelDim),
        MakeMat(weights.embedder_input_embedding.data(), kModelDim),
        weights.embedder_input_embedding.scale(), /*add=*/nullptr,
        MakeMat(activations.logits.All(), kVocabSize), pool);
    for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
      float* HWY_RESTRICT logits = activations.logits.Batch(query_idx);
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

template <class TConfig>
void GenerateSingleT(const ByteStorageT& weights_u8,
                     const RuntimeConfig& runtime_config,
                     const PromptTokens& prompt, size_t pos, KVCache& kv_cache,
                     hwy::ThreadPool& pool, TimingInfo& timing_info) {
  const size_t num_queries = 1;
  const size_t qbatch_start = 0;

  Activations activations;
  activations.Allocate<TConfig>(num_queries);

  const MultiplePromptsTokens prompts(&prompt, num_queries);
  const KVCaches kv_caches{&kv_cache, num_queries};

  GenerateT<TConfig>(weights_u8, activations, runtime_config, prompts, pos,
                     qbatch_start, kv_caches, pool, timing_info);
}

template <class TConfig>
void GenerateBatchT(const ByteStorageT& weights_u8,
                    const RuntimeConfig& runtime_config,
                    const MultiplePromptsTokens& prompts, size_t pos,
                    const KVCaches& kv_caches, hwy::ThreadPool& pool,
                    TimingInfo& timing_info) {
  HWY_ASSERT(prompts.size() == kv_caches.size());
  // Griffin does not support query batching.
  const size_t max_qbatch_size =
      (TConfig::kGriffinLayers > 0) ? 1 : runtime_config.decode_qbatch_size;

  Activations activations;
  activations.Allocate<TConfig>(max_qbatch_size);

  const size_t num_queries = prompts.size();
  for (size_t qbatch_start = 0; qbatch_start < num_queries;
       qbatch_start += max_qbatch_size) {
    // Generate one batch of tokens from `qbatch_size` queries.
    const size_t qbatch_size =
        HWY_MIN(num_queries - qbatch_start, max_qbatch_size);
    const MultiplePromptsTokens qbatch_prompts(&prompts[qbatch_start],
                                               qbatch_size);
    const KVCaches qbatch_kv(&kv_caches[qbatch_start], qbatch_size);
    GenerateT<TConfig>(weights_u8, activations, runtime_config, qbatch_prompts,
                       pos, qbatch_start, qbatch_kv, pool, timing_info);
  }
}

}  // namespace HWY_NAMESPACE

#if HWY_ONCE

// These are extern functions defined by instantiations/*.cc, which include this
// 'header' after defining GEMMA_CONFIG, which is for function overloading.
void GenerateSingle(  // NOLINT(misc-definitions-in-headers)
    GEMMA_CONFIG, const ByteStorageT& weights_u8,
    const RuntimeConfig& runtime_config, const PromptTokens& prompt, size_t pos,
    KVCache& kv_cache, hwy::ThreadPool& pool, TimingInfo& timing_info) {
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(GenerateSingleT<GEMMA_CONFIG>)
  (weights_u8, runtime_config, prompt, pos, kv_cache, pool, timing_info);
}

void GenerateBatch(  // NOLINT(misc-definitions-in-headers)
    GEMMA_CONFIG, const ByteStorageT& weights_u8,
    const RuntimeConfig& runtime_config, const MultiplePromptsTokens& prompts,
    size_t pos, const KVCaches& kv_caches, hwy::ThreadPool& pool,
    TimingInfo& timing_info) {
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(GenerateBatchT<GEMMA_CONFIG>)
  (weights_u8, runtime_config, prompts, pos, kv_caches, pool, timing_info);
}

#endif  // HWY_ONCE

}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_INL_H_
