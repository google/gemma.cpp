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
#include "ops/matvec-inl.h"
#include "ops/ops-inl.h"
#include "util/threading.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/bit_set.h"
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

// Different functions use different naming conventions for the number of
// tokens. Functions that are query-independent, such as RMSNorm*, call the
// count `num_interleaved`. Functions that are query-dependent, such as
// `Attention`, use separate `num_tokens` and `num_queries`.

// TODO: add batch query support for Griffin (QueriesPos).
template <class TConfig>
HWY_NOINLINE void GriffinRecurrent(
    size_t batch_start, size_t num_tokens, size_t layer,
    Activations& activations, const CompressedLayer<TConfig>* layer_weights,
    const KVCaches& kv_caches, hwy::ThreadPool& pool) {
  PROFILER_ZONE("Gen.Griffin");
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
    float* out_ptr = activations.att_sums.Batch(batch_idx);
    MatVecAdd<kModelDim, kModelDim>(
        layer_weights->griffin.linear_out_w, 0, x,
        layer_weights->griffin.linear_out_biases.data_scale1(),
        activations.even_odd.All(), out_ptr, pool);
  }
}

// Wrapper class; holds arguments in member variables to shorten call sites.
template <class TConfig>
class GemmaAttention {
  static constexpr size_t kCacheLayerSize = CacheLayerSize<TConfig>()();
  static constexpr size_t kCachePosSize = CachePosSize<TConfig>()();
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kKVHeads = TConfig::kKVHeads;
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kQStride = Activations::QStride<TConfig>();
  static constexpr size_t kSeqLen = TConfig::kSeqLen;
  static constexpr bool kIsMHA = Activations::IsMHA<TConfig>();

  // The attention window usually starts at 0 unless unless `pos` is larger than
  // the attention window size, then it is `pos` - window_size + 1.
  static HWY_INLINE size_t StartPos(size_t pos, size_t layer) {
    const size_t att_window_size = TConfig::kAttentionWindowSizes[layer];
    return pos - std::min(att_window_size - 1, pos);
  }

  template <typename T>
  HWY_INLINE void PositionalEncodingQK(const T* qk, size_t pos, size_t layer,
                                       const float mul, T* qk_out) {
    const float* inv_timescale = activations_.inv_timescale.Const();
    // PostQKType::Rope
    (void)layer;
    if (TConfig::kUseHalfRope) {
      hwy::CopyBytes(qk, qk_out, kQKVDim * sizeof(*qk));
      Rope(qk_out, kQKVDim / 2, inv_timescale, pos);
      MulByConst(mul, qk_out, kQKVDim);
    } else {
      RopeAndMulBy(mul, qk, kQKVDim, inv_timescale, pos, qk_out);
    }
  }

  // Fills activations.q and computes KV. For kIsMHA, a single MatMul suffices
  // and we later copy KV from q to KVCache. Otherwise, a second MatMul writes
  // KV directly to KVCache.
  HWY_NOINLINE void ComputeQKV(const size_t num_interleaved) {
    PROFILER_ZONE("Gen.Attention.QKV");
    // For the computation of Q, K, and V, it is useful to remember that
    // qkv_einsum_w has shape [(kHeads + kKVHeads * 2), kKQVDim, kModelDim]
    // and kQStride = kQKVDim * (kIsMHA ? 3 : 1);

    const auto pre_att_rms_out =
        MakeMat(activations_.pre_att_rms_out.All(), kModelDim);
    MatMul_4x4</*kAdd=*/false>(
        num_interleaved, pre_att_rms_out,
        MakeMat(layer_weights_.qkv_einsum_w.data(), kModelDim),
        layer_weights_.qkv_einsum_w.scale(), /*add=*/nullptr,
        MakeMat(activations_.q.All(), kHeads * kQStride), pool_);

    if constexpr (kIsMHA) {
      static_assert(TConfig::kInterleaveQKV, "MHA implies interleaved");
      // Multi-Head Attention a.k.a. "use_qkv_einsum" computed QKV already.
    } else {
      // Single query and no wraparound means we can use a matmul and write
      // directly into the KV cache with a stride of kCachePosSize.
      if (num_queries_ == 1 &&
          queries_pos_[0] + num_tokens_ <= div_seq_len_.GetDivisor()) {
        const size_t kv_ofs =
            queries_pos_[0] * kCachePosSize + layer_ * kCacheLayerSize;
        // KV structure is [k, v, k, v, ....] = kKVHeads pairs of (k, v).
        float* HWY_RESTRICT kv = kv_caches_[0].kv_cache.get() + kv_ofs;
        MatMul_4x4</*kAdd=*/false>(
            num_tokens_, pre_att_rms_out,
            MakeMat(layer_weights_.qkv_einsum_w.data(), kModelDim, kModelDim,
                    kHeads * kQKVDim * kModelDim),
            layer_weights_.qkv_einsum_w.scale(), /*add=*/nullptr,
            MakeMat(kv, kKVHeads * 2 * kQKVDim, kCachePosSize), pool_);
      } else {
        // Proceed row by row because there will be wraparound.
        for (size_t interleaved_idx = 0; interleaved_idx < num_interleaved;
             ++interleaved_idx) {
          const float* x = activations_.pre_att_rms_out.Batch(interleaved_idx);
          const size_t query_idx = interleaved_idx % num_queries_;
          const size_t batch_idx = interleaved_idx / num_queries_;
          KVCache& kv_cache = kv_caches_[query_idx];
          const size_t cache_pos =
              div_seq_len_.Remainder(queries_pos_[query_idx] + batch_idx);
          const size_t kv_offset =
              cache_pos * kCachePosSize + layer_ * kCacheLayerSize;
          float* HWY_RESTRICT kv = kv_cache.kv_cache.get() + kv_offset;
          // KV structure is [k, v, k, v, ....] = kKVHeads pairs of (k, v).
          MatVec<kKVHeads * 2 * kQKVDim, kModelDim>(
              layer_weights_.qkv_einsum_w, kHeads * kQKVDim * kModelDim, x,
              activations_.even_odd.All(), kv, pool_);
        }
      }
    }

    // Apply positional encodings for K (and copy KV to cache if MHA).
    pool_.Run(
        0, kKVHeads * num_interleaved,
        [&](uint64_t task, size_t /*thread*/) HWY_ATTR {
          const size_t head = task % kKVHeads;
          const size_t interleaved_idx = task / kKVHeads;
          const size_t query_idx = interleaved_idx % num_queries_;
          const size_t batch_idx = interleaved_idx / num_queries_;
          const size_t pos = queries_pos_[query_idx] + batch_idx;
          const size_t cache_pos = div_seq_len_.Remainder(pos);
          const size_t kv_offset = cache_pos * kCachePosSize +
                                   layer_ * kCacheLayerSize +
                                   head * kQKVDim * 2;
          KVCache& kv_cache = kv_caches_[query_idx];
          float* HWY_RESTRICT kv = kv_cache.kv_cache.get() + kv_offset;
          const float* HWY_RESTRICT mha_kv =
              activations_.q.Batch(interleaved_idx) + head * kQStride + kQKVDim;

          // Copy from `q` if MHA, or apply in-place.
          PositionalEncodingQK(kIsMHA ? mha_kv : kv, pos, layer_, 1.0f, kv);

          // If MHA, also copy V into KVCache.
          if (kIsMHA) {
            hwy::CopyBytes(mha_kv + kQKVDim, kv + kQKVDim,
                           kQKVDim * sizeof(*kv));
          }
        });
  }

  // Computes Q.K scores, which are "logits" (or scores) stored to head_att.
  HWY_INLINE void QDotK(const size_t start_pos, const size_t pos,
                        const size_t head_offset, const float* HWY_RESTRICT q,
                        const KVCache& kv_cache, float* HWY_RESTRICT head_att) {
    if (HWY_LIKELY(pos <= kSeqLen)) {
      // Slightly faster: no wraparound.
      for (size_t pos2 = start_pos; pos2 <= pos; ++pos2) {
        const size_t kv_offset =
            pos2 * kCachePosSize + layer_ * kCacheLayerSize + head_offset;
        const float* HWY_RESTRICT k = &kv_cache.kv_cache[kv_offset];
        const float score = Dot(q, k, kQKVDim);
        head_att[pos2] = score;
      }
    } else {
      for (size_t pos2 = start_pos; pos2 <= pos; ++pos2) {
        const size_t cache_pos = div_seq_len_.Remainder(pos2);
        const size_t kv_offset =
            cache_pos * kCachePosSize + layer_ * kCacheLayerSize + head_offset;
        const float* HWY_RESTRICT k = &kv_cache.kv_cache[kv_offset];
        const float score = Dot(q, k, kQKVDim);
        head_att[pos2 % kSeqLen] = score;
      }
    }
  }

  // Accumulates the sum of v (from `kv_cache`) * probability (`head_att`) into
  // `att_out`. Equivalent in gemma/modules.py:
  // encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)
  static HWY_INLINE void WeightedSumV(const size_t start_pos, const size_t pos,
                                      const float* HWY_RESTRICT head_att,
                                      const size_t layer,
                                      const size_t head_offset,
                                      const hwy::Divisor& div_seq_len,
                                      const KVCache& kv_cache,
                                      float* HWY_RESTRICT att_out) {
    hwy::ZeroBytes(att_out, kQKVDim * sizeof(*att_out));

    if (HWY_LIKELY(pos <= kSeqLen)) {
      // Slightly faster: no wraparound.
      for (size_t pos2 = start_pos; pos2 <= pos; ++pos2) {
        const size_t kv_offset =
            pos2 * kCachePosSize + layer * kCacheLayerSize + head_offset;
        const float* HWY_RESTRICT v =
            kv_cache.kv_cache.get() + kv_offset + kQKVDim;
        MulByConstAndAdd(head_att[pos2], v, att_out, kQKVDim);
      }
    } else {
      for (size_t pos2 = start_pos; pos2 <= pos; ++pos2) {
        const size_t cache_pos = div_seq_len.Remainder(pos2);
        const size_t kv_offset =
            cache_pos * kCachePosSize + layer * kCacheLayerSize + head_offset;
        const float* HWY_RESTRICT v =
            kv_cache.kv_cache.get() + kv_offset + kQKVDim;
        MulByConstAndAdd(head_att[pos2 % kSeqLen], v, att_out, kQKVDim);
      }
    }
  }

  HWY_NOINLINE void DotSoftmaxWeightedSum(const size_t num_interleaved) {
    PROFILER_ZONE("Gen.Attention.DotSoftmax");
    GEMMA_CONSTEXPR_SQRT float kQueryScale = ChooseQueryScale<TConfig>();

    // A "head group" in the context of GQA refers to a collection of query
    // heads that share the same key and value heads.
    static_assert((kHeads % kKVHeads) == 0,
                  "query heads must be a multiple of key-value heads");
    constexpr size_t kHeadGroups = kHeads / kKVHeads;

    // For each head (token, query), compute Q.K, softmax, and weighted V.
    pool_.Run(0, kHeads * num_interleaved,
              [&](uint64_t task, size_t /*thread*/) HWY_ATTR {
                const size_t head = task % kHeads;
                const size_t interleaved_idx = task / kHeads;
                const size_t query_idx = interleaved_idx % num_queries_;
                const size_t batch_idx = interleaved_idx / num_queries_;
                const size_t head_offset = (head / kHeadGroups) * kQKVDim * 2;
                KVCache& kv_cache = kv_caches_[query_idx];
                float* HWY_RESTRICT q =
                    activations_.q.Batch(interleaved_idx) + head * kQStride;

                // Apply rope and scaling to Q.
                const size_t pos = queries_pos_[query_idx] + batch_idx;
                PositionalEncodingQK(q, pos, layer_, kQueryScale, q);

                const size_t start_pos = StartPos(pos, layer_);

                float* HWY_RESTRICT head_att =
                    activations_.att.Batch(interleaved_idx) + head * kSeqLen;
                QDotK(start_pos, pos, head_offset, q, kv_cache, head_att);
                // SoftMax with optional SoftCap yields "probabilities" in
                // head_att.
                const size_t head_att_len = std::min(pos + 1, kSeqLen);
                MaybeLogitsSoftCap(TConfig::kAttCap, head_att, head_att_len);
                Softmax(head_att, head_att_len);

                float* HWY_RESTRICT att_out =
                    activations_.att_out.Batch(interleaved_idx) +
                    head * kQKVDim;
                WeightedSumV(start_pos, pos, head_att, layer_, head_offset,
                             div_seq_len_, kv_cache, att_out);
              });
  }

  // Sums encoded (`att_out`) over num_heads (`kHeads`) and head_dim (`kQKVDim`)
  // into output (`layer_out`).
  HWY_NOINLINE void SumHeads(const size_t num_interleaved) {
    PROFILER_ZONE("Gen.Attention.SumHeads");
    constexpr bool kAdd = TConfig::kSoftmaxAttnOutputBiases;
    const float* bias =
        kAdd ? layer_weights_.attention_output_biases.data_scale1() : nullptr;

    // att_weights and att_out are concatenated heads, each of length kQKVDim.
    // Thus the [num_interleaved, kModelDim] matmul output is the sum over
    // heads. Compare gemma/modules.py:
    // attn_output = self.attn_vec_einsum('BTNH,NHD->BTD', encoded)
    MatMul_4x4<kAdd>(
        num_interleaved, MakeMat(activations_.att_out.All(), kHeads * kQKVDim),
        MakeMat(layer_weights_.att_weights.data(), kHeads * kQKVDim),
        layer_weights_.attn_vec_einsum_w.scale(), bias,
        MakeMat(activations_.att_sums.All(), kModelDim), pool_);
  }

 public:
  GemmaAttention(const QueriesPos& queries_pos, size_t num_tokens, size_t layer,
                 Activations& activations,
                 const CompressedLayer<TConfig>* layer_weights,
                 const hwy::Divisor& div_seq_len, const KVCaches& kv_caches,
                 hwy::ThreadPool& pool)
      : queries_pos_(queries_pos),
        num_queries_(queries_pos.size()),
        num_tokens_(num_tokens),
        layer_(layer),
        activations_(activations),
        layer_weights_(*layer_weights),
        div_seq_len_(div_seq_len),
        kv_caches_(kv_caches),
        pool_(pool) {
    HWY_DASSERT(num_queries_ <= kv_caches_.size());
  }

  HWY_INLINE void operator()() {
    const size_t num_interleaved = num_tokens_ * num_queries_;
    ComputeQKV(num_interleaved);
    DotSoftmaxWeightedSum(num_interleaved);
    SumHeads(num_interleaved);
  }

 private:
  const QueriesPos& queries_pos_;
  const size_t num_queries_;
  const size_t num_tokens_;
  const size_t layer_;
  Activations& activations_;
  const CompressedLayer<TConfig>& layer_weights_;
  const hwy::Divisor& div_seq_len_;
  const KVCaches& kv_caches_;
  hwy::ThreadPool& pool_;
};

template <class TConfig>
HWY_NOINLINE void Attention(LayerAttentionType type,
                            const QueriesPos& queries_pos, size_t num_tokens,
                            size_t layer, Activations& activations,
                            const CompressedLayer<TConfig>* layer_weights,
                            const hwy::Divisor& div_seq_len,
                            const KVCaches& kv_caches, hwy::ThreadPool& pool) {
  if (type == LayerAttentionType::kGemma) {
    GemmaAttention<TConfig>(queries_pos, num_tokens, layer, activations,
                            layer_weights, div_seq_len, kv_caches, pool)();
  } else {
    // Only reached if the model is Griffin. `if constexpr` prevents generating
    // this code for non-Griffin models.
    if constexpr (TConfig::kGriffinLayers > 0) {
      HWY_ASSERT(queries_pos.size() == 1);
      GriffinRecurrent<TConfig>(queries_pos[0], num_tokens, layer, activations,
                                layer_weights, kv_caches, pool);
    }
  }
}

template <class TConfig, typename T>
HWY_NOINLINE void Activation(T* HWY_RESTRICT c1, T* HWY_RESTRICT c2,
                             size_t count) {
  PROFILER_ZONE("Gen.Activation");
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
    const QueriesPos& queries_pos, size_t num_tokens, size_t layer,
    const CompressedLayer<TConfig>* layer_weights, Activations& activations,
    const hwy::Divisor& div_seq_len, const KVCaches& kv_caches,
    hwy::ThreadPool& pool) {
  constexpr size_t kModelDim = TConfig::kModelDim;
  const size_t num_interleaved = num_tokens * queries_pos.size();
  auto type = TConfig::kLayerConfig[layer];
  size_t layer_of_type =
      NumLayersOfTypeBefore(TConfig::kLayerConfig, type, layer);

  RMSNormBatched(num_interleaved, activations.x.All(),
                 layer_weights->pre_attention_norm_scale.data_scale1(),
                 activations.pre_att_rms_out.All(), kModelDim);

  Attention<TConfig>(type, queries_pos, num_tokens, layer_of_type, activations,
                     layer_weights, div_seq_len, kv_caches, pool);

  PostNorm<TConfig>(num_interleaved, layer_weights->post_attention_norm_scale,
                    activations.att_sums.All());

  ResidualConnection<TConfig>(num_interleaved, activations.att_sums.All(),
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

// Prefill and Transformer() advance positions in-place.
using QueriesMutablePos = hwy::Span<size_t>;

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
 public:
  // `tbatch_size` is the number of tokens from one query to prefill at a time.
  template <class TConfig>
  void Init(size_t num_queries, size_t tbatch_size, PerClusterPools& pools) {
    PROFILER_ZONE("Init.Prefill");
    HWY_ASSERT(num_queries != 0);
    HWY_ASSERT(activations_.empty());  // only call once.

    // Allocate one activation per query, not outer worker, because the common
    // case is a single query. If we allocate the lesser of the two, it is
    // unclear how to choose an unused activation in Prefill.
    activations_.resize(num_queries);

    if (num_queries == 1) {
      activations_[0].Allocate<TConfig>(tbatch_size);
    } else {
      // Allocating in parallel can save 30 ms. We might have more workers than
      // queries/tasks, so do not check the `thread` argument.
      pools.Outer().Run(0, num_queries,
                        [this, tbatch_size](uint64_t qi, size_t /*thread*/) {
                          activations_[qi].Allocate<TConfig>(tbatch_size);
                        });
    }
  }

  template <class TConfig>
  HWY_NOINLINE void Prefill(const QueriesPromptTokens& queries_prompt,
                            const size_t prefill_per_query,
                            const QueriesMutablePos& queries_pos,
                            const size_t query_idx_start,
                            const CompressedWeights<TConfig>& weights,
                            const RuntimeConfig& runtime_config,
                            const hwy::Divisor& div_seq_len,
                            const KVCaches& kv_caches, PerClusterPools& pools) {
    PROFILER_ZONE("Gen.Prefill");
    const size_t num_queries = queries_prompt.size();
    HWY_ASSERT(kv_caches.size() == num_queries);
    const size_t max_tbatch_size = activations_[0].x.BatchSize();

    // For each query (parallel): an outer worker processes all its tokens.
    // `qi` is relative to the batch, not the global query index.
    pools.Outer().Run(
        0, num_queries, [&](const uint64_t qi, size_t qthread) HWY_ATTR {
          Activations& activations = activations_[qi];
          hwy::ThreadPool& inner_pool = pools.Inner(qthread);

          // Single query at a time, so pass slices of the spans because
          // GemmaAttention will only access the first KV cache and position.
          KVCaches single_kv_cache(&kv_caches[qi], 1);
          QueriesPos single_query_pos(&queries_pos[qi], 1);

          // For each batch of tokens in the query:
          for (size_t tbatch_start = 0; tbatch_start < prefill_per_query;
               tbatch_start += max_tbatch_size) {
            // Fill activations.x (much faster than TransformerLayer).
            const size_t tbatch_size =
                HWY_MIN(max_tbatch_size, prefill_per_query - tbatch_start);
            for (size_t ti = 0; ti < tbatch_size; ++ti) {
              const int token = queries_prompt[qi][tbatch_start + ti];
              const size_t pos = queries_pos[qi] + ti;
              EmbedToken<TConfig>(token, ti, pos, weights, activations.x);
            }

            // Transformer with one batch of tokens from a single query.
            for (size_t layer = 0; layer < TConfig::kLayers; ++layer) {
              const auto* layer_weights = weights.GetLayer(layer);
              TransformerLayer<TConfig>(single_query_pos, tbatch_size, layer,
                                        layer_weights, activations, div_seq_len,
                                        single_kv_cache, inner_pool);
            }

            // NOTE: we unconditionally call StreamToken, even if EOS.
            for (size_t ti = 0; ti < tbatch_size; ++ti) {
              const size_t pos = queries_pos[qi] + ti;
              const int token = queries_prompt[qi][pos];
              runtime_config.StreamToken(query_idx_start + qi, pos, token,
                                         0.0f);
            }

            queries_pos[qi] += tbatch_size;
          }  // for tbatch_start
        });
  }

 private:
  std::vector<Activations> activations_;  // One per query, filled by Init.
};

// Generates one token for each query. `queries_token` is the previous token
// from each query, and `queries_pos` are their position in the sequence.
template <class TConfig>
HWY_NOINLINE void Transformer(
    const QueriesToken& queries_token, const QueriesMutablePos& queries_pos,
    const CompressedWeights<TConfig>& weights, Activations& activations,
    const hwy::Divisor& div_seq_len, const KVCaches& kv_caches,
    hwy::ThreadPool& pool, const LayersOutputFunc& layers_output,
    const ActivationsObserverFunc& activations_observer) {
  constexpr size_t kModelDim = TConfig::kModelDim;
  const size_t num_queries = queries_token.size();
  HWY_DASSERT(queries_pos.size() == num_queries);

  if (layers_output) {
    for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
      const float token_f = queries_token[query_idx];
      layers_output(query_idx, queries_pos[query_idx], "tokens", -1, &token_f,
                    1);
    }
  }

  for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
    EmbedToken<TConfig>(queries_token[query_idx], query_idx,
                        queries_pos[query_idx], weights, activations.x);
  }

  for (size_t layer = 0; layer < TConfig::kLayers; ++layer) {
    const CompressedLayer<TConfig>* layer_weights = weights.GetLayer(layer);
    TransformerLayer<TConfig>(queries_pos, /*num_tokens=*/1, layer,
                              layer_weights, activations, div_seq_len,
                              kv_caches, pool);

    if (activations_observer) {
      activations_observer(queries_pos, layer, activations);
    }
  }

  RMSNormInplaceBatched(num_queries, weights.final_norm_scale.data_scale1(),
                        activations.x.All(), kModelDim);

  if (activations_observer) {
    activations_observer(queries_pos, -1, activations);
  }
  for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
    queries_pos[query_idx] += 1;
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

// Returns the min and max number of tokens for all queries.
static void ScanQueryLengths(const QueriesPromptTokens& queries_prompt,
                             size_t& min_prompt_size, size_t& max_prompt_size) {
  const size_t num_queries = queries_prompt.size();
  min_prompt_size = hwy::LimitsMax<size_t>();
  max_prompt_size = 0;
  for (size_t i = 0; i < num_queries; ++i) {
    min_prompt_size = std::min(min_prompt_size, queries_prompt[i].size());
    max_prompt_size = std::max(max_prompt_size, queries_prompt[i].size());
  }
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

// Generates one continuation for each query in `queries_prompt`, which is one
// qbatch whose size is at most the `batch_size` passed to
// `activations.Allocate`.
//
// `queries_pos` stores the KV cache position for each query. In the first turn
// of a chat, pos = 0; we increment each query's position after each token.
//
// `query_idx_start` is the query_idx of the first query in the batch, so that
// `StreamFunc` gets the global query index, not relative to the batch.
//
// `kv_caches` is for the batch, size must match `queries_prompt`.
template <class TConfig>
void GenerateT(const ByteStorageT& weights_u8, Activations& activations,
               const RuntimeConfig& runtime_config,
               const QueriesPromptTokens& queries_prompt,
               const QueriesPos& queries_pos_in, const size_t query_idx_start,
               const KVCaches& kv_caches, PerClusterPools& pools,
               TimingInfo& timing_info) {
  constexpr size_t kModelDim = TConfig::kModelDim;
  constexpr size_t kVocabSize = TConfig::kVocabSize;
  const CompressedWeights<TConfig>& weights =
      *reinterpret_cast<const CompressedWeights<TConfig>*>(weights_u8.get());

  // TODO: remove once all parallel sections support hierarchical parallelism.
  hwy::ThreadPool& pool = pools.Inner(0);

  // Copy so we can increment without requiring users to pass in a mutable span.
  std::vector<size_t> queries_pos_copy(queries_pos_in.cbegin(),
                                       queries_pos_in.cend());
  const QueriesMutablePos queries_mutable_pos(queries_pos_copy.data(),
                                              queries_pos_copy.size());

  const size_t num_queries = queries_prompt.size();
  HWY_ASSERT(num_queries <= 4096);  // TokenStreamer uses BitSet4096.
  HWY_ASSERT(num_queries <= activations.x.BatchSize());
  HWY_ASSERT(queries_pos_in.size() == num_queries);
  HWY_ASSERT(kv_caches.size() == num_queries);
  const hwy::Divisor div_seq_len(static_cast<uint32_t>(kv_caches[0].seq_len));

  size_t min_prompt_size, max_prompt_size;
  ScanQueryLengths(queries_prompt, min_prompt_size, max_prompt_size);

  size_t max_tokens = runtime_config.max_tokens;
  size_t max_generated_tokens = runtime_config.max_generated_tokens;
  RangeChecks<TConfig>(max_tokens, max_generated_tokens, max_prompt_size);
  for (size_t pos : queries_pos_copy) {
    if (pos >= max_tokens) {
      fprintf(stderr, "Warning: pos %zu >= max_tokens %zu, aborting.\n", pos,
              max_tokens);
      return;
    }
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
    // TODO: move to Gemma, reuse across calls to Generate.
    PrefillState prefill;
    prefill.Init<TConfig>(num_queries, runtime_config.prefill_tbatch_size,
                          pools);
    prefill_start = hwy::platform::Now();
    prefill.Prefill<TConfig>(queries_prompt, prefill_per_query,
                             queries_mutable_pos, query_idx_start, weights,
                             runtime_config, div_seq_len, kv_caches, pools);
    timing_info.NotifyPrefill(prefill_per_query * num_queries, prefill_start);
    // queries_pos are incremented by Prefill.
  }

  // Storage for the last generated token from each query, passed to the next
  // Transformer() call.
  std::vector<int> gen_tokens(num_queries);

  // Stream the last prompt token from each query and fill gen_tokens.
  TokenStreamer token_streamer(runtime_config);
  for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
    gen_tokens[query_idx] = queries_prompt[query_idx][prefill_per_query];
    (void)token_streamer(query_idx_start + query_idx,
                         queries_mutable_pos[query_idx], gen_tokens[query_idx],
                         0.0f);
  }

  const double gen_start = hwy::platform::Now();
  for (size_t gen = 0; gen < HWY_MIN(max_tokens, max_generated_tokens); ++gen) {
    // Decode generates one token per query and increments queries_mutable_pos.
    Transformer<TConfig>(QueriesToken(gen_tokens.data(), num_queries),
                         queries_mutable_pos, weights, activations, div_seq_len,
                         kv_caches, pool, runtime_config.layers_output,
                         runtime_config.activations_observer);
    // queries_pos are incremented by Transformer.

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
      MaybeLogitsSoftCap(TConfig::kFinalCap, logits, kVocabSize);
      Softmax(logits, kVocabSize);
      const int token = sample_token(logits, kVocabSize);
      timing_info.NotifyGenerated(prefill_start, gen_start);

      const bool is_eos =
          token_streamer(query_idx_start + query_idx,
                         queries_mutable_pos[query_idx], token, logits[token]);
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
                     PerClusterPools& pools, TimingInfo& timing_info) {
  constexpr size_t kNumQueries = 1;
  const size_t qbatch_start = 0;

  Activations activations;
  activations.Allocate<TConfig>(kNumQueries);

  const QueriesPromptTokens prompt_span(&prompt, kNumQueries);
  QueriesPos pos_span(&pos, kNumQueries);
  const KVCaches kv_caches{&kv_cache, kNumQueries};

  GenerateT<TConfig>(weights_u8, activations, runtime_config, prompt_span,
                     pos_span, qbatch_start, kv_caches, pools, timing_info);
}

template <class TConfig>
void GenerateBatchT(const ByteStorageT& weights_u8,
                    const RuntimeConfig& runtime_config,
                    const QueriesPromptTokens& queries_prompt,
                    const QueriesPos& queries_pos, const KVCaches& kv_caches,
                    PerClusterPools& pools, TimingInfo& timing_info) {
  const size_t num_queries = queries_prompt.size();
  HWY_ASSERT(queries_pos.size() == num_queries);
  HWY_ASSERT(kv_caches.size() == num_queries);
  // Griffin does not support query batching.
  const size_t max_qbatch_size =
      (TConfig::kGriffinLayers > 0) ? 1 : runtime_config.decode_qbatch_size;

  Activations activations;
  activations.Allocate<TConfig>(max_qbatch_size);

  for (size_t qbatch_start = 0; qbatch_start < num_queries;
       qbatch_start += max_qbatch_size) {
    // Generate one batch of tokens from `qbatch_size` queries.
    const size_t qbatch_size =
        HWY_MIN(num_queries - qbatch_start, max_qbatch_size);
    const QueriesPromptTokens qbatch_prompts(&queries_prompt[qbatch_start],
                                             qbatch_size);
    QueriesPos qbatch_pos(&queries_pos[qbatch_start], qbatch_size);
    const KVCaches qbatch_kv(&kv_caches[qbatch_start], qbatch_size);
    GenerateT<TConfig>(weights_u8, activations, runtime_config, qbatch_prompts,
                       qbatch_pos, qbatch_start, qbatch_kv, pools, timing_info);
  }
}

}  // namespace HWY_NAMESPACE

#if HWY_ONCE

// These are extern functions defined by instantiations/*.cc, which include this
// 'header' after defining GEMMA_CONFIG, which is for function overloading.
void GenerateSingle(  // NOLINT(misc-definitions-in-headers)
    GEMMA_CONFIG, const ByteStorageT& weights_u8,
    const RuntimeConfig& runtime_config, const PromptTokens& prompt, size_t pos,
    KVCache& kv_cache, PerClusterPools& pools, TimingInfo& timing_info) {
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(GenerateSingleT<GEMMA_CONFIG>)
  (weights_u8, runtime_config, prompt, pos, kv_cache, pools, timing_info);
}

void GenerateBatch(  // NOLINT(misc-definitions-in-headers)
    GEMMA_CONFIG, const ByteStorageT& weights_u8,
    const RuntimeConfig& runtime_config,
    const QueriesPromptTokens& queries_prompt, const QueriesPos& queries_pos,
    const KVCaches& kv_caches, PerClusterPools& pools,
    TimingInfo& timing_info) {
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(GenerateBatchT<GEMMA_CONFIG>)
  (weights_u8, runtime_config, queries_prompt, queries_pos, kv_caches, pools,
   timing_info);
}

#endif  // HWY_ONCE

}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_INL_H_
