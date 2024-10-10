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

#include <stddef.h>
#include <stdio.h>

#include <algorithm>  // std::min
#include <type_traits>
#include <vector>

#include "gemma/activations.h"
#include "gemma/common.h"
#include "gemma/configs.h"
#include "gemma/gemma.h"
#include "gemma/weights.h"
// Placeholder for internal test4, do not remove
#include "paligemma/image.h"
#include "util/allocator.h"
#include "util/threading.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/bit_set.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/timer.h"

// Include guard (still compiled once per target)
#if defined(THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_INL_H_
#undef THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_INL_H_
#else
#define THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_INL_H_
#endif

#include "hwy/highway.h"
// After highway.h
#include "ops/matmul-inl.h"
#include "ops/matvec-inl.h"
#include "ops/ops-inl.h"
#include "hwy/profiler.h"  // also uses SIMD

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
    const KVCaches& kv_caches) {
  PROFILER_ZONE("Gen.Griffin");
  KVCache& kv_cache = kv_caches[0];
  hwy::ThreadPool& pool = activations.env.Pool();
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
        layer_weights->griffin.linear_out_biases.data_scale1(), out_ptr, pool);
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

  // The attention window usually starts at 0 unless `pos` is larger than
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
        ConstMat(activations_.pre_att_rms_out.All(), kModelDim);
    const auto w_q1 =
        layer_weights_.qkv_einsum_w.data() == nullptr
            ? ConstMat(layer_weights_.qkv_einsum_w1.data(), kModelDim)
            : ConstMat(layer_weights_.qkv_einsum_w.data(), kModelDim);
    const auto w_q2 =
        layer_weights_.qkv_einsum_w.data() == nullptr
            ? ConstMat(layer_weights_.qkv_einsum_w2.data(), kModelDim)
            : ConstMat(layer_weights_.qkv_einsum_w.data(), kModelDim, kModelDim,
                       kHeads * kQKVDim * kModelDim);
    MatMul</*kAdd=*/false>(num_interleaved, pre_att_rms_out, w_q1,
                           layer_weights_.qkv_einsum_w.scale(), /*add=*/nullptr,
                           activations_.env,
                           MutableMat(activations_.q.All(), kHeads * kQStride));

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
        MatMul</*kAdd=*/false>(
            num_tokens_, pre_att_rms_out, w_q2,
            layer_weights_.qkv_einsum_w.scale(), /*add=*/nullptr,
            activations_.env,
            MutableMat(kv, kKVHeads * 2 * kQKVDim, kCachePosSize));
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
          if (layer_weights_.qkv_einsum_w.data() == nullptr) {
            MatVec<kKVHeads * 2 * kQKVDim, kModelDim>(
                layer_weights_.qkv_einsum_w2, 0, x, kv, pool_);
          } else {
            MatVec<kKVHeads * 2 * kQKVDim, kModelDim>(
                layer_weights_.qkv_einsum_w, kHeads * kQKVDim * kModelDim, x,
                kv, pool_);
          }
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
  HWY_INLINE void QDotK(const size_t start_pos, const size_t last_pos,
                        const size_t head_offset, const float* HWY_RESTRICT q,
                        const KVCache& kv_cache, float* HWY_RESTRICT head_att) {
    if (HWY_LIKELY(last_pos < kSeqLen)) {
      // Slightly faster: no wraparound.
      for (size_t pos = start_pos; pos <= last_pos; ++pos) {
        const size_t kv_offset =
            pos * kCachePosSize + layer_ * kCacheLayerSize + head_offset;
        const float* HWY_RESTRICT k = &kv_cache.kv_cache[kv_offset];
        const float score = Dot(q, k, kQKVDim);
        head_att[pos] = score;
      }
    } else {
      for (size_t pos = start_pos; pos <= last_pos; ++pos) {
        const size_t cache_pos = div_seq_len_.Remainder(pos);
        const size_t kv_offset =
            cache_pos * kCachePosSize + layer_ * kCacheLayerSize + head_offset;
        const float* HWY_RESTRICT k = &kv_cache.kv_cache[kv_offset];
        const float score = Dot(q, k, kQKVDim);
        head_att[pos % kSeqLen] = score;
      }
    }
  }

  // Accumulates the sum of v (from `kv_cache`) * probability (`head_att`) into
  // `att_out`. Equivalent in gemma/modules.py:
  // encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)
  static HWY_INLINE void WeightedSumV(
      const size_t start_pos, const size_t last_pos,
      const float* HWY_RESTRICT head_att, const size_t layer,
      const size_t head_offset, const hwy::Divisor& div_seq_len,
      const KVCache& kv_cache, float* HWY_RESTRICT att_out) {
    hwy::ZeroBytes(att_out, kQKVDim * sizeof(*att_out));

    if (HWY_LIKELY(last_pos < kSeqLen)) {
      // Slightly faster: no wraparound.
      for (size_t pos = start_pos; pos <= last_pos; ++pos) {
        const size_t kv_offset =
            pos * kCachePosSize + layer * kCacheLayerSize + head_offset;
        const float* HWY_RESTRICT v =
            kv_cache.kv_cache.get() + kv_offset + kQKVDim;
        MulByConstAndAdd(head_att[pos], v, att_out, kQKVDim);
      }
    } else {
      for (size_t pos = start_pos; pos <= last_pos; ++pos) {
        const size_t cache_pos = div_seq_len.Remainder(pos);
        const size_t kv_offset =
            cache_pos * kCachePosSize + layer * kCacheLayerSize + head_offset;
        const float* HWY_RESTRICT v =
            kv_cache.kv_cache.get() + kv_offset + kQKVDim;
        MulByConstAndAdd(head_att[pos % kSeqLen], v, att_out, kQKVDim);
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
                size_t last_pos = pos;
                const size_t prefix_end = queries_prefix_end_[query_idx];
                if (prefix_end > 0 && prefix_end - 1 > last_pos) {
                  // last_pos in QDotK and WeightedSumV is inclusive.
                  last_pos = prefix_end - 1;
                }

                float* HWY_RESTRICT head_att =
                    activations_.att.Batch(interleaved_idx) + head * kSeqLen;
                QDotK(start_pos, last_pos, head_offset, q, kv_cache, head_att);
                // SoftMax with optional SoftCap yields "probabilities" in
                // head_att.
                const size_t head_att_len = std::min(last_pos + 1, kSeqLen);
                MaybeLogitsSoftCap(TConfig::kAttCap, head_att, head_att_len);
                Softmax(head_att, head_att_len);

                float* HWY_RESTRICT att_out =
                    activations_.att_out.Batch(interleaved_idx) +
                    head * kQKVDim;
                WeightedSumV(start_pos, last_pos, head_att, layer_, head_offset,
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
    MatMul<kAdd>(
        num_interleaved, ConstMat(activations_.att_out.All(), kHeads * kQKVDim),
        ConstMat(layer_weights_.att_weights.data(), kHeads * kQKVDim),
        layer_weights_.att_weights.scale(), bias, activations_.env,
        MutableMat(activations_.att_sums.All(), kModelDim));
  }

 public:
  // Constructor with explicit initialization of queries_prefix_end. This is
  // needed for the Prefix-LM style attention. For standard causal attention,
  // the other constructor can be used.
  GemmaAttention(const QueriesPos& queries_pos,
                 const QueriesPos& queries_prefix_end, size_t num_tokens,
                 size_t layer, Activations& activations,
                 const CompressedLayer<TConfig>* layer_weights,
                 const hwy::Divisor& div_seq_len, const KVCaches& kv_caches)
      : queries_pos_(queries_pos),
        queries_prefix_end_(queries_prefix_end),
        num_queries_(queries_pos.size()),
        num_tokens_(num_tokens),
        layer_(layer),
        activations_(activations),
        layer_weights_(*layer_weights),
        div_seq_len_(div_seq_len),
        kv_caches_(kv_caches),
        pool_(activations.env.Pool()) {
    HWY_DASSERT(num_queries_ <= kv_caches_.size());
  }
  // Constructor with default initialization to 0 for queries_prefix_end.
  GemmaAttention(const QueriesPos& queries_pos, size_t num_tokens, size_t layer,
                 Activations& activations,
                 const CompressedLayer<TConfig>* layer_weights,
                 const hwy::Divisor& div_seq_len, const KVCaches& kv_caches)
      : queries_pos_(queries_pos),
        queries_prefix_end_vec_(queries_pos.size(), 0),
        queries_prefix_end_(queries_prefix_end_vec_.data(),
                            queries_prefix_end_vec_.size()),
        num_queries_(queries_pos.size()),
        num_tokens_(num_tokens),
        layer_(layer),
        activations_(activations),
        layer_weights_(*layer_weights),
        div_seq_len_(div_seq_len),
        kv_caches_(kv_caches),
        pool_(activations.env.Pool()) {
    HWY_DASSERT(num_queries_ <= kv_caches_.size());
  }

  // Full attention computation in three steps.
  HWY_INLINE void operator()() {
    const size_t num_interleaved = num_tokens_ * num_queries_;
    ComputeQKV(num_interleaved);
    DotSoftmaxWeightedSum(num_interleaved);
    SumHeads(num_interleaved);
  }

 private:
  const QueriesPos& queries_pos_;
  const std::vector<size_t> queries_prefix_end_vec_;
  const QueriesPos queries_prefix_end_;
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
HWY_NOINLINE void Attention(
    LayerAttentionType type, const QueriesPos& queries_pos,
    const QueriesPos& queries_prefix_end, size_t num_tokens, size_t layer,
    Activations& activations, const CompressedLayer<TConfig>* layer_weights,
    const hwy::Divisor& div_seq_len, const KVCaches& kv_caches) {
  if (type == LayerAttentionType::kGemma) {
    GemmaAttention<TConfig>(queries_pos, queries_prefix_end, num_tokens, layer,
                            activations, layer_weights, div_seq_len,
                            kv_caches)();
  } else {
    // Only reached if the model is Griffin. `if constexpr` prevents generating
    // this code for non-Griffin models.
    if constexpr (TConfig::kGriffinLayers > 0) {
      HWY_ASSERT(queries_pos.size() == 1);
      GriffinRecurrent<TConfig>(queries_pos[0], num_tokens, layer, activations,
                                layer_weights, kv_caches);
    }
  }
}

// Wrapper class; holds arguments in member variables to shorten call sites.
// The main differences to GemmaAttention are:
// - no KV Cache necessary, attention is always all-to-all and not causal.
// - no potential wrap-around, attention always goes from 0 to kSeqLen.
// - no need for batching, as we are always computing attention for kSeqLen
//   tokens.
// This results in a much simpler implementation. However, to avoid duplicating
// code, we should still consider merging the two classes.
// TODO(keysers): Refactor to share code with GemmaAttention.
template <class TConfig>
class VitAttention {
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kKVHeads = TConfig::kKVHeads;
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kQStride = 3 * kQKVDim;
  static constexpr size_t kSeqLen = TConfig::kSeqLen;

  // Computes Q, K, V for all heads, stored in activations_.q.
  HWY_NOINLINE void ComputeQKV() {
    PROFILER_ZONE("Gen.VitAttention.QKV");
    const auto y =
        ConstMat(activations_.pre_att_rms_out.All(), kModelDim);
    auto& qkv = activations_.q;
    HWY_ASSERT(qkv.BatchSize() == num_tokens_);
    HWY_ASSERT(qkv.Len() == kHeads * kQStride);
    MatMul</*kAdd=*/true>(
        num_tokens_, y,
        ConstMat(layer_weights_.vit.qkv_einsum_w.data_scale1(), kModelDim),
        /*scale=*/1.0f, layer_weights_.vit.qkv_einsum_b.data_scale1(),
        activations_.env, MutableMat(qkv.All(), qkv.Len()));
  }

  HWY_NOINLINE void DotSoftmaxWeightedSum() {
    GEMMA_CONSTEXPR_SQRT float kQueryScale =
        1.0f / Sqrt(static_cast<float>(TConfig::kQKVDim));
    PROFILER_ZONE("Gen.VitAttention.DotSoftmax");
    // A "head group" in the context of GQA refers to a collection of query
    // heads that share the same key and value heads.
    static_assert(kHeads == kKVHeads, "Vit expects MHA");

    // Compute Q.K, softmax, and weighted V.
    pool_.Run(0, kHeads * num_tokens_,
              [&](uint64_t task, size_t /*thread*/) HWY_ATTR {
                const size_t head = task % kHeads;
                const size_t token = task / kHeads;
                // Compute Q.K scores, which are "logits" stored in head_att.
                float* HWY_RESTRICT q =
                    activations_.q.Batch(token) + head * kQStride;
                MulByConst(kQueryScale, q, kQKVDim);
                float* HWY_RESTRICT head_att =
                    activations_.att.Batch(token) + head * kSeqLen;
                for (size_t i = 0; i < kSeqLen; ++i) {
                  float* HWY_RESTRICT k =
                      activations_.q.Batch(i) + head * kQStride + kQKVDim;
                  head_att[i] = Dot(q, k, kQKVDim);  // score = q.k
                }
                // SoftMax yields "probabilities" in head_att.
                Softmax(head_att, kSeqLen);
                // Compute weighted sum of v into att_out.
                float* HWY_RESTRICT att_out =
                    activations_.att_out.Batch(token) + head * kQKVDim;
                hwy::ZeroBytes(att_out, kQKVDim * sizeof(*att_out));
                for (size_t i = 0; i < kSeqLen; ++i) {
                  float* HWY_RESTRICT v =
                      activations_.q.Batch(i) + head * kQStride + 2 * kQKVDim;
                  MulByConstAndAdd(head_att[i], v, att_out, kQKVDim);
                }
              });
  }

  // Sums encoded (`att_out`) over num_heads (`kHeads`) and head_dim (`kQKVDim`)
  // into output (`att_sums`).
  HWY_NOINLINE void SumHeads() {
    PROFILER_ZONE("Gen.VitAttention.SumHeads");
    auto* bias = layer_weights_.vit.attn_out_b.data_scale1();
    auto att_out = ConstMat(activations_.att_out.All(), kHeads * kQKVDim);
    auto att_weights = ConstMat(layer_weights_.vit.attn_out_w.data_scale1(),
                                kHeads * kQKVDim);
    auto att_sums = MutableMat(activations_.att_sums.All(), kModelDim);
    // att_weights and att_out are concatenated heads, each of length kQKVDim.
    // Thus the [num_tokens_, kModelDim] matmul output is the sum over heads.
    MatMul</*kAdd=*/true>(num_tokens_, att_out, att_weights, /*scale=*/1.0f,
                          bias, activations_.env, att_sums);
  }

 public:
  VitAttention(size_t num_tokens, size_t layer, Activations& activations,
               const CompressedLayer<TConfig>* layer_weights)
      : num_tokens_(num_tokens),
        layer_(layer),
        activations_(activations),
        layer_weights_(*layer_weights),
        pool_(activations.env.Pool()) {}

  HWY_INLINE void operator()() {
    ComputeQKV();
    DotSoftmaxWeightedSum();
    SumHeads();
  }

 private:
  const size_t num_tokens_;
  const size_t layer_;
  Activations& activations_;
  const CompressedLayer<TConfig>& layer_weights_;
  hwy::ThreadPool& pool_;
};

template <class TConfig, typename T>
HWY_NOINLINE void Activation(T* HWY_RESTRICT c1, T* HWY_RESTRICT c2,
                             size_t count) {
  PROFILER_ZONE("Gen.Activation");
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<T>;
  using VF = hn::Vec<DF>;
  // ActivationType::Gelu
  if (c2 == nullptr) {  // No multiplier, just Gelu.
    Gelu(c1, count);
    return;
  };
  // Has multiplier, Gelu(c1) * c2.
  hn::Transform1(DF(), c1, count, c2, [](DF df, VF v, VF mul) HWY_ATTR {
    return hn::Mul(mul, Gelu(df, v));
  });
}

template <class TConfig>
HWY_NOINLINE void FFW(Activations& activations, size_t num_interleaved,
                      const CompressedLayer<TConfig>* layer_weights) {
  PROFILER_ZONE("Gen.FFW");
  constexpr size_t kModelDim = TConfig::kModelDim;
  constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;
  constexpr bool kAddBias = TConfig::kFFBiases;
  constexpr bool kIsVit = TConfig::kLayerConfig[0] == LayerAttentionType::kVit;
  using WeightType =
      hwy::If<kIsVit,
              typename CompressedLayer<TConfig>::WeightF32OrBF16,
              typename CompressedLayer<TConfig>::Weight>;
  HWY_DASSERT(num_interleaved <= activations.bf_pre_ffw_rms_out.BatchSize());

  // Define slightly more readable names for the weights and activations.
  const auto x = ConstMat(activations.bf_pre_ffw_rms_out.All(), kModelDim);
  Mat<const WeightType> w1;
  const float* bias1 = nullptr;
  Mat<const WeightType> w2;
  const float* bias2 = nullptr;
  float scale = 1.0f;
  Mat<const WeightType> w_output;
  const float* output_bias = nullptr;
  float output_scale = 1.0f;
  auto hidden_activations = MutableMat(activations.C1.All(), kFFHiddenDim);
  auto multiplier = MutableMat(activations.C2.All(), kFFHiddenDim);
  auto ffw_out = MutableMat(activations.ffw_out.All(), kModelDim);

  // For some of the weights and activations, it depends on the config where to
  // get them from or whether to use them at all.
  if constexpr (kAddBias && !kIsVit) {
    bias1 = layer_weights->ffw_gating_biases.data_scale1();
    bias2 = bias1 + kFFHiddenDim;
    output_bias = layer_weights->ffw_output_biases.data_scale1();
  }
  if constexpr (!kIsVit) {
    w1 = layer_weights->gating_einsum_w.data() == nullptr
             ? ConstMat(layer_weights->gating_einsum_w1.data(), kModelDim)
             : ConstMat(layer_weights->gating_einsum_w.data(), kModelDim);
    w2 = layer_weights->gating_einsum_w.data() == nullptr
             ? ConstMat(layer_weights->gating_einsum_w2.data(), kModelDim)
             : ConstMat(layer_weights->gating_einsum_w.data(), kModelDim,
                        kModelDim, kModelDim * kFFHiddenDim);
    scale = layer_weights->gating_einsum_w.data() == nullptr
                ? layer_weights->gating_einsum_w1.scale()
                : layer_weights->gating_einsum_w.scale();
    w_output = ConstMat(layer_weights->linear_w.data(), kFFHiddenDim);
    output_scale = layer_weights->linear_w.scale();
  } else {
    w1 = ConstMat(layer_weights->vit.linear_0_w.data_scale1(), kModelDim);
    bias1 = layer_weights->vit.linear_0_b.data_scale1();
    multiplier.ptr = nullptr;
    w_output =
        ConstMat(layer_weights->vit.linear_1_w.data_scale1(), kFFHiddenDim);
    output_bias = layer_weights->vit.linear_1_b.data_scale1();
  }

  // Compute the hidden layer activations.
  MatMul<kAddBias>(num_interleaved, x, w1, scale, bias1, activations.env,
                   hidden_activations);
  if constexpr (!kIsVit) {
    MatMul<kAddBias>(num_interleaved, x, w2, scale, bias2, activations.env,
                     multiplier);
  }

  // Activation (Gelu) and maybe multiply by gate. Store activations in act.
  Activation<TConfig>(hidden_activations.ptr, multiplier.ptr,
                      kFFHiddenDim * num_interleaved);

  // Hidden layer -> output layer.
  MatMul<kAddBias>(num_interleaved, ConstMat(hidden_activations), w_output,
                   output_scale, output_bias, activations.env, ffw_out);
}

// `batch_idx` indicates which row of `x` to write to.
// `pos` is the *token*'s position, not the start of the batch, because this is
// called for batches of tokens in prefill, but batches of queries in decode.
template <class TConfig>
HWY_NOINLINE void EmbedToken(int token, size_t batch_idx, size_t pos,
                             size_t pos_in_prompt,
                             const CompressedWeights<TConfig>& weights,
                             RowVectorBatch<float>& x,
                             const ImageTokens* image_tokens) {
  // Image tokens just need to be copied.
  if (image_tokens != nullptr && pos_in_prompt < image_tokens->BatchSize()) {
    hwy::CopyBytes(image_tokens->Batch(pos_in_prompt), x.Batch(batch_idx),
                   x.Len() * sizeof(x.Const()[0]));
    return;
  }

  constexpr size_t kModelDim = TConfig::kModelDim;
  constexpr size_t kVocabSize = TConfig::kVocabSize;
  GEMMA_CONSTEXPR_EMBSCALING const float kEmbScaling =
      EmbeddingScaling<TConfig>();

  HWY_DASSERT(token >= 0);
  HWY_DASSERT(token < static_cast<int>(kVocabSize));

  const hn::ScalableTag<float> df;
  DecompressAndZeroPad(
      df,
      MakeSpan(weights.embedder_input_embedding.data(), kVocabSize * kModelDim),
      token * kModelDim, x.Batch(batch_idx), kModelDim);
  MulByConst(kEmbScaling * weights.embedder_input_embedding.scale(),
             x.Batch(batch_idx), kModelDim);
  if constexpr (TConfig::kAbsolutePE) {
    AddAbsolutePositionalEmbeddings(x.Batch(batch_idx), kModelDim, pos);
  }
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
    const QueriesPos& queries_pos, const QueriesPos& queries_prefix_end,
    size_t num_tokens, size_t layer,
    const CompressedLayer<TConfig>* layer_weights, Activations& activations,
    const hwy::Divisor& div_seq_len, const KVCaches& kv_caches) {
  constexpr size_t kModelDim = TConfig::kModelDim;
  const size_t num_interleaved = num_tokens * queries_pos.size();
  auto type = TConfig::kLayerConfig[layer];
  size_t layer_of_type =
      NumLayersOfTypeBefore(TConfig::kLayerConfig, type, layer);

  RMSNormBatched(num_interleaved, activations.x.All(),
                 layer_weights->pre_attention_norm_scale.data_scale1(),
                 activations.pre_att_rms_out.All(), kModelDim);

  Attention<TConfig>(type, queries_pos, queries_prefix_end, num_tokens,
                     layer_of_type, activations, layer_weights, div_seq_len,
                     kv_caches);

  PostNorm<TConfig>(num_interleaved, layer_weights->post_attention_norm_scale,
                    activations.att_sums.All());

  ResidualConnection<TConfig>(num_interleaved, activations.att_sums.All(),
                              activations.x.All(), layer_weights,
                              /*is_attention=*/true);

  RMSNormBatched(num_interleaved, activations.x.All(),
                 layer_weights->pre_ffw_norm_scale.data_scale1(),
                 activations.bf_pre_ffw_rms_out.All(), kModelDim);

  FFW<TConfig>(activations, num_interleaved, layer_weights);

  PostNorm<TConfig>(num_interleaved, layer_weights->post_ffw_norm_scale,
                    activations.ffw_out.All());

  ResidualConnection<TConfig>(num_interleaved, activations.ffw_out.All(),
                              activations.x.All(), layer_weights,
                              /*is_attention=*/false);
}

// Vit transformer layer. Some comments below refer to the Vit implementation in
// the Big Vision codebase. See
// github.com/google-research/big_vision/blob/main/big_vision/models/vit.py
// TODO(keysers): consider adding a wrapper for both LayerNorm with RMSNorm and
// try mergig this with TransformerLayer.
template <class TConfig>
HWY_NOINLINE void VitTransformerLayer(
    size_t num_tokens, size_t layer,
    const CompressedLayer<TConfig>* layer_weights, Activations& activations) {
  constexpr size_t kModelDim = TConfig::kModelDim;
  auto type = TConfig::kLayerConfig[layer];
  HWY_ASSERT(type == LayerAttentionType::kVit);

  auto& x = activations.x;
  HWY_ASSERT(x.BatchSize() == num_tokens);
  HWY_ASSERT(x.Len() == kModelDim);

  // y = nn.LayerNorm()(x)
  // y ~ pre_att_rms_out
  LayerNormBatched(num_tokens, x.All(),
                   layer_weights->vit.layer_norm_0_scale.data_scale1(),
                   layer_weights->vit.layer_norm_0_bias.data_scale1(),
                   activations.pre_att_rms_out.All(), kModelDim);
  // y = out["sa"] = nn.MultiHeadDotProductAttention(...)(y, y)
  // y ~ att_sums
  VitAttention<TConfig>(num_tokens, layer, activations, layer_weights)();

  // x = out["+sa"] = x + y
  AddFromBatched(num_tokens, activations.att_sums.All(), x.All(), kModelDim);

  // y = nn.LayerNorm()(x)
  // y ~ bf_pre_ffw_rms_out
  LayerNormBatched(num_tokens, x.All(),
                   layer_weights->vit.layer_norm_1_scale.data_scale1(),
                   layer_weights->vit.layer_norm_1_bias.data_scale1(),
                   activations.bf_pre_ffw_rms_out.All(), kModelDim);

  // y = out["mlp"] = MlpBlock(...)(y)
  // y ~ ffw_out
  FFW<TConfig>(activations, num_tokens, layer_weights);

  // x = out["+mlp"] = x + y
  AddFromBatched(num_tokens, activations.ffw_out.All(), x.All(), kModelDim);
}

// Prefill() and Transformer() increment positions in-place.
using QueriesMutablePos = hwy::Span<size_t>;

// Populates KV cache for batches of tokens from one query at a time.
template <class TConfig>
HWY_NOINLINE void Prefill(
    const QueriesPromptTokens& queries_prompt,
    const QueriesMutablePos& queries_pos, const QueriesPos& queries_prefix_end,
    const size_t query_idx_start, const CompressedWeights<TConfig>& weights,
    Activations& activations, const RuntimeConfig& runtime_config,
    const hwy::Divisor& div_seq_len, const KVCaches& kv_caches) {
  PROFILER_ZONE("Gen.Prefill");
  const size_t num_queries = queries_prompt.size();
  HWY_ASSERT(queries_pos.size() == num_queries);
  HWY_ASSERT(queries_prefix_end.size() == num_queries);
  HWY_ASSERT(kv_caches.size() == num_queries);

  // Batches are important for amortizing loading weights over multiple tokens.
  // This is possible in prefill because we know all tokens beforehand, whereas
  // decode depends on the previous output token. However, each prefill batch of
  // a query requires that preceding batches already wrote to the KV cache,
  // hence we sequentially loop over token batches. We can reduce the number of
  // iterations by increasing the batch size, but this also increases arithmetic
  // intensity, and so we are eventually compute-limited. We could devote some
  // threads to parallelizing over queries, but for simplicity we assign them
  // all to MatMul.
  const size_t max_tbatch_size = activations.x.BatchSize();

  // For each query. `qi` is within the batch, not the global query index.
  for (size_t qi = 0; qi < num_queries; ++qi) {
    // Single query at a time, so pass slices of the spans because
    // GemmaAttention will only access the first KV cache and position.
    QueriesPos single_query_pos(&queries_pos[qi], 1);
    QueriesPos single_query_prefix_end(&queries_prefix_end[qi], 1);
    KVCaches single_kv_cache(&kv_caches[qi], 1);

    const size_t prompt_size = queries_prompt[qi].size();
    // In autoregressive mode, we don't need to prefill the last token, so - 1.
    size_t prefill_this_query = prompt_size - 1;
    const size_t prefix_end_this_query = queries_prefix_end[qi];
    // We can't attend beyond the prompt_size.
    HWY_ASSERT(prefix_end_this_query <= prompt_size);
    // Special case: if the prefix includes the last token, we need to prefill
    // the last token, too. However, we need to rewind this for the generation
    // of the first token. So we need to keep track of this.
    // TODO: consider implementing masking instead of this logic?
    const bool attend_to_last_token =
        (prefill_this_query < prefix_end_this_query);
    if (attend_to_last_token) {
      // The difference can be at most 1.
      prefill_this_query += 1;
      HWY_ASSERT(prefill_this_query == prefix_end_this_query);
    }
    // In prefix-LM mode, we need to look at all the tokens for the prefix in
    // one iteration through the layers, so we need a large enough batch size.
    HWY_ASSERT(prefix_end_this_query == 0 ||
               max_tbatch_size >= prefill_this_query);

    // For each batch of tokens in the query:
    for (size_t tbatch_start = 0; tbatch_start < prefill_this_query;
         tbatch_start += max_tbatch_size) {
      const size_t tbatch_size =
          HWY_MIN(max_tbatch_size, prefill_this_query - tbatch_start);

      // Fill activations.x (much faster than TransformerLayer).
      for (size_t ti = 0; ti < tbatch_size; ++ti) {
        const size_t pos = queries_pos[qi] + ti;
        const size_t pos_in_prompt = tbatch_start + ti;
        const int token = queries_prompt[qi][pos_in_prompt];
        EmbedToken<TConfig>(token, ti, pos, pos_in_prompt, weights,
                            activations.x, runtime_config.image_tokens);
      }

      // Transformer with one batch of tokens from a single query.
      for (size_t layer = 0; layer < TConfig::kLayers; ++layer) {
        const auto* layer_weights = weights.GetLayer(layer);
        TransformerLayer<TConfig>(single_query_pos, single_query_prefix_end,
                                  tbatch_size, layer, layer_weights,
                                  activations, div_seq_len, single_kv_cache);
      }

      // NOTE: we unconditionally call StreamToken, even if EOS.
      for (size_t ti = 0; ti < tbatch_size; ++ti) {
        const size_t pos = queries_pos[qi] + ti;
        const size_t pos_in_prompt = tbatch_start + ti;
        const int token = queries_prompt[qi][pos_in_prompt];
        if (pos_in_prompt < prompt_size - 1) {
          runtime_config.StreamToken(query_idx_start + qi, pos, token, 0.0f);
        } else {
          // The last token will be streamed later and we should only get here
          // if we need to attend to the last token because it is in the prefix.
          HWY_ASSERT(attend_to_last_token);
        }
      }

      queries_pos[qi] += tbatch_size;
    }  // for tbatch_start
    if (attend_to_last_token) {
      // We need to rewind the position for the last token that we only
      // attended to to make sure the prefix LM sees everything.
      // This means we duplicate work on the last prompt token in autoregressive
      // decoding. Alternatives: (1) real masking; (2) always prefill the last
      // token and only generate the next one from the already prefilled
      // activations.
      queries_pos[qi] -= 1;
    }
  }
}

// Gets the patches of the image and embeds them with the image embedding
// kernel. The result is stored in activations.x.
template <class TConfig>
HWY_NOINLINE void EmbedImagePatches(const Image& image,
                                    const CompressedWeights<TConfig>& weights,
                                    Activations& activations) {
  static constexpr size_t kModelDim = TConfig::VitConfig::kModelDim;
  static constexpr size_t kPatchWidth = TConfig::VitConfig::kPatchWidth;
  static constexpr size_t kSeqLen = TConfig::VitConfig::kSeqLen;
  constexpr size_t kPatchSize = kPatchWidth * kPatchWidth * 3;
  HWY_ASSERT(weights.vit_img_embedding_kernel.NumElements() ==
             kPatchSize * kModelDim);
  HWY_ASSERT(activations.x.Len() == kModelDim);
  std::vector<hwy::AlignedFreeUniquePtr<float[]>> image_patches(kSeqLen);
  for (size_t i = 0; i < kSeqLen; ++i) {
    image_patches[i] = hwy::AllocateAligned<float>(kPatchSize);
    image.GetPatch(i, image_patches[i].get());
  }
  // img/embedding/kernel has original shape (14, 14, 3, 1152)
  // H x W x C x D transposed to D x (H x W x C) so here (1152, 14 * 14 * 3)
  // image_patches is (256, 14 * 14 * 3)
  // This could be done as one MatMul like:
  // RowVectorBatch<float> image_patches(kSeqLen, kPatchSize);
  // [Get patches]
  // MatMul</*kAdd=*/true>(
  //       kVitSeqLen, ConstMat(image_patches.All(), kPatchSize),
  //       ConstMat(weights.vit_img_embedding_kernel.data_scale1(), kPatchSize),
  //       /*scale=*/1.0f, weights.vit_img_embedding_bias.data_scale1(),
  //       activations.env, MutableMat(activations.x.All(), kVitModelDim));
  // However, MatMul currently requires that
  //   A.cols % (2 * hn::Lanes(hn::ScalableTag<MulT>())) == 0
  // which is not the case here. We should relax that requirement on MatMul and
  // then use the above. For now, we rely on MatVecAdd instead.
  for (size_t i = 0; i < kSeqLen; ++i) {
    MatVecAdd<kModelDim, kPatchSize>(
        weights.vit_img_embedding_kernel, 0, image_patches[i].get(),
        weights.vit_img_embedding_bias.data_scale1(), activations.x.Batch(i),
        activations.env.Pools().Outer());
  }
  // Add position embeddings.
  AddFrom(weights.vit_img_pos_embedding.data_scale1(), activations.x.All(),
          kSeqLen * kModelDim);
}

// Prefills the image tokens with the ViT encoder.
template <class TConfig>
HWY_NOINLINE void PrefillVit(const CompressedWeights<TConfig>& weights,
                             const RuntimeConfig& runtime_config,
                             const Image& image, ImageTokens& image_tokens,
                             Activations& activations) {
  PROFILER_ZONE("Gen.PrefillVit");
  const size_t num_tokens = TConfig::VitConfig::kSeqLen;
  const size_t kVitModelDim = TConfig::VitConfig::kModelDim;
  HWY_ASSERT(num_tokens == activations.x.BatchSize());
  // Embed the image patches.
  EmbedImagePatches<TConfig>(image, weights, activations);
  // Go through all layers.
  for (size_t layer = 0; layer < TConfig::VitConfig::kLayers; ++layer) {
    const auto* layer_weights = weights.GetVitLayer(layer);
    VitTransformerLayer<typename TConfig::VitConfig>(
        num_tokens, layer, layer_weights, activations);
  }
  // Final Layernorm.
  LayerNormBatched(num_tokens, activations.x.All(),
                   weights.vit_encoder_norm_scale.data_scale1(),
                   weights.vit_encoder_norm_bias.data_scale1(),
                   activations.x.All(), kVitModelDim);

  // Apply head embedding into image_tokens of size of the LLM kModelDim.
  MatMul</*kAdd=*/true>(
      num_tokens, ConstMat(activations.x.All(), kVitModelDim),
      ConstMat(weights.vit_img_head_kernel.data_scale1(), kVitModelDim),
      /*scale=*/1.0f, weights.vit_img_head_bias.data_scale1(), activations.env,
      MutableMat(image_tokens.All(), TConfig::kModelDim));
}

// Generates one token for each query. `queries_token` is the previous token
// from each query, and `queries_pos` are their position in the sequence.
template <class TConfig>
HWY_NOINLINE void Transformer(
    const QueriesToken& queries_token, const QueriesMutablePos& queries_pos,
    const QueriesPos& queries_prefix_end,
    const CompressedWeights<TConfig>& weights, Activations& activations,
    const hwy::Divisor& div_seq_len, const KVCaches& kv_caches,
    const LayersOutputFunc& layers_output,
    const ActivationsObserverFunc& activations_observer) {
  constexpr size_t kModelDim = TConfig::kModelDim;
  const size_t num_queries = queries_token.size();
  HWY_DASSERT(queries_pos.size() == num_queries);
  HWY_DASSERT(queries_prefix_end.size() == num_queries);

  if (layers_output) {
    for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
      const float token_f = queries_token[query_idx];
      layers_output(query_idx, queries_pos[query_idx], "tokens", -1, &token_f,
                    1);
    }
  }

  for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
    EmbedToken<TConfig>(queries_token[query_idx], query_idx,
                        queries_pos[query_idx], /*pos_in_prompt=*/0, weights,
                        activations.x, /*image_tokens=*/nullptr);
  }

  for (size_t layer = 0; layer < TConfig::kLayers; ++layer) {
    const CompressedLayer<TConfig>* layer_weights = weights.GetLayer(layer);
    TransformerLayer<TConfig>(queries_pos, queries_prefix_end, /*num_tokens=*/1,
                              layer, layer_weights, activations, div_seq_len,
                              kv_caches);

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
static size_t MaxQueryLength(const QueriesPromptTokens& queries_prompt) {
  size_t max_prompt_size = 0;
  for (size_t i = 0; i < queries_prompt.size(); ++i) {
    max_prompt_size = std::max(max_prompt_size, queries_prompt[i].size());
  }
  return max_prompt_size;
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

template <class TConfig>
SampleFunc ChooseSampleFunc(const RuntimeConfig& runtime_config) {
  constexpr size_t kTopK = TConfig::kTopK;

  // If user provided a sample_func, use it.
  if (runtime_config.sample_func) return runtime_config.sample_func;

  // Fast path for top-1 with no accept_token.
  if (kTopK == 1 && !runtime_config.accept_token) {
    return [](float* logits, size_t vocab_size) HWY_ATTR -> TokenAndProb {
      PROFILER_ZONE("Gen.Sample Top1");
      return Top1OfSoftmax(logits, vocab_size);
    };
  }

  // General case: Softmax with top-k sampling.
  return [&runtime_config](float* logits,
                           size_t vocab_size) HWY_ATTR -> TokenAndProb {
    PROFILER_ZONE("Gen.Sample general");
    Softmax(logits, vocab_size);
    const int token = SampleTopK<kTopK>(logits, vocab_size, *runtime_config.gen,
                                        runtime_config.temperature,
                                        runtime_config.accept_token);
    return TokenAndProb{.token = token, .prob = logits[token]};
  };
}

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
               const QueriesPos& queries_pos_in,
               const QueriesPos& queries_prefix_end,
               const size_t query_idx_start, const KVCaches& kv_caches,
               TimingInfo& timing_info) {
  constexpr size_t kModelDim = TConfig::kModelDim;
  constexpr size_t kVocabSize = TConfig::kVocabSize;
  const CompressedWeights<TConfig>& weights =
      *reinterpret_cast<const CompressedWeights<TConfig>*>(weights_u8.get());

  // Copy so we can increment without requiring users to pass in a mutable span.
  std::vector<size_t> queries_pos_copy(queries_pos_in.cbegin(),
                                       queries_pos_in.cend());
  QueriesMutablePos queries_mutable_pos(queries_pos_copy.data(),
                                        queries_pos_copy.size());
  // For the first turn, qpos remains 0. Otherwise, rewind the previous EOS.
  // Background: for multiturn, Gemma 2 expects only <end_of_turn>, not EOS. The
  // previous `Generate` called `StreamToken` for the last token (EOS), hence
  // our caller's qpos is 1 too high. This must be corrected because we didn't
  // write to the KV cache at that position, so MSAN would complain.
  for (size_t& qpos : queries_mutable_pos) {
    qpos = qpos == 0 ? 0 : qpos - 1;
  }
  // Sanity check: prompts should not be empty, nor start with EOS.
  for (size_t query_idx = 0; query_idx < queries_prompt.size(); ++query_idx) {
    const PromptTokens& prompt = queries_prompt[query_idx];
    HWY_ASSERT(prompt.size() != 0 && prompt[0] != runtime_config.eos_id);
  }

  const size_t num_queries = queries_prompt.size();
  HWY_ASSERT(num_queries <= 4096);  // TokenStreamer uses BitSet4096.
  HWY_ASSERT(num_queries <= activations.x.BatchSize());
  HWY_ASSERT(queries_pos_in.size() == num_queries);
  HWY_ASSERT(kv_caches.size() == num_queries);
  const hwy::Divisor div_seq_len(static_cast<uint32_t>(kv_caches[0].seq_len));

  size_t max_prompt_size = MaxQueryLength(queries_prompt);
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

  const SampleFunc sample_token = ChooseSampleFunc<TConfig>(runtime_config);

  // Prefill stops before min_prompt_size - 1 because the last prompt
  // token is the first input token for generation.
  const double prefill_start = hwy::platform::Now();
  // If tbatch is larger than the qbatch we already have in `activations`, then
  // allocate prefill_activations, otherwise reuse.
  const bool use_prefill_activations =
      runtime_config.prefill_tbatch_size > activations.x.BatchSize();
  Activations prefill_activations;
  if (use_prefill_activations) {
    prefill_activations.Allocate<TConfig>(runtime_config.prefill_tbatch_size,
                                          activations.env.Pools());
  }
  Prefill<TConfig>(queries_prompt, queries_mutable_pos, queries_prefix_end,
                   query_idx_start, weights,
                   use_prefill_activations ? prefill_activations : activations,
                   runtime_config, div_seq_len, kv_caches);
  // Compute the number of tokens that were prefilled and notify timing_info.
  size_t prefilled_tokens = 0;
  for (size_t qi = 0; qi < num_queries; ++qi) {
    prefilled_tokens += queries_prompt[qi].size() - 1;
  }
  timing_info.NotifyPrefill(prefilled_tokens, prefill_start);
  // queries_pos are incremented by Prefill.

  // Storage for the last generated token from each query, passed to the next
  // Transformer() call.
  std::vector<int> gen_tokens(num_queries);

  // Stream the last prompt token from each query and fill gen_tokens.
  TokenStreamer token_streamer(runtime_config);
  for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
    size_t last_token_pos_in_prompt =
        queries_mutable_pos[query_idx] - queries_pos_in[query_idx];
    gen_tokens[query_idx] = queries_prompt[query_idx][last_token_pos_in_prompt];
    (void)token_streamer(query_idx_start + query_idx,
                         queries_mutable_pos[query_idx], gen_tokens[query_idx],
                         0.0f);
  }

  const double gen_start = hwy::platform::Now();
  for (size_t gen = 0; gen < HWY_MIN(max_tokens, max_generated_tokens); ++gen) {
    // Decode generates one token per query and increments queries_mutable_pos.
    Transformer<TConfig>(
        QueriesToken(gen_tokens.data(), num_queries), queries_mutable_pos,
        queries_prefix_end, weights, activations, div_seq_len, kv_caches,
        runtime_config.layers_output, runtime_config.activations_observer);
    // queries_pos are incremented by Transformer.

    bool all_queries_eos = true;
    {
      PROFILER_ZONE("Gen.EmbeddingMatmul");
      // Compute logits from last layer activations.
      MatMul</*kAdd=*/false>(
          num_queries, ConstMat(activations.x.All(), kModelDim),
          ConstMat(weights.embedder_input_embedding.data(), kModelDim),
          weights.embedder_input_embedding.scale(), /*add=*/nullptr,
          activations.env, MutableMat(activations.logits.All(), kVocabSize));
    }
    PROFILER_ZONE("Gen.Softcap+Sample+Stream");
    for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
      float* HWY_RESTRICT logits = activations.logits.Batch(query_idx);
      MaybeLogitsSoftCap(TConfig::kFinalCap, logits, kVocabSize);
      const TokenAndProb tp = sample_token(logits, kVocabSize);
      timing_info.NotifyGenerated(prefill_start, gen_start);

      const bool is_eos =
          token_streamer(query_idx_start + query_idx,
                         queries_mutable_pos[query_idx], tp.token, tp.prob);
      all_queries_eos &= is_eos;
      gen_tokens[query_idx] = is_eos ? runtime_config.eos_id : tp.token;
    }
    if (all_queries_eos) break;
  }  // foreach token to generate

  timing_info.NotifyGenerateDone(gen_start);
}

template <class TConfig>
void GenerateSingleT(const ByteStorageT& weights_u8,
                     const RuntimeConfig& runtime_config,
                     const PromptTokens& prompt, size_t pos, size_t prefix_end,
                     KVCache& kv_cache, PerClusterPools& pools,
                     TimingInfo& timing_info) {
  constexpr size_t kNumQueries = 1;
  const size_t qbatch_start = 0;

  // TODO: move into Gemma?
  Activations activations;
  activations.Allocate<TConfig>(kNumQueries, pools);

  const QueriesPromptTokens queries_prompt(&prompt, kNumQueries);
  QueriesPos queries_pos(&pos, kNumQueries);
  const QueriesPos queries_prefix_end(&prefix_end, kNumQueries);
  const KVCaches kv_caches{&kv_cache, kNumQueries};

  GenerateT<TConfig>(weights_u8, activations, runtime_config, queries_prompt,
                     queries_pos, queries_prefix_end, qbatch_start, kv_caches,
                     timing_info);
}

template <class TConfig>
void GenerateBatchT(const ByteStorageT& weights_u8,
                    const RuntimeConfig& runtime_config,
                    const QueriesPromptTokens& queries_prompt,
                    const QueriesPos& queries_pos,
                    const QueriesPos& queries_prefix_end,
                    const KVCaches& kv_caches, PerClusterPools& pools,
                    TimingInfo& timing_info) {
  const size_t num_queries = queries_prompt.size();
  HWY_ASSERT(queries_pos.size() == num_queries);
  HWY_ASSERT(kv_caches.size() == num_queries);
  // Griffin does not support query batching.
  const size_t max_qbatch_size =
      (TConfig::kGriffinLayers > 0) ? 1 : runtime_config.decode_qbatch_size;

  Activations activations;
  activations.Allocate<TConfig>(max_qbatch_size, pools);

  for (size_t qbatch_start = 0; qbatch_start < num_queries;
       qbatch_start += max_qbatch_size) {
    // Generate one batch of tokens from `qbatch_size` queries.
    const size_t qbatch_size =
        HWY_MIN(num_queries - qbatch_start, max_qbatch_size);
    const QueriesPromptTokens qbatch_prompts(&queries_prompt[qbatch_start],
                                             qbatch_size);
    QueriesPos qbatch_pos(&queries_pos[qbatch_start], qbatch_size);
    const QueriesPos qbatch_prefix_end(&queries_prefix_end[qbatch_start],
                                             qbatch_size);
    const KVCaches qbatch_kv(&kv_caches[qbatch_start], qbatch_size);
    GenerateT<TConfig>(weights_u8, activations, runtime_config, qbatch_prompts,
                       qbatch_pos, qbatch_prefix_end, qbatch_start, qbatch_kv,
                       timing_info);
  }
}

template <class TConfig>
void GenerateImageTokensT(const ByteStorageT& weights_u8,
                          const RuntimeConfig& runtime_config,
                          const Image& image, ImageTokens& image_tokens,
                          PerClusterPools& pools) {
  if constexpr (TConfig::VitConfig::kLayers == 0) {
    return;
  } else {
    Activations prefill_activations;
    RuntimeConfig prefill_runtime_config = runtime_config;
    prefill_runtime_config.prefill_tbatch_size = TConfig::VitConfig::kSeqLen;
    prefill_activations.Allocate<typename TConfig::VitConfig>(
        prefill_runtime_config.prefill_tbatch_size, pools);
    // Weights are for the full PaliGemma model, not just the ViT part.
    const CompressedWeights<TConfig>& weights =
        *reinterpret_cast<const CompressedWeights<TConfig>*>(weights_u8.get());
    PrefillVit<TConfig>(weights, prefill_runtime_config, image, image_tokens,
                        prefill_activations);
  }
}

}  // namespace HWY_NAMESPACE

#if HWY_ONCE

// These are extern functions defined by instantiations/*.cc, which include this
// 'header' after defining GEMMA_CONFIG, which is for function overloading.
void GenerateSingle(  // NOLINT(misc-definitions-in-headers)
    GEMMA_CONFIG, const ByteStorageT& weights_u8,
    const RuntimeConfig& runtime_config, const PromptTokens& prompt, size_t pos,
    size_t prefix_end, KVCache& kv_cache, PerClusterPools& pools,
    TimingInfo& timing_info) {
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(GenerateSingleT<GEMMA_CONFIG>)
  (weights_u8, runtime_config, prompt, pos, prefix_end, kv_cache, pools,
   timing_info);
}

void GenerateBatch(  // NOLINT(misc-definitions-in-headers)
    GEMMA_CONFIG, const ByteStorageT& weights_u8,
    const RuntimeConfig& runtime_config,
    const QueriesPromptTokens& queries_prompt, const QueriesPos& queries_pos,
    const QueriesPos& queries_prefix_end, const KVCaches& kv_caches,
    PerClusterPools& pools, TimingInfo& timing_info) {
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(GenerateBatchT<GEMMA_CONFIG>)
  (weights_u8, runtime_config, queries_prompt, queries_pos, queries_prefix_end,
   kv_caches, pools, timing_info);
}

void GenerateImageTokens(  // NOLINT(misc-definitions-in-headers)
    GEMMA_CONFIG, const ByteStorageT& weights_u8,
    const RuntimeConfig& runtime_config, const Image& image,
    ImageTokens& image_tokens, PerClusterPools& pools) {
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(GenerateImageTokensT<GEMMA_CONFIG>)
  (weights_u8, runtime_config, image, image_tokens, pools);
}

#endif  // HWY_ONCE

}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_INL_H_
