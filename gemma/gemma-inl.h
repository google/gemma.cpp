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

#include <math.h>  // sqrtf
#include <stddef.h>
#include <stdio.h>

#include <algorithm>  // std::min
#include <vector>

#include "compression/compress.h"
#include "gemma/activations.h"
#include "gemma/common.h"
#include "gemma/configs.h"
#include "gemma/gemma.h"
#include "gemma/kv_cache.h"
#include "gemma/weights.h"
#include "paligemma/image.h"
#include "util/allocator.h"
#include "util/basics.h"
#include "util/threading.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/bit_set.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"
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

#ifndef GEMMA_TYPE
#if HWY_IDE
// Provide a definition so the IDE does not complain.
#define GEMMA_TYPE float
#else
#error "Only include from instantiations/*.cc, which must define GEMMA_TYPE"
#endif  // HWY_IDE
#endif  // GEMMA_TYPE

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// Different functions use different naming conventions for the number of
// tokens. Functions that are query-independent, such as RMSNorm*, call the
// count `num_interleaved`. Functions that are query-dependent, such as
// `Attention`, use separate `num_tokens` and `num_queries`.

// TODO: add batch query support for Griffin (QueriesPos).
template <typename T>
HWY_NOINLINE void GriffinRecurrent(size_t batch_start, size_t num_tokens,
                                   size_t layer, Activations& activations,
                                   const LayerWeightsPtrs<T>* layer_weights,
                                   const KVCaches& kv_caches) {
  PROFILER_ZONE("Gen.Griffin");
  KVCache& kv_cache = kv_caches[0];
  hwy::ThreadPool& pool = activations.env->parallel.Pools().Pool(0);
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  const size_t model_dim = layer_weights->layer_config.model_dim;
  const size_t conv_1d_width = layer_weights->layer_config.conv1d_width;
  const size_t heads = layer_weights->layer_config.heads;

  // X / Y linear layers.
  for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
    float* HWY_RESTRICT y = activations.griffin_y.Batch(batch_idx);
    float* HWY_RESTRICT x = activations.griffin_x.Batch(batch_idx);
    TwoMatVecAdd(layer_weights->griffin.linear_x_w,
                 layer_weights->griffin.linear_y_w, 0, model_dim, model_dim,
                 activations.pre_att_rms_out.Batch(batch_idx),
                 /*add0=*/layer_weights->griffin.linear_x_biases.data_scale1(),
                 /*add1=*/layer_weights->griffin.linear_y_biases.data_scale1(),
                 /*out0=*/x, /*out1=*/y, pool);
    Gelu(y, model_dim);
  }

  // Conv1D.
  for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
    const size_t pos = batch_start + batch_idx;
    float* HWY_RESTRICT x = activations.griffin_x.Batch(batch_idx);
    HWY_FULL(float) df;
    HWY_DASSERT(model_dim % hn::Lanes(df) == 0);
    const size_t layer_offset = layer * model_dim * (conv_1d_width - 1);

    // cache[i] = input at time t-i.
    float* HWY_RESTRICT cache[kMaxConv1DWidth];
    cache[0] = x;
    for (size_t i = 1; i < conv_1d_width; i++) {
      cache[i] =
          kv_cache.conv1d_cache.get() + layer_offset +
          ((pos + conv_1d_width - 1 - i) % (conv_1d_width - 1)) * model_dim;
    }
    for (size_t i = 0; i < model_dim; i += hn::Lanes(df)) {
      auto xv = hn::Load(df, x + i);
      auto accum0 =
          hn::Load(df, layer_weights->griffin.conv_biases.data_scale1() + i);
      auto accum1 = hn::Zero(df);
      HWY_ASSERT_M(conv_1d_width % 2 == 0, "Conv width must be even");
      for (size_t l = 0; 2 * l < conv_1d_width; l++) {
        auto wv0 =
            hn::Load(df, layer_weights->griffin.conv_w.data_scale1() +
                             (conv_1d_width - 1 - 2 * l) * model_dim + i);
        auto wv1 =
            hn::Load(df, layer_weights->griffin.conv_w.data_scale1() +
                             (conv_1d_width - 2 - 2 * l) * model_dim + i);
        accum0 = hn::MulAdd(wv0, hn::Load(df, cache[l * 2] + i), accum0);
        accum1 = hn::MulAdd(wv1, hn::Load(df, cache[l * 2 + 1] + i), accum1);
      }
      hn::Store(hn::Add(accum0, accum1), df, x + i);
      hn::Store(xv, df, cache[HWY_MAX(conv_1d_width, 1) - 1] + i);
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
        kv_cache.rglru_cache.get() + layer * model_dim;

    pool.Run(0, heads, [&](const uint64_t head, size_t /*thread*/) HWY_ATTR {
      const size_t kHeadDim = model_dim / heads;
      const size_t kMatrixSize = kHeadDim * kHeadDim;
      size_t head_offset = head * kHeadDim;
      TwoOfsMatVecAddLoop(
          layer_weights->griffin.gate_w, kMatrixSize * head,
          kMatrixSize * (heads + head), kHeadDim, kHeadDim, x + head_offset,
          /*add0=*/layer_weights->griffin.gate_biases.data_scale1() +
              head_offset,
          /*add1=*/layer_weights->griffin.gate_biases.data_scale1() +
              model_dim + head_offset,
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
    MatVecAdd(layer_weights->griffin.linear_out_w, 0, model_dim, model_dim, x,
              layer_weights->griffin.linear_out_biases.data_scale1(), out_ptr,
              pool);
  }
}

// Wrapper class; holds arguments in member variables to shorten call sites.
template <typename T>
class GemmaAttention {
  // The attention window usually starts at 0 unless `pos` is larger than
  // the attention window size, then it is `pos` - window_size + 1.
  HWY_INLINE size_t StartPos(size_t pos, size_t layer) {
    const size_t att_window_size =
        activations_.weights_config.attention_window_sizes[layer];
    return pos - std::min(att_window_size - 1, pos);
  }

  template <typename U>
  HWY_INLINE void PositionalEncodingQK(U* qk, size_t pos, size_t layer,
                                       const float mul) {
    // qk is either q or k, so qkv_dim is the length we operate on.
    const size_t qkv_dim = layer_config_.qkv_dim;
    const float* inv_timescale = activations_.inv_timescale.Const();
    // PostQKType::Rope
    (void)layer;
    if (layer_weights_.layer_config.post_qk == PostQKType::HalfRope) {
      Rope(qk, qkv_dim / 2, inv_timescale, pos);
      if (mul != 1.0f) MulByConst(mul, qk, qkv_dim);
    } else {
      RopeAndMulBy(mul, qk, qkv_dim, inv_timescale, pos);
    }
  }

  // Fills activations.q and computes KV. For is_mha_, a single MatMul suffices
  // and we later copy KV from q to KVCache. Otherwise, a second MatMul writes
  // KV directly to KVCache.
  HWY_NOINLINE void ComputeQKV(const size_t num_interleaved) {
    PROFILER_ZONE("Gen.Attention.QKV");
    const size_t model_dim = layer_config_.model_dim;
    const size_t qkv_dim = layer_config_.qkv_dim;
    const size_t heads = layer_config_.heads;
    const size_t kv_heads = layer_config_.kv_heads;

    const auto pre_att_rms_out =
        ConstMatFromBatch(num_interleaved, activations_.pre_att_rms_out);
    auto w_q1 = layer_weights_.qkv_einsum_w.data()
                    ? ConstMatFromWeights(layer_weights_.qkv_einsum_w)
                    : ConstMatFromWeights(layer_weights_.qkv_einsum_w1);
    // The original qkv_einsum_w has shape [(heads + kv_heads * 2), kKQVDim,
    // model_dim], which we reshaped to (heads + kv_heads * 2) * kKQVDim rows.
    // We must shrink to the actual size because MatMul verifies
    // `B.extents.rows == C.Cols()`. If MHA, `QStride() == 3 * qkv_dim` and all
    // rows are used. Otherwise, `QStride() == qkv_dim` and KV will be
    // computed in the second MatMul.
    const size_t w1_rows = heads * layer_config_.QStride();
    w_q1.ShrinkRows(w1_rows);
    MatMul(pre_att_rms_out, w_q1,
           /*add=*/nullptr, *activations_.env, RowPtrFromBatch(activations_.q));

    if (is_mha_) {
      // Multi-Head Attention a.k.a. "use_qkv_einsum" computed QKV already.
    } else {
      auto w_q2 = layer_weights_.qkv_einsum_w.data()
                      ? ConstMatFromWeights(layer_weights_.qkv_einsum_w,
                                            w1_rows * model_dim)
                      : ConstMatFromWeights(layer_weights_.qkv_einsum_w2);
      // KV structure is [k, v, k, v, ....] = kv_heads pairs of (k, v).
      const size_t w_rows_kv_cols = kv_heads * 2 * qkv_dim;
      w_q2.ShrinkRows(w_rows_kv_cols);

      // Single query and no wraparound means we can use a matmul and write
      // directly into the KV cache with a stride of cache_pos_size_.
      if (num_queries_ == 1 &&
          queries_pos_[0] + num_tokens_ <= div_seq_len_.GetDivisor()) {
        const size_t kv_ofs =
            queries_pos_[0] * cache_pos_size_ + layer_ * cache_layer_size_;
        float* HWY_RESTRICT kv = kv_caches_[0].kv_cache.get() + kv_ofs;
        RowPtrF kv_rows(kv, w_rows_kv_cols);
        kv_rows.SetStride(cache_pos_size_);
        MatMul(pre_att_rms_out, w_q2,
               /*add=*/nullptr, *activations_.env, kv_rows);
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
              cache_pos * cache_pos_size_ + layer_ * cache_layer_size_;
          float* HWY_RESTRICT kv = kv_cache.kv_cache.get() + kv_offset;
          if (layer_weights_.qkv_einsum_w.data()) {
            MatVec(layer_weights_.qkv_einsum_w, heads * qkv_dim * model_dim,
                   w_rows_kv_cols, model_dim, x, kv, pool_);
          } else {
            MatVec(layer_weights_.qkv_einsum_w2, 0,  //
                   w_rows_kv_cols, model_dim, x, kv, pool_);
          }
        }
      }
    }  // !is_mha_

    // Apply positional encodings for K (and copy KV to cache if MHA).
    pool_.Run(0, kv_heads * num_interleaved,
              [&](uint64_t task, size_t /*thread*/) HWY_ATTR {
                const size_t head = task % kv_heads;
                const size_t interleaved_idx = task / kv_heads;
                const size_t query_idx = interleaved_idx % num_queries_;
                const size_t batch_idx = interleaved_idx / num_queries_;
                const size_t pos = queries_pos_[query_idx] + batch_idx;
                const size_t cache_pos = div_seq_len_.Remainder(pos);
                const size_t kv_offset = cache_pos * cache_pos_size_ +
                                         layer_ * cache_layer_size_ +
                                         head * qkv_dim * 2;
                KVCache& kv_cache = kv_caches_[query_idx];
                float* HWY_RESTRICT kv = kv_cache.kv_cache.get() + kv_offset;
                // If MHA, copy computed K and V into KVCache.
                if (is_mha_) {
                  const float* HWY_RESTRICT mha_kv =
                      activations_.q.Batch(interleaved_idx) + head * q_stride_ +
                      qkv_dim;
                  hwy::CopyBytes(mha_kv, kv, 2 * qkv_dim * sizeof(*kv));
                }

                // Apply further processing to K.
                PositionalEncodingQK(kv, pos, layer_, /*mul=*/1.0f);
              });
  }

  // Computes Q.K scores, which are "logits" (or scores) stored to head_att.
  HWY_INLINE void QDotK(const size_t start_pos, const size_t last_pos,
                        const size_t head_offset, const float* HWY_RESTRICT q,
                        const KVCache& kv_cache, float* HWY_RESTRICT head_att) {
    const size_t qkv_dim = layer_config_.qkv_dim;
    if (HWY_LIKELY(last_pos < activations_.seq_len)) {
      // Slightly faster: no wraparound.
      for (size_t pos = start_pos; pos <= last_pos; ++pos) {
        const size_t kv_offset =
            pos * cache_pos_size_ + layer_ * cache_layer_size_ + head_offset;
        const float* HWY_RESTRICT k = &kv_cache.kv_cache[kv_offset];
        const float score = Dot(q, k, qkv_dim);
        head_att[pos] = score;
      }
    } else {
      for (size_t pos = start_pos; pos <= last_pos; ++pos) {
        const size_t cache_pos = div_seq_len_.Remainder(pos);
        const size_t kv_offset = cache_pos * cache_pos_size_ +
                                 layer_ * cache_layer_size_ + head_offset;
        const float* HWY_RESTRICT k = &kv_cache.kv_cache[kv_offset];
        const float score = Dot(q, k, qkv_dim);
        head_att[pos % activations_.seq_len] = score;
      }
    }
  }

  // Accumulates the sum of v (from `kv_cache`) * probability (`head_att`) into
  // `att_out`. Equivalent in gemma/modules.py:
  // encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)
  HWY_INLINE void WeightedSumV(const size_t start_pos, const size_t last_pos,
                               const float* HWY_RESTRICT head_att,
                               const size_t layer, const size_t head_offset,
                               const hwy::Divisor& div_seq_len,
                               const KVCache& kv_cache,
                               float* HWY_RESTRICT att_out) const {
    const size_t qkv_dim = layer_config_.qkv_dim;
    hwy::ZeroBytes(att_out, qkv_dim * sizeof(*att_out));

    if (HWY_LIKELY(last_pos < activations_.seq_len)) {
      // Slightly faster: no wraparound.
      for (size_t pos = start_pos; pos <= last_pos; ++pos) {
        const size_t kv_offset =
            pos * cache_pos_size_ + layer * cache_layer_size_ + head_offset;
        const float* HWY_RESTRICT v =
            kv_cache.kv_cache.get() + kv_offset + qkv_dim;
        MulByConstAndAdd(head_att[pos], v, att_out, qkv_dim);
      }
    } else {
      for (size_t pos = start_pos; pos <= last_pos; ++pos) {
        const size_t cache_pos = div_seq_len.Remainder(pos);
        const size_t kv_offset = cache_pos * cache_pos_size_ +
                                 layer * cache_layer_size_ + head_offset;
        const float* HWY_RESTRICT v =
            kv_cache.kv_cache.get() + kv_offset + qkv_dim;
        MulByConstAndAdd(head_att[pos % activations_.seq_len], v, att_out,
                         qkv_dim);
      }
    }
  }

  HWY_NOINLINE void DotSoftmaxWeightedSum(const size_t num_interleaved) {
    PROFILER_ZONE("Gen.Attention.DotSoftmax");
    const float query_scale = ChooseQueryScale(activations_.weights_config);

    // A "head group" in the context of GQA refers to a collection of query
    // heads that share the same key and value heads.
    const size_t kHeadGroups = layer_config_.heads / layer_config_.kv_heads;

    // For each head (token, query), compute Q.K, softmax, and weighted V.
    pool_.Run(0, layer_config_.heads * num_interleaved,
              [&](uint64_t task, size_t /*thread*/) HWY_ATTR {
                const size_t head = task % layer_config_.heads;
                const size_t interleaved_idx = task / layer_config_.heads;
                const size_t query_idx = interleaved_idx % num_queries_;
                const size_t batch_idx = interleaved_idx / num_queries_;
                const size_t qkv_dim = layer_config_.qkv_dim;
                const size_t head_offset = (head / kHeadGroups) * qkv_dim * 2;
                KVCache& kv_cache = kv_caches_[query_idx];
                float* HWY_RESTRICT q =
                    activations_.q.Batch(interleaved_idx) + head * q_stride_;

                // Apply rope and scaling to Q.
                const size_t pos = queries_pos_[query_idx] + batch_idx;
                PositionalEncodingQK(q, pos, layer_, query_scale);

                const size_t start_pos = StartPos(pos, layer_);
                size_t last_pos = pos;
                const size_t prefix_end = queries_prefix_end_[query_idx];
                if (prefix_end > 0 && prefix_end - 1 > last_pos) {
                  // last_pos in QDotK and WeightedSumV is inclusive.
                  last_pos = prefix_end - 1;
                }

                float* HWY_RESTRICT head_att =
                    activations_.att.Batch(interleaved_idx) +
                    head * activations_.seq_len;
                QDotK(start_pos, last_pos, head_offset, q, kv_cache, head_att);
                // SoftMax with optional SoftCap yields "probabilities" in
                // head_att.
                const size_t head_att_len =
                    std::min(last_pos + 1, activations_.seq_len);
                MaybeLogitsSoftCap(activations_.weights_config.att_cap,
                                   head_att, head_att_len);
                Softmax(head_att, head_att_len);

                float* HWY_RESTRICT att_out =
                    activations_.att_out.Batch(interleaved_idx) +
                    head * qkv_dim;
                WeightedSumV(start_pos, last_pos, head_att, layer_, head_offset,
                             div_seq_len_, kv_cache, att_out);
              });
  }

  // Sums encoded (`att_out`) over num_heads (`layer_config_.heads`) and
  // head_dim (`qkv_dim`) into output (`layer_out`).
  HWY_NOINLINE void SumHeads(const size_t num_interleaved) {
    PROFILER_ZONE("Gen.Attention.SumHeads");
    // att_weights and att_out are concatenated heads, each of length
    // layer_config_.qkv_dim. Thus the [num_interleaved,
    // layer_config_.model_dim] matmul output is the sum over heads. Compare
    // gemma/modules.py: attn_output = self.attn_vec_einsum('BTNH,NHD->BTD',
    // encoded)
    HWY_DASSERT(layer_config_.model_dim > 0);
    HWY_DASSERT(layer_config_.heads > 0);
    HWY_DASSERT(layer_config_.qkv_dim > 0);
    HWY_DASSERT(layer_weights_.att_weights.data() != nullptr);
    HWY_DASSERT(activations_.att_out.All() != nullptr);
    HWY_DASSERT(activations_.att_sums.All() != nullptr);

    const float* add =
        layer_weights_.layer_config.softmax_attn_output_biases
            ? layer_weights_.attention_output_biases.data_scale1()
            : nullptr;
    MatMul(ConstMatFromBatch(num_interleaved, activations_.att_out),
           ConstMatFromWeights(layer_weights_.att_weights), add,
           *activations_.env, RowPtrFromBatch(activations_.att_sums));
  }

 public:
  // Constructor with explicit initialization of queries_prefix_end. This is
  // needed for the Prefix-LM style attention. For standard causal attention,
  // the other constructor can be used.
  GemmaAttention(const QueriesPos& queries_pos,
                 const QueriesPos& queries_prefix_end, size_t num_tokens,
                 size_t layer, Activations& activations,
                 const LayerWeightsPtrs<T>* layer_weights,
                 const hwy::Divisor& div_seq_len, const KVCaches& kv_caches)
      : GemmaAttention(queries_pos, &queries_prefix_end, num_tokens, layer,
                       activations, layer_weights, div_seq_len, kv_caches) {}
  // Constructor with default initialization to 0 for queries_prefix_end.
  GemmaAttention(const QueriesPos& queries_pos, size_t num_tokens, size_t layer,
                 Activations& activations,
                 const LayerWeightsPtrs<T>* layer_weights,
                 const hwy::Divisor& div_seq_len, const KVCaches& kv_caches)
      : GemmaAttention(queries_pos, nullptr, num_tokens, layer, activations,
                       layer_weights, div_seq_len, kv_caches) {}

  // Full attention computation in three steps.
  HWY_INLINE void operator()() {
    const size_t num_interleaved = num_tokens_ * num_queries_;
    ComputeQKV(num_interleaved);
    DotSoftmaxWeightedSum(num_interleaved);
    SumHeads(num_interleaved);
  }

 private:
  // Delegated Constructor that does most of the common work.
  GemmaAttention(const QueriesPos& queries_pos,
                 const QueriesPos* queries_prefix_end, size_t num_tokens,
                 size_t layer, Activations& activations,
                 const LayerWeightsPtrs<T>* layer_weights,
                 const hwy::Divisor& div_seq_len, const KVCaches& kv_caches)
      : queries_pos_(queries_pos),
        num_queries_(queries_pos.size()),
        num_tokens_(num_tokens),
        layer_(layer),
        layer_config_(layer_weights->layer_config),
        q_stride_(layer_config_.QStride()),
        cache_layer_size_(layer_weights->layer_config.CacheLayerSize()),
        cache_pos_size_(activations.cache_pos_size),
        is_mha_(layer_config_.IsMHA()),
        activations_(activations),
        layer_weights_(*layer_weights),
        div_seq_len_(div_seq_len),
        kv_caches_(kv_caches),
        pool_(activations.env->parallel.Pools().Pool(0)) {
    HWY_DASSERT(num_queries_ <= kv_caches_.size());
    HWY_DASSERT_M((layer_config_.heads % layer_config_.kv_heads) == 0,
                  "query heads must be a multiple of key-value heads");
    if (queries_prefix_end != nullptr) {
      queries_prefix_end_ = *queries_prefix_end;
    } else {
      queries_prefix_end_vec_.assign(num_queries_, 0);
      queries_prefix_end_ = QueriesPos(queries_prefix_end_vec_.data(),
                                       queries_prefix_end_vec_.size());
    }
  }

  const QueriesPos& queries_pos_;
  std::vector<size_t> queries_prefix_end_vec_;
  QueriesPos queries_prefix_end_;
  const size_t num_queries_;
  const size_t num_tokens_;
  const size_t layer_;
  const LayerConfig& layer_config_;
  const size_t q_stride_ = 0;
  const size_t cache_layer_size_ = 0;
  const size_t cache_pos_size_ = 0;
  const bool is_mha_ = false;

  Activations& activations_;
  const LayerWeightsPtrs<T>& layer_weights_;
  const hwy::Divisor& div_seq_len_;
  const KVCaches& kv_caches_;
  hwy::ThreadPool& pool_;
};

template <typename T>
HWY_NOINLINE void Attention(
    LayerAttentionType type, const QueriesPos& queries_pos,
    const QueriesPos& queries_prefix_end, size_t num_tokens, size_t layer,
    Activations& activations, const LayerWeightsPtrs<T>* layer_weights,
    const hwy::Divisor& div_seq_len, const KVCaches& kv_caches) {
  if (type == LayerAttentionType::kGemma) {
    GemmaAttention<T>(queries_pos, queries_prefix_end, num_tokens, layer,
                      activations, layer_weights, div_seq_len, kv_caches)();
  } else {
    // Only reached if the model is Griffin.
    // The kv_caches are allocated only for the griffin layers, so we need to
    // map the layer index to the griffin layer index.
    auto type = layer_weights->layer_config.type;
    size_t layer_of_type =
        activations.weights_config.NumLayersOfTypeBefore(type, layer);
    HWY_ASSERT(queries_pos.size() == 1);
    GriffinRecurrent(queries_pos[0], num_tokens, layer_of_type, activations,
                     layer_weights, kv_caches);
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
template <typename T>
class VitAttention {
  // Computes Q, K, V for all heads, stored in activations_.q.
  HWY_NOINLINE void ComputeQKV() {
    PROFILER_ZONE("Gen.VitAttention.QKV");
    auto& qkv = activations_.q;
    HWY_ASSERT(qkv.BatchSize() == num_tokens_);
    HWY_ASSERT(qkv.Cols() == layer_config_.heads * 3 * layer_config_.qkv_dim);
    MatMul(ConstMatFromBatch(num_tokens_, activations_.pre_att_rms_out),
           ConstMatFromWeights(layer_weights_.vit.qkv_einsum_w),
           layer_weights_.vit.qkv_einsum_b.data_scale1(), *activations_.env,
           RowPtrFromBatch(qkv));
  }

  HWY_NOINLINE void DotSoftmaxWeightedSum() {
    const size_t qkv_dim = layer_config_.qkv_dim;
    const size_t heads = layer_config_.heads;
    HWY_ASSERT_M(heads == layer_config_.kv_heads, "Vit expects MHA");
    const size_t seq_len = activations_.seq_len;
    const float query_scale = 1.0f / sqrtf(static_cast<float>(qkv_dim));
    PROFILER_ZONE("Gen.VitAttention.DotSoftmax");

    // Compute Q.K, softmax, and weighted V.
    pool_.Run(0, layer_config_.heads * num_tokens_,
              [&](uint64_t task, size_t /*thread*/) HWY_ATTR {
                const size_t head = task % layer_config_.heads;
                const size_t token = task / layer_config_.heads;
                // Compute Q.K scores, which are "logits" stored in head_att.
                float* HWY_RESTRICT q =
                    activations_.q.Batch(token) + head * 3 * qkv_dim;
                MulByConst(query_scale, q, qkv_dim);
                float* HWY_RESTRICT head_att =
                    activations_.att.Batch(token) + head * activations_.seq_len;
                for (size_t i = 0; i < seq_len; ++i) {
                  float* HWY_RESTRICT k =
                      activations_.q.Batch(i) + head * 3 * qkv_dim + qkv_dim;
                  head_att[i] = Dot(q, k, qkv_dim);  // score = q.k
                }
                // SoftMax yields "probabilities" in head_att.
                Softmax(head_att, seq_len);
                // Compute weighted sum of v into att_out.
                float* HWY_RESTRICT att_out =
                    activations_.att_out.Batch(token) + head * qkv_dim;
                hwy::ZeroBytes(att_out, qkv_dim * sizeof(*att_out));
                for (size_t i = 0; i < seq_len; ++i) {
                  float* HWY_RESTRICT v = activations_.q.Batch(i) +
                                          head * 3 * qkv_dim + 2 * qkv_dim;
                  MulByConstAndAdd(head_att[i], v, att_out, qkv_dim);
                }
              });
  }

  // Sums encoded (`att_out`) over num_heads (`layer_config_.heads`) and
  // head_dim (`qkv_dim`) into output (`att_sums`).
  HWY_NOINLINE void SumHeads() {
    PROFILER_ZONE("Gen.VitAttention.SumHeads");
    auto* bias = layer_weights_.vit.attn_out_b.data_scale1();
    // att_weights and att_out are concatenated heads, each of length
    // qkv_dim. Thus the [num_tokens_, layer_config_.model_dim]
    // matmul output is the sum over heads.
    auto att_out = ConstMatFromBatch(num_tokens_, activations_.att_out);
    auto att_weights = ConstMatFromWeights(layer_weights_.vit.attn_out_w);
    auto att_sums = RowPtrFromBatch(activations_.att_sums);
    MatMul(att_out, att_weights, bias, *activations_.env, att_sums);
  }

 public:
  VitAttention(size_t num_tokens, size_t layer, Activations& activations,
               const LayerWeightsPtrs<T>* layer_weights)
      : num_tokens_(num_tokens),
        layer_(layer),
        activations_(activations),
        layer_weights_(*layer_weights),
        layer_config_(layer_weights->layer_config),
        pool_(activations.env->parallel.Pools().Pool(0)) {}

  HWY_INLINE void operator()() {
    ComputeQKV();
    DotSoftmaxWeightedSum();
    SumHeads();
  }

 private:
  const size_t num_tokens_;
  const size_t layer_;
  Activations& activations_;
  const LayerWeightsPtrs<T>& layer_weights_;
  const LayerConfig& layer_config_;
  hwy::ThreadPool& pool_;
};

template <typename T>
HWY_NOINLINE void Activation(ActivationType activation, T* HWY_RESTRICT c1,
                             T* HWY_RESTRICT c2, size_t count) {
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

template <typename T>
HWY_NOINLINE void FFWNoVit(Activations& activations, size_t num_interleaved,
                           const LayerWeightsPtrs<T>* layer_weights) {
  PROFILER_ZONE("Gen.FFW");
  const size_t model_dim = layer_weights->layer_config.model_dim;
  const size_t ffh_hidden_dim = layer_weights->layer_config.ff_hidden_dim;
  HWY_DASSERT(num_interleaved <= activations.bf_pre_ffw_rms_out.BatchSize());

  const bool add_bias = layer_weights->layer_config.ff_biases;
  const float* bias1 =
      add_bias ? layer_weights->ffw_gating_biases.data_scale1() : nullptr;
  const float* bias2 = add_bias ? bias1 + ffh_hidden_dim : nullptr;
  const float* output_bias =
      add_bias ? layer_weights->ffw_output_biases.data_scale1() : nullptr;

  // Define slightly more readable names for the weights and activations.
  const auto x =
      ConstMatFromBatch(num_interleaved, activations.bf_pre_ffw_rms_out);

  auto hidden_activations = RowPtrFromBatch(activations.C1);
  auto multiplier = RowPtrFromBatch(activations.C2);
  auto ffw_out = RowPtrFromBatch(activations.ffw_out);

  // gating_einsum_w holds two half-matrices. We plan to change the importer to
  // avoid this confusion by splitting into gating_einsum_w1 and
  // gating_einsum_w2.
  const bool split = !!layer_weights->gating_einsum_w.data();
  auto w1 = split ? ConstMatFromWeights(layer_weights->gating_einsum_w)
                  : ConstMatFromWeights(layer_weights->gating_einsum_w1);
  auto w2 = split ? ConstMatFromWeights(layer_weights->gating_einsum_w,
                                        model_dim * ffh_hidden_dim)
                  : ConstMatFromWeights(layer_weights->gating_einsum_w2);
  if (split) {
    // Ensure that B.Extents().row matches C.Cols() because MatMul checks that.
    w1.ShrinkRows(ffh_hidden_dim);
    w2.ShrinkRows(ffh_hidden_dim);
  }
  auto w_output = ConstMatFromWeights(layer_weights->linear_w);

  // Compute the hidden layer activations.
  MatMul(x, w1, bias1, *activations.env, hidden_activations);
  MatMul(x, w2, bias2, *activations.env, multiplier);

  // Activation (Gelu) and maybe multiply by gate. Store activations in act.
  Activation(layer_weights->layer_config.activation, hidden_activations.Row(0),
             multiplier.Row(0), ffh_hidden_dim * num_interleaved);

  // Hidden layer -> output layer.
  auto activations_mat = MakeConstMat(
      hidden_activations.Row(0), Extents2D(num_interleaved, ffh_hidden_dim));

  MatMul(activations_mat, w_output, output_bias, *activations.env, ffw_out);
}

// Same as FFWNoVit, but with different layer_weights members and no second
// gating matrix.
template <typename T>
HWY_NOINLINE void FFWVit(Activations& activations, size_t num_interleaved,
                         const LayerWeightsPtrs<T>* layer_weights) {
  PROFILER_ZONE("Gen.FFW");
  const size_t ff_hidden_dim = layer_weights->layer_config.ff_hidden_dim;
  HWY_DASSERT(num_interleaved <= activations.bf_pre_ffw_rms_out.BatchSize());

  const bool add_bias = layer_weights->layer_config.ff_biases;
  const float* bias1 =
      add_bias ? layer_weights->vit.linear_0_b.data_scale1() : nullptr;
  const float* output_bias =
      add_bias ? layer_weights->vit.linear_1_b.data_scale1() : nullptr;

  // Define slightly more readable names for the weights and activations.
  const auto x =
      ConstMatFromBatch(num_interleaved, activations.bf_pre_ffw_rms_out);

  auto hidden_activations = RowPtrFromBatch(activations.C1);
  auto ffw_out = RowPtrFromBatch(activations.ffw_out);

  auto w1 = ConstMatFromWeights(layer_weights->vit.linear_0_w);
  auto w_output = ConstMatFromWeights(layer_weights->vit.linear_1_w);

  // Compute the hidden layer activations.
  MatMul(x, w1, bias1, *activations.env, hidden_activations);

  // Activation (Gelu), store in act.
  RowPtrF multiplier = RowPtrF(nullptr, 0);
  Activation(layer_weights->layer_config.activation, hidden_activations.Row(0),
             multiplier.Row(0), ff_hidden_dim * num_interleaved);

  // Hidden layer -> output layer.
  auto activations_mat = MakeConstMat(
      hidden_activations.Row(0), Extents2D(num_interleaved, ff_hidden_dim));

  MatMul(activations_mat, w_output, output_bias, *activations.env, ffw_out);
}

// `batch_idx` indicates which row of `x` to write to.
// `pos` is the *token*'s position, not the start of the batch, because this is
// called for batches of tokens in prefill, but batches of queries in decode.
template <typename T>
HWY_NOINLINE void EmbedToken(int token, size_t batch_idx, size_t pos,
                             size_t pos_in_prompt,
                             const ModelWeightsPtrs<T>& weights,
                             RowVectorBatch<float>& x,
                             const ImageTokens* image_tokens) {
  // Image tokens just need to be copied.
  if (image_tokens != nullptr && pos_in_prompt < image_tokens->BatchSize()) {
    hwy::CopyBytes(image_tokens->Batch(pos_in_prompt), x.Batch(batch_idx),
                   x.Cols() * sizeof(x.Const()[0]));
    return;
  }

  const size_t model_dim = weights.weights_config.model_dim;
  const size_t vocab_size = weights.weights_config.vocab_size;
  const float emb_scaling = EmbeddingScaling(model_dim);

  HWY_DASSERT(token >= 0);
  HWY_DASSERT(token < static_cast<int>(vocab_size));

  const hn::ScalableTag<float> df;
  DecompressAndZeroPad(
      df,
      MakeSpan(weights.embedder_input_embedding.data(), vocab_size * model_dim),
      token * model_dim, x.Batch(batch_idx), model_dim);
  MulByConst(emb_scaling * weights.embedder_input_embedding.scale(),
             x.Batch(batch_idx), model_dim);
  if (weights.weights_config.absolute_pe) {
    AddAbsolutePositionalEmbeddings(x.Batch(batch_idx), model_dim, pos);
  }
}

template <typename Weights, typename T>
HWY_NOINLINE void ResidualConnection(
    size_t num_interleaved, T* HWY_RESTRICT other, T* HWY_RESTRICT x,
    const LayerWeightsPtrs<Weights>* layer_weights, bool is_attention) {
  // ResidualType::Add
  AddFromBatched(num_interleaved, other, x,
                 layer_weights->layer_config.model_dim);
}

template <typename WeightT, typename InOutT>
void PostNorm(PostNormType post_norm, size_t num_interleaved,
              const WeightT& weights, InOutT* inout) {
  if (post_norm == PostNormType::Scale) {
    RMSNormInplaceBatched(num_interleaved, weights.data_scale1(), inout,
                          weights.NumElements());
  }
}

template <typename T>
HWY_NOINLINE void TransformerLayer(const QueriesPos& queries_pos,
                                   const QueriesPos& queries_prefix_end,
                                   size_t num_tokens, size_t cache_layer_idx,
                                   const LayerWeightsPtrs<T>* layer_weights,
                                   Activations& activations,
                                   const hwy::Divisor& div_seq_len,
                                   const KVCaches& kv_caches) {
  const size_t model_dim = activations.weights_config.model_dim;
  const size_t num_interleaved = num_tokens * queries_pos.size();
  auto type = layer_weights->layer_config.type;

  RMSNormBatched(num_interleaved, activations.x.All(),
                 layer_weights->pre_attention_norm_scale.data_scale1(),
                 activations.pre_att_rms_out.All(), model_dim);

  Attention(type, queries_pos, queries_prefix_end, num_tokens, cache_layer_idx,
            activations, layer_weights, div_seq_len, kv_caches);

  PostNorm(layer_weights->layer_config.post_norm, num_interleaved,
           layer_weights->post_attention_norm_scale,
           activations.att_sums.All());

  ResidualConnection(num_interleaved, activations.att_sums.All(),
                     activations.x.All(), layer_weights, /*is_attention=*/true);

  RMSNormBatched(num_interleaved, activations.x.All(),
                 layer_weights->pre_ffw_norm_scale.data_scale1(),
                 activations.bf_pre_ffw_rms_out.All(), model_dim);

  if (layer_weights->layer_config.type == LayerAttentionType::kVit) {
    FFWVit(activations, num_interleaved, layer_weights);
  } else {
    FFWNoVit(activations, num_interleaved, layer_weights);
  }

  PostNorm(layer_weights->layer_config.post_norm, num_interleaved,
           layer_weights->post_ffw_norm_scale, activations.ffw_out.All());

  ResidualConnection(num_interleaved, activations.ffw_out.All(),
                     activations.x.All(), layer_weights,
                     /*is_attention=*/false);
}

// Vit transformer layer. Some comments below refer to the Vit implementation in
// the Big Vision codebase. See
// github.com/google-research/big_vision/blob/main/big_vision/models/vit.py
// TODO(keysers): consider adding a wrapper for both LayerNorm with RMSNorm and
// try merging this with TransformerLayer.
template <typename T>
HWY_NOINLINE void VitTransformerLayer(size_t num_tokens, size_t layer,
                                      const LayerWeightsPtrs<T>* layer_weights,
                                      Activations& activations) {
  const size_t model_dim = activations.weights_config.model_dim;
  auto type = layer_weights->layer_config.type;
  HWY_DASSERT(type == LayerAttentionType::kVit);
  (void)type;

  auto& x = activations.x;
  HWY_DASSERT(x.BatchSize() == num_tokens);
  HWY_DASSERT(x.Cols() == model_dim);

  // y = nn.LayerNorm()(x)
  // y ~ pre_att_rms_out
  LayerNormBatched(num_tokens, x.All(),
                   layer_weights->vit.layer_norm_0_scale.data_scale1(),
                   layer_weights->vit.layer_norm_0_bias.data_scale1(),
                   activations.pre_att_rms_out.All(), model_dim);

  // y = out["sa"] = nn.MultiHeadDotProductAttention(...)(y, y)
  // y ~ att_sums
  VitAttention<T>(num_tokens, layer, activations, layer_weights)();

  // x = out["+sa"] = x + y
  AddFromBatched(num_tokens, activations.att_sums.All(), x.All(), model_dim);

  // y = nn.LayerNorm()(x)
  // y ~ bf_pre_ffw_rms_out
  LayerNormBatched(num_tokens, x.All(),
                   layer_weights->vit.layer_norm_1_scale.data_scale1(),
                   layer_weights->vit.layer_norm_1_bias.data_scale1(),
                   activations.bf_pre_ffw_rms_out.All(), model_dim);

  // y = out["mlp"] = MlpBlock(...)(y)
  // y ~ ffw_out
  FFWVit(activations, num_tokens, layer_weights);

  // x = out["+mlp"] = x + y
  AddFromBatched(num_tokens, activations.ffw_out.All(), x.All(), model_dim);
}

// Prefill() and Transformer() increment positions in-place.
using QueriesMutablePos = hwy::Span<size_t>;

// Populates KV cache for batches of tokens from one query at a time.
template <typename T>
HWY_NOINLINE void Prefill(
    const QueriesPromptTokens& queries_prompt,
    const QueriesMutablePos& queries_pos, const QueriesPos& queries_prefix_end,
    const size_t query_idx_start, const ModelWeightsPtrs<T>& weights,
    Activations& activations, const RuntimeConfig& runtime_config,
    const hwy::Divisor& div_seq_len, const KVCaches& kv_caches) {
  PROFILER_ZONE("Gen.Prefill");
  const size_t num_queries = queries_prompt.size();
  HWY_DASSERT(queries_pos.size() == num_queries);
  HWY_DASSERT(queries_prefix_end.size() == num_queries);
  HWY_DASSERT(kv_caches.size() == num_queries);

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
        EmbedToken(token, ti, pos, pos_in_prompt, weights, activations.x,
                   runtime_config.image_tokens);
      }

      // Transformer with one batch of tokens from a single query.
      for (size_t layer = 0;
           layer < weights.weights_config.layer_configs.size(); ++layer) {
        const auto* layer_weights = weights.GetLayer(layer);
        TransformerLayer(single_query_pos, single_query_prefix_end, tbatch_size,
                         layer, layer_weights, activations, div_seq_len,
                         single_kv_cache);
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
template <typename T>
HWY_NOINLINE void EmbedImagePatches(const Image& image,
                                    const ModelWeightsPtrs<T>& weights,
                                    Activations& activations) {
  const size_t model_dim = weights.weights_config.vit_config.model_dim;
  const size_t patch_width = weights.weights_config.vit_config.patch_width;
  const size_t seq_len = weights.weights_config.vit_config.seq_len;
  const size_t patch_size = patch_width * patch_width * 3;
  HWY_DASSERT(weights.vit_img_embedding_kernel.NumElements() ==
              patch_size * model_dim);
  HWY_DASSERT(activations.x.Cols() == model_dim);
  std::vector<hwy::AlignedFreeUniquePtr<float[]>> image_patches(seq_len);
  for (size_t i = 0; i < seq_len; ++i) {
    image_patches[i] = hwy::AllocateAligned<float>(patch_size);
    image.GetPatch(i, image_patches[i].get());
  }
  // img/embedding/kernel has original shape (14, 14, 3, 1152)
  // H x W x C x D transposed to D x (H x W x C) so here (1152, 14 * 14 * 3)
  // image_patches is (256, 14 * 14 * 3)
  // This could be done as one MatMul like:
  // RowVectorBatch<float> image_patches(kSeqLen, kPatchSize);
  // [Get patches]
  // MatMul(
  //       MatFromBatch(kVitSeqLen, image_patches),
  //       MatFromWeights(weights.vit_img_embedding_kernel),
  //       weights.vit_img_embedding_bias.data_scale1(), *activations.env,
  //       RowPtrF(activations.x.All(), kVitModelDim));
  // However, MatMul currently requires that
  //   A.cols % (2 * hn::Lanes(hn::ScalableTag<MulT>())) == 0
  // which is not the case here. We should relax that requirement on MatMul and
  // then use the above. For now, we rely on MatVecAdd instead.
  for (size_t i = 0; i < seq_len; ++i) {
    MatVecAdd(
        weights.vit_img_embedding_kernel, 0, model_dim, patch_size,
        image_patches[i].get(), weights.vit_img_embedding_bias.data_scale1(),
        activations.x.Batch(i), activations.env->parallel.Pools().Pool(0));
  }
  // Add position embeddings.
  AddFrom(weights.vit_img_pos_embedding.data_scale1(), activations.x.All(),
          seq_len * model_dim);
}

// Prefills the image tokens with the ViT encoder.
template <typename T>
HWY_NOINLINE void PrefillVit(const ModelWeightsPtrs<T>& weights,
                             const RuntimeConfig& runtime_config,
                             const Image& image, ImageTokens& image_tokens,
                             Activations& activations) {
  PROFILER_ZONE("Gen.PrefillVit");
  const size_t num_tokens = weights.weights_config.vit_config.seq_len;
  const size_t vit_model_dim = weights.weights_config.vit_config.model_dim;
  HWY_ASSERT(num_tokens == activations.x.BatchSize());
  // Embed the image patches.
  EmbedImagePatches(image, weights, activations);
  // Go through all layers.
  for (size_t layer = 0;
       layer < weights.weights_config.vit_config.layer_configs.size();
       ++layer) {
    const auto* layer_weights = weights.GetVitLayer(layer);
    VitTransformerLayer(num_tokens, layer, layer_weights, activations);
  }
  // Final Layernorm.
  LayerNormBatched(num_tokens, activations.x.All(),
                   weights.vit_encoder_norm_scale.data_scale1(),
                   weights.vit_encoder_norm_bias.data_scale1(),
                   activations.x.All(), vit_model_dim);

  // Apply head embedding into image_tokens of size of the LLM kModelDim.
  MatMul(ConstMatFromBatch(num_tokens, activations.x),
         ConstMatFromWeights(weights.vit_img_head_kernel),
         weights.vit_img_head_bias.data_scale1(), *activations.env,
         RowPtrFromBatch(image_tokens));
}

// Generates one token for each query. `queries_token` is the previous token
// from each query, and `queries_pos` are their position in the sequence.
template <typename T>
HWY_NOINLINE void Transformer(
    const QueriesToken& queries_token, const QueriesMutablePos& queries_pos,
    const QueriesPos& queries_prefix_end, const ModelWeightsPtrs<T>& weights,
    Activations& activations, const hwy::Divisor& div_seq_len,
    const KVCaches& kv_caches, const LayersOutputFunc& layers_output,
    const ActivationsObserverFunc& activations_observer) {
  const size_t model_dim = weights.weights_config.model_dim;
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
    EmbedToken(queries_token[query_idx], query_idx, queries_pos[query_idx],
               /*pos_in_prompt=*/0, weights, activations.x,
               /*image_tokens=*/nullptr);
  }

  for (size_t layer = 0; layer < weights.c_layers.size(); ++layer) {
    const LayerWeightsPtrs<T>* layer_weights = weights.GetLayer(layer);
    TransformerLayer(queries_pos, queries_prefix_end, /*num_tokens=*/1, layer,
                     layer_weights, activations, div_seq_len, kv_caches);

    if (activations_observer) {
      activations_observer(queries_pos, layer, activations);
    }
  }

  RMSNormInplaceBatched(num_queries, weights.final_norm_scale.data_scale1(),
                        activations.x.All(), model_dim);

  if (activations_observer) {
    activations_observer(queries_pos, -1, activations);
  }
  for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
    queries_pos[query_idx] += 1;
  }
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

HWY_INLINE SampleFunc ChooseSampleFunc(const RuntimeConfig& runtime_config) {
  // If user provided a sample_func, use it.
  if (runtime_config.sample_func) return runtime_config.sample_func;

  // Fast path for top-1 with no accept_token.
  if (runtime_config.top_k == 1 && !runtime_config.accept_token) {
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
    const int token = SampleTopK(
        logits, runtime_config.top_k, vocab_size, *runtime_config.gen,
        runtime_config.temperature, runtime_config.accept_token);
    return TokenAndProb{.token = token, .prob = logits[token]};
  };
}

template <typename T>
// Runs one decode step for all the queries in the batch. Returns true if all
// queries are at <end_of_sentence>.
bool DecodeStepT(const ModelWeightsPtrs<T>& weights,
                 const RuntimeConfig& runtime_config,
                 const QueriesPromptTokens& queries_prompt,
                 const size_t query_idx_start, const KVCaches& kv_caches,
                 const QueriesPos& queries_prefix_end,
                 const hwy::Divisor div_seq_len, const size_t vocab_size,
                 const SampleFunc& sample_token, double prefill_start,
                 double gen_start, Activations& activations,
                 TokenStreamer& token_streamer, std::vector<int>& gen_tokens,
                 TimingInfo& timing_info,
                 const QueriesMutablePos& queries_mutable_pos) {
  const size_t num_queries = queries_prompt.size();
  // Decode generates one token per query and increments
  // queries_mutable_pos.
  Transformer(QueriesToken(gen_tokens.data(), num_queries), queries_mutable_pos,
              queries_prefix_end, weights, activations, div_seq_len, kv_caches,
              runtime_config.layers_output,
              runtime_config.activations_observer);
  // queries_pos are incremented by Transformer.

  bool all_queries_eos = true;
  {
    PROFILER_ZONE("Gen.EmbeddingMatmul");
    // Compute logits from last layer activations.
    MatMul(ConstMatFromBatch(num_queries, activations.x),
           ConstMatFromWeights(weights.embedder_input_embedding),
           /*add=*/nullptr, *activations.env,
           RowPtrFromBatch(activations.logits));
  }
  PROFILER_ZONE("Gen.Softcap+Sample+Stream");
  for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
    float* HWY_RESTRICT logits = activations.logits.Batch(query_idx);
    MaybeLogitsSoftCap(weights.weights_config.final_cap, logits, vocab_size);
    const TokenAndProb tp = sample_token(logits, vocab_size);
    timing_info.NotifyGenerated(prefill_start, gen_start);

    const bool is_eos =
        token_streamer(query_idx_start + query_idx,
                       queries_mutable_pos[query_idx], tp.token, tp.prob);
    all_queries_eos &= is_eos;
    gen_tokens[query_idx] = is_eos ? runtime_config.eos_id : tp.token;
  }
  return all_queries_eos;
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
template <typename T>
void GenerateT(const ModelWeightsStorage& model, Activations& activations,
               const RuntimeConfig& runtime_config,
               const QueriesPromptTokens& queries_prompt,
               const QueriesPos& queries_pos_in,
               const QueriesPos& queries_prefix_end,
               const size_t query_idx_start, const KVCaches& kv_caches,
               TimingInfo& timing_info) {
  // Griffin assumes that the recurrent block cache is zero-initialized.
  for (size_t i = 0; i < kv_caches.size(); ++i) {
    if (queries_pos_in[i] == 0) {
      kv_caches[i].ZeroGriffinCache();  // No-op for non-Griffin models.
    }
  }

  // Copy so we can increment without requiring users to pass in a mutable span.
  std::vector<size_t> queries_pos_copy(queries_pos_in.cbegin(),
                                       queries_pos_in.cend());
  const QueriesMutablePos queries_mutable_pos(queries_pos_copy.data(),
                                              queries_pos_copy.size());

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
  const ModelWeightsPtrs<T>& weights = *model.GetWeightsOfType<T>();
  size_t max_prompt_size = MaxQueryLength(queries_prompt);
  size_t max_generated_tokens = runtime_config.max_generated_tokens;
  RangeChecks(weights.weights_config, max_generated_tokens, max_prompt_size);
  const SampleFunc sample_token = ChooseSampleFunc(runtime_config);

  // Prefill stops before min_prompt_size - 1 because the last prompt
  // token is the first input token for generation.
  const double prefill_start = hwy::platform::Now();
  // If tbatch is larger than the qbatch we already have in `activations`, then
  // allocate prefill_activations, otherwise reuse.
  const bool use_prefill_activations =
      runtime_config.prefill_tbatch_size > activations.x.BatchSize();
  Activations prefill_activations(weights.weights_config);
  if (use_prefill_activations) {
    prefill_activations.Allocate(runtime_config.prefill_tbatch_size,
                                 activations.env);
  }
  Prefill(queries_prompt, queries_mutable_pos, queries_prefix_end,
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

  {
    const size_t vocab_size = model.Config().vocab_size;
    const double gen_start = hwy::platform::Now();
    for (size_t gen = 0; gen < max_generated_tokens; ++gen) {
      bool all_queries_eos = DecodeStepT<T>(
          weights, runtime_config, queries_prompt, query_idx_start, kv_caches,
          queries_prefix_end, div_seq_len, vocab_size, sample_token,
          prefill_start, gen_start, activations, token_streamer, gen_tokens,
          timing_info, queries_mutable_pos);
      if (all_queries_eos) break;
    }  // foreach token to generate
    timing_info.NotifyGenerateDone(gen_start);
  }
}

template <typename T>
void GenerateSingleT(const ModelWeightsStorage& model,
                     const RuntimeConfig& runtime_config,
                     const PromptTokens& prompt, size_t pos, size_t prefix_end,
                     KVCache& kv_cache, MatMulEnv* env,
                     TimingInfo& timing_info) {
  constexpr size_t kNumQueries = 1;
  const size_t qbatch_start = 0;

  // TODO: move into Gemma?
  Activations activations(model.Config());
  activations.Allocate(kNumQueries, env);

  const QueriesPromptTokens queries_prompt(&prompt, kNumQueries);
  QueriesPos queries_pos(&pos, kNumQueries);
  const QueriesPos queries_prefix_end(&prefix_end, kNumQueries);
  const KVCaches kv_caches{&kv_cache, kNumQueries};

  GenerateT<T>(model, activations, runtime_config, queries_prompt, queries_pos,
               queries_prefix_end, qbatch_start, kv_caches, timing_info);
}

template <typename T>
void GenerateBatchT(const ModelWeightsStorage& model,
                    const RuntimeConfig& runtime_config,
                    const QueriesPromptTokens& queries_prompt,
                    const QueriesPos& queries_pos,
                    const QueriesPos& queries_prefix_end,
                    const KVCaches& kv_caches, MatMulEnv* env,
                    TimingInfo& timing_info) {
  const size_t num_queries = queries_prompt.size();
  HWY_ASSERT(queries_pos.size() == num_queries);
  HWY_ASSERT(kv_caches.size() == num_queries);
  // Griffin does not support query batching.
  size_t max_qbatch_size = runtime_config.decode_qbatch_size;
  for (const auto& layer_config : model.Config().layer_configs) {
    if (layer_config.type == LayerAttentionType::kGriffinRecurrentBlock) {
      max_qbatch_size = 1;
      break;
    }
  }

  Activations activations(model.Config());
  activations.Allocate(max_qbatch_size, env);

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
    GenerateT<T>(model, activations, runtime_config, qbatch_prompts, qbatch_pos,
                 qbatch_prefix_end, qbatch_start, qbatch_kv, timing_info);
  }
}

template <typename T>
void GenerateImageTokensT(const ModelWeightsStorage& model,
                          const RuntimeConfig& runtime_config,
                          const Image& image, ImageTokens& image_tokens,
                          MatMulEnv* env) {
  if (model.Config().vit_config.layer_configs.empty()) {
    HWY_ABORT("Model does not support generating image tokens.");
  }
  RuntimeConfig prefill_runtime_config = runtime_config;
  ModelConfig vit_config = GetVitConfig(model.Config());
  prefill_runtime_config.prefill_tbatch_size = vit_config.seq_len;
  Activations prefill_activations(vit_config);
  prefill_activations.Allocate(vit_config.seq_len, env);
  // Weights are for the full PaliGemma model, not just the ViT part.
  PrefillVit(*model.GetWeightsOfType<T>(), prefill_runtime_config, image,
             image_tokens, prefill_activations);
}

}  // namespace HWY_NAMESPACE

#if HWY_ONCE

// These are extern functions defined by instantiations/*.cc, which include this
// 'header' after defining GEMMA_CONFIG, which is for function overloading.
void GenerateSingle(  // NOLINT(misc-definitions-in-headers)
    GEMMA_TYPE, const ModelWeightsStorage& model,
    const RuntimeConfig& runtime_config, const PromptTokens& prompt, size_t pos,
    size_t prefix_end, KVCache& kv_cache, MatMulEnv* env,
    TimingInfo& timing_info) {
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(GenerateSingleT<GEMMA_TYPE>)
  (model, runtime_config, prompt, pos, prefix_end, kv_cache, env, timing_info);
}

void GenerateBatch(  // NOLINT(misc-definitions-in-headers)
    GEMMA_TYPE, const ModelWeightsStorage& model,
    const RuntimeConfig& runtime_config,
    const QueriesPromptTokens& queries_prompt, const QueriesPos& queries_pos,
    const QueriesPos& queries_prefix_end, const KVCaches& kv_caches,
    MatMulEnv* env, TimingInfo& timing_info) {
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(GenerateBatchT<GEMMA_TYPE>)
  (model, runtime_config, queries_prompt, queries_pos, queries_prefix_end,
   kv_caches, env, timing_info);
}

void GenerateImageTokens(  // NOLINT(misc-definitions-in-headers)
    GEMMA_TYPE, const ModelWeightsStorage& model,
    const RuntimeConfig& runtime_config, const Image& image,
    ImageTokens& image_tokens, MatMulEnv* env) {
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(GenerateImageTokensT<GEMMA_TYPE>)
  (model, runtime_config, image, image_tokens, env);
}

#endif  // HWY_ONCE

}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_INL_H_
