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
#include <stdint.h>
#include <stdio.h>

#include <algorithm>  // std::min
#include <vector>

#include "gemma/activations.h"
#include "gemma/common.h"  // EmbeddingScaling
#include "gemma/configs.h"
#include "gemma/gemma.h"
#include "gemma/kv_cache.h"
#include "gemma/weights.h"
#include "paligemma/image.h"
#include "util/mat.h"
#include "util/threading_context.h"
#include "hwy/aligned_allocator.h"  // Span
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
// `Attention`, use separate `num_tokens` and `num_queries`. `num_tokens` is the
// number of tokens from one query: 1 for decode, otherwise prefill_tbatch_size.

template <typename T>
HWY_NOINLINE void GriffinRecurrent(const QueriesPos& queries_pos,
                                   size_t num_tokens, size_t griffin_layer,
                                   Activations& activations,
                                   const LayerWeightsPtrs<T>* layer_weights,
                                   const KVCaches& kv_caches) {
  PROFILER_ZONE("Gen.Griffin");
  hwy::ThreadPool& pool = activations.env->ctx.pools.Pool(0);
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  const D df;

  const size_t model_dim = layer_weights->layer_config.model_dim;
  HWY_DASSERT(model_dim % hn::Lanes(df) == 0);

  const size_t heads = layer_weights->layer_config.heads;
  const size_t conv_1d_width = layer_weights->layer_config.conv1d_width;
  HWY_ASSERT_M(conv_1d_width % 2 == 0, "Conv width must be even");
  const size_t kHeadDim = model_dim / heads;
  const size_t kMatrixSize = kHeadDim * kHeadDim;

  const size_t num_queries = queries_pos.size();
  const hwy::Divisor div_num_q(static_cast<uint32_t>(num_queries));
  const size_t num_interleaved = num_tokens * num_queries;

  // X / Y linear layers.
  // TODO: MatMul
  HWY_DASSERT(activations.griffin_y.Rows() == activations.griffin_x.Rows());
  HWY_DASSERT(num_interleaved == activations.griffin_y.Rows());
  for (size_t r = 0; r < num_interleaved; ++r) {
    float* HWY_RESTRICT y = activations.griffin_y.Row(r);
    float* HWY_RESTRICT x = activations.griffin_x.Row(r);
    TwoMatVecAdd(layer_weights->griffin.linear_x_w,
                 layer_weights->griffin.linear_y_w, 0, model_dim, model_dim,
                 activations.pre_att_rms_out.Row(r),
                 /*add0=*/layer_weights->griffin.linear_x_biases.PackedScale1(),
                 /*add1=*/layer_weights->griffin.linear_y_biases.PackedScale1(),
                 /*out0=*/x, /*out1=*/y, pool);
    Gelu(y, model_dim);
  }

  // Conv1D.
  for (size_t interleaved_idx = 0; interleaved_idx < num_interleaved;
       ++interleaved_idx) {
    const size_t query_idx = div_num_q.Remainder(interleaved_idx);
    const size_t batch_idx = div_num_q.Divide(interleaved_idx);
    const size_t pos = queries_pos[query_idx] + batch_idx;
    float* HWY_RESTRICT x = activations.griffin_x.Row(query_idx);

    // cache[i] = input at time t-i.
    float* HWY_RESTRICT cache[kMaxConv1DWidth];
    cache[0] = x;
    for (size_t i = 1; i < conv_1d_width; i++) {
      cache[i] =
          kv_caches[query_idx].conv1d_cache.Row(griffin_layer) +
          ((pos + conv_1d_width - 1 - i) % (conv_1d_width - 1)) * model_dim;
    }
    for (size_t i = 0; i < model_dim; i += hn::Lanes(df)) {
      auto xv = hn::Load(df, x + i);
      auto accum0 =
          hn::Load(df, layer_weights->griffin.conv_biases.PackedScale1() + i);
      auto accum1 = hn::Zero(df);
      for (size_t l = 0; 2 * l < conv_1d_width; l++) {
        auto wv0 =
            hn::Load(df, layer_weights->griffin.conv_w.PackedScale1() +
                             (conv_1d_width - 1 - 2 * l) * model_dim + i);
        auto wv1 =
            hn::Load(df, layer_weights->griffin.conv_w.PackedScale1() +
                             (conv_1d_width - 2 - 2 * l) * model_dim + i);
        accum0 = hn::MulAdd(wv0, hn::Load(df, cache[l * 2] + i), accum0);
        accum1 = hn::MulAdd(wv1, hn::Load(df, cache[l * 2 + 1] + i), accum1);
      }
      hn::Store(hn::Add(accum0, accum1), df, x + i);
      hn::Store(xv, df, cache[HWY_MAX(conv_1d_width, 1) - 1] + i);
    }
  }

  // RGLRU
  for (size_t interleaved_idx = 0; interleaved_idx < num_interleaved;
       ++interleaved_idx) {
    const size_t query_idx = div_num_q.Remainder(interleaved_idx);
    const size_t batch_idx = div_num_q.Divide(interleaved_idx);
    const size_t pos = queries_pos[query_idx] + batch_idx;

    float* HWY_RESTRICT x = activations.griffin_x.Row(query_idx);
    float* HWY_RESTRICT y = activations.griffin_y.Row(query_idx);
    float* HWY_RESTRICT gate_x = activations.griffin_gate_x.Row(query_idx);
    float* HWY_RESTRICT a = activations.griffin_multiplier.Row(query_idx);
    float* HWY_RESTRICT rnn_state =
        kv_caches[query_idx].rglru_cache.Row(griffin_layer);

    pool.Run(0, heads, [&](const uint64_t head, size_t /*thread*/) HWY_ATTR {
      size_t head_offset = head * kHeadDim;
      TwoOfsMatVecAddLoop(
          layer_weights->griffin.gate_w, kMatrixSize * head,
          kMatrixSize * (heads + head), kHeadDim, kHeadDim, x + head_offset,
          /*add0=*/layer_weights->griffin.gate_biases.PackedScale1() +
              head_offset,
          /*add1=*/layer_weights->griffin.gate_biases.PackedScale1() +
              model_dim + head_offset,
          /*out0=*/gate_x + head_offset, /*out1=*/a + head_offset);
      Sigmoid(gate_x + head_offset, kHeadDim);
      Sigmoid(a + head_offset, kHeadDim);
      const auto fn_mul = [](D d, hn::Vec<D> x, hn::Vec<D> gate_x)
                          HWY_ATTR { return hn::Mul(x, gate_x); };
      hn::Transform1(D(), a + head_offset, kHeadDim,
                     layer_weights->griffin.a.PackedScale1() + head_offset,
                     fn_mul);
      hn::Transform1(D(), x + head_offset, kHeadDim, gate_x + head_offset,
                     fn_mul);
      // RNN scan
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
  }  // interleaved_idx

  // Final linear layer.
  // TODO: MatMul
  for (size_t r = 0; r < num_interleaved; ++r) {
    float* HWY_RESTRICT x = activations.griffin_x.Row(r);
    float* out_ptr = activations.att_sums.Row(r);
    MatVecAdd(layer_weights->griffin.linear_out_w, 0, model_dim, model_dim, x,
              layer_weights->griffin.linear_out_biases.PackedScale1(), out_ptr,
              pool);
  }
}  // GriffinRecurrent

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
    const float* inv_timescale = activations_.inv_timescale.Packed();
    bool is_global_layer =
        activations_.weights_config.attention_window_sizes[layer] ==
        activations_.seq_len;
    // TODO: add a config flag instead of hardcoding the model.
    if (is_global_layer && IsVLM(activations_.weights_config.model)) {
      inv_timescale = activations_.inv_timescale_global.Packed();
    }
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

    // The original qkv_einsum_w has shape [(heads + kv_heads * 2), kKQVDim,
    // model_dim], which we reshaped to (heads + kv_heads * 2) * kKQVDim rows.
    // We must shrink to the actual size because MatMul verifies
    // `B.extents.rows == C.Cols()`. If MHA, `QStride() == 3 * qkv_dim` and all
    // rows are used. Otherwise, `QStride() == qkv_dim` and KV will be
    // computed in the second MatMul.
    const size_t w1_rows = heads * layer_config_.QStride();
    HWY_DASSERT(layer_weights_.qkv_einsum_w1.Rows() == w1_rows);
    MatMul(activations_.pre_att_rms_out, layer_weights_.qkv_einsum_w1,
           /*add=*/nullptr, *activations_.env, RowPtrFromMat(activations_.q));

    if (is_mha_) {
      // Multi-Head Attention a.k.a. "use_qkv_einsum" computed QKV already.
    } else {
      // KV structure is [k, v, k, v, ....] = kv_heads pairs of (k, v).
      const size_t w_rows_kv_cols = kv_heads * 2 * qkv_dim;
      HWY_DASSERT(layer_weights_.qkv_einsum_w2.Rows() == w_rows_kv_cols);

      // Single query and no wraparound means we can use a matmul and write
      // directly into the KV cache with a stride of cache_pos_size_.
      if (num_queries_ == 1 &&
          queries_pos_[0] + num_tokens_ <= div_seq_len_.GetDivisor()) {
        const size_t kv_ofs =
            queries_pos_[0] * cache_pos_size_ + layer_ * cache_layer_size_;
        float* HWY_RESTRICT kv = kv_caches_[0].kv_cache.get() + kv_ofs;
        RowPtrF kv_rows(kv, w_rows_kv_cols);
        kv_rows.SetStride(cache_pos_size_);
        MatMul(activations_.pre_att_rms_out, layer_weights_.qkv_einsum_w2,
               /*add=*/nullptr, *activations_.env, kv_rows);
      } else {
        // Proceed row by row because there will be wraparound.
        for (size_t interleaved_idx = 0; interleaved_idx < num_interleaved;
             ++interleaved_idx) {
          const float* x = activations_.pre_att_rms_out.Row(interleaved_idx);
          const size_t query_idx = interleaved_idx % num_queries_;
          const size_t batch_idx = interleaved_idx / num_queries_;
          KVCache& kv_cache = kv_caches_[query_idx];
          const size_t cache_pos =
              div_seq_len_.Remainder(queries_pos_[query_idx] + batch_idx);
          const size_t kv_offset =
              cache_pos * cache_pos_size_ + layer_ * cache_layer_size_;
          float* HWY_RESTRICT kv = kv_cache.kv_cache.get() + kv_offset;
          MatVec(layer_weights_.qkv_einsum_w2, 0, w_rows_kv_cols, model_dim, x,
                 kv, pool_);
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
                      activations_.q.Row(interleaved_idx) + head * q_stride_ +
                      qkv_dim;
                  hwy::CopyBytes(mha_kv, kv, 2 * qkv_dim * sizeof(*kv));
                }

                // Apply further processing to K.
                if (layer_weights_.key_norm_scale.HasPtr()) {
                  RMSNormInplace(layer_weights_.key_norm_scale.PackedScale1(),
                                 0, kv, qkv_dim);
                }
                PositionalEncodingQK(kv, pos, layer_, /*mul=*/1.0f);
              });
  }

  // Computes Q.K scores, which are "logits" (or scores) stored to att.
  // `k` is a strided view of the kv cache with dimensions [seq_len, qkv_dim].
  HWY_INLINE void QDotK(const size_t start_pos, const size_t last_pos,
                        const float* HWY_RESTRICT q, const MatPtrT<float>& k,
                        float* HWY_RESTRICT att) {
    const size_t qkv_dim = layer_config_.qkv_dim;
    if (HWY_LIKELY(last_pos < activations_.seq_len)) {
      // Slightly faster: no wraparound.
      for (size_t pos = start_pos; pos <= last_pos; ++pos) {
        const float* HWY_RESTRICT k_ptr = k.Row(pos);
        const float score = Dot(q, k_ptr, qkv_dim);
        att[pos] = score;
      }
    } else {
      for (size_t pos = start_pos; pos <= last_pos; ++pos) {
        const size_t cache_pos = div_seq_len_.Remainder(pos);
        const float* HWY_RESTRICT k_ptr = k.Row(cache_pos);
        const float score = Dot(q, k_ptr, qkv_dim);
        att[pos % activations_.seq_len] = score;
      }
    }
  }

  // Accumulates the sum of v (from `kv_cache`) * probability (`att`) into
  // `att_out`. Equivalent in gemma/modules.py:
  // encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)
  // `v` is a strided view of the kv cache with dimensions [seq_len, qkv_dim].
  HWY_INLINE void WeightedSumV(const size_t start_pos, const size_t last_pos,
                               const float* HWY_RESTRICT att,
                               const MatPtrT<float>& v,
                               float* HWY_RESTRICT att_out) const {
    const size_t qkv_dim = layer_config_.qkv_dim;
    hwy::ZeroBytes(att_out, qkv_dim * sizeof(*att_out));

    if (HWY_LIKELY(last_pos < activations_.seq_len)) {
      // Slightly faster: no wraparound.
      for (size_t pos = start_pos; pos <= last_pos; ++pos) {
        const float* HWY_RESTRICT v_ptr = v.Row(pos);
        MulByConstAndAdd(att[pos], v_ptr, att_out, qkv_dim);
      }
    } else {
      for (size_t pos = start_pos; pos <= last_pos; ++pos) {
        const size_t cache_pos = div_seq_len_.Remainder(pos);
        const float* HWY_RESTRICT v_ptr = v.Row(cache_pos);
        MulByConstAndAdd(att[pos % activations_.seq_len], v_ptr, att_out,
                         qkv_dim);
      }
    }
  }

 public:
  // Calculates the attention outputs for a single q.
  HWY_INLINE void SingleDotSoftmaxWeightedSum(
      float* HWY_RESTRICT q, const MatPtrT<float>& k, const MatPtrT<float>& v,
      float* HWY_RESTRICT att, float* HWY_RESTRICT att_out,
      const float query_scale, const size_t pos, const size_t start_pos,
      const size_t last_pos) {
    const size_t qkv_dim = layer_config_.qkv_dim;

    // Apply rope and scaling to Q.
    if (layer_weights_.query_norm_scale.HasPtr()) {
      RMSNormInplace(layer_weights_.query_norm_scale.PackedScale1(), 0, q,
                     qkv_dim);
    }
    PositionalEncodingQK(q, pos, layer_, query_scale);

    QDotK(start_pos, last_pos, q, k, att);

    // SoftMax with optional SoftCap yields "probabilities" in att.
    const size_t att_len = std::min(last_pos + 1, activations_.seq_len);
    MaybeLogitsSoftCap(activations_.weights_config.att_cap, att, att_len);
    Softmax(att, att_len);

    WeightedSumV(start_pos, last_pos, att, v, att_out);
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

                float* HWY_RESTRICT q =
                    activations_.q.Row(interleaved_idx) + head * q_stride_;
                float* HWY_RESTRICT att =
                    activations_.att.Row(interleaved_idx) +
                    head * activations_.seq_len;
                float* HWY_RESTRICT att_out =
                    activations_.att_out.Row(interleaved_idx) + head * qkv_dim;

                // Make strided views into the kv cache entries for the current
                // query and head.
                KVCache& kv_cache = kv_caches_[query_idx];
                const size_t kv_head_offset =
                    layer_ * cache_layer_size_ + head_offset;
                MatPtrT<float> k("k_view",
                                 Extents2D(kv_cache.seq_len, qkv_dim));
                k.SetPtr(kv_cache.kv_cache.get() + kv_head_offset,
                         /*stride=*/cache_pos_size_);
                MatPtrT<float> v("v_view",
                                 Extents2D(kv_cache.seq_len, qkv_dim));
                v.SetPtr(kv_cache.kv_cache.get() + kv_head_offset + qkv_dim,
                         /*stride=*/cache_pos_size_);

                // Find the token position in the query and calculate the range
                // of cache positions to attend to.
                const size_t pos = queries_pos_[query_idx] + batch_idx;
                const size_t start_pos = StartPos(pos, layer_);
                size_t last_pos = pos;
                const size_t prefix_end = queries_prefix_end_[query_idx];
                if (prefix_end > 0 && prefix_end - 1 > last_pos) {
                  // last_pos in QDotK and WeightedSumV is inclusive.
                  last_pos = prefix_end - 1;
                }

                SingleDotSoftmaxWeightedSum(q, k, v, att, att_out, query_scale,
                                            pos, start_pos, last_pos);
              });
  }

 private:
  // Sums encoded (`att_out`) over num_heads (`layer_config_.heads`) and
  // head_dim (`qkv_dim`) into output (`layer_out`).
  HWY_NOINLINE void SumHeads() {
    PROFILER_ZONE("Gen.Attention.SumHeads");
    // att_weights and att_out are concatenated heads, each of length
    // layer_config_.qkv_dim. Thus the [num_interleaved,
    // layer_config_.model_dim] matmul output is the sum over heads. Compare
    // gemma/modules.py: attn_output = self.attn_vec_einsum('BTNH,NHD->BTD',
    // encoded)
    HWY_DASSERT(layer_config_.model_dim != 0 && layer_config_.heads != 0 &&
                layer_config_.qkv_dim != 0);
    HWY_DASSERT(layer_weights_.att_weights.HasPtr());
    HWY_DASSERT(activations_.att_out.HasPtr());
    HWY_DASSERT(activations_.att_sums.HasPtr());

    const float* add =
        layer_weights_.layer_config.softmax_attn_output_biases
            ? layer_weights_.attention_output_biases.PackedScale1()
            : nullptr;
    MatMul(activations_.att_out, layer_weights_.att_weights, add,
           *activations_.env, RowPtrFromMat(activations_.att_sums));
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
                       activations, layer_weights, div_seq_len, kv_caches,
                       activations.env->ctx) {}
  // Constructor with default initialization to 0 for queries_prefix_end.
  GemmaAttention(const QueriesPos& queries_pos, size_t num_tokens, size_t layer,
                 Activations& activations,
                 const LayerWeightsPtrs<T>* layer_weights,
                 const hwy::Divisor& div_seq_len, const KVCaches& kv_caches)
      : GemmaAttention(queries_pos, nullptr, num_tokens, layer, activations,
                       layer_weights, div_seq_len, kv_caches,
                       activations.env->ctx) {}
  // Constructor with an explicit ThreadingContext. This is needed for
  // experimental code that invokes methods that do not use `activations.env`.
  // Callers should not have to construct an `activations.env` just to pass in
  // the threading context.
  GemmaAttention(const QueriesPos& queries_pos,
                 const QueriesPos& queries_prefix_end, size_t num_tokens,
                 size_t layer, Activations& activations,
                 const LayerWeightsPtrs<T>* layer_weights,
                 const hwy::Divisor& div_seq_len, const KVCaches& kv_caches,
                 ThreadingContext& ctx)
      : GemmaAttention(queries_pos, &queries_prefix_end, num_tokens, layer,
                       activations, layer_weights, div_seq_len, kv_caches,
                       ctx) {}

  // Full attention computation in three steps.
  HWY_INLINE void operator()() {
    const size_t num_interleaved = num_tokens_ * num_queries_;
    ComputeQKV(num_interleaved);
    DotSoftmaxWeightedSum(num_interleaved);
    SumHeads();
  }

 private:
  // Delegated Constructor that does most of the common work.
  GemmaAttention(const QueriesPos& queries_pos,
                 const QueriesPos* queries_prefix_end, size_t num_tokens,
                 size_t layer, Activations& activations,
                 const LayerWeightsPtrs<T>* layer_weights,
                 const hwy::Divisor& div_seq_len, const KVCaches& kv_caches,
                 ThreadingContext& ctx)
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
        pool_(ctx.pools.Pool(0)) {
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
    HWY_DASSERT(type == LayerAttentionType::kGriffinRecurrentBlock);
    // KVCache conv1d_cache and rglru_cache have one row per *Griffin* layer,
    // so map `layer` to the Griffin layer index.
    const size_t griffin_layer =
        activations.weights_config.NumLayersOfTypeBefore(type, layer);
    GriffinRecurrent(queries_pos, num_tokens, griffin_layer, activations,
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
    HWY_ASSERT(qkv.Rows() == num_tokens_);
    HWY_ASSERT(qkv.Cols() == layer_config_.heads * 3 * layer_config_.qkv_dim);
    MatMul(activations_.pre_att_rms_out, layer_weights_.vit.qkv_einsum_w,
           layer_weights_.vit.qkv_einsum_b.PackedScale1(), *activations_.env,
           RowPtrFromMat(qkv));
  }

  // TODO(philculliton): transition fully to MatMul.
  HWY_NOINLINE void DotSoftmaxWeightedSumMatrix() {
    const size_t qkv_dim = layer_config_.qkv_dim;
    const size_t heads = layer_config_.heads;
    HWY_ASSERT_M(heads == layer_config_.kv_heads, "Vit expects MHA");
    const size_t seq_len = activations_.seq_len;
    const float query_scale = 1.0f / sqrtf(static_cast<float>(qkv_dim));
    PROFILER_ZONE("Gen.VitAttention.DotSoftmax");

    // Shift Q, K, VT to MatStorageT.
    MatStorageT<float> Q("Q2", Extents2D(num_tokens_, qkv_dim),
                         MatPadding::kPacked);
    MatStorageT<float> K("K2", Extents2D(seq_len, qkv_dim),
                         MatPadding::kPacked);
    MatStorageT<float> C("C2", Extents2D(num_tokens_, seq_len),
                         MatPadding::kPacked);

    // Initialize att_out to zero prior to head loop.
    ZeroInit(activations_.att_out);

    for (size_t head = 0; head < heads; ++head) {
      pool_.Run(0, num_tokens_, [&](uint64_t task, size_t /*thread*/) HWY_ATTR {
        const size_t token = task;
        float* HWY_RESTRICT q = activations_.q.Row(token) + head * 3 * qkv_dim;
        // TODO: shift to MatMul with A.scale once MatMul is confirmed working
        MulByConst(query_scale, q, qkv_dim);
        hwy::CopyBytes(q, Q.Row(token), qkv_dim * sizeof(float));
      });

      pool_.Run(0, seq_len, [&](uint64_t task, size_t /*thread*/) HWY_ATTR {
        const size_t seq_idx = task;
        float* HWY_RESTRICT k =
            activations_.q.Row(seq_idx) + head * 3 * qkv_dim + qkv_dim;
        hwy::CopyBytes(k, K.Row(seq_idx), qkv_dim * sizeof(float));
      });

      // this produces C, a (num_tokens_, seq_len) matrix of dot products
      MatMul(Q, K, nullptr, *activations_.env, RowPtrFromMat(C));

      pool_.Run(0, num_tokens_, [&](uint64_t task, size_t /*thread*/) HWY_ATTR {
        float* HWY_RESTRICT c = C.Row(task);
        Softmax(c, C.Cols());
      });

      pool_.Run(0, num_tokens_, [&](uint64_t task, size_t /*thread*/) HWY_ATTR {
        size_t token = task;
        float* HWY_RESTRICT att_out =
            activations_.att_out.Row(token) + head * qkv_dim;
        for (size_t i = 0; i < seq_len; ++i) {
          float* HWY_RESTRICT v =
              activations_.q.Row(i) + head * 3 * qkv_dim + 2 * qkv_dim;
          MulByConstAndAdd(C.Row(token)[i], v, att_out, qkv_dim);
        }
      });
    }
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
                    activations_.q.Row(token) + head * 3 * qkv_dim;
                MulByConst(query_scale, q, qkv_dim);
                float* HWY_RESTRICT head_att =
                    activations_.att.Row(token) + head * activations_.seq_len;
                for (size_t i = 0; i < seq_len; ++i) {
                  float* HWY_RESTRICT k =
                      activations_.q.Row(i) + head * 3 * qkv_dim + qkv_dim;
                  head_att[i] = Dot(q, k, qkv_dim);  // score = q.k
                }
                // SoftMax yields "probabilities" in head_att.
                Softmax(head_att, seq_len);
                // Compute weighted sum of v into att_out.
                float* HWY_RESTRICT att_out =
                    activations_.att_out.Row(token) + head * qkv_dim;
                hwy::ZeroBytes(att_out, qkv_dim * sizeof(*att_out));
                for (size_t i = 0; i < seq_len; ++i) {
                  float* HWY_RESTRICT v =
                      activations_.q.Row(i) + head * 3 * qkv_dim + 2 * qkv_dim;
                  MulByConstAndAdd(head_att[i], v, att_out, qkv_dim);
                }
              });
  }

  // Sums encoded (`att_out`) over num_heads (`layer_config_.heads`) and
  // head_dim (`qkv_dim`) into output (`att_sums`).
  HWY_NOINLINE void SumHeads() {
    PROFILER_ZONE("Gen.VitAttention.SumHeads");
    auto* bias = layer_weights_.vit.attn_out_b.PackedScale1();
    // att_weights and att_out are concatenated heads, each of length
    // qkv_dim. Thus the [num_tokens_, layer_config_.model_dim]
    // matmul output is the sum over heads.
    auto att_sums = RowPtrFromMat(activations_.att_sums);
    MatMul(activations_.att_out, layer_weights_.vit.attn_out_w, bias,
           *activations_.env, att_sums);
  }

 public:
  VitAttention(size_t num_tokens, size_t layer, Activations& activations,
               const LayerWeightsPtrs<T>* layer_weights)
      : num_tokens_(num_tokens),
        layer_(layer),
        activations_(activations),
        layer_weights_(*layer_weights),
        layer_config_(layer_weights->layer_config),
        pool_(activations.env->ctx.pools.Pool(0)) {}

  HWY_INLINE void operator()() {
    ComputeQKV();
    if (activations_.weights_config.wrapping == PromptWrapping::GEMMA_VLM) {
      DotSoftmaxWeightedSumMatrix();
    } else {
      DotSoftmaxWeightedSum();
    }
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
                             const T* HWY_RESTRICT c2, size_t count) {
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

// No C2 multiplier.
template <class Mat>
void ActivationBatched(ActivationType activation, Mat& c1) {
  using T = typename Mat::T;
  for (size_t i = 0; i < c1.Rows(); ++i) {
    // Cast to correct type so type deduction works.
    Activation(activation, c1.Row(i), static_cast<const T*>(nullptr),
               c1.Cols());
  }
}

template <class Mat>
void ActivationBatched(ActivationType activation, Mat& c1, const Mat* c2) {
  using T = typename Mat::T;
  HWY_DASSERT(c1.SameShape(*c2));
  if (c2 && c2->HasPtr()) {
    for (size_t i = 0; i < c1.Rows(); ++i) {
      Activation(activation, c1.Row(i), c2->Row(i), c1.Cols());
    }
  } else {  // No multiplier
    for (size_t i = 0; i < c1.Rows(); ++i) {
      Activation(activation, c1.Row(i), static_cast<const T*>(nullptr),
                 c1.Cols());
    }
  }
}

template <typename T>
HWY_NOINLINE void FFWNoVit(Activations& activations,
                           const LayerWeightsPtrs<T>* layer_weights) {
  PROFILER_ZONE("Gen.FFW");
  const size_t ffh_hidden_dim = layer_weights->layer_config.ff_hidden_dim;

  const bool add_bias = layer_weights->layer_config.ff_biases;
  const float* bias1 =
      add_bias ? layer_weights->ffw_gating_biases.PackedScale1() : nullptr;
  const float* bias2 = add_bias ? bias1 + ffh_hidden_dim : nullptr;
  const float* output_bias =
      add_bias ? layer_weights->ffw_output_biases.PackedScale1() : nullptr;

  // Compute the hidden layer activations.
  MatMul(activations.pre_ffw_rms_out, layer_weights->gating_einsum_w1, bias1,
         *activations.env, RowPtrFromMat(activations.C1));
  MatMul(activations.pre_ffw_rms_out, layer_weights->gating_einsum_w2, bias2,
         *activations.env, RowPtrFromMat(activations.C2));

  // Activation (Gelu) and maybe multiply by gate. Store activations in act.
  ActivationBatched(layer_weights->layer_config.activation, activations.C1,
                    &activations.C2);

  // Hidden layer -> output layer.
  MatMul(activations.C1, layer_weights->linear_w, output_bias, *activations.env,
         RowPtrFromMat(activations.ffw_out));
}

// Same as FFWNoVit, but with different layer_weights members and no second
// gating matrix.
template <typename T>
HWY_NOINLINE void FFWVit(Activations& activations,
                         const LayerWeightsPtrs<T>* layer_weights) {
  PROFILER_ZONE("Gen.FFW.ViT");

  const bool add_bias = layer_weights->layer_config.ff_biases;
  const float* bias1 =
      add_bias ? layer_weights->vit.linear_0_b.PackedScale1() : nullptr;
  const float* output_bias =
      add_bias ? layer_weights->vit.linear_1_b.PackedScale1() : nullptr;

  // Compute the hidden layer activations.
  MatMul(activations.pre_ffw_rms_out, layer_weights->vit.linear_0_w, bias1,
         *activations.env, RowPtrFromMat(activations.C1));

  // Activation (Gelu), store in C1.
  ActivationBatched(layer_weights->layer_config.activation, activations.C1);

  // Hidden layer -> output layer.
  MatMul(activations.C1, layer_weights->vit.linear_1_w, output_bias,
         *activations.env, RowPtrFromMat(activations.ffw_out));
}

// `batch_idx` indicates which row of `x` to write to.
// `pos` is the *token*'s position, not the start of the batch, because this is
// called for batches of tokens in prefill, but batches of queries in decode.
//
// For GEMMA_VLM, image tokens are copied into -2 locations (per the Gemma 3
// spec) until we run out of image tokens. This allows for a multi-image prompt
// if -2 locations with appropriate begin/end image tokens are created by the
// calling application.
template <typename T>
HWY_NOINLINE void EmbedMMToken(int token, size_t batch_idx, size_t pos,
                               size_t pos_in_prompt,
                               const ModelWeightsPtrs<T>& weights,
                               MatStorageT<float>& x,
                               const ImageTokens* image_tokens,
                               size_t& image_token_position) {
  // Image tokens just need to be copied.
  if (weights.weights_config.wrapping == PromptWrapping::GEMMA_VLM &&
      image_tokens != nullptr && token == -2 &&
      image_token_position < image_tokens->Rows()) {
    hwy::CopyBytes(image_tokens->Row(image_token_position), x.Row(batch_idx),
                   x.Cols() * x.ElementBytes());
    image_token_position++;
    return;
  }

  if (weights.weights_config.wrapping == PromptWrapping::PALIGEMMA &&
      image_tokens != nullptr && pos_in_prompt < image_tokens->Rows()) {
    hwy::CopyBytes(image_tokens->Row(pos_in_prompt), x.Row(batch_idx),
                   x.Cols() * x.ElementBytes());
    return;
  }

  const size_t model_dim = weights.weights_config.model_dim;
  const float emb_scaling = EmbeddingScaling(model_dim);

  HWY_DASSERT(token >= 0);
  HWY_DASSERT(token < static_cast<int>(weights.weights_config.vocab_size));

  const hn::ScalableTag<float> df;
  // Using `Stride` to compute the offset works for both NUQ (because we use an
  // offset and NUQ is never padded) and padded, because non-NUQ types are
  // seekable, hence the offset can also skip any padding.
  const size_t embedding_ofs =
      token * weights.embedder_input_embedding.Stride();
  HWY_ASSERT(weights.embedder_input_embedding.Cols() == model_dim);
  const auto embedding_span = MakeSpan(weights.embedder_input_embedding.Row(0),
                                       embedding_ofs + model_dim);
  DecompressAndZeroPad(df, embedding_span, embedding_ofs, x.Row(batch_idx),
                       model_dim);
  MulByConst(emb_scaling * weights.embedder_input_embedding.Scale(),
             x.Row(batch_idx), model_dim);
  if (weights.weights_config.absolute_pe) {
    AddAbsolutePositionalEmbeddings(x.Row(batch_idx), model_dim, pos);
  }
}

// `batch_idx` indicates which row of `x` to write to.
// `pos` is the *token*'s position, not the start of the batch, because this is
// called for batches of tokens in prefill, but batches of queries in decode.
// This version of the function doesn't track internal image token position.
template <typename T>
HWY_NOINLINE void EmbedToken(int token, size_t batch_idx, size_t pos,
                             size_t pos_in_prompt,
                             const ModelWeightsPtrs<T>& weights,
                             MatStorageT<float>& x,
                             const ImageTokens* image_tokens) {
  size_t image_token_position = 0;
  EmbedMMToken<T>(token, batch_idx, pos, pos_in_prompt, weights, x,
                  image_tokens, image_token_position);
}

template <typename T2, class LayerWeights>
HWY_NOINLINE void ResidualConnection(const MatPtrT<T2>& other,
                                     MatPtrT<float>& HWY_RESTRICT x,
                                     const LayerWeights* layer_weights,
                                     bool is_attention) {
  // ResidualType::Add
  AddFromBatched(other, x);
}

template <typename WeightT, typename InOutT>
void PostNorm(PostNormType post_norm, const MatPtrT<WeightT>& weights,
              MatPtrT<InOutT>& inout) {
  HWY_DASSERT(weights.Rows() == 1);
  if (post_norm == PostNormType::Scale) {
    RMSNormInplaceBatched(weights, inout);
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
  auto type = layer_weights->layer_config.type;

  RMSNormBatched(activations.x, layer_weights->pre_attention_norm_scale,
                 activations.pre_att_rms_out);

  Attention(type, queries_pos, queries_prefix_end, num_tokens, cache_layer_idx,
            activations, layer_weights, div_seq_len, kv_caches);

  PostNorm(layer_weights->layer_config.post_norm,
           layer_weights->post_attention_norm_scale, activations.att_sums);

  ResidualConnection(activations.att_sums, activations.x, layer_weights,
                     /*is_attention=*/true);

  RMSNormBatched(activations.x, layer_weights->pre_ffw_norm_scale,
                 activations.pre_ffw_rms_out);

  if (layer_weights->layer_config.type == LayerAttentionType::kVit) {
    FFWVit(activations, layer_weights);
  } else {
    FFWNoVit(activations, layer_weights);
  }

  PostNorm(layer_weights->layer_config.post_norm,
           layer_weights->post_ffw_norm_scale, activations.ffw_out);

  ResidualConnection(activations.ffw_out, activations.x, layer_weights,
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
  (void)model_dim;

  auto& x = activations.x;
  HWY_DASSERT(x.Rows() == num_tokens);
  HWY_DASSERT(x.Cols() == model_dim);

  // y = nn.LayerNorm()(x)
  // y ~ pre_att_rms_out
  LayerNormBatched(x, layer_weights->vit.layer_norm_0_scale,
                   layer_weights->vit.layer_norm_0_bias,
                   activations.pre_att_rms_out);

  // y = out["sa"] = nn.MultiHeadDotProductAttention(...)(y, y)
  // y ~ att_sums
  VitAttention<T>(num_tokens, layer, activations, layer_weights)();

  // x = out["+sa"] = x + y
  AddFromBatched(activations.att_sums, x);

  // y = nn.LayerNorm()(x)
  // y ~ pre_ffw_rms_out
  LayerNormBatched(x, layer_weights->vit.layer_norm_1_scale,
                   layer_weights->vit.layer_norm_1_bias,
                   activations.pre_ffw_rms_out);

  // y = out["mlp"] = MlpBlock(...)(y)
  // y ~ ffw_out
  FFWVit(activations, layer_weights);

  // x = out["+mlp"] = x + y
  AddFromBatched(activations.ffw_out, x);
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
  const size_t max_tbatch_size = runtime_config.prefill_tbatch_size;

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
      activations.SetBatchSize(tbatch_size);

      // Fill activations.x (much faster than TransformerLayer).
      size_t image_token_position = 0;
      for (size_t ti = 0; ti < tbatch_size; ++ti) {
        const size_t pos = queries_pos[qi] + ti;
        const size_t pos_in_prompt = tbatch_start + ti;
        const int token = queries_prompt[qi][pos_in_prompt];
        EmbedMMToken(token, ti, pos, pos_in_prompt, weights, activations.x,
                     runtime_config.image_tokens, image_token_position);
      }

      // Transformer with one batch of tokens from a single query.
      for (size_t layer = 0;
           layer < weights.weights_config.layer_configs.size(); ++layer) {
        const LayerWeightsPtrs<T>* layer_weights = weights.GetLayer(layer);
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
  HWY_DASSERT(weights.vit_img_embedding_kernel.Rows() == model_dim);
  HWY_DASSERT(weights.vit_img_embedding_kernel.Cols() == patch_size);
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
  // MatStorageT<float> image_patches("patches", Extents2D(kSeqLen,
  //   kPatchSize), MatPadding::kPacked);
  // [Get patches]
  // MatMul(
  //       MatFromBatch(kVitSeqLen, image_patches),
  //       MatFromWeights(weights.vit_img_embedding_kernel),
  //       weights.vit_img_embedding_bias.PackedScale1(), *activations.env,
  //       RowPtrF(activations.x.Row(0), kVitModelDim));
  // However, MatMul currently requires that
  //   A.cols % (2 * hn::Lanes(hn::ScalableTag<MulT>())) == 0
  // which is not the case here. We should relax that requirement on MatMul and
  // then use the above. For now, we rely on MatVecAdd instead.
  for (size_t i = 0; i < seq_len; ++i) {
    MatVecAdd(weights.vit_img_embedding_kernel, 0, model_dim, patch_size,
              image_patches[i].get(),
              weights.vit_img_embedding_bias.PackedScale1(),
              activations.x.Row(i), activations.env->ctx.pools.Pool(0));
  }
  // Add position embeddings.
  AddFromBatched(weights.vit_img_pos_embedding, activations.x);
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
  HWY_ASSERT(num_tokens == activations.x.Rows());
  // Embed the image patches.
  EmbedImagePatches(image, weights, activations);
  // Go through all layers.
  for (size_t layer = 0;
       layer < weights.weights_config.vit_config.layer_configs.size();
       ++layer) {
    const LayerWeightsPtrs<T>* layer_weights = weights.VitLayer(layer);
    VitTransformerLayer(num_tokens, layer, layer_weights, activations);
  }
  // Final Layernorm.
  LayerNormBatched(activations.x, weights.vit_encoder_norm_scale,
                   weights.vit_encoder_norm_bias, activations.x);

  if (weights.weights_config.wrapping == PromptWrapping::GEMMA_VLM) {
    activations.x = AvgPool4x4(activations.x);

    // Apply soft embedding norm before input projection.
    RMSNormInplace(weights.mm_embed_norm.PackedScale1(), 0,
                   activations.x.Row(0), vit_model_dim);
  }

  // Apply head embedding into image_tokens of size of the LLM kModelDim.
  MatMul(activations.x, weights.vit_img_head_kernel,
         weights.vit_img_head_bias.PackedScale1(), *activations.env,
         RowPtrFromMat(image_tokens));
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

  size_t image_token_position = 0;
  for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
    EmbedMMToken(queries_token[query_idx], query_idx, queries_pos[query_idx],
                 /*pos_in_prompt=*/0, weights, activations.x,
                 /*image_tokens=*/nullptr, image_token_position);
  }

  for (size_t layer = 0; layer < weights.c_layers.size(); ++layer) {
    const LayerWeightsPtrs<T>* layer_weights = weights.GetLayer(layer);
    TransformerLayer(queries_pos, queries_prefix_end, /*num_tokens=*/1, layer,
                     layer_weights, activations, div_seq_len, kv_caches);

    if (activations_observer) {
      activations_observer(queries_pos, layer, activations);
    }
  }

  RMSNormInplaceBatched(weights.final_norm_scale, activations.x);

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
  explicit TokenStreamer(const RuntimeConfig& runtime_config,
                         const ModelConfig& model_config)
      : runtime_config_(runtime_config), model_config_(model_config) {}

  // Returns whether the query was already at, or has just reached, the end of
  // the stream: either via token == eos_id, or StreamToken returning false.
  bool operator()(size_t query_idx, size_t pos, int token, float prob) {
    if (HWY_UNLIKELY(is_eos_.Get(query_idx))) return true;

    if (!runtime_config_.StreamToken(query_idx, pos, token, prob) ||
        model_config_.IsEOS(token)) {
      is_eos_.Set(query_idx);
      return true;
    }

    return false;
  }

 private:
  const RuntimeConfig& runtime_config_;
  const ModelConfig& model_config_;
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
    return FusedSoftmaxAndSampleTopK(
        logits, runtime_config.top_k, vocab_size, *runtime_config.gen,
        runtime_config.temperature, runtime_config.accept_token);
  };
}

template <typename T>
// Runs one decode step for all the queries in the batch. Returns true if all
// queries are at <end_of_sentence>.
bool DecodeStepT(const ModelConfig& config, const ModelWeightsPtrs<T>& weights,
                 const RuntimeConfig& runtime_config,
                 const QueriesPromptTokens& queries_prompt,
                 const size_t query_idx_start, const KVCaches& kv_caches,
                 const QueriesPos& queries_prefix_end,
                 const hwy::Divisor div_seq_len, const size_t vocab_size,
                 const SampleFunc& sample_token, Activations& activations,
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

  HWY_DASSERT(num_queries == activations.x.Rows());
  bool all_queries_eos = true;
  {
    PROFILER_ZONE("Gen.EmbeddingMatmul");
    // Compute logits from last layer activations.
    MatMul(activations.x, weights.embedder_input_embedding,
           /*add=*/nullptr, *activations.env,
           RowPtrFromMat(activations.logits));
  }
  PROFILER_ZONE("Gen.Softcap+Sample+Stream");
  for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
    float* HWY_RESTRICT logits = activations.logits.Row(query_idx);
    MaybeLogitsSoftCap(weights.weights_config.final_cap, logits, vocab_size);
    const TokenAndProb tp = sample_token(logits, vocab_size);
    timing_info.NotifyGenerated();

    const bool is_eos =
        token_streamer(query_idx_start + query_idx,
                       queries_mutable_pos[query_idx], tp.token, tp.prob);
    all_queries_eos &= is_eos;
    gen_tokens[query_idx] = is_eos ? config.eos_id : tp.token;
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
void GenerateT(const ModelStore& model, const ModelWeightsPtrs<T>& weights,
               Activations& activations, const RuntimeConfig& runtime_config,
               const QueriesPromptTokens& queries_prompt,
               const QueriesPos& queries_pos_in,
               const QueriesPos& queries_prefix_end,
               const size_t query_idx_start, const KVCaches& kv_caches,
               TimingInfo& timing_info) {
  HWY_ASSERT(queries_pos_in.size() == kv_caches.size());
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
    HWY_ASSERT(prompt.size() != 0 && prompt[0] != model.Config().eos_id);
  }

  const size_t num_queries = queries_prompt.size();
  HWY_ASSERT(num_queries <= 4096);  // TokenStreamer uses BitSet4096.
  HWY_ASSERT(num_queries <= activations.x.Rows());
  HWY_ASSERT(queries_pos_in.size() == num_queries);
  HWY_ASSERT(kv_caches.size() == num_queries);
  const hwy::Divisor div_seq_len(static_cast<uint32_t>(kv_caches[0].seq_len));
  size_t max_prompt_size = MaxQueryLength(queries_prompt);
  size_t max_generated_tokens = runtime_config.max_generated_tokens;
  RangeChecks(weights.weights_config, max_generated_tokens, max_prompt_size);
  const SampleFunc sample_token = ChooseSampleFunc(runtime_config);

  // Prefill stops before min_prompt_size - 1 because the last prompt
  // token is the first input token for generation.
  timing_info.prefill_start = hwy::platform::Now();
  // Note that Prefill calls activations.SetBatchSize, so we reset it below.
  Prefill(queries_prompt, queries_mutable_pos, queries_prefix_end,
          query_idx_start, weights, activations, runtime_config, div_seq_len,
          kv_caches);
  // Compute the number of tokens that were prefilled and notify timing_info.
  size_t prefilled_tokens = 0;
  for (size_t qi = 0; qi < num_queries; ++qi) {
    prefilled_tokens += queries_prompt[qi].size() - 1;
  }
  timing_info.NotifyPrefill(prefilled_tokens);
  // queries_pos are incremented by Prefill.
  activations.SetBatchSize(num_queries);

  // Storage for the last generated token from each query, passed to the next
  // Transformer() call.
  std::vector<int> gen_tokens(num_queries);

  // Stream the last prompt token from each query and fill gen_tokens.
  TokenStreamer token_streamer(runtime_config, model.Config());
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
    timing_info.generate_start = hwy::platform::Now();
    for (size_t gen = 0; gen < max_generated_tokens; ++gen) {
      bool all_queries_eos = DecodeStepT<T>(
          model.Config(), weights, runtime_config, queries_prompt,
          query_idx_start, kv_caches, queries_prefix_end, div_seq_len,
          vocab_size, sample_token, activations, token_streamer, gen_tokens,
          timing_info, queries_mutable_pos);
      if (all_queries_eos) break;
    }  // foreach token to generate
    timing_info.NotifyGenerateDone();
  }
}

template <typename T>
void GenerateSingleT(const ModelStore& model,
                     const ModelWeightsPtrs<T>& weights,
                     const RuntimeConfig& runtime_config,
                     const PromptTokens& prompt, size_t pos, size_t prefix_end,
                     KVCache& kv_cache, MatMulEnv* env,
                     TimingInfo& timing_info) {
  constexpr size_t kNumQueries = 1;
  const size_t qbatch_start = 0;

  const size_t max_batch_size =
      HWY_MAX(kNumQueries, runtime_config.prefill_tbatch_size);
  // TODO: move into Gemma?
  Activations activations(model.Config(), max_batch_size, env);

  const QueriesPromptTokens queries_prompt(&prompt, kNumQueries);
  QueriesPos queries_pos(&pos, kNumQueries);
  const QueriesPos queries_prefix_end(&prefix_end, kNumQueries);
  const KVCaches kv_caches{&kv_cache, kNumQueries};

  GenerateT<T>(model, weights, activations, runtime_config, queries_prompt,
               queries_pos, queries_prefix_end, qbatch_start, kv_caches,
               timing_info);
}

template <typename T>
void GenerateBatchT(const ModelStore& model,
                    const ModelWeightsPtrs<T>& weights,
                    const RuntimeConfig& runtime_config,
                    const QueriesPromptTokens& queries_prompt,
                    const QueriesPos& queries_pos,
                    const QueriesPos& queries_prefix_end,
                    const KVCaches& kv_caches, MatMulEnv* env,
                    TimingInfo& timing_info) {
  const size_t num_queries = queries_prompt.size();
  HWY_ASSERT(queries_pos.size() == num_queries);
  HWY_ASSERT(kv_caches.size() >= num_queries);
  const size_t max_qbatch_size = runtime_config.decode_qbatch_size;
  const size_t max_batch_size =
      HWY_MAX(max_qbatch_size, runtime_config.prefill_tbatch_size);

  Activations activations(model.Config(), max_batch_size, env);

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
    GenerateT<T>(model, weights, activations, runtime_config, qbatch_prompts,
                 qbatch_pos, qbatch_prefix_end, qbatch_start, qbatch_kv,
                 timing_info);
  }
}

template <typename T>
void GenerateImageTokensT(const ModelStore& model,
                          const ModelWeightsPtrs<T>& weights,
                          const RuntimeConfig& runtime_config,
                          const Image& image, ImageTokens& image_tokens,
                          MatMulEnv* env) {
  if (model.Config().vit_config.layer_configs.empty()) {
    HWY_ABORT("Model does not support generating image tokens.");
  }
  RuntimeConfig prefill_runtime_config = runtime_config;
  ModelConfig vit_config = GetVitConfig(model.Config());
  prefill_runtime_config.prefill_tbatch_size =
      vit_config.seq_len / (vit_config.pool_dim * vit_config.pool_dim);
  Activations prefill_activations(vit_config, vit_config.seq_len, env);
  // Weights are for the full PaliGemma model, not just the ViT part.
  PrefillVit(weights, prefill_runtime_config, image, image_tokens,
             prefill_activations);
}

}  // namespace HWY_NAMESPACE

#if HWY_ONCE

// These are extern functions defined by instantiations/*.cc, which include this
// 'header' after defining `GEMMA_TYPE`.
void GenerateSingle(  // NOLINT(misc-definitions-in-headers)
    const ModelStore& model, const ModelWeightsPtrs<GEMMA_TYPE>& weights,
    const RuntimeConfig& runtime_config, const PromptTokens& prompt, size_t pos,
    size_t prefix_end, KVCache& kv_cache, MatMulEnv* env,
    TimingInfo& timing_info) {
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(GenerateSingleT<GEMMA_TYPE>)
  (model, weights, runtime_config, prompt, pos, prefix_end, kv_cache, env,
   timing_info);
}

void GenerateBatch(  // NOLINT(misc-definitions-in-headers)
    const ModelStore& model, const ModelWeightsPtrs<GEMMA_TYPE>& weights,
    const RuntimeConfig& runtime_config,
    const QueriesPromptTokens& queries_prompt, const QueriesPos& queries_pos,
    const QueriesPos& queries_prefix_end, const KVCaches& kv_caches,
    MatMulEnv* env, TimingInfo& timing_info) {
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(GenerateBatchT<GEMMA_TYPE>)
  (model, weights, runtime_config, queries_prompt, queries_pos,
   queries_prefix_end, kv_caches, env, timing_info);
}

void GenerateImageTokens(  // NOLINT(misc-definitions-in-headers)
    const ModelStore& model, const ModelWeightsPtrs<GEMMA_TYPE>& weights,
    const RuntimeConfig& runtime_config, const Image& image,
    ImageTokens& image_tokens, MatMulEnv* env) {
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(GenerateImageTokensT<GEMMA_TYPE>)
  (model, weights, runtime_config, image, image_tokens, env);
}

#endif  // HWY_ONCE

}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_INL_H_
