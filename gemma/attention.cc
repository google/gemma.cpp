// Copyright 2025 Google LLC
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

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "gemma/activations.h"
#include "gemma/gemma.h"
#include "gemma/weights.h"
#include "util/threading.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"

// Compiles this file for multiple architectures via "foreach_target.h", to
// which we pass the filename via macro 'argument'.
// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/attention.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "ops/ops-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// Computes Q.K scores, which are "logits" (or scores) stored to att.
// `k` is a strided view of the kv cache with dimensions [seq_len, qkv_dim].
static HWY_INLINE void QDotK(const size_t start_pos, const size_t last_pos,
                             const hwy::Divisor& div_seq_len,
                             const float* HWY_RESTRICT q,
                             const MatPtrT<float>& k, float* HWY_RESTRICT att) {
  if (HWY_LIKELY(last_pos < static_cast<size_t>(div_seq_len.GetDivisor()))) {
    // Slightly faster: no wraparound.
    for (size_t pos = start_pos; pos <= last_pos; ++pos) {
      const float score = Dot(q, k.Row(pos), k.Cols());
      att[pos] = score;
    }
  } else {
    for (size_t pos = start_pos; pos <= last_pos; ++pos) {
      const size_t pos_modulo = div_seq_len.Remainder(pos);
      const float score = Dot(q, k.Row(pos_modulo), k.Cols());
      att[pos_modulo] = score;
    }
  }
}

template <typename U>
static void PositionalEncodingQK(U* qk, const size_t qkv_dim,
                                 const size_t layer_idx,
                                 const LayerWeightsPtrs& layer,
                                 const AttentionActivations& activations,
                                 const size_t pos, const float mul = 1.0f) {
  const PostQKType& post_qk = layer.layer_config.post_qk;
  // qk is either q or k, so qkv_dim is the length we operate on.
  const float* inv_timescale = activations.inv_timescale.PackedScale1();
  bool is_global_layer = activations.IsGlobalLayer(layer_idx);
  // TODO: add a config flag instead of hardcoding the model.
  if (is_global_layer && IsVLM(activations.config.model)) {
    inv_timescale = activations.inv_timescale_global.PackedScale1();
  }
  // PostQKType::Rope
  if (post_qk == PostQKType::HalfRope) {
    Rope(qk, qkv_dim / 2, inv_timescale, pos);
    if (mul != 1.0f) MulByConst(mul, qk, qkv_dim);
  } else {
    RopeAndMulBy(mul, qk, qkv_dim, inv_timescale, pos);
  }
}

// Accumulates the sum of v (from `kv_cache`) * probability (`att`) into
// `att_out`. Equivalent in gemma/modules.py:
// encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)
// `v` is a strided view of the kv cache with dimensions [seq_len, qkv_dim].
static HWY_INLINE void WeightedSumV(const size_t start_pos,
                                    const size_t last_pos,
                                    const hwy::Divisor& div_seq_len,
                                    const float* HWY_RESTRICT att,
                                    const MatPtrT<float>& v,
                                    float* HWY_RESTRICT att_out) {
  const size_t qkv_dim = v.Cols();
  hwy::ZeroBytes(att_out, qkv_dim * sizeof(*att_out));

  if (HWY_LIKELY(last_pos < static_cast<size_t>(div_seq_len.GetDivisor()))) {
    // Slightly faster: no wraparound.
    for (size_t pos = start_pos; pos <= last_pos; ++pos) {
      MulByConstAndAdd(att[pos], v.Row(pos), att_out, v.Cols());
    }
  } else {
    for (size_t pos = start_pos; pos <= last_pos; ++pos) {
      const size_t pos_modulo = div_seq_len.Remainder(pos);
      const float* HWY_RESTRICT v_ptr = v.Row(pos_modulo);
      MulByConstAndAdd(att[pos_modulo], v_ptr, att_out, v.Cols());
    }
  }
}

// Calculates the attention outputs for a single q, which may be updated
// in place for RMSNorm.
void SingleDotSoftmaxWeightedSum(
    const size_t pos, const size_t start_pos, const size_t last_pos,
    float* HWY_RESTRICT q, const MatPtrT<float>& k, const MatPtrT<float>& v,
    const size_t layer_idx, const LayerWeightsPtrs& layer,
    const AttentionActivations& activations, float* HWY_RESTRICT att,
    float* HWY_RESTRICT att_out) {
  const size_t qkv_dim = layer.layer_config.qkv_dim;
  const float att_cap = activations.config.att_cap;
  const float query_scale = activations.query_scale;
  const size_t seq_len =
      static_cast<size_t>(activations.div_seq_len.GetDivisor());

  // Apply rope and scaling to Q.
  if (layer.query_norm_scale.HasPtr()) {
    CallUpcasted(&layer.query_norm_scale, [&](const auto* weights_t) {
      RMSNormInplace(weights_t->PackedScale1(), 0, q, qkv_dim);
    });
  }
  PositionalEncodingQK(q, qkv_dim, layer_idx, layer, activations, pos,
                       query_scale);

  QDotK(start_pos, last_pos, activations.div_seq_len, q, k, att);

  // SoftMax with optional SoftCap yields "probabilities" in att.
  const size_t att_len = HWY_MIN(last_pos + 1, seq_len);
  MaybeLogitsSoftCap(att_cap, att, att_len);
  Softmax(att, att_len);

  WeightedSumV(start_pos, last_pos, activations.div_seq_len, att, v, att_out);
}

// The attention window usually starts at 0 unless `pos` is larger than
// the attention window size, then it is `pos` - window_size + 1.
static HWY_INLINE size_t StartPos(size_t pos, const ModelConfig& config,
                                  size_t layer_idx) {
  const size_t att_window_size = config.attention_window_sizes[layer_idx];
  return pos - HWY_MIN(att_window_size - 1, pos);
}

void DotSoftmaxWeightedSum(const size_t num_tokens, const size_t layer_idx,
                           const LayerWeightsPtrs& layer,
                           AttentionActivations& activations, QBatch& qbatch,
                           NestedPools& pools) {
  PROFILER_ZONE("Gen.Attention.DotSoftmax");
  const hwy::Divisor div_qbatch(qbatch.Size());
  const LayerConfig& layer_config = layer.layer_config;
  const size_t qkv_dim = layer_config.qkv_dim;

  // A "head group" in the context of GQA refers to a collection of query
  // heads that share the same key and value heads.
  const size_t kHeadGroups = layer_config.heads / layer_config.kv_heads;

  const size_t cache_layer_size = layer_config.CacheLayerSize();
  const size_t seq_len =
      static_cast<size_t>(activations.div_seq_len.GetDivisor());

  // For each head/token/query, compute Q.K, softmax, and weighted V.

  // Statically partition token/query across packages.
  const size_t num_tq = num_tokens * div_qbatch.GetDivisor();
  const IndexRangePartition tq_ranges =
      StaticPartition(IndexRange(0, num_tq), pools.NumPackages(), 1);
  ParallelizeOneRange(
      tq_ranges, pools.AllPackages(),
      [&](const IndexRange& tq_range, const size_t pkg_idx) {
        pools.AllClusters(pkg_idx).Run(
            tq_range.begin(), tq_range.end(),
            [&](const size_t tq_idx, const size_t cluster_idx) {
              const size_t qi = div_qbatch.Remainder(tq_idx);
              const size_t batch_idx = div_qbatch.Divide(tq_idx);
              auto& kv_cache = qbatch.KV(qi).kv_cache;

              // Find the token position in the query and calculate
              // the range of cache positions to attend to.
              const size_t pos = qbatch.Pos(qi) + batch_idx;
              const size_t start_pos =
                  StartPos(pos, activations.config, layer_idx);
              size_t last_pos = pos;
              const size_t prefix_end = qbatch.PrefixEnd(qi);
              if (prefix_end > 0 && prefix_end - 1 > last_pos) {
                // last_pos in QDotK and WeightedSumV is inclusive.
                last_pos = prefix_end - 1;
              }

              pools.Cluster(pkg_idx, cluster_idx)
                  .Run(
                      0, layer_config.heads,
                      [&](const size_t head, size_t thread) HWY_ATTR {
                        const size_t head_offset =
                            (head / kHeadGroups) * qkv_dim * 2;

                        float* HWY_RESTRICT q =
                            activations.q.Row(tq_idx) + head * qkv_dim;

                        float* HWY_RESTRICT att =
                            activations.att.Row(tq_idx) + head * seq_len;
                        float* HWY_RESTRICT att_out =
                            activations.att_out.Row(tq_idx) + head * qkv_dim;

                        // Make strided read-only views into the kv cache for
                        // this query and head.
                        const size_t kv_head_offset =
                            layer_idx * cache_layer_size + head_offset;
                        MatPtrT<float> k("k_view", Extents2D(seq_len, qkv_dim));
                        k.SetPtr(kv_cache.Row(0) + kv_head_offset,
                                 kv_cache.Stride());
                        MatPtrT<float> v("v_view", Extents2D(seq_len, qkv_dim));
                        v.SetPtr(kv_cache.Row(0) + kv_head_offset + qkv_dim,
                                 kv_cache.Stride());

                        SingleDotSoftmaxWeightedSum(pos, start_pos, last_pos, q,
                                                    k, v, layer_idx, layer,
                                                    activations, att, att_out);
                      });
            });
      });
}

// Different functions use different naming conventions for the number of
// tokens. Functions that are query-independent, such as RMSNorm*, call the
// count `num_interleaved`. Functions that are query-dependent, such as
// `Attention`, use separate `num_tokens` and `num_queries`. `num_tokens` is the
// number of tokens from one query: 1 for decode, otherwise prefill_tbatch_size.

// Fills activations.q and writes to KV cache.
static HWY_INLINE void ComputeQKV(size_t num_tokens, const size_t layer_idx,
                                  const LayerWeightsPtrs& layer,
                                  AttentionActivations& activations,
                                  const QBatch& qbatch, const int flags,
                                  MatMulEnv& env) {
  PROFILER_ZONE("Gen.Attention.QKV");
  const hwy::Divisor div_qbatch(qbatch.Size());
  const size_t num_interleaved = num_tokens * div_qbatch.GetDivisor();
  const LayerConfig& layer_config = layer.layer_config;
  const size_t qkv_dim = layer_config.qkv_dim;
  const size_t kv_heads = layer_config.kv_heads;
  const size_t cache_layer_size = layer_config.CacheLayerSize();

  // The original qkv_einsum_w has shape [(heads + kv_heads * 2), qkv_dim,
  // model_dim], which we reshaped to (heads + kv_heads * 2) * qkv_dim rows.
  CallMatMul(activations.pre_att_rms_out, layer.qkv_einsum_w1,
             /*add=*/nullptr, env, activations.q);

  // Set up MatMul row pointers for writing to KV, which consists of
  // `kv_heads` pairs of (k, v) vectors. This safely handles wraparound
  // because rows are computed modulo seq_len.
  MatPtrT<float> kv_rows("kv", Extents2D(activations.pre_att_rms_out.Rows(),
                                         layer.qkv_einsum_w2.Rows()));
  for (size_t interleaved_idx = 0; interleaved_idx < num_interleaved;
       ++interleaved_idx) {
    const size_t qi = div_qbatch.Remainder(interleaved_idx);
    const size_t batch_idx = div_qbatch.Divide(interleaved_idx);
    const size_t cache_pos =
        activations.div_seq_len.Remainder(qbatch.Pos(qi) + batch_idx);
    env.row_ptrs[0][interleaved_idx] = reinterpret_cast<uint8_t*>(
        qbatch.KV(qi).kv_cache.Row(cache_pos) + layer_idx * cache_layer_size);
  }
  kv_rows.AttachRowPtrs(env.row_ptrs[0].get());
  CallMatMul(activations.pre_att_rms_out, layer.qkv_einsum_w2,
             /*add=*/nullptr, env, kv_rows);

  // Apply positional encodings for K.
  // Note that 2D parallelism is not worth the fork/join overhead because the
  // tasks are very lightweight.
  env.ctx.pools.Pool(0).Run(
      0, kv_heads * num_interleaved,
      [&](uint64_t task, size_t /*thread*/) HWY_ATTR {
        const size_t head = task % kv_heads;
        const size_t interleaved_idx = task / kv_heads;
        const size_t qi = div_qbatch.Remainder(interleaved_idx);
        const size_t batch_idx = div_qbatch.Divide(interleaved_idx);
        const size_t pos = qbatch.Pos(qi) + batch_idx;
        const size_t cache_pos = activations.div_seq_len.Remainder(pos);
        auto& kv_cache = qbatch.KV(qi).kv_cache;
        float* HWY_RESTRICT kv = kv_cache.Row(cache_pos) +
                                 layer_idx * cache_layer_size +
                                 head * qkv_dim * 2;

        // Apply further processing to K.
        if (layer.key_norm_scale.HasPtr()) {
          CallUpcasted(&layer.key_norm_scale, [&](const auto* weights_t) {
            RMSNormInplace(weights_t->PackedScale1(), 0, kv, qkv_dim);
          });
        }

        PositionalEncodingQK(kv, qkv_dim, layer_idx, layer, activations, pos);
      });
}

// Sums encoded (`att_out`) over num_heads (`layer_config.heads`) and
// head_dim (`qkv_dim`) into output (`layer_out`).
static HWY_INLINE void SumHeads(const LayerWeightsPtrs& layer,
                                AttentionActivations& activations,
                                MatMulEnv& env) {
  PROFILER_ZONE("Gen.Attention.SumHeads");
  const LayerConfig& layer_config = layer.layer_config;
  // att_weights and att_out are concatenated heads, each of length
  // layer_config.qkv_dim. Thus the [num_interleaved,
  // layer_config.model_dim] matmul output is the sum over heads. Compare
  // gemma/modules.py: attn_output = self.attn_vec_einsum('BTNH,NHD->BTD',
  // encoded)
  HWY_DASSERT(layer_config.model_dim != 0 && layer_config.heads != 0 &&
              layer_config.qkv_dim != 0);
  const float* add = layer_config.softmax_attn_output_biases
                         ? layer.attention_output_biases.PackedScale1()
                         : nullptr;
  CallMatMul(activations.att_out, layer.att_weights, add, env,
             activations.att_sums);
}

void GemmaAttention(size_t num_tokens, const size_t layer_idx,
                    const LayerWeightsPtrs& layer,
                    AttentionActivations& activations, QBatch& qbatch,
                    MatMulEnv& env, int flags) {
  const LayerConfig& layer_config = layer.layer_config;
  HWY_DASSERT(!layer_config.IsMHA());  // No longer supported.
  HWY_DASSERT_M((layer_config.heads % layer_config.kv_heads) == 0,
                "query heads must be a multiple of key-value heads");
  (void)layer_config;  // only used in HWY_DASSERT

  ComputeQKV(num_tokens, layer_idx, layer, activations, qbatch, flags, env);
  DotSoftmaxWeightedSum(num_tokens, layer_idx, layer, activations, qbatch,
                        env.ctx.pools);
  SumHeads(layer, activations, env);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();
