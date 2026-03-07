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

#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#include "util/zones.h"
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include "gemma/activations.h"
#include "gemma/configs.h"  // kMaxQKVDim
#include "gemma/weights.h"
#include "util/threading.h"
#include "util/threading_context.h"
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
#include "compression/compress-inl.h"
#include "gemma/flash_attention.h"
#include "gemma/gemma-inl.h"
#include "ops/ops-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// Returns the number of floats per vector (aka NF).
size_t FloatsPerVector() {
  using DF = hn::ScalableTag<float>;
  const DF df;
  return hn::Lanes(df);
}

// The k-cache and v-cache are setup without knowing NF. So if it hasn't been
// done already, reshape it to take NF into account.
void MaybeReshapeCache(const MatPtrT<KV_t>& kv, MatPtrT<KV_t>& cache) {
  if (kv.Cols() > cache.Cols()) {
    cache.ReshapePackedRowsToCols(2 * FloatsPerVector());
  }
}

// Transposes a single row of the kv cache into the k-cache and v-cache.
void TransposeKVCacheRow(const KV_t* HWY_RESTRICT kv, KV_t* HWY_RESTRICT k,
                         KV_t* HWY_RESTRICT v, size_t qkv_dim) {
  // This is inefficient, as the writes are scattered over cache lines, but it
  // is a tiny fraction of the overall computation, and it is linear in the
  // token length.
  const size_t kFloatsPerTile = 2 * FloatsPerVector();
  for (size_t i = 0; i < qkv_dim; i += 2) {
    k[i * kFloatsPerTile] = kv[i];
    k[i * kFloatsPerTile + 1] = kv[i + 1];
  }
  for (size_t i = 0; i < qkv_dim; i += kFloatsPerTile) {
    for (size_t j = 0; j < kFloatsPerTile; j++) {
      v[i * kFloatsPerTile + j] = kv[i + j + qkv_dim];
    }
  }
}

// Computes Q.K scores, which are "logits" (or scores) stored to att.
// `k` is a strided view of the kv cache with dimensions [seq_len, qkv_dim].
static HWY_INLINE void QDotK(const size_t start_pos, const size_t last_pos,
                             const hwy::Divisor& div_seq_len,
                             const float* HWY_RESTRICT q,
                             const MatPtrT<KV_t>& k, float* HWY_RESTRICT att,
                             ThreadingContext& ctx, const size_t worker) {
  GCPP_ZONE(ctx, worker, Zones::kGenAttentionQDotK);
  const hn::ScalableTag<BF16> dbf;
  const size_t qkv_dim = k.Cols();
  HWY_ALIGN BF16 q_bf[kMaxQKVDim];

  CompressPerThread tls;
  const hn::ScalableTag<float> df;
  CompressTraits<BF16>::Compress(df, q, qkv_dim, tls, MakeSpan(q_bf, qkv_dim),
                                 0);

  // --seq_len must be large enough to avoid wraparound.
  HWY_DASSERT(last_pos < static_cast<size_t>(div_seq_len.GetDivisor()));
  for (size_t pos = start_pos; pos <= last_pos; ++pos) {
    const float score =
        Dot(dbf, MakeConstSpan(q_bf, qkv_dim), 0, k.Row(pos), qkv_dim);
    att[pos] = score;
  }
}

void PositionalEncodingQK(float* qk, const size_t layer_idx,
                          const AttentionActivationsPtrs& activations,
                          ThreadingContext& ctx, const size_t worker,
                          const size_t pos, const float mul) {
  const LayerConfig& layer_config = activations.config.layer_configs[layer_idx];
  const size_t qkv_dim = layer_config.qkv_dim;
  const PostQKType& post_qk = layer_config.post_qk;
  // qk is either q or k, so qkv_dim is the length we operate on.
  const float* inv_timescale = activations.inv_timescale.PackedScale1();
  const bool is_global_layer = activations.config.IsGlobalLayer(layer_idx);
  if (is_global_layer && activations.config.use_global_timescale) {
    inv_timescale = activations.inv_timescale_global.PackedScale1();
  }
  // PostQKType::Rope
  if (post_qk == PostQKType::HalfRope) {
    Rope(qk, qkv_dim / 2, inv_timescale, pos, ctx, worker);
    if (mul != 1.0f) MulByConst(mul, qk, qkv_dim);
  } else {
    RopeAndMulBy(mul, qk, qkv_dim, inv_timescale, pos, ctx, worker);
  }
}

// Accumulates the sum of v (from `kv_cache`) * probability (`att`) into
// `att_out`. Equivalent in gemma/modules.py:
// encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)
// `v` is a strided view of the kv cache with dimensions [seq_len, qkv_dim].
static HWY_INLINE void WeightedSumV(
    const size_t start_pos, const size_t last_pos,
    const hwy::Divisor& div_seq_len, const float* HWY_RESTRICT att,
    const MatPtrT<KV_t>& v, float* HWY_RESTRICT att_out, ThreadingContext& ctx,
    const size_t worker) {
  // --seq_len must be large enough to avoid wraparound.
  HWY_DASSERT(last_pos < static_cast<size_t>(div_seq_len.GetDivisor()));
  // TODO: replace with MatMul(att, v) after it supports non-transposed B.
  MulByConstTo(att[start_pos], v.Row(start_pos), att_out, v.Cols(), ctx,
               worker);
  for (size_t pos = start_pos + 1; pos <= last_pos; ++pos) {
    MulByConstAndAdd(att[pos], v.Row(pos), att_out, v.Cols());
  }
}

// Calculates the attention outputs for a single q, which may be updated
// in place for RMSNorm.
void SingleDotSoftmaxWeightedSum(
    const size_t q_pos, const size_t kv_start_pos, const size_t kv_last_pos,
    float* HWY_RESTRICT q, const MatPtrT<KV_t>& k, const MatPtrT<KV_t>& v,
    const MatPtr& query_norm_scale, const size_t layer_idx,
    const AttentionActivationsPtrs& activations, float* HWY_RESTRICT att,
    float* HWY_RESTRICT att_out, const SMOptions& sm_options,
    ThreadingContext& ctx, const size_t worker) {
  const float att_cap = activations.config.att_cap;
  const float query_scale = activations.query_scale;
  // --seq_len must be large enough to avoid wraparound.
  HWY_DASSERT(kv_last_pos < activations.SeqLen());
  const LayerConfig& layer_config = activations.config.layer_configs[layer_idx];

  // Apply rope and scaling to Q.
  if (query_norm_scale.HasPtr()) {
    CallUpcasted(&query_norm_scale, [&](const auto* weights_t) {
      RMSNormInplace(weights_t->PackedScale1(), /*w_ofs=*/0, q,
                     layer_config.qkv_dim, ctx, worker);
    });
  }

  PositionalEncodingQK(q, layer_idx, activations, ctx, worker, q_pos,
                       query_scale);

  QDotK(kv_start_pos, kv_last_pos, activations.div_seq_len, q, k, att, ctx,
        worker);

  // SoftMax with optional SoftCap yields "probabilities" in att.
  const Logits logits(att, kv_last_pos + 1);
  MaybeLogitsSoftCap(att_cap, logits, ctx, worker);
  Softmax(logits, ctx, worker, /*temperature=*/1.0f, sm_options);

  WeightedSumV(kv_start_pos, kv_last_pos, activations.div_seq_len, att, v,
               att_out, ctx, worker);
}

// The attention window usually starts at 0 unless `pos` is larger than
// the attention window size, then it is `pos` - window_size + 1.
size_t StartPos(size_t pos, const ModelConfig& config, size_t layer_idx) {
  const size_t att_window_size = config.attention_window_sizes[layer_idx];
  return pos - HWY_MIN(att_window_size - 1, pos);
}

void DotSoftmaxWeightedSum(const size_t num_tokens, const size_t layer_idx,
                           const MatPtr& query_norm_scale,
                           AttentionActivationsPtrs& activations,
                           QBatch& qbatch, ThreadingContext& ctx) {
  GCPP_ZONE(ctx, 0, Zones::kGenAttentionDotSoftmaxWeightedSumInclusive);

  const hwy::Divisor div_qbatch(qbatch.Size());
  const LayerConfig& layer_config = activations.config.layer_configs[layer_idx];
  const size_t qkv_dim = layer_config.qkv_dim;

  // A "head group" in the context of GQA refers to a collection of query
  // heads that share the same key and value heads.
  const size_t kHeadGroups = layer_config.heads / layer_config.kv_heads;

  const size_t cache_layer_size = layer_config.CacheLayerSize();
  const size_t seq_len = activations.SeqLen();
  // All layers should have the same number of heads.
  HWY_DASSERT(activations.div_heads.GetDivisor() == layer_config.heads);

  // For each head/token/query, compute Q.K, softmax, and weighted V.
  const auto func = [&](const size_t task, size_t worker) HWY_ATTR {
    const size_t tq_idx = activations.div_heads.Divide(task);
    const size_t head = activations.div_heads.Remainder(task);
    GCPP_ZONE(ctx, worker, Zones::kGenAttentionDotSoftmaxWeightedSumPar);

    const size_t qi = div_qbatch.Remainder(tq_idx);
    const size_t token_idx = div_qbatch.Divide(tq_idx);
    auto& kv_cache = qbatch.KV(qi).kv_cache;

    // Find the token position in the query and calculate
    // the range of cache positions to attend to.
    const size_t pos = qbatch.Pos(qi) + token_idx;
    const size_t start_pos = StartPos(pos, activations.config, layer_idx);
    size_t last_pos = pos;
    const size_t prefix_end = qbatch.PrefixEnd(qi);
    if (prefix_end > 0 && prefix_end - 1 > last_pos) {
      // last_pos in QDotK and WeightedSumV is inclusive.
      last_pos = prefix_end - 1;
    }

    float* HWY_RESTRICT q = activations.q.Row(tq_idx) + head * qkv_dim;
    float* HWY_RESTRICT att = activations.att.Row(tq_idx) + head * seq_len;
    float* HWY_RESTRICT att_out =
        activations.att_out.Row(tq_idx) + head * qkv_dim;
    SMOptions sm_options{.max_out = activations.softmax_max.Row(tq_idx) + head,
                         .d_out = activations.softmax_d.Row(tq_idx) + head};

    // Make strided read-only views into the kv cache for
    // this query and head.
    const size_t head_offset = (head / kHeadGroups) * qkv_dim * 2;
    const size_t kv_head_offset = layer_idx * cache_layer_size + head_offset;
    MatPtrT<KV_t> k("k_view", Extents2D(seq_len, qkv_dim));
    k.SetPtr(kv_cache.Row(0) + kv_head_offset, kv_cache.Stride());
    MatPtrT<KV_t> v("v_view", Extents2D(seq_len, qkv_dim));
    v.SetPtr(kv_cache.Row(0) + kv_head_offset + qkv_dim, kv_cache.Stride());

    constexpr size_t offset = 0;  // placeholder, do not remove
    SingleDotSoftmaxWeightedSum(pos + offset, start_pos, last_pos, q, k, v,
                                query_norm_scale, layer_idx, activations, att,
                                att_out, sm_options, ctx, worker);
  };

  {
    PROFILER_ZONE("Gen.Attention.DotSoftmax.ForkJoin");
    // Full parallelism is helpful, kAcrossClusters is insufficient.
    HierarchicalParallelFor(
        num_tokens * div_qbatch.GetDivisor() * layer_config.heads, ctx,
        Callers::kAttDotSoftmaxWeightedSum, func);
  }
}

// Different functions use different naming conventions for the number of
// tokens. Functions that are query-independent, such as RMSNorm*, call the
// count `num_interleaved`. Functions that are query-dependent, such as
// `Attention`, use separate `num_tokens` and `num_queries`. `num_tokens` is the
// number of tokens from one query: 1 for decode, otherwise prefill_tbatch_size.

// Fills activations.q and writes to KV cache.
static HWY_INLINE void ComputeQKV(size_t num_tokens, const size_t layer_idx,
                                  const LayerWeightsPtrs& layer,
                                  AttentionActivationsPtrs& activations,
                                  const QBatch& qbatch, const int flags,
                                  MatMulEnv& env) {
  GCPP_ZONE(env.ctx, hwy::Profiler::GlobalIdx(),
            Zones::kGenAttentionComputeQKV);

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
  MatPtrT<KV_t> kv_rows("kv", Extents2D(activations.pre_att_rms_out.Rows(),
                                        layer.qkv_einsum_w2.Rows()));
  for (size_t interleaved_idx = 0; interleaved_idx < num_interleaved;
       ++interleaved_idx) {
    // Index into qbatch, within [0, qbatch.Size()]
    const size_t qi = div_qbatch.Remainder(interleaved_idx);
    // Index along token sequence, within [0, num_tokens)
    const size_t token_idx = div_qbatch.Divide(interleaved_idx);
    const size_t cache_pos = qbatch.Pos(qi) + token_idx;
    // --seq_len must be large enough to avoid wraparound.
    HWY_DASSERT(cache_pos < activations.SeqLen());

    env.row_ptrs[0][interleaved_idx] = reinterpret_cast<uint8_t*>(
        qbatch.KV(qi).kv_cache.Row(cache_pos) + layer_idx * cache_layer_size);
  }
  kv_rows.AttachRowPtrs(env.row_ptrs[0].get());
  CallMatMul(activations.pre_att_rms_out, layer.qkv_einsum_w2,
             /*add=*/nullptr, env, kv_rows);
  for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
    MaybeReshapeCache(qbatch.KV(qi).kv_cache, qbatch.KV(qi).k_cache);
    MaybeReshapeCache(qbatch.KV(qi).kv_cache, qbatch.KV(qi).v_cache);
  }
  const size_t kFloatsPerVector = FloatsPerVector();

  // Apply positional encodings for K.
  // Note that 2D parallelism is not worth the fork/join overhead because the
  // tasks are very lightweight.
  ParallelFor(
      Parallelism::kFlat, kv_heads * num_interleaved, env.ctx,
      /*cluster_idx=*/0, Callers::kAttComputeQKV,
      [&](size_t task, size_t worker) HWY_ATTR {
        const size_t head = task % kv_heads;
        const size_t interleaved_idx = task / kv_heads;
        const size_t qi = div_qbatch.Remainder(interleaved_idx);
        const size_t token_idx = div_qbatch.Divide(interleaved_idx);
        const size_t cache_pos = qbatch.Pos(qi) + token_idx;
        // --seq_len must be large enough to avoid wraparound.
        HWY_DASSERT(cache_pos < activations.SeqLen());
        auto& kv_cache = qbatch.KV(qi).kv_cache;
        KV_t* HWY_RESTRICT kv = kv_cache.Row(cache_pos) +
                                layer_idx * cache_layer_size +
                                head * qkv_dim * 2;
        // Note that k_cache and v_cache are different shapes.
        // The innermost dimension of k is 2 values from qkv_dim because they
        // are going to be used in a BF16 dot product involving pairs of
        // values over NF k positions.
        // The innermost dimension of v is 2NF values from qkv_dim because they
        // will be loaded into a BF16 vector to be scaled and added to the
        // cached attention output in 2 NF-sized registers.
        // TODO(rays): factor out these calculations into functions.
        auto& k_cache = qbatch.KV(qi).k_cache;
        KV_t* HWY_RESTRICT k =
            k_cache.Row(cache_pos / (2 * kFloatsPerVector)) +
            (layer_idx * cache_layer_size + head * qkv_dim * 2) *
                kFloatsPerVector +
            (cache_pos % (2 * kFloatsPerVector)) * 2;
        auto& v_cache = qbatch.KV(qi).v_cache;
        KV_t* HWY_RESTRICT v =
            v_cache.Row(cache_pos / (2 * kFloatsPerVector)) +
            (layer_idx * cache_layer_size + head * qkv_dim * 2) *
                kFloatsPerVector +
            (cache_pos % (2 * kFloatsPerVector)) * 2 * kFloatsPerVector;

        HWY_ALIGN float kv_f32[2 * kMaxQKVDim];
        const hn::ScalableTag<float> df;
        DecompressAndZeroPad(df, MakeSpan(kv, 2 * qkv_dim), 0, kv_f32,
                             2 * qkv_dim);

        // Apply further processing to K.
        if (layer.key_norm_scale.HasPtr()) {
          CallUpcasted(&layer.key_norm_scale, [&](const auto* weights_t) {
            RMSNormInplace(weights_t->PackedScale1(), /*w_ofs=*/0, kv_f32,
                           qkv_dim, env.ctx, worker);
          });
        }

        constexpr size_t offset = 0;  // placeholder, do not remove
        PositionalEncodingQK(kv_f32, layer_idx, activations, env.ctx, worker,
                             cache_pos + offset,
                             /*mul=*/1.0f);
        CompressPerThread tls;
        Compress(kv_f32, 2 * qkv_dim, tls, MakeSpan(kv, 2 * qkv_dim), 0);
        // This is inefficient, as multiple threads are writing the same K
        // cache line, but the input is generated by a matmul, so it is
        // difficult to change, and it probably isn't significant.
        TransposeKVCacheRow(kv, k, v, qkv_dim);
      });
}

void GemmaAttention(size_t num_tokens, const size_t layer_idx,
                    const LayerWeightsPtrs& layer,
                    AttentionActivationsPtrs& activations, QBatch& qbatch,
                    MatMulEnv& env, AttentionImpl attention_impl, int flags) {
  GCPP_ZONE(env.ctx, hwy::Profiler::GlobalIdx(), Zones::kGenAttention);

  const LayerConfig& layer_config = layer.layer_config;
  HWY_DASSERT(!layer_config.IsMHA());  // No longer supported.
  HWY_DASSERT_M((layer_config.heads % layer_config.kv_heads) == 0,
                "query heads must be a multiple of key-value heads");
  (void)layer_config;  // only used in HWY_DASSERT

  ComputeQKV(num_tokens, layer_idx, layer, activations, qbatch, flags, env);
  if (attention_impl == AttentionImpl::kOld) {
    DotSoftmaxWeightedSum(num_tokens, layer_idx, layer.query_norm_scale,
                          activations, qbatch, env.ctx);
  } else {
    // * 2 does not help on Turin.
    FlashAttention(num_tokens,
                   /*target_parallelism=*/env.ctx.pools.MaxWorkers() *
                       AttentionActivations::kThreadReplicationFactor,
                   layer_idx, layer.query_norm_scale, activations, qbatch,
                   env.ctx, attention_impl);
  }
  SumHeads(layer, activations, env);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();
