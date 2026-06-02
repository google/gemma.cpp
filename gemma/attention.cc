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
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include "gemma/activations.h"
#include "gemma/configs.h"  // kMaxQKVDim
#include "gemma/kv_cache.h"
#include "gemma/query.h"
#include "gemma/weights.h"
#include "ops/matmul.h"
#include "util/threading.h"
#include "util/threading_context.h"
#include "util/zones.h"
#include "hwy/base.h"
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
#include "gemma/attention.h"  // includes highway.h
#include "gemma/flash_attention.h"
#include "gemma/gemma-inl.h"
#include "ops/ops-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// Transposes a single row of the kv cache into the k-cache and v-cache.
void TransposeKVCacheRow(const KV_t* HWY_RESTRICT kv, KV_t* HWY_RESTRICT k,
                         KV_t* HWY_RESTRICT v, size_t qkv_dim) {
  // This is inefficient, as the writes are scattered over cache lines, but it
  // is a tiny fraction of the overall computation, and it is linear in the
  // token length.
  const size_t kFloatsPerTile = 2 * FloatsPerVector();
  const size_t kRoundedQkvDim = hwy::RoundUpTo(qkv_dim, kMaxBF16PerVector);
  for (size_t i = 0; i < qkv_dim; i += 2) {
    k[i * kFloatsPerTile] = kv[i];
    k[i * kFloatsPerTile + 1] = kv[i + 1];
  }
  for (size_t i = qkv_dim; i < kRoundedQkvDim; i += 2) {
    k[i * kFloatsPerTile] = hwy::ConvertScalarTo<KV_t>(0.0f);
    k[i * kFloatsPerTile + 1] = hwy::ConvertScalarTo<KV_t>(0.0f);
  }
  for (size_t i = 0; i < qkv_dim; i += kFloatsPerTile) {
    if (i + kFloatsPerTile <= qkv_dim) {
      for (size_t j = 0; j < kFloatsPerTile; j++) {
        v[i * kFloatsPerTile + j] = kv[i + j + qkv_dim];
      }
    } else {
      for (size_t j = 0; j < qkv_dim - i; j++) {
        v[i * kFloatsPerTile + j] = kv[i + j + qkv_dim];
      }
      for (size_t j = qkv_dim - i; j < kFloatsPerTile; j++) {
        v[i * kFloatsPerTile + j] = hwy::ConvertScalarTo<KV_t>(0.0f);
      }
    }
  }
  for (size_t i = hwy::RoundUpTo(qkv_dim, kFloatsPerTile); i < kRoundedQkvDim;
       i += kFloatsPerTile) {
    for (size_t j = 0; j < kFloatsPerTile; j++) {
      v[i * kFloatsPerTile + j] = hwy::ConvertScalarTo<KV_t>(0.0f);
    }
  }
}

// Zeros out a part of k and v that corresponds to out-of-bounds cache
// positions.
void TransposeOOBKVCacheRow(KV_t* HWY_RESTRICT k, KV_t* HWY_RESTRICT v,
                            size_t qkv_dim) {
  const size_t kFloatsPerTile = 2 * FloatsPerVector();
  const size_t kRoundedQkvDim = hwy::RoundUpTo(qkv_dim, kMaxBF16PerVector);
  for (size_t i = 0; i < kRoundedQkvDim; i += 2) {
    k[i * kFloatsPerTile] = hwy::ConvertScalarTo<KV_t>(0.0f);
    k[i * kFloatsPerTile + 1] = hwy::ConvertScalarTo<KV_t>(0.0f);
  }
  for (size_t i = 0; i < kRoundedQkvDim; i += kFloatsPerTile) {
    for (size_t j = 0; j < kFloatsPerTile; j++) {
      v[i * kFloatsPerTile + j] = hwy::ConvertScalarTo<KV_t>(0.0f);
    }
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
    MaybeReshapeCache(qbatch.KV(qi).cache->KOrVDefaultCols(),
                      qbatch.KV(qi).k_cache);
    MaybeReshapeCache(qbatch.KV(qi).cache->KOrVDefaultCols(),
                      qbatch.KV(qi).v_cache);
  }
  const size_t kFloatsPerVector = FloatsPerVector();
  const size_t kRoundedTokens =
      hwy::RoundUpTo(num_tokens, 2 * kFloatsPerVector);
  const size_t kRoundedNumInterleaved =
      kRoundedTokens * div_qbatch.GetDivisor();

  // Apply positional encodings for K.
  // Note that 2D parallelism is not worth the fork/join overhead because the
  // tasks are very lightweight.
  ParallelFor(
      Parallelism::kFlat, kv_heads * kRoundedNumInterleaved, env.ctx,
      /*cluster_idx=*/0, Callers::kAttComputeQKV,
      [&](size_t task, size_t worker) HWY_ATTR {
        const size_t head = task % kv_heads;
        const size_t interleaved_idx = task / kv_heads;
        const size_t qi = div_qbatch.Remainder(interleaved_idx);
        const size_t token_idx = div_qbatch.Divide(interleaved_idx);
        const size_t cache_pos = qbatch.Pos(qi) + token_idx;
        if (token_idx >= kRoundedTokens) {
          return;
        }
        // The innermost dimension of v is 2NF values from qkv_dim because they
        // will be loaded into a BF16 vector to be scaled and added to the
        // cached attention output in 2 NF-sized registers.
        auto& k_cache = qbatch.KV(qi).k_cache;
        KV_t* HWY_RESTRICT k =
            k_cache.Row(cache_pos / (2 * kFloatsPerVector)) +
            qbatch.KV(qi).cache->KOffset(layer_idx, head, kFloatsPerVector,
                                         cache_pos);
        auto& v_cache = qbatch.KV(qi).v_cache;
        KV_t* HWY_RESTRICT v =
            v_cache.Row(cache_pos / (2 * kFloatsPerVector)) +
            qbatch.KV(qi).cache->VOffset(layer_idx, head, kFloatsPerVector,
                                         cache_pos);
        if (token_idx >= num_tokens) {
          // Create a zero-filled K/V pair for padding for out-of-sequence
          // tokens.
          TransposeOOBKVCacheRow(k, v, qkv_dim);
          return;
        }
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
  FlashAttention(num_tokens,
                 /*target_parallelism=*/env.ctx.pools.MaxWorkers() *
                     AttentionActivations::kThreadReplicationFactor,
                 layer_idx, layer.query_norm_scale, activations, qbatch,
                 env.ctx, attention_impl);
  SumHeads(layer, activations, env);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();
