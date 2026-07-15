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

// Defines Gemma member functions which dynamic-dispatch into the SIMD
// implementations in gemma-inl.h.

#include "gemma/gemma.h"

#include <cmath>
#include <limits>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include "gemma/tensor_stats.h"
#include "util/zones.h"

// Compiles this file for multiple architectures via "foreach_target.h", to
// which we pass the filename via macro 'argument'.
// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/gemma.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "deepseek/deepseek.h"  // includes highway.h
#include "gemma/attention.h"  // includes highway.h
#include "gemma/gemma-inl.h"
#include "gemma/gemma4_moe.h"  // includes highway.h
#include "gemma/generate_internal.h"
#include "gemma/tiled_attention.h"  // includes highway.h
#include "gemma/vit.h"              // includes highway.h

#ifndef GEMMA_CC_ONCE
#define GEMMA_CC_ONCE

#include <math.h>  // sqrtf
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

#include "gemma/configs.h"
#include "gemma/model_store.h"
#include "gemma/tokenizer.h"
#include "gemma/weights.h"
#include "io/blob_store.h"
#include "io/io.h"  // Path
#include "ops/matmul.h"
#include "paligemma/image.h"
#include "util/basics.h"
#include "util/threading_context.h"
#include "hwy/aligned_allocator.h"  // Span
#include "hwy/base.h"
#include "hwy/timer.h"

// Require opt-in to debug/introspection functions to eliminate their overhead.
HWY_INLINE_VAR constexpr bool kObserver = false;

#endif  // GEMMA_CC_ONCE

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

void Attention(LayerAttentionType type, const size_t num_tokens,
               const size_t layer_idx, const LayerWeightsPtrs& layer,
               Activations& activations, QBatch& qbatch, MatMulEnv& env) {
  const int kFlags = 0;
  if (activations.attention_impl == AttentionImpl::kFlashTransposedQs ||
      activations.attention_impl == AttentionImpl::kFlashTransposedQsBF16 ||
      activations.attention_impl == AttentionImpl::kFlashTransposedQsInt16) {
    TiledAttention(activations.attention_impl, num_tokens, layer_idx, layer,
                   activations.attention, qbatch, env, kFlags);
    return;
  }

  if (type == LayerAttentionType::kGemma) {
    GemmaAttention(num_tokens, layer_idx, layer, activations.attention, qbatch,
                   env, activations.attention_impl, kFlags);
  }
}

HWY_NOINLINE void TransformerLayer(const size_t num_tokens,
                                   const size_t layer_idx,
                                   const LayerWeightsPtrs& layer,
                                   Activations& activations, QBatch& qbatch,
                                   MatMulEnv& env) {
  const LayerConfig& layer_config = layer.layer_config;
  if (layer_config.type == LayerAttentionType::kDeepSeekMLA) {
    DeepSeekTransformerLayer(num_tokens, layer_idx, layer, activations, qbatch,
                             env);
    return;
  }
  if (layer_config.IsMoE() &&
      activations.attention.config.model == Model::GEMMA4_26B_MOE) {
    Gemma4MoETransformerLayer(num_tokens, layer_idx, layer, activations, qbatch,
                              env);
    return;
  }

  RMSNormBatched(activations.x, layer.pre_attention_norm_scale,
                 activations.attention.pre_att_rms_out, env.ctx);

  Attention(layer_config.type, num_tokens, layer_idx, layer, activations,
            qbatch, env);

  PostNorm(layer_config.post_norm, layer.post_attention_norm_scale,
           activations.attention.att_sums, env.ctx);

  ResidualConnection(activations.attention.att_sums, activations.x, layer,
                     /*is_attention=*/true, env.ctx);

  RMSNormBatched(activations.x, layer.pre_ffw_norm_scale,
                 activations.pre_ffw_rms_out, env.ctx);

  if (layer_config.type == LayerAttentionType::kVit) {
    FFWVit(layer, activations, env);
  } else {
    FFWNoVit(layer, activations, env);
  }

  PostNorm(layer_config.post_norm, layer.post_ffw_norm_scale,
           activations.ffw_out, env.ctx);

  ResidualConnection(activations.ffw_out, activations.x, layer,
                     /*is_attention=*/false, env.ctx);
  if (layer_config.ple_dim > 0) {
    // 1. Gate: [batch, model_dim] @ [model_dim, ple_dim] -> [batch, ple_dim]
    // Use activations.x_bf to convert activations.x
    for (size_t r = 0; r < num_tokens; ++r) {
      for (size_t c = 0; c < layer_config.model_dim; ++c) {
        activations.x_bf.Row(r)[c] = BF16(activations.x.Row(r)[c]);
      }
    }

    // Use pre-allocated activations.gate_out (BF16)
    CallMatMul(activations.x_bf, layer.ple_gate, /*add=*/nullptr, env,
               activations.gate_out);

    // 2. Activation and Element-wise multiply
    const size_t ple_dim = layer_config.ple_dim;
    const size_t layer_offset = layer_idx * ple_dim;

    ParallelFor(Parallelism::kFlat, num_tokens, env.ctx, /*cluster_idx=*/0,
                Callers::kActivationBatched,
                [&](uint64_t token_idx, size_t worker) HWY_ATTR {
                  BF16* g_row = activations.gate_out.Row(token_idx);
                  const float* p_row =
                      activations.ple_embeds.Row(token_idx) + layer_offset;
                  namespace hn = hwy::HWY_NAMESPACE;
                  using DF = hn::ScalableTag<float>;
                  const DF df;
                  Decompress1AndCompressInplace(
                      df, g_row, ple_dim, p_row, 0,
                      [](auto df, auto v_gate, auto v_embed) HWY_ATTR {
                        return hn::Mul(Gelu(df, v_gate), v_embed);
                      });
                });

    // 3. Projection: [batch, ple_dim] @ [ple_dim, model_dim] -> [batch,
    // model_dim]
    CallMatMul(activations.gate_out, layer.ple_proj, /*add=*/nullptr, env,
               activations.ffw_out);

    // 4. Norm and Residual Add
    RMSNormInplaceBatched(layer.post_ple_ns, activations.ffw_out, env.ctx);

    ParallelFor(Parallelism::kFlat, num_tokens, env.ctx, /*cluster_idx=*/0,
                Callers::kOpsAddFromBatched,
                [&](uint64_t token_idx, size_t worker) {
                  AddFrom(activations.ffw_out.Row(token_idx),
                          activations.x.Row(token_idx), layer_config.model_dim,
                          env.ctx, worker);
                });
  }

  if (layer.skip_scale.HasPtr()) {
    float skip_scale_val = 1.0f;
    if (layer.skip_scale.GetType() == Type::kF32) {
      skip_scale_val = static_cast<const float*>(layer.skip_scale.Packed())[0];
    } else if (layer.skip_scale.GetType() == Type::kBF16) {
      skip_scale_val = hwy::ConvertScalarTo<float>(
          static_cast<const BF16*>(layer.skip_scale.Packed())[0]);
    } else {
      HWY_ABORT("Unexpected skip_scale type: %d",
                static_cast<int>(layer.skip_scale.GetType()));
    }
    for (size_t r = 0; r < activations.x.Rows(); ++r) {
      MulByConst(skip_scale_val, activations.x.Row(r), activations.x.Cols());
    }
  }
}

// Returns the scale value to use for the embedding (basically sqrt model_dim).
static float EmbeddingScaling(size_t model_dim) {
  // Round to bf16 to match Gemma's Embedder, which casts before mul.
  return hwy::ConvertScalarTo<float>(
      hwy::ConvertScalarTo<BF16>(sqrtf(static_cast<float>(model_dim))));
}

static HWY_INLINE void EmbedTokenFromWeights(int token, size_t x_row,
                                             const ModelConfig& model_config,
                                             const MatPtr& embedding,
                                             MatStorageT<float>& x) {
  const size_t model_dim = model_config.model_dim;
  // DeepSeek does not scale embeddings by sqrt(model_dim).
  const float emb_scaling =
      model_config.HasMLA() ? 1.0f : EmbeddingScaling(model_dim);

  HWY_DASSERT(token >= 0);
  HWY_DASSERT(token < static_cast<int>(model_config.vocab_size));

  CallUpcasted(&embedding, [&](const auto* weights_t) {
    // Using `Stride` to compute the offset works for both NUQ (because we use
    // an offset and NUQ is never padded) and padded, because non-NUQ types are
    // seekable, hence the offset can also skip any padding.
    const size_t embedding_ofs = token * weights_t->Stride();
    HWY_ASSERT(weights_t->Cols() == model_dim);
    const auto embedding_span =
        MakeSpan(weights_t->Row(0), embedding_ofs + model_dim);
    DecompressAndZeroPad(hn::ScalableTag<float>(), embedding_span,
                         embedding_ofs, x.Row(x_row), model_dim);
    MulByConst(emb_scaling * weights_t->Scale(), x.Row(x_row), model_dim);
  });
}

static void StreamAndUpdateEOS(size_t qi, size_t pos, int token,
                               float prob, const ModelConfig& config,
                               const RuntimeConfig& runtime_config,
                               QBatch& qbatch, bool update_pos,
                               hwy::BitSet4096<>& non_eos);

static HWY_INLINE SampleFunc ChooseSampleFunc(const RuntimeConfig& runtime_config,
                                              const AesCtrEngine& engine,
                                              ThreadingContext& ctx);

static void ValidateT5GemmaFreshGeneration(size_t pos, size_t prefix_end) {
  if (pos != 0 || prefix_end != 0) {
    HWY_ABORT(
        "T5Gemma currently supports fresh seq2seq generation only; got pos=%zu "
        "and prefix_end=%zu.",
        pos, prefix_end);
  }
}

static void InitT5GemmaEncoderCache(const ModelConfig& config,
                                    const PromptTokens& prompt,
                                    T5GemmaEncoderCache& cache,
                                    const Allocator& allocator) {
  HWY_ASSERT(prompt.size() != 0);
  cache.source_len = prompt.size();
  cache.hidden_states =
      MatStorageT<float>("t5_enc", Extents2D(cache.source_len, config.model_dim),
                         allocator, MatPadding::kOdd);
  cache.cross_keys.resize(config.decoder_layer_configs.size());
  cache.cross_values.resize(config.decoder_layer_configs.size());
  for (size_t layer_idx = 0; layer_idx < config.decoder_layer_configs.size();
       ++layer_idx) {
    const LayerConfig& layer_config = config.decoder_layer_configs[layer_idx];
    const Extents2D extents(cache.source_len,
                            layer_config.kv_heads * layer_config.qkv_dim);
    cache.cross_keys[layer_idx] =
        MatStorageT<float>("t5_cross_k", extents, allocator, MatPadding::kOdd);
    cache.cross_values[layer_idx] =
        MatStorageT<float>("t5_cross_v", extents, allocator, MatPadding::kOdd);
  }
  cache.pad_mask.resize(cache.source_len);
  for (size_t i = 0; i < cache.source_len; ++i) {
    cache.pad_mask[i] = prompt[i] == T5GEMMA_PAD_ID ? 1 : 0;
  }
}

static void AttachT5GemmaEncoderCaches(
    const ModelConfig& config, AllQueries& all_queries,
    std::vector<T5GemmaEncoderCache>& encoder_caches,
    const Allocator& allocator) {
  encoder_caches.resize(all_queries.NumQueries());
  for (size_t qi = 0; qi < all_queries.NumQueries(); ++qi) {
    ValidateT5GemmaFreshGeneration(all_queries[qi].initial_pos,
                                   all_queries[qi].prefix_end);
    InitT5GemmaEncoderCache(config, all_queries[qi].prompt, encoder_caches[qi],
                            allocator);
    all_queries[qi].t5gemma_encoder_cache = &encoder_caches[qi];
    all_queries[qi].mutable_pos = 0;
    all_queries[qi].prev_token = BOS_ID;
  }
}

static HWY_INLINE size_t T5GemmaEncoderWindowSize(const ModelConfig& config,
                                                  size_t layer_idx) {
  if (layer_idx < config.encoder_attention_window_sizes.size()) {
    return config.encoder_attention_window_sizes[layer_idx];
  }
  return config.max_seq_len;
}

static HWY_INLINE bool T5GemmaEncoderCanAttend(const ModelConfig& config,
                                               const T5GemmaEncoderCache& cache,
                                               size_t layer_idx,
                                               size_t query_pos,
                                               size_t key_pos) {
  if (cache.pad_mask[key_pos]) return false;
  const size_t window_size = T5GemmaEncoderWindowSize(config, layer_idx);
  if (window_size >= config.max_seq_len) return true;
  const size_t distance = query_pos > key_pos ? query_pos - key_pos
                                              : key_pos - query_pos;
  return distance <= window_size;
}

static void T5GemmaApplyRope(const LayerConfig& layer_config,
                             const MatPtrT<float>& inv_timescale, float scale,
                             MatStorageT<float>& q,
                             MatStorageT<float>& kv,
                             ThreadingContext& ctx) {
  const size_t source_len = q.Rows();
  const size_t qkv_dim = layer_config.qkv_dim;
  for (size_t pos = 0; pos < source_len; ++pos) {
    for (size_t head = 0; head < layer_config.heads; ++head) {
      float* q_row = q.Row(pos) + head * qkv_dim;
      if (layer_config.post_qk == PostQKType::HalfRope) {
        Rope(q_row, qkv_dim / 2, inv_timescale.PackedScale1(), pos, ctx,
             /*worker=*/0);
        if (scale != 1.0f) MulByConst(scale, q_row, qkv_dim);
      } else {
        RopeAndMulBy(scale, q_row, qkv_dim, inv_timescale.PackedScale1(), pos,
                     ctx, /*worker=*/0);
      }
    }
    for (size_t kv_head = 0; kv_head < layer_config.kv_heads; ++kv_head) {
      float* k_row = kv.Row(pos) + kv_head * 2 * qkv_dim;
      if (layer_config.post_qk == PostQKType::HalfRope) {
        Rope(k_row, qkv_dim / 2, inv_timescale.PackedScale1(), pos, ctx,
             /*worker=*/0);
      } else {
        RopeAndMulBy(/*mul=*/1.0f, k_row, qkv_dim,
                     inv_timescale.PackedScale1(), pos, ctx, /*worker=*/0);
      }
    }
  }
}

static void T5GemmaEncoderAttentionReference(
    const ModelConfig& config, size_t layer_idx,
    const T5GemmaEncoderLayerWeightsPtrs& layer,
    T5GemmaEncoderCache& encoder_cache, MatStorageT<BF16>& pre_att_rms_out,
    MatStorageT<float>& q, MatStorageT<float>& kv,
    MatStorageT<float>& att_out, MatStorageT<float>& att_sums,
    const MatPtrT<float>& inv_timescale, MatMulEnv& env) {
  const LayerConfig& layer_config = layer.layer_config;
  const size_t source_len = encoder_cache.source_len;
  const size_t qkv_dim = layer_config.qkv_dim;
  const size_t heads = layer_config.heads;
  const size_t kv_heads = layer_config.kv_heads;
  const size_t heads_per_kv = heads / kv_heads;

  CallMatMul(pre_att_rms_out, layer.qkv_einsum_w1, /*add=*/nullptr, env, q);
  CallMatMul(pre_att_rms_out, layer.qkv_einsum_w2, /*add=*/nullptr, env, kv);
  T5GemmaApplyRope(layer_config, inv_timescale,
                   /*scale=*/1.0f / sqrtf(static_cast<float>(qkv_dim)), q, kv,
                   env.ctx);

  for (size_t query_pos = 0; query_pos < source_len; ++query_pos) {
    for (size_t head = 0; head < heads; ++head) {
      const size_t kv_head = head / heads_per_kv;
      const float* query = q.Row(query_pos) + head * qkv_dim;

      float max_score = -std::numeric_limits<float>::infinity();
      for (size_t key_pos = 0; key_pos < source_len; ++key_pos) {
        if (!T5GemmaEncoderCanAttend(config, encoder_cache, layer_idx,
                                     query_pos, key_pos)) {
          continue;
        }
        const float* key = kv.Row(key_pos) + kv_head * 2 * qkv_dim;
        float score = 0.0f;
        for (size_t dim = 0; dim < qkv_dim; ++dim) {
          score += query[dim] * key[dim];
        }
        score = MaybeLogitsSoftCap(config.att_cap, score);
        max_score = std::max(max_score, score);
      }

      float denom = 0.0f;
      float* out = att_out.Row(query_pos) + head * qkv_dim;
      std::fill(out, out + qkv_dim, 0.0f);
      if (max_score == -std::numeric_limits<float>::infinity()) continue;

      for (size_t key_pos = 0; key_pos < source_len; ++key_pos) {
        if (!T5GemmaEncoderCanAttend(config, encoder_cache, layer_idx,
                                     query_pos, key_pos)) {
          continue;
        }
        const float* key = kv.Row(key_pos) + kv_head * 2 * qkv_dim;
        const float* value = key + qkv_dim;
        float score = 0.0f;
        for (size_t dim = 0; dim < qkv_dim; ++dim) {
          score += query[dim] * key[dim];
        }
        score = MaybeLogitsSoftCap(config.att_cap, score);
        const float weight = expf(score - max_score);
        denom += weight;
        for (size_t dim = 0; dim < qkv_dim; ++dim) {
          out[dim] += weight * value[dim];
        }
      }
      const float inv_denom = denom == 0.0f ? 0.0f : 1.0f / denom;
      for (size_t dim = 0; dim < qkv_dim; ++dim) {
        out[dim] *= inv_denom;
      }
    }
  }

  CallMatMul(att_out, layer.att_weights, /*add=*/nullptr, env, att_sums);
}

static void T5GemmaEncoderFFW(const T5GemmaEncoderLayerWeightsPtrs& layer,
                              MatStorageT<float>& hidden_states,
                              MatStorageT<BF16>& pre_ffw_rms_out,
                              MatStorageT<BF16>& c1, MatStorageT<BF16>& c2,
                              MatStorageT<float>& ffw_out, MatMulEnv& env) {
  RMSNormBatched(hidden_states, layer.pre_ffw_norm_scale, pre_ffw_rms_out,
                 env.ctx);

#if GEMMA_FUSED_FFN
  const LayerConfig& layer_config = layer.layer_config;
  const auto fused = [&](RowPtrsBF C1, IndexRange range_r, IndexRange range_c,
                         StridedViewBF C2, size_t worker) {
    Activation(layer_config.activation, C1, range_r, range_c, C2, env.ctx,
               worker);
  };
  MMOptions options;
  options.SetFunc(fused);
  CallTwoMatMul(pre_ffw_rms_out, layer.gating_einsum_w1,
                layer.gating_einsum_w2, env, c1, options);
#else
  CallMatMul(pre_ffw_rms_out, layer.gating_einsum_w1, /*add=*/nullptr, env, c1);
  CallMatMul(pre_ffw_rms_out, layer.gating_einsum_w2, /*add=*/nullptr, env, c2);
  ActivationBatched(layer.layer_config.activation, c1, &c2, env.ctx);
#endif

  CallMatMul(c1, layer.linear_w, /*add=*/nullptr, env, ffw_out);
  RMSNormInplaceBatched(layer.post_ffw_norm_scale, ffw_out, env.ctx);
  AddFromBatched(ffw_out, hidden_states, env.ctx);
}

static void T5GemmaEncoderLayerReference(
    const ModelConfig& config, size_t layer_idx,
    const T5GemmaEncoderLayerWeightsPtrs& layer,
    T5GemmaEncoderCache& encoder_cache, MatStorageT<BF16>& pre_att_rms_out,
    MatStorageT<float>& q, MatStorageT<float>& kv,
    MatStorageT<float>& att_out, MatStorageT<float>& att_sums,
    MatStorageT<BF16>& pre_ffw_rms_out, MatStorageT<BF16>& c1,
    MatStorageT<BF16>& c2, MatStorageT<float>& ffw_out,
    const MatPtrT<float>& inv_timescale, MatMulEnv& env) {
  RMSNormBatched(encoder_cache.hidden_states, layer.pre_attention_norm_scale,
                 pre_att_rms_out, env.ctx);
  T5GemmaEncoderAttentionReference(config, layer_idx, layer, encoder_cache,
                                   pre_att_rms_out, q, kv, att_out, att_sums,
                                   inv_timescale, env);
  RMSNormInplaceBatched(layer.post_attention_norm_scale, att_sums, env.ctx);
  AddFromBatched(att_sums, encoder_cache.hidden_states, env.ctx);

  T5GemmaEncoderFFW(layer, encoder_cache.hidden_states, pre_ffw_rms_out, c1, c2,
                    ffw_out, env);
}

static void T5GemmaEncode(const ModelConfig& config, const WeightsPtrs& weights,
                          const PromptTokens& prompt,
                          T5GemmaEncoderCache& encoder_cache,
                          Activations& activations, MatMulEnv& env) {
  GCPP_ZONE(env.ctx, hwy::Profiler::GlobalIdx(), Zones::kGenEmbed);
  HWY_ASSERT(config.is_encoder_decoder);
  HWY_ASSERT(encoder_cache.source_len == prompt.size());
  HWY_ASSERT(encoder_cache.hidden_states.Rows() == prompt.size());
  HWY_ASSERT(encoder_cache.hidden_states.Cols() == config.model_dim);
  HWY_ASSERT(encoder_cache.pad_mask.size() == prompt.size());

  for (size_t pos = 0; pos < prompt.size(); ++pos) {
    EmbedTokenFromWeights(prompt[pos], pos, config,
                          weights.t5gemma_encoder_embedding,
                          encoder_cache.hidden_states);
    HWY_DASSERT(encoder_cache.pad_mask[pos] ==
                (prompt[pos] == T5GEMMA_PAD_ID ? 1 : 0));
  }

  const size_t source_len = encoder_cache.source_len;
  activations.SetT5EncoderSourceLen(source_len);

  for (size_t layer_idx = 0; layer_idx < weights.t5gemma_encoder_layers.size();
       ++layer_idx) {
    T5GemmaEncoderLayerReference(
        config, layer_idx, weights.t5gemma_encoder_layers[layer_idx],
        encoder_cache, activations.t5_encoder_pre_att_rms_out,
        activations.t5_encoder_q, activations.t5_encoder_kv,
        activations.t5_encoder_att_out, activations.t5_encoder_att_sums,
        activations.t5_encoder_pre_ffw_rms_out, activations.t5_encoder_c1,
        activations.t5_encoder_c2, activations.t5_encoder_ffw_out,
        activations.t5_encoder_inv_timescale, env);
  }
  RMSNormInplaceBatched(weights.t5gemma_encoder_final_norm_scale,
                        encoder_cache.hidden_states, env.ctx);
}

static void T5GemmaPrecomputeCrossAttentionKV(
    const WeightsPtrs& weights, T5GemmaEncoderCache& encoder_cache,
    MatMulEnv& env) {
  HWY_ASSERT(encoder_cache.cross_keys.size() ==
             weights.t5gemma_decoder_layers.size());
  HWY_ASSERT(encoder_cache.cross_values.size() ==
             weights.t5gemma_decoder_layers.size());
  for (size_t layer_idx = 0; layer_idx < weights.t5gemma_decoder_layers.size();
       ++layer_idx) {
    const T5GemmaDecoderLayerWeightsPtrs& layer =
        weights.t5gemma_decoder_layers[layer_idx];
    MatStorageT<float>& cross_k = encoder_cache.cross_keys[layer_idx];
    MatStorageT<float>& cross_v = encoder_cache.cross_values[layer_idx];
    cross_k.AllocateAndAttachRowPtrs(env.row_ptrs);
    cross_v.AllocateAndAttachRowPtrs(env.row_ptrs);
    CallMatMul(encoder_cache.hidden_states, layer.cross_k_einsum_w,
               /*add=*/nullptr, env, cross_k);
    CallMatMul(encoder_cache.hidden_states, layer.cross_v_einsum_w,
               /*add=*/nullptr, env, cross_v);
  }
}

static void T5GemmaEncodeAllQueries(const ModelConfig& config,
                                    const WeightsPtrs& weights,
                                    AllQueries& all_queries,
                                    Activations& activations, MatMulEnv& env) {
  for (size_t qi = 0; qi < all_queries.NumQueries(); ++qi) {
    T5GemmaEncoderCache* encoder_cache =
        all_queries[qi].t5gemma_encoder_cache;
    HWY_DASSERT(encoder_cache != nullptr);
    T5GemmaEncode(config, weights, all_queries[qi].prompt, *encoder_cache,
                  activations, env);
    T5GemmaPrecomputeCrossAttentionKV(weights, *encoder_cache, env);
  }
}

static size_t T5GemmaPromptTokenCount(const AllQueries& all_queries) {
  size_t tokens = 0;
  for (size_t qi = 0; qi < all_queries.NumQueries(); ++qi) {
    tokens += all_queries[qi].prompt.size();
  }
  return tokens;
}

static void T5GemmaEmbedDecoderTokens(const ModelConfig& config,
                                      const WeightsPtrs& weights,
                                      Activations& activations,
                                      QBatch& qbatch,
                                      MatMulEnv& env) {
  activations.SetBatchSize(qbatch.Size());
  for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
    EmbedTokenFromWeights(qbatch.PrevToken(qi), qi, config,
                          weights.t5gemma_decoder_embedding, activations.x);
  }
}

static HWY_INLINE size_t T5GemmaDecoderWindowSize(const ModelConfig& config,
                                                  size_t layer_idx) {
  if (layer_idx < config.decoder_attention_window_sizes.size()) {
    return config.decoder_attention_window_sizes[layer_idx];
  }
  return config.max_seq_len;
}

static HWY_INLINE size_t T5GemmaDecoderStartPos(const ModelConfig& config,
                                                size_t layer_idx,
                                                size_t query_pos) {
  const size_t window_size = T5GemmaDecoderWindowSize(config, layer_idx);
  if (window_size >= config.max_seq_len || query_pos < window_size) return 0;
  return query_pos + 1 - window_size;
}

static void T5GemmaApplyDecoderRope(const LayerConfig& layer_config,
                                    const MatPtrT<float>& inv_timescale,
                                    float scale, MatPtrT<float>& q,
                                    MatStorageT<float>& kv, QBatch& qbatch,
                                    ThreadingContext& ctx) {
  const size_t qkv_dim = layer_config.qkv_dim;
  for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
    const size_t pos = qbatch.Pos(qi);
    for (size_t head = 0; head < layer_config.heads; ++head) {
      float* q_row = q.Row(qi) + head * qkv_dim;
      if (layer_config.post_qk == PostQKType::HalfRope) {
        Rope(q_row, qkv_dim / 2, inv_timescale.PackedScale1(), pos, ctx,
             /*worker=*/0);
        if (scale != 1.0f) MulByConst(scale, q_row, qkv_dim);
      } else {
        RopeAndMulBy(scale, q_row, qkv_dim, inv_timescale.PackedScale1(), pos,
                     ctx, /*worker=*/0);
      }
    }
    for (size_t kv_head = 0; kv_head < layer_config.kv_heads; ++kv_head) {
      float* k_row = kv.Row(qi) + kv_head * 2 * qkv_dim;
      if (layer_config.post_qk == PostQKType::HalfRope) {
        Rope(k_row, qkv_dim / 2, inv_timescale.PackedScale1(), pos, ctx,
             /*worker=*/0);
      } else {
        RopeAndMulBy(/*mul=*/1.0f, k_row, qkv_dim,
                     inv_timescale.PackedScale1(), pos, ctx, /*worker=*/0);
      }
    }
  }
}

static void T5GemmaWriteDecoderKV(const T5GemmaDecoderLayerWeightsPtrs& layer,
                                  size_t layer_idx, const MatStorageT<float>& kv,
                                  QBatch& qbatch) {
  const LayerConfig& layer_config = layer.layer_config;
  const size_t cache_layer_size = layer_config.CacheLayerSize();
  for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
    if (qbatch.KV(qi).IsTiled()) {
      HWY_ABORT(
          "T5Gemma reference decoder self-attention currently requires the "
          "plain KV cache; tiled/transposed KV cache support is not wired yet.");
    }
    const size_t pos = qbatch.Pos(qi);
    HWY_ASSERT(pos < qbatch.KV(qi).SeqLen());
    KV_t* dst =
        qbatch.KV(qi).kv_cache.Row(pos) + layer_idx * cache_layer_size;
    const float* src = kv.Row(qi);
    for (size_t i = 0; i < cache_layer_size; ++i) {
      dst[i] = hwy::ConvertScalarTo<KV_t>(src[i]);
    }
  }
}

static void T5GemmaDecoderSelfAttentionReference(
    const ModelConfig& config, size_t layer_idx,
    const T5GemmaDecoderLayerWeightsPtrs& layer, Activations& activations,
    MatStorageT<float>& kv, const MatPtrT<float>& inv_timescale,
    QBatch& qbatch, MatMulEnv& env) {
  const LayerConfig& layer_config = layer.layer_config;
  const size_t qkv_dim = layer_config.qkv_dim;
  const size_t heads = layer_config.heads;
  const size_t kv_heads = layer_config.kv_heads;
  const size_t heads_per_kv = heads / kv_heads;
  const size_t cache_layer_size = layer_config.CacheLayerSize();

  RMSNormBatched(activations.x, layer.pre_self_attention_norm_scale,
                 activations.attention.pre_att_rms_out, env.ctx);
  CallMatMul(activations.attention.pre_att_rms_out, layer.self_qkv_einsum_w1,
             /*add=*/nullptr, env, activations.attention.q);
  CallMatMul(activations.attention.pre_att_rms_out, layer.self_qkv_einsum_w2,
             /*add=*/nullptr, env, kv);
  T5GemmaApplyDecoderRope(layer_config, inv_timescale,
                          /*scale=*/1.0f / sqrtf(static_cast<float>(qkv_dim)),
                          activations.attention.q, kv, qbatch, env.ctx);
  T5GemmaWriteDecoderKV(layer, layer_idx, kv, qbatch);

  for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
    const size_t query_pos = qbatch.Pos(qi);
    const size_t start_pos =
        T5GemmaDecoderStartPos(config, layer_idx, query_pos);
    for (size_t head = 0; head < heads; ++head) {
      const size_t kv_head = head / heads_per_kv;
      const float* query = activations.attention.q.Row(qi) + head * qkv_dim;

      float max_score = -std::numeric_limits<float>::infinity();
      for (size_t key_pos = start_pos; key_pos <= query_pos; ++key_pos) {
        const KV_t* key =
            qbatch.KV(qi).kv_cache.Row(key_pos) + layer_idx * cache_layer_size +
            kv_head * 2 * qkv_dim;
        float score = 0.0f;
        for (size_t dim = 0; dim < qkv_dim; ++dim) {
          score += query[dim] * hwy::ConvertScalarTo<float>(key[dim]);
        }
        score = MaybeLogitsSoftCap(config.att_cap, score);
        max_score = std::max(max_score, score);
      }

      float denom = 0.0f;
      float* out = activations.attention.att_out.Row(qi) + head * qkv_dim;
      std::fill(out, out + qkv_dim, 0.0f);
      for (size_t key_pos = start_pos; key_pos <= query_pos; ++key_pos) {
        const KV_t* key =
            qbatch.KV(qi).kv_cache.Row(key_pos) + layer_idx * cache_layer_size +
            kv_head * 2 * qkv_dim;
        const KV_t* value = key + qkv_dim;
        float score = 0.0f;
        for (size_t dim = 0; dim < qkv_dim; ++dim) {
          score += query[dim] * hwy::ConvertScalarTo<float>(key[dim]);
        }
        score = MaybeLogitsSoftCap(config.att_cap, score);
        const float weight = expf(score - max_score);
        denom += weight;
        for (size_t dim = 0; dim < qkv_dim; ++dim) {
          out[dim] += weight * hwy::ConvertScalarTo<float>(value[dim]);
        }
      }
      const float inv_denom = denom == 0.0f ? 0.0f : 1.0f / denom;
      for (size_t dim = 0; dim < qkv_dim; ++dim) out[dim] *= inv_denom;
    }
  }

  CallMatMul(activations.attention.att_out, layer.self_att_weights,
             /*add=*/nullptr, env, activations.attention.att_sums);
  RMSNormInplaceBatched(layer.post_self_attention_norm_scale,
                        activations.attention.att_sums, env.ctx);
  AddFromBatched(activations.attention.att_sums, activations.x, env.ctx);
}

static void T5GemmaDecoderCrossAttentionReference(
    const ModelConfig& config, size_t layer_idx,
    const T5GemmaDecoderLayerWeightsPtrs& layer, Activations& activations,
    QBatch& qbatch, MatMulEnv& env) {
  const LayerConfig& layer_config = layer.layer_config;
  const size_t qkv_dim = layer_config.qkv_dim;
  const size_t heads = layer_config.heads;
  const size_t kv_heads = layer_config.kv_heads;
  const size_t heads_per_kv = heads / kv_heads;

  RMSNormBatched(activations.x, layer.pre_cross_attention_norm_scale,
                 activations.attention.pre_att_rms_out, env.ctx);
  CallMatMul(activations.attention.pre_att_rms_out, layer.cross_q_einsum_w,
             /*add=*/nullptr, env, activations.attention.q);

  for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
    const T5GemmaEncoderCache* encoder_cache = qbatch.T5EncoderCache(qi);
    HWY_DASSERT(encoder_cache != nullptr);
    HWY_DASSERT(layer_idx < encoder_cache->cross_keys.size());
    HWY_DASSERT(layer_idx < encoder_cache->cross_values.size());
    const size_t source_len = encoder_cache->source_len;
    const MatStorageT<float>& cross_k = encoder_cache->cross_keys[layer_idx];
    const MatStorageT<float>& cross_v = encoder_cache->cross_values[layer_idx];
    HWY_DASSERT(cross_k.Rows() == source_len);
    HWY_DASSERT(cross_v.Rows() == source_len);
    HWY_DASSERT(cross_k.Cols() == kv_heads * qkv_dim);
    HWY_DASSERT(cross_v.Cols() == kv_heads * qkv_dim);

    for (size_t head = 0; head < heads; ++head) {
      const size_t kv_head = head / heads_per_kv;
      const float* query = activations.attention.q.Row(qi) + head * qkv_dim;

      float max_score = -std::numeric_limits<float>::infinity();
      for (size_t source_pos = 0; source_pos < source_len; ++source_pos) {
        if (encoder_cache->pad_mask[source_pos]) continue;
        const float* key = cross_k.Row(source_pos) + kv_head * qkv_dim;
        float score = 0.0f;
        for (size_t dim = 0; dim < qkv_dim; ++dim) {
          score += query[dim] * key[dim];
        }
        score = MaybeLogitsSoftCap(
            config.att_cap, score / sqrtf(static_cast<float>(qkv_dim)));
        max_score = std::max(max_score, score);
      }

      float denom = 0.0f;
      float* out = activations.attention.att_out.Row(qi) + head * qkv_dim;
      std::fill(out, out + qkv_dim, 0.0f);
      if (max_score == -std::numeric_limits<float>::infinity()) continue;

      for (size_t source_pos = 0; source_pos < source_len; ++source_pos) {
        if (encoder_cache->pad_mask[source_pos]) continue;
        const float* key = cross_k.Row(source_pos) + kv_head * qkv_dim;
        const float* value = cross_v.Row(source_pos) + kv_head * qkv_dim;
        float score = 0.0f;
        for (size_t dim = 0; dim < qkv_dim; ++dim) {
          score += query[dim] * key[dim];
        }
        score = MaybeLogitsSoftCap(
            config.att_cap, score / sqrtf(static_cast<float>(qkv_dim)));
        const float weight = expf(score - max_score);
        denom += weight;
        for (size_t dim = 0; dim < qkv_dim; ++dim) {
          out[dim] += weight * value[dim];
        }
      }
      const float inv_denom = denom == 0.0f ? 0.0f : 1.0f / denom;
      for (size_t dim = 0; dim < qkv_dim; ++dim) out[dim] *= inv_denom;
    }
  }

  CallMatMul(activations.attention.att_out, layer.cross_att_weights,
             /*add=*/nullptr, env, activations.attention.att_sums);
  RMSNormInplaceBatched(layer.post_cross_attention_norm_scale,
                        activations.attention.att_sums, env.ctx);
  AddFromBatched(activations.attention.att_sums, activations.x, env.ctx);
}

static void T5GemmaDecoderFFW(const T5GemmaDecoderLayerWeightsPtrs& layer,
                              Activations& activations, MatMulEnv& env) {
  RMSNormBatched(activations.x, layer.pre_ffw_norm_scale,
                 activations.pre_ffw_rms_out, env.ctx);

#if GEMMA_FUSED_FFN
  const LayerConfig& layer_config = layer.layer_config;
  const auto fused = [&](RowPtrsBF C1, IndexRange range_r, IndexRange range_c,
                         StridedViewBF C2, size_t worker) {
    Activation(layer_config.activation, C1, range_r, range_c, C2, env.ctx,
               worker);
  };
  MMOptions options;
  options.SetFunc(fused);
  CallTwoMatMul(activations.pre_ffw_rms_out, layer.gating_einsum_w1,
                layer.gating_einsum_w2, env, activations.C1, options);
#else
  CallMatMul(activations.pre_ffw_rms_out, layer.gating_einsum_w1,
             /*add=*/nullptr, env, activations.C1);
  CallMatMul(activations.pre_ffw_rms_out, layer.gating_einsum_w2,
             /*add=*/nullptr, env, activations.C2);
  ActivationBatched(layer.layer_config.activation, activations.C1,
                    &activations.C2, env.ctx);
#endif

  CallMatMul(activations.C1, layer.linear_w, /*add=*/nullptr, env,
             activations.ffw_out);
  RMSNormInplaceBatched(layer.post_ffw_norm_scale, activations.ffw_out,
                        env.ctx);
  AddFromBatched(activations.ffw_out, activations.x, env.ctx);
}

static void T5GemmaComputeLogitsChunked(const WeightsPtrs& weights,
                                        Activations& activations,
                                        MatMulEnv& env,
                                        uint8_t** chunk_row_ptrs,
                                        const hwy::BitSet4096<>* non_eos,
                                        int* greedy_tokens,
                                        float* greedy_logits) {
  constexpr size_t kLogitsChunk = kMaxNC;
  const size_t vocab_size = weights.t5gemma_decoder_embedding.Rows();
  const size_t model_dim = weights.t5gemma_decoder_embedding.Cols();
  for (size_t start = 0; start < vocab_size; start += kLogitsChunk) {
    const size_t rows = std::min(kLogitsChunk, vocab_size - start);
    MatPtr embedding_chunk("dec_emb_chunk",
                           weights.t5gemma_decoder_embedding.GetType(),
                           Extents2D(rows, model_dim));
    embedding_chunk.SetScale(weights.t5gemma_decoder_embedding.Scale());
    embedding_chunk.SetPtr(
        const_cast<uint8_t*>(
            weights.t5gemma_decoder_embedding.RowBytes(start)),
        weights.t5gemma_decoder_embedding.Stride());

    MatPtrT<float> logits_chunk(
        "logits_chunk", Extents2D(activations.logits.Rows(), rows));
    logits_chunk.SetPtr(activations.logits.Row(0) + start,
                        activations.logits.Stride());
    for (size_t qi = 0; qi < activations.logits.Rows(); ++qi) {
      chunk_row_ptrs[qi] =
          reinterpret_cast<uint8_t*>(activations.logits.Row(qi) + start);
    }
    logits_chunk.AttachRowPtrs(chunk_row_ptrs);

    CallMatMul(activations.x_bf, embedding_chunk, /*add=*/nullptr, env,
               logits_chunk);

    if (greedy_tokens != nullptr) {
      for (size_t qi = 0; qi < activations.logits.Rows(); ++qi) {
        if (non_eos != nullptr && !non_eos->Get(qi)) continue;
        const TokenAndProb chunk_best =
            ArgmaxAndMax(Logits(activations.logits.Row(qi) + start, rows));
        if (chunk_best.prob > greedy_logits[qi]) {
          greedy_logits[qi] = chunk_best.prob;
          greedy_tokens[qi] = static_cast<int>(start + chunk_best.token);
        }
      }
    }
  }
}

static void T5GemmaDecoderTransformer(const ModelConfig& config,
                                      const WeightsPtrs& weights,
                                      Activations& activations,
                                      MatStorageT<float>& decoder_kv,
                                      const MatPtrT<float>& inv_timescale,
                                      QBatch& qbatch, MatMulEnv& env) {
  for (size_t layer_idx = 0; layer_idx < weights.t5gemma_decoder_layers.size();
       ++layer_idx) {
    const T5GemmaDecoderLayerWeightsPtrs& layer =
        weights.t5gemma_decoder_layers[layer_idx];
    T5GemmaDecoderSelfAttentionReference(config, layer_idx, layer, activations,
                                         decoder_kv, inv_timescale, qbatch,
                                         env);
    T5GemmaDecoderCrossAttentionReference(config, layer_idx, layer,
                                          activations, qbatch, env);
    T5GemmaDecoderFFW(layer, activations, env);
  }
}

static bool T5GemmaUseGreedyFastPath(const RuntimeConfig& runtime_config) {
  return !runtime_config.sample_func && !runtime_config.accept_token &&
         runtime_config.top_k == 1 && runtime_config.temperature == 0.0f;
}

static void T5GemmaSampleAndStream(const ModelConfig& config,
                                   const RuntimeConfig& runtime_config,
                                   const WeightsPtrs& weights,
                                   const SampleFunc& sample_token,
                                   Activations& activations, QBatch& qbatch,
                                   MatMulEnv& env, uint8_t** chunk_row_ptrs,
                                   std::vector<int>& greedy_tokens,
                                   std::vector<float>& greedy_logits,
                                   hwy::BitSet4096<>& non_eos,
                                   TimingInfo& timing_info) {
  HWY_DASSERT(qbatch.Size() == activations.x.Rows());

  RMSNormBatched(activations.x, weights.t5gemma_decoder_final_norm_scale,
                 activations.x_bf, env.ctx);
  const bool greedy_fast_path = T5GemmaUseGreedyFastPath(runtime_config);
  if (greedy_fast_path) {
    std::fill(greedy_tokens.begin(), greedy_tokens.end(), 0);
    std::fill(greedy_logits.begin(), greedy_logits.end(),
              -std::numeric_limits<float>::infinity());
  }
  {
    GCPP_ZONE(env.ctx, /*worker=*/0, Zones::kGenEmbeddingMatmul);
    T5GemmaComputeLogitsChunked(
        weights, activations, env, chunk_row_ptrs,
        greedy_fast_path ? &non_eos : nullptr,
        greedy_fast_path ? greedy_tokens.data() : nullptr,
        greedy_fast_path ? greedy_logits.data() : nullptr);
  }
  if (!greedy_fast_path) {
    MaybeLogitsSoftCapBatched(config.final_cap, activations.logits, non_eos,
                              env.ctx);
  }

  timing_info.NotifyGenerated(non_eos.Count());
  ParallelFor(
      Parallelism::kFlat, qbatch.Size(), env.ctx,
      /*cluster_idx=*/0, Callers::kSampleAndStream,
      [&](size_t qi, size_t worker) {
        if (!non_eos.Get(qi)) return;

        const size_t pos = qbatch.Pos(qi);
        TokenAndProb tp;
        if (greedy_fast_path) {
          tp.token = greedy_tokens[qi];
          tp.prob = 1.0f;
        } else {
          tp = sample_token(qi, pos, activations.logits.RowSpan(qi), worker);
        }
        activations.sampled.Row(qi)[0] = static_cast<uint32_t>(pos);
        activations.sampled.Row(qi)[1] = static_cast<uint32_t>(tp.token);
        activations.sampled.Row(qi)[2] = hwy::BitCastScalar<uint32_t>(tp.prob);
      });

  non_eos.Foreach([&](size_t qi) {
    const size_t pos = activations.sampled.Row(qi)[0];
    const int token = static_cast<int>(activations.sampled.Row(qi)[1]);
    const float prob =
        hwy::BitCastScalar<float>(activations.sampled.Row(qi)[2]);
    StreamAndUpdateEOS(qi, pos, token, prob, config, runtime_config, qbatch,
                       /*update_pos=*/true, non_eos);
  });
}

static void T5GemmaGenerateT(const ModelConfig& config,
                             const RuntimeConfig& runtime_config,
                             const AesCtrEngine& engine,
                             const WeightsPtrs& weights,
                             Activations& activations, QBatch& qbatch,
                             MatMulEnv& env, TimingInfo& timing_info) {
  hwy::BitSet4096<> non_eos;
  for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
    non_eos.Set(qi);
    qbatch.PrevToken(qi) = BOS_ID;
  }

  const SampleFunc sample_token =
      ChooseSampleFunc(runtime_config, engine, env.ctx);
  const size_t max_gen_steps =
      HWY_MIN(runtime_config.max_generated_tokens, qbatch.KV(0).SeqLen());
  const LayerConfig& layer_config = config.decoder_layer_configs[0];
  MatStorageT<float> decoder_kv(
      MatFactory("t5_d_kv", qbatch.Size(),
                 2 * layer_config.kv_heads * layer_config.qkv_dim,
                 env.ctx.allocator));
  decoder_kv.AllocateAndAttachRowPtrs(env.row_ptrs);
  MatStorageT<float> inv_timescale =
      CreateInvTimescale(env.ctx.allocator, layer_config.qkv_dim,
                         layer_config.post_qk == PostQKType::HalfRope);
  auto logits_chunk_row_ptrs = hwy::AllocateAligned<uint8_t*>(qbatch.Size());
  std::vector<int> greedy_tokens(qbatch.Size());
  std::vector<float> greedy_logits(qbatch.Size());
  timing_info.generate_start = hwy::platform::Now();
  for (size_t gen = 0; gen < max_gen_steps && non_eos.Any(); ++gen) {
    T5GemmaEmbedDecoderTokens(config, weights, activations, qbatch, env);
    T5GemmaDecoderTransformer(config, weights, activations, decoder_kv,
                              inv_timescale, qbatch, env);
    T5GemmaSampleAndStream(config, runtime_config, weights, sample_token,
                           activations, qbatch, env,
                           logits_chunk_row_ptrs.get(), greedy_tokens,
                           greedy_logits, non_eos, timing_info);
  }
  timing_info.NotifyGenerateDone();
}

// `x_row` indicates which row of `x` to write to.
// `pos` is the *token*'s position for `AddAbsolutePositionalEmbeddings`, not
// the start of the batch, because this is called for batches of tokens in
// prefill, but batches of queries in decode.
//
// For GEMMA_VLM, image tokens are copied into -2 locations (per the Gemma 3
// spec) until we run out of image tokens. This allows for a multi-image prompt
// if -2 locations with appropriate begin/end image tokens are created by the
// calling application.
// Returns new image_token_position.
HWY_NOINLINE size_t EmbedMMToken(int token, size_t x_row, size_t pos,
                                 size_t pos_in_prompt,
                                 const ModelConfig& model_config,
                                 const WeightsPtrs& weights,
                                 MatStorageT<float>& x, ThreadingContext& ctx,
                                 const ImageTokens* image_tokens,
                                 size_t image_token_position) {
  GCPP_ZONE(ctx, hwy::Profiler::GlobalIdx(), Zones::kGenEmbed);

  // Image tokens just need to be copied.
  if (model_config.wrapping == PromptWrapping::GEMMA_VLM &&
      image_tokens != nullptr && token == -2 &&
      image_token_position < image_tokens->Rows()) {
    hwy::CopyBytes(image_tokens->Row(image_token_position), x.Row(x_row),
                   x.Cols() * x.ElementBytes());
    return image_token_position + 1;
  }

  if (model_config.wrapping == PromptWrapping::PALIGEMMA &&
      image_tokens != nullptr && pos_in_prompt < image_tokens->Rows()) {
    hwy::CopyBytes(image_tokens->Row(pos_in_prompt), x.Row(x_row),
                   x.Cols() * x.ElementBytes());
    return image_token_position;
  }

  EmbedTokenFromWeights(token, x_row, model_config,
                        weights.embedder_input_embedding, x);

  if (model_config.absolute_pe) {
    AddAbsolutePositionalEmbeddings(x.Row(x_row), model_config.model_dim, pos);
  }
  return image_token_position;
}

static HWY_NOINLINE void ComputePLEEmbeddings(size_t tbatch_size,
                                              const std::vector<int>& tokens,
                                              const ModelConfig& config,
                                              const WeightsPtrs& weights,
                                              Activations& activations,
                                              MatMulEnv& env) {
  if (config.ple_dim == 0) return;

  // 1. Convert activations.x (float) to activations.x_bf (BF16)
  for (size_t r = 0; r < tbatch_size; ++r) {
    for (size_t c = 0; c < config.model_dim; ++c) {
      activations.x_bf.Row(r)[c] = BF16(activations.x.Row(r)[c]);
    }
  }

  // 2. CallMatMul for the context projection (with folded scale)
  const float scale_proj = 1.0f / sqrtf(static_cast<float>(config.model_dim));
  MatPtr scaled_ple_model_proj = weights.ple_model_proj;
  scaled_ple_model_proj.SetScale(scaled_ple_model_proj.Scale() * scale_proj);

  CallMatMul(activations.x_bf, scaled_ple_model_proj, /*add=*/nullptr, env,
             activations.ple_embeds);

  // 3. Apply the model projection scale (Folded into MatMul above, so this is empty)
  const size_t ple_total_dim = config.num_layers * config.ple_dim;

  // 4. RMSNorm (applied to each layer's embedding independently)
  CallUpcasted(&weights.ple_proj_norm, [&](const auto* weights_t) {
    ParallelFor(Parallelism::kFlat, tbatch_size, env.ctx, /*cluster_idx=*/0,
                Callers::kOpsRMSNormInplaceBatched,
                [&](uint64_t token_idx, size_t worker) {
                  float* row = activations.ple_embeds.Row(token_idx);
                  for (size_t layer = 0; layer < config.num_layers; ++layer) {
                    float* slice = row + layer * config.ple_dim;
                    RMSNormInplace(weights_t->PackedScale1(), /*w_ofs=*/0,
                                   slice, config.ple_dim, env.ctx, worker);
                  }
                });
  });

  // 5. Add token embedding and apply input scale
  const float scale_input = 1.0f / sqrtf(2.0f);
  // Use pre-allocated activations.ple_token_emb
  float* token_emb = activations.ple_token_emb.data();
  for (size_t r = 0; r < tbatch_size; ++r) {
    int token = tokens[r];
    CallUpcasted(&weights.ple_embeddings, [&](const auto* weights_t) HWY_ATTR {
      const size_t embedding_ofs = token * weights_t->Stride();
      const auto embedding_span =
          MakeSpan(weights_t->Row(0), embedding_ofs + ple_total_dim);
      const hn::ScalableTag<float> df;
      DecompressAndZeroPad(df, embedding_span, embedding_ofs, token_emb,
                           ple_total_dim);

      const float token_scale =
          sqrtf(static_cast<float>(config.ple_dim)) * weights_t->Scale();
      const float scaled_token_scale = token_scale * scale_input;
      float* out_row = activations.ple_embeds.Row(r);

      // Vectorized embedding loop (aligned with precomputed scale intent)
      using DF = hn::ScalableTag<float>;
      using VF = hn::Vec<DF>;
      const DF df_float;
      const VF v_scale_input = hn::Set(df_float, scale_input);
      const VF v_scaled_token_scale = hn::Set(df_float, scaled_token_scale);

      Decompress1AndCompressInplace(
          df_float, out_row, ple_total_dim, token_emb, /*p1_ofs=*/0,
          [&](DF df, VF v_out, VF v_emb) HWY_ATTR -> VF {
            VF v_scaled_out = hn::Mul(v_out, v_scale_input);
            return hn::MulAdd(v_emb, v_scaled_token_scale, v_scaled_out);
          });
    });
  }
}

// Populates KV cache for batches of tokens from one query at a time. This is
// called if prompts are longer than the query batch size, and also in
// prefix-LM mode (end > 0), which must see all tokens in one batch.
static HWY_NOINLINE void PrefillTBatch(const ModelConfig& config,
                                       const RuntimeConfig& runtime_config,
                                       const WeightsPtrs& weights,
                                       Activations& activations, QBatch& qbatch,
                                       MatMulEnv& env,
                                       hwy::BitSet4096<>& non_eos) {
  PROFILER_ZONE("Gen.PrefillT");

  // Batches are important for amortizing loading weights over multiple tokens.
  // This is possible in prefill because we know all tokens beforehand, whereas
  // decode depends on the previous output token. However, each prefill batch of
  // a query requires that preceding batches already wrote to the KV cache,
  // hence we sequentially loop over token batches. We can reduce the number of
  // iterations by increasing the batch size, but this also increases arithmetic
  // intensity, and so we are eventually compute-limited. TransformerLayer uses
  // all available threads, so we do not also parallelize over queries, but note
  // that PrefillQBatch uses queries as the batch dimension.
  const size_t max_tbatch_size = runtime_config.prefill_tbatch_size;

  // For each query. `qi` is within the batch, not the global query index.
  for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
    non_eos.Set(qi);

    // One query at a time, batching will be the query's prompt tokens.
    QBatch qbatch_1 = qbatch.Single(qi);

    const size_t prompt_size = qbatch_1.Prompt(0).size();
    // In autoregressive mode, we don't need to prefill the last token, so - 1.
    size_t prefill_this_query = prompt_size - 1;
    const size_t prefix_end_this_query = qbatch_1.PrefixEnd(0);
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
      std::vector<int> tbatch_tokens;
      if (config.ple_dim > 0) {
        tbatch_tokens.reserve(tbatch_size);
      }
      activations.token_ids.resize(tbatch_size);
      for (size_t ti = 0; ti < tbatch_size; ++ti) {
        const size_t pos = qbatch_1.Pos(0) + ti;
        const size_t pos_in_prompt = tbatch_start + ti;
        HWY_DASSERT(pos_in_prompt < prompt_size);
        const int token = qbatch_1.Prompt(0)[pos_in_prompt];
        if (config.ple_dim > 0) {
          tbatch_tokens.push_back(token);
        }
        activations.token_ids[ti] = token;
        image_token_position = EmbedMMToken(
            token, ti, pos, pos_in_prompt, config, weights, activations.x,
            env.ctx, runtime_config.image_tokens, image_token_position);
        // NOTE: we unconditionally call StreamToken, even if EOS.
        if (pos_in_prompt < prompt_size - 1) {
          runtime_config.StreamToken(qbatch_1.QueryIdx(0), pos, token, 0.0f);
        } else {
          // The last token will be streamed later and we should only get here
          // if we need to attend to the last token because it is in the prefix.
          HWY_ASSERT(attend_to_last_token);
        }
      }
      if (config.ple_dim > 0) {
        ComputePLEEmbeddings(tbatch_size, tbatch_tokens, config, weights,
                             activations, env);
      }

      // mHC (DeepSeek V4): fan the embeddings out into residual streams.
      DeepSeekMaybeInitHCStreams(activations, env);

      // Transformer with one batch of tokens from a single query. No need to
      // set `PrevToken` because we already did the embedding above.
      for (size_t layer_idx = 0; layer_idx < config.layer_configs.size();
           ++layer_idx) {
        TransformerLayer(tbatch_size, layer_idx, *weights.GetLayer(layer_idx),
                         activations, qbatch_1, env);
      }

      // Speculative decoding (DeepSeek V4): keep the MTP block's KV cache in
      // sync with the prompt. Row ti pairs with the next prompt token, which
      // is always available because prefill stops before the last token.
      if (HWY_UNLIKELY(runtime_config.use_mtp && !weights.mtp_layers.empty() &&
                       prefix_end_this_query == 0)) {
        std::vector<int> next(tbatch_size);
        for (size_t ti = 0; ti < tbatch_size; ++ti) {
          next[ti] = qbatch_1.Prompt(0)[tbatch_start + ti + 1];
        }
        DeepSeekMTPStep(tbatch_size, next.data(), /*compute_logits=*/false,
                        weights, activations, qbatch_1, env);
      }

      qbatch_1.MutablePos(0) += tbatch_size;
    }  // for tbatch_start
    if (attend_to_last_token) {
      // We need to rewind the position for the last token that we only
      // attended to to make sure the prefix LM sees everything.
      // This means we duplicate work on the last prompt token in autoregressive
      // decoding. Alternatives: (1) real masking; (2) always prefill the last
      // token and only generate the next one from the already prefilled
      // activations.
      qbatch_1.MutablePos(0) -= 1;
    }
  }
}

static void MaybeObserve(const RuntimeConfig& runtime_config,
                         Activations& activations, QBatch& qbatch,
                         int layer_idx) {
  if constexpr (kObserver) {
    if (HWY_UNLIKELY(runtime_config.activations_observer)) {
      runtime_config.activations_observer(
          QueriesPos(&qbatch.MutablePos(0), qbatch.Size()), layer_idx,
          activations);
    }
  }
}

// Embeds PrevToken (one from each query) and calls each TransformerLayer.
// Called by query-batched `PrefillQBatch` and `GenerateT`, but not the
// token-batched `PrefillTBatch`, which supports image embedding.
HWY_NOINLINE void Transformer(const ModelConfig& config,
                              const RuntimeConfig& runtime_config,
                              const WeightsPtrs& weights,
                              Activations& activations, QBatch& qbatch,
                              MatMulEnv& env) {
  if constexpr (kObserver) {
    if (HWY_UNLIKELY(runtime_config.layers_output)) {
      for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
        const float token_f = qbatch.PrevToken(qi);
        runtime_config.layers_output(qbatch.QueryIdx(qi), qbatch.Pos(qi),
                                     "tokens", -1, &token_f, 1);
      }
    }
  }

  // TODO: parallelize?
  std::vector<int> tbatch_tokens;
  if (config.ple_dim > 0) {
    tbatch_tokens.reserve(qbatch.Size());
  }
  activations.token_ids.resize(qbatch.Size());
  for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
    const int token = qbatch.PrevToken(qi);
    if (config.ple_dim > 0) {
      tbatch_tokens.push_back(token);
    }
    activations.token_ids[qi] = token;
    EmbedMMToken(token, qi, qbatch.Pos(qi),
                 /*pos_in_prompt=*/0, config, weights, activations.x, env.ctx,
                 /*image_tokens=*/nullptr, /*image_token_position=*/0);
  }
  if (config.ple_dim > 0) {
    ComputePLEEmbeddings(qbatch.Size(), tbatch_tokens, config, weights,
                         activations, env);
  }

  // mHC (DeepSeek V4): fan the embedding out into parallel residual streams.
  DeepSeekMaybeInitHCStreams(activations, env);

  for (size_t layer_idx = 0; layer_idx < weights.c_layers.size(); ++layer_idx) {
    TransformerLayer(/*num_tokens=*/1, layer_idx, *weights.GetLayer(layer_idx),
                     activations, qbatch, env);

    MaybeObserve(runtime_config, activations, qbatch, layer_idx);
  }

  // mHC: collapse the residual streams back into x before the final norm.
  DeepSeekMaybeFinalizeHCStreams(weights, activations, env);
}

// Populates KV cache for the batch queries, one token at a time.
static HWY_NOINLINE void PrefillQBatch(const size_t max_prompt_size,
                                       const ModelConfig& config,
                                       const RuntimeConfig& runtime_config,
                                       const WeightsPtrs& weights,
                                       Activations& activations, QBatch& qbatch,
                                       MatMulEnv& env,
                                       hwy::BitSet4096<>& non_eos) {
  PROFILER_ZONE("Gen.PrefillQ");

  for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
    non_eos.Set(qi);

    // Should only be called for autoregressive (non-prefix-LM) prefill.
    HWY_DASSERT(qbatch.PrefixEnd(qi) == 0);
  }

  // In autoregressive mode, we don't prefill the last token, hence - 1.
  for (size_t pos_in_prompt = 0; pos_in_prompt < max_prompt_size - 1;
       ++pos_in_prompt) {
    for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
      int token = config.eos_id;
      if (pos_in_prompt < qbatch.Prompt(qi).size() - 1) {
        token = qbatch.Prompt(qi)[pos_in_prompt];
        // Ignore StreamToken return value because requesting to stop does not
        // make sense during prefill.
        (void)runtime_config.StreamToken(qbatch.QueryIdx(qi), pos_in_prompt,
                                         token, 0.0f);
        qbatch.MutablePos(qi) = pos_in_prompt;
      } else {
        // This prevents the kv cache of eos_id to be written to last prefilled
        // token.
        qbatch.MutablePos(qi) = qbatch.Prompt(qi).size();
      }

      qbatch.PrevToken(qi) = token;
    }

    // The input (PrevToken) is one token from each query in the batch.
    // Do not call `SampleAndStream` because it computes logits for token
    // probabilities, which are not required for the prompt tokens.
    Transformer(config, runtime_config, weights, activations, qbatch, env);
  }

  for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
    qbatch.MutablePos(qi) = qbatch.Prompt(qi).size() - 1;
  }
}

// Calls `StreamToken`, writes the token to `PrevToken` for use by subsequent
// `Transformer`, and increments `MutablePos`. Also updates `non_eos` if the
// query is at the end of its sequence.
static void StreamAndUpdateEOS(const size_t qi, size_t pos, int token,
                               const float prob, const ModelConfig& config,
                               const RuntimeConfig& runtime_config,
                               QBatch& qbatch, bool update_pos,
                               hwy::BitSet4096<>& non_eos) {
  HWY_DASSERT(non_eos.Get(qi));  // otherwise, should not be called.

  if (HWY_UNLIKELY(
          !runtime_config.StreamToken(qbatch.QueryIdx(qi), pos, token, prob))) {
    // User decided to stop: set token to primary EOS to trigger IsEOS below.
    token = config.eos_id;
    HWY_DASSERT(config.IsEOS(token));
  }

  qbatch.PrevToken(qi) = token;
  qbatch.MutablePos(qi) += update_pos ? 1 : 0;

  // Primary or secondary EOS: mark query as EOS, but still increment (for
  // multi-turn, we should still keep the prior EOS).
  if (HWY_UNLIKELY(config.IsEOS(token))) non_eos.Clear(qi);
}

// Final norm x -> x_bf. DeepSeek V4 checkpoints store the true norm scale;
// gemma weights are exported as scale-1 for the (1 + w) RMSNorm. Shared with
// the speculative driver (deepseek_spec.cc), which must match this exactly.
HWY_NOINLINE void FinalNormBatched(const ModelConfig& config,
                                   const WeightsPtrs& weights,
                                   Activations& activations, MatMulEnv& env) {
  if (HWY_UNLIKELY(config.model_family_version == 4 && config.HasMLA())) {
    DeepSeekFinalNorm(weights, activations, env);
  } else {
    RMSNormBatched(activations.x, weights.final_norm_scale, activations.x_bf,
                   env.ctx);
  }
}

// Logits from x_bf (after FinalNormBatched). DeepSeek models have an untied
// output head; others reuse the input embedding. Also shared with the
// speculative driver.
HWY_NOINLINE void FinalLogits(const WeightsPtrs& weights,
                              Activations& activations, MatMulEnv& env) {
  GCPP_ZONE(env.ctx, /*worker=*/0, Zones::kGenEmbeddingMatmul);
  const MatPtr& output_head = weights.lm_head.HasPtr()
                                  ? weights.lm_head
                                  : weights.embedder_input_embedding;
  CallMatMul(activations.x_bf, output_head,
             /*add=*/nullptr, env, activations.logits);
}

// Must be called after Transformer: either after prefill, or during decode.
// Computes logits, samples and streams the token.
static void SampleAndStream(const ModelConfig& config,
                            const RuntimeConfig& runtime_config,
                            const WeightsPtrs& weights,
                            const SampleFunc& sample_token,
                            Activations& activations, QBatch& qbatch,
                            MatMulEnv& env, hwy::BitSet4096<>& non_eos,
                            TimingInfo& timing_info) {
  HWY_DASSERT(qbatch.Size() == activations.x.Rows());

  FinalNormBatched(config, weights, activations, env);

  MaybeObserve(runtime_config, activations, qbatch, -1);

  FinalLogits(weights, activations, env);
  PROFILER_ZONE("Gen.Softcap+Sample+Stream");

  MaybeLogitsSoftCapBatched(config.final_cap, activations.logits, non_eos,
                            env.ctx);

  timing_info.NotifyGenerated(non_eos.Count());

  ParallelFor(
      Parallelism::kFlat, qbatch.Size(), env.ctx,
      /*cluster_idx=*/0, Callers::kSampleAndStream,
      [&](size_t qi, size_t worker) {
        if (!non_eos.Get(qi)) return;

        // We streamed all prefill tokens, but pos is still one behind
        // because we started generation at pos = prompt.size() - 1.
        // We want the pos argument to match the number of calls to
        // `StreamToken`, as expected by the caller.
        const size_t pos = qbatch.Pos(qi) + 1;

        const TokenAndProb tp =
            sample_token(qi, pos, activations.logits.RowSpan(qi), worker);
        // `sampled` is padded, which prevents false sharing.
        activations.sampled.Row(qi)[0] = static_cast<uint32_t>(pos);
        activations.sampled.Row(qi)[1] = static_cast<uint32_t>(tp.token);
        activations.sampled.Row(qi)[2] = hwy::BitCastScalar<uint32_t>(tp.prob);
      });

  // Sequentially, because `StreamToken` is not yet thread-safe.
  non_eos.Foreach([&](size_t qi) {
    const size_t pos = activations.sampled.Row(qi)[0];
    const int token = static_cast<int>(activations.sampled.Row(qi)[1]);
    const float prob =
        hwy::BitCastScalar<float>(activations.sampled.Row(qi)[2]);
    StreamAndUpdateEOS(qi, pos, token, prob, config, runtime_config, qbatch,
                       /*update_pos=*/true, non_eos);
  });
}

static HWY_INLINE SampleFunc
ChooseSampleFunc(const RuntimeConfig& runtime_config,
                 const AesCtrEngine& engine, ThreadingContext& ctx) {
  // If user provided a sample_func, use it.
  if (runtime_config.sample_func) return runtime_config.sample_func;

  // Fast path for top-1 with no accept_token.
  if (runtime_config.top_k == 1 && !runtime_config.accept_token) {
    return [&](size_t /*qi*/, size_t /*pos*/, Logits logits, size_t worker)
               HWY_ATTR -> TokenAndProb {
                 GCPP_ZONE(ctx, worker, Zones::kGenSampleTop1);
                 return Top1OfSoftmax(logits);
               };
  }

  // General case: Softmax with top-k sampling.
  return [&](size_t qi, size_t pos, Logits logits,
             size_t worker) HWY_ATTR -> TokenAndProb {
    GCPP_ZONE(ctx, worker, Zones::kGenSampleTopK);
    // We want a different sequence for each batch element and position.
    const uint64_t stream = (static_cast<uint64_t>(qi) << 32) | pos;
    RngStream gen(engine, stream);
    return FusedSoftmaxAndSampleTopK(logits, runtime_config.top_k, gen,
                                     runtime_config.temperature,
                                     runtime_config.accept_token, ctx, worker);
  };
}

size_t PrefillTBatchOrQBatch(const ModelConfig& config,
                             const RuntimeConfig& runtime_config,
                             const WeightsPtrs& weights,
                             Activations& activations, QBatch& qbatch,
                             MatMulEnv& env, TimingInfo& timing_info) {
  size_t max_prompt_size = 0;
  bool all_prefix_end_are_zero = true;
  size_t total_prefill_tokens = 0;  // only for throughput stats.
  const size_t seq_len = qbatch.KV(0).SeqLen();
  for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
    const PromptTokens& prompt = qbatch.Prompt(qi);
    // Sanity check: prompts should not be empty. Note that multi-turn prompts
    // start with <end_of_turn>.
    HWY_ASSERT(prompt.size() != 0);

    max_prompt_size = HWY_MAX(max_prompt_size, prompt.size());

    // Prefill stops before size - 1 because the last prompt token is the
    // first input token for generation.
    total_prefill_tokens += prompt.size() - 1;

    all_prefix_end_are_zero &= qbatch.PrefixEnd(qi) == 0;

    // We use a single divisor, so all sequence lengths must be the same.
    HWY_ASSERT(qbatch.KV(qi).SeqLen() == seq_len);
  }
  if (max_prompt_size > seq_len) {
    HWY_ABORT(
        "max_prompt_size = %zu, seq_len = %zu, increase --seq_len to at least "
        "that.",
        max_prompt_size, seq_len);
  }
  HWY_ASSERT(activations.attention.div_seq_len.GetDivisor() == seq_len);

  hwy::BitSet4096<> non_eos;  // indexed by qi

  timing_info.prefill_start = hwy::platform::Now();
  // Batch over the larger of prompt length, or queries.
  if ((qbatch.Size() > max_prompt_size) && all_prefix_end_are_zero) {
    activations.SetBatchSize(qbatch.Size());  // required before PrefillQBatch
    PrefillQBatch(max_prompt_size, config, runtime_config, weights, activations,
                  qbatch, env, non_eos);
  } else {
    PrefillTBatch(config, runtime_config, weights, activations, qbatch, env,
                  non_eos);
    activations.SetBatchSize(qbatch.Size());  // Restore after PrefillTBatch.
  }
  HWY_DASSERT(non_eos.Count() == qbatch.Size());
  timing_info.NotifyPrefill(total_prefill_tokens);
  // queries_pos have been incremented by Prefill.

  size_t max_gen_steps = runtime_config.max_generated_tokens;
  if (max_prompt_size + max_gen_steps > seq_len) {
    HWY_WARN("prefill %zu + max_gen_steps %zu > seq_len %zu, truncating.",
             max_prompt_size, max_gen_steps, seq_len);
    max_gen_steps = seq_len - max_prompt_size;
  }

  return max_gen_steps;
}

void StreamAndUpdateEOSAfterPrefill(const ModelConfig& config,
                                    const RuntimeConfig& runtime_config,
                                    QBatch& qbatch, hwy::BitSet4096<>& non_eos,
                                    size_t qi) {
  const size_t last_pos_in_prompt = qbatch.Pos(qi) - qbatch.InitialPos(qi);

  const size_t pos = qbatch.Pos(qi);  // during prefill, pos is still correct.
  // In autoregressive mode, we have not prefilled the last token, so do
  // not advance.
  const bool update_pos = (qbatch.Pos(qi) < qbatch.PrefixEnd(qi));
  StreamAndUpdateEOS(qi, pos, qbatch.Prompt(qi)[last_pos_in_prompt], 0.0f,
                     config, runtime_config, qbatch, update_pos, non_eos);
}

void SetWeightStats(const LayerWeightsPtrs& layer, Activations& a,
                    ThreadingContext& ctx) {
  const size_t layer_idx = layer.layer_idx;
  if (layer.layer_config.IsMoE()) {
    const size_t expert_idx = 0;
    a.s_w_expert_in1.Notify(layer_idx, layer.moe_gating_einsum_w1[expert_idx],
                            ctx, kTensorStatsIsWeight);
    a.s_w_expert_in2.Notify(layer_idx, layer.moe_gating_einsum_w2[expert_idx],
                            ctx, kTensorStatsIsWeight);
    a.s_w_expert_hidden.Notify(layer_idx, layer.moe_linear_w[expert_idx], ctx,
                               kTensorStatsIsWeight);
  }
  a.s_w_gating_einsum_w1.Notify(layer_idx, layer.gating_einsum_w1, ctx,
                                kTensorStatsIsWeight);
  a.s_w_gating_einsum_w2.Notify(layer_idx, layer.gating_einsum_w2, ctx,
                                kTensorStatsIsWeight);
  a.s_w_linear_w.Notify(layer_idx, layer.linear_w, ctx, kTensorStatsIsWeight);
}

// Decode: generates one continuation token for each query in `qbatch`.
static void GenerateT(const ModelConfig& config,
                      const RuntimeConfig& runtime_config,
                      const AesCtrEngine& engine, const WeightsPtrs& weights,
                      Activations& activations, QBatch& qbatch, MatMulEnv& env,
                      TimingInfo& timing_info) {
  for (const LayerWeightsPtrs& layer : weights.c_layers) {
    SetWeightStats(layer, activations, env.ctx);
  }

  if (HWY_UNLIKELY(runtime_config.use_mtp)) {
    if (!weights.mtp_layers.empty() && qbatch.Size() == 1 &&
        runtime_config.top_k == 1 && !runtime_config.sample_func &&
        !runtime_config.accept_token) {
      GenerateSpecV4(config, runtime_config, weights, activations, qbatch, env,
                     timing_info);
      return;
    }
    HWY_WARN(
        "use_mtp requires MTP weights, a single query and top_k == 1 (greedy);"
        " falling back to normal decoding.");
  }

  MaybePrint(2, timing_info.verbosity, "[ BEGIN PHASE: prefill ]");
  const size_t max_gen_steps = PrefillTBatchOrQBatch(
      config, runtime_config, weights, activations, qbatch, env, timing_info);
  // No-op if the profiler is disabled, but useful to separate prefill and
  // generate phases for profiling.
  if constexpr (PROFILER_ENABLED) {
    fprintf(stderr, "\n");
  }
  env.ctx.profiler.PrintResults();

  hwy::BitSet4096<> non_eos;  // indexed by qi

  // Stream the last prompt token from each query, fill activations.gen_tokens.
  for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
    non_eos.Set(qi);
    StreamAndUpdateEOSAfterPrefill(config, runtime_config, qbatch, non_eos, qi);
  }

  const SampleFunc sample_token =
      ChooseSampleFunc(runtime_config, engine, env.ctx);

  MaybePrint(2, timing_info.verbosity, "\n[ BEGIN PHASE: generate ]\n");

  timing_info.generate_start = hwy::platform::Now();
  for (size_t gen = 0; gen < max_gen_steps && non_eos.Any(); ++gen) {
    Transformer(config, runtime_config, weights, activations, qbatch, env);
    SampleAndStream(config, runtime_config, weights, sample_token, activations,
                    qbatch, env, non_eos, timing_info);
  }
  timing_info.NotifyGenerateDone();
}

// Same as GenerateT, but uses ContinuousQBatch.
static void GenerateTWithContinuousBatching(
    const ModelConfig& config, const RuntimeConfig& runtime_config,
    const AesCtrEngine& engine, const WeightsPtrs& weights,
    Activations& activations, AllQueries& all_queries, MatMulEnv& env,
    TimingInfo& timing_info) {
  const size_t qbatch_size = runtime_config.decode_qbatch_size;

  QBatch qbatch(0, qbatch_size, all_queries);
  ContinuousQBatch prefill_batch(qbatch_size, all_queries);

  hwy::BitSet4096<> non_eos;
  const SampleFunc sample_token =
      ChooseSampleFunc(runtime_config, engine, env.ctx);

  size_t query_inserted = 0;
  while (non_eos.Any() || query_inserted < all_queries.NumQueries()) {
    for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
      // Continue if qi slot is still processing.
      if (non_eos.Get(qi)) continue;
      // Collect the kv_cache from the qi slot in the qbatch to the
      // available_kv_caches_ in the prefill_batch.
      prefill_batch.MaybeReleaseKV(qbatch.Single(qi));

      // Prefill if no available prefilled queries to insert.
      if (prefill_batch.ShouldPrefill()) {
        prefill_batch.SetupNextBatchForPrefill();
        PrefillTBatchOrQBatch(config, runtime_config, weights, activations,
                              prefill_batch, env, timing_info);
        activations.SetBatchSize(qbatch.Size());
      }

      // Get the next query to insert to the generate batch.
      std::optional<size_t> qi_to_insert = prefill_batch.GetNextToInsert();
      if (qi_to_insert) {
        qbatch.Insert(qi_to_insert.value(), qi);
        query_inserted++;

        non_eos.Set(qi);
        StreamAndUpdateEOSAfterPrefill(config, runtime_config, qbatch, non_eos,
                                       qi);
      }
    }

    Transformer(config, runtime_config, weights, activations, qbatch, env);
    SampleAndStream(config, runtime_config, weights, sample_token, activations,
                    qbatch, env, non_eos, timing_info);
  }
  timing_info.NotifyGenerateDone();
}

void GenerateSingleT(const PromptTokens& prompt, size_t pos, size_t prefix_end,
                     const ModelConfig& config,
                     const RuntimeConfig& runtime_config,
                     const AesCtrEngine& engine, const WeightsPtrs& weights,
                     KVCache& kv_cache, MatMulEnv& env,
                     TimingInfo& timing_info) {
  if (config.is_encoder_decoder) {
    ValidateT5GemmaFreshGeneration(pos, prefix_end);
    std::vector<T5GemmaEncoderCache> encoder_caches;
    AllQueries all_queries(prompt, pos, prefix_end,
                           hwy::Span<KVCache>(&kv_cache, 1));
    AttachT5GemmaEncoderCaches(config, all_queries, encoder_caches,
                               env.ctx.allocator);
    Activations activations(runtime_config, config, /*batch_size=*/1,
                            kv_cache.SeqLen(), env.ctx, env.row_ptrs);
    timing_info.prefill_start = hwy::platform::Now();
    T5GemmaEncodeAllQueries(config, weights, all_queries, activations, env);
    timing_info.NotifyPrefill(T5GemmaPromptTokenCount(all_queries));
    QBatch qbatch(/*start=*/0, /*max_size=*/1, all_queries);
    T5GemmaGenerateT(config, runtime_config, engine, weights, activations,
                     qbatch, env, timing_info);
    return;
  }
  Activations activations(runtime_config, config,
                          runtime_config.prefill_tbatch_size, kv_cache.SeqLen(),
                          env.ctx, env.row_ptrs);

  AllQueries all_queries(prompt, pos, prefix_end,
                         hwy::Span<KVCache>(&kv_cache, 1));
  QBatch qbatch(/*start=*/0, /*max_size=*/1, all_queries);
  GenerateT(config, runtime_config, engine, weights, activations, qbatch, env,
            timing_info);
}

// Splits the input into batches of at most `runtime_config.decode_qbatch_size`
// queries, and calls `GenerateT` on each batch.
void GenerateBatchT(const ModelConfig& config,
                    const RuntimeConfig& runtime_config,
                    const AesCtrEngine& engine, const WeightsPtrs& weights,
                    AllQueries& all_queries, MatMulEnv& env,
                    TimingInfo& timing_info) {
  if (config.is_encoder_decoder) {
    std::vector<T5GemmaEncoderCache> encoder_caches;
    AttachT5GemmaEncoderCaches(config, all_queries, encoder_caches,
                               env.ctx.allocator);
    const size_t max_batch_size = HWY_MAX(runtime_config.decode_qbatch_size,
                                          runtime_config.prefill_tbatch_size);
    Activations activations(runtime_config, config, max_batch_size,
                            all_queries[0].kv_cache.SeqLen(), env.ctx,
                            env.row_ptrs);
    timing_info.prefill_start = hwy::platform::Now();
    T5GemmaEncodeAllQueries(config, weights, all_queries, activations, env);
    timing_info.NotifyPrefill(T5GemmaPromptTokenCount(all_queries));
    QBatch qbatch(/*start=*/0, runtime_config.decode_qbatch_size, all_queries);
    T5GemmaGenerateT(config, runtime_config, engine, weights, activations,
                     qbatch, env, timing_info);
    return;
  }
  const size_t max_batch_size = HWY_MAX(runtime_config.decode_qbatch_size,
                                        runtime_config.prefill_tbatch_size);
  Activations activations(runtime_config, config, max_batch_size,
                          all_queries[0].kv_cache.SeqLen(), env.ctx,
                          env.row_ptrs);

  if (runtime_config.use_continuous_batching) {
    GenerateTWithContinuousBatching(config, runtime_config, engine, weights,
                                    activations, all_queries, env, timing_info);
  } else {
    for (size_t start = 0; start < all_queries.NumQueries();
         start += runtime_config.decode_qbatch_size) {
      QBatch qbatch(start, runtime_config.decode_qbatch_size, all_queries);
      // Generate a batch of one token for each of `qbatch.Size()` queries.
      GenerateT(config, runtime_config, engine, weights, activations, qbatch,
                env, timing_info);
    }
  }
}

void GenerateImageTokensT(const ModelConfig& config,
                          const RuntimeConfig& runtime_config, size_t seq_len,
                          const WeightsPtrs& weights, const Image& image,
                          ImageTokens& image_tokens, MatMulEnv& env,
                          TimingInfo& timing_info) {
  const ModelConfig vit_config = GetVitConfig(config);
  const size_t num_tokens = vit_config.max_seq_len;

  MaybePrint(2, timing_info.verbosity, "\n[ BEGIN PHASE: image_token_gen ]\n");
  timing_info.NotifyImageTokenStart();

  {
    GCPP_ZONE(env.ctx, hwy::Profiler::GlobalIdx(), Zones::kGenImageTokens);
    if (config.vit_config.layer_configs.empty()) {
      HWY_ABORT("Model does not support generating image tokens.");
    }
    RuntimeConfig prefill_runtime_config = runtime_config;
    prefill_runtime_config.prefill_tbatch_size =
        num_tokens / (vit_config.pool_dim * vit_config.pool_dim);
    Activations prefill_activations(runtime_config, vit_config, num_tokens,
                                    num_tokens, env.ctx, env.row_ptrs);
    // Weights are for the full PaliGemma model, not just the ViT part.
    PrefillVit(config, weights, prefill_runtime_config, image, image_tokens,
               prefill_activations, env);
  }  // end GCPP_ZONE before we print results.

  // No-op if the profiler is disabled. Printing now ensures that the
  // `PrintResults` after prefill does not include the image token part.
  env.ctx.profiler.PrintResults();

  timing_info.NotifyImageTokenDone(num_tokens);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {
HWY_EXPORT(GenerateSingleT);
HWY_EXPORT(GenerateBatchT);
HWY_EXPORT(GenerateImageTokensT);

Gemma::Gemma(const GemmaArgs& args, ThreadingContext& ctx)
    : reader_(args.loader.weights),
      model_(reader_, args.loader.tokenizer, args.loader.wrapping),
      weights_(model_.Config()),
      chat_template_(model_.Tokenizer(), model_.Config().model),
      inference_(args.inference),
      aes_ctr_engine_(args.inference.deterministic) {
  if (args.inference.seq_len > model_.Config().max_seq_len) {
    HWY_WARN(
        "Overriding model's max_seq_len=%u with user provided seq_len=%zu.",
        model_.Config().max_seq_len, args.inference.seq_len);
    model_.MutableConfig().SetMaxSeqLen(args.inference.seq_len);
  }
  // Negligible CPU time in the ctor body (except ReadFromBlobs).
  weight_read_mode_ = weights_.ReadFromBlobs(model_, reader_, args.loader,
                                             args.inference, mat_owners_, ctx);
  // Read everything into memory, or `weights_.mapped_` keeps the mapping alive.
  reader_.CloseFile();
}

Gemma::~Gemma() = default;

void Gemma::Save(const Path& weights_path, ThreadingContext& ctx) const {
  BlobWriter writer(weights_path, ctx);
  const std::vector<uint32_t> serialized_mat_ptrs =
      weights_.AddTensorDataToWriter(writer);
  WriteSingleFile(model_.Config(), model_.Tokenizer(), serialized_mat_ptrs,
                  writer);
}

void Gemma::Generate(const RuntimeConfig& runtime_config,
                     const PromptTokens& prompt, size_t pos, size_t prefix_end,
                     KVCache& kv_cache, MatMulEnv& env,
                     TimingInfo& timing_info) const {
  env.ctx.pools.MaybeStartSpinning(runtime_config.use_spinning);

  HWY_DYNAMIC_DISPATCH(GenerateSingleT)(
      prompt, pos, prefix_end, model_.Config(), runtime_config, aes_ctr_engine_,
      weights_, kv_cache, env, timing_info);

  env.ctx.pools.MaybeStopSpinning(runtime_config.use_spinning);
}

void Gemma::GenerateBatch(const RuntimeConfig& runtime_config,
                          AllQueries& all_queries, MatMulEnv& env,
                          TimingInfo& timing_info) const {
  env.ctx.pools.MaybeStartSpinning(runtime_config.use_spinning);

  HWY_DYNAMIC_DISPATCH(GenerateBatchT)(model_.Config(), runtime_config,
                                       aes_ctr_engine_, weights_, all_queries,
                                       env, timing_info);

  env.ctx.pools.MaybeStopSpinning(runtime_config.use_spinning);
}

void Gemma::GenerateImageTokens(const RuntimeConfig& runtime_config,
                                size_t seq_len, const Image& image,
                                ImageTokens& image_tokens, MatMulEnv& env,
                                TimingInfo& timing_info) const {
  env.ctx.pools.MaybeStartSpinning(runtime_config.use_spinning);

  HWY_DYNAMIC_DISPATCH(GenerateImageTokensT)(model_.Config(), runtime_config,
                                             seq_len, weights_, image,
                                             image_tokens, env, timing_info);

  env.ctx.pools.MaybeStopSpinning(runtime_config.use_spinning);
}

ContinuousQBatch::ContinuousQBatch(size_t max_size, AllQueries& queries)
    : QBatch(0, max_size, queries) {
  for (size_t i = start_; i < queries_.NumQueries(); ++i) {
    if (!queries_[i].kv_cache.IsEmpty()) {
      // Put the kv_cache to the available_kv_caches_ instead; leaving the
      // kv_cache in the queries_ is very confusing. This simplifies the logic
      // of kv_cache management.
      available_kv_caches_.push_back(queries_[i].kv_cache);
      queries_[i].kv_cache = KVCachePtr();
    }
  }
}

bool ContinuousQBatch::ShouldPrefill() const {
  const bool no_available_to_insert = next_to_insert_ == next_to_prefill_;
  const int more_queries_to_prefill = next_to_prefill_ < queries_.NumQueries();
  return no_available_to_insert && more_queries_to_prefill;
}

void ContinuousQBatch::SetupNextBatchForPrefill() {
  start_ = next_to_prefill_;
  size_ = HWY_MIN(max_size_, queries_.NumQueries() - start_);
  HWY_DASSERT(size_ != 0);
  HWY_DASSERT(start_ + size_ <= queries_.NumQueries());
  query_idx_.clear();
  query_idx_.reserve(size_);
  for (size_t i = 0; i < size_; ++i) {
    const size_t next_query_idx = start_ + i;
    query_idx_.push_back(next_query_idx);
    HWY_ASSERT(queries_[next_query_idx].kv_cache.IsEmpty());
    queries_[next_query_idx].kv_cache = available_kv_caches_.back();
    available_kv_caches_.pop_back();
  }
  next_to_prefill_ += size_;
}

std::optional<size_t> ContinuousQBatch::GetNextToInsert() {
  if (next_to_insert_ == next_to_prefill_) {
    return std::nullopt;
  }
  next_to_insert_++;
  return next_to_insert_ - 1;
}

void ContinuousQBatch::MaybeReleaseKV(const QBatch& from) {
  const int query_to_collect = from.QueryIdx(0);
  // Only collect if the query to collect is not the same as the next query to
  // insert. This happens at the beginning of each Generate call.
  if (query_to_collect != next_to_insert_) {
    // Only clear the KV cache if there are more queries to insert; Otherwise
    // we get a crash because Transformer will still access that KV cache.
    if (next_to_insert_ < queries_.NumQueries()) {
      available_kv_caches_.push_back(from.KV(0));
      ZeroInit(from.KV(0).kv_cache);
      from.KV(0) = KVCachePtr();
    }
  }
}

}  // namespace gcpp
#endif  // HWY_ONCE
