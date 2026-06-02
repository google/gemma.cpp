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

#include <math.h>  // sqrtf
#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#include "gemma/flash_structs.h"
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include "gemma/activations.h"
#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "gemma/weights.h"
#include "paligemma/image.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"

// Compiles this file for multiple architectures via "foreach_target.h", to
// which we pass the filename via macro 'argument'.
// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/vit.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "gemma/attention.h"
#include "gemma/flash_attention.h"
#include "gemma/gemma-inl.h"
#include "ops/ops-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// Wrapper class; holds arguments in member variables to shorten call sites.
// The main differences to GemmaAttention are:
// - no KV Cache necessary, attention is always all-to-all and not causal.
// - no potential wrap-around, attention always goes from 0 to kSeqLen.
// - no need for batching, as we are always computing attention for kSeqLen
//   tokens.
// This results in a much simpler implementation. However, to avoid duplicating
// code, we should still consider merging the two classes.
// TODO(keysers): Refactor to share code with GemmaAttention.
class VitAttention {
  // Computes Q, K, V for all heads, stored in activations_.q.
  HWY_NOINLINE void ComputeQKV() {
    PROFILER_ZONE("Gen.VitAttention.QKV");
    auto& qkv = activations_.attention.q;
    HWY_ASSERT(qkv.Rows() == num_tokens_);
    HWY_ASSERT(qkv.Cols() == layer_config_.heads * 3 * layer_config_.qkv_dim);
    CallMatMul(activations_.attention.pre_att_rms_out, layer_.vit.qkv_einsum_w,
               layer_.vit.qkv_einsum_b.PackedScale1(), env_, qkv);
  }

  // Applies the query scale to the query and converts to QType.
  template <typename QKVType, typename QType>
  void ScaleQuery(const MatPtrT<QKVType>& qkv, const size_t num_tokens,
                  const size_t heads, const size_t qkv_dim,
                  const float query_scale, MatPtrT<QType>& q_output) {
    ParallelFor(Parallelism::kWithinCluster, heads, env_.ctx,
                /*cluster_idx=*/0, Callers::kFlashAttention,
                [&](size_t head, size_t worker) {
                  size_t q_offset = head * qkv_dim;
                  for (size_t token = 0; token < num_tokens; ++token) {
                    const float* HWY_RESTRICT src_q =
                        qkv.Row(token) + q_offset * 3;
                    QType* HWY_RESTRICT dst_q = q_output.Row(token) + q_offset;
                    for (size_t i = 0; i < qkv_dim; ++i) {
                      dst_q[i] = hwy::ConvertScalarTo<QType>(
                          hwy::ConvertScalarTo<float>(src_q[i]) * query_scale);
                    }
                  }
                });
  }

  // Transposes K and V and converts to KVType.
  template <typename QKVType, typename KVType>
  void TransposeKAndV(const MatPtrT<QKVType>& qkv, const size_t num_tokens,
                      const size_t heads, const size_t qkv_dim,
                      MatPtrT<KVType>& k_output, MatPtrT<KVType>& v_output) {
    using DF = hn::ScalableTag<float>;
    const DF df;
    const size_t kNF = hn::Lanes(df);
    const size_t kNumTokensH = hwy::DivCeil(num_tokens, 2 * kNF);
    const size_t kRoundedKVDim = hwy::RoundUpTo(qkv_dim, 2 * kNF);
    ParallelFor(
        Parallelism::kWithinCluster, heads, env_.ctx,
        /*cluster_idx=*/0, Callers::kFlashAttention,
        [&](size_t head, size_t worker) {
          const size_t qkv_offset = head * 3 * qkv_dim;
          const size_t k_or_v_offset = head * 2 * kNF * kRoundedKVDim;
          for (size_t token_h = 0; token_h < kNumTokensH; ++token_h) {
            KVType* HWY_RESTRICT dst_k = k_output.Row(token_h);
            KVType* HWY_RESTRICT dst_v = v_output.Row(token_h);
            size_t dst_k_index = k_or_v_offset;
            for (size_t q = 0; q < qkv_dim; q += 2) {
              for (size_t token_l = 0; token_l < 2 * kNF;
                   ++token_l, dst_k_index += 2) {
                const QKVType* HWY_RESTRICT src_k =
                    qkv.Row(token_h * 2 * kNF + token_l) + qkv_offset + qkv_dim;
                dst_k[dst_k_index] = hwy::ConvertScalarTo<KVType>(src_k[q]);
                dst_k[dst_k_index + 1] =
                    hwy::ConvertScalarTo<KVType>(src_k[q + 1]);
              }
            }
            size_t dst_v_index = k_or_v_offset;
            for (size_t q = 0; q < qkv_dim; q += 2 * kNF) {
              for (size_t token_l = 0; token_l < 2 * kNF; ++token_l) {
                const QKVType* HWY_RESTRICT src_v =
                    qkv.Row(token_h * 2 * kNF + token_l) + qkv_offset +
                    qkv_dim * 2;
                if (q + 2 * kNF <= qkv_dim) {
                  for (size_t q_l = 0; q_l < 2 * kNF; ++q_l) {
                    dst_v[dst_v_index++] =
                        hwy::ConvertScalarTo<KVType>(src_v[q + q_l]);
                  }
                } else {
                  for (size_t q_l = 0; q_l < qkv_dim - q; ++q_l) {
                    dst_v[dst_v_index++] =
                        hwy::ConvertScalarTo<KVType>(src_v[q + q_l]);
                  }
                }
              }
            }
            // Zero out the padding area.
            // In the loops above, the dst_k loop has written 2kNF x 2
            // consecutive elements for each q +=2, and the dst_v loop has
            // written 2kNF x 2kNF consecutive elements for each q += 2 * kNF.
            // Both of them therefore write 2kNF elements for each increment of
            // q, so we can combine both into a single loop for the padding.
            // This could be further simplified by writing a zero vector.
            for (size_t q = qkv_dim; q < kRoundedKVDim; ++q) {
              for (size_t token_l = 0; token_l < 2 * kNF; ++token_l) {
                dst_k[dst_k_index++] = hwy::ConvertScalarTo<KVType>(0.0f);
                dst_v[dst_v_index++] = hwy::ConvertScalarTo<KVType>(0.0f);
              }
            }
          }
        });
  }

  // Computes the flash attention parameters. This is mostly about deciding on
  // the tile sizes and filling the param structs with the correct offsets.
  template <typename QType, typename KVType>
  void ComputeParams(const uint32_t num_tokens, const size_t seq_len,
                     const size_t heads, const uint32_t qkv_dim,
                     const MatPtrT<QType>& q, const MatPtrT<KVType>& k,
                     const MatPtrT<KVType>& v, const MatPtrT<float>& att_out,
                     std::vector<Tile148Params>& flash_params) {
    flash_params.clear();
    for (uint32_t head = 0; head < heads; ++head) {
      uint32_t token = 0;
      while (token + k8xNFVTileSize <= num_tokens) {
        flash_params.push_back(Tile148Params{
            .v_tile_size = k8xNFVTileSize,
            .qi_index = token,
            .kv_head = head,
        });
        token += k8xNFVTileSize;
      }
      if (token + k4xNFVTileSize <= num_tokens) {
        flash_params.push_back(Tile148Params{
            .v_tile_size = k4xNFVTileSize,
            .qi_index = token,
            .kv_head = head,
        });
        token += k4xNFVTileSize;
      }
      while (token < num_tokens) {
        flash_params.push_back(Tile148Params{
            .v_tile_size = 1,
            .qi_index = token,
            .kv_head = head,
        });
        token += 1;
      }
    }
    for (auto& param : flash_params) {
      param.min_start_pos = 0;
      param.max_last_pos = num_tokens - 1;
      for (size_t i = 0; i < param.v_tile_size; ++i) {
        param.q_offsets[i] =
            q.Row(param.qi_index + i) + param.kv_head * qkv_dim - q.Row(0);
        param.out_offsets[i] = att_out.Row(param.qi_index + i) +
                               param.kv_head * qkv_dim - att_out.Row(0);
        param.start_pos[i] = 0;
        param.last_pos[i] = num_tokens - 1;
      }
    }
  }

  // Runs the flash attention algorithm on Q, K, V.
  HWY_NOINLINE void FlashAttention() {
    GCPP_ZONE(env_.ctx, 0, Zones::kVitFlashAttentionInclusive);
    const size_t qkv_dim = layer_config_.qkv_dim;
    const size_t heads = layer_config_.heads;
    HWY_ASSERT_M(heads == layer_config_.kv_heads, "Vit expects MHA");
    const size_t kNF = FloatsPerVector();
    const size_t kRoundedKVDim = hwy::RoundUpTo(qkv_dim, 2 * kNF);
    auto& attn = activations_.attention;
    const size_t seq_len = static_cast<size_t>(attn.div_seq_len.GetDivisor());
    if (attn.vit_K_T.Rows() >= seq_len) {
      attn.vit_K_T.ReshapePackedRowsToCols(2 * kNF);
      attn.vit_V_T.ReshapePackedRowsToCols(2 * kNF);
    }
    const float query_scale = 1.0f / sqrtf(static_cast<float>(qkv_dim));
    ScaleQuery(attn.q, num_tokens_, heads, qkv_dim, query_scale, attn.q_bf);
    TransposeKAndV(attn.q, num_tokens_, heads, qkv_dim, attn.vit_K_T,
                   attn.vit_V_T);
    ComputeParams(num_tokens_, seq_len, heads, qkv_dim, attn.q_bf, attn.vit_K_T,
                  attn.vit_V_T, attn.att_out, attn.flash_params);
    size_t num_tasks = attn.flash_params.size();

    // For each param, compute fused flash Q.K, softmax and weighted V.
    const auto func = [&, &ctx = env_.ctx](const size_t task,
                                           size_t worker) HWY_ATTR {
      GCPP_ZONE(ctx, worker, Zones::kFlashAttentionFlashAttention);
      auto& param = attn.flash_params[task];
      MatPtrT<KV_t> kT("k_T_view", Extents2D(hwy::DivCeil(seq_len, 2 * kNF),
                                             kRoundedKVDim * 2 * kNF));
      kT.SetPtr(attn.vit_K_T.Row(0) + param.kv_head * kRoundedKVDim * 2 * kNF,
                attn.vit_K_T.Stride());
      MatPtrT<KV_t> vT("v_T_view", Extents2D(hwy::DivCeil(seq_len, 2 * kNF),
                                             kRoundedKVDim * 2 * kNF));
      vT.SetPtr(attn.vit_V_T.Row(0) + param.kv_head * kRoundedKVDim * 2 * kNF,
                attn.vit_V_T.Stride());
      DispatchDispatchTileFlashAttention148(
          param, attn.q_bf, kT, vT, /*layer_idx=*/0, attn, attn.att_out,
          qkv_dim, ctx, worker, /*attention_impl=*/AttentionImpl::kFlash);
    };

    {
      PROFILER_ZONE("Gen.VitFlashAttention.ForkJoin");
      // Full parallelism is helpful, SmallParallelFor is insufficient.
      HierarchicalParallelFor(num_tasks, env_.ctx, Callers::kFlashAttention,
                              func);
    }
  }

  // Sums encoded (`att_out`) over num_heads (`layer_config_.heads`) and
  // head_dim (`qkv_dim`) into output (`att_sums`).
  HWY_NOINLINE void SumHeads() {
    auto* bias = layer_.vit.attn_out_b.PackedScale1();
    // att_weights and att_out are concatenated heads, each of length
    // qkv_dim. Thus the [num_tokens_, layer_config_.model_dim]
    // matmul output is the sum over heads.
    CallMatMul(activations_.attention.att_out, layer_.vit.attn_out_w, bias,
               env_, activations_.attention.att_sums);
  }

 public:
  VitAttention(size_t num_tokens, size_t layer_idx, Activations& activations,
               const LayerWeightsPtrs& layer, MatMulEnv& env)
      : num_tokens_(num_tokens),
        activations_(activations),
        layer_(layer),
        layer_config_(layer.layer_config),
        env_(env),
        pool_(env_.ctx.pools.Pool(0)),
        caller1_(env_.ctx.pool_callers.Get(Callers::kVitDotSoftmax1)),
        caller2_(env_.ctx.pool_callers.Get(Callers::kVitDotSoftmax2)),
        caller3_(env_.ctx.pool_callers.Get(Callers::kVitDotSoftmax3)),
        caller4_(env_.ctx.pool_callers.Get(Callers::kVitDotSoftmax4)) {}

  HWY_INLINE void operator()() {
    ComputeQKV();
    FlashAttention();
    SumHeads();
  }

 private:
  const size_t num_tokens_;
  Activations& activations_;
  const LayerWeightsPtrs& layer_;
  const LayerConfig& layer_config_;
  MatMulEnv& env_;
  hwy::ThreadPool& pool_;
  hwy::pool::Caller caller1_;
  hwy::pool::Caller caller2_;
  hwy::pool::Caller caller3_;
  hwy::pool::Caller caller4_;
};

// Same as FFWNoVit, but with different layer members and no second
// gating matrix.
void FFWVit(const LayerWeightsPtrs& layer, Activations& activations,
            MatMulEnv& env) {
  PROFILER_ZONE("Gen.FFW.ViT");
  const LayerConfig& layer_config = layer.layer_config;

  const bool add_bias = layer_config.ff_biases;
  const float* bias1 = add_bias ? layer.vit.linear_0_b.PackedScale1() : nullptr;
  const float* output_bias =
      add_bias ? layer.vit.linear_1_b.PackedScale1() : nullptr;

  // Compute the hidden layer activations.
  CallMatMul(activations.pre_ffw_rms_out, layer.vit.linear_0_w, bias1, env,
             activations.C1);

  // Activation (Gelu), store in C1.
  ActivationBatched(layer_config.activation, activations.C1, env.ctx);

  // Hidden layer -> output layer.
  CallMatMul(activations.C1, layer.vit.linear_1_w, output_bias, env,
             activations.ffw_out);
}

// Vit transformer layer. Some comments below refer to the Vit implementation in
// the Big Vision codebase. See
// github.com/google-research/big_vision/blob/main/big_vision/models/vit.py
// TODO(keysers): consider adding a wrapper for both LayerNorm with RMSNorm and
// try merging this with TransformerLayer.
void VitTransformerLayer(size_t num_tokens, const size_t layer_idx,
                         const LayerWeightsPtrs& layer,
                         Activations& activations, MatMulEnv& env) {
  const size_t model_dim = activations.attention.config.model_dim;
  auto type = layer.layer_config.type;
  HWY_DASSERT(type == LayerAttentionType::kVit);
  (void)type;
  (void)model_dim;

  auto& x = activations.x;
  HWY_DASSERT(x.Rows() == num_tokens);
  HWY_DASSERT(x.Cols() == model_dim);

  // y = nn.LayerNorm()(x)
  // y ~ pre_att_rms_out
  LayerNormBatched(x, layer.vit.layer_norm_0_scale, layer.vit.layer_norm_0_bias,
                   activations.attention.pre_att_rms_out);

  // y = out["sa"] = nn.MultiHeadDotProductAttention(...)(y, y)
  // y ~ att_sums
  VitAttention(num_tokens, layer_idx, activations, layer, env)();

  // x = out["+sa"] = x + y
  AddFromBatched(activations.attention.att_sums, x, env.ctx);

  // y = nn.LayerNorm()(x)
  // y ~ pre_ffw_rms_out
  LayerNormBatched(x, layer.vit.layer_norm_1_scale, layer.vit.layer_norm_1_bias,
                   activations.pre_ffw_rms_out);

  // y = out["mlp"] = MlpBlock(...)(y)
  // y ~ ffw_out
  FFWVit(layer, activations, env);

  // x = out["+mlp"] = x + y
  AddFromBatched(activations.ffw_out, x, env.ctx);
}

// Gets the patches of the image and embeds them with the image embedding
// kernel. The result is stored in activations.x.
static HWY_NOINLINE void EmbedImagePatches(const Image& image,
                                           const ModelConfig& model_config,
                                           const WeightsPtrs& weights,
                                           Activations& activations,
                                           MatMulEnv& env) {
  const size_t model_dim = model_config.vit_config.model_dim;
  const size_t patch_width = model_config.vit_config.patch_width;
  const size_t num_tokens = model_config.vit_config.seq_len;
  const size_t patch_area = patch_width * patch_width * 3;
  const hwy::Divisor div_patch_dim(patch_width);
  HWY_DASSERT(weights.vit_img_embedding_kernel.Rows() == model_dim);
  HWY_DASSERT(weights.vit_img_embedding_kernel.Cols() == patch_area);
  HWY_DASSERT(activations.x.Cols() == model_dim);
  (void)model_dim;
  // img/embedding/kernel has original shape (14, 14, 3, 1152)
  // H x W x C x D transposed to D x (H x W x C) so here (1152, 14 * 14 * 3)
  // image_patches is (256, 14 * 14 * 3)
  // Must be padded, see `DoDecompressA`.
  MatStorageT<float> image_patches("patches", Extents2D(num_tokens, patch_area),
                                   env.ctx.allocator, MatPadding::kOdd);
  for (size_t i = 0; i < num_tokens; ++i) {
    image.GetPatch(i, div_patch_dim, image_patches.Row(i));
  }
  CallMatMul(image_patches, weights.vit_img_embedding_kernel,
             weights.vit_img_embedding_bias.PackedScale1(), env, activations.x);
  // Add position embeddings.
  CallUpcastedActivation(&weights.vit_img_pos_embedding,
                         [&](const auto* weights_t) {
                           AddFromBatched(*weights_t, activations.x, env.ctx);
                         });
}

// Prefills the image tokens with the ViT encoder.
void PrefillVit(const ModelConfig& model_config, const WeightsPtrs& weights,
                const RuntimeConfig& runtime_config, const Image& image,
                ImageTokens& image_tokens, Activations& activations,
                MatMulEnv& env) {
  PROFILER_ZONE("Gen.PrefillVit");
  const size_t num_tokens = model_config.vit_config.seq_len;
  const size_t vit_model_dim = model_config.vit_config.model_dim;
  HWY_ASSERT(num_tokens == activations.x.Rows());
  // Embed the image patches.
  EmbedImagePatches(image, model_config, weights, activations, env);
  // Go through all layers.
  for (size_t layer_idx = 0;
       layer_idx < model_config.vit_config.layer_configs.size(); ++layer_idx) {
    VitTransformerLayer(num_tokens, layer_idx, *weights.VitLayer(layer_idx),
                        activations, env);
  }
  // Final Layernorm.
  LayerNormBatched(activations.x, weights.vit_encoder_norm_scale,
                   weights.vit_encoder_norm_bias, activations.x);

  if (model_config.wrapping == PromptWrapping::GEMMA_VLM) {
    activations.x = AvgPool4x4(activations.x, env.ctx.allocator);

    // Apply soft embedding norm before input projection.
    CallUpcasted(&weights.mm_embed_norm, [&](const auto* weights_t) {
      RMSNormInplace(weights_t->PackedScale1(), /*w_ofs=*/0,
                     activations.x.Row(0), vit_model_dim, env.ctx,
                     hwy::Profiler::GlobalIdx());
    });
  }

  // Apply head embedding into image_tokens of size of the LLM kModelDim.
  CallMatMul(activations.x, weights.vit_img_head_kernel,
             weights.vit_img_head_bias.PackedScale1(), env, image_tokens);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();
