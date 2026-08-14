// Copyright 2026 Google LLC
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
#define HWY_TARGET_INCLUDE "gemma/vit_gemma4.cc" // NOLINT
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

void AssertValidImage(const ModelConfig& model_config, const Image& image) {
  HWY_ASSERT(image.width() > 0 && image.height() > 0);
  HWY_ASSERT(image.width() <= 65536 && image.height() <= 65536);
  HWY_ASSERT(image.size() ==
             static_cast<size_t>(image.width()) * image.height() * 3);
  HWY_ASSERT(image.data() != nullptr);
  HWY_ASSERT(model_config.vit_config.patch_width > 0 &&
             model_config.vit_config.patch_width <= 256);
  HWY_ASSERT(model_config.vit_config.seq_len > 0);
  HWY_ASSERT(model_config.vit_config.pool_dim > 0 &&
             model_config.vit_config.pool_dim <= 16);
  HWY_ASSERT(model_config.vit_config.seq_len >=
             model_config.vit_config.pool_dim * model_config.vit_config.pool_dim);
}

void GetAspectRatioPreservingSize(int height, int width, int patch_size,
                                  int max_patches, int pooling_kernel_size,
                                  int& target_height, int& target_width) {
  HWY_ASSERT(height > 0 && width > 0);
  HWY_ASSERT(height <= 65536 && width <= 65536);
  HWY_ASSERT(patch_size > 0 && patch_size <= 256);
  HWY_ASSERT(max_patches > 0);
  HWY_ASSERT(pooling_kernel_size > 0 && pooling_kernel_size <= 16);
  HWY_ASSERT(max_patches >= pooling_kernel_size * pooling_kernel_size);

  float total_px = static_cast<float>(height) * static_cast<float>(width);
  float target_px = max_patches * (patch_size * patch_size);
  float factor = std::sqrt(target_px / total_px);
  float ideal_height = factor * height;
  float ideal_width = factor * width;
  int side_mult = pooling_kernel_size * patch_size;  // 3 * 16 = 48

  target_height =
      static_cast<int>(std::floor(ideal_height / side_mult)) * side_mult;
  target_width =
      static_cast<int>(std::floor(ideal_width / side_mult)) * side_mult;

  if (target_height == 0 && target_width == 0) {
    HWY_ABORT("Attempting to resize to a 0 x 0 image.");
  }

  int max_side_length =
      (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;
  if (target_height == 0) {
    target_height = side_mult;
    target_width = std::min(
        static_cast<int>(std::floor(static_cast<float>(width) / height)) *
            side_mult,
        max_side_length);
  } else if (target_width == 0) {
    target_width = side_mult;
    target_height = std::min(
        static_cast<int>(std::floor(static_cast<float>(height) / width)) *
            side_mult,
        max_side_length);
  }

  if (target_height * target_width > target_px) {
    HWY_ABORT("Resized image exceeds max patches.");
  }
}

static HWY_NOINLINE void EmbedImagePatchesGemma4(
    const Image& image, const ModelConfig& model_config,
    const WeightsPtrs& weights, Activations& activations, MatMulEnv& env) {
  const size_t model_dim = model_config.vit_config.model_dim;      // 768
  const size_t patch_width = model_config.vit_config.patch_width;  // 16
  const size_t max_patches = model_config.vit_config.seq_len;      // 2520
  const size_t patch_area = patch_width * patch_width * 3;         // 768
  const hwy::Divisor div_patch_dim(patch_width);

  const size_t num_real_patches =
      (image.height() / patch_width) * (image.width() / patch_width);
  HWY_ASSERT(num_real_patches <= max_patches);

  MatStorageT<float> image_patches("patches",
                                   Extents2D(max_patches, patch_area),
                                   env.ctx.allocator, MatPadding::kOdd);
  gcpp::ZeroInit(image_patches);

  for (size_t i = 0; i < num_real_patches; ++i) {
    float* row = image_patches.Row(i);
    image.GetPatch(i, div_patch_dim, row);
  }

  CallMatMul(image_patches, weights.vit_img_embedding_kernel,
             weights.vit_img_embedding_bias.PackedScale1(), env, activations.x);

  const size_t real_patch_width = image.width() / patch_width;

  CallUpcastedActivation(
      &weights.vit_img_pos_embedding, [&](const auto* pos_emb_t) {
        for (size_t i = 0; i < num_real_patches; ++i) {
          size_t y = i / real_patch_width;
          size_t x = i % real_patch_width;

          HWY_DASSERT(x < 10240);
          HWY_DASSERT(y < 10240);

          const auto* x_emb = pos_emb_t->Row(2 * x);
          const auto* y_emb = pos_emb_t->Row(2 * y + 1);
          float* act_row = activations.x.Row(i);

          for (size_t d = 0; d < model_dim; ++d) {
            act_row[d] +=
                hwy::ConvertScalarTo<float>(x_emb[d]) + hwy::ConvertScalarTo<float>(y_emb[d]);
          }
        }
      });
}

class VitGemma4Attention {
 public:
  VitGemma4Attention(size_t num_tokens, size_t num_real_patches,
                     size_t layer_idx, Activations& activations,
                     const LayerWeightsPtrs& layer, MatMulEnv& env,
                     size_t real_patch_width)
      : num_tokens_(num_tokens),
        num_real_patches_(num_real_patches),
        activations_(activations),
        layer_(layer),
        layer_config_(layer.layer_config),
        env_(env),
        real_patch_width_(real_patch_width),
        pool_(env_.ctx.pools.Pool(0)),
        caller1_(env_.ctx.pool_callers.Get(Callers::kVitDotSoftmax1)),
        caller2_(env_.ctx.pool_callers.Get(Callers::kVitDotSoftmax2)),
        caller3_(env_.ctx.pool_callers.Get(Callers::kVitDotSoftmax3)),
        caller4_(env_.ctx.pool_callers.Get(Callers::kVitDotSoftmax4)) {}

  HWY_INLINE void operator()() {
    ComputeQKV();
    ApplyQKVNorm();
    FlashAttention();
    SumHeads();
  }

 private:
  HWY_NOINLINE void ApplyQKVNorm() {
    PROFILER_ZONE("Gen.VitGemma4Attention.QKVNorm");
    const size_t heads = layer_config_.heads;
    const size_t kv_heads = layer_config_.kv_heads;
    const size_t qkv_dim = layer_config_.qkv_dim;
    auto& qkv = activations_.attention.q;

    float inv_timescale[16];
    for (size_t dim = 0; dim < 16; ++dim) {
      inv_timescale[dim] =
          1.0f / std::pow(100.0f, static_cast<float>(2 * dim) / 32.0f);
    }

    // Q-norm
    if (layer_.query_norm_scale.HasPtr()) {
      CallUpcasted(&layer_.query_norm_scale, [&](const auto* weights_t) {
        ParallelFor(Parallelism::kWithinCluster, num_real_patches_, env_.ctx,
                    /*cluster_idx=*/0, Callers::kFlashAttention,
                    [&](size_t token, size_t worker) {
                      const size_t y = token / real_patch_width_;
                      const size_t x = token % real_patch_width_;
                      for (size_t h = 0; h < heads; ++h) {
                        float* q_ptr = qkv.Row(token) + h * qkv_dim;
                        RMSNormInplace(weights_t->PackedScale1(), /*w_ofs=*/0,
                                       q_ptr, qkv_dim, env_.ctx, worker);
                        Rope(q_ptr, 32, inv_timescale, static_cast<int>(x),
                             env_.ctx, worker);
                        Rope(q_ptr + 32, 32, inv_timescale, static_cast<int>(y),
                             env_.ctx, worker);
                      }
                    });
      });
    } else if (layer_config_.use_qk_norm) {
      ParallelFor(Parallelism::kWithinCluster, num_real_patches_, env_.ctx,
                  /*cluster_idx=*/0, Callers::kFlashAttention,
                  [&](size_t token, size_t worker) {
                    const size_t y = token / real_patch_width_;
                    const size_t x = token % real_patch_width_;
                    for (size_t h = 0; h < heads; ++h) {
                      float* q_ptr = qkv.Row(token) + h * qkv_dim;
                      RMSNormNoScaleInplace(q_ptr, qkv_dim, env_.ctx, worker);
                      Rope(q_ptr, 32, inv_timescale, static_cast<int>(x),
                           env_.ctx, worker);
                      Rope(q_ptr + 32, 32, inv_timescale, static_cast<int>(y),
                           env_.ctx, worker);
                    }
                  });
    }

    // K-norm
    if (layer_.key_norm_scale.HasPtr()) {
      CallUpcasted(&layer_.key_norm_scale, [&](const auto* weights_t) {
        ParallelFor(Parallelism::kWithinCluster, num_real_patches_, env_.ctx,
                    /*cluster_idx=*/0, Callers::kFlashAttention,
                    [&](size_t token, size_t worker) {
                      const size_t y = token / real_patch_width_;
                      const size_t x = token % real_patch_width_;
                      for (size_t h = 0; h < kv_heads; ++h) {
                        float* k_ptr =
                            qkv.Row(token) + heads * qkv_dim + 2 * h * qkv_dim;
                        RMSNormInplace(weights_t->PackedScale1(), /*w_ofs=*/0,
                                       k_ptr, qkv_dim, env_.ctx, worker);
                        Rope(k_ptr, 32, inv_timescale, static_cast<int>(x),
                             env_.ctx, worker);
                        Rope(k_ptr + 32, 32, inv_timescale, static_cast<int>(y),
                             env_.ctx, worker);
                      }
                    });
      });
    } else if (layer_config_.use_qk_norm) {
      ParallelFor(Parallelism::kWithinCluster, num_real_patches_, env_.ctx,
                  /*cluster_idx=*/0, Callers::kFlashAttention,
                  [&](size_t token, size_t worker) {
                    const size_t y = token / real_patch_width_;
                    const size_t x = token % real_patch_width_;
                    for (size_t h = 0; h < kv_heads; ++h) {
                      float* k_ptr =
                          qkv.Row(token) + heads * qkv_dim + 2 * h * qkv_dim;
                      RMSNormNoScaleInplace(k_ptr, qkv_dim, env_.ctx, worker);
                      Rope(k_ptr, 32, inv_timescale, static_cast<int>(x),
                           env_.ctx, worker);
                      Rope(k_ptr + 32, 32, inv_timescale, static_cast<int>(y),
                           env_.ctx, worker);
                    }
                  });
    }

    ParallelFor(Parallelism::kWithinCluster, num_real_patches_, env_.ctx,
                /*cluster_idx=*/0, Callers::kFlashAttention,
                [&](size_t token, size_t worker) {
                  for (size_t h = 0; h < kv_heads; ++h) {
                    float* v_ptr = qkv.Row(token) + heads * qkv_dim +
                                   (2 * h + 1) * qkv_dim;
                    RMSNormNoScaleInplace(v_ptr, qkv_dim, env_.ctx, worker);
                  }
                });
  }

  HWY_NOINLINE void ComputeQKV() {
    PROFILER_ZONE("Gen.VitGemma4Attention.QKV");
    auto& qkv = activations_.attention.q;
    const size_t heads = layer_config_.heads;
    const size_t kv_heads = layer_config_.kv_heads;
    const size_t qkv_dim = layer_config_.qkv_dim;

    HWY_ASSERT(qkv.Rows() == num_tokens_);
    HWY_ASSERT(qkv.Cols() == (heads + 2 * kv_heads) * qkv_dim);

    // Q part:
    MatPtrT<float> q_part("q_part", Extents2D(num_tokens_, heads * qkv_dim));
    q_part.SetPtr(qkv.Row(0), qkv.Stride());
    CallMatMul(activations_.attention.pre_att_rms_out, layer_.qkv_einsum_w1,
               /*bias=*/nullptr, env_, q_part);

    // KV part:
    MatPtrT<float> kv_part("kv_part",
                           Extents2D(num_tokens_, 2 * kv_heads * qkv_dim));
    kv_part.SetPtr(qkv.Row(0) + heads * qkv_dim, qkv.Stride());
    CallMatMul(activations_.attention.pre_att_rms_out, layer_.qkv_einsum_w2,
               /*bias=*/nullptr, env_, kv_part);
  }

  // Applies the query scale to the query and converts to QType.
  template <typename QKVType, typename QType>
  void ScaleQuery(const MatPtrT<QKVType>& qkv, const size_t num_tokens,
                  const size_t heads, const size_t qkv_dim,
                  const float query_scale, MatPtrT<QType>& q_output) {
    ParallelFor(
        Parallelism::kWithinCluster, heads, env_.ctx,
        /*cluster_idx=*/0, Callers::kFlashAttention,
        [&](size_t head, size_t worker) {
          size_t src_q_offset = head * qkv_dim;
          size_t dst_q_offset = head * qkv_dim;
          for (size_t token = 0; token < num_tokens; ++token) {
            const float* HWY_RESTRICT src_q = qkv.Row(token) + src_q_offset;
            QType* HWY_RESTRICT dst_q = q_output.Row(token) + dst_q_offset;
            if (token < num_real_patches_) {
              for (size_t i = 0; i < qkv_dim; ++i) {
                dst_q[i] = hwy::ConvertScalarTo<QType>(
                    hwy::ConvertScalarTo<float>(src_q[i]) * query_scale);
              }
            } else {
              for (size_t i = 0; i < qkv_dim; ++i) {
                dst_q[i] = hwy::ConvertScalarTo<QType>(0.0f);
              }
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
          const size_t k_offset = heads * qkv_dim + 2 * head * qkv_dim;
          const size_t v_offset = heads * qkv_dim + (2 * head + 1) * qkv_dim;

          const size_t k_or_v_output_offset = head * 2 * kNF * kRoundedKVDim;
          for (size_t token_h = 0; token_h < kNumTokensH; ++token_h) {
            KVType* HWY_RESTRICT dst_k = k_output.Row(token_h);
            KVType* HWY_RESTRICT dst_v = v_output.Row(token_h);
            size_t dst_k_index = k_or_v_output_offset;
            for (size_t q = 0; q < qkv_dim; q += 2) {
              for (size_t token_l = 0; token_l < 2 * kNF;
                   ++token_l, dst_k_index += 2) {
                const size_t token_idx = token_h * 2 * kNF + token_l;
                if (token_idx < num_real_patches_) {
                  const QKVType* HWY_RESTRICT src_k =
                      qkv.Row(token_idx) + k_offset;
                  dst_k[dst_k_index] = hwy::ConvertScalarTo<KVType>(src_k[q]);
                  dst_k[dst_k_index + 1] =
                      hwy::ConvertScalarTo<KVType>(src_k[q + 1]);
                } else {
                  dst_k[dst_k_index] = hwy::ConvertScalarTo<KVType>(0.0f);
                  dst_k[dst_k_index + 1] = hwy::ConvertScalarTo<KVType>(0.0f);
                }
              }
            }
            size_t dst_v_index = k_or_v_output_offset;
            for (size_t q = 0; q < qkv_dim; q += 2 * kNF) {
              for (size_t token_l = 0; token_l < 2 * kNF; ++token_l) {
                const size_t token_idx = token_h * 2 * kNF + token_l;
                if (token_idx < num_real_patches_) {
                  const QKVType* HWY_RESTRICT src_v =
                      qkv.Row(token_idx) + v_offset;
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
                } else {
                  const size_t cnt =
                      (q + 2 * kNF <= qkv_dim) ? (2 * kNF) : (qkv_dim - q);
                  for (size_t q_l = 0; q_l < cnt; ++q_l) {
                    dst_v[dst_v_index++] = hwy::ConvertScalarTo<KVType>(0.0f);
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
    const float query_scale = 1.0f;
    ScaleQuery(attn.q, num_tokens_, heads, qkv_dim, query_scale, attn.q_bf);
    TransposeKAndV(attn.q, num_tokens_, heads, qkv_dim, attn.vit_K_T,
                   attn.vit_V_T);
    // Zero-out the remaining query outputs to prevent garbage propagation
    for (size_t token = num_real_patches_; token < num_tokens_; ++token) {
      float* out_row = attn.att_out.Row(token);
      for (size_t d = 0; d < heads * qkv_dim; ++d) {
        out_row[d] = 0.0f;
      }
    }
    ComputeParams(num_real_patches_, num_real_patches_, heads, qkv_dim,
                  attn.q_bf, attn.vit_K_T, attn.vit_V_T, attn.att_out,
                  attn.flash_params);
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
      PROFILER_ZONE("Gen.VitGemma4FlashAttention.ForkJoin");
      // Full parallelism is helpful, SmallParallelFor is insufficient.
      HierarchicalParallelFor(num_tasks, env_.ctx, Callers::kFlashAttention,
                              func);
    }
  }

  // Sums encoded (`att_out`) over num_heads (`layer_config_.heads`) and
  // head_dim (`qkv_dim`) into output (`att_sums`).
  HWY_NOINLINE void SumHeads() {
    CallMatMul(activations_.attention.att_out, layer_.attn_vec_einsum_w,
               /*bias=*/nullptr, env_, activations_.attention.att_sums);
  }

 private:
  const size_t num_tokens_;
  const size_t num_real_patches_;
  Activations& activations_;
  const LayerWeightsPtrs& layer_;
  const LayerConfig& layer_config_;
  MatMulEnv& env_;
  const size_t real_patch_width_;
  hwy::ThreadPool& pool_;
  hwy::pool::Caller caller1_;
  hwy::pool::Caller caller2_;
  hwy::pool::Caller caller3_;
  hwy::pool::Caller caller4_;
};

void VitGemma4TransformerLayer(size_t num_tokens, size_t num_real_patches,
                               const size_t layer_idx,
                               const LayerWeightsPtrs& layer,
                               Activations& activations, MatMulEnv& env,
                               size_t real_patch_width) {
  RMSNormBatched(activations.x, layer.pre_attention_norm_scale,
                 activations.attention.pre_att_rms_out, env.ctx);

  VitGemma4Attention(num_tokens, num_real_patches, layer_idx, activations,
                     layer, env, real_patch_width)();

  PostNorm(layer.layer_config.post_norm, layer.post_attention_norm_scale,
           activations.attention.att_sums, env.ctx);

  AddFromBatched(activations.attention.att_sums, activations.x, env.ctx);

  RMSNormBatched(activations.x, layer.pre_ffw_norm_scale,
                 activations.pre_ffw_rms_out, env.ctx);

  FFWNoVit(layer, activations, env);

  PostNorm(layer.layer_config.post_norm, layer.post_ffw_norm_scale,
           activations.ffw_out, env.ctx);

  AddFromBatched(activations.ffw_out, activations.x, env.ctx);
}

// Prefills the image tokens with the ViT encoder.
void PrefillVitGemma4(const ModelConfig& model_config,
                      const WeightsPtrs& weights,
                      const RuntimeConfig& runtime_config, const Image& image,
                      ImageTokens& image_tokens, Activations& activations,
                      MatMulEnv& env) {
  PROFILER_ZONE("Gen.PrefillVitGemma4");
  AssertValidImage(model_config, image);
  const size_t max_patches = model_config.vit_config.seq_len;      // 2520
  const size_t vit_model_dim = model_config.vit_config.model_dim;  // 768
  const size_t pooling_kernel_size = model_config.vit_config.pool_dim;
  HWY_ASSERT(max_patches == activations.x.Rows());

  // 1. Aspect-ratio-preserving resize
  Image resized_image = image;
  int target_height, target_width;
  GetAspectRatioPreservingSize(image.height(), image.width(),
                               model_config.vit_config.patch_width,  // 16
                               max_patches,
                               pooling_kernel_size,
                               target_height, target_width);
  resized_image.ResizeBilinear(target_width, target_height);

  // 2. Embed patches & add position embeddings
  EmbedImagePatchesGemma4(resized_image, model_config, weights, activations,
                          env);

  const size_t patch_width = model_config.vit_config.patch_width;  // 16
  const size_t real_patch_width = target_width / patch_width;

  const size_t num_real_patches = (resized_image.height() / patch_width) *
                                  (resized_image.width() / patch_width);

  // 3. Go through all layers.
  for (size_t layer_idx = 0;
       layer_idx < model_config.vit_config.layer_configs.size(); ++layer_idx) {
    VitGemma4TransformerLayer(max_patches, num_real_patches, layer_idx,
                              *weights.VitLayer(layer_idx), activations, env,
                              real_patch_width);
  }

  // 4. Gemma 4 ViT has no final encoder norm (unlike PaliGemma/SigLIP which uses
  // LayerNorm with scale/bias). Output feeds directly into pooling.

  // 5. Pooling (3x3 average pooling)
  const size_t H_p = target_height / patch_width;
  const size_t W_p = target_width / patch_width;
  const size_t H_g = H_p / 3;
  const size_t W_g = W_p / 3;
  const size_t num_pooled_tokens = H_g * W_g;
  const size_t max_soft_tokens = max_patches / 9;  // 280

  HWY_ASSERT(num_pooled_tokens <= max_soft_tokens);

  MatStorageT<float> pooled("pooled", Extents2D(max_soft_tokens, vit_model_dim),
                            env.ctx.allocator, MatPadding::kOdd);
  gcpp::ZeroInit(pooled);

  for (size_t gy = 0; gy < H_g; ++gy) {
    for (size_t gx = 0; gx < W_g; ++gx) {
      const size_t g_idx = gy * W_g + gx;
      float* dst_row = pooled.Row(g_idx);
      for (size_t dy = 0; dy < 3; ++dy) {
        for (size_t dx = 0; dx < 3; ++dx) {
          const size_t py = 3 * gy + dy;
          const size_t px = 3 * gx + dx;
          const size_t p_idx = py * W_p + px;
          const float* src_row = activations.x.Row(p_idx);
          for (size_t d = 0; d < vit_model_dim; ++d) {
            dst_row[d] += src_row[d];
          }
        }
      }
      for (size_t d = 0; d < vit_model_dim; ++d) {
        dst_row[d] /= 9.0f;
      }
    }
  }

  // 6. Unscaled RMSNorm (embedding_pre_projection_norm in HF
  // Gemma4MultimodalEmbedder)
  RMSNormNoScaleInplaceBatched(pooled, env.ctx);

  // 7. Project to LLM space
  // Apply head embedding into image_tokens of size of the LLM kModelDim.
  CallMatMul(pooled, weights.vit_img_head_kernel,
             /*bias=*/nullptr, env, image_tokens);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();
