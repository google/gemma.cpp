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

  // TODO(philculliton): transition fully to MatMul.
  HWY_NOINLINE void DotSoftmaxWeightedSumMatrix() {
    const size_t qkv_dim = layer_config_.qkv_dim;
    const size_t heads = layer_config_.heads;
    HWY_ASSERT_M(heads == layer_config_.kv_heads, "Vit expects MHA");
    const size_t seq_len =
        static_cast<size_t>(activations_.attention.div_seq_len.GetDivisor());
    const float query_scale = 1.0f / sqrtf(static_cast<float>(qkv_dim));
    PROFILER_ZONE("Gen.VitAttention.DotSoftmax");

    // Shift Q, K, VT to MatStorageT.
    MatStorageT<float> Q("Q2", Extents2D(num_tokens_, qkv_dim),
                         env_.ctx.allocator, MatPadding::kPacked);
    MatStorageT<float> K("K2", Extents2D(seq_len, qkv_dim), env_.ctx.allocator,
                         MatPadding::kPacked);
    MatStorageT<float> C("C2", Extents2D(num_tokens_, seq_len),
                         env_.ctx.allocator, MatPadding::kPacked);

    // Initialize att_out to zero prior to head loop.
    ZeroInit(activations_.attention.att_out);

    for (size_t head = 0; head < heads; ++head) {
      pool_.Run(0, num_tokens_, [&](uint64_t task, size_t worker) HWY_ATTR {
        const size_t token = task;
        float* HWY_RESTRICT q =
            activations_.attention.q.Row(token) + head * 3 * qkv_dim;
        // TODO: shift to MatMul with A.scale once MatMul is confirmed working
        MulByConst(query_scale, q, qkv_dim, env_.ctx.profiler, worker);
        hwy::CopyBytes(q, Q.Row(token), qkv_dim * sizeof(float));
      });

      pool_.Run(0, seq_len, [&](uint64_t task, size_t /*thread*/) HWY_ATTR {
        const size_t seq_idx = task;
        float* HWY_RESTRICT k = activations_.attention.q.Row(seq_idx) +
                                head * 3 * qkv_dim + qkv_dim;
        hwy::CopyBytes(k, K.Row(seq_idx), qkv_dim * sizeof(float));
      });

      // this produces C, a (num_tokens_, seq_len) matrix of dot products
      CallMatMul(Q, K, nullptr, env_, C);

      pool_.Run(0, num_tokens_, [&](uint64_t task, size_t worker) HWY_ATTR {
        float* HWY_RESTRICT c = C.Row(task);
        Softmax(c, C.Cols(), env_.ctx.profiler, worker);
      });

      pool_.Run(0, num_tokens_, [&](uint64_t task, size_t worker) HWY_ATTR {
        size_t token = task;
        float* HWY_RESTRICT att_out =
            activations_.attention.att_out.Row(token) + head * qkv_dim;
        for (size_t i = 0; i < seq_len; ++i) {
          float* HWY_RESTRICT v = activations_.attention.q.Row(i) +
                                  head * 3 * qkv_dim + 2 * qkv_dim;
          MulByConstAndAdd(C.Row(token)[i], v, att_out, qkv_dim,
                           env_.ctx.profiler, worker);
        }
      });
    }
  }

  HWY_NOINLINE void DotSoftmaxWeightedSum() {
    const size_t qkv_dim = layer_config_.qkv_dim;
    const size_t heads = layer_config_.heads;
    HWY_ASSERT_M(heads == layer_config_.kv_heads, "Vit expects MHA");
    const size_t seq_len =
        static_cast<size_t>(activations_.attention.div_seq_len.GetDivisor());
    const float query_scale = 1.0f / sqrtf(static_cast<float>(qkv_dim));
    PROFILER_ZONE("Gen.VitAttention.DotSoftmax");

    // Compute Q.K, softmax, and weighted V.
    pool_.Run(0, layer_config_.heads * num_tokens_,
              [&](uint64_t task, size_t worker) HWY_ATTR {
                const size_t head = task % layer_config_.heads;
                const size_t token = task / layer_config_.heads;
                // Compute Q.K scores, which are "logits" stored in head_att.
                float* HWY_RESTRICT q =
                    activations_.attention.q.Row(token) + head * 3 * qkv_dim;
                MulByConst(query_scale, q, qkv_dim, env_.ctx.profiler, worker);
                float* HWY_RESTRICT head_att =
                    activations_.attention.att.Row(token) + head * seq_len;
                for (size_t i = 0; i < seq_len; ++i) {
                  float* HWY_RESTRICT k = activations_.attention.q.Row(i) +
                                          head * 3 * qkv_dim + qkv_dim;
                  head_att[i] = Dot(q, k, qkv_dim);  // score = q.k
                }
                // SoftMax yields "probabilities" in head_att.
                Softmax(head_att, seq_len, env_.ctx.profiler, worker);
                // Compute weighted sum of v into att_out.
                float* HWY_RESTRICT att_out =
                    activations_.attention.att_out.Row(token) + head * qkv_dim;
                hwy::ZeroBytes(att_out, qkv_dim * sizeof(*att_out));
                for (size_t i = 0; i < seq_len; ++i) {
                  float* HWY_RESTRICT v = activations_.attention.q.Row(i) +
                                          head * 3 * qkv_dim + 2 * qkv_dim;
                  MulByConstAndAdd(head_att[i], v, att_out, qkv_dim,
                                   env_.ctx.profiler, worker);
                }
              });
  }

  // Sums encoded (`att_out`) over num_heads (`layer_config_.heads`) and
  // head_dim (`qkv_dim`) into output (`att_sums`).
  HWY_NOINLINE void SumHeads() {
    PROFILER_ZONE("Gen.VitAttention.SumHeads");
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
        pool_(env_.ctx.pools.Pool(0)) {}

  HWY_INLINE void operator()() {
    ComputeQKV();
    if (activations_.attention.config.wrapping == PromptWrapping::GEMMA_VLM) {
      DotSoftmaxWeightedSumMatrix();
    } else {
      DotSoftmaxWeightedSum();
    }
    SumHeads();
  }

 private:
  const size_t num_tokens_;
  Activations& activations_;
  const LayerWeightsPtrs& layer_;
  const LayerConfig& layer_config_;
  MatMulEnv& env_;
  hwy::ThreadPool& pool_;
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
  const size_t patch_size = patch_width * patch_width * 3;
  HWY_DASSERT(weights.vit_img_embedding_kernel.Rows() == model_dim);
  HWY_DASSERT(weights.vit_img_embedding_kernel.Cols() == patch_size);
  HWY_DASSERT(activations.x.Cols() == model_dim);
  (void)model_dim;
  // img/embedding/kernel has original shape (14, 14, 3, 1152)
  // H x W x C x D transposed to D x (H x W x C) so here (1152, 14 * 14 * 3)
  // image_patches is (256, 14 * 14 * 3)
  // Must be padded, see `DoDecompressA`.
  MatStorageT<float> image_patches("patches", Extents2D(num_tokens, patch_size),
                                   env.ctx.allocator, MatPadding::kOdd);
  for (size_t i = 0; i < num_tokens; ++i) {
    image.GetPatch(i, image_patches.Row(i));
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
      RMSNormInplace(weights_t->PackedScale1(), 0, activations.x.Row(0),
                     vit_model_dim, env.ctx.profiler, hwy::Profiler::Thread());
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
