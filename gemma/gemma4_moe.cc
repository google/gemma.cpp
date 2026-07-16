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

#include <algorithm>  // std::copy_n
#include <atomic>
#include <cmath>
#include <vector>

#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#include "util/zones.h"
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include "gemma/activations.h"
#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "gemma/tensor_stats.h"
#include "gemma/weights.h"
#include "hwy/contrib/sort/order.h"
#include "hwy/profiler.h"

// Compiles this file for multiple architectures via "foreach_target.h", to
// which we pass the filename via macro 'argument'.
// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/gemma4_moe.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "gemma/attention.h"  // includes highway.h
#include "gemma/gemma-inl.h"
#include "ops/ops-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

static constexpr size_t kMaxExperts = 128;

static constexpr uint32_t FloatToUint32Sortkey(float val) {
  uint32_t temp = hwy::BitCastScalar<uint32_t>(val);
  return temp & 0x80000000 ? ~temp : temp ^ 0x80000000;
}

static constexpr float Uint32SortkeyToFloat(uint32_t val) {
  return hwy::BitCastScalar<float>(val & 0x80000000 ? val ^ 0x80000000 : ~val);
}

template <typename OT>
void Gemma4PreNorm(size_t norm_num_groups, const MatStorageT<float>& x,
                   const MatPtr& weights, MatPtrT<OT>& out,
                   ThreadingContext& ctx) {
  if (norm_num_groups > 1) {
    GroupedRMSNormBatched(x, weights, out, norm_num_groups, ctx);
  } else {
    RMSNormBatched(x, weights, out, ctx);
  }
}

template <typename OT>
void Gemma4PostNorm(PostNormType post_norm, size_t norm_num_groups,
                    const MatPtr& weights, MatPtrT<OT>& inout,
                    ThreadingContext& ctx) {
  if (post_norm == PostNormType::Scale) {
    if (norm_num_groups > 1) {
      GroupedRMSNormInplaceBatched(weights, inout, norm_num_groups, ctx);
    } else {
      RMSNormInplaceBatched(weights, inout, ctx);
    }
  }
}

// Gemma 4 MoE struct: contains the MoE-specific FFW logic for Gemma 4 models.
struct Gemma4MoE {
  // Fills `activations.per_token` and `expert_sizes`.
  // Router weights always include 1 (due to softmax).
  static HWY_NOINLINE void ChooseExperts(const LayerWeightsPtrs& layer,
                                         Activations& activations,
                                         std::atomic<uint32_t>* expert_sizes,
                                         MatMulEnv& env) {
    const size_t num_experts = layer.layer_config.NumExperts();
    const size_t experts_per_token =
        layer.layer_config.NumExpertsPerDatapoint();
    const size_t num_tokens = activations.x.Rows();
    const size_t model_dim = layer.layer_config.model_dim;

    activations.router_in.OverrideRows(num_tokens);

    const float scale_factor = 1.0f / sqrtf(static_cast<float>(model_dim));
    const bool has_router_scale = layer.router_scale.HasPtr();
    const MatPtrT<BF16> router_scale(layer.router_scale);
    const BF16* scale_ptr = has_router_scale ? router_scale.Row(0) : nullptr;

    for (size_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
      const float* pre_ffw_row = activations.x.Row(token_idx);
      BF16* router_in_row = activations.router_in.Row(token_idx);

      for (size_t col = 0; col < model_dim; ++col) {
        router_in_row[col] = hwy::ConvertScalarTo<BF16>(pre_ffw_row[col]);
      }
    }

    RMSNormNoScaleInplaceBatched(activations.router_in, env.ctx);

    // TODO(philculliton): Use a float buffer for router_in to avoid the
    // BF16->float->BF16 round-trip, and precompute scale_factor * router_scale
    // once rather than per-token. Per the CL comment: we are converting to
    // bf16, but then converting back to float below. Should we set up a
    // router_in_row_f32 so we can just keep it as float? (That would help if
    // num_tokens>>1, because we could precompute * scale_factor once.)
    for (size_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
      BF16* router_in_row = activations.router_in.Row(token_idx);
      if (has_router_scale) {
        for (size_t col = 0; col < model_dim; ++col) {
          router_in_row[col] = hwy::ConvertScalarTo<BF16>(
              hwy::ConvertScalarTo<float>(router_in_row[col]) * scale_factor *
              hwy::ConvertScalarTo<float>(scale_ptr[col]));
        }
      } else {
        for (size_t col = 0; col < model_dim; ++col) {
          router_in_row[col] = hwy::ConvertScalarTo<BF16>(
              hwy::ConvertScalarTo<float>(router_in_row[col]) * scale_factor);
        }
      }
    }

    activations.router_logits.OverrideRows(num_tokens);
    activations.s_router_in.Notify(layer.layer_idx, activations.router_in,
                                   env.ctx);
    CallMatMul(activations.router_in, layer.moe_router,
               /*add=*/nullptr, env, activations.router_logits);
    activations.s_router_logits.Notify(layer.layer_idx,
                                       activations.router_logits, env.ctx);

    // Lightweight tasks, use within-cluster to minimize fork-join overhead.
    ParallelFor(Parallelism::kWithinCluster, num_tokens, env.ctx,
                /*cluster_idx=*/0, Callers::kMoEChooseExperts,
                [&](size_t token_idx, size_t worker) HWY_ATTR {
                  GCPP_ZONE(env.ctx, worker, Zones::kGenMoEFFWExpertsChoose);

                  // Sort the expert_idx by router logits.
                  const float* HWY_RESTRICT logits =
                      activations.router_logits.Row(token_idx);
                  hwy::K32V32 logits_expert_indices[kMaxExperts];
                  for (uint32_t expert_idx = 0; expert_idx < num_experts;
                       ++expert_idx) {
                    logits_expert_indices[expert_idx] = {
                        .value = expert_idx,
                        .key = FloatToUint32Sortkey(logits[expert_idx])};
                  }
                  hwy::HWY_NAMESPACE::VQSortStatic(logits_expert_indices,
                                                   num_experts,
                                                   hwy::SortDescending());
                  // For softmax below.
                  const float max_logit =
                      Uint32SortkeyToFloat(logits_expert_indices[0].key);

                  float sum_exp = 0.0f;
                  for (uint32_t expert_idx = 0; expert_idx < num_experts;
                       ++expert_idx) {
                    sum_exp += expf(logits[expert_idx] - max_logit);
                  }

                  // Compute expert weights for the top `experts_per_token`
                  // logits and store which experts are activated for this
                  // token.
                  PerToken* per_token = activations.GetPerToken(token_idx);
                  float top_k_weight_sum = 0.0f;
                  for (size_t i = 0; i < experts_per_token; ++i) {
                    const size_t expert_idx = logits_expert_indices[i].value;
                    HWY_DASSERT(expert_idx < num_experts);

                    const float logit =
                        Uint32SortkeyToFloat(logits_expert_indices[i].key);

                    const uint32_t row = expert_sizes[expert_idx].fetch_add(
                        1, std::memory_order_relaxed);

                    float weight = expf(logit - max_logit) / sum_exp;
                    per_token[i].weight = weight;
                    top_k_weight_sum += weight;

                    per_token[i].expert_idx = static_cast<uint16_t>(expert_idx);
                    per_token[i].row_idx = static_cast<uint16_t>(row);
                  }

                  // Renormalize top-k weights to sum to 1.
                  for (size_t i = 0; i < experts_per_token; ++i) {
                    per_token[i].weight /= top_k_weight_sum;
                  }
                });
  }

  // Computes `activations.ffw_expert_out` for one activated expert.
  static HWY_NOINLINE void ComputeExpertOutput(
      const LayerWeightsPtrs& layer, Activations& activations,
      const size_t expert_idx, const size_t expert_size,
      const uint16_t* HWY_RESTRICT expert_tokens, Parallelism parallelism,
      const size_t cluster_idx, MatMulEnv& env) {
    HWY_DASSERT(expert_idx < layer.layer_config.NumExperts());
    HWY_DASSERT(expert_size != 0);  // Only called for activated experts.
    HWY_DASSERT(cluster_idx < activations.per_cluster.size());
    Activations::PerCluster& per_cluster = activations.per_cluster[cluster_idx];

    const size_t model_dim = activations.attention.config.model_dim;

    const size_t worker = env.ctx.Worker(cluster_idx);
    GCPP_ZONE(env.ctx, worker, Zones::kGenMoEFFWExperts);

    MatPtrT<BF16>& tmp_in = per_cluster.ffw_expert_in;
    MatPtrT<BF16>& expert_out = activations.ffw_expert_out[expert_idx];
    tmp_in.OverrideRows(expert_size);
    expert_out.OverrideRows(expert_size);

    // Gather tokens activated by this expert.
    {
      GCPP_ZONE(env.ctx, worker, Zones::kGenMoEFFWExpertsGather);
      for (size_t i = 0; i < expert_size; ++i) {
        const size_t token_idx = expert_tokens[i];
        std::copy_n(activations.pre_ffw_rms_out.Row(token_idx), model_dim,
                    tmp_in.Row(i));
      }
    }

    activations.s_expert_in.Notify(layer.layer_idx, tmp_in, env.ctx, 0,
                                   cluster_idx, parallelism);

    const ActivationType activation = layer.layer_config.activation;
    MatPtrT<BF16>& C1 = per_cluster.moe_C1;
    C1.OverrideRows(expert_size);

    // Compute the hidden layer activations.
    const auto fused = [&](RowPtrsBF C1, IndexRange range_r, IndexRange range_c,
                           StridedViewBF C2, size_t worker) {
      Activation(activation, C1, range_r, range_c, C2, env.ctx, worker);
    };
    MMOptions options{.cluster_idx = static_cast<uint32_t>(cluster_idx),
                      .parallelism = parallelism};

    if constexpr (GEMMA_FUSED_FFN) {
      options.SetFunc(fused);
      CallTwoMatMul(tmp_in, layer.moe_gating_einsum_w1[expert_idx],
                    layer.moe_gating_einsum_w2[expert_idx], env, C1, options);
      options.func = nullptr;  // next MatMul is normal/not fused.
    } else {
      MatPtrT<BF16>& C2 = per_cluster.moe_C2;
      C2.OverrideRows(expert_size);

      CallMatMul(tmp_in, layer.moe_gating_einsum_w1[expert_idx],
                 /*add=*/nullptr, env, C1, options);

      CallMatMul(tmp_in, layer.moe_gating_einsum_w2[expert_idx],
                 /*add=*/nullptr, env, C2, options);

      // Activation (Gelu) and maybe multiply by gate.
      ActivationBatched(activation, C1, &C2, env.ctx, cluster_idx,
                        options.parallelism);
    }

    activations.s_expert_hidden.Notify(layer.layer_idx, C1, env.ctx, 0,
                                       cluster_idx, parallelism);

    // Hidden layer -> output layer.
    size_t expert_ff_hidden_dim = layer.moe_gating_einsum_w1[expert_idx].Rows();
    MatStorageT<BF16> C1_narrow("C1_n",
                                Extents2D(expert_size, expert_ff_hidden_dim),
                                env.ctx.allocator, MatPadding::kOdd);
    for (size_t i = 0; i < expert_size; ++i) {
      memcpy(C1_narrow.Row(i), C1.Row(i), expert_ff_hidden_dim * sizeof(BF16));
    }


    CallMatMul(C1_narrow, layer.moe_linear_w[expert_idx],
               /*add=*/nullptr, env, expert_out, options);

    if (layer.p_expert_sc.HasPtr()) {
      const MatPtrT<BF16> sc_mat(layer.p_expert_sc);
      const BF16* sc_data = sc_mat.Row(0);
      const float expert_scale =
          hwy::ConvertScalarTo<float>(sc_data[expert_idx]);
      for (size_t i = 0; i < expert_size; ++i) {
        BF16* row_data = expert_out.Row(i);
        for (size_t j = 0; j < model_dim; ++j) {
          row_data[j] = hwy::ConvertScalarTo<BF16>(
              hwy::ConvertScalarTo<float>(row_data[j]) * expert_scale);
        }
      }
    }

    activations.s_expert_out.Notify(layer.layer_idx, expert_out, env.ctx, 0,
                                    cluster_idx, parallelism);
  }

  // Computes `activations.ffw_expert_out` for each expert.
  static HWY_NOINLINE void ComputeAllExpertOutputs(
      const LayerWeightsPtrs& layer, Activations& activations,
      const uint32_t* HWY_RESTRICT expert_sizes, MatMulEnv& env) {
    HWY_ALIGN uint32_t expert_pos[kMaxExperts];
    const size_t num_experts = layer.layer_config.NumExperts();
    const size_t num_tokens = activations.x.Rows();
    const size_t experts_per_token =
        layer.layer_config.NumExpertsPerDatapoint();
    const uint32_t expert_size_sum =
        ExclusivePrefixSum(expert_sizes, num_experts, expert_pos);
    HWY_DASSERT(expert_size_sum == num_tokens * experts_per_token);
    (void)expert_size_sum;
    HWY_DASSERT(expert_pos[0] == 0);
    HWY_DASSERT(expert_pos[num_experts - 1] + expert_sizes[num_experts - 1] ==
                expert_size_sum);

    uint32_t activated_expert_idx[kMaxExperts];
    size_t num_activated_experts = 0;

    for (size_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
      const PerToken* per_token = activations.GetPerToken(token_idx);
      for (size_t i = 0; i < experts_per_token; ++i) {
        const size_t expert_idx = per_token[i].expert_idx;
        const size_t row = per_token[i].row_idx;
        activations.expert_tokens[expert_pos[expert_idx] + row] =
            static_cast<uint16_t>(token_idx);

        activated_expert_idx[num_activated_experts] =
            static_cast<uint32_t>(expert_idx);
        num_activated_experts += (row == 0);
      }
    }

    Parallelism outer = Parallelism::kAcrossClusters;
    Parallelism inner = Parallelism::kWithinCluster;
    if (num_activated_experts < activations.per_cluster.size()) {
      outer = Parallelism::kNone;
      inner = Parallelism::kHierarchical;
    }
    ParallelFor(outer, num_activated_experts, env.ctx,
                /*cluster_idx=*/0, Callers::kMoEComputeAllExpertOutputs,
                [&](size_t expert_idx_idx, size_t cluster_idx) {
                  const size_t expert_idx =
                      activated_expert_idx[expert_idx_idx];
                  ComputeExpertOutput(
                      layer, activations, expert_idx, expert_sizes[expert_idx],
                      &activations.expert_tokens[expert_pos[expert_idx]], inner,
                      cluster_idx, env);
                });
  }

  // For each token in parallel, fills `activations.ffw_out` with the weighted
  // sum.
  static HWY_NOINLINE void WeightedSumOfExperts(
      const LayerWeightsPtrs& layer, Activations& activations,
      const uint32_t* HWY_RESTRICT expert_sizes, MatMulEnv& env) {
    HWY_DASSERT(layer.layer_config.IsMoE());
    HWY_DASSERT(layer.layer_config.NumExperts() <= kMaxExperts);
    const size_t experts_per_token =
        layer.layer_config.NumExpertsPerDatapoint();
    const size_t model_dim = activations.attention.config.model_dim;

    ParallelFor(
        Parallelism::kFlat, activations.x.Rows(), env.ctx,
        /*cluster_idx=*/0, Callers::kMoEWeightedSumOfExperts,
        [&](size_t token_idx, size_t worker) {
          GCPP_ZONE(env.ctx, worker, Zones::kGenMoEFFWWeightedSum);
          const PerToken* per_token = activations.GetPerToken(token_idx);

          const auto get_expert_row = [&](size_t i) {
            const size_t expert_idx = per_token[i].expert_idx;
            HWY_DASSERT(expert_idx < layer.layer_config.NumExperts());
            const size_t row = per_token[i].row_idx;
            HWY_DASSERT(row < expert_sizes[expert_idx]);
            return activations.ffw_expert_out[expert_idx].Row(row);
          };
          
          // First expert: write to `ffw_out` without accumulating.
          MulByConstTo(per_token[0].weight, get_expert_row(0),
                       activations.ffw_out.Row(token_idx), model_dim, env.ctx,
                       worker);
          
          // Subsequent experts: accumulate into `ffw_out`.
          for (size_t i = 1; i < experts_per_token; ++i) {
            MulByConstAndAdd(per_token[i].weight, get_expert_row(i),
                             activations.ffw_out.Row(token_idx), model_dim);
          }
        });
  }

  static HWY_NOINLINE void MoEFFW(const LayerWeightsPtrs& layer,
                                  Activations& activations, MatMulEnv& env) {
    GCPP_ZONE(env.ctx, /*global_idx=*/0, Zones::kGenMoEFFW);

    HWY_ALIGN std::atomic<uint32_t> atomic_expert_sizes[kMaxExperts] = {};
    ChooseExperts(layer, activations, atomic_expert_sizes, env);

    // Copy to normal array so we can call `ExclusivePrefixSum`.
    HWY_ALIGN uint32_t expert_sizes[kMaxExperts];
    const size_t num_experts = layer.layer_config.NumExperts();
    for (size_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
      expert_sizes[expert_idx] =
          atomic_expert_sizes[expert_idx].load(std::memory_order_acquire);
    }

    ComputeAllExpertOutputs(layer, activations, expert_sizes, env);
    WeightedSumOfExperts(layer, activations, expert_sizes, env);
  }
};

void Gemma4MoETransformerLayer(size_t num_tokens, const size_t layer_idx,
                               const LayerWeightsPtrs& layer,
                               Activations& activations, QBatch& qbatch,
                               MatMulEnv& env) {
  GCPP_ZONE(env.ctx, hwy::Profiler::GlobalIdx(),
            Zones::kGenTotalTransformerLayer);

  const ModelConfig& config = activations.attention.config;
  const size_t norm_num_groups = config.norm_num_groups;
  const size_t num_layers = config.num_layers;

  auto pre_norm = [&](const MatPtr& weights, auto& out) HWY_ATTR {
    Gemma4PreNorm(norm_num_groups, activations.x, weights, out, env.ctx);
  };

  auto post_norm = [&](PostNormType type, const MatPtr& weights,
                       auto& inout) HWY_ATTR {
    Gemma4PostNorm(type, norm_num_groups, weights, inout, env.ctx);
  };

  auto rms_norm_inplace = [&](const MatPtr& weights, auto& inout) HWY_ATTR {
    size_t w_ofs =
        weights.Rows() > 1 ? (layer_idx / (num_layers / weights.Rows())) : 0;
    RMSNormInplaceBatched(weights, inout, env.ctx, w_ofs);
  };

  size_t kv_cache_layer_idx = layer_idx;

  pre_norm(layer.pre_attention_norm_scale,
           activations.attention.pre_att_rms_out);

  HWY_DASSERT(layer.layer_config.type == LayerAttentionType::kGemma);
  HWY_DASSERT(qbatch.PrefixEnd(0) == 0);  // expect causal attention
  int flags = 0;
  GemmaAttention(num_tokens, kv_cache_layer_idx, layer, activations.attention,
                 qbatch, env, activations.attention_impl, flags);

  post_norm(layer.layer_config.post_norm, layer.post_attention_norm_scale,
            activations.attention.att_sums);

  ResidualConnection(activations.attention.att_sums, activations.x, layer,
                     /*is_attention=*/true, env.ctx);

  // Dual-Path FFW
  pre_norm(
      layer.pre_ffw2_ns.HasPtr() ? layer.pre_ffw2_ns : layer.pre_ffw_norm_scale,
      activations.pre_ffw_rms_out);

  // Shared MLP Path
  FFWNoVit(layer, activations, env);  // writes to activations.ffw_out

  // Reuse att_sums as temporary buffer for Shared MLP output
  for (size_t r = 0; r < num_tokens; ++r) {
    const float* from_row = activations.ffw_out.Row(r);
    BF16* to_row = activations.attention.att_sums.Row(r);
    for (size_t c = 0; c < config.model_dim; ++c) {
      to_row[c] = hwy::ConvertScalarTo<BF16>(from_row[c]);
    }
  }

  if (layer.post_ffw2_ns.HasPtr()) {
    rms_norm_inplace(layer.post_ffw2_ns, activations.attention.att_sums);
  }

  // MoE Path
  pre_norm(layer.pre_ffw_norm_scale, activations.pre_ffw_rms_out);

  Gemma4MoE::MoEFFW(layer, activations, env);  // writes to activations.ffw_out

  if (layer.post_ffw1_ns.HasPtr()) {
    rms_norm_inplace(layer.post_ffw1_ns, activations.ffw_out);
  }

  // Combine & Final Norm (Fix for dual-path combination)
  AddFromBatched(activations.attention.att_sums, activations.ffw_out, env.ctx);

  if (layer.post_ffw_norm_scale.HasPtr()) {
    post_norm(layer.layer_config.post_norm, layer.post_ffw_norm_scale,
              activations.ffw_out);
  }

  ResidualConnection(activations.ffw_out, activations.x, layer,
                     /*is_attention=*/false, env.ctx);

  // Apply skip_scale AFTER residual connection
  // (HF: hidden_states *= self.layer_scalar, applied to full output)
  if (layer.skip_scale.HasPtr()) {
    const float skip_scale_val = CallUpcastedActivation(
        &layer.skip_scale, [](const auto* mat) -> float {
          return hwy::ConvertScalarTo<float>(mat->Row(0)[0]);
        });
    for (size_t r = 0; r < num_tokens; ++r) {
      MulByConst(skip_scale_val, activations.x.Row(r), activations.x.Cols());
    }
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();
