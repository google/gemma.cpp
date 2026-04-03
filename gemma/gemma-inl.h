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

// Transformer components shared between vit.cc and attention.cc.

#include <stddef.h>
#include <stdint.h>

#include "gemma/activations.h"
#include "gemma/configs.h"
#include "gemma/weights.h"
#include "ops/matmul.h"
#include "util/mat.h"
#include "util/threading.h"
#include "util/zones.h"
#include "hwy/profiler.h"

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
#include "ops/ops-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// For use by Vit even if !GEMMA_FUSED_FFN.
template <typename T1, typename T2>
void Activation(ActivationType activation, T1* HWY_RESTRICT c1,
                const T2* HWY_RESTRICT c2, const size_t count,
                ThreadingContext& ctx, const size_t worker) {
  GCPP_ZONE(ctx, worker, Zones::kGenActivation);
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  using VF = hn::Vec<DF>;
  // ActivationType::Gelu
  if (c2 == nullptr) {  // No multiplier, just Gelu.
    Gelu(c1, count);
    return;
  };
  // Has multiplier, Gelu(c1) * c2.
  Decompress1AndCompressInplace(DF(), c1, count, c2, /*p1_ofs=*/0,
                                [](DF df, VF v1, VF v2) HWY_ATTR -> VF {
                                  return hn::Mul(v2, Gelu(df, v1));
                                });
}

// No C2 multiplier - used by Vit.
template <class Mat>
void ActivationBatched(
    ActivationType activation, Mat& c1, ThreadingContext& ctx,
    size_t cluster_idx = 0,
    ParallelismStrategy parallelism = ParallelismStrategy::kFlat) {
  using T = typename Mat::T;
  ParallelFor(parallelism, c1.Rows(), ctx, cluster_idx,
              Callers::kActivationBatched, [&](uint64_t task, size_t worker) {
                Activation(activation, c1.Row(task),
                           static_cast<const T*>(nullptr), c1.Cols(), ctx,
                           worker);
              });
}

#if GEMMA_FUSED_FFN

// Called during TwoMatMul.
static inline void Activation(ActivationType activation, const RowPtrsBF C1,
                              const IndexRange range_r,
                              const IndexRange range_c, const StridedViewBF C2,
                              ThreadingContext& ctx, const size_t worker) {
  GCPP_ZONE(ctx, worker, Zones::kGenActivationFused);
  const size_t cols = range_c.Num();
  HWY_DASSERT(C2.Cols() == cols);
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  using VF = hn::Vec<DF>;
  // Gated: Gelu(c1) * c2.
  for (size_t ir = 0; ir < range_r.Num(); ++ir) {
    Decompress1AndCompressInplace(
        DF(), C1.Row(range_r.begin() + ir) + range_c.begin(), cols, C2.Row(ir),
        /*p1_ofs*/ 0, [](DF df, VF v1, VF v2) HWY_ATTR -> VF {
          return hn::Mul(v2, Gelu(df, v1));
        });
  }
}

#endif  // GEMMA_FUSED_FFN

// Only used if !GEMMA_FUSED_FFN, but define anyway so that we can check
// using if constexpr rather than #if, which interferes with code folding.
template <class Mat1, class Mat2>
HWY_NOINLINE void ActivationBatched(
    ActivationType activation, Mat1& c1, const Mat2* c2, ThreadingContext& ctx,
    size_t cluster_idx = 0,
    ParallelismStrategy parallelism = ParallelismStrategy::kFlat) {
  HWY_DASSERT(c1.SameShape(*c2));
  if (c2 && c2->HasPtr()) {
    ParallelFor(parallelism, c1.Rows(), ctx, cluster_idx,
                Callers::kActivationBatched, [&](uint64_t task, size_t worker) {
                  Activation(activation, c1.Row(task), c2->Row(task), c1.Cols(),
                             ctx, worker);
                });
  } else {  // No multiplier
    ParallelFor(parallelism, c1.Rows(), ctx, cluster_idx,
                Callers::kActivationBatched, [&](uint64_t task, size_t worker) {
                  Activation(activation, c1.Row(task),
                             static_cast<const typename Mat2::T*>(nullptr),
                             c1.Cols(), ctx, worker);
                });
  }
}

template <typename T2, class LayerWeights>
HWY_NOINLINE void ResidualConnection(const MatPtrT<T2>& other,
                                     MatPtrT<float>& HWY_RESTRICT x,
                                     const LayerWeights& layer,
                                     bool is_attention, ThreadingContext& ctx) {
  // ResidualType::Add
  AddFromBatched(other, x, ctx);
}

template <typename InOutT>
void PostNorm(PostNormType post_norm, const MatPtr& weights,
              MatPtrT<InOutT>& inout, ThreadingContext& ctx) {
  HWY_DASSERT(weights.Rows() == 1);
  if (post_norm == PostNormType::Scale) {
    RMSNormInplaceBatched(weights, inout, ctx);
  }
}

// Gemma 4 MoE feed-forward with top-k routing.
static inline void FFWMoE(const LayerWeightsPtrs& layer,
                            Activations& activations, MatMulEnv& env) {
  GCPP_ZONE(env.ctx, hwy::Profiler::GlobalIdx(), Zones::kGenFFW);
  const LayerConfig& layer_config = layer.layer_config;
  const size_t num_experts = layer_config.num_experts;
  const size_t top_k = layer_config.top_k_experts;
  const size_t model_dim = layer_config.model_dim;
  const size_t batch_size = activations.pre_ffw_rms_out.Rows();

  // Compute router logits: [batch, num_experts].
  CallMatMul(activations.pre_ffw_rms_out, layer.ffn_gate_in_w,
             /*add=*/nullptr, env, activations.C1);

  // Find top-k experts per token and compute softmax routing weights.
  // C2 stores [expert_idx_0, weight_0, expert_idx_1, weight_1, ...].
  ParallelFor(ParallelismStrategy::kFlat, batch_size, env.ctx,
              /*cluster_idx=*/0, Callers::kSampleAndStream,
              [&](size_t bi, size_t /*worker*/) {
    float* router_logits = activations.C1.Row(bi);
    float* routing = activations.C2.Row(bi);
    // Selection sort for top-k.
    for (size_t k = 0; k < top_k; ++k) {
      size_t best_idx = 0;
      float best_val = -1e30f;
      for (size_t e = 0; e < num_experts; ++e) {
        bool done = false;
        for (size_t j = 0; j < k; ++j) {
          if (static_cast<size_t>(routing[j * 2]) == e) { done = true; break; }
        }
        if (!done && router_logits[e] > best_val) {
          best_val = router_logits[e];
          best_idx = e;
        }
      }
      routing[k * 2] = static_cast<float>(best_idx);
      routing[k * 2 + 1] = best_val;
    }
    // Softmax over top-k logits.
    float max_val = routing[1];
    for (size_t k = 1; k < top_k; ++k) {
      if (routing[k * 2 + 1] > max_val) max_val = routing[k * 2 + 1];
    }
    float sum_exp = 0.0f;
    for (size_t k = 0; k < top_k; ++k) {
      routing[k * 2 + 1] = expf(routing[k * 2 + 1] - max_val);
      sum_exp += routing[k * 2 + 1];
    }
    for (size_t k = 0; k < top_k; ++k) routing[k * 2 + 1] /= sum_exp;
  });

  // Save routing data before the expert loop overwrites C2.
  // Each token stores [expert_idx_0, weight_0, ..., expert_idx_k, weight_k].
  const size_t routing_cols = top_k * 2;
  std::vector<float> routing_data(batch_size * routing_cols);
  for (size_t bi = 0; bi < batch_size; ++bi) {
    const float* src = activations.C2.Row(bi);
    float* dst = routing_data.data() + bi * routing_cols;
    for (size_t c = 0; c < routing_cols; ++c) dst[c] = src[c];
  }

  // Zero out the output buffer.
  memset(activations.ffw_out.PackedData(), 0,
         batch_size * model_dim * sizeof(float));

  // Process each expert: gather tokens, compute, scatter weighted results.
  for (size_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
    // Count tokens routed to this expert.
    size_t expert_token_count = 0;
    for (size_t bi = 0; bi < batch_size; ++bi) {
      const float* routing = routing_data.data() + bi * routing_cols;
      for (size_t k = 0; k < top_k; ++k) {
        if (static_cast<size_t>(routing[k * 2]) == expert_idx) {
          ++expert_token_count;
          break;
        }
      }
    }
    if (expert_token_count == 0) continue;

    // Gather tokens for this expert, scaled by routing weight.
    size_t write_idx = 0;
    for (size_t bi = 0; bi < batch_size; ++bi) {
      const float* routing = routing_data.data() + bi * routing_cols;
      for (size_t k = 0; k < top_k; ++k) {
        if (static_cast<size_t>(routing[k * 2]) == expert_idx) {
          const float weight = routing[k * 2 + 1];
          const float* input = activations.pre_ffw_rms_out.Row(bi);
          float* dest = activations.C1.Row(write_idx);
          for (size_t d = 0; d < model_dim; ++d) dest[d] = input[d] * weight;
          ++write_idx;
          break;
        }
      }
    }

    // Expert FFN: gate * up, then down.
    CallMatMul(activations.C1, layer.ffn_gate_w,
               /*add=*/nullptr, env, activations.C1);
    CallMatMul(activations.pre_ffw_rms_out, layer.ffn_up_w,
               /*add=*/nullptr, env, activations.C2);
    ActivationBatched(layer_config.activation, activations.C1,
                      &activations.C2, env.ctx);
    CallMatMul(activations.C1, layer.ffn_down_w,
               /*add=*/nullptr, env, activations.C1);

    // Scatter results back to output buffer.
    write_idx = 0;
    for (size_t bi = 0; bi < batch_size; ++bi) {
      const float* routing = routing_data.data() + bi * routing_cols;
      for (size_t k = 0; k < top_k; ++k) {
        if (static_cast<size_t>(routing[k * 2]) == expert_idx) {
          float* output = activations.ffw_out.Row(bi);
          const float* expert_out = activations.C1.Row(write_idx);
          for (size_t d = 0; d < model_dim; ++d) output[d] += expert_out[d];
          ++write_idx;
          break;
        }
      }
    }
  }  // for each expert

  if (layer.ffn_output_w.HasPtr()) {
    CallMatMul(activations.ffw_out, layer.ffn_output_w,
               /*add=*/nullptr, env, activations.ffw_out);
  }
}


static inline void FFWNoVit(const LayerWeightsPtrs& layer,
                            Activations& activations, MatMulEnv& env) {
  GCPP_ZONE(env.ctx, hwy::Profiler::GlobalIdx(), Zones::kGenFFW);
  const LayerConfig& layer_config = layer.layer_config;

  HWY_DASSERT(!layer_config.ff_biases);  // Only used in Vit.

#if GEMMA_FUSED_FFN
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
  // Compute the hidden layer activations.
  CallMatMul(activations.pre_ffw_rms_out, layer.gating_einsum_w1, nullptr, env,
             activations.C1);
  CallMatMul(activations.pre_ffw_rms_out, layer.gating_einsum_w2, nullptr, env,
             activations.C2);
  // Activation (Gelu) and maybe multiply by gate. Store activations in act.
  ActivationBatched(layer_config.activation, activations.C1, &activations.C2,
                    env.ctx);
#endif

  // Hidden layer -> output layer.
  CallMatMul(activations.C1, layer.linear_w, nullptr, env, activations.ffw_out);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_INL_H_
