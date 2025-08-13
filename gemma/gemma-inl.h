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

template <typename T>
void Activation(ActivationType activation, T* HWY_RESTRICT c1,
                const T* HWY_RESTRICT c2, const size_t count, hwy::Profiler& p,
                const size_t worker) {
  static const auto zone = p.AddZone("Gen.Activation");
  PROFILER_ZONE3(p, worker, zone);
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
void ActivationBatched(ActivationType activation, Mat& c1,
                       ThreadingContext& ctx) {
  using T = typename Mat::T;
  SmallParallelFor(c1.Rows(), ctx.pools, [&](uint64_t task, size_t worker) {
    // Cast to correct type so type deduction works.
    Activation(activation, c1.Row(task), static_cast<const T*>(nullptr),
               c1.Cols(), ctx.profiler, worker);
  });
}

template <class Mat>
HWY_NOINLINE void ActivationBatched(ActivationType activation, Mat& c1,
                                    const Mat* c2, ThreadingContext& ctx) {
  using T = typename Mat::T;
  HWY_DASSERT(c1.SameShape(*c2));
  if (c2 && c2->HasPtr()) {
    SmallParallelFor(c1.Rows(), ctx.pools, [&](uint64_t task, size_t worker) {
      Activation(activation, c1.Row(task), c2->Row(task), c1.Cols(),
                 ctx.profiler, worker);
    });
  } else {  // No multiplier
    SmallParallelFor(c1.Rows(), ctx.pools, [&](uint64_t task, size_t worker) {
      Activation(activation, c1.Row(task), static_cast<const T*>(nullptr),
                 c1.Cols(), ctx.profiler, worker);
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

static inline void FFWNoVit(const LayerWeightsPtrs& layer,
                            Activations& activations, MatMulEnv& env) {
  static const auto zone =
      env.ctx.profiler.AddZone("Gen.FFW", hwy::ProfilerFlags::kInclusive);
  PROFILER_ZONE3(env.ctx.profiler, hwy::Profiler::Thread(), zone);
  const LayerConfig& layer_config = layer.layer_config;
  const size_t ffh_hidden_dim = layer_config.ff_hidden_dim;

  const bool add_bias = layer_config.ff_biases;
  const float* bias1 =
      add_bias ? layer.ffw_gating_biases.PackedScale1() : nullptr;
  const float* bias2 = add_bias ? bias1 + ffh_hidden_dim : nullptr;
  const float* output_bias =
      add_bias ? layer.ffw_output_biases.PackedScale1() : nullptr;

  // Compute the hidden layer activations.
  CallMatMul(activations.pre_ffw_rms_out, layer.gating_einsum_w1, bias1, env,
             activations.C1);
  CallMatMul(activations.pre_ffw_rms_out, layer.gating_einsum_w2, bias2, env,
             activations.C2);

  // Activation (Gelu) and maybe multiply by gate. Store activations in act.
  ActivationBatched(layer_config.activation, activations.C1, &activations.C2,
                    env.ctx);

  // Hidden layer -> output layer.
  CallMatMul(activations.C1, layer.linear_w, output_bias, env,
             activations.ffw_out);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_INL_H_
