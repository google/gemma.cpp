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

#include "gemma/activations.h"
#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "gemma/weights.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"

// Compiles this file for multiple architectures via "foreach_target.h", to
// which we pass the filename via macro 'argument'.
// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/griffin.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "ops/matvec-inl.h"
#include "ops/ops-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// Different functions use different naming conventions for the number of
// tokens. Functions that are query-independent, such as RMSNorm*, call the
// count `num_interleaved`. Functions that are query-dependent, such as
// `Attention`, use separate `num_tokens` and `num_queries`. `num_tokens` is the
// number of tokens from one query: 1 for decode, otherwise prefill_tbatch_size.

void GriffinRecurrent(const QueriesPos& queries_pos, size_t num_tokens,
                      size_t griffin_layer, Activations& activations,
                      const LayerWeightsPtrs* layer_weights,
                      const KVCaches& kv_caches, MatMulEnv& env) {
  PROFILER_ZONE("Gen.Griffin");
  hwy::ThreadPool& pool = env.ctx.pools.Pool(0);
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  const D df;

  const size_t model_dim = layer_weights->layer_config.model_dim;
  HWY_DASSERT(model_dim % hn::Lanes(df) == 0);

  const size_t heads = layer_weights->layer_config.heads;
  const size_t conv_1d_width = layer_weights->layer_config.conv1d_width;
  HWY_ASSERT_M(conv_1d_width % 2 == 0, "Conv width must be even");
  const size_t kHeadDim = model_dim / heads;
  const size_t kMatrixSize = kHeadDim * kHeadDim;

  const size_t num_queries = queries_pos.size();
  const hwy::Divisor div_num_q(static_cast<uint32_t>(num_queries));
  const size_t num_interleaved = num_tokens * num_queries;

  // X / Y linear layers.
  // TODO: MatMul
  HWY_DASSERT(activations.griffin_y.Rows() == activations.griffin_x.Rows());
  HWY_DASSERT(num_interleaved == activations.griffin_y.Rows());
  CallUpcastedSame(
      &layer_weights->griffin.linear_x_w, &layer_weights->griffin.linear_y_w,
      [&](const auto* wx, const auto* wy) {
        for (size_t r = 0; r < num_interleaved; ++r) {
          float* HWY_RESTRICT y = activations.griffin_y.Row(r);
          float* HWY_RESTRICT x = activations.griffin_x.Row(r);
          TwoMatVecAdd(
              *wx, *wy, 0, model_dim, model_dim,
              activations.pre_att_rms_out.Row(r),
              /*add0=*/layer_weights->griffin.linear_x_biases.PackedScale1(),
              /*add1=*/layer_weights->griffin.linear_y_biases.PackedScale1(),
              /*out0=*/x, /*out1=*/y, pool);
          Gelu(y, model_dim);
        }
      });

  // Conv1D.
  for (size_t interleaved_idx = 0; interleaved_idx < num_interleaved;
       ++interleaved_idx) {
    const size_t query_idx = div_num_q.Remainder(interleaved_idx);
    const size_t batch_idx = div_num_q.Divide(interleaved_idx);
    const size_t pos = queries_pos[query_idx] + batch_idx;
    float* HWY_RESTRICT x = activations.griffin_x.Row(query_idx);

    // cache[i] = input at time t-i.
    float* HWY_RESTRICT cache[kMaxConv1DWidth];
    cache[0] = x;
    for (size_t i = 1; i < conv_1d_width; i++) {
      cache[i] =
          kv_caches[query_idx].conv1d_cache.Row(griffin_layer) +
          ((pos + conv_1d_width - 1 - i) % (conv_1d_width - 1)) * model_dim;
    }
    for (size_t i = 0; i < model_dim; i += hn::Lanes(df)) {
      auto xv = hn::Load(df, x + i);
      auto accum0 =
          hn::Load(df, layer_weights->griffin.conv_biases.PackedScale1() + i);
      auto accum1 = hn::Zero(df);
      for (size_t l = 0; 2 * l < conv_1d_width; l++) {
        auto wv0 =
            hn::Load(df, layer_weights->griffin.conv_w.PackedScale1() +
                             (conv_1d_width - 1 - 2 * l) * model_dim + i);
        auto wv1 =
            hn::Load(df, layer_weights->griffin.conv_w.PackedScale1() +
                             (conv_1d_width - 2 - 2 * l) * model_dim + i);
        accum0 = hn::MulAdd(wv0, hn::Load(df, cache[l * 2] + i), accum0);
        accum1 = hn::MulAdd(wv1, hn::Load(df, cache[l * 2 + 1] + i), accum1);
      }
      hn::Store(hn::Add(accum0, accum1), df, x + i);
      hn::Store(xv, df, cache[HWY_MAX(conv_1d_width, 1) - 1] + i);
    }
  }

  // RGLRU
  for (size_t interleaved_idx = 0; interleaved_idx < num_interleaved;
       ++interleaved_idx) {
    const size_t query_idx = div_num_q.Remainder(interleaved_idx);
    const size_t batch_idx = div_num_q.Divide(interleaved_idx);
    const size_t pos = queries_pos[query_idx] + batch_idx;

    float* HWY_RESTRICT x = activations.griffin_x.Row(query_idx);
    float* HWY_RESTRICT y = activations.griffin_y.Row(query_idx);
    float* HWY_RESTRICT gate_x = activations.griffin_gate_x.Row(query_idx);
    float* HWY_RESTRICT a = activations.griffin_multiplier.Row(query_idx);
    float* HWY_RESTRICT rnn_state =
        kv_caches[query_idx].rglru_cache.Row(griffin_layer);

    pool.Run(0, heads, [&](const uint64_t head, size_t /*thread*/) HWY_ATTR {
      size_t head_offset = head * kHeadDim;
      CallUpcasted(&layer_weights->griffin.gate_w, [&](const auto* gate_w) {
        TwoOfsMatVecAddLoop(
            *gate_w, kMatrixSize * head, kMatrixSize * (heads + head), kHeadDim,
            kHeadDim, x + head_offset,
            /*add0=*/layer_weights->griffin.gate_biases.PackedScale1() +
                head_offset,
            /*add1=*/layer_weights->griffin.gate_biases.PackedScale1() +
                model_dim + head_offset,
            /*out0=*/gate_x + head_offset, /*out1=*/a + head_offset);
      });
      Sigmoid(gate_x + head_offset, kHeadDim);
      Sigmoid(a + head_offset, kHeadDim);
      const auto fn_mul = [](D d, hn::Vec<D> x, hn::Vec<D> gate_x)
                              HWY_ATTR { return hn::Mul(x, gate_x); };
      hn::Transform1(D(), a + head_offset, kHeadDim,
                     layer_weights->griffin.a.PackedScale1() + head_offset,
                     fn_mul);
      hn::Transform1(D(), x + head_offset, kHeadDim, gate_x + head_offset,
                     fn_mul);
      // RNN scan
      HWY_DASSERT(kHeadDim % hn::Lanes(df) == 0);
      for (size_t i = 0; i < kHeadDim; i += hn::Lanes(df)) {
        auto log_a = hn::Load(df, a + head_offset + i);
        auto gated_x = hn::Load(df, x + head_offset + i);
        auto rnn = hn::Load(df, rnn_state + head_offset + i);
        auto a = hn::Exp(df, log_a);
        auto x_multiplier = hn::Sqrt(hn::NegMulAdd(a, a, hn::Set(df, 1.0f)));
        if (pos == 0) {
          x_multiplier = hn::Set(df, 1.0f);
        }
        auto new_x = hn::MulAdd(x_multiplier, gated_x, hn::Mul(a, rnn));
        hn::Store(new_x, df, rnn_state + head_offset + i);

        // Join branches.
        auto yv = hn::Load(df, y + head_offset + i);
        auto pre_out = hn::Mul(yv, new_x);
        hn::Store(pre_out, df, x + head_offset + i);
      }
    });
  }  // interleaved_idx

  // Final linear layer.
  CallMatMul(activations.griffin_x, layer_weights->griffin.linear_out_w,
             layer_weights->griffin.linear_out_biases.PackedScale1(), env,
             activations.att_sums);
}  // GriffinRecurrent

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();
