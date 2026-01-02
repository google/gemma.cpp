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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#include "gemma/flash_structs.h"
#include "util/threading_context.h"
#include "util/zones.h"
#include "hwy/base.h"
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include "gemma/activations.h"
#include "gemma/configs.h"  // kMaxQKVDim
#include "util/threading.h"
#include "hwy/profiler.h"

// Compiles this file for multiple architectures via "foreach_target.h", to
// which we pass the filename via macro 'argument'.
// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/flash_attention.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"
#include "gemma/attention.h"
#include "ops/matmul-inl.h"
#include "ops/ops-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

static constexpr size_t kNFx8HTileSize = 8;
static constexpr float kNegInf = -std::numeric_limits<float>::max() / 64.0f;
// Transposes q into q_t.
// Both are 4D tensors stuffed into a 2-D MatPtrT.
// q has shape [batch, qbatch][head, qkv_dim].
// q_t has shape [qkv_dim][qbatch, head, batch] in order to make the maximum
// possible consecutive elements have the same KV.
static void TransposeQ(const MatPtrT<float>& q, MatPtrT<BF16>& q_t,
                       const size_t qbatch_size, ThreadingContext& ctx) {
  // Group floats by the number of floats in a cache line.
  const size_t kNF = ctx.cache_info.LineBytes() / sizeof(float);
  const size_t num_heads = q.Cols() / q_t.Rows();
  const size_t batch_size = q.Rows() / qbatch_size;
  const auto func = [&](const size_t task, size_t worker) HWY_ATTR {
    GCPP_ZONE(ctx, worker, Zones::kFlashAttentionTransposeQ);
    for (size_t lane = 0; lane < kNF; ++lane) {
      size_t q_row = task * kNF + lane;
      if (q_row >= q_t.Rows()) break;
      BF16* HWY_RESTRICT qt_row = q_t.Row(q_row);
      for (size_t qi = 0; qi < qbatch_size; ++qi) {
        for (size_t h = 0; h < num_heads; ++h) {
          for (size_t b = 0; b < batch_size; ++b) {
            qt_row[(qi * num_heads + h) * batch_size + b] =
                hwy::ConvertScalarTo<BF16>(
                    q.Row(b * qbatch_size + qi)[h * q_t.Rows() + q_row]);
          }
        }
      }
    }
  };
  {
    const size_t num_tasks = hwy::DivCeil(q_t.Rows(), kNF);
    // Better than kFlat.
    ParallelFor(Parallelism::kHierarchical, num_tasks, ctx,
                /*cluster_idx=*/0, Callers::kFlashTransposeQ, func);
  }
}

// Updates q in place for RMSNorm and positional encoding.
void RMSNormAndPositionalEncoding(const size_t num_tokens, const QBatch& qbatch,
                                  MatPtrT<float>& q,
                                  const MatPtr& query_norm_scale,
                                  const size_t layer_idx,
                                  const AttentionActivationsPtrs& activations,
                                  ThreadingContext& ctx) {
  const LayerConfig& layer_config = activations.config.layer_configs[layer_idx];
  const float query_scale = activations.query_scale;
  const hwy::Divisor div_qbatch(qbatch.Size());
  const auto func = [&](const size_t task, size_t worker) HWY_ATTR {
    GCPP_ZONE(ctx, worker, Zones::kFlashAttentionRmsNormAndPositionalEncoding);
    size_t qi = div_qbatch.Remainder(task);
    size_t batch_idx = div_qbatch.Divide(task);
    for (size_t h = 0; h < layer_config.heads; ++h) {
      const size_t tq_idx = qbatch.Size() * batch_idx + qi;
      // Find the token position in the query and calculate
      // the range of cache positions to attend to.
      constexpr size_t offset = 0;  // placeholder, do not remove
      const size_t pos = qbatch.Pos(qi) + batch_idx + offset;
      float* HWY_RESTRICT q_row = q.Row(tq_idx) + h * layer_config.qkv_dim;
      // Apply rope and scaling to Q.
      if (query_norm_scale.HasPtr()) {
        CallUpcasted(&query_norm_scale, [&](const auto* weights_t) {
          RMSNormInplace(weights_t->PackedScale1(), /*w_ofs=*/0, q_row,
                         layer_config.qkv_dim, ctx, worker);
        });
      }
      PositionalEncodingQK(q_row, layer_idx, activations, ctx, worker, pos,
                           query_scale);
    }
  };
  {
    // kHierarchical is not worth the extra sync overhead because the tasks are
    // very lightweight.
    ParallelFor(Parallelism::kFlat, num_tokens * qbatch.Size(), ctx,
                /*cluster_idx=*/0, Callers::kFlashRMSNormAndPositionalEncoding,
                func);
  }
}

// Handles a single v row of flash attention for a single q.k dot product.
void HWY_INLINE SingleFlashAttentionStep(float x, float cap, float& old_max,
                                         float& old_d,
                                         const float* HWY_RESTRICT v,
                                         const size_t v_cols,
                                         float* HWY_RESTRICT att_out) {
  if (cap > 0.0f) {
    // Compute tanh(x / cap) * cap, being LogitsSoftCap on the scalar x.
    x = cap * std::tanh(x / cap);
  }
  float m = std::max(x, old_max);
  x = std::exp(x - m);
  float scale = old_d * std::exp(old_max - m);
  old_d = x + scale;
  old_max = m;
  float one_over_d = 1.0f / old_d;
  scale *= one_over_d;
  x *= one_over_d;
  MulByConst(scale, att_out, v_cols);
  MulByConstAndAdd(x, v, att_out, v_cols);
}

// Calculates the complete attention outputs for a single row of q.
void SingleFlashAttention(const size_t start_pos, const size_t last_pos,
                          const BF16* HWY_RESTRICT q, const MatPtrT<KV_t>& k,
                          const MatPtrT<KV_t>& v, const size_t layer_idx,
                          const AttentionActivationsPtrs& activations,
                          float* HWY_RESTRICT att_out, ThreadingContext& ctx,
                          const size_t worker) {
  GCPP_ZONE(ctx, worker, Zones::kFlashAttentionSingleFlashAttention);
  const hn::ScalableTag<BF16> dbf;
  const size_t qkv_dim = k.Cols();

  const size_t pos_mod = activations.div_seq_len.Remainder(start_pos);
  // TODO: Mixed-mode can be further improved for Turin: we can demote right
  // before we do the dot product instruction, rather than promote both to f32.
  // But some potential accuracy loss there, needs evaluation first.
  float m = Dot(dbf, MakeConstSpan(q, qkv_dim), 0, k.Row(pos_mod), qkv_dim);
  if (float cap = activations.config.att_cap; cap > 0.0f) {
    // Compute tanh(x / cap) * cap, being LogitsSoftCap on the scalar x.
    m = cap * std::tanh(m / cap);
  }
  float d = 1.0f;
  // This is just a copy of the first token.
  MulByConstTo(d, v.Row(pos_mod), att_out, v.Cols(), ctx, worker);
  for (size_t pos = start_pos + 1; pos <= last_pos; ++pos) {
    const size_t pos_mod = activations.div_seq_len.Remainder(pos);
    float x = Dot(dbf, MakeConstSpan(q, qkv_dim), 0, k.Row(pos_mod), qkv_dim);
    SingleFlashAttentionStep(x, activations.config.att_cap, m, d,
                             v.Row(pos_mod), v.Cols(), att_out);
  }
}

// Computes and returns a single vector of NF Q.K dot products, which represents
// the dot products of NF rows of Q for a single K timestep.
template <class DF, class VF = hn::Vec<DF>>
VF QDotKVector(DF df, const uint32_t* HWY_RESTRICT q_offsets,
               const size_t k_pos, const MatPtrT<BF16>& q,
               const MatPtrT<KV_t>& k) {
  const hn::ScalableTag<BF16> dbf;
  const size_t qkv_dim = k.Cols();

  hn::TFromD<DF> results[hn::MaxLanes(df)];
  for (size_t i = 0; i < hn::Lanes(df); ++i) {
    results[i] = Dot(dbf, MakeConstSpan(q.Row(0) + q_offsets[i], qkv_dim), 0,
                     k.Row(k_pos), qkv_dim);
  }
  return hn::LoadU(df, results);
}

// Returns an NF Q rows by 8 K rows tile of Q.K dot products.
// This is the result of NF rows of Q against 8 K timesteps, with positions
// given by k_pos[0..7]. Q has been transposed so that the NF rows are read in
// consecutive elements, and other columns by adding q_stride.
template <class DF, class VF = hn::Vec<DF>>
void QDotKTile(DF df, const BF16* HWY_RESTRICT q, const size_t q_stride,
               const MatPtrT<KV_t>& k, const size_t* k_pos, VF& sum0, VF& sum1,
               VF& sum2, VF& sum3, VF& sum4, VF& sum5, VF& sum6, VF& sum7) {
  constexpr size_t kHTileSize = kNFx8HTileSize;
  sum0 = hn::Zero(df);
  sum1 = hn::Zero(df);
  sum2 = hn::Zero(df);
  sum3 = hn::Zero(df);
  sum4 = hn::Zero(df);
  sum5 = hn::Zero(df);
  sum6 = hn::Zero(df);
  sum7 = hn::Zero(df);
  const float* HWY_RESTRICT k_row[kHTileSize];
  for (size_t i = 0; i < kHTileSize; ++i) {
    k_row[i] = k.Row(k_pos[i]);
  }

  const hn::Rebind<BF16, DF> dbfh;
  using VBF = hn::Vec<decltype(dbfh)>;

  for (size_t i = 0; i < k.Cols(); ++i) {
    const VBF q_vec_bf = hn::Load(dbfh, q);
    const VF q_vec = hn::PromoteTo(df, q_vec_bf);
    VF k_0 = hn::Set(df, k_row[0][i]);
    sum0 = hn::MulAdd(q_vec, k_0, sum0);
    VF k_1 = hn::Set(df, k_row[1][i]);
    sum1 = hn::MulAdd(q_vec, k_1, sum1);
    VF k_2 = hn::Set(df, k_row[2][i]);
    sum2 = hn::MulAdd(q_vec, k_2, sum2);
    VF k_3 = hn::Set(df, k_row[3][i]);
    sum3 = hn::MulAdd(q_vec, k_3, sum3);
    VF k_4 = hn::Set(df, k_row[4][i]);
    sum4 = hn::MulAdd(q_vec, k_4, sum4);
    VF k_5 = hn::Set(df, k_row[5][i]);
    sum5 = hn::MulAdd(q_vec, k_5, sum5);
    VF k_6 = hn::Set(df, k_row[6][i]);
    sum6 = hn::MulAdd(q_vec, k_6, sum6);
    VF k_7 = hn::Set(df, k_row[7][i]);
    sum7 = hn::MulAdd(q_vec, k_7, sum7);
    q += q_stride;
  }
}

// Returns the element-wise maximum of 8 vectors, in a single vector.
template <class DF, class VF = hn::Vec<DF>>
VF HWY_INLINE ElementwiseMaxOf8(DF df, const VF& x0, const VF& x1, const VF& x2,
                                const VF& x3, const VF& x4, const VF& x5,
                                const VF& x6, const VF& x7) {
  VF m0 = hn::Max(x0, x1);
  VF m1 = hn::Max(x2, x3);
  VF m2 = hn::Max(x4, x5);
  VF m3 = hn::Max(x6, x7);
  m0 = hn::Max(m0, m1);
  m2 = hn::Max(m2, m3);
  return hn::Max(m0, m2);
}

// Returns the element-wise sum of 8 vectors, in a single vector.
template <class DF, class VF = hn::Vec<DF>>
VF HWY_INLINE ElementwiseSumOf8(DF df, const VF& x0, const VF& x1, const VF& x2,
                                const VF& x3, const VF& x4, const VF& x5,
                                const VF& x6, const VF& x7) {
  VF sum0 = hn::Add(x0, x1);
  VF sum1 = hn::Add(x2, x3);
  VF sum2 = hn::Add(x4, x5);
  VF sum3 = hn::Add(x6, x7);
  sum0 = hn::Add(sum0, sum1);
  sum2 = hn::Add(sum2, sum3);
  return hn::Add(sum0, sum2);
}

// Sweeps a tile of NF Q rows by 8 K timesteps accumulators from start_pos to
// min_last_pos, then sweeps the remaining timesteps in the range (min_last_pos,
// max_last_pos].
void TileFlashAttention(
    const MatPtrT<BF16>& q, const uint32_t* HWY_RESTRICT q_offsets,
    const StridedView<BF16>& qT, const MatPtrT<KV_t>& k, const size_t start_pos,
    const uint32_t* HWY_RESTRICT last_pos, const size_t min_last_pos,
    const size_t max_last_pos, const MatPtrT<KV_t>& v, const size_t layer_idx,
    const AttentionActivationsPtrs& activations, MatPtrT<float>& att_out,
    const uint32_t* HWY_RESTRICT out_offsets, ThreadingContext& ctx,
    const size_t worker) {
  GCPP_ZONE(ctx, worker, Zones::kFlashAttentionTileFlashAttention);
  constexpr size_t kHTileSize = kNFx8HTileSize;
  using DF = hn::ScalableTag<float>;
  const DF df;
  using VF = hn::Vec<DF>;
  using DI = hn::ScalableTag<uint32_t>;
  const DI di;
  using VI = hn::Vec<DI>;
  const size_t kVTileSize = hn::Lanes(df);
  for (size_t i = 0; i < kVTileSize; ++i) {
    hwy::ZeroBytes(att_out.Row(0) + out_offsets[i],
                   v.Cols() * sizeof(att_out.Row(0)[0]));
  }
  VI lasts = hn::LoadU(di, last_pos);
  VF old_m = hn::Set(df, -std::numeric_limits<float>::max() / 2.0f);
  VF old_d = hn::Zero(df);
  const BF16* HWY_RESTRICT qT_row = qT.Row(0);
  const size_t qT_stride = qT.Stride();
  size_t position = start_pos;
  while (position + kHTileSize - 1 <= min_last_pos) {
    size_t k_pos[kHTileSize];
    for (size_t i = 0; i < kHTileSize; ++i) {
      k_pos[i] = activations.div_seq_len.Remainder(position + i);
    }
    VF x0, x1, x2, x3, x4, x5, x6, x7;
    QDotKTile(df, qT_row, qT_stride, k, k_pos, x0, x1, x2, x3, x4, x5, x6, x7);
    if (activations.config.att_cap > 0.0f) {
      // Compute tanh(x / cap) * cap, being LogitsSoftCap on the tile.
      VF cap = hn::Set(df, activations.config.att_cap);
      VF one_over_cap = hn::Div(hn::Set(df, 1.0f), cap);
      x0 = hn::Mul(cap, hn::Tanh(df, hn::Mul(x0, one_over_cap)));
      x1 = hn::Mul(cap, hn::Tanh(df, hn::Mul(x1, one_over_cap)));
      x2 = hn::Mul(cap, hn::Tanh(df, hn::Mul(x2, one_over_cap)));
      x3 = hn::Mul(cap, hn::Tanh(df, hn::Mul(x3, one_over_cap)));
      x4 = hn::Mul(cap, hn::Tanh(df, hn::Mul(x4, one_over_cap)));
      x5 = hn::Mul(cap, hn::Tanh(df, hn::Mul(x5, one_over_cap)));
      x6 = hn::Mul(cap, hn::Tanh(df, hn::Mul(x6, one_over_cap)));
      x7 = hn::Mul(cap, hn::Tanh(df, hn::Mul(x7, one_over_cap)));
    }
    VF m = ElementwiseMaxOf8(df, x0, x1, x2, x3, x4, x5, x6, x7);
    m = hn::Max(old_m, m);
    x0 = hn::Exp(df, hn::Sub(x0, m));
    x1 = hn::Exp(df, hn::Sub(x1, m));
    x2 = hn::Exp(df, hn::Sub(x2, m));
    x3 = hn::Exp(df, hn::Sub(x3, m));
    x4 = hn::Exp(df, hn::Sub(x4, m));
    x5 = hn::Exp(df, hn::Sub(x5, m));
    x6 = hn::Exp(df, hn::Sub(x6, m));
    x7 = hn::Exp(df, hn::Sub(x7, m));
    VF scale = hn::Mul(old_d, hn::Exp(df, hn::Sub(old_m, m)));
    old_d = ElementwiseSumOf8(df, x0, x1, x2, x3, x4, x5, x6, x7);
    old_d = hn::Add(scale, old_d);
    old_m = m;
    VF one_over_d = hn::Div(hn::Set(df, 1.0f), old_d);
    scale = hn::Mul(scale, one_over_d);
    x0 = hn::Mul(x0, one_over_d);
    x1 = hn::Mul(x1, one_over_d);
    x2 = hn::Mul(x2, one_over_d);
    x3 = hn::Mul(x3, one_over_d);
    x4 = hn::Mul(x4, one_over_d);
    x5 = hn::Mul(x5, one_over_d);
    x6 = hn::Mul(x6, one_over_d);
    x7 = hn::Mul(x7, one_over_d);
    MulByConstAndAddTile(df, scale, x0, x1, x2, x3, x4, x5, x6, x7, v, k_pos,
                         att_out.Row(0), out_offsets, v.Cols());
    position += kHTileSize;
  }
  while (position <= max_last_pos) {
    size_t k_pos = activations.div_seq_len.Remainder(position);
    VF x0 = QDotKVector(df, q_offsets, k_pos, q, k);
    if (activations.config.att_cap > 0.0f) {
      // Compute tanh(x / cap) * cap, being LogitsSoftCap on the vector.
      VF cap = hn::Set(df, activations.config.att_cap);
      VF one_over_cap = hn::Div(hn::Set(df, 1.0f), cap);
      x0 = hn::Mul(cap, hn::Tanh(df, hn::Mul(x0, one_over_cap)));
    }
    // Past the last position, x0 doesn't count.
    auto mask = hn::Gt(hn::Set(di, position), lasts);
    VF causal_offset = hn::MaskedSet(df, RebindMask(df, mask),
                                     std::numeric_limits<float>::max() / 2.0f);
    x0 = hn::Sub(x0, causal_offset);
    VF m = hn::Max(old_m, x0);
    x0 = hn::Exp(df, hn::Sub(x0, m));
    VF scale = hn::Mul(old_d, hn::Exp(df, hn::Sub(old_m, m)));
    old_m = m;
    old_d = hn::Add(scale, x0);
    VF one_over_d = hn::Div(hn::Set(df, 1.0f), old_d);
    x0 = hn::Mul(x0, one_over_d);
    scale = hn::Mul(scale, one_over_d);
    MulByConstAndAddVector(df, scale, x0, v, k_pos, att_out.Row(0), out_offsets,
                           v.Cols());
    ++position;
  }
}

// Returns an 4 Q rows by NF K tile of Q.K dot products, in single precision.
// This is the result of 4 rows of Q against NF K timesteps, with positions
// given by k_offsets[0..NF].
template <class DF, class VF = hn::Vec<DF>>
void QDotKTilex4(DF df, const BF16* HWY_RESTRICT q,
                 const uint32_t* HWY_RESTRICT q_offsets, const MatPtrT<KV_t>& k,
                 const int32_t* HWY_RESTRICT k_offsets, VF& sum0, VF& sum1,
                 VF& sum2, VF& sum3) {
  sum0 = hn::Zero(df);
  sum1 = hn::Zero(df);
  sum2 = hn::Zero(df);
  sum3 = hn::Zero(df);
  const float* HWY_RESTRICT k_base = k.Row(0);
  using DI = hn::ScalableTag<int32_t>;
  const DI di;
  using VI = hn::Vec<DI>;
  VI k_offsets_vec = hn::LoadU(di, k_offsets);
  for (size_t i = 0; i < k.Cols(); ++i) {
    VF k_vec = hn::GatherIndex(df, k_base + i, k_offsets_vec);
    VF q_0 = hn::Set(df, hwy::ConvertScalarTo<float>(q[q_offsets[0] + i]));
    sum0 = hn::MulAdd(q_0, k_vec, sum0);
    VF q_1 = hn::Set(df, hwy::ConvertScalarTo<float>(q[q_offsets[1] + i]));
    sum1 = hn::MulAdd(q_1, k_vec, sum1);
    VF q_2 = hn::Set(df, hwy::ConvertScalarTo<float>(q[q_offsets[2] + i]));
    sum2 = hn::MulAdd(q_2, k_vec, sum2);
    VF q_3 = hn::Set(df, hwy::ConvertScalarTo<float>(q[q_offsets[3] + i]));
    sum3 = hn::MulAdd(q_3, k_vec, sum3);
  }
}

// Handles NF v rows of flash attention for NF q.k dot products from one q row.
template <class DF, class VF = hn::Vec<DF>>
float HWY_INLINE SingleFlashAttentionRowVector(DF df, VF& x, float& old_max,
                                               float& old_d) {
  float m = hn::ReduceMax(df, x);
  m = std::max(m, old_max);
  x = hn::Exp(df, hn::Sub(x, hn::Set(df, m)));
  float scale = old_d * std::exp(old_max - m);
  old_d = hn::ReduceSum(df, x) + scale;
  old_max = m;
  if (old_d > 0.0f) {
    const float one_over_d = 1.0f / old_d;
    scale *= one_over_d;
    x = hn::Mul(x, hn::Set(df, one_over_d));
  } else {
    scale = 0.0f;
    x = hn::Zero(df);
  }
  return scale;
}

// Reduces each of x and stores in following lanes of max (tested with float32)
template <class DF, typename T = hn::TFromD<DF>,
          class DF4 = hn::CappedTag<T, 4>, class VF4 = hn::Vec<DF4>,
          class VF = hn::Vec<DF>, typename F>
static VF4 HWY_INLINE Reduce4(DF df, VF x_0, VF x_1, VF x_2, VF x_3,
                              F reducer) {
  const DF4 df4;
  constexpr size_t kMaxLanes = hn::MaxLanes(df);
  HWY_LANES_CONSTEXPR size_t kLanes = hn::Lanes(df);
  HWY_ALIGN T x_transposed[4 * kMaxLanes];
  hn::StoreInterleaved4<DF>(x_0, x_1, x_2, x_3, df, x_transposed);
  VF4 result = hn::Load(df4, x_transposed);
  for (int i = 1; i < kLanes; ++i) {
    result = reducer(result, hn::Load(df4, x_transposed + i * 4));
  }
  return result;
}

// Handles Up to 4 Q rows by NF*2 timesteps of flash attention.
template <int kNumQueries, class DF, class VF = hn::Vec<DF>>
static void HWY_INLINE FlashAttentionTileStepAndApplySoftCap(
    DF df, float att_cap, float one_over_att_cap, VF& x_0_p0, VF& x_0_p1,
    VF& x_1_p0, VF& x_1_p1, VF& x_2_p0, VF& x_2_p1, VF& x_3_p0, VF& x_3_p1,
    float* HWY_RESTRICT old_max, float* HWY_RESTRICT old_d,
    float* HWY_RESTRICT scales) {
  using DF4 = hn::CappedTag<float, 4>;
  const DF4 df4;
  using VF4 = hn::Vec<DF4>;
  static_assert(kNumQueries >= 1 && kNumQueries <= 4);
  VF4 new_max = hn::Set(df4, kNegInf);
  VF max_0, max_1, max_2, max_3 = hn::Zero(df);
  max_0 = hn::Max(x_0_p0, x_0_p1);
  if constexpr (kNumQueries >= 2) {
    max_1 = hn::Max(x_1_p0, x_1_p1);
  }
  if constexpr (kNumQueries >= 3) {
    max_2 = hn::Max(x_2_p0, x_2_p1);
  }
  if constexpr (kNumQueries >= 4) {
    max_3 = hn::Max(x_3_p0, x_3_p1);
  }
  if constexpr (kNumQueries == 1) {
    new_max = hn::InsertLane(new_max, 0, hn::ReduceMax(df, max_0));
  } else {
    new_max = Reduce4(df, max_0, max_1, max_2, max_3,
                      [](auto a, auto b) HWY_ATTR { return hn::Max(a, b); });
  }
  if (att_cap > 0.0f) {
    VF4 cap = hn::Set(df4, att_cap);
    VF4 one_over_cap = hn::Set(df4, one_over_att_cap);
    new_max = hn::Mul(cap, hn::Tanh(df4, hn::Mul(new_max, one_over_cap)));
  }
  VF4 old_max_vf = hn::Set(df4, kNegInf);
  old_max_vf = hn::LoadU(df4, old_max);
  new_max = hn::Max(new_max, old_max_vf);
  auto changed_max = hn::Gt(new_max, hn::Set(df4, kNegInf));
  // TODO figure out what was wrong with broadcasts and change to that.
  hn::StoreU(new_max, df4, old_max);
  if constexpr (kNumQueries >= 1) {
    const VF new_max_0 = hn::Set(df, old_max[0]);
    x_0_p0 = hn::Exp(df, hn::Sub(x_0_p0, new_max_0));
    x_0_p1 = hn::Exp(df, hn::Sub(x_0_p1, new_max_0));
  }
  if constexpr (kNumQueries >= 2) {
    const VF new_max_0 = hn::Set(df, old_max[1]);
    x_1_p0 = hn::Exp(df, hn::Sub(x_1_p0, new_max_0));
    x_1_p1 = hn::Exp(df, hn::Sub(x_1_p1, new_max_0));
  }
  if constexpr (kNumQueries >= 3) {
    const VF new_max_0 = hn::Set(df, old_max[2]);
    x_2_p0 = hn::Exp(df, hn::Sub(x_2_p0, new_max_0));
    x_2_p1 = hn::Exp(df, hn::Sub(x_2_p1, new_max_0));
  }
  if constexpr (kNumQueries >= 4) {
    const VF new_max_0 = hn::Set(df, old_max[3]);
    x_3_p0 = hn::Exp(df, hn::Sub(x_3_p0, new_max_0));
    x_3_p1 = hn::Exp(df, hn::Sub(x_3_p1, new_max_0));
  }
  VF4 old_d_vf = hn::Set(df4, 0.0f);
  old_d_vf = hn::LoadU(df4, old_d);
  VF4 scale = hn::Mul(old_d_vf, hn::Exp(df4, hn::Sub(old_max_vf, new_max)));

  VF4 x_sum = hn::Zero(df4);
  if constexpr (kNumQueries == 1) {
    x_sum = hn::Set(df4, hn::ReduceSum(df, x_0_p0) + hn::ReduceSum(df, x_0_p1));
  } else {
    VF x_0_sum = hn::Add(x_0_p0, x_0_p1);
    VF x_1_sum = hn::Add(x_1_p0, x_1_p1);
    VF x_2_sum = hn::Add(x_2_p0, x_2_p1);
    VF x_3_sum = hn::Add(x_3_p0, x_3_p1);
    x_sum = Reduce4(df, x_0_sum, x_1_sum, x_2_sum, x_3_sum,
                    [](auto a, auto b) HWY_ATTR { return hn::Add(a, b); });
  }
  old_d_vf = hn::Add(scale, x_sum);
  auto non_zero_mask = hn::Gt(old_d_vf, hn::Set(df4, 0.0f));
  const VF zero = hn::Zero(df);
  const VF4 zero4 = hn::Zero(df4);
  const VF4 one_over_d =
      hn::MaskedDivOr(zero4, non_zero_mask, hn::Set(df4, 1.0f), old_d_vf);
  HWY_ALIGN float tmp_one_over_d[4];
  hn::Store(one_over_d, df4, tmp_one_over_d);
  hn::BlendedStore(old_d_vf, changed_max, df4, old_d);
  scale = hn::Mul(scale, one_over_d);
  hn::BlendedStore(scale, changed_max, df4, scales);
  if (hn::ExtractLane(old_d_vf, 0) > 0.0f && scales[0] != 1.0f) {
    const VF one_over_d_0 = hn::Set(df, tmp_one_over_d[0]);
    x_0_p0 = hn::Mul(x_0_p0, one_over_d_0);
    x_0_p1 = hn::Mul(x_0_p1, one_over_d_0);
  } else {
    x_0_p0 = zero;
    x_0_p1 = zero;
  }
  if constexpr (kNumQueries >= 2) {
    if (hn::ExtractLane(old_d_vf, 1) > 0.0f && scales[1] != 1.0f) {
      const VF one_over_d_1 = hn::Set(df, tmp_one_over_d[1]);
      x_1_p0 = hn::Mul(x_1_p0, one_over_d_1);
      x_1_p1 = hn::Mul(x_1_p1, one_over_d_1);
    } else {
      x_1_p0 = zero;
      x_1_p1 = zero;
    }
  }
  if constexpr (kNumQueries >= 3) {
    if (hn::ExtractLane(old_d_vf, 2) > 0.0f && scales[2] != 1.0f) {
      const VF one_over_d_2 = hn::Set(df, tmp_one_over_d[2]);
      x_2_p0 = hn::Mul(x_2_p0, one_over_d_2);
      x_2_p1 = hn::Mul(x_2_p1, one_over_d_2);
    } else {
      x_2_p0 = zero;
      x_2_p1 = zero;
    }
  }
  if constexpr (kNumQueries >= 4) {
    if (hn::ExtractLane(old_d_vf, 3) > 0.0f && scales[3] != 1.0f) {
      const VF one_over_d_3 = hn::Set(df, tmp_one_over_d[3]);
      x_3_p0 = hn::Mul(x_3_p0, one_over_d_3);
      x_3_p1 = hn::Mul(x_3_p1, one_over_d_3);
    } else {
      x_3_p0 = zero;
      x_3_p1 = zero;
    }
  }
}

// Implements flash attention for a strip of 4 query vectors.
// It iterates through timesteps in K from `start_pos` up to `max_last_pos`.
// Timesteps up to `min_last_pos` (*) are processed in tiles of shape 4 Q rows
// by NF timesteps in K for efficiency while timesteps between `min_last_pos +
// 1` and `max_last_pos` are processed one-by-one to handle differing `last_pos`
// values within the strip.
// (*) Actually, it only iterates through
// `min_last_pos - (min_last_pos + 1 - start_pos) % NF` in tiles, as the tiled
// computation can, for obvious reasons, only process an integer number of
// tiles.
//
// @param q The query matrix [batch_size * q_heads, qkv_dim] in BF16 format.
// @param q_offsets Offsets from `q.Row(0)` to the start of the 4 query
//   vectors to be processed in this tile.
// @param k Key matrix [seq_len, qkv_dim] from KV cache.
// @param start_pos The first token position in the KV cache to attend to.
// @param last_pos An array of 4 indices giving the last token position
//   (inclusive) that each of the 4 queries may attend to.
// @param min_last_pos The minimum value in `last_pos`. Timesteps up to this
//   position can be processed efficiently in batches.
// @param max_last_pos The maximum value in `last_pos`. Timesteps between
//   `min_last_pos + 1` and this position are processed individually to
//   respect each query's `last_pos` limit.
// @param v Value matrix [seq_len, qkv_dim] from KV cache.
// @param layer_idx The index of the current transformer layer.
// @param activations Attention configurations and buffers.
// @param att_out Output buffer for attention results.
// @param out_offsets Offsets from `att_out.Row(0)` to store the 4 output
//   vectors.
// @param ctx Threading context.
// @param worker Worker thread index.
Tile4FlashState TileFlashAttention4(
    const MatPtrT<BF16>& q, const uint32_t* HWY_RESTRICT q_offsets,
    const MatPtrT<KV_t>& k, const size_t start_pos,
    const uint32_t* HWY_RESTRICT last_pos, const size_t min_last_pos,
    const size_t max_last_pos, const MatPtrT<KV_t>& v, const size_t layer_idx,
    const AttentionActivationsPtrs& activations, MatPtrT<float>& att_out,
    const uint32_t* HWY_RESTRICT out_offsets, ThreadingContext& ctx,
    const size_t worker) {
  GCPP_ZONE(ctx, worker, Zones::kFlashAttentionTileFlashAttention4);
  using DF = hn::ScalableTag<float>;
  const DF df;
  using VF = hn::Vec<DF>;
  constexpr size_t kMaxNF = hn::MaxLanes(df);
  const size_t kHTileSize = hn::Lanes(df);
  HWY_DASSERT(kHTileSize <= kMaxNF);
  constexpr size_t kVTileSize = 4;
  float scales[kVTileSize];
  for (size_t i = 0; i < kVTileSize; ++i) {
    hwy::ZeroBytes(att_out.Row(0) + out_offsets[i],
                   v.Cols() * sizeof(att_out.Row(0)[0]));
  }
  Tile4FlashState state;
  size_t position = start_pos;
  while (position + kHTileSize - 1 <= min_last_pos) {
    int32_t k_offsets[kMaxNF];
    size_t v_pos[kMaxNF];
    for (size_t i = 0; i < kHTileSize; ++i) {
      v_pos[i] = activations.div_seq_len.Remainder(position + i);
      k_offsets[i] = k.Row(v_pos[i]) - k.Row(0);
    }
    VF x0, x1, x2, x3;
    QDotKTilex4(df, q.Row(0), q_offsets, k, k_offsets, x0, x1, x2, x3);
    if (activations.config.att_cap > 0.0f) {
      // Compute tanh(x / cap) * cap, being LogitsSoftCap on the tile.
      VF cap = hn::Set(df, activations.config.att_cap);
      VF one_over_cap = hn::Div(hn::Set(df, 1.0f), cap);
      x0 = hn::Mul(cap, hn::Tanh(df, hn::Mul(x0, one_over_cap)));
      x1 = hn::Mul(cap, hn::Tanh(df, hn::Mul(x1, one_over_cap)));
      x2 = hn::Mul(cap, hn::Tanh(df, hn::Mul(x2, one_over_cap)));
      x3 = hn::Mul(cap, hn::Tanh(df, hn::Mul(x3, one_over_cap)));
    }
    scales[0] = SingleFlashAttentionRowVector(df, x0, state.row_states[0].max,
                                              state.row_states[0].d);
    scales[1] = SingleFlashAttentionRowVector(df, x1, state.row_states[1].max,
                                              state.row_states[1].d);
    scales[2] = SingleFlashAttentionRowVector(df, x2, state.row_states[2].max,
                                              state.row_states[2].d);
    scales[3] = SingleFlashAttentionRowVector(df, x3, state.row_states[3].max,
                                              state.row_states[3].d);
    MulByConstAndAddTile4(df, scales, x0, x1, x2, x3, v, v_pos, att_out.Row(0),
                          out_offsets, v.Cols());
    position += kHTileSize;
  }
  const hn::ScalableTag<BF16> dbf;
  const size_t qkv_dim = k.Cols();

  while (position <= max_last_pos) {
    size_t k_pos = activations.div_seq_len.Remainder(position);
    if (position <= last_pos[0]) {
      // Past the last position, x0 doesn't count.
      float x0 = Dot(dbf, MakeConstSpan(q.Row(0) + q_offsets[0], qkv_dim), 0,
                     k.Row(k_pos), qkv_dim);
      SingleFlashAttentionStep(x0, activations.config.att_cap,
                               state.row_states[0].max, state.row_states[0].d,
                               v.Row(k_pos), v.Cols(),
                               att_out.Row(0) + out_offsets[0]);
    }
    if (position <= last_pos[1]) {
      // Past the last position, x1 doesn't count.
      float x1 = Dot(dbf, MakeConstSpan(q.Row(0) + q_offsets[1], qkv_dim), 0,
                     k.Row(k_pos), qkv_dim);
      SingleFlashAttentionStep(x1, activations.config.att_cap,
                               state.row_states[1].max, state.row_states[1].d,
                               v.Row(k_pos), v.Cols(),
                               att_out.Row(0) + out_offsets[1]);
    }
    if (position <= last_pos[2]) {
      // Past the last position, x2 doesn't count.
      float x2 = Dot(dbf, MakeConstSpan(q.Row(0) + q_offsets[2], qkv_dim), 0,
                     k.Row(k_pos), qkv_dim);
      SingleFlashAttentionStep(x2, activations.config.att_cap,
                               state.row_states[2].max, state.row_states[2].d,
                               v.Row(k_pos), v.Cols(),
                               att_out.Row(0) + out_offsets[2]);
    }
    if (position <= last_pos[3]) {
      // Past the last position, x3 doesn't count.
      float x3 = Dot(dbf, MakeConstSpan(q.Row(0) + q_offsets[3], qkv_dim), 0,
                     k.Row(k_pos), qkv_dim);
      SingleFlashAttentionStep(x3, activations.config.att_cap,
                               state.row_states[3].max, state.row_states[3].d,
                               v.Row(k_pos), v.Cols(),
                               att_out.Row(0) + out_offsets[3]);
    }
    ++position;
  }
  return state;
}

// Rounds n to a number that can be used as the number of Q rows in a tile
// of flash attention.
static size_t RoundToSuitablePowerOf2(size_t n) {
  if (n < 4) return 1;
  if (n < 8) return 4;
  if (n < 16) return 8;
  if (n < 32) return 16;
  return 32;
}

// The vertical tile size is determined by the ability to use tiling and the
// target_parallelism. In practice the possible tile sizes in order of
// preference for efficiency are kNF, 4, 1, where kNF is likely to be 4 8 or
// 16. The final tile size is chosen to be the largest possible that allows
// for target_parallelism parallel tasks.
size_t GetVTileSize(size_t kNF, size_t num_head_groups, size_t num_tokens,
                    size_t total_tasks, size_t target_parallelism) {
  const size_t kMaxEqualK =
      RoundToSuitablePowerOf2(num_head_groups * num_tokens);
  const size_t kMinTileSize = (total_tasks / 4 >= target_parallelism) ? 4 : 1;
  return (kNF <= kMaxEqualK && total_tasks / kNF >= target_parallelism)
             ? kNF
             : std::min(kMinTileSize, kMaxEqualK);
}

// The nominal aim of attention is to combine 3 inputs Q[L,D], K[L,D], V[L,D]
// into a single output O[L,D].
// Conventional attention first computes A[L,L] = Q . KT
// followed by A = softmax(A) (over invididual rows).
// Then A is multiplied by V to get O[L,D].
// For each row of O, this takes a read of one row of Q L times, all of K,
// 3 write/reads of one row of A, read all of V, and read/write the one row of O
// L times. Ignoring the computation for now, and focusing just on memory,
// the one row of O takes L(4D+3) reads and L(D+3) writes.
// For the whole of Q, this is L^2(4D+3) reads and L^2(D+3) writes.
//
// Flash attention fuses these operations together, and has 3 operating modes:
// 1. NF rows of the result computed using tiles of registers of shape NFx8.
// 2. 4 rows of the result computed using tiles of registers of shape 4xNF.
// 3. One row (of Q and the result) at a time.
// In all cases the intermediate result (Q.KT) is never stored to memory.
// NF is the number of float lanes in a register, being 16 for AVX3. The softmax
// is converted to streaming form using the algorithm from:
// https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf.
// Q is transposed to Q_T[D,L] to make the dot product computation efficient.
//
// In mode 1:
// QDotKTileFloat computes NF Q rows x 8 K timesteps of Q.K dot products in one
// go, reducing reads of Q by 8 and reads of K by NF. The streaming softmax is
// computed entirely in registers, and a further NF registers to accumulate the
// results of the product of the softmax and V, reduce the number of reads of V
// by NF, and the reads/writes of O by 8.
// The reads are thus reduced to 2DL^2(1/8+1/NF) and writes reduced to DL^2/8,
// which on AVX3 is an overall reduction by about a factor of 10.
// Mode 1 can only be accessed if there is a large Qbatch size, or in multi-turn
// prefill, since in other cases, there is either a single K timestep (prefill)
// or a single num_heads set of Q rows (decode).
//
// In mode 2, the 4 rows of Q are computed against NF K timesteps in a tile,
// reducing the reads of Q by NF, and the reads of K by 4. The softmax and
// accumulation of the result is done in registers, cutting the reads of V by 4.
// The reads/writes of O are reduced by a factor of NF.
// The overall reduction is limited by the need to use gather to load K.
// Transposing K would be possible, but is complicated by the wraparound.
// Mode 2 can be used in all cases when there are at least 4 attention heads,
// but it may be prefereable to use mode 3 when the batch size is small to
// maximise parallelism.
//
// In mode 3, a single row of Q is computed against a single K timestep at a
// time, using SingleFlashAttention. In this case there is no reduction in the
// reads of Q or K, or V, or O, but the reads/writes of the intermediate A are
// still eliminated.
//
// A further complication is that real attention is not as simple as documented
// in the paper and above. There are multiple query heads, differing KV, and
// different sequence lengths, so a lot of the work in FlashAttention is making
// sure that a collection of q rows with the same KV and sequence length are
// grouped together so that mode 1 or 2 can be used, and choosing which of the
// 3 modes to use for best efficiency.
void FlashAttention(const size_t num_tokens, const size_t target_parallelism,
                    const size_t layer_idx, const MatPtr& query_norm_scale,
                    AttentionActivationsPtrs& activations, QBatch& qbatch,
                    ThreadingContext& ctx) {
  GCPP_ZONE(ctx, 0, Zones::kFlashAttentionInclusive);
  RMSNormAndPositionalEncoding(num_tokens, qbatch, activations.q,
                               query_norm_scale, layer_idx, activations, ctx);
  const hwy::Divisor div_qbatch(qbatch.Size());
  // Compress q to q_bf.
  ParallelFor(
      Parallelism::kWithinCluster, activations.q.Rows(), ctx,
      /*cluster_idx=*/0, Callers::kFlashAttention,
      [&](size_t row, size_t worker) {
        CompressPerThread tls;
        const hn::ScalableTag<float> df;
        CompressTraits<BF16>::Compress(
            df, activations.q.Row(row), activations.q.Cols(), tls,
            MakeSpan(activations.q_bf.Row(row), activations.q_bf.Cols()), 0);
      });
  const LayerConfig& layer_config = activations.config.layer_configs[layer_idx];
  const size_t qkv_dim = layer_config.qkv_dim;

  // A "head group" in the context of GQA refers to a collection of query
  // heads that share the same key and value heads.
  const size_t kHeadGroups = layer_config.heads / layer_config.kv_heads;
  const size_t cache_layer_size = layer_config.CacheLayerSize();
  const size_t seq_len =
      static_cast<size_t>(activations.div_seq_len.GetDivisor());
  const size_t token_batch = num_tokens * div_qbatch.GetDivisor();
  const size_t total_tasks = token_batch * layer_config.heads;

  using DF = hn::ScalableTag<float>;
  const DF df;
  const size_t kNF = hn::Lanes(df);
  constexpr size_t kMaxNF = hn::MaxLanes(df);
  HWY_DASSERT(kNF <= kMaxNF);
  const size_t kVTileSize = GetVTileSize(kNF, kHeadGroups, num_tokens,
                                         total_tasks, target_parallelism);
  // Only transpose Q if we are using tiling.
  if (kVTileSize == kNF) {
    size_t max_last = 0, min_start = std::numeric_limits<size_t>::max();
    for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
      size_t pos = qbatch.Pos(qi);
      const size_t start = StartPos(pos, activations.config, layer_idx);
      pos += num_tokens - 1;
      const size_t end = qbatch.PrefixEnd(qi);
      if (end > 0 && end - 1 > pos) {
        pos = end - 1;
      }
      max_last = std::max(max_last, pos);
      min_start = std::min(min_start, start);
    }
    if (max_last - min_start + 1 >= kNFx8HTileSize) {
      // q has shape [batch, qbatch][head, qkv_dim].
      // We transpose it to [qkv_dim][qbatch, head, batch] in order to make the
      // maximum possible number of consecutive columns have the same KV
      // matrices. Each thread will process a tile of NF columns of QT so the
      // starting column index of QT is just the task index * kVTileSize.
      TransposeQ(activations.q, activations.q_T, qbatch.Size(), ctx);
    }
  }
  const size_t num_thread_tasks = hwy::DivCeil(total_tasks, kVTileSize);
  const hwy::Divisor div_tokens(num_tokens);
  // All layers should have the same number of heads.
  HWY_DASSERT(activations.div_heads.GetDivisor() == layer_config.heads);

  // For each head/token/query, compute fused flash Q.K, softmax and weighted V.
  const auto func = [&](const size_t task, size_t worker) HWY_ATTR {
    GCPP_ZONE(ctx, worker, Zones::kFlashAttentionFlashAttention);
    // Offsets into original Q for each row in the tile.
    uint32_t q_offsets[kMaxNF];
    // Offsets into att_out for each row in the tile.
    uint32_t out_offsets[kMaxNF];
    // Start positions for each row in the tile.
    size_t start_positions[kMaxNF];
    // Last positions for each row in the tile. Inclusive.
    uint32_t last_pos[kMaxNF];
    // min and max last positions across all rows in the tile determines when
    // TileFlashAttention switches to single vector mode to handle the
    // ragged sequence lengths.
    size_t min_last_pos = std::numeric_limits<size_t>::max();
    size_t max_last_pos = 0;
    // Indices into the qbatch.KV for each row in the tile.
    size_t qi_indices[kMaxNF];
    // Indices into the kv_cache for each row in the tile.
    size_t kv_offsets[kMaxNF];
    // first_task is [qbatch, head, token].
    const size_t first_task = task * kVTileSize;
    const size_t last_task = first_task + kVTileSize - 1;
    bool use_tile_attention = kVTileSize > 1 && last_task < total_tasks;
    for (size_t offset = 0;
         offset < kVTileSize && first_task + offset < total_tasks; ++offset) {
      const size_t batch_idx = div_tokens.Remainder(first_task + offset);
      const size_t qh = div_tokens.Divide(first_task + offset);
      const size_t head = activations.div_heads.Remainder(qh);
      const size_t qi = activations.div_heads.Divide(qh);
      const size_t tq_idx = div_qbatch.GetDivisor() * batch_idx + qi;
      qi_indices[offset] = qi;

      // Find the token position in the query and calculate
      // the range of cache positions to attend to.
      const size_t pos = qbatch.Pos(qi) + batch_idx;
      const size_t start_pos = StartPos(pos, activations.config, layer_idx);
      start_positions[offset] = start_pos;
      size_t last = pos;
      const size_t prefix_end = qbatch.PrefixEnd(qi);
      if (prefix_end > 0 && prefix_end - 1 > last) {
        // last_pos in `TileFlashAttention` is inclusive.
        last = prefix_end - 1;
      }
      last_pos[offset] = last;
      min_last_pos = HWY_MIN(min_last_pos, last);
      max_last_pos = HWY_MAX(max_last_pos, last);
      q_offsets[offset] = activations.q_bf.Row(tq_idx) + head * qkv_dim -
                          activations.q_bf.Row(0);
      out_offsets[offset] = activations.att_out.Row(tq_idx) + head * qkv_dim -
                            activations.att_out.Row(0);
      const size_t kv_index = head / kHeadGroups;
      const size_t head_offset = kv_index * qkv_dim * 2;
      kv_offsets[offset] = layer_idx * cache_layer_size + head_offset;
      // If any of the parameters in this if statement differ within this task,
      // then we can't use TileFlashAttention. TileFlashAttention requires that
      // all rows in the tile have the same K and V matrices, and Q starts at
      // the same position. The end positions do not have to be the equal.
      if (start_positions[offset] != start_positions[0] ||
          qi_indices[offset] != qi_indices[0] ||
          kv_offsets[offset] != kv_offsets[0]) {
        use_tile_attention = false;
      }
    }
    for (size_t offset = 0;
         offset < kVTileSize && first_task + offset < total_tasks; ++offset) {
      auto& kv_cache = qbatch.KV(qi_indices[offset]).kv_cache;
      MatPtrT<KV_t> k("k_view", Extents2D(seq_len, qkv_dim));
      k.SetPtr(kv_cache.Row(0) + kv_offsets[offset], kv_cache.Stride());
      MatPtrT<KV_t> v("v_view", Extents2D(seq_len, qkv_dim));
      v.SetPtr(kv_cache.Row(0) + kv_offsets[offset] + qkv_dim,
               kv_cache.Stride());
      if (use_tile_attention) {
        // To avoid duplicating the code to setup K and V, the call to
        // TileFlashAttention is inside the loop over tasks, even though it
        // handles all rows in the task at once.
        StridedView<BF16> qT =
            StridedView<BF16>(activations.q_T.Row(0) + first_task, kVTileSize,
                              activations.q_T.Stride());
        if (kVTileSize == kNF) {
          // We can still use TileFlashAttention even if we didn't transpose Q
          // above. The condition used for transposing Q above is more general
          // and easier to compute than the condition used within
          // TileFlashAttention that min_last_pos - start_positions[offset] <
          // kNFx8HTileSize. In this case, qT is never used. Some tasks might
          // use qT and some might not, which is why the more general condition
          // is used above to catch all cases where qT will be used.
          TileFlashAttention(activations.q_bf, q_offsets, qT, k,
                             start_positions[offset], last_pos, min_last_pos,
                             max_last_pos, v, layer_idx, activations,
                             activations.att_out, out_offsets, ctx, worker);
        } else if (kVTileSize == 4) {
          TileFlashAttention4(activations.q_bf, q_offsets, k,
                              start_positions[offset], last_pos, min_last_pos,
                              max_last_pos, v, layer_idx, activations,
                              activations.att_out, out_offsets, ctx, worker);
        } else {
          HWY_UNREACHABLE;
        }
        break;
      } else {
        SingleFlashAttention(start_positions[offset], last_pos[offset],
                             activations.q_bf.Row(0) + q_offsets[offset], k, v,
                             layer_idx, activations,
                             activations.att_out.Row(0) + out_offsets[offset],
                             ctx, worker);
      }
    }
  };

  {
    PROFILER_ZONE("Gen.FlashAttention.ForkJoin");
    // Full parallelism is helpful, SmallParallelFor is insufficient.
    HierarchicalParallelFor(num_thread_tasks, ctx, Callers::kFlashAttention,
                            func);
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();
