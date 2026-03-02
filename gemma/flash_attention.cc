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
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#include "gemma/flash_structs.h"
#include "gemma/kv_cache.h"
#include "gemma/query.h"
#include "util/basics.h"
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
#include "hwy/contrib/math/fast_math-inl.h"

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
HWY_INLINE void SingleFlashAttentionStep(float x, float cap, float& old_max,
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
static HWY_INLINE VF4 Reduce4(DF df, VF x_0, VF x_1, VF x_2, VF x_3,
                              F reducer) {
  const DF4 df4;
  constexpr size_t kMaxLanes = hn::MaxLanes(df);
  HWY_LANES_CONSTEXPR size_t kLanes = hn::Lanes(df);
  HWY_ALIGN T x_transposed[4 * kMaxLanes];
  hn::StoreInterleaved4(x_0, x_1, x_2, x_3, df, x_transposed);
  VF x01 =
      reducer(hn::Load(df, x_transposed), hn::Load(df, x_transposed + kLanes));
  VF x23 = reducer(hn::Load(df, x_transposed + 2 * kLanes),
                   hn::Load(df, x_transposed + 3 * kLanes));
  VF x0123 = reducer(x01, x23);
  hn::Store(x0123, df, x_transposed);

  VF4 result = hn::Load(df4, x_transposed);
  for (int i = 1; i < kLanes / 4; ++i) {
    result = reducer(result, hn::Load(df4, x_transposed + i * 4));
  }
  return result;
}

// Returns vector with 8 lanes. Shouldn't be on architectures with less than 8
// lanes per vector.
template <class DF, typename T = hn::TFromD<DF>,
          class DF8 = hn::CappedTag<T, 8>, class VF8 = hn::Vec<DF8>,
          class VF = hn::Vec<DF>, typename F>
static HWY_INLINE VF8 Reduce8(DF df, VF x_0, VF x_1, VF x_2, VF x_3, VF x_4,
                              VF x_5, VF x_6, VF x_7, F reducer) {
  auto res0123 = Reduce4(df, x_0, x_1, x_2, x_3, reducer);
  auto res4567 = Reduce4(df, x_4, x_5, x_6, x_7, reducer);

  using DF4 = hn::CappedTag<T, 4>;
  const DF4 df4;
  const DF8 df8;
  HWY_ALIGN T buf[8];
  hn::Store(res0123, df4, buf);
  hn::Store(res4567, df4, buf + 4);
  return hn::Load(df8, buf);
}

// Handles Up to 4 Q rows by NF*2 timesteps of flash attention.
template <int kNumQueries, class DF, class VF = hn::Vec<DF>>
static HWY_INLINE void FlashAttentionTileStepAndApplySoftCap4(
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
  hn::StoreU(new_max, df4, old_max);
  auto apply_exp = [&](int i, VF& x_p0, VF& x_p1) HWY_ATTR {
    const VF new_max_i = hn::Set(df, old_max[i]);
    x_p0 = hn::FastExp(df, hn::Sub(x_p0, new_max_i));
    x_p1 = hn::FastExp(df, hn::Sub(x_p1, new_max_i));
  };
  if constexpr (kNumQueries >= 1) {
    apply_exp(0, x_0_p0, x_0_p1);
  }
  if constexpr (kNumQueries >= 2) {
    apply_exp(1, x_1_p0, x_1_p1);
  }
  if constexpr (kNumQueries >= 3) {
    apply_exp(2, x_2_p0, x_2_p1);
  }
  if constexpr (kNumQueries >= 4) {
    apply_exp(3, x_3_p0, x_3_p1);
  }
  VF4 old_d_vf = hn::Set(df4, 0.0f);
  old_d_vf = hn::LoadU(df4, old_d);

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
  VF4 scale = hn::Mul(old_d_vf, hn::Exp(df4, hn::Sub(old_max_vf, new_max)));
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
  // same as lambda
  auto mul_or_zero = [&](VF& x_p0, VF& x_p1, int i) HWY_ATTR {
    if (HWY_LIKELY(old_d[i] > 0.0f && scales[i] != 1.0f)) {
      const VF one_over_d_i = hn::Set(df, tmp_one_over_d[i]);
      x_p0 = hn::Mul(x_p0, one_over_d_i);
      x_p1 = hn::Mul(x_p1, one_over_d_i);
    } else {
      x_p0 = zero;
      x_p1 = zero;
    }
  };
  mul_or_zero(x_0_p0, x_0_p1, 0);
  if constexpr (kNumQueries >= 2) {
    mul_or_zero(x_1_p0, x_1_p1, 1);
  }
  if constexpr (kNumQueries >= 3) {
    mul_or_zero(x_2_p0, x_2_p1, 2);
  }
  if constexpr (kNumQueries >= 4) {
    mul_or_zero(x_3_p0, x_3_p1, 3);
  }
}

template <int kNumQueries, class DF, class VF = hn::Vec<DF>>
static HWY_INLINE void FlashAttentionTileStepAndApplySoftCap8(
    DF df, float att_cap, float one_over_att_cap, VF& x_0_p0, VF& x_0_p1,
    VF& x_1_p0, VF& x_1_p1, VF& x_2_p0, VF& x_2_p1, VF& x_3_p0, VF& x_3_p1,
    VF& x_4_p0, VF& x_4_p1, VF& x_5_p0, VF& x_5_p1, VF& x_6_p0, VF& x_6_p1,
    VF& x_7_p0, VF& x_7_p1, float* HWY_RESTRICT old_max,
    float* HWY_RESTRICT old_d, float* HWY_RESTRICT scales) {
  using DF8 = hn::CappedTag<float, 8>;
  const DF8 df8;
  using VF8 = hn::Vec<DF8>;
  static_assert(kNumQueries >= 1 && kNumQueries <= 8);
  VF8 new_max = hn::Set(df8, kNegInf);
  VF max_0, max_1, max_2, max_3, max_4, max_5, max_6, max_7 = hn::Zero(df);
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
  if constexpr (kNumQueries >= 5) {
    max_4 = hn::Max(x_4_p0, x_4_p1);
  }
  if constexpr (kNumQueries >= 6) {
    max_5 = hn::Max(x_5_p0, x_5_p1);
  }
  if constexpr (kNumQueries >= 7) {
    max_6 = hn::Max(x_6_p0, x_6_p1);
  }
  if constexpr (kNumQueries >= 8) {
    max_7 = hn::Max(x_7_p0, x_7_p1);
  }

  if constexpr (kNumQueries == 1) {
    new_max = hn::InsertLane(new_max, 0, hn::ReduceMax(df, max_0));
  } else {
    new_max =
        Reduce8(df, max_0, max_1, max_2, max_3, max_4, max_5, max_6, max_7,
                [](auto a, auto b) HWY_ATTR { return hn::Max(a, b); });
  }
  if (att_cap > 0.0f) {
    VF8 cap = hn::Set(df8, att_cap);
    VF8 one_over_cap = hn::Set(df8, one_over_att_cap);
    new_max = hn::Mul(cap, hn::Tanh(df8, hn::Mul(new_max, one_over_cap)));
  }
  VF8 old_max_vf = hn::Set(df8, kNegInf);
  old_max_vf = hn::LoadU(df8, old_max);
  new_max = hn::Max(new_max, old_max_vf);
  auto changed_max = hn::Gt(new_max, hn::Set(df8, kNegInf));
  hn::StoreU(new_max, df8, old_max);

  auto apply_exp = [&](int i, VF& x_p0, VF& x_p1) HWY_ATTR {
    const VF new_max_i = hn::Set(df, old_max[i]);
    x_p0 = hn::Exp(df, hn::Sub(x_p0, new_max_i));
    x_p1 = hn::Exp(df, hn::Sub(x_p1, new_max_i));
  };

  if constexpr (kNumQueries >= 1) {
    apply_exp(0, x_0_p0, x_0_p1);
  }
  if constexpr (kNumQueries >= 2) {
    apply_exp(1, x_1_p0, x_1_p1);
  }
  if constexpr (kNumQueries >= 3) {
    apply_exp(2, x_2_p0, x_2_p1);
  }
  if constexpr (kNumQueries >= 4) {
    apply_exp(3, x_3_p0, x_3_p1);
  }
  if constexpr (kNumQueries >= 5) {
    apply_exp(4, x_4_p0, x_4_p1);
  }
  if constexpr (kNumQueries >= 6) {
    apply_exp(5, x_5_p0, x_5_p1);
  }
  if constexpr (kNumQueries >= 7) {
    apply_exp(6, x_6_p0, x_6_p1);
  }
  if constexpr (kNumQueries >= 8) {
    apply_exp(7, x_7_p0, x_7_p1);
  }
  VF8 old_d_vf = hn::Set(df8, 0.0f);
  old_d_vf = hn::LoadU(df8, old_d);

  VF8 x_sum = hn::Zero(df8);
  if constexpr (kNumQueries == 1) {
    x_sum = hn::Set(df8, hn::ReduceSum(df, x_0_p0) + hn::ReduceSum(df, x_0_p1));
  } else {
    VF x_0_sum = hn::Add(x_0_p0, x_0_p1);
    VF x_1_sum = hn::Add(x_1_p0, x_1_p1);
    VF x_2_sum = hn::Add(x_2_p0, x_2_p1);
    VF x_3_sum = hn::Add(x_3_p0, x_3_p1);
    VF x_4_sum = hn::Add(x_4_p0, x_4_p1);
    VF x_5_sum = hn::Add(x_5_p0, x_5_p1);
    VF x_6_sum = hn::Add(x_6_p0, x_6_p1);
    VF x_7_sum = hn::Add(x_7_p0, x_7_p1);
    x_sum = Reduce8(df, x_0_sum, x_1_sum, x_2_sum, x_3_sum, x_4_sum, x_5_sum,
                    x_6_sum, x_7_sum,
                    [](auto a, auto b) HWY_ATTR { return hn::Add(a, b); });
  }
  VF8 scale = hn::Mul(old_d_vf, hn::Exp(df8, hn::Sub(old_max_vf, new_max)));
  old_d_vf = hn::Add(scale, x_sum);
  auto non_zero_mask = hn::Gt(old_d_vf, hn::Set(df8, 0.0f));
  const VF zero = hn::Zero(df);
  const VF8 zero8 = hn::Zero(df8);
  const VF8 one_over_d =
      hn::MaskedDivOr(zero8, non_zero_mask, hn::Set(df8, 1.0f), old_d_vf);
  HWY_ALIGN float tmp_one_over_d[8];
  hn::Store(one_over_d, df8, tmp_one_over_d);
  hn::BlendedStore(old_d_vf, changed_max, df8, old_d);
  scale = hn::Mul(scale, one_over_d);
  hn::BlendedStore(scale, changed_max, df8, scales);
  auto mul_or_zero = [&](VF& x_p0, VF& x_p1, int i) HWY_ATTR {
    if (HWY_LIKELY(old_d[i] > 0.0f && scales[i] != 1.0f)) {
      const VF one_over_d_i = hn::Set(df, tmp_one_over_d[i]);
      x_p0 = hn::Mul(x_p0, one_over_d_i);
      x_p1 = hn::Mul(x_p1, one_over_d_i);
    } else {
      x_p0 = zero;
      x_p1 = zero;
    }
  };
  mul_or_zero(x_0_p0, x_0_p1, 0);
  if constexpr (kNumQueries >= 2) {
    mul_or_zero(x_1_p0, x_1_p1, 1);
  }
  if constexpr (kNumQueries >= 3) {
    mul_or_zero(x_2_p0, x_2_p1, 2);
  }
  if constexpr (kNumQueries >= 4) {
    mul_or_zero(x_3_p0, x_3_p1, 3);
  }
  if constexpr (kNumQueries >= 5) {
    mul_or_zero(x_4_p0, x_4_p1, 4);
  }
  if constexpr (kNumQueries >= 6) {
    mul_or_zero(x_5_p0, x_5_p1, 5);
  }
  if constexpr (kNumQueries >= 7) {
    mul_or_zero(x_6_p0, x_6_p1, 6);
  }
  if constexpr (kNumQueries >= 8) {
    mul_or_zero(x_7_p0, x_7_p1, 7);
  }
}
template <int kNumQueries, class DF, class VF = hn::Vec<DF>>
static HWY_INLINE void FlashAttentionTileStepAndApplySoftCap(
    DF df, float att_cap, float one_over_att_cap, VF& x_0_p0, VF& x_0_p1,
    VF& x_1_p0, VF& x_1_p1, VF& x_2_p0, VF& x_2_p1, VF& x_3_p0, VF& x_3_p1,
    VF& x_4_p0, VF& x_4_p1, VF& x_5_p0, VF& x_5_p1, VF& x_6_p0, VF& x_6_p1,
    VF& x_7_p0, VF& x_7_p1, float* HWY_RESTRICT old_max,
    float* HWY_RESTRICT old_d, float* HWY_RESTRICT scales, size_t q_group_idx,
    size_t kNumQueriesPerGroup) {
  constexpr int kFirstHalfAmountOfQueries = std::min(kNumQueries, 4);
  [[maybe_unused]] constexpr int kSecondHalfAmountOfQueries =
      kNumQueries - kFirstHalfAmountOfQueries;
  if constexpr (kNumQueries <= 4) {
    FlashAttentionTileStepAndApplySoftCap4<kFirstHalfAmountOfQueries>(
        df, att_cap, one_over_att_cap, x_0_p0, x_0_p1, x_1_p0, x_1_p1, x_2_p0,
        x_2_p1, x_3_p0, x_3_p1, old_max + (q_group_idx)*kNumQueriesPerGroup,
        old_d + (q_group_idx)*kNumQueriesPerGroup, scales);
  } else {
#if HWY_MAX_BYTES <= 16
    FlashAttentionTileStepAndApplySoftCap4<4>(
        df, att_cap, one_over_att_cap, x_0_p0, x_0_p1, x_1_p0, x_1_p1, x_2_p0,
        x_2_p1, x_3_p0, x_3_p1, old_max + (q_group_idx)*kNumQueriesPerGroup,
        old_d + (q_group_idx)*kNumQueriesPerGroup, scales);
    FlashAttentionTileStepAndApplySoftCap4<kSecondHalfAmountOfQueries>(
        df, att_cap, one_over_att_cap, x_4_p0, x_4_p1, x_5_p0, x_5_p1, x_6_p0,
        x_6_p1, x_7_p0, x_7_p1,
        old_max + (q_group_idx + 1) * kNumQueriesPerGroup,
        old_d + (q_group_idx + 1) * kNumQueriesPerGroup,
        scales + kNumQueriesPerGroup);
#else
    FlashAttentionTileStepAndApplySoftCap8<kNumQueries>(
        df, att_cap, one_over_att_cap, x_0_p0, x_0_p1, x_1_p0, x_1_p1, x_2_p0,
        x_2_p1, x_3_p0, x_3_p1, x_4_p0, x_4_p1, x_5_p0, x_5_p1, x_6_p0, x_6_p1,
        x_7_p0, x_7_p1, old_max + (q_group_idx)*kNumQueriesPerGroup,
        old_d + (q_group_idx)*kNumQueriesPerGroup, scales);
#endif
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

template <int kNumQueries, typename Q_T, class DQ_T, class VQ_T = hn::Vec<DQ_T>,
          typename T>
static HWY_INLINE void QDotKTilexUpTo8TransposedKDoubleWidth(
    DQ_T df, const Q_T* HWY_RESTRICT q, const Q_T* HWY_RESTRICT q2,
    const T* HWY_RESTRICT k_transposed_tile, size_t qkv_dim, VQ_T& sum0_p0,
    VQ_T& sum0_p1, VQ_T& sum1_p0, VQ_T& sum1_p1, VQ_T& sum2_p0, VQ_T& sum2_p1,
    VQ_T& sum3_p0, VQ_T& sum3_p1, VQ_T& sum4_p0, VQ_T& sum4_p1, VQ_T& sum5_p0,
    VQ_T& sum5_p1, VQ_T& sum6_p0, VQ_T& sum6_p1, VQ_T& sum7_p0, VQ_T& sum7_p1) {
  const PackedSpan<const T> k_transposed_span =
      MakeConstSpan(k_transposed_tile, gcpp::KVCache::kTileSize * qkv_dim);
  HWY_DASSERT(kNumQueries <= 8);
  HWY_DASSERT(gcpp::KVCache::kTileSize >=
              hn::Lanes(df) * 2);  // So we can decompress 2 lanes at a time.
  sum0_p0 = hn::Zero(df);
  sum0_p1 = hn::Zero(df);
  if constexpr (kNumQueries >= 2) {
    sum1_p0 = hn::Zero(df);
    sum1_p1 = hn::Zero(df);
  }
  if constexpr (kNumQueries >= 3) {
    sum2_p0 = hn::Zero(df);
    sum2_p1 = hn::Zero(df);
  }
  if constexpr (kNumQueries >= 4) {
    sum3_p0 = hn::Zero(df);
    sum3_p1 = hn::Zero(df);
  }
  if constexpr (kNumQueries >= 5) {
    sum4_p0 = hn::Zero(df);
    sum4_p1 = hn::Zero(df);
  }
  if constexpr (kNumQueries >= 6) {
    sum5_p0 = hn::Zero(df);
    sum5_p1 = hn::Zero(df);
  }
  if constexpr (kNumQueries >= 7) {
    sum6_p0 = hn::Zero(df);
    sum6_p1 = hn::Zero(df);
  }
  if constexpr (kNumQueries >= 8) {
    sum7_p0 = hn::Zero(df);
    sum7_p1 = hn::Zero(df);
  }

  constexpr int kFirstHalfAmountOfQueries = std::min(kNumQueries, 4);
  constexpr int kSecondHalfAmountOfQueries =
      kNumQueries - kFirstHalfAmountOfQueries;
  HWY_UNROLL(1)
  for (size_t i = 0; i < qkv_dim; ++i) {
    VQ_T k_vec1, k_vec2;
    if constexpr (HWY_TARGET == HWY_AVX2) {
      hwy::Prefetch(k_transposed_span.ptr + (i + 3) * gcpp::KVCache::kTileSize);
      hwy::Prefetch(k_transposed_span.ptr + (i + 4) * gcpp::KVCache::kTileSize);
    }
    Decompress2(df, k_transposed_span, i * gcpp::KVCache::kTileSize, k_vec1,
                k_vec2);
    sum0_p0 = hn::MulAdd(
        k_vec1, hn::Set(df, q[i * kFirstHalfAmountOfQueries + 0]), sum0_p0);
    sum0_p1 = hn::MulAdd(
        k_vec2, hn::Set(df, q[i * kFirstHalfAmountOfQueries + 0]), sum0_p1);
    if constexpr (kNumQueries >= 2) {
      sum1_p0 = hn::MulAdd(
          k_vec1, hn::Set(df, q[i * kFirstHalfAmountOfQueries + 1]), sum1_p0);
      sum1_p1 = hn::MulAdd(
          k_vec2, hn::Set(df, q[i * kFirstHalfAmountOfQueries + 1]), sum1_p1);
    }
    if constexpr (kNumQueries >= 3) {
      sum2_p0 = hn::MulAdd(
          k_vec1, hn::Set(df, q[i * kFirstHalfAmountOfQueries + 2]), sum2_p0);
      sum2_p1 = hn::MulAdd(
          k_vec2, hn::Set(df, q[i * kFirstHalfAmountOfQueries + 2]), sum2_p1);
    }
    if constexpr (kNumQueries >= 4) {
      sum3_p0 = hn::MulAdd(
          k_vec1, hn::Set(df, q[i * kFirstHalfAmountOfQueries + 3]), sum3_p0);
      sum3_p1 = hn::MulAdd(
          k_vec2, hn::Set(df, q[i * kFirstHalfAmountOfQueries + 3]), sum3_p1);
    }
    if constexpr (kNumQueries >= 5) {
      sum4_p0 = hn::MulAdd(
          k_vec1, hn::Set(df, q2[i * kSecondHalfAmountOfQueries + 0]), sum4_p0);
      sum4_p1 = hn::MulAdd(
          k_vec2, hn::Set(df, q2[i * kSecondHalfAmountOfQueries + 0]), sum4_p1);
    }
    if constexpr (kNumQueries >= 6) {
      sum5_p0 = hn::MulAdd(
          k_vec1, hn::Set(df, q2[i * kSecondHalfAmountOfQueries + 1]), sum5_p0);
      sum5_p1 = hn::MulAdd(
          k_vec2, hn::Set(df, q2[i * kSecondHalfAmountOfQueries + 1]), sum5_p1);
    }
    if constexpr (kNumQueries >= 7) {
      sum6_p0 = hn::MulAdd(
          k_vec1, hn::Set(df, q2[i * kSecondHalfAmountOfQueries + 2]), sum6_p0);
      sum6_p1 = hn::MulAdd(
          k_vec2, hn::Set(df, q2[i * kSecondHalfAmountOfQueries + 2]), sum6_p1);
    }
    if constexpr (kNumQueries >= 8) {
      sum7_p0 = hn::MulAdd(
          k_vec1, hn::Set(df, q2[i * kSecondHalfAmountOfQueries + 3]), sum7_p0);
      sum7_p1 = hn::MulAdd(
          k_vec2, hn::Set(df, q2[i * kSecondHalfAmountOfQueries + 3]), sum7_p1);
    }
  }
}

template <int kNumQueries, class DF, class VF = hn::Vec<DF>, typename T>
static HWY_INLINE void QDotKTilexUpTo8TransposedKDoubleWidthBF16(
    DF df, const BF16* HWY_RESTRICT q, const BF16* HWY_RESTRICT q2,
    const T* HWY_RESTRICT k_transposed_tile, size_t qkv_dim, VF& sum0_p0,
    VF& sum0_p1, VF& sum1_p0, VF& sum1_p1, VF& sum2_p0, VF& sum2_p1,
    VF& sum3_p0, VF& sum3_p1, VF& sum4_p0, VF& sum4_p1, VF& sum5_p0,
    VF& sum5_p1, VF& sum6_p0, VF& sum6_p1, VF& sum7_p0, VF& sum7_p1) {
  using DBF = hn::ScalableTag<BF16>;
  const DBF dbf;
  using VBF = hn::Vec<DBF>;
  const PackedSpan<const T> k_transposed_span =
      MakeConstSpan(k_transposed_tile, gcpp::KVCache::kTileSize * qkv_dim);
  [[maybe_unused]] HWY_LANES_CONSTEXPR size_t lanes_bf16 = hn::Lanes(dbf);
  HWY_DASSERT(hn::Lanes(dbf) <= gcpp::KVCache::kTileSize);
  HWY_DASSERT(kNumQueries <= 8);
  HWY_DASSERT(gcpp::KVCache::kTileSize >=
              hn::Lanes(df) * 2);  // So we can decompress 2 lanes at a time.
  sum0_p0 = hn::Zero(df);
  sum0_p1 = hn::Zero(df);
  if constexpr (kNumQueries >= 2) {
    sum1_p0 = hn::Zero(df);
    sum1_p1 = hn::Zero(df);
  }
  if constexpr (kNumQueries >= 3) {
    sum2_p0 = hn::Zero(df);
    sum2_p1 = hn::Zero(df);
  }
  if constexpr (kNumQueries >= 4) {
    sum3_p0 = hn::Zero(df);
    sum3_p1 = hn::Zero(df);
  }
  if constexpr (kNumQueries >= 5) {
    sum4_p0 = hn::Zero(df);
    sum4_p1 = hn::Zero(df);
  }
  if constexpr (kNumQueries >= 6) {
    sum5_p0 = hn::Zero(df);
    sum5_p1 = hn::Zero(df);
  }
  if constexpr (kNumQueries >= 7) {
    sum6_p0 = hn::Zero(df);
    sum6_p1 = hn::Zero(df);
  }
  if constexpr (kNumQueries >= 8) {
    sum7_p0 = hn::Zero(df);
    sum7_p1 = hn::Zero(df);
  }
  VF helper_sum0_p0 = hn::Zero(df), helper_sum0_p1 = hn::Zero(df);
  VF helper_sum1_p0 = hn::Zero(df), helper_sum1_p1 = hn::Zero(df);
  VF helper_sum2_p0 = hn::Zero(df), helper_sum2_p1 = hn::Zero(df);
  VF helper_sum3_p0 = hn::Zero(df), helper_sum3_p1 = hn::Zero(df);
  VF helper_sum4_p0 = hn::Zero(df), helper_sum4_p1 = hn::Zero(df);
  VF helper_sum5_p0 = hn::Zero(df), helper_sum5_p1 = hn::Zero(df);
  VF helper_sum6_p0 = hn::Zero(df), helper_sum6_p1 = hn::Zero(df);
  VF helper_sum7_p0 = hn::Zero(df), helper_sum7_p1 = hn::Zero(df);
  const float* q_float_ptr = HWY_RCAST_ALIGNED(const float*, q);
  const float* q2_float_ptr = HWY_RCAST_ALIGNED(const float*, q2);
  constexpr int kFirstHalfAmountOfQueries = std::min(kNumQueries, 4);
  constexpr int kSecondHalfAmountOfQueries =
      kNumQueries - kFirstHalfAmountOfQueries;

  for (size_t i = 0; i < qkv_dim / 2; i++) {
    VBF k_vec1, k_vec2;
    Decompress2(dbf, k_transposed_span, i * 2 * gcpp::KVCache::kTileSize,
                k_vec1, k_vec2);

    VF q_0_as_float = hn::Set(df, q_float_ptr[i * kFirstHalfAmountOfQueries]);
    VBF q_0 = hn::BitCast(dbf, q_0_as_float);
    sum0_p0 =
        hn::ReorderWidenMulAccumulate(df, k_vec1, q_0, sum0_p0, helper_sum0_p0);
    sum0_p1 =
        hn::ReorderWidenMulAccumulate(df, k_vec2, q_0, sum0_p1, helper_sum0_p1);
    if constexpr (kNumQueries >= 2) {
      VF q_1_as_float =
          hn::Set(df, q_float_ptr[i * kFirstHalfAmountOfQueries + 1]);
      VBF q_1 = hn::BitCast(dbf, q_1_as_float);
      sum1_p0 = hn::ReorderWidenMulAccumulate(df, k_vec1, q_1, sum1_p0,
                                              helper_sum1_p0);
      sum1_p1 = hn::ReorderWidenMulAccumulate(df, k_vec2, q_1, sum1_p1,
                                              helper_sum1_p1);
    }
    if constexpr (kNumQueries >= 3) {
      VF q_2_as_float =
          hn::Set(df, q_float_ptr[i * kFirstHalfAmountOfQueries + 2]);
      VBF q_2 = hn::BitCast(dbf, q_2_as_float);
      sum2_p0 = hn::ReorderWidenMulAccumulate(df, k_vec1, q_2, sum2_p0,
                                              helper_sum2_p0);
      sum2_p1 = hn::ReorderWidenMulAccumulate(df, k_vec2, q_2, sum2_p1,
                                              helper_sum2_p1);
    }
    if constexpr (kNumQueries >= 4) {
      VF q_3_as_float =
          hn::Set(df, q_float_ptr[i * kFirstHalfAmountOfQueries + 3]);
      VBF q_3 = hn::BitCast(dbf, q_3_as_float);
      sum3_p0 = hn::ReorderWidenMulAccumulate(df, k_vec1, q_3, sum3_p0,
                                              helper_sum3_p0);
      sum3_p1 = hn::ReorderWidenMulAccumulate(df, k_vec2, q_3, sum3_p1,
                                              helper_sum3_p1);
    }
    if constexpr (kNumQueries >= 5) {
      VF q_4_as_float =
          hn::Set(df, q2_float_ptr[i * kSecondHalfAmountOfQueries + 0]);
      VBF q_4 = hn::BitCast(dbf, q_4_as_float);
      sum4_p0 = hn::ReorderWidenMulAccumulate(df, k_vec1, q_4, sum4_p0,
                                              helper_sum4_p0);
      sum4_p1 = hn::ReorderWidenMulAccumulate(df, k_vec2, q_4, sum4_p1,
                                              helper_sum4_p1);
    }
    if constexpr (kNumQueries >= 6) {
      VF q_5_as_float =
          hn::Set(df, q2_float_ptr[i * kSecondHalfAmountOfQueries + 1]);
      VBF q_5 = hn::BitCast(dbf, q_5_as_float);
      sum5_p0 = hn::ReorderWidenMulAccumulate(df, k_vec1, q_5, sum5_p0,
                                              helper_sum5_p0);
      sum5_p1 = hn::ReorderWidenMulAccumulate(df, k_vec2, q_5, sum5_p1,
                                              helper_sum5_p1);
    }
    if constexpr (kNumQueries >= 7) {
      VF q_6_as_float =
          hn::Set(df, q2_float_ptr[i * kSecondHalfAmountOfQueries + 2]);
      VBF q_6 = hn::BitCast(dbf, q_6_as_float);
      sum6_p0 = hn::ReorderWidenMulAccumulate(df, k_vec1, q_6, sum6_p0,
                                              helper_sum6_p0);
      sum6_p1 = hn::ReorderWidenMulAccumulate(df, k_vec2, q_6, sum6_p1,
                                              helper_sum6_p1);
    }
    if constexpr (kNumQueries >= 8) {
      VF q_7_as_float =
          hn::Set(df, q2_float_ptr[i * kSecondHalfAmountOfQueries + 3]);
      VBF q_7 = hn::BitCast(dbf, q_7_as_float);
      sum7_p0 = hn::ReorderWidenMulAccumulate(df, k_vec1, q_7, sum7_p0,
                                              helper_sum7_p0);
      sum7_p1 = hn::ReorderWidenMulAccumulate(df, k_vec2, q_7, sum7_p1,
                                              helper_sum7_p1);
    }
  }
#if HWY_NATIVE_DOT_BF16 == 0
  sum0_p0 = hn::Add(sum0_p0, helper_sum0_p0);
  sum0_p1 = hn::Add(sum0_p1, helper_sum0_p1);
  if constexpr (kNumQueries >= 2) {
    sum1_p0 = hn::Add(sum1_p0, helper_sum1_p0);
    sum1_p1 = hn::Add(sum1_p1, helper_sum1_p1);
  }
  if constexpr (kNumQueries >= 3) {
    sum2_p0 = hn::Add(sum2_p0, helper_sum2_p0);
    sum2_p1 = hn::Add(sum2_p1, helper_sum2_p1);
  }
  if constexpr (kNumQueries >= 4) {
    sum3_p0 = hn::Add(sum3_p0, helper_sum3_p0);
    sum3_p1 = hn::Add(sum3_p1, helper_sum3_p1);
  }
  if constexpr (kNumQueries >= 5) {
    sum4_p0 = hn::Add(sum4_p0, helper_sum4_p0);
    sum4_p1 = hn::Add(sum4_p1, helper_sum4_p1);
  }
  if constexpr (kNumQueries >= 6) {
    sum5_p0 = hn::Add(sum5_p0, helper_sum5_p0);
    sum5_p1 = hn::Add(sum5_p1, helper_sum5_p1);
  }
  if constexpr (kNumQueries >= 7) {
    sum6_p0 = hn::Add(sum6_p0, helper_sum6_p0);
    sum6_p1 = hn::Add(sum6_p1, helper_sum6_p1);
  }
  if constexpr (kNumQueries >= 8) {
    sum7_p0 = hn::Add(sum7_p0, helper_sum7_p0);
    sum7_p1 = hn::Add(sum7_p1, helper_sum7_p1);
  }
#endif
}

template <int kVTileSize, class DF, class VF = hn::Vec<DF>>
static HWY_INLINE void ApplySoftCap(DF df, float att_cap, float one_over_cap,
                                    VF& x0, VF& x1, VF& x2, VF& x3, VF& x4,
                                    VF& x5, VF& x6, VF& x7) {
  if (att_cap > 0.0f) {
    VF cap = hn::Set(df, att_cap);
    VF one_over_cap_vec = hn::Set(df, one_over_cap);
    x0 = hn::Mul(cap, hn::CallTanh(df, hn::Mul(x0, one_over_cap_vec)));
    if constexpr (kVTileSize >= 2) {
      x1 = hn::Mul(cap, hn::CallTanh(df, hn::Mul(x1, one_over_cap_vec)));
    }
    if constexpr (kVTileSize >= 3) {
      x2 = hn::Mul(cap, hn::CallTanh(df, hn::Mul(x2, one_over_cap_vec)));
    }
    if constexpr (kVTileSize >= 4) {
      x3 = hn::Mul(cap, hn::CallTanh(df, hn::Mul(x3, one_over_cap_vec)));
    }
    if constexpr (kVTileSize >= 5) {
      x4 = hn::Mul(cap, hn::CallTanh(df, hn::Mul(x4, one_over_cap_vec)));
    }
    if constexpr (kVTileSize >= 6) {
      x5 = hn::Mul(cap, hn::CallTanh(df, hn::Mul(x5, one_over_cap_vec)));
    }
    if constexpr (kVTileSize >= 7) {
      x6 = hn::Mul(cap, hn::CallTanh(df, hn::Mul(x6, one_over_cap_vec)));
    }
    if constexpr (kVTileSize >= 8) {
      x7 = hn::Mul(cap, hn::CallTanh(df, hn::Mul(x7, one_over_cap_vec)));
    }
  }
}

template <int kNumQueries, class DF, class VF = hn::Vec<DF>, typename DU,
          class VU = hn::Vec<DU>>
static HWY_NOINLINE void ApplyMasking(
    DF df, DU du, size_t position,
    const size_t* HWY_RESTRICT first_pos_per_query,
    const size_t* HWY_RESTRICT last_pos_per_query, VF& x0_p0, VF& x0_p1,
    VF& x1_p0, VF& x1_p1, VF& x2_p0, VF& x2_p1, VF& x3_p0, VF& x3_p1, VF& x4_p0,
    VF& x4_p1, VF& x5_p0, VF& x5_p1, VF& x6_p0, VF& x6_p1, VF& x7_p0,
    VF& x7_p1) {
  VU lane_indices = hn::Iota(du, 0);
  HWY_LANES_CONSTEXPR size_t kTileSize = hn::Lanes(df);
  auto per_lane_pos_p0 = hn::Add(hn::Set(du, position), lane_indices);
  auto per_lane_pos_p1 =
      hn::Add(hn::Set(du, position + kTileSize), lane_indices);

  VF neg_inf = hn::Set(df, kNegInf);

  auto apply_mask_for_query = [&](int query_idx, VF& x_p0, VF& x_p1) HWY_ATTR {
    const size_t first_pos = first_pos_per_query[query_idx];
    const size_t last_pos = last_pos_per_query[query_idx];

    auto valid_tokens_mask_p0 = hn::Ge(per_lane_pos_p0, hn::Set(du, first_pos));
    valid_tokens_mask_p0 = hn::And(
        valid_tokens_mask_p0, hn::Le(per_lane_pos_p0, hn::Set(du, last_pos)));
    x_p0 =
        hn::IfThenElse(hn::RebindMask(df, valid_tokens_mask_p0), x_p0, neg_inf);

    auto valid_tokens_mask_p1 = hn::Ge(per_lane_pos_p1, hn::Set(du, first_pos));
    valid_tokens_mask_p1 = hn::And(
        valid_tokens_mask_p1, hn::Le(per_lane_pos_p1, hn::Set(du, last_pos)));
    x_p1 =
        hn::IfThenElse(hn::RebindMask(df, valid_tokens_mask_p1), x_p1, neg_inf);
  };

  if constexpr (kNumQueries >= 1) {
    apply_mask_for_query(0, x0_p0, x0_p1);
  }
  if constexpr (kNumQueries >= 2) {
    apply_mask_for_query(1, x1_p0, x1_p1);
  }
  if constexpr (kNumQueries >= 3) {
    apply_mask_for_query(2, x2_p0, x2_p1);
  }
  if constexpr (kNumQueries >= 4) {
    apply_mask_for_query(3, x3_p0, x3_p1);
  }
  if constexpr (kNumQueries >= 5) {
    apply_mask_for_query(4, x4_p0, x4_p1);
  }
  if constexpr (kNumQueries >= 6) {
    apply_mask_for_query(5, x5_p0, x5_p1);
  }
  if constexpr (kNumQueries >= 7) {
    apply_mask_for_query(6, x6_p0, x6_p1);
  }
  if constexpr (kNumQueries >= 8) {
    apply_mask_for_query(7, x7_p0, x7_p1);
  }
}

// Performs tiled flash attention for arbitrary number of queries
// It depends on kv being tiled.
// Runs 2 loops one over tiles, and inner one over queries(up to 4 at a time).
// It moves NF*2 timesteps forward in kv at a time.
// Args:
// kvs - hwy::Span of MatPtrT<KV_T> of shape (kvs, (tile_count, qkv_dim *
// kTileSize * 2)) This span allows to pass kv cache that is not contiguous,
// all except for the last one should have theirs row count be true,
// as it will be used to figure out when to switch to the next one.
// q_T_in_groups_up_to_4 - Span of float* All except last float*
// should have (qkv_dim, 4) Last one can have any size up to 4.
// start_pos_per_query - start position in kv to start attention from ()
// last_pos_per_query - last position in kv to attend to (exclusive)
// queries_per_timestep - how many queries begin/end on the same timestep
// attention_shape - see struct definition for more details.
// att_cap - soft cap on attention logits
// att_out - MatPtrT<float> of shape (q_count, qkv_dim)
// exp_denominator_sums and max_logits: float* of shape:
// (RountedUpTo(q_count,4),)
// Need to be have multiple of 4 elements alocated and
// be initizalized If you need to compute over multiple chunks of kv's you can
// keep values between calls to this function and avoid explicit merge.
template <typename KV_T, typename Q_T>
HWY_NOINLINE void TileFlashAttentionReturnExpSumsAndMaxLogits(
    const hwy::Span<const MatPtrT<KV_T>> kvs, int q_count,
    const hwy::Span<const Q_T * HWY_RESTRICT> q_T_in_groups_up_to_4,
    hwy::Span<const size_t> start_pos_per_query,
    hwy::Span<const size_t> last_pos_per_query, const float att_cap,
    MatPtrT<float>& att_out, float* HWY_RESTRICT exp_denominator_sums,
    float* HWY_RESTRICT max_logits) {
  using DF = hn::ScalableTag<float>;
  const DF df;
  using VF = hn::Vec<DF>;
  using DU = hn::ScalableTag<uint32_t>;
  [[maybe_unused]] const DU du;
  constexpr int kTileSize = gcpp::KVCache::kTileSize;
  HWY_LANES_CONSTEXPR size_t kHTileSize = hn::Lanes(df);
  constexpr int kNumQueriesPerGroup = 4;
  constexpr int kNumQueriesPerLoop =
      (!HWY_ARCH_X86 || (HWY_TARGET <= HWY_AVX3)) ? 8 : 4;
  constexpr int kNumGroupsPerLoop = kNumQueriesPerLoop / kNumQueriesPerGroup;
  const size_t full_groups_of_queries = q_count / kNumQueriesPerGroup;
  const size_t num_loops = hwy::DivCeil(q_count, kNumQueriesPerLoop);
  const size_t qkv_dim = att_out.Cols();
  HWY_DASSERT(kHTileSize <= hn::MaxLanes(df));
  HWY_LANES_CONSTEXPR size_t step_size = kHTileSize * 2;
  size_t smallest_start_pos = std::numeric_limits<size_t>::max();
  size_t largest_last_pos = std::numeric_limits<size_t>::min();
  for (size_t i = 0; i < start_pos_per_query.size(); ++i) {
    smallest_start_pos = std::min(smallest_start_pos, start_pos_per_query[i]);
    largest_last_pos = std::max(largest_last_pos, last_pos_per_query[i]);
  }
  // start / end positions per group of 4 queries.
  std::vector<size_t, hwy::AlignedAllocator<size_t>> pos_data(num_loops * 4);
  hwy::Span<size_t> min_start_pos_per_group(pos_data.data(), num_loops);
  hwy::Span<size_t> max_start_pos_per_group(pos_data.data() + num_loops,
                                            num_loops);
  hwy::Span<size_t> min_last_pos_per_group(pos_data.data() + 2 * num_loops,
                                           num_loops);
  hwy::Span<size_t> max_last_pos_per_group(pos_data.data() + 3 * num_loops,
                                           num_loops);

  for (size_t i = 0; i < num_loops; ++i) {
    size_t min_start = std::numeric_limits<size_t>::max();
    size_t max_start = 0;
    size_t min_last = std::numeric_limits<size_t>::max();
    size_t max_last = 0;
    for (int j = 0; j < kNumQueriesPerLoop; ++j) {
      if (i * kNumQueriesPerLoop + j < q_count) {
        min_start = std::min(min_start,
                             start_pos_per_query[i * kNumQueriesPerLoop + j]);
        max_start = std::max(max_start,
                             start_pos_per_query[i * kNumQueriesPerLoop + j]);
        min_last =
            std::min(min_last, last_pos_per_query[i * kNumQueriesPerLoop + j]);
        max_last =
            std::max(max_last, last_pos_per_query[i * kNumQueriesPerLoop + j]);
      }
    }
    min_start_pos_per_group[i] = min_start;
    max_start_pos_per_group[i] = max_start;
    min_last_pos_per_group[i] = min_last;
    max_last_pos_per_group[i] = max_last;
  }
  const size_t base_pos = smallest_start_pos - (smallest_start_pos % kTileSize);
  const size_t rem = smallest_start_pos % kTileSize;
  const size_t num_skipped_sub_tiles = rem / step_size;
  size_t position = base_pos + num_skipped_sub_tiles * step_size;
  [[maybe_unused]] float one_over_cap = 1.0f / att_cap;
  std::vector<MatPtrT<float>> att_out_per_query;
  att_out_per_query.reserve(num_loops);
  for (size_t i = 0; i < num_loops; ++i) {
    att_out_per_query.emplace_back("att_out",
                                   Extents2D(kNumQueriesPerLoop, qkv_dim));
    att_out_per_query.back().SetPtr(att_out.Row(i * kNumQueriesPerLoop),
                                    att_out.Stride());
  }
  size_t current_kv_start_offset = 0;
  size_t current_kv_idx = 0;

  auto inner_loop = [&]<int kNumQueries>(int q_group_idx) HWY_ATTR {
    int loop_idx = q_group_idx / (kNumQueriesPerLoop / kNumQueriesPerGroup);
    if (position + step_size <= min_start_pos_per_group[loop_idx] ||
        position > max_last_pos_per_group[loop_idx]) {
      return;
    }
    VF x_0_p_0, x_0_p_1, x_1_p_0, x_1_p_1, x_2_p_0, x_2_p_1, x_3_p_0, x_3_p_1;
    VF x_4_p_0, x_4_p_1, x_5_p_0, x_5_p_1, x_6_p_0, x_6_p_1, x_7_p_0, x_7_p_1;
    const size_t pos_in_tile = position % kTileSize;
    // tile base can point to same tile as previous loop iteration, hence no
    // HWY_RESTRICT
    // KVs are unaligned and we only use unaligned loads in this implementation.
    const KV_T* tile_base =
        reinterpret_cast<const KV_T*>(kvs[current_kv_idx].RowBytes(
            (position - current_kv_start_offset) / kTileSize));

    const KV_T* v_tile =
        tile_base + qkv_dim * kTileSize + (pos_in_tile)*qkv_dim;
    const Q_T* q_group = q_T_in_groups_up_to_4[q_group_idx];
    const Q_T* q2_group = nullptr;
    if (kNumQueries > 4) {
      q2_group = q_T_in_groups_up_to_4[q_group_idx + 1];
    }
    if constexpr (IsF32<Q_T>()) {
      const KV_T* k_transposed_tile = tile_base + pos_in_tile;
      QDotKTilexUpTo8TransposedKDoubleWidth<kNumQueries>(
          df, q_group, q2_group, k_transposed_tile, qkv_dim, x_0_p_0, x_0_p_1,
          x_1_p_0, x_1_p_1, x_2_p_0, x_2_p_1, x_3_p_0, x_3_p_1, x_4_p_0,
          x_4_p_1, x_5_p_0, x_5_p_1, x_6_p_0, x_6_p_1, x_7_p_0, x_7_p_1);
    } else if constexpr (IsBF16<Q_T>()) {
      const KV_T* k_transposed_tile = tile_base + pos_in_tile * 2;
      QDotKTilexUpTo8TransposedKDoubleWidthBF16<kNumQueries>(
          df, q_group, q2_group, k_transposed_tile, qkv_dim, x_0_p_0, x_0_p_1,
          x_1_p_0, x_1_p_1, x_2_p_0, x_2_p_1, x_3_p_0, x_3_p_1, x_4_p_0,
          x_4_p_1, x_5_p_0, x_5_p_1, x_6_p_0, x_6_p_1, x_7_p_0, x_7_p_1);
    } else {
      static_assert(
          false,
          "Query type type not supported, only float and BF16 are supported");
    }

    constexpr int kFirstHalfAmountOfQueries = std::min(kNumQueries, 4);
    constexpr int kSecondHalfAmountOfQueries =
        kNumQueries - kFirstHalfAmountOfQueries;
    ApplySoftCap<kFirstHalfAmountOfQueries * 2>(
        df, att_cap, one_over_cap, x_0_p_0, x_0_p_1, x_1_p_0, x_1_p_1, x_2_p_0,
        x_2_p_1, x_3_p_0, x_3_p_1);
    if constexpr (kNumQueries > 4) {
      ApplySoftCap<kSecondHalfAmountOfQueries * 2>(
          df, att_cap, one_over_cap, x_4_p_0, x_4_p_1, x_5_p_0, x_5_p_1,
          x_6_p_0, x_6_p_1, x_7_p_0, x_7_p_1);
    }

    if (position < max_start_pos_per_group[loop_idx] ||
        position + step_size - 1 > min_last_pos_per_group[loop_idx]) {
      ApplyMasking<kNumQueries>(
          df, du, position,
          start_pos_per_query.data() + q_group_idx * kNumQueriesPerGroup,
          last_pos_per_query.data() + q_group_idx * kNumQueriesPerGroup,
          x_0_p_0, x_0_p_1, x_1_p_0, x_1_p_1, x_2_p_0, x_2_p_1, x_3_p_0,
          x_3_p_1, x_4_p_0, x_4_p_1, x_5_p_0, x_5_p_1, x_6_p_0, x_6_p_1,
          x_7_p_0, x_7_p_1);
    }
    HWY_ALIGN float scales[kNumQueriesPerLoop];
    // HWY_UNROLL(kNumQueriesPerLoop)
    for (size_t i = 0; i < kNumQueriesPerLoop; ++i) {
      scales[i] = 1.0f;
    }
    FlashAttentionTileStepAndApplySoftCap<kNumQueries>(
        df, 0.0f, 1.0f, x_0_p_0, x_0_p_1, x_1_p_0, x_1_p_1, x_2_p_0, x_2_p_1,
        x_3_p_0, x_3_p_1, x_4_p_0, x_4_p_1, x_5_p_0, x_5_p_1, x_6_p_0, x_6_p_1,
        x_7_p_0, x_7_p_1, max_logits, exp_denominator_sums, scales, q_group_idx,
        kNumQueriesPerGroup);
    if constexpr (IsF32<Q_T>()) {
      MulByConstAndAddTileUpTo8<kNumQueries>(
          df, scales, x_0_p_0, x_0_p_1, x_1_p_0, x_1_p_1, x_2_p_0, x_2_p_1,
          x_3_p_0, x_3_p_1, x_4_p_0, x_4_p_1, x_5_p_0, x_5_p_1, x_6_p_0,
          x_6_p_1, x_7_p_0, x_7_p_1, v_tile, att_out_per_query[loop_idx]);
    } else if constexpr (IsBF16<Q_T>()) {
      MulByConstAndAddTileUpTo8_BF16<kNumQueries>(
          df, scales, x_0_p_0, x_0_p_1, x_1_p_0, x_1_p_1, x_2_p_0, x_2_p_1,
          x_3_p_0, x_3_p_1, x_4_p_0, x_4_p_1, x_5_p_0, x_5_p_1, x_6_p_0,
          x_6_p_1, x_7_p_0, x_7_p_1, v_tile, att_out_per_query[loop_idx]);
    }
  };

  while (position <= largest_last_pos) {
    while (position - current_kv_start_offset >=
           kvs[current_kv_idx].Rows() * kTileSize) {
      current_kv_start_offset += kvs[current_kv_idx].Rows() * kTileSize;
      current_kv_idx++;
    }
    int group_idx = 0;
    for (; group_idx + kNumGroupsPerLoop <= full_groups_of_queries;
         group_idx += kNumGroupsPerLoop) {
      inner_loop.template operator()<kNumQueriesPerLoop>(group_idx);
    }
    if (group_idx < full_groups_of_queries) {
      inner_loop.template operator()<4>(group_idx);
      group_idx++;
    }
    switch (q_count % kNumQueriesPerGroup) {
      case 1:
        inner_loop.template operator()<1>(group_idx);
        break;
      case 2:
        inner_loop.template operator()<2>(group_idx);
        break;
      case 3:
        inner_loop.template operator()<3>(group_idx);
        break;
      default:
        break;
    }

    position += step_size;
  }
}

void DispatchTileFlashAttentionReturnExpSumsAndMaxLogits(
    hwy::Span<const MatPtr> kvs, int q_count,
    const hwy::Span<const float* HWY_RESTRICT> q_T_in_groups_up_to_4,
    hwy::Span<const size_t> start_pos_per_query,
    hwy::Span<const size_t> last_pos_per_query, const float att_cap,
    MatPtrT<float>& att_out, float* HWY_RESTRICT exp_denominator_sums,
    float* HWY_RESTRICT max_logits) {
  CallUpcastedKVs(kvs, [&](const auto& kv_t) {
    return TileFlashAttentionReturnExpSumsAndMaxLogits(
        kv_t, q_count, q_T_in_groups_up_to_4, start_pos_per_query,
        last_pos_per_query, att_cap, att_out, exp_denominator_sums, max_logits);
  });
}

void DispatchTileFlashAttentionReturnExpSumsAndMaxLogitsBF16(
    hwy::Span<const MatPtr> kvs, int q_count,
    const hwy::Span<const BF16 * HWY_RESTRICT> q_T_in_groups_up_to_4,
    hwy::Span<const size_t> start_pos_per_query,
    hwy::Span<const size_t> last_pos_per_query, const float att_cap,
    MatPtrT<float>& att_out, float* HWY_RESTRICT exp_denominator_sums,
    float* HWY_RESTRICT max_logits) {
  CallUpcastedKVs(kvs, [&](const auto& kv_t) {
    return TileFlashAttentionReturnExpSumsAndMaxLogits(
        kv_t, q_count, q_T_in_groups_up_to_4, start_pos_per_query,
        last_pos_per_query, att_cap, att_out, exp_denominator_sums, max_logits);
  });
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
                    ThreadingContext& ctx, AttentionImpl attention_impl) {
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
