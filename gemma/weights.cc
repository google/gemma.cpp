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

#include "gemma/weights.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <mutex>  // NOLINT
#include <string>
#include <vector>

#include "compression/compress.h"
#include "compression/types.h"
#include "gemma/configs.h"
#include "gemma/gemma_args.h"
#include "gemma/model_store.h"
#include "gemma/weights_internal.h"
#include "hwy/base.h"
#include "hwy/highway.h"
#include "hwy/profiler.h"
#include "io/blob_store.h"
#include "util/mat.h"
#include "util/threading_context.h"
#include "util/zones.h"

// TODO: move into foreach_target
#include "compression/compress-inl.h"

namespace gcpp {

using weights_internal::TensorToRead;

static std::mutex g_mat_owners_mutex;

// Copies att_weights from `attn_vec_einsum_w`.
void LayerWeightsPtrs::InitAttWeights(std::vector<MatOwner>& mat_owners,
                                      const Allocator& allocator) {
  // We only use this tensor for Gemma layers.
  if (layer_config.type != LayerAttentionType::kGemma) return;

  // Files must have one or the other.
  HWY_ASSERT(attn_vec_einsum_w.HasPtr() ^ att_weights.HasPtr());
  // Done if we already read the transposed tensor.
  if (att_weights.HasPtr() && !attn_vec_einsum_w.HasPtr()) return;

  // NUQ is handled by a specialization in weights.cc.
  HWY_ASSERT(attn_vec_einsum_w.GetType() != Type::kNUQ);

  const size_t model_dim = layer_config.model_dim;
  const size_t heads = layer_config.heads;
  const size_t qkv_dim = layer_config.qkv_dim;

  // Reshape [heads, model_dim, qkv_dim] to [model_dim, heads * qkv_dim].
  att_weights.SetType(attn_vec_einsum_w.GetType());
  HWY_ASSERT(att_weights.Rows() == model_dim);
  HWY_ASSERT(att_weights.Cols() == heads * qkv_dim);
  HWY_ASSERT(attn_vec_einsum_w.Rows() == heads * model_dim);
  HWY_ASSERT(attn_vec_einsum_w.Cols() == qkv_dim);

  const MatPadding padding = DefaultPadding(att_weights.GetType());
  {
    std::lock_guard<std::mutex> lock(g_mat_owners_mutex);
    mat_owners.push_back(MatOwner());
    mat_owners.back().AllocateFor(att_weights, allocator, padding);
  }

  if (IsPacked(att_weights.GetType())) {
    const size_t cols = heads * qkv_dim;
    const size_t src_row_bytes = PackedEnd(att_weights.GetType(), qkv_dim);
    const size_t dst_row_bytes = PackedEnd(att_weights.GetType(), cols);
    HWY_ASSERT(dst_row_bytes == heads * src_row_bytes);

    uint8_t* dst_ptr = att_weights.RowBytes(0);
    const uint8_t* src_ptr = attn_vec_einsum_w.RowBytes(0);

    for (size_t m = 0; m < model_dim; ++m) {
      uint8_t* dst_row = dst_ptr + m * dst_row_bytes;
      for (size_t h = 0; h < heads; ++h) {
        size_t src_row_idx = h * model_dim + m;
        const uint8_t* src_row = src_ptr + src_row_idx * src_row_bytes;
        hwy::CopyBytes(src_row, dst_row + h * src_row_bytes, src_row_bytes);
      }
    }
  } else {
    const size_t T_bytes = att_weights.ElementBytes();
    for (size_t m = 0; m < model_dim; ++m) {
      uint8_t* HWY_RESTRICT out_row = att_weights.RowBytes(m);
      for (size_t h = 0; h < heads; ++h) {
        hwy::CopyBytes(attn_vec_einsum_w.RowBytes(h * model_dim + m),
                       out_row + h * qkv_dim * T_bytes, qkv_dim * T_bytes);
      }
    }
  }
  att_weights.SetScale(attn_vec_einsum_w.Scale());
}

static void SplitPackedMatrix(MatPtr& parent, size_t split_row, MatPtr& w1,
                              MatPtr& w2) {
  const size_t stride = parent.Stride();
  uint8_t* base_ptr = parent.RowBytes(0);
  w1.SetPtr(base_ptr, stride);

  if (IsPacked(parent.GetType())) {
    const size_t split_bytes = PackedEnd(parent.GetType(), split_row * stride);
    w2.SetPtr(base_ptr + split_bytes, stride);
    return;
  }
  w2.SetPtr(parent.RowBytes(split_row), stride);
}

// For FFN. Fast, only updates pointers.
void LayerWeightsPtrs::SplitW1() {
  // Used for Gemma layers; FFWVit uses different tensors.
  if (layer_config.type == LayerAttentionType::kVit) return;
  if (layer_config.IsMoE()) return;

  // Files have both or neither of w1 and w2.
  HWY_ASSERT(gating_einsum_w1.HasPtr() == gating_einsum_w2.HasPtr());
  // w is mutually exclusive with w1 and w2 in the file.
  HWY_ASSERT(gating_einsum_w.HasPtr() ^ gating_einsum_w1.HasPtr());
  // Done if we already read split tensors. Note that they are not
  // necessarily the same type.
  if (gating_einsum_w1.HasPtr() && !gating_einsum_w.HasPtr()) return;

  const size_t ff_hidden_dim = layer_config.ff_hidden_dim;
  HWY_ASSERT(gating_einsum_w.Rows() == 2 * ff_hidden_dim);
  HWY_ASSERT(gating_einsum_w1.Rows() == ff_hidden_dim);
  HWY_ASSERT(gating_einsum_w2.Rows() == ff_hidden_dim);
  // Cols are the model_dim but we don't have ModelConfig here.
  HWY_ASSERT(gating_einsum_w1.Cols() == gating_einsum_w.Cols());
  HWY_ASSERT(gating_einsum_w2.Cols() == gating_einsum_w.Cols());

  SplitPackedMatrix(gating_einsum_w, ff_hidden_dim, gating_einsum_w1,
                    gating_einsum_w2);
  gating_einsum_w1.SetType(gating_einsum_w.GetType());
  gating_einsum_w2.SetType(gating_einsum_w.GetType());
  gating_einsum_w1.SetScale(gating_einsum_w.Scale());
  gating_einsum_w2.SetScale(gating_einsum_w.Scale());
  gating_einsum_w.SetPtr(nullptr, gating_einsum_w.Cols());
}

// For attention, which might not have a w2. Fast, only updates pointers.
void LayerWeightsPtrs::SplitAttW1() {
  // We only use this tensor for Gemma layers.
  if (layer_config.type != LayerAttentionType::kGemma) return;

  // w is mutually exclusive with w1 in the file.
  HWY_ASSERT(qkv_einsum_w.HasPtr() ^ qkv_einsum_w1.HasPtr());
  // Done if we already read split tensors. Note that w2 does not exist for
  // MHA, and otherwise might not be the same type.
  if (qkv_einsum_w1.HasPtr() && !qkv_einsum_w.HasPtr()) return;

  // If w2 was loaded from a separate tensor (e.g. attn/k_einsum/w for
  // single-kv-head models), only w1 needs to be split from the combined
  // qkv_einsum_w. w2 already has its data.
  if (qkv_einsum_w2.HasPtr()) {
    const size_t w1_rows = layer_config.heads * layer_config.qkv_dim;
    HWY_ASSERT(qkv_einsum_w.Rows() == w1_rows + qkv_einsum_w2.Rows());
    HWY_ASSERT(qkv_einsum_w1.Rows() == w1_rows);
    HWY_ASSERT(qkv_einsum_w1.Cols() == qkv_einsum_w.Cols());

    const size_t stride = qkv_einsum_w.Stride();
    qkv_einsum_w1.SetPtr(qkv_einsum_w.RowBytes(0), stride);
    qkv_einsum_w1.SetType(qkv_einsum_w.GetType());
    qkv_einsum_w1.SetScale(qkv_einsum_w.Scale());
    qkv_einsum_w.SetPtr(nullptr, qkv_einsum_w.Cols());
    return;
  }

  const size_t w1_rows = layer_config.heads * layer_config.qkv_dim;
  const size_t w2_rows = layer_config.kv_heads * 2 * layer_config.qkv_dim;
  HWY_ASSERT(qkv_einsum_w.Rows() == w1_rows + w2_rows);
  HWY_ASSERT(qkv_einsum_w1.Rows() == w1_rows);
  HWY_ASSERT(qkv_einsum_w2.Rows() == w2_rows);
  // Cols are the model_dim but we don't have ModelConfig here.
  HWY_ASSERT(qkv_einsum_w1.Cols() == qkv_einsum_w.Cols());
  HWY_ASSERT(qkv_einsum_w2.Cols() == qkv_einsum_w.Cols());

  SplitPackedMatrix(qkv_einsum_w, w1_rows, qkv_einsum_w1, qkv_einsum_w2);
  qkv_einsum_w1.SetType(qkv_einsum_w.GetType());
  qkv_einsum_w2.SetType(qkv_einsum_w.GetType());
  qkv_einsum_w1.SetScale(qkv_einsum_w.Scale());
  qkv_einsum_w2.SetScale(qkv_einsum_w.Scale());
  qkv_einsum_w.SetPtr(nullptr, qkv_einsum_w.Cols());
}

static void InitAttWeightsGeneric(const LayerConfig& layer_config,
                                  MatPtr& attn_vec_einsum_w,
                                  MatPtr& att_weights,
                                  std::vector<MatOwner>& mat_owners,
                                  const Allocator& allocator) {
  HWY_ASSERT(attn_vec_einsum_w.HasPtr() ^ att_weights.HasPtr());
  if (att_weights.HasPtr() && !attn_vec_einsum_w.HasPtr()) return;
  HWY_ASSERT(attn_vec_einsum_w.GetType() != Type::kNUQ);

  const size_t model_dim = layer_config.model_dim;
  const size_t heads = layer_config.heads;
  const size_t qkv_dim = layer_config.qkv_dim;

  att_weights.SetType(attn_vec_einsum_w.GetType());
  HWY_ASSERT(att_weights.Rows() == model_dim);
  HWY_ASSERT(att_weights.Cols() == heads * qkv_dim);
  HWY_ASSERT(attn_vec_einsum_w.Rows() == heads * model_dim);
  HWY_ASSERT(attn_vec_einsum_w.Cols() == qkv_dim);

  {
    std::lock_guard<std::mutex> lock(g_mat_owners_mutex);
    mat_owners.push_back(MatOwner());
    mat_owners.back().AllocateFor(att_weights, allocator, MatPadding::kOdd);
  }

  const size_t T_bytes = att_weights.ElementBytes();
  for (size_t m = 0; m < model_dim; ++m) {
    uint8_t* HWY_RESTRICT out_row = att_weights.RowBytes(m);
    for (size_t h = 0; h < heads; ++h) {
      hwy::CopyBytes(attn_vec_einsum_w.RowBytes(h * model_dim + m),
                     out_row + h * qkv_dim * T_bytes, qkv_dim * T_bytes);
    }
  }
  att_weights.SetScale(attn_vec_einsum_w.Scale());
}

static void SplitGateGeneric(const LayerConfig& layer_config,
                             MatPtr& gating_einsum_w, MatPtr& gating_einsum_w1,
                             MatPtr& gating_einsum_w2) {
  HWY_ASSERT(gating_einsum_w1.HasPtr() == gating_einsum_w2.HasPtr());
  HWY_ASSERT(gating_einsum_w.HasPtr() ^ gating_einsum_w1.HasPtr());
  if (gating_einsum_w1.HasPtr() && !gating_einsum_w.HasPtr()) return;

  const size_t ff_hidden_dim = layer_config.ff_hidden_dim;
  HWY_ASSERT(gating_einsum_w.Rows() == 2 * ff_hidden_dim);
  HWY_ASSERT(gating_einsum_w1.Rows() == ff_hidden_dim);
  HWY_ASSERT(gating_einsum_w2.Rows() == ff_hidden_dim);
  HWY_ASSERT(gating_einsum_w1.Cols() == gating_einsum_w.Cols());
  HWY_ASSERT(gating_einsum_w2.Cols() == gating_einsum_w.Cols());

  const size_t stride = gating_einsum_w.Stride();
  gating_einsum_w1.SetPtr(gating_einsum_w.RowBytes(0), stride);
  gating_einsum_w2.SetPtr(gating_einsum_w.RowBytes(ff_hidden_dim), stride);
  gating_einsum_w1.SetType(gating_einsum_w.GetType());
  gating_einsum_w2.SetType(gating_einsum_w.GetType());
  gating_einsum_w1.SetScale(gating_einsum_w.Scale());
  gating_einsum_w2.SetScale(gating_einsum_w.Scale());
  gating_einsum_w.SetPtr(nullptr, gating_einsum_w.Cols());
}

static void SplitQKVGeneric(const LayerConfig& layer_config,
                            MatPtr& qkv_einsum_w, MatPtr& qkv_einsum_w1,
                            MatPtr& qkv_einsum_w2) {
  HWY_ASSERT(qkv_einsum_w.HasPtr() ^ qkv_einsum_w1.HasPtr());
  if (qkv_einsum_w1.HasPtr() && !qkv_einsum_w.HasPtr()) return;

  const size_t w1_rows = layer_config.heads * layer_config.qkv_dim;
  const size_t w2_rows = layer_config.kv_heads * 2 * layer_config.qkv_dim;
  HWY_ASSERT(qkv_einsum_w.Rows() == w1_rows + w2_rows);
  HWY_ASSERT(qkv_einsum_w1.Rows() == w1_rows);
  HWY_ASSERT(qkv_einsum_w2.Rows() == w2_rows);
  HWY_ASSERT(qkv_einsum_w1.Cols() == qkv_einsum_w.Cols());
  HWY_ASSERT(qkv_einsum_w2.Cols() == qkv_einsum_w.Cols());

  const size_t stride = qkv_einsum_w.Stride();
  qkv_einsum_w1.SetPtr(qkv_einsum_w.RowBytes(0), stride);
  qkv_einsum_w2.SetPtr(qkv_einsum_w.RowBytes(w1_rows), stride);
  qkv_einsum_w1.SetType(qkv_einsum_w.GetType());
  qkv_einsum_w2.SetType(qkv_einsum_w.GetType());
  qkv_einsum_w1.SetScale(qkv_einsum_w.Scale());
  qkv_einsum_w2.SetScale(qkv_einsum_w.Scale());
  qkv_einsum_w.SetPtr(nullptr, qkv_einsum_w.Cols());
}

static void HWY_MAYBE_UNUSED InitAttWeightsI8(
    const LayerConfig& layer_config, MatPtrT<I8Stream>& attn_vec_einsum_w,
    MatPtrT<I8Stream>& att_weights, std::vector<MatOwner>& mat_owners,
    ThreadingContext& ctx) {
  if (!attn_vec_einsum_w.HasPtr()) return;
  HWY_ASSERT(attn_vec_einsum_w.GetType() == Type::kI8);

  att_weights.SetType(Type::kI8);

  {
    std::lock_guard<std::mutex> lock(g_mat_owners_mutex);
    mat_owners.emplace_back();
    mat_owners.back().AllocateFor(att_weights, ctx.allocator,
                                  MatPadding::kPacked);
  }

  const size_t model_dim = layer_config.model_dim;
  const size_t heads = layer_config.heads;
  const size_t qkv_dim = layer_config.qkv_dim;

  // Reshape [kHeads, kModelDim, kQKVDim] to [kModelDim, kHeads * kQKVDim].
  hwy::AlignedFreeUniquePtr<float[]> attn_vec_einsum_w_tmp =
      hwy::AllocateAligned<float>(model_dim * heads * qkv_dim);
  hwy::AlignedFreeUniquePtr<float[]> att_weights_tmp =
      hwy::AllocateAligned<float>(model_dim * heads * qkv_dim);

  const hwy::HWY_NAMESPACE::ScalableTag<float> df;
  HWY_NAMESPACE::DecompressAndZeroPad(df, attn_vec_einsum_w.Span(), 0,
                                      attn_vec_einsum_w_tmp.get(),
                                      model_dim * heads * qkv_dim);

  for (size_t m = 0; m < model_dim; ++m) {
    float* HWY_RESTRICT out_row = att_weights_tmp.get() + m * heads * qkv_dim;
    for (size_t h = 0; h < heads; ++h) {
      hwy::CopyBytes(
          attn_vec_einsum_w_tmp.get() + h * model_dim * qkv_dim + m * qkv_dim,
          out_row + h * qkv_dim, qkv_dim * sizeof(float));
    }
  }

  CompressWorkingSet work;
  HWY_NAMESPACE::Compress(att_weights_tmp.get(), model_dim * heads * qkv_dim,
                          work, att_weights.Span(),
                          /*packed_ofs=*/0, ctx);

  att_weights.SetScale(attn_vec_einsum_w.Scale());
}

static void HWY_MAYBE_UNUSED SplitW1I8(const LayerConfig& layer_config,
                                       MatPtrT<I8Stream>& gating_einsum_w,
                                       MatPtrT<I8Stream>& gating_einsum_w1,
                                       MatPtrT<I8Stream>& gating_einsum_w2,
                                       std::vector<MatOwner>& mat_owners,
                                       ThreadingContext& ctx) {
  // Files have both or neither of w1 and w2.
  HWY_ASSERT(gating_einsum_w1.HasPtr() == gating_einsum_w2.HasPtr());
  // w is mutually exclusive with w1 and w2 in the file.
  HWY_ASSERT(gating_einsum_w.HasPtr() ^ gating_einsum_w1.HasPtr());
  // Done if we already read split tensors.
  if (gating_einsum_w1.HasPtr() && !gating_einsum_w.HasPtr()) return;
  // Nothing to do if w is not present.
  if (!gating_einsum_w.HasPtr()) return;

  HWY_ASSERT(gating_einsum_w.GetType() == Type::kI8);

  const size_t ff_hidden_dim = layer_config.ff_hidden_dim;
  const size_t model_dim = gating_einsum_w.Cols();
  HWY_ASSERT(gating_einsum_w.Rows() == 2 * ff_hidden_dim);
  HWY_ASSERT(gating_einsum_w1.Rows() == ff_hidden_dim);
  HWY_ASSERT(gating_einsum_w2.Rows() == ff_hidden_dim);
  HWY_ASSERT(gating_einsum_w1.Cols() == model_dim);
  HWY_ASSERT(gating_einsum_w2.Cols() == model_dim);

  gating_einsum_w1.SetType(Type::kI8);
  gating_einsum_w2.SetType(Type::kI8);

  {
    std::lock_guard<std::mutex> lock(g_mat_owners_mutex);
    mat_owners.emplace_back();
    mat_owners.back().AllocateFor(gating_einsum_w1, ctx.allocator,
                                  MatPadding::kPacked);
    mat_owners.emplace_back();
    mat_owners.back().AllocateFor(gating_einsum_w2, ctx.allocator,
                                  MatPadding::kPacked);
  }

  const size_t total_size = gating_einsum_w.Rows() * gating_einsum_w.Cols();
  hwy::AlignedFreeUniquePtr<float[]> w_tmp =
      hwy::AllocateAligned<float>(total_size);

  const hwy::HWY_NAMESPACE::ScalableTag<float> df;
  HWY_NAMESPACE::DecompressAndZeroPad(df, gating_einsum_w.Span(), 0,
                                      w_tmp.get(), total_size);

  const size_t split_size = ff_hidden_dim * model_dim;
  float* w1_tmp = w_tmp.get();
  float* w2_tmp = w_tmp.get() + split_size;

  CompressWorkingSet work;
  HWY_NAMESPACE::Compress(w1_tmp, split_size, work, gating_einsum_w1.Span(), 0,
                          ctx);
  HWY_NAMESPACE::Compress(w2_tmp, split_size, work, gating_einsum_w2.Span(), 0,
                          ctx);

  gating_einsum_w1.SetScale(1.0f);
  gating_einsum_w2.SetScale(1.0f);

  gating_einsum_w.SetPtr(nullptr, gating_einsum_w.Cols());
}

static void HWY_MAYBE_UNUSED SplitAttW1I8(const LayerConfig& layer_config,
                                          MatPtrT<I8Stream>& qkv_einsum_w,
                                          MatPtrT<I8Stream>& qkv_einsum_w1,
                                          MatPtrT<I8Stream>& qkv_einsum_w2,
                                          std::vector<MatOwner>& mat_owners,
                                          ThreadingContext& ctx) {
  // w is mutually exclusive with w1 in the file.
  HWY_ASSERT(qkv_einsum_w.HasPtr() ^ qkv_einsum_w1.HasPtr());
  // Done if we already read split tensors.
  if (qkv_einsum_w1.HasPtr() && !qkv_einsum_w.HasPtr()) return;
  // Nothing to do if w is not present.
  if (!qkv_einsum_w.HasPtr()) return;

  HWY_ASSERT(qkv_einsum_w.GetType() == Type::kI8);

  // If w2 was loaded from a separate tensor (e.g. attn/k_einsum/w for
  // single-kv-head models), only w1 needs to be split from the combined
  // qkv_einsum_w. w2 already has its data.
  if (qkv_einsum_w2.HasPtr()) {
    const size_t model_dim = qkv_einsum_w.Cols();
    const size_t w1_rows = layer_config.heads * layer_config.qkv_dim;
    HWY_ASSERT(qkv_einsum_w.Rows() == w1_rows + qkv_einsum_w2.Rows());
    HWY_ASSERT(qkv_einsum_w1.Rows() == w1_rows);
    HWY_ASSERT(qkv_einsum_w1.Cols() == model_dim);

    qkv_einsum_w1.SetType(Type::kI8);
    {
      std::lock_guard<std::mutex> lock(g_mat_owners_mutex);
      mat_owners.emplace_back();
      mat_owners.back().AllocateFor(qkv_einsum_w1, ctx.allocator,
                                    MatPadding::kPacked);
    }

    const size_t w1_size = w1_rows * model_dim;
    hwy::AlignedFreeUniquePtr<float[]> w_tmp =
        hwy::AllocateAligned<float>(w1_size);

    const hwy::HWY_NAMESPACE::ScalableTag<float> df;
    HWY_NAMESPACE::DecompressAndZeroPad(df, qkv_einsum_w.Span(), 0, w_tmp.get(),
                                        w1_size);

    CompressWorkingSet work;
    HWY_NAMESPACE::Compress(w_tmp.get(), w1_size, work, qkv_einsum_w1.Span(), 0,
                            ctx);

    qkv_einsum_w1.SetScale(1.0f);
    qkv_einsum_w.SetPtr(nullptr, qkv_einsum_w.Cols());
    return;
  }

  const size_t model_dim = qkv_einsum_w.Cols();
  const size_t w1_rows = layer_config.heads * layer_config.qkv_dim;
  const size_t w2_rows = layer_config.kv_heads * 2 * layer_config.qkv_dim;
  HWY_ASSERT(qkv_einsum_w.Rows() == w1_rows + w2_rows);
  HWY_ASSERT(qkv_einsum_w1.Rows() == w1_rows);
  HWY_ASSERT(qkv_einsum_w2.Rows() == w2_rows);
  HWY_ASSERT(qkv_einsum_w1.Cols() == model_dim);
  HWY_ASSERT(qkv_einsum_w2.Cols() == model_dim);

  qkv_einsum_w1.SetType(Type::kI8);
  qkv_einsum_w2.SetType(Type::kI8);

  {
    std::lock_guard<std::mutex> lock(g_mat_owners_mutex);
    mat_owners.emplace_back();
    mat_owners.back().AllocateFor(qkv_einsum_w1, ctx.allocator,
                                  MatPadding::kPacked);
    mat_owners.emplace_back();
    mat_owners.back().AllocateFor(qkv_einsum_w2, ctx.allocator,
                                  MatPadding::kPacked);
  }

  const size_t total_size = qkv_einsum_w.Rows() * qkv_einsum_w.Cols();
  hwy::AlignedFreeUniquePtr<float[]> w_tmp =
      hwy::AllocateAligned<float>(total_size);

  const hwy::HWY_NAMESPACE::ScalableTag<float> df;
  HWY_NAMESPACE::DecompressAndZeroPad(df, qkv_einsum_w.Span(), 0, w_tmp.get(),
                                      total_size);

  const size_t w1_size = w1_rows * model_dim;
  const size_t w2_size = w2_rows * model_dim;
  float* w1_tmp = w_tmp.get();
  float* w2_tmp = w_tmp.get() + w1_size;

  CompressWorkingSet work;
  HWY_NAMESPACE::Compress(w1_tmp, w1_size, work, qkv_einsum_w1.Span(), 0, ctx);
  HWY_NAMESPACE::Compress(w2_tmp, w2_size, work, qkv_einsum_w2.Span(), 0, ctx);

  qkv_einsum_w1.SetScale(1.0f);
  qkv_einsum_w2.SetScale(1.0f);

  qkv_einsum_w.SetPtr(nullptr, qkv_einsum_w.Cols());
}

// Must be called after reading weights via `ForEachTensor`.
// TODO: exporters should bake this into the weights already.
// WARNING: called from multiple threads; `mat_owners` requires a lock.
void LayerWeightsPtrs::Fixup(Model model, std::vector<MatOwner>& mat_owners,
                             ThreadingContext& ctx) {
  if (attn_vec_einsum_w.GetType() == Type::kI8) {
    MatPtrT<I8Stream> attn_vec_einsum_w_i8(attn_vec_einsum_w);
    MatPtrT<I8Stream> att_weights_i8(att_weights);
    InitAttWeightsI8(layer_config, attn_vec_einsum_w_i8, att_weights_i8,
                     mat_owners, ctx);
    attn_vec_einsum_w = attn_vec_einsum_w_i8;
    att_weights = att_weights_i8;
  } else {
    InitAttWeights(mat_owners, ctx.allocator);
  }

  if (gating_einsum_w.GetType() == Type::kI8) {
    MatPtrT<I8Stream> gating_einsum_w_i8(gating_einsum_w);
    MatPtrT<I8Stream> gating_einsum_w1_i8(gating_einsum_w1);
    MatPtrT<I8Stream> gating_einsum_w2_i8(gating_einsum_w2);
    SplitW1I8(layer_config, gating_einsum_w_i8, gating_einsum_w1_i8,
              gating_einsum_w2_i8, mat_owners, ctx);
    gating_einsum_w = gating_einsum_w_i8;
    gating_einsum_w1 = gating_einsum_w1_i8;
    gating_einsum_w2 = gating_einsum_w2_i8;
  } else {
    SplitW1();
  }

  if (qkv_einsum_w.GetType() == Type::kI8) {
    MatPtrT<I8Stream> qkv_einsum_w_i8(qkv_einsum_w);
    MatPtrT<I8Stream> qkv_einsum_w1_i8(qkv_einsum_w1);
    MatPtrT<I8Stream> qkv_einsum_w2_i8(qkv_einsum_w2);
    SplitAttW1I8(layer_config, qkv_einsum_w_i8, qkv_einsum_w1_i8,
                 qkv_einsum_w2_i8, mat_owners, ctx);
    qkv_einsum_w = qkv_einsum_w_i8;
    qkv_einsum_w1 = qkv_einsum_w1_i8;
    qkv_einsum_w2 = qkv_einsum_w2_i8;
  } else {
    SplitAttW1();
    // Interleave K/V heads in qkv_einsum_w2: the exporter writes
    // [K0..Kn, V0..Vn] but the runtime expects [K0, V0, K1, V1, ...].
    // TODO(philculliton): either (1) fix the exporter to emit interleaved
    // layout directly, or (2) replace this model check with a general
    // structural condition (e.g. !IsMHA() && kv_heads > 1) once we
    // verify it doesn't regress other multi-kv-head models.
    // This applies to Gemma 4 global layers; the model check will be expanded.
    if (model == Model::GEMMA4_26B_MOE && layer_config.kv_heads == 2 &&
        layer_config.qkv_dim == 512) {
      const size_t old_stride = qkv_einsum_w2.Stride();
      const size_t elem_bytes = qkv_einsum_w2.ElementBytes();
      const size_t old_row_bytes = old_stride * elem_bytes;
      const size_t kv_heads = layer_config.kv_heads;
      const size_t total_bytes = qkv_einsum_w2.Rows() * old_row_bytes;
      hwy::AlignedFreeUniquePtr<uint8_t[]> tmp =
          hwy::AllocateAligned<uint8_t>(total_bytes);
      hwy::CopyBytes(qkv_einsum_w2.RowBytes(0), tmp.get(), total_bytes);

      {
        std::lock_guard<std::mutex> lock(g_mat_owners_mutex);
        mat_owners.emplace_back();
        mat_owners.back().AllocateFor(qkv_einsum_w2, ctx.allocator,
                                      MatPadding::kPacked);
      }

      const size_t new_row_bytes = qkv_einsum_w2.Cols() * elem_bytes;
      const size_t qkv_dim = layer_config.qkv_dim;
      const uint8_t* src_ptr = tmp.get();
      for (size_t i = 0; i < kv_heads; ++i) {
        for (size_t row = 0; row < qkv_dim; ++row) {
          hwy::CopyBytes(src_ptr + (i * qkv_dim + row) * old_row_bytes,
                         qkv_einsum_w2.RowBytes((2 * i) * qkv_dim + row),
                         new_row_bytes);
          hwy::CopyBytes(
              src_ptr + ((kv_heads + i) * qkv_dim + row) * old_row_bytes,
              qkv_einsum_w2.RowBytes((2 * i + 1) * qkv_dim + row),
              new_row_bytes);
        }
      }
    }
  }
}

void T5GemmaEncoderLayerWeightsPtrs::Fixup(std::vector<MatOwner>& mat_owners,
                                           ThreadingContext& ctx) {
  InitAttWeightsGeneric(layer_config, attn_vec_einsum_w, att_weights,
                        mat_owners, ctx.allocator);
  SplitGateGeneric(layer_config, gating_einsum_w, gating_einsum_w1,
                   gating_einsum_w2);
  SplitQKVGeneric(layer_config, qkv_einsum_w, qkv_einsum_w1, qkv_einsum_w2);
}

void T5GemmaDecoderLayerWeightsPtrs::Fixup(std::vector<MatOwner>& mat_owners,
                                           ThreadingContext& ctx) {
  InitAttWeightsGeneric(layer_config, self_attn_vec_einsum_w, self_att_weights,
                        mat_owners, ctx.allocator);
  InitAttWeightsGeneric(layer_config, cross_attn_vec_einsum_w,
                        cross_att_weights, mat_owners, ctx.allocator);
  SplitGateGeneric(layer_config, gating_einsum_w, gating_einsum_w1,
                   gating_einsum_w2);
  SplitQKVGeneric(layer_config, self_qkv_einsum_w, self_qkv_einsum_w1,
                  self_qkv_einsum_w2);
}

// Zero-initializes only the allocated tensors in `*this`.
void WeightsPtrs::ZeroInit() {
  ForEachTensor(nullptr, nullptr, [](const TensorArgs& t) {
    if (!t.mat.HasPtr() || t.mat.GetType() == Type::kUnknown) return;
    gcpp::ZeroInit(t.mat);
  });
}

// Copies only the allocated tensors in `*this` from tensors in `other`.
void WeightsPtrs::CopyFrom(const WeightsPtrs& other) {
  ForEachTensor(const_cast<WeightsPtrs*>(&other), nullptr,
                [](const TensorArgs& t) {
                  if (!t.mat.HasPtr()) return;
                  HWY_ASSERT(t.other_mat1 && t.other_mat1->HasPtr());
                  CopyMat(*t.other_mat1, t.mat);
                });
}

// For reshaping file tensors to the shape expected by the code. This would
// ideally already happen in the importer. Called by `ReadFromBlobs`.
void WeightsPtrs::Fixup(std::vector<MatOwner>& mat_owners,
                        ThreadingContext& ctx) {
  const size_t cluster_idx = 0;
  if (config_.is_encoder_decoder) {
    ParallelFor(Parallelism::kFlat, t5gemma_encoder_layers.size(), ctx,
                cluster_idx, Callers::kFixupWeights,
                [&](uint64_t layer, size_t /*worker*/) {
                  t5gemma_encoder_layers[layer].Fixup(mat_owners, ctx);
                });

    ParallelFor(Parallelism::kFlat, t5gemma_decoder_layers.size(), ctx,
                cluster_idx, Callers::kFixupWeights,
                [&](uint64_t layer, size_t /*worker*/) {
                  t5gemma_decoder_layers[layer].Fixup(mat_owners, ctx);
                });
    return;
  }

  ParallelFor(Parallelism::kFlat, c_layers.size(), ctx, cluster_idx,
              Callers::kFixupWeights, [&](uint64_t layer, size_t /*worker*/) {
                GetLayer(layer)->Fixup(config_.model, mat_owners, ctx);
              });

  ParallelFor(Parallelism::kFlat, vit_layers.size(), ctx, cluster_idx,
              Callers::kFixupWeights, [&](uint64_t layer, size_t /*worker*/) {
                VitLayer(layer)->Fixup(config_.model, mat_owners, ctx);
              });

  for (LayerWeightsPtrs& mtp : mtp_layers) {
    mtp.Fixup(config_.model, mat_owners, ctx);
  }
}

std::vector<uint32_t> WeightsPtrs::AddTensorDataToWriter(
    BlobWriter& writer) const {
  std::vector<uint32_t> serialized_mat_ptrs;
  // ForEachTensor is non-const but the lambda does not modify *this.
  const_cast<WeightsPtrs*>(this)->ForEachTensor(
      nullptr, nullptr, [&](const TensorArgs& t) {
        if (t.flags & TensorArgs::kMaybeRead && !t.mat.HasPtr()) return;
        HWY_ASSERT_M(t.mat.HasPtr(), t.mat.Name());
        writer.Add(t.mat.Name(), t.mat.Packed(), t.mat.PackedBytes());
        t.mat.AppendTo(serialized_mat_ptrs);
      });
  return serialized_mat_ptrs;
}

// Decides whether to read or map based on heuristics and user override.
WeightsPtrs::Mode weights_internal::ChooseMode(uint64_t file_bytes,
                                               const LoaderArgs& loader,
                                               const InferenceArgs& inference,
                                               const Allocator& allocator) {
  Tristate to_bf16 = loader.to_bf16;
  Tristate map = loader.map;

  // An explicit request to convert the embedding requires owned memory. Do
  // not let the automatic mapping heuristic override that request. An
  // explicit --map=1 is still honored and diagnosed in ReadFromBlobs.
  if (loader.sfp_embedding == Tristate::kTrue && map == Tristate::kDefault) {
    map = Tristate::kFalse;
  }

  // Disable mapping if not padded to the base page size.
  if (file_bytes % allocator.BasePageBytes() != 0) {
    if (map == Tristate::kTrue) {  // Only complain if explicitly requested.
      HWY_WARN("Unable to map non-padded file (%zu, %zu), reading instead.",
               static_cast<size_t>(file_bytes >> 10),
               allocator.BasePageBytes());
    }
    map = Tristate::kFalse;
  }

  // Check for user override:
  if (to_bf16 == Tristate::kTrue && map == Tristate::kTrue) {
    HWY_WARN("Cannot have to_bf16 && map, to_bf16 takes precedence.");
  }
  if (to_bf16 == Tristate::kTrue) return WeightsPtrs::Mode::kReadBF16;
  if (map == Tristate::kTrue) return WeightsPtrs::Mode::kMap;

  if (to_bf16 == Tristate::kDefault) {
    // Heuristic: sub-bf16 compression is not helpful if compute-bound.
    to_bf16 = (inference.decode_qbatch_size >= 128) ? Tristate::kTrue
                                                    : Tristate::kFalse;
  }

  if (map == Tristate::kDefault) {
    // Heuristic: map if large fraction of total. Do not decide based on
    // `FreeMiB` because it  is generally low.
    const size_t file_mib = file_bytes >> 20;
    const size_t total_mib = allocator.TotalMiB();
    if (file_mib > total_mib) {
      HWY_WARN("Weight file %zu MiB > detected memory %zu MiB.",
               static_cast<size_t>(file_mib), total_mib);
    }
    // Large fraction of total.
    map = (file_mib >= total_mib / 3) ? Tristate::kTrue : Tristate::kFalse;
  }

  // If the `map` heuristic triggers, use that for safety.
  if (map == Tristate::kTrue) return WeightsPtrs::Mode::kMap;
  return (to_bf16 == Tristate::kTrue) ? WeightsPtrs::Mode::kReadBF16
                                      : WeightsPtrs::Mode::kRead;
}

// Allocates multiple in parallel and binds to NUMA nodes.
static void AllocateAndBindAll(std::vector<TensorToRead>& tensors,
                               const WeightsPtrs::Mode mode,
                               std::vector<MatOwner>& owners,
                               ThreadingContext& ctx) {
  const size_t start = owners.size();
  owners.resize(start + tensors.size());

  // Allocate in parallel because faulting in large tensors is slow.
  ParallelFor(
      Parallelism::kFlat, tensors.size(), ctx, /*cluster_idx=*/0,
      Callers::kAllocateAndBindAll, [&](uint64_t task, size_t /*thread*/) {
        TensorToRead& tensor = tensors[task];
        MatPtr& mat = *tensor.mat;

        tensor.prev_type = mat.GetType();
        tensor.prev_packed_bytes = mat.PackedBytes();
        // Only worthwhile from 16/32-bit types; the others are already <= 8
        // bits, and NUQ is smaller than SFP.
        if (tensor.to_sfp && tensor.prev_type != Type::kF32 &&
            tensor.prev_type != Type::kBF16) {
          tensor.to_sfp = false;
        }
        if (tensor.to_sfp) {
          mat.SetType(Type::kSFP);
          // We only care about MatMul inputs; skip F32 or small tensors.
        } else if (tensor.prev_type == Type::kF32 || mat.Rows() < 1024) {
          tensor.keep_type = true;
          tensor.padding = MatPadding::kPacked;  // single I/O for simplicity
        } else if (mode == WeightsPtrs::Mode::kReadBF16) {
          mat.SetType(Type::kBF16);
        }

        owners[start + task].AllocateFor(*tensor.mat, ctx.allocator,
                                         tensor.padding);
      });
}

// Mode == kMap. CPU time is negligible.
static void MapAll(const std::vector<TensorToRead>& tensors,
                   const MapPtr& mapped, uint64_t file_bytes) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    // SetPtr does not change the stride, but it is expected to be packed
    // because that is what Compress() writes to the file.
    const size_t mat_bytes = tensors[i].mat->PackedBytes();
    // Ensure blob size matches that computed from metadata.
    HWY_ASSERT_M(mat_bytes == tensors[i].range.bytes, tensors[i].mat->Name());
    // Ensure the blob lies within the file mapping.
    const uint64_t offset = tensors[i].range.offset;
    HWY_ASSERT_M(offset + mat_bytes <= file_bytes, tensors[i].mat->Name());

    tensors[i].mat->SetPtr(const_cast<uint8_t*>(mapped.get() + offset),
                           tensors[i].mat->Stride());
  }
}

// Mode == kReadBF16:

template <typename T>
static void DecompressToBF16(MatPtr& mat,
                             const hwy::AlignedFreeUniquePtr<uint8_t[]>& buf) {
  hwy::HWY_NAMESPACE::ScalableTag<BF16> dbf;
  const size_t cols = mat.Cols();

  const size_t num_packed = CompressedArrayElements<T>(mat.Extents().Area());
  const PackedSpan<T> packed{HWY_RCAST_ALIGNED(T*, buf.get()), num_packed};

  size_t packed_ofs = 0;
  for (size_t r = 0; r < mat.Rows(); ++r, packed_ofs += cols) {
    HWY_NAMESPACE::DecompressAndZeroPad(
        dbf, packed, packed_ofs, HWY_RCAST_ALIGNED(BF16*, mat.RowBytes(r)),
        cols);
  }
}

static void ReadAllToBF16(const std::vector<TensorToRead>& tensors,
                          const BlobReader& reader, ThreadingContext& ctx) {
  // Especially TSAN is slow enough to warrant hierarchical parallelism.
  const Parallelism parallelism =
      HWY_IS_DEBUG_BUILD ? Parallelism::kHierarchical : Parallelism::kFlat;
  ParallelFor(parallelism, tensors.size(), ctx, /*cluster_idx=*/0,
              Callers::kReadAllToBF16, [&](uint64_t task, size_t thread) {
                GCPP_ZONE(ctx, thread, Zones::kStartupWeightsReadAllToBF16);
                const TensorToRead& tensor = tensors[task];
                MatPtr& mat = *tensor.mat;
                // Validate blob size matches allocated buffer before any read.
                // MapAll (line ~557) and MakeBatches (line ~645) both assert
                // this; this path was the only one missing the check.
                HWY_ASSERT_M(tensor.range.bytes == tensor.prev_packed_bytes,
                             mat.Name());

                if (tensor.keep_type) {
                  HWY_ASSERT(reader.file().Read(
                      tensor.range.offset, tensor.range.bytes, mat.Packed()));
                  return;
                }

                // Read to a temporary buffer.
                const hwy::AlignedFreeUniquePtr<uint8_t[]> buf =
                    hwy::AllocateAligned<uint8_t>(tensor.range.bytes);
                HWY_ASSERT(reader.file().Read(tensor.range.offset,
                                              tensor.range.bytes, buf.get()));

                if constexpr (GEMMA_ENABLE_NUQ) {
                  if (tensor.prev_type == Type::kNUQ) {
                    return DecompressToBF16<NuqStream>(*tensor.mat, buf);
                  }
                }
                switch (tensor.prev_type) {
                  case Type::kF32:
                    return DecompressToBF16<float>(*tensor.mat, buf);
                  case Type::kBF16:
                    return DecompressToBF16<BF16>(*tensor.mat, buf);
                  case Type::kSFP:
                    return DecompressToBF16<SfpStream>(*tensor.mat, buf);
                  case Type::kQ4_0:
                    return DecompressToBF16<Q4_0Stream>(*tensor.mat, buf);
                  case Type::kMXFP4:
                    return DecompressToBF16<MxFp4Stream>(*tensor.mat, buf);
                  default:
                    HWY_ABORT("Unsupported type %s",
                              TypeName(tensor.prev_type));
                }
              });
}

// Tensors flagged `to_sfp`, in any mode that reads rather than maps:

// Number of rows to read per parallel task. Rows are grouped so that the reads
// are large enough to be efficient, but small enough that the transient buffers
// are negligible: staging the whole tensor would defeat the purpose of
// compressing it.
static size_t RowsPerChunk(size_t row_bytes) {
  constexpr size_t kTargetBytes = 4 * 1024 * 1024;
  return HWY_MAX(size_t{1}, kTargetBytes / HWY_MAX(size_t{1}, row_bytes));
}

// Holds the per-task buffers for one chunk of `rows_per_chunk` rows, so that
// both passes below can reuse the same code.
template <typename T>
class Chunk {
 public:
  Chunk(size_t cols, size_t rows_per_chunk)
      : cols_(cols), rows_per_chunk_(rows_per_chunk) {
    const hwy::HWY_NAMESPACE::ScalableTag<float> df;
    const size_t NF = hwy::HWY_NAMESPACE::Lanes(df);
    buf_ = hwy::AllocateAligned<uint8_t>(rows_per_chunk * cols * sizeof(T));
    // `DecompressAndZeroPad` writes whole vectors, and `compress-inl.h`
    // requires up to two of them beyond the requested count.
    raw_ = hwy::AllocateAligned<float>(
        hwy::RoundUpTo(rows_per_chunk * cols, NF) + 4 * NF);
    HWY_ASSERT(buf_ && raw_);
  }

  // Reads rows `[begin, end)` from the file, where the tensor is stored as
  // type `T` and packed, hence `mat.Stride()` does not apply. Returns the
  // decompressed values, ignoring `mat.Scale()`.
  float* Decompress(const TensorToRead& tensor, const BlobReader& reader,
                    size_t begin, size_t end) {
    HWY_DASSERT(end - begin <= rows_per_chunk_);
    const size_t row_bytes = cols_ * sizeof(T);
    const size_t num = (end - begin) * cols_;
    HWY_ASSERT(reader.file().Read(tensor.range.offset + begin * row_bytes,
                                  num * sizeof(T), buf_.get()));

    // Rows are contiguous in both source and destination, hence decompress the
    // entire chunk at once: this is faster, and prevents the zero padding from
    // overwriting the start of the next row.
    const hwy::HWY_NAMESPACE::ScalableTag<float> df;
    const PackedSpan<T> packed{HWY_RCAST_ALIGNED(T*, buf_.get()), num};
    HWY_NAMESPACE::DecompressAndZeroPad(df, packed, 0, raw_.get(), num);
    return raw_.get();
  }

 private:
  size_t cols_;
  size_t rows_per_chunk_;
  hwy::AlignedFreeUniquePtr<uint8_t[]> buf_;
  hwy::AlignedFreeUniquePtr<float[]> raw_;
};

// Returns the largest magnitude in the tensor, ignoring `mat.Scale()`.
template <typename T>
static float MaxAbs(const TensorToRead& tensor, const BlobReader& reader,
                    size_t rows_per_chunk, size_t num_chunks,
                    ThreadingContext& ctx) {
  const MatPtr& mat = *tensor.mat;
  const size_t rows = mat.Rows();
  const size_t cols = mat.Cols();
  // Indexed by chunk rather than by worker, so that we need not know how many
  // workers `ParallelFor` will use.
  std::vector<float> chunk_max(num_chunks, 0.0f);

  ParallelFor(Parallelism::kFlat, num_chunks, ctx, /*cluster_idx=*/0,
              Callers::kReadAllToSFP, [&](uint64_t chunk, size_t thread) {
                GCPP_ZONE(ctx, thread, Zones::kStartupWeightsReadAllToSFP);
                const size_t begin = chunk * rows_per_chunk;
                const size_t end = HWY_MIN(begin + rows_per_chunk, rows);
                Chunk<T> buffers(cols, rows_per_chunk);
                const float* raw =
                    buffers.Decompress(tensor, reader, begin, end);

                float maxabs = 0.0f;
                for (size_t i = 0; i < (end - begin) * cols; ++i) {
                  maxabs = HWY_MAX(maxabs, hwy::ScalarAbs(raw[i]));
                }
                chunk_max[chunk] = maxabs;
              });

  float maxabs = 0.0f;
  for (const float m : chunk_max) maxabs = HWY_MAX(maxabs, m);
  return maxabs;
}

// Reads the tensor as stored in the file (type `T`) and compresses it into
// `mat`, whose type `AllocateAndBindAll` already changed to `Type::kSFP`.
// The file is read twice because `SfpStream` encodes a limited range of
// magnitudes, hence we may need a per-tensor scale, which requires knowing the
// largest magnitude before encoding anything. The second read is typically
// served from the OS cache.
template <typename T>
static void CompressToSFP(const TensorToRead& tensor, const BlobReader& reader,
                          ThreadingContext& ctx) {
  MatPtr& mat = *tensor.mat;
  const size_t rows = mat.Rows();
  const size_t cols = mat.Cols();
  const size_t rows_per_chunk = RowsPerChunk(cols * sizeof(T));
  const size_t num_chunks = hwy::DivCeil(rows, rows_per_chunk);

  const float prev_scale = mat.Scale();
  const float maxabs =
      MaxAbs<T>(tensor, reader, rows_per_chunk, num_chunks, ctx) * prev_scale;
  const float scale =
      (maxabs <= SfpStream::kMax) ? 1.0f : maxabs / SfpStream::kMax;
  const float mul = prev_scale / scale;

  ParallelFor(Parallelism::kFlat, num_chunks, ctx, /*cluster_idx=*/0,
              Callers::kReadAllToSFP, [&](uint64_t chunk, size_t thread) {
                GCPP_ZONE(ctx, thread, Zones::kStartupWeightsReadAllToSFP);
                const size_t begin = chunk * rows_per_chunk;
                const size_t end = HWY_MIN(begin + rows_per_chunk, rows);
                Chunk<T> buffers(cols, rows_per_chunk);
                float* raw = buffers.Decompress(tensor, reader, begin, end);

                if (mul != 1.0f) {
                  for (size_t i = 0; i < (end - begin) * cols; ++i) {
                    // Clamp because rounding may still exceed `kMax`.
                    const float magn =
                        HWY_MIN(SfpStream::kMax, hwy::ScalarAbs(raw[i] * mul));
                    raw[i] = hwy::ScalarCopySign(magn, raw[i]);
                  }
                }

                // Row by row because destination rows are padded, whereas `raw`
                // is not. This is safe because `SfpStream` is a per-value
                // encoding.
                CompressPerThread tls;  // unused by SFP, which is stateless
                for (size_t r = begin; r < end; ++r) {
                  const PackedSpan<SfpStream> row{
                      HWY_RCAST_ALIGNED(SfpStream*, mat.RowBytes(r)), cols};
                  HWY_NAMESPACE::Compress(raw + (r - begin) * cols, cols, tls,
                                          row,
                                          /*packed_ofs=*/0);
                }
              });

  mat.SetScale(scale);
}

void weights_internal::ReadAllToSFP(const std::vector<TensorToRead>& tensors,
                                    const BlobReader& reader,
                                    ThreadingContext& ctx) {
  PROFILER_ZONE("Startup.Weights.ReadAllToSFP");
  // Usually a single (large) tensor, hence parallelize within, not across.
  for (const TensorToRead& tensor : tensors) {
    // CompressToSFP derives read sizes from the tensor shape. Ensure those
    // reads cannot cross the blob boundary if metadata is inconsistent.
    HWY_ASSERT_M(tensor.range.bytes == tensor.prev_packed_bytes,
                 tensor.mat->Name());
    switch (tensor.prev_type) {
      case Type::kF32:
        CompressToSFP<float>(tensor, reader, ctx);
        break;
      case Type::kBF16:
        CompressToSFP<BF16>(tensor, reader, ctx);
        break;
      default:
        HWY_ABORT("Unsupported type %s", TypeName(tensor.prev_type));
    }
  }
}

// Mode == kRead:

static std::vector<IOBatch> MakeBatches(
    const std::vector<TensorToRead>& tensors, const uint64_t file_bytes) {
  PROFILER_ZONE("Startup.Weights.MakeBatches");
  std::vector<IOBatch> batches;
  batches.reserve(tensors.size());

  for (size_t i = 0; i < tensors.size(); ++i) {
    const BlobRange& range = tensors[i].range;
    MatPtr& mat = *tensors[i].mat;
    uint64_t offset = range.offset;
    HWY_ASSERT(range.End() <= file_bytes);

    batches.emplace_back(offset, range.key_idx);
    if (mat.IsPacked()) {
      HWY_ASSERT(range.bytes == mat.PackedBytes());
      if (!batches.back().Add(mat.Packed(), range.bytes)) {
        // This should not happen if tensors are < 2GB.
        // If it does, we need to chunk. For now, let's assume it doesn't.
        HWY_ABORT("Packed tensor too large for a single IO batch.");
      }
      offset += range.bytes;
    } else {
      const size_t file_bytes_per_row = mat.Cols() * mat.ElementBytes();
      const size_t mem_stride_bytes = mat.Stride() * mat.ElementBytes();
      uint8_t* row_bytes = mat.RowBytes(0);
      for (size_t r = 0; r < mat.Rows(); ++r) {
        if (!batches.back().Add(row_bytes,
                                file_bytes_per_row)) {  // Full batch.
          batches.emplace_back(offset, range.key_idx);
          // Adding to an empty batch is always successful.
          HWY_ASSERT(batches.back().Add(row_bytes, file_bytes_per_row));
        }
        offset += file_bytes_per_row;
        row_bytes += mem_stride_bytes;
      }
    }
    if (offset != range.End()) {
      HWY_ABORT(
          "MISMATCH tensor %zu '%s': offset=%zu range.End()=%zu "
          "range.bytes=%zu rows=%zu cols=%zu elem=%zu packed=%d",
          i, tensors[i].mat->Name(), static_cast<size_t>(offset),
          static_cast<size_t>(range.End()), static_cast<size_t>(range.bytes),
          mat.Rows(), mat.Cols(), mat.ElementBytes(), mat.IsPacked());
    }
  }

  HWY_ASSERT(batches.size() >= tensors.size());
  return batches;
}

// Parallel synchronous I/O. Note that O_DIRECT seems undesirable because we
// want to use the OS cache between consecutive runs.
static void ReadBatches(const BlobReader& reader,
                        const std::vector<IOBatch>& batches,
                        ThreadingContext& ctx) {
  // >5x speedup from parallel reads when cached.
  ParallelFor(Parallelism::kHierarchical, batches.size(), ctx,
              /*cluster_idx=*/0, Callers::kReadBatches,
              [&](uint64_t task, size_t thread) {
                GCPP_ZONE(ctx, thread, Zones::kStartupWeightsReadBatches);
                const IOBatch& batch = batches[task];
                const std::string& key = reader.Keys()[batch.KeyIdx()];
                const uint64_t bytes_read = batch.Read(reader.file());
                if (bytes_read != batch.TotalBytes()) {
                  HWY_ABORT("Read failed for %s from %zu, %zu bytes; got %zu.",
                            key.c_str(), static_cast<size_t>(batch.Offset()),
                            static_cast<size_t>(batch.TotalBytes()),
                            static_cast<size_t>(bytes_read));
                }
              });
}

// Aborts on error. Updates `mode` to the actual mode used. Returns mapped
// memory or nullptr if `kMap` was not used.
static MapPtr MapOrReadAll(std::vector<TensorToRead>& tensors,
                           BlobReader& reader, WeightsPtrs::Mode* mode,
                           std::vector<MatOwner>& mat_owners,
                           ThreadingContext& ctx) {
  if (*mode == WeightsPtrs::Mode::kMap) {
    if (MapPtr mapped = reader.Map()) {
      MapAll(tensors, mapped, reader.file().FileSize());
      return mapped;
    }
    HWY_WARN("Failed to map file (%zu KiB), reading instead.",
             static_cast<size_t>(reader.file_bytes() >> 10));
    // If we wanted to map but failed, memory is probably not plentiful, so
    // fall through to kRead because kReadBF16 requires more memory.
    *mode = WeightsPtrs::Mode::kRead;
  }

  {
    PROFILER_ZONE("Startup.Weights.Allocate");
    // NOTE: this changes the stride of `mats`!
    AllocateAndBindAll(tensors, *mode, mat_owners, ctx);
  }

  // `MakeBatches` and `ReadAllToBF16` read into the destination rows, hence
  // tensors that require a compression pass are handled separately.
  std::vector<TensorToRead> to_sfp, rest;
  for (const TensorToRead& tensor : tensors) {
    (tensor.to_sfp ? to_sfp : rest).push_back(tensor);
  }
  if (!to_sfp.empty()) weights_internal::ReadAllToSFP(to_sfp, reader, ctx);

  if (*mode == WeightsPtrs::Mode::kReadBF16) {
    ReadAllToBF16(rest, reader, ctx);
    return MapPtr();
  }

  const std::vector<IOBatch> batches = MakeBatches(rest, reader.file_bytes());
  ReadBatches(reader, batches, ctx);
  return MapPtr();
}

WeightsPtrs::Mode WeightsPtrs::ReadFromBlobs(const ModelStore& model,
                                             BlobReader& reader,
                                             const LoaderArgs& loader,
                                             const InferenceArgs& inference,
                                             std::vector<MatOwner>& mat_owners,
                                             ThreadingContext& ctx) {
  PROFILER_ZONE("Startup.Weights.ReadFromBlobs");

  // List of tensors to read/map, and where from.
  std::vector<TensorToRead> tensors;

  // Enumerate all weights (negligible cost).
  ForEachTensor(nullptr, nullptr, [&](const TensorArgs& t) HWY_ATTR {
    size_t key_idx;
    if (model.FindAndUpdateMatPtr(t.mat, key_idx)) {
      const bool is_compressed = IsCompressed(t.mat.GetType());
      const MatPadding padding =
          (is_compressed || (t.flags & TensorArgs::kPacked))
              ? MatPadding::kPacked
              : MatPadding::kOdd;
      tensors.push_back(
          {.mat = &t.mat, .range = reader.Range(key_idx), .padding = padding});
      return;
    }
    if (t.flags & TensorArgs::kMaybeRead) return;  // optional and not found.
    HWY_ABORT("Tensor %s is required but not found in file.", t.mat.Name());
  });

  Mode mode = weights_internal::ChooseMode(reader.file_bytes(), loader,
                                           inference, ctx.allocator);

  // Compressing the input embedding to SFP halves its footprint. For models
  // with tied input/output embeddings, it also halves the weight bandwidth of
  // the per-token logits MatMul.
  if (loader.sfp_embedding == Tristate::kTrue) {
    if (mode == Mode::kMap) {
      HWY_WARN("Cannot have sfp_embedding && map, ignoring sfp_embedding.");
    } else {
      for (TensorToRead& tensor : tensors) {
        if (tensor.mat == &embedder_input_embedding) tensor.to_sfp = true;
      }
    }
  }

  mapped_ = MapOrReadAll(tensors, reader, &mode, mat_owners, ctx);

  {
    PROFILER_ZONE("Startup.Fixup");
    Fixup(mat_owners, ctx);
  }
  return mode;
}

}  // namespace gcpp
