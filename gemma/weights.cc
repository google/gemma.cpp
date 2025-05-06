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
#include <stdio.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "compression/blob_store.h"
#include "compression/compress.h"
#include "compression/shared.h"
#include "gemma/configs.h"
#include "gemma/model_store.h"
#include "util/mat.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/highway.h"
#include "hwy/profiler.h"
#include "hwy/stats.h"

// TODO: move into foreach_target; this is only used for NUQ Reshape.
#include "compression/compress-inl.h"

namespace gcpp {

template <>
void LayerWeightsPtrs<NuqStream>::Reshape() {
  if (!attn_vec_einsum_w.HasPtr()) return;
  HWY_ASSERT(attn_vec_einsum_w.GetType() == Type::kNUQ);

  HWY_ASSERT(att_weights.HasPtr());
  HWY_ASSERT(att_weights.GetType() == Type::kNUQ);

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
  hwy::ThreadPool pool(0);
  HWY_NAMESPACE::Compress(att_weights_tmp.get(), model_dim * heads * qkv_dim,
                          work, att_weights.Span(),
                          /*packed_ofs=*/0, pool);

  att_weights.SetScale(attn_vec_einsum_w.Scale());
}

// Aborts on error.
static void MapOrRead(const std::vector<MatPtr*>& mats, BlobReader& reader,
                      const std::vector<BlobRange>& ranges,
                      MatOwners& mat_owners, const MatPadding padding,
                      hwy::ThreadPool& pool) {
  HWY_ASSERT(mats.size() == ranges.size());

  if (reader.IsMapped()) {
    PROFILER_ZONE("Startup.Weights.Map");
    for (size_t i = 0; i < mats.size(); ++i) {
      // SetPtr does not change the stride, but it is expected to be packed
      // because that is what Compress() writes to the file.
      const size_t mat_bytes = mats[i]->PackedBytes();
      // Ensure blob size matches that computed from metadata.
      HWY_ASSERT_M(mat_bytes == ranges[i].bytes, mats[i]->Name());

      hwy::Span<const uint8_t> span = reader.MappedSpan<uint8_t>(ranges[i]);
      HWY_ASSERT(span.size() == mat_bytes);
      mats[i]->SetPtr(const_cast<uint8_t*>(span.data()), mats[i]->Stride());
    }
    return;
  }

  PROFILER_ZONE("Startup.Weights.AllocateAndEnqueue");

  // NOTE: this changes the stride of `mats`!
  mat_owners.AllocateFor(mats, padding, pool);

  // Enqueue the read requests, one per row in each tensor.
  for (size_t i = 0; i < mats.size(); ++i) {
    uint64_t offset = ranges[i].offset;
    const size_t file_bytes_per_row = mats[i]->Cols() * mats[i]->ElementBytes();
    // Caution, `RowT` requires knowledge of the actual type. We instead use
    // the first row, which is the same for any type, and advance the *byte*
    // pointer by the *byte* stride.
    const size_t mem_stride_bytes = mats[i]->Stride() * mats[i]->ElementBytes();
    uint8_t* row = mats[i]->RowT<uint8_t>(0);
    for (size_t r = 0; r < mats[i]->Rows(); ++r) {
      reader.Enqueue(BlobRange{.offset = offset,
                               .bytes = file_bytes_per_row,
                               .key_idx = ranges[i].key_idx},
                     row);
      offset += file_bytes_per_row;
      row += mem_stride_bytes;
      // Keep the in-memory row padding uninitialized so msan detects any use.
    }
  }

  reader.ReadAll(pool);
}

void WeightsOwner::ReadOrAllocate(const ModelStore& model, BlobReader& reader,
                                  hwy::ThreadPool& pool) {
  // List of tensors to read/map, and where from.
  std::vector<MatPtr*> mats;
  std::vector<BlobRange> ranges;

  // Padding is inserted when reading row by row, except for NUQ tensors.
  const MatPadding padding = MatPadding::kOdd;

  AllocatePointer(model.Config());

  // Enumerate all weights (negligible cost).
  CallT([&](const auto& weights) {
    weights->ForEachTensor(nullptr, nullptr, [&](const TensorArgs& t) {
      if (t.flags & TensorArgs::kOnlyAllocate) {
        mat_owners_.AllocateFor(t.mat, padding);
        return;
      }
      size_t key_idx;
      if (model.FindAndUpdateMatPtr(t.mat, key_idx)) {
        mats.push_back(&t.mat);
        ranges.push_back(reader.Range(key_idx));
        return;
      }
      if (t.flags & TensorArgs::kMaybeRead) return;  // optional and not found.
      HWY_ABORT("Tensor %s is required but not found in file.", t.mat.Name());
    });
  });

  MapOrRead(mats, reader, ranges, mat_owners_, padding, pool);

  Reshape(pool);
}

// Allocates `*_weights_`, but not yet the tensors inside. This is split out
// of `CallT` because that is const, hence it would pass a const& of the
// `unique_ptr` to its lambda, but we want to reset the pointer.
void WeightsOwner::AllocatePointer(const ModelConfig& config) {
  switch (weight_type_) {
    case Type::kSFP:
      sfp_weights_.reset(new ModelWeightsPtrs<SfpStream>(config));
      break;
    case Type::kNUQ:
      nuq_weights_.reset(new ModelWeightsPtrs<NuqStream>(config));
      break;
    case Type::kF32:
      float_weights_.reset(new ModelWeightsPtrs<float>(config));
      break;
    case Type::kBF16:
      bf16_weights_.reset(new ModelWeightsPtrs<BF16>(config));
      break;
    default:
      HWY_ABORT("Unsupported weight type %s.", TypeName(weight_type_));
  }
}

// Gemma calls `WeightsOwner::ReadOrAllocate`, but test code instead calls
// `WeightsPtrs::AllocateForTest`, so the implementation is there, and here
// we only type-dispatch.
void WeightsOwner::AllocateForTest(const ModelConfig& config,
                                   hwy::ThreadPool& pool) {
  PROFILER_ZONE("Startup.AllocateWeights");

  AllocatePointer(config);
  CallT([&](const auto& weights) {
    weights->AllocateForTest(mat_owners_, pool);
  });
}

void WeightsOwner::ZeroInit() {
  PROFILER_FUNC;
  CallT([](const auto& weights) { weights->ZeroInit(); });
}

void WeightsOwner::RandInit(float stddev, std::mt19937& gen) {
  PROFILER_FUNC;
  float_weights_->RandInit(stddev, gen);
}

void WeightsOwner::LogWeightStatsF32() {
  size_t total_weights = 0;
  HWY_ASSERT(weight_type_ == Type::kF32);  // Only for float weights.
  float_weights_->ForEachTensor(
      nullptr, nullptr, [&total_weights](const TensorArgs& t) {
        if (t.mat.Scale() != 1.0f) {
          printf("[scale=%f] ", t.mat.Scale());
        }
        hwy::Stats stats;
        HWY_ASSERT(t.mat.GetType() == Type::kF32);
        for (size_t r = 0; r < t.mat.Rows(); ++r) {
          const float* HWY_RESTRICT row = t.mat.RowT<float>(r);
          for (size_t c = 0; c < t.mat.Cols(); ++c) {
            stats.Notify(row[c]);
          }
        }
        printf("%-20s  %12zu   %13.10f   %8.5f   %13.10f\n", t.mat.Name(),
               t.mat.Rows() * t.mat.Cols(), stats.Min(), stats.Mean(),
               stats.Max());

        total_weights += t.mat.Rows() * t.mat.Cols();
      });
  printf("%-20s  %12zu\n", "Total", total_weights);
}

void WeightsOwner::Reshape(hwy::ThreadPool& pool) {
  PROFILER_ZONE("Startup.Reshape");
  CallT([&pool](const auto& weights) { weights->Reshape(pool); });
}

std::vector<uint32_t> WeightsOwner::AddTensorDataToWriter(
    BlobWriter& writer) const {
  std::vector<uint32_t> serialized_mat_ptrs;
  CallT([&](const auto& weights) {
    weights->ForEachTensor(nullptr, nullptr, [&](const TensorArgs& t) {
      if (t.flags & TensorArgs::kOnlyAllocate) return;
      if (t.flags & TensorArgs::kMaybeRead && !t.mat.HasPtr()) return;
      HWY_ASSERT_M(t.mat.HasPtr(), t.mat.Name());
      writer.Add(t.mat.Name(), t.mat.Packed(), t.mat.PackedBytes());
      t.mat.AppendTo(serialized_mat_ptrs);
    });
  });
  return serialized_mat_ptrs;
}

}  // namespace gcpp
