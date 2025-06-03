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
#include "gemma/model_store.h"
#include "io/blob_store.h"
#include "ops/matmul.h"  // MMParallel
#include "util/mat.h"
#include "util/threading_context.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/highway.h"
#include "hwy/profiler.h"

// TODO: move into foreach_target; this is only used for NUQ Fixup.
#include "compression/compress-inl.h"

namespace gcpp {

// Copies att_weights from `attn_vec_einsum_w`.
void LayerWeightsPtrs::InitAttWeights(std::vector<MatOwner>& mat_owners) {
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

  {
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    mat_owners.push_back(MatOwner());
    mat_owners.back().AllocateFor(att_weights, MatPadding::kOdd);
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

// For FFN. Fast, only updates pointers.
void LayerWeightsPtrs::SplitW1() {
  // Used for Gemma and Griffin layers; FFWVit uses different tensors.
  if (layer_config.type == LayerAttentionType::kVit) return;

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

  const size_t stride = gating_einsum_w.Stride();
  gating_einsum_w1.SetPtr(gating_einsum_w.RowBytes(0), stride);
  gating_einsum_w2.SetPtr(gating_einsum_w.RowBytes(ff_hidden_dim), stride);
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

  const size_t w1_rows = layer_config.heads * layer_config.qkv_dim;
  const size_t w2_rows = layer_config.kv_heads * 2 * layer_config.qkv_dim;
  HWY_ASSERT(qkv_einsum_w.Rows() == w1_rows + w2_rows);
  HWY_ASSERT(qkv_einsum_w1.Rows() == w1_rows);
  HWY_ASSERT(qkv_einsum_w2.Rows() == w2_rows);
  // Cols are the model_dim but we don't have ModelConfig here.
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

// Must be called after reading weights via `ForEachTensor`.
// TODO: exporters should bake this into the weights already.
// WARNING: called from multiple threads; `mat_owners` requires a lock.
void LayerWeightsPtrs::Fixup(std::vector<MatOwner>& mat_owners) {
  // TODO(janwas): handle NUQ
  InitAttWeights(mat_owners);
  SplitW1();
  SplitAttW1();
}

static void HWY_MAYBE_UNUSED InitAttWeightsNUQ(
    const LayerConfig& layer_config, MatPtrT<NuqStream>& attn_vec_einsum_w,
    MatPtrT<NuqStream>& att_weights, std::vector<MatOwner>& mat_owners) {
  if (!attn_vec_einsum_w.HasPtr()) return;
  HWY_ASSERT(attn_vec_einsum_w.GetType() == Type::kNUQ);

  HWY_ASSERT(att_weights.HasPtr());
  att_weights.SetType(Type::kNUQ);

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

static void HWY_MAYBE_UNUSED SplitW1NUQ(const LayerConfig& layer_config) {
  // TODO(janwas): implement.
}

// Zero-initializes only the allocated tensors in `*this`.
void ModelWeightsPtrs::ZeroInit() {
  ForEachTensor(nullptr, nullptr, [](const TensorArgs& t) {
    if (!t.mat.HasPtr()) return;
    gcpp::ZeroInit(t.mat);
  });
}

// Copies only the allocated tensors in `*this` from tensors in `other`.
void ModelWeightsPtrs::CopyFrom(const ModelWeightsPtrs& other) {
  ForEachTensor(const_cast<ModelWeightsPtrs*>(&other), nullptr,
                [](const TensorArgs& t) {
                  if (!t.mat.HasPtr()) return;
                  HWY_ASSERT(t.other_mat1 && t.other_mat1->HasPtr());
                  CopyMat(*t.other_mat1, t.mat);
                });
}

// For reshaping file tensors to the shape expected by the code. This would
// ideally already happen in the importer. Called by WeightsOwner::Fixup.
void ModelWeightsPtrs::Fixup(std::vector<MatOwner>& mat_owners,
                             hwy::ThreadPool& pool) {
  pool.Run(0, c_layers.size(), [&](uint64_t layer, size_t /*thread*/) {
    GetLayer(layer)->Fixup(mat_owners);
  });

  pool.Run(0, vit_layers.size(), [&](uint64_t layer, size_t /*thread*/) {
    VitLayer(layer)->Fixup(mat_owners);
  });
}

std::vector<uint32_t> ModelWeightsPtrs::AddTensorDataToWriter(
    BlobWriter& writer) const {
  std::vector<uint32_t> serialized_mat_ptrs;
  // ForEachTensor is non-const but the lambda does not modify *this.
  const_cast<ModelWeightsPtrs*>(this)->ForEachTensor(
      nullptr, nullptr, [&](const TensorArgs& t) {
        if (t.flags & TensorArgs::kMaybeRead && !t.mat.HasPtr()) return;
        HWY_ASSERT_M(t.mat.HasPtr(), t.mat.Name());
        writer.Add(t.mat.Name(), t.mat.Packed(), t.mat.PackedBytes());
        t.mat.AppendTo(serialized_mat_ptrs);
      });
  return serialized_mat_ptrs;
}

struct TensorToRead {
  MatPtr* mat;
  BlobRange range;
  // Some tensors opt out of padding via kPacked flags.
  MatPadding padding;
};

// Allocates multiple in parallel and binds to NUMA nodes.
static void AllocateAndBindAll(const std::vector<TensorToRead>& tensors,
                               std::vector<MatOwner>& owners,
                               hwy::ThreadPool& pool) {
  const size_t start = owners.size();
  owners.resize(start + tensors.size());

  MMParallel parallel(ThreadingContext::Get());

  // Allocate in parallel because faulting in large tensors is slow.
  pool.Run(0, tensors.size(), [&](uint64_t task, size_t /*thread*/) {
    owners[start + task].AllocateFor(*tensors[task].mat, tensors[task].padding);
    // TODO(janwas): MatMul outputs will later also be BF16.
    BindB(*tensors[task].mat, sizeof(float), parallel);
  });
}

// Parallel I/O into allocated memory, or mapped view of file. The latter is
// better when the file is huge, but page faults add noise to measurements.
enum class Mode { kRead, kMap };

// Decides whether to read or map based on heuristics and user override.
static Mode ChooseMode(uint64_t file_bytes, Tristate map) {
  const Allocator& allocator = ThreadingContext::Get().allocator;
  // User has explicitly requested a map or read via args.
  if (map == Tristate::kTrue) return Mode::kMap;
  if (map == Tristate::kFalse) return Mode::kRead;
  // Else: use heuristics to choose. Note that `FreeMiB` is generally low
  // because idle memory is used as cache, so do not use it to decide.
  const size_t file_mib = file_bytes >> 20;
  const size_t total_mib = allocator.TotalMiB();
  if (file_mib > total_mib) {
    HWY_WARN("Weight file %zu MiB > detected memory %zu MiB.",
             static_cast<size_t>(file_mib), total_mib);
  }
  // Large fraction of total.
  if (file_mib >= total_mib / 3) return Mode::kMap;
  // Big enough that even parallel loading wouldn't be quick.
  if (file_mib > 50 * 1024) return Mode::kMap;
  return Mode::kRead;
}

static MapPtr MapFileOrNull(File& file, uint64_t file_bytes) {
  const Allocator& allocator = ThreadingContext::Get().allocator;
  if (file_bytes % allocator.BasePageBytes() == 0) {
    MapPtr mapped = file.Map();
    if (!mapped) {
      HWY_WARN("Failed to map file (%zu KiB), reading instead.",
               static_cast<size_t>(file_bytes >> 10));
    }
  } else {
    HWY_WARN("Unable to map non-padded file (%zu, %zu), reading instead.",
             static_cast<size_t>(file_bytes >> 10), allocator.BasePageBytes());
  }
  return MapPtr();
}

static void MapAll(const std::vector<TensorToRead>& tensors,
                   const MapPtr& mapped) {
  PROFILER_ZONE("Startup.Weights.Map");
  for (size_t i = 0; i < tensors.size(); ++i) {
    // SetPtr does not change the stride, but it is expected to be packed
    // because that is what Compress() writes to the file.
    const size_t mat_bytes = tensors[i].mat->PackedBytes();
    // Ensure blob size matches that computed from metadata.
    HWY_ASSERT_M(mat_bytes == tensors[i].range.bytes, tensors[i].mat->Name());

    tensors[i].mat->SetPtr(
        const_cast<uint8_t*>(mapped.get() + tensors[i].range.offset),
        tensors[i].mat->Stride());
  }
}

static std::vector<IOBatch> MakeBatches(
    const std::vector<TensorToRead>& tensors, const uint64_t file_bytes) {
  PROFILER_ZONE("Startup.Weights.MakeBatches");
  // Batches must be contiguous but blobs are padded, hence at least one
  // batch per tensor, and more when tensor rows exceed the batch size.
  std::vector<IOBatch> batches;
  batches.reserve(tensors.size());

  for (size_t i = 0; i < tensors.size(); ++i) {
    const BlobRange& range = tensors[i].range;
    MatPtr& mat = *tensors[i].mat;
    uint64_t offset = range.offset;
    HWY_ASSERT(range.End() <= file_bytes);

    batches.emplace_back(offset, range.key_idx);
    const size_t file_bytes_per_row = mat.Cols() * mat.ElementBytes();
    const size_t mem_stride_bytes = mat.Stride() * mat.ElementBytes();
    uint8_t* row_bytes = mat.RowBytes(0);
    for (size_t r = 0; r < mat.Rows(); ++r) {
      if (!batches.back().Add(row_bytes, file_bytes_per_row)) {  // Full batch.
        batches.emplace_back(offset, range.key_idx);
        // Adding to an empty batch is always successful.
        HWY_ASSERT(batches.back().Add(row_bytes, file_bytes_per_row));
      }
      offset += file_bytes_per_row;
      row_bytes += mem_stride_bytes;
      // Keep the in-memory row padding uninitialized so msan detects any use.
    }
    HWY_ASSERT(offset == range.End());
  }

  HWY_ASSERT(batches.size() >= tensors.size());
  return batches;
}

// Parallel synchronous I/O. Note that O_DIRECT seems undesirable because we
// want to use the OS cache between consecutive runs.
static void ReadBatches(const BlobReader& reader,
                        const std::vector<IOBatch>& batches,
                        hwy::ThreadPool& pool) {
  PROFILER_ZONE("Startup.Weights.Read");
  // >5x speedup from parallel reads when cached.
  pool.Run(0, batches.size(), [&](uint64_t i, size_t /*thread*/) {
    const IOBatch& batch = batches[i];
    const std::string& key = reader.Keys()[batch.KeyIdx()];
    const uint64_t bytes_read = batch.Read(reader.file());
    if (bytes_read != batch.TotalBytes()) {
      HWY_ABORT("Read failed for %s from %zu, %zu bytes; got %zu.", key.c_str(),
                static_cast<size_t>(batch.Offset()),
                static_cast<size_t>(batch.TotalBytes()),
                static_cast<size_t>(bytes_read));
    }
  });
}

// Aborts on error.
static void MapOrReadAll(const std::vector<TensorToRead>& tensors,
                         BlobReader& reader, Tristate map,
                         std::vector<MatOwner>& mat_owners,
                         hwy::ThreadPool& pool) {
  if (ChooseMode(reader.file_bytes(), map) == Mode::kMap) {
    MapPtr mapped = MapFileOrNull(reader.file(), reader.file_bytes());
    if (mapped) {
      MapAll(tensors, mapped);
      return;
    }
  }  // otherwise fall through to read mode

  {
    PROFILER_ZONE("Startup.Weights.Allocate");
    // NOTE: this changes the stride of `mats`!
    AllocateAndBindAll(tensors, mat_owners, pool);
  }

  const std::vector<IOBatch> batches =
      MakeBatches(tensors, reader.file_bytes());
  ReadBatches(reader, batches, pool);
}

void ModelWeightsPtrs::ReadFromBlobs(const ModelStore& model,
                                     BlobReader& reader, Tristate map,
                                     std::vector<MatOwner>& mat_owners,
                                     hwy::ThreadPool& pool) {
  // List of tensors to read/map, and where from.
  std::vector<TensorToRead> tensors;

  // Enumerate all weights (negligible cost).
  ForEachTensor(nullptr, nullptr, [&](const TensorArgs& t) {
    const MatPadding padding = (t.flags & TensorArgs::kPacked)
                                   ? MatPadding::kPacked
                                   : MatPadding::kOdd;
    size_t key_idx;
    if (model.FindAndUpdateMatPtr(t.mat, key_idx)) {
      tensors.push_back(
          {.mat = &t.mat, .range = reader.Range(key_idx), .padding = padding});
      return;
    }
    if (t.flags & TensorArgs::kMaybeRead) return;  // optional and not found.
    HWY_ABORT("Tensor %s is required but not found in file.", t.mat.Name());
  });

  MapOrReadAll(tensors, reader, map, mat_owners, pool);

  {
    PROFILER_ZONE("Startup.Fixup");
    Fixup(mat_owners, pool);
  }
}

}  // namespace gcpp
