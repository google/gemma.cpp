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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_KV_CACHE_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_KV_CACHE_H_

#include <stddef.h>
#include <stdint.h>

#include <optional>
#include <vector>

#include "gemma/configs.h"     // ModelConfig
#include "gemma/gemma_args.h"  // InferenceArgs
#include "util/basics.h"       // BF16
#include "util/mat.h"
#include "hwy/base.h"

namespace gcpp {

using KV_t = BF16;
using KV_microscale_t = BF16;
struct KVCache;

// A non-owning view of a KVCache.
struct KVCachePtr {
  bool IsEmpty() const;
  size_t SeqLen() const;

  bool IsTiled() const;
  MatPtrT<KV_t> kv_cache;
  KVCache* cache = nullptr;
};

struct KVCache {
  // Both entry points select the same layout for the same attention backend.
  KVCache(const ModelConfig& config, const InferenceArgs& inference_args,
          const Allocator& allocator);
  KVCache(const ModelConfig& config, const InferenceArgs& inference_args,
          AttentionImpl attention_impl, const Allocator& allocator,
          std::optional<Type> kv_cache_type = std::nullopt);

  // Returns an independent snapshot, including compact storage and aliases.
  KVCache Copy() const;
  size_t SeqLen() const { return seq_len_; }

  // Prepare compact BF16 storage before projecting a batch. Growth preserves
  // live history, including partial SIMD tiles. The SIMD width is fixed after
  // the first use of this cache.
  void PrepareLayer(size_t layer, size_t num_tokens, size_t pos,
                    size_t tile_size);
  MatPtrT<KV_t> FlashK(size_t layer, size_t head) const;
  MatPtrT<KV_t> FlashV(size_t layer, size_t head) const;
  size_t LayerCapacity(size_t layer) const;
  void Clear();
  size_t AllocatedBytes() const;

  bool IsTiled() const { return !kv_head_ptrs.empty(); }

  // Returns chronological spans of fixed-size tiles for the transposed
  // backends. start_pos may lie inside the first returned tile.
  std::vector<MatPtr> GetPointers(size_t layer_idx, size_t kv_head_idx,
                                  size_t start_pos,
                                  bool is_global_layer) const {
    HWY_DASSERT(IsTiled() && attention_impl_ != AttentionImpl::kFlash);
    MatPtr source =
        kv_head_ptrs[layer_kv_head_offsets[layer_idx] + kv_head_idx];
    if (is_global_layer) return {source};
    const size_t first = (start_pos / kTileSize) % source.Rows();
    MatPtr tail("kv_start", source.GetType(),
                Extents2D(source.Rows() - first, source.Cols()));
    tail.SetPtr(source.RowBytes(first), source.Stride());
    tail.SetLayout(source.GetLayout());
    return {tail, source};
  }

  // Saved sizes for computing offsets into the KV cache.
  size_t num_layers = 0;
  size_t kv_heads = 0;
  size_t qkv_dim = 0;

  // Cumulative non-uniform offset tables
  std::vector<uint32_t> layer_flat_offsets;
  std::vector<uint32_t> layer_kv_head_offsets;

  // DeepSeek V4 per-query incremental compressor state (kv_state/score_state
  // per layer, plus the indexer compressor's on CSA layers), f32. One row;
  // `ds_state_offsets[layer]` is the element offset of a layer's segment.
  // Zero-sized unless the model has V4 compressor layers. deepseek.cc
  // re-initializes a layer's segment when it processes cache position 0.
  MatStorageT<float> ds_state;
  // Same layout as `ds_state`: per-layer boundary snapshot for speculative
  // decoding (state after the committed token of a verify step). Written by
  // deepseek.cc when `Activations::ds_snapshot_after` is set; restored
  // wholesale by the driver when a draft is rejected.
  MatStorageT<float> ds_state_snapshot;
  std::vector<uint32_t> ds_state_offsets;
  static constexpr size_t kTileSize = 32;

  // Compact storage is indexed by owning layer and head. Shared layers reuse
  // their source's offsets. Each row contains one K/V tile; each head has its
  // own stride so heterogeneous head dimensions need no padding to a maximum.
  // BF16 Flash tiles use 2 * SIMD float lanes; other backends use kTileSize.
  // In a Flash tile, K holds dimension pairs across tokens, followed by V
  // blocked by the SIMD tile width. FlashK/FlashV expose the two halves with
  // a stride spanning the whole tile.
  std::vector<MatPtr> kv_head_ptrs;
  MatStorageT<KV_t> kv_cache;  // [seq_len, layers * kv_heads * qkv_dim * 2]
  KVCachePtr ToPtr() {
    return KVCachePtr{
        .kv_cache = kv_cache,
        .cache = this,
    };
  }

 private:
  size_t seq_len_ = 0;
  AttentionImpl attention_impl_ = AttentionImpl::kFlash;
  size_t flash_tile_size_ = 0;
  std::vector<MatOwner> kv_head_owners_;
  std::vector<size_t> head_windows_;
  std::vector<size_t> head_dims_;
  std::vector<size_t> layer_heads_;
  void ResizeHead(size_t head, size_t rows, size_t pos);
  explicit KVCache(const Allocator& allocator) : allocator_(allocator) {}
  const Allocator& allocator_;
};

inline bool KVCachePtr::IsEmpty() const {
  return cache ? cache->SeqLen() == 0 : kv_cache.Rows() == 0;
}

inline size_t KVCachePtr::SeqLen() const {
  return cache ? cache->SeqLen() : kv_cache.Rows();
}

inline bool KVCachePtr::IsTiled() const {
  return cache != nullptr && cache->IsTiled();
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_KV_CACHE_H_
