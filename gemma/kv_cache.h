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

#include <optional>
#include <utility>
#include <vector>

#include "gemma/configs.h"     // ModelConfig
#include "gemma/gemma_args.h"  // InferenceArgs
#include "util/basics.h"       // BF16
#include "util/mat.h"
#include "hwy/base.h"

namespace gcpp {

using KV_t = BF16;
struct KVCache;

// A non-owning view of a KVCache.
struct KVCachePtr {
  bool IsEmpty() const { return kv_cache.Rows() == 0; }
  size_t SeqLen() const;

  bool IsTiled() const;
  MatPtrT<KV_t> kv_cache;
  MatPtrT<KV_t> k_cache;
  MatPtrT<KV_t> v_cache;
  KVCache* cache = nullptr;
};

struct KVCache {
  KVCache(const ModelConfig& config, const InferenceArgs& inference_args,
          const Allocator& allocator);
  KVCache(const ModelConfig& config, const InferenceArgs& inference_args,
          const RuntimeConfig& runtime_config, const Allocator& allocator);
  // Returns a deep copy of the KVCache. Use explicit function instead of
  // copy ctor to make the cost explicit.
  KVCache Copy();

  size_t SeqLen() const {
    if (IsTiled()) {
      return tiled_seq_len.value();
    }
    return kv_cache.Rows();
  }

  bool IsTiled() const {
    return tiled_seq_len.has_value();
  }

  // This function returns a vector of pointers and handles wraparound for local
  // layers.
  // You can use this function to get kv's,
  // it will slice internal circular buffer and give you parts of it that are in
  // order. Keep in mind that this gives out pointers to tiles, and for local
  // layers start_pos might be in a middle of the first tile. At start_pos %
  // kTileSize
  std::vector<MatPtr> GetPointers(int layer_idx, int kv_head_idx,
                                                int num_kv_heads, int start_pos,
                                                bool is_global_layer) {
    if (!IsTiled()) {
      HWY_ABORT("This function is only meant to be used with tiled KV caches.");
    }
    MatPtr& source_ptr = kv_head_ptrs[layer_idx * num_kv_heads + kv_head_idx];
    if (is_global_layer) {
      return {source_ptr};
    }
    size_t start_tile_mod_window = (start_pos / kTileSize) % source_ptr.Rows();
    size_t start_len = source_ptr.Rows() - start_tile_mod_window;
    MatPtr start_ptr("kv_start", source_ptr.GetType(),
                     Extents2D(start_len, source_ptr.Cols()));
    start_ptr.SetPtr(source_ptr.RowBytes(start_tile_mod_window),
                     source_ptr.Cols());
    return {start_ptr, source_ptr};
  }

  static constexpr size_t kTileSize = 32;
  std::optional<uint32_t> tiled_seq_len = std::nullopt;
  // Default Format
  // If tiled_seq_len is not set, then the kv_cache is assumed to be [seq_len,
  // layers * kv_heads * qkv_dim * 2].
  //
  // Tiled Format
  // If tiled_seq_len is set, the kv cache is stored in tiled format.
  // Allocations must happen in full tiles.
  // The order of dimensions on rows is: [layer, kv_head, tile].
  // The total number of rows is:
  //  num_layers * num_kv_heads * (tiled_seq_len / kTileSize).
  // Each tile (containing kTileSize elements from the sequence) can be thought
  // of as storing K^T and V, where K is shaped [kTileSize, qkv_dim].

  // Type erased kv cache. It's compact because local layers are allocated as
  // circular buffers.
  MatPtr compact_kv_cache_ptr;
  MatOwner compact_kv_cache;
  // Pointers to the raw KV storage indexed by layer and head. This helps
  // accessing the tiles even though different layers may have a different
  // number of tiles in storage. All pointers point into compact_kv_cache.

  // To access the tiles of (layer_idx, head_idx), index the array with
  // layer_idx * num_kv_heads + kv_head_idx.
  // Or use GetPointers function.

  // The returned MatPtr will have one tile per row. The number of rows for
  // global layers is max_seq_len/kTileSize. For local layers it is slightly
  // more than attention_window_size[layer_idx] / kTileSize. For local layers, a
  // given token_idx is in row (token_idx / kTileSize) %
  // kv_head_ptrs[...].Rows().
  std::vector<MatPtr> kv_head_ptrs;
  MatStorageT<KV_t> kv_cache;  // [seq_len, layers * kv_heads * qkv_dim * 2]
  // The format of k_cache indicates that there are pairs of values from
  // qkv_dim in groups of 2x kFloatsPerVector(=NF) elements from the sequence,
  // in groups of qkv_dim/2 elements in groups of kv_heads elements.
  // This enables sequential loading of the data when filling 2 vectors with
  // NF sequence elements of pairs of BF16 qkv values. The next vector then
  // continues reading the rest of qkv.
  // [seq_len / 2NF, layers * kv_heads * qkv_dim/2 * 2NF * 2]
  MatStorageT<KV_t> k_cache;
  // v_cache is formatted to allow sequential access to V during scaling and
  // update of att_out.
  // Originally [seq_len, layers * kv_heads * qkv_dim]
  // v_cache is transposed to:
  // [layers, kv_heads, seq_len, qkv_dim], reshaped to:
  // [layers, kv_heads, seq_len/(2NF), 2NF, qkv_dim/(2NF), 2NF]
  // then transposed to:
  // [seq_len/(2NF), layers, kv_heads, qkv_dim/(2NF), 2NF, 2NF]
  // and finally packed in a 2D MatStorageT as:
  // [seq_len/(2NF), layers * kv_heads * qkv_dim/(2NF) * 2NF * 2NF]
  // This allows sequential reads of 2NF registers each of 2NF BF16 values,
  // repeatedly until all of qkv_dim is read.
  MatStorageT<KV_t> v_cache;

  KVCachePtr ToPtr() {
    return KVCachePtr{
        .kv_cache = kv_cache,
        .k_cache = k_cache,
        .v_cache = v_cache,
    };
  }

 private:
  const Allocator& allocator_;

  // For use by other ctor and Copy()
  KVCache(const Extents2D& kv_extents, const Allocator& allocator);
};

inline size_t KVCachePtr::SeqLen() const {
  if (IsTiled()) {
    return cache->tiled_seq_len.value();
  }
  return kv_cache.Rows();
}

inline bool KVCachePtr::IsTiled() const {
  // MPU code create a KVCachePtr without kv_cache.
  return cache != nullptr && cache->tiled_seq_len.has_value();
}

// Convenience function to create views into KVCaches.
std::vector<KVCachePtr> ToKVCachePtrs(const hwy::Span<KVCache>& kv_caches);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_KV_CACHE_H_
