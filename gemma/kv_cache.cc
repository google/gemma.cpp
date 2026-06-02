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

#include "gemma/kv_cache.h"

#include <stddef.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "compression/types.h"
#include "gemma/configs.h"
#include "gemma/gemma_args.h"
#include "util/mat.h"  // ZeroInit
#include "hwy/base.h"    // HWY_MAX

namespace gcpp {

// Number of rows for KV cache. Note that both rows and cols are u32, and
// the total number of elements can exceed 2^32.
static size_t CappedSeqLen(const ModelConfig& config,
                           const InferenceArgs& inference_args) {
  if (inference_args.seq_len > config.max_seq_len) {
    HWY_WARN("Capping seq_len %zu to config.max_seq_len %u.",
             inference_args.seq_len, config.max_seq_len);
    return config.max_seq_len;
  }
  return inference_args.seq_len;
}

KVCache::KVCache(const Extents2D& kv_extents, size_t num_layers,
                 size_t kv_heads, size_t qkv_dim, const Allocator& allocator)
    : num_layers(num_layers),
      kv_heads(kv_heads),
      qkv_dim(qkv_dim),
      rounded_qkv_dim(hwy::RoundUpTo(qkv_dim, kMaxBF16PerVector)),
      kv_cache("kv", kv_extents, allocator, MatPadding::kOdd),
      // WARNING: the rows and cols of k_cache and v_cache will be modified
      // before use!
      // The rows will be reduced by a factor of 2xkFloatsPerVector, and the
      // cols will be increased by 2xkFloatsPerVector on first use. This is to
      // avoid making KVCache another class that has to be duplicated for each
      // machine architecture, since kFloatsPerVector is architecture dependent.
      // The change is shape is safe only if the padding is kPacked.
      k_cache("k",
              Extents2D(hwy::RoundUpTo(kv_extents.rows, kMaxBF16PerVector),
                        KOrVDefaultCols()),
              allocator, MatPadding::kPacked),
      v_cache("v",
              Extents2D(hwy::RoundUpTo(kv_extents.rows, kMaxBF16PerVector),
                        KOrVDefaultCols()),
              allocator, MatPadding::kPacked),
      allocator_(allocator) {}

KVCache::KVCache(const ModelConfig& config, const InferenceArgs& inference_args,
                 const Allocator& allocator)
    : KVCache(
          Extents2D(CappedSeqLen(config, inference_args), config.KVCacheCols()),
          config.layer_configs.size(), config.layer_configs[0].kv_heads,
          config.layer_configs[0].qkv_dim, allocator) {}

KVCache::KVCache(const ModelConfig& config, const InferenceArgs& inference_args,
                 const RuntimeConfig& runtime_config,
                 const Allocator& allocator)
    : allocator_(allocator) {
  // clang-format off
  if (runtime_config.attention_impl == AttentionImpl::kFlashTransposedQs ||
      runtime_config.attention_impl == AttentionImpl::kFlashTransposedQsInt16 ||
      runtime_config.attention_impl == AttentionImpl::kFlashTransposedQsBF16
      ) {
    // clang-format on
    const size_t num_tiles =
        hwy::DivCeil(CappedSeqLen(config, inference_args), kTileSize);
    tiled_seq_len = num_tiles * kTileSize;
    Type kv_cache_type;
    if (runtime_config.attention_impl == AttentionImpl::kFlashTransposedQsBF16
    ) {
      kv_cache_type = runtime_config.kv_cache_type.value_or(Type::kBF16);
    } else if (runtime_config.attention_impl ==
               AttentionImpl::kFlashTransposedQsInt16) {
      if (runtime_config.kv_cache_type.has_value() &&
          runtime_config.kv_cache_type.value() != Type::kInt8) {
        HWY_WARN(
            "You are have set kv_cache_type to %s, but you are using "
            "FlashTransposedQsInt16 attention implementation which only "
            "supports Int8. kv_cache_type will be set to Int8.",
            runtime_config.kv_cache_type.value());
      }
      kv_cache_type = Type::kInt8;
    } else {
      kv_cache_type = runtime_config.kv_cache_type.value_or(Type::kF32);
    }

    int tile_length = 2 * config.layer_configs[0].qkv_dim * kTileSize;
    if (kv_cache_type == Type::kInt8) {
      // microscaling
      tile_length += 2 * sizeof(BF16) * kTileSize;
    }
    auto num_tiles_per_head = [](size_t window_size, size_t prefill_tbatch_size,
                                 size_t max_seq_len) {
      return hwy::DivCeil(
          std::min(max_seq_len, window_size + prefill_tbatch_size), kTileSize);
    };

    size_t total_num_tiles = 0;
    for (size_t window_size : config.attention_window_sizes) {
      total_num_tiles +=
          num_tiles_per_head(window_size, runtime_config.prefill_tbatch_size,
                             config.max_seq_len) *
          config.layer_configs[0].kv_heads;
    }
    Extents2D extents(total_num_tiles, tile_length);
    compact_kv_cache_ptr = MatPtr("kv_tiled", kv_cache_type, extents);
    compact_kv_cache.AllocateFor(compact_kv_cache_ptr, allocator,
                                 MatPadding::kPacked);
    total_num_tiles = 0;
    kv_head_ptrs.reserve(config.attention_window_sizes.size() *
                         config.layer_configs[0].kv_heads);
    for (size_t window_size : config.attention_window_sizes) {
      for (size_t kv = 0; kv < config.layer_configs[0].kv_heads; ++kv) {
        size_t num_tiles_per_kv_head =
            num_tiles_per_head(window_size, runtime_config.prefill_tbatch_size,
                               config.max_seq_len);
        MatPtr kv_ptr("kv_ptr", kv_cache_type,
                      Extents2D(num_tiles_per_kv_head, tile_length));
        kv_ptr.SetPtr(compact_kv_cache_ptr.RowBytes(total_num_tiles),
                      compact_kv_cache_ptr.Stride());
        kv_head_ptrs.emplace_back(std::move(kv_ptr));
        total_num_tiles += num_tiles_per_kv_head;
      }
    }
  } else {
    kv_cache = MatStorageT<KV_t>(
        "kv",
        Extents2D(CappedSeqLen(config, inference_args), config.KVCacheCols()),
        allocator, MatPadding::kOdd);
  }
}

KVCache KVCache::Copy() {
  KVCache copy(kv_cache.Extents(), num_layers, kv_heads, qkv_dim, allocator_);

  CopyMat(kv_cache, copy.kv_cache);
  return copy;
}

std::vector<KVCachePtr> ToKVCachePtrs(const hwy::Span<KVCache>& kv_caches) {
  std::vector<KVCachePtr> ptrs;
  ptrs.reserve(kv_caches.size());
  for (size_t i = 0; i < kv_caches.size(); ++i) {
    ptrs.push_back(kv_caches[i].ToPtr());
  }
  return ptrs;
}

}  // namespace gcpp
