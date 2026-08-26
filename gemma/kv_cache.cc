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

static const std::vector<LayerConfig>& KVLayerConfigs(
    const ModelConfig& config) {
  return config.is_encoder_decoder ? config.decoder_layer_configs
                                   : config.layer_configs;
}

static const std::vector<uint32_t>& KVAttentionWindowSizes(
    const ModelConfig& config) {
  return config.is_encoder_decoder ? config.decoder_attention_window_sizes
                                   : config.attention_window_sizes;
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
      allocator_(allocator) {
  layer_flat_offsets.resize(num_layers, 0);
  layer_k_v_offsets.resize(num_layers, 0);
  layer_kv_head_offsets.resize(num_layers, 0);
  rounded_qkv_dims.resize(num_layers, static_cast<uint32_t>(rounded_qkv_dim));
  size_t flat_accum = 0;
  size_t k_v_accum = 0;
  size_t kv_head_accum = 0;
  for (size_t i = 0; i < num_layers; ++i) {
    layer_flat_offsets[i] = static_cast<uint32_t>(flat_accum);
    flat_accum += 2 * kv_heads * qkv_dim;
    layer_k_v_offsets[i] = static_cast<uint32_t>(k_v_accum);
    k_v_accum += kv_heads * rounded_qkv_dim;
    layer_kv_head_offsets[i] = static_cast<uint32_t>(kv_head_accum);
    kv_head_accum += kv_heads;
  }
  // NOTE: k_v_cols is intentionally left at 0 (default). It serves as a
  // sentinel for MaybeReshapeCache: when k_v_cols == cache.Cols(), the reshape
  // fires. The 2-arg constructor path relies on k_v_cols == 0 to skip reshape.
}

// Allocates and zero-initializes the DeepSeek V4 incremental compressor state
// if any layer needs it, and fills the per-layer offset table.
static void InitDSState(const ModelConfig& config, const Allocator& allocator,
                        MatStorageT<float>& ds_state,
                        MatStorageT<float>& ds_state_snapshot,
                        std::vector<uint32_t>& ds_state_offsets) {
  const size_t num_layers = config.layer_configs.size();
  ds_state_offsets.resize(num_layers, 0);
  size_t accum = 0;
  for (size_t i = 0; i < num_layers; ++i) {
    ds_state_offsets[i] = static_cast<uint32_t>(accum);
    accum += config.layer_configs[i].DSStateSize();
  }
  // The MTP block is dense (no compressor state), but give it an offset entry
  // so `ds_state_offsets[num_layers]` is valid.
  if (config.num_mtp_layers > 0) {
    for (size_t i = 0; i < config.num_mtp_layers; ++i) {
      ds_state_offsets.push_back(static_cast<uint32_t>(accum));
    }
  }
  if (accum == 0) return;
  ds_state = MatStorageT<float>("ds_state", Extents2D(1, accum), allocator,
                                MatPadding::kPacked);
  ZeroInit(ds_state);
  // Boundary snapshot for speculative decoding: state after each verified
  // token of a verify step, restored if a draft is rejected.
  ds_state_snapshot = MatStorageT<float>("ds_snap", Extents2D(32, accum),
                                         allocator, MatPadding::kPacked);
  ZeroInit(ds_state_snapshot);
}

// Support heterogeneous layer configurations (common in Gemma 4 architectures),
// where different layers can have varying attention shapes (e.g., mixing local
// layers with smaller qkv_dim/more heads and global layers with larger
// qkv_dim/fewer heads).
//
// Rather than assuming uniform layer sizes, we dynamically compute and store
// cumulative offsets for each layer to allow correct indexing into the
// flattened KV cache.
KVCache::KVCache(const ModelConfig& config, const InferenceArgs& inference_args,
                 const Allocator& allocator)
    : allocator_(allocator) {
  const std::vector<LayerConfig>& kv_layer_configs = KVLayerConfigs(config);

  HWY_ASSERT(!kv_layer_configs.empty());
  num_layers = kv_layer_configs.size();

  // 1. Build non-uniform offset tables dynamically
  layer_flat_offsets.resize(num_layers, 0);
  layer_k_v_offsets.resize(num_layers, 0);
  layer_kv_head_offsets.resize(num_layers, 0);
  rounded_qkv_dims.resize(num_layers, 0);

  size_t flat_accum = 0;
  size_t k_v_accum = 0;
  size_t kv_head_accum = 0;

  for (size_t i = 0; i < num_layers; ++i) {
    layer_flat_offsets[i] = static_cast<uint32_t>(flat_accum);
    flat_accum += kv_layer_configs[i].CacheLayerSize();

    layer_k_v_offsets[i] = static_cast<uint32_t>(k_v_accum);
    size_t rounded_dim =
        hwy::RoundUpTo(kv_layer_configs[i].qkv_dim, kMaxBF16PerVector);
    rounded_qkv_dims[i] = static_cast<uint32_t>(rounded_dim);
    k_v_accum += kv_layer_configs[i].kv_heads * rounded_dim;

    layer_kv_head_offsets[i] = static_cast<uint32_t>(kv_head_accum);
    kv_head_accum += config.layer_configs[i].kv_heads;
  }
  k_v_cols = static_cast<uint32_t>(k_v_accum);

  // Since we also store legacy homogeneous variables, we default them to Layer
  // 0 values.
  kv_heads = kv_layer_configs[0].kv_heads;
  qkv_dim = kv_layer_configs[0].qkv_dim;
  rounded_qkv_dim = hwy::RoundUpTo(qkv_dim, kMaxBF16PerVector);

  const size_t rows = CappedSeqLen(config, inference_args);
  const size_t cols = config.KVCacheCols();
  kv_cache = MatStorageT<KV_t>("kv", Extents2D(rows, cols), allocator,
                               MatPadding::kOdd);
  k_cache = MatStorageT<KV_t>(
      "k", Extents2D(hwy::RoundUpTo(rows, kMaxBF16PerVector), k_v_cols),
      allocator, MatPadding::kPacked);
  v_cache = MatStorageT<KV_t>(
      "v", Extents2D(hwy::RoundUpTo(rows, kMaxBF16PerVector), k_v_cols),
      allocator, MatPadding::kPacked);
  const size_t num_tiles = hwy::DivCeil(rows, kTileSize);
  tiled_seq_len = num_tiles * kTileSize;
  // Trailing segment for the MTP block (indexed as layer `num_layers`).
  if (config.num_mtp_layers > 0) {
    const size_t mtp_size = config.MTPLayerConfig().CacheLayerSize();
    for (size_t i = 0; i < config.num_mtp_layers; ++i) {
      layer_flat_offsets.push_back(static_cast<uint32_t>(flat_accum));
      flat_accum += mtp_size;
    }
  }
  InitDSState(config, allocator, ds_state, ds_state_snapshot, ds_state_offsets);
}

KVCache::KVCache(const ModelConfig& config, const InferenceArgs& inference_args,
                 const RuntimeConfig& runtime_config,
                 const Allocator& allocator)
    : allocator_(allocator) {
  const std::vector<LayerConfig>& kv_layer_configs = KVLayerConfigs(config);
  const std::vector<uint32_t>& kv_attention_window_sizes =
      KVAttentionWindowSizes(config);

  num_layers = kv_layer_configs.size();

  // 1. Build non-uniform offset tables dynamically
  layer_flat_offsets.resize(num_layers, 0);
  layer_k_v_offsets.resize(num_layers, 0);
  layer_kv_head_offsets.resize(num_layers, 0);
  rounded_qkv_dims.resize(num_layers, 0);

  size_t flat_accum = 0;
  size_t k_v_accum = 0;
  size_t kv_head_accum = 0;
  size_t max_qkv_dim = 0;
  size_t max_kv_heads = 0;

  for (size_t i = 0; i < num_layers; ++i) {
    layer_flat_offsets[i] = static_cast<uint32_t>(flat_accum);
    flat_accum += kv_layer_configs[i].CacheLayerSize();

    layer_k_v_offsets[i] = static_cast<uint32_t>(k_v_accum);
    size_t rounded_dim =
        hwy::RoundUpTo(kv_layer_configs[i].qkv_dim, kMaxBF16PerVector);
    rounded_qkv_dims[i] = static_cast<uint32_t>(rounded_dim);
    k_v_accum += kv_layer_configs[i].kv_heads * rounded_dim;

    layer_kv_head_offsets[i] = static_cast<uint32_t>(kv_head_accum);
    kv_head_accum += config.layer_configs[i].kv_heads;

    max_qkv_dim = HWY_MAX(max_qkv_dim, kv_layer_configs[i].qkv_dim);
    max_kv_heads = HWY_MAX(max_kv_heads, kv_layer_configs[i].kv_heads);
  }
  k_v_cols = static_cast<uint32_t>(k_v_accum);

  // Since we also store legacy homogeneous variables (used by tests/old code),
  // we default them to Layer 0 values.
  kv_heads = kv_layer_configs[0].kv_heads;
  qkv_dim = kv_layer_configs[0].qkv_dim;
  rounded_qkv_dim = hwy::RoundUpTo(qkv_dim, kMaxBF16PerVector);

  // clang-format off
  if (runtime_config.attention_impl == AttentionImpl::kFlash ||
      runtime_config.attention_impl == AttentionImpl::kFlashTransposedQs ||
      runtime_config.attention_impl == AttentionImpl::kFlashTransposedQsInt16 ||
      runtime_config.attention_impl == AttentionImpl::kFlashTransposedQsInt8 ||
      runtime_config.attention_impl == AttentionImpl::kFlashTransposedQsBF16 ||
      runtime_config.attention_impl ==  AttentionImpl::kFlashMatrixAccumulation ||
      runtime_config.attention_impl == AttentionImpl::kInt8MatrixAccumulation
      ) {
    // clang-format on
    kv_cache = MatStorageT<KV_t>(
        "kv",
        Extents2D(CappedSeqLen(config, inference_args), config.KVCacheCols()),
        allocator, MatPadding::kOdd);
    k_cache = MatStorageT<KV_t>(
        "k",
        Extents2D(hwy::RoundUpTo(CappedSeqLen(config, inference_args),
                                 kMaxBF16PerVector),
                  k_v_cols),
        allocator, MatPadding::kPacked);
    v_cache = MatStorageT<KV_t>(
        "v",
        Extents2D(hwy::RoundUpTo(CappedSeqLen(config, inference_args),
                                 kMaxBF16PerVector),
                  k_v_cols),
        allocator, MatPadding::kPacked);
    const size_t num_tiles =
        hwy::DivCeil(CappedSeqLen(config, inference_args), kTileSize);
    tiled_seq_len = num_tiles * kTileSize;
    Type kv_cache_type;
    if (runtime_config.attention_impl ==
        AttentionImpl::kFlashMatrixAccumulation) {
      kv_cache_type = runtime_config.kv_cache_type.value_or(Type::kBF16);
    } else if (runtime_config.attention_impl ==
                   AttentionImpl::kFlashTransposedQsBF16
    ) {
      kv_cache_type = runtime_config.kv_cache_type.value_or(Type::kBF16);
    } else if (runtime_config.attention_impl ==
                   AttentionImpl::kFlashTransposedQsInt16 ||
               runtime_config.attention_impl ==
                   AttentionImpl::kFlashTransposedQsInt8 ||
               runtime_config.attention_impl ==
                   AttentionImpl::kInt8MatrixAccumulation) {
      if (runtime_config.kv_cache_type.has_value() &&
          runtime_config.kv_cache_type.value() != Type::kInt8) {
        HWY_WARN(
            "You are have set kv_cache_type to %s, but you are using "
            "an attention implementation which only "
            "supports Int8. kv_cache_type will be set to Int8.",
            TypeName(runtime_config.kv_cache_type.value()));
      }
      kv_cache_type = Type::kInt8;
    } else {
      kv_cache_type = runtime_config.kv_cache_type.value_or(Type::kF32);
    }

    // Allocate tile size using max_qkv_dim to prevent out-of-bounds corruption
    int max_tile_length = 2 * max_qkv_dim * kTileSize;
    if (kv_cache_type == Type::kInt8) {
      // microscaling
      max_tile_length += 2 * sizeof(BF16) * kTileSize;
      if (runtime_config.attention_impl ==
          AttentionImpl::kFlashTransposedQsInt8) {
        // K sums
        max_tile_length += sizeof(int32_t) * kTileSize;
      }
    }
    auto num_tiles_per_head = [](size_t window_size, size_t prefill_tbatch_size,
                                 size_t max_seq_len) {
      return hwy::DivCeil(
          std::min(max_seq_len, window_size + prefill_tbatch_size), kTileSize);
    };

    size_t total_local_num_tiles = 0;
    size_t total_global_num_tiles = 0;
    size_t local_tile_length = 0;
    size_t global_tile_length = 0;

    for (size_t i = 0; i < num_layers; ++i) {
      size_t num_tiles = num_tiles_per_head(kv_attention_window_sizes[i],
                                            runtime_config.prefill_tbatch_size,
                                            config.max_seq_len) *
                         kv_layer_configs[i].kv_heads;

      size_t tile_len = 2 * kv_layer_configs[i].qkv_dim * kTileSize;
      if (kv_cache_type == Type::kInt8) {
        tile_len += 2 * sizeof(BF16) * kTileSize;
        if (runtime_config.attention_impl ==
            AttentionImpl::kFlashTransposedQsInt8) {
          // K sums
          tile_len += sizeof(int32_t) * kTileSize;
        }
      }

      if (kv_attention_window_sizes[i] == config.max_seq_len) {
        total_global_num_tiles += num_tiles;
        global_tile_length = tile_len;
      } else {
        total_local_num_tiles += num_tiles;
        local_tile_length = tile_len;
      }
    }

    if (total_local_num_tiles > 0) {
      Extents2D local_extents(total_local_num_tiles, local_tile_length);
      compact_local_kv_cache_ptr =
          MatPtr("kv_tiled_local", kv_cache_type, local_extents);
      if (runtime_config.attention_impl ==
          AttentionImpl::kFlashMatrixAccumulation) {
        compact_local_kv_cache_ptr.SetLayout(
            MatPtr::Layout::kBF16MatrixAccumulation);
      } else if (runtime_config.attention_impl ==
                 AttentionImpl::kInt8MatrixAccumulation) {
        compact_local_kv_cache_ptr.SetLayout(
            MatPtr::Layout::kInt8MatrixAccumulation);
      }
      compact_local_kv_cache.AllocateFor(compact_local_kv_cache_ptr, allocator,
                                         MatPadding::kPacked);
    }

    if (total_global_num_tiles > 0) {
      Extents2D global_extents(total_global_num_tiles, global_tile_length);
      compact_global_kv_cache_ptr =
          MatPtr("kv_tiled_global", kv_cache_type, global_extents);
      if (runtime_config.attention_impl ==
          AttentionImpl::kFlashMatrixAccumulation) {
        compact_global_kv_cache_ptr.SetLayout(
            MatPtr::Layout::kBF16MatrixAccumulation);
      } else if (runtime_config.attention_impl ==
                 AttentionImpl::kInt8MatrixAccumulation) {
        compact_global_kv_cache_ptr.SetLayout(
            MatPtr::Layout::kInt8MatrixAccumulation);
      }
      compact_global_kv_cache.AllocateFor(compact_global_kv_cache_ptr,
                                          allocator,
                                          MatPadding::kPacked);
    }

    if (compact_global_kv_cache_ptr.HasPtr()) {
      compact_kv_cache_ptr = compact_global_kv_cache_ptr;
    } else {
      compact_kv_cache_ptr = compact_local_kv_cache_ptr;
    }

    size_t local_tiles_processed = 0;
    size_t global_tiles_processed = 0;
    kv_head_ptrs.clear();
    kv_head_ptrs.reserve(kv_head_accum);
    for (size_t i = 0; i < num_layers; ++i) {
      size_t layer_tile_length = 2 * kv_layer_configs[i].qkv_dim * kTileSize;
      if (kv_cache_type == Type::kInt8) {
        layer_tile_length += 2 * sizeof(BF16) * kTileSize;
        if (runtime_config.attention_impl ==
            AttentionImpl::kFlashTransposedQsInt8) {
          // K sums
          layer_tile_length += sizeof(int32_t) * kTileSize;
        }
      }
      bool is_global = kv_attention_window_sizes[i] == config.max_seq_len;
      for (size_t kv = 0; kv < kv_layer_configs[i].kv_heads; ++kv) {
        size_t num_tiles_per_kv_head = num_tiles_per_head(
            kv_attention_window_sizes[i], runtime_config.prefill_tbatch_size,
            config.max_seq_len);
        MatPtr kv_ptr("kv_ptr", kv_cache_type,
                      Extents2D(num_tiles_per_kv_head, layer_tile_length));
        if (is_global) {
          kv_ptr.SetPtr(
              compact_global_kv_cache_ptr.RowBytes(global_tiles_processed),
              compact_global_kv_cache_ptr.Stride());
          global_tiles_processed += num_tiles_per_kv_head;
        } else {
          kv_ptr.SetPtr(
              compact_local_kv_cache_ptr.RowBytes(local_tiles_processed),
              compact_local_kv_cache_ptr.Stride());
          local_tiles_processed += num_tiles_per_kv_head;
        }
        if (runtime_config.attention_impl ==
            AttentionImpl::kFlashMatrixAccumulation) {
          kv_ptr.SetLayout(MatPtr::Layout::kBF16MatrixAccumulation);
        } else if (runtime_config.attention_impl ==
                   AttentionImpl::kInt8MatrixAccumulation) {
          kv_ptr.SetLayout(MatPtr::Layout::kInt8MatrixAccumulation);
        }
        kv_head_ptrs.emplace_back(std::move(kv_ptr));
      }
    }
  } else {
    kv_cache = MatStorageT<KV_t>(
        "kv",
        Extents2D(CappedSeqLen(config, inference_args), config.KVCacheCols()),
        allocator, MatPadding::kOdd);
  }
  if (config.num_mtp_layers > 0) {
    const size_t mtp_size = config.MTPLayerConfig().CacheLayerSize();
    for (size_t i = 0; i < config.num_mtp_layers; ++i) {
      layer_flat_offsets.push_back(static_cast<uint32_t>(flat_accum));
      flat_accum += mtp_size;
    }
  }
  InitDSState(config, allocator, ds_state, ds_state_snapshot, ds_state_offsets);
}

KVCache KVCache::Copy() {
  KVCache copy(kv_cache.Extents(), num_layers, kv_heads, qkv_dim, allocator_);

  CopyMat(kv_cache, copy.kv_cache);
  if (compact_local_kv_cache_ptr.HasPtr()) {
    CopyMat(compact_local_kv_cache_ptr, copy.compact_local_kv_cache_ptr);
  }
  if (compact_global_kv_cache_ptr.HasPtr()) {
    CopyMat(compact_global_kv_cache_ptr, copy.compact_global_kv_cache_ptr);
  }
  copy.compact_kv_cache_ptr = compact_global_kv_cache_ptr.HasPtr()
                                  ? copy.compact_global_kv_cache_ptr
                                  : copy.compact_local_kv_cache_ptr;
  copy.tiled_seq_len = tiled_seq_len;
  if (ds_state.Rows() > 0) {
    copy.ds_state = MatStorageT<float>("ds_state", ds_state.Extents(),
                                       allocator_, MatPadding::kPacked);
    CopyMat(ds_state, copy.ds_state);
    copy.ds_state_snapshot =
        MatStorageT<float>("ds_snap", ds_state_snapshot.Extents(), allocator_,
                           MatPadding::kPacked);
    CopyMat(ds_state_snapshot, copy.ds_state_snapshot);
    copy.ds_state_offsets = ds_state_offsets;
  }
  copy.layer_flat_offsets = layer_flat_offsets;
  copy.layer_k_v_offsets = layer_k_v_offsets;
  copy.rounded_qkv_dims = rounded_qkv_dims;
  copy.layer_kv_head_offsets = layer_kv_head_offsets;
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
