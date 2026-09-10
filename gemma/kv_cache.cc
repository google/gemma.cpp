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

static std::optional<Type> KVCacheType(const InferenceArgs& inference_args) {
  const auto& name = inference_args.kv_cache_type;
  if (name.empty()) return std::nullopt;
  if (name == "int8" || name == "i8") return Type::kInt8;
  if (name == "bf16") return Type::kBF16;
  if (name == "f32" || name == "float") return Type::kF32;
  HWY_ABORT("Unknown kv_cache_type: %s", name.c_str());
}

KVCache::KVCache(const ModelConfig& config, const InferenceArgs& inference_args,
                 const Allocator& allocator)
    : KVCache(config, inference_args,
              GetAttentionImpl(inference_args.attention_impl), allocator) {}

KVCache::KVCache(const ModelConfig& config, const InferenceArgs& inference_args,
                 AttentionImpl attention_impl, const Allocator& allocator,
                 std::optional<Type> kv_cache_type)
    : seq_len_(CappedSeqLen(config, inference_args)),
      attention_impl_(attention_impl),
      allocator_(allocator) {
  const auto& layers = KVLayerConfigs(config);
  const auto& windows = KVAttentionWindowSizes(config);
  HWY_ASSERT(!layers.empty() && seq_len_ != 0);
  HWY_ASSERT(windows.size() == layers.size());
  num_layers = layers.size();
  kv_heads = layers[0].kv_heads;
  qkv_dim = layers[0].qkv_dim;
  layer_flat_offsets.resize(num_layers);
  layer_kv_head_offsets.resize(num_layers);
  layer_heads_.resize(num_layers);

  size_t flat_cols = 0;
  for (size_t i = 0; i < num_layers; ++i) {
    const auto& layer = layers[i];
    layer_heads_[i] = layer.kv_heads;
    if (!layer.HasOwnKVCache()) {
      const size_t source = static_cast<size_t>(layer.kv_share_layer_idx);
      HWY_ASSERT(source < i);
      HWY_ASSERT(layer.kv_heads == layers[source].kv_heads &&
                 layer.qkv_dim == layers[source].qkv_dim);
      layer_flat_offsets[i] = layer_flat_offsets[source];
      layer_kv_head_offsets[i] = layer_kv_head_offsets[source];
      for (size_t h = 0; h < layer.kv_heads; ++h) {
        auto& window = head_windows_[layer_kv_head_offsets[i] + h];
        window = HWY_MAX(window, windows[i]);
      }
      continue;
    }
    layer_flat_offsets[i] = static_cast<uint32_t>(flat_cols);
    flat_cols += layer.CacheLayerSize();
    layer_kv_head_offsets[i] = static_cast<uint32_t>(head_windows_.size());
    for (size_t h = 0; h < layer.kv_heads; ++h) {
      HWY_ASSERT(windows[i] != 0);
      head_windows_.push_back(windows[i]);
      head_dims_.push_back(layer.qkv_dim);
    }
  }

  // DeepSeek, recurrent, and encoder/decoder attention still consume flat KV.
  const bool needs_flat =
      config.is_encoder_decoder || config.num_mtp_layers > 0 ||
      std::any_of(layers.begin(), layers.end(), [](const LayerConfig& layer) {
        return layer.type != LayerAttentionType::kGemma;
      });
  if (needs_flat) {
    kv_cache =
        MatStorageT<KV_t>("kv", Extents2D(seq_len_, config.KVCacheCols()),
                          allocator, MatPadding::kOdd);
  }
  for (size_t i = 0; i < config.num_mtp_layers; ++i) {
    layer_flat_offsets.push_back(static_cast<uint32_t>(flat_cols));
    flat_cols += config.MTPLayerConfig().CacheLayerSize();
  }
  InitDSState(config, allocator, ds_state, ds_state_snapshot, ds_state_offsets);
  if (config.is_encoder_decoder ||
      std::none_of(layers.begin(), layers.end(), [](const LayerConfig& layer) {
        return layer.type == LayerAttentionType::kGemma;
      })) {
    return;
  }

  if (!kv_cache_type.has_value()) kv_cache_type = KVCacheType(inference_args);
  Type type = kv_cache_type.value_or(Type::kF32);
  MatPtr::Layout layout = MatPtr::Layout::kFlat;
  if (attention_impl == AttentionImpl::kFlash ||
      attention_impl == AttentionImpl::kFlashTransposedQsBF16 ||
      attention_impl == AttentionImpl::kFlashMatrixAccumulation) {
    type = kv_cache_type.value_or(Type::kBF16);
  }
  if (attention_impl == AttentionImpl::kFlash) type = Type::kBF16;
  if (attention_impl == AttentionImpl::kFlashTransposedQsInt16 ||
      attention_impl == AttentionImpl::kFlashTransposedQsInt8 ||
      attention_impl == AttentionImpl::kInt8MatrixAccumulation) {
    if (kv_cache_type.has_value() && *kv_cache_type != Type::kInt8) {
      HWY_WARN("This attention implementation requires an Int8 KV cache.");
    }
    type = Type::kInt8;
  }
  if (attention_impl == AttentionImpl::kFlashMatrixAccumulation) {
    layout = MatPtr::Layout::kBF16MatrixAccumulation;
  } else if (attention_impl == AttentionImpl::kInt8MatrixAccumulation) {
    layout = MatPtr::Layout::kInt8MatrixAccumulation;
  }
  kv_head_owners_.resize(head_windows_.size());
  kv_head_ptrs.reserve(head_windows_.size());
  for (size_t h = 0; h < head_windows_.size(); ++h) {
    const bool flash = attention_impl == AttentionImpl::kFlash;
    const size_t tile = flash ? kMaxBF16PerVector : kTileSize;
    const size_t dim =
        flash ? hwy::RoundUpTo(head_dims_[h], tile) : head_dims_[h];
    size_t cols = 2 * dim * tile;
    if (type == Type::kInt8) {
      cols += 2 * sizeof(BF16) * tile;
      if (attention_impl == AttentionImpl::kFlashTransposedQsInt8) {
        cols += sizeof(int32_t) * tile;
      }
    }
    // Include the full batch and trailing padding before rounding, so writes
    // cannot wrap onto the oldest token still visible to this batch.
    const size_t wanted = HWY_MIN(
        seq_len_, head_windows_[h] - 1 +
                      HWY_MAX(size_t{1}, inference_args.prefill_tbatch_size) +
                      tile - 1);
    MatPtr ptr("kv_head", type, Extents2D(hwy::DivCeil(wanted, tile), cols));
    ptr.SetLayout(layout);
    kv_head_owners_[h].AllocateFor(ptr, allocator, MatPadding::kPacked);
    kv_head_ptrs.push_back(ptr);
  }
}

size_t KVCache::LayerCapacity(size_t layer) const {
  const size_t tile =
      attention_impl_ == AttentionImpl::kFlash
          ? (flash_tile_size_ ? flash_tile_size_ : kMaxBF16PerVector)
          : kTileSize;
  return kv_head_ptrs[layer_kv_head_offsets[layer]].Rows() * tile;
}

void KVCache::PrepareLayer(size_t layer, size_t num_tokens, size_t pos,
                           size_t tile_size) {
  HWY_ASSERT(attention_impl_ == AttentionImpl::kFlash);
  HWY_ASSERT(pos <= seq_len_ && num_tokens <= seq_len_ - pos);
  HWY_ASSERT(tile_size != 0 && kMaxBF16PerVector % tile_size == 0);
  if (flash_tile_size_ == 0) {
    // Allocation is independent of the dispatched SIMD target. No values have
    // been stored yet, so split allocation tiles into native K/V tiles once.
    for (auto& ptr : kv_head_ptrs) {
      const size_t factor = kMaxBF16PerVector / tile_size;
      MatPtr reshaped("kv_head", ptr.GetType(),
                      Extents2D(ptr.Rows() * factor, ptr.Cols() / factor));
      reshaped.SetPtr(ptr.RowBytes(0), reshaped.Cols());
      ptr = reshaped;
    }
    flash_tile_size_ = tile_size;
  }
  HWY_ASSERT(flash_tile_size_ == tile_size);
  for (size_t h = 0; h < layer_heads_[layer]; ++h) {
    const size_t head = layer_kv_head_offsets[layer] + h;
    const size_t wanted =
        HWY_MIN(seq_len_, head_windows_[head] - 1 +
                              HWY_MAX(size_t{1}, num_tokens) + tile_size - 1);
    const size_t rows = hwy::DivCeil(wanted, tile_size);
    if (rows > kv_head_ptrs[head].Rows()) ResizeHead(head, rows, pos);
  }
}

void KVCache::ResizeHead(size_t head, size_t rows, size_t pos) {
  auto& old = kv_head_ptrs[head];
  MatPtr next("kv_head", old.GetType(), Extents2D(rows, old.Cols()));
  next.SetLayout(old.GetLayout());
  MatOwner owner;
  owner.AllocateFor(next, allocator_, MatPadding::kPacked);
  const size_t first = pos - HWY_MIN(pos, head_windows_[head] - 1);
  for (size_t t = first / flash_tile_size_;
       t < hwy::DivCeil(pos, flash_tile_size_); ++t) {
    hwy::CopyBytes(old.RowBytes(t % old.Rows()), next.RowBytes(t % rows),
                   old.Cols() * old.ElementBytes());
  }
  old = next;
  kv_head_owners_[head] = std::move(owner);
}

MatPtrT<KV_t> KVCache::FlashK(size_t layer, size_t head) const {
  HWY_DASSERT(attention_impl_ == AttentionImpl::kFlash &&
              flash_tile_size_ != 0);
  MatPtrT<KV_t> view(kv_head_ptrs[layer_kv_head_offsets[layer] + head]);
  view.OverrideCols(view.Cols() / 2);
  return view;
}

MatPtrT<KV_t> KVCache::FlashV(size_t layer, size_t head) const {
  auto view = FlashK(layer, head);
  view.SetPtr(view.Row(0) + view.Cols(), view.Stride());
  return view;
}

void KVCache::Clear() {
  if (kv_cache.HasPtr()) ZeroInit(kv_cache);
  for (auto& ptr : kv_head_ptrs) ZeroInit(ptr);
  if (ds_state.HasPtr()) ZeroInit(ds_state);
  if (ds_state_snapshot.HasPtr()) ZeroInit(ds_state_snapshot);
}

size_t KVCache::AllocatedBytes() const {
  const auto bytes = [](const MatPtr& mat) {
    return mat.Rows() * mat.Stride() * mat.ElementBytes();
  };
  size_t total = bytes(kv_cache) + bytes(ds_state) + bytes(ds_state_snapshot);
  for (const auto& ptr : kv_head_ptrs) total += bytes(ptr);
  return total;
}

KVCache KVCache::Copy() const {
  KVCache copy(allocator_);
  copy.seq_len_ = seq_len_;
  copy.attention_impl_ = attention_impl_;
  copy.flash_tile_size_ = flash_tile_size_;
  copy.num_layers = num_layers;
  copy.kv_heads = kv_heads;
  copy.qkv_dim = qkv_dim;
  copy.layer_flat_offsets = layer_flat_offsets;
  copy.layer_kv_head_offsets = layer_kv_head_offsets;
  copy.layer_heads_ = layer_heads_;
  copy.head_windows_ = head_windows_;
  copy.head_dims_ = head_dims_;
  copy.kv_head_ptrs = kv_head_ptrs;
  copy.kv_head_owners_.resize(kv_head_ptrs.size());
  for (size_t h = 0; h < kv_head_ptrs.size(); ++h) {
    copy.kv_head_owners_[h].AllocateFor(copy.kv_head_ptrs[h], allocator_,
                                        MatPadding::kPacked);
    CopyMat(kv_head_ptrs[h], copy.kv_head_ptrs[h]);
  }
  if (kv_cache.HasPtr()) {
    copy.kv_cache = MatStorageT<KV_t>("kv", kv_cache.Extents(), allocator_,
                                      MatPadding::kOdd);
    CopyMat(kv_cache, copy.kv_cache);
  }
  if (ds_state.HasPtr()) {
    copy.ds_state = MatStorageT<float>("ds_state", ds_state.Extents(),
                                       allocator_, MatPadding::kPacked);
    CopyMat(ds_state, copy.ds_state);
    copy.ds_state_snapshot =
        MatStorageT<float>("ds_snap", ds_state_snapshot.Extents(), allocator_,
                           MatPadding::kPacked);
    CopyMat(ds_state_snapshot, copy.ds_state_snapshot);
  }
  copy.ds_state_offsets = ds_state_offsets;
  return copy;
}

}  // namespace gcpp
