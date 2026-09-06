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

#include <utility>

#include "gemma/configs.h"
#include "gemma/gemma_args.h"
#include "hwy/base.h"  // HWY_MAX
#include "util/mat.h"  // CopyMat

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

// ComputeQKV writes a whole batch before attention reads it. In a local
// layer, its first query can need window - 1 preceding tokens in addition
// to all num_tokens newly written rows. A ring of only window rows is unsafe.
static size_t LayerRows(size_t seq_len, size_t window, size_t num_tokens) {
  HWY_ASSERT(window != 0);
  return HWY_MIN(seq_len, window - 1 + HWY_MAX(size_t{1}, num_tokens));
}

KVCache::KVCache(const ModelConfig& config, const InferenceArgs& inference_args,
                 const Allocator& allocator)
    : KVCache(CappedSeqLen(config, inference_args), allocator) {
  HWY_ASSERT(seq_len_ != 0);
  HWY_ASSERT(config.attention_window_sizes.size() ==
             config.layer_configs.size());
  layers_.reserve(config.layer_configs.size());
  for (size_t i = 0; i < config.layer_configs.size(); ++i) {
    const size_t window = config.attention_window_sizes[i];
    layers_.emplace_back(
        window, LayerRows(seq_len_, window, inference_args.prefill_tbatch_size),
        config.layer_configs[i].CacheLayerSize(), allocator_);
  }
}

void KVCache::PrepareLayer(size_t layer_idx, size_t num_tokens, size_t pos) {
  auto& layer = layers_[layer_idx];
  const size_t rows = LayerRows(seq_len_, layer.window, num_tokens);
  if (rows <= layer.cache.Rows()) return;

  MatStorageT<KV_t> grown("kv", Extents2D(rows, layer.cache.Cols()), allocator_,
                          MatPadding::kOdd);
  const hwy::Divisor div_rows(static_cast<uint32_t>(rows));
  const size_t first = pos - HWY_MIN(pos, layer.cache.Rows());
  for (size_t p = first; p < pos; ++p) {
    hwy::CopyBytes(layer.cache.Row(layer.div_rows.Remainder(p)),
                   grown.Row(div_rows.Remainder(p)),
                   layer.cache.Cols() * sizeof(KV_t));
  }
  layer.cache = std::move(grown);
  layer.div_rows = div_rows;
}

size_t KVCache::AllocatedBytes() const {
  size_t bytes = 0;
  for (const auto& layer : layers_) {
    bytes += layer.cache.Rows() * layer.cache.Stride() * sizeof(KV_t);
  }
  return bytes;
}

KVCache KVCache::Copy() const {
  KVCache copy(seq_len_, allocator_);
  copy.layers_.reserve(layers_.size());
  for (const auto& layer : layers_) {
    copy.layers_.emplace_back(layer.window, layer.cache.Rows(),
                              layer.cache.Cols(), allocator_);
    CopyMat(layer.cache, copy.layers_.back().cache);
  }
  return copy;
}

}  // namespace gcpp
