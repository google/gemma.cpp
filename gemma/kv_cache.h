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

#include <vector>

#include "gemma/configs.h"     // ModelConfig
#include "gemma/gemma_args.h"  // InferenceArgs
#include "hwy/base.h"          // Divisor
#include "util/basics.h"       // BF16
#include "util/mat.h"

namespace gcpp {

using KV_t = float;

struct KVCache {
  KVCache(const ModelConfig& config, const InferenceArgs& inference_args,
          const Allocator& allocator);

  KVCache(KVCache&&) = default;
  KVCache(const KVCache&) = delete;
  KVCache& operator=(const KVCache&) = delete;

  // Returns an independent snapshot, including the current ring capacities.
  KVCache Copy() const;

  // Logical context limit; independent of each layer's physical ring size.
  size_t SeqLen() const { return seq_len_; }

  MatStorageT<KV_t>& LayerCache(size_t layer_idx) {
    return layers_[layer_idx].cache;
  }
  const MatStorageT<KV_t>& LayerCache(size_t layer_idx) const {
    return layers_[layer_idx].cache;
  }

  KV_t* Row(size_t layer_idx, size_t pos) {
    auto& layer = layers_[layer_idx];
    return layer.cache.Row(layer.div_rows.Remainder(pos));
  }
  const KV_t* Row(size_t layer_idx, size_t pos) const {
    const auto& layer = layers_[layer_idx];
    return layer.cache.Row(layer.div_rows.Remainder(pos));
  }

  // Called before writing an entire token batch. Runtime batch sizes can be
  // larger than the size used to construct the cache. Grow without discarding
  // the preceding history that early queries in the batch still need.
  void PrepareLayer(size_t layer_idx, size_t num_tokens, size_t pos);

  // KV buffer capacity, including row padding (not process RSS).
  size_t AllocatedBytes() const;

 private:
  struct LayerStorage {
    LayerStorage(size_t window, size_t rows, size_t cols,
                 const Allocator& allocator)
        : window(window),
          cache("kv", Extents2D(rows, cols), allocator, MatPadding::kOdd),
          div_rows(static_cast<uint32_t>(rows)) {}

    size_t window;
    MatStorageT<KV_t> cache;
    hwy::Divisor div_rows;
  };

  KVCache(size_t seq_len, const Allocator& allocator)
      : seq_len_(seq_len), allocator_(allocator) {}

  size_t seq_len_;
  const Allocator& allocator_;
  std::vector<LayerStorage> layers_;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_KV_CACHE_H_
