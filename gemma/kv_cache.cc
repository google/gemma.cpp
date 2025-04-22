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

#include <algorithm>

#include "gemma/configs.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"  // ZeroBytes

namespace gcpp {

void KVCache::ZeroGriffinCache() {
  if (conv1d_cache_size != 0) {
    hwy::ZeroBytes(conv1d_cache.get(),
                   conv1d_cache_size * sizeof(conv1d_cache[0]));
  }
  if (rglru_cache_size != 0) {
    hwy::ZeroBytes(rglru_cache.get(),
                   rglru_cache_size * sizeof(rglru_cache[0]));
  }
}

// prefill_tbatch_size is the maximum number of tokens from one query to
// prefill at a time.
KVCache KVCache::Create(const ModelConfig& weights_config,
                        size_t prefill_tbatch_size) {
  KVCache kv_cache = {};

  const size_t size_cache_pos = weights_config.CachePosSize();
  if (size_cache_pos != 0) {
    // Allocate more so that prefill can always access one batch, even if
    // near the end of the sequence.
    kv_cache.seq_len = weights_config.seq_len + prefill_tbatch_size;
    kv_cache.kv_cache =
        hwy::AllocateAligned<float>(kv_cache.seq_len * size_cache_pos);
  }

  const size_t num_griffin_layers = weights_config.NumLayersOfType(
      LayerAttentionType::kGriffinRecurrentBlock);
  // TODO(patrickms): Add query batching support for Griffin.
  if (num_griffin_layers > 0) {
    uint32_t conv1d_width = 0;
    for (const auto& layer_config : weights_config.layer_configs) {
      conv1d_width = std::max(conv1d_width, layer_config.conv1d_width);
    }
    const size_t conv1d_cache_size =
        num_griffin_layers * (conv1d_width == 0 ? 0 : conv1d_width - 1) *
        weights_config.model_dim;
    kv_cache.conv1d_cache_size = conv1d_cache_size;
    if (conv1d_cache_size != 0) {
      kv_cache.conv1d_cache = hwy::AllocateAligned<float>(conv1d_cache_size);
    }

    const size_t rglru_cache_size =
        num_griffin_layers * weights_config.model_dim;
    kv_cache.rglru_cache_size = rglru_cache_size;
    if (rglru_cache_size != 0) {
      kv_cache.rglru_cache = hwy::AllocateAligned<float>(rglru_cache_size);
    }
  }  // num_griffin_layers

  return kv_cache;
}

KVCache KVCache::Copy(const ModelConfig& weights_config,
                      size_t prefill_tbatch_size) {
  KVCache kv_cache_copy = Create(weights_config, prefill_tbatch_size);

  const size_t size_cache_pos = weights_config.CachePosSize();
  if (size_cache_pos != 0) {
    std::copy(kv_cache.get(), kv_cache.get() + size_cache_pos * seq_len,
              kv_cache_copy.kv_cache.get());
  }

  const size_t num_griffin_layers = weights_config.NumLayersOfType(
      LayerAttentionType::kGriffinRecurrentBlock);
  if (num_griffin_layers > 0) {
    if (conv1d_cache_size != 0) {
      std::copy(conv1d_cache.get(), conv1d_cache.get() + conv1d_cache_size,
                kv_cache_copy.conv1d_cache.get());
    }
    if (rglru_cache_size != 0) {
      std::copy(rglru_cache.get(),
                rglru_cache.get() + rglru_cache_size * sizeof(rglru_cache[0]),
                kv_cache_copy.rglru_cache.get());
    }
  }
  return kv_cache_copy;
}

}  // namespace gcpp
