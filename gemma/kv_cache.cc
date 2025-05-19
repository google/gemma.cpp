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

#include <algorithm>  // std::copy

#include "gemma/configs.h"
#include "util/mat.h"  // ZeroInit
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"  // ZeroBytes

namespace gcpp {

void KVCache::ZeroGriffinCache() {
  if (griffin_layers == 0) return;
  ZeroInit(conv1d_cache);
  ZeroInit(rglru_cache);
}

static size_t GriffinConv1dCols(const ModelConfig& config) {
  size_t conv1d_width = 0;
  for (const auto& layer_config : config.layer_configs) {
    conv1d_width = HWY_MAX(conv1d_width, layer_config.conv1d_width);
  }
  // The row offset, in blocks of model_dim is computed mod (conv1d_width - 1),
  // hence allocate conv1d_width * model_dim total columns.
  return conv1d_width * config.model_dim;
}

// prefill_tbatch_size is the maximum number of tokens from one query to
// prefill at a time.
KVCache::KVCache(const ModelConfig& config, size_t prefill_tbatch_size)
    : griffin_layers(
          config.NumLayersOfType(LayerAttentionType::kGriffinRecurrentBlock)),
      conv1d_cache("conv1d_cache",
                   Extents2D(griffin_layers, GriffinConv1dCols(config)),
                   MatPadding::kOdd),
      rglru_cache("rglru_cache", Extents2D(griffin_layers, config.model_dim),
                  MatPadding::kOdd) {
  // TODO: move to MatStorageT.
  const size_t size_cache_pos = config.CachePosSize();
  if (size_cache_pos != 0) {
    // Allocate more so that prefill can always access one batch, even if
    // near the end of the sequence.
    seq_len = config.seq_len + prefill_tbatch_size;
    kv_cache = hwy::AllocateAligned<float>(seq_len * size_cache_pos);
  }
}

KVCache KVCache::Copy(const ModelConfig& weights_config,
                      size_t prefill_tbatch_size) {
  KVCache copy(weights_config, prefill_tbatch_size);

  const size_t size_cache_pos = weights_config.CachePosSize();
  if (size_cache_pos != 0) {
    std::copy(kv_cache.get(), kv_cache.get() + size_cache_pos * seq_len,
              copy.kv_cache.get());
  }

  if (conv1d_cache.HasPtr()) {
    CopyMat(conv1d_cache, copy.conv1d_cache);
  }
  if (rglru_cache.HasPtr()) {
    CopyMat(rglru_cache, copy.rglru_cache);
  }

  return copy;
}

}  // namespace gcpp
