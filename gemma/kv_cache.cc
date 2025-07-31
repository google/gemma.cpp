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

#include "gemma/configs.h"
#include "gemma/gemma_args.h"
#include "util/mat.h"  // ZeroInit
#include "hwy/base.h"    // HWY_MAX

namespace gcpp {

void KVCache::ZeroGriffinCache() {
  if (conv1d_cache.Rows() == 0) return;
  ZeroInit(conv1d_cache);
  ZeroInit(rglru_cache);
}

static size_t GriffinLayers(const ModelConfig& config) {
  return config.NumLayersOfType(LayerAttentionType::kGriffinRecurrentBlock);
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

KVCache::KVCache(const Extents2D& conv1d_extents,
                 const Extents2D& rglru_extents, const Extents2D& kv_extents,
                 const Allocator& allocator)
    : conv1d_cache("conv1d_cache", conv1d_extents, allocator, MatPadding::kOdd),
      rglru_cache("rglru_cache", rglru_extents, allocator, MatPadding::kOdd),
      kv_cache("kv", kv_extents, allocator, MatPadding::kOdd),
      allocator_(allocator) {}

KVCache::KVCache(const ModelConfig& config, const InferenceArgs& inference_args,
                 const Allocator& allocator)
    : KVCache(
          Extents2D(GriffinLayers(config), GriffinConv1dCols(config)),
          Extents2D(GriffinLayers(config), config.model_dim),
          Extents2D(CappedSeqLen(config, inference_args), config.KVCacheCols()),
          allocator) {}

KVCache KVCache::Copy() {
  KVCache copy(conv1d_cache.Extents(), rglru_cache.Extents(),
               kv_cache.Extents(), allocator_);

  if (conv1d_cache.Rows() != 0) {
    CopyMat(conv1d_cache, copy.conv1d_cache);
    CopyMat(rglru_cache, copy.rglru_cache);
  }

  CopyMat(kv_cache, copy.kv_cache);

  return copy;
}

}  // namespace gcpp
