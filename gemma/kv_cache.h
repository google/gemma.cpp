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

#include "gemma/common.h"  // Model
#include "hwy/aligned_allocator.h"

namespace gcpp {

struct KVCache {
  size_t seq_len = 0;  // = kSeqLen + prefill_tbatch_size

  // seq_len * kGemmaLayers * kKVHeads * kQKVDim * 2
  hwy::AlignedFreeUniquePtr<float[]> kv_cache;

  // (kConv1dWidth - 1) * kModelDim * kGriffinLayers
  hwy::AlignedFreeUniquePtr<float[]> conv1d_cache;
  size_t conv1d_cache_size = 0;

  // kModelDim * kGriffinLayers
  hwy::AlignedFreeUniquePtr<float[]> rglru_cache;
  size_t rglru_cache_size = 0;

  // Zero-initialize the Griffin recurrent block cache, i.e. the conv1d_cache
  // and rglru_cache.
  void ZeroGriffinCache();

  static KVCache Create(const ModelConfig& weights_config,
                        size_t prefill_tbatch_size);
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_KV_CACHE_H_
