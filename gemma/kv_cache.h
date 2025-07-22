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

#include "gemma/configs.h"  // ModelConfig
#include "gemma/gemma_args.h"  // InferenceArgs
#include "util/basics.h"       // BF16
#include "util/mat.h"

namespace gcpp {

using KV_t = float;

struct KVCache {
  KVCache(const ModelConfig& config, const InferenceArgs& inference_args,
          const Allocator& allocator);

  // Returns a deep copy of the KVCache. Use explicit function instead of
  // copy ctor to make the cost explicit.
  KVCache Copy();

  // Zero-initialize the Griffin recurrent block cache, i.e. the conv1d_cache
  // and rglru_cache.
  void ZeroGriffinCache();

  size_t SeqLen() const { return kv_cache.Rows(); }

  // [griffin_layers, griffin_conv1d_cols * model_dim]
  MatStorageT<float> conv1d_cache;
  MatStorageT<float> rglru_cache;  // [griffin_layers, model_dim]

  MatStorageT<KV_t> kv_cache;  // [seq_len, layers * kv_heads * qkv_dim * 2]

 private:
  const Allocator& allocator_;

  // For use by other ctor and Copy()
  KVCache(const Extents2D& conv1d_extents, const Extents2D& rglru_extents,
          const Extents2D& kv_extents, const Allocator& allocator);
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_KV_CACHE_H_
