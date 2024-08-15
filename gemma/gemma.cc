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

// Defines Gemma member functions; the actual implementations are in
// gemma-inl.h, included from instantiations/*.cc.

#include "gemma/gemma.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <utility>  // std::move

#include "compression/io.h"  // Path
#include "gemma/common.h"
#include "gemma/weights.h"
#include "util/threading.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/highway.h"

namespace gcpp {

Gemma::Gemma(const Path& tokenizer_path, const Path& weights,
             const ModelInfo& info, PerClusterPools& pools)
    : pools_(pools), tokenizer_(tokenizer_path), info_(info) {
  weights_u8_ =
      LoadCompressedWeights(weights, info.model, info.weight, pools_.Inner(0));
}

Gemma::Gemma(GemmaTokenizer&& tokenizer, const ModelInfo& info,
             PerClusterPools& pools)
    : pools_(pools), tokenizer_(std::move(tokenizer)), info_(info) {
  HWY_ASSERT(info.weight == Type::kF32);
  weights_u8_ = CallForModel<float, AllocateCompressedWeights>(info.model,
                                                               pools_.Inner(0));
}

Gemma::~Gemma() {
  CallForModelAndWeight<DeleteCompressedWeights>(info_.model, info_.weight,
                                                 weights_u8_);
}

// There are >100 instantiations of the inference code. To reduce compile time,
// we shard them across multiple translation units in instantiations/*.cc.
// This declares the functions defined there. We use overloading because
// explicit instantiations are still too slow to compile.
#define GEMMA_DECLARE(CONFIGT, TWEIGHT)                                        \
  extern void GenerateSingle(CONFIGT<TWEIGHT>, const ByteStorageT& weights_u8, \
                             const RuntimeConfig& runtime_config,              \
                             const PromptTokens& prompt, size_t pos,           \
                             KVCache& kv_cache, PerClusterPools& pools,        \
                             TimingInfo& timing_info);                         \
  extern void GenerateBatch(                                                   \
      CONFIGT<TWEIGHT>, const ByteStorageT& weights_u8,                        \
      const RuntimeConfig& runtime_config, const QueriesPromptTokens& prompts, \
      const QueriesPos& queries_pos, const KVCaches& kv_caches,                \
      PerClusterPools& pools, TimingInfo& timing_info);
GEMMA_FOREACH_CONFIG_AND_WEIGHT(GEMMA_DECLARE);

// Adapters to select from the above overloads via CallForModelAndWeight.
template <class TConfig>
struct GenerateSingleT {
  void operator()(const ByteStorageT& weights_u8,
                  const RuntimeConfig& runtime_config,
                  const PromptTokens& prompt, size_t pos, KVCache& kv_cache,
                  PerClusterPools& pools, TimingInfo& timing_info) const {
    GenerateSingle(TConfig(), weights_u8, runtime_config, prompt, pos, kv_cache,
                   pools, timing_info);
  }
};

template <class TConfig>
struct GenerateBatchT {
  void operator()(const ByteStorageT& weights_u8,
                  const RuntimeConfig& runtime_config,
                  const QueriesPromptTokens& queries_prompt,
                  const QueriesPos& queries_pos, const KVCaches& kv_caches,
                  PerClusterPools& pools, TimingInfo& timing_info) const {
    GenerateBatch(TConfig(), weights_u8, runtime_config, queries_prompt,
                  queries_pos, kv_caches, pools, timing_info);
  }
};

void Gemma::Generate(const RuntimeConfig& runtime_config,
                     const PromptTokens& prompt, size_t pos, KVCache& kv_cache,
                     TimingInfo& timing_info) {
  pools_.StartSpinning();

  CallForModelAndWeight<GenerateSingleT>(info_.model, info_.weight, weights_u8_,
                                         runtime_config, prompt, pos, kv_cache,
                                         pools_, timing_info);

  pools_.StopSpinning();
}

void Gemma::GenerateBatch(const RuntimeConfig& runtime_config,
                          const QueriesPromptTokens& queries_prompt,
                          const QueriesPos& queries_pos,
                          const KVCaches& kv_caches, TimingInfo& timing_info) {
  pools_.StartSpinning();

  CallForModelAndWeight<GenerateBatchT>(
      info_.model, info_.weight, weights_u8_, runtime_config, queries_prompt,
      queries_pos, kv_caches, pools_, timing_info);

  pools_.StopSpinning();
}

}  // namespace gcpp
