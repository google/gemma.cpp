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

#include <string>
#include <utility>  // std::move
#include <vector>

#include "compression/io.h"  // Path
#include "compression/shared.h"
#include "gemma/common.h"
#include "gemma/weights.h"
#include "ops/ops-inl.h"
#include "paligemma/image.h"
#include "util/threading.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/highway.h"

namespace gcpp {

Gemma::Gemma(const Path& tokenizer_path, const Path& weights,
             const ModelInfo& info, NestedPools& pools)
    : env_(pools), tokenizer_(tokenizer_path) {
  model_.Load(weights, info.model, info.weight, info.wrapping,
              env_.parallel.Pools().Pool(0),
              /*tokenizer_proto=*/nullptr);
}

Gemma::Gemma(const Path& weights, NestedPools& pools) : env_(pools) {
  std::string tokenizer_proto;
  model_.Load(weights, Model::UNKNOWN, Type::kUnknown, PromptWrapping::GEMMA_IT,
              env_.parallel.Pools().Pool(0), &tokenizer_proto);
  tokenizer_.Deserialize(tokenizer_proto);
}

Gemma::Gemma(GemmaTokenizer&& tokenizer, const ModelInfo& info,
             NestedPools& pools)
    : env_(pools), tokenizer_(std::move(tokenizer)) {
  HWY_ASSERT(info.weight == Type::kF32);
  model_.Allocate(info.model, info.weight, env_.parallel.Pools().Pool(0));
}

Gemma::~Gemma() {
}

// There are >=3 types of the inference code. To reduce compile time,
// we shard them across multiple translation units in instantiations/*.cc.
// This declares the functions defined there. We use overloading because
// explicit instantiations are still too slow to compile.
#define GEMMA_DECLARE(TWEIGHT)                                                 \
  extern void GenerateSingle(TWEIGHT, const ModelWeightsStorage& model,        \
                             const RuntimeConfig& runtime_config,              \
                             const PromptTokens& prompt, size_t pos,           \
                             size_t prefix_end, KVCache& kv_cache,             \
                             MatMulEnv* env, TimingInfo& timing_info);         \
  extern void GenerateBatch(                                                   \
      TWEIGHT, const ModelWeightsStorage& model,                               \
      const RuntimeConfig& runtime_config, const QueriesPromptTokens& prompts, \
      const QueriesPos& queries_pos, const QueriesPos& queries_prefix_end,     \
      const KVCaches& kv_caches, MatMulEnv* env, TimingInfo& timing_info);     \
  extern void GenerateImageTokens(TWEIGHT, const ModelWeightsStorage& model,   \
                                  const RuntimeConfig& runtime_config,         \
                                  const Image& image,                          \
                                  ImageTokens& image_tokens, MatMulEnv* env);
GEMMA_DECLARE(float)
GEMMA_DECLARE(BF16)
GEMMA_DECLARE(NuqStream)
GEMMA_DECLARE(SfpStream)

// Adapters to select from the above overloads via CallForModelWeight.
template <class TConfig>
struct GenerateSingleT {
  void operator()(const ModelWeightsStorage& model,
                  const RuntimeConfig& runtime_config,
                  const PromptTokens& prompt, size_t pos, size_t prefix_end,
                  KVCache& kv_cache, MatMulEnv* env,
                  TimingInfo& timing_info) const {
    GenerateSingle(TConfig(), model, runtime_config, prompt, pos, prefix_end,
                   kv_cache, env, timing_info);
  }
};

template <class TConfig>
struct GenerateBatchT {
  void operator()(const ModelWeightsStorage& model,
                  const RuntimeConfig& runtime_config,
                  const QueriesPromptTokens& queries_prompt,
                  const QueriesPos& queries_pos,
                  const QueriesPos& queries_prefix_end,
                  const KVCaches& kv_caches, MatMulEnv* env,
                  TimingInfo& timing_info) const {
    GenerateBatch(TConfig(), model, runtime_config, queries_prompt, queries_pos,
                  queries_prefix_end, kv_caches, env, timing_info);
  }
};

template <class TConfig>
struct GenerateImageTokensT {
  void operator()(const ModelWeightsStorage& model,
                  const RuntimeConfig& runtime_config, const Image& image,
                  ImageTokens& image_tokens, MatMulEnv* env) const {
    GenerateImageTokens(TConfig(), model, runtime_config, image, image_tokens,
                        env);
  }
};

void Gemma::Generate(const RuntimeConfig& runtime_config,
                     const PromptTokens& prompt, size_t pos, size_t prefix_end,
                     KVCache& kv_cache, TimingInfo& timing_info) {
  env_.parallel.Pools().MaybeStartSpinning(runtime_config.use_spinning);

  model_.CallForModelWeight<GenerateSingleT>(
      runtime_config, prompt, pos, prefix_end, kv_cache, &env_, timing_info);

  env_.parallel.Pools().MaybeStopSpinning(runtime_config.use_spinning);
}

void Gemma::GenerateBatch(const RuntimeConfig& runtime_config,
                          const QueriesPromptTokens& queries_prompt,
                          const QueriesPos& queries_pos,
                          const QueriesPos& queries_prefix_end,
                          const KVCaches& kv_caches, TimingInfo& timing_info) {
  // If we did not get passed prefix ends (size 0), assume 0 and pass that on.
  QueriesPos mutable_queries_prefix_end = queries_prefix_end;
  std::vector<size_t> prefix_end_vec;
  if (queries_prefix_end.size() == 0) {
    prefix_end_vec.resize(queries_prompt.size(), 0);
    mutable_queries_prefix_end =
        QueriesPos(prefix_end_vec.data(), prefix_end_vec.size());
  }

  env_.parallel.Pools().MaybeStartSpinning(runtime_config.use_spinning);

  model_.CallForModelWeight<GenerateBatchT>(
      runtime_config, queries_prompt, queries_pos, mutable_queries_prefix_end,
      kv_caches, &env_, timing_info);

  env_.parallel.Pools().MaybeStopSpinning(runtime_config.use_spinning);
}

void Gemma::GenerateImageTokens(const RuntimeConfig& runtime_config,
                                const Image& image, ImageTokens& image_tokens) {
  env_.parallel.Pools().MaybeStartSpinning(runtime_config.use_spinning);

  model_.CallForModelWeight<GenerateImageTokensT>(runtime_config, image,
                                                  image_tokens, &env_);

  env_.parallel.Pools().MaybeStopSpinning(runtime_config.use_spinning);
}

// Non-template functions moved from gemma-inl.h to avoid ODR violations.

void RangeChecks(const ModelConfig& weights_config,
                 size_t& max_generated_tokens, const size_t prompt_size) {
  if (!weights_config.use_local_attention) {
    if (max_generated_tokens > weights_config.seq_len) {
      fprintf(stderr,
              "WARNING: max_generated_tokens %zu > kSeqLen %u, truncating.\n",
              max_generated_tokens, weights_config.seq_len);
      max_generated_tokens = weights_config.seq_len;
    }
  }
  HWY_ASSERT(prompt_size > 0);
}

}  // namespace gcpp
