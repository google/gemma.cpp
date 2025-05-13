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

#include <memory>
#include <utility>  // std::move
#include <vector>

// Placeholder for internal header, do not modify.
#include "compression/types.h"
#include "gemma/configs.h"
#include "gemma/model_store.h"
#include "gemma/tokenizer.h"
#include "gemma/weights.h"
#include "io/blob_store.h"
#include "io/io.h"  // Path
#include "ops/matmul.h"
#include "paligemma/image.h"
#include "util/threading_context.h"
#include "hwy/base.h"

namespace gcpp {

// Internal init must run before I/O. This helper function takes care of that,
// plus calling `SetArgs`.
MatMulEnv MakeMatMulEnv(const ThreadingArgs& threading_args) {
  // Placeholder for internal init, do not modify.

  ThreadingContext::SetArgs(threading_args);
  return MatMulEnv(ThreadingContext::Get());
}

Gemma::Gemma(const LoaderArgs& loader, MatMulEnv& env)
    : env_(env),
      reader_(BlobReader::Make(loader.weights, loader.map)),
      model_(*reader_, loader.tokenizer, loader.wrapping),
      weights_(model_.Config().weight),
      chat_template_(model_.Tokenizer(), model_.Config().model) {
  weights_.ReadFromBlobs(model_, *reader_, env_.ctx.pools.Pool());
  reader_.reset();
}

Gemma::Gemma(const ModelConfig& config, GemmaTokenizer&& tokenizer,
             MatMulEnv& env)
    : env_(env),
      model_(config, std::move(tokenizer)),
      weights_(config.weight),
      chat_template_(model_.Tokenizer(), model_.Config().model) {
  HWY_ASSERT(config.weight == Type::kF32);
  weights_.AllocateForTest(config, env_.ctx.pools.Pool(0));
}

Gemma::~Gemma() = default;

void Gemma::Save(const Path& weights_path, hwy::ThreadPool& pool) const {
  BlobWriter writer;
  const std::vector<uint32_t> serialized_mat_ptrs =
      weights_.AddTensorDataToWriter(writer);
  WriteSingleFile(model_.Config(), model_.Tokenizer(), serialized_mat_ptrs,
                  writer, env_.ctx.pools.Pool(), weights_path);
}

// There are >=3 types of the inference code. To reduce compile time,
// we shard them across multiple translation units in instantiations/*.cc.
// This declares the functions defined there. We use overloading because
// explicit instantiations are still too slow to compile.
// TODO: we want to move toward type-erasing, where we check the tensor type at
// each usage. Then we would have a single function, passing `WeightsOwner`
// instead of `WeightsPtrs<T>`.
#define GEMMA_DECLARE(WEIGHT_TYPE)                                             \
  extern void GenerateSingle(                                                  \
      const ModelStore& model, const ModelWeightsPtrs<WEIGHT_TYPE>& weights,  \
      const RuntimeConfig& runtime_config, const PromptTokens& prompt,         \
      size_t pos, size_t prefix_end, KVCache& kv_cache, MatMulEnv* env,        \
      TimingInfo& timing_info);                                                \
  extern void GenerateBatch(                                                   \
      const ModelStore& model, const ModelWeightsPtrs<WEIGHT_TYPE>& weights,  \
      const RuntimeConfig& runtime_config, const QueriesPromptTokens& prompts, \
      const QueriesPos& queries_pos, const QueriesPos& queries_prefix_end,     \
      const KVCaches& kv_caches, MatMulEnv* env, TimingInfo& timing_info);     \
  extern void GenerateImageTokens(                                             \
      const ModelStore& model, const ModelWeightsPtrs<WEIGHT_TYPE>& weights,  \
      const RuntimeConfig& runtime_config, const Image& image,                 \
      ImageTokens& image_tokens, MatMulEnv* env);
GEMMA_DECLARE(float)
GEMMA_DECLARE(BF16)
GEMMA_DECLARE(NuqStream)
GEMMA_DECLARE(SfpStream)

void Gemma::Generate(const RuntimeConfig& runtime_config,
                     const PromptTokens& prompt, size_t pos, size_t prefix_end,
                     KVCache& kv_cache, TimingInfo& timing_info) const {
  env_.ctx.pools.MaybeStartSpinning(runtime_config.use_spinning);

  weights_.CallT([&](auto& weights) {
    GenerateSingle(model_, *weights, runtime_config, prompt, pos, prefix_end,
                   kv_cache, &env_, timing_info);
  });

  env_.ctx.pools.MaybeStopSpinning(runtime_config.use_spinning);
}

void Gemma::GenerateBatch(const RuntimeConfig& runtime_config,
                          const QueriesPromptTokens& queries_prompt,
                          const QueriesPos& queries_pos,
                          const QueriesPos& queries_prefix_end,
                          const KVCaches& kv_caches,
                          TimingInfo& timing_info) const {
  // If we did not get passed prefix ends (size 0), assume 0 and pass that on.
  QueriesPos mutable_queries_prefix_end = queries_prefix_end;
  std::vector<size_t> prefix_end_vec;
  if (queries_prefix_end.size() == 0) {  // hwy::Span lacks empty()
    prefix_end_vec.resize(queries_prompt.size(), 0);
    mutable_queries_prefix_end =
        QueriesPos(prefix_end_vec.data(), prefix_end_vec.size());
  }

  env_.ctx.pools.MaybeStartSpinning(runtime_config.use_spinning);

  weights_.CallT([&](auto& weights) {
    gcpp::GenerateBatch(model_, *weights, runtime_config, queries_prompt,
                        queries_pos, mutable_queries_prefix_end, kv_caches,
                        &env_, timing_info);
  });

  env_.ctx.pools.MaybeStopSpinning(runtime_config.use_spinning);
}

void Gemma::GenerateImageTokens(const RuntimeConfig& runtime_config,
                                const Image& image,
                                ImageTokens& image_tokens) const {
  env_.ctx.pools.MaybeStartSpinning(runtime_config.use_spinning);

  weights_.CallT([&](auto& weights) {
    gcpp::GenerateImageTokens(model_, *weights, runtime_config, image,
                              image_tokens, &env_);
  });

  env_.ctx.pools.MaybeStopSpinning(runtime_config.use_spinning);
}

}  // namespace gcpp
