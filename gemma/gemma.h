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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_H_

#include <stdio.h>

#include <vector>

// IWYU pragma: begin_exports
#include "gemma/activations.h"
#include "gemma/configs.h"
#include "gemma/gemma_args.h"
#include "gemma/kv_cache.h"
#include "gemma/model_store.h"
#include "gemma/weights.h"
#include "io/blob_store.h"
#include "io/io.h"       // Path
#include "ops/matmul.h"  // MatMulEnv
#include "paligemma/image.h"
#include "util/basics.h"  // TokenAndProb
#include "util/threading_context.h"
#include "hwy/timer.h"
// IWYU pragma: end_exports

namespace gcpp {

struct PerQuery {
  PromptTokens prompt;

  // Position in the KV cache: initially zero for the first turn, or when
  // multi-turn is NOT desired. Incremented by prefill and `StreamAndUpdateEOS`.
  size_t mutable_pos;
  // Allows computing the last prefill token as `mutable_pos - initial_pos`,
  // which might differ from `prompt.size() - 1` for prefix-LM.
  size_t initial_pos;
  // Zero for causal attention, or the end of the prefix for prefix-LM style
  // attention in Paligemma.
  size_t prefix_end;

  KVCache& kv_cache;

  // Previous token generated for this query, or the last prompt token. Will be
  // fed into the next Transformer() call.
  int prev_token = 0;
};

// Array of `PerQuery`. Referenced by `QBatch` and passed to `GenerateBatch`.
struct AllQueries {
  AllQueries() = default;

  // For `GenerateSingleT`: same prompt/pos, replicated for each KV cache.
  AllQueries(const PromptTokens& prompt, size_t pos, size_t prefix_end,
             const hwy::Span<KVCache>& kv_caches) {
    per_query_.reserve(kv_caches.size());
    for (size_t i = 0; i < kv_caches.size(); ++i) {
      HWY_ASSERT(kv_caches[i].SeqLen() == kv_caches[0].SeqLen());
      per_query_.push_back(PerQuery{
          .prompt = prompt,
          .mutable_pos = pos,
          .initial_pos = pos,
          .prefix_end = prefix_end,
          .kv_cache = kv_caches[i],
      });
    }
  }

  // Batch of queries with initial position set to zero. Causal attention
  // is requested via empty or all-zero `prefix_end`.
  AllQueries(
      const hwy::Span<const PromptTokens>& prompts,
      const hwy::Span<KVCache>& kv_caches,
      const hwy::Span<const size_t>& prefix_end = hwy::Span<const size_t>()) {
    HWY_ASSERT(prompts.size() == kv_caches.size());
    HWY_ASSERT(prompts.size() == prefix_end.size() || prefix_end.size() == 0);
    per_query_.reserve(kv_caches.size());
    for (size_t i = 0; i < kv_caches.size(); ++i) {
      HWY_ASSERT(kv_caches[i].SeqLen() == kv_caches[0].SeqLen());
      per_query_.push_back(PerQuery{
          .prompt = prompts[i],
          .mutable_pos = 0,
          .initial_pos = 0,
          .prefix_end = prefix_end.size() == 0 ? 0 : prefix_end[i],
          .kv_cache = kv_caches[i],
      });
    }
  }

  void Reserve(size_t size) { per_query_.reserve(size); }
  void Append(const PerQuery& query) { per_query_.push_back(query); }

  size_t NumQueries() const { return per_query_.size(); }

  PerQuery& operator[](size_t query_idx) {
    HWY_DASSERT(query_idx < NumQueries());
    return per_query_[query_idx];
  }
  const PerQuery& operator[](size_t query_idx) const {
    HWY_DASSERT(query_idx < NumQueries());
    return per_query_[query_idx];
  }

 private:
  std::vector<PerQuery> per_query_;
};

// View into AllQueries: either a batch of queries, or a single query for use
// in PrefillTBatch or GenerateSingleT. Cheap to create because it holds a
// reference to AllQueries.
class QBatch {
 public:
  QBatch(size_t start, size_t max_size, AllQueries& queries)
      : start_(start),
        max_size_(max_size),
        queries_(queries),
        size_(HWY_MIN(max_size_, queries_.NumQueries() - start_)) {
    HWY_ASSERT(max_size_ <= 4096);  // non_eos uses `BitSet4096`.
    HWY_DASSERT(size_ != 0);
    HWY_DASSERT(start_ + size_ <= queries_.NumQueries());
  }

  // Returns a single-query view starting at `qi` relative to this batch.
  QBatch Single(size_t qi) const { return QBatch(start_ + qi, 1, queries_); }

  // How many queries in this batch, <= `queries_.NumQueries()` and `max_size_`.
  size_t Size() const { return size_; }

  // Returns index for use with `AllQueries` and `BatchStreamToken`.
  size_t QueryIdx(size_t qi) const {
    HWY_DASSERT(qi < size_);
    return start_ + qi;
  }

  // Accessor functions to bridge the previous SoA and current AoS layout.
  const PromptTokens& Prompt(size_t qi) const {
    return queries_[QueryIdx(qi)].prompt;
  }
  size_t Pos(size_t qi) const { return queries_[QueryIdx(qi)].mutable_pos; }
  size_t& MutablePos(size_t qi) { return queries_[QueryIdx(qi)].mutable_pos; }
  size_t InitialPos(size_t qi) const {
    return queries_[QueryIdx(qi)].initial_pos;
  }
  size_t PrefixEnd(size_t qi) const {
    return queries_[QueryIdx(qi)].prefix_end;
  }
  KVCache& KV(size_t qi) const { return queries_[QueryIdx(qi)].kv_cache; }
  int& PrevToken(size_t qi) { return queries_[QueryIdx(qi)].prev_token; }

 private:
  size_t start_;
  size_t max_size_;
  AllQueries& queries_;
  size_t size_;
};

struct TimingInfo {
  // be sure to populate prefill_start before calling NotifyPrefill.
  void NotifyPrefill(size_t tokens) {
    prefill_duration = hwy::platform::Now() - prefill_start;
    prefill_tokens = tokens;
    time_to_first_token = 0.0;
    tokens_generated = 0;
  }

  // be sure to populate prefill_start and generate_start before calling
  // NotifyGenerated.
  void NotifyGenerated() {
    ++tokens_generated;
    if (HWY_UNLIKELY(tokens_generated == 1)) {
      time_to_first_token = hwy::platform::Now() - prefill_start;
      if (verbosity >= 1) {
        double prefill_tok_sec =
            static_cast<double>(prefill_tokens) / prefill_duration;
        fprintf(stderr,
                "\n\n[ Timing info ] Prefill: %d ms for %zu prompt tokens "
                "(%.2f tokens / sec); Time to first token: %d ms\n",
                static_cast<int>(prefill_duration * 1000), prefill_tokens,
                prefill_tok_sec, static_cast<int>(time_to_first_token * 1000));
      }
    }
    if (verbosity >= 2 && tokens_generated % 128 == 0) {
      double gen_tok_sec = static_cast<double>(tokens_generated) /
                           (hwy::platform::Now() - generate_start);
      fprintf(stderr,
              "\n\n[ Timing info ] %zu tokens generated "
              "(avg speed %.2f tokens / sec)\n\n",
              tokens_generated, gen_tok_sec);
    }
  }

  // be sure to populate generate_start before calling NotifyGenerateDone.
  void NotifyGenerateDone() {
    generate_duration = hwy::platform::Now() - generate_start;
    if (verbosity >= 1) {
      double gen_tok_sec =
          static_cast<double>(tokens_generated) / generate_duration;
      fprintf(stderr,
              "\n[ Timing info ] Generate: %d ms for %zu tokens (%.2f tokens / "
              "sec)\n",
              static_cast<int>(generate_duration * 1000), tokens_generated,
              gen_tok_sec);
    }
  }

  int verbosity = 0;
  double prefill_start = 0;
  double generate_start = 0;
  double prefill_duration = 0;
  size_t prefill_tokens = 0;
  double time_to_first_token = 0;
  double generate_duration = 0;
  size_t tokens_generated = 0;
};

// After construction, all methods are const and thread-compatible if using
// separate ThreadingContext for each thread.
class Gemma {
 public:
  // Reads weights/config/tokenizer from the `BlobStore` at `loader.weights`.
  // `ctx` is only used to read tensors, but it is typically also referenced
  // by the `MatMulEnv` passed to the Generate* methods.
  Gemma(const LoaderArgs& loader, const InferenceArgs& inference,
        ThreadingContext& ctx);
  ~Gemma();

  const ModelConfig& Config() const { return model_.Config(); }
  const GemmaTokenizer& Tokenizer() const { return model_.Tokenizer(); }
  const WeightsPtrs& Weights() const { return weights_; }
  WeightsPtrs::Mode WeightReadMode() const { return weight_read_mode_; }
  const GemmaChatTemplate& ChatTemplate() const { return chat_template_; }
  const InferenceArgs& Inference() const { return inference_; }

  void Save(const Path& weights_path, NestedPools& pools) const;

  // `pos` is the position in the KV cache. Users are responsible for
  // incrementing it in the `*StreamFunc`, or setting to zero for single-turn.
  void Generate(const RuntimeConfig& runtime_config, const PromptTokens& prompt,
                size_t pos, KVCache& kv_cache, MatMulEnv& env,
                TimingInfo& timing_info) const {
    Generate(runtime_config, prompt, pos, /*prefix_end=*/0, kv_cache, env,
             timing_info);
  }
  // For prefix-LM style attention, we can pass the end of the prefix.
  void Generate(const RuntimeConfig& runtime_config, const PromptTokens& prompt,
                size_t pos, size_t prefix_end, KVCache& kv_cache,
                MatMulEnv& env, TimingInfo& timing_info) const;

  void GenerateBatch(const RuntimeConfig& runtime_config,
                     AllQueries& all_queries, MatMulEnv& env,
                     TimingInfo& timing_info) const;

  // Generates the image tokens by running the image encoder ViT.
  void GenerateImageTokens(const RuntimeConfig& runtime_config, size_t seq_len,
                           const Image& image, ImageTokens& image_tokens,
                           MatMulEnv& env) const;

 private:
  BlobReader reader_;
  ModelStore model_;
  std::vector<MatOwner> mat_owners_;
  WeightsPtrs weights_;
  WeightsPtrs::Mode weight_read_mode_;
  GemmaChatTemplate chat_template_;
  InferenceArgs inference_;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_H_
