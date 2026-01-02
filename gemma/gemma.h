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

#include <optional>
#include <vector>

// IWYU pragma: begin_exports
#include "gemma/activations.h"
#include "gemma/configs.h"
#include "gemma/gemma_args.h"
#include "gemma/kv_cache.h"
#include "gemma/model_store.h"
#include "gemma/query.h"
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

// Used for continuous batching.
class ContinuousQBatch : public QBatch {
 public:
  ContinuousQBatch(size_t max_size, AllQueries& queries);

  // Whether we should prefill the next batch, i.e. next_to_insert_ ==
  // next_to_prefill_.
  bool ShouldPrefill() const;

  // Setup the query_idx_ to point to the next group of queries to prefill.
  void SetupNextBatchForPrefill();

  // Get the next query to insert to the generate batch.
  std::optional<size_t> GetNextToInsert();

  // Collect the kv_cache from QBatch to available_kv_caches_.
  void MaybeReleaseKV(const QBatch& from);

 public:
  int next_to_prefill_ = 0;
  int next_to_insert_ = 0;
  std::vector<KVCachePtr> available_kv_caches_;
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
  void NotifyGenerated(size_t batch_size) {
    generation_steps += 1;
    const bool is_first = (tokens_generated == 0);
    tokens_generated += batch_size;
    if (HWY_UNLIKELY(is_first)) {
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
    if (HWY_UNLIKELY(verbosity >= 2 && tokens_generated % 1024 == 0)) {
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
  size_t generation_steps = 0;
};

// After construction, all methods are const and thread-compatible if using
// separate `ThreadingContext` and `MatMulEnv` for each concurrent `Generate`.
class Gemma {
 public:
  // Reads weights/config/tokenizer from `BlobStore` at `args.loader.weights`.
  // `ctx` is only used to read tensors and not stored. Calls to `Generate*`
  // may reference the same, or other `ThreadingContext` via `MatMulEnv`.
  Gemma(const GemmaArgs& args, ThreadingContext& ctx);

  // Deprecated prior interface for backwards compatibility.
  Gemma(const LoaderArgs& loader, const InferenceArgs& inference,
        ThreadingContext& ctx)
      : Gemma(GemmaArgs(loader, ThreadingArgs(), inference), ctx) {}

  ~Gemma();

  const ModelConfig& Config() const { return model_.Config(); }
  const GemmaTokenizer& Tokenizer() const { return model_.Tokenizer(); }
  const WeightsPtrs& Weights() const { return weights_; }
  WeightsPtrs::Mode WeightReadMode() const { return weight_read_mode_; }
  const GemmaChatTemplate& ChatTemplate() const { return chat_template_; }
  const InferenceArgs& Inference() const { return inference_; }

  void Save(const Path& weights_path, ThreadingContext& ctx) const;

  // `pos` is the position in the KV cache. Users are responsible for
  // incrementing it in the `*StreamFunc`, or setting to zero for single-turn.
  // All `Generate*` may be called concurrently if `env` and the
  // `ThreadingContext` it references are both distinct.
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
  AesCtrEngine aes_ctr_engine_;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_H_
