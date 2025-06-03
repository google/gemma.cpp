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

MatMulEnv MakeMatMulEnv(const ThreadingArgs& threading_args);

using KVCaches = hwy::Span<KVCache>;

class Gemma {
 public:
  // Reads weights/config/tokenizer from the `BlobStore` at `loader.weights`.
  // `env` must remain valid for the lifetime of this Gemma.
  Gemma(const LoaderArgs& loader, MatMulEnv& env);

  ~Gemma();

  MatMulEnv& Env() const { return env_; }
  // TODO: rename to Config()
  const ModelConfig& GetModelConfig() const { return model_.Config(); }
  const GemmaTokenizer& Tokenizer() const { return model_.Tokenizer(); }
  const ModelWeightsPtrs& Weights() const { return weights_; }
  const GemmaChatTemplate& ChatTemplate() const { return chat_template_; }

  void Save(const Path& weights_path, hwy::ThreadPool& pool) const;

  // `pos` is the position in the KV cache. Users are responsible for
  // incrementing it in the `*StreamFunc`, or setting to zero for single-turn.
  void Generate(const RuntimeConfig& runtime_config, const PromptTokens& prompt,
                size_t pos, KVCache& kv_cache, TimingInfo& timing_info) const {
    Generate(runtime_config, prompt, pos, /*prefix_end=*/0, kv_cache,
             timing_info);
  }
  // For prefix-LM style attention, we can pass the end of the prefix.
  void Generate(const RuntimeConfig& runtime_config, const PromptTokens& prompt,
                size_t pos, size_t prefix_end, KVCache& kv_cache,
                TimingInfo& timing_info) const;

  // `queries_pos` are the positions in the KV cache. Users are responsible for
  // incrementing them in `BatchStreamFunc`, or setting to zero for single-turn.
  void GenerateBatch(const RuntimeConfig& runtime_config,
                     const QueriesPromptTokens& queries_prompt,
                     const QueriesPos& queries_pos, const KVCaches& kv_caches,
                     TimingInfo& timing_info) const {
    GenerateBatch(runtime_config, queries_prompt, queries_pos,
                  /*queries_prefix_end=*/{}, kv_caches, timing_info);
  }
  // For prefix-LM style attention, we can pass the ends of the prefixes.
  void GenerateBatch(const RuntimeConfig& runtime_config,
                     const QueriesPromptTokens& queries_prompt,
                     const QueriesPos& queries_pos,
                     const QueriesPos& queries_prefix_end,
                     const KVCaches& kv_caches, TimingInfo& timing_info) const;

  // Generates the image tokens by running the image encoder ViT.
  void GenerateImageTokens(const RuntimeConfig& runtime_config,
                           const Image& image, ImageTokens& image_tokens) const;

 private:
  MatMulEnv& env_;
  BlobReader reader_;
  ModelStore model_;
  std::vector<MatOwner> mat_owners_;
  ModelWeightsPtrs weights_;
  GemmaChatTemplate chat_template_;
};

void RangeChecks(const ModelConfig& weights_config,
                 size_t& max_generated_tokens, size_t prompt_size);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_H_
