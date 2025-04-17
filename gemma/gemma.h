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

#include <functional>
#include <random>
#include <string>
#include <vector>

// IWYU pragma: begin_exports
#include "compression/io.h"  // Path
#include "gemma/activations.h"
#include "gemma/common.h"
#include "gemma/configs.h"
#include "gemma/kv_cache.h"
#include "gemma/tokenizer.h"
#include "gemma/weights.h"
#include "ops/matmul.h"  // MatMulEnv
#include "paligemma/image.h"
#include "util/basics.h"  // TokenAndProb
#include "util/mat.h"     // RowVectorBatch
#include "util/threading_context.h"
#include "hwy/timer.h"
// IWYU pragma: end_exports
#include "hwy/aligned_allocator.h"  // Span

namespace gcpp {
using PromptTokens = hwy::Span<const int>;

// Batches of independent queries have their own prompt, previous token,
// position in the sequence, and KVCache.
using QueriesPromptTokens = hwy::Span<const PromptTokens>;
using QueriesToken = hwy::Span<const int>;
using QueriesPos = hwy::Span<const size_t>;
using KVCaches = hwy::Span<KVCache>;

// StreamFunc is called with (token, probability). For prompt tokens,
// probability is 0.0f. StreamFunc should return false to stop generation and
// true to continue generation.
using StreamFunc = std::function<bool(int, float)>;
// BatchStreamFunc is called with (query_idx, pos, token, probability).
// For prompt tokens, probability is 0.0f.
// StreamFunc should return false to stop generation and true to continue.
using BatchStreamFunc = std::function<bool(size_t, size_t, int, float)>;
// If not empty, AcceptFunc is called with token. It should return false for
// tokens you don't want to generate and true for tokens you want to generate.
using AcceptFunc = std::function<bool(int, float)>;
// If not empty, SampleFunc is called with the logits for the next token, which
// it may modify/overwrite, and its return value is the next generated token
// together with its probability.
using SampleFunc = std::function<TokenAndProb(float*, size_t)>;
// If not empty, LayersOutputFunc is called for layer outputs, specified with:
// - index of query within containing batch (if any); zero otherwise.
// - position in the tokens sequence
// - name of the data, e.g. "tokens" for token IDs
// - layer index (or -1 for global outputs)
// - pointer to the data array
// - size of the data array
using LayersOutputFunc = std::function<void(size_t, size_t, const std::string&,
                                            int, const float*, size_t)>;
// If not empty, ActivationsObserverFunc is invoked after each layer with:
// - per-query position within the tokens sequence
// - layer index (or -1 for post-norm output)
// - activations
using ActivationsObserverFunc =
    std::function<void(const QueriesPos& queries_pos, int, const Activations&)>;

// ImageTokens are represented as a RowVectorBatch, where each "batch" index
// corresponds to a token for an image patch as computed by the image encoder.
using ImageTokens = RowVectorBatch<float>;

// RuntimeConfig holds configuration for a single generation run.
struct RuntimeConfig {
  // If not empty, batch_stream_token is called for each token in the batch,
  // instead of stream_token.
  bool StreamToken(size_t query_idx, size_t pos, int token, float prob) const {
    if (batch_stream_token) {
      return batch_stream_token(query_idx, pos, token, prob);
    }
    return stream_token(token, prob);
  }

  // Limit on the number of tokens generated.
  size_t max_generated_tokens;

  // These defaults are overridden by InferenceArgs::CopyTo(*this):
  // Max tokens per batch during prefill.
  size_t prefill_tbatch_size = 256;
  // Max queries per batch (one token from each) during decode.
  size_t decode_qbatch_size = 16;

  // Sampling-related parameters.
  float temperature;     // Temperature for sampling.
  size_t top_k = kTopK;  // Top-k for sampling.
  std::mt19937* gen;     // Random number generator used for sampling.

  int verbosity;  // Controls verbosity of printed messages.

  // Functions operating on the generated tokens.
  StreamFunc stream_token;
  BatchStreamFunc batch_stream_token;
  AcceptFunc accept_token;         // if empty, accepts all tokens.
  SampleFunc sample_func;          // if empty, uses SampleTopK.

  // Observer callbacks for intermediate data.
  LayersOutputFunc layers_output;  // if not empty, called after each layer.
  ActivationsObserverFunc activations_observer;  // if set, called per-layer.

  // If not empty, these point to the image tokens and are used in the
  // PaliGemma prefix-LM style attention.
  const ImageTokens *image_tokens = nullptr;

  // Whether to use thread spinning to reduce barrier synchronization latency.
  // Mutable so we can change kDefault to kTrue/kFalse during Generate, because
  // RuntimeConfig is const there and is not passed to the Gemma ctor. This
  // default decision is likely sufficient because it is based on whether
  // threads are successfully pinned.
  mutable Tristate use_spinning = Tristate::kDefault;

  // End-of-sequence token.
  int eos_id = EOS_ID;
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

// Internal init must run before I/O; calling it from GemmaEnv() is too late.
// This helper function takes care of the internal init plus calling `SetArgs`.
MatMulEnv MakeMatMulEnv(const ThreadingArgs& threading_args);

class Gemma {
 public:
  // Reads old format weights file and tokenizer file.
  // `env` must remain valid for the lifetime of this Gemma.
  Gemma(const Path& tokenizer_path, const Path& weights, const ModelInfo& info,
        MatMulEnv& env);
  // Reads new format weights file that contains everything in a single file.
  // `env` must remain valid for the lifetime of this Gemma.
  Gemma(const Path& weights, MatMulEnv& env);
  // Allocates weights, caller is responsible for filling them.
  Gemma(GemmaTokenizer&& tokenizer, const ModelInfo& info, MatMulEnv& env);
  ~Gemma();

  MatMulEnv& Env() const { return env_; }
  const ModelConfig& GetModelConfig() const { return model_.Config(); }
  // DEPRECATED
  ModelInfo Info() const {
    return ModelInfo({.model = model_.Config().model,
                      .wrapping = model_.Config().wrapping,
                      .weight = model_.Config().weight});
  }
  const GemmaTokenizer& Tokenizer() const { return tokenizer_; }
  const GemmaChatTemplate& ChatTemplate() const { return chat_template_; }
  const ModelWeightsStorage& Weights() const { return model_; }
  ModelWeightsStorage& MutableWeights() { return model_; }
  void Save(const Path& weights, hwy::ThreadPool& pool) {
    std::string tokenizer_proto = tokenizer_.Serialize();
    model_.Save(tokenizer_proto, weights, pool);
  }

  // `pos` is the position in the KV cache. Users are responsible for
  // incrementing it in the `*StreamFunc`, or setting to zero for single-turn.
  void Generate(const RuntimeConfig& runtime_config, const PromptTokens& prompt,
                size_t pos, KVCache& kv_cache, TimingInfo& timing_info) {
    Generate(runtime_config, prompt, pos, /*prefix_end=*/0, kv_cache,
             timing_info);
  }
  // For prefix-LM style attention, we can pass the end of the prefix.
  void Generate(const RuntimeConfig& runtime_config, const PromptTokens& prompt,
                size_t pos, size_t prefix_end, KVCache& kv_cache,
                TimingInfo& timing_info);

  // `queries_pos` are the positions in the KV cache. Users are responsible for
  // incrementing them in `BatchStreamFunc`, or setting to zero for single-turn.
  void GenerateBatch(const RuntimeConfig& runtime_config,
                     const QueriesPromptTokens& queries_prompt,
                     const QueriesPos& queries_pos, const KVCaches& kv_caches,
                     TimingInfo& timing_info) {
    GenerateBatch(runtime_config, queries_prompt, queries_pos,
                  /*queries_prefix_end=*/{}, kv_caches, timing_info);
  }
  // For prefix-LM style attention, we can pass the ends of the prefixes.
  void GenerateBatch(const RuntimeConfig& runtime_config,
                     const QueriesPromptTokens& queries_prompt,
                     const QueriesPos& queries_pos,
                     const QueriesPos& queries_prefix_end,
                     const KVCaches& kv_caches, TimingInfo& timing_info);

  // Generates the image tokens by running the image encoder ViT.
  void GenerateImageTokens(const RuntimeConfig& runtime_config,
                           const Image& image, ImageTokens& image_tokens);

 private:
  MatMulEnv& env_;

  GemmaTokenizer tokenizer_;
  GemmaChatTemplate chat_template_;
  // Type-erased so that this can be defined in the header.
  ModelWeightsStorage model_;
};

void RangeChecks(const ModelConfig& weights_config,
                 size_t& max_generated_tokens, size_t prompt_size);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_H_
