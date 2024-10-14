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

#include <functional>
#include <random>
#include <string>
#include <vector>

// IWYU pragma: begin_exports
#include "compression/io.h"  // Path
#include "gemma/activations.h"
#include "gemma/common.h"
#include "gemma/kv_cache.h"
#include "gemma/tokenizer.h"
#include "paligemma/image.h"
#include "util/allocator.h"  // RowVectorBatch
#include "util/basics.h"     // TokenAndProb
#include "util/threading.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
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
  size_t prefill_tbatch_size = 32;
  // Max queries per batch (one token from each) during decode.
  size_t decode_qbatch_size = 16;

  float temperature;  // Temperature for sampling.
  int verbosity;  // Controls verbosity of printed messages.
  std::mt19937* gen;  // Random number generator used for sampling.

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
  bool use_spinning = true;

  // End-of-sequence token.
  int eos_id = EOS_ID;
};

struct TimingInfo {
  void NotifyPrefill(size_t tokens, double start) {
    prefill_duration = hwy::platform::Now() - start;
    prefill_tokens = tokens;
    time_to_first_token = 0.0;
    tokens_generated = 0;
  }

  void NotifyGenerated(double prefill_start, double gen_start) {
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
                           (hwy::platform::Now() - gen_start);
      fprintf(stderr,
              "\n\n[ Timing info ] %zu tokens generated "
              "(avg speed %.2f tokens / sec)\n\n",
              tokens_generated, gen_tok_sec);
    }
  }

  void NotifyGenerateDone(double gen_start) {
    generate_duration = hwy::platform::Now() - gen_start;
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
  double prefill_duration = 0;
  size_t prefill_tokens = 0;
  double time_to_first_token = 0;
  double generate_duration = 0;
  size_t tokens_generated = 0;
};

// ModelConfigInfo holds model configuration details: number of layers, etc.
struct ModelConfigInfo {
  const int layers;
  const int model_dim;
  const int heads;
  const int kv_heads;
  const int qkv_dim;
};

class Gemma {
 public:
  Gemma(const Path& tokenizer_path, const Path& weights, const ModelInfo& info,
        PerClusterPools& pools);

  // Allocates weights, caller is responsible for filling them.
  Gemma(GemmaTokenizer&& tokenizer, const ModelInfo& info,
        PerClusterPools& pools);
  ~Gemma();

  ModelConfigInfo ModelConfig() const;
  const ModelInfo& Info() const { return info_; }
  const GemmaTokenizer& Tokenizer() const { return tokenizer_; }
  const ByteStorageT& Weights() const { return weights_u8_; }
  ByteStorageT& MutableWeights() { return weights_u8_; }

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
  PerClusterPools& pools_;

  GemmaTokenizer tokenizer_;
  // Type-erased so that this can be defined in the header.
  ByteStorageT weights_u8_;
  ModelInfo info_;
};

// Adds BOS token and possibly 'turn' annotations, which depend on `training`
// and `pos`, the number of tokens decoded so far; returns the corresponding
// tokens. Asserts that tokenization is successful.
std::vector<int> WrapAndTokenize(const GemmaTokenizer& tokenizer,
                                 const ModelInfo& info, size_t pos,
                                 std::string& prompt);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_H_
