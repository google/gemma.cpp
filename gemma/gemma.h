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
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/timer.h"
// IWYU pragma: end_exports
#include "hwy/aligned_allocator.h"  // Span
#include "hwy/base.h"  // hwy::bfloat16_t

namespace gcpp {

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
// If not empty, SampleFunc is called with the probability distribution for the
// next token, and its return value is used as the next generated token.
using SampleFunc = std::function<int(const float*, size_t)>;
// If not empty, LayersOutputFunc is called for layer outputs, specified with:
// - index of query within containing batch (if any); zero otherwise.
// - position in the tokens sequence
// - name of the data, e.g. "tokens", "blocks", "final_norm"
// - layer index (or -1 for global outputs), e.g. "blocks" exposes x per-layer
// - pointer to the data array
// - size of the data array
using LayersOutputFunc =
    std::function<void(size_t, size_t, const std::string&, int, const float*, size_t)>;

struct RuntimeConfig {
  bool StreamToken(size_t query_idx, size_t pos, int token, float prob) const {
    if (batch_stream_token) {
      return batch_stream_token(query_idx, pos, token, prob);
    }
    return stream_token(token, prob);
  }

  size_t max_tokens;
  size_t max_generated_tokens;

  // These defaults are overridden by InferenceArgs::CopyTo(*this):
  // Max tokens per batch during prefill.
  size_t prefill_tbatch_size = 32;
  // Max queries per batch (one token from each) during decode.
  size_t decode_qbatch_size = 16;

  float temperature;
  int verbosity;
  std::mt19937* gen;
  StreamFunc stream_token;
  BatchStreamFunc batch_stream_token;
  AcceptFunc accept_token;  // if empty, accepts all tokens.
  SampleFunc sample_func;   // if empty, uses SampleTopK.
  LayersOutputFunc layers_output;  // if not empty, called after each layer.
  int eos_id = EOS_ID;
};

struct TimingInfo {
  void NotifyPrefill(size_t tokens, double start) {
    prefill_tok_sec =
        static_cast<double>(tokens) / (hwy::platform::Now() - start);
    gen_tok_sec = 0.0;
    time_to_first_token = 0.0;
    tokens_generated = 0;
  }

  void NotifyGenerated(double prefill_start) {
    ++tokens_generated;
    if (HWY_UNLIKELY(tokens_generated == 1)) {
      time_to_first_token = hwy::platform::Now() - prefill_start;
    }
  }

  void NotifyGenerateDone(double gen_start) {
    gen_tok_sec = static_cast<double>(tokens_generated) /
                  (hwy::platform::Now() - gen_start);
  }

  double prefill_tok_sec;
  double gen_tok_sec;
  double time_to_first_token;
  size_t tokens_generated;
};

using PromptTokens = hwy::Span<const int>;
using MultiplePromptsTokens = hwy::Span<const PromptTokens>;
using KVCaches = hwy::Span<KVCache>;

class Gemma {
 public:
  Gemma(const Path& tokenizer_path, const Path& weights, const ModelInfo& info,
        hwy::ThreadPool& pool);

  // Allocates weights, caller is responsible for filling them.
  Gemma(GemmaTokenizer&& tokenizer, const ModelInfo& info,
        hwy::ThreadPool& pool);
  ~Gemma();

  const ModelInfo& Info() const { return info_; }
  const GemmaTokenizer& Tokenizer() const { return tokenizer_; }
  const ByteStorageT& Weights() const { return weights_u8_; }

  void Generate(const RuntimeConfig& runtime_config, const PromptTokens& prompt,
                size_t start_pos, KVCache& kv_cache, TimingInfo& timing_info);

  void GenerateBatch(const RuntimeConfig& runtime_config,
                     const MultiplePromptsTokens& prompts, size_t start_pos,
                     const KVCaches& kv_caches, TimingInfo& timing_info);

 private:
  hwy::ThreadPool& pool_;

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

// DEPRECATED, call Gemma::Generate directly.
HWY_INLINE void GenerateGemma(Gemma& gemma, const RuntimeConfig& runtime_config,
                              const std::vector<int>& prompt, size_t start_pos,
                              KVCache& kv_cache, hwy::ThreadPool& /*pool*/,
                              TimingInfo& timing_info) {
  gemma.Generate(runtime_config, prompt, start_pos, kv_cache, timing_info);
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_H_
