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

// Shared between various frontends.

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_ARGS_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_ARGS_H_

#include <stddef.h>
#include <stdio.h>

#include <functional>
#include <random>
#include <string>

#include "io/io.h"       // Path
#include "ops/matmul.h"  // MMStorage::kMax*
#include "util/args.h"
#include "util/basics.h"  // Tristate
#include "util/mat.h"
#include "hwy/aligned_allocator.h"  // Span
#include "hwy/base.h"               // HWY_ABORT
#include "hwy/profiler.h"

namespace gcpp {

struct LoaderArgs : public ArgsBase<LoaderArgs> {
  LoaderArgs(int argc, char* argv[]) { InitAndParse(argc, argv); }
  LoaderArgs(const std::string& tokenizer_path,
             const std::string& weights_path) {
    Init();  // Init sets to defaults, so assignments must come after Init().
    tokenizer.path = tokenizer_path;
    weights.path = weights_path;
  };

  Path tokenizer;
  Path weights;  // weights file location
  Tristate map;
  Tristate to_bf16;
  Tristate wrapping;

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(tokenizer, "tokenizer", Path(),
            "Path name of tokenizer model; only required for pre-2025 format.");
    visitor(weights, "weights", Path(),
            "Path name of model weights (.sbs) file.\n  Required argument.\n");
    visitor(map, "map", Tristate::kDefault,
            "Enable memory-mapping? -1 = auto, 0 = no, 1 = yes.");
    visitor(to_bf16, "to_bf16", Tristate::kDefault,
            "Convert weights to bf16? -1 = auto, 0 = no, 1 = yes.");
    visitor(wrapping, "wrapping", Tristate::kDefault,
            "Enable prompt wrapping? Specify 0 for pre-2025 format PT models.");
  }
};

using PromptTokens = hwy::Span<const int>;

// Batches of independent queries have their own prompt, previous token,
// position in the sequence, and KVCache.
using QueriesPromptTokens = hwy::Span<const PromptTokens>;
using QueriesToken = hwy::Span<const int>;
using QueriesPos = hwy::Span<const size_t>;

// ImageTokens are represented as a matrix, where each row corresponds
// to a token for an image patch as computed by the image encoder.
using ImageTokens = MatStorageT<float>;

// StreamFunc is called with (token, probability). For prompt tokens,
// probability is 0.0f. StreamFunc should return false to stop generation and
// true to continue generation.
using StreamFunc = std::function<bool(int, float)>;
// BatchStreamFunc is called with (query_idx, pos, token, probability).
// For prompt tokens, probability is 0.0f. Generation continues if this returns
// true and stops if it returns false. Note that query_idx is absolute, not
// relative to the batch.
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
struct Activations;
using ActivationsObserverFunc =
    std::function<void(const QueriesPos& queries_pos, int, const Activations&)>;

// RuntimeConfig holds configuration for a single generation run.
// TODO: move into InferenceArgs, use that directly.
struct RuntimeConfig {
  // If non-null, `batch_stream_token` is called for each token in the batch,
  // otherwise `stream_token`. `query_idx` is absolute, not batch-relative.
  bool StreamToken(size_t query_idx, size_t pos, int token, float prob) const {
    PROFILER_ZONE("Gen.StreamToken");
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
  float temperature;  // Temperature for sampling.

  size_t top_k = 1;           // Top-k for sampling.
  std::mt19937* gen;          // Random number generator used for sampling.

  int verbosity;  // Controls verbosity of printed messages.

  // Functions operating on the generated tokens.
  StreamFunc stream_token;
  BatchStreamFunc batch_stream_token;
  AcceptFunc accept_token;  // if empty, accepts all tokens.
  SampleFunc sample_func;   // if empty, uses SampleTopK.

  // Observer callbacks for intermediate data.
  LayersOutputFunc layers_output;  // if not empty, called after each layer.
  ActivationsObserverFunc activations_observer;  // if set, called per-layer.

  // If not empty, these point to the image tokens and are used in the
  // PaliGemma prefix-LM style attention.
  const ImageTokens* image_tokens = nullptr;

  // Whether to use thread spinning to reduce barrier synchronization latency.
  // Mutable so we can change kDefault to kTrue/kFalse during Generate, because
  // RuntimeConfig is const there and is not passed to the Gemma ctor. This
  // default decision is likely sufficient because it is based on whether
  // threads are successfully pinned.
  mutable Tristate use_spinning = Tristate::kDefault;
};

struct InferenceArgs : public ArgsBase<InferenceArgs> {
  InferenceArgs(int argc, char* argv[]) { InitAndParse(argc, argv); }
  InferenceArgs() { Init(); };

  bool IsInteractive() const { return prompt.empty() && prompt_file.Empty(); }

  int verbosity;

  size_t seq_len;
  size_t max_generated_tokens;

  size_t prefill_tbatch_size;
  size_t decode_qbatch_size;

  float temperature;
  size_t top_k;
  bool deterministic;
  bool multiturn;
  Path image_file;

  std::string prompt;  // Bypasses std::getline
  // For prompts longer than the Linux terminal's 4K line edit buffer.
  Path prompt_file;
  std::string eot_line;

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(verbosity, "verbosity", 1,
            "Show verbose developer information\n    0 = only print generation "
            "output\n    1 = standard user-facing terminal ui\n    2 = show "
            "developer/debug info).\n    Default = 1.",
            1);

    visitor(seq_len, "seq_len", size_t{8192},
            "Sequence length, capped by ModelConfig.max_seq_len.");
    visitor(max_generated_tokens, "max_generated_tokens", size_t{4096},
            "Maximum number of tokens to generate.");

    visitor(prefill_tbatch_size, "prefill_tbatch", size_t{256},
            "Prefill: max tokens per batch.");
    visitor(decode_qbatch_size, "decode_qbatch", size_t{16},
            "Decode: max queries per batch.");

    visitor(temperature, "temperature", 1.0f, "Temperature for top-K", 2);
    visitor(top_k, "top_k", size_t{1}, "Number of top-K tokens to sample from",
            2);
    visitor(deterministic, "deterministic", false,
            "Make top-k sampling deterministic", 2);
    visitor(multiturn, "multiturn", false,
            "Multiturn mode\n    0 = clear KV cache after every "
            "interaction\n    1 = continue KV cache after every interaction\n  "
            "  Default : 0 (conversation "
            "resets every turn)");
    visitor(image_file, "image_file", Path(), "Image file to load.");

    visitor(prompt, "prompt", std::string(""),
            "Initial prompt for non-interactive mode. When specified, "
            "generates a response and exits.",
            1);
    visitor(prompt_file, "prompt_file", Path(),
            "Path to file containing the prompt for non-interactive mode. When "
            " specified, generates a response and exits.",
            1);

    visitor(
        eot_line, "eot_line", std::string(""),
        "End of turn line. "
        "When you specify this, the prompt will be all lines "
        "before the line where only the given string appears.\n    Default = "
        "When a newline is encountered, that signals the end of the turn.",
        2);
  }

  void CopyTo(RuntimeConfig& runtime_config) const {
    runtime_config.max_generated_tokens = max_generated_tokens;
    runtime_config.prefill_tbatch_size = prefill_tbatch_size;
    runtime_config.decode_qbatch_size = decode_qbatch_size;
    if (prefill_tbatch_size > MMStorage::kMaxM) {
      HWY_ABORT(
          "prefill_tbatch_size %zu > kMaxM %zu: specify a smaller value, "
          "or increase the constant in MMStorage.\n",
          prefill_tbatch_size, MMStorage::kMaxM);
    }
    if (decode_qbatch_size > MMStorage::kMaxM) {
      HWY_ABORT(
          "decode_qbatch_size %zu > kMaxM %zu: specify a smaller value, "
          "or increase the constant in MMStorage.\n",
          decode_qbatch_size, MMStorage::kMaxM);
    }

    runtime_config.temperature = temperature;
    runtime_config.top_k = top_k;
  }
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_ARGS_H_
