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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_H_

#include <algorithm>
#include <cctype>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

// copybara:import_next_line:gemma_cpp
#include "compression/compress.h"  // SfpStream/NuqStream
// copybara:end
// copybara:import_next_line:gemma_cpp
#include "configs.h"  // kSeqLen
// copybara:end
// copybara:import_next_line:gemma_cpp
#include "util/args.h"  // ArgsBase
// copybara:end
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"  // hwy::bfloat16_t
#include "hwy/contrib/thread_pool/thread_pool.h"
// copybara:import_next_line:sentencepiece
#include "src/sentencepiece_processor.h"
// copybara:end

namespace gcpp {

// Allowable types for GEMMA_WEIGHT_T (can be specified at compilation time):
// float, hwy::bfloat16_t, SfpStream, NuqStream
#ifndef GEMMA_WEIGHT_T
#define GEMMA_WEIGHT_T SfpStream
#endif  // !GEMMA_WEIGHT_T
using WeightT = GEMMA_WEIGHT_T;

using EmbedderInputT = hwy::bfloat16_t;
constexpr size_t kPrefillBatchSize = 16;
constexpr bool kSystemPrompt = false;

struct KVCache {
  hwy::AlignedFreeUniquePtr<float[]>
      key_cache;  // batch_size * kSeqLen * kLayers * kKVHeads * kQKVDim
  hwy::AlignedFreeUniquePtr<float[]>
      value_cache;  // batch_size * kSeqLen * kLayers * kKVHeads * kQKVDim
};

// Model variants: see configs.h for details.
enum class Model { GEMMA_2B, GEMMA_7B };
enum class ModelTraining { GEMMA_IT, GEMMA_PT };

struct LoaderArgs : public ArgsBase<LoaderArgs> {
  LoaderArgs(int argc, char* argv[]) { InitAndParse(argc, argv); }

  static std::string ToLower(const std::string& text) {
    std::string result = text;
    std::transform(begin(result), end(result), begin(result),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
  }

  gcpp::Model ModelType() const {
    const std::string model_type_lc = ToLower(model_type);
    if (model_type_lc == "2b-pt" || model_type_lc == "2b-it") {
      return gcpp::Model::GEMMA_2B;
    } else {
      return gcpp::Model::GEMMA_7B;
    }
  }

  gcpp::ModelTraining ModelTraining() const {
    const std::string model_type_lc = ToLower(model_type);
    if (model_type_lc == "7b-pt" || model_type_lc == "2b-pt") {
      return gcpp::ModelTraining::GEMMA_PT;
    } else {
      return gcpp::ModelTraining::GEMMA_IT;
    }
  }

  // Returns error string or nullptr if OK.
  const char* Validate() const {
    const std::string model_type_lc = ToLower(model_type);
    if (model_type_lc != "2b-pt" && model_type_lc != "7b-pt" &&
        model_type_lc != "2b-it" && model_type_lc != "7b-it") {
      return "Model type must be 2b-pt, 7b-pt, 2b-it, or "
             "7b-it.";
    }
    if (tokenizer.path.empty()) {
      return "Missing --tokenizer flag, a file for the tokenizer is required.";
    }
    if (model_type.empty()) {
      return "Missing --model flag, need to specify either 2b-pt, 7b-pt, "
             "2b-it, or 7b-it.";
    }
    if (cache.path.empty()) {
      return "Missing --compressed_weights flag, a file for the compressed "
             "model.";
    }
    return nullptr;
  }

  Path tokenizer;
  Path model;  // uncompressed weights OR
  Path cache;  // compressed weights
  std::string model_type;

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(tokenizer, "tokenizer", Path(),
            "Path name of tokenizer model file.\n    Required argument.");
    visitor(
        cache, "compressed_weights", Path(),
        "Path name of compressed weights file, regenerated from `--weights` "
        "file if "
        "the compressed weights file does not exist.\n    Required argument.");
    visitor(model_type, "model", std::string(),
            "Model type\n    2b-it (2B parameters, instruction-tuned)\n    "
            "2b-pt (2B parameters, pretrained)\n    7b-it (7B parameters "
            "instruction-tuned)\n    7b-pt (7B parameters, pretrained)\n"
            "    Required argument.");
    visitor(model, "weights", Path(),
            "Path name of model weights (.sbs) file. Only required if "
            "compressed_weights file is not present and needs to be "
            "regenerated. This parameter is only required for compressing"
            "new model weight exports, otherwise it is not needed.");
  }
};

struct GemmaInterface;

struct Gemma {
  Gemma(const LoaderArgs& args, hwy::ThreadPool& pool);
  ~Gemma();  // must be defined after GemmaInterface's dtor is defined.

  const sentencepiece::SentencePieceProcessor& Tokenizer() const;

  std::unique_ptr<GemmaInterface> impl_;
  gcpp::ModelTraining model_training;
};

// StreamFunc is called with (token, probability). For prompt tokens,
// probability is 0.0f.
using StreamFunc = std::function<bool(int, float)>;
using AcceptFunc = std::function<bool(int)>;

struct InferenceArgs : public ArgsBase<InferenceArgs> {
  InferenceArgs(int argc, char* argv[]) { InitAndParse(argc, argv); }

  size_t max_tokens;
  size_t max_generated_tokens;

  float temperature;
  bool deterministic;
  bool multiturn;

  // Returns error string or nullptr if OK.
  const char* Validate() const {
    if (max_tokens > gcpp::kSeqLen) {
      return "max_tokens is larger than the maximum sequence length (see "
             "configs.h).";
    }
    if (max_generated_tokens > max_tokens) {
      return "Maximum number of generated tokens is larger than the maximum "
             "total tokens.";
    }
    return nullptr;
  }

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(max_tokens, "max_tokens", size_t{3072},
            "Maximum number of tokens in prompt + generation.");
    visitor(max_generated_tokens, "max_generated_tokens", size_t{2048},
            "Maximum number of tokens to generate.");

    visitor(temperature, "temperature", 1.0f, "Temperature for top-K", 2);
    visitor(deterministic, "deterministic", false,
            "Make top-k sampling deterministic", 2);
    visitor(multiturn, "multiturn", false,
            "Multiturn mode (if 0, this clears the KV cache after every "
            "interaction without quitting)\n    Default : 0 (conversation "
            "resets every turn)");
  }
};

void GenerateGemma(Gemma& gemma, const InferenceArgs& args,
                   const std::vector<int>& prompt, size_t start_pos,
                   hwy::ThreadPool& pool, hwy::ThreadPool& inner_pool,
                   const StreamFunc& stream_token,
                   const AcceptFunc& accept_token, std::mt19937& g,
                   int verbosity);

constexpr int EOS_ID = 1;

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_H_
