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

#include <functional>
#include <memory>
#include <random>
#include <vector>

// copybara:import_next_line:gemma_cpp
#include "compression/compress.h"  // SfpStream/NuqStream
// copybara:import_next_line:gemma_cpp
#include "configs.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"  // hwy::bfloat16_t
#include "hwy/contrib/thread_pool/thread_pool.h"
// copybara:import_next_line:gemma_cpp
#include "util/args.h"  // Path
// copybara:import_next_line:sentencepiece
#include "src/sentencepiece_processor.h"

namespace gcpp {

using GemmaWeightT = GEMMA_WEIGHT_T;
using EmbedderInputT = hwy::bfloat16_t;
constexpr size_t kPrefillBatchSize = 16;
constexpr bool kSystemPrompt = false;

struct KVCache {
  hwy::AlignedFreeUniquePtr<float[]>
      key_cache;  // kSeqLen * kNumGemmaLayers * kKVHeads * kQKVDim
  hwy::AlignedFreeUniquePtr<float[]>
      value_cache;  // kSeqLen * kNumGemmaLayers * kKVHeads * kQKVDim
  hwy::AlignedFreeUniquePtr<float[]>
      conv1d_cache;  // (kConv1dWidth - 1) * kModelDim * kNumGriffinLayers
  hwy::AlignedFreeUniquePtr<float[]>
      rglru_cache;  // kModelDim * kNumGriffinLayers
};

// Model variants: see configs.h for details.
enum class Model { GEMMA_2B, GEMMA_7B, GRIFFIN_2B };
enum class ModelTraining { GEMMA_IT, GEMMA_PT };

// Returns error string or nullptr if OK.
// Thread-hostile.
const char* ParseModelTypeAndTraining(const std::string& model_flag,
                                      Model& model, ModelTraining& training);

struct RuntimeConfig {
  size_t max_tokens;
  size_t max_generated_tokens;
  float temperature;
  int verbosity;
};

struct GemmaInterface;

class GemmaTokenizer {
 public:
  virtual bool Encode(const std::string& input,
                      std::vector<std::string>* pieces) const = 0;
  virtual bool Encode(const std::string& input,
                      std::vector<int>* pieces) const = 0;
  virtual bool Decode(const std::vector<int>& ids,
                      std::string* detokenized) const = 0;
};

struct Gemma {
  Gemma(const Path& tokenizer_path, const Path& weights, Model model_type,
        hwy::ThreadPool& pool);
  ~Gemma();  // must be defined after GemmaInterface's dtor is defined.
  const GemmaTokenizer* Tokenizer() const;
  std::unique_ptr<GemmaInterface> impl_;
};

KVCache CreateKVCache(Model type);  // convenient workaround for now
KVCache CreateKVCache(size_t size_cache_pos, size_t seq_len,
                      size_t conv1d_cache_size, size_t rglru_cache_size);

// StreamFunc is called with (token, probability). For prompt tokens,
// probability is 0.0f.
using StreamFunc = std::function<bool(int, float)>;
using AcceptFunc = std::function<bool(int)>;

void GenerateGemma(Gemma& gemma, size_t max_tokens, size_t max_generated_tokens,
                   float temperature, const std::vector<int>& prompt,
                   size_t start_pos, KVCache& kv_cache, hwy::ThreadPool& pool,
                   hwy::ThreadPool& inner_pool, const StreamFunc& stream_token,
                   const AcceptFunc& accept_token, std::mt19937& gen,
                   int verbosity);

// Convenience function for the common case:
// - Bundle runtime parameters as RuntimeConfig
// - No threadpools within threadpools (inner_pool = dummy)
// - All tokens accepted
void GenerateGemma(Gemma& gemma, RuntimeConfig runtime_config,
                   const std::vector<int>& prompt, size_t start_pos,
                   KVCache& kv_cache, hwy::ThreadPool& pool,
                   const StreamFunc& stream_token, std::mt19937& gen);

void CompressWeights(gcpp::Model model, const Path& weights,
                     const Path& compressed_weights, hwy::ThreadPool& pool);

float ComputeCrossEntropy(Gemma& gemma, size_t max_tokens,
                          const std::vector<int>& prompt, KVCache& kv_cache,
                          hwy::ThreadPool& pool, hwy::ThreadPool& inner_pool,
                          int verbosity);

constexpr int EOS_ID = 1;

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_H_
