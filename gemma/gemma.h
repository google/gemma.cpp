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
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "compression/io.h"  // Path
#include "gemma/common.h"
#include "gemma/configs.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"  // hwy::bfloat16_t
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

using GemmaWeightT = GEMMA_WEIGHT_T;
using EmbedderInputT = hwy::bfloat16_t;
// Will be called for layers output with:
// - position in the tokens sequence
// - name of the data, p.ex. "tokens", "block.1", "final_norm"
// - pointer to the data array
// - size of the data array
using LayersOutputT =
    std::function<void(int, const std::string&, const float*, size_t)>;
constexpr size_t kPrefillBatchSize = 16;
constexpr bool kSystemPrompt = false;

struct KVCache {
  hwy::AlignedFreeUniquePtr<float[]>
      kv_cache;  // kSeqLen * kGemmaLayers * kKVHeads * kQKVDim * 2
  hwy::AlignedFreeUniquePtr<float[]>
      conv1d_cache;  // (kConv1dWidth - 1) * kModelDim * kGriffinLayers
  hwy::AlignedFreeUniquePtr<float[]>
      rglru_cache;  // kModelDim * kGriffinLayers
};

enum class ModelTraining { GEMMA_IT, GEMMA_PT };

// Returns error string or nullptr if OK.
// Thread-hostile.
const char* ParseModelTypeAndTraining(const std::string& model_flag,
                                      Model& model, ModelTraining& training);

// StreamFunc is called with (token, probability). For prompt tokens,
// probability is 0.0f. StreamFunc should return False to stop generation and
// True to continue generation.
using StreamFunc = std::function<bool(int, float)>;
// AcceptFunc is called with token. It should return False for tokens you don't
// want to generate and True for tokens you want to generate.
using AcceptFunc = std::function<bool(int)>;

constexpr int EOS_ID = 1;

struct RuntimeConfig {
  size_t max_tokens;
  size_t max_generated_tokens;
  float temperature;
  int verbosity;
  std::mt19937* gen;
  const StreamFunc& stream_token;
  const AcceptFunc& accept_token;
  int eos_id = EOS_ID;
};

struct GemmaInterface;

class GemmaTokenizer {
 public:
  virtual ~GemmaTokenizer() = default;
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
  ~Gemma();  // must be defined after the GemmaInterface dtor is defined.
  const GemmaTokenizer* Tokenizer() const;
  std::unique_ptr<GemmaInterface> impl_;
};

struct TimingInfo {
  double prefill_tok_sec = 0.0;
  double gen_tok_sec = 0.0;
  double time_to_first_token = 0;
};

KVCache CreateKVCache(Model type);  // convenient workaround for now
KVCache CreateKVCache(size_t size_cache_pos, size_t seq_len,
                      size_t conv1d_cache_size, size_t rglru_cache_size);

// Bundle runtime parameters as RuntimeConfig
// layers_output is optional; if set - it will be called with the activations
// output after applying each layer.
void GenerateGemma(Gemma& gemma, const RuntimeConfig& runtime_config,
                   const std::vector<int>& prompt, size_t start_pos,
                   KVCache& kv_cache, hwy::ThreadPool& pool,
                   TimingInfo& timing_info,
                   LayersOutputT* layers_output = nullptr);

void GenerateGemma(Model model, const ByteStorageT& weights,
                   ByteStorageT& inference_state,
                   RuntimeConfig runtime_config,
                   const std::vector<int>& prompt, size_t start_pos,
                   KVCache& kv_cache, hwy::ThreadPool& pool,
                   TimingInfo& timing_info);

ByteStorageT LoadWeights(const Path& weights, Model model,
                         hwy::ThreadPool& pool);

ByteStorageT AllocateInferenceState(Model model);

void CompressWeights(gcpp::Model model, const Path& weights,
                     const Path& compressed_weights, hwy::ThreadPool& pool);

float ComputeCrossEntropy(Gemma& gemma, size_t max_tokens,
                          const std::vector<int>& prompt, KVCache& kv_cache,
                          hwy::ThreadPool& pool, int verbosity);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_H_
