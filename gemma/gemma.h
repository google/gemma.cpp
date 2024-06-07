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
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"  // hwy::bfloat16_t
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

constexpr size_t kPrefillBatchSize = 16;
constexpr bool kSystemPrompt = false;

struct KVCache {
  hwy::AlignedFreeUniquePtr<float[]>
      kv_cache;  // kSeqLen * kGemmaLayers * kKVHeads * kQKVDim * 2
  hwy::AlignedFreeUniquePtr<float[]>
      conv1d_cache;  // (kConv1dWidth - 1) * kModelDim * kGriffinLayers
  hwy::AlignedFreeUniquePtr<float[]>
      rglru_cache;  // kModelDim * kGriffinLayers

  static KVCache Create(Model type);
};

constexpr int EOS_ID = 1;

class GemmaTokenizer {
 public:
  GemmaTokenizer();
  explicit GemmaTokenizer(const Path& tokenizer_path);

  // must come after definition of Impl
  ~GemmaTokenizer();
  GemmaTokenizer(GemmaTokenizer&& other);
  GemmaTokenizer& operator=(GemmaTokenizer&& other);

  bool Encode(const std::string& input, std::vector<std::string>* pieces) const;
  bool Encode(const std::string& input, std::vector<int>* pieces) const;
  bool Decode(const std::vector<int>& ids, std::string* detokenized) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// StreamFunc is called with (token, probability). For prompt tokens,
// probability is 0.0f. StreamFunc should return False to stop generation and
// True to continue generation.
using StreamFunc = std::function<bool(int, float)>;
// AcceptFunc is called with token. It should return False for tokens you don't
// want to generate and True for tokens you want to generate.
using AcceptFunc = std::function<bool(int)>;
// CustomSampleFunc is called with the probability distribution for the next
// token, and its return value is used as the next generated token.
using CustomSampleFunc = std::function<int(const float*, size_t)>;

struct RuntimeConfig {
  size_t max_tokens;
  size_t max_generated_tokens;
  float temperature;
  int verbosity;
  std::mt19937* gen;
  const StreamFunc& stream_token;
  const AcceptFunc& accept_token;
  const CustomSampleFunc* sample_func = nullptr;
  int eos_id = EOS_ID;
};

struct TimingInfo {
  double prefill_tok_sec = 0.0;
  double gen_tok_sec = 0.0;
  double time_to_first_token = 0;
};

// Will be called for layers output with:
// - position in the tokens sequence
// - name of the data, p.ex. "tokens", "block.1", "final_norm"
// - pointer to the data array
// - size of the data array
using LayersOutputT =
    std::function<void(int, const std::string&, const float*, size_t)>;

class Gemma {
 public:
  Gemma(const Path& tokenizer_path, const Path& weights, Model model_type,
        hwy::ThreadPool& pool);

  // Allocates weights, caller is responsible for filling them.
  Gemma(GemmaTokenizer&& tokenizer, Model model_type, hwy::ThreadPool& pool);
  ~Gemma();

  Model ModelType() const { return model_type_; }
  const GemmaTokenizer& Tokenizer() const { return tokenizer_; }
  const ByteStorageT& Weights() const { return weights_u8_; }
  const ByteStorageT& Prefill() const { return prefill_u8_; }
  const ByteStorageT& Decode() const { return decode_u8_; }

  // layers_output is optional; if set - it will be called with the activations
  // output after applying each layer.
  void Generate(const RuntimeConfig& runtime_config,
                const std::vector<int>& prompt, size_t start_pos,
                KVCache& kv_cache, TimingInfo& timing_info,
                LayersOutputT* layers_output = nullptr);

 private:
  hwy::ThreadPool& pool_;

  GemmaTokenizer tokenizer_;
  // Type-erased so that this can be defined in the header, without requiring
  // forwarding functions.
  ByteStorageT weights_u8_;
  ByteStorageT prefill_u8_;
  ByteStorageT decode_u8_;
  Model model_type_;
};

// DEPRECATED, call Gemma::Generate directly.
HWY_INLINE void GenerateGemma(Gemma& gemma, const RuntimeConfig& runtime_config,
                              const std::vector<int>& prompt, size_t start_pos,
                              KVCache& kv_cache, hwy::ThreadPool& /*pool*/,
                              TimingInfo& timing_info,
                              LayersOutputT* layers_output) {
  gemma.Generate(runtime_config, prompt, start_pos, kv_cache, timing_info,
                 layers_output);
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_H_
