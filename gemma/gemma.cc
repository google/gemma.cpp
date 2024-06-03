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

// Lightweight C++ implementation of the gemma model.

// Compiles this file for multiple architectures via "foreach_target.h", to
// which we pass the filename via macro 'argument'.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/gemma.cc"  // NOLINT
#include "hwy/foreach_target.h"        // IWYU pragma: keep
// Must come after foreach_target.h to avoid redefinition errors.
#include "compression/compress-inl.h"
#include "gemma/common-inl.h"
#include "gemma/ops.h"
#include "hwy/contrib/matvec/matvec-inl.h"
#include "hwy/highway.h"

// Non-SIMD includes and types. Note that HWY_ONCE is only true on the last
// compile pass, whereas we want this defined in the first.
#ifndef GEMMA_ONCE
#define GEMMA_ONCE

#include <math.h>  // sqrtf
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <regex>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#include "compression/compress.h"
#include "compression/io.h"  // Path
#include "gemma/configs.h"
#include "gemma/gemma.h"
#include "gemma/weights.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"
#include "hwy/timer.h"
// copybara:import_next_line:sentencepiece
#include "src/sentencepiece_processor.h"

// Setting this to true disables fread() calls that read the model file.
constexpr bool kDryRunFread = false;

// Setting this to false will load and use uncompressed weights.
constexpr bool kWeightsAreCompressed = true;

// Set this to true to debug tokenizer tokens.
constexpr bool kShowTokenization = false;

namespace gcpp {

float ScaleWeights(float* data, size_t len) {
  float maxabs = 0.0;
  for (size_t i = 0; i < len; ++i) {
    maxabs = std::max(maxabs, std::abs(data[i]));
  }
  const float kMaxRange = 1.875f;
  if (maxabs <= kMaxRange) {
    return 1.0f;
  }
  const float scale = maxabs / kMaxRange;
  const float inv_scale = 1.0f / scale;
  for (size_t i = 0; i < len; ++i) {
    data[i] *= inv_scale;
  }
  return scale;
}

template <typename TConfig>
hwy::AlignedFreeUniquePtr<uint8_t[]> LoadWeights(
    const Path& checkpoint, hwy::ThreadPool& pool,
    bool scale_for_compression = false) {
  PROFILER_ZONE("Startup.LoadWeights");
  if (!checkpoint.Exists()) {
    HWY_ABORT("The model weights file '%s' does not exist.",
              checkpoint.path.c_str());
  }

  ByteStorageT weights_u8 = AllocateWeights<float, TConfig>(pool);
  auto* weights = reinterpret_cast<WeightsF<TConfig>*>(weights_u8.get());

  size_t scale_pos = 0;
  FILE* fptr;
  if constexpr (kDryRunFread) {
    fprintf(stderr, "Dry-Run, not reading model-file.\n");
  } else {
    fptr = fopen(checkpoint.path.c_str(), "rb");
    if (fptr == nullptr) {
      HWY_ABORT("Failed to open model file %s - does it exist?",
                checkpoint.path.c_str());
    }
  }
  bool ok = true;
  uint64_t total_size = 0;
  auto do_fread = [&](void* var, int layer, const char* name, size_t size) {
    if (layer == -1) {
      fprintf(stderr, "Loading Parameters (size %zu): %s\n", size, name);
    } else {
      fprintf(stderr, "Loading Parameters (layer=%d, size %zu): %s\n", layer,
              size, name);
    }
    if constexpr (!kDryRunFread) {
      ok &= 1 == fread(var, size, 1, fptr);
      total_size += size;
    }
  };
  do_fread(&(weights->embedder_input_embedding), -1, "embedder_input_embedding",
           sizeof(weights->embedder_input_embedding));
  do_fread(&(weights->final_norm_scale), -1, "final_norm_scale",
           sizeof(weights->final_norm_scale));
  for (size_t layer = 0; layer < TConfig::kLayers; ++layer) {
    auto type = TConfig::kLayerConfig[layer];
    LayerF<TConfig>* layer_view = weights->GetLayer(layer);

#define READ_WEIGHTS(name)                                                 \
  do {                                                                     \
    do_fread(&(layer_view->name), layer, #name, sizeof(layer_view->name)); \
  } while (0)

#define SCALE_WEIGHTS(name)                                               \
  do {                                                                    \
    if (ok && !kDryRunFread && scale_for_compression) {                   \
      weights->scales[scale_pos++] =                                      \
          ScaleWeights(layer_view->name.data(), layer_view->name.size()); \
    }                                                                     \
  } while (0)
    // Make sure we don't have uninitialized memory.
    hwy::ZeroBytes(layer_view, sizeof(*layer_view));
    if (type == LayerAttentionType::kGemma) {
      READ_WEIGHTS(attn_vec_einsum_w);
      READ_WEIGHTS(qkv_einsum_w);
      SCALE_WEIGHTS(attn_vec_einsum_w);
      SCALE_WEIGHTS(qkv_einsum_w);
    } else {
      READ_WEIGHTS(griffin.linear_x_w);
      READ_WEIGHTS(griffin.linear_x_biases);
      READ_WEIGHTS(griffin.linear_y_w);
      READ_WEIGHTS(griffin.linear_y_biases);
      READ_WEIGHTS(griffin.linear_out_w);
      READ_WEIGHTS(griffin.linear_out_biases);
      READ_WEIGHTS(griffin.conv_w);
      READ_WEIGHTS(griffin.conv_biases);
      READ_WEIGHTS(griffin.gate_w);
      READ_WEIGHTS(griffin.gate_biases);
      READ_WEIGHTS(griffin.a);
      SCALE_WEIGHTS(griffin.linear_x_w);
      SCALE_WEIGHTS(griffin.linear_y_w);
      SCALE_WEIGHTS(griffin.linear_out_w);
      SCALE_WEIGHTS(griffin.gate_w);
    }
    READ_WEIGHTS(gating_einsum_w);
    READ_WEIGHTS(linear_w);
    SCALE_WEIGHTS(gating_einsum_w);
    SCALE_WEIGHTS(linear_w);
    READ_WEIGHTS(pre_attention_norm_scale);
    READ_WEIGHTS(pre_ffw_norm_scale);
    if (TConfig::kPostNormScale) {
      READ_WEIGHTS(post_attention_norm_scale);
      READ_WEIGHTS(post_ffw_norm_scale);
    }
    if (TConfig::kFFBiases) {
      READ_WEIGHTS(ffw_gating_biases);
      READ_WEIGHTS(ffw_output_biases);
    }
    if (TConfig::kSoftmaxAttnOutputBiases &&
        type == LayerAttentionType::kGemma) {
      READ_WEIGHTS(attention_output_biases);
    }
#undef READ_WEIGHTS
  }
  if (!ok) {
    HWY_ABORT(
        "Failed to read from %s - might be a directory, or too small? "
        "expected size: %d kB",
        checkpoint.path.c_str(), static_cast<uint32_t>(total_size >> 10));
  }
  if (!kDryRunFread) {
    HWY_ASSERT(0 == fclose(fptr));
    if (scale_for_compression) {
      HWY_ASSERT(scale_pos == TConfig::kNumTensorScales);
    }
  }
  return weights_u8;
}

template <class TConfig>
struct CompressedLayer {
  // No ctor/dtor, allocated via AllocateAligned.

  using TLayer = gcpp::LayerF<TConfig>;
  using WeightT = typename TConfig::WeightT;

  static constexpr size_t kHeads = TLayer::kHeads;
  static constexpr size_t kKVHeads = TLayer::kKVHeads;
  static constexpr size_t kModelDim = TLayer::kModelDim;
  static constexpr size_t kQKVDim = TLayer::kQKVDim;
  static constexpr size_t kFFHiddenDim = TLayer::kFFHiddenDim;
  static constexpr size_t kAttVecEinsumWSize = TLayer::kAttVecEinsumWSize;
  static constexpr size_t kQKVEinsumWSize = TLayer::kQKVEinsumWSize;
  static constexpr size_t kGatingEinsumWSize = TLayer::kGatingEinsumWSize;
  static constexpr size_t kConv1dWidth = TLayer::kConv1dWidth;
  static constexpr bool kFFBiases = TLayer::kFFBiases;
  static constexpr bool kPostNormScale = TConfig::kPostNormScale;
  static constexpr size_t kAOBiasDim = TLayer::kAOBiasDim;
  static constexpr size_t kGriffinDim = TLayer::kGriffinDim;

  // Compressed Parameters

  template <class T, size_t N>
  using ArrayT = CompressedArray<T, N>;

  union {
    struct {
      ArrayT<WeightT, kAttVecEinsumWSize> attn_vec_einsum_w;
      ArrayT<WeightT, kQKVEinsumWSize> qkv_einsum_w;
      ArrayT<float, kAOBiasDim> attention_output_biases;
    };

    struct {
      ArrayT<WeightT, kGriffinDim * kGriffinDim> linear_x_w;
      ArrayT<float, kGriffinDim> linear_x_biases;
      ArrayT<WeightT, kGriffinDim * kGriffinDim> linear_y_w;
      ArrayT<float, kGriffinDim> linear_y_biases;
      ArrayT<WeightT, kGriffinDim * kGriffinDim> linear_out_w;
      ArrayT<float, kGriffinDim> linear_out_biases;
      ArrayT<float, TConfig::kConv1dWidth * kGriffinDim> conv_w;
      ArrayT<float, kGriffinDim> conv_biases;
      ArrayT<WeightT, kGriffinDim * kGriffinDim / kHeads * 2> gate_w;
      ArrayT<float, kGriffinDim * 2> gate_biases;
      ArrayT<float, kGriffinDim> a;
    } griffin;
  };

  ArrayT<WeightT, TLayer::kGatingEinsumWSize> gating_einsum_w;
  ArrayT<WeightT, kModelDim * kFFHiddenDim> linear_w;
  // We don't yet have an RMSNorm that accepts all WeightT.
  ArrayT<hwy::bfloat16_t, kModelDim> pre_attention_norm_scale;
  ArrayT<hwy::bfloat16_t, kModelDim> pre_ffw_norm_scale;
  ArrayT<hwy::bfloat16_t, kPostNormScale ? kModelDim : 0>
      post_attention_norm_scale;
  ArrayT<hwy::bfloat16_t, kPostNormScale ? kModelDim : 0> post_ffw_norm_scale;

  ArrayT<float, kFFBiases ? 2 * kFFHiddenDim : 0> ffw_gating_biases;
  ArrayT<float, kFFBiases ? kModelDim : 0> ffw_output_biases;
};

// Array instead of single large allocation for parallel mem init. Split out
// of CompressedWeights so that only these pointers are initialized, not the
// CompressedArray.
template <class TConfig>
struct CompressedLayerPointers {
  explicit CompressedLayerPointers(hwy::ThreadPool& pool) {
    pool.Run(0, TConfig::kLayers, [this](uint64_t task, size_t /*thread*/) {
      this->c_layers[task] = hwy::AllocateAligned<CompressedLayer<TConfig>>(1);
    });
  }

  using CLayer = CompressedLayer<TConfig>;
  std::array<hwy::AlignedFreeUniquePtr<CLayer[]>, TConfig::kLayers> c_layers;
};

template <class TConfig>
struct CompressedWeights {
  // No ctor/dtor, allocated via AllocateAligned.

  CompressedArray<EmbedderInputT, TConfig::kVocabSize * TConfig::kModelDim>
      embedder_input_embedding;

  CompressedArray<hwy::bfloat16_t, TConfig::kModelDim> final_norm_scale;

  // Must be last so that the other arrays remain aligned.
  CompressedLayerPointers<TConfig> c_layer_ptrs;

  const CompressedLayer<TConfig>* GetLayer(size_t layer) const {
    return c_layer_ptrs.c_layers[layer].get();
  }
  CompressedLayer<TConfig>* GetLayer(size_t layer) {
    return c_layer_ptrs.c_layers[layer].get();
  }
};

template <class TConfig>
using WeightsT = hwy::If<kWeightsAreCompressed, CompressedWeights<TConfig>,
                         WeightsF<TConfig>>;

// Aligned.
template <class TConfig, size_t TBatchSize>
struct Activations {
  static constexpr size_t kBatchSize = TBatchSize;
  using LayerConfig = LayerF<TConfig>;
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kKVHeads = TConfig::kKVHeads;
  static constexpr size_t kCacheLayerSize = kKVHeads * kQKVDim * 2;
  static constexpr size_t kCachePosSize =
      TConfig::kGemmaLayers * kCacheLayerSize;
  static constexpr size_t kQDim = kHeads == kKVHeads ? kQKVDim * 3 : kQKVDim;

  std::array<float, kBatchSize * kModelDim> x;  // input
  std::array<float, kBatchSize * kModelDim> pre_att_rms_out;
  std::array<float, kBatchSize * kHeads * kQDim> q;  // query vector
  std::array<float, kBatchSize * kHeads * TConfig::kSeqLen>
      att;                                                   // attention vector
  std::array<float, kBatchSize * kHeads * kQKVDim> att_out;  // attention output
  std::array<float, kHeads * kBatchSize * kModelDim>
      att_post1;  // attention output after linear transformation, per head
  std::array<float, kBatchSize * kModelDim>
      att_post2;  // accumulation of attention outputs over heads
  std::array<hwy::bfloat16_t, kBatchSize * kModelDim> bf_pre_ffw_rms_out;
  std::array<float, kBatchSize * TConfig::kFFHiddenDim * 2> ffw_hidden;
  // bf_ version can't be used until GeluMulToBF16 issue in FFW() is resolved.
  // std::array<hwy::bfloat16_t, kBatchSize * 2 * TConfig::kFFHiddenDim>
  //     bf_ffw_hidden;
  std::array<float, kBatchSize * kModelDim> ffw_out;
  std::array<float, kBatchSize * TConfig::kVocabSize> logits;

  // For bf16/f32 vectors * bf16 matrix: faster to unpack once beforehand, into
  // per-thread storage.
  std::array<float, kModelDim * kMaxThreads> even_odd;

  // Griffin layer internal activations
  static constexpr size_t kGriffinDim =
      TConfig::kGriffinLayers > 0 ? kModelDim : 0;
  std::array<float, kBatchSize * kGriffinDim> griffin_x;
  std::array<float, kBatchSize * kGriffinDim> griffin_y;
  std::array<float, kBatchSize * kGriffinDim> griffin_gate_x;
  std::array<float, kBatchSize * kGriffinDim> griffin_multiplier;
};

template<typename TConfig>
struct InferenceState {
  Activations<TConfig, kPrefillBatchSize> prefill;
  HWY_ALIGN Activations<TConfig, 1> state;

  static ByteStorageT Allocate() {
    return hwy::AllocateAligned<uint8_t>(sizeof(InferenceState<TConfig>));
  }
};

// GemmaImpl is a template and thus cannot be exposed in gemma.h, hence we
// define an abstract base class.
struct GemmaInterface {
  virtual ~GemmaInterface() = default;

  virtual const GemmaTokenizer* Tokenizer() const = 0;

  virtual void Generate(const RuntimeConfig& runtime_config,
                        const std::vector<int>& prompt, size_t start_pos,
                        KVCache& kv_cache, hwy::ThreadPool& pool,
                        TimingInfo& timing_info,
                        LayersOutputT* layers_output) = 0;

  virtual float ComputeCrossEntropy(size_t max_tokens,
                                    const std::vector<int>& prompt,
                                    KVCache& kv_cache, hwy::ThreadPool& pool,
                                    int verbosity) = 0;
};

template <class Config>
KVCache CreateKVCacheT() {
  constexpr size_t kConv1dWidth = Config::kConv1dWidth;
  return CreateKVCache(
      Config::kGemmaLayers * Config::kKVHeads * Config::kQKVDim,
      Config::kSeqLen + kPrefillBatchSize,
      Config::kGriffinLayers * (kConv1dWidth == 0 ? 0 : kConv1dWidth - 1) *
          Config::kModelDim,
      Config::kGriffinLayers * Config::kModelDim);
}

KVCache CreateKVCache(Model type) {
  switch (type) {
    case Model::GEMMA_2B:
      return CreateKVCacheT<ConfigGemma2B>();
    case Model::GEMMA_7B:
      return CreateKVCacheT<ConfigGemma7B>();
    case Model::GRIFFIN_2B:
      return CreateKVCacheT<ConfigGriffin2B>();
    case Model::GEMMA_TINY:
      return CreateKVCacheT<ConfigGemmaTiny>();
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(type));
  }
}

class GemmaTokenizerImpl : public GemmaTokenizer {
 public:
  GemmaTokenizerImpl(
      std::unique_ptr<sentencepiece::SentencePieceProcessor>&& impl)
      : impl_(std::move(impl)) {}
  bool Encode(const std::string& input,
              std::vector<std::string>* pieces) const override {
    return impl_->Encode(input, pieces).ok();
  }
  bool Encode(const std::string& input,
              std::vector<int>* pieces) const override {
    if constexpr (kShowTokenization) {
      bool is_ok = impl_->Encode(input, pieces).ok();
      for (int i = 0; i < static_cast<int>(pieces->size()); i++) {
        fprintf(stderr, "%3d: %d\n", i, (*pieces)[i]);
      }
      return is_ok;
    } else {
      return impl_->Encode(input, pieces).ok();
    }
  }
  // Given a sequence of ids, decodes it into a detokenized output.
  bool Decode(const std::vector<int>& ids,
              std::string* detokenized) const override {
    return impl_->Decode(ids, detokenized).ok();
  }

 private:
  std::unique_ptr<sentencepiece::SentencePieceProcessor> impl_;
};

namespace {
template <class Config>
void DeleteLayersPtrs(CompressedWeights<Config>* c_weights) {
  c_weights->c_layer_ptrs.~CompressedLayerPointers<Config>();
}
template <class Config>
void DeleteLayersPtrs(WeightsF<Config>* weights) {
  weights->layer_ptrs.~LayerPointers<float, Config>();
}
}  // namespace

template <class Config>
struct GemmaImpl : public GemmaInterface {
  GemmaImpl(std::unique_ptr<sentencepiece::SentencePieceProcessor>& tokenizer,
            hwy::AlignedFreeUniquePtr<uint8_t[]>& weights_u8,
            hwy::ThreadPool& pool);

  ~GemmaImpl() {
    WeightsT<Config>* weights =
        reinterpret_cast<WeightsT<Config>*>(weights_u8.get());
    DeleteLayersPtrs(weights);
  }

  const GemmaTokenizer* Tokenizer() const override { return &tokenizer; }

  void Generate(const RuntimeConfig& runtime_config,
                const std::vector<int>& prompt, size_t start_pos,
                KVCache& kv_cache, hwy::ThreadPool& pool,
                TimingInfo& timing_info, LayersOutputT* layers_output) override;

  float ComputeCrossEntropy(size_t max_tokens, const std::vector<int>& prompt,
                            KVCache& kv_cache, hwy::ThreadPool& pool,
                            int verbosity) override;

  GemmaTokenizerImpl tokenizer;
  hwy::AlignedFreeUniquePtr<uint8_t[]> weights_u8;
  hwy::AlignedUniquePtr<Activations<Config, kPrefillBatchSize>> prefill;
  hwy::AlignedUniquePtr<Activations<Config, 1>> state;
};

template <class TConfig>
std::string TokenString(GemmaImpl<TConfig>& gemma, int token) {
  std::string token_str;
  gemma.Tokenizer()->Decode({token}, &token_str);
  return "'" + std::regex_replace(token_str, std::regex("\n"), "\\n") + "'";
}

}  // namespace gcpp
#endif  // GEMMA_ONCE

// SIMD code, compiled once per target.
HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

template <size_t kBatchSize, typename LayerT, class TConfig>
HWY_NOINLINE void GriffinRecurrent(
    size_t batch_start, size_t num_tokens, size_t layer,
    Activations<TConfig, kBatchSize>& activations, const LayerT* layer_weights,
    KVCache& kv_cache, hwy::ThreadPool& pool) {
  PROFILER_ZONE("Gen.Griffin");
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  HWY_DASSERT(num_tokens <= kBatchSize);
  static constexpr size_t kModelDim =
      gcpp::Activations<TConfig, kBatchSize>::kModelDim;
  static constexpr size_t kConv1dWidth = TConfig::kConv1dWidth;
  static constexpr size_t kHeads = TConfig::kHeads;

  // X / Y linear layers.
  for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
    const size_t batch_offset = batch_idx * kModelDim;
    float* HWY_RESTRICT y = activations.griffin_y.data() + batch_offset;
    float* HWY_RESTRICT x = activations.griffin_x.data() + batch_offset;
    TwoMatVecAdd<kModelDim, kModelDim>(
        layer_weights->griffin.linear_x_w, layer_weights->griffin.linear_y_w, 0,
        activations.pre_att_rms_out.data() + batch_offset,
        /*add0=*/layer_weights->griffin.linear_x_biases.data(),
        /*add1=*/layer_weights->griffin.linear_y_biases.data(), /*out0=*/x,
        /*out1=*/y, pool);
    Gelu(y, kModelDim);
  }

  // Conv1D.
  for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
    const size_t batch_offset = batch_idx * kModelDim;
    const size_t pos = batch_start + batch_idx;
    float* HWY_RESTRICT x = activations.griffin_x.data() + batch_offset;
    HWY_FULL(float) df;
    HWY_DASSERT(kModelDim % Lanes(df) == 0);
    const size_t layer_offset = layer * kModelDim * (kConv1dWidth - 1);

    // cache[i] = input at time t-i.
    float* HWY_RESTRICT cache[HWY_MAX(kConv1dWidth, 1)];
    cache[0] = x;
    for (size_t i = 1; i < kConv1dWidth; i++) {
      cache[i] =
          kv_cache.conv1d_cache.get() + layer_offset +
          ((pos + kConv1dWidth - 1 - i) % (kConv1dWidth - 1)) * kModelDim;
    }
    for (size_t i = 0; i < kModelDim; i += Lanes(df)) {
      auto xv = hn::Load(df, x + i);
      auto accum0 =
          hn::Load(df, layer_weights->griffin.conv_biases.data() + i);
      auto accum1 = hn::Zero(df);
      static_assert(kConv1dWidth % 2 == 0, "Conv width must be even");
      for (size_t l = 0; 2 * l < kConv1dWidth; l++) {
        auto wv0 = hn::Load(df, layer_weights->griffin.conv_w.data() +
                                (kConv1dWidth - 1 - 2 * l) * kModelDim + i);
        auto wv1 = hn::Load(df, layer_weights->griffin.conv_w.data() +
                                (kConv1dWidth - 2 - 2 * l) * kModelDim + i);
        accum0 = hn::MulAdd(wv0, hn::Load(df, cache[l * 2] + i), accum0);
        accum1 = hn::MulAdd(wv1, hn::Load(df, cache[l * 2 + 1] + i), accum1);
      }
      hn::Store(hn::Add(accum0, accum1), df, x + i);
      hn::Store(xv, df, cache[HWY_MAX(kConv1dWidth, 1) - 1] + i);
    }
  }

  // RGLRU
  for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
    const size_t batch_offset = batch_idx * kModelDim;
    const size_t pos = batch_start + batch_idx;
    float* HWY_RESTRICT y = activations.griffin_y.data() + batch_offset;
    float* HWY_RESTRICT x = activations.griffin_x.data() + batch_offset;
    float* HWY_RESTRICT gate_x =
        activations.griffin_gate_x.data() + batch_offset;
    float* HWY_RESTRICT a =
        activations.griffin_multiplier.data() + batch_offset;
    float* HWY_RESTRICT rnn_state =
        kv_cache.rglru_cache.get() + layer * kModelDim;

    pool.Run(0, kHeads, [&](const uint64_t head, size_t /*thread*/) HWY_ATTR {
      constexpr size_t kHeadDim = kModelDim / kHeads;
      constexpr size_t kMatrixSize = kHeadDim * kHeadDim;
      size_t head_offset = head * kHeadDim;
      TwoOfsMatVecAddLoop<kHeadDim, kHeadDim>(
          layer_weights->griffin.gate_w, kMatrixSize * head,
          kMatrixSize * (kHeads + head), x + head_offset,
          /*add0=*/layer_weights->griffin.gate_biases.data() + head_offset,
          /*add1=*/layer_weights->griffin.gate_biases.data() + kModelDim +
              head_offset,
          /*out0=*/gate_x + head_offset, /*out1=*/a + head_offset);
      Sigmoid(gate_x + head_offset, kHeadDim);
      Sigmoid(a + head_offset, kHeadDim);
      const auto fn_mul = [](D d, hn::Vec<D> x, hn::Vec<D> gate_x)
                          HWY_ATTR { return hn::Mul(x, gate_x); };
      hn::Transform1(D(), a + head_offset, kHeadDim,
                     layer_weights->griffin.a.data() + head_offset, fn_mul);
      hn::Transform1(D(), x + head_offset, kHeadDim, gate_x + head_offset,
                     fn_mul);
      // RNN scan
      HWY_FULL(float) df;
      HWY_DASSERT(kHeadDim % Lanes(df) == 0);
      for (size_t i = 0; i < kHeadDim; i += Lanes(df)) {
        auto log_a = hn::Load(df, a + head_offset + i);
        auto gated_x = hn::Load(df, x + head_offset + i);
        auto rnn = hn::Load(df, rnn_state + head_offset + i);
        auto a = hn::Exp(df, log_a);
        auto x_multiplier = hn::Sqrt(hn::NegMulAdd(a, a, hn::Set(df, 1.0)));
        if (pos == 0) {
          x_multiplier = hn::Set(df, 1.0);
        }
        auto new_x = hn::MulAdd(x_multiplier, gated_x, hn::Mul(a, rnn));
        hn::Store(new_x, df, rnn_state + head_offset + i);

        // Join branches.
        auto yv = hn::Load(df, y + head_offset + i);
        auto pre_out = hn::Mul(yv, new_x);
        hn::Store(pre_out, df, x + head_offset + i);
      }
    });
  }

  // Final linear layer.
  for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
    const size_t batch_offset = batch_idx * kModelDim;
    float* HWY_RESTRICT x = activations.griffin_x.data() + batch_offset;
    float* out_ptr = activations.att_post2.data() + batch_idx * kModelDim;
    MatVecAdd<kModelDim, kModelDim>(
        layer_weights->griffin.linear_out_w, 0, x,
        layer_weights->griffin.linear_out_biases.data(),
        activations.even_odd.data(), out_ptr, pool);
  }
}

template <size_t kBatchSize, typename LayerT, class TConfig>
HWY_NOINLINE void Attention(size_t batch_start, size_t num_tokens, size_t layer,
                            Activations<TConfig, kBatchSize>& activations,
                            const LayerT* layer_weights, KVCache& kv_cache,
                            hwy::ThreadPool& pool) {
  PROFILER_ZONE("Gen.Attention");
  HWY_DASSERT(num_tokens <= kBatchSize);
  static constexpr size_t kQKVDim = gcpp::Activations<TConfig, 1>::kQKVDim;
  static constexpr size_t kCachePosSize =
      gcpp::Activations<TConfig, kBatchSize>::kCachePosSize;
  static constexpr size_t kCacheLayerSize =
      gcpp::Activations<TConfig, kBatchSize>::kCacheLayerSize;
  static constexpr size_t kModelDim =
      gcpp::Activations<TConfig, kBatchSize>::kModelDim;
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kKVHeads = TConfig::kKVHeads;
  static constexpr size_t kSeqLen = TConfig::kSeqLen;
  static const float kQueryScale =
      static_cast<float>(1.0 / sqrt(static_cast<double>(kQKVDim)));

  auto Attn = [&](float* q, uint64_t head, size_t head_offset, size_t batch_idx,
                  size_t thread) HWY_ATTR {
    const size_t pos = batch_start + batch_idx;
    // Calculate scores
    float* HWY_RESTRICT head_att = activations.att.data() +
                                   head * kSeqLen +
                                   batch_idx * kHeads * kSeqLen;

    Rope(q, TConfig::kUseHalfRope ? kQKVDim / 2 : kQKVDim, pos);
    MulByConst(kQueryScale, q, kQKVDim);

    // Compute Q dot K scores
    const size_t start_pos = pos - std::min(kSeqLen - 1, pos);
    for (size_t pos2 = start_pos; pos2 <= pos; ++pos2) {
      const size_t cache_pos = pos2 % (kSeqLen + kPrefillBatchSize);
      const size_t kv_offset = cache_pos * kCachePosSize +
                               layer * kCacheLayerSize + head_offset;
      const float* HWY_RESTRICT k2 = kv_cache.kv_cache.get() + kv_offset;
      const float score = Dot(q, k2, kQKVDim);
      head_att[pos2 % kSeqLen] = score;
    }
    Softmax(head_att, std::min(pos + 1, kSeqLen));

    // Weighted summation
    float* HWY_RESTRICT att_out = activations.att_out.data() + head * kQKVDim +
                                  batch_idx * kHeads * kQKVDim;
    hwy::ZeroBytes(att_out, kQKVDim * sizeof(*att_out));
    for (size_t pos2 = start_pos; pos2 <= pos; ++pos2) {
      const size_t cache_pos = pos2 % (kSeqLen + kPrefillBatchSize);
      const size_t kv_offset = cache_pos * kCachePosSize +
                               layer * kCacheLayerSize + head_offset;
      float* HWY_RESTRICT v2 = kv_cache.kv_cache.get() + kv_offset + kQKVDim;
      MulByConstAndAdd(head_att[pos2 % kSeqLen], v2, att_out, kQKVDim);
    }
  };

  for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
    const float* x = activations.pre_att_rms_out.data() + batch_idx * kModelDim;
    // QKV projections:
    if constexpr (kHeads == kKVHeads) {
      // Multi-Head Attention calculates qkv using q as scratch space.
      static_assert(TConfig::kInterleaveQKV);
      float* HWY_RESTRICT qkv =
          activations.q.data() + batch_idx * kHeads * kQKVDim * 3;
      MatVec<kHeads * kQKVDim * 3, kModelDim>(layer_weights->qkv_einsum_w, 0, x,
                                              activations.even_odd.data(), qkv,
                                              pool);
    } else {
      const size_t pos = batch_start + batch_idx;
      float* HWY_RESTRICT q =
          activations.q.data() + batch_idx * kHeads * kQKVDim;
      MatVec<kHeads * kQKVDim, kModelDim>(layer_weights->qkv_einsum_w, 0, x,
                                          activations.even_odd.data(), q, pool);

      const size_t cache_pos = pos % (kSeqLen + kPrefillBatchSize);
      const size_t kv_offset =
          cache_pos * kCachePosSize + layer * kCacheLayerSize;
      float* HWY_RESTRICT kv = kv_cache.kv_cache.get() + kv_offset;
      MatVec<kKVHeads * kQKVDim * 2, kModelDim>(
          layer_weights->qkv_einsum_w, kHeads * kQKVDim * kModelDim, x,
          activations.even_odd.data(), kv, pool);
    }
  }

  // Positional encodings for k:
  const size_t num_kv_tasks = kKVHeads * num_tokens;
  pool.Run(0, num_kv_tasks, [&](const uint64_t task, size_t thread) HWY_ATTR {
    const size_t head = task % kKVHeads;
    const size_t batch_idx = task / kKVHeads;
    const size_t pos = batch_start + batch_idx;
    const size_t cache_pos = pos % (kSeqLen + kPrefillBatchSize);
    const size_t kv_offset = cache_pos * kCachePosSize +
                             layer * kCacheLayerSize + head * kQKVDim * 2;
    float* HWY_RESTRICT kv = kv_cache.kv_cache.get() + kv_offset;
    if constexpr (kHeads == kKVHeads) {
      // For MHA, copy kv into the KV cache from scratch space (see above).
      const float* HWY_RESTRICT q =
          activations.q.data() + (batch_idx * kHeads + head) * kQKVDim * 3;
      memcpy(kv, q + kQKVDim, 2 * kQKVDim * sizeof(float));
    }
    Rope(kv, TConfig::kUseHalfRope ? kQKVDim / 2 : kQKVDim, pos);
  });

  static_assert((TConfig::kHeads % TConfig::kKVHeads) == 0,
                "query heads must be a multiple of key-value heads");
  static constexpr size_t kGroupHeads = TConfig::kHeads / TConfig::kKVHeads;
  static constexpr size_t kQOffsetScale = (kHeads == kKVHeads) ? 3 : 1;
  const size_t num_q_tasks = kHeads * num_tokens;
  pool.Run(0, num_q_tasks, [&](const uint64_t task, size_t thread) HWY_ATTR {
    const size_t head = task % kHeads;
    const size_t batch_idx = task / kHeads;
    const size_t head_offset = (head / kGroupHeads) * kQKVDim * 2;
    float* HWY_RESTRICT q = activations.q.data() + (batch_idx * kHeads + head) *
                                                       kQKVDim * kQOffsetScale;
    Attn(q, head, head_offset, batch_idx, thread);
  });

  for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
    // TODO(szabadka) Use a single MatVecAdd like in GriffinRecurrent() after
    // rearranging the weights.
    float* HWY_RESTRICT att_out =
        activations.att_out.data() + batch_idx * kHeads * kQKVDim;
    float* HWY_RESTRICT layer_out =
        activations.att_post2.data() + batch_idx * kModelDim;
    MatVecT</*kAdd=*/TConfig::kSoftmaxAttnOutputBiases, kModelDim, kQKVDim>(
        layer_weights->attn_vec_einsum_w, 0, att_out,
        layer_weights->attention_output_biases.data(),
        activations.even_odd.data(), layer_out, pool);
    for (size_t head = 1; head < kHeads; ++head) {
      float* HWY_RESTRICT head_out =
          activations.att_post1.data() + head * kBatchSize * kModelDim;
      MatVec<kModelDim, kQKVDim>(
          layer_weights->attn_vec_einsum_w, head * kModelDim * kQKVDim,
          att_out + head * kQKVDim,
          activations.even_odd.data(), head_out, pool);
      AddFrom(head_out, layer_out, kModelDim);
    }
  }
}

template <size_t kBatchSize, typename LayerT, typename TConfig>
HWY_NOINLINE void FFW(Activations<TConfig, kBatchSize>& activations,
                      size_t num_tokens, const LayerT* layer_weights,
                      hwy::ThreadPool& pool) {
  HWY_DASSERT(num_tokens <= kBatchSize);
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;
  float* HWY_RESTRICT even_odd = activations.even_odd.data();

  for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
    const size_t hidden_offset = batch_idx * kFFHiddenDim * 2;
    PROFILER_ZONE("Gen.FFW.GatedGELU");
    const hwy::bfloat16_t* HWY_RESTRICT vec =
        activations.bf_pre_ffw_rms_out.data() + batch_idx * kModelDim;
    float* HWY_RESTRICT out = activations.ffw_hidden.data() + hidden_offset;
    float* HWY_RESTRICT out_mul = out + kFFHiddenDim;

    // Same matrix, first and second half of rows. Could fuse into one MatVec.
    MatVecT</*kAdd=*/TConfig::kFFBiases, kFFHiddenDim, kModelDim>(
        layer_weights->gating_einsum_w, kFFHiddenDim * kModelDim, vec,
        TConfig::kFFBiases ?
        layer_weights->ffw_gating_biases.data() + kFFHiddenDim : nullptr,
        even_odd, out_mul, pool);
    // Gate, will go through the nonlinearity.
    MatVecT</*kAdd=*/TConfig::kFFBiases, kFFHiddenDim, kModelDim>(
        layer_weights->gating_einsum_w, 0, vec,
        layer_weights->ffw_gating_biases.data(), even_odd, out, pool);

    namespace hn = hwy::HWY_NAMESPACE;
    using DF = hn::ScalableTag<float>;
    using VF = hn::Vec<DF>;
    hn::Transform1(DF(), out, kFFHiddenDim, out_mul,
                   [](DF df, VF v, VF mul)
                       HWY_ATTR { return hn::Mul(mul, Gelu(df, v)); });
  }

  for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
    PROFILER_ZONE("Gen.FFW\\GatedGELU");
    const size_t hidden_offset = batch_idx * kFFHiddenDim * 2;
    MatVecT</*kAdd=*/TConfig::kFFBiases, kModelDim, kFFHiddenDim>(
        layer_weights->linear_w, 0,
        activations.ffw_hidden.data() + hidden_offset,
        layer_weights->ffw_output_biases.data(), even_odd,
        activations.ffw_out.data() + batch_idx * kModelDim, pool);
  }
}

template <size_t kBatchSize, typename WeightArrayT, typename TConfig>
HWY_NOINLINE void Prefill(const int* tokens, size_t num_tokens, size_t pos,
                          const WeightArrayT& weights,
                          Activations<TConfig, kBatchSize>& activations,
                          KVCache& kv_cache, hwy::ThreadPool& pool) {
  PROFILER_ZONE("Gen.Prefill\\Att\\FFW");
  static constexpr size_t kModelDim = TConfig::kModelDim;
  GEMMA_CONSTEXPR_EMBSCALING const float kEmbScaling =
      EmbeddingScaling<TConfig>();

  pool.Run(
      0, num_tokens, [&](const uint64_t token_idx, size_t /*thread*/) HWY_ATTR {
        const int token = tokens[token_idx];
        HWY_ASSERT(token >= 0);
        HWY_ASSERT(token < TConfig::kVocabSize);
        Decompress(weights.embedder_input_embedding, token * kModelDim,
                   activations.x.data() + token_idx * kModelDim, kModelDim);
        MulByConst(kEmbScaling, activations.x.data() + token_idx * kModelDim,
                   kModelDim);
        if constexpr (TConfig::kAbsolutePE) {
          AddAbsolutePositionalEmbeddings(
              activations.x.data() + token_idx * kModelDim, TConfig::kModelDim,
              pos);
        };
      });

  for (size_t layer = 0; layer < TConfig::kLayers; ++layer) {
    auto type = TConfig::kLayerConfig[layer];
    const auto* layer_weights = weights.GetLayer(layer);
    size_t layer_of_type =
        NumLayersOfTypeBefore(TConfig::kLayerConfig, type, layer);

    for (size_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
      RMSNorm(activations.x.data() + token_idx * kModelDim,
              layer_weights->pre_attention_norm_scale.data(),
              activations.pre_att_rms_out.data() + token_idx * kModelDim,
              kModelDim);
    }
    if (type == LayerAttentionType::kGemma) {
      Attention<kBatchSize>(pos, num_tokens, layer_of_type, activations,
                            layer_weights, kv_cache, pool);
    } else {
      GriffinRecurrent<kBatchSize>(pos, num_tokens, layer_of_type, activations,
                                   layer_weights, kv_cache, pool);
    }

    pool.Run(0, num_tokens, [&](const uint64_t token_idx,
                                size_t /*thread*/) HWY_ATTR {
      if (TConfig::kPostNormScale) {
        RMSNormInplace(layer_weights->post_attention_norm_scale.data(),
                       activations.att_post2.data() + token_idx * kModelDim,
                       kModelDim);
      }
      AddFrom(activations.att_post2.data() + token_idx * kModelDim,
              activations.x.data() + token_idx * kModelDim, kModelDim);
      RMSNorm(activations.x.data() + token_idx * kModelDim,
              layer_weights->pre_ffw_norm_scale.data(),
              activations.bf_pre_ffw_rms_out.data() + token_idx * kModelDim,
              kModelDim);
    });
    FFW<kBatchSize>(activations, num_tokens, layer_weights, pool);
    for (size_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
      if (TConfig::kPostNormScale) {
        RMSNormInplace(layer_weights->post_ffw_norm_scale.data(),
                       activations.ffw_out.data() + token_idx * kModelDim,
                       kModelDim);
      }
      AddFrom(activations.ffw_out.data() + token_idx * kModelDim,
              activations.x.data() + token_idx * kModelDim, kModelDim);
    }
  }  // foreach layer

  pool.Run(
      0, num_tokens, [&](const uint64_t token_idx, size_t /*thread*/) HWY_ATTR {
        RMSNormInplace(weights.final_norm_scale.data(),
                       activations.x.data() + token_idx * kModelDim, kModelDim);
      });
}

// n = 1 specialization
template <typename WeightArrayT, class TConfig>
void Transformer(int token, size_t pos, const WeightArrayT& weights,
                 Activations<TConfig, 1>& activations, KVCache& kv_cache,
                 hwy::ThreadPool& pool, LayersOutputT* layers_output) {
  if (layers_output != nullptr) {
    float token_f = token;
    (*layers_output)(pos, "Tokens", &token_f, 1);
  }
  static constexpr size_t kModelDim = TConfig::kModelDim;
  Decompress(weights.embedder_input_embedding, token * kModelDim,
             activations.x.data(), kModelDim);

  GEMMA_CONSTEXPR_EMBSCALING const float kEmbScaling =
      EmbeddingScaling<TConfig>();
  MulByConst(kEmbScaling, activations.x.data(), kModelDim);
  if constexpr (TConfig::kAbsolutePE) {
    AddAbsolutePositionalEmbeddings(activations.x.data(), TConfig::kModelDim,
                                    pos);
  };
  for (size_t layer = 0; layer < TConfig::kLayers; ++layer) {
    auto type = TConfig::kLayerConfig[layer];
    const auto* layer_weights = weights.GetLayer(layer);
    size_t layer_of_type =
        NumLayersOfTypeBefore(TConfig::kLayerConfig, type, layer);
    RMSNorm(activations.x.data(),
            layer_weights->pre_attention_norm_scale.data(),
            activations.pre_att_rms_out.data(), kModelDim);
    if (type == LayerAttentionType::kGemma) {
      Attention<1>(pos, 1, layer_of_type, activations, layer_weights, kv_cache,
                   pool);
    } else {
      GriffinRecurrent<1>(pos, 1, layer_of_type, activations, layer_weights,
                          kv_cache, pool);
    }
    if (TConfig::kPostNormScale) {
      RMSNormInplace(layer_weights->post_attention_norm_scale.data(),
                     activations.att_post2.data(), kModelDim);
    }
    AddFrom(activations.att_post2.data(), activations.x.data(), kModelDim);
    RMSNorm(activations.x.data(), layer_weights->pre_ffw_norm_scale.data(),
            activations.bf_pre_ffw_rms_out.data(), kModelDim);
    FFW<1>(activations, /* num_tokens = */ 1, layer_weights, pool);
    if (TConfig::kPostNormScale) {
      RMSNormInplace(layer_weights->post_ffw_norm_scale.data(),
                     activations.ffw_out.data(), kModelDim);
    }
    AddFrom(activations.ffw_out.data(), activations.x.data(), kModelDim);
    if (layers_output != nullptr) {
      std::string block_name = "blocks." + std::to_string(layer);
      (*layers_output)(pos, block_name, activations.x.data(), kModelDim);
    }
  }
  RMSNormInplace(weights.final_norm_scale.data(), activations.x.data(),
                 kModelDim);
  if (layers_output != nullptr) {
    (*layers_output)(pos, "final_norm", activations.x.data(), kModelDim);
  }
}

template <class TConfig>
void RangeChecks(size_t& max_tokens, size_t& max_generated_tokens,
                 size_t& prompt_size) {
  if (!TConfig::kUseLocalAttention) {
    if (max_tokens > TConfig::kSeqLen) {
      fprintf(stderr, "WARNING: max_tokens %zu > kSeqLen %d, truncating.\n",
              max_tokens, TConfig::kSeqLen);
      max_tokens = static_cast<size_t>(TConfig::kSeqLen);
    }
  }

  if (max_generated_tokens > max_tokens) {
    fprintf(stderr,
            "WARNING: max_generated_tokens %zu > max_tokens %zu, truncating.\n",
            max_generated_tokens, max_tokens);
    max_generated_tokens = max_tokens - 1;
  }

  if (!TConfig::kUseLocalAttention) {
    if (prompt_size + max_generated_tokens > max_tokens) {
      fprintf(stderr,
              "WARNING: prompt_size %zu + max_generated_tokens %zu > "
              "max_tokens %zu, truncating to ",
              prompt_size, max_generated_tokens, max_tokens);
      prompt_size = std::min(prompt_size, max_tokens - max_generated_tokens);
      fprintf(stderr, "%zu\n", prompt_size);
    }
  }
}

template <class TConfig, template<typename> typename WeightsType>
void GenerateImpl(const WeightsType<TConfig>& weights,
                  Activations<TConfig, kPrefillBatchSize>& prefill_activations,
                  Activations<TConfig, 1>& activations,
                  const RuntimeConfig& runtime_config,
                  const std::vector<int>& prompt, size_t pos, KVCache& kv_cache,
                  hwy::ThreadPool& pool, TimingInfo& timing_info,
                  LayersOutputT* layers_output) {
  static constexpr size_t kVocabSize = TConfig::kVocabSize;
  size_t prompt_size = prompt.size();
  size_t max_tokens = runtime_config.max_tokens;
  size_t max_generated_tokens = runtime_config.max_generated_tokens;
  RangeChecks<TConfig>(max_tokens, max_generated_tokens, prompt_size);
  if (pos >= max_tokens) {
    fprintf(stderr, "Warning: pos %zu >= max_tokens %zu, aborting.\n", pos,
            max_tokens);
    return;
  }
  HWY_ASSERT(prompt_size > 0);

  // pos indexes the KV cache. In the first turn of a chat, pos = 0.
  //
  // After the first turn, pos gets passed in with > 0 corresponding to the
  // current token position in the KV cache.
  //
  // pos_offset keeps track of the relative position within the turn, starting
  // at 0 each turn. During prefill, pos_offset corresponds to the index into
  // the prompt vector.
  //
  // In single-turn (non-chat) usage, pos and pos_offset start at 0 and are
  // always equal.
  size_t pos_offset = 0;  // offset relative to pos
  const double prefill_start = hwy::platform::Now();

  // Prefill stops before prompt_size - 1 since the last prompt token is the
  // first input token for generation.
  while (pos_offset < prompt_size - 1) {
    const size_t batch_size =
        std::min(kPrefillBatchSize, prompt_size - 1 - pos_offset);
    HWY_DASSERT(batch_size <= kPrefillBatchSize);
    HWY_DASSERT(pos_offset + batch_size <= prompt_size - 1);
    const int* batch_tokens = prompt.data() + pos_offset;
    Prefill<kPrefillBatchSize>(batch_tokens, batch_size, pos, weights,
                               prefill_activations, kv_cache, pool);
    for (size_t idx = 0; idx < batch_size; ++idx) {
      if (!runtime_config.stream_token(batch_tokens[idx], 0.0f)) return;
    }
    pos += batch_size;
    pos_offset += batch_size;
  }

  if (runtime_config.verbosity >= 2) {
    const double prefill_end = hwy::platform::Now();
    timing_info.prefill_tok_sec =
        static_cast<double>(pos_offset) / (prefill_end - prefill_start);
  }

  const double gen_start = hwy::platform::Now();

  HWY_DASSERT(pos_offset == prompt_size - 1);

  size_t pos_gen_start = pos_offset;
  int token = prompt.at(pos_offset);
  runtime_config.stream_token(token, 0);
  for (size_t generate_pos = 0;
       pos < max_tokens && generate_pos < max_generated_tokens;
       ++pos, ++pos_offset, ++generate_pos) {
    const bool is_generating_phase = pos_offset >= prompt_size - 1;
    Transformer(token, pos, weights, activations, kv_cache, pool,
                layers_output);
    float* final_activation = activations.x.data();
    // The condition below is always true if we are doing Prefill above.
    // We keep it here for clarity so that the code is correct even if Prefill
    // is disabled.
    if (is_generating_phase) {
      PROFILER_ZONE("Gen.Embedding");
      // Generation phase
      MatVec<kVocabSize, TConfig::kModelDim>(
          weights.embedder_input_embedding, 0, final_activation,
          activations.even_odd.data(), activations.logits.data(), pool);
      // Barrier: must have all logits so we can subtract max.
      Softmax(activations.logits.data(), kVocabSize);
      token = SampleTopK<TConfig::kTopK>(
          activations.logits.data(), kVocabSize, *runtime_config.gen,
          runtime_config.temperature, runtime_config.accept_token);
      if (!runtime_config.stream_token(token, activations.logits[token])) {
        token = runtime_config.eos_id;
      }
      if (generate_pos == 0) {
        timing_info.time_to_first_token = hwy::platform::Now() - gen_start;
      }
    } else {
      // We would take this branch if we were not doing Prefill but would
      // process the tokens of the prompt one at a time.
      token = prompt.at(pos_offset + 1);
      if (!runtime_config.stream_token(token, 0)) {
        token = runtime_config.eos_id;
      }
    }
    if (token == runtime_config.eos_id) {
      if (runtime_config.verbosity >= 2) {
        const double gen_end = hwy::platform::Now();
        timing_info.gen_tok_sec =
            static_cast<double>(pos_offset - pos_gen_start) /
            (gen_end - gen_start);
      }
      break;
    }
  }
}

template <class TConfig>
void GenerateImpl(GemmaImpl<TConfig>& gemma,
                  const RuntimeConfig& runtime_config,
                  const std::vector<int>& prompt, size_t pos, KVCache& kv_cache,
                  hwy::ThreadPool& pool, TimingInfo& timing_info,
                  LayersOutputT* layers_output) {
  const WeightsT<TConfig>& weights =
      *reinterpret_cast<WeightsT<TConfig>*>(gemma.weights_u8.get());
  GenerateImpl<TConfig, WeightsT>(
      weights, *gemma.prefill.get(), *gemma.state.get(), runtime_config, prompt,
      pos, kv_cache, pool, timing_info, layers_output);
}

template <class TConfig>
void GenerateImpl(const ByteStorageT& weights_u8,
                  ByteStorageT& inference_state_u8,
                  const RuntimeConfig& runtime_config,
                  const std::vector<int>& prompt, size_t pos,
                  KVCache& kv_cache, hwy::ThreadPool& pool,
                  TimingInfo& timing_info, LayersOutputT* layers_output) {
  const WeightsF<TConfig>& weights =
      *reinterpret_cast<const WeightsF<TConfig>*>(weights_u8.get());
  InferenceState<TConfig>& inference_state =
      *reinterpret_cast<InferenceState<TConfig>*>(inference_state_u8.get());
  GenerateImpl<TConfig, WeightsF>(
      weights, inference_state.prefill, inference_state.state, runtime_config,
      prompt, pos, kv_cache, pool, timing_info, layers_output);
}

void GenerateImplT(Model model, const ByteStorageT& weights_u8,
                   ByteStorageT& inference_state_u8,
                   const RuntimeConfig& runtime_config,
                   const std::vector<int>& prompt, size_t pos,
                   KVCache& kv_cache, hwy::ThreadPool& pool,
                   TimingInfo& timing_info, LayersOutputT* layers_output) {
  switch (model) {
    case Model::GEMMA_2B:
      GenerateImpl<ConfigGemma2B>(
          weights_u8, inference_state_u8, runtime_config, prompt, pos, kv_cache,
          pool, timing_info, layers_output);
      break;
    case Model::GEMMA_TINY:
      GenerateImpl<ConfigGemmaTiny>(
          weights_u8, inference_state_u8, runtime_config, prompt, pos, kv_cache,
          pool, timing_info, layers_output);
      break;
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

#define TOKEN(token_id) TokenString(gemma, token_id).c_str()

template <class TConfig>
void LogTopK(GemmaImpl<TConfig>& gemma, float* logits, float* dist, size_t len,
             size_t k) {
  std::vector<std::pair<float, int>> sorted(len);
  for (size_t i = 0; i < len; ++i) {
    sorted[i] = std::make_pair(dist[i], static_cast<int>(i));
  }
  std::sort(sorted.begin(), sorted.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
              if (a.first != b.first) {
                return a.first > b.first;
              }
              return a.second < b.second;
            });
  for (size_t i = 0; i < k; ++i) {
    printf("  [#%-2d token %6d = %-12s  %.2e  %f]\n", static_cast<int>(i + 1),
           sorted[i].second, TOKEN(sorted[i].second), sorted[i].first,
           logits[sorted[i].second]);
  }
}

template <class TConfig>
float ComputeCrossEntropyImpl(GemmaImpl<TConfig>& gemma, size_t max_tokens,
                              const std::vector<int>& prompt, KVCache& kv_cache,
                              hwy::ThreadPool& pool, int verbosity) {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kVocabSize = TConfig::kVocabSize;
  Activations<TConfig, 1>& activations = *gemma.state.get();
  const WeightsT<TConfig>& weights =
      *reinterpret_cast<const WeightsT<TConfig>*>(gemma.weights_u8.get());
  std::vector<float> logits(kVocabSize);
  Softmax(activations.logits.data(), kVocabSize);
  float total_entropy = 0.0f;
  for (size_t pos = 0; pos < max_tokens && pos < prompt.size(); ++pos) {
    if (verbosity >= 4) {
      LogTopK(gemma, logits.data(), activations.logits.data(), kVocabSize, 10);
    }
    const int token = prompt[pos];
    const float prob = activations.logits[token];
    if (verbosity >= 3) {
      printf("pos %4zu token %6d = %-12s  %.10e  %14.10f bits\n", pos, token,
             TOKEN(token), prob, -std::log(prob) / std::log(2.0));
    }
    total_entropy -= std::max(std::log(prob), -64.0f);
    if (verbosity >= 2 && pos % 100 == 99) {
      printf("Processed %zu tokens, cross-entropy per token: %f\n", pos + 1,
             total_entropy / std::log(2.0) / (pos + 1));
    }
    Transformer(token, pos, weights, activations, kv_cache, pool,
                /*layers_output=*/nullptr);
    MatVec<kVocabSize, kModelDim>(
        weights.embedder_input_embedding, 0, activations.x.data(),
        activations.even_odd.data(), activations.logits.data(), pool);
    LogitsSoftCap(30.0f, activations.logits.data(), kVocabSize);
    memcpy(logits.data(), activations.logits.data(),
           kVocabSize * sizeof(logits[0]));
    Softmax(activations.logits.data(), kVocabSize);
  }
  return total_entropy / std::log(2.0);
}

#undef TOKEN

// Calls func(name, float*, CompressedArray&) for each tensor. float* is null
// if weights = null, which happens during the first call where we attempt to
// load from cache.
//
// This avoids repeating the list of tensors between loading and compressing.
template <class TConfig, class Func>
void ForEachTensor(const WeightsF<TConfig>* weights,
                   CompressedWeights<TConfig>& c_weights, Func& func) {
  func("c_embedding",
       weights ? weights->embedder_input_embedding.data() : nullptr,
       c_weights.embedder_input_embedding);
  func("c_final_norm", weights ? weights->final_norm_scale.data() : nullptr,
       c_weights.final_norm_scale);

  char name_buf[16];
  for (int layer_idx = 0; layer_idx < TConfig::kLayers; ++layer_idx) {
    auto type = TConfig::kLayerConfig[layer_idx];
    const size_t idx = static_cast<size_t>(layer_idx);
    const LayerF<TConfig>* layer = weights ? weights->GetLayer(idx) : nullptr;
    CompressedLayer<TConfig>* layer_weights = c_weights.GetLayer(idx);

#define CALL_FUNC(name, member)                                \
  snprintf(name_buf, sizeof(name_buf), name "_%d", layer_idx); \
  func(name_buf, layer ? layer->member.data() : nullptr, layer_weights->member)

    CALL_FUNC("pre_ff_ns", pre_ffw_norm_scale);
    CALL_FUNC("gating_ein", gating_einsum_w);
    CALL_FUNC("linear_w", linear_w);
    if (type == LayerAttentionType::kGemma) {
      CALL_FUNC("qkv_ein", qkv_einsum_w);
      CALL_FUNC("att_ein", attn_vec_einsum_w);
    } else {
      CALL_FUNC("gr_lin_x_w", griffin.linear_x_w);
      CALL_FUNC("gr_lin_x_b", griffin.linear_x_biases);
      CALL_FUNC("gr_lin_y_w", griffin.linear_y_w);
      CALL_FUNC("gr_lin_y_b", griffin.linear_y_biases);
      CALL_FUNC("gr_lin_out_w", griffin.linear_out_w);
      CALL_FUNC("gr_lin_out_b", griffin.linear_out_biases);
      CALL_FUNC("gr_conv_w", griffin.conv_w);
      CALL_FUNC("gr_conv_b", griffin.conv_biases);
      CALL_FUNC("gr_gate_w", griffin.gate_w);
      CALL_FUNC("gr_gate_b", griffin.gate_biases);
      CALL_FUNC("gr_a", griffin.a);
    }
    CALL_FUNC("pre_att_ns", pre_attention_norm_scale);
    if (TConfig::kPostNormScale) {
      CALL_FUNC("post_att_ns", post_attention_norm_scale);
      CALL_FUNC("post_ff_ns", post_ffw_norm_scale);
    }

    if (TConfig::kFFBiases) {
      CALL_FUNC("ffw_gat_b", ffw_gating_biases);
      CALL_FUNC("ffw_out_b", ffw_output_biases);
    }

    if (TConfig::kSoftmaxAttnOutputBiases &&
        type == LayerAttentionType::kGemma) {
      CALL_FUNC("attn_ob", attention_output_biases);
    }
#undef CALL_FUNC
  }
}

template <class TConfig>
hwy::AlignedFreeUniquePtr<uint8_t[]> LoadCompressedWeights(
    const Path& weights, hwy::ThreadPool& pool) {
  PROFILER_ZONE("Startup.LoadCompressedWeights");
  if (!weights.Exists()) {
    HWY_ABORT("The model weights file '%s' does not exist.",
              weights.path.c_str());
  }

  // Allocate compressed weights.
  using CWeights = CompressedWeights<TConfig>;
  hwy::AlignedFreeUniquePtr<uint8_t[]> c_weights_u8 =
      hwy::AllocateAligned<uint8_t>(sizeof(CWeights));
  CWeights* c_weights = reinterpret_cast<CWeights*>(c_weights_u8.get());
  new (&c_weights->c_layer_ptrs) CompressedLayerPointers<TConfig>(pool);

  std::array<float, TConfig::kNumTensorScales> scales;
  CacheLoader loader(weights);
  ForEachTensor<TConfig>(nullptr, *c_weights, loader);
  loader.LoadScales(scales.data(), scales.size());
  if (!loader.ReadAll(pool)) {
    HWY_ABORT("Failed to load model weights.");
  }
  if (TConfig::kNumTensorScales > 0) {
    size_t scale_pos = 0;
    for (int layer_idx = 0; layer_idx < TConfig::kLayers; ++layer_idx) {
      auto type = TConfig::kLayerConfig[layer_idx];
      const size_t idx = static_cast<size_t>(layer_idx);
      CompressedLayer<TConfig>* layer_weights = c_weights->GetLayer(idx);
      if (type == LayerAttentionType::kGemma) {
        layer_weights->attn_vec_einsum_w.set_scale(scales[scale_pos++]);
        layer_weights->qkv_einsum_w.set_scale(scales[scale_pos++]);
      } else {
        layer_weights->griffin.linear_x_w.set_scale(scales[scale_pos++]);
        layer_weights->griffin.linear_y_w.set_scale(scales[scale_pos++]);
        layer_weights->griffin.linear_out_w.set_scale(scales[scale_pos++]);
        layer_weights->griffin.gate_w.set_scale(scales[scale_pos++]);
      }
      layer_weights->gating_einsum_w.set_scale(scales[scale_pos++]);
      layer_weights->linear_w.set_scale(scales[scale_pos++]);
    }
    HWY_ASSERT(scale_pos == TConfig::kNumTensorScales);
  }
  return c_weights_u8;
}

template <class TConfig>
void CompressWeights(const Path& weights_path,
                     const Path& compressed_weights_path,
                     hwy::ThreadPool& pool) {
  if (!weights_path.Exists()) {
    HWY_ABORT("The model weights file '%s' does not exist.",
              weights_path.path.c_str());
  }

  // Allocate compressed weights.
  using CWeights = CompressedWeights<TConfig>;
  hwy::AlignedFreeUniquePtr<uint8_t[]> c_weights_u8 =
      hwy::AllocateAligned<uint8_t>(sizeof(CWeights));
  CWeights* c_weights = reinterpret_cast<CWeights*>(c_weights_u8.get());
  new (&c_weights->c_layer_ptrs) CompressedLayerPointers<TConfig>(pool);

  // Get weights, compress, and store.
  const bool scale_for_compression = TConfig::kNumTensorScales > 0;
  const hwy::AlignedFreeUniquePtr<uint8_t[]> weights_u8 =
      LoadWeights<TConfig>(weights_path, pool, scale_for_compression);
  WeightsF<TConfig>* weights =
      reinterpret_cast<WeightsF<TConfig>*>(weights_u8.get());
  Compressor compressor(pool);
  ForEachTensor<TConfig>(weights, *c_weights, compressor);
  compressor.AddScales(weights->scales.data(), weights->scales.size());
  compressor.WriteAll(pool, compressed_weights_path);

  weights->layer_ptrs.~LayerPointers<float, TConfig>();
  c_weights->c_layer_ptrs.~CompressedLayerPointers<TConfig>();
}

void CompressWeightsT(gcpp::Model model, const Path& weights,
                      const Path& compressed_weights, hwy::ThreadPool& pool) {
  switch (model) {
    case Model::GEMMA_2B:
      CompressWeights<ConfigGemma2B>(weights, compressed_weights, pool);
      break;
    case Model::GEMMA_7B:
      CompressWeights<ConfigGemma7B>(weights, compressed_weights, pool);
      break;
    case Model::GRIFFIN_2B:
      CompressWeights<ConfigGriffin2B>(weights, compressed_weights, pool);
      break;
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {

HWY_EXPORT(CompressWeightsT);
HWY_EXPORT(GenerateImplT);

KVCache CreateKVCache(size_t size_cache_pos, size_t seq_len,
                      size_t conv1d_cache_size, size_t rglru_cache_size) {
  KVCache kv_cache = {};
  if (size_cache_pos != 0) {
    kv_cache.kv_cache =
        hwy::AllocateAligned<float>(seq_len * size_cache_pos * 2);
  }
  if (conv1d_cache_size != 0) {
    kv_cache.conv1d_cache = hwy::AllocateAligned<float>(conv1d_cache_size);
    hwy::ZeroBytes(kv_cache.conv1d_cache.get(),
                   conv1d_cache_size * sizeof(kv_cache.conv1d_cache[0]));
  }
  if (rglru_cache_size != 0) {
    kv_cache.rglru_cache = hwy::AllocateAligned<float>(rglru_cache_size);
    hwy::ZeroBytes(kv_cache.rglru_cache.get(),
                   rglru_cache_size * sizeof(kv_cache.rglru_cache[0]));
  }
  return kv_cache;
}

template <class Config>
GemmaImpl<Config>::GemmaImpl(
    std::unique_ptr<sentencepiece::SentencePieceProcessor>& tokenizer,
    hwy::AlignedFreeUniquePtr<uint8_t[]>& weights_u8, hwy::ThreadPool& pool)
    : tokenizer(GemmaTokenizerImpl(std::move(tokenizer))),
      weights_u8(std::move(weights_u8)),
      prefill(hwy::MakeUniqueAligned<Activations<Config, kPrefillBatchSize>>()),
      state(hwy::MakeUniqueAligned<Activations<Config, 1>>()) {}

template <typename Config>
void GemmaImpl<Config>::Generate(const RuntimeConfig& runtime_config,
                                 const std::vector<int>& prompt,
                                 size_t start_pos, KVCache& kv_cache,
                                 hwy::ThreadPool& pool, TimingInfo& timing_info,
                                 LayersOutputT* layers_output) {
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(GenerateImpl<Config>)
  (*this, runtime_config, prompt, start_pos, kv_cache, pool, timing_info,
   layers_output);
}

template <typename Config>
float GemmaImpl<Config>::ComputeCrossEntropy(size_t max_tokens,
                                             const std::vector<int>& prompt,
                                             KVCache& kv_cache,
                                             hwy::ThreadPool& pool,
                                             int verbosity) {
  HWY_EXPORT_T(ComputeCrossEntropyT, ComputeCrossEntropyImpl<Config>);
  return HWY_DYNAMIC_DISPATCH_T(ComputeCrossEntropyT)(
      *this, max_tokens, prompt, kv_cache, pool, verbosity);
}

template <class Config>
GemmaImpl<Config>* CreateGemmaImpl(const Path& tokenizer_path,
                                   const Path& weights, hwy::ThreadPool& pool) {
  std::unique_ptr<sentencepiece::SentencePieceProcessor> tokenizer;
  {
    PROFILER_ZONE("Startup.tokenizer");
    tokenizer = std::make_unique<sentencepiece::SentencePieceProcessor>();
    if (!tokenizer->Load(tokenizer_path.path).ok()) {
      HWY_ABORT("Failed to load the tokenizer file.");
    }
  }

  hwy::AlignedFreeUniquePtr<uint8_t[]> weights_u8;
  if constexpr (kWeightsAreCompressed) {
    HWY_EXPORT_T(LoadCompressedWeightsT, LoadCompressedWeights<Config>);
    weights_u8 = HWY_DYNAMIC_DISPATCH_T(LoadCompressedWeightsT)(weights, pool);
  } else {
    weights_u8 = LoadWeights<Config>(weights, pool);
  }
  return new GemmaImpl<Config>(tokenizer, weights_u8, pool);
}

Gemma::Gemma(const Path& tokenizer_path, const Path& weights, Model model_type,
             hwy::ThreadPool& pool) {
  switch (model_type) {
    case Model::GEMMA_2B:
      impl_.reset(CreateGemmaImpl<ConfigGemma2B>(tokenizer_path, weights, pool));
      break;
    case Model::GEMMA_7B:
      impl_.reset(CreateGemmaImpl<ConfigGemma7B>(tokenizer_path, weights, pool));
      break;
    case Model::GRIFFIN_2B:
      impl_.reset(CreateGemmaImpl<ConfigGriffin2B>(tokenizer_path, weights, pool));
      break;
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model_type));
  }
}

Gemma::~Gemma() = default;  // after GemmaInterface is defined

const GemmaTokenizer* Gemma::Tokenizer() const { return impl_->Tokenizer(); }

void GenerateGemma(Gemma& gemma, const RuntimeConfig& runtime_config,
                   const std::vector<int>& prompt, size_t start_pos,
                   KVCache& kv_cache, hwy::ThreadPool& pool,
                   TimingInfo& timing_info,
                   LayersOutputT* layers_output) {
  pool.SetWaitMode(hwy::PoolWaitMode::kSpin);
  gemma.impl_->Generate(runtime_config, prompt, start_pos, kv_cache, pool,
                        timing_info, layers_output);
  pool.SetWaitMode(hwy::PoolWaitMode::kBlock);
}

void GenerateGemma(Model model, const ByteStorageT& weights,
                   ByteStorageT& inference_state,
                   RuntimeConfig runtime_config,
                   const std::vector<int>& prompt, size_t start_pos,
                   KVCache& kv_cache, hwy::ThreadPool& pool,
                   TimingInfo& timing_info) {
  HWY_DYNAMIC_DISPATCH(GenerateImplT)(
      model, weights, inference_state, runtime_config, prompt, start_pos,
      kv_cache, pool, timing_info, /*layers_output=*/nullptr);
}

ByteStorageT LoadWeights(const Path& weights, Model model,
                         hwy::ThreadPool& pool) {
  return HWY_DYNAMIC_DISPATCH(LoadWeightsT)(model, weights, pool);
}

ByteStorageT AllocateInferenceState(Model model) {
  switch (model) {
    case Model::GEMMA_2B:
      return InferenceState<ConfigGemma2B>::Allocate();
    case Model::GEMMA_7B:
      return InferenceState<ConfigGemma7B>::Allocate();
    case Model::GRIFFIN_2B:
      return InferenceState<ConfigGriffin2B>::Allocate();
    case Model::GEMMA_TINY:
      return InferenceState<ConfigGemmaTiny>::Allocate();
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

void CompressWeights(gcpp::Model model, const Path& weights,
                     const Path& compressed_weights, hwy::ThreadPool& pool) {
  HWY_DYNAMIC_DISPATCH(CompressWeightsT)
  (model, weights, compressed_weights, pool);
}

float ComputeCrossEntropy(Gemma& gemma, size_t max_tokens,
                          const std::vector<int>& prompt, KVCache& kv_cache,
                          hwy::ThreadPool& pool, int verbosity) {
  pool.SetWaitMode(hwy::PoolWaitMode::kSpin);
  const float result = gemma.impl_->ComputeCrossEntropy(
      max_tokens, prompt, kv_cache, pool, verbosity);
  pool.SetWaitMode(hwy::PoolWaitMode::kBlock);
  return result;
}

namespace {
constexpr const char* kModelFlags[] = {"2b-pt", "7b-pt", "gr2b-pt",
                                       "2b-it", "7b-it", "gr2b-it",
                                       "tiny"};
constexpr Model kModelTypes[] = {Model::GEMMA_2B,   Model::GEMMA_7B,
                                 Model::GRIFFIN_2B, Model::GEMMA_2B,
                                 Model::GEMMA_7B,   Model::GRIFFIN_2B,
                                 Model::GEMMA_TINY};
constexpr ModelTraining kModelTraining[] = {
    ModelTraining::GEMMA_PT, ModelTraining::GEMMA_PT, ModelTraining::GEMMA_PT,
    ModelTraining::GEMMA_IT, ModelTraining::GEMMA_IT, ModelTraining::GEMMA_IT,
    ModelTraining::GEMMA_IT};
}  // namespace

const char* ParseModelTypeAndTraining(const std::string& model_flag,
                                      Model& model, ModelTraining& training) {
  constexpr size_t kNum = std::end(kModelFlags) - std::begin(kModelFlags);
  static char kErrorMessageBuffer[kNum * 8 + 1024] =
      "Invalid or missing model flag, need to specify one of ";
  for (size_t i = 0; i + 1 < kNum; i++) {
    strcat(kErrorMessageBuffer, kModelFlags[i]);  // NOLINT
    strcat(kErrorMessageBuffer, ", ");            // NOLINT
  }
  strcat(kErrorMessageBuffer, kModelFlags[kNum - 1]);  // NOLINT
  strcat(kErrorMessageBuffer, ".");                    // NOLINT
  std::string model_type_lc = model_flag;
  std::transform(begin(model_type_lc), end(model_type_lc), begin(model_type_lc),
                 [](unsigned char c) { return std::tolower(c); });
  for (size_t i = 0; i < kNum; i++) {
    if (kModelFlags[i] == model_type_lc) {
      model = kModelTypes[i];
      training = kModelTraining[i];
      return nullptr;
    }
  }
  return kErrorMessageBuffer;
}

}  // namespace gcpp
#endif  // HWY_ONCE
