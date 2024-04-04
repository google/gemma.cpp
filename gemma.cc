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
#define HWY_TARGET_INCLUDE "gemma.cc"  // NOLINT
#include "hwy/foreach_target.h"        // IWYU pragma: keep
// Must come after foreach_target.h to avoid redefinition errors.
// copybara:import_next_line:gemma_cpp
#include "compression/compress-inl.h"
// copybara:import_next_line:gemma_cpp
#include "ops.h"
#include "hwy/contrib/matvec/matvec-inl.h"
#include "hwy/highway.h"
#include "hwy/profiler.h"
#include "hwy/timer.h"
// copybara:import_next_line:gemma_cpp
#include "util/args.h"  // Path

// Non-SIMD includes and types. Note that HWY_ONCE is only true on the last
// compile pass, whereas we want this defined in the first.
#ifndef GEMMA_ONCE
#define GEMMA_ONCE

#include <math.h>  // sqrtf
#include <stddef.h>
#include <stdio.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>  // NOLINT
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

// copybara:import_next_line:gemma_cpp
#include "compression/compress.h"
// copybara:import_next_line:gemma_cpp
#include "configs.h"
// copybara:import_next_line:gemma_cpp
#include "gemma.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
// copybara:import_next_line:sentencepiece
#include "src/sentencepiece_processor.h"

// Setting this to true disables fread() calls that read the model file.
constexpr bool kDryRunFread = false;

// Setting this to false will load and use uncompressed weights.
constexpr bool kWeightsAreCompressed = true;

namespace gcpp {

template <class TConfig>
struct Layer {
  Layer() = default;
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kKVHeads = TConfig::kKVHeads;
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;
  static constexpr size_t kAttVecEinsumWSize = kHeads * kQKVDim * kModelDim;
  static constexpr size_t kQKVEinsumWSize =
      (kHeads + 2 * kKVHeads) * kQKVDim * kModelDim;
  // 2x for (gelu gating vector, gated vector)
  static constexpr size_t kGatingEinsumWSize = 2 * kFFHiddenDim * kModelDim;

  std::array<float, kAttVecEinsumWSize> attn_vec_einsum_w;
  std::array<float, kQKVEinsumWSize> qkv_einsum_w;
  std::array<float, kGatingEinsumWSize> gating_einsum_w;
  std::array<float, kModelDim * kFFHiddenDim> linear_w;
  std::array<float, kModelDim> pre_attention_norm_scale;
  std::array<float, kModelDim> pre_ffw_norm_scale;
};

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

// Array instead of single large allocation for parallel mem init. Split out of
// Weights so that only these pointers are initialized.
template <class TConfig>
struct LayerPointers {
  explicit LayerPointers(hwy::ThreadPool& pool) {
    pool.Run(0, TConfig::kLayers, [this](uint64_t task, size_t /*thread*/) {
      this->layers[task] = hwy::AllocateAligned<Layer<TConfig>>(1);
    });
  }

  using TLayer = Layer<TConfig>;
  std::array<hwy::AlignedFreeUniquePtr<TLayer[]>, TConfig::kLayers> layers;
};

template <class TConfig>
struct Weights {
  // No ctor/dtor, allocated via AllocateAligned.

  std::array<float, TConfig::kVocabSize * TConfig::kModelDim>
      embedder_input_embedding;

  std::array<float, TConfig::kModelDim> final_norm_scale;

  LayerPointers<TConfig> layer_ptrs;

  std::array<float, TConfig::kNumTensorScales> scales;

  const Layer<TConfig>* GetLayer(size_t layer) const {
    return layer_ptrs.layers[layer].get();
  }
  Layer<TConfig>* GetLayer(size_t layer) {
    return layer_ptrs.layers[layer].get();
  }
};

template <typename TConfig>
hwy::AlignedFreeUniquePtr<uint8_t[]> LoadWeights(
    const Path& checkpoint, hwy::ThreadPool& pool,
    bool scale_for_compression = false) {
  PROFILER_ZONE("Startup.LoadWeights");
  if (!std::filesystem::exists(checkpoint.path)) {
    HWY_ABORT("The model weights file '%s' does not exist.",
              checkpoint.path.c_str());
  }

  using TWeights = Weights<TConfig>;
  hwy::AlignedFreeUniquePtr<uint8_t[]> weights_u8 =
      hwy::AllocateAligned<uint8_t>(sizeof(TWeights));
  TWeights* weights = reinterpret_cast<TWeights*>(weights_u8.get());
  new (&weights->layer_ptrs) LayerPointers<TConfig>(pool);

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
    Layer<TConfig>* layer_view = weights->GetLayer(layer);

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
    READ_WEIGHTS(attn_vec_einsum_w);
    READ_WEIGHTS(qkv_einsum_w);
    SCALE_WEIGHTS(attn_vec_einsum_w);
    SCALE_WEIGHTS(qkv_einsum_w);
    READ_WEIGHTS(gating_einsum_w);
    READ_WEIGHTS(linear_w);
    SCALE_WEIGHTS(gating_einsum_w);
    SCALE_WEIGHTS(linear_w);
    READ_WEIGHTS(pre_attention_norm_scale);
    READ_WEIGHTS(pre_ffw_norm_scale);
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

  using TLayer = gcpp::Layer<TConfig>;
  using WeightT = typename TConfig::WeightT;

  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;

  // Compressed Parameters
  // We don't yet have an RMSNorm that accepts all WeightT.
  CompressedArray<hwy::bfloat16_t, kModelDim> pre_attention_norm_scale;
  CompressedArray<hwy::bfloat16_t, kModelDim> pre_ffw_norm_scale;
  CompressedArray<WeightT, TLayer::kGatingEinsumWSize> gating_einsum_w;
  CompressedArray<WeightT, kModelDim * kFFHiddenDim> linear_w;
  CompressedArray<WeightT, TLayer::kQKVEinsumWSize> qkv_einsum_w;
  CompressedArray<WeightT, TLayer::kAttVecEinsumWSize> attn_vec_einsum_w;
};

// Array instead of single large allocation for parallel mem init. Split out of
// CompressedWeights so that only these pointers are initialized, not the
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
                         Weights<TConfig>>;

// Aligned.
template <class TConfig, size_t TBatchSize>
struct Activations {
  static constexpr size_t kBatchSize = TBatchSize;
  using LayerConfig = Layer<TConfig>;
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kKVHeads = TConfig::kKVHeads;
  static constexpr size_t kCachePosSize = TConfig::kLayers * kKVHeads * kQKVDim;
  static constexpr size_t kCacheLayerSize = kKVHeads * kQKVDim;

  std::array<float, kBatchSize * kModelDim> x;  // input
  std::array<float, kBatchSize * kModelDim> pre_att_rms_out;
  std::array<float, kBatchSize * kHeads * kQKVDim> q;  // query vector
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
};

// GemmaImpl is a template and thus cannot be exposed in gemma.h, hence we
// define an abstract base class.
struct GemmaInterface {
  virtual ~GemmaInterface() = default;

  virtual const sentencepiece::SentencePieceProcessor* Tokenizer() const = 0;

  virtual void Generate(size_t max_tokens, size_t max_generated_tokens,
                        float temperature, const std::vector<int>& prompt,
                        size_t start_pos, KVCache& kv_cache,
                        hwy::ThreadPool& pool, hwy::ThreadPool& inner_pool,
                        const StreamFunc& stream_token,
                        const AcceptFunc& accept_token, std::mt19937& gen,
                        int verbosity) = 0;
};

template <class Config>
KVCache CreateKVCache() {
  return CreateKVCache(Config::kLayers * Config::kKVHeads * Config::kQKVDim,
                       Config::kSeqLen);
}

KVCache CreateKVCache(Model type) {
  switch (type) {
    case Model::GEMMA_2B:
      return CreateKVCache<ConfigGemma2B>();
    case Model::GEMMA_7B:
      return CreateKVCache<ConfigGemma7B>();
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(type));
  }
}

namespace {
template <class Config>
void DeleteLayersPtrs(CompressedWeights<Config>* c_weights) {
  c_weights->c_layer_ptrs.~CompressedLayerPointers<Config>();
}
template <class Config>
void DeleteLayersPtrs(Weights<Config>* weights) {
  weights->layer_ptrs.~LayerPointers<Config>();
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

  const sentencepiece::SentencePieceProcessor* Tokenizer() const override {
    return tokenizer.get();
  }

  void Generate(size_t max_tokens, size_t max_generated_tokens,
                float temperature, const std::vector<int>& prompt,
                size_t start_pos, KVCache& kv_cache, hwy::ThreadPool& pool,
                hwy::ThreadPool& inner_pool, const StreamFunc& stream_token,
                const AcceptFunc& accept_token, std::mt19937&,
                int verbosity) override;

  std::unique_ptr<sentencepiece::SentencePieceProcessor> tokenizer;
  hwy::AlignedFreeUniquePtr<uint8_t[]> weights_u8;
  hwy::AlignedUniquePtr<Activations<Config, kPrefillBatchSize>> prefill;
  hwy::AlignedUniquePtr<Activations<Config, 1>> state;
};

}  // namespace gcpp
#endif  // GEMMA_ONCE

// SIMD code, compiled once per target.
HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

template <size_t kBatchSize, typename LayerT, class TConfig>
HWY_NOINLINE void Attention(size_t batch_start, size_t batch_idx, size_t layer,
                            Activations<TConfig, kBatchSize>& activations,
                            const LayerT* layer_weights, KVCache& kv_cache,
                            hwy::ThreadPool& pool) {
  PROFILER_ZONE("Gen.Attention");
  const size_t pos = batch_start + batch_idx;
  HWY_DASSERT(batch_idx < kBatchSize);
  static constexpr size_t kQKVDim = gcpp::Activations<TConfig, 1>::kQKVDim;
  static constexpr size_t kCachePosSize =
      gcpp::Activations<TConfig, kBatchSize>::kCachePosSize;
  static constexpr size_t kCacheLayerSize =
      gcpp::Activations<TConfig, kBatchSize>::kCacheLayerSize;
  static constexpr size_t kModelDim =
      gcpp::Activations<TConfig, kBatchSize>::kModelDim;
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kKVHeads = TConfig::kKVHeads;
  static const float kQueryScale =
      static_cast<float>(1.0 / sqrt(static_cast<double>(kQKVDim)));

  float* x = activations.pre_att_rms_out.data() + batch_idx * kModelDim;

  auto ProjQ = [&](uint64_t head, size_t head_offset) HWY_ATTR {
    float* HWY_RESTRICT q =
        activations.q.data() + head * kQKVDim + batch_idx * kHeads * kQKVDim;

    MatVecLoop<kQKVDim, kModelDim>(layer_weights->qkv_einsum_w,
                                   head_offset + 0 * kQKVDim * kModelDim, x, q);
  };

  auto ProjKV = [&](size_t k_offset, size_t v_offset,
                    size_t kv_offset) HWY_ATTR {
    float* HWY_RESTRICT k = kv_cache.key_cache.get() + kv_offset;
    float* HWY_RESTRICT v = kv_cache.value_cache.get() + kv_offset;

    TwoOfsMatVecLoop<kQKVDim, kModelDim>(layer_weights->qkv_einsum_w, k_offset,
                                         v_offset, x, k, v);

    Rope(k, kQKVDim, pos);
  };

  auto Attn = [&](uint64_t head, size_t head_offset) HWY_ATTR {
    // Calculate scores
    float* HWY_RESTRICT q =
        activations.q.data() + head * kQKVDim + batch_idx * kHeads * kQKVDim;
    float* HWY_RESTRICT head_att = activations.att.data() +
                                   head * TConfig::kSeqLen +
                                   batch_idx * kHeads * kQKVDim;

    Rope(q, kQKVDim, pos);
    MulByConst(kQueryScale, q, kQKVDim);

    // Compute Q dot K scores
    for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
      const size_t cache_offset =
          pos2 * kCachePosSize + layer * kCacheLayerSize + head_offset;
      const float* HWY_RESTRICT k2 = kv_cache.key_cache.get() + cache_offset;
      const float score = Dot(q, k2, kQKVDim);
      head_att[pos2] = score;
    }
    Softmax(head_att, pos + 1);

    // Weighted summation
    float* HWY_RESTRICT att_out = activations.att_out.data() + head * kQKVDim +
                                  batch_idx * kHeads * kQKVDim;
    hwy::ZeroBytes(att_out, kQKVDim * sizeof(*att_out));
    for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
      const size_t cache_offset =
          pos2 * kCachePosSize + layer * kCacheLayerSize + head_offset;
      float* HWY_RESTRICT v2 = kv_cache.value_cache.get() + cache_offset;
      MulByConstAndAdd(head_att[pos2], v2, att_out, kQKVDim);
    }
    // linear projection from kQKVDim back to kModelDim, sum projections
    // across heads
    float* HWY_RESTRICT head_out =
        head == 0
            ? activations.att_post2.data() + batch_idx * kModelDim
            : activations.att_post1.data() + head * kBatchSize * kModelDim;
    MatVecLoop<kModelDim, kQKVDim>(layer_weights->attn_vec_einsum_w,
                                   head * kModelDim * kQKVDim, att_out,
                                   head_out);
  };

  if constexpr (kHeads == kKVHeads) {
    // Multi-Head Attention
    pool.Run(0, kHeads, [&](const uint64_t head, size_t /*thread*/) HWY_ATTR {
      const size_t head_offset = head * 3 * kQKVDim * kModelDim;

      ProjQ(head, head_offset);

      const size_t k_offset = head_offset + 1 * kQKVDim * kModelDim;
      const size_t v_offset = head_offset + 2 * kQKVDim * kModelDim;
      const size_t kv_offset =
          pos * kCachePosSize + layer * kCacheLayerSize + head * kQKVDim;

      ProjKV(k_offset, v_offset, kv_offset);

      Attn(head, head * kQKVDim);
    });
  } else {
    // Multi-Query Attention
    constexpr const size_t q_offset = kHeads * kQKVDim * kModelDim;
    constexpr const size_t k_offset = q_offset + 0 * kQKVDim * kModelDim;
    constexpr const size_t v_offset = q_offset + 1 * kQKVDim * kModelDim;
    const size_t kv_offset = pos * kCachePosSize + layer * kCacheLayerSize;
    ProjKV(k_offset, v_offset, kv_offset);

    pool.Run(0, kHeads, [&](const uint64_t head, size_t /*thread*/) HWY_ATTR {
      ProjQ(head, head * kQKVDim * kModelDim);
      Attn(head, 0);
    });
  }

  // accumulate output across all heads into att_post2. head 0 already wrote
  // directly to att_post2.
  for (size_t head = 1; head < kHeads; ++head) {
    AddFrom(activations.att_post1.data() + head * kBatchSize * kModelDim,
            activations.att_post2.data() + batch_idx * kModelDim, kModelDim);
  }
}

template <size_t kBatchSize, typename LayerT, typename TConfig>
HWY_NOINLINE void FFW(Activations<TConfig, kBatchSize>& activations,
                      size_t batch_idx, const LayerT* layer_weights,
                      hwy::ThreadPool& pool) {
  HWY_DASSERT(batch_idx < kBatchSize);
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;
  const size_t hidden_offset = batch_idx * kFFHiddenDim * 2;

  {
    PROFILER_ZONE("Gen.FFW.GatedGELU");
    const hwy::bfloat16_t* HWY_RESTRICT vec =
        activations.bf_pre_ffw_rms_out.data() + batch_idx * kModelDim;
    float* HWY_RESTRICT out = activations.ffw_hidden.data() + hidden_offset;
    float* HWY_RESTRICT out_mul = out + kFFHiddenDim;

    // Same matrix, first and second half of rows. Could fuse into one MatVec,
    // but separating them could help on NUMA e.g. multiple sockets.
    MatVec<kFFHiddenDim, kModelDim>(layer_weights->gating_einsum_w,
                                    kFFHiddenDim * kModelDim, vec, out_mul,
                                    pool);

    // Gate, will go through the nonlinearity.
    MatVec<kFFHiddenDim, kModelDim>(layer_weights->gating_einsum_w, 0, vec, out,
                                    pool);

    namespace hn = hwy::HWY_NAMESPACE;
    using DF = hn::ScalableTag<float>;
    using VF = hn::Vec<DF>;
    hn::Transform1(DF(), out, kFFHiddenDim, out_mul,
                   [](DF df, VF v, VF mul)
                       HWY_ATTR { return hn::Mul(mul, Gelu(df, v)); });
  }

  PROFILER_ZONE("Gen.FFW\\GatedGELU");
  MatVec<kModelDim, kFFHiddenDim>(
      layer_weights->linear_w, 0, activations.ffw_hidden.data() + hidden_offset,
      activations.ffw_out.data() + batch_idx * kModelDim, pool);
}

// `EmbeddingScaling` can be constexpr only if `Sqrt` and `hwy::ConvertScalarTo`
// are both constexpr
#if HWY_COMPILER_GCC_ACTUAL
#define GEMMA_CONSTEXPR_EMBSCALING HWY_BF16_CONSTEXPR
#else
#define GEMMA_CONSTEXPR_EMBSCALING
#endif

template <typename TConfig>
GEMMA_CONSTEXPR_EMBSCALING float EmbeddingScaling() {
  // Round to bf16 to match Gemma's Embedder, which casts before mul.
  return hwy::ConvertScalarTo<float>(hwy::ConvertScalarTo<hwy::bfloat16_t>(
      Sqrt(static_cast<float>(TConfig::kModelDim))));
}

template <size_t kBatchSize, typename WeightArrayT, typename TConfig>
HWY_NOINLINE void Prefill(const int* tokens, size_t num_tokens, size_t pos,
                          const WeightArrayT& weights,
                          Activations<TConfig, kBatchSize>& activations,
                          KVCache& kv_cache, hwy::ThreadPool& pool,
                          hwy::ThreadPool& inner_pool) {
  PROFILER_ZONE("Gen.Prefill\\Att\\FFW");
  static constexpr size_t kModelDim = TConfig::kModelDim;
  GEMMA_CONSTEXPR_EMBSCALING const float kEmbScaling =
      EmbeddingScaling<TConfig>();

  pool.Run(
      0, num_tokens, [&](const uint64_t token_idx, size_t /*thread*/) HWY_ATTR {
        const int token = tokens[token_idx];
        Decompress(weights.embedder_input_embedding, token * kModelDim,
                   activations.x.data() + token_idx * kModelDim, kModelDim);
        MulByConst(kEmbScaling, activations.x.data() + token_idx * kModelDim,
                   kModelDim);
      });

  for (size_t layer = 0; layer < TConfig::kLayers; ++layer) {
    const auto* layer_weights = weights.GetLayer(layer);

    for (size_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
      RMSNorm(activations.x.data() + token_idx * kModelDim,
              layer_weights->pre_attention_norm_scale.data(),
              activations.pre_att_rms_out.data() + token_idx * kModelDim,
              kModelDim);
      Attention<kBatchSize>(pos, token_idx, layer, activations, layer_weights,
                            kv_cache, pool);
    }

    // TODO: sink the loop into these functions, i.e. make them matmuls.
    pool.Run(
        0, num_tokens,
        [&](const uint64_t token_idx, size_t thread_id) HWY_ATTR {
          AddFrom(activations.att_post2.data() + token_idx * kModelDim,
                  activations.x.data() + token_idx * kModelDim, kModelDim);
          RMSNorm(activations.x.data() + token_idx * kModelDim,
                  layer_weights->pre_ffw_norm_scale.data(),
                  activations.bf_pre_ffw_rms_out.data() + token_idx * kModelDim,
                  kModelDim);
          FFW<kBatchSize>(activations, token_idx, layer_weights, inner_pool);
          AddFrom(activations.ffw_out.data() + token_idx * kModelDim,
                  activations.x.data() + token_idx * kModelDim, kModelDim);
        });
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
                 hwy::ThreadPool& pool, hwy::ThreadPool& inner_pool) {
  static constexpr size_t kLayers = TConfig::kLayers;
  static constexpr size_t kModelDim = TConfig::kModelDim;

  Decompress(weights.embedder_input_embedding, token * kModelDim,
             activations.x.data(), kModelDim);

  GEMMA_CONSTEXPR_EMBSCALING const float kEmbScaling =
      EmbeddingScaling<TConfig>();
  MulByConst(kEmbScaling, activations.x.data(), kModelDim);

  for (size_t layer = 0; layer < kLayers; ++layer) {
    const auto* layer_weights = weights.GetLayer(layer);
    RMSNorm(activations.x.data(),
            layer_weights->pre_attention_norm_scale.data(),
            activations.pre_att_rms_out.data(), kModelDim);
    Attention<1>(pos, 0, layer, activations, layer_weights, kv_cache, pool);
    AddFrom(activations.att_post2.data(), activations.x.data(), kModelDim);
    RMSNorm(activations.x.data(), layer_weights->pre_ffw_norm_scale.data(),
            activations.bf_pre_ffw_rms_out.data(), kModelDim);
    FFW<1>(activations, /* batch_idx = */ 0, layer_weights, pool);
    AddFrom(activations.ffw_out.data(), activations.x.data(), kModelDim);
  }
  RMSNormInplace(weights.final_norm_scale.data(), activations.x.data(),
                 kModelDim);
}

template <class TConfig>
void RangeChecks(size_t& max_tokens, size_t& max_generated_tokens,
                 size_t& prompt_size) {
  if (max_tokens > TConfig::kSeqLen) {
    fprintf(stderr, "WARNING: max_tokens %zu > kSeqLen %d, truncating.\n",
            max_tokens, TConfig::kSeqLen);
    max_tokens = static_cast<size_t>(TConfig::kSeqLen);
  }

  if (max_generated_tokens > max_tokens) {
    fprintf(stderr,
            "WARNING: max_generated_tokens %zu > max_tokens %zu, truncating.\n",
            max_generated_tokens, max_tokens);
    max_generated_tokens = max_tokens - 1;
  }

  if (prompt_size + max_generated_tokens > max_tokens) {
    fprintf(stderr,
            "WARNING: prompt_size %zu + max_generated_tokens %zu > kSeqLen "
            "%d, truncating.\n",
            prompt_size, max_generated_tokens, TConfig::kSeqLen);
    prompt_size = max_tokens - max_generated_tokens;
  }
}

template <class TConfig>
void GenerateImpl(GemmaImpl<TConfig>& gemma, size_t max_tokens,
                  size_t max_generated_tokens, float temperature,
                  const std::vector<int>& prompt, size_t pos, KVCache& kv_cache,
                  hwy::ThreadPool& pool, hwy::ThreadPool& inner_pool,
                  const StreamFunc& stream_token,
                  const AcceptFunc& accept_token, std::mt19937& gen,
                  int verbosity) {
  static constexpr size_t kVocabSize = TConfig::kVocabSize;
  Activations<TConfig, 1>& activations = *gemma.state.get();
  Activations<TConfig, kPrefillBatchSize>& prefill_activations =
      *gemma.prefill.get();

  const WeightsT<TConfig>& weights =
      *reinterpret_cast<WeightsT<TConfig>*>(gemma.weights_u8.get());

  size_t prompt_size = prompt.size();
  RangeChecks<TConfig>(max_tokens, max_generated_tokens, prompt_size);
  if (pos >= max_tokens) {
    fprintf(stderr, "Warning: pos %zu >= max_tokens %zu, aborting.\n", pos,
            max_tokens);
    return;
  }

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
                               prefill_activations, kv_cache, pool, inner_pool);
    for (size_t idx = 0; idx < batch_size; ++idx) {
      stream_token(batch_tokens[idx], 0.0f);
    }
    pos += batch_size;
    pos_offset += batch_size;
  }

  if (verbosity >= 2) {
    // in the future this output should not occur in GenerateImpl but instead
    // should be available as observable state for frontend code to handle I/O.
    const double prefill_end = hwy::platform::Now();
    const double prefill_tok_sec =
        static_cast<double>(pos_offset) / (prefill_end - prefill_start);
    std::cout << "\n[ Prefill tokens / sec = " << prefill_tok_sec << " ]";
  }

  const double gen_start = hwy::platform::Now();

  HWY_DASSERT(pos_offset == prompt_size - 1);

  size_t pos_gen_start = pos_offset;
  int token = prompt.at(pos_offset);
  stream_token(token, 0);
  for (size_t generate_pos = 0;
       pos < max_tokens && generate_pos < max_generated_tokens;
       ++pos, ++pos_offset, ++generate_pos) {
    Transformer(token, pos, weights, activations, kv_cache, pool, inner_pool);
    float* final_activation = activations.x.data();
    // The condition below is always true if we are doing Prefill above.
    // We keep it here for clarity so that the code is correct even if Prefill
    // is disabled.
    if (pos_offset >= prompt_size - 1) {
      PROFILER_ZONE("Gen.Embedding");
      // Generation phase
      MatVec<kVocabSize, TConfig::kModelDim>(weights.embedder_input_embedding,
                                             0, final_activation,
                                             activations.logits.data(), pool);
      // Barrier: must have all logits so we can subtract max.
      Softmax(activations.logits.data(), kVocabSize);
      token = SampleTopK<TConfig::kTopK>(activations.logits.data(), kVocabSize,
                                         gen, temperature, accept_token);
      if (!stream_token(token, activations.logits[token])) {
        token = EOS_ID;
      }
    } else {
      // We would take this branch if we were not doing Prefill but would
      // process the tokens of the prompt one at a time.
      token = prompt.at(pos_offset + 1);
      stream_token(token, 0);
    }
    if (token == EOS_ID) {
      if (verbosity >= 2) {
        const double gen_end = hwy::platform::Now();
        const double gen_tok_sec =
            static_cast<double>(pos_offset - pos_gen_start) /
            (gen_end - gen_start);
        std::cout << "\n[ Generation tokens / sec = " << gen_tok_sec << " ]\n";
      }
      break;
    }
  }
}

void Generate2B(GemmaImpl<ConfigGemma2B>& gemma, size_t max_tokens,
                size_t max_generated_tokens, float temperature,
                const std::vector<int>& prompt, size_t start_pos,
                KVCache& kv_cache, hwy::ThreadPool& pool,
                hwy::ThreadPool& inner_pool, const StreamFunc& stream_token,
                const AcceptFunc& accept_token, std::mt19937& gen,
                int verbosity) {
  GenerateImpl(gemma, max_tokens, max_generated_tokens, temperature, prompt,
               start_pos, kv_cache, pool, inner_pool, stream_token,
               accept_token, gen, verbosity);
}

void Generate7B(GemmaImpl<ConfigGemma7B>& gemma, size_t max_tokens,
                size_t max_generated_tokens, float temperature,
                const std::vector<int>& prompt, size_t start_pos,
                KVCache& kv_cache, hwy::ThreadPool& pool,
                hwy::ThreadPool& inner_pool, const StreamFunc& stream_token,
                const AcceptFunc& accept_token, std::mt19937& gen,
                int verbosity) {
  GenerateImpl(gemma, max_tokens, max_generated_tokens, temperature, prompt,
               start_pos, kv_cache, pool, inner_pool, stream_token,
               accept_token, gen, verbosity);
}

// Calls func(name, float*, CompressedArray&) for each tensor. float* is null
// if weights = null, which happens during the first call where we attempt to
// load from cache.
//
// This avoids repeating the list of tensors between loading and compressing.
template <class TConfig, class Func>
void ForEachTensor(const Weights<TConfig>* weights,
                   CompressedWeights<TConfig>& c_weights, Func& func) {
  func("c_embedding",
       weights ? weights->embedder_input_embedding.data() : nullptr,
       c_weights.embedder_input_embedding);
  func("c_final_norm", weights ? weights->final_norm_scale.data() : nullptr,
       c_weights.final_norm_scale);

  char name_buf[16];
  for (int layer_idx = 0; layer_idx < TConfig::kLayers; ++layer_idx) {
    const size_t idx = static_cast<size_t>(layer_idx);
    const Layer<TConfig>* layer = weights ? weights->GetLayer(idx) : nullptr;
    CompressedLayer<TConfig>* layer_weights = c_weights.GetLayer(idx);

#define CALL_FUNC(name, member)                                \
  snprintf(name_buf, sizeof(name_buf), name "_%d", layer_idx); \
  func(name_buf, layer ? layer->member.data() : nullptr, layer_weights->member)

    CALL_FUNC("pre_ff_ns", pre_ffw_norm_scale);
    CALL_FUNC("gating_ein", gating_einsum_w);
    CALL_FUNC("linear_w", linear_w);
    CALL_FUNC("qkv_ein", qkv_einsum_w);
    CALL_FUNC("att_ein", attn_vec_einsum_w);
    CALL_FUNC("pre_att_ns", pre_attention_norm_scale);
#undef CALL_FUNC
  }
}

template <class TConfig>
hwy::AlignedFreeUniquePtr<uint8_t[]> LoadCompressedWeights(
    const Path& weights, hwy::ThreadPool& pool) {
  PROFILER_ZONE("Startup.LoadCache");
  if (!std::filesystem::exists(weights.path)) {
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
  CacheLoader loader(weights.path.c_str());
  ForEachTensor<TConfig>(nullptr, *c_weights, loader);
  loader.LoadScales(scales.data(), scales.size());
  if (!loader.ReadAll(pool)) {
    HWY_ABORT("Failed to load model weights.");
  }
  if (TConfig::kNumTensorScales > 0) {
    size_t scale_pos = 0;
    for (int layer_idx = 0; layer_idx < TConfig::kLayers; ++layer_idx) {
      const size_t idx = static_cast<size_t>(layer_idx);
      CompressedLayer<TConfig>* layer_weights = c_weights->GetLayer(idx);
      layer_weights->attn_vec_einsum_w.set_scale(scales[scale_pos++]);
      layer_weights->qkv_einsum_w.set_scale(scales[scale_pos++]);
      layer_weights->gating_einsum_w.set_scale(scales[scale_pos++]);
      layer_weights->linear_w.set_scale(scales[scale_pos++]);
    }
    HWY_ASSERT(scale_pos == TConfig::kNumTensorScales);
  }
  return c_weights_u8;
}

// Type-erased because this function is called via a function pointer.
hwy::AlignedFreeUniquePtr<uint8_t[]> LoadCompressedWeightsT(
    gcpp::Model model, const Path& weights, hwy::ThreadPool& pool) {
  switch (model) {
    case Model::GEMMA_2B:
      return LoadCompressedWeights<ConfigGemma2B>(weights, pool);
    case Model::GEMMA_7B:
      return LoadCompressedWeights<ConfigGemma7B>(weights, pool);
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

hwy::AlignedFreeUniquePtr<uint8_t[]> LoadWeightsT(gcpp::Model model,
                                                  const Path& weights,
                                                  hwy::ThreadPool& pool) {
  switch (model) {
    case Model::GEMMA_2B:
      return LoadWeights<ConfigGemma2B>(weights, pool);
    case Model::GEMMA_7B:
      return LoadWeights<ConfigGemma7B>(weights, pool);
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

template <class TConfig>
void CompressWeights(const Path& weights_path,
                     const Path& compressed_weights_path,
                     hwy::ThreadPool& pool) {
  if (!std::filesystem::exists(weights_path.path)) {
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
  Weights<TConfig>* weights =
      reinterpret_cast<Weights<TConfig>*>(weights_u8.get());
  Compressor compressor(pool);
  ForEachTensor<TConfig>(weights, *c_weights, compressor);
  compressor.AddScales(weights->scales.data(), weights->scales.size());
  compressor.WriteAll(pool, compressed_weights_path.path.c_str());

  weights->layer_ptrs.~LayerPointers<TConfig>();
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
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model));
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {

HWY_EXPORT(LoadCompressedWeightsT);
HWY_EXPORT(LoadWeightsT);
HWY_EXPORT(CompressWeightsT);
HWY_EXPORT(Generate2B);
HWY_EXPORT(Generate7B);

KVCache CreateKVCache(size_t size_cache_pos, size_t seq_len) {
  KVCache kv_cache = {};
  kv_cache.key_cache = hwy::AllocateAligned<float>(seq_len * size_cache_pos);
  kv_cache.value_cache = hwy::AllocateAligned<float>(seq_len * size_cache_pos);
  return kv_cache;
}

template <class Config>
GemmaImpl<Config>::GemmaImpl(
    std::unique_ptr<sentencepiece::SentencePieceProcessor>& tokenizer,
    hwy::AlignedFreeUniquePtr<uint8_t[]>& weights_u8, hwy::ThreadPool& pool)
    : tokenizer(std::move(tokenizer)),
      weights_u8(std::move(weights_u8)),
      prefill(hwy::MakeUniqueAligned<Activations<Config, kPrefillBatchSize>>()),
      state(hwy::MakeUniqueAligned<Activations<Config, 1>>()) {}

template <>
void GemmaImpl<ConfigGemma2B>::Generate(
    size_t max_tokens, size_t max_generated_tokens, float temperature,
    const std::vector<int>& prompt, size_t start_pos, KVCache& kv_cache,
    hwy::ThreadPool& pool, hwy::ThreadPool& inner_pool,
    const StreamFunc& stream_token, const AcceptFunc& accept_token,
    std::mt19937& gen, int verbosity) {
  HWY_DYNAMIC_DISPATCH(Generate2B)
  (*this, max_tokens, max_generated_tokens, temperature, prompt, start_pos,
   kv_cache, pool, inner_pool, stream_token, accept_token, gen, verbosity);
}
template <>
void GemmaImpl<ConfigGemma7B>::Generate(
    size_t max_tokens, size_t max_generated_tokens, float temperature,
    const std::vector<int>& prompt, size_t start_pos, KVCache& kv_cache,
    hwy::ThreadPool& pool, hwy::ThreadPool& inner_pool,
    const StreamFunc& stream_token, const AcceptFunc& accept_token,
    std::mt19937& gen, int verbosity) {
  HWY_DYNAMIC_DISPATCH(Generate7B)
  (*this, max_tokens, max_generated_tokens, temperature, prompt, start_pos,
   kv_cache, pool, inner_pool, stream_token, accept_token, gen, verbosity);
}

Gemma::Gemma(const Path& tokenizer_path, const Path& weights, Model model_type,
             hwy::ThreadPool& pool) {
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
    weights_u8 =
        HWY_DYNAMIC_DISPATCH(LoadCompressedWeightsT)(model_type, weights, pool);
  } else {
    weights_u8 = HWY_DYNAMIC_DISPATCH(LoadWeightsT)(model_type, weights, pool);
  }

  switch (model_type) {
    case Model::GEMMA_2B:
      impl_.reset(new GemmaImpl<ConfigGemma2B>(tokenizer, weights_u8, pool));
      break;
    case Model::GEMMA_7B:
      impl_.reset(new GemmaImpl<ConfigGemma7B>(tokenizer, weights_u8, pool));
      break;
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model_type));
  }
}

Gemma::~Gemma() = default;  // after GemmaInterface is defined

const sentencepiece::SentencePieceProcessor* Gemma::Tokenizer() const {
  return impl_->Tokenizer();
}

void GenerateGemma(Gemma& gemma, size_t max_tokens, size_t max_generated_tokens,
                   float temperature, const std::vector<int>& prompt,
                   size_t start_pos, KVCache& kv_cache, hwy::ThreadPool& pool,
                   hwy::ThreadPool& inner_pool, const StreamFunc& stream_token,
                   const AcceptFunc& accept_token, std::mt19937& gen,
                   int verbosity) {
  pool.SetWaitMode(hwy::PoolWaitMode::kSpin);
  gemma.impl_->Generate(max_tokens, max_generated_tokens, temperature, prompt,
                        start_pos, kv_cache, pool, inner_pool, stream_token,
                        accept_token, gen, verbosity);
  pool.SetWaitMode(hwy::PoolWaitMode::kBlock);
}

void GenerateGemma(Gemma& gemma, RuntimeConfig runtime_config,
                   const std::vector<int>& prompt, size_t start_pos,
                   KVCache& kv_cache, hwy::ThreadPool& pool,
                   const StreamFunc& stream_token, std::mt19937& gen) {
  hwy::ThreadPool inner_pool(0);
  GenerateGemma(
      gemma, runtime_config.max_tokens, runtime_config.max_generated_tokens,
      runtime_config.temperature, prompt, start_pos, kv_cache, pool, inner_pool,
      stream_token, [](int) { return true; }, gen, runtime_config.verbosity);
}

void CompressWeights(gcpp::Model model, const Path& weights,
                     const Path& compressed_weights, hwy::ThreadPool& pool) {
  HWY_DYNAMIC_DISPATCH(CompressWeightsT)
  (model, weights, compressed_weights, pool);
}

}  // namespace gcpp
#endif  // HWY_ONCE
