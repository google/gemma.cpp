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
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// Must come after foreach_target.h to avoid redefinition errors.
// copybara:import_next_line:gemma_cpp
#include "compression/compress-inl.h"
// copybara:import_next_line:gemma_cpp
#include "ops.h"
// copybara:import_next_line:gemma_cpp
#include "util/args.h"  // Path
#include "hwy/contrib/matvec/matvec-inl.h"
#include "hwy/highway.h"
#include "hwy/profiler.h"
#include "hwy/timer.h"

// Non-SIMD includes and types. Note that HWY_ONCE is only true on the last
// compile pass, whereas we want this defined in the first.
#ifndef GEMMA_ONCE
#define GEMMA_ONCE

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

namespace gcpp {

template <class TConfig>
struct Layer {
  Layer() = default;
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;
  static constexpr size_t kAttVecEinsumWSize = kHeads * kQKVDim * kModelDim;
  // 3x for (query, key, value)
  static constexpr size_t kQKVEinsumWSize = 3 * kHeads * kQKVDim * kModelDim;
  // 2x for (gelu gating vector, gated vector)
  static constexpr size_t kGatingEinsumWSize = 2 * kFFHiddenDim * kModelDim;

  std::array<float, kAttVecEinsumWSize> attn_vec_einsum_w;
  std::array<float, kQKVEinsumWSize> qkv_einsum_w;
  std::array<float, kGatingEinsumWSize> gating_einsum_w;
  std::array<float, kModelDim * kFFHiddenDim> linear_w;
  std::array<float, kModelDim> pre_attention_norm_scale;
  std::array<float, kModelDim> pre_ffw_norm_scale;
};

template <class TConfig>
struct Weights {
  Weights() = default;

  hwy::AlignedUniquePtr<Layer<TConfig>[]> layers;  // kLayers

  std::array<float, TConfig::kVocabSize * TConfig::kModelDim>
      embedder_input_embedding;

  std::array<float, TConfig::kModelDim> final_norm_scale;
};

// Only called if cached loading fails.
template <typename TConfig>
hwy::AlignedUniquePtr<Weights<TConfig>> LoadWeights(const Path& checkpoint) {
  PROFILER_ZONE("Startup.LoadWeights");
  using TWeights = Weights<TConfig>;
  hwy::AlignedUniquePtr<TWeights> weights = hwy::MakeUniqueAligned<TWeights>();
  weights->layers =
      hwy::MakeUniqueAlignedArray<Layer<TConfig>>(TConfig::kLayers);

  FILE* fptr;
  fptr = fopen(checkpoint.path.c_str(), "rb");
  if (fptr == nullptr) {
    HWY_ABORT("Failed to open model file %s - does it exist?",
              checkpoint.path.c_str());
  }
  bool ok = true;
  ok &= 1 == fread(&(weights->embedder_input_embedding),
                   sizeof(weights->embedder_input_embedding), 1, fptr);
  ok &= 1 == fread(&(weights->final_norm_scale),
                   sizeof(weights->final_norm_scale), 1, fptr);
  for (size_t layer = 0; layer < TConfig::kLayers; ++layer) {
    Layer<TConfig>* layer_view = &weights->layers[layer];
    ok &= 1 == fread(&layer_view->attn_vec_einsum_w,
                     sizeof(layer_view->attn_vec_einsum_w), 1, fptr);
    ok &= 1 == fread(&layer_view->qkv_einsum_w,
                     sizeof(layer_view->qkv_einsum_w), 1, fptr);
    ok &= 1 == fread(&layer_view->gating_einsum_w,
                     sizeof(layer_view->gating_einsum_w), 1, fptr);
    ok &= 1 ==
          fread(&layer_view->linear_w, sizeof(layer_view->linear_w), 1, fptr);
    ok &= 1 == fread(&layer_view->pre_attention_norm_scale,
                     sizeof(layer_view->pre_attention_norm_scale), 1, fptr);
    ok &= 1 == fread(&layer_view->pre_ffw_norm_scale,
                     sizeof(layer_view->pre_ffw_norm_scale), 1, fptr);
  }
  if (!ok) {
    HWY_ABORT("Failed to read from %s - might be a directory, or too small?",
              checkpoint.path.c_str());
  }
  HWY_ASSERT(0 == fclose(fptr));
  return weights;
}

template <class TConfig>
struct CompressedLayer {
  // No ctor/dtor, allocated via AllocateAligned.

  using TLayer = gcpp::Layer<TConfig>;

  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;

  // Compressed Parameters
  // We don't yet have an RMSNorm that accepts all WeightT.
  CompressedArray<hwy::bfloat16_t, kModelDim> c_pre_attention_norm_scale;
  CompressedArray<hwy::bfloat16_t, kModelDim> c_pre_ffw_norm_scale;
  CompressedArray<WeightT, TLayer::kGatingEinsumWSize> c_gating_einsum_w;
  CompressedArray<WeightT, kModelDim * kFFHiddenDim> c_linear_w;
  CompressedArray<WeightT, TLayer::kQKVEinsumWSize> c_qkv_einsum_w;
  CompressedArray<WeightT, TLayer::kAttVecEinsumWSize> c_attn_vec_einsum_w;
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
      c_embedder_input_embedding;

  CompressedArray<hwy::bfloat16_t, TConfig::kModelDim> c_final_norm_scale;

  // Must be last so that the other arrays remain aligned.
  CompressedLayerPointers<TConfig> c_layer_ptrs;

  const CompressedLayer<TConfig>* CLayer(size_t layer) const {
    return c_layer_ptrs.c_layers[layer].get();
  }
  CompressedLayer<TConfig>* CLayer(size_t layer) {
    return c_layer_ptrs.c_layers[layer].get();
  }
};

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

  virtual const sentencepiece::SentencePieceProcessor& Tokenizer() const = 0;

  // TODO: group pool/callbacks into struct
  virtual void Generate(const InferenceArgs& args,
                        const std::vector<int>& prompt, size_t start_pos,
                        hwy::ThreadPool& pool, hwy::ThreadPool& inner_pool,
                        const StreamFunc& stream_token,
                        const AcceptFunc& accept_token, std::mt19937& gen,
                        int verbosity) = 0;
};

template <class Config>
struct GemmaImpl : public GemmaInterface {
  GemmaImpl(const LoaderArgs& args, hwy::ThreadPool& pool);

  ~GemmaImpl() {
    using CWeights = CompressedWeights<Config>;
    CWeights* c_weights = reinterpret_cast<CWeights*>(compressed_weights.get());
    c_weights->c_layer_ptrs.~CompressedLayerPointers<Config>();
  }

  const sentencepiece::SentencePieceProcessor& Tokenizer() const {
    return tokenizer;
  }

  void Generate(const InferenceArgs& args, const std::vector<int>& prompt,
                size_t start_pos, hwy::ThreadPool& pool,
                hwy::ThreadPool& inner_pool, const StreamFunc& stream_token,
                const AcceptFunc& accept_token, std::mt19937&, int verbosity);

  sentencepiece::SentencePieceProcessor tokenizer;

  // CompressedWeights<Config>
  hwy::AlignedFreeUniquePtr<uint8_t[]> compressed_weights;
  hwy::AlignedUniquePtr<Activations<Config, kPrefillBatchSize>> prefill;
  hwy::AlignedUniquePtr<Activations<Config, 1>> state;
  KVCache kv_cache;
};

}  // namespace gcpp
#endif  // GEMMA_ONCE

// SIMD code, compiled once per target.
HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

template <class TConfig, size_t kBatchSize>
HWY_NOINLINE void Attention(size_t batch_start, size_t batch_idx, size_t layer,
                            Activations<TConfig, kBatchSize>& activations,
                            const CompressedLayer<TConfig>* c_layer,
                            KVCache& kv_cache, hwy::ThreadPool& pool) {
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
  const float kQueryScale = 1.0 / sqrtf(static_cast<float>(kQKVDim));

  pool.Run(0, kHeads, [&](const uint64_t head, size_t /*thread*/) HWY_ATTR {
    // linear projections to QKV
    const size_t head_offset =
        3 * kQKVDim * kModelDim;  // 3x for QKV dimensions
    const size_t q_offset = head * head_offset + 0 * kQKVDim * kModelDim;
    const size_t k_offset = head * head_offset + 1 * kQKVDim * kModelDim;
    const size_t v_offset = head * head_offset + 2 * kQKVDim * kModelDim;

    float* HWY_RESTRICT q =
        activations.q.data() + head * kQKVDim + batch_idx * kHeads * kQKVDim;

    const size_t batch_offset = batch_idx * kModelDim;

    MatVecLoop<kQKVDim, kModelDim>(
        c_layer->c_qkv_einsum_w, q_offset,
        activations.pre_att_rms_out.data() + batch_offset, q);

    const size_t kv_offset =
        pos * kCachePosSize + layer * kCacheLayerSize + head * kQKVDim;

    TwoOfsMatVecLoop<kQKVDim, kModelDim>(
        c_layer->c_qkv_einsum_w, k_offset, v_offset,
        activations.pre_att_rms_out.data() + batch_offset,
        kv_cache.key_cache.get() + kv_offset,
        kv_cache.value_cache.get() + kv_offset);

    // Calculate scores
    float* HWY_RESTRICT head_att = activations.att.data() +
                                   head * TConfig::kSeqLen +
                                   batch_idx * kHeads * kQKVDim;

    Rope(q, kQKVDim, pos);
    Rope(kv_cache.key_cache.get() + kv_offset, kQKVDim, pos);
    MulByConst(kQueryScale, q, kQKVDim);
    // Compute Q dot K scores
    for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
      const size_t cache_offset =
          pos2 * kCachePosSize + layer * kCacheLayerSize + head * kQKVDim;
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
          pos2 * kCachePosSize + layer * kCacheLayerSize + head * kQKVDim;
      float* HWY_RESTRICT v2 = kv_cache.value_cache.get() + cache_offset;
      MulByConstAndAdd(head_att[pos2], v2, att_out, kQKVDim);
    }
    // linear projection from kQKVDim back to kModelDim, sum projections
    // across heads
    float* HWY_RESTRICT head_out =
        head == 0
            ? activations.att_post2.data() + batch_idx * kModelDim
            : activations.att_post1.data() + head * kBatchSize * kModelDim;
    MatVecLoop<kModelDim, kQKVDim>(c_layer->c_attn_vec_einsum_w,
                                   head * kModelDim * kQKVDim, att_out,
                                   head_out);
  });

  // accumulate output across all heads into att_post2. head 0 already wrote
  // directly to att_post2.
  for (size_t head = 1; head < kHeads; ++head) {
    AddFrom(activations.att_post1.data() + head * kBatchSize * kModelDim,
            activations.att_post2.data() + batch_idx * kModelDim, kModelDim);
  }
}

template <typename TConfig, size_t kBatchSize>
HWY_NOINLINE void FFW(Activations<TConfig, kBatchSize>& activations,
                      size_t batch_idx, const CompressedLayer<TConfig>* c_layer,
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
    MatVec<kFFHiddenDim, kModelDim>(c_layer->c_gating_einsum_w,
                                    kFFHiddenDim * kModelDim, vec, out_mul,
                                    pool);

    // Gate, will go through the nonlinearity.
    MatVec<kFFHiddenDim, kModelDim>(c_layer->c_gating_einsum_w, 0, vec, out,
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
      c_layer->c_linear_w, 0, activations.ffw_hidden.data() + hidden_offset,
      activations.ffw_out.data() + batch_idx * kModelDim, pool);
}

template <typename TConfig, size_t kBatchSize>
HWY_NOINLINE void Prefill(const int* tokens, size_t num_tokens, size_t pos,
                          const CompressedWeights<TConfig>& c_weights,
                          Activations<TConfig, kBatchSize>& activations,
                          KVCache& kv_cache, hwy::ThreadPool& pool,
                          hwy::ThreadPool& inner_pool) {
  PROFILER_ZONE("Gen.Prefill\\Att\\FFW");
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static const float kEmbScaling = sqrtf(static_cast<float>(kModelDim));

  pool.Run(
      0, num_tokens, [&](const uint64_t token_idx, size_t /*thread*/) HWY_ATTR {
        const int token = tokens[token_idx];
        Decompress(c_weights.c_embedder_input_embedding, token * kModelDim,
                   activations.x.data() + token_idx * kModelDim, kModelDim);
        MulByConst(kEmbScaling, activations.x.data() + token_idx * kModelDim,
                   kModelDim);
      });

  for (size_t layer = 0; layer < TConfig::kLayers; ++layer) {
    const CompressedLayer<TConfig>* c_layer = c_weights.CLayer(layer);

    for (size_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
      RMSNorm(activations.x.data() + token_idx * kModelDim,
              c_layer->c_pre_attention_norm_scale.data(),
              activations.pre_att_rms_out.data() + token_idx * kModelDim,
              kModelDim);
      Attention<TConfig, kBatchSize>(pos, token_idx, layer, activations,
                                     c_layer, kv_cache, pool);
    }

    // TODO: sink the loop into these functions, i.e. make them matmuls.
    pool.Run(
        0, num_tokens,
        [&](const uint64_t token_idx, size_t thread_id) HWY_ATTR {
          AddFrom(activations.att_post2.data() + token_idx * kModelDim,
                  activations.x.data() + token_idx * kModelDim, kModelDim);
          RMSNorm(activations.x.data() + token_idx * kModelDim,
                  c_layer->c_pre_ffw_norm_scale.data(),
                  activations.bf_pre_ffw_rms_out.data() + token_idx * kModelDim,
                  kModelDim);
          FFW<TConfig, kBatchSize>(activations, token_idx, c_layer, inner_pool);
          AddFrom(activations.ffw_out.data() + token_idx * kModelDim,
                  activations.x.data() + token_idx * kModelDim, kModelDim);
        });
  }  // foreach layer

  pool.Run(
      0, num_tokens, [&](const uint64_t token_idx, size_t /*thread*/) HWY_ATTR {
        RMSNormInplace(c_weights.c_final_norm_scale.data(),
                       activations.x.data() + token_idx * kModelDim, kModelDim);
      });
}

// n = 1 specialization
template <class TConfig>
void Transformer(int token, size_t pos,
                 const CompressedWeights<TConfig>& c_weights,
                 Activations<TConfig, 1>& activations, KVCache& kv_cache,
                 hwy::ThreadPool& pool, hwy::ThreadPool& inner_pool) {
  static constexpr size_t kLayers = TConfig::kLayers;
  static constexpr size_t kModelDim = TConfig::kModelDim;

  static const float kEmbScaling = sqrtf(static_cast<float>(kModelDim));

  Decompress(c_weights.c_embedder_input_embedding, token * kModelDim,
             activations.x.data(), kModelDim);

  MulByConst(kEmbScaling, activations.x.data(), kModelDim);

  for (size_t layer = 0; layer < kLayers; ++layer) {
    const CompressedLayer<TConfig>* c_layer = c_weights.CLayer(layer);
    RMSNorm(activations.x.data(), c_layer->c_pre_attention_norm_scale.data(),
            activations.pre_att_rms_out.data(), kModelDim);
    Attention<TConfig, 1>(pos, 0, layer, activations, c_layer, kv_cache, pool);
    AddFrom(activations.att_post2.data(), activations.x.data(), kModelDim);
    RMSNorm(activations.x.data(), c_layer->c_pre_ffw_norm_scale.data(),
            activations.bf_pre_ffw_rms_out.data(), kModelDim);
    FFW<TConfig, 1>(activations, /* batch_idx = */ 0, c_layer, pool);
    AddFrom(activations.ffw_out.data(), activations.x.data(), kModelDim);
  }
  RMSNormInplace(c_weights.c_final_norm_scale.data(), activations.x.data(),
                 kModelDim);
}

template <class TConfig>
void GenerateImpl(GemmaImpl<TConfig>& gemma, const InferenceArgs& args,
                  const std::vector<int>& prompt, size_t pos,
                  hwy::ThreadPool& pool, hwy::ThreadPool& inner_pool,
                  const StreamFunc& stream_token,
                  const AcceptFunc& accept_token, std::mt19937& gen,
                  int verbosity) {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kVocabSize = TConfig::kVocabSize;
  static constexpr size_t kTopK = TConfig::kTopK;
  Activations<TConfig, 1>& activations = *gemma.state.get();
  Activations<TConfig, kPrefillBatchSize>& prefill_activations =
      *gemma.prefill.get();
  const CompressedWeights<TConfig>& c_weights =
      *reinterpret_cast<CompressedWeights<TConfig>*>(
          gemma.compressed_weights.get());
  KVCache& kv_cache = gemma.kv_cache;
  int token;

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
  double prefill_start = hwy::platform::Now();

  // Prefill stops before prompt.size() - 1 since the last prompt token is the
  // first input token for generation.
  while (pos_offset < prompt.size() - 1) {
    const size_t end_offset =
        std::min(kPrefillBatchSize, prompt.size() - 1 - pos_offset);
    HWY_DASSERT(end_offset < prompt.size());
    const int* batch_tokens = prompt.data() + pos_offset;
    Prefill<TConfig, kPrefillBatchSize>(batch_tokens, end_offset, pos,
                                        c_weights, prefill_activations,
                                        kv_cache, pool, inner_pool);
    for (size_t idx = 0; idx < end_offset; ++idx) {
      stream_token(batch_tokens[idx], 0.0);
    }
    pos += end_offset;
    pos_offset += end_offset;
  }

  if (verbosity >= 2) {
    // in the future this output should not occur in GenerateImpl but instead
    // should be available as observable state for frontend code to handle I/O.
    double prefill_end = hwy::platform::Now();
    const double prefill_tok_sec = pos_offset / (prefill_end - prefill_start);
    std::cout << "\n[ Prefill tokens / sec = " << prefill_tok_sec << " ]\n";
  }

  double gen_start = hwy::platform::Now();

  HWY_DASSERT(pos_offset == prompt.size() - 1);

  if (verbosity >= 2) {
    // Provide usage warnings if max_new_tokens is out of range.
    if (args.max_generated_tokens > args.max_tokens) {
      std::cout << "Warning: max_new_tokens should be <= max_tokens"
                << std::endl;
    } else if ((prompt.size() + args.max_generated_tokens) > args.max_tokens) {
      std::cout << "Warning: Prompt size + max_new_tokens exceeds max_tokens."
                << std::endl;
    }
  }

  auto pos_gen_start = pos_offset;
  token = prompt.at(pos_offset);
  size_t generate_pos = 0;
  for (; pos < args.max_tokens && generate_pos < args.max_generated_tokens;
       ++pos, ++pos_offset, ++generate_pos) {
    Transformer(token, pos, c_weights, activations, kv_cache, pool, inner_pool);
    float* final_activation = activations.x.data();
    if (pos_offset >= prompt.size()) {
      PROFILER_ZONE("Gen.Embedding");
      // Generation phase
      MatVec<kVocabSize, kModelDim>(c_weights.c_embedder_input_embedding, 0,
                                    final_activation, activations.logits.data(),
                                    pool);
      // Barrier: must have all logits so we can subtract max.
      Softmax(activations.logits.data(), kVocabSize);
      token = SampleTopK<kTopK>(activations.logits.data(), kVocabSize, gen,
                                args.temperature, accept_token);
    }
    if (!stream_token(token, activations.logits[token])) {
      token = EOS_ID;
    }
    if (token == EOS_ID) {
      if (verbosity >= 2) {
        double gen_end = hwy::platform::Now();
        const double gen_tok_sec =
            (pos_offset - pos_gen_start) / (gen_end - gen_start);
        std::cout << "\n[ Generation tokens / sec = " << gen_tok_sec << " ]\n";
      }
      break;
    }
  }
}

void Generate2B(GemmaImpl<ConfigGemma2B>& gemma, const InferenceArgs& args,
                const std::vector<int>& prompt, size_t start_pos,
                hwy::ThreadPool& pool, hwy::ThreadPool& inner_pool,
                const StreamFunc& stream_token, const AcceptFunc& accept_token,
                std::mt19937& gen, int verbosity) {
  GenerateImpl(gemma, args, prompt, start_pos, pool, inner_pool, stream_token,
               accept_token, gen, verbosity);
}

void Generate7B(GemmaImpl<ConfigGemma7B>& gemma, const InferenceArgs& args,
                const std::vector<int>& prompt, size_t start_pos,
                hwy::ThreadPool& pool, hwy::ThreadPool& inner_pool,
                const StreamFunc& stream_token, const AcceptFunc& accept_token,
                std::mt19937& gen, int verbosity) {
  GenerateImpl(gemma, args, prompt, start_pos, pool, inner_pool, stream_token,
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
       c_weights.c_embedder_input_embedding);
  func("c_final_norm", weights ? weights->final_norm_scale.data() : nullptr,
       c_weights.c_final_norm_scale);

  char name[16];
  for (int layer_idx = 0; layer_idx < static_cast<int>(TConfig::kLayers);
       ++layer_idx) {
    const size_t idx = static_cast<size_t>(layer_idx);
    Layer<TConfig>* layer = weights ? &weights->layers[idx] : nullptr;
    CompressedLayer<TConfig>* c_layer = c_weights.CLayer(idx);

    snprintf(name, sizeof(name), "pre_ff_ns_%d", layer_idx);
    func(name, layer ? layer->pre_ffw_norm_scale.data() : nullptr,
         c_layer->c_pre_ffw_norm_scale);

    snprintf(name, sizeof(name), "gating_ein_%d", layer_idx);
    func(name, layer ? layer->gating_einsum_w.data() : nullptr,
         c_layer->c_gating_einsum_w);

    snprintf(name, sizeof(name), "linear_w_%d", layer_idx);
    func(name, layer ? layer->linear_w.data() : nullptr, c_layer->c_linear_w);
    snprintf(name, sizeof(name), "qkv_ein_%d", layer_idx);

    func(name, layer ? layer->qkv_einsum_w.data() : nullptr,
         c_layer->c_qkv_einsum_w);
    snprintf(name, sizeof(name), "att_ein_%d", layer_idx);

    func(name, layer ? layer->attn_vec_einsum_w.data() : nullptr,
         c_layer->c_attn_vec_einsum_w);

    snprintf(name, sizeof(name), "pre_att_ns_%d", layer_idx);
    func(name, layer ? layer->pre_attention_norm_scale.data() : nullptr,
         c_layer->c_pre_attention_norm_scale);
  }
}

template <class TConfig>
hwy::AlignedFreeUniquePtr<uint8_t[]> GetCompressedWeights(
    const Path& model, const Path& cache, hwy::ThreadPool& pool) {
  PROFILER_ZONE("Startup.LoadCache");

  if (!std::filesystem::exists(model.path) &&
      !std::filesystem::exists(cache.path)) {
    HWY_ABORT(
        "Either the model weights (--weights) or cached compressed weights "
        "(--compressed_weights) must exist.");
  }

  // Allocate compressed weights.
  using CWeights = CompressedWeights<TConfig>;
  hwy::AlignedFreeUniquePtr<uint8_t[]> c_weights_u8 =
      hwy::AllocateAligned<uint8_t>(sizeof(CWeights));
  CWeights* c_weights = reinterpret_cast<CWeights*>(c_weights_u8.get());
  new (&c_weights->c_layer_ptrs) CompressedLayerPointers<TConfig>(pool);

  // First attempt to load them from cache, without requiring weights.
  CacheLoader loader(cache.path.c_str());
  ForEachTensor<TConfig>(nullptr, *c_weights, loader);
  if (loader.ReadAll(pool)) return c_weights_u8;

  // Get weights, compress, and store in cache.
  hwy::AlignedUniquePtr<Weights<TConfig>> weights = LoadWeights<TConfig>(model);
  Compressor compressor(pool);
  ForEachTensor<TConfig>(weights.get(), *c_weights, compressor);
  compressor.WriteAll(pool, cache.path.c_str());

  return c_weights_u8;
}

// Type-erased because this function is called via a function pointer.
hwy::AlignedFreeUniquePtr<uint8_t[]> GetCompressedWeightsT(
    const LoaderArgs& args, hwy::ThreadPool& pool) {
  switch (args.ModelType()) {
    case Model::GEMMA_2B:
      return GetCompressedWeights<ConfigGemma2B>(args.model, args.cache, pool);
    case Model::GEMMA_7B:
      return GetCompressedWeights<ConfigGemma7B>(args.model, args.cache, pool);
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(args.ModelType()));
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {

HWY_EXPORT(GetCompressedWeightsT);
HWY_EXPORT(Generate2B);
HWY_EXPORT(Generate7B);

KVCache CreateKVCache(size_t size_cache_pos, size_t seq_len) {
  KVCache kv_cache = {};
  kv_cache.key_cache = hwy::AllocateAligned<float>(seq_len * size_cache_pos);
  kv_cache.value_cache = hwy::AllocateAligned<float>(seq_len * size_cache_pos);
  return kv_cache;
}

template <class Config>
GemmaImpl<Config>::GemmaImpl(const LoaderArgs& args, hwy::ThreadPool& pool)
    : compressed_weights(
          HWY_DYNAMIC_DISPATCH(GetCompressedWeightsT)(args, pool)),
      prefill(hwy::MakeUniqueAligned<Activations<Config, kPrefillBatchSize>>()),
      state(hwy::MakeUniqueAligned<Activations<Config, 1>>()),
      kv_cache(
          CreateKVCache(Config::kLayers * Config::kKVHeads * Config::kQKVDim,
                        Config::kSeqLen)) {
  PROFILER_ZONE("Startup.tokenizer");

  HWY_ASSERT(tokenizer.Load(args.tokenizer.path).ok());
}

template <>
void GemmaImpl<ConfigGemma2B>::Generate(const InferenceArgs& args,
                                        const std::vector<int>& prompt,
                                        size_t start_pos, hwy::ThreadPool& pool,
                                        hwy::ThreadPool& inner_pool,
                                        const StreamFunc& stream_token,
                                        const AcceptFunc& accept_token,
                                        std::mt19937& gen, int verbosity) {
  HWY_DYNAMIC_DISPATCH(Generate2B)
  (*this, args, prompt, start_pos, pool, inner_pool, stream_token, accept_token,
   gen, verbosity);
}
template <>
void GemmaImpl<ConfigGemma7B>::Generate(const InferenceArgs& args,
                                        const std::vector<int>& prompt,
                                        size_t start_pos, hwy::ThreadPool& pool,
                                        hwy::ThreadPool& inner_pool,
                                        const StreamFunc& stream_token,
                                        const AcceptFunc& accept_token,
                                        std::mt19937& gen, int verbosity) {
  HWY_DYNAMIC_DISPATCH(Generate7B)
  (*this, args, prompt, start_pos, pool, inner_pool, stream_token, accept_token,
   gen, verbosity);
}

Gemma::Gemma(const LoaderArgs& args, hwy::ThreadPool& pool) {
  const Model model_type = args.ModelType();
  model_training = args.ModelTraining();
  switch (model_type) {
    case Model::GEMMA_2B:
      impl_.reset(new GemmaImpl<ConfigGemma2B>(args, pool));
      break;
    case Model::GEMMA_7B:
      impl_.reset(new GemmaImpl<ConfigGemma7B>(args, pool));
      break;
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(model_type));
  }
}
Gemma::~Gemma() = default;  // after GemmaInterface is defined

const sentencepiece::SentencePieceProcessor& Gemma::Tokenizer() const {
  return impl_->Tokenizer();
}

void GenerateGemma(Gemma& gemma, const InferenceArgs& args,
                   const std::vector<int>& prompt, size_t start_pos,
                   hwy::ThreadPool& pool, hwy::ThreadPool& inner_pool,
                   const StreamFunc& stream_token,
                   const AcceptFunc& accept_token, std::mt19937& gen,
                   int verbosity) {
  pool.SetWaitMode(hwy::PoolWaitMode::kSpin);
  gemma.impl_->Generate(args, prompt, start_pos, pool, inner_pool, stream_token,
                        accept_token, gen, verbosity);
  pool.SetWaitMode(hwy::PoolWaitMode::kBlock);
}

}  // namespace gcpp
#endif  // HWY_ONCE
