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
#include "gemma/ops.h"
#include "hwy/contrib/matvec/matvec-inl.h"
#include "hwy/highway.h"

// Non-SIMD includes and types. Note that HWY_ONCE is only true on the last
// compile pass, whereas we want this defined in the first.
#ifndef GEMMA_ONCE
#define GEMMA_ONCE

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "compression/io.h"  // Path
#include "gemma/common.h"
#include "gemma/configs.h"
#include "gemma/gemma.h"
#include "gemma/weights.h"
// Placeholder for internal test1, do not remove
// Placeholder for internal test4, do not remove
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"
#include "hwy/timer.h"
// copybara:import_next_line:sentencepiece
#include "src/sentencepiece_processor.h"

namespace gcpp {

// Set this to true to debug tokenizer tokens.
constexpr bool kShowTokenization = false;

// Must be aligned.
template <class TConfig, size_t kBatchSize>
struct Activations {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kKVHeads = TConfig::kKVHeads;
  static constexpr bool kIsMHA = kHeads == kKVHeads;  // Multi-Head Attention
  // Stride between subsequent queries. Each of Q, K, V are of length kQKVDim,
  // but for MHA we store them as Q,K,V, Q,K,V, .. instead of Q..Q, K..K, V..V.
  static constexpr size_t kQStride = kQKVDim * (kIsMHA ? 3 : 1);

  std::array<float, kBatchSize * kModelDim> x;  // input
  std::array<float, kBatchSize * kModelDim> pre_att_rms_out;
  std::array<float, kBatchSize * kHeads * kQStride>
      q;  // query vector
  std::array<float, kBatchSize * kHeads * TConfig::kSeqLen>
      att;  // attention vector
  std::array<float, kBatchSize * kHeads * kQKVDim>
      att_out;  // attention output
  std::array<float, kHeads * kBatchSize * kModelDim>
      att_post1;  // attention output after linear transformation, per head
  std::array<float, kBatchSize * kModelDim>
      att_post2;  // accumulation of attention outputs over heads
  std::array<hwy::bfloat16_t, kBatchSize * kModelDim>
      bf_pre_ffw_rms_out;
  std::array<float, kBatchSize * TConfig::kFFHiddenDim * 2>
      ffw_hidden;

  // For FFW MatMul.
  std::array<float, kBatchSize * TConfig::kFFHiddenDim> C1;
  std::array<float, kBatchSize * TConfig::kFFHiddenDim> C2;

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
  std::array<float, kBatchSize * kGriffinDim>
      griffin_multiplier;
};

namespace {

template <class TConfig>
struct CreateKVCache {
  KVCache operator()() const {
    KVCache kv_cache = {};

    const size_t size_cache_pos = CachePosSize<TConfig>()();
    if (size_cache_pos != 0) {
      const size_t seq_len =
          (TConfig::kSeqLen + kPrefillBatchSize);
      kv_cache.kv_cache = hwy::AllocateAligned<float>(seq_len * size_cache_pos);
    }

    // TODO(patrickms): Add query batching support for Griffin.
    if (TConfig::kGriffinLayers) {
      constexpr size_t kConv1dWidth = TConfig::kConv1dWidth;
      const size_t conv1d_cache_size =
          TConfig::kGriffinLayers * (kConv1dWidth == 0 ? 0 : kConv1dWidth - 1) *
          TConfig::kModelDim;
      if (conv1d_cache_size != 0) {
        kv_cache.conv1d_cache = hwy::AllocateAligned<float>(conv1d_cache_size);
        hwy::ZeroBytes(kv_cache.conv1d_cache.get(),
                       conv1d_cache_size * sizeof(kv_cache.conv1d_cache[0]));
      }

      const size_t rglru_cache_size =
          TConfig::kGriffinLayers * TConfig::kModelDim;
      if (rglru_cache_size != 0) {
        kv_cache.rglru_cache = hwy::AllocateAligned<float>(rglru_cache_size);
        hwy::ZeroBytes(kv_cache.rglru_cache.get(),
                       rglru_cache_size * sizeof(kv_cache.rglru_cache[0]));
      }
    }  // kGriffinLayers

    return kv_cache;
  }
};

}  // namespace

KVCache KVCache::Create(Model model_type) {
  // TWeight=float is a placeholder and unused because CreateKVCache does not
  // use TConfig::Weight.
  return CallForModel</*TWeight=*/float, CreateKVCache>(model_type);
}

class GemmaTokenizer::Impl {
 public:
  Impl() = default;
  explicit Impl(const Path& tokenizer_path) {
    PROFILER_ZONE("Startup.tokenizer");
    spp_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
    if (!spp_->Load(tokenizer_path.path).ok()) {
      HWY_ABORT("Failed to load the tokenizer file.");
    }
  }

  bool Encode(const std::string& input,
              std::vector<std::string>* pieces) const {
    return spp_ && spp_->Encode(input, pieces).ok();
  }

  bool Encode(const std::string& input, std::vector<int>* ids) const {
    if constexpr (kShowTokenization) {
      bool is_ok = spp_ && spp_->Encode(input, ids).ok();
      for (int i = 0; i < static_cast<int>(ids->size()); i++) {
        fprintf(stderr, "%3d: %d\n", i, (*ids)[i]);
      }
      return is_ok;
    } else {
      return spp_ && spp_->Encode(input, ids).ok();
    }
  }

  // Given a sequence of ids, decodes it into a detokenized output.
  bool Decode(const std::vector<int>& ids, std::string* detokenized) const {
    return spp_ && spp_->Decode(ids, detokenized).ok();
  }

 private:
  std::unique_ptr<sentencepiece::SentencePieceProcessor> spp_;
};

GemmaTokenizer::GemmaTokenizer(const Path& tokenizer_path) {
  impl_ = std::make_unique<Impl>(tokenizer_path);
}

// Default suffices, but they must be defined after GemmaTokenizer::Impl.
GemmaTokenizer::GemmaTokenizer() = default;
GemmaTokenizer::~GemmaTokenizer() = default;
GemmaTokenizer::GemmaTokenizer(GemmaTokenizer&& other) = default;
GemmaTokenizer& GemmaTokenizer::operator=(GemmaTokenizer&& other) = default;

bool GemmaTokenizer::Encode(const std::string& input,
                            std::vector<std::string>* pieces) const {
  return impl_->Encode(input, pieces);
}

bool GemmaTokenizer::Encode(const std::string& input,
                            std::vector<int>* ids) const {
  return impl_->Encode(input, ids);
}

// Given a sequence of ids, decodes it into a detokenized output.
bool GemmaTokenizer::Decode(const std::vector<int>& ids,
                            std::string* detokenized) const {
  return impl_->Decode(ids, detokenized);
}

// Placeholder for internal test2, do not remove

}  // namespace gcpp
#endif  // GEMMA_ONCE

// SIMD code, compiled once per target.
HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace {

template <class TConfig, size_t kBatchSize, size_t kQueryBatchSize>
HWY_NOINLINE void GriffinRecurrent(
    size_t batch_start, size_t num_tokens, size_t num_queries, size_t layer,
    Activations<TConfig, kBatchSize * kQueryBatchSize>& activations,
    const CompressedLayer<TConfig>* layer_weights,
    const std::vector<KVCache*>& kv_caches, hwy::ThreadPool& pool) {
  PROFILER_ZONE("Gen.Griffin");
  static_assert(kQueryBatchSize == 1,
                "Griffin does not support batched queries.");
  HWY_DASSERT(num_queries == 1);  // TODO: add batch query support for Griffin.
  KVCache& kv_cache = *kv_caches[0];
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  HWY_DASSERT(num_tokens <= kBatchSize);
  static constexpr size_t kModelDim =
      gcpp::Activations<TConfig, kBatchSize * kQueryBatchSize>::kModelDim;
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
    HWY_DASSERT(kModelDim % hn::Lanes(df) == 0);
    const size_t layer_offset = layer * kModelDim * (kConv1dWidth - 1);

    // cache[i] = input at time t-i.
    float* HWY_RESTRICT cache[HWY_MAX(kConv1dWidth, 1)];
    cache[0] = x;
    for (size_t i = 1; i < kConv1dWidth; i++) {
      cache[i] =
          kv_cache.conv1d_cache.get() + layer_offset +
          ((pos + kConv1dWidth - 1 - i) % (kConv1dWidth - 1)) * kModelDim;
    }
    for (size_t i = 0; i < kModelDim; i += hn::Lanes(df)) {
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
      HWY_DASSERT(kHeadDim % hn::Lanes(df) == 0);
      for (size_t i = 0; i < kHeadDim; i += hn::Lanes(df)) {
        auto log_a = hn::Load(df, a + head_offset + i);
        auto gated_x = hn::Load(df, x + head_offset + i);
        auto rnn = hn::Load(df, rnn_state + head_offset + i);
        auto a = hn::Exp(df, log_a);
        auto x_multiplier = hn::Sqrt(hn::NegMulAdd(a, a, hn::Set(df, 1.0f)));
        if (pos == 0) {
          x_multiplier = hn::Set(df, 1.0f);
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

template <class TConfig, size_t kBatchSize, size_t kQueryBatchSize>
HWY_NOINLINE void Attention(
    size_t batch_and_query_start, size_t num_tokens, size_t num_queries,
    size_t layer,
    Activations<TConfig, kBatchSize * kQueryBatchSize>& activations,
    const CompressedLayer<TConfig>* layer_weights,
    const std::vector<KVCache*>& kv_caches,
    hwy::ThreadPool& pool) {
  PROFILER_ZONE("Gen.Attention");
  HWY_DASSERT(num_tokens <= kBatchSize);
  HWY_DASSERT(num_queries <= kQueryBatchSize);
  HWY_DASSERT(batch_and_query_start % num_queries == 0);
  using TActivations = Activations<TConfig, kBatchSize * kQueryBatchSize>;
  constexpr size_t kQKVDim = TActivations::kQKVDim;
  constexpr size_t kQStride = TActivations::kQStride;
  constexpr size_t kCachePosSize = CachePosSize<TConfig>()();
  constexpr size_t kCacheLayerSize = CacheLayerSize<TConfig>()();
  constexpr size_t kModelDim = TActivations::kModelDim;
  constexpr size_t kHeads = TConfig::kHeads;
  constexpr size_t kKVHeads = TConfig::kKVHeads;
  constexpr size_t kSeqLen = TConfig::kSeqLen;
  GEMMA_CONSTEXPR_SQRT const float kQueryScale =
      1.0f / Sqrt(static_cast<float>(kQKVDim));
  constexpr bool kIsMHA = TActivations::kIsMHA;  // Multi-Head Attention
  const size_t batch_start = batch_and_query_start / num_queries;
  const size_t num_tokens_and_queries = num_tokens * num_queries;

  // If MHA, this also computes KV, which we copy to the KV cache below.
  static_assert(!kIsMHA || TConfig::kInterleaveQKV);  // MHA => interleaved
  MatMul_4x4_Batch<kModelDim, kHeads * kQStride>(
      num_tokens_and_queries, activations.pre_att_rms_out.data(),
      layer_weights->qkv_einsum_w.data(), activations.q.data(), pool);

  for (size_t batch_and_query_idx = 0;
       batch_and_query_idx < num_tokens_and_queries; ++batch_and_query_idx) {
    const float* x = activations.pre_att_rms_out.data() + batch_and_query_idx
                     * kModelDim;
    const size_t query_idx = batch_and_query_idx % num_queries;
    const size_t batch_idx = batch_and_query_idx / num_queries;
    KVCache& kv_cache = *kv_caches[query_idx];
    // QKV projections:
    if constexpr (!kIsMHA) {
      const size_t pos = batch_start + batch_idx;
      const size_t cache_pos = pos % (kSeqLen + kPrefillBatchSize);
      const size_t kv_offset =
          cache_pos * kCachePosSize + layer * kCacheLayerSize;
      float* HWY_RESTRICT kv = kv_cache.kv_cache.get() + kv_offset;
      // TODO: requires MatMul support for offsets.
      MatVec<kKVHeads * kQKVDim * 2, kModelDim>(
          layer_weights->qkv_einsum_w, kHeads * kQKVDim * kModelDim, x,
          activations.even_odd.data(), kv, pool);
    }
  }

  // Positional encodings for kv:
  pool.Run(
      0, kKVHeads * num_tokens_and_queries,
      [&](uint64_t task, size_t thread) HWY_ATTR {
        const size_t head = task % kKVHeads;
        const size_t batch_and_query_idx = task / kKVHeads;
        const size_t query_idx = batch_and_query_idx % num_queries;
        const size_t batch_idx = batch_and_query_idx / num_queries;
        const size_t pos = batch_start + batch_idx;
        const size_t cache_pos = pos % (kSeqLen + kPrefillBatchSize);
        const size_t kv_offset = cache_pos * kCachePosSize +
                                 layer * kCacheLayerSize + head * kQKVDim * 2;
        KVCache& kv_cache = *kv_caches[query_idx];
        float* HWY_RESTRICT kv = kv_cache.kv_cache.get() + kv_offset;
        if constexpr (kIsMHA) {
          // For MHA, copy kv into the KV cache from scratch space (see above).
          const float* HWY_RESTRICT q =
              activations.q.data() + (batch_and_query_idx * kHeads
                                      + head) * kQStride;
          // Skip past the Q part of `q`, and copy KV to `kv`.
          memcpy(kv, q + kQKVDim, 2 * kQKVDim * sizeof(float));
        }
        Rope(kv, TConfig::kUseHalfRope ? kQKVDim / 2 : kQKVDim, pos);
      });

  static_assert((kHeads % kKVHeads) == 0,
                "query heads must be a multiple of key-value heads");
  constexpr size_t kGroupHeads = kHeads / kKVHeads;
  pool.Run(0, kHeads * num_tokens_and_queries,
           [&](uint64_t task, size_t thread) HWY_ATTR {
    const size_t head = task % kHeads;
    const size_t batch_and_query_idx = task / kHeads;
    const size_t query_idx = batch_and_query_idx % num_queries;
    const size_t batch_idx = batch_and_query_idx / num_queries;
    const size_t head_offset = (head / kGroupHeads) * kQKVDim * 2;
    KVCache& kv_cache = *kv_caches[query_idx];
    float* HWY_RESTRICT q =
        activations.q.data() + (batch_and_query_idx * kHeads + head) * kQStride;

    const size_t pos = batch_start + batch_idx;
    // Calculate scores
    float* HWY_RESTRICT head_att =
        activations.att.data() + head * kSeqLen
        + batch_and_query_idx * kHeads * kSeqLen;

    Rope(q, TConfig::kUseHalfRope ? kQKVDim / 2 : kQKVDim, pos);
    MulByConst(kQueryScale, q, kQKVDim);

    // Compute Q dot K scores
    const size_t start_pos = pos - std::min(kSeqLen - 1, pos);
    for (size_t pos2 = start_pos; pos2 <= pos; ++pos2) {
      const size_t cache_pos = pos2 % (kSeqLen + kPrefillBatchSize);
      const size_t kv_offset =
          cache_pos * kCachePosSize + layer * kCacheLayerSize + head_offset;
      const float* HWY_RESTRICT k2 = kv_cache.kv_cache.get() + kv_offset;
      const float score = Dot(q, k2, kQKVDim);
      head_att[pos2 % kSeqLen] = score;
    }
    const size_t head_att_len = std::min(pos + 1, kSeqLen);
    if constexpr (TConfig::kAttCap > 0.0f) {
      LogitsSoftCap(TConfig::kAttCap, head_att, head_att_len);
    }
    Softmax(head_att, head_att_len);

    // Weighted summation
    float* HWY_RESTRICT att_out = activations.att_out.data() + head * kQKVDim +
                                  batch_and_query_idx * kHeads * kQKVDim;
    hwy::ZeroBytes(att_out, kQKVDim * sizeof(*att_out));
    for (size_t pos2 = start_pos; pos2 <= pos; ++pos2) {
      const size_t cache_pos = pos2 % (kSeqLen + kPrefillBatchSize);
      const size_t kv_offset =
          cache_pos * kCachePosSize + layer * kCacheLayerSize + head_offset;
      float* HWY_RESTRICT v2 = kv_cache.kv_cache.get() + kv_offset + kQKVDim;
      MulByConstAndAdd(head_att[pos2 % kSeqLen], v2, att_out, kQKVDim);
    }
  });

  for (size_t batch_and_query_idx = 0;
       batch_and_query_idx < num_tokens_and_queries; ++batch_and_query_idx) {
    // TODO(szabadka) Use a single MatVecAdd like in GriffinRecurrent() after
    // rearranging the weights.
    float* HWY_RESTRICT att_out =
        activations.att_out.data() + batch_and_query_idx * kHeads * kQKVDim;
    float* HWY_RESTRICT layer_out =
        activations.att_post2.data() + batch_and_query_idx * kModelDim;
    MatVecT</*kAdd=*/TConfig::kSoftmaxAttnOutputBiases, kModelDim, kQKVDim>(
        layer_weights->attn_vec_einsum_w, 0, att_out,
        layer_weights->attention_output_biases.data(),
        activations.even_odd.data(), layer_out, pool);
    for (size_t head = 1; head < kHeads; ++head) {
      // TODO(patrickms): Check this calculation
      float* HWY_RESTRICT head_out =
          activations.att_post1.data() +
          head * kBatchSize * kQueryBatchSize * kModelDim;
      // TODO: requires MatMul support for offsets.
      MatVec<kModelDim, kQKVDim>(
          layer_weights->attn_vec_einsum_w, head * kModelDim * kQKVDim,
          att_out + head * kQKVDim,
          activations.even_odd.data(), head_out, pool);
      AddFrom(head_out, layer_out, kModelDim);
    }
  }
}

template <class TConfig, size_t kBatchSize>
HWY_NOINLINE void FFW(Activations<TConfig, kBatchSize>& activations,
                      size_t num_tokens,
                      const CompressedLayer<TConfig>* layer_weights,
                      hwy::ThreadPool& pool) {
  HWY_DASSERT(num_tokens <= kBatchSize);
  constexpr size_t kModelDim = TConfig::kModelDim;
  constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;
  float* HWY_RESTRICT even_odd = activations.even_odd.data();

  // TODO: MatMul does not yet support adding another matrix to the result.
  if constexpr (!TConfig::kFFBiases) {
    PROFILER_ZONE("Gen.FFW.GatedGELU");

    // MatMul expects col-major B, which is what we have: kModelDim consecutive
    // elements in memory, repeated kFFHiddenDim times.
    const auto b1 = layer_weights->gating_einsum_w.data();
    constexpr size_t kColsA = kModelDim;
    constexpr size_t kColsB = kFFHiddenDim;
    const auto b2 = b1 + kColsA * kColsB;
    auto A = activations.bf_pre_ffw_rms_out.data();
    // Will go through GELU.
    MatMul_4x4_Batch<kColsA, kColsB>(num_tokens, A, b1, activations.C1.data(),
                                     pool);
    // What to multiply by.
    MatMul_4x4_Batch<kColsA, kColsB>(num_tokens, A, b2, activations.C2.data(),
                                     pool);

    // Gelu and multiply by gate.
    namespace hn = hwy::HWY_NAMESPACE;
    using DF = hn::ScalableTag<float>;
    using VF = hn::Vec<DF>;
    hn::Transform1(DF(), activations.C1.data(), kFFHiddenDim * num_tokens,
                   activations.C2.data(), [](DF df, VF v, VF mul) HWY_ATTR {
                     return hn::Mul(mul, Gelu(df, v));
                   });

    MatMul_4x4_Batch<kFFHiddenDim, kModelDim>(num_tokens, activations.C1.data(),
                                              layer_weights->linear_w.data(),
                                              activations.ffw_out.data(), pool);
  } else {
    for (size_t batch_idx = 0; batch_idx < num_tokens; ++batch_idx) {
      const size_t hidden_offset = batch_idx * kFFHiddenDim * 2;
      const hwy::bfloat16_t* HWY_RESTRICT vec =
          activations.bf_pre_ffw_rms_out.data() + batch_idx * kModelDim;
      float* HWY_RESTRICT out = activations.ffw_hidden.data() + hidden_offset;
      float* HWY_RESTRICT out_mul = out + kFFHiddenDim;

      PROFILER_ZONE("Gen.FFW.GatedGELU");
      // Same matrix, first and second half of rows. Could fuse into one MatVec.
      MatVecT<TConfig::kFFBiases, kFFHiddenDim, kModelDim>(
          layer_weights->gating_einsum_w, kFFHiddenDim * kModelDim, vec,
          TConfig::kFFBiases
              ? layer_weights->ffw_gating_biases.data() + kFFHiddenDim
              : nullptr,
          even_odd, out_mul, pool);
      // Gate, will go through the nonlinearity.
      MatVecT<TConfig::kFFBiases, kFFHiddenDim, kModelDim>(
          layer_weights->gating_einsum_w, 0, vec,
          layer_weights->ffw_gating_biases.data(), even_odd, out, pool);

      namespace hn = hwy::HWY_NAMESPACE;
      using DF = hn::ScalableTag<float>;
      using VF = hn::Vec<DF>;
      hn::Transform1(DF(), out, kFFHiddenDim, out_mul,
                     [](DF df, VF v, VF mul)
                         HWY_ATTR { return hn::Mul(mul, Gelu(df, v)); });

      MatVecT</*kAdd=*/TConfig::kFFBiases, kModelDim, kFFHiddenDim>(
          layer_weights->linear_w, 0,
          activations.ffw_hidden.data() + hidden_offset,
          layer_weights->ffw_output_biases.data(), even_odd,
          activations.ffw_out.data() + batch_idx * kModelDim, pool);
    }
  }
}

template <class TConfig, size_t kBatchSize>
HWY_NOINLINE void EmbedToken(int token, size_t token_idx, size_t pos,
                             const CompressedWeights<TConfig>& weights,
                             Activations<TConfig, kBatchSize>& activations) {
  constexpr size_t kModelDim = TConfig::kModelDim;
  GEMMA_CONSTEXPR_EMBSCALING const float kEmbScaling =
      EmbeddingScaling<TConfig>();
  HWY_DASSERT(token >= 0);
  HWY_DASSERT(token < TConfig::kVocabSize);
  Decompress(weights.embedder_input_embedding, token * kModelDim,
             activations.x.data() + token_idx * kModelDim, kModelDim);
  MulByConst(kEmbScaling, activations.x.data() + token_idx * kModelDim,
             kModelDim);
  if constexpr (TConfig::kAbsolutePE) {
    AddAbsolutePositionalEmbeddings(
        activations.x.data() + token_idx * kModelDim, kModelDim,
        pos + token_idx);
  };
}

template <class TConfig, size_t kBatchSize, size_t kQueryBatchSize>
HWY_NOINLINE void TransformerLayer(
    size_t num_tokens, size_t num_queries, size_t pos, size_t layer,
    const CompressedLayer<TConfig>* layer_weights,
    Activations<TConfig, kBatchSize * kQueryBatchSize>& activations,
    const std::vector<KVCache*>& kv_caches, hwy::ThreadPool& pool) {
  constexpr size_t kModelDim = TConfig::kModelDim;
  const size_t num_tokens_and_queries = num_tokens * num_queries;
  auto type = TConfig::kLayerConfig[layer];
  size_t layer_of_type =
      NumLayersOfTypeBefore(TConfig::kLayerConfig, type, layer);
  RMSNormBatched<kBatchSize * kQueryBatchSize>(
      num_tokens_and_queries, activations.x.data(),
      layer_weights->pre_attention_norm_scale.data(),
      activations.pre_att_rms_out.data(), kModelDim);
  if (type == LayerAttentionType::kGemma) {
    Attention<TConfig, kBatchSize, kQueryBatchSize>(
        pos, num_tokens, num_queries, layer_of_type, activations,
              layer_weights, kv_caches, pool);
  } else {
    // This Griffin layers should never exist unless the model is a Griffin
    // model. This conditional prevents the compiler from generating code for
    // this branch when building a non-Griffin model, since we have static
    // asserts about the query batch size for Griffin layers.
    if constexpr (TConfig::kGriffinLayers > 0) {
      GriffinRecurrent<TConfig, kBatchSize, kQueryBatchSize>(
          pos, num_tokens, num_queries, layer_of_type, activations,
                      layer_weights, kv_caches, pool);
    }
  }
  if (TConfig::kPostNormScale) {
    RMSNormInplaceBatched<kBatchSize * kQueryBatchSize>(
        num_tokens_and_queries,
        layer_weights->post_attention_norm_scale.data(),
        activations.att_post2.data(), kModelDim);
  }
  AddFromBatched<kBatchSize * kQueryBatchSize>(num_tokens_and_queries,
                                               activations.att_post2.data(),
                                               activations.x.data(), kModelDim);
  RMSNormBatched<kBatchSize * kQueryBatchSize>(
      num_tokens_and_queries, activations.x.data(),
      layer_weights->pre_ffw_norm_scale.data(),
      activations.bf_pre_ffw_rms_out.data(), kModelDim);
  FFW<TConfig, kBatchSize * kQueryBatchSize>(
      activations, num_tokens_and_queries, layer_weights, pool);
  if (TConfig::kPostNormScale) {
    RMSNormInplaceBatched<kBatchSize * kQueryBatchSize>(
        num_tokens_and_queries, layer_weights->post_ffw_norm_scale.data(),
        activations.ffw_out.data(), kModelDim);
  }
  AddFromBatched<kBatchSize * kQueryBatchSize>(
      num_tokens_and_queries, activations.ffw_out.data(),
      activations.x.data(), kModelDim);
}

template <class TConfig, size_t kBatchSize, size_t kQueryBatchSize>
HWY_NOINLINE void Prefill(
    const int* tokens, size_t num_tokens, size_t num_queries, size_t pos,
    const CompressedWeights<TConfig>& weights,
    Activations<TConfig, kBatchSize * kQueryBatchSize>& activations,
    const std::vector<KVCache*>& kv_caches, hwy::ThreadPool& pool) {
  HWY_DASSERT(num_queries <= kQueryBatchSize);
  const size_t minibatch_size = std::min(num_tokens, kBatchSize);
  PROFILER_ZONE("Gen.Prefill\\Att\\FFW");
  // TODO(patrickms): Try to hoist pool.Run out of the loop.
  for (size_t i = 0; i < num_tokens; i += minibatch_size) {
    const size_t offset = i * num_queries;
    const size_t current_token_count = std::min(
        minibatch_size, num_tokens - i);
    pool.Run(0, current_token_count * num_queries,
            [&](const uint64_t token_idx, size_t /*thread*/) HWY_ATTR {
              EmbedToken<TConfig, kBatchSize * kQueryBatchSize>(
                  tokens[token_idx + offset], token_idx, pos + offset,
                  weights, activations);
            });

    for (size_t layer = 0; layer < TConfig::kLayers; ++layer) {
      const auto* layer_weights = weights.GetLayer(layer);
      TransformerLayer<TConfig, kBatchSize, kQueryBatchSize>(
          current_token_count, num_queries, pos + offset , layer, layer_weights,
          activations, kv_caches, pool);
    }
  }
}

// Compute the transformer for a batch of input tokens. During generation,
// we usually have num_tokens == 1 (and also kBatchSize == 1).
template <class TConfig, size_t kBatchSize, size_t kQueryBatchSize>
HWY_NOINLINE void Transformer(
    const int* tokens, size_t num_tokens, size_t num_queries, size_t pos,
    const CompressedWeights<TConfig>& weights,
    Activations<TConfig, kBatchSize * kQueryBatchSize>& activations,
    const std::vector<KVCache*>& kv_caches,
    hwy::ThreadPool& pool,
    const LayersOutputFunc& layers_output) {
  HWY_ASSERT(num_tokens <= kBatchSize);
  const size_t num_tokens_and_queries = num_tokens * num_queries;
  if (layers_output) {
    for (size_t token_idx = 0; token_idx < num_tokens_and_queries;
         ++token_idx) {
      float token_f = tokens[token_idx];
      layers_output(pos + token_idx, "Tokens", &token_f, 1);
    }
  }
  constexpr size_t kModelDim = TConfig::kModelDim;
  for (size_t token_idx = 0; token_idx < num_tokens_and_queries; ++token_idx) {
    EmbedToken<TConfig, kBatchSize * kQueryBatchSize>(
        tokens[token_idx], token_idx, pos, weights, activations);
  }

  for (size_t layer = 0; layer < TConfig::kLayers; ++layer) {
    const CompressedLayer<TConfig>* layer_weights = weights.GetLayer(layer);
    TransformerLayer<TConfig, kBatchSize, kQueryBatchSize>(
        num_tokens, num_queries, pos, layer, layer_weights,
        activations, kv_caches, pool);

    if (layers_output) {
      const std::string block_name = "blocks." + std::to_string(layer);
      for (size_t token_idx = 0; token_idx < num_tokens_and_queries;
           ++token_idx) {
        layers_output(pos + token_idx, block_name,
                      activations.x.data() + token_idx * kModelDim, kModelDim);
      }
    }
  }

  RMSNormInplaceBatched<kBatchSize * kQueryBatchSize>(
      num_tokens * num_queries, weights.final_norm_scale.data(),
      activations.x.data(), kModelDim);
  if (layers_output) {
    for (size_t token_idx = 0; token_idx < num_tokens_and_queries;
         ++token_idx) {
      layers_output(pos + token_idx, "final_norm",
                    activations.x.data() + token_idx * kModelDim, kModelDim);
    }
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

  HWY_ASSERT(prompt_size > 0);
}

template <class TConfig, size_t kBatchSize>
Activations<TConfig, kBatchSize>& GetActivations(
    const ByteStorageT& state_u8) {
  return *reinterpret_cast<Activations<TConfig, kBatchSize>*>(
      state_u8.get());
}

}  // namespace

// Placeholder for internal test3, do not remove

bool StreamToken(size_t query_idx, size_t pos, int token, float weight,
                 const RuntimeConfig& runtime_config) {
  if (runtime_config.batch_stream_token) {
    return runtime_config.batch_stream_token(query_idx, pos, token, weight);
  }
  return runtime_config.stream_token(token, weight);
}

template <class TConfig, size_t kQueryBatchSize>
void GenerateT(const ByteStorageT& weights_u8, const ByteStorageT& prefill_u8,
               const ByteStorageT& decode_u8,
               const RuntimeConfig& runtime_config,
               const hwy::Span<const hwy::Span<int>>& prompts, size_t pos,
               const size_t query_index_offset,
               const std::vector<KVCache*>& kv_caches, hwy::ThreadPool& pool,
               TimingInfo& timing_info) {
  constexpr size_t kAdjustedPrefillBatchSize =
      std::max((size_t)1, kPrefillBatchSize / kQueryBatchSize);
  static_assert(kAdjustedPrefillBatchSize >= kMinAdjustedPrefillBatchSize);
  const size_t num_queries = prompts.size();
  HWY_DASSERT(num_queries <= kQueryBatchSize);
  pos *= num_queries;  // position in (num_queries) interleaved token sequence.
  const CompressedWeights<TConfig>& weights =
      *reinterpret_cast<const CompressedWeights<TConfig>*>(weights_u8.get());
  auto& prefill_activations =
      GetActivations<TConfig,
                     kAdjustedPrefillBatchSize * kQueryBatchSize>(prefill_u8);
  auto& activations = GetActivations<TConfig, kQueryBatchSize>(decode_u8);

  size_t min_prompt_size =  (size_t)-1;
  size_t max_prompt_size = 0;
  for (int i=0; i < prompts.size(); ++i) {
    min_prompt_size = std::min(min_prompt_size, prompts[i].size());
    max_prompt_size = std::max(max_prompt_size, prompts[i].size());
  }

  std::vector<int> prompt;
  prompt.reserve(max_prompt_size * prompts.size());
  for (int i = 0; i < max_prompt_size; ++i) {
    for (int j=0; j < prompts.size(); ++j) {
      if (i < prompts[j].size()) {
        prompt.push_back(prompts[j][i]);
      } else {
        prompt.push_back(0);
      }
    }
  }

  constexpr size_t kVocabSize = TConfig::kVocabSize;

  size_t max_tokens = runtime_config.max_tokens;
  size_t max_generated_tokens = runtime_config.max_generated_tokens;
  RangeChecks<TConfig>(max_tokens, max_generated_tokens, max_prompt_size);
  if (pos >= max_tokens) {
    fprintf(stderr, "Warning: pos %zu >= max_tokens %zu, aborting.\n", pos,
            max_tokens);
    return;
  }

  // If no sample_func is provided, we use top-k sampling.
  const SampleFunc sample_token =
      runtime_config.sample_func
          ? runtime_config.sample_func
          : [&](const float* logits, size_t vocab_size) -> int {
    return SampleTopK<TConfig::kTopK>(logits, vocab_size, *runtime_config.gen,
                                      runtime_config.temperature,
                                      runtime_config.accept_token);
  };

  std::vector<bool> reached_eos(num_queries);
  std::fill(reached_eos.begin(), reached_eos.end(), false);

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
  // Used to keep track of how many tokens are processed per prompt,
  // so that we know when to start generating tokens.
  size_t single_prompt_pos_offset = 0;
  const double prefill_start = hwy::platform::Now();

  // Prefill stops before prompt_size - 1 since the last prompt token is the
  // first input token for generation.
  while (single_prompt_pos_offset < min_prompt_size - 1) {
    const size_t batch_size = std::min(
        kPrefillBatchSize, min_prompt_size - 1 - single_prompt_pos_offset);
    const size_t batch_and_query_size = batch_size * num_queries;
    HWY_DASSERT(batch_size <= kPrefillBatchSize);
    HWY_DASSERT(single_prompt_pos_offset + batch_size <= min_prompt_size - 1);
    HWY_DASSERT(pos_offset + batch_size <= (min_prompt_size - 1) * num_queries);
    const int* batch_tokens = prompt.data() + pos_offset;
    Prefill<TConfig, kAdjustedPrefillBatchSize, kQueryBatchSize>(
        batch_tokens, batch_size, num_queries, pos, weights,
            prefill_activations, kv_caches, pool);
    for (size_t idx = 0; idx < batch_size; ++idx) {
      bool all_tokens_eos = true;
      for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
        if (reached_eos[query_idx]) continue;
        if (StreamToken(
                query_idx + query_index_offset, single_prompt_pos_offset,
                batch_tokens[idx * num_queries + query_idx], 0.0f,
                runtime_config)) {
          all_tokens_eos = false;
        } else {
          reached_eos[query_idx] = true;
        }
      }
      if (all_tokens_eos) {
        return;
      }
    }
    pos += batch_and_query_size;
    pos_offset += batch_and_query_size;
    single_prompt_pos_offset += batch_size;
  }

  timing_info.prefill_tok_sec =
      static_cast<double>(pos_offset) / (hwy::platform::Now() - prefill_start);

  // Start generation.
  const double gen_start = hwy::platform::Now();
  HWY_DASSERT(single_prompt_pos_offset == min_prompt_size - 1);
  size_t pos_gen_start = pos_offset;
  int token = prompt.at(pos_offset);
  std::vector<int>::const_iterator first = prompt.begin() + pos_offset;
  std::vector<int>::const_iterator last = first + num_queries;
  std::vector<int> gen_tokens(first, last);
  // The loop below is not yet prepared for decode batch size > 1.
  HWY_ASSERT(kDecodeBatchSize == 1);
  bool all_tokens_eos = true;
  for (size_t i=0; i < num_queries; ++i) {
    if (reached_eos[i]) continue;
    if (StreamToken(i + query_index_offset,
                    single_prompt_pos_offset, gen_tokens[i], 0.0f,
                    runtime_config)) {
      all_tokens_eos = false;
    } else {
      reached_eos[i] = true;
    }
  }
  if (all_tokens_eos) {
    return;
  }
  for (size_t generate_pos = 0;
       generate_pos < max_tokens && generate_pos < max_generated_tokens;
       ++single_prompt_pos_offset, ++generate_pos) {
    Transformer<TConfig, kDecodeBatchSize, kQueryBatchSize>(
        gen_tokens.data(), kDecodeBatchSize, num_queries, pos, weights,
        activations, kv_caches, pool, runtime_config.layers_output);
    float token_logit = 0.0f;
    // The condition below is always true if we are doing Prefill above.
    // We keep it here for clarity so that the code is correct even if Prefill
    // is disabled.
    bool all_tokens_eos = true;
    float* x = activations.x.data();
    float* logits = activations.logits.data();
    for (size_t i = 0; i < num_queries; ++i, ++pos, ++pos_offset,
                x += TConfig::kModelDim, logits += kVocabSize) {
      const size_t prompt_size = prompts[i].size();
      const bool is_generating_phase =
          (single_prompt_pos_offset >= prompt_size - 1);
      if (is_generating_phase) {
        PROFILER_ZONE("Gen.Embedding");
        // Compute logits from last layer activations.
        MatVec<kVocabSize, TConfig::kModelDim>(
            weights.embedder_input_embedding, 0, x, activations.even_odd.data(),
            logits, pool);
        if constexpr (TConfig::kFinalCap > 0.0f) {
          LogitsSoftCap(TConfig::kFinalCap, activations.logits.data(),
                        kVocabSize);
        }
        // Barrier: must have all logits so we can subtract max.
        Softmax(logits, kVocabSize);
        token = sample_token(logits, kVocabSize);
        token_logit = logits[token];
        if (generate_pos == 0) {
          timing_info.time_to_first_token = hwy::platform::Now() - gen_start;
        }
      } else {
        // We would take this branch if we were not doing Prefill but would
        // process the tokens of the prompt one at a time.
        token = prompt.at(pos_offset);
        token_logit = 0.0f;
      }

      if (!reached_eos[i]) {
        if (!StreamToken(i + query_index_offset, single_prompt_pos_offset+1,
                         token, token_logit, runtime_config)) {
          token = runtime_config.eos_id;
        }
        if (token != runtime_config.eos_id) {
          all_tokens_eos = false;
        } else {
          reached_eos[i] = true;
        }
      }
      gen_tokens[i] = token;
    }
    if (all_tokens_eos) {
      break;
    }
  }
  timing_info.gen_tok_sec = static_cast<double>(pos_offset - pos_gen_start) /
                            (hwy::platform::Now() - gen_start);
}

template <class TConfig>
void GenerateOneQueryT(const ByteStorageT& weights_u8,
                       const ByteStorageT& prefill_u8,
                       const ByteStorageT& decode_u8,
                       const RuntimeConfig& runtime_config,
                       const std::vector<int>& prompt, size_t pos,
                       KVCache& kv_cache, hwy::ThreadPool& pool,
                       TimingInfo& timing_info) {
  std::vector<hwy::Span<int>> prompt_vector = {
    hwy::Span<int>(const_cast<int*>(prompt.data()), prompt.size())};
  const hwy::Span<const hwy::Span<int>> prompts(
      prompt_vector.data(), prompt_vector.size());
  std::vector<KVCache*> kv_caches = {&kv_cache};
  GenerateT<TConfig, 1>(weights_u8, prefill_u8, decode_u8,
                        runtime_config, prompts, pos, 0,
                        kv_caches, pool, timing_info);
}

template <class TConfig>
void GenerateBatchT(const ByteStorageT& weights_u8,
                    const ByteStorageT& prefill_u8,
                    const ByteStorageT& decode_u8,
                    const RuntimeConfig& runtime_config,
                    const hwy::Span<const hwy::Span<int>>& prompts,
                    size_t pos, const std::vector<KVCache*>& kv_caches,
                    hwy::ThreadPool& pool,
                    TimingInfo& timing_info) {
  // Disable query batching for Griffin models.
  constexpr size_t kQueryBatchSize =
      (TConfig::kGriffinLayers > 0) ? 1 : kBatchedQueryBatchSize;
  for (size_t i = 0; i < prompts.size(); i += kQueryBatchSize) {
    const size_t num_queries = std::min(prompts.size() - i, kQueryBatchSize);
    const hwy::Span<const hwy::Span<int>> current_prompts(
      prompts.data() + i, num_queries);
    GenerateT<TConfig, kQueryBatchSize>(weights_u8, prefill_u8, decode_u8,
                                        runtime_config, current_prompts,
                                        pos, i, kv_caches, pool, timing_info);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {

namespace {
template <typename TConfig>
struct AllocateState {
  void operator()(ByteStorageT& prefill, ByteStorageT& decode) const {
    // When batching queries, the prefill batch size is reduced by a factor
    // of kBatchedQueryBatchSize
    prefill = AllocateSizeof<
        Activations<TConfig,
                    kMinAdjustedPrefillBatchSize * kBatchedQueryBatchSize>>();
    decode = AllocateSizeof<
        Activations<TConfig, kDecodeBatchSize * kBatchedQueryBatchSize>>();
  }
};

}  // namespace

Gemma::Gemma(const Path& tokenizer_path, const Path& weights, Model model_type,
             Type weight_type, hwy::ThreadPool& pool)
    : pool_(pool),
      tokenizer_(tokenizer_path),
      model_type_(model_type),
      weight_type_(weight_type) {
  weights_u8_ = LoadCompressedWeights(weights, model_type, weight_type, pool);
  CallForModelAndWeight<AllocateState>(model_type, weight_type, prefill_u8_,
                                       decode_u8_);
}

Gemma::Gemma(GemmaTokenizer&& tokenizer, Model model_type, Type weight_type,
             hwy::ThreadPool& pool)
    : pool_(pool),
      tokenizer_(std::move(tokenizer)),
      model_type_(model_type),
      weight_type_(weight_type) {
  HWY_ASSERT(weight_type == Type::kF32);
  weights_u8_ = CallForModel<float, AllocateCompressedWeights>(
      model_type, pool);
  CallForModelAndWeight<AllocateState>(model_type, weight_type, prefill_u8_,
                                       decode_u8_);
}

Gemma::~Gemma() {
  CallForModelAndWeight<DeleteCompressedWeights>(model_type_, weight_type_,
                                                 weights_u8_);
}

void Gemma::Generate(const RuntimeConfig& runtime_config,
                     const std::vector<int>& prompt, size_t start_pos,
                     KVCache& kv_cache, TimingInfo& timing_info) {
  pool_.SetWaitMode(hwy::PoolWaitMode::kSpin);

  GEMMA_EXPORT_AND_DISPATCH(
      model_type_, weight_type_, GenerateOneQueryT,
      (weights_u8_, prefill_u8_, decode_u8_, runtime_config, prompt, start_pos,
       kv_cache, pool_, timing_info));

  pool_.SetWaitMode(hwy::PoolWaitMode::kBlock);
}

void Gemma::GenerateBatch(const RuntimeConfig& runtime_config,
                          const hwy::Span<const hwy::Span<int>>& prompts,
                          size_t start_pos,
                          const std::vector<KVCache*>& kv_caches,
                          TimingInfo& timing_info) {
  pool_.SetWaitMode(hwy::PoolWaitMode::kSpin);

  GEMMA_EXPORT_AND_DISPATCH(
      model_type_, weight_type_, GenerateBatchT,
      (weights_u8_, prefill_u8_, decode_u8_, runtime_config, prompts, start_pos,
       kv_caches, pool_, timing_info));

  pool_.SetWaitMode(hwy::PoolWaitMode::kBlock);
}

std::vector<int> WrapAndTokenize(const GemmaTokenizer& tokenizer,
                                 const ModelTraining training, size_t pos,
                                 std::string& prompt) {
  // Instruction-tuned models are trained to expect control tokens.
  if (training == ModelTraining::GEMMA_IT) {
    // Prepend "<end_of_turn>" if this is a multi-turn dialogue continuation.
    const std::string start = (pos == 0)
                                  ? "<start_of_turn>user\n"
                                  : "<end_of_turn>\n<start_of_turn>user\n";
    prompt = start + prompt + "<end_of_turn>\n<start_of_turn>model\n";
  }

  std::vector<int> tokens;
  HWY_ASSERT(tokenizer.Encode(prompt, &tokens));
  // Both pre-trained and instruction-tuned require BOS as first token.
  if (pos == 0) {
    tokens.insert(tokens.begin(), BOS_ID);
  }
  return tokens;
}

}  // namespace gcpp
#endif  // HWY_ONCE
