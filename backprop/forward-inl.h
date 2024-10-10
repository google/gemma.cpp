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

// Include guard for non-SIMD code.
#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_FORWARD_INL_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_FORWARD_INL_H_

#include <stddef.h>
#include <stdint.h>

#include <cmath>
#include <vector>

#include "backprop/activations.h"
#include "gemma/common.h"
#include "gemma/configs.h"
#include "util/allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_FORWARD_INL_H_

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_FORWARD_TOGGLE) == defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_FORWARD_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_FORWARD_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_FORWARD_TOGGLE
#endif

#include "hwy/highway.h"
// After highway.h
#include "ops/matvec-inl.h"
#include "ops/ops-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

template <typename ArrayT>
void InputEmbedding(const ArrayT& weights, const std::vector<int>& prompt,
                    const float scaling, float* HWY_RESTRICT output,
                    size_t model_dim, size_t vocab_size) {
  const hn::ScalableTag<float> df;
  HWY_ASSERT(!prompt.empty());
  for (size_t pos = 0; pos < prompt.size() - 1; ++pos) {
    int token = prompt[pos];
    DecompressAndZeroPad(df, MakeSpan(weights.data(), model_dim * vocab_size),
                         token * model_dim, output + pos * model_dim,
                         model_dim);
    MulByConst(scaling, output + pos * model_dim, model_dim);
  }
}

template<typename WT, typename XT, typename OutT>
void ApplyRMSNorm(const WT* HWY_RESTRICT weights, const XT* HWY_RESTRICT x,
                  size_t model_dim, size_t num_tokens,
                  OutT* HWY_RESTRICT output,
                  hwy::ThreadPool& pool) {
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    const size_t offset = pos * model_dim;
    RMSNorm(x + offset, weights, output + offset, model_dim);
  }
}

static HWY_NOINLINE float CrossEntropyLoss(const float* HWY_RESTRICT probs,
                                           const std::vector<int>& prompt,
                                           size_t context_size,
                                           size_t vocab_size,
                                           hwy::ThreadPool& pool) {
  HWY_ASSERT(!prompt.empty());
  float loss = 0.0f;
  for (size_t pos = 0; pos < prompt.size() - 1; ++pos) {
    if (pos + 1 < context_size) {
      continue;  // next token is part of context, don't try to predict it
    }
    const int next_token = prompt[pos + 1];
    loss += std::log(probs[pos * vocab_size + next_token]);
  }
  float scaling = -1.0 / std::log(2.0);
  return loss * scaling;
}

template <typename TConfig, typename LayerT>
void ApplyForwardLayer(const LayerT& weights,
                       ForwardLayer<float, TConfig>& activations,
                       size_t num_tokens, float* HWY_RESTRICT output,
                       const RowVectorBatch<float>& inv_timescale,
                       hwy::ThreadPool& pool) {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kSeqLen = TConfig::kSeqLen;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kHeads = TConfig::kHeads;
  static const float kQueryScale =
      static_cast<float>(1.0 / sqrt(static_cast<double>(kQKVDim)));
  HWY_ASSERT(num_tokens <= kSeqLen);

  ApplyRMSNorm(weights.pre_attention_norm_scale.data(),
               activations.input.data(), kModelDim, num_tokens,
               activations.pre_att_rms_out.data(), pool);

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    MatVec<(kHeads + 2) * kQKVDim, kModelDim>(
        weights.qkv_einsum_w, 0,
        activations.pre_att_rms_out.data() + pos * kModelDim,
        activations.qkv.data() + pos * (kHeads + 2) * kQKVDim, pool);
  }
  const size_t num_tasks = kHeads * num_tokens;

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    float* HWY_RESTRICT k =
        activations.qkv.data() + (pos * (kHeads + 2) + kHeads) * kQKVDim;
    Rope(k, kQKVDim, inv_timescale.Const(), pos);
  }
  pool.Run(0, num_tasks, [&](const uint64_t task, size_t thread) HWY_ATTR {
    const size_t head = task % kHeads;
    const size_t pos = task / kHeads;
    float* HWY_RESTRICT q =
        activations.qkv.data() + (pos * (kHeads + 2) + head) * kQKVDim;
    Rope(q, kQKVDim, inv_timescale.Const(), pos);
    MulByConst(kQueryScale, q, kQKVDim);
  });

  pool.Run(0, num_tasks, [&](const uint64_t task, size_t thread) HWY_ATTR {
    const size_t head = task % kHeads;
    const size_t pos = task / kHeads;
    const float* HWY_RESTRICT q =
        activations.qkv.data() + (pos * (kHeads + 2) + head) * kQKVDim;
    float* HWY_RESTRICT head_att =
        activations.att.data() + (pos * kHeads + head) * kSeqLen;
    for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
      const float* HWY_RESTRICT k2 =
          activations.qkv.data() + (pos2 * (kHeads + 2) + kHeads) * kQKVDim;
      const float score = Dot(q, k2, kQKVDim);
      head_att[pos2] = score;
    }
  });

  pool.Run(0, num_tasks, [&](const uint64_t task, size_t thread) HWY_ATTR {
    const size_t head = task % kHeads;
    const size_t pos = task / kHeads;
    float* HWY_RESTRICT head_att =
        activations.att.data() + (pos * kHeads + head) * kSeqLen;
    Softmax(head_att, pos + 1);
  });

  pool.Run(0, num_tasks, [&](const uint64_t task, size_t thread) HWY_ATTR {
    const size_t head = task % kHeads;
    const size_t pos = task / kHeads;
    const float* HWY_RESTRICT head_att =
        activations.att.data() + (pos * kHeads + head) * kSeqLen;
    float* HWY_RESTRICT att_out =
        activations.att_out.data() + (pos * kHeads + head) * kQKVDim;
    hwy::ZeroBytes(att_out, kQKVDim * sizeof(*att_out));
    for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
      float* HWY_RESTRICT v2 =
          activations.qkv.data() + (pos2 * (kHeads + 2) + kHeads + 1) * kQKVDim;
      MulByConstAndAdd(head_att[pos2], v2, att_out, kQKVDim);
    }
  });

  activations.attention_out.ZeroInit();
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    for (size_t head = 0; head < kHeads; ++head) {
      MatVec<kModelDim, kQKVDim>(
          weights.attn_vec_einsum_w, head * kModelDim * kQKVDim,
          activations.att_out.data() + pos * kHeads * kQKVDim + head * kQKVDim,
          activations.att_post1.data() + pos * kModelDim, pool);
      AddFrom(activations.att_post1.data() + pos * kModelDim,
              activations.attention_out.data() + pos * kModelDim, kModelDim);
    }
  }

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    AddFrom(activations.input.data() + pos * kModelDim,
            activations.attention_out.data() + pos * kModelDim, kModelDim);
  }

  ApplyRMSNorm(weights.pre_ffw_norm_scale.data(),
               activations.attention_out.data(), kModelDim, num_tokens,
               activations.bf_pre_ffw_rms_out.data(), pool);
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    MatVec<kFFHiddenDim * 2, kModelDim>(
        weights.gating_einsum_w, 0,
        activations.bf_pre_ffw_rms_out.data() + pos * kModelDim,
        activations.ffw_hidden.data() + pos * kFFHiddenDim * 2, pool);
  }
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    const size_t hidden_offset = pos * kFFHiddenDim * 2;
    const float* HWY_RESTRICT out =
        activations.ffw_hidden.data() + hidden_offset;
    const float* HWY_RESTRICT out_mul = out + kFFHiddenDim;
    float* HWY_RESTRICT out_gated =
        activations.ffw_hidden_gated.data() + pos * kFFHiddenDim;
    namespace hn = hwy::HWY_NAMESPACE;
    using DF = hn::ScalableTag<float>;
    DF df;
    for (size_t i = 0; i < kFFHiddenDim; i += Lanes(df)) {
      const auto y = hn::Load(df, out + i);
      const auto x = hn::Load(df, out_mul + i);
      hn::Store(hn::Mul(x, Gelu(df, y)), df, out_gated + i);
    }
  }
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    MatVec<kModelDim, kFFHiddenDim>(
        weights.linear_w, 0,
        activations.ffw_hidden_gated.data() + pos * kFFHiddenDim,
        output + pos * kModelDim, pool);
  }
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    AddFrom(activations.attention_out.data() + pos * kModelDim,
            output + pos * kModelDim, kModelDim);
  }
}

template <typename TConfig, typename WeightsT, typename LayerT>
float CrossEntropyLossForwardPass(const std::vector<int>& prompt,
                                  size_t context_size, const WeightsT& weights,
                                  ForwardPass<float, TConfig>& forward,
                                  const RowVectorBatch<float>& inv_timescale,
                                  hwy::ThreadPool& pool) {
  static constexpr size_t kVocabSize = TConfig::kVocabSize;
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kLayers = TConfig::kLayers;
  const float kEmbScaling = EmbeddingScaling<TConfig>();
  static_assert(!TConfig::kAbsolutePE);
  static_assert(TConfig::kPostNorm == PostNormType::None);
  static_assert(TConfig::kKVHeads == 1);

  HWY_DASSERT(context_size > 0);
  HWY_DASSERT(context_size < prompt.size());
  const size_t num_tokens = prompt.size() - 1;

  InputEmbedding(weights.embedder_input_embedding, prompt, kEmbScaling,
                 forward.layers[0].input.data(), kModelDim, kVocabSize);

  for (size_t layer = 0; layer < kLayers; ++layer) {
    auto type = TConfig::kLayerConfig[layer];
    // TODO(szabadka) Implement Griffin layer.
    HWY_ASSERT(type == LayerAttentionType::kGemma);
    float* HWY_RESTRICT output = layer + 1 < kLayers ?
                                 forward.layers[layer + 1].input.data() :
                                 forward.final_layer_output.data();
    ApplyForwardLayer<TConfig, LayerT>(*weights.GetLayer(layer),
                                       forward.layers[layer], num_tokens,
                                       output, inv_timescale, pool);
  }

  ApplyRMSNorm(weights.final_norm_scale.data(),
               forward.final_layer_output.data(),
               kModelDim, num_tokens, forward.final_norm_output.data(), pool);

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    MatVec<kVocabSize, kModelDim>(
        weights.embedder_input_embedding, 0,
        forward.final_norm_output.data() + pos * kModelDim,
        forward.logits.data() + pos * kVocabSize, pool);
  }

  if constexpr (TConfig::kFinalCap > 0.0f) {
    for (size_t pos = 0; pos < num_tokens; ++pos) {
      LogitsSoftCap(TConfig::kFinalCap,
                    forward.logits.data() + pos * kVocabSize, kVocabSize);
    }
  }

  hwy::CopyBytes(forward.logits.data(), forward.probs.data(),
                 num_tokens * kVocabSize * sizeof(forward.logits.At(0)));

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    Softmax(forward.probs.data() + pos * kVocabSize, kVocabSize);
  }

  return CrossEntropyLoss(forward.probs.data(), prompt, context_size,
                          kVocabSize, pool);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
