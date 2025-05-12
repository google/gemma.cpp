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
#include "gemma/common.h"  // EmbeddingScaling
#include "gemma/configs.h"
#include "gemma/weights.h"
#include "util/allocator.h"
#include "util/mat.h"
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

template <typename T>
void InputEmbedding(const MatPtrT<T>& weights, const std::vector<int>& prompt,
                    const float scaling, float* HWY_RESTRICT output,
                    size_t model_dim, size_t vocab_size) {
  const hn::ScalableTag<float> df;
  HWY_ASSERT(!prompt.empty());
  for (size_t pos = 0; pos < prompt.size() - 1; ++pos) {
    int token = prompt[pos];
    const auto span = weights.Span();
    HWY_ASSERT(span.num == model_dim * vocab_size);
    DecompressAndZeroPad(df, span, token * model_dim, output + pos * model_dim,
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
    RMSNorm(x + offset, weights, 0, output + offset, model_dim);
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

template <typename T>
void ApplyForwardLayer(const LayerWeightsPtrs<T>& weights,
                       ForwardLayer<float>& activations, size_t num_tokens,
                       float* HWY_RESTRICT output,
                       const MatStorageT<float>& inv_timescale,
                       hwy::ThreadPool& pool) {
  const LayerConfig& config = weights.layer_config;
  const size_t model_dim = config.model_dim;
  const size_t kSeqLen = activations.input.Rows();
  const size_t kQKVDim = config.qkv_dim;
  const size_t kHeads = config.heads;
  static const float query_scale =
      static_cast<float>(1.0 / sqrt(static_cast<double>(kQKVDim)));
  HWY_ASSERT(num_tokens <= kSeqLen);

  ApplyRMSNorm(weights.pre_attention_norm_scale.Packed(),
               activations.input.Packed(), model_dim, num_tokens,
               activations.pre_att_rms_out.Packed(), pool);

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    MatVec(weights.qkv_einsum_w, 0, (kHeads + 2) * kQKVDim, model_dim,
           activations.pre_att_rms_out.Packed() + pos * model_dim,
           activations.qkv.Packed() + pos * (kHeads + 2) * kQKVDim, pool);
  }
  const size_t num_tasks = kHeads * num_tokens;

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    float* HWY_RESTRICT k =
        activations.qkv.Packed() + (pos * (kHeads + 2) + kHeads) * kQKVDim;
    Rope(k, kQKVDim, inv_timescale.PackedScale1(), pos);
  }
  pool.Run(0, num_tasks, [&](const uint64_t task, size_t thread) HWY_ATTR {
    const size_t head = task % kHeads;
    const size_t pos = task / kHeads;
    float* HWY_RESTRICT q =
        activations.qkv.Packed() + (pos * (kHeads + 2) + head) * kQKVDim;
    Rope(q, kQKVDim, inv_timescale.PackedScale1(), pos);
    MulByConst(query_scale, q, kQKVDim);
  });

  pool.Run(0, num_tasks, [&](const uint64_t task, size_t thread) HWY_ATTR {
    const size_t head = task % kHeads;
    const size_t pos = task / kHeads;
    const float* HWY_RESTRICT q =
        activations.qkv.Packed() + (pos * (kHeads + 2) + head) * kQKVDim;
    float* HWY_RESTRICT head_att =
        activations.att.Packed() + (pos * kHeads + head) * kSeqLen;
    for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
      const float* HWY_RESTRICT k2 =
          activations.qkv.Packed() + (pos2 * (kHeads + 2) + kHeads) * kQKVDim;
      const float score = Dot(q, k2, kQKVDim);
      head_att[pos2] = score;
    }
  });

  pool.Run(0, num_tasks, [&](const uint64_t task, size_t thread) HWY_ATTR {
    const size_t head = task % kHeads;
    const size_t pos = task / kHeads;
    float* HWY_RESTRICT head_att =
        activations.att.Packed() + (pos * kHeads + head) * kSeqLen;
    Softmax(head_att, pos + 1);
  });

  pool.Run(0, num_tasks, [&](const uint64_t task, size_t thread) HWY_ATTR {
    const size_t head = task % kHeads;
    const size_t pos = task / kHeads;
    const float* HWY_RESTRICT head_att =
        activations.att.Packed() + (pos * kHeads + head) * kSeqLen;
    float* HWY_RESTRICT att_out =
        activations.att_out.Packed() + (pos * kHeads + head) * kQKVDim;
    hwy::ZeroBytes(att_out, kQKVDim * sizeof(*att_out));
    for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
      float* HWY_RESTRICT v2 = activations.qkv.Packed() +
                               (pos2 * (kHeads + 2) + kHeads + 1) * kQKVDim;
      MulByConstAndAdd(head_att[pos2], v2, att_out, kQKVDim);
    }
  });

  ZeroInit(activations.attention_out);
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    for (size_t head = 0; head < kHeads; ++head) {
      MatVec(weights.attn_vec_einsum_w, head * model_dim * kQKVDim, model_dim,
             kQKVDim,
             activations.att_out.Packed() + pos * kHeads * kQKVDim +
                 head * kQKVDim,
             activations.att_post1.Packed() + pos * model_dim, pool);
      AddFrom(activations.att_post1.Packed() + pos * model_dim,
              activations.attention_out.Packed() + pos * model_dim, model_dim);
    }
  }

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    AddFrom(activations.input.Packed() + pos * model_dim,
            activations.attention_out.Packed() + pos * model_dim, model_dim);
  }

  ApplyRMSNorm(weights.pre_ffw_norm_scale.Packed(),
               activations.attention_out.Packed(), model_dim, num_tokens,
               activations.pre_ffw_rms_out.Packed(), pool);
  const size_t kFFHiddenDim = config.ff_hidden_dim;
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    MatVec(weights.gating_einsum_w, 0, kFFHiddenDim * 2, model_dim,
           activations.pre_ffw_rms_out.Packed() + pos * model_dim,
           activations.ffw_hidden.Packed() + pos * kFFHiddenDim * 2, pool);
  }
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    const size_t hidden_offset = pos * kFFHiddenDim * 2;
    const float* HWY_RESTRICT out =
        activations.ffw_hidden.Packed() + hidden_offset;
    const float* HWY_RESTRICT out_mul = out + kFFHiddenDim;
    float* HWY_RESTRICT out_gated =
        activations.ffw_hidden_gated.Packed() + pos * kFFHiddenDim;
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
    MatVec(weights.linear_w, 0, model_dim, kFFHiddenDim,
           activations.ffw_hidden_gated.Packed() + pos * kFFHiddenDim,
           output + pos * model_dim, pool);
  }
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    AddFrom(activations.attention_out.Packed() + pos * model_dim,
            output + pos * model_dim, model_dim);
  }
}

template <typename T>
float CrossEntropyLossForwardPass(const std::vector<int>& prompt,
                                  size_t context_size,
                                  const ModelWeightsPtrs<T>& weights,
                                  ForwardPass<float>& forward,
                                  const MatStorageT<float>& inv_timescale,
                                  hwy::ThreadPool& pool) {
  const ModelConfig& config = weights.weights_config;
  const size_t vocab_size = config.vocab_size;
  const size_t model_dim = config.model_dim;
  const size_t layers = config.layer_configs.size();
  const float emb_scaling = EmbeddingScaling(model_dim);
  HWY_ASSERT(!config.absolute_pe);
  HWY_ASSERT(config.layer_configs[0].post_norm == PostNormType::None);
  HWY_ASSERT(config.layer_configs[0].kv_heads == 1);

  HWY_DASSERT(context_size > 0);
  HWY_DASSERT(context_size < prompt.size());
  const size_t num_tokens = prompt.size() - 1;

  InputEmbedding(weights.embedder_input_embedding, prompt, emb_scaling,
                 forward.layers[0].input.Packed(), model_dim, vocab_size);

  for (size_t layer = 0; layer < config.layer_configs.size(); ++layer) {
    auto type = config.layer_configs[layer].type;
    // TODO(szabadka) Implement Griffin layer.
    HWY_ASSERT(type == LayerAttentionType::kGemma);
    float* HWY_RESTRICT output = layer + 1 < layers
                                     ? forward.layers[layer + 1].input.Packed()
                                     : forward.final_layer_output.Packed();
    ApplyForwardLayer(*weights.GetLayer(layer), forward.layers[layer],
                      num_tokens, output, inv_timescale, pool);
  }

  ApplyRMSNorm(weights.final_norm_scale.Packed(),
               forward.final_layer_output.Packed(), model_dim, num_tokens,
               forward.final_norm_output.Packed(), pool);

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    MatVec(weights.embedder_input_embedding, 0, vocab_size, model_dim,
           forward.final_norm_output.Packed() + pos * model_dim,
           forward.logits.Packed() + pos * vocab_size, pool);
  }

  if (config.final_cap > 0.0f) {
    for (size_t pos = 0; pos < num_tokens; ++pos) {
      LogitsSoftCap(config.final_cap,
                    forward.logits.Packed() + pos * vocab_size, vocab_size);
    }
  }

  CopyMat(forward.logits, forward.probs);

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    Softmax(forward.probs.Packed() + pos * vocab_size, vocab_size);
  }

  return CrossEntropyLoss(forward.probs.Packed(), prompt, context_size,
                          vocab_size, pool);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
