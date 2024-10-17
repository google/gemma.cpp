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

// Implementation of the Vector-Jacobian Products (VJP) of the individual
// operations of the forward pass.

// Include guard for non-SIMD code.
#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_BACKWARD_INL_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_BACKWARD_INL_H_

#include <stddef.h>

#include <cmath>
#include <vector>

#include "backprop/activations.h"
#include "backprop/prompt.h"
#include "gemma/common.h"
#include "gemma/weights.h"
#include "util/allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_BACKWARD_INL_H_

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_BACKWARD_TOGGLE) == defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_BACKWARD_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_BACKWARD_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_BACKWARD_TOGGLE
#endif

#include "hwy/highway.h"
// After highway.h
#include "ops/matmul-inl.h"
#include "ops/ops-inl.h"
#include "hwy/contrib/dot/dot-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

HWY_INLINE void MatMulVJP(const float* HWY_RESTRICT weights,  // kRows * kCols,
                          const float* HWY_RESTRICT x,  // num_tokens * kCols
                          const float* HWY_RESTRICT v,  // num_tokens * kRows
                          size_t cols, size_t rows, size_t num_tokens,
                          float* HWY_RESTRICT grad_w,  // kRows * kCols,
                          float* HWY_RESTRICT grad_x,  // num_tokens * kCols
                          hwy::ThreadPool& pool) {
  hwy::ZeroBytes(grad_x, num_tokens * cols * sizeof(grad_x[0]));
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    const size_t voffs = pos * rows;
    const size_t xoffs = pos * cols;
    for (size_t j = 0; j < rows; ++j) {
      MulByConstAndAdd(v[voffs + j], &x[xoffs], &grad_w[j * cols], cols);
      MulByConstAndAdd(v[voffs + j], &weights[j * cols], &grad_x[xoffs], cols);
    }
  }
}

HWY_INLINE void MultiHeadMatMulVJP(
    const float* HWY_RESTRICT weights,  // heads * kRows * kCols
    const float* HWY_RESTRICT x,        // num_tokens * heads * kCols
    const float* HWY_RESTRICT v,        // num_tokens * kRows
    size_t heads, size_t cols, size_t rows, size_t num_tokens,
    float* HWY_RESTRICT grad_w,  // heads * kRows * kCols
    float* HWY_RESTRICT grad_x,  // num_tokens * heads * kCols
    hwy::ThreadPool& pool) {
  hwy::ZeroBytes(grad_x, num_tokens * heads * cols * sizeof(grad_x[0]));
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    for (size_t j = 0; j < rows; ++j) {
      for (size_t h = 0; h < heads; ++h) {
        MulByConstAndAdd(v[pos * rows + j], &x[pos * heads * cols + h * cols],
                         &grad_w[h * rows * cols + j * cols], cols);
        MulByConstAndAdd(v[pos * rows + j],
                         &weights[h * rows * cols + j * cols],
                         &grad_x[pos * heads * cols + h * cols], cols);
      }
    }
  }
}

template <class D, HWY_IF_F32_D(D)>
static HWY_INLINE hn::Vec<D> DGelu(D d, hn::Vec<D> v) {
  const hn::Vec<D> kMul = hn::Set(d, 0.044715f);
  const hn::Vec<D> kSqrt2OverPi = hn::Set(d, 0.797884560804236f);
  const hn::Vec<D> kHalf = hn::Set(d, 0.5f);
  const hn::Vec<D> kOne = hn::Set(d, 1.0f);
  // kSqrtOverPi*3*kMul
  const hn::Vec<D> kMulv2 = hn::Set(d, 0.1070322244f);

  const hn::Vec<D> v2 = hn::Mul(v, v);
  const hn::Vec<D> v3 = hn::Mul(v2, v);
  const hn::Vec<D> arg = hn::Mul(kSqrt2OverPi, hn::MulAdd(kMul, v3, v));
  const hn::Vec<D> tanh = hn::Tanh(d, arg);
  const hn::Vec<D> cdf = hn::MulAdd(kHalf, tanh, kHalf);
  const hn::Vec<D> dtanh = hn::Sub(kOne, hn::Mul(tanh, tanh));
  const hn::Vec<D> darg = hn::MulAdd(kMulv2, v2, kSqrt2OverPi);
  return hn::MulAdd(kHalf, hn::Mul(v, hn::Mul(dtanh, darg)), cdf);
}

static HWY_NOINLINE void SoftmaxVJP(const float* HWY_RESTRICT forward,
                                    float* HWY_RESTRICT backward,
                                    const size_t size) {
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  const D d;

  const auto offset =
      hn::Set(d, hn::Dot::Compute<0>(d, forward, backward, size));
  hn::Transform1(
      d, backward, size, forward,
      [&offset](const auto d, const auto v, const auto y)
      HWY_ATTR { return hn::Mul(y, hn::Sub(v, offset)); });
}

static HWY_NOINLINE void RMSNormVJP(
    const float* HWY_RESTRICT weights, const float* HWY_RESTRICT x,
    const float* HWY_RESTRICT v, size_t model_dim, size_t num_tokens,
    float* HWY_RESTRICT grad_w, float* HWY_RESTRICT grad_x,
    hwy::ThreadPool& pool) {
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    const size_t offset = pos * model_dim;
    const float ss = detail::RMSNormMul(x + offset, model_dim);

    for (size_t i = 0; i < model_dim; ++i) {
      grad_w[i] += v[offset + i] * x[offset + i] * ss;
    }
    const float ss3 = ss * ss * ss / StaticCast<float>(model_dim);
    float tmp = 0.0f;
    for (size_t i = 0; i < model_dim; ++i) {
      tmp += (1.0f + weights[i]) * v[offset + i] * x[offset + i];
    }
    tmp *= ss3;
    for (size_t i = 0; i < model_dim; ++i) {
      grad_x[offset + i] = ss * (1.0f + weights[i]) * v[offset + i] -
                           tmp * x[offset + i];
    }
  }
}

static HWY_NOINLINE void InputEmbeddingVJP(
    const float* weights, const std::vector<int>& prompt,
    const float scaling, const float* HWY_RESTRICT v,
    float* HWY_RESTRICT grad, size_t model_dim) {
  HWY_ASSERT(!prompt.empty());
  for (size_t pos = 0; pos < prompt.size() - 1; ++pos) {
    int token = prompt[pos];
    MulByConstAndAdd(scaling, v + pos * model_dim,
                     grad + token * model_dim, model_dim);
  }
}

template <typename T>
void LayerVJP(const LayerWeightsPtrs<T>& weights,
              const ForwardLayer<float>& forward,
              const float* HWY_RESTRICT next_layer_grad, size_t num_tokens,
              LayerWeightsPtrs<T>& grad, ForwardLayer<float>& backward,
              const RowVectorBatch<float>& inv_timescale,
              hwy::ThreadPool& pool) {
  const LayerConfig& config = weights.layer_config;
  const size_t model_dim = config.model_dim;
  const size_t qkv_dim = config.qkv_dim;
  const size_t heads = config.heads;
  const size_t seq_len = forward.input.Rows();
  const size_t ff_hidden_dim = config.ff_hidden_dim;
  const float query_scale =
      static_cast<float>(1.0 / sqrt(static_cast<double>(qkv_dim)));
  HWY_ASSERT(num_tokens <= seq_len);

  MatMulVJP(weights.linear_w.data(), forward.ffw_hidden_gated.data(),
            next_layer_grad, ff_hidden_dim, model_dim, num_tokens,
            grad.linear_w.data(), backward.ffw_hidden_gated.data(), pool);

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    const size_t hidden_offset = pos * ff_hidden_dim * 2;
    const float* HWY_RESTRICT f_out = forward.ffw_hidden.data() + hidden_offset;
    const float* HWY_RESTRICT f_out_mul = f_out + ff_hidden_dim;
    const float* HWY_RESTRICT b_out_gated =
        backward.ffw_hidden_gated.data() + pos * ff_hidden_dim;
    float* HWY_RESTRICT b_out = backward.ffw_hidden.data() + hidden_offset;
    float* HWY_RESTRICT b_out_mul = b_out + ff_hidden_dim;
    namespace hn = hwy::HWY_NAMESPACE;
    using DF = hn::ScalableTag<float>;
    DF df;
    for (size_t i = 0; i < ff_hidden_dim; i += Lanes(df)) {
      const auto y = Load(df, f_out + i);
      const auto x = Load(df, f_out_mul + i);
      const auto v = Load(df, b_out_gated + i);
      hn::Store(hn::Mul(v, Gelu(df, y)), df, b_out_mul + i);
      hn::Store(hn::Mul(v, hn::Mul(x, DGelu(df, y))), df, b_out + i);
    }
  }

  MatMulVJP(weights.gating_einsum_w.data(), forward.bf_pre_ffw_rms_out.data(),
            backward.ffw_hidden.data(), model_dim, ff_hidden_dim * 2,
            num_tokens, grad.gating_einsum_w.data(),
            backward.bf_pre_ffw_rms_out.data(), pool);
  RMSNormVJP(weights.pre_ffw_norm_scale.data(), forward.attention_out.data(),
             backward.bf_pre_ffw_rms_out.data(), model_dim, num_tokens,
             grad.pre_ffw_norm_scale.data(), backward.attention_out.data(),
             pool);

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    AddFrom(next_layer_grad + pos * model_dim,
            backward.attention_out.data() + pos * model_dim, model_dim);
  }

  backward.qkv.ZeroInit();

  MultiHeadMatMulVJP(weights.attn_vec_einsum_w.data(), forward.att_out.data(),
                     backward.attention_out.data(), heads, qkv_dim, model_dim,
                     num_tokens, grad.attn_vec_einsum_w.data(),
                     backward.att_out.data(), pool);

  for (size_t head = 0; head < heads; ++head) {
    for (size_t pos = 0; pos < num_tokens; ++pos) {
      const size_t aoffset = head * seq_len + pos * heads * seq_len;
      const float* HWY_RESTRICT f_head_att = forward.att.data() + aoffset;
      const float* HWY_RESTRICT b_att_out =
          backward.att_out.data() + (pos * heads + head) * qkv_dim;
      float* HWY_RESTRICT b_head_att = backward.att.data() + aoffset;
      for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
        const size_t v2offs = (pos2 * (heads + 2) + heads + 1) * qkv_dim;
        const float* HWY_RESTRICT f_v2 = forward.qkv.data() + v2offs;
        float* HWY_RESTRICT b_v2 = backward.qkv.data() + v2offs;
        b_head_att[pos2] = Dot(b_att_out, f_v2, qkv_dim);
        MulByConstAndAdd(f_head_att[pos2], b_att_out, b_v2, qkv_dim);
      }
    }
  }

  for (size_t head = 0; head < heads; ++head) {
    for (size_t pos = 0; pos < num_tokens; ++pos) {
      const size_t aoffset = head * seq_len + pos * heads * seq_len;
      const float* HWY_RESTRICT f_head_att = forward.att.data() + aoffset;
      float* HWY_RESTRICT b_head_att = backward.att.data() + aoffset;
      SoftmaxVJP(f_head_att, b_head_att, pos + 1);
    }
  }

  for (size_t head = 0; head < heads; ++head) {
    for (size_t pos = 0; pos < num_tokens; ++pos) {
      const size_t qoffs = (pos * (heads + 2) + head) * qkv_dim;
      const size_t aoffs = head * seq_len + pos * heads * seq_len;
      const float* HWY_RESTRICT f_q = forward.qkv.data() + qoffs;
      const float* HWY_RESTRICT b_head_att = backward.att.data() + aoffs;
      float* HWY_RESTRICT b_q = backward.qkv.data() + qoffs;
      for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
        const size_t k2offs = (pos2 * (heads + 2) + heads) * qkv_dim;
        const float* HWY_RESTRICT f_k2 = forward.qkv.data() + k2offs;
        float* HWY_RESTRICT b_k2 = backward.qkv.data() + k2offs;
        MulByConstAndAdd(b_head_att[pos2], f_k2, b_q, qkv_dim);
        MulByConstAndAdd(b_head_att[pos2], f_q, b_k2, qkv_dim);
      }
    }
  }

  for (int pos = 0; pos < static_cast<int>(num_tokens); ++pos) {
    float* HWY_RESTRICT b_kv =
        backward.qkv.data() + (pos * (heads + 2) + heads) * qkv_dim;
    Rope(b_kv, qkv_dim, inv_timescale.Const(), -pos);
  }

  for (size_t head = 0; head < heads; ++head) {
    for (size_t pos = 0; pos < num_tokens; ++pos) {
      float* HWY_RESTRICT b_q =
          backward.qkv.data() + (pos * (heads + 2) + head) * qkv_dim;
      MulByConst(query_scale, b_q, qkv_dim);
      Rope(b_q, qkv_dim, inv_timescale.Const(), -pos);
    }
  }

  MatMulVJP(weights.qkv_einsum_w.data(), forward.pre_att_rms_out.data(),
            backward.qkv.data(), model_dim, (heads + 2) * qkv_dim, num_tokens,
            grad.qkv_einsum_w.data(), backward.pre_att_rms_out.data(), pool);
  RMSNormVJP(weights.pre_attention_norm_scale.data(), forward.input.data(),
             backward.pre_att_rms_out.data(), model_dim, num_tokens,
             grad.pre_attention_norm_scale.data(), backward.input.data(), pool);
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    AddFrom(backward.attention_out.data() + pos * model_dim,
            backward.input.data() + pos * model_dim, model_dim);
  }
}

static HWY_NOINLINE void SoftcapVJP(const float cap,
                                    const float* HWY_RESTRICT forward,
                                    float* HWY_RESTRICT backward,
                                    const size_t size) {
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  const D d;

  const auto one = hn::Set(d, 1.0f);
  const auto vcap = hn::Set(d, cap);
  const auto vinv_cap = hn::Div(hn::Set(d, 1.0f), vcap);
  hn::Transform1(d, backward, size, forward,
                 [&](const auto d, const auto v, const auto y) HWY_ATTR {
                   const auto scaled = hn::Mul(vinv_cap, y);  // = tanh
                   return hn::Mul(v, hn::Sub(one, hn::Mul(scaled, scaled)));
                 });
}

static HWY_NOINLINE void CrossEntropyLossGrad(
    const float* HWY_RESTRICT x, float* HWY_RESTRICT grad,
    const Prompt& prompt, size_t vocab_size) {
  HWY_ASSERT(!prompt.tokens.empty());
  const float scaling = -1.0 / std::log(2.0);
  size_t num_tokens = prompt.tokens.size() - 1;
  hwy::ZeroBytes(grad, num_tokens * vocab_size * sizeof(grad[0]));
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    if (pos + 1 < prompt.context_size) {
      continue;
    }
    const int next_token = prompt.tokens[pos + 1];
    grad[pos * vocab_size + next_token] =
        scaling / x[pos * vocab_size + next_token];
  }
}

template <typename T>
void CrossEntropyLossBackwardPassInl(const Prompt& prompt,
                                     const ModelWeightsPtrs<T>& weights,
                                     const ForwardPass<float>& forward,
                                     ModelWeightsPtrs<T>& grad,
                                     ForwardPass<float>& backward,
                                     RowVectorBatch<float>& inv_timescale,
                                     hwy::ThreadPool& pool) {
  const ModelConfig& config = weights.weights_config;
  const size_t kVocabSize = config.vocab_size;
  const size_t model_dim = config.model_dim;
  const size_t kLayers = config.layer_configs.size();
  const float kEmbScaling = EmbeddingScaling(model_dim);
  HWY_ASSERT(!config.absolute_pe);
  HWY_ASSERT(config.layer_configs[0].post_norm == PostNormType::None);
  HWY_ASSERT(config.layer_configs[0].kv_heads == 1);

  HWY_DASSERT(prompt.context_size > 0);
  HWY_DASSERT(prompt.context_size < prompt.tokens.size());
  const size_t num_tokens = prompt.tokens.size() - 1;

  CrossEntropyLossGrad(forward.probs.data(), backward.logits.data(), prompt,
                       kVocabSize);

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    SoftmaxVJP(forward.probs.data() + pos * kVocabSize,
               backward.logits.data() + pos * kVocabSize,
               kVocabSize);
  }

  if (config.final_cap > 0.0f) {
    for (size_t pos = 0; pos < num_tokens; ++pos) {
      SoftcapVJP(config.final_cap, forward.logits.data() + pos * kVocabSize,
                 backward.logits.data() + pos * kVocabSize, kVocabSize);
    }
  }

  MatMulVJP(weights.embedder_input_embedding.data(),
            forward.final_norm_output.data(), backward.logits.data(), model_dim,
            kVocabSize, num_tokens, grad.embedder_input_embedding.data(),
            backward.final_norm_output.data(), pool);

  RMSNormVJP(weights.final_norm_scale.data(), forward.final_layer_output.data(),
             backward.final_norm_output.data(), model_dim, num_tokens,
             grad.final_norm_scale.data(), backward.final_layer_output.data(),
             pool);

  for (int layer = static_cast<int>(kLayers) - 1; layer >= 0; --layer) {
    auto layer_config = config.layer_configs[layer];
    // TODO(szabadka) Implement Griffin layer vjp.
    HWY_ASSERT(layer_config.type == LayerAttentionType::kGemma);
    float* next_layer_grad = layer + 1 < kLayers
                             ? backward.layers[layer + 1].input.data()
                             : backward.final_layer_output.data();
    LayerVJP(*weights.GetLayer(layer), forward.layers[layer], next_layer_grad,
             num_tokens, *grad.GetLayer(layer), backward.layers[layer],
             inv_timescale, pool);
  }

  InputEmbeddingVJP(weights.embedder_input_embedding.data(), prompt.tokens,
                    kEmbScaling, backward.layers[0].input.data(),
                    grad.embedder_input_embedding.data(), model_dim);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
