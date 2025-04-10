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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_BACKWARD_SCALAR_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_BACKWARD_SCALAR_H_

#include <stddef.h>
#include <string.h>

#include <cmath>
#include <vector>

#include "backprop/activations.h"
#include "backprop/common_scalar.h"
#include "backprop/prompt.h"
#include "gemma/common.h"  // EmbeddingScaling
#include "gemma/weights.h"

namespace gcpp {
template<typename T>
void MatMulVJPT(const T* w, const T* x, const T* dy, T* dw, T* dx,
                size_t N, size_t M, size_t K) {
  memset(dx, 0, M * K * sizeof(dx[0]));
  for (size_t i = 0; i < K; ++i) {
    for (size_t j = 0; j < N; ++j) {
      MulByConstAndAddT(dy[i * N + j], &x[i * M], &dw[j * M], M);
      MulByConstAndAddT(dy[i * N + j], &w[j * M], &dx[i * M], M);
    }
  }
}
template<typename T>
void MultiHeadMatMulVJPT(const T* w, const T* x, const T* dy, T* dw, T* dx,
                         size_t H, size_t N, size_t M, size_t K) {
  memset(dx, 0, H * M * K * sizeof(dx[0]));
  for (size_t i = 0; i < K; ++i) {
    for (size_t j = 0; j < N; ++j) {
      for (size_t h = 0; h < H; ++h) {
        MulByConstAndAddT(dy[i * N + j], &x[i * H * M + h * M],
                          &dw[h * N * M + j * M], M);
        MulByConstAndAddT(dy[i * N + j], &w[h * N * M + j * M],
                          &dx[i * H * M + h * M], M);
      }
    }
  }
}

template<typename T>
void RMSNormVJPT(const T* w, const T* x, const T* dy, T* dw, T* dx,
                 size_t N, size_t K) {
  for (size_t i = 0; i < K; ++i) {
    constexpr T eps(1e-6);
    T ss = SquaredL2(x + i * N, N);
    ss = T(1.0) / std::sqrt(ss / T(N) + eps);
    for (size_t j = 0; j < N; ++j) {
      dw[j] += dy[i * N + j] * x[i * N + j] * ss;
    }
    const T ss3 = ss * ss * ss / T(N);
    T tmp = 0.0;
    for (size_t j = 0; j < N; ++j) {
      tmp += (T(1.0) + w[j]) * dy[i* N + j] * x[i * N + j];
    }
    tmp *= ss3;
    for (size_t j = 0; j < N; ++j) {
      dx[i * N + j] = ss * (T(1.0) + w[j]) * dy[i* N + j] - tmp * x[i * N + j];
    }
  }
}
template<typename T>
void SoftmaxVJPT(const T* y, T* dy, size_t N) {
  T sum = {};
  for (size_t i = 0; i < N; ++i) {
    sum += y[i] * dy[i];
  }
  for (size_t i = 0; i < N; ++i) {
    dy[i] = y[i] * (dy[i] - sum);
  }
}
template<typename T>
void SoftmaxVJPT(const T* y, T* dy, size_t N, size_t K) {
  for (size_t i = 0; i < K; ++i) {
    SoftmaxVJPT(y + i * N, dy + i * N, N);
  }
}

template<typename T>
T GeluDerivative(T x) {
  static const T kMul = 0.044715;
  static const T kSqrt2OverPi = 0.797884560804236;
  static const T kMul2 = kSqrt2OverPi * T(3.0) * kMul;

  const T x2 = x * x;
  const T x3 = x2 * x;
  const T arg = kSqrt2OverPi * (kMul * x3 + x);
  const T tanh = std::tanh(arg);
  const T cdf = T(0.5) * (T(1.0) + tanh);
  const T dtanh = T(1.0) - tanh * tanh;
  const T darg = kMul2 * x2 + kSqrt2OverPi;
  return T(0.5) * x * dtanh * darg + cdf;
}

template<typename T>
void GatedGeluVJP(const T* in, const T* d_out, T* d_in, size_t N, size_t K) {
  for (size_t i = 0; i < K; ++i) {
    const T* x1 = in + i * 2 * N;
    const T* x2 = x1 + N;
    const T* v = d_out + i * N;
    T* dx1 = d_in + i * 2 * N;
    T* dx2 = dx1 + N;
    for (size_t j = 0; j < N; ++j) {
      dx1[j] = v[j] * x2[j] * GeluDerivative(x1[j]);
      dx2[j] = v[j] * Gelu(x1[j]);
    }
  }
}

template <typename T>
void MaskedAttentionVJP(const T* qkv, const T* doutput, T* dqkv,
                        size_t num_tokens, size_t kHeads, size_t qkv_dim,
                        size_t seq_len) {
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    const size_t offset = pos * (kHeads + 2) * qkv_dim;
    memset(dqkv + offset, 0, (kHeads + 1) * qkv_dim * sizeof(qkv[0]));
  }
  for (size_t head = 0; head < kHeads; ++head) {
    for (size_t pos = 0; pos < num_tokens; ++pos) {
      const size_t qoffs = (pos * (kHeads + 2) + head) * qkv_dim;
      const size_t aoffs = head * seq_len + pos * kHeads * seq_len;
      const T* q = qkv + qoffs;
      const T* dout = doutput + aoffs;
      T* dq = dqkv + qoffs;
      for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
        const size_t koffs = (pos2 * (kHeads + 2) + kHeads) * qkv_dim;
        const T* k = qkv + koffs;
        T* dk = dqkv + koffs;
        MulByConstAndAddT(dout[pos2], k, dq, qkv_dim);
        MulByConstAndAddT(dout[pos2], q, dk, qkv_dim);
      }
    }
  }
}

template <typename T>
void MaskedSoftmaxVJPT(const T* y, T* dy, size_t num_tokens, size_t kHeads,
                       size_t seq_len) {
  for (size_t head = 0; head < kHeads; ++head) {
    for (size_t pos = 0; pos < num_tokens; ++pos) {
      size_t offset = pos * kHeads * seq_len + head * seq_len;
      SoftmaxVJPT(y + offset, dy + offset, pos + 1);
      memset(dy + offset + pos + 1, 0, (seq_len - pos - 1) * sizeof(T));
    }
  }
}

template <typename T>
void MixByAttentionVJP(const T* qkv, const T* attention, const T* doutput,
                       T* dqkv, T* dattention, size_t num_tokens, size_t kHeads,
                       size_t qkv_dim, size_t seq_len) {
  auto v_offset = [&](size_t pos) {
    return (pos * (kHeads + 2) + kHeads + 1) * qkv_dim;
  };
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    memset(&dqkv[v_offset(pos)], 0, qkv_dim * sizeof(qkv[0]));
  }
  for (size_t head = 0; head < kHeads; ++head) {
    for (size_t pos = 0; pos < num_tokens; ++pos) {
      const size_t offset = head * qkv_dim + pos * kHeads * qkv_dim;
      const size_t aoffset = head * seq_len + pos * kHeads * seq_len;
      const T* att = &attention[aoffset];
      const T* dout = &doutput[offset];
      T* datt = &dattention[aoffset];
      for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
        datt[pos2] = DotT(dout, &qkv[v_offset(pos2)], qkv_dim);
        MulByConstAndAddT(att[pos2], dout, &dqkv[v_offset(pos2)], qkv_dim);
      }
    }
  }
}

template<typename T>
void InputEmbeddingVJPT(const T* w, const std::vector<int>& tokens, T scaling,
                        const T* dy, T* dw, size_t N) {
  const size_t num_tokens = tokens.empty() ? 0 : tokens.size() - 1;
  for (size_t i = 0; i < num_tokens; ++i) {
    int token = tokens[i];
    MulByConstAndAddT(scaling, dy + i * N, dw + token * N, N);
  }
}

template <typename T>
void LayerVJP(const LayerWeightsPtrs<T>& weights,
              const ForwardLayer<T>& forward, const T* dy,
              LayerWeightsPtrs<T>& grad, ForwardLayer<T>& backward,
              size_t num_tokens) {
  const LayerConfig& layer_config = weights.layer_config;
  const size_t model_dim = layer_config.model_dim;
  const size_t seq_len = forward.input.Rows();
  const size_t qkv_dim = layer_config.qkv_dim;
  const size_t kHeads = layer_config.heads;
  const size_t kFFHiddenDim = layer_config.ff_hidden_dim;
  const T kQueryScale = 1.0 / std::sqrt(T(qkv_dim));

  MatMulVJPT(weights.linear_w.Packed(), forward.ffw_hidden_gated.Packed(), dy,
             grad.linear_w.Packed(), backward.ffw_hidden_gated.Packed(),
             model_dim, kFFHiddenDim, num_tokens);

  GatedGeluVJP(forward.ffw_hidden.Packed(), backward.ffw_hidden_gated.Packed(),
               backward.ffw_hidden.Packed(), kFFHiddenDim, num_tokens);

  MatMulVJPT(weights.gating_einsum_w.Packed(),
             forward.bf_pre_ffw_rms_out.Packed(), backward.ffw_hidden.Packed(),
             grad.gating_einsum_w.Packed(),
             backward.bf_pre_ffw_rms_out.Packed(), kFFHiddenDim * 2, model_dim,
             num_tokens);

  RMSNormVJPT(
      weights.pre_ffw_norm_scale.Packed(), forward.attention_out.Packed(),
      backward.bf_pre_ffw_rms_out.Packed(), grad.pre_ffw_norm_scale.Packed(),
      backward.attention_out.Packed(), model_dim, num_tokens);

  AddFromT(dy, backward.attention_out.Packed(), num_tokens * model_dim);

  MultiHeadMatMulVJPT(
      weights.attn_vec_einsum_w.Packed(), forward.att_out.Packed(),
      backward.attention_out.Packed(), grad.attn_vec_einsum_w.Packed(),
      backward.att_out.Packed(), kHeads, model_dim, qkv_dim, num_tokens);

  MixByAttentionVJP(forward.qkv.Packed(), forward.att.Packed(),
                    backward.att_out.Packed(), backward.qkv.Packed(),
                    backward.att.Packed(), num_tokens, kHeads, qkv_dim,
                    seq_len);

  MaskedSoftmaxVJPT(forward.att.Packed(), backward.att.Packed(), num_tokens,
                    kHeads, seq_len);

  MaskedAttentionVJP(forward.qkv.Packed(), backward.att.Packed(),
                     backward.qkv.Packed(), num_tokens, kHeads, qkv_dim,
                     seq_len);

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    T* qkv = backward.qkv.Packed() + pos * (kHeads + 2) * qkv_dim;
    MulByConstT(kQueryScale, qkv, kHeads * qkv_dim);
  }

  for (int pos = 0; pos < num_tokens; ++pos) {
    T* qkv = backward.qkv.Packed() + pos * (kHeads + 2) * qkv_dim;
    for (size_t h = 0; h <= kHeads; ++h) {
      Rope(qkv + h * qkv_dim, qkv_dim, -pos);
    }
  }

  MatMulVJPT(weights.qkv_einsum_w.Packed(), forward.pre_att_rms_out.Packed(),
             backward.qkv.Packed(), grad.qkv_einsum_w.Packed(),
             backward.pre_att_rms_out.Packed(), (kHeads + 2) * qkv_dim,
             model_dim, num_tokens);
  RMSNormVJPT(weights.pre_attention_norm_scale.Packed(), forward.input.Packed(),
              backward.pre_att_rms_out.Packed(),
              grad.pre_attention_norm_scale.Packed(), backward.input.Packed(),
              model_dim, num_tokens);

  AddFromT(backward.attention_out.Packed(), backward.input.Packed(),
           num_tokens * model_dim);
}

template <typename T>
void SoftcapVJPT(float cap, const T* y, T* dy, size_t N) {
  const T inv_cap = T{1.0} / static_cast<T>(cap);
  for (size_t i = 0; i < N; ++i) {
    T scaled = y[i] * inv_cap;  // tanh
    dy[i] *= (T{1.0} - scaled * scaled);
  }
}

template<typename T>
void CrossEntropyLossGrad(const T* x, T* dx, const Prompt& prompt, size_t V) {
  T scaling = -1.0 / std::log(2.0);
  const std::vector<int> tokens = prompt.tokens;
  const size_t num_tokens = tokens.empty() ? 0 : tokens.size() - 1;
  memset(dx, 0, V * num_tokens * sizeof(x[0]));
  for (size_t i = 0; i < num_tokens; ++i) {
    if (i + 1 < prompt.context_size) {
      continue;
    }
    const int next_token = tokens[i + 1];
    dx[i * V + next_token] = scaling / x[i * V + next_token];
  }
}

template <typename T>
void CrossEntropyLossBackwardPass(const Prompt& prompt,
                                  const ModelWeightsPtrs<T>& weights,
                                  const ForwardPass<T>& forward,
                                  ModelWeightsPtrs<T>& grad,
                                  ForwardPass<T>& backward) {
  const ModelConfig& config = weights.weights_config;
  const size_t model_dim = config.model_dim;
  const size_t vocab_size = config.vocab_size;
  const size_t layers = config.layer_configs.size();
  const std::vector<int> tokens = prompt.tokens;
  const size_t num_tokens = tokens.empty() ? 0 : tokens.size() - 1;

  CrossEntropyLossGrad(forward.probs.Packed(), backward.logits.Packed(), prompt,
                       vocab_size);

  SoftmaxVJPT(forward.probs.Packed(), backward.logits.Packed(), vocab_size,
              num_tokens);

  if (config.final_cap > 0.0f) {
    for (size_t i = 0; i < num_tokens; ++i) {
      SoftcapVJPT(config.final_cap, forward.logits.Packed() + i * vocab_size,
                  backward.logits.Packed() + i * vocab_size, vocab_size);
    }
  }

  MatMulVJPT(weights.embedder_input_embedding.Packed(),
             forward.final_norm_output.Packed(), backward.logits.Packed(),
             grad.embedder_input_embedding.Packed(),
             backward.final_norm_output.Packed(), vocab_size, model_dim,
             num_tokens);

  RMSNormVJPT(
      weights.final_norm_scale.Packed(), forward.final_layer_output.Packed(),
      backward.final_norm_output.Packed(), grad.final_norm_scale.Packed(),
      backward.final_layer_output.Packed(), model_dim, num_tokens);

  for (int layer = static_cast<int>(layers) - 1; layer >= 0; --layer) {
    T* next_layer_grad = layer + 1 < layers
                             ? backward.layers[layer + 1].input.Packed()
                             : backward.final_layer_output.Packed();
    LayerVJP(*weights.GetLayer(layer), forward.layers[layer], next_layer_grad,
             *grad.GetLayer(layer), backward.layers[layer], num_tokens);
  }

  const T kEmbScaling = EmbeddingScaling(model_dim);
  InputEmbeddingVJPT(weights.embedder_input_embedding.Packed(), tokens,
                     kEmbScaling, backward.layers[0].input.Packed(),
                     grad.embedder_input_embedding.Packed(), model_dim);
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_BACKWARD_SCALAR_H_
