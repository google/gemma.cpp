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
#include <complex>
#include <vector>

#include "backprop/common_scalar.h"
#include "backprop/prompt.h"
#include "gemma/activations.h"
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
    const size_t offset = i * N;
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


template<typename T>
void MaskedAttentionVJP(const T* qkv, const T* doutput, T* dqkv,
                        size_t num_tokens, size_t kHeads, size_t kQKVDim,
                        size_t kSeqLen) {
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    const size_t offset = pos * (kHeads + 2) * kQKVDim;
    memset(dqkv + offset, 0, (kHeads + 1) * kQKVDim * sizeof(qkv[0]));
  }
  for (size_t head = 0; head < kHeads; ++head) {
    for (size_t pos = 0; pos < num_tokens; ++pos) {
      const size_t qoffs = (pos * (kHeads + 2) + head) * kQKVDim;
      const size_t aoffs = head * kSeqLen + pos * kHeads * kSeqLen;
      const T* q = qkv + qoffs;
      const T* dout = doutput + aoffs;
      T* dq = dqkv + qoffs;
      for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
        const size_t koffs = (pos2 * (kHeads + 2) + kHeads) * kQKVDim;
        const T* k = qkv + koffs;
        T* dk = dqkv + koffs;
        MulByConstAndAddT(dout[pos2], k, dq, kQKVDim);
        MulByConstAndAddT(dout[pos2], q, dk, kQKVDim);
      }
    }
  }
}

template<typename T>
void MaskedSoftmaxVJPT(const T* y, T* dy, size_t num_tokens,
                       size_t kHeads, size_t kSeqLen) {
  for (size_t head = 0; head < kHeads; ++head) {
    for (size_t pos = 0; pos < num_tokens; ++pos) {
      size_t offset = pos * kHeads * kSeqLen + head * kSeqLen;
      SoftmaxVJPT(y + offset, dy + offset, pos + 1);
      memset(dy + offset + pos + 1, 0, (kSeqLen - pos - 1) * sizeof(T));
    }
  }
}

template<typename T>
void MixByAttentionVJP(const T* qkv, const T* attention, const T* doutput,
                       T* dqkv, T* dattention, size_t num_tokens,
                       size_t kHeads, size_t kQKVDim, size_t kSeqLen) {
  auto v_offset = [&](size_t pos) {
    return (pos * (kHeads + 2) + kHeads + 1) * kQKVDim;
  };
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    memset(&dqkv[v_offset(pos)], 0, kQKVDim * sizeof(qkv[0]));
  }
  for (size_t head = 0; head < kHeads; ++head) {
    for (size_t pos = 0; pos < num_tokens; ++pos) {
      const size_t offset = head * kQKVDim + pos * kHeads * kQKVDim;
      const size_t aoffset = head * kSeqLen + pos * kHeads * kSeqLen;
      const T* att = &attention[aoffset];
      const T* dout = &doutput[offset];
      T* datt = &dattention[aoffset];
      for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
        datt[pos2] = DotT(dout, &qkv[v_offset(pos2)], kQKVDim);
        MulByConstAndAddT(att[pos2], dout, &dqkv[v_offset(pos2)], kQKVDim);
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

template<typename T, typename TConfig>
void LayerVJP(const Layer<T, TConfig>& weights,
              const ForwardLayer<T, TConfig>& forward,
              const T* dy,
              Layer<T, TConfig>& grad,
              ForwardLayer<T, TConfig>& backward,
              size_t num_tokens) {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kSeqLen = TConfig::kSeqLen;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;
  static const T kQueryScale = 1.0 / std::sqrt(T(kQKVDim));

  MatMulVJPT(weights.linear_w.data(), forward.ffw_hidden_gated.data(),
             dy, grad.linear_w.data(), backward.ffw_hidden_gated.data(),
             kModelDim, kFFHiddenDim, num_tokens);

  GatedGeluVJP(forward.ffw_hidden.data(), backward.ffw_hidden_gated.data(),
               backward.ffw_hidden.data(), kFFHiddenDim, num_tokens);

  MatMulVJPT(weights.gating_einsum_w.data(), forward.bf_pre_ffw_rms_out.data(),
             backward.ffw_hidden.data(), grad.gating_einsum_w.data(),
             backward.bf_pre_ffw_rms_out.data(), kFFHiddenDim * 2, kModelDim,
             num_tokens);

  RMSNormVJPT(weights.pre_ffw_norm_scale.data(), forward.attention_out.data(),
              backward.bf_pre_ffw_rms_out.data(),
              grad.pre_ffw_norm_scale.data(), backward.attention_out.data(),
              kModelDim, num_tokens);

  AddFromT(dy, backward.attention_out.data(), num_tokens * kModelDim);

  MultiHeadMatMulVJPT(weights.attn_vec_einsum_w.data(), forward.att_out.data(),
                      backward.attention_out.data(),
                      grad.attn_vec_einsum_w.data(),
                      backward.att_out.data(),
                      kHeads, kModelDim, kQKVDim, num_tokens);

  MixByAttentionVJP(forward.qkv.data(), forward.att.data(),
                    backward.att_out.data(), backward.qkv.data(),
                    backward.att.data(), num_tokens, kHeads, kQKVDim,
                    kSeqLen);

  MaskedSoftmaxVJPT(forward.att.data(), backward.att.data(),
                    num_tokens, kHeads, kSeqLen);

  MaskedAttentionVJP(forward.qkv.data(), backward.att.data(),
                     backward.qkv.data(), num_tokens, kHeads, kQKVDim, kSeqLen);

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    T* qkv = backward.qkv.data() + pos * (kHeads + 2) * kQKVDim;
    MulByConstT(kQueryScale, qkv, kHeads * kQKVDim);
  }

  for (int pos = 0; pos < num_tokens; ++pos) {
    T* qkv = backward.qkv.data() + pos * (kHeads + 2) * kQKVDim;
    for (size_t h = 0; h <= kHeads; ++h) {
      Rope(qkv + h * kQKVDim, kQKVDim, -pos);
    }
  }

  MatMulVJPT(weights.qkv_einsum_w.data(), forward.pre_att_rms_out.data(),
             backward.qkv.data(), grad.qkv_einsum_w.data(),
            backward.pre_att_rms_out.data(),
            (kHeads + 2) * kQKVDim, kModelDim, num_tokens);
  RMSNormVJPT(weights.pre_attention_norm_scale.data(), forward.input.data(),
              backward.pre_att_rms_out.data(),
              grad.pre_attention_norm_scale.data(),
              backward.input.data(), kModelDim, num_tokens);

  AddFromT(backward.attention_out.data(), backward.input.data(),
           num_tokens * kModelDim);
}

template<typename T>
void SoftcapVJPT(const T* y, T* dy, size_t N) {
  T cap = 30.0;
  T inv_cap = T(1.0) / cap;
  for (size_t i = 0; i < N; ++i) {
    T scaled = y[i] * inv_cap;
    dy[i] *= (T(1.0) - scaled * scaled);
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

template<typename T, typename TConfig>
void CrossEntropyLossBackwardPass(const Prompt& prompt,
                                  const Weights<T, TConfig>& weights,
                                  const ForwardPass<T, TConfig>& forward,
                                  Weights<T, TConfig>& grad,
                                  ForwardPass<T, TConfig>& backward) {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kVocabSize = TConfig::kVocabSize;
  static constexpr size_t kLayers = TConfig::kLayers;
  const std::vector<int> tokens = prompt.tokens;
  const size_t num_tokens = tokens.empty() ? 0 : tokens.size() - 1;

  CrossEntropyLossGrad(forward.probs.data(), backward.logits.data(), prompt,
                       kVocabSize);

  SoftmaxVJPT(forward.probs.data(), backward.logits.data(),
              kVocabSize, num_tokens);

  SoftcapVJPT(forward.logits.data(), backward.logits.data(),
              num_tokens * kVocabSize);

  MatMulVJPT(weights.embedder_input_embedding.data(),
             forward.final_norm_output.data(),
             backward.logits.data(),
             grad.embedder_input_embedding.data(),
             backward.final_norm_output.data(),
             kVocabSize, kModelDim, num_tokens);

  RMSNormVJPT(weights.final_norm_scale.data(),
              forward.final_layer_output.data(),
              backward.final_norm_output.data(),
              grad.final_norm_scale.data(),
              backward.final_layer_output.data(), kModelDim, num_tokens);

  for (int layer = static_cast<int>(kLayers) - 1; layer >= 0; --layer) {
    T* next_layer_grad = layer + 1 < kLayers
                         ? backward.layers[layer + 1].input.data()
                         : backward.final_layer_output.data();
    LayerVJP(*weights.GetLayer(layer), forward.layers[layer], next_layer_grad,
             *grad.GetLayer(layer), backward.layers[layer], num_tokens);
  }

  const T kEmbScaling = EmbeddingScaling(kModelDim);
  InputEmbeddingVJPT(weights.embedder_input_embedding.data(),
                     tokens, kEmbScaling, backward.layers[0].input.data(),
                     grad.embedder_input_embedding.data(), kModelDim);
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_BACKWARD_SCALAR_H_
