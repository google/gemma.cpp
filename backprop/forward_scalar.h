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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_FORWARD_SCALAR_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_FORWARD_SCALAR_H_

#include <stddef.h>
#include <string.h>

#include <cmath>
#include <complex>
#include <vector>

#include "backprop/activations.h"
#include "backprop/common_scalar.h"
#include "backprop/prompt.h"
#include "gemma/common.h"  // EmbeddingScaling
#include "gemma/weights.h"
#include "hwy/base.h"

namespace gcpp {

// w is N x M matrix in row-major order, x is M x K matrix in column-major order
// y = w * x is N x K matrix in column-major order.
template<typename T>
void MatMulT(const T* w, const T* x, T* y, size_t N, size_t M, size_t K) {
  for (size_t i = 0; i < K; ++i) {
    for (size_t j = 0; j < N; ++j) {
      y[i * N + j] = DotT(&w[j * M], &x[i * M], M);
    }
  }
}

// w is H concatenated N x M matrix in row-major order, x is HM x K matrix in
// column-major order and y = w' * x is N x K matrix in column-major order,
// where w' is the rearrangement of w into an N x HM matrix.
template<typename T>
void MultiHeadMatMul(const T* w, const T* x, T* y, size_t H, size_t N,
                     size_t M, size_t K) {
  memset(y, 0, N * K * sizeof(y[0]));
  for (size_t i = 0; i < K; ++i) {
    for (size_t h = 0; h < H; ++h) {
      for (size_t j = 0; j < N; ++j) {
        y[i * N + j] += DotT(&w[h * N * M + j * M], &x[i * H * M + h * M], M);
      }
    }
  }
}

template<typename T>
void RMSNormT(const T* w, const T* x, T* out, size_t N, size_t K) {
  constexpr T eps(1e-6);
  for (size_t i = 0; i < K; ++i) {
    T ss = SquaredL2(x + i * N, N);
    ss = T(1.0) / std::sqrt(ss / T(N) + eps);
    for (size_t j = 0; j < N; j++) {
      out[i * N + j] = (T(1.0) + w[j]) * (ss * x[i * N + j]);
    }
  }
}
template<typename T>
void Softmax(T* x, size_t N) {
  T sum = {};
  auto maxreal = std::real(x[0]);
  for (size_t i = 1; i < N; ++i) {
    if (std::real(x[i]) > maxreal) {
      maxreal = std::real(x[i]);
    }
  }
  for (size_t i = 0; i < N; ++i) {
    x[i] = std::exp(x[i] - maxreal);
    sum += x[i];
  }
  T scale = T(1.0) / sum;
  for (size_t i = 0; i < N; ++i) {
    x[i] *= scale;
  }
}
template<typename T>
void Softmax(T* x, size_t N, size_t K) {
  for (size_t i = 0; i < K; ++i) {
    Softmax(x + i * N, N);
  }
}
template <typename T>
void Softcap(float cap, T* x, size_t N) {
  const T inv_cap = T{1.0} / static_cast<T>(cap);
  for (size_t i = 0; i < N; ++i) {
    x[i] = static_cast<T>(cap) * std::tanh(x[i] * inv_cap);
  }
}

template<typename T>
void GatedGelu(const T* in, T* out, size_t N, size_t K) {
  for (size_t i = 0; i < K; ++i) {
    const T* x1 = in + i * 2 * N;
    const T* x2 = x1 + N;
    T* y = out + i * N;
    for (size_t j = 0; j < N; ++j) {
      y[j] = x2[j] * Gelu(x1[j]);
    }
  }
}

template<typename T>
void InputEmbedding(const T* w, const std::vector<int>& tokens, T scaling,
                    T* y, size_t N) {
  HWY_ASSERT(w != nullptr);
  HWY_ASSERT(y != nullptr);
  const size_t num_tokens = tokens.empty() ? 0 : tokens.size() - 1;
  for (size_t i = 0; i < num_tokens; ++i) {
    int token = tokens[i];
    memcpy(y + i * N, w + token * N, N * sizeof(y[0]));
    MulByConstT(scaling, y + i * N, N);
  }
}

template <typename T>
void MaskedAttention(const T* qkv, T* output, size_t num_tokens, size_t heads,
                     size_t qkv_dim, size_t seq_len) {
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    for (size_t head = 0; head < heads; ++head) {
      const size_t qoffset = pos * (heads + 2) * qkv_dim;
      const size_t aoffset = pos * heads * seq_len + head * seq_len;
      const T* q = qkv + qoffset + head * qkv_dim;
      for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
        const T* k = qkv + (pos2 * (heads + 2) + heads) * qkv_dim;
        output[aoffset + pos2] = DotT(q, k, qkv_dim);
      }
    }
  }
}
template <typename T>
void MaskedSoftmax(T* x, size_t num_tokens, size_t heads, size_t seq_len) {
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    for (size_t head = 0; head < heads; ++head) {
      size_t offset = pos * heads * seq_len + head * seq_len;
      Softmax(x + offset, pos + 1);
      memset(x + offset + pos + 1, 0, (seq_len - pos - 1) * sizeof(T));
    }
  }
}
template <typename T>
void MixByAttention(const T* qkv, const T* attention, T* output,
                    size_t num_tokens, size_t heads, size_t qkv_dim,
                    size_t seq_len) {
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    for (size_t head = 0; head < heads; ++head) {
      const T* att = &attention[pos * heads * seq_len + head * seq_len];
      T* out = &output[head * qkv_dim + pos * heads * qkv_dim];
      memset(out, 0, qkv_dim * sizeof(out[0]));
      for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
        size_t v_offset = (pos2 * (heads + 2) + heads + 1) * qkv_dim;
        const T* v = &qkv[v_offset];
        MulByConstAndAddT(att[pos2], v, out, qkv_dim);
      }
    }
  }
}
template <typename T>
void ApplyLayer(const LayerWeightsPtrs<T>& weights,
                ForwardLayer<T>& activations, size_t num_tokens, T* output) {
  const LayerConfig& layer_config = weights.layer_config;
  const size_t model_dim = layer_config.model_dim;
  const size_t seq_len = activations.input.Rows();
  const size_t qkv_dim = layer_config.qkv_dim;
  const size_t heads = layer_config.heads;
  const size_t ff_hidden_dim = layer_config.ff_hidden_dim;
  static const T query_scale = T(1.0) / std::sqrt(T(qkv_dim));

  RMSNormT(weights.pre_attention_norm_scale.Packed(),
           activations.input.Packed(), activations.pre_att_rms_out.Packed(),
           model_dim, num_tokens);

  MatMulT(weights.qkv_einsum_w.Packed(), activations.pre_att_rms_out.Packed(),
          activations.qkv.Packed(), (heads + 2) * qkv_dim, model_dim,
          num_tokens);

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    T* qkv = activations.qkv.Packed() + pos * (heads + 2) * qkv_dim;
    for (size_t h = 0; h <= heads; ++h) {
      Rope(qkv + h * qkv_dim, qkv_dim, pos);
    }
  }

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    T* qkv = activations.qkv.Packed() + pos * (heads + 2) * qkv_dim;
    MulByConstT(query_scale, qkv, heads * qkv_dim);
  }

  MaskedAttention(activations.qkv.Packed(), activations.att.Packed(),
                  num_tokens, heads, qkv_dim, seq_len);

  MaskedSoftmax(activations.att.Packed(), num_tokens, heads, seq_len);

  MixByAttention(activations.qkv.Packed(), activations.att.Packed(),
                 activations.att_out.Packed(), num_tokens, heads, qkv_dim,
                 seq_len);

  MultiHeadMatMul(weights.attn_vec_einsum_w.Packed(),
                  activations.att_out.Packed(),
                  activations.attention_out.Packed(), heads, model_dim, qkv_dim,
                  num_tokens);

  AddFromT(activations.input.Packed(), activations.attention_out.Packed(),
           num_tokens * model_dim);

  RMSNormT(weights.pre_ffw_norm_scale.Packed(),
           activations.attention_out.Packed(),
           activations.pre_ffw_rms_out.Packed(), model_dim, num_tokens);

  MatMulT(weights.gating_einsum_w.Packed(),
          activations.pre_ffw_rms_out.Packed(), activations.ffw_hidden.Packed(),
          ff_hidden_dim * 2, model_dim, num_tokens);

  GatedGelu(activations.ffw_hidden.Packed(),
            activations.ffw_hidden_gated.Packed(), ff_hidden_dim, num_tokens);

  MatMulT(weights.linear_w.Packed(), activations.ffw_hidden_gated.Packed(),
          output, model_dim, ff_hidden_dim, num_tokens);

  AddFromT(activations.attention_out.Packed(), output, num_tokens * model_dim);
}

template<typename T>
T CrossEntropyLoss(const T* x, const Prompt& prompt, size_t V) {
  T loss = {};
  const std::vector<int> tokens = prompt.tokens;
  const size_t num_tokens = tokens.empty() ? 0 : tokens.size() - 1;
  for (size_t i = 0; i < num_tokens; ++i) {
    if (i + 1 < prompt.context_size) {
      continue;  // next token is part of context, don't try to predict it
    }
    const int next_token = tokens[i + 1];
    loss += std::log(x[i * V + next_token]);
  }
  T scaling = -1.0 / std::log(2.0);
  return loss * scaling;
}

template <typename T>
T CrossEntropyLossForwardPass(const Prompt& prompt,
                              const ModelWeightsPtrs<T>& weights,
                              ForwardPass<T>& forward) {
  const ModelConfig& config = weights.weights_config;
  const size_t model_dim = config.model_dim;
  const size_t vocab_size = config.vocab_size;
  const size_t layers = config.layer_configs.size();
  const std::vector<int> tokens = prompt.tokens;
  const size_t num_tokens = tokens.empty() ? 0 : tokens.size() - 1;

  const T kEmbScaling = EmbeddingScaling(model_dim);
  InputEmbedding(weights.embedder_input_embedding.Packed(), tokens, kEmbScaling,
                 forward.layers[0].input.Packed(), model_dim);

  for (size_t layer = 0; layer < layers; ++layer) {
    T* output = layer + 1 < layers ? forward.layers[layer + 1].input.Packed()
                                   : forward.final_layer_output.Packed();
    ApplyLayer(*weights.GetLayer(layer), forward.layers[layer], num_tokens,
               output);
  }

  RMSNormT(weights.final_norm_scale.Packed(),
           forward.final_layer_output.Packed(),
           forward.final_norm_output.Packed(), model_dim, num_tokens);

  MatMulT(weights.embedder_input_embedding.Packed(),
          forward.final_norm_output.Packed(), forward.logits.Packed(),
          vocab_size, model_dim, num_tokens);

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    if (config.final_cap > 0.0f) {
      Softcap(config.final_cap, forward.logits.Packed() + pos * vocab_size,
              vocab_size);
    }
  }

  CopyMat(forward.logits, forward.probs);
  Softmax(forward.probs.Packed(), vocab_size, num_tokens);

  return CrossEntropyLoss(forward.probs.Packed(), prompt, vocab_size);
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_FORWARD_SCALAR_H_
