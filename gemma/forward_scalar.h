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

#include "gemma/activations.h"
#include "gemma/common_scalar.h"
#include "gemma/prompt.h"
#include "gemma/weights.h"

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
template<typename T>
void Softcap(T* x, size_t N) {
  T cap = 30.0;
  T inv_cap = T(1.0) / cap;
  for (size_t i = 0; i < N; ++i) {
    x[i] = cap * std::tanh(x[i] * inv_cap);
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
  for (size_t i = 0; i + 1 < tokens.size(); ++i) {
    int token = tokens[i];
    memcpy(y + i * N, w + token * N, N * sizeof(y[0]));
    MulByConstT(scaling, y + i * N, N);
  }
}

template<typename T>
void MaskedAttention(const T* qkv, T* output, size_t num_tokens,
                     size_t kHeads, size_t kQKVDim, size_t kSeqLen) {
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    for (size_t head = 0; head < kHeads; ++head) {
      const size_t qoffset = pos * (kHeads + 2) * kQKVDim;
      const size_t aoffset = pos * kHeads * kSeqLen + head * kSeqLen;
      const T* q = qkv + qoffset + head * kQKVDim;
      for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
        const T* k = qkv + (pos2 * (kHeads + 2) + kHeads) * kQKVDim;
        output[aoffset + pos2] = DotT(q, k, kQKVDim);
      }
    }
  }
}
template<typename T>
void MaskedSoftmax(T* x, size_t num_tokens, size_t kHeads, size_t kSeqLen) {
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    for (size_t head = 0; head < kHeads; ++head) {
      size_t offset = pos * kHeads * kSeqLen + head * kSeqLen;
      Softmax(x + offset, pos + 1);
      memset(x + offset + pos + 1, 0, (kSeqLen - pos - 1) * sizeof(T));
    }
  }
}
template<typename T>
void MixByAttention(const T* qkv, const T* attention, T* output,
                    size_t num_tokens, size_t kHeads, size_t kQKVDim,
                    size_t kSeqLen) {
  for (size_t pos = 0; pos < num_tokens; ++pos) {
    for (size_t head = 0; head < kHeads; ++head) {
      const T* att = &attention[pos * kHeads * kSeqLen + head * kSeqLen];
      T* out = &output[head * kQKVDim + pos * kHeads * kQKVDim];
      memset(out, 0, kQKVDim * sizeof(out[0]));
      for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
        size_t v_offset = (pos2 * (kHeads + 2) + kHeads + 1) * kQKVDim;
        const T* v = &qkv[v_offset];
        MulByConstAndAddT(att[pos2], v, out, kQKVDim);
      }
    }
  }
}
template<typename T, typename TConfig>
void ApplyLayer(const Layer<T, TConfig>& weights,
                ForwardLayer<T, TConfig>& activations,
                size_t num_tokens, T* output) {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kSeqLen = TConfig::kSeqLen;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;
  static const T kQueryScale = T(1.0) / std::sqrt(T(kQKVDim));

  RMSNormT(weights.pre_attention_norm_scale.data(), activations.input.data(),
           activations.pre_att_rms_out.data(), kModelDim, num_tokens);

  MatMulT(weights.qkv_einsum_w.data(), activations.pre_att_rms_out.data(),
          activations.qkv.data(), (kHeads + 2) * kQKVDim, kModelDim,
          num_tokens);

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    T* qkv = activations.qkv.data() + pos * (kHeads + 2) * kQKVDim;
    for (size_t h = 0; h <= kHeads; ++h) {
      Rope(qkv + h * kQKVDim, kQKVDim, pos);
    }
  }

  for (size_t pos = 0; pos < num_tokens; ++pos) {
    T* qkv = activations.qkv.data() + pos * (kHeads + 2) * kQKVDim;
    MulByConstT(kQueryScale, qkv, kHeads * kQKVDim);
  }

  MaskedAttention(activations.qkv.data(), activations.att.data(),
                  num_tokens, kHeads, kQKVDim, kSeqLen);

  MaskedSoftmax(activations.att.data(), num_tokens, kHeads, kSeqLen);

  MixByAttention(activations.qkv.data(), activations.att.data(),
                 activations.att_out.data(), num_tokens, kHeads, kQKVDim,
                 kSeqLen);

  MultiHeadMatMul(weights.attn_vec_einsum_w.data(), activations.att_out.data(),
                  activations.attention_out.data(), kHeads, kModelDim, kQKVDim,
                  num_tokens);

  AddFromT(activations.input.data(), activations.attention_out.data(),
           num_tokens * kModelDim);

  RMSNormT(weights.pre_ffw_norm_scale.data(), activations.attention_out.data(),
           activations.bf_pre_ffw_rms_out.data(), kModelDim, num_tokens);

  MatMulT(weights.gating_einsum_w.data(), activations.bf_pre_ffw_rms_out.data(),
          activations.ffw_hidden.data(), kFFHiddenDim * 2, kModelDim,
          num_tokens);

  GatedGelu(activations.ffw_hidden.data(), activations.ffw_hidden_gated.data(),
            kFFHiddenDim, num_tokens);

  MatMulT(weights.linear_w.data(), activations.ffw_hidden_gated.data(),
          output, kModelDim, kFFHiddenDim, num_tokens);

  AddFromT(activations.attention_out.data(), output, num_tokens * kModelDim);
}

template<typename T>
T CrossEntropyLoss(const T* x, const Prompt& prompt, size_t V) {
  T loss = {};
  for (size_t i = 0; i + 1 < prompt.tokens.size(); ++i) {
    if (i + 1 < prompt.context_size) {
      continue;  // next token is part of context, don't try to predict it
    }
    const int next_token = prompt.tokens[i + 1];
    loss += std::log(x[i * V + next_token]);
  }
  T scaling = -1.0 / std::log(2.0);
  return loss * scaling;
}

template<typename T, typename TConfig>
T CrossEntropyLossForwardPass(const Prompt& prompt,
                              const Weights<T, TConfig>& weights,
                              ForwardPass<T, TConfig>& forward) {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kVocabSize = TConfig::kVocabSize;
  static constexpr size_t kLayers = TConfig::kLayers;
  const size_t num_tokens = prompt.tokens.size() - 1;

  const T kEmbScaling = EmbeddingScaling(kModelDim);
  InputEmbedding(weights.embedder_input_embedding.data(), prompt.tokens,
                 kEmbScaling, forward.layers[0].input.data(), kModelDim);

  for (size_t layer = 0; layer < kLayers; ++layer) {
    T* output = layer + 1 < kLayers ?
                forward.layers[layer + 1].input.data() :
                forward.final_layer_output.data();
    ApplyLayer(*weights.GetLayer(layer), forward.layers[layer], num_tokens,
               output);
  }

  RMSNormT(weights.final_norm_scale.data(),
           forward.final_layer_output.data(),
           forward.final_norm_output.data(), kModelDim, num_tokens);

  MatMulT(weights.embedder_input_embedding.data(),
          forward.final_norm_output.data(),
          forward.logits.data(), kVocabSize, kModelDim, num_tokens);

  Softcap(forward.logits.data(), num_tokens * kVocabSize);

  memcpy(forward.probs.data(), forward.logits.data(),
         num_tokens * kVocabSize * sizeof(forward.logits[0]));
  Softmax(forward.probs.data(), kVocabSize, num_tokens);

  return CrossEntropyLoss(forward.probs.data(), prompt, kVocabSize);
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_FORWARD_SCALAR_H_
