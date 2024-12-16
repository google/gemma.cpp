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

#include "backprop/backward_scalar.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>  // memcpy

#include <complex>
#include <limits>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "backprop/activations.h"
#include "backprop/common_scalar.h"
#include "backprop/forward_scalar.h"
#include "backprop/prompt.h"
#include "backprop/sampler.h"
#include "backprop/test_util.h"
#include "compression/compress.h"
#include "gemma/configs.h"
#include "gemma/weights.h"

namespace gcpp {

TEST(BackPropTest, MatMulVJP) {
  static const size_t kRows = 8;
  static const size_t kCols = 64;
  static const size_t kTokens = 5;
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  MatStorageT<T> weights("weights", kRows, kCols);
  MatStorageT<T> x("x", kTokens, kCols);
  MatStorageT<T> grad("grad", kRows, kCols);
  MatStorageT<T> dx("dx", kTokens, kCols);
  MatStorageT<TC> c_weights("c_weights", kRows, kCols);
  MatStorageT<TC> c_x("c_x", kTokens, kCols);
  MatStorageT<TC> c_y("c_y", kTokens, kRows);
  MatStorageT<T> dy("dy", kTokens, kRows);

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(weights, 1.0 * (1 << iter), gen);
    RandInit(x, 1.0 * (1 << iter), gen);
    RandInit(dy, 1.0, gen);
    Complexify(weights, c_weights);
    Complexify(x, c_x);
    auto func = [&]() {
      MatMulT(c_weights.data(), c_x.data(), c_y.data(), kRows, kCols, kTokens);
      return DotT(dy.data(), c_y.data(), kTokens * kRows);
    };
    grad.ZeroInit();
    MatMulVJPT(weights.data(), x.data(), dy.data(), grad.data(), dx.data(),
               kRows, kCols, kTokens);
    TestGradient(dx, c_x, func, 1e-11, 1e-12, __LINE__);
    TestGradient(grad, c_weights, func, 1e-14, 1e-12, __LINE__);
  }
}

TEST(BackPropTest, MultiHeadMatMulVJP) {
  static const size_t kRows = 2;
  static const size_t kCols = 16;
  static const size_t kHeads = 4;
  static const size_t kTokens = 3;
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  MatStorageT<T> weights("weights", kRows, kCols * kHeads);
  MatStorageT<T> x("x", kTokens, kCols * kHeads);
  MatStorageT<T> grad("grad", kRows, kCols * kHeads);
  MatStorageT<T> dx("dx", kTokens, kCols * kHeads);
  MatStorageT<TC> c_weights("c_weights", kRows, kCols * kHeads);
  MatStorageT<TC> c_x("c_x", kTokens, kCols * kHeads);
  MatStorageT<TC> c_y("c_y", kTokens, kRows);
  MatStorageT<T> dy("dy", kTokens, kRows);

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(weights, 1.0 * (1 << iter), gen);
    RandInit(x, 1.0 * (1 << iter), gen);
    RandInit(dy, 1.0, gen);
    Complexify(weights, c_weights);
    Complexify(x, c_x);
    auto func = [&]() {
      MultiHeadMatMul(c_weights.data(), c_x.data(), c_y.data(), kHeads, kRows,
                      kCols, kTokens);
      return DotT(dy.data(), c_y.data(), kTokens * kRows);
    };
    grad.ZeroInit();
    MultiHeadMatMulVJPT(weights.data(), x.data(), dy.data(), grad.data(),
                        dx.data(), kHeads, kRows, kCols, kTokens);
    TestGradient(dx, c_x, func, 1e-15, 1e-13, __LINE__);
    TestGradient(grad, c_weights, func, 1e-15, 1e-13, __LINE__);
  }
}

TEST(BackPropTest, RMSNormVJP) {
  static const size_t K = 2;
  static const size_t N = 64;
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  MatStorageT<T> weights("weights", N, 1);
  MatStorageT<T> grad("grad", N, 1);
  MatStorageT<T> x("x", K, N);
  MatStorageT<T> dx("dx", K, N);
  MatStorageT<T> dy("dy", K, N);
  MatStorageT<TC> c_weights("c_weights", N, 1);
  MatStorageT<TC> c_x("c_x", K, N);
  MatStorageT<TC> c_y("c_y", K, N);

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(weights, 1.0 * (1 << iter), gen);
    RandInit(x, 1.0 * (1 << iter), gen);
    Complexify(weights, c_weights);
    Complexify(x, c_x);
    RandInit(dy, 1.0, gen);
    auto func = [&]() {
      RMSNormT(c_weights.data(), c_x.data(), c_y.data(), N, K);
      return DotT(dy.data(), c_y.data(), K * N);
    };
    grad.ZeroInit();
    RMSNormVJPT(weights.data(), x.data(), dy.data(), grad.data(), dx.data(),
                N, K);
    TestGradient(dx, c_x, func, 1e-15, 1e-14, __LINE__);
    TestGradient(grad, c_weights, func, 1e-15, 1e-14, __LINE__);
  }
}

TEST(BackPropTest, SoftmaxVJP) {
  static const size_t N = 64;
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  MatStorageT<T> x("x", N, 1);
  MatStorageT<T> dx("dx", N, 1);
  MatStorageT<T> dy("dy", N, 1);
  MatStorageT<TC> c_x("c_x", N, 1);
  MatStorageT<TC> c_y("c_y", N, 1);

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(x, 1.0 * (1 << iter), gen);
    Complexify(x, c_x);
    RandInit(dy, 1.0, gen);
    auto func = [&]() {
      memcpy(c_y.data(), c_x.data(), c_x.SizeBytes());
      Softmax(c_y.data(), N);
      return DotT(dy.data(), c_y.data(), N);
    };
    Softmax(x.data(), N);
    memcpy(dx.data(), dy.data(), dx.SizeBytes());
    SoftmaxVJPT(x.data(), dx.data(), N);
    TestGradient(dx, c_x, func, 1e-15, 1e-15, __LINE__);
  }
}

TEST(BackPropTest, MaskedSoftmaxVJP) {
  static const size_t kSeqLen = 16;
  static const size_t kHeads = 2;
  static const size_t kTokens = 14;
  static const size_t N = kTokens * kHeads * kSeqLen;
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  MatStorageT<T> x("x", N, 1);
  MatStorageT<T> dy("dy", N, 1);
  MatStorageT<T> dx("dx", N, 1);
  MatStorageT<TC> c_x("c_x", N, 1);
  MatStorageT<TC> c_y("c_y", N, 1);
  dx.ZeroInit();

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(x, 1.0 * (1 << iter), gen);
    Complexify(x, c_x);
    RandInit(dy, 1.0, gen);
    auto func = [&]() {
      memcpy(c_y.data(), c_x.data(),
             kTokens * kHeads * kSeqLen * sizeof(c_x.At(0)));
      MaskedSoftmax(c_y.data(), kTokens, kHeads, kSeqLen);
      return DotT(dy.data(), c_y.data(), N);
    };
    MaskedSoftmax(x.data(), kTokens, kHeads, kSeqLen);
    memcpy(dx.data(), dy.data(), kTokens * kHeads * kSeqLen * sizeof(dx.At(0)));
    MaskedSoftmaxVJPT(x.data(), dx.data(), kTokens, kHeads, kSeqLen);
    TestGradient(dx, c_x, func, 1e-14, 1e-15, __LINE__);
  }
}

TEST(BackPropTest, SoftcapVJP) {
  static const size_t N = 64;
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  MatStorageT<T> x("x", N, 1);
  MatStorageT<T> dx("dx", N, 1);
  MatStorageT<T> dy("dy", N, 1);
  MatStorageT<TC> c_x("c_x", N, 1);
  MatStorageT<TC> c_y("c_y", N, 1);

  constexpr float kCap = 30.0f;
  for (int iter = 0; iter < 10; ++iter) {
    RandInit(x, 1.0 * (1 << iter), gen);
    Complexify(x, c_x);
    RandInit(dy, 1.0, gen);
    auto func = [&]() {
      memcpy(c_y.data(), c_x.data(), N * sizeof(c_x.At(0)));
      Softcap(kCap, c_y.data(), N);
      return DotT(dy.data(), c_y.data(), N);
    };
    Softcap(kCap, x.data(), N);
    memcpy(dx.data(), dy.data(), dx.SizeBytes());
    SoftcapVJPT(kCap, x.data(), dx.data(), N);
    TestGradient(dx, c_x, func, 1e-15, 1e-14, __LINE__);
  }
}

TEST(BackPropTest, CrossEntropyLossGrad) {
  static const size_t K = 8;
  static const size_t V = 64;
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  MatStorageT<T> x("x", K, V);
  MatStorageT<T> dx("dx", K, V);
  MatStorageT<TC> c_x("c_x", K, V);
  Prompt prompt;
  prompt.tokens = { 0, 1, 2, 3, 0, 3, 2, 1, 0 };

  const float kCap = 30.0f;
  for (int iter = 0; iter < 10; ++iter) {
    prompt.context_size = 1 + (iter % 6);
    RandInit(x, 1.0 * (1 << iter), gen);
    Softcap(kCap, x.data(), V * K);
    Softmax(x.data(), V, K);
    CrossEntropyLossGrad(x.data(), dx.data(), prompt, V);
    Complexify(x, c_x);
    auto func = [&]() {
      return CrossEntropyLoss(c_x.data(), prompt, V);
    };
    TestGradient(dx, c_x, func, 1e-100, 1e-15, __LINE__);
  }
}

TEST(BackPropTest, GatedGeluVJP) {
  static const size_t K = 2;
  static const size_t N = 64;
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  MatStorageT<T> x("x", K, 2 * N);
  MatStorageT<T> dx("dx", K, 2 * N);
  MatStorageT<T> dy("dy", K, N);
  MatStorageT<TC> c_x("c_x", K, 2 * N);
  MatStorageT<TC> c_y("c_y", K, N);

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(x, 1.0, gen);
    Complexify(x, c_x);
    RandInit(dy, 1.0, gen);
    auto func = [&]() {
      GatedGelu(c_x.data(), c_y.data(), N, K);
      return DotT(dy.data(), c_y.data(), N * K);
    };
    GatedGeluVJP(x.data(), dy.data(), dx.data(), N, K);
    TestGradient(dx, c_x, func, 1e-15, 1e-15, __LINE__);
  }
}

TEST(BackPropTest, MaskedAttentionVJP) {
  static const size_t kSeqLen = 16;
  static const size_t kHeads = 2;
  static const size_t kQKVDim = 8;
  static const size_t kTokens = 14;
  static const size_t kQKVSize = kSeqLen * (kHeads + 2) * kQKVDim;
  static const size_t kOutSize = kTokens * kHeads * kSeqLen;
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  MatStorageT<T> x("x", kQKVSize, 1);
  MatStorageT<T> dx("dx", kQKVSize, 1);
  MatStorageT<T> dy("dy", kOutSize, 1);
  MatStorageT<TC> c_x("c_x", kQKVSize, 1);
  MatStorageT<TC> c_y("c_y", kOutSize, 1);
  dx.ZeroInit();
  c_y.ZeroInit();

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(x, 1.0, gen);
    Complexify(x, c_x);
    RandInit(dy, 1.0, gen);
    auto func = [&]() {
      MaskedAttention(c_x.data(), c_y.data(), kTokens, kHeads, kQKVDim,
                      kSeqLen);
      return DotT(dy.data(), c_y.data(), kOutSize);
    };
    MaskedAttentionVJP(x.data(), dy.data(), dx.data(),
                       kTokens, kHeads, kQKVDim, kSeqLen);
    TestGradient(dx, c_x, func, 1e-14, 1e-15, __LINE__);
  }
}

TEST(BackPropTest, MixByAttentionVJP) {
  static const size_t kSeqLen = 16;
  static const size_t kHeads = 2;
  static const size_t kQKVDim = 8;
  static const size_t kTokens = 14;
  static const size_t kQKVSize = kSeqLen * (kHeads + 2) * kQKVDim;
  static const size_t kAttnSize = kSeqLen * kHeads * kSeqLen;
  static const size_t kOutSize = kSeqLen * kHeads * kQKVDim;
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  MatStorageT<T> qkv("qkv", kQKVSize, 1);
  MatStorageT<T> dqkv("dqkv", kQKVSize, 1);
  MatStorageT<T> attn("attn", kAttnSize, 1);
  MatStorageT<T> dattn("dattn", kAttnSize, 1);
  MatStorageT<T> dy("dy", kOutSize, 1);
  MatStorageT<TC> c_qkv("c_qkv", kQKVSize, 1);
  MatStorageT<TC> c_attn("c_attn", kAttnSize, 1);
  MatStorageT<TC> c_y("c_y", kOutSize, 1);
  dqkv.ZeroInit();
  dattn.ZeroInit();
  c_y.ZeroInit();

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(qkv, 1.0, gen);
    RandInit(attn, 1.0, gen);
    Complexify(qkv, c_qkv);
    Complexify(attn, c_attn);
    RandInit(dy, 1.0, gen);
    auto func = [&]() {
      MixByAttention(c_qkv.data(), c_attn.data(), c_y.data(),
                     kTokens, kHeads, kQKVDim, kSeqLen);
      return DotT(dy.data(), c_y.data(), kOutSize);
    };
    MixByAttentionVJP(qkv.data(), attn.data(), dy.data(), dqkv.data(),
                      dattn.data(), kTokens, kHeads, kQKVDim, kSeqLen);
    TestGradient(dqkv, c_qkv, func, 1e-14, 1e-15, __LINE__);
    TestGradient(dattn, c_attn, func, 1e-14, 1e-15, __LINE__);
  }
}

TEST(BackPropTest, InputEmbeddingVJP) {
  static const size_t kSeqLen = 8;
  static const size_t kVocabSize = 4;
  static const size_t kModelDim = 16;
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  MatStorageT<T> weights("weights", kVocabSize, kModelDim);
  MatStorageT<T> grad("grad", kVocabSize, kModelDim);
  MatStorageT<T> dy("dy", kSeqLen, kModelDim);
  MatStorageT<TC> c_weights("c_weights", kVocabSize, kModelDim);
  MatStorageT<TC> c_y("c_y", kSeqLen, kModelDim);
  std::vector<int> tokens = { 0, 1, 2, 3, 0, 1, 2 };
  size_t num_tokens = tokens.size() - 1;

  for (size_t iter = 0; iter < 10; ++iter) {
    RandInit(weights, 1.0, gen);
    RandInit(dy, 1.0, gen);
    Complexify(weights, c_weights);
    auto func = [&]() {
      InputEmbedding(c_weights.data(), tokens, TC(3.0), c_y.data(), kModelDim);
      return DotT(dy.data(), c_y.data(), num_tokens * kModelDim);
    };
    grad.ZeroInit();
    InputEmbeddingVJPT(weights.data(), tokens, 3.0, dy.data(), grad.data(),
                       kModelDim);
    TestGradient(grad, c_weights, func, 1e-16, 1e-14, __LINE__);
  }
}

static ModelConfig TestConfig() {
  ModelConfig config;
  config.scale_names = {"att_ein",      "qkv_ein",   "gr_lin_x_w", "gr_lin_y_w",
                        "gr_lin_out_w", "gr_gate_w", "gating_ein", "linear_w"};
  config.model_dim = 32;
  config.vocab_size = 12;
  config.seq_len = 18;
  LayerConfig layer_config;
  layer_config.model_dim = config.model_dim;
  layer_config.ff_hidden_dim = 48;
  layer_config.heads = 3;
  layer_config.kv_heads = 1;
  layer_config.qkv_dim = 12;
  config.layer_configs = {2, layer_config};
  config.num_tensor_scales = 4 * config.layer_configs.size();
  config.query_scale = QueryScaleType::SqrtKeySize;
  config.attention_window_sizes = FixedAttentionWindowSizes<2>(32);
  // This is required for optimize_test to pass.
  config.final_cap = 30.0f;
  return config;
}

TEST(BackPropTest, LayerVJP) {
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  ModelConfig config = TestConfig();
  TensorIndex tensor_index(config, /*llm_layer_idx=*/0, /*img_layer_idx=*/-1,
                           /*reshape_att=*/false);
  const size_t kOutputSize = config.seq_len * config.model_dim;
  LayerWeightsPtrs<T> weights(config.layer_configs[0], tensor_index);
  LayerWeightsPtrs<T> grad(config.layer_configs[0], tensor_index);
  ForwardLayer<T> forward(config.layer_configs[0], config.seq_len);
  ForwardLayer<T> backward(config.layer_configs[0], config.seq_len);
  LayerWeightsPtrs<TC> c_weights(config.layer_configs[0], tensor_index);
  ForwardLayer<TC> c_forward(config.layer_configs[0], config.seq_len);
  MatStorageT<T> y("y", kOutputSize, 1);
  MatStorageT<T> dy("dy", kOutputSize, 1);
  MatStorageT<TC> c_y("c_y", kOutputSize, 1);
  const size_t num_tokens = 3;
  std::vector<MatStorage> layer_storage;
  weights.Allocate(layer_storage);
  grad.Allocate(layer_storage);
  c_weights.Allocate(layer_storage);
  backward.input.ZeroInit();

  for (size_t iter = 0; iter < 10; ++iter) {
    RandInit(weights, 1.0, gen);
    RandInit(forward.input, 1.0, gen);
    RandInit(dy, 1.0, gen);
    Complexify(weights, c_weights);
    Complexify(forward.input, c_forward.input);
    auto func = [&]() {
      ApplyLayer(c_weights, c_forward, num_tokens, c_y.data());
      return DotT(dy.data(), c_y.data(), num_tokens * config.model_dim);
    };
    grad.ZeroInit(/*layer_idx=*/0);
    ApplyLayer(weights, forward, num_tokens, y.data());
    LayerVJP(weights, forward, dy.data(), grad, backward, num_tokens);
    TestGradient(backward.input, c_forward.input, func, 1e-11, 5e-11,
                 __LINE__);
    TestGradient(grad, c_weights, func, 1e-11);
  }
}

TEST(BackPropTest, EndToEnd) {
  std::mt19937 gen(42);
  using T = double;
  using TC = std::complex<T>;
  ModelConfig config = TestConfig();
  WeightsWrapper<T> weights(config);
  WeightsWrapper<T> grad(config);
  ForwardPass<T> forward(config);
  ForwardPass<T> backward(config);
  WeightsWrapper<TC> c_weights(config);
  ForwardPass<TC> c_forward(config);

  ReverseSequenceSampler training_task({0, 0, 1, 1});
  std::vector<Prompt> batch = training_task.SampleBatch(3, gen);

  for (const Prompt& prompt : batch) {
    ReverseSequenceSampler::LogPrompt(prompt);
    RandInit(weights.get(), 1.0, gen);
    CrossEntropyLossForwardPass(prompt, weights.get(), forward);
    grad.ZeroInit();
    CrossEntropyLossBackwardPass(
        prompt, weights.get(), forward, grad.get(), backward);

    Complexify(weights.get(), c_weights.get());
    auto func = [&]() {
      return CrossEntropyLossForwardPass(prompt, c_weights.get(), c_forward);
    };

    TestGradient(grad.get(), c_weights.get(), func, 1e-11);
  }
}

template <typename T>
void MulByConstAndAddT(T c, const LayerWeightsPtrs<T>& x,
                       LayerWeightsPtrs<T>& out) {
  MulByConstAndAddT(c, x.pre_attention_norm_scale,
                    out.pre_attention_norm_scale);
  MulByConstAndAddT(c, x.attn_vec_einsum_w, out.attn_vec_einsum_w);
  MulByConstAndAddT(c, x.qkv_einsum_w, out.qkv_einsum_w);
  MulByConstAndAddT(c, x.pre_ffw_norm_scale, out.pre_ffw_norm_scale);
  MulByConstAndAddT(c, x.gating_einsum_w, out.gating_einsum_w);
  MulByConstAndAddT(c, x.linear_w, out.linear_w);
}

template <typename T>
void MulByConstAndAddT(T c, const ModelWeightsPtrs<T>& x,
                       ModelWeightsPtrs<T>& out) {
  const size_t layers = x.c_layers.size();
  MulByConstAndAddT(c, x.embedder_input_embedding,
                    out.embedder_input_embedding);
  MulByConstAndAddT(c, x.final_norm_scale, out.final_norm_scale);
  for (size_t i = 0; i < layers; ++i) {
    MulByConstAndAddT(c, *x.GetLayer(i), *out.GetLayer(i));
  }
}

// Evaluates forward pass on a batch.
template <typename T>
T CrossEntropyLossForwardPass(const std::vector<Prompt>& batch,
                              const WeightsWrapper<T>& weights,
                              ForwardPass<T>& forward) {
  T loss = 0.0;
  for (const Prompt& prompt : batch) {
    loss += CrossEntropyLossForwardPass(prompt, weights.get(), forward);
  }
  T scale = 1.0 / batch.size();
  return loss * scale;
}

// Evaluates forward pass on a batch by applying gradient with the given
// learning rate. Does not update weights, but uses the given tmp weights
// instead.
template <typename T>
T CrossEntropyLossForwardPass(T learning_rate, const std::vector<Prompt>& batch,
                              const WeightsWrapper<T>& weights,
                              const WeightsWrapper<T>& grad,
                              WeightsWrapper<T>& tmp, ForwardPass<T>& forward) {
  tmp.CopyFrom(weights);
  const T scale = -learning_rate / batch.size();
  MulByConstAndAddT(scale, grad.get(), tmp.get());
  return CrossEntropyLossForwardPass(batch, tmp, forward);
}

// Uses line search in the negative gradient direction to update weights. We do
// this so that we can test that each step during the gradient descent can
// decrease the objective function value.
template <typename T>
T FindOptimalUpdate(const WeightsWrapper<T>& grad, WeightsWrapper<T>& weights,
                    WeightsWrapper<T>& tmp, ForwardPass<T>& forward,
                    const std::vector<Prompt>& batch, T loss,
                    T initial_learning_rate) {
  T lr0 = initial_learning_rate;
  T loss0 = CrossEntropyLossForwardPass(
      lr0, batch, weights, grad, tmp, forward);
  for (size_t iter = 0; iter < 30; ++iter) {
    T lr1 = lr0 * 0.5;
    T loss1 = CrossEntropyLossForwardPass(
        lr1, batch, weights, grad, tmp, forward);
    if (loss0 < loss && loss1 >= loss0) {
      break;
    }
    loss0 = loss1;
    lr0 = lr1;
  }
  for (size_t iter = 0; iter < 30; ++iter) {
    T lr1 = lr0 * 2.0;
    T loss1 = CrossEntropyLossForwardPass(
        lr1, batch, weights, grad, tmp, forward);
    if (loss1 >= loss0) {
      break;
    }
    loss0 = loss1;
    lr0 = lr1;
  }
  const T scale = -lr0 / batch.size();
  MulByConstAndAddT(scale, grad.get(), weights.get());
  return lr0;
}

TEST(BackProptest, Convergence) {
  std::mt19937 gen(42);
  using T = float;
  using TC = std::complex<double>;
  ModelConfig config = TestConfig();
  WeightsWrapper<T> weights(config);
  WeightsWrapper<T> grad(config);
  WeightsWrapper<T> tmp(config);
  ForwardPass<T> forward(config);
  ForwardPass<T> backward(config);
  WeightsWrapper<TC> c_weights(config);
  ForwardPass<TC> c_forward(config);
  constexpr size_t kBatchSize = 5;
  ReverseSequenceSampler training_task({0, 0, 0, 1, 1});
  T learning_rate = 0.01;

  RandInit(weights.get(), T(1.0), gen);

  printf("Sample batch:\n");
  for (size_t i = 0; i < 10; ++i) {
    ReverseSequenceSampler::LogPrompt(training_task.Sample(gen));
  }

  T prev_loss = std::numeric_limits<T>::max();
  bool stop = false;
  size_t step = 0;
  while (!stop) {
    T loss = 0.0;
    grad.ZeroInit();
    std::mt19937 sgen(42);
    std::vector<Prompt> batch = training_task.SampleBatch(kBatchSize, sgen);
    for (const Prompt& prompt : batch) {
      loss += CrossEntropyLossForwardPass(prompt, weights.get(), forward);
      CrossEntropyLossBackwardPass(
          prompt, weights.get(), forward, grad.get(), backward);
    }

    if (step % 250 == 0) {
      printf("Checking gradient...\n");
      Complexify(weights.get(), c_weights.get());
      auto func = [&]() {
        TC scale = batch.size();
        return CrossEntropyLossForwardPass(batch, c_weights, c_forward) * scale;
      };

      TestGradient(grad.get(), c_weights.get(), func, 5e-3f);
    }

    loss /= batch.size();
    EXPECT_LT(loss, prev_loss);
    stop = step >= 10000 || loss < 1e-2;
    if (step % 10 == 0 || stop) {
      printf("step: %5zu  loss: %.15f  learning_rate: %.15f\n",
             step, loss, learning_rate);
    }
    if (!stop) {
      learning_rate = FindOptimalUpdate(
          grad, weights, tmp, forward, batch, loss, learning_rate);
      ++step;
    }
    prev_loss = loss;
  }
  EXPECT_LT(step, 1000);
}

}  // namespace gcpp
