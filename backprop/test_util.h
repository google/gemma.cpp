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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_TEST_UTIL_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_TEST_UTIL_H_

#include <array>
#include <complex>
#include <random>

#include "gemma/weights.h"
#include "gtest/gtest.h"

namespace gcpp {

template<typename T, size_t kLen>
void RandInit(std::array<T, kLen>& x, T stddev, std::mt19937& gen) {
  std::normal_distribution<T> dist(0.0, stddev);
  for (size_t i = 0; i < kLen; ++i) {
    x[i] = dist(gen);
  }
}

template<typename T, typename TConfig>
void RandInit(Layer<T, TConfig>& w, T stddev, std::mt19937& gen) {
  RandInit(w.pre_attention_norm_scale, stddev, gen);
  RandInit(w.attn_vec_einsum_w, stddev, gen);
  RandInit(w.qkv_einsum_w, stddev, gen);
  RandInit(w.pre_ffw_norm_scale, stddev, gen);
  RandInit(w.gating_einsum_w, stddev, gen);
  RandInit(w.linear_w, stddev, gen);
}

template<typename T, typename TConfig>
void RandInit(Weights<T, TConfig>& w, T stddev, std::mt19937& gen) {
  static constexpr size_t kLayers = TConfig::kLayers;
  RandInit(w.embedder_input_embedding, stddev, gen);
  RandInit(w.final_norm_scale, stddev, gen);
  for (size_t i = 0; i < kLayers; ++i) {
    RandInit(*w.GetLayer(i), stddev, gen);
  }
}

template<typename T, typename U, size_t kLen>
void Complexify(const std::array<T, kLen>& x,
                std::array<std::complex<U>, kLen>& c_x) {
  for (size_t i = 0; i < kLen; ++i) {
    c_x[i] = std::complex<U>(x[i], 0.0);
  }
}


template<typename T, typename U, typename TConfig>
void Complexify(const Layer<T, TConfig>& w,
                Layer<std::complex<U>, TConfig>& c_w) {
  Complexify(w.pre_attention_norm_scale, c_w.pre_attention_norm_scale);
  Complexify(w.attn_vec_einsum_w, c_w.attn_vec_einsum_w);
  Complexify(w.qkv_einsum_w, c_w.qkv_einsum_w);
  Complexify(w.pre_ffw_norm_scale, c_w.pre_ffw_norm_scale);
  Complexify(w.gating_einsum_w, c_w.gating_einsum_w);
  Complexify(w.linear_w, c_w.linear_w);
}

template<typename T, typename U, typename TConfig>
void Complexify(const Weights<T, TConfig>& w,
                Weights<std::complex<U>, TConfig>& c_w) {
  static constexpr size_t kLayers = TConfig::kLayers;
  Complexify(w.embedder_input_embedding, c_w.embedder_input_embedding);
  Complexify(w.final_norm_scale, c_w.final_norm_scale);
  for (size_t i = 0; i < kLayers; ++i) {
    Complexify(*w.GetLayer(i), *c_w.GetLayer(i));
  }
}

template<typename T, typename U, size_t N>
void TestNear(const std::array<T, N>& actual, const std::array<U, N>& expected,
              double max_abs_err, double max_rel_err, int line) {
  double sum0 = 0;
  double sum1 = 0;
  double sum01 = 0;
  for (size_t i = 0; i < N; ++i) {
    sum0 += actual[i] * actual[i];
    sum1 += expected[i] * expected[i];
    sum01 += actual[i] * expected[i];
    ASSERT_NEAR(actual[i], expected[i],
                std::max(max_abs_err, std::abs(expected[i]) * max_rel_err))
        << "line: " << line << " dim=" << N << " i=" << i;
  }
  if (sum0 > 1e-40) {
    double norm_dot = sum01 / std::sqrt(sum0) / std::sqrt(sum1);
    ASSERT_NEAR(norm_dot, 1.0, 1e-7)
        << "line: " << line << " sum0: " << sum0  << " sum1: " << sum1
        << " sum01: " << sum01;
  }
}

// Compute gradient with the finite difference method in the complex plane.
// If f : R->R is the tested function and F : C->C is its extension on the
// complex plane so that F is complex differentiable in x, then
//
//   F(x + ih) = F(x) + ih F'(x) + O(h^2) F''(x)
//
// which means that
//
//   F'(x) ~= Imag(F(x + ih)) / h
//
// This method is more numerically stable than the real-valued finite difference
// method since we don't need to subtract floating point numbers that are near
// to each other.
template<typename T, typename U, size_t N, typename FUNC>
void TestGradient(const std::array<T, N>& grad,
                  std::array<std::complex<U>, N>& x, FUNC func,
                  U step, T max_abs_err, T max_rel_err, int line) {
  std::array<T, N> exp_grad;
  const U inv_step = 1.0 / step;
  for (size_t i = 0; i < N; ++i) {
    const U x0 = std::real(x[i]);
    const std::complex<U> x1 = std::complex<U>(x0, step);
    x[i] = x1;
    const std::complex<U> f1 = func();
    exp_grad [i] = std::imag(f1) * inv_step;
    x[i] = x0;
  }
  TestNear(grad, exp_grad, max_abs_err, max_rel_err, line);
}

template<size_t N, typename FUNC>
void TestGradient(const std::array<float, N>& grad,
                  std::array<std::complex<float>, N>& x, FUNC func,
                  float max_abs_err, float max_rel_error, int line) {
  TestGradient(grad, x, func, 1e-30f, max_abs_err, max_rel_error, line);
}

template<size_t N, typename FUNC>
void TestGradient(const std::array<float, N>& grad,
                  std::array<std::complex<double>, N>& x, FUNC func,
                  float max_abs_err, float max_rel_error, int line) {
  TestGradient(grad, x, func, 1e-50, max_abs_err, max_rel_error, line);
}

template<size_t N, typename FUNC>
void TestGradient(const std::array<double, N>& grad,
                  std::array<std::complex<double>, N>& x, FUNC func,
                  double max_abs_err, double max_rel_error, int line) {
  TestGradient(grad, x, func, 1e-50, max_abs_err, max_rel_error, line);
}

template<typename T, typename U, typename TConfig, typename FUNC>
void TestGradient(const Layer<T, TConfig>& grad,
                  Layer<std::complex<U>, TConfig>& c_weights,
                  FUNC func, T max_err) {
  TestGradient(grad.pre_attention_norm_scale,
               c_weights.pre_attention_norm_scale,
               func, max_err, max_err, __LINE__);
  TestGradient(grad.attn_vec_einsum_w, c_weights.attn_vec_einsum_w,
               func, max_err, max_err, __LINE__);
  TestGradient(grad.qkv_einsum_w, c_weights.qkv_einsum_w,
               func, max_err, max_err, __LINE__);
  TestGradient(grad.pre_ffw_norm_scale, c_weights.pre_ffw_norm_scale,
               func, max_err, max_err, __LINE__);
  TestGradient(grad.gating_einsum_w, c_weights.gating_einsum_w,
               func, max_err, max_err, __LINE__);
  TestGradient(grad.linear_w, c_weights.linear_w,
               func, max_err, max_err, __LINE__);
}

template<typename T, typename U, typename TConfig, typename FUNC>
void TestGradient(const Weights<T, TConfig>& grad,
                  Weights<std::complex<U>, TConfig>& c_weights,
                  FUNC func, T max_err) {
  TestGradient(grad.embedder_input_embedding,
                 c_weights.embedder_input_embedding,
                 func,  2 * max_err, max_err, __LINE__);
  TestGradient(grad.final_norm_scale, c_weights.final_norm_scale,
               func, max_err, max_err, __LINE__);
  for (int i = 0; i < TConfig::kLayers; ++i) {
    TestGradient(*grad.GetLayer(i), *c_weights.GetLayer(i), func, max_err);
  }
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_TEST_UTIL_H_
