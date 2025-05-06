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

#include <stddef.h>

#include <cmath>
#include <complex>

#include "gtest/gtest.h"
#include "gemma/configs.h"
#include "gemma/weights.h"
#include "util/mat.h"
#include "util/threading_context.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

template <typename T, typename U>
void Complexify(const MatPtrT<T>& x, MatPtrT<std::complex<U>>& c_x) {
  for (size_t r = 0; r < x.Rows(); ++r) {
    const T* row = x.Row(r);
    std::complex<U>* c_row = c_x.Row(r);
    for (size_t c = 0; c < x.Cols(); ++c) {
      c_row[c] = std::complex<U>(row[c], 0.0);
    }
  }
}

template <typename T, typename U>
void Complexify(const LayerWeightsPtrs<T>& w, LayerWeightsPtrs<U>& c_w) {
  Complexify(w.pre_attention_norm_scale, c_w.pre_attention_norm_scale);
  Complexify(w.attn_vec_einsum_w, c_w.attn_vec_einsum_w);
  Complexify(w.qkv_einsum_w, c_w.qkv_einsum_w);
  Complexify(w.pre_ffw_norm_scale, c_w.pre_ffw_norm_scale);
  Complexify(w.gating_einsum_w, c_w.gating_einsum_w);
  Complexify(w.linear_w, c_w.linear_w);
}

template <typename T, typename U>
void Complexify(const ModelWeightsPtrs<T>& w, ModelWeightsPtrs<U>& c_w) {
  const size_t kLayers = w.c_layers.size();
  Complexify(w.embedder_input_embedding, c_w.embedder_input_embedding);
  Complexify(w.final_norm_scale, c_w.final_norm_scale);
  for (size_t i = 0; i < kLayers; ++i) {
    Complexify(*w.GetLayer(i), *c_w.GetLayer(i));
  }
}

// Somewhat duplicates `WeightsOwner`, but that has neither double nor
// complex types allowed and it would cause code bloat to add them there.
template <typename T>
class WeightsWrapper {
 public:
  explicit WeightsWrapper(const ModelConfig& config) : weights_(config) {
    hwy::ThreadPool& pool = ThreadingContext2::Get().pools.Pool();
    weights_.AllocateForTest(owners_, pool);
  }

  const ModelWeightsPtrs<T>& get() const { return weights_; }
  ModelWeightsPtrs<T>& get() { return weights_; }

 private:
  MatOwners owners_;
  ModelWeightsPtrs<T> weights_;
};

template <typename T, typename U>
void TestNear(const MatPtrT<T>& actual, const MatPtrT<U>& expected,
              double max_abs_err, double max_rel_err, int line_test,
              int line_util) {
  // TODO: consider compensated sum.
  double sum0 = 0;
  double sum1 = 0;
  double sum01 = 0;
  for (size_t r = 0; r < actual.Rows(); ++r) {
    const T* actual_row = actual.Row(r);
    const U* expected_row = expected.Row(r);
    for (size_t c = 0; c < actual.Cols(); ++c) {
      sum0 += actual_row[c] * actual_row[c];
      sum1 += expected_row[c] * expected_row[c];
      sum01 += actual_row[c] * expected_row[c];
      ASSERT_NEAR(
          actual_row[c], expected_row[c],
          std::max(max_abs_err, std::abs(expected_row[c]) * max_rel_err))
          << "test line " << line_test << "test_util.h line " << line_util
          << " r " << r << " c " << c;
    }
  }
  if (sum0 > 1e-16) {
    double norm_dot = sum01 / std::sqrt(sum0) / std::sqrt(sum1);
    ASSERT_NEAR(norm_dot, 1.0, 3e-6)
        << "test line " << line_test << " test_util.h line " << line_util
        << " sum0: " << sum0 << " sum1: " << sum1 << " sum01: " << sum01;
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
template <typename FUNC, typename T, typename U>
void TestGradient(const MatPtrT<T>& grad, MatPtrT<std::complex<U>>& x,
                  FUNC func, U step, T max_abs_err, T max_rel_err,
                  int line_test, int line_util) {
  MatStorageT<T> exp_grad = MakePacked<T>("exp_grad", x.Rows(), x.Cols());
  const U inv_step = 1.0 / step;
  for (size_t r = 0; r < x.Rows(); ++r) {
    std::complex<U>* x_row = x.Row(r);
    T* exp_row = exp_grad.Row(r);
    for (size_t c = 0; c < x.Cols(); ++c) {
      const U x0 = std::real(x_row[c]);
      const std::complex<U> x1 = std::complex<U>(x0, step);
      x_row[c] = x1;
      const std::complex<U> f1 = func();
      exp_row[c] = std::imag(f1) * inv_step;
      x_row[c] = x0;
    }
  }
  TestNear(grad, exp_grad, max_abs_err, max_rel_err, line_test, line_util);
}

template <typename FUNC>
void TestGradient(const MatPtrT<float>& grad, MatPtrT<std::complex<float>>& x,
                  FUNC func, float max_abs_err, float max_rel_error,
                  int line_test, int line_util) {
  TestGradient(grad, x, func, 1e-30f, max_abs_err, max_rel_error, line_test,
               line_util);
}

template <typename FUNC, typename T>
void TestGradient(const MatPtrT<T>& grad, MatPtrT<std::complex<double>>& x,
                  FUNC func, T max_abs_err, T max_rel_error, int line_test,
                  int line_util) {
  TestGradient(grad, x, func, 1e-50, max_abs_err, max_rel_error, line_test,
               line_util);
}

template <typename T, typename U, typename FUNC>
void TestGradient(const LayerWeightsPtrs<T>& grad,
                  LayerWeightsPtrs<U>& c_weights, FUNC func, T max_err,
                  int line_test) {
  TestGradient(grad.pre_attention_norm_scale,
               c_weights.pre_attention_norm_scale, func, max_err, max_err,
               line_test, __LINE__);
  TestGradient(grad.attn_vec_einsum_w, c_weights.attn_vec_einsum_w, func,
               max_err, max_err, line_test, __LINE__);
  TestGradient(grad.qkv_einsum_w, c_weights.qkv_einsum_w, func, max_err,
               max_err, line_test, __LINE__);
  TestGradient(grad.pre_ffw_norm_scale, c_weights.pre_ffw_norm_scale, func,
               max_err, max_err, line_test, __LINE__);
  TestGradient(grad.gating_einsum_w, c_weights.gating_einsum_w, func, max_err,
               max_err, line_test, __LINE__);
  TestGradient(grad.linear_w, c_weights.linear_w, func, max_err, max_err,
               line_test, __LINE__);
}

template <typename T, typename U, typename FUNC>
void TestGradient(const ModelWeightsPtrs<T>& grad,
                  ModelWeightsPtrs<U>& c_weights, FUNC func, T max_err,
                  int line_test) {
  TestGradient(grad.embedder_input_embedding,
               c_weights.embedder_input_embedding, func, 2 * max_err, max_err,
               line_test, __LINE__);
  TestGradient(grad.final_norm_scale, c_weights.final_norm_scale, func, max_err,
               max_err, line_test, __LINE__);
  for (size_t i = 0; i < grad.c_layers.size(); ++i) {
    TestGradient(*grad.GetLayer(i), *c_weights.GetLayer(i), func, max_err,
                 line_test);
  }
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_TEST_UTIL_H_
