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

#include "backprop/optimizer.h"

#include <cmath>
#include <random>

#include "gemma/common.h"
#include "gemma/configs.h"
#include "gemma/weights.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

namespace {

class WeightInitializer {
 public:
  WeightInitializer(std::mt19937& gen) : dist_(0.0f, 1.0f), gen_(gen) {}

  template <size_t N>
  void operator()(const char* name, std::array<float, N>& tensor) {
    for (size_t i = 0; i < N; ++i) {
      tensor[i] = dist_(gen_);
    }
  }
 private:
  std::normal_distribution<float> dist_;
  std::mt19937& gen_;
};

template <typename TConfig>
struct RandInitWeightsT {
  void operator()(const ByteStorageT& weights_u8, hwy::ThreadPool& pool,
                  std::mt19937& gen) const {
    auto& weights = *reinterpret_cast<WeightsF<TConfig>*>(weights_u8.get());
    // TODO(szabadka) Use the same weight initialization method as in the python
    // version.
    WeightInitializer init(gen);
    ForEachTensor1<float, TConfig>(init, weights);
  }
};

class AdamUpdater {
 public:
  explicit AdamUpdater(float alpha, float beta1, float beta2, float epsilon,
                       size_t t)
      : alpha_(alpha), beta1_(beta1), beta2_(beta2), cbeta1_(1.0f - beta1),
        cbeta2_(1.0f - beta2), norm1_(1.0 / (1.0 - std::pow(beta1, t))),
        norm2_(1.0 / (1.0 - std::pow(beta2, t))), epsilon_(epsilon) {}

  template <size_t kCapacity>
  void operator()(const char* name, const std::array<float, kCapacity>& grad,
                  std::array<float, kCapacity>& weights,
                  std::array<float, kCapacity>& grad_m,
                  std::array<float, kCapacity>& grad_v) {
    for (size_t i = 0; i < kCapacity; ++i) {
      grad_m[i] *= beta1_;
      grad_m[i] += cbeta1_ * grad[i];
      grad_v[i] *= beta2_;
      grad_v[i] += cbeta2_ * grad[i] * grad[i];
      const float mhat = grad_m[i] * norm1_;
      const float vhat = grad_v[i] * norm2_;
      weights[i] -= alpha_ * mhat / (std::sqrt(vhat) + epsilon_);
    }
  }

 private:
  float alpha_;
  float beta1_;
  float beta2_;
  float cbeta1_;
  float cbeta2_;
  float norm1_;
  float norm2_;
  float epsilon_;
};

template <typename TConfig>
struct AdamUpdateT {
  void operator()(const ByteStorageT& grad_u8, float alpha, float beta1,
                  float beta2, float epsilon, size_t t,
                  const ByteStorageT& weights_u8, const ByteStorageT& grad_m_u8,
                  const ByteStorageT& grad_v_u8, hwy::ThreadPool& pool) const {
    const auto& grad =
        *reinterpret_cast<const WeightsF<TConfig>*>(grad_u8.get());
    auto& weights = *reinterpret_cast<WeightsF<TConfig>*>(weights_u8.get());
    auto& grad_m = *reinterpret_cast<WeightsF<TConfig>*>(grad_m_u8.get());
    auto& grad_v = *reinterpret_cast<WeightsF<TConfig>*>(grad_v_u8.get());
    AdamUpdater updater(alpha, beta1, beta2, epsilon, t);
    ForEachTensor4<float, TConfig>(updater, grad, weights, grad_m, grad_v);
  }
};

}  // namespace

void RandInitWeights(Model model, const ByteStorageT& weights,
                     hwy::ThreadPool& pool,
                     std::mt19937& gen) {
  CallFunctorForModel<RandInitWeightsT>(model, weights, pool, gen);
}

void AdamUpdate(Model model, const ByteStorageT& grad, float alpha, float beta1,
                float beta2, float epsilon, size_t t,
                const ByteStorageT& weights, const ByteStorageT& grad_m,
                const ByteStorageT& grad_v, hwy::ThreadPool& pool) {
  CallFunctorForModel<AdamUpdateT>(model, grad, alpha, beta1, beta2, epsilon, t,
                                   weights, grad_m, grad_v, pool);
}

}  // namespace gcpp
