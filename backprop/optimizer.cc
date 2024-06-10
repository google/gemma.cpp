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
#include "gemma/weights.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

namespace {

class WeightInitializer {
 public:
  WeightInitializer(std::mt19937& gen) : dist_(0.0f, 1.0f), gen_(gen) {}

  template <size_t N>
  void operator()(const char* name, CompressedArray<float, N>& tensor) {
    float* data = tensor.data();
    for (size_t i = 0; i < N; ++i) {
      data[i] = dist_(gen_);
    }
    tensor.set_scale(1.0f);
  }
 private:
  std::normal_distribution<float> dist_;
  std::mt19937& gen_;
};

template <typename TConfig>
struct RandInitWeightsT {
  void operator()(const ByteStorageT& weights_u8, hwy::ThreadPool& pool,
                  std::mt19937& gen) const {
    auto& weights =
        *reinterpret_cast<CompressedWeights<TConfig>*>(weights_u8.get());
    // TODO(szabadka) Use the same weight initialization method as in the python
    // version.
    WeightInitializer init(gen);
    ForEachTensor1<TConfig>(init, weights);
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
  void operator()(const char* name,
                  const CompressedArray<float, kCapacity>& grad,
                  CompressedArray<float, kCapacity>& weights,
                  CompressedArray<float, kCapacity>& grad_m,
                  CompressedArray<float, kCapacity>& grad_v) {
    const float* HWY_RESTRICT g = grad.data();
    float* HWY_RESTRICT w = weights.data();
    float* HWY_RESTRICT m = grad_m.data();
    float* HWY_RESTRICT v = grad_v.data();
    for (size_t i = 0; i < kCapacity; ++i) {
      m[i] *= beta1_;
      m[i] += cbeta1_ * g[i];
      v[i] *= beta2_;
      v[i] += cbeta2_ * g[i] * g[i];
      const float mhat = m[i] * norm1_;
      const float vhat = v[i] * norm2_;
      w[i] -= alpha_ * mhat / (std::sqrt(vhat) + epsilon_);
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
    using TWeights = CompressedWeights<TConfig>;
    const auto& grad = *reinterpret_cast<const TWeights*>(grad_u8.get());
    auto& weights = *reinterpret_cast<TWeights*>(weights_u8.get());
    auto& grad_m = *reinterpret_cast<TWeights*>(grad_m_u8.get());
    auto& grad_v = *reinterpret_cast<TWeights*>(grad_v_u8.get());
    AdamUpdater updater(alpha, beta1, beta2, epsilon, t);
    ForEachTensor4<TConfig>(updater, grad, weights, grad_m, grad_v);
  }
};

}  // namespace

void RandInitWeights(Model model_type, Type weight_type,
                     const ByteStorageT& weights, hwy::ThreadPool& pool,
                     std::mt19937& gen) {
  HWY_ASSERT(weight_type == Type::kF32);
  CallForModel<float, RandInitWeightsT>(model_type, weights, pool, gen);
}

void AdamUpdate(Model model_type, Type weight_type, const ByteStorageT& grad,
                float alpha, float beta1, float beta2, float epsilon, size_t t,
                const ByteStorageT& weights, const ByteStorageT& grad_m,
                const ByteStorageT& grad_v, hwy::ThreadPool& pool) {
  HWY_ASSERT(weight_type == Type::kF32);
  CallForModel<float, AdamUpdateT>(model_type, grad, alpha, beta1, beta2,
                                   epsilon, t, weights, grad_m, grad_v, pool);
}

}  // namespace gcpp
