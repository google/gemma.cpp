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
struct RandInitWeights {
  void operator()(ByteStorageT& weights_u8, hwy::ThreadPool& pool,
                  std::mt19937& gen) const {
    auto& weights = *reinterpret_cast<WeightsF<TConfig>*>(weights_u8.get());
    // TODO(szabadka) Use the same weight initialization method as in the python
    // version.
    WeightInitializer init(gen);
    ForEachTensor1<float, TConfig>(init, weights);
  }
};

class WeightUpdater {
 public:
  explicit WeightUpdater(float lr) : lr_(lr) {}

  template <size_t kCapacity>
  void operator()(const char* name, const std::array<float, kCapacity>& grad,
                  std::array<float, kCapacity>& weights) {
    for (size_t i = 0; i < kCapacity; ++i) {
      weights[i] += lr_ * grad[i];
    }
  }

 private:
  float lr_;
};

template <typename TConfig>
struct UpdateWeightsT {
  void operator()(const ByteStorageT& grad_u8, float scale,
                  ByteStorageT& weights_u8, hwy::ThreadPool& pool) const {
    const auto& grad =
        *reinterpret_cast<const WeightsF<TConfig>*>(grad_u8.get());
    auto& weights = *reinterpret_cast<WeightsF<TConfig>*>(weights_u8.get());
    WeightUpdater updater(scale);
    ForEachTensor2<float, TConfig>(updater, grad, weights);
  }
};

}  // namespace

void UpdateWeights(Model model, const ByteStorageT& grad, float scale,
                   ByteStorageT& weights, hwy::ThreadPool& pool) {
  CallFunctorForModel<UpdateWeightsT>(model, grad, scale, weights, pool);
}

}  // namespace gcpp
