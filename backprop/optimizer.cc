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

#include "compression/compress.h"
#include "gemma/common.h"
#include "gemma/weights.h"
#include "util/allocator.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

namespace {

class AdamUpdater {
 public:
  explicit AdamUpdater(float alpha, float beta1, float beta2, float epsilon,
                       size_t t)
      : alpha_(alpha), beta1_(beta1), beta2_(beta2), cbeta1_(1.0f - beta1),
        cbeta2_(1.0f - beta2), norm1_(1.0 / (1.0 - std::pow(beta1, t))),
        norm2_(1.0 / (1.0 - std::pow(beta2, t))), epsilon_(epsilon) {}

  void operator()(const char* name, const MatPtr& grad, MatPtr& weights,
                  MatPtr& grad_m, MatPtr& grad_v) {
    const float* HWY_RESTRICT g = grad.data<float>();
    float* HWY_RESTRICT w = weights.data<float>();
    float* HWY_RESTRICT m = grad_m.data<float>();
    float* HWY_RESTRICT v = grad_v.data<float>();
    for (size_t i = 0; i < grad.NumElements(); ++i) {
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

void AdamUpdate(ModelWeightsPtrs<float>* grad, float alpha, float beta1,
                float beta2, float epsilon, size_t t,
                ModelWeightsPtrs<float>* weights,
                ModelWeightsPtrs<float>* grad_m,
                ModelWeightsPtrs<float>* grad_v, hwy::ThreadPool& pool) {
  AdamUpdater updater(alpha, beta1, beta2, epsilon, t);
  ModelWeightsPtrs<float>::ForEachTensor(
      {grad, weights, grad_m, grad_v}, ForEachType::kLoadNoToc,
      [&updater](const char* name, hwy::Span<MatPtr*> tensors) {
        updater(name, *tensors[0], *tensors[1], *tensors[2], *tensors[3]);
      });
}

}  // namespace

void AdamUpdate(Type weight_type, const ModelWeightsStorage& grad, float alpha,
                float beta1, float beta2, float epsilon, size_t t,
                const ModelWeightsStorage& weights,
                const ModelWeightsStorage& grad_m,
                const ModelWeightsStorage& grad_v, hwy::ThreadPool& pool) {
  HWY_ASSERT(weight_type == Type::kF32);
  AdamUpdate(grad.GetWeightsOfType<float>(), alpha, beta1, beta2, epsilon, t,
             weights.GetWeightsOfType<float>(),
             grad_m.GetWeightsOfType<float>(), grad_v.GetWeightsOfType<float>(),
             pool);
}

}  // namespace gcpp
