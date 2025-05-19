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

#include "gemma/weights.h"
#include "util/allocator.h"
#include "util/mat.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

namespace {

using MatPtrF = MatPtrT<float>;

// Split into two classes so that ForEachTensor only requires two "other"
// arguments. This is anyway useful for locality, because `grad` only feeds
// into `grad_m` and `grad_v` here.
class AdamUpdateMV {
 public:
  AdamUpdateMV(float beta1, float beta2, size_t t)
      : beta1_(beta1),
        beta2_(beta2),
        cbeta1_(1.0f - beta1),
        cbeta2_(1.0f - beta2),
        norm1_(1.0 / (1.0 - std::pow(beta1, t))),
        norm2_(1.0 / (1.0 - std::pow(beta2, t))) {}

  void operator()(const MatPtrF& grad, MatPtrF& grad_m, MatPtrF& grad_v) {
    for (size_t r = 0; r < grad.Rows(); ++r) {
      const float* HWY_RESTRICT g = grad.Row(r);
      float* HWY_RESTRICT m = grad_m.Row(r);
      float* HWY_RESTRICT v = grad_v.Row(r);
      for (size_t c = 0; c < grad.Cols(); ++c) {
        m[c] *= beta1_;
        m[c] += cbeta1_ * g[c];
        v[c] *= beta2_;
        v[c] += cbeta2_ * g[c] * g[c];
      }
    }
  }

 private:
  float beta1_;
  float beta2_;
  float cbeta1_;
  float cbeta2_;
  float norm1_;
  float norm2_;
};

// Updates `weights` based on the updated `grad_m` and `grad_v` from above.
class AdamUpdateW {
 public:
  AdamUpdateW(float alpha, float beta1, float beta2, float epsilon, size_t t)
      : alpha_(alpha),
        norm1_(1.0 / (1.0 - std::pow(beta1, t))),
        norm2_(1.0 / (1.0 - std::pow(beta2, t))),
        epsilon_(epsilon) {}

  void operator()(MatPtrF& weights, const MatPtrF& grad_m,
                  const MatPtrF& grad_v) {
    for (size_t r = 0; r < weights.Rows(); ++r) {
      float* HWY_RESTRICT w = weights.Row(r);
      const float* HWY_RESTRICT m = grad_m.Row(r);
      const float* HWY_RESTRICT v = grad_v.Row(r);
      for (size_t c = 0; c < weights.Cols(); ++c) {
        const float mhat = m[c] * norm1_;
        const float vhat = v[c] * norm2_;
        w[c] -= alpha_ * mhat / (std::sqrt(vhat) + epsilon_);
      }
    }
  }

 private:
  float alpha_;
  float norm1_;
  float norm2_;
  float epsilon_;
};

void AdamUpdate(ModelWeightsPtrs<float>* grad, float alpha, float beta1,
                float beta2, float epsilon, size_t t,
                ModelWeightsPtrs<float>* weights,
                ModelWeightsPtrs<float>* grad_m,
                ModelWeightsPtrs<float>* grad_v, hwy::ThreadPool& pool) {
  AdamUpdateMV update_mv(beta1, beta2, t);
  grad->ForEachTensor(grad_m, grad_v, [&update_mv](const TensorArgs& t) {
    const MatPtrF grad_f(t.mat);
    MatPtrF grad_m_f(*t.other_mat1);
    MatPtrF grad_v_f(*t.other_mat2);
    update_mv(grad_f, grad_m_f, grad_v_f);
  });

  AdamUpdateW update_w(alpha, beta1, beta2, epsilon, t);
  weights->ForEachTensor(grad_m, grad_v, [&update_w](const TensorArgs& t) {
    MatPtrF weights_f(t.mat);
    const MatPtrF grad_m_f(*t.other_mat1);
    const MatPtrF grad_v_f(*t.other_mat2);
    update_w(weights_f, grad_m_f, grad_v_f);
  });
}

}  // namespace

void AdamUpdate(const WeightsOwner& grad, float alpha, float beta1, float beta2,
                float epsilon, size_t t, const WeightsOwner& weights,
                const WeightsOwner& grad_m, const WeightsOwner& grad_v,
                hwy::ThreadPool& pool) {
  AdamUpdate(grad.GetF32(), alpha, beta1, beta2, epsilon, t, weights.GetF32(),
             grad_m.GetF32(), grad_v.GetF32(), pool);
}

}  // namespace gcpp
