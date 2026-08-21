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

#ifndef THIRD_PARTY_GEMMA_CPP_OPS_FP_ARITH_H_
#define THIRD_PARTY_GEMMA_CPP_OPS_FP_ARITH_H_

#include "hwy/base.h"

// Non-SIMD building blocks for floating-point arithmetic.

namespace gcpp {

// Returns `sum` and `err` such that `sum + err` is exactly equal to `a + b`,
// despite floating-point rounding. `sum` is already the best estimate for the
// addition, so do not directly add `err` to it.
//
// Knuth98/Moller65. Unlike FastTwoSum, this does not require any relative
// ordering of the exponents of a and b. 6 ops.
template <typename T, HWY_IF_FLOAT3264(T)>
static inline T TwoSum(T a, T b, T& err) {
  const T sum = a + b;
  const T a2 = sum - b;
  const T b2 = sum - a2;
  const T err_a = a - a2;
  const T err_b = b - b2;
  err = err_a + err_b;
  return sum;
}

// Accumulates numbers with about twice the precision of T using 7 * n FLOPS.
// Rump/Ogita/Oishi08, Algorithm 6.11 in Handbook of Floating-Point Arithmetic.
template <typename T>
class CascadedSummation {
 public:
  void Notify(T t) {
    T err;
    sum_ = TwoSum(sum_, t, err);
    sum_err_ += err;
  }

  void Assimilate(const CascadedSummation& other) {
    Notify(other.sum_);
    sum_err_ += other.sum_err_;
  }

  // Allows users to observe how much difference the extra precision made.
  T Err() const { return sum_err_; }

  // Returns the sum of all `t` passed to `Notify`.
  T Total() const { return sum_ + sum_err_; }

 private:
  T sum_ = T{0};
  T sum_err_ = T{0};
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_OPS_FP_ARITH_H_
