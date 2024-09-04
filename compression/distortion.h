// Copyright 2023 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_GEMMA_CPP_COMPRESSION_DISTORTION_H_
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_DISTORTION_H_

#include <math.h>  // pow
#include <stddef.h>
#include <stdio.h>

#include <vector>

#include "hwy/aligned_allocator.h"  // HWY_ALIGNMENT
#include "hwy/base.h"               // ScalarAbs
#include "hwy/contrib/sort/vqsort.h"
#include "hwy/stats.h"

namespace gcpp {

// Returns `sum` and `err` such that `sum + err` is exactly equal to `a + b`,
// despite floating-point rounding. `sum` is already the best estimate for the
// addition, so do not directly add `err` to it.
//
// Knuth98/Moller65. Unlike FastTwoSum, this does not require any relative
// ordering of the exponents of a and b. 6 ops.
// TODO: move to and use in Highway stats.h?
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

// Summarizes the error of a distortion (e.g. quantization) applied to a series
// of numbers.
// Users should check all four resulting metrics (NumExact, NumRoundedToZero,
// GeomeanValueDivL1, WeightedAverageL1) because each covers different aspects.
class DistortionStats {
 public:
  void Notify(float original, float distorted) {
    (void)padding_;  // prevent unused member warning

    const bool rounded_to_zero = (original != 0.0f) && (distorted == 0.0f);
    // We expect original == 0 is not distorted (can be exactly represented).
    HWY_ASSERT(original != 0.0f || distorted == 0.0f);

    s_original_.Notify(original);
    const float l1f = hwy::ScalarAbs(original - distorted);
    const double l1 = static_cast<double>(l1f);
    s_l1_.Notify(l1f);
    if (l1f != 0.0f) {
      l1_.push_back(l1f);
    }
    sum_l1_.Notify(l1f);
    if (rounded_to_zero) sum_l1_rounded_.Notify(l1f);

    // Event counts
    {
      n_ += 1;
      // Rounding (small) negative numbers to 0 does not influence dot products
      // as much as an actual sign flip, so do not count them.
      n_sign_flip_ +=
          ((original < 0.0f) != (distorted < 0.0f)) && !rounded_to_zero;
      n_exact_ += (original == distorted);
      n_rounded_to_zero += rounded_to_zero;
    }

    // Signal to noise ratio (Shannon's channel capacity, NOT the L2-based and
    // logarithmic PSNR) to estimate the ratios of original to the L1 norm.
    if (l1f != 0.0) {  // prevent division by zero
      const double snr =
          1.0 + static_cast<double>(hwy::ScalarAbs(original)) / l1;
      // For numerical purposes (prevents overflow). A hierarchical geomean
      // could also work, but that is more complex and not necessarily better.
      // We will return exp() of the arithmetic mean.
      sum_log_snr_ += log(snr);
      num_snr_ += 1;
    }
  }

  void Assimilate(const DistortionStats& other) {
    s_original_.Assimilate(other.s_original_);
    s_l1_.Assimilate(other.s_l1_);
    sum_l1_.Assimilate(other.sum_l1_);
    sum_l1_rounded_.Assimilate(other.sum_l1_rounded_);
    l1_.insert(l1_.end(), other.l1_.begin(), other.l1_.end());

    n_ += other.n_;
    n_sign_flip_ += other.n_sign_flip_;
    n_exact_ += other.n_exact_;
    n_rounded_to_zero += other.n_rounded_to_zero;

    sum_log_snr_ += other.sum_log_snr_;
    num_snr_ += other.num_snr_;
  }

  size_t NumExact() const { return n_exact_; }
  size_t NumSignFlip() const { return n_sign_flip_; }
  size_t NumRoundedToZero() const { return n_rounded_to_zero; }
  // Total absolute error.
  double SumL1() const { return sum_l1_.Total(); }
  // Total absolute error for numbers that were rounded to zero.
  double SumL1Rounded() const { return sum_l1_rounded_.Total(); }

  // Returns geomean of 1 + S/N (Shannon channel capacity). This is computed via
  // the ratio of input magnitude to nonzero L1 norms. Higher is better.
  double GeomeanValueDivL1() const {
    if (num_snr_ == 0) return 0.0;
    return exp(sum_log_snr_ / static_cast<double>(num_snr_));
  }

  // Returns weighted average of nonzero L1 norms. Those further from the median
  // L1 norm are much more heavily weighted, such that this behaves more like
  // the L-infinity norm, but still includes all differences, not just the max.
  // Lower is better, magnitude depends on the input magnitude.
  double WeightedAverageL1() const {
    if (l1_.empty()) return 0.0f;  // all exact

    std::vector<float> weights(l1_);  // copy so we can modify
    const float median = [&weights]() {
      const size_t mid = weights.size() / 2;
      // We just want the median; partial sort is faster.
      hwy::VQSelect(weights.data(), weights.size(), mid, hwy::SortAscending());
      return weights[mid];
    }();
    weights = l1_;  // restore original order

    // Replace with distance from median (might have too few samples for mode).
    float max_abs = -1.0f;
    for (float& d : weights) {
      d = hwy::ScalarAbs(d - median);
      max_abs = HWY_MAX(max_abs, d);
    }
    HWY_ASSERT(max_abs >= 0.0f);
    // All equal - return the distance value to prevent division by zero.
    if (max_abs == 0.0f) return median;

    // Normalize to max difference and exponentiate.
    const double inv_max = 1.0 / static_cast<double>(max_abs);
    double sum_weights = 0.0;
    for (float& w : weights) {
      const double normalized = static_cast<double>(w) * inv_max;
      const double amplified = exp(4.0 * normalized * normalized);
      sum_weights += amplified;
      w = static_cast<float>(amplified);
    }
    // At least 1.0 per weight, plus more for at least one weight because we
    // verified via max_abs that not all are equal.
    HWY_ASSERT(sum_weights > static_cast<double>(weights.size()));

    // Return weighted average.
    double weighted_sum = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
      weighted_sum += l1_[i] * weights[i];
    }
    return weighted_sum / sum_weights;
  }

  hwy::Stats& L1() { return s_l1_; }
  hwy::Stats& Original() { return s_original_; }

 private:
  hwy::Stats s_original_;
  hwy::Stats s_l1_;
  CascadedSummation<double> sum_l1_;          // all
  CascadedSummation<double> sum_l1_rounded_;  // only if rounded_to_zero
  std::vector<float> l1_;

  // Event counts
  size_t n_ = 0;
  size_t n_sign_flip_ = 0;
  size_t n_exact_ = 0;
  size_t n_rounded_to_zero = 0;

  double sum_log_snr_ = 0.0;
  size_t num_snr_ = 0;

  uint8_t padding_[HWY_ALIGNMENT];  // prevents false sharing
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_DISTORTION_H_
