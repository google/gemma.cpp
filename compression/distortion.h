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

#include "hwy/base.h"  // ScalarAbs

namespace gcpp {

class DistortionStats {
 public:
  void Notify(float original, float distorted) {
    (void)padding_;  // prevent unused member warning

    const double l1 = hwy::ScalarAbs(original - distorted);

    if (l1 > max_l1_) {
      max_l1_ = l1;
      max_idx_ = n_;
    }

    const double pow3 = l1 * l1 * l1;
    sum_pow3_ += pow3;
    sum_pow6_ += pow3 * pow3;
    n_ += 1;

    // Avoid division by zero, which happens when there is no error. NumExact()
    // reports the number of times this happens.
    if (l1 != 0.0) {
      const double rel = 1.0 + hwy::ScalarAbs(original) / l1;
      // Logarithm is required to prevent overflow. A hierarchical geomean
      // could also work, but that is more complex and not necessarily better.
      sum_log_rel_ += log(rel);
      num_rel_ += 1;
    }
  }

  void Assimilate(const DistortionStats& other) {
    if (other.max_l1_ > max_l1_) {
      max_l1_ = other.max_l1_;
      max_idx_ = other.max_idx_;
    }

    sum_pow3_ += other.sum_pow3_;
    sum_pow6_ += other.sum_pow6_;
    n_ += other.n_;

    sum_log_rel_ += other.sum_log_rel_;
    num_rel_ += other.num_rel_;
  }

  size_t NumExact() const { return n_ - num_rel_; }

  double GeomeanValueDivL1() const {
    if (num_rel_ == 0) return 0.0;
    return exp(sum_log_rel_ / static_cast<double>(num_rel_));
  }

  double PNorm() const {
    // p-norms are a compromise between max-norm (penalizes the largest error
    // without dilution, but does not notice any other errors) and L1 (all
    // errors contribute, but large errors are diluted by smaller ones).
    const double norm3 = pow(sum_pow3_ / static_cast<double>(n_), 1.0 / 3);
    const double norm6 = pow(sum_pow6_ / static_cast<double>(n_), 1.0 / 6);
    return 0.5 * (norm3 + norm6);
  }

  size_t MaxIndex() const { return max_idx_; }
  double MaxL1() const { return max_l1_; }

 private:
  size_t n_ = 0;
  size_t max_idx_ = 0;  // index that had l1 = max_l1_.
  double max_l1_ = -1.0;

  double sum_pow3_ = 0.0;
  double sum_pow6_ = 0.0;

  double sum_log_rel_ = 0.0;
  size_t num_rel_ = 0;
  double padding_;  // prevents false sharing
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_DISTORTION_H_
