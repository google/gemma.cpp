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

#ifndef THIRD_PARTY_GEMMA_CPP_COMPRESSION_TEST_UTIL_H_
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_TEST_UTIL_H_

#include <stddef.h>
#include <stdint.h>

#include <cmath>

#include "hwy/base.h"

// IWYU pragma: begin_exports
// copybara:import_next_line:gemma_cpp
#include "compression/distortion.h"
// copybara:import_next_line:gemma_cpp
#include "compression/stats.h"
#include "hwy/tests/test_util.h"  // RandomState
// IWYU pragma: end_exports

namespace gcpp {

// Returns random Gaussian (mean=0, stddev=1/3 similar to expected weights)
// using the central limit theorem. Avoid std::normal_distribution for
// consistent cross-platform output.
HWY_INLINE double RandomGaussian(hwy::RandomState& rng) {
  uint64_t sum = 0;
  constexpr int kReps = 40;
  for (int rep = 0; rep < kReps; ++rep) {
    sum += hwy::Random32(&rng) & 0xFFFFF;
  }
  const double sum_f =
      static_cast<double>(sum) / static_cast<double>(0xFFFFF * kReps);
  HWY_ASSERT(0.0 <= sum_f && sum_f <= 1.0);
  const double plus_minus_1 = 2.0 * sum_f - 1.0;
  HWY_ASSERT(-1.0 <= plus_minus_1 && plus_minus_1 <= 1.0);
  // Normalize by stddev of sum of uniform random scaled to [-1, 1].
  return plus_minus_1 * std::sqrt(kReps / 3.0);
};

HWY_INLINE void VerifyGaussian(Stats& stats) {
  const double stddev = stats.StandardDeviation();
  HWY_ASSERT(-0.01 <= stats.Mean() && stats.Mean() <= 0.01);
  HWY_ASSERT(0.30 <= stddev && stddev <= 0.35);
  HWY_ASSERT(-1.1 <= stats.Min() && stats.Min() <= -0.9);
  HWY_ASSERT(0.9 <= stats.Max() && stats.Max() <= 1.1);
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_TEST_UTIL_H_
