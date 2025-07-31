// Copyright 2024 Google LLC
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

#include "compression/distortion.h"

#include <stdio.h>

#include "compression/types.h"  // SfpStream::kMax
#include "util/test_util.h"
#include "hwy/nanobenchmark.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"  // HWY_ASSERT_EQ

namespace gcpp {
namespace {

#if !HWY_TEST_STANDALONE
class DistortionTest : public testing::Test {};
#endif

TEST(DistortionTest, TestCascadedSummation) {
  CascadedSummation<double> cs;
  // Example from Priest92. Exact sum is 2.
  const double kHuge = 9007199254740992.0 * hwy::Unpredictable1();  // 2^53
  const double kNeg = -4503599627370495.0 * hwy::Unpredictable1();  // -(2^52-1)
  const double kIn[6] = {kHuge, kHuge - 2.0, kNeg, kNeg, kNeg, kNeg};
  for (double in : kIn) {
    cs.Notify(in);
  }
  HWY_ASSERT_EQ(2.0, cs.Total());
}

// Number of exact and rounded-to-zero matches expectations.
TEST(DistortionTest, TestCounts) {
  // Arbitrary positive/negative original, zero distorted.
  DistortionStats stats;
  for (size_t i = 1; i < 10; ++i) {
    stats.Notify(i / 100.0f, 0.0f);
    stats.Notify(i / -100.0f, 0.0f);
  }
  HWY_ASSERT(stats.NumExact() == 0);
  HWY_ASSERT(stats.NumRoundedToZero() == 18);

  // Add some exact (same):
  size_t num_exact = 0;
  for (float x = 0.0f; x <= 1.5f; x += 0.25f) {
    stats.Notify(x, x);
    stats.Notify(-x, -x);
    num_exact += 2;
  }
  HWY_ASSERT_EQ(num_exact, stats.NumExact());
  HWY_ASSERT(stats.NumRoundedToZero() == 18);  // unchanged
}

// Few large differences are diluted in SNR but not WeightedAverageL1.
TEST(DistortionTest, TestDilution) {
  DistortionStats stats;
  for (size_t i = 0; i < 100; ++i) {
    stats.Notify(0.998f, 0.999f);  // small
  }
  HWY_ASSERT(IsInside(900.0, 1000.0, stats.GeomeanValueDivL1()));
  // All-equal WeightedSum is exact.
  HWY_ASSERT(IsNear(0.001, stats.WeightedAverageL1()));

  // Now add a large difference:
  stats.Notify(SfpStream::kMax - 0.0625f,
               SfpStream::kMax);  // max magnitude, 3-bit mantissa
  // .. WeightedAverageL1 is closer to it.
  HWY_ASSERT(IsInside(0.020, 0.025, stats.WeightedAverageL1()));

  // Add a small and large difference:
  stats.Notify((1.75f - 0.125f) / 1024, 1.75f / 1024);  // small, 2-bit mantissa
  stats.Notify(-SfpStream::kMax + 0.0625f,
               -SfpStream::kMax);  // larger negative
  // .. SNR is still barely affected.
  HWY_ASSERT(IsInside(890.0, 900.0, stats.GeomeanValueDivL1()));
  // .. WeightedAverageL1 is higher after another large error.
  HWY_ASSERT(IsInside(0.030, 0.035, stats.WeightedAverageL1()));

  // With these inputs, none are exact nor round to zero.
  HWY_ASSERT(stats.NumExact() == 0);
  HWY_ASSERT(stats.NumRoundedToZero() == 0);
  HWY_ASSERT_EQ(0.0, stats.SumL1Rounded());
  HWY_ASSERT(IsInside(0.220, 0.23, stats.SumL1()));
}

}  // namespace
}  // namespace gcpp

HWY_TEST_MAIN();
