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

#ifndef HWY_DISABLED_TARGETS
// Exclude HWY_SCALAR due to 2x bf16 -> f32.
#define HWY_DISABLED_TARGETS HWY_SCALAR
#endif

#include <stddef.h>
#include <stdio.h>

#include <algorithm>  // std::swap
#include <array>
#include <cmath>
#include <random>

#include "util/allocator.h"
#include "util/threading.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/stats.h"
#include "hwy/timer.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ops/dot_test.cc"
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"
#include "ops/dot-inl.h"
#include "hwy/profiler.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

using Array = hwy::AlignedFreeUniquePtr<float[]>;

// Returns normalized value in [-1, 1).
float RandomFloat(std::mt19937& rng) {
  const uint32_t exp = hwy::BitCastScalar<uint32_t>(1.0f);
  const uint32_t mantissa_mask = hwy::MantissaMask<float>();
  const uint32_t representation = exp | (rng() & mantissa_mask);
  const float f12 = hwy::BitCastScalar<float>(representation);
  HWY_DASSERT(1.0f <= f12 && f12 < 2.0f);  // exponent is 2^0, only mantissa
  const float f = (2.0f * (f12 - 1.0f)) - 1.0f;
  HWY_DASSERT(-1.0f <= f && f < 1.0f);
  return f;
}

// Based on Algorithm 6.1 from "Accurate Sum and Dot Product".
// `num` is the size of a, b[, and buf] and must be larger than 2 and even.
void GenerateIllConditionedInputs(double target_cond, size_t num,
                                  float* HWY_RESTRICT a, float* HWY_RESTRICT b,
                                  double* HWY_RESTRICT buf, std::mt19937& rng) {
  PROFILER_FUNC;
  HWY_ASSERT(target_cond >= 1.0);
  HWY_ASSERT(num % 2 == 0);
  const size_t half = num / 2;
  const hn::ScalableTag<float> df;

  const int max_exp = static_cast<int>(std::log2(target_cond) / 2.0);
  std::uniform_int_distribution<int> e_dist(0, max_exp);

  // First half: random exponents and mantissas
  for (size_t i = 0; i < half; ++i) {
    // Ensure the min and max exponents are used.
    const int e = i == 0 ? 0 : i == 1 ? max_exp : e_dist(rng);
    a[i] = RandomFloat(rng) * (1 << e);
    b[i] = RandomFloat(rng) * (1 << e);
  }

  // Zero-init second half for DotExact
  for (size_t i = half; i < num; ++i) {
    a[i] = 0.0f;
    b[i] = 0.0f;
  }

  const float a_exp_step = max_exp / (half - 1);
  float a_exp = max_exp;  // max_exp downto 0
  for (size_t i = half; i < num; ++i, a_exp -= a_exp_step) {
    const int e = static_cast<int>(a_exp);
    HWY_DASSERT(e >= 0);
    a[i] = RandomFloat(rng) * (1 << e);
    const float r = RandomFloat(rng) * (1 << e);
    if (a[i] == 0.0f) {
      b[i] = 0.0f;
    } else {
      // This is called >100K times. CompensatedDot is much faster than ExactDot
      // and just about as accurate, but requires multiples of two vectors.
      // const float exact = ExactDot(a, b, i, buf);
      (void)buf;
      const size_t padded = hwy::RoundUpTo(i, 2 * hn::Lanes(df));
      const float exact = CompensatedDot(df, a, /*w_ofs=*/0, b, padded);
      b[i] = r - exact / a[i];
    }
  }

  // Fisher-Yates shuffle of both a and b simultaneously - std::shuffle only
  // shuffles one array, and we want the same permutation for both.
  for (size_t i = num - 1; i != 0; --i) {
    std::uniform_int_distribution<size_t> dist(0, i);
    const size_t j = dist(rng);

    std::swap(a[i], a[j]);
    std::swap(b[i], b[j]);
  }
}

template <typename T, size_t kNum>
void PrintStats(const char* caption, const std::array<T, kNum>& values) {
  hwy::Stats stats;
  for (T t : values) {
    stats.Notify(static_cast<float>(t));
  }
  fprintf(stderr, "%s %s\n", caption, stats.ToString().c_str());
}

void TestAllDot() {
  // Skip EMU128 and old x86, include SSE4 because it tests the non-FMA path.
  if (HWY_TARGET == HWY_EMU128 || HWY_TARGET == HWY_SSSE3 ||
      HWY_TARGET == HWY_SSE2) {
    return;
  }

  hn::ScalableTag<float> df;

  constexpr size_t kMaxThreads = 8;
  std::mt19937 rngs[kMaxThreads];
  for (size_t i = 0; i < kMaxThreads; ++i) {
    rngs[i].seed(12345 + 65537 * i);
  }

  constexpr size_t kReps = hn::AdjustedReps(200);
  const size_t num = 24 * 1024;
  PerClusterPools pools(/*max_clusters=*/1, kMaxThreads, /*pin=*/1);
  RowVectorBatch<float> a(kMaxThreads, num);
  RowVectorBatch<float> b(kMaxThreads, num);
  RowVectorBatch<double> bufs(kMaxThreads, num);

  const double target_cond = 1e12;
  std::array<double, kReps> conds;
  std::array<uint32_t, kReps> ulps_fast;
  std::array<uint32_t, kReps> ulps_comp;
  std::array<double, kReps> t_fast;
  std::array<double, kReps> t_comp;

  constexpr size_t kTimeReps = 3;

  pools.Inner(0).Run(0, kReps, [&](const uint32_t rep, size_t thread) {
    float* HWY_RESTRICT pa = a.Batch(thread);
    float* HWY_RESTRICT pb = b.Batch(thread);
    double* HWY_RESTRICT buf = bufs.Batch(thread);
    GenerateIllConditionedInputs(target_cond, num, pa, pb, buf, rngs[thread]);
    conds[rep] = ConditionNumber(df, pa, pb, num);

    const float dot_exact = ExactDot(pa, pb, num, buf);

    float dot_fast = 0.0f;
    float dot_comp = 0.0f;

    double elapsed = hwy::HighestValue<double>();
    for (int rep = 0; rep < kTimeReps; ++rep) {
      const double start = hwy::platform::Now();
      dot_fast += SimpleDot(df, pa, 0, pb, num);
      elapsed = HWY_MIN(elapsed, hwy::platform::Now() - start);
    }
    dot_fast /= kTimeReps;
    t_fast[rep] = elapsed;

    elapsed = hwy::HighestValue<double>();
    for (size_t r = 0; r < kTimeReps; ++r) {
      const double start = hwy::platform::Now();
      dot_comp += CompensatedDot(df, pa, /*w_ofs=*/0, pb, num);
      elapsed = HWY_MIN(elapsed, hwy::platform::Now() - start);
    }
    dot_comp /= kTimeReps;
    t_comp[rep] = elapsed;

    ulps_fast[rep] = hwy::detail::ComputeUlpDelta(dot_fast, dot_exact);
    ulps_comp[rep] = hwy::detail::ComputeUlpDelta(dot_comp, dot_exact);
    fprintf(stderr, "cond %.1E: %15.7E %15.7E %15.7E ulp %5u %1u\n", conds[rep],
            dot_exact, dot_fast, dot_comp, ulps_fast[rep], ulps_comp[rep]);
  });

  PROFILER_PRINT_RESULTS();
  PrintStats("cond", conds);
  PrintStats("ulp fast", ulps_fast);
  PrintStats("ulp comp", ulps_comp);
  PrintStats("t fast", t_fast);
  PrintStats("t comp", t_comp);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gcpp {
HWY_BEFORE_TEST(DotTest);
HWY_EXPORT_AND_TEST_P(DotTest, TestAllDot);
HWY_AFTER_TEST();

}  // namespace gcpp

#endif
