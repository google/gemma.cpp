// Copyright 2025 Google LLC
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

#include "util/basics.h"

#include <stddef.h>
#include <stdio.h>

#include "hwy/base.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/timer.h"

namespace gcpp {
namespace {

TEST(BasicsTest, EngineIsDeterministic) {
  const AesCtrEngine engine1(/*deterministic=*/true);
  const AesCtrEngine engine2(/*deterministic=*/true);
  RngStream rng1(engine1, 0);
  RngStream rng2(engine2, 0);
  // Remember for later testing after resetting the stream.
  const uint64_t r0 = rng1();
  const uint64_t r1 = rng1();
  // Not consecutive values. This could actually happen due to the extra XOR,
  // but given the deterministic seeding here, we know it will not.
  HWY_ASSERT(r0 != r1);
  // Let rng2 catch up.
  HWY_ASSERT(r0 == rng2());
  HWY_ASSERT(r1 == rng2());

  for (size_t i = 0; i < 1000; ++i) {
    HWY_ASSERT(rng1() == rng2());
  }

  // Reset counter, ensure it matches the prior sequence.
  rng1 = RngStream(engine1, 0);
  HWY_ASSERT(r0 == rng1());
  HWY_ASSERT(r1 == rng1());
}

TEST(BasicsTest, EngineIsSeeded) {
  AesCtrEngine engine1(/*deterministic=*/true);
  AesCtrEngine engine2(/*deterministic=*/false);
  RngStream rng1(engine1, 0);
  RngStream rng2(engine2, 0);
  // It would be very unlucky to have even one 64-bit value match, and two are
  // extremely unlikely.
  const uint64_t a0 = rng1();
  const uint64_t a1 = rng1();
  const uint64_t b0 = rng2();
  const uint64_t b1 = rng2();
  HWY_ASSERT(a0 != b0 || a1 != b1);
}

TEST(BasicsTest, StreamsDiffer) {
  AesCtrEngine engine(/*deterministic=*/true);
  // Compare random streams for more coverage than just the first N streams.
  RngStream rng_for_stream(engine, 0);
  for (size_t i = 0; i < 1000; ++i) {
    RngStream rng1(engine, rng_for_stream());
    RngStream rng2(engine, rng_for_stream());
    // It would be very unlucky to have even one 64-bit value match, and two are
    // extremely unlikely.
    const uint64_t a0 = rng1();
    const uint64_t a1 = rng1();
    const uint64_t b0 = rng2();
    const uint64_t b1 = rng2();
    HWY_ASSERT(a0 != b0 || a1 != b1);
  }
}

// If not close to 50% 1-bits, the RNG is quite broken.
TEST(BasicsTest, BitDistribution) {
  AesCtrEngine engine(/*deterministic=*/true);
  RngStream rng(engine, 0);
  constexpr size_t kU64 = 2 * 1000 * 1000;
  const hwy::Timestamp t0;
  uint64_t one_bits = 0;
  for (size_t i = 0; i < kU64; ++i) {
    one_bits += hwy::PopCount(rng());
  }
  const uint64_t total_bits = kU64 * 64;
  const double one_ratio = static_cast<double>(one_bits) / total_bits;
  const double elapsed = hwy::SecondsSince(t0);
  fprintf(stderr, "1-bit ratio %.5f, %.1f M/s\n", one_ratio,
          kU64 / elapsed * 1E-6);
  HWY_ASSERT(0.4999 <= one_ratio && one_ratio <= 0.5001);
}

TEST(BasicsTest, ChiSquared) {
  AesCtrEngine engine(/*deterministic=*/true);
  RngStream rng(engine, 0);
  constexpr size_t kU64 = 1 * 1000 * 1000;

  // Test each byte separately.
  for (size_t shift = 0; shift < 64; shift += 8) {
    size_t counts[256] = {};
    for (size_t i = 0; i < kU64; ++i) {
      const size_t byte = (rng() >> shift) & 0xFF;
      counts[byte]++;
    }

    double chi_squared = 0.0;
    const double expected = static_cast<double>(kU64) / 256.0;
    for (size_t i = 0; i < 256; ++i) {
      const double diff = static_cast<double>(counts[i]) - expected;
      chi_squared += diff * diff / expected;
    }
    // Should be within ~0.5% and 99.5% percentiles. See
    // https://www.medcalc.org/manual/chi-square-table.php
    if (chi_squared < 196.0 || chi_squared > 311.0) {
      HWY_ABORT("Chi-squared byte %zu: %.5f \n", shift / 8, chi_squared);
    }
  }
}

}  // namespace
}  // namespace gcpp
HWY_TEST_MAIN();
