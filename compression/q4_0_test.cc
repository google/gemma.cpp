// Copyright 2026 Google LLC
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
#define HWY_DISABLED_TARGETS (HWY_SCALAR | HWY_SVE)
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <algorithm>
#include <cmath>

#include "util/test_util.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "compression/q4_0_test.cc"
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/q4_0-inl.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

static constexpr size_t kBlockSize = Q4_0Stream::kBlockSize;

struct TestQuantize {
  template <typename T, class D>
  HWY_INLINE void operator()(T /*unused*/, D d) {
    const size_t total = kBlockSize;
    const hn::ScalableTag<float> df;

    auto in = hwy::AllocateAligned<float>(total);
    auto dec1 = hwy::AllocateAligned<T>(total);
    auto dec2 = hwy::AllocateAligned<T>(total);
    auto dec3 = hwy::AllocateAligned<T>(total);
    auto packed = hwy::AllocateAligned<Q4_0Stream>(Q4_0Stream::PackedEnd(total));
    HWY_ASSERT(in && dec1 && dec2 && dec3 && packed);
    const auto packed_span = MakeSpan(packed.get(), total);

    hwy::RandomState rng;
    float max_abs = 0.0f;
    for (size_t i = 0; i < total; ++i) {
      in[i] = static_cast<float>(RandomGaussian(rng));
      max_abs = std::max(max_abs, std::abs(in[i]));
    }

    // Quantize and dequantize
    Q4_0Codec::Enc(df, in.get(), total, packed_span, 0);
    Q4_0Codec::DecompressAndZeroPad(d, MakeConst(packed_span), 0, dec1.get(), total);

    // Max theoretical quantization error for 4-bit symmetric quant is scale/2.
    // scale is max_val / -8.0f (or max_val / 8.0f).
    // So max error is max_abs / 16.0f.
    // We add a small epsilon for BF16/FP16 conversion errors.
    const float tolerance = (max_abs / 16.0f) + 0.02f;

    for (size_t i = 0; i < total; ++i) {
      const float expected = in[i];
      const float actual = hwy::ConvertScalarTo<float>(dec1[i]);
      EXPECT_NEAR(expected, actual, tolerance)
          << "At index " << i << " expected " << expected << " actual "
          << actual << " (max_abs=" << max_abs << ", tolerance=" << tolerance << ")";
    }

    // Verify Dec2 dequantization path
    const size_t N = hn::Lanes(d);
    if (N <= 16) {
      hn::Vec<D> raw0, raw1;
      // Dec2 decompresses 2 vectors of size N = 2 * N total elements
      // packed_ofs must be a multiple of 2 * N
      Q4_0Codec::Dec2(d, MakeConst(packed_span), 0, raw0, raw1);
      auto dec_dec2 = hwy::AllocateAligned<T>(2 * N);
      hn::StoreU(raw0, d, dec_dec2.get());
      hn::StoreU(raw1, d, dec_dec2.get() + N);

      for (size_t i = 0; i < 2 * N; ++i) {
        const float expected = in[i];
        const float actual = hwy::ConvertScalarTo<float>(dec_dec2[i]);
        EXPECT_NEAR(expected, actual, tolerance)
            << "Dec2 index " << i << " expected " << expected << " actual "
            << actual;
      }
    }
  }
};

void TestQuantizeBF16() { hn::ForGEVectors<128, TestQuantize>()(BF16()); }
void TestQuantizeF32() { hn::ForGEVectors<128, TestQuantize>()(float()); }

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gcpp {
HWY_BEFORE_TEST(Q4_0Test);
HWY_EXPORT_AND_TEST_P(Q4_0Test, TestQuantizeBF16);
HWY_EXPORT_AND_TEST_P(Q4_0Test, TestQuantizeF32);
HWY_AFTER_TEST();
}  // namespace gcpp

#endif
