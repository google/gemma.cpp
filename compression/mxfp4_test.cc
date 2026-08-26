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

#include "compression/types.h"
#include "util/basics.h"
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_SCALAR | HWY_SVE)
#endif

#include <stddef.h>
#include <stdint.h>

#include "util/test_util.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util.h"
#include "hwy/timer.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "compression/mxfp4_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/mxfp4-inl.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

struct TestMxFp4Exact {
  template <typename T, class D>
  HWY_INLINE void operator()(T /*unused*/, D d) {
    const size_t total = MxFp4Stream::kBlockSize;
    const hn::ScalableTag<float> df;

    auto in = hwy::AllocateAligned<float>(total);
    auto dec = hwy::AllocateAligned<T>(total);
    auto packed =
        hwy::AllocateAligned<MxFp4Stream>(MxFp4Stream::PackedEnd(total));
    HWY_ASSERT(in && dec && packed);
    const auto packed_span = MakeSpan(packed.get(), total);

    // Exact representable values in e2m1 * 2^0 scale
    const float kExactVals[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                 2.0f, 3.0f, 4.0f, 6.0f};
    for (size_t i = 0; i < total; ++i) {
      float v = kExactVals[i % 8];
      if (i % 2 == 1) v = -v;
      in[i] = v;
    }

    MxFp4Codec::Enc(df, in.get(), total, packed_span, 0);
    MxFp4Codec::DecompressAndZeroPad(d, MakeConst(packed_span), 0, dec.get(),
                                     total);

    for (size_t i = 0; i < total; ++i) {
      const float expected = in[i];
      const float actual = hwy::ConvertScalarTo<float>(dec[i]);
      EXPECT_NEAR(expected, actual, 1e-4f) << "At index " << i;
    }
  }
};

struct TestMxFp4Random {
  template <typename T, class D>
  HWY_INLINE void operator()(T /*unused*/, D d) {
    const size_t total = 2 * MxFp4Stream::kBlockSize;
    const hn::ScalableTag<float> df;

    auto in = hwy::AllocateAligned<float>(total);
    auto dec = hwy::AllocateAligned<T>(total);
    auto packed =
        hwy::AllocateAligned<MxFp4Stream>(MxFp4Stream::PackedEnd(total));
    HWY_ASSERT(in && dec && packed);
    const auto packed_span = MakeSpan(packed.get(), total);

    hwy::RandomState rng;
    for (size_t i = 0; i < total; ++i) {
      in[i] = static_cast<float>(RandomGaussian(rng)) * 2.0f;
    }

    MxFp4Codec::Enc(df, in.get(), total, packed_span, 0);
    MxFp4Codec::DecompressAndZeroPad(d, MakeConst(packed_span), 0, dec.get(),
                                     total);

    // Verify Dec2 dequantization path
    const size_t N = hn::Lanes(d);
    if (N <= 16) {
      hn::Vec<D> raw0, raw1;
      MxFp4Codec::Dec2(d, MakeConst(packed_span), 0, raw0, raw1);
      auto dec_dec2 = hwy::AllocateAligned<T>(2 * N);
      hn::StoreU(raw0, d, dec_dec2.get());
      hn::StoreU(raw1, d, dec_dec2.get() + N);

      for (size_t i = 0; i < 2 * N; ++i) {
        EXPECT_NEAR(hwy::ConvertScalarTo<float>(dec[i]),
                    hwy::ConvertScalarTo<float>(dec_dec2[i]), 1e-5f)
            << "At Dec2 index " << i;
      }
    }
  }
};

struct TestEncDec {
  template <typename T, class D>
  HWY_INLINE void operator()(T /*unused*/, D d) {
    const hn::Repartition<uint8_t, D> du8;
    const hn::ScalableTag<float> df;

    constexpr size_t total =
        4 * 1024 * 1024;  // 4M elements (8 MB BF16 / 16 MB float)
    auto in = hwy::AllocateAligned<float>(total);
    auto dec = hwy::AllocateAligned<T>(total);
    auto packed =
        hwy::AllocateAligned<MxFp4Stream>(MxFp4Stream::PackedEnd(total));
    HWY_ASSERT(in && dec && packed);
    const auto packed_span = MakeSpan(packed.get(), total);

    hwy::RandomState rng;
    for (size_t i = 0; i < total; ++i) {
      in[i] = static_cast<float>(RandomGaussian(rng)) * 2.0f;
    }

    MxFp4Codec::Enc(df, in.get(), total, packed_span, 0);

    // Warm up
    for (size_t w = 0; w < 5; ++w) {
      MxFp4Codec::DecompressAndZeroPad(d, MakeConst(packed_span), 0, dec.get(),
                                       total);
    }

    constexpr size_t kReps = 30;
    double dec_elapsed_min = hwy::HighestValue<double>();
    double dec_elapsed_sum = 0.0;
    for (size_t rep = 0; rep < kReps; ++rep) {
      const double t0 = hwy::platform::Now();
      MxFp4Codec::DecompressAndZeroPad(d, MakeConst(packed_span), 0, dec.get(),
                                       total);
      const double t1 = hwy::platform::Now();
      const double dt = t1 - t0;
      dec_elapsed_min = HWY_MIN(dec_elapsed_min, dt);
      dec_elapsed_sum += dt;
    }
    const double dec_mbs_peak = total * sizeof(T) * 1E-6 / dec_elapsed_min;
    const double dec_mbs_avg =
        total * sizeof(T) * 1E-6 / (dec_elapsed_sum / kReps);
    const char* type_str = hwy::IsSame<T, BF16>() ? "BF16" : "float";
    fprintf(stderr,
            "[%s] Target: %-10s | VecBytes: %2zu | Peak: %8.2f GB/s | Avg: "
            "%8.2f GB/s\n",
            type_str, hwy::TargetName(HWY_TARGET), Lanes(du8),
            dec_mbs_peak * 1e-3, dec_mbs_avg * 1e-3);
  }
};

void TestAllExact() {
  hn::ForGEVectors<128, TestMxFp4Exact>()(float());
  hn::ForGEVectors<128, TestMxFp4Exact>()(BF16());
}

void TestAllRandom() {
  hn::ForGEVectors<128, TestMxFp4Random>()(float());
  hn::ForGEVectors<128, TestMxFp4Random>()(BF16());
}

void TestAllEncDec() {
  hn::ForGEVectors<128, TestEncDec>()(BF16());
  hn::ForGEVectors<128, TestEncDec>()(float());
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {
HWY_BEFORE_TEST(MxFp4Test);
HWY_EXPORT_AND_TEST_P(MxFp4Test, TestAllExact);
HWY_EXPORT_AND_TEST_P(MxFp4Test, TestAllRandom);
HWY_EXPORT_AND_TEST_P(MxFp4Test, TestAllEncDec);
HWY_AFTER_TEST();
}  // namespace gcpp
#endif
