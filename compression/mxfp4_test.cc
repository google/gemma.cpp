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
    if (N * sizeof(T) <= 64) {
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

struct TestMxFp4AllScalesAndOffsets {
  template <typename T, class D>
  HWY_INLINE void operator()(T /*unused*/, D d) {
    using Bits = hwy::If<sizeof(T) == 2, uint16_t, uint32_t>;
    constexpr size_t kBlockSize = MxFp4Stream::kBlockSize;
    constexpr size_t kNumScales = 256;
    constexpr size_t kTotal = kNumScales * kBlockSize;
    const size_t N = hn::Lanes(d);

    auto packed = hwy::AllocateAligned<MxFp4Stream>(kNumScales);
    auto dec = hwy::AllocateAligned<T>(hwy::RoundUpTo(kTotal, N));
    HWY_ASSERT(packed && dec);

    for (size_t s = 0; s < kNumScales; ++s) {
      packed[s].scale_e8m0 = static_cast<uint8_t>(s);
      for (size_t c2 = 0; c2 < 16; ++c2) {
        const uint8_t lo = static_cast<uint8_t>(c2);
        const uint8_t hi = static_cast<uint8_t>(15 - c2);
        packed[s].bytes[c2] = static_cast<uint8_t>(lo | (hi << 4));
      }
    }

    const auto packed_span = MakeConstSpan(packed.get(), kTotal);
    MxFp4Codec::DecompressAndZeroPad(d, packed_span, 0, dec.get(), kTotal);

    static constexpr float kE2M1[16] = {
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    for (size_t s = 0; s < kNumScales; ++s) {
      const float scale_f =
          (s == 0 || s == 0xFF)
              ? 0.0f
              : hwy::BitCastScalar<float>(static_cast<uint32_t>(s) << 23);
      for (size_t c2 = 0; c2 < 16; ++c2) {
        for (size_t half = 0; half < 2; ++half) {
          const size_t nib = (half == 0) ? c2 : (15 - c2);
          const Bits exp_bits = hwy::BitCastScalar<Bits>(
              hwy::ConvertScalarTo<T>(kE2M1[nib] * scale_f));
          const Bits got_bits =
              hwy::BitCastScalar<Bits>(dec[s * kBlockSize + half * 16 + c2]);
          // For BF16, 0.5 * 2^-126 = 2^-127 is subnormal in F32/BF16 and may
          // flush to signed zero (matching hardware VCVTNE2PS2BF16).
          if (sizeof(T) == 2 && s == 1 && (nib == 1 || nib == 9) &&
              got_bits == static_cast<Bits>(exp_bits & 0x8000u)) {
            continue;
          }
          HWY_ASSERT_EQ(exp_bits, got_bits);
        }
      }
    }

    // Verify unaligned packed_ofs, unaligned raw output pointer, zero-padding,
    // and no overwrite past RoundUpTo(num, Lanes(d)).
    constexpr size_t kSubTotal = 4 * kBlockSize;
    constexpr uint8_t kScales[4] = {0, 1, 127, 128};
    auto sub_packed = hwy::AllocateAligned<MxFp4Stream>(4);
    auto ref = hwy::AllocateAligned<T>(hwy::RoundUpTo(kSubTotal, N));
    auto out_buf = hwy::AllocateAligned<T>(kSubTotal + N + 2);
    HWY_ASSERT(sub_packed && ref && out_buf);

    for (size_t b = 0; b < 4; ++b) {
      sub_packed[b] = packed[kScales[b]];
    }
    const auto sub_span = MakeConstSpan(sub_packed.get(), kSubTotal);
    MxFp4Codec::DecompressAndZeroPad(d, sub_span, 0, ref.get(), kSubTotal);

    const T sentinel_val = hwy::ConvertScalarTo<T>(0.12345f);
    const Bits kSentinel = hwy::BitCastScalar<Bits>(sentinel_val);
    const Bits kZero = 0;
    for (size_t ofs : {size_t{0}, size_t{1}, size_t{7}, size_t{16},
                       size_t{31}}) {
      for (size_t num : {size_t{1}, size_t{5}, size_t{17}, size_t{32},
                         size_t{33}, size_t{63}, size_t{65}}) {
        if (ofs + num > kSubTotal) continue;
        const size_t padded = hwy::RoundUpTo(num, N);
        T* unaligned_out = out_buf.get() + 1;
        for (size_t i = 0; i <= padded; ++i) {
          unaligned_out[i] = sentinel_val;
        }
        MxFp4Codec::DecompressAndZeroPad(d, sub_span, ofs, unaligned_out, num);
        for (size_t i = 0; i < num; ++i) {
          HWY_ASSERT_EQ(hwy::BitCastScalar<Bits>(ref[ofs + i]),
                        hwy::BitCastScalar<Bits>(unaligned_out[i]));
        }
        for (size_t i = num; i < padded; ++i) {
          HWY_ASSERT_EQ(kZero, hwy::BitCastScalar<Bits>(unaligned_out[i]));
        }
        HWY_ASSERT_EQ(kSentinel,
                      hwy::BitCastScalar<Bits>(unaligned_out[padded]));
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

void TestAllScalesAndOffsets() {
  hn::ForGEVectors<128, TestMxFp4AllScalesAndOffsets>()(float());
  hn::ForGEVectors<128, TestMxFp4AllScalesAndOffsets>()(BF16());
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
HWY_EXPORT_AND_TEST_P(MxFp4Test, TestAllScalesAndOffsets);
HWY_EXPORT_AND_TEST_P(MxFp4Test, TestAllEncDec);
HWY_AFTER_TEST();
}  // namespace gcpp
#endif
