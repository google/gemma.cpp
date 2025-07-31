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

#include "compression/types.h"
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <set>

#include "compression/distortion.h"
#include "util/test_util.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/timer.h"
// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "compression/sfp_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/sfp-inl.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// Decode
float F32FromSFP8(uint32_t sfp) {
  HWY_ASSERT(sfp < 256);
  HWY_ASSERT(sfp != 0x80);  // -0 is reserved

  const uint32_t sign32 = (sfp & 0x80) << 24;
  sfp &= 0x7F;
  const bool large_e = sfp >= 64;
  const size_t m_bits = large_e ? 3 : 2;
  uint32_t m = sfp & ((1u << m_bits) - 1u);
  size_t e = sfp >> m_bits;
  if (sfp == 0) return 0.0f;
  const uint32_t e_bias = large_e ? 15 : 23;
  const uint32_t exp32 = static_cast<uint32_t>(127 + e - e_bias) << 23;
  const uint32_t mnt32 = m << (23 - m_bits);
  const uint32_t binary32 = sign32 | exp32 | mnt32;
  float result;
  hwy::CopySameSize(&binary32, &result);
  return result;
}

// Used for HWY_AVX3_DL and newer.
void PrintTables() {
  if (HWY_ONCE && false) {
    uint8_t hi[128];
    fprintf(stderr, "lo\n");
    for (uint32_t sfp = 0; sfp < 128; ++sfp) {
      const uint32_t u = hwy::BitCastScalar<uint32_t>(F32FromSFP8(sfp));
      // Lower bits are zero, hence we can truncate instead of rounding to bf16.
      HWY_ASSERT((u & 0xFFFF) == 0);
      fprintf(stderr, "0x%02X,", (u >> 16) & 0xFF);
      hi[sfp] = u >> 24;
    }
    fprintf(stderr, "\nhi\n");
    for (uint32_t sfp = 0; sfp < 128; ++sfp) {
      fprintf(stderr, "0x%02X,", hi[sfp]);
    }
    fprintf(stderr, "\n");
  }
}

void TestAllUnique() {
  std::set<float> unique;
  for (uint32_t sfp = 0; sfp < 256; ++sfp) {
    if (sfp == 0x80) continue;  // -0 is reserved
    unique.insert(F32FromSFP8(sfp));
  }
  HWY_ASSERT_EQ(size_t{255}, unique.size());
  if (false) {
    for (float f : unique) {
      fprintf(stderr, "%e\n", f);
    }
  }
}

// For deriving the new shift-based decoder, which is 3 ops faster than the
// previous "assemble from binary32 bits" method.
void TestAllFastDecode() {
  for (size_t sfp = 0; sfp < 128; ++sfp) {
    const float f = F32FromSFP8(sfp);
    const uint32_t u = hwy::BitCastScalar<uint32_t>(f);
    const uint32_t lo = (u >> 16) & 0xFF;
    const uint32_t hi = u >> 24;
    const bool is_small = sfp < 0x40;
    const uint32_t base = is_small ? 0x34 : 0x38;
    const uint32_t fast_lo = (sfp << (is_small ? 5 : 4)) & 0xFF;
    uint32_t fast_hi = base + (sfp >> (is_small ? 3 : 4));
    if (sfp == 0) fast_hi = 0;

    // fprintf(stderr, "sfp %2zx -> %6.3E %x %x\n", sfp, f, lo, hi);
    if (fast_lo != lo || fast_hi != hi) {
      HWY_ABORT(
          "mismatch sfp %2zx -> %6.3E lo %2x fastLo %2x hi %2x fastHi %2x\n",
          sfp, f, lo, fast_lo, hi, fast_hi);
    }
  }
}

// ------------------------------ Foreach compressed representation

// Encode
HWY_INLINE uint32_t SFP8FromF32(float f) {
  HWY_ASSERT(-SfpStream::kMax <= f && f <= SfpStream::kMax);

  constexpr uint32_t kMaskM = hwy::MantissaMask<float>();
  uint32_t binary32;
  hwy::CopySameSize(&f, &binary32);
  const uint32_t s = (binary32 & hwy::SignMask<float>()) >> 24;
  binary32 &= ~hwy::SignMask<float>();
  f = hwy::ScalarAbs(f);

  // >= 1.1111 * 2^-8 rounds up to 1.0*2^-7.
  bool large_e = (f >= 0.007568359375f);

  const uint32_t org_binary32 = binary32;
  const uint32_t m32 = binary32 & kMaskM;
  binary32 = (binary32 & ~kMaskM) | m32;
  size_t m_bits = large_e ? 3 : 2;
  const uint32_t is_odd = (m32 >> (23 - m_bits)) & 1;
  const uint32_t round = is_odd + (1u << (23 - m_bits - 1)) - 1;
  const uint32_t rounded = binary32 + round;

  // >= 1.111 also rounds up, but only if it was considered !large_e before.
  if (f >= 0.00732421875f) {
    large_e = true;
    m_bits = 3;
  }

  uint32_t m = (kMaskM & rounded) >> (23 - m_bits);
  int32_t e = (rounded >> 23) - 127;

  if (e <= -23) {
    // 2^-23 is the smallest normal exponent. Zero has e = -127. Do not set the
    // SFP sign bit because the encoding for -0 is reserved.
    if (e < -23) return 0;
    // e = 2^-23: round up mantissa because m=0 encodes 0.0f.
    if (m == 0) m = 1;
  }

  if (false) {
    fprintf(stderr, "in %x round %x rounded %x e %d m %x large_e %d\n",
            org_binary32, round, rounded, e, m, large_e);
  }
  uint32_t e_sfp = e + (large_e ? 15 : 23);
  HWY_ASSERT(e_sfp < 16);

  const uint32_t encoded = (e_sfp << m_bits) | m | s;
  HWY_ASSERT(encoded < 256);
  return encoded;
}

// For every possible encoding: ensure re-encoding the decoded value matches it.
struct TestDecEnc {
  template <class T, class D>
  HWY_INLINE void operator()(T /*unused*/, D d) {
    const hn::RepartitionToWide<D> d16;
    const hn::Rebind<BF16, decltype(d16)> dbf;
    const hn::Repartition<float, D> df;
    for (uint32_t encoded = 0; encoded < 256; ++encoded) {
      if (encoded == 0x80) continue;  // -0 is reserved
      const float decoded = F32FromSFP8(encoded);
      const uint32_t encoded2 = SFP8FromF32(decoded);

      hn::Vec<D> dec_lo, dec_hi;
      SfpCodec::DecBytes(d, hn::Set(d, encoded), dec_lo, dec_hi);
      const hn::Vec<decltype(dbf)> dec =
          hn::BitCast(dbf, hn::ZipLower(d16, dec_lo, dec_hi));
      const float vdecoded = hn::GetLane(hn::PromoteLowerTo(df, dec));
      const uint32_t vencoded2 =
          hn::GetLane(SfpCodec::EncBytes(d, dec_lo, dec_hi));

      if (decoded != vdecoded || encoded2 != vencoded2 || encoded != encoded2) {
        HWY_ABORT("enc %u -> dec %E=%x=%E -> enc %u %u\n", encoded, decoded,
                  hwy::BitCastScalar<uint32_t>(decoded), vdecoded, encoded2,
                  vencoded2);
      }
    }
  }
};

void TestAllDecEnc() { hn::ForGEVectors<32, TestDecEnc>()(uint8_t()); }

// ------------------------------ Golden (known values)

// Generate values, encode, decode back to that value.
struct TestGolden {
  template <class T, class D>
  HWY_INLINE void operator()(T /*unused*/, D d) {
    const hn::Repartition<float, D> df;
    const hn::Repartition<BF16, D> dbf;
    const hn::RebindToUnsigned<decltype(dbf)> d16;

    struct Golden {
      float in;
      float out;
    };
    const Golden golden[] = {
        // All mantissa bits set, all discarded zero (no rounding)
        {0.46875f, 0.46875f},
        {0.9375f, 0.9375f},
        // All mantissa bits set, one below it set (round up to pow2)
        {0.484375f, 0.5f},
        {0.96875f, 1.0f},
        // Lowest mantissa bit set, all discarded zero (no rounding)
        {0.28125f, 0.28125f},
        {0.5625f, 0.5625f},
        // Lowest mantissa bit set, one below it set (round up to even)
        {0.296875f, 0.3125f},
        {0.59375f, 0.625f},
        // All mantissa zero, all discarded set (round up)
        {0.279296875f, 0.28125f},
        {0.55859375f, 0.5625f},
        // All mantissa zero, one below it set (round DOWN to pow2)
        {0.265625f, 0.25f},
        {0.53125f, 0.5f},

        // At inflection point: 1.max*2^-8 rounds up to 1.0*2^-7
        {0.0068359375f, 0.0068359375f},  // 1.11 -> 1.11
        {0.00732421875f, 0.0078125f},    // 1.111 -> 1.11[1] -> 1.0
        {0.007568359375f, 0.0078125f},   // 1.1111 -> 1.0

        // Above 1.0: no longer special-cased.
        {1.0f, 1.0f},
        {1.0625f, 1.0f},  // 1.000100

        // Smallest normal exponents - we no longer use subnormals.
        {2.384185791015625E-7f, 2.384185791015625E-7f},  // 1.00p-22
        {1.49011611938E-07f, 1.49011611938E-07f},        // 1.01p-23
        {1.19209289551E-07f, 1.49011611938E-07f},        // 1.00p-23 -> 1.01p-23
        {5.96046447754E-08f, 0.0f},                      // 1.00p-24 -> 0
        {8.94069671631E-08f, 0.0f},                      // 1.10p-24 -> 0
        {1.11758708954E-07f, 1.49011611938E-07f},        // 1.111p-24-> 1.01p-23

        // 1100_010 * 2^-7 rounds down to 110
        {0.013841f, 0.013671875f},
    };
    constexpr size_t kNumGolden = sizeof(golden) / sizeof(Golden);
    for (uint32_t s : {0, 1}) {
      for (size_t i = 0; i < kNumGolden; ++i) {
        const float in = s ? -golden[i].in : golden[i].in;
        const float out = s ? -golden[i].out : golden[i].out;
        const hn::Vec<decltype(dbf)> in_bf =
            hn::OrderedDemote2To(dbf, hn::Set(df, in), hn::Set(df, in));
        const uint32_t encoded = SFP8FromF32(in);
        const uint32_t vencoded = hn::GetLane(SfpCodec::EncBytes(
            d, hn::BitCast(d, in_bf),
            hn::BitCast(d, hn::ShiftRight<8>(hn::BitCast(d16, in_bf)))));
        const float decoded = F32FromSFP8(encoded);
        hn::Vec<D> dec_lo, dec_hi;
        SfpCodec::DecBytes(d, hn::Set(d, encoded), dec_lo, dec_hi);
        const hn::Vec<decltype(dbf)> dec =
            hn::BitCast(dbf, hn::ZipLower(d16, dec_lo, dec_hi));
        const float vdecoded = hn::GetLane(hn::PromoteLowerTo(df, dec));

        if (decoded != vdecoded || decoded != out || encoded != vencoded) {
          HWY_ABORT("@%zu in %E dec %E %E golden %E\n", i, in, decoded,
                    vdecoded, golden[i].out);
        }
      }  // i
    }    // s
  }
};

void TestAllGolden() {
  // Full vectors only, other tests cover partial vectors.
  TestGolden()(uint8_t(), hn::ScalableTag<uint8_t>());
}

// ------------------------------ Order

// Store 8-bit iota, decode, encode, check iota == packed. This ensures
// Enc/Dec are preserving the order independent of vector length.
struct TestOrder {
  template <class T, class DBF>
  HWY_INLINE void operator()(T /*unused*/, DBF dbf) {
    const size_t N16 = hn::Lanes(dbf);

    for (size_t num = 1; num < 6 * N16; ++num) {
      const size_t padded = hwy::RoundUpTo(num, N16);

      auto iota = hwy::AllocateAligned<SfpStream>(num);
      auto packed = hwy::AllocateAligned<SfpStream>(num);
      auto bf = hwy::AllocateAligned<BF16>(padded);
      HWY_ASSERT(iota && packed && bf);
      for (size_t i = 0; i < num; ++i) {
        // Clear sign bit so we can also check that bf is in ascending order.
        iota[i].byte = i & 127;
      }

      SfpCodec::DecompressAndZeroPad(dbf, MakeConstSpan(iota.get(), num), 0,
                                     bf.get(), num);
      for (size_t i = num; i < padded; ++i) {
        if (hwy::ConvertScalarTo<float>(bf[i]) != 0.0f) {
          HWY_ABORT("num %zu padded %zu i %zu: not padded", num, padded, i);
        }
      }

      SfpCodec::Enc(dbf, bf.get(), num, packed.get());

      for (size_t i = 0; i < num; ++i) {
        if (iota[i].byte != packed[i].byte) {
          HWY_ABORT("@%zu: %d %d\n", i, iota[i].byte, packed[i].byte);
        }
      }
    }
  }
};

void TestAllOrder() { hn::ForGEVectors<32, TestOrder>()(BF16()); }

// ------------------------------ Foreach bf16 input

// Checks the distortion from an encode and decode round trip. Unlike
// `TestShortLengthsT` in compress_test, this covers large `num` and
// prints the enc/dec throughput.
struct TestEncDec {
  template <class T, class DBF>
  HWY_INLINE void operator()(T /*unused*/, DBF dbf) {
    const hn::Repartition<uint8_t, DBF> du8;

    // We only use the upper 4 of 7 bf16 mantissa bits, so force the lower three
    // bits to zero to reduce the number of inputs.
    constexpr size_t kStep = 8;
    const size_t max = 0x8000 / 8;

    auto in = hwy::AllocateAligned<T>(max);
    auto packed = hwy::AllocateAligned<SfpStream>(max);
    auto dec = hwy::AllocateAligned<T>(max);  // already padded
    HWY_ASSERT(in && packed && dec);
    size_t num = 0;
    for (size_t i = 0; i < max; ++i) {
      const uint16_t bits = i * kStep;
      const float f = hwy::F32FromBF16(hwy::BitCastScalar<T>(bits));
      // Keep if within range
      if (hwy::ScalarIsFinite(f) && f <= SfpStream::kMax) {
        in[num] = hwy::BF16FromF32(f);
        in[num + 1] = hwy::BF16FromF32(-f);
        num += 2;
      }
    }

    double enc_elapsed = hwy::HighestValue<double>();
    double dec_elapsed = hwy::HighestValue<double>();
    for (size_t rep = 0; rep < 100; ++rep) {
      const double t0 = hwy::platform::Now();
      SfpCodec::Enc(dbf, in.get(), num, packed.get());
      const double t1 = hwy::platform::Now();
      SfpCodec::DecompressAndZeroPad(dbf, MakeConstSpan(packed.get(), num), 0,
                                     dec.get(), num);
      const double t2 = hwy::platform::Now();
      enc_elapsed = HWY_MIN(enc_elapsed, t1 - t0);
      dec_elapsed = HWY_MIN(dec_elapsed, t2 - t1);
    }
    const double enc_mbs = num * sizeof(T) * 1E-6 / enc_elapsed;
    const double dec_mbs = num * sizeof(T) * 1E-6 / dec_elapsed;
    fprintf(stderr, "Vec size %zu Enc %.2f MB/s Dec %.2f MB/s\n", Lanes(du8),
            enc_mbs, dec_mbs);

    {
      double sum = 0.0;
      DistortionStats stats;
      for (size_t i = 0; i < num; ++i) {
        const float out = hwy::F32FromBF16(dec[i]);
        sum += hwy::ConvertScalarTo<double>(hwy::ScalarAbs(in[i]));
        stats.Notify(hwy::ConvertScalarTo<float>(in[i]), out);
      }
      const double avg_in = sum / num;
      const double snr = stats.GeomeanValueDivL1();
      const double wl1 = stats.WeightedAverageL1();
      if (false) {
        fprintf(stderr,
                "Num inputs %zu, avg %.3E, exact %zu round0 %zu (sum %E) snr "
                "%.2f wL1 %f\n",
                num, avg_in, stats.NumExact(), stats.NumRoundedToZero(),
                stats.SumL1Rounded(), snr, wl1);
      }
      HWY_ASSERT(stats.Original().Count() == stats.L1().Count());
      // Inputs are in [-SfpStream::kMax, SfpStream::kMax], symmetric, and
      // heavy-tailed.
      HWY_ASSERT(stats.Original().Min() == -SfpStream::kMax);
      HWY_ASSERT(stats.Original().Max() == SfpStream::kMax);
      HWY_ASSERT(gcpp::IsInside(-1E-6, 1E-6, stats.Original().Mean()));
      HWY_ASSERT(gcpp::IsInside(-1E-6, 1E-6, stats.Original().Skewness()));
      HWY_ASSERT(gcpp::IsInside(80.0, 100.0, stats.Original().Kurtosis()));
      // Absolute errors are in [0, 0.0625], and (heavy) right-tailed.
      HWY_ASSERT(stats.L1().Min() == 0.0f);
      HWY_ASSERT(stats.L1().Max() == 0.0625f);
      HWY_ASSERT(gcpp::IsInside(4E-4, 5E-4, stats.L1().Mean()));
      HWY_ASSERT(gcpp::IsInside(10.0, 15.0, stats.L1().Skewness()));
      HWY_ASSERT(gcpp::IsInside(150.0, 200.0, stats.L1().Kurtosis()));
      // SNR is low because many *tiny* numbers are rounded to zero.
      HWY_ASSERT_EQ(3322, stats.NumRoundedToZero());
      HWY_ASSERT(gcpp::IsInside(5E-6, 6E-6, stats.SumL1Rounded()));
      HWY_ASSERT(gcpp::IsInside(1.880, 1.885, stats.SumL1()));
      HWY_ASSERT_EQ(256, stats.NumExact());
      HWY_ASSERT_EQ(0, stats.NumSignFlip());
      HWY_ASSERT(gcpp::IsInside(2.70, 2.75, snr));
      HWY_ASSERT(gcpp::IsInside(0.010, 0.011, wl1));  // = half of mean |x|.
    }
  }
};

void TestAllEncDec() { hn::ForGEVectors<32, TestEncDec>()(BF16()); }

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {
HWY_BEFORE_TEST(SfpTest);
HWY_EXPORT_AND_TEST_P(SfpTest, PrintTables);
HWY_EXPORT_AND_TEST_P(SfpTest, TestAllUnique);
HWY_EXPORT_AND_TEST_P(SfpTest, TestAllFastDecode);
HWY_EXPORT_AND_TEST_P(SfpTest, TestAllDecEnc);
HWY_EXPORT_AND_TEST_P(SfpTest, TestAllGolden);
HWY_EXPORT_AND_TEST_P(SfpTest, TestAllOrder);
HWY_EXPORT_AND_TEST_P(SfpTest, TestAllEncDec);
HWY_AFTER_TEST();
}  // namespace gcpp
#endif  // HWY_ONCE
