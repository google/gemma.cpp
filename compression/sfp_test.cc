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

// We use ConcatEven/Odd which are not supported. Use HWY_EMU128 instead.
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS HWY_SCALAR
#endif

// copybara:import_next_line:gemma_cpp
#include "compression/sfp.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <random>
#include <set>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "compression/sfp_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// Any highway.h must come after foreach_target.h
// copybara:import_next_line:gemma_cpp
#include "compression/distortion.h"
// copybara:import_next_line:gemma_cpp
#include "compression/sfp-inl.h"
#include "hwy/highway.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"
#include "hwy/timer.h"

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

// ------------------------------ Foreach compressed representation

// Encode
HWY_INLINE uint32_t SFP8FromF32(float f) {
  HWY_ASSERT(-1.875f <= f && f <= 1.875f);

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
    const hn::Rebind<hwy::bfloat16_t, decltype(d16)> dbf;
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
    const hn::Repartition<hwy::bfloat16_t, D> dbf;
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

// ------------------------------ Foreach bf16 input

// Generate all values, encode, decode back.
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
    auto dec = hwy::AllocateAligned<T>(max);
    HWY_ASSERT(in && packed && dec);
    size_t num = 0;
    for (size_t i = 0; i < max; ++i) {
      const uint16_t bits = i * kStep;
      const float f = hwy::F32FromBF16(hwy::BitCastScalar<T>(bits));
      // Keep if within range
      if (hwy::ScalarIsFinite(f) && f <= 1.875f) {
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
      SfpCodec::Dec(dbf, packed.get(), num, dec.get());
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
      const double avg = sum / num;
      fprintf(stderr, "Avg magnitude %.3E, p-norm %.3E snr %.2f @%zu = %.4E\n",
              avg, stats.PNorm(), stats.GeomeanValueDivL1(), stats.MaxIndex(),
              stats.MaxL1());
    }
  }
};

void TestAllEncDec() { hn::ForGEVectors<32, TestEncDec>()(hwy::bfloat16_t()); }

// ------------------------------ Order

// Store 8-bit iota, decode, encode, check iota == packed. This ensures
// Enc/Dec are preserving the order independent of vector length.
struct TestOrder {
  template <class T, class DBF>
  HWY_INLINE void operator()(T /*unused*/, DBF dbf) {
    const hn::Repartition<uint8_t, DBF> du8;

    const size_t num = 10 * hn::Lanes(du8) / 3;

    auto iota = hwy::AllocateAligned<SfpStream>(num);
    auto packed = hwy::AllocateAligned<SfpStream>(num);
    auto bf = hwy::AllocateAligned<hwy::bfloat16_t>(num);
    HWY_ASSERT(iota && packed && bf);
    for (size_t i = 0; i < num; ++i) {
      // Clear sign bit so we can also check that bf is in ascending order.
      iota[i].byte = i & 127;
    }

    SfpCodec::Dec(dbf, iota.get(), num, bf.get());
    SfpCodec::Enc(dbf, bf.get(), num, packed.get());

    for (size_t i = 0; i < num; ++i) {
      if (iota[i].byte != packed[i].byte) {
        HWY_ABORT("@%zu: %d %d\n", i, iota[i].byte, packed[i].byte);
      }
    }
  }
};

void TestAllOrder() { hn::ForGEVectors<32, TestOrder>()(hwy::bfloat16_t()); }

// ------------------------------ Dot

struct TestDot {
  template <typename T, class D>
  HWY_INLINE void operator()(T /*unused*/, D d) {
    const hn::Repartition<float, D> df;
    const size_t num = 384;
    auto in = hwy::AllocateAligned<T>(num);
    auto dec = hwy::AllocateAligned<T>(num);
    auto vec = hwy::AllocateAligned<T>(num);
    auto sfp = hwy::AllocateAligned<SfpStream>(num);
    HWY_ASSERT(in && dec && vec && sfp);

    std::mt19937 rng(123);
    std::normal_distribution<float> dist{0.001f, 0.3f};
    for (size_t i = 0; i < num; ++i) {
      in[i] = hwy::ConvertScalarTo<T>(dist(rng));
      vec[i] = hwy::ConvertScalarTo<T>(dist(rng));
    }
    // This changes the correlation between in and vec, which considerably
    // affects the error of the result.
    std::shuffle(in.get(), in.get() + num, rng);

    SfpCodec::Enc(d, in.get(), num, sfp.get());

    double actual = 0.0;
    double elapsed = hwy::HighestValue<double>();
    for (size_t rep = 0; rep < 200; ++rep) {
      hn::Vec<decltype(df)> sum0 = hn::Zero(df);
      hn::Vec<decltype(df)> sum1 = hn::Zero(df);
      hn::Vec<decltype(df)> sum2 = hn::Zero(df);
      hn::Vec<decltype(df)> sum3 = hn::Zero(df);
      const double t0 = hwy::platform::Now();
      SfpCodec::Dot(df, sfp.get(), num, vec.get(), sum0, sum1, sum2, sum3);
      const double t1 = hwy::platform::Now();
      elapsed = HWY_MIN(elapsed, t1 - t0);
      sum0 = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
      actual = hn::ReduceSum(df, sum0);
    }

    SfpCodec::Dec(d, sfp.get(), num, dec.get());
    fprintf(stderr, "Vec %zu Dot %.2f MB/s\n", Lanes(d) * sizeof(T),
            num * sizeof(T) * 1E-6 / elapsed);

    double expected = 0.0;   // using original input
    double expected2 = 0.0;  // using decoded SFP
    for (size_t i = 0; i < num; ++i) {
      expected += hwy::ConvertScalarTo<double>(in[i]) *
                  hwy::ConvertScalarTo<double>(vec[i]);
      expected2 += hwy::ConvertScalarTo<double>(dec[i]) *
                   hwy::ConvertScalarTo<double>(vec[i]);
    }
    const double l1 = hwy::ScalarAbs(expected - actual);
    const double snr = 1.0 + hwy::ScalarAbs(expected) / l1;
    fprintf(stderr, "expected %.3f e2 %.4f actual %.4f l1 %E snr %.2f\n",
            expected, expected2, actual, l1, snr);
    HWY_ASSERT(hwy::ScalarAbs(expected2 - actual) < 1E-4);
    const double expected_l1 = sizeof(T) == 2 ? 1.52E-2 : 1.15E-2;
    const double expected_snr = sizeof(T) == 2 ? 80.1f : 104.9f;
    HWY_ASSERT(expected_l1 <= l1 && l1 < 1.02f * expected_l1);
    HWY_ASSERT(expected_snr <= snr && snr < 1.01f * expected_snr);
  }
};

void TestAllDotF32() {
  const hn::ForGEVectors<128, TestDot> test;
  test(float());
}
void TestAllDotBF16() {
  const hn::ForGEVectors<128, TestDot> test;
  test(hwy::bfloat16_t());
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gcpp {
HWY_BEFORE_TEST(SfpTest);
HWY_EXPORT_AND_TEST_P(SfpTest, TestAllUnique);
HWY_EXPORT_AND_TEST_P(SfpTest, TestAllDecEnc);
HWY_EXPORT_AND_TEST_P(SfpTest, TestAllGolden);
HWY_EXPORT_AND_TEST_P(SfpTest, TestAllEncDec);
HWY_EXPORT_AND_TEST_P(SfpTest, TestAllOrder);
HWY_EXPORT_AND_TEST_P(SfpTest, TestAllDotF32);
HWY_EXPORT_AND_TEST_P(SfpTest, TestAllDotBF16);
}  // namespace gcpp

#endif
