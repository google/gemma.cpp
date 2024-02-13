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

// Normal include guard to placate lint.
#ifndef THIRD_PARTY_GEMMA_CPP_COMPRESSION_SFP_INL_H_
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_SFP_INL_H_

#include <stddef.h>
#include <stdint.h>

// copybara:import_next_line:gemma_cpp
#include "compression/sfp.h"
#include "hwy/base.h"

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_SFP_INL_H_

// Actual per-target include guard.
#if defined(THIRD_PARTY_GEMMA_CPP_SFP_INL_TOGGLE) == defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_SFP_INL_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_SFP_INL_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_SFP_INL_TOGGLE
#endif

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// For unsigned numbers with MSB zero, signed comparison is faster on x86.
template <class DU>
HWY_INLINE hn::Mask<DU> SignedGt(DU du, hn::Vec<DU> a, hn::Vec<DU> b) {
  const hn::RebindToSigned<DU> di;
  return hn::RebindMask(du, hn::Gt(BitCast(di, a), hn::BitCast(di, b)));
}
template <class DU>
HWY_INLINE hn::Mask<DU> SignedLt(DU du, hn::Vec<DU> a, hn::Vec<DU> b) {
  return SignedGt(du, b, a);
}

// Encode/decode functions.
class SfpCodec {
 public:
  // Returns 8-bit packed representation of `lo` and `hi` bytes of bf16. 31 ops.
  // Implementation detail, public because called by test.
  template <class D, HWY_IF_U8_D(D)>
  static HWY_INLINE hn::Vec<D> EncBytes(D d, const hn::Vec<D> lo,
                                        const hn::Vec<D> hi) {
    const hn::Vec<D> k1 = hn::Set(d, 1u);
    const hn::Vec<D> k80 = hn::Set(d, 0x80u);

    // Copy sign for later insertion.
    const hn::Vec<D> sign_in_msb = hi;
    // Biased exponent = lower 7 bits of hi and MSB of lo. Modified below.
    hn::Vec<D> biased_e = hn::Or(hn::Add(hi, hi), hn::ShiftRight<7>(lo));
    HWY_ASSERT(hn::AllTrue(d, hn::Lt(biased_e, k80)));  // <= 2^0

    // Clear MSB to isolate the mantissa and enable signed comparisons, then
    // shift right by *one* (plus 1 to undo the prior add/left-shift) to leave
    // headroom for overflow during rounding.
    const hn::Vec<D> m6 = hn::ShiftRight<2>(hn::Add(lo, lo));

    // The place to round depends on whether the exponent is large (>= -7) - if
    // so, we retain three mantissa bits, otherwise two. However, rounding can
    // also cause the exponent to increase. We first choose a threshold that
    // rounds up to 1.0*2^-7 for both two and three bit mantissas:
    // >= 1.1111 * 2^-8 (0.007568359375). This entails the exponent being
    // greater, or equal and the mantissa > (1111000 >> 1) - 1 = 0x3B.
    const hn::Vec<D> kMinLargeE = hn::Set(d, 127 - 8);
    const hn::Mask<D> is_large_before_round = hn::Or(
        SignedGt(d, biased_e, kMinLargeE),
        hn::And(hn::Eq(biased_e, kMinLargeE), SignedGt(d, m6, Set(d, 0x3B))));

    // To retain the most-significant 3 or 2 mantissa bits, we will right-shift
    // by is_large_before_round ? 3 : 4. Variable Shr is expensive for 8-bit
    // elements, so (<< 1) if is_large_before_round, then always (>> 4).
    const hn::Vec<D> m_shl4 =
        hn::MaskedAddOr(m6, is_large_before_round, m6, m6);

    // Before shifting (truncation), round to nearest even to reduce bias. If
    // the lowest remaining mantissa bit is odd, increase the offset. Example
    // with the lowest remaining bit (left) and next lower two bits; the
    // latter, plus two more, will be truncated.
    // 0[00] +  1 =  0[01]
    // 0[01] +  1 =  0[10]
    // 0[10] +  1 =  0[11]  (round down toward even)
    // 0[11] +  1 =  1[00]  (round up)
    // 1[00] + 10 =  1[10]
    // 1[01] + 10 =  1[11]
    // 1[10] + 10 = C0[00]  (round up toward even with C=1 carry out)
    // 1[11] + 10 = C0[01]  (round up toward even with C=1 carry out)
    const hn::Vec<D> odd_bit = hn::And(hn::ShiftRight<4>(m_shl4), k1);
    const hn::Vec<D> rounded = hn::Add(m_shl4, hn::Add(odd_bit, Set(d, 7)));
    // Update the exponent if rounding overflowed.
    const hn::Vec<D> carry_bit =
        hn::IfThenElse(is_large_before_round, k80, hn::Set(d, 0x40u));
    const hn::Vec<D> carry_clear = hn::AndNot(carry_bit, rounded);
    HWY_DASSERT(hn::AllTrue(d, hn::Lt(carry_clear, carry_bit)));
    const hn::Mask<D> is_overflow = hn::Ne(carry_clear, rounded);
    biased_e = hn::MaskedAddOr(biased_e, is_overflow, biased_e, k1);
    HWY_DASSERT(hn::AllTrue(d, hn::Lt(biased_e, Set(d, 128))));

    // Detect if zero or the min exponent.
    const hn::Vec<D> kMinNormal = hn::Set(d, 127 - 23);
    const hn::Mask<D> is_zero = SignedLt(d, biased_e, kMinNormal);
    const hn::Mask<D> is_min = hn::Eq(biased_e, kMinNormal);

    // 1.1110xxx * 2^-8 was considered small above, and thus rounded up to 2^-7,
    // which the decoder will consider large, and expect 3 mantissa bits. If we
    // set the threshold above to 1.111, then it does NOT round up. Thus we
    // check exponent >= -7 *after* rounding.
    const hn::Mask<D> is_large = SignedGt(d, biased_e, hn::Set(d, 127 - 8));

    // To extract and pack the mantissa, only is_large matters. Either it
    // matches is_large_before_round, or the rounding resulted in mantissa=0, so
    // we either extract two or three bits by shifting out the lower 5..6 bits.
    // is_large_before is_large  rounded     want
    //         0           0     0Cmm????     mm
    //         0           1     0100????    000
    //         1           0     impossible   -
    //         1           1     Cmmm???0    mmm
    hn::Vec<D> m = hn::ShiftRight<4>(carry_clear);
    HWY_DASSERT(hn::AllTrue(
        d, SignedLt(d, m,
                    hn::IfThenElse(is_large, hn::Set(d, 8), hn::Set(d, 4)))));

    // 1.0 * 2^-23 has the same encoding as zero, so round it up to 1.01.
    m = hn::MaskedMaxOr(m, is_min, m, k1);

    const hn::Vec<D> e_bias = hn::IfThenElse(
        is_large,
        hn::Set(d, hwy::BitCastScalar<uint8_t>(static_cast<int8_t>(15 - 127))),
        hn::Set(d, hwy::BitCastScalar<uint8_t>(static_cast<int8_t>(23 - 127))));
    const hn::Vec<D> e = hn::Add(biased_e, e_bias);
    HWY_DASSERT(
        hn::AllTrue(d, hn::Lt(hn::IfThenZeroElse(is_zero, e), hn::Set(d, 16))));

    // Shift exponent left 2 or 3 bits to make space for `m`.
    const hn::Vec<D> em =
        hn::Or(m, hn::ShiftLeft<2>(hn::MaskedAddOr(e, is_large, e, e)));
    HWY_DASSERT(hn::AllTrue(d, hn::Lt(hn::IfThenZeroElse(is_zero, em), k80)));
    const hn::Vec<D> encoded = hn::BitwiseIfThenElse(k80, sign_in_msb, em);
    // Doing this last ensures -0 is replaced with 0.
    return hn::IfThenZeroElse(is_zero, encoded);
  }

  // Decodes u8 `encoded` into `lo` and `hi` bytes of bf16. 12 ops.
  // Implementation detail, public because called by test.
  template <class D, HWY_IF_U8_D(D)>
  static HWY_INLINE void DecBytes(D d, hn::Vec<D> encoded, hn::Vec<D>& lo,
                                  hn::Vec<D>& hi) {
    const hn::Vec<D> k0 = hn::Zero(d);
    const hn::Vec<D> k80 = hn::Set(d, 0x80u);

    HWY_DASSERT(hn::AllTrue(d, hn::Ne(encoded, k80)));  // -0 is reserved
    // Copy sign for later insertion via BitwiseIfThenElse.
    const hn::Vec<D> sign_in_msb = encoded;
    encoded = hn::AndNot(k80, encoded);

    // Special-case zero, negated so we can use MaskedAddOr. Signed comparison
    // is fine because we have cleared the sign bit.
    const hn::Mask<D> is_nonzero = SignedGt(d, encoded, k0);
    // If MSB is clear, we have two mantissa bits, otherwise three.
    const hn::Mask<D> is_small_e = SignedLt(d, encoded, hn::Set(d, 64));
    // If is_small_e, add/left-shift 0xxxx.mm to 0xxxx.mm0; else keep 1xxx.mmm.
    const hn::Vec<D> e4m3 =
        hn::MaskedAddOr(encoded, is_small_e, encoded, encoded);
    HWY_DASSERT(hn::AllTrue(d, hn::Lt(e4m3, k80)));
    const hn::Vec<D> e = hn::ShiftRight<3>(e4m3);  // 4-bit exponent only
    HWY_DASSERT(hn::AllTrue(d, hn::Lt(e, Set(d, 16u))));
    // The encoded exponent for 2^0 is 15, so subtract 15. Add 127 for the
    // binary32/bf16 bias. Subtract another 8 if is_small_e because its lowest
    // encoded value (0) should be less than the lowest 'large' exponent 2^-7.
    const hn::Vec<D> e_bias = hn::IfThenElse(
        is_small_e, hn::Set(d, 127u - 15u - 8u), hn::Set(d, 127u - 15u));
    // Special-case zero or add e_bias. If encoded=0, e and e4m3 are zero, but
    // we must zero e_bias to get the desired all-zero bf16.
    const hn::Vec<D> biased_e = hn::MaskedAddOr(k0, is_nonzero, e_bias, e);
    // The decoded binary32 exponent should be at most 2^0.
    HWY_DASSERT(hn::AllTrue(d, hn::Lt(biased_e, k80)));

    // Shift the MSB of e4m3's mantissa into the MSB of the bf16 mantissa.
    const hn::Vec<D> m7 = hn::ShiftLeft<4>(e4m3);
    // Lower byte of bf16 = exponent LSB || mantissa.
    lo = hn::BitwiseIfThenElse(k80, hn::ShiftLeft<7>(biased_e), m7);
    // Upper byte of bf16 = sign || lower 7 bits of exponent.
    hi = hn::BitwiseIfThenElse(k80, sign_in_msb, hn::ShiftRight<1>(biased_e));
  }

  // Encodes `num` bf16 values from `in_bf` to `out_packed`.
  template <class DBF, HWY_IF_BF16_D(DBF)>
  static HWY_INLINE void Enc(DBF dbf, const hwy::bfloat16_t* HWY_RESTRICT in_bf,
                             size_t num, SfpStream* HWY_RESTRICT out_packed) {
    const hn::Repartition<uint8_t, DBF> d8;
    using V8 = hn::Vec<decltype(d8)>;
    const size_t N16 = hn::Lanes(dbf);

    size_t i = 0;
    if (num >= 2 * N16) {
      HWY_UNROLL(1)
      for (; i <= num - 2 * N16; i += 2 * N16) {
        const V8 packed = Enc2B(dbf, in_bf + i);
        hn::StoreU(packed, d8, &out_packed->byte + i);
      }
    }

    const size_t remaining = num - i;
    HWY_DASSERT(remaining < 2 * N16);
    if (remaining != 0) {
      HWY_ALIGN hwy::bfloat16_t padded[2 * hn::MaxLanes(dbf)];
      hwy::ZeroBytes(padded, sizeof(padded));
      hwy::CopyBytes(in_bf + i, padded, remaining * sizeof(padded[0]));
      const V8 packed = Enc2B(dbf, padded);
      hn::StoreN(packed, d8, &out_packed->byte + i, remaining);
    }
  }

  // Encodes `num` f32 values from `in_f` to `packed`.
  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void Enc(DF df, const float* HWY_RESTRICT in_f, size_t num,
                             SfpStream* HWY_RESTRICT out_packed) {
    const hn::Repartition<uint8_t, DF> d8;
    using V8 = hn::Vec<decltype(d8)>;
    const size_t NF = hn::Lanes(df);

    size_t i = 0;
    if (num >= 4 * NF) {
      HWY_UNROLL(1)
      for (; i <= num - 4 * NF; i += 4 * NF) {
        const V8 packed = Enc4F(df, in_f + i);
        hn::StoreU(packed, d8, &out_packed->byte + i);
      }
    }

    const size_t remaining = num - i;
    HWY_DASSERT(remaining < 4 * NF);
    if (remaining != 0) {
      HWY_ALIGN float padded[4 * hn::MaxLanes(df)];
      hwy::ZeroBytes(padded, sizeof(padded));
      hwy::CopyBytes(in_f + i, padded, remaining * sizeof(padded[0]));
      const V8 packed = Enc4F(df, padded);
      hn::StoreN(packed, d8, &out_packed->byte + i, remaining);
    }
  }

  // Decodes `num` values from `in_packed` to `out_bf`.
  template <class DBF, HWY_IF_BF16_D(DBF)>
  static HWY_INLINE void Dec(DBF dbf, const SfpStream* HWY_RESTRICT in_packed,
                             size_t num, hwy::bfloat16_t* HWY_RESTRICT out_bf) {
    const hn::Repartition<uint8_t, DBF> d8;
    using V8 = hn::Vec<decltype(d8)>;
    using VBF = hn::Vec<decltype(dbf)>;
    const size_t N16 = hn::Lanes(dbf);

    size_t i = 0;
    if (num >= 2 * N16) {
      HWY_UNROLL(1)
      for (; i <= num - 2 * N16; i += 2 * N16) {
        const V8 packed = hn::LoadU(d8, &in_packed->byte + i);
        VBF bf0, bf1;
        Dec2B(dbf, packed, bf0, bf1);
        hn::StoreU(bf0, dbf, out_bf + i);
        hn::StoreU(bf1, dbf, out_bf + i + N16);
      }
    }

    const size_t remaining = num - i;
    HWY_DASSERT(remaining < 2 * N16);
    if (remaining != 0) {
      const V8 packed = hn::LoadN(d8, &in_packed->byte + i, remaining);
      HWY_ALIGN hwy::bfloat16_t padded[2 * hn::MaxLanes(dbf)];
      VBF bf0, bf1;
      Dec2B(dbf, packed, bf0, bf1);
      hn::StoreU(bf0, dbf, padded);
      hn::StoreU(bf1, dbf, padded + N16);
      hwy::CopyBytes(padded, out_bf + i, remaining * sizeof(padded[0]));
    }
  }

  // Decodes `num` values from `in_packed` to `out_f`.
  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void Dec(DF df, const SfpStream* HWY_RESTRICT in_packed,
                             size_t num, float* HWY_RESTRICT out_f) {
    const hn::Repartition<uint8_t, DF> d8;
    using V8 = hn::Vec<decltype(d8)>;
    using VF = hn::Vec<decltype(df)>;
    const size_t NF = hn::Lanes(df);

    size_t i = 0;
    if (num >= 4 * NF) {
      HWY_UNROLL(1)
      for (; i <= num - 4 * NF; i += 4 * NF) {
        const V8 packed = hn::LoadU(d8, &in_packed->byte + i);
        VF f0, f1, f2, f3;
        Dec4F(df, packed, f0, f1, f2, f3);
        hn::StoreU(f0, df, out_f + i + NF * 0);
        hn::StoreU(f1, df, out_f + i + NF * 1);
        hn::StoreU(f2, df, out_f + i + NF * 2);
        hn::StoreU(f3, df, out_f + i + NF * 3);
      }
    }

    const size_t remaining = num - i;
    HWY_DASSERT(remaining < 4 * NF);
    if (remaining != 0) {
      const V8 packed = hn::LoadN(d8, &in_packed->byte + i, remaining);
      HWY_ALIGN float padded[4 * hn::MaxLanes(df)];
      VF f0, f1, f2, f3;
      Dec4F(df, packed, f0, f1, f2, f3);
      hn::StoreU(f0, df, padded + NF * 0);
      hn::StoreU(f1, df, padded + NF * 1);
      hn::StoreU(f2, df, padded + NF * 2);
      hn::StoreU(f3, df, padded + NF * 3);
      hwy::CopyBytes(padded, out_f + i, remaining * sizeof(padded[0]));
    }
  }

  // Fused decode and dot product with bf16 into four output accumulators.
  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void Dot(DF df, const SfpStream* HWY_RESTRICT in_packed,
                             size_t num,
                             const hwy::bfloat16_t* HWY_RESTRICT vec_aligned,
                             hn::Vec<DF>& sum0, hn::Vec<DF>& sum1,
                             hn::Vec<DF>& sum2, hn::Vec<DF>& sum3) {
    const hn::Repartition<uint8_t, DF> d8;
    const hn::Repartition<hwy::bfloat16_t, DF> dbf;
    using V8 = hn::Vec<decltype(d8)>;
    using VBF = hn::Vec<decltype(dbf)>;
    const size_t N16 = hn::Lanes(dbf);

    size_t i = 0;
    if (num >= 2 * N16) {
      HWY_UNROLL(1)
      for (; i <= num - 2 * N16; i += 2 * N16) {
        const V8 packed = hn::LoadU(d8, &in_packed->byte + i);
        const VBF v0 = hn::LoadU(dbf, vec_aligned + i);
        const VBF v1 = hn::LoadU(dbf, vec_aligned + i + N16);
        VBF bf0, bf1;
        Dec2B(dbf, packed, bf0, bf1);
        sum0 = hn::ReorderWidenMulAccumulate(df, bf0, v0, sum0, sum1);
        sum2 = hn::ReorderWidenMulAccumulate(df, bf1, v1, sum2, sum3);
      }
    }

    const size_t remaining = num - i;
    if (remaining != 0) {
      const V8 packed = hn::LoadN(d8, &in_packed->byte + i, remaining);
      HWY_ALIGN hwy::bfloat16_t padded[2 * hn::MaxLanes(dbf)];
      hwy::ZeroBytes(padded, sizeof(padded));
      hwy::CopyBytes(vec_aligned + i, padded, remaining * sizeof(padded[0]));
      const VBF v0 = hn::LoadU(dbf, padded);
      const VBF v1 = hn::LoadU(dbf, padded + N16);
      VBF bf0, bf1;
      Dec2B(dbf, packed, bf0, bf1);
      sum0 = hn::ReorderWidenMulAccumulate(df, bf0, v0, sum0, sum1);
      sum2 = hn::ReorderWidenMulAccumulate(df, bf1, v1, sum2, sum3);
    }
  }

  // Fused decode and dot product with f32 into four output accumulators.
  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void Dot(DF df, const SfpStream* HWY_RESTRICT in_packed,
                             size_t num, const float* HWY_RESTRICT vec_aligned,
                             hn::Vec<DF>& sum0, hn::Vec<DF>& sum1,
                             hn::Vec<DF>& sum2, hn::Vec<DF>& sum3) {
    const hn::Repartition<uint8_t, DF> d8;
    using V8 = hn::Vec<decltype(d8)>;
    using VF = hn::Vec<decltype(df)>;
    const size_t NF = hn::Lanes(df);

    size_t i = 0;
    if (num >= 4 * NF) {
      HWY_UNROLL(1)
      for (; i <= num - 4 * NF; i += 4 * NF) {
        const V8 packed = hn::LoadU(d8, &in_packed->byte + i);
        const VF v0 = hn::LoadU(df, vec_aligned + i + NF * 0);
        const VF v1 = hn::LoadU(df, vec_aligned + i + NF * 1);
        const VF v2 = hn::LoadU(df, vec_aligned + i + NF * 2);
        const VF v3 = hn::LoadU(df, vec_aligned + i + NF * 3);
        VF f0, f1, f2, f3;
        Dec4F(df, packed, f0, f1, f2, f3);
        sum0 = hn::MulAdd(f0, v0, sum0);
        sum1 = hn::MulAdd(f1, v1, sum1);
        sum2 = hn::MulAdd(f2, v2, sum2);
        sum3 = hn::MulAdd(f3, v3, sum3);
      }
    }

    const size_t remaining = num - i;
    if (remaining != 0) {
      const V8 packed = hn::LoadN(d8, &in_packed->byte + i, remaining);
      HWY_ALIGN float padded[4 * hn::MaxLanes(df)];
      hwy::ZeroBytes(padded, sizeof(padded));
      hwy::CopyBytes(vec_aligned + i, padded, remaining * sizeof(padded[0]));
      const VF v0 = hn::LoadU(df, padded + NF * 0);
      const VF v1 = hn::LoadU(df, padded + NF * 1);
      const VF v2 = hn::LoadU(df, padded + NF * 2);
      const VF v3 = hn::LoadU(df, padded + NF * 3);
      VF f0, f1, f2, f3;
      Dec4F(df, packed, f0, f1, f2, f3);
      sum0 = hn::MulAdd(f0, v0, sum0);
      sum1 = hn::MulAdd(f1, v1, sum1);
      sum2 = hn::MulAdd(f2, v2, sum2);
      sum3 = hn::MulAdd(f3, v3, sum3);
    }
  }

 private:
  // Wrappers to avoid code duplication across float/bf16 input types and
  // the main loop/remainder.

  // Returns vector of packed bytes for callers to StoreU or StoreN.
  template <class D16, HWY_IF_U16_D(D16),
            class V8 = hn::Vec<hn::Repartition<uint8_t, D16>>>
  static HWY_INLINE V8 Enc2U(D16 d16, const hn::Vec<D16> w0,
                             const hn::Vec<D16> w1) {
    const hn::Repartition<uint8_t, D16> d8;

    // Although more expensive on AVX3, in-order packing enables streaming
    // decompression without fixed-size packets.
    const V8 lo = hn::ConcatEven(d8, hn::BitCast(d8, w1), hn::BitCast(d8, w0));
    const V8 hi = hn::ConcatOdd(d8, hn::BitCast(d8, w1), hn::BitCast(d8, w0));
    return EncBytes(d8, lo, hi);
  }

  template <class DBF, HWY_IF_BF16_D(DBF),
            class V8 = hn::Vec<hn::Repartition<uint8_t, DBF>>>
  static HWY_INLINE V8 Enc2B(DBF dbf, const hwy::bfloat16_t* HWY_RESTRICT in) {
    const hn::Repartition<uint16_t, DBF> d16;
    const size_t N16 = hn::Lanes(d16);
    using V16 = hn::Vec<decltype(d16)>;

    const V16 w0 = hn::BitCast(d16, hn::LoadU(dbf, in));
    const V16 w1 = hn::BitCast(d16, hn::LoadU(dbf, in + N16));
    return Enc2U(d16, w0, w1);
  }

  template <class DF, HWY_IF_F32_D(DF),
            class V8 = hn::Vec<hn::Repartition<uint8_t, DF>>>
  static HWY_INLINE V8 Enc4F(DF df, const float* HWY_RESTRICT in) {
    const hn::Repartition<uint16_t, DF> d16;
    const hn::Repartition<hwy::bfloat16_t, DF> dbf;
    using VF = hn::Vec<decltype(df)>;
    using V16 = hn::Vec<decltype(d16)>;
    const size_t NF = hn::Lanes(df);

    const VF f0 = hn::LoadU(df, in + NF * 0);
    const VF f1 = hn::LoadU(df, in + NF * 1);
    const VF f2 = hn::LoadU(df, in + NF * 2);
    const VF f3 = hn::LoadU(df, in + NF * 3);
    // Chop off the lower 16 bits; EncBytes still rounds properly.
    const V16 w0 = hn::BitCast(d16, hn::OrderedDemote2To(dbf, f0, f1));
    const V16 w1 = hn::BitCast(d16, hn::OrderedDemote2To(dbf, f2, f3));
    return Enc2U(d16, w0, w1);
  }

  template <class D16, HWY_IF_U16_D(D16),
            class V8 = hn::Vec<hn::Repartition<uint8_t, D16>>>
  static HWY_INLINE void Dec2U(D16 d16, V8 packed, hn::Vec<D16>& w0,
                               hn::Vec<D16>& w1) {
    const hn::Repartition<uint8_t, D16> d8;
    V8 lo, hi;
    DecBytes(d8, packed, lo, hi);
    w0 = hn::BitCast(d16, hn::InterleaveWholeLower(d8, lo, hi));
    w1 = hn::BitCast(d16, hn::InterleaveWholeUpper(d8, lo, hi));
  }

  template <class DBF, HWY_IF_BF16_D(DBF),
            class V8 = hn::Vec<hn::Repartition<uint8_t, DBF>>>
  static HWY_INLINE void Dec2B(DBF dbf, V8 packed, hn::Vec<DBF>& bf0,
                               hn::Vec<DBF>& bf1) {
    const hn::Repartition<uint16_t, DBF> d16;
    using V16 = hn::Vec<decltype(d16)>;
    V16 w0, w1;
    Dec2U(d16, packed, w0, w1);
    bf0 = hn::BitCast(dbf, w0);
    bf1 = hn::BitCast(dbf, w1);
  }

  template <class DF, HWY_IF_F32_D(DF),
            class V8 = hn::Vec<hn::Repartition<uint8_t, DF>>>
  static HWY_INLINE void Dec4F(DF df, V8 packed, hn::Vec<DF>& f0,
                               hn::Vec<DF>& f1, hn::Vec<DF>& f2,
                               hn::Vec<DF>& f3) {
    const hn::Repartition<hwy::bfloat16_t, DF> dbf;
    using VBF = hn::Vec<decltype(dbf)>;
    VBF bf0, bf1;
    Dec2B(dbf, packed, bf0, bf1);
    f0 = hn::PromoteLowerTo(df, bf0);
    f1 = hn::PromoteUpperTo(df, bf0);
    f2 = hn::PromoteLowerTo(df, bf1);
    f3 = hn::PromoteUpperTo(df, bf1);
  }
};  // SfpCodec

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_SFP_INL_H_
