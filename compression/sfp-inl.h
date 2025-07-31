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

#include "compression/types.h"
#include "hwy/base.h"

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_SFP_INL_H_

// Actual per-target include guard.
#if defined(THIRD_PARTY_GEMMA_CPP_SFP_INL_TOGGLE) == defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_SFP_INL_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_SFP_INL_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_SFP_INL_TOGGLE
#endif

#include "hwy/detect_targets.h"
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

  // Decodes u8 `encoded` into `lo` and `hi` bytes of bf16. 3 ops (AVX-512).
#if HWY_TARGET <= HWY_AVX3_DL || HWY_IDE
  template <class D, HWY_IF_U8_D(D), HWY_IF_V_SIZE_D(D, 64)>
  static HWY_INLINE void DecBytes(D d, hn::Vec<D> encoded, hn::Vec<D>& lo,
                                  hn::Vec<D>& hi) {
    const hn::Vec<D> k80 = hn::Set(d, 0x80u);
    HWY_DASSERT(hn::AllTrue(d, hn::Ne(encoded, k80)));  // -0 is reserved

    // Two 2x64 table lookups for lo/hi.
    alignas(64) static constexpr uint8_t kTblL0[64] = {
        0x00, 0x20, 0x40, 0x60, 0x80, 0xA0, 0xC0, 0xE0, 0x00, 0x20, 0x40,
        0x60, 0x80, 0xA0, 0xC0, 0xE0, 0x00, 0x20, 0x40, 0x60, 0x80, 0xA0,
        0xC0, 0xE0, 0x00, 0x20, 0x40, 0x60, 0x80, 0xA0, 0xC0, 0xE0, 0x00,
        0x20, 0x40, 0x60, 0x80, 0xA0, 0xC0, 0xE0, 0x00, 0x20, 0x40, 0x60,
        0x80, 0xA0, 0xC0, 0xE0, 0x00, 0x20, 0x40, 0x60, 0x80, 0xA0, 0xC0,
        0xE0, 0x00, 0x20, 0x40, 0x60, 0x80, 0xA0, 0xC0, 0xE0};
    alignas(64) static constexpr uint8_t kTblL1[64] = {
        0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xA0,
        0xB0, 0xC0, 0xD0, 0xE0, 0xF0, 0x00, 0x10, 0x20, 0x30, 0x40, 0x50,
        0x60, 0x70, 0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0, 0x00,
        0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xA0, 0xB0,
        0xC0, 0xD0, 0xE0, 0xF0, 0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60,
        0x70, 0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0};
    alignas(64) static constexpr uint8_t kTblH0[64] = {
        0x00, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x35, 0x35, 0x35,
        0x35, 0x35, 0x35, 0x35, 0x35, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36,
        0x36, 0x36, 0x37, 0x37, 0x37, 0x37, 0x37, 0x37, 0x37, 0x37, 0x38,
        0x38, 0x38, 0x38, 0x38, 0x38, 0x38, 0x38, 0x39, 0x39, 0x39, 0x39,
        0x39, 0x39, 0x39, 0x39, 0x3A, 0x3A, 0x3A, 0x3A, 0x3A, 0x3A, 0x3A,
        0x3A, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B};
    alignas(64) static constexpr uint8_t kTblH1[64] = {
        0x3C, 0x3C, 0x3C, 0x3C, 0x3C, 0x3C, 0x3C, 0x3C, 0x3C, 0x3C, 0x3C,
        0x3C, 0x3C, 0x3C, 0x3C, 0x3C, 0x3D, 0x3D, 0x3D, 0x3D, 0x3D, 0x3D,
        0x3D, 0x3D, 0x3D, 0x3D, 0x3D, 0x3D, 0x3D, 0x3D, 0x3D, 0x3D, 0x3E,
        0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E,
        0x3E, 0x3E, 0x3E, 0x3E, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F,
        0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F};
    const hn::Vec<D> tblL0 = hn::LoadU(d, kTblL0);
    const hn::Vec<D> tblL1 = hn::LoadU(d, kTblL1);
    const hn::Vec<D> tblH0 = hn::LoadU(d, kTblH0);
    const hn::Vec<D> tblH1 = hn::LoadU(d, kTblH1);
#if HWY_IDE  // only let the IDE see portable code.
    const auto idx = hn::IndicesFromVec(hn::AndNot(k80, encoded));
#else  // AVX-512-specific: index MSB is ignored, no need to clear.
    const hn::Indices512<uint8_t> idx{encoded.raw};
#endif
    hi = hn::TwoTablesLookupLanes(d, tblH0, tblH1, idx);
    lo = hn::TwoTablesLookupLanes(d, tblL0, tblL1, idx);
    hi = hn::OrAnd(hi, encoded, k80);  // Insert sign bit
  }

// Generic is only required for partial vectors (too small for tables).
#undef SFP_IF_GENERIC_DEC
#define SFP_IF_GENERIC_DEC(D) HWY_IF_V_SIZE_LE_D(D, 32)
#else
// Always enable the generic decoder.
#undef SFP_IF_GENERIC_DEC
#define SFP_IF_GENERIC_DEC(D) void* yes = nullptr
#endif

  // Decodes u8 `encoded` into `lo` and `hi` bytes of bf16. 9 ops.
  template <class D, HWY_IF_U8_D(D), SFP_IF_GENERIC_DEC(D)>
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
    // If bit 6 is clear, we have two mantissa bits, otherwise three.
    const hn::Mask<D> is_small_e = SignedLt(d, encoded, hn::Set(d, 64));
    // For encoded in [1, 8), hi = 0x34; encoded = 0x40 => hi = 0x3C including
    // (encoded >> 4) == 4, so add 0x38.
    const hn::Vec<D> e_bias =
        hn::IfThenElse(is_small_e, hn::Set(d, 0x34), hn::Set(d, 0x38));

    // The low byte of bf16 is encoded << (is_small_e ? 5 : 4).
    const hn::Vec<D> shl1_if_small =
        hn::MaskedAddOr(encoded, is_small_e, encoded, encoded);
    lo = hn::ShiftLeft<4>(shl1_if_small);
    // Lower 4 bits always zero.
    HWY_DASSERT(hn::AllTrue(d, hn::Eq(hn::And(lo, Set(d, 15u)), hn::Zero(d))));

    // The upper byte of bf16 is e_bias + (encoded >> (is_small_e ? 3 : 4)).
    const hn::Vec<D> shr_3_or_4 = hn::ShiftRight<4>(shl1_if_small);
    // .. except when encoded=0: hi = 0, and lo is already 0.
    const hn::Vec<D> e7 = hn::MaskedAddOr(k0, is_nonzero, e_bias, shr_3_or_4);
    HWY_DASSERT(hn::AllTrue(d, hn::Lt(e7, Set(d, 64u))));  // <= 0x3F
    // .. also insert the sign bit.
    hi = hn::BitwiseIfThenElse(k80, sign_in_msb, e7);
  }

  // Encodes `num` bf16 values from `in_bf` to `out_packed`. Their magnitude
  // must be at most SfpStream::kMax.
  template <class DBF, HWY_IF_BF16_D(DBF)>
  static HWY_INLINE void Enc(DBF dbf, const BF16* HWY_RESTRICT in_bf,
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
      HWY_ALIGN BF16 padded[2 * hn::MaxLanes(dbf)];
      hwy::ZeroBytes(padded, sizeof(padded));
      hwy::CopyBytes(in_bf + i, padded, remaining * sizeof(padded[0]));
      const V8 packed = Enc2B(dbf, padded);
      hn::StoreN(packed, d8, &out_packed->byte + i, remaining);
    }
  }

  // Encodes `num` f32 values from `in_f` to `packed`. Their magnitude
  // must be at most SfpStream::kMax.
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

  template <class DBF16, HWY_IF_BF16_D(DBF16),
            class V8 = hn::Vec<hn::Repartition<uint8_t, DBF16>>>
  static HWY_INLINE void Dec2(DBF16 dbf16, V8 packed, hn::Vec<DBF16>& raw0,
                              hn::Vec<DBF16>& raw1) {
    Dec2B(dbf16, packed, raw0, raw1);
  }

  template <class DF, HWY_IF_F32_D(DF),
            class V8 = hn::Vec<hn::Twice<hn::Rebind<uint8_t, DF>>>>
  static HWY_INLINE void Dec2(DF df, V8 packed, hn::Vec<DF>& raw0,
                              hn::Vec<DF>& raw1) {
    const hn::Rebind<BF16, DF> dbf;  // half-vector
    using VBF = hn::Vec<decltype(dbf)>;
    VBF bf0, bf1;
    Dec2B(dbf, packed, bf0, bf1);
    raw0 = hn::PromoteTo(df, bf0);
    raw1 = hn::PromoteTo(df, bf1);
  }

  // Decompresses to (arbitrary) `num` BF16 elements in `raw_bf`, then appends
  // `[0, hn::Lanes(dbf))` zeroes as required to round `num` up to one vector,
  // if it is not already. DBF argument is provided by nuq-inl.h.
  template <class DBF, HWY_IF_BF16_D(DBF)>
  static HWY_INLINE void DecompressAndZeroPad(
      DBF dbf, const PackedSpan<const SfpStream>& packed, size_t packed_ofs,
      BF16* HWY_RESTRICT raw_bf, size_t num) {
    const hn::Repartition<uint8_t, DBF> d8;
    using V8 = hn::Vec<decltype(d8)>;
    using VBF = hn::Vec<decltype(dbf)>;
    const size_t N16 = hn::Lanes(dbf);

    const uint8_t* HWY_RESTRICT base = &packed.ptr->byte + packed_ofs;

    size_t i = 0;
    if (num >= 2 * N16) {
      HWY_UNROLL(1)
      for (; i <= num - 2 * N16; i += 2 * N16) {
        const V8 packed = hn::LoadU(d8, base + i);
        VBF bf0, bf1;
        Dec2B(dbf, packed, bf0, bf1);
        hn::StoreU(bf0, dbf, raw_bf + i);
        hn::StoreU(bf1, dbf, raw_bf + i + N16);
      }
    }

    const size_t remaining = num - i;
    HWY_DASSERT(remaining < 2 * N16);
    if (remaining != 0) {
      const V8 packed = hn::LoadN(d8, base + i, remaining);
      VBF bf0, bf1;
      Dec2B(dbf, packed, bf0, bf1);
      // If at most one vector, the first store adds zero padding. Check before
      // storing the second, because callers only pad to one vector.
      hn::StoreU(bf0, dbf, raw_bf + i);
      if (remaining > N16) hn::StoreU(bf1, dbf, raw_bf + i + N16);
    }
  }

  // Decompresses to (arbitrary) `num` float elements in `raw_f`, then appends
  // `[0, hn::Lanes(df))` zeroes as required to round `num` up to one vector,
  // if it is not already.
  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void DecompressAndZeroPad(
      DF df, const PackedSpan<const SfpStream>& packed, size_t packed_ofs,
      float* HWY_RESTRICT raw_f, size_t num) {
    const hn::Repartition<uint8_t, DF> d8;
    using V8 = hn::Vec<decltype(d8)>;
    using VF = hn::Vec<decltype(df)>;
    const size_t NF = hn::Lanes(df);

    const uint8_t* HWY_RESTRICT base = &packed.ptr->byte + packed_ofs;

    size_t i = 0;
    if (num >= 4 * NF) {
      HWY_UNROLL(1)
      for (; i <= num - 4 * NF; i += 4 * NF) {
        const V8 packed = hn::LoadU(d8, base + i);
        VF f0, f1, f2, f3;
        Dec4F(df, packed, f0, f1, f2, f3);
        hn::StoreU(f0, df, raw_f + i + NF * 0);
        hn::StoreU(f1, df, raw_f + i + NF * 1);
        hn::StoreU(f2, df, raw_f + i + NF * 2);
        hn::StoreU(f3, df, raw_f + i + NF * 3);
      }
    }

    const size_t remaining = num - i;
    HWY_DASSERT(remaining < 4 * NF);
    if (HWY_UNLIKELY(remaining != 0)) {
      const V8 packed = hn::LoadN(d8, base + i, remaining);
      VF f0, f1, f2, f3;
      Dec4F(df, packed, f0, f1, f2, f3);
      // We are only guaranteed one vector of padding, so cannot unconditionally
      // store four vectors. `StoreN` would work, at the cost of saturated
      // subtraction and creating masks. Because we know that `raw_f` is padded
      // to at least one vector, we can instead store entire vectors and only
      // make the address conditional, which potentially avoids branches.
      // Separate per-vector storage may avoid conflicts.
      HWY_ALIGN float buf[4 * hn::MaxLanes(df)];
      hn::StoreU(f0, df, raw_f + i);
      hn::StoreU(f1, df, (remaining > 1 * NF ? (raw_f + i) : buf) + 1 * NF);
      hn::StoreU(f2, df, (remaining > 2 * NF ? (raw_f + i) : buf) + 2 * NF);
      hn::StoreU(f3, df, (remaining > 3 * NF ? (raw_f + i) : buf) + 3 * NF);
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
  static HWY_INLINE V8 Enc2B(DBF dbf, const BF16* HWY_RESTRICT in) {
    const hn::Repartition<uint16_t, DBF> d16;
    const size_t N16 = hn::Lanes(d16);
    using V16 = hn::Vec<decltype(d16)>;

    const V16 w0 = hn::BitCast(d16, hn::LoadU(dbf, in));
    const V16 w1 = hn::BitCast(d16, hn::LoadU(dbf, in + N16));
    return Enc2U(d16, w0, w1);
  }

  // Truncates two f32 to bf16, in lane order, without rounding (see Enc4F).
  template <class DBF, class DF = hn::RepartitionToWide<DBF>>
  static HWY_INLINE hn::Vec<DBF> Truncate2To(DBF dbf, hn::Vec<DF> f0,
                                             hn::Vec<DF> f1) {
    const hn::RebindToUnsigned<DBF> d16;
    using V16 = hn::Vec<decltype(d16)>;
    const V16 u0 = BitCast(d16, f0);
    const V16 u1 = BitCast(d16, f1);
    return BitCast(DBF(), HWY_IS_LITTLE_ENDIAN ? ConcatOdd(d16, u1, u0)
                                               : ConcatEven(d16, u1, u0));
  }

  template <class DF, HWY_IF_F32_D(DF),
            class V8 = hn::Vec<hn::Repartition<uint8_t, DF>>>
  static HWY_INLINE V8 Enc4F(DF df, const float* HWY_RESTRICT in) {
    const hn::Repartition<uint16_t, DF> d16;
    const hn::Repartition<BF16, DF> dbf;
    using VF = hn::Vec<decltype(df)>;
    using V16 = hn::Vec<decltype(d16)>;
    const size_t NF = hn::Lanes(df);

    const VF f0 = hn::LoadU(df, in + NF * 0);
    const VF f1 = hn::LoadU(df, in + NF * 1);
    const VF f2 = hn::LoadU(df, in + NF * 2);
    const VF f3 = hn::LoadU(df, in + NF * 3);
    // Chop off the lower 16 bits instead of OrderedDemote2To, which rounds to
    // the nearest bf16, because EncBytes will round again.
    const V16 w0 = hn::BitCast(d16, Truncate2To(dbf, f0, f1));
    const V16 w1 = hn::BitCast(d16, Truncate2To(dbf, f2, f3));
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
    const hn::Repartition<BF16, DF> dbf;
    using VBF = hn::Vec<decltype(dbf)>;
    VBF bf0, bf1;
    Dec2B(dbf, packed, bf0, bf1);
    f0 = hn::PromoteLowerTo(df, bf0);
    f1 = hn::PromoteUpperTo(df, bf0);
    f2 = hn::PromoteLowerTo(df, bf1);
    f3 = hn::PromoteUpperTo(df, bf1);
  }

  // TODO: currently unused, but keep for potential later MatMul packing.
  template <class DBF, HWY_IF_BF16_D(DBF),
            class V8 = hn::Vec<hn::Repartition<uint8_t, DBF>>>
  static HWY_INLINE void DecEvenOdd(DBF dbf, V8 packed, hn::Vec<DBF>& even,
                                    hn::Vec<DBF>& odd) {
    const hn::Repartition<uint8_t, DBF> d8;
    V8 lo, hi;
    DecBytes(d8, packed, lo, hi);
    // (Supported since Highway 1.2)
    even = hn::BitCast(dbf, hn::InterleaveEven(d8, lo, hi));
    odd = hn::BitCast(dbf, hn::InterleaveOdd(d8, lo, hi));
  }

  template <class DF, HWY_IF_F32_D(DF),
            class V8 = hn::Vec<hn::Repartition<uint8_t, DF>>>
  static HWY_INLINE void DecEvenOddF(DF df, V8 packed, hn::Vec<DF>& even0,
                                     hn::Vec<DF>& odd0, hn::Vec<DF>& even1,
                                     hn::Vec<DF>& odd1) {
    const hn::Repartition<BF16, DF> dbf;
    using VBF = hn::Vec<decltype(dbf)>;
    VBF even_bf, odd_bf;
    DecEvenOdd(dbf, packed, even_bf, odd_bf);
    even0 = hn::PromoteLowerTo(df, even_bf);
    odd0 = hn::PromoteLowerTo(df, odd_bf);
    even1 = hn::PromoteUpperTo(df, even_bf);
    odd1 = hn::PromoteUpperTo(df, odd_bf);
  }
};  // SfpCodec

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_SFP_INL_H_
