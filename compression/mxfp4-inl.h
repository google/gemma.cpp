// Copyright 2026 Google LLC
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

#ifndef THIRD_PARTY_GEMMA_CPP_COMPRESSION_MXFP4_INL_H_
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_MXFP4_INL_H_

#include <stddef.h>
#include <stdint.h>

#include <cmath>

#include "compression/types.h"
#include "hwy/base.h"

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_MXFP4_INL_H_

#if defined(THIRD_PARTY_GEMMA_CPP_COMPRESSION_MXFP4_INL_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_COMPRESSION_MXFP4_INL_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_COMPRESSION_MXFP4_INL_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_MXFP4_INL_TOGGLE
#endif

#include "hwy/contrib/math/math-inl.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

class MxFp4Codec {
  static constexpr size_t kBlockSize = MxFp4Stream::kBlockSize;

  static HWY_INLINE size_t BlockByteOffset(size_t index) {
    return (index / kBlockSize) * sizeof(MxFp4Stream);
  }

  static HWY_INLINE float DecodeE8M0ScaleToF32(uint8_t e8m0_scale) {
    if (e8m0_scale == 0xFF || e8m0_scale == 0) {
      return 0.0f;
    }
    const uint32_t bits = static_cast<uint32_t>(e8m0_scale) << 23;
    return hwy::BitCastScalar<float>(bits);
  }

  static HWY_INLINE uint8_t EncodeNibble(float v) {
    HWY_ALIGN static constexpr uint8_t kEncodeLut[32] = {
        0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6,
        6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
    uint8_t sign = 0;
    if (v < 0.0f) {
      sign = 0x8;
      v = -v;
    }
    const size_t lut_idx = HWY_MIN(size_t{31}, static_cast<size_t>(v * 4.0f));
    return sign | kEncodeLut[lut_idx];
  }

  // BF16 bytes for E2M1 * 2^0 (row 0, even exp_delta) and E2M1 * 2^1 (row 1,
  // odd exp_delta). Selecting row (exp_delta & 1) absorbs the bit-7 exponent
  // LSB carry so scaling adds (exp_delta >> 1) directly to the high byte.
  alignas(16) static constexpr uint8_t kLutLo[2][16] = {
      {0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0,
       0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0},
      {0x00, 0x80, 0x00, 0x40, 0x80, 0xC0, 0x00, 0x40,
       0x00, 0x80, 0x00, 0x40, 0x80, 0xC0, 0x00, 0x40},
  };
  alignas(16) static constexpr uint8_t kLutHi[2][16] = {
      {0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40,
       0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0},
      {0x00, 0x3F, 0x40, 0x40, 0x40, 0x40, 0x41, 0x41,
       0x80, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0, 0xC1, 0xC1},
  };

  template <class DU8H, class DF, HWY_IF_F32_D(DF),
            class VU8H = hn::Vec<DU8H>, class VF = hn::Vec<DF>,
            class V8>
  static HWY_INLINE VF NibblesToF32(DF df, DU8H du8h,
                                    const VU8H& nibbles, const VF& vd,
                                    const V8& lut_lo, const V8& lut_hi) {
    using DU8_4X = hn::Repartition<uint8_t, DF>;
    using DBF = hn::Rebind<BF16, DF>;
    using DU8_2X = hn::Repartition<uint8_t, DBF>;
    using VBF = hn::Vec<DBF>;

    const DU8_4X du8_4x;
    const DU8_2X du8_2x;
    const auto nibbles_full = hn::ResizeBitCast(du8_4x, nibbles);
    const auto lo_full = hn::TableLookupBytes(lut_lo, nibbles_full);
    const auto hi_full = hn::TableLookupBytes(lut_hi, nibbles_full);

    const auto lo_2x = hn::ResizeBitCast(du8_2x, lo_full);
    const auto hi_2x = hn::ResizeBitCast(du8_2x, hi_full);

    const auto combined_2x = hn::InterleaveWholeLower(du8_2x, lo_2x, hi_2x);

    const DBF dbf;
    const VBF bf_vec = hn::BitCast(dbf, combined_2x);
    const VF f_vec = hn::PromoteTo(df, bf_vec);
    return hn::Mul(f_vec, vd);
  }

  template <class DF, class DU8H, class VF = hn::Vec<DF>, class V8>
  static HWY_INLINE void Dequantize2Lanes(
      DF df, DU8H du8h, const uint8_t* HWY_RESTRICT qs_ptr, size_t block_ofs,
      const VF& vd, const V8& lut_lo, const V8& lut_hi, VF& raw0, VF& raw1) {
    using VU8H = hn::Vec<DU8H>;
    VU8H nibbles0, nibbles1;
#if HWY_HAVE_CONSTEXPR_LANES
    if constexpr (hn::Lanes(df) >= 16) {
#else
    if (hn::Lanes(df) >= 16) {
#endif
      const VU8H raw_bytes = hn::LoadU(du8h, qs_ptr);
      nibbles0 = hn::And(raw_bytes, hn::Set(du8h, 0x0F));
      nibbles1 = hn::ShiftRight<4>(raw_bytes);
    } else {
      const bool upper = (block_ofs >= 16);
      const size_t byte_ofs = upper ? (block_ofs - 16) : block_ofs;
      const VU8H bytes0 = hn::LoadU(du8h, qs_ptr + byte_ofs);
      const VU8H bytes1 =
          hn::LoadU(du8h, qs_ptr + byte_ofs + hn::Lanes(df));
      if (upper) {
        nibbles0 = hn::ShiftRight<4>(bytes0);
        nibbles1 = hn::ShiftRight<4>(bytes1);
      } else {
        const VU8H mask = hn::Set(du8h, 0x0F);
        nibbles0 = hn::And(bytes0, mask);
        nibbles1 = hn::And(bytes1, mask);
      }
    }
    raw0 = NibblesToF32(df, du8h, nibbles0, vd, lut_lo, lut_hi);
    raw1 = NibblesToF32(df, du8h, nibbles1, vd, lut_lo, lut_hi);
  }

  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void DequantizeBlock(DF df,
                                         const uint8_t* HWY_RESTRICT block_ptr,
                                         float* HWY_RESTRICT raw) {
    using DU8H = hn::Rebind<uint8_t, DF>;
    using DU8 = hn::Repartition<uint8_t, DF>;
    using VU8 = hn::Vec<DU8>;
    using VF = hn::Vec<DF>;

    const DU8H du8h;
    const DU8 du8;
    const VU8 lut_lo = hn::LoadDup128(du8, kLutLo[0]);
    const VU8 lut_hi = hn::LoadDup128(du8, kLutHi[0]);
    HWY_LANES_CONSTEXPR size_t NF = hn::Lanes(df);
    const uint8_t e8m0 = block_ptr[0];
    const float scale_f = DecodeE8M0ScaleToF32(e8m0);
    const VF vd = hn::Set(df, scale_f);

    const uint8_t* qs_ptr = block_ptr + 1;
    const size_t num_vectors = 32 / (2 * NF);

    for (size_t v_idx = 0; v_idx < num_vectors; ++v_idx) {
      VF out0, out1;
      Dequantize2Lanes(df, du8h, qs_ptr, v_idx * 2 * NF, vd, lut_lo, lut_hi,
                       out0, out1);
      hn::StoreU(out0, df, raw + v_idx * 2 * NF);
      hn::StoreU(out1, df, raw + v_idx * 2 * NF + NF);
    }
  }

  template <class DBF, HWY_IF_BF16_D(DBF)>
  static HWY_INLINE void DequantizeBlock(DBF dbf,
                                         const uint8_t* HWY_RESTRICT block_ptr,
                                         BF16* HWY_RESTRICT raw) {
    const uint8_t e8m0 = block_ptr[0];
#if HWY_IS_LITTLE_ENDIAN && !HWY_HAVE_SCALABLE
    // For e8m0 in [1, 252], normal BF16 exponents stay in [1, 254] and the
    // single subnormal magnitude (0.5 * 2^-126 = 2^-127) flushes to signed
    // zero, matching hardware VCVTNE2PS2BF16. Scales {0, 253, 254, 255} require
    // zero/Inf handling and fall through to the F32 path below.
    if (HWY_LIKELY(static_cast<uint8_t>(e8m0 - 1u) < 252u)) {
#if HWY_TARGET <= HWY_AVX3
      if constexpr (hn::MaxLanes(dbf) == 32) {
        HWY_ALIGN static constexpr uint16_t kLutBaseBF16[32] = {
            0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
            0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0,
            0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
            0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0,
        };

        HWY_ALIGN static constexpr uint16_t kNonZeroMaskBF16[32] = {
            0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
            0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
            0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
            0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        };

        using DU16x16 = hn::FixedTag<uint16_t, 16>;
        using DU16x32 = hn::FixedTag<uint16_t, 32>;
        using DU8x16 = hn::FixedTag<uint8_t, 16>;

        const DU16x16 du16x16;
        const DU16x32 du16x32;
        const DU8x16 du8x16;

        const uint16_t delta_bits =
            static_cast<uint16_t>((static_cast<uint32_t>(e8m0) - 127u) << 7);
        const auto delta_vec = hn::And(hn::Set(du16x32, delta_bits),
                                       hn::Load(du16x32, kNonZeroMaskBF16));
        const auto lut_u16 =
            hn::Add(hn::Load(du16x32, kLutBaseBF16), delta_vec);

        const uint8_t* qs_ptr = block_ptr + 1;
        const auto raw16 = hn::LoadU(du8x16, qs_ptr);
        const auto lo_nibs = hn::And(raw16, hn::Set(du8x16, 0x0F));
        const auto hi_nibs = hn::ShiftRight<4>(raw16);

        const auto idx_lo = hn::PromoteTo(du16x16, lo_nibs);
        const auto idx_hi = hn::PromoteTo(du16x16, hi_nibs);
        const auto idx512 = hn::Combine(du16x32, idx_hi, idx_lo);

        const auto w_u16 =
            hn::TableLookupLanes(lut_u16, hn::IndicesFromVec(du16x32, idx512));
        hn::StoreU(hn::BitCast(dbf, w_u16), dbf, raw);
        return;
      }
#endif
      if constexpr (hn::MaxLanes(dbf) == 16 || hn::MaxLanes(dbf) == 8) {
        alignas(16) static constexpr uint8_t kNonZeroMask8[16] = {
            0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        };
        const hn::Repartition<uint8_t, DBF> du8;
        const size_t odd = (e8m0 ^ 1u) & 1u;
        const uint8_t exp_hi_delta =
            static_cast<uint8_t>(((e8m0 - 1u) >> 1) - 63u);
        const auto lut_lo = hn::LoadDup128(du8, kLutLo[odd]);
        const auto lut_hi = hn::Add(
            hn::LoadDup128(du8, kLutHi[odd]),
            hn::And(hn::Set(du8, exp_hi_delta),
                    hn::LoadDup128(du8, kNonZeroMask8)));

        if constexpr (hn::MaxLanes(dbf) == 16) {
          const hn::Half<decltype(du8)> du8x16;
          const hn::Repartition<uint64_t, decltype(du8x16)> du64x2;
          const auto raw16 = hn::LoadU(du8x16, block_ptr + 1);
          const auto lo64 =
              hn::BitCast(du64x2, hn::And(raw16, hn::Set(du8x16, 0x0F)));
          const auto hi64 = hn::BitCast(du64x2, hn::ShiftRight<4>(raw16));
          // Per-block InterleaveLower/Upper is safe without InterleaveWhole*:
          // du64x2 (Half<du8>) is 128-bit (one block), and Combine places
          // [lo[0..7], hi[0..7]] in block 0 and [lo[8..15], hi[8..15]] in
          // block 1 of 256-bit du8, so per-block byte interleaves below emit
          // contiguous BF16[0..15] and BF16[16..31] without cross-lane
          // permutes.
          const auto pair0 =
              hn::BitCast(du8x16, hn::InterleaveLower(du64x2, lo64, hi64));
          const auto pair1 =
              hn::BitCast(du8x16, hn::InterleaveUpper(du64x2, lo64, hi64));
          const auto nibs = hn::Combine(du8, pair1, pair0);

          const auto b_lo = hn::TableLookupBytes(lut_lo, nibs);
          const auto b_hi = hn::TableLookupBytes(lut_hi, nibs);
          hn::StoreU(hn::BitCast(dbf, hn::InterleaveLower(du8, b_lo, b_hi)),
                     dbf, raw);
          hn::StoreU(hn::BitCast(dbf, hn::InterleaveUpper(du8, b_lo, b_hi)),
                     dbf, raw + 16);
        } else {
          // MaxLanes(dbf) == 8: du8 is 128-bit (a single block), where
          // InterleaveLower/Upper and InterleaveWholeLower/Upper are identical.
          const auto raw16 = hn::LoadU(du8, block_ptr + 1);
          const auto lo_nibs = hn::And(raw16, hn::Set(du8, 0x0F));
          const auto hi_nibs = hn::ShiftRight<4>(raw16);

          const auto b_lo0 = hn::TableLookupBytes(lut_lo, lo_nibs);
          const auto b_hi0 = hn::TableLookupBytes(lut_hi, lo_nibs);
          hn::StoreU(hn::BitCast(dbf, hn::InterleaveLower(du8, b_lo0, b_hi0)),
                     dbf, raw);
          hn::StoreU(hn::BitCast(dbf, hn::InterleaveUpper(du8, b_lo0, b_hi0)),
                     dbf, raw + 8);

          const auto b_lo1 = hn::TableLookupBytes(lut_lo, hi_nibs);
          const auto b_hi1 = hn::TableLookupBytes(lut_hi, hi_nibs);
          hn::StoreU(hn::BitCast(dbf, hn::InterleaveLower(du8, b_lo1, b_hi1)),
                     dbf, raw + 16);
          hn::StoreU(hn::BitCast(dbf, hn::InterleaveUpper(du8, b_lo1, b_hi1)),
                     dbf, raw + 24);
        }
        return;
      }
    }
#endif
    using DF = hn::Repartition<float, DBF>;
    using VF = hn::Vec<DF>;
    using DU8H = hn::Rebind<uint8_t, DF>;
    using DU8 = hn::Repartition<uint8_t, DF>;
    using VU8 = hn::Vec<DU8>;
    using VBF = hn::Vec<DBF>;

    const DF df;
    const DU8H du8h;
    const DU8 du8;
    const VU8 lut_lo = hn::LoadDup128(du8, kLutLo[0]);
    const VU8 lut_hi = hn::LoadDup128(du8, kLutHi[0]);
    HWY_LANES_CONSTEXPR size_t NF = hn::Lanes(df);
    const float scale_f = DecodeE8M0ScaleToF32(e8m0);
    const VF vd = hn::Set(df, scale_f);

    const uint8_t* qs_ptr = block_ptr + 1;
    const size_t num_vectors = 32 / (2 * NF);

    for (size_t v_idx = 0; v_idx < num_vectors; ++v_idx) {
      VF out0, out1;
      Dequantize2Lanes(df, du8h, qs_ptr, v_idx * 2 * NF, vd, lut_lo, lut_hi,
                       out0, out1);
      const VBF out_bf = hn::OrderedDemote2To(dbf, out0, out1);
      hn::StoreU(out_bf, dbf, raw + v_idx * 2 * NF);
    }
  }

  template <class DF, class VF = hn::Vec<DF>>
  static HWY_INLINE void QuantizeBlock(DF df, const float* HWY_RESTRICT raw,
                                       uint8_t* HWY_RESTRICT block_ptr) {
    VF max_vec = hn::Zero(df);
    for (size_t i = 0; i < kBlockSize; i += hn::Lanes(df)) {
      max_vec = hn::Max(max_vec, hn::Abs(hn::LoadU(df, raw + i)));
    }
    const float max_abs = hn::ReduceMax(df, max_vec);

    if (max_abs == 0.0f) {
      hwy::ZeroBytes(block_ptr, sizeof(MxFp4Stream));
      return;
    }

    using D1 = hn::FixedTag<float, 1>;
    using V1 = hn::Vec<D1>;
    const D1 d1;
    const V1 val = hn::Set(d1, max_abs / 6.0f);
    int exp = static_cast<int>(hn::GetLane(hn::Ceil(hn::Log2(d1, val))));
    exp = HWY_MAX(-126, HWY_MIN(127, exp));
    block_ptr[0] = static_cast<uint8_t>(exp + 127);

    const float inv_scale = std::ldexp(1.0f, -exp);
    uint8_t* qs_ptr = block_ptr + 1;
    for (size_t c2 = 0; c2 < 16; ++c2) {
      const uint8_t lo = EncodeNibble(raw[c2] * inv_scale);
      const uint8_t hi = EncodeNibble(raw[c2 + 16] * inv_scale);
      qs_ptr[c2] = static_cast<uint8_t>(lo | (hi << 4));
    }
  }

 public:
  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void Enc(DF df, const float* HWY_RESTRICT raw,
                             const size_t num,
                             const PackedSpan<MxFp4Stream>& packed,
                             size_t packed_ofs) {
    HWY_DASSERT(packed_ofs % kBlockSize == 0);
    const size_t num_blocks = hwy::DivCeil(num, kBlockSize);

    uint8_t* block_ptr =
        reinterpret_cast<uint8_t*>(packed.ptr) + BlockByteOffset(packed_ofs);
    for (size_t b = 0; b < num_blocks; ++b) {
      const size_t block_num = HWY_MIN(num - b * kBlockSize, kBlockSize);

      if (block_num == kBlockSize) {
        QuantizeBlock(df, raw + b * kBlockSize, block_ptr);
      } else {
        HWY_ALIGN float temp[kBlockSize] = {};
        hwy::CopyBytes(raw + b * kBlockSize, temp, block_num * sizeof(float));
        QuantizeBlock(df, temp, block_ptr);
      }
      block_ptr += sizeof(MxFp4Stream);
    }
  }

  template <class DF, HWY_IF_F32_D(DF), class VF = hn::Vec<DF>>
  static HWY_INLINE void Dec2(DF df,
                              const PackedSpan<const MxFp4Stream>& packed,
                              const size_t packed_ofs, VF& raw0, VF& raw1) {
    using DU8H = hn::Rebind<uint8_t, DF>;
    using DU8 = hn::Repartition<uint8_t, DF>;
    using VU8 = hn::Vec<DU8>;

    const DU8H du8h;
    const DU8 du8;
    const VU8 lut_lo = hn::LoadDup128(du8, kLutLo[0]);
    const VU8 lut_hi = hn::LoadDup128(du8, kLutHi[0]);
    const uint8_t* block_ptr = reinterpret_cast<const uint8_t*>(packed.ptr) +
                               BlockByteOffset(packed_ofs);
    const uint8_t e8m0 = block_ptr[0];
    const float scale_f = DecodeE8M0ScaleToF32(e8m0);
    const VF vd = hn::Set(df, scale_f);

    const uint8_t* qs_ptr = block_ptr + 1;
    const size_t block_ofs = packed_ofs % kBlockSize;

    Dequantize2Lanes(df, du8h, qs_ptr, block_ofs, vd, lut_lo, lut_hi, raw0,
                     raw1);
  }

  template <class DBF, HWY_IF_BF16_D(DBF), class VBF = hn::Vec<DBF>>
  static HWY_INLINE void Dec2(DBF dbf,
                              const PackedSpan<const MxFp4Stream>& packed,
                              const size_t packed_ofs, VBF& raw0, VBF& raw1) {
    using DF = hn::Repartition<float, DBF>;
    using VF = hn::Vec<DF>;
    const DF df;
    HWY_LANES_CONSTEXPR size_t NF = hn::Lanes(df);
    VF raw0_f, raw1_f, raw2_f, raw3_f;
    Dec2(df, packed, packed_ofs + 0 * 2 * NF, raw0_f, raw1_f);
    Dec2(df, packed, packed_ofs + 1 * 2 * NF, raw2_f, raw3_f);
    raw0 = hn::OrderedDemote2To(dbf, raw0_f, raw1_f);
    raw1 = hn::OrderedDemote2To(dbf, raw2_f, raw3_f);
  }

  template <class D, typename Raw = hn::TFromD<D>>
  static HWY_INLINE void DecompressAndZeroPad(
      D d, const PackedSpan<const MxFp4Stream>& packed, size_t packed_ofs,
      Raw* HWY_RESTRICT raw, size_t num) {
    if (num == 0) return;

    HWY_LANES_CONSTEXPR size_t N = hn::Lanes(d);
    const size_t padded_num = hwy::RoundUpTo(num, N);
    if (padded_num > num) {
      hwy::ZeroBytes(raw + num, (padded_num - num) * sizeof(Raw));
    }

    size_t current_packed_ofs = packed_ofs;
    Raw* HWY_RESTRICT current_raw = raw;
    size_t num_to_decompress = num;

    if (size_t within_block = current_packed_ofs % kBlockSize;
        within_block != 0) {
      const size_t remaining_in_block = kBlockSize - within_block;
      const size_t num_in_first_block =
          HWY_MIN(num_to_decompress, remaining_in_block);

      const uint8_t* block_ptr = reinterpret_cast<const uint8_t*>(packed.ptr) +
                                 BlockByteOffset(current_packed_ofs);
      HWY_ALIGN Raw temp[kBlockSize];
      DequantizeBlock(d, block_ptr, temp);
      hwy::CopyBytes(temp + within_block, current_raw,
                     num_in_first_block * sizeof(Raw));

      current_packed_ofs += num_in_first_block;
      current_raw += num_in_first_block;
      num_to_decompress -= num_in_first_block;
    }

    if (num_to_decompress == 0) return;

    HWY_DASSERT(current_packed_ofs % kBlockSize == 0);

    const size_t num_full_blocks = num_to_decompress / kBlockSize;
    const uint8_t* block_ptr = reinterpret_cast<const uint8_t*>(packed.ptr) +
                               BlockByteOffset(current_packed_ofs);
    for (size_t b = 0; b < num_full_blocks; ++b) {
      DequantizeBlock(d, block_ptr, current_raw);
      block_ptr += sizeof(MxFp4Stream);
      current_raw += kBlockSize;
    }
    current_packed_ofs += num_full_blocks * kBlockSize;

    const size_t remaining = num_to_decompress % kBlockSize;
    if (remaining != 0) {
      HWY_ALIGN Raw temp[kBlockSize];
      DequantizeBlock(d, block_ptr, temp);
      hwy::CopyBytes(temp, current_raw, remaining * sizeof(Raw));
    }
  }
};

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_MXFP4_INL_TOGGLE
