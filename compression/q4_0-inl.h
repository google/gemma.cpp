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

#ifndef THIRD_PARTY_GEMMA_CPP_COMPRESSION_Q4_0_INL_H_
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_Q4_0_INL_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <cmath>

#include "compression/types.h"
#include "hwy/base.h"

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_Q4_0_INL_H_

#if defined(THIRD_PARTY_GEMMA_CPP_COMPRESSION_Q4_0_INL_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_COMPRESSION_Q4_0_INL_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_COMPRESSION_Q4_0_INL_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_Q4_0_INL_TOGGLE
#endif

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

class Q4_0Codec {
  using ScaleT = hwy::bfloat16_t;
  static constexpr size_t kBlockSize = 32;

  static constexpr size_t BlockByteOffset(size_t packed_ofs) {
    const size_t kBytesPerBlock = sizeof(ScaleT) + kBlockSize / 2;
    return (packed_ofs / kBlockSize) * kBytesPerBlock;
  }

  template <class DF, class VF = hn::Vec<DF>>
  static HWY_INLINE void StoreRaw(DF df, VF val_f, float* HWY_RESTRICT raw) {
    hn::StoreU(val_f, df, raw);
  }

  template <class DF, class VF = hn::Vec<DF>>
  static HWY_INLINE void StoreRaw(DF df, VF val_f, hwy::bfloat16_t* HWY_RESTRICT raw) {
    const hn::Rebind<hwy::bfloat16_t, DF> dbf;
    hn::StoreU(hn::DemoteTo(dbf, val_f), dbf, raw);
  }

  template <class DU8, class DI8, class DF, class VF = hn::Vec<DF>>
  static HWY_INLINE VF DequantizeLanes(DF df, DU8 du8, DI8 di8,
                                       const uint8_t* qs_ptr, size_t block_ofs,
                                       size_t lane_ofs, size_t N, VF vd) {
    const size_t pos = block_ofs + lane_ofs;
    const bool upper = (pos >= 16);
    const size_t byte_ofs = upper ? (pos - 16) : pos;

    const auto raw_bytes = hn::LoadU(du8, qs_ptr + byte_ofs);
    const auto mask = hn::Set(du8, 0x0F);
    const auto offset = hn::Set(di8, 8);

    // Branchless: compute both paths, select via IfThenElse to avoid
    // branch misprediction when `upper` is not resolved at compile time.
    const auto shifted = hn::ShiftRight<4>(raw_bytes);
    const auto masked = hn::And(raw_bytes, mask);
    const auto vmask = hn::SetMask(du8, upper);
    auto nibbles = hn::IfThenElse(vmask, shifted, masked);

    auto signed_nibbles = hn::BitCast(di8, nibbles);
    signed_nibbles = hn::Sub(signed_nibbles, offset);

    const hn::Rebind<int16_t, decltype(di8)> di16;
    const hn::Rebind<int32_t, decltype(di16)> di32;
    const auto val_i16 = hn::PromoteTo(di16, signed_nibbles);
    const auto val_i32 = hn::PromoteTo(di32, val_i16);
    const auto val_f = hn::ConvertTo(df, val_i32);

    return hn::Mul(val_f, vd);
  }

  template <class D, typename Raw = hn::TFromD<D>>
  static HWY_INLINE void DequantizeBlock(D d, const uint8_t* HWY_RESTRICT block_ptr,
                                         Raw* HWY_RESTRICT raw) {
    const hn::Repartition<float, D> df;
    const hn::Rebind<int32_t, decltype(df)> di32;
    const hn::Rebind<int16_t, decltype(di32)> di16;
    const hn::Rebind<int8_t, decltype(di16)> di8;
    const hn::Rebind<uint8_t, decltype(di8)> du8;

    using T = ScaleT;
    T scale;
    hwy::CopyBytes(block_ptr, &scale, sizeof(T));
    const float scale_f = hwy::F32FromBF16(scale);
    const auto vd = hn::Set(df, scale_f);

    const uint8_t* qs_ptr = block_ptr + sizeof(T);
    const size_t N = hn::Lanes(df);
    const size_t num_vectors = 32 / N;

    for (size_t v_idx = 0; v_idx < num_vectors; ++v_idx) {
      const auto out = DequantizeLanes(df, du8, di8, qs_ptr, 0, v_idx * N, N, vd);
      StoreRaw(df, out, raw + v_idx * N);
    }
  }

  template <class DF, class VF = hn::Vec<DF>>
  static HWY_INLINE void QuantizeBlock(DF df, const float* HWY_RESTRICT raw,
                                       uint8_t* HWY_RESTRICT block_ptr) {
    float max_abs = 0.0f;
    float max_val = 0.0f;

    for (size_t i = 0; i < 32; ++i) {
      const float v = raw[i];
      if (std::abs(v) > max_abs) {
        max_abs = std::abs(v);
        max_val = v;
      }
    }

    const float d = max_val / -8.0f;
    const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

    const ScaleT scale = hwy::ConvertScalarTo<ScaleT>(d);
    hwy::CopyBytes(&scale, block_ptr, sizeof(ScaleT));

    uint8_t* qs_ptr = block_ptr + sizeof(ScaleT);

    for (size_t j = 0; j < 16; ++j) {
      const float x0 = raw[j] * id;
      const float x1 = raw[j + 16] * id;

      const int32_t xi0 = std::min<int32_t>(
          15, std::max<int32_t>(0, static_cast<int32_t>(std::round(x0 + 8.0f))));
      const int32_t xi1 = std::min<int32_t>(
          15, std::max<int32_t>(0, static_cast<int32_t>(std::round(x1 + 8.0f))));

      qs_ptr[j] = static_cast<uint8_t>(xi0 | (xi1 << 4));
    }
  }

 public:
  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void Enc(DF df, const float* HWY_RESTRICT raw,
                             const size_t num,
                             const PackedSpan<Q4_0Stream>& packed,
                             size_t packed_ofs) {
    HWY_DASSERT(packed_ofs % kBlockSize == 0);
    const size_t num_blocks = hwy::DivCeil(num, kBlockSize);

    for (size_t b = 0; b < num_blocks; ++b) {
      const size_t current_packed_ofs = packed_ofs + b * kBlockSize;
      uint8_t* block_ptr =
          &packed.ptr->byte + BlockByteOffset(current_packed_ofs);
      const size_t block_num = HWY_MIN(num - b * kBlockSize, kBlockSize);

      if (block_num == kBlockSize) {
        QuantizeBlock(df, raw + b * kBlockSize, block_ptr);
      } else {
        float temp[kBlockSize] = {};
        memcpy(temp, raw + b * kBlockSize, block_num * sizeof(float));
        QuantizeBlock(df, temp, block_ptr);
      }
    }
  }

  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void Dec2(DF df, const PackedSpan<const Q4_0Stream>& packed,
                              const size_t packed_ofs, hn::Vec<DF>& raw0,
                              hn::Vec<DF>& raw1) {
    const hn::Rebind<int32_t, DF> di32;
    const hn::Rebind<int16_t, decltype(di32)> di16;
    const hn::Rebind<int8_t, decltype(di16)> di8;
    const hn::Rebind<uint8_t, decltype(di8)> du8;

    using T = ScaleT;
    const size_t N = hn::Lanes(df);

    const uint8_t* block_ptr = &packed.ptr->byte + BlockByteOffset(packed_ofs);
    T scale;
    hwy::CopyBytes(block_ptr, &scale, sizeof(T));
    const float scale_f = hwy::F32FromBF16(scale);
    const auto vd = hn::Set(df, scale_f);

    const uint8_t* qs_ptr = block_ptr + sizeof(T);
    const size_t block_ofs = packed_ofs % kBlockSize;

    raw0 = DequantizeLanes(df, du8, di8, qs_ptr, block_ofs, 0, N, vd);
    raw1 = DequantizeLanes(df, du8, di8, qs_ptr, block_ofs, N, N, vd);
  }

  template <class DBF, HWY_IF_BF16_D(DBF)>
  static HWY_INLINE void Dec2(DBF dbf, const PackedSpan<const Q4_0Stream>& packed,
                              const size_t packed_ofs, hn::Vec<DBF>& raw0,
                              hn::Vec<DBF>& raw1) {
    const hn::Repartition<float, decltype(dbf)> df;
    using VF = hn::Vec<decltype(df)>;
    const size_t NF = hn::Lanes(df);

    VF raw0_f, raw1_f, raw2_f, raw3_f;
    Dec2(df, packed, packed_ofs + 0 * 2 * NF, raw0_f, raw1_f);
    Dec2(df, packed, packed_ofs + 1 * 2 * NF, raw2_f, raw3_f);

    raw0 = hn::OrderedDemote2To(dbf, raw0_f, raw1_f);
    raw1 = hn::OrderedDemote2To(dbf, raw2_f, raw3_f);
  }

  template <class D, typename Raw = hn::TFromD<D>>
  static HWY_INLINE void DecompressAndZeroPad(
      D d, const PackedSpan<const Q4_0Stream>& packed, size_t packed_ofs,
      Raw* HWY_RESTRICT raw, size_t num) {
    if (num == 0) return;

    const size_t N = hn::Lanes(d);
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

      const uint8_t* block_ptr =
          &packed.ptr->byte + BlockByteOffset(current_packed_ofs);
      HWY_ALIGN Raw temp[kBlockSize];
      DequantizeBlock(d, block_ptr, temp);
      memcpy(current_raw, temp + within_block,
             num_in_first_block * sizeof(Raw));

      current_packed_ofs += num_in_first_block;
      current_raw += num_in_first_block;
      num_to_decompress -= num_in_first_block;
    }

    if (num_to_decompress == 0) return;

    HWY_DASSERT(current_packed_ofs % kBlockSize == 0);

    const size_t num_full_blocks = num_to_decompress / kBlockSize;
    for (size_t b = 0; b < num_full_blocks; ++b) {
      const uint8_t* block_ptr =
          &packed.ptr->byte + BlockByteOffset(current_packed_ofs);
      DequantizeBlock(d, block_ptr, current_raw);
      current_packed_ofs += kBlockSize;
      current_raw += kBlockSize;
    }

    const size_t remaining = num_to_decompress % kBlockSize;
    if (remaining != 0) {
      const uint8_t* block_ptr =
          &packed.ptr->byte + BlockByteOffset(current_packed_ofs);
      HWY_ALIGN Raw temp[kBlockSize];
      DequantizeBlock(d, block_ptr, temp);
      memcpy(current_raw, temp, remaining * sizeof(Raw));
    }
  }
};

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_Q4_0_INL_TOGGLE
