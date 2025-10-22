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

// Normal include guard.
#ifndef THIRD_PARTY_GEMMA_CPP_COMPRESSION_INT_INL_H_
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_INT_INL_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <cstdint>
#include <cstdio>

#include "compression/types.h"
#include "util/basics.h"
#include "hwy/base.h"
#include "hwy/print-inl.h"

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_INT_INL_H_

// Actual per-target include guard.
#if defined(THIRD_PARTY_GEMMA_CPP_COMPRESSION_INT_INL_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_COMPRESSION_INT_INL_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_COMPRESSION_INT_INL_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_INT_INL_TOGGLE
#endif

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Encode/decode functions.
class IntCodec {
  using ScaleT = hwy::bfloat16_t;
  static constexpr size_t kGroupSize = I8Stream::kGroupSize;

  // Offset (in bytes) of a group's start for packed_ofs (in elements) within a
  // set of groups.
  static constexpr size_t GroupByteOffset(size_t packed_ofs) {
    const size_t kBytesPerGroup = (2 * sizeof(ScaleT)) + kGroupSize;
    return (packed_ofs / kGroupSize) * kBytesPerGroup;
  }

 public:
  template <class DBF, HWY_IF_BF16_D(DBF)>
  static HWY_INLINE void DequantizeGroup(
      DBF dbf, const PackedSpan<const I8Stream>& packed, size_t packed_ofs,
      hwy::bfloat16_t* HWY_RESTRICT raw, size_t num) {
    using T = ScaleT;
    const hn::ScalableTag<float> df;
    const hn::Rebind<int32_t, decltype(df)> di32;
    const hn::Rebind<int16_t, decltype(di32)> di16;
    const hn::Rebind<int8_t, decltype(di16)> di8;
    const hn::Twice<hn::Rebind<hwy::bfloat16_t, decltype(df)>> dbf16;

    const size_t N = hn::Lanes(di8);
    const size_t N16 = hn::Lanes(dbf16);
    using VI8 = hn::Vec<decltype(di8)>;
    using VF = hn::Vec<decltype(df)>;

    T inv_scale, zeropoint;
    hwy::CopyBytes(&packed.ptr->i + GroupByteOffset(packed_ofs), &inv_scale,
                   sizeof(T));
    hwy::CopyBytes(&packed.ptr->i + GroupByteOffset(packed_ofs) + sizeof(T),
                   &zeropoint, sizeof(T));

    float inv_scale_f = hwy::ConvertScalarTo<float>(inv_scale);
    float zeropoint_f = hwy::ConvertScalarTo<float>(zeropoint);

    VF inv_scale_vec = hn::Set(df, inv_scale_f);
    VF zeroscale_vec = hn::Set(df, -zeropoint_f * (inv_scale_f));

    // Then iterate over remainder of packed, extracting num / N vectors and
    // inserting into raw.
    const size_t g_num = HWY_MIN(num, kGroupSize);

    const size_t current_offset = GroupByteOffset(packed_ofs) +
                                  (2 * sizeof(T)) + (packed_ofs % kGroupSize);
    size_t i = 0;
    for (i = 0; i + 4 * N <= g_num; i += 4 * N) {
      const VI8 val0 =
          hn::LoadU(di8, &packed.ptr->i + current_offset + i + 0 * N);
      const VI8 val1 =
          hn::LoadU(di8, &packed.ptr->i + current_offset + i + 1 * N);
      const VI8 val2 =
          hn::LoadU(di8, &packed.ptr->i + current_offset + i + 2 * N);
      const VI8 val3 =
          hn::LoadU(di8, &packed.ptr->i + current_offset + i + 3 * N);

      const VF val0_f =
          hn::ConvertTo(df, hn::PromoteTo(di32, hn::PromoteTo(di16, val0)));
      const VF val1_f =
          hn::ConvertTo(df, hn::PromoteTo(di32, hn::PromoteTo(di16, val1)));
      const VF val2_f =
          hn::ConvertTo(df, hn::PromoteTo(di32, hn::PromoteTo(di16, val2)));
      const VF val3_f =
          hn::ConvertTo(df, hn::PromoteTo(di32, hn::PromoteTo(di16, val3)));

      VF dequantized_val0 = hn::MulAdd(inv_scale_vec, val0_f, zeroscale_vec);
      VF dequantized_val1 = hn::MulAdd(inv_scale_vec, val1_f, zeroscale_vec);
      VF dequantized_val2 = hn::MulAdd(inv_scale_vec, val2_f, zeroscale_vec);
      VF dequantized_val3 = hn::MulAdd(inv_scale_vec, val3_f, zeroscale_vec);

      hn::StoreU(
          hn::OrderedDemote2To(dbf16, dequantized_val0, dequantized_val1),
          dbf16, raw + i + 0 * N16);
      hn::StoreU(
          hn::OrderedDemote2To(dbf16, dequantized_val2, dequantized_val3),
          dbf16, raw + i + 1 * N16);
    }
    for (; i + N <= g_num; i += N) {
      const VI8 val0 = hn::LoadU(di8, &packed.ptr->i + current_offset + i);
      const VF val0_f =
          hn::ConvertTo(df, hn::PromoteTo(di32, hn::PromoteTo(di16, val0)));
      VF dequantized_val0 = hn::MulAdd(inv_scale_vec, val0_f, zeroscale_vec);
      const hn::Rebind<hwy::bfloat16_t, decltype(df)> dbf_half;
      hn::StoreU(hn::DemoteTo(dbf_half, dequantized_val0), dbf_half, raw + i);
    }
    if (i < g_num) {
      const size_t remaining = g_num - i;
      const VI8 val0 =
          hn::LoadN(di8, &packed.ptr->i + current_offset + i, remaining);
      const VF val0_f =
          hn::ConvertTo(df, hn::PromoteTo(di32, hn::PromoteTo(di16, val0)));
      VF dequantized_val0 = hn::MulAdd(inv_scale_vec, val0_f, zeroscale_vec);
      const hn::Rebind<hwy::bfloat16_t, decltype(df)> dbf_half;
      hn::StoreN(hn::DemoteTo(dbf_half, dequantized_val0), dbf_half, raw + i,
                 remaining);
    }
  }

  // Dequantizes `num` floats from `packed` into `raw`. `packed` points to
  // compressed storage and `packed_ofs` indicates the destination offset
  // within it, in number of elements. Scaling values are interleaved with int
  // values to allow for easier unpacking.
  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void DequantizeGroup(
      DF df, const PackedSpan<const I8Stream>& packed, size_t packed_ofs,
      float* HWY_RESTRICT raw, size_t num) {
    using T = ScaleT;
    const hn::Rebind<int32_t, decltype(df)> di32;
    const hn::Rebind<int16_t, decltype(di32)> di16;
    const hn::Rebind<int8_t, decltype(di16)> di8;

    const size_t N = hn::Lanes(di8);
    const size_t N32 = hn::Lanes(df);
    using VI8 = hn::Vec<decltype(di8)>;
    using VF = hn::Vec<decltype(df)>;

    // HWY_ASSERT(num % 2 * N == 0);

    // Load scale and zero point from the beginning - ensure correct pointer
    // offset.
    T inv_scale, zeropoint;
    hwy::CopyBytes(&packed.ptr->i + GroupByteOffset(packed_ofs), &inv_scale,
                   sizeof(T));
    hwy::CopyBytes(&packed.ptr->i + GroupByteOffset(packed_ofs) + sizeof(T),
                   &zeropoint, sizeof(T));

    float inv_scale_f = hwy::ConvertScalarTo<float>(inv_scale);
    float zeropoint_f = hwy::ConvertScalarTo<float>(zeropoint);

    VF inv_scale_vec = hn::Set(df, inv_scale_f);
    VF zeroscale_vec = hn::Set(df, -zeropoint_f * (inv_scale_f));

    // Then iterate over remainder of packed, extracting num / N vectors and
    // inserting into raw.
    const size_t g_num = HWY_MIN(num, kGroupSize);

    const size_t current_offset = GroupByteOffset(packed_ofs) +
                                  (2 * sizeof(T)) + (packed_ofs % kGroupSize);

    size_t i = 0;
    for (; i + 2 * N <= g_num; i += 2 * N) {
      const VI8 val0 =
          hn::LoadU(di8, &packed.ptr->i + current_offset + i + 0 * N);
      const VI8 val1 =
          hn::LoadU(di8, &packed.ptr->i + current_offset + i + 1 * N);

      const VF val0_f =
          hn::ConvertTo(df, hn::PromoteTo(di32, hn::PromoteTo(di16, val0)));
      const VF val1_f =
          hn::ConvertTo(df, hn::PromoteTo(di32, hn::PromoteTo(di16, val1)));

      VF dequantized_val0 = hn::MulAdd(inv_scale_vec, val0_f, zeroscale_vec);
      VF dequantized_val1 = hn::MulAdd(inv_scale_vec, val1_f, zeroscale_vec);

      hn::StoreU(dequantized_val0, df, raw + i + 0 * N32);
      hn::StoreU(dequantized_val1, df, raw + i + 1 * N32);
    }
    for (; i + N <= g_num; i += N) {
      const VI8 val0 = hn::LoadU(di8, &packed.ptr->i + current_offset + i);
      const VF val0_f =
          hn::ConvertTo(df, hn::PromoteTo(di32, hn::PromoteTo(di16, val0)));
      VF dequantized_val0 = hn::MulAdd(inv_scale_vec, val0_f, zeroscale_vec);
      hn::StoreU(dequantized_val0, df, raw + i);
    }
    if (i < g_num) {
      const size_t remaining = g_num - i;
      const VI8 val0 =
          hn::LoadN(di8, &packed.ptr->i + current_offset + i, remaining);
      const VF val0_f =
          hn::ConvertTo(df, hn::PromoteTo(di32, hn::PromoteTo(di16, val0)));
      VF dequantized_val0 = hn::MulAdd(inv_scale_vec, val0_f, zeroscale_vec);
      hn::StoreN(dequantized_val0, df, raw + i, remaining);
    }
  }

  // Quantizes `num` floats from `raw` into `packed`. `packed` points to
  // compressed storage and `packed_ofs` indicates the destination offset
  // within it, in number of elements. Scaling values are interleaved with
  // int values to allow for easier unpacking.
  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void QuantizeGroup(DF df, const float* HWY_RESTRICT raw,
                                       size_t num,
                                       const PackedSpan<I8Stream>& packed,
                                       size_t packed_ofs) {
    using T = ScaleT;
    const hn::Repartition<int32_t, DF> di32;
    const hn::Half<hn::Repartition<int16_t, decltype(di32)>> di16;
    const hn::Half<hn::Repartition<int8_t, decltype(di16)>> di8;

    const size_t N = hn::Lanes(df);
    using VI8 = hn::Vec<decltype(di8)>;
    using VF = hn::Vec<decltype(df)>;

    HWY_DASSERT(packed_ofs % kGroupSize == 0);
    HWY_DASSERT(num % 2 * N == 0);

    // Calculate min/max using SIMD
    float min_val = hwy::HighestValue<float>();
    float max_val = hwy::LowestValue<float>();
    VF vmin = hn::Set(df, hwy::HighestValue<float>());
    VF vmax = hn::Set(df, hwy::LowestValue<float>());

    size_t j = 0;
    for (; j + N <= num; j += N) {
      const VF xi = hn::LoadU(df, raw + j);
      vmin = hn::Min(vmin, xi);
      vmax = hn::Max(vmax, xi);
    }

    min_val = hn::ReduceMin(df, vmin);
    max_val = hn::ReduceMax(df, vmax);

    for (; j < num; ++j) {
      min_val = HWY_MIN(min_val, raw[j]);
      max_val = HWY_MAX(max_val, raw[j]);
    }

    // Calculate range, scale and zeropoint
    float x_range = max_val - min_val;
    x_range = x_range == 0.0f ? 1.0f : x_range;
    const float scale_f = 255.0f / x_range;
    const float zeropoint_f = static_cast<float>(
        static_cast<int32_t>(-scale_f * min_val - 128.0f));  // Correct casting

    const T scale = hwy::ConvertScalarTo<T>(scale_f);
    // inv_scale is used for all dequantization.
    const T inv_scale = hwy::ConvertScalarTo<T>(1.0f / scale_f);
    const T zeropoint = hwy::ConvertScalarTo<T>(zeropoint_f);
    memcpy(&packed.ptr->i + GroupByteOffset(packed_ofs), &inv_scale, sizeof(T));
    memcpy(&packed.ptr->i + GroupByteOffset(packed_ofs) + sizeof(T), &zeropoint,
           sizeof(T));

    const size_t g_num = HWY_MIN(num, kGroupSize);

    VF mul = hn::Set(df, hwy::ConvertScalarTo<float>(scale));
    VF add = hn::Set(df, hwy::ConvertScalarTo<float>(zeropoint));

    const size_t current_offset = GroupByteOffset(packed_ofs) +
                                  (2 * sizeof(T)) + (packed_ofs % kGroupSize);

    size_t i = 0;
    for (; i + 2 * N <= g_num; i += 2 * N) {
      const VI8 val0 = hn::DemoteTo(
          di8,
          hn::DemoteTo(di16, NearestInt(hn::MulAdd(
                                 mul, hn::LoadU(df, raw + i + 0 * N), add))));
      const VI8 val1 = hn::DemoteTo(
          di8,
          hn::DemoteTo(di16, NearestInt(hn::MulAdd(
                                 mul, hn::LoadU(df, raw + i + 1 * N), add))));

      hn::StoreU(val0, di8, &packed.ptr->i + current_offset + i + 0 * N);
      hn::StoreU(val1, di8, &packed.ptr->i + current_offset + i + 1 * N);
    }

    size_t remaining = g_num - i;

    HWY_DASSERT(remaining < 2 * N);
    if (HWY_UNLIKELY(remaining == 0)) return;

    if (remaining > N) {
      const VI8 val0 = hn::DemoteTo(
          di8, hn::DemoteTo(di16, NearestInt(hn::MulAdd(
                                      mul, hn::LoadU(df, raw + i), add))));
      hn::StoreU(val0, di8, &packed.ptr->i + current_offset + i);

      const size_t remaining1 = remaining - N;
      const VI8 val1 = hn::DemoteTo(
          di8,
          hn::DemoteTo(di16,
                       NearestInt(hn::MulAdd(
                           mul, hn::LoadN(df, raw + i + N, remaining1), add))));
      hn::StoreN(val1, di8, &packed.ptr->i + current_offset + i + N,
                 remaining1);
    } else {  // remaining <= N
      const VI8 val0 = hn::DemoteTo(
          di8, hn::DemoteTo(di16,
                            NearestInt(hn::MulAdd(
                                mul, hn::LoadN(df, raw + i, remaining), add))));
      hn::StoreN(val0, di8, &packed.ptr->i + current_offset + i, remaining);
    }
  }

  // Encodes `num` floats from `raw` into `packed`. `packed` points to
  // compressed storage and `packed_ofs` indicates the destination offset
  // within it, in number of elements. Scaling values are interleaved with
  // int
  // values to allow for easier unpacking.
  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void Enc(DF df, const float* HWY_RESTRICT raw,
                             const size_t num,
                             const PackedSpan<I8Stream>& packed,
                             size_t packed_ofs) {
    HWY_ASSERT(packed_ofs % kGroupSize == 0);

    const size_t num_groups = hwy::DivCeil(num, kGroupSize);

    size_t current_offset = packed_ofs;
    for (size_t g = 0; g < num_groups; ++g) {
      const size_t g_num = HWY_MIN(num - g * kGroupSize, kGroupSize);
      const float* HWY_RESTRICT g_in = raw + g * kGroupSize;

      QuantizeGroup(df, g_in, g_num, packed, current_offset);
      current_offset += g_num;
    }
  }

  // Decompresses to two bf16 vectors. `packed_ofs` must be a multiple of two
  // vectors so that we only have to load one group's table.
  template <class DBF, HWY_IF_BF16_D(DBF)>
  static HWY_INLINE void Dec2(DBF dbf, const PackedSpan<const I8Stream>& packed,
                              const size_t packed_ofs, hn::Vec<DBF>& raw0,
                              hn::Vec<DBF>& raw1) {
    const hn::Repartition<float, decltype(dbf)> df;
    using VF = hn::Vec<decltype(df)>;
    const size_t NF = hn::Lanes(df);

    HWY_ASSERT(packed_ofs % 2 * NF == 0);

    VF raw0_f, raw1_f, raw2_f, raw3_f;
    Dec2(df, packed, packed_ofs + 0 * 2 * NF, raw0_f, raw1_f);
    Dec2(df, packed, packed_ofs + 1 * 2 * NF, raw2_f, raw3_f);

    raw0 = hn::OrderedDemote2To(dbf, raw0_f, raw1_f);
    raw1 = hn::OrderedDemote2To(dbf, raw2_f, raw3_f);
  }

  // Decompresses to two f32 vectors. `packed_ofs` must be a multiple of two
  // vectors so that we only have to load one group's table.
  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void Dec2(DF df, const PackedSpan<const I8Stream>& packed,
                              const size_t packed_ofs, hn::Vec<DF>& raw0,
                              hn::Vec<DF>& raw1) {
    using T = ScaleT;
    const hn::Rebind<int32_t, decltype(df)> di32;
    const hn::Rebind<int16_t, decltype(di32)> di16;
    const hn::Rebind<int8_t, decltype(di16)> di8;

    const size_t N = hn::Lanes(di8);
    using VI8 = hn::Vec<decltype(di8)>;
    using VF = hn::Vec<decltype(df)>;

    T inv_scale, zeropoint;
    hwy::CopyBytes(&packed.ptr->i + GroupByteOffset(packed_ofs), &inv_scale,
                   sizeof(T));
    hwy::CopyBytes(&packed.ptr->i + GroupByteOffset(packed_ofs) + sizeof(T),
                   &zeropoint, sizeof(T));

    float inv_scale_f = hwy::ConvertScalarTo<float>(inv_scale);
    float zeropoint_f = hwy::ConvertScalarTo<float>(zeropoint);

    VF inv_scale_vec = hn::Set(df, inv_scale_f);
    VF zeroscale_vec = hn::Set(df, -zeropoint_f * (inv_scale_f));

    const size_t current_offset = GroupByteOffset(packed_ofs) +
                                  (2 * sizeof(T)) + (packed_ofs % kGroupSize);

    const VI8 val0 = hn::LoadU(di8, &packed.ptr->i + current_offset + 0 * N);
    const VI8 val1 = hn::LoadU(di8, &packed.ptr->i + current_offset + 1 * N);

    const VF val0_f =
        hn::ConvertTo(df, hn::PromoteTo(di32, hn::PromoteTo(di16, val0)));
    const VF val1_f =
        hn::ConvertTo(df, hn::PromoteTo(di32, hn::PromoteTo(di16, val1)));

    raw0 = hn::MulAdd(inv_scale_vec, val0_f, zeroscale_vec);
    raw1 = hn::MulAdd(inv_scale_vec, val1_f, zeroscale_vec);
  }

  template <class D, typename Raw = hn::TFromD<D>>
  static HWY_INLINE void DecompressAndZeroPad(
      D d, const PackedSpan<const I8Stream>& packed, size_t packed_ofs,
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

    if (size_t within_group = current_packed_ofs % kGroupSize;
        within_group != 0) {
      const size_t remaining_in_group = kGroupSize - within_group;
      const size_t num_in_first_group =
          HWY_MIN(num_to_decompress, remaining_in_group);
      DequantizeGroup(d, packed, current_packed_ofs, current_raw,
                      num_in_first_group);
      current_packed_ofs += num_in_first_group;
      current_raw += num_in_first_group;
      num_to_decompress -= num_in_first_group;
    }

    if (num_to_decompress == 0) return;

    HWY_DASSERT(current_packed_ofs % kGroupSize == 0);

    const size_t num_full_groups = num_to_decompress / kGroupSize;
    for (size_t g = 0; g < num_full_groups; ++g) {
      DequantizeGroup(d, packed, current_packed_ofs, current_raw, kGroupSize);
      current_packed_ofs += kGroupSize;
      current_raw += kGroupSize;
    }

    const size_t remaining = num_to_decompress % kGroupSize;
    if (remaining != 0) {
      DequantizeGroup(d, packed, current_packed_ofs, current_raw, remaining);
    }
  }
};  // IntCodec

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_INT_INL_H_
