// Copyright 2024 Google LLC
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

// Include guard for headers.
#ifndef THIRD_PARTY_GEMMA_CPP_COMPRESSION_COMPRESS_INL_H_
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_COMPRESS_INL_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <memory>
#include <vector>

#include "compression/compress.h"  // IWYU pragma: export
#include "compression/distortion.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/timer.h"

#if COMPRESS_STATS
#include <cmath>  // lroundf
#endif

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_COMPRESS_INL_H_

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_COMPRESS_TOGGLE) == defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_COMPRESS_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_COMPRESS_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_COMPRESS_TOGGLE
#endif

#include "hwy/highway.h"
// After highway.h
#include "compression/nuq-inl.h"
#include "compression/sfp-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

#ifdef HWY_IS_TEST
static constexpr bool kIsTest = true;
#else
static constexpr bool kIsTest = false;
#endif

// Enables generic code independent of compression type.
template <typename T>  // primary, must specialize
struct CompressTraits {};

// Used by MatMul for f32 weights or activations, if native
// `ReorderWidenMulAccumulate` is available.
template <>
struct CompressTraits<float> {
  using Packed = float;

  template <class DF, HWY_IF_F32_D(DF), class VF = hn::Vec<DF>>
  static HWY_INLINE void Compress(DF /*df*/, const float* HWY_RESTRICT raw,
                                  size_t num, CompressPerThread& /*tls*/,
                                  const PackedSpan<Packed>& packed,
                                  const size_t packed_ofs) {
    hwy::CopyBytes(raw, packed.ptr + packed_ofs, num * sizeof(raw[0]));
  }

  template <class DF, HWY_IF_F32_D(DF), class VF = hn::Vec<DF>>
  static void Store2(DF df, VF raw0, VF raw1, const PackedSpan<Packed>& packed,
                     const size_t packed_ofs) {
    const size_t NF = hn::Lanes(df);
    hn::StoreU(raw0, df, packed.ptr + packed_ofs);
    hn::StoreU(raw1, df, packed.ptr + packed_ofs + NF);
  }

  template <class DBF16, HWY_IF_BF16_D(DBF16), class VBF16 = hn::Vec<DBF16>>
  static HWY_INLINE void Load2(DBF16 dbf16,
                               const PackedSpan<const Packed>& packed,
                               const size_t packed_ofs, VBF16& raw0,
                               VBF16& raw1) {
    const hn::Repartition<float, decltype(dbf16)> df;
    using VF = hn::Vec<decltype(df)>;
    const size_t NF = hn::Lanes(df);
    const VF f0 = hn::LoadU(df, packed.ptr + packed_ofs + 0 * NF);
    const VF f1 = hn::LoadU(df, packed.ptr + packed_ofs + 1 * NF);
    const VF f2 = hn::LoadU(df, packed.ptr + packed_ofs + 2 * NF);
    const VF f3 = hn::LoadU(df, packed.ptr + packed_ofs + 3 * NF);
    raw0 = hn::OrderedDemote2To(dbf16, f0, f1);
    raw1 = hn::OrderedDemote2To(dbf16, f2, f3);
  }

  template <class DF, HWY_IF_F32_D(DF), class VF = hn::Vec<DF>>
  static HWY_INLINE void Load2(DF df, const PackedSpan<const Packed>& packed,
                               const size_t packed_ofs, VF& raw0, VF& raw1) {
    const size_t N = hn::Lanes(df);
    raw0 = hn::LoadU(df, packed.ptr + packed_ofs);
    raw1 = hn::LoadU(df, packed.ptr + packed_ofs + N);
  }

  template <class DD, HWY_IF_F64_D(DD), class VD = hn::Vec<DD>>
  static HWY_INLINE void Load2(DD dd, const PackedSpan<const Packed>& packed,
                               const size_t packed_ofs, VD& raw0, VD& raw1) {
    const hn::Rebind<float, DD> df;
    using VF = hn::Vec<decltype(df)>;
    const size_t NF = hn::Lanes(df);
    // Two half loads are likely cheaper than one full + UpperHalf.
    const VF f0 = hn::LoadU(df, packed.ptr + packed_ofs + 0 * NF);
    const VF f1 = hn::LoadU(df, packed.ptr + packed_ofs + 1 * NF);
    raw0 = hn::PromoteTo(dd, f0);
    raw1 = hn::PromoteTo(dd, f1);
  }

  template <class DBF, HWY_IF_BF16_D(DBF)>
  static HWY_INLINE void DecompressAndZeroPad(
      DBF dbf, const PackedSpan<const Packed>& packed, const size_t packed_ofs,
      BF16* HWY_RESTRICT raw, size_t num) {
    const hn::Repartition<float, decltype(dbf)> df;
    using VF = hn::Vec<decltype(df)>;
    const size_t NF = hn::Lanes(df);

    size_t i = 0;
    if (num >= 2 * NF) {
      for (; i <= num - 2 * NF; i += 2 * NF) {
        const VF f0 = hn::LoadU(df, packed.ptr + packed_ofs + i);
        const VF f1 = hn::LoadU(df, packed.ptr + packed_ofs + i + NF);
        hn::StoreU(hn::OrderedDemote2To(dbf, f0, f1), dbf, raw + i);
      }
    }
    const size_t remaining = num - i;
    HWY_DASSERT(remaining < 2 * NF);
    if (HWY_UNLIKELY(remaining != 0)) {
      const size_t remaining2 = remaining - HWY_MIN(remaining, NF);
      const VF f0 = hn::LoadN(df, packed.ptr + packed_ofs + i, remaining);
      const VF f1 = hn::LoadN(df, packed.ptr + packed_ofs + i + NF, remaining2);
      hn::StoreU(hn::OrderedDemote2To(dbf, f0, f1), dbf, raw + i);
    }
  }

  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void DecompressAndZeroPad(
      DF df, const PackedSpan<const Packed>& packed, const size_t packed_ofs,
      float* HWY_RESTRICT raw, size_t num) {
    using VF = hn::Vec<decltype(df)>;
    const size_t NF = hn::Lanes(df);

    size_t i = 0;
    if (num >= NF) {
      for (; i <= num - NF; i += NF) {
        const VF vf = hn::LoadU(df, packed.ptr + packed_ofs + i);
        hn::StoreU(vf, df, raw + i);
      }
    }
    const size_t remaining = num - i;
    HWY_DASSERT(remaining < NF);
    if (HWY_UNLIKELY(remaining != 0)) {
      const VF vf = hn::LoadN(df, packed.ptr + packed_ofs + i, remaining);
      hn::StoreU(vf, df, raw + i);  // adds zero padding
    }
  }

  template <class DD, HWY_IF_F64_D(DD)>
  static HWY_INLINE void DecompressAndZeroPad(
      DD dd, const PackedSpan<const Packed>& packed, const size_t packed_ofs,
      double* HWY_RESTRICT raw, size_t num) {
    const hn::Rebind<float, DD> df;
    using VF = hn::Vec<decltype(df)>;
    const size_t ND = hn::Lanes(dd);

    size_t i = 0;
    if (num >= ND) {
      for (; i <= num - ND; i += ND) {
        const VF vf = hn::LoadU(df, packed.ptr + packed_ofs + i);
        hn::StoreU(hn::PromoteTo(dd, vf), dd, raw + i);
      }
    }
    const size_t remaining = num - i;
    HWY_DASSERT(remaining < ND);
    if (HWY_UNLIKELY(remaining != 0)) {
      const VF vf = hn::LoadN(df, packed.ptr + packed_ofs + i, remaining);
      hn::StoreU(hn::PromoteTo(dd, vf), dd, raw + i);  // adds zero padding
    }
  }
};

template <>
struct CompressTraits<BF16> {
  using Packed = BF16;

  // Note: it is fine for the lower 16 mantissa bits of `raw` to be nonzero
  // because we round rather than truncate.
  template <class DF, HWY_IF_F32_D(DF), class VF = hn::Vec<DF>>
  static HWY_INLINE void Compress(DF df, const float* HWY_RESTRICT raw,
                                  size_t num, CompressPerThread& tls,
                                  const PackedSpan<Packed>& packed,
                                  const size_t packed_ofs) {
    const hn::Repartition<BF16, decltype(df)> dbf;
    const size_t NF = hn::Lanes(df);

    size_t i = 0;
    if (num >= 2 * NF) {
      for (; i <= num - 2 * NF; i += 2 * NF) {
        const VF raw0 = hn::LoadU(df, raw + i);
        const VF raw1 = hn::LoadU(df, raw + i + NF);

        hn::StoreU(hn::OrderedDemote2To(dbf, raw0, raw1), dbf,
                   packed.ptr + packed_ofs + i);

        if (COMPRESS_STATS) {
          DistortionStats stats;
          for (size_t j = 0; j < 2 * NF; ++j) {
            stats.Notify(raw[i + j],
                         hwy::F32FromBF16(packed.ptr[packed_ofs + i + j]));
          }
          tls.stats.Notify(stats);
        }
      }
    }

    const size_t remaining = num - i;
    HWY_DASSERT(remaining < 2 * NF);
    if (remaining != 0) {
      const VF raw0 = hn::LoadN(df, raw + i, remaining);
      const size_t remaining1 = remaining - HWY_MIN(remaining, NF);
      const VF raw1 = hn::LoadN(df, raw + i + NF, remaining1);

      hn::StoreN(hn::OrderedDemote2To(dbf, raw0, raw1), dbf,
                 packed.ptr + packed_ofs + i, remaining);

      if (COMPRESS_STATS) {
        DistortionStats stats;
        for (size_t j = 0; j < remaining; ++j) {
          stats.Notify(raw[i + j],
                       hwy::F32FromBF16(packed.ptr[packed_ofs + i + j]));
        }
        tls.stats.Notify(stats);
      }
    }
  }

  template <class DF, HWY_IF_F32_D(DF), class VF = hn::Vec<DF>>
  static void Store2(DF df, VF raw0, VF raw1, const PackedSpan<Packed>& packed,
                     const size_t packed_ofs) {
    const hn::Repartition<BF16, decltype(df)> dbf;
    hn::StoreU(hn::OrderedDemote2To(dbf, raw0, raw1), dbf,
               packed.ptr + packed_ofs);
  }

  template <class DBF16, HWY_IF_BF16_D(DBF16)>
  static HWY_INLINE void Load2(DBF16 dbf16,
                               const PackedSpan<const Packed>& packed,
                               const size_t packed_ofs, hn::Vec<DBF16>& raw0,
                               hn::Vec<DBF16>& raw1) {
    const size_t N16 = hn::Lanes(dbf16);
    raw0 = hn::LoadU(dbf16, packed.ptr + packed_ofs);
    raw1 = hn::LoadU(dbf16, packed.ptr + packed_ofs + N16);
  }

  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void Load2(DF df, const PackedSpan<const Packed>& packed,
                               const size_t packed_ofs, hn::Vec<DF>& raw0,
                               hn::Vec<DF>& raw1) {
    const hn::Repartition<BF16, decltype(df)> dbf;
    using VBF = hn::Vec<decltype(dbf)>;
    const VBF packed0 = hn::LoadU(dbf, packed.ptr + packed_ofs);
    raw0 = hn::PromoteLowerTo(df, packed0);
    raw1 = hn::PromoteUpperTo(df, packed0);
  }

  template <class DBF, HWY_IF_BF16_D(DBF)>
  static HWY_INLINE void DecompressAndZeroPad(
      DBF dbf, const PackedSpan<const Packed>& packed, const size_t packed_ofs,
      BF16* HWY_RESTRICT raw, size_t num) {
    using VBF = hn::Vec<decltype(dbf)>;
    const size_t N16 = hn::Lanes(dbf);

    size_t i = 0;
    if (num >= N16) {
      for (; i <= num - N16; i += N16) {
        const VBF packed0 = hn::LoadU(dbf, packed.ptr + packed_ofs + i);
        hn::StoreU(packed0, dbf, raw + i);
      }
    }

    const size_t remaining = num - i;
    HWY_DASSERT(remaining < N16);
    if (HWY_UNLIKELY(remaining != 0)) {
      const VBF packed0 =
          hn::LoadN(dbf, packed.ptr + packed_ofs + i, remaining);
      hn::StoreU(packed0, dbf, raw + i);
    }
  }

#if 0
  template <class DBF, HWY_IF_BF16_D(DBF)>
  static HWY_INLINE void DecompressAndZeroPad(
      DBF dbf, const PackedSpan<const Packed>& packed, const size_t packed_ofs,
      BF16* HWY_RESTRICT raw, size_t num) {
    const size_t N16 = hn::Lanes(dbf);

#if 1
    const BF16* HWY_RESTRICT start = packed.ptr + packed_ofs;
    using VBF = hn::Vec<decltype(dbf)>;
    size_t i = 0;
    if (num >= 4 * N16) {
      for (; i <= num - 4 * N16; i += 4 * N16) {
        const VBF packed0 = hn::LoadU(dbf, start + i + 0 * N16);
        const VBF packed1 = hn::LoadU(dbf, start + i + 1 * N16);
        const VBF packed2 = hn::LoadU(dbf, start + i + 2 * N16);
        const VBF packed3 = hn::LoadU(dbf, start + i + 3 * N16);
        hn::StoreU(packed0, dbf, raw + i + 0 * N16);
        hn::StoreU(packed1, dbf, raw + i + 1 * N16);
        hn::StoreU(packed2, dbf, raw + i + 2 * N16);
        hn::StoreU(packed3, dbf, raw + i + 3 * N16);
      }
    }

    for (; i < num; i += N16) {
      const size_t remaining = num - i;
      const VBF packed0 = hn::LoadN(dbf, start + i, remaining);
      hn::StoreU(packed0, dbf, raw + i);
    }
#else
    hwy::CopyBytes(packed.ptr + packed_ofs, raw, num * sizeof(BF16));
    hwy::ZeroBytes(raw + num, (hwy::RoundUpTo(num, N16) - num) * sizeof(BF16));
#endif
  }
#endif

  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void DecompressAndZeroPad(
      DF df, const PackedSpan<const Packed>& packed, const size_t packed_ofs,
      float* HWY_RESTRICT raw, size_t num) {
    const hn::Repartition<BF16, decltype(df)> dbf;
    using VF = hn::Vec<decltype(df)>;
    using VBF = hn::Vec<decltype(dbf)>;
    const size_t NF = hn::Lanes(df);

    size_t i = 0;
    if (num >= 2 * NF) {
      for (; i <= num - 2 * NF; i += 2 * NF) {
        VF raw0, raw1;
        Load2(df, packed, packed_ofs + i, raw0, raw1);
        hn::StoreU(raw0, df, raw + i);
        hn::StoreU(raw1, df, raw + i + NF);
      }
    }

    const size_t remaining = num - i;
    HWY_DASSERT(remaining < 2 * NF);
    if (HWY_UNLIKELY(remaining != 0)) {
      const VBF packed0 =
          hn::LoadN(dbf, packed.ptr + packed_ofs + i, remaining);
      const VF raw0 = hn::PromoteLowerTo(df, packed0);
      const VF raw1 = hn::PromoteUpperTo(df, packed0);
      // If at most one vector, the first store adds zero padding. Check before
      // storing the second, because callers only pad to one vector.
      hn::StoreU(raw0, df, raw + i);
      if (remaining > NF) hn::StoreU(raw1, df, raw + i + NF);
    }
  }
};

// Switching floating point: 8-bit, 2..3 mantissa bits.
template <>
struct CompressTraits<SfpStream> {
  using Packed = SfpStream;

  // Callers are responsible for scaling `raw` such that its magnitudes do not
  // exceed `SfpStream::kMax`. See CompressedArray::Scale().
  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void Compress(DF df, const float* HWY_RESTRICT raw,
                                  size_t num, CompressPerThread& tls,
                                  const PackedSpan<Packed>& packed,
                                  const size_t packed_ofs) {
    SfpCodec::Enc(df, raw, num, packed.ptr + packed_ofs);

    if constexpr (COMPRESS_STATS) {
      const hn::Repartition<BF16, DF> dbf;
      auto distorted =
          hwy::AllocateAligned<BF16>(hwy::RoundUpTo(num, hn::Lanes(dbf)));
      SfpCodec::DecompressAndZeroPad(dbf, MakeConst(packed), packed_ofs,
                                     distorted.get(), num);
      DistortionStats stats;
      for (size_t i = 0; i < num; ++i) {
        stats.Notify(raw[i], hwy::F32FromBF16(distorted[i]));
      }
      tls.stats.Notify(stats);
    }
  }

  template <class D>  // Caller checks this is f32 or bf16
  static HWY_INLINE void Load2(D d, const PackedSpan<const Packed>& packed,
                               const size_t packed_ofs, hn::Vec<D>& raw0,
                               hn::Vec<D>& raw1) {
    const hn::Twice<hn::Rebind<uint8_t, D>> d8;
    using V8 = hn::Vec<decltype(d8)>;
    const V8 v8 = hn::LoadU(d8, &packed.ptr->byte + packed_ofs);
    SfpCodec::Dec2(d, v8, raw0, raw1);
  }

  // Store2 is not yet implemented.

  template <class D, typename Raw>
  static HWY_INLINE void DecompressAndZeroPad(
      D d, const PackedSpan<const Packed>& packed, const size_t packed_ofs,
      Raw* HWY_RESTRICT raw, const size_t num) {
    SfpCodec::DecompressAndZeroPad(d, packed, packed_ofs, raw, num);
  }
};

// Nonuniform quantization, 4.5 bits per element, two separate streams.
template <>
struct CompressTraits<NuqStream> {
  using Packed = NuqStream;

  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void Compress(DF df, const float* HWY_RESTRICT raw,
                                  size_t num, CompressPerThread& tls,
                                  const PackedSpan<Packed>& packed,
                                  const size_t packed_ofs) {
    if (!tls.buf) tls.buf = std::make_unique<NuqStream::ClusterBuf>();
    NuqCodec::Enc(df, raw, num, *tls.buf, packed, packed_ofs);

    if constexpr (COMPRESS_STATS) {
      for (size_t i = 0; i < num; ++i) {
        tls.stats.NotifyIn(static_cast<int>(lroundf(raw[i] * 100.0f + 500.0f)));
      }

      const hn::Repartition<BF16, DF> dbf;
      const size_t N16 = hn::Lanes(dbf);
      auto distorted = hwy::AllocateAligned<BF16>(hwy::RoundUpTo(num, N16));
      NuqCodec::DecompressAndZeroPad(dbf, MakeConst(packed), packed_ofs,
                                     distorted.get(), num);
      DistortionStats stats;
      for (size_t i = 0; i < num; ++i) {
        stats.Notify(raw[i], hwy::F32FromBF16(distorted[i]));
      }
      tls.stats.Notify(stats);
    }
  }

  template <class D>  // Caller checks this is f32 or bf16
  static HWY_INLINE void Load2(D d, const PackedSpan<const Packed>& packed,
                               const size_t packed_ofs, hn::Vec<D>& raw0,
                               hn::Vec<D>& raw1) {
    NuqCodec::Dec2(d, packed, packed_ofs, raw0, raw1);
  }

  // Store2 is not yet implemented.

  template <class D, typename Raw>
  static HWY_INLINE void DecompressAndZeroPad(
      D d, const PackedSpan<const Packed>& packed, const size_t packed_ofs,
      Raw* raw, const size_t num) {
    NuqCodec::DecompressAndZeroPad(d, packed, packed_ofs, raw, num);
  }
};

// Compresses `num` elements of `raw` to `packed` starting at `packed_ofs`,
// which is useful for compressing sub-regions of an array.
template <typename Packed>
HWY_NOINLINE void Compress(const float* HWY_RESTRICT raw, size_t num,
                           CompressWorkingSet& work,
                           const PackedSpan<Packed>& packed,
                           const size_t packed_ofs, hwy::ThreadPool& pool) {
  packed.BoundsCheck(packed_ofs, num);
  work.tls.resize(pool.NumWorkers());
  if constexpr (COMPRESS_STATS) {
    for (auto& tls : work.tls) {
      tls.stats.Reset();
    }
  }

  const bool want_bench = COMPRESS_STATS || !kIsTest;
  const double t0 = want_bench ? hwy::platform::Now() : 0.0;

  using Traits = CompressTraits<hwy::RemoveConst<Packed>>;
  constexpr size_t kBatch = 8192;
  const size_t num_batches = hwy::DivCeil(num, kBatch);
  pool.Run(0, num_batches,
           [&](const uint32_t idx_batch, size_t thread) HWY_ATTR {
             const hn::ScalableTag<float> df;

             const size_t my_pos = idx_batch * kBatch;
             const size_t my_num =
                 idx_batch == num_batches - 1 ? (num - my_pos) : kBatch;
             Traits::Compress(df, raw + my_pos, my_num, work.tls[thread],
                              packed, packed_ofs + my_pos);
           });

  if (want_bench) {  // Avoids log spam in tests
    const double t1 = hwy::platform::Now();
    const double mb = static_cast<double>(num) * sizeof(raw[0]) * 1E-6;
    const double mbps = mb / (t1 - t0);
    fprintf(stderr, "Compress %.1f MB/s\n", mbps);
  }

  if constexpr (COMPRESS_STATS) {
    for (size_t i = 1; i < work.tls.size(); ++i) {
      work.tls[0].stats.Assimilate(work.tls[i].stats);
    }
    work.tls[0].stats.PrintAll();
  }
}

// Same as above, but without parallelization nor benchmarking.
template <typename Packed>
HWY_NOINLINE void Compress(const float* HWY_RESTRICT raw, size_t num,
                           CompressPerThread& tls,
                           const PackedSpan<Packed>& packed,
                           const size_t packed_ofs) {
  packed.BoundsCheck(packed_ofs, num);
  using Traits = CompressTraits<hwy::RemoveConst<Packed>>;
  const hn::ScalableTag<float> df;
  Traits::Compress(df, raw, num, tls, packed, packed_ofs);
}

// Stores two f32 vectors to f32 or bf16.
template <class DF, typename Packed, HWY_IF_F32_D(DF), class VF = hn::Vec<DF>>
void Compress2(DF df, VF raw0, VF raw1, const PackedSpan<Packed>& packed,
               const size_t packed_ofs) {
  static_assert(hwy::IsSameEither<Packed, float, BF16>());
  packed.BoundsCheck(packed_ofs, 2 * hn::Lanes(df));
  using Traits = CompressTraits<hwy::RemoveConst<Packed>>;
  Traits::Store2(df, raw0, raw1, packed, packed_ofs);
}

namespace detail {

// Compile-time-only check that `DRaw` and `Packed` are compatible. This makes
// for better error messages than "no matching function found".
template <class DRaw, typename Packed>
HWY_INLINE void VerifyRawAndPackedForDecompress() {
  using TRaw = hn::TFromD<DRaw>;
  // We can decompress any Packed to f32 or BF16, or f32 to f64.
  static_assert(hwy::IsSameEither<TRaw, float, BF16>() ||
                (IsF32<Packed>() && hwy::IsSame<TRaw, double>()));
}

}  // namespace detail

// Decompresses from any type of `packed`, to two vectors of `float/BF16`, or
// `double`, if `Packed` is `float`.
template <class DRaw, typename Packed, class VRaw = hn::Vec<DRaw>>
HWY_INLINE void Decompress2(DRaw d, const PackedSpan<Packed>& packed,
                            const size_t packed_ofs, VRaw& raw0, VRaw& raw1) {
  detail::VerifyRawAndPackedForDecompress<DRaw, Packed>();
  packed.BoundsCheck(packed_ofs, 2 * hn::Lanes(d));
  using Traits = CompressTraits<hwy::RemoveCvRef<Packed>>;
  Traits::Load2(d, MakeConst(packed), packed_ofs, raw0, raw1);
}

// Decompresses from any type of `packed`, starting at (any) `packed_ofs`, to
// (any) `num` elements in `raw`, then appends `[0, hn::Lanes(d))` zeroes as
// required to round `num` up to one vector, if it is not already. The caller is
// responsible for scaling `raw` to the original range because `EmbedMMToken`
// also wants to scale the decompressed elements.
// `TRaw` can be `float/BF16`, or `double` if `Packed` is `float`.
template <class DRaw, typename Packed, typename TRaw = hn::TFromD<DRaw>>
HWY_INLINE void DecompressAndZeroPad(DRaw d, const PackedSpan<Packed>& packed,
                                     const size_t packed_ofs, TRaw* raw,
                                     size_t num) {
  detail::VerifyRawAndPackedForDecompress<DRaw, Packed>();
  packed.BoundsCheck(packed_ofs, num);
  using Traits = CompressTraits<hwy::RemoveCvRef<Packed>>;
  Traits::DecompressAndZeroPad(d, MakeConst(packed), packed_ofs, raw, num);
}

// Invokes `kernel` for the `v.num` elements of `w` and `v`. Decompresses from
// both into groups of four vectors with lane type `Kernel::Raw`, passes them to
// `kernel.Update4`; loads the final vector(s) with zero-padding, then passes
// them to `kernel.Update1`, then returns `kernel.Reduce`. `v.num` is not
// required to be a multiple of the vector length.
//
// Both `w` and `v` can be any packed type. To support random access in `w`
// even if it is `NuqStream`, we ignore `w.num` and provide a `w_ofs`, but no
// `v_ofs` because it is always 0 in our use cases. `D` only serves to specify
// the vector size/fraction.
//
// `kernel` is const& so we can pass an rvalue argument, but can contain
// mutable state, though not vectors (see highway.h). In addition to the groups
// of four input vectors, we pass eight state vectors with lane type specified
// by `Kernel::State`, which is typically `float` but may differ if `Raw` is
// `double`, or `WT` and `VT` are `BF16`.
//
// Decoupling decompression and remainder handling from the actual usage of the
// vectors makes it easier to implement various dot product and sum algorithms.
// This is similar to `hwy/contrib/unroller`, but less general and relies on
// `DecompressAndZeroPad`.
template <class D, typename WT, typename VT, class Kernel>
HWY_INLINE float DecompressAndCall(D, const PackedSpan<const WT>& w,
                                   const size_t w_ofs,
                                   const PackedSpan<const VT> v,
                                   const Kernel& kernel) {
  // Decompressed inputs
  using Raw = typename Kernel::template Raw<WT, VT>;
  const hn::Repartition<Raw, D> d_raw;
  using VRaw = hn::Vec<decltype(d_raw)>;
  VRaw w0, w1, w2, w3, v0, v1, v2, v3;

  // State for Kernel
  const hn::Repartition<typename Kernel::State, D> d_state;
  using VState = hn::Vec<decltype(d_state)>;
  VState sum0 = hn::Zero(d_state);
  VState sum1 = hn::Zero(d_state);
  VState sum2 = hn::Zero(d_state);
  VState sum3 = hn::Zero(d_state);
  VState comp0 = hn::Zero(d_state);
  VState comp1 = hn::Zero(d_state);
  VState comp2 = hn::Zero(d_state);
  VState comp3 = hn::Zero(d_state);

  const size_t N = hn::Lanes(d_raw);
  size_t i = 0;
  if (v.num >= 4 * N) {
    for (; i <= v.num - 4 * N; i += 4 * N) {
      Decompress2(d_raw, w, w_ofs + i + 0 * N, w0, w1);
      Decompress2(d_raw, w, w_ofs + i + 2 * N, w2, w3);
      Decompress2(d_raw, v, i + 0 * N, v0, v1);
      Decompress2(d_raw, v, i + 2 * N, v2, v3);

      kernel.Update4(d_raw, w0, w1, w2, w3, v0, v1, v2, v3, sum0, sum1, sum2,
                     sum3, comp0, comp1, comp2, comp3);
    }
  }

  size_t remaining = v.num - i;
  HWY_DASSERT(remaining < 4 * N);
  if (HWY_UNLIKELY(remaining != 0)) {
    HWY_ALIGN Raw padded_w[4 * hn::MaxLanes(d_raw)];
    HWY_ALIGN Raw padded_v[4 * hn::MaxLanes(d_raw)];
    DecompressAndZeroPad(d_raw, w, w_ofs + i, padded_w, remaining);
    DecompressAndZeroPad(d_raw, v, i, padded_v, remaining);

    // 1..4 whole vectors, possibly zero-padded.
    for (size_t padded_pos = 0; padded_pos < remaining; padded_pos += N) {
      const VRaw w0 = hn::Load(d_raw, padded_w + padded_pos);
      const VRaw v0 = hn::Load(d_raw, padded_v + padded_pos);
      kernel.Update1(d_raw, w0, v0, sum0, comp0);
    }
  }

  return kernel.Reduce(d_state, sum0, sum1, sum2, sum3, comp0, comp1, comp2,
                       comp3);
}

// Same as above, but single input array. Used by RMSNorm.
template <class D, typename VT, class Kernel>
HWY_INLINE float DecompressAndCall(D, const PackedSpan<const VT> v,
                                   const Kernel& kernel) {
  // Decompressed inputs
  using Raw = typename Kernel::template Raw<VT, VT>;
  const hn::Repartition<Raw, D> d_raw;
  using VRaw = hn::Vec<decltype(d_raw)>;
  VRaw v0, v1, v2, v3;

  // State for Kernel
  const hn::Repartition<typename Kernel::State, D> d_state;
  using VState = hn::Vec<decltype(d_state)>;
  VState sum0 = hn::Zero(d_state);
  VState sum1 = hn::Zero(d_state);
  VState sum2 = hn::Zero(d_state);
  VState sum3 = hn::Zero(d_state);
  VState comp0 = hn::Zero(d_state);
  VState comp1 = hn::Zero(d_state);
  VState comp2 = hn::Zero(d_state);
  VState comp3 = hn::Zero(d_state);

  const size_t N = hn::Lanes(d_raw);
  size_t i = 0;
  if (v.num >= 4 * N) {
    for (; i <= v.num - 4 * N; i += 4 * N) {
      Decompress2(d_raw, v, i + 0 * N, v0, v1);
      Decompress2(d_raw, v, i + 2 * N, v2, v3);

      kernel.Update4(d_raw, v0, v1, v2, v3, v0, v1, v2, v3, sum0, sum1, sum2,
                     sum3, comp0, comp1, comp2, comp3);
    }
  }

  size_t remaining = v.num - i;
  HWY_DASSERT(remaining < 4 * N);
  if (HWY_UNLIKELY(remaining != 0)) {
    HWY_ALIGN Raw padded_v[4 * hn::MaxLanes(d_raw)];
    DecompressAndZeroPad(d_raw, v, i, padded_v, remaining);

    // 1..4 whole vectors, possibly zero-padded.
    for (size_t padded_pos = 0; padded_pos < remaining; padded_pos += N) {
      const VRaw v0 = hn::Load(d_raw, padded_v + padded_pos);
      kernel.Update1(d_raw, v0, v0, sum0, comp0);
    }
  }

  return kernel.Reduce(d_state, sum0, sum1, sum2, sum3, comp0, comp1, comp2,
                       comp3);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
