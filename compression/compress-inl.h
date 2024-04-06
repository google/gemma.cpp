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

#include <array>

// copybara:import_next_line:gemma_cpp
#include "compression/blob_store.h"
// copybara:import_next_line:gemma_cpp
#include "compression/compress.h"
// copybara:import_next_line:gemma_cpp
#include "compression/distortion.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/timer.h"

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_COMPRESS_INL_H_

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_COMPRESS_TOGGLE) == defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_COMPRESS_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_COMPRESS_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_COMPRESS_TOGGLE
#endif

// copybara:import_next_line:gemma_cpp
#include "compression/nuq-inl.h"
// copybara:import_next_line:gemma_cpp
#include "compression/sfp-inl.h"
#include "hwy/contrib/dot/dot-inl.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Enables generic code independent of compression type.
template <typename T>  // primary, must specialize
struct CompressTraits {};

template <>
struct CompressTraits<float> {
  using MatT = float;

  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void Compress(DF df, const float* HWY_RESTRICT in,
                                  size_t num, CompressPerThread& tls,
                                  size_t /*out_capacity*/,
                                  MatT* HWY_RESTRICT out, size_t out_ofs) {
    using VF = hn::Vec<decltype(df)>;
    const size_t N = hn::Lanes(df);
    HWY_DASSERT(num >= 2 * N && num % (2 * N) == 0);

    for (size_t i = 0; i < num; i += 2 * N) {
      const VF in0 = hn::LoadU(df, in + i);
      const VF in1 = hn::LoadU(df, in + i + N);
      hn::StoreU(in0, df, out + out_ofs + i);
      hn::StoreU(in1, df, out + out_ofs + i + N);
    }
  }

  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void Decompress(DF df, size_t /*in_capacity*/,
                                    const MatT* HWY_RESTRICT in, size_t in_ofs,
                                    float* HWY_RESTRICT out, size_t num) {
    using VF = hn::Vec<decltype(df)>;
    const size_t N = hn::Lanes(df);
    HWY_DASSERT(num >= 2 * N && num % (2 * N) == 0);

    for (size_t i = 0; i < num; i += 2 * N) {
      const VF in0 = hn::LoadU(df, in + in_ofs + i);
      const VF in1 = hn::LoadU(df, in + in_ofs + i + N);
      hn::StoreU(in0, df, out + i);
      hn::StoreU(in1, df, out + i + N);
    }
  }

  // VecT can be float or hwy::bfloat16_t.
  template <class DF, typename VecT, HWY_IF_F32_D(DF)>
  static HWY_INLINE float Dot(DF df, size_t /*in_capacity*/,
                              const MatT* HWY_RESTRICT in, size_t in_ofs,
                              const VecT* HWY_RESTRICT vec_aligned,
                              size_t num) {
    HWY_DASSERT(num >= hn::Lanes(df) && (num % hn::Lanes(df)) == 0);
    HWY_DASSERT(hn::IsAligned(df, vec_aligned));
    constexpr int kAssumptions =
        hn::Dot::kAtLeastOneVector | hn::Dot::kMultipleOfVector;
    // vec_aligned must be the second argument because hn::Dot supports f32*bf16
    // and f32*f32.
    return hn::Dot::Compute<kAssumptions>(df, in + in_ofs, vec_aligned, num);
  }
};

template <>
struct CompressTraits<hwy::bfloat16_t> {
  using MatT = hwy::bfloat16_t;

  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void Compress(DF df, const float* HWY_RESTRICT in,
                                  size_t num, CompressPerThread& tls,
                                  size_t /*out_capacity*/,
                                  MatT* HWY_RESTRICT out, size_t out_ofs) {
    const hn::RebindToUnsigned<decltype(df)> du;
    const hn::Repartition<hwy::bfloat16_t, decltype(df)> dbf;
    using VF = hn::Vec<decltype(df)>;
    const size_t N = hn::Lanes(df);

    hn::Vec<decltype(du)> or_sum = hn::Zero(du);

    size_t i = 0;
    if (num >= 2 * N) {
      for (; i <= num - 2 * N; i += 2 * N) {
        const VF in0 = hn::LoadU(df, in + i);
        const VF in1 = hn::LoadU(df, in + i + N);

        // Sticky bits so we can warn if any lower bits were set.
        or_sum = hn::Or3(or_sum, hn::BitCast(du, in0), hn::BitCast(du, in1));
        hn::StoreU(hn::OrderedDemote2To(dbf, in0, in1), dbf, out + out_ofs + i);

        if (COMPRESS_STATS) {
          DistortionStats stats;
          for (size_t j = 0; j < 2 * N; ++j) {
            stats.Notify(in[i + j], hwy::F32FromBF16(out[out_ofs + i + j]));
          }
          tls.stats.Notify(stats);
        }
      }
    }

    const size_t remaining = num - i;
    if (remaining != 0) {
      const VF in0 = hn::LoadN(df, in + i, remaining);
      const size_t remaining1 = remaining - HWY_MIN(remaining, N / 2);
      const VF in1 = hn::LoadN(df, in + i + N, remaining1);

      // Sticky bits so we can warn if any lower bits were set.
      or_sum = hn::Or3(or_sum, hn::BitCast(du, in0), hn::BitCast(du, in1));
      hn::StoreU(hn::OrderedDemote2To(dbf, in0, in1), dbf, out + out_ofs + i);

      if (COMPRESS_STATS) {
        DistortionStats stats;
        for (size_t j = 0; j < remaining; ++j) {
          stats.Notify(in[i + j], hwy::F32FromBF16(out[out_ofs + i + j]));
        }
        tls.stats.Notify(stats);
      }
    }

    // If the lower 16 bits are not zero, we should implement rounding.
    or_sum = hn::And(or_sum, hn::Set(du, 0xFFFF));
    if (!hn::AllTrue(du, hn::Eq(or_sum, hn::Zero(du)))) {
      // fprintf(stderr, "Warning: Lossy truncation.");
    }
  }

  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void Decompress(DF df, size_t /*in_capacity*/,
                                    const MatT* HWY_RESTRICT in, size_t in_ofs,
                                    float* HWY_RESTRICT out, size_t num) {
    const hn::Repartition<hwy::bfloat16_t, decltype(df)> dbf;
    using VBF = hn::Vec<decltype(dbf)>;
    using VF = hn::Vec<decltype(df)>;
    const size_t N16 = hn::Lanes(dbf);

    size_t i = 0;
    if (num >= N16) {
      for (i = 0; i <= num - N16; i += N16) {
        const VBF in16 = hn::LoadU(dbf, in + in_ofs + i);
        const VF in0 = hn::PromoteLowerTo(df, in16);
        const VF in1 = hn::PromoteUpperTo(df, in16);
        hn::StoreU(in0, df, out + i);
        hn::StoreU(in1, df, out + i + N16 / 2);
      }
    }

    const size_t remaining = num - i;
    if (remaining != 0) {
      const VBF in16 = hn::LoadN(dbf, in + in_ofs + i, remaining);
      const VF in0 = hn::PromoteLowerTo(df, in16);
      const VF in1 = hn::PromoteUpperTo(df, in16);
      hn::StoreN(in0, df, out + i, remaining);
      // Avoid wraparound, potentially store nothing.
      const size_t remaining1 = remaining - HWY_MIN(remaining, N16 / 2);
      hn::StoreN(in1, df, out + i + N16 / 2, remaining1);
    }
  }

  // VecT can be float or hwy::bfloat16_t.
  template <class DF, typename VecT, HWY_IF_F32_D(DF)>
  static HWY_INLINE float Dot(DF df, size_t /*in_capacity*/,
                              const MatT* HWY_RESTRICT in, size_t in_ofs,
                              const VecT* HWY_RESTRICT vec_aligned,
                              size_t num) {
    HWY_DASSERT(num >= hn::Lanes(df) && (num % hn::Lanes(df)) == 0);
    HWY_DASSERT(hn::IsAligned(df, vec_aligned));

    const hn::Repartition<VecT, decltype(df)> d_vec;

    constexpr int kAssumptions =
        hn::Dot::kAtLeastOneVector | hn::Dot::kMultipleOfVector;
    // vec_aligned must be first argument because hn::Dot supports f32*bf16 and
    // bf16*bf16.
    return hn::Dot::Compute<kAssumptions>(d_vec, vec_aligned, in + in_ofs, num);
  }
};

template <>
struct CompressTraits<SfpStream> {
  using MatT = SfpStream;

  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void Compress(DF df, const float* in, size_t num,
                                  CompressPerThread& tls,
                                  size_t /*out_capacity*/, MatT* out,
                                  size_t out_ofs) {
    SfpCodec::Enc(df, in, num, out + out_ofs);

    if (COMPRESS_STATS) {
      const hn::Repartition<hwy::bfloat16_t, DF> dbf;
      auto distorted = hwy::AllocateAligned<hwy::bfloat16_t>(num);
      SfpCodec::Dec(dbf, out + out_ofs, num, distorted.get());
      DistortionStats stats;
      for (size_t i = 0; i < num; ++i) {
        stats.Notify(in[i], hwy::F32FromBF16(distorted[i]));
      }
      tls.stats.Notify(stats);
    }
  }

  template <class D, typename OutT>
  static HWY_INLINE void Decompress(D d, size_t /*in_capacity*/, const MatT* in,
                                    size_t in_ofs, OutT* out, size_t num) {
    SfpCodec::Dec(d, in + in_ofs, num, out);
  }

  template <class DF, typename VecT, HWY_IF_F32_D(DF)>
  static HWY_INLINE float Dot(DF df, size_t /*in_capacity*/, const MatT* in,
                              size_t in_ofs, const VecT* vec_aligned,
                              size_t num) {
    using VF = hn::Vec<decltype(df)>;
    VF sum0 = hn::Zero(df);
    VF sum1 = hn::Zero(df);
    VF sum2 = hn::Zero(df);
    VF sum3 = hn::Zero(df);

    SfpCodec::Dot(df, in + in_ofs, num, vec_aligned, sum0, sum1, sum2, sum3);

    // Reduction tree: sum of all accumulators, then their lanes
    sum0 = hn::Add(sum0, sum1);
    sum2 = hn::Add(sum2, sum3);
    sum0 = hn::Add(sum0, sum2);
    return hn::ReduceSum(df, sum0);
  }
};

template <>
struct CompressTraits<NuqStream> {
  using MatT = NuqStream;

  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void Compress(DF df, const float* in, size_t num,
                                  CompressPerThread& tls, size_t out_capacity,
                                  MatT* out, size_t out_ofs) {
    NuqCodec::Enc(df, in, num, tls.buf, out_capacity, out, out_ofs);

    if (COMPRESS_STATS) {
      for (size_t i = 0; i < num; ++i) {
        tls.stats.NotifyIn(static_cast<int>(lroundf(in[i] * 100.0f + 500.0f)));
      }

      const hn::Repartition<hwy::bfloat16_t, DF> dbf;
      auto distorted = hwy::AllocateAligned<hwy::bfloat16_t>(num);
      NuqCodec::Dec(dbf, out_capacity, out, out_ofs, distorted.get(), num);
      DistortionStats stats;
      for (size_t i = 0; i < num; ++i) {
        stats.Notify(in[i], hwy::F32FromBF16(distorted[i]));
      }
      tls.stats.Notify(stats);
    }
  }

  template <class D, typename OutT>
  static HWY_INLINE void Decompress(D d, size_t in_capacity, const MatT* in,
                                    size_t in_ofs, OutT* out, size_t num) {
    NuqCodec::Dec(d, in_capacity, in, in_ofs, out, num);
  }

  template <class DF, typename VecT, HWY_IF_F32_D(DF)>
  static HWY_INLINE float Dot(DF df, size_t in_capacity, const MatT* in,
                              size_t in_ofs,
                              const VecT* HWY_RESTRICT vec_aligned,
                              size_t num) {
    using VF = hn::Vec<decltype(df)>;
    VF sum0 = hn::Zero(df);
    VF sum1 = hn::Zero(df);
    VF sum2 = hn::Zero(df);
    VF sum3 = hn::Zero(df);

    NuqCodec::Dot(df, in_capacity, in, in_ofs, vec_aligned, num, sum0, sum1,
                  sum2, sum3);

    // Reduction tree: sum of all accumulators, then their lanes
    sum0 = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
    return hn::ReduceSum(df, sum0);
  }
};

// Compresses `num` inputs to `out` starting at `out_ofs`. This can be used for
// compressing sub-regions of an array.
template <typename MatT>
HWY_NOINLINE void Compress(const float* in, size_t num,
                           CompressWorkingSet& work, size_t out_capacity,
                           MatT* out, size_t out_ofs, hwy::ThreadPool& pool) {
  HWY_DASSERT(out_ofs + num <= out_capacity);
  work.tls.resize(pool.NumThreads());
  if (COMPRESS_STATS) {
    for (auto& tls : work.tls) {
      tls.stats.Reset();
    }
  }

  const double t0 = hwy::platform::Now();

  using Traits = CompressTraits<MatT>;
  constexpr size_t kBatch = 8192;
  const size_t num_batches = hwy::DivCeil(num, kBatch);
  pool.Run(0, num_batches,
           [&](const uint32_t idx_batch, size_t thread) HWY_ATTR {
             const hn::ScalableTag<float> df;

             const size_t in_ofs = idx_batch * kBatch;
             const size_t my_num =
                 idx_batch == num_batches - 1 ? (num - in_ofs) : kBatch;
             Traits::Compress(df, in + in_ofs, my_num, work.tls[thread],
                              out_capacity, out, out_ofs + in_ofs);
           });

  const double t1 = hwy::platform::Now();
  const double mb = static_cast<double>(num) * sizeof(in[0]) * 1E-6;
  const double mbps = mb / (t1 - t0);
  fprintf(stderr, "Compress %.1f MB/s\n", mbps);

  if (COMPRESS_STATS) {
    for (size_t i = 1; i < work.tls.size(); ++i) {
      work.tls[0].stats.Assimilate(work.tls[i].stats);
    }
    work.tls[0].stats.PrintAll();
  }
}

// Compresses an entire std::array into `out`, which is assumed to have exactly
// that much capacity.
template <size_t kCapacity, typename MatT>
HWY_INLINE void Compress(const std::array<float, kCapacity>& in,
                         CompressWorkingSet& work,
                         CompressedArray<MatT, kCapacity>& compressed,
                         hwy::ThreadPool& pool) {
  Compress(in.data(), kCapacity, work, kCapacity, compressed.data(), 0, pool);
}

// Decompresses `num` values from `compressed` starting at `compressed_ofs`.
template <typename ArrayT, typename OutT>
HWY_NOINLINE void Decompress(const ArrayT& compressed, size_t compressed_ofs,
                             OutT* out, size_t num) {
  HWY_DASSERT(compressed_ofs + num <= compressed.size());
  const hn::ScalableTag<OutT> d;
  using Traits = CompressTraits<typename ArrayT::value_type>;
  Traits::Decompress(d, compressed.size(), compressed.data(), compressed_ofs,
                     out, num);
}

// As above, but with threading and benchmarking.
template <typename MatT, size_t kCapacity, typename OutT>
HWY_INLINE void Decompress(const CompressedArray<MatT, kCapacity>& compressed,
                           size_t compressed_ofs, OutT* out, size_t num,
                           hwy::ThreadPool& pool) {
  HWY_DASSERT(compressed_ofs + num <= compressed.size());
  const double t0 = hwy::platform::Now();

  using Traits = CompressTraits<MatT>;
  constexpr size_t kBatch = 8192;
  const size_t num_batches = hwy::DivCeil(num, kBatch);
  pool.Run(
      0, num_batches, [&](const uint32_t idx_batch, size_t thread) HWY_ATTR {
        const hn::ScalableTag<OutT> d;

        const size_t ofs = idx_batch * kBatch;
        const size_t num = idx_batch == num_batches - 1 ? (num - ofs) : kBatch;
        Traits::Decompress(d, compressed.size(), compressed.data(),
                           compressed_ofs + ofs, out + ofs, num);
      });

  const double t1 = hwy::platform::Now();
  const double mb = num * sizeof(MatT) * 1E-6;
  const double mbps = mb / (t1 - t0);
  fprintf(stderr, "Decompress %.1f MB/s\n", mbps);
}

// Returns dot product with `vec_aligned` of length `num`.
template <class DF, typename ArrayT, typename VecT>
HWY_INLINE float Dot(DF df, const ArrayT& compressed, size_t compressed_ofs,
                     const VecT* vec_aligned, size_t num) {
  HWY_DASSERT(compressed_ofs + num <= compressed.size());
  HWY_DASSERT(hn::IsAligned(df, vec_aligned));
  using Traits = CompressTraits<typename ArrayT::value_type>;
  return Traits::Dot(df, compressed.size(), compressed.data(), compressed_ofs,
                     vec_aligned, num);
}

// Returns dot product with `vec_aligned` of length `num`.
template <class DF, typename MatT, size_t kCapacity, typename VecT>
HWY_INLINE float Dot(DF df, const CompressedArray<MatT, kCapacity>& compressed,
                     size_t compressed_ofs, const VecT* vec_aligned,
                     size_t num) {
  HWY_DASSERT(compressed_ofs + num <= compressed.size());
  HWY_DASSERT(hn::IsAligned(df, vec_aligned));
  using Traits = CompressTraits<MatT>;
  return (compressed.scale() * Traits::Dot(df, compressed.size(),
                                           compressed.data(), compressed_ofs,
                                           vec_aligned, num));
}

// Callback used by ForeachTensor.
class Compressor {
 public:
  explicit Compressor(hwy::ThreadPool& pool) : pool_(pool) {}

  // Called for each tensor; compresses it and stores to the cache.
  template <typename MatT, size_t kCapacity>
  void operator()(const char* name, const float* weights,
                  CompressedArray<MatT, kCapacity>& compressed) {
    fprintf(stderr, "Regenerating %s (%zuM), please wait\n", name,
            kCapacity / (1000 * 1000));
    Compress(weights, kCapacity, work_, kCapacity, compressed.data(), 0, pool_);
    writer_.Add(CacheKey<MatT>(name), compressed.data(),
                compressed.CompressedSize());
  }

  void AddScales(float* scales, size_t len) {
    if (len) {
      writer_.Add(CacheKey<float>("scales"), scales, len * sizeof(scales[0]));
    }
  }

  void WriteAll(hwy::ThreadPool& pool, const char* blob_filename) {
    const BlobError err = writer_.WriteAll(pool, blob_filename);
    if (err != 0) {
      fprintf(stderr, "Failed to write blobs to %s (error %d)\n", blob_filename,
              err);
    }
  }

 private:
  CompressWorkingSet work_;
  hwy::ThreadPool& pool_;
  BlobWriter writer_;
};

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
