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
#ifndef THIRD_PARTY_GEMMA_CPP_COMPRESSION_NUQ_INL_H_
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_NUQ_INL_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <cstdio>

#include "compression/types.h"
#include "util/basics.h"
#include "hwy/base.h"

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_NUQ_INL_H_

// Actual per-target include guard.
#if defined(THIRD_PARTY_GEMMA_CPP_COMPRESSION_NUQ_INL_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_COMPRESSION_NUQ_INL_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_COMPRESSION_NUQ_INL_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_NUQ_INL_TOGGLE
#endif

#include "hwy/highway.h"
// After highway.h
#include "compression/sfp-inl.h"
#include "hwy/contrib/sort/vqsort-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// For internal use by NuqCodec.
class NuqClustering {
  static constexpr size_t kGroupSize = NuqStream::kGroupSize;

  // To go from sorted order back to the original order in O(1), we store the
  // original index in the lower bits of the float32 mantissa, which means they
  // are sorted alongside the value.
  struct FloatPayload {
    // Resets payload to zero; useful for displaying the actual value.
    static HWY_INLINE float Clear(float f) {
      const uint32_t binary32 = hwy::BitCastScalar<uint32_t>(f);
      return hwy::BitCastScalar<float>(binary32 &
                                       ~static_cast<uint32_t>(kGroupSize - 1));
    }

    // Sets payload to `bits`.
    static HWY_INLINE float Set(float f, size_t bits) {
      HWY_DASSERT(bits < kGroupSize);
      const uint32_t binary32 = hwy::BitCastScalar<uint32_t>(Clear(f));
      return hwy::BitCastScalar<float>(static_cast<uint32_t>(binary32 | bits));
    }

    // Obtains the payload (index) previously set by `Set`.
    static HWY_INLINE size_t Get(float f) {
      return hwy::BitCastScalar<uint32_t>(f) &
             static_cast<uint32_t>(kGroupSize - 1);
    }
  };

  // Cumulative sums for O(1) mean and interval sums.
  class ClusterCost {
    // Ensures it is safe to load a vector from the last element.
    static constexpr size_t kMaxLanes = hn::MaxLanes(hn::ScalableTag<float>());

    // Initialization value for table elements where `valid` is false.
    static constexpr float kSentinel = -1.0f;

   public:
    explicit ClusterCost(const float* HWY_RESTRICT sorted) {
      double cumsum = 0.0;
      double cumsum2 = 0.0;
      dcumsum_[0] = 0.0;
      cumsum_[0] = cumsum2_[0] = 0.0f;
      for (size_t i = 0; i < kGroupSize; ++i) {
        const float x = FloatPayload::Clear(sorted[i]);
        cumsum += x;
        cumsum2 += static_cast<double>(x) * x;
        dcumsum_[1 + i] = cumsum;
        cumsum_[1 + i] = static_cast<float>(cumsum);
        cumsum2_[1 + i] = static_cast<float>(cumsum2);
      }

      const hn::ScalableTag<float> df;
      using VF = hn::Vec<decltype(df)>;
      const VF k1 = hn::Set(df, 1.0f);
      const size_t N = hn::Lanes(df);
      HWY_DASSERT(kGroupSize % N == 0);

      // Precomputed length and reciprocal.
      for (size_t len = 0; len < kGroupSize; len += N) {
        const VF vlen = hn::Iota(df, static_cast<int32_t>(len));
        hn::StoreU(vlen, df, len_ + kMaxLanes + len);
        hn::StoreU(hn::Div(k1, vlen), df, inv_len_ + kMaxLanes + len);
      }
      // len = kGroupSize is legitimate, e.g., for all-equal weights.
      len_[kMaxLanes + kGroupSize] = static_cast<float>(kGroupSize);
      inv_len_[kMaxLanes + kGroupSize] = 1.0f / static_cast<float>(kGroupSize);
      // len = 0 can happen, but valid is false for that lane.
      len_[kMaxLanes + 0] = kSentinel;
      inv_len_[kMaxLanes + 0] = kSentinel;

      // Ensure it is safe to load a vector from the last element.
      for (size_t i = 0; i < kMaxLanes; ++i) {
        constexpr size_t kEnd = kGroupSize + 1;
        cumsum_[kEnd + i] = cumsum_[kGroupSize];
        cumsum2_[kEnd + i] = cumsum2_[kGroupSize];
        len_[kMaxLanes + kEnd + i] = len_[kMaxLanes + kGroupSize];
        inv_len_[kMaxLanes + kEnd + i] = inv_len_[kMaxLanes + kGroupSize];
      }
      // For inv_len_ we also prepend MaxLanes in case first > last.
      for (size_t i = 0; i < kMaxLanes; ++i) {
        len_[i] = kSentinel;
        inv_len_[i] = kSentinel;
      }
    }

    // Returns cost (L2 norm) for a single cluster, used for backtracking.
    double SumOfSorted(size_t first, size_t last) const {
      HWY_DASSERT(first < kGroupSize);
      HWY_DASSERT(last < kGroupSize);
      return dcumsum_[last + 1] - dcumsum_[first];
    }

    // Returns vector of costs of clustering first..last + i with their means.
    // O(1) thanks to cumulative sums, which works for Lp-norms with p > 1; we
    // choose p=2 for simplicity (fewer terms). Caller ignores lanes where
    // `!valid[i]`, i.e. `first > last + i`.
    template <class DF, class MF = hn::Mask<DF>, class VF = hn::Vec<DF>>
    VF SumCosts(DF df, size_t first, size_t last, MF valid) const {
      HWY_DASSERT(first < kGroupSize);
      HWY_DASSERT(last < kGroupSize);

      VF inv_len;
      const VF vlen = Lengths(df, first, last, valid, inv_len);

      const VF u_lo = hn::Set(df, cumsum_[first]);
      const VF u_lo2 = hn::Set(df, cumsum2_[first]);
      const VF hi = hn::LoadU(df, cumsum_ + last + 1);
      const VF hi2 = hn::LoadU(df, cumsum2_ + last + 1);
      const VF sum = hn::Sub(hi, u_lo);
      const VF sum2 = hn::Sub(hi2, u_lo2);

      // Sum of L2 over i in [first, last] = (x[i] - mu)^2. `sum` and `sum2` are
      // the cumulative sums of x and x^2, so expand to `sum x^2 + sum x * -2 *
      // mu + sum mu^2`. The last term is the sum of a constant, hence `mu^2 *
      // len`. Thus we have: `sum2 + mu * (-2 * sum + mu * len)`. We avoid a
      // (-)2 constant by adding.
      const VF mu = hn::Mul(sum, inv_len);  // mean := sum[i] / len[i]
      const VF two_sum = hn::Add(sum, sum);
      const VF l2 = hn::MulAdd(mu, hn::MulSub(mu, vlen, two_sum), sum2);
      // mu can have some roundoff error. To avoid multiple redundant clusters,
      // clamp to zero.
      return hn::ZeroIfNegative(l2);
    }

   private:
    // Returns precomputed lengths of [first, last + i] and their reciprocals.
    template <class DF, class VF = hn::Vec<DF>, class MF = hn::Mask<DF>>
    VF Lengths(DF df, size_t first, size_t last, MF valid, VF& inv_len) const {
      const int len = static_cast<int>(last) - static_cast<int>(first) + 1;
      HWY_DASSERT(kMaxLanes + len >= 0);
      HWY_DASSERT(len <= static_cast<int>(kGroupSize));
      // last + i are contiguous, hence single loads instead of gather.
      const VF vlen = hn::LoadU(df, len_ + kMaxLanes + len);
      inv_len = hn::LoadU(df, inv_len_ + kMaxLanes + len);

      if constexpr (HWY_IS_DEBUG_BUILD) {
        // Sanity check: no valid lanes are sentinels, all invalid are.
        const VF sentinel = hn::Set(df, kSentinel);
        const MF bad = hn::Eq(vlen, sentinel);
        const MF inv_bad = hn::Eq(inv_len, sentinel);
        HWY_DASSERT(hn::AllFalse(df, hn::And(valid, bad)));
        HWY_DASSERT(hn::AllFalse(df, hn::And(valid, inv_bad)));
        HWY_DASSERT(hn::AllTrue(df, hn::Or(valid, bad)));
        HWY_DASSERT(hn::AllTrue(df, hn::Or(valid, inv_bad)));
      }

      return vlen;
    }

    // Float has enough precision for our relatively small kGroupSize (256).
    // Element i = sums of [0..i-1].
    float cumsum_[kGroupSize + 1 + kMaxLanes];
    float cumsum2_[kGroupSize + 1 + kMaxLanes];
    float len_[kMaxLanes + kGroupSize + 1 + kMaxLanes];      // = vlen[i]
    float inv_len_[kMaxLanes + kGroupSize + 1 + kMaxLanes];  // = 1 / vlen[i]

    double dcumsum_[kGroupSize + 1];  // for SumOfSorted
  };

  // Dynamic programming step: returns costs of clustering 0..last+i, where the
  // rightmost clusters start at `first`. Called for each `idx_cluster`,
  // `first`, and `last`; vectorized across `last`. `first` may be greater than
  // `last`. `valid[i]` is `first <= last + i`.
  template <class DF, class VF = hn::Vec<DF>, class MF = hn::Mask<DF>>
  static HWY_INLINE VF
  ClusterDynProg(DF df, const NuqStream::AlignedMatrix<float>& costs,
                 const ClusterCost& cc, const size_t idx_cluster,
                 const size_t first, const size_t last, const MF valid) {
    HWY_DASSERT(idx_cluster != 0);
    HWY_DASSERT(0 != first && first < kGroupSize);
    HWY_DASSERT(last < kGroupSize);
    HWY_DASSERT(last % hn::Lanes(df) == 0);  // Called in steps of N

    // Cost of clustering 0..first-1 with one fewer cluster than now.
    const VF prev = hn::Set(df, costs(idx_cluster - 1, first - 1));
    // Eq2: add to that the cost of another cluster from first..last.
    return hn::Add(prev, cc.SumCosts(df, first, last, valid));
  }

 public:
  // Clusters `num <= kGroupSize` values in `x`, which need not be sorted
  // already nor aligned, by choosing and filling `centers` (size `kClusters`,
  // ascending order, not necessarily equal to one of the `x`). Fills `indices`
  // with the index of the cluster to which each `x` belongs (16-bit for
  // bit-packing). `buf` is per-thread.
  //
  // Returns the number of unused clusters, i.e., the starting index within
  // `centers`; prior centers are zero-initialized.
  //
  // O(kClusters * kGroupSize * kGroupSize), but the constant factors are so low
  // that this is about 5 times as fast as the O(kClusters * kGroupSize) SMAWK
  // as implemented in FAISS, for our kGroupSize of 256.
  template <class DF>
  static HWY_NOINLINE size_t ClusterExactL2(DF df, const float* HWY_RESTRICT x,
                                            size_t num,
                                            NuqStream::ClusterBuf& buf,
                                            float* HWY_RESTRICT centers,
                                            uint16_t* HWY_RESTRICT indices) {
    HWY_DASSERT(num <= kGroupSize);
    const hn::RebindToSigned<decltype(df)> di;
    using VF = hn::Vec<decltype(df)>;
    using MF = hn::Mask<decltype(df)>;
    using VI = hn::Vec<decltype(di)>;
    const size_t N = hn::Lanes(df);
    HWY_DASSERT(kGroupSize % N == 0);

    HWY_ALIGN float sorted_and_i[kGroupSize];
    for (size_t i = 0; i < num; ++i) {
      sorted_and_i[i] = FloatPayload::Set(x[i], i);
    }
    if (num != kGroupSize) {
      // Initialize the rest of the group. Use an existing value so we do not
      // waste a cluster on a sentinel value. Arbitrarily choose the largest.
      float max = -1E38f;
      for (size_t i = 0; i < num; ++i) {
        max = HWY_MAX(max, x[i]);
      }
      for (size_t i = num; i < kGroupSize; ++i) {
        sorted_and_i[i] = FloatPayload::Set(max, i);
      }
    }
    hn::VQSortStatic(sorted_and_i, kGroupSize, hwy::SortAscending());
    ClusterCost cc(sorted_and_i);  // ignores payload bits.

    // Reference: https://arxiv.org/abs/1701.07204
    // costs[k-1][m] is the lowest cost of clustering x1..m into k clusters.
    NuqStream::AlignedMatrix<float>& costs = buf.costs;
    // argmin[k][m] is the starting index within sorted_and_i[] of the k-th
    // cluster.
    NuqStream::AlignedMatrix<int32_t>& argmin = buf.argmin;

    // Fill first row of `costs` and `argmin`: single cluster, iterate over all
    // `last`.
    {
      const size_t cluster_idx = 0;
      const size_t first = 0;
      const VI vfirst = hn::Set(di, static_cast<int32_t>(first));
      const MF all_valid = hn::FirstN(df, N);  // first <= last is always true
      for (size_t last = 0; last < kGroupSize; last += N) {
        const VF vcosts = cc.SumCosts(df, first, last, all_valid);
        hn::Store(vcosts, df, &costs(cluster_idx, last));
        hn::Store(vfirst, di, &argmin(cluster_idx, last));
      }
    }

    constexpr size_t kClusters = NuqStream::kClusters;
    for (size_t cluster_idx = 1; cluster_idx < kClusters; ++cluster_idx) {
      // For vectors of `last + i` with `i < N`:
      for (size_t last = 0; last < kGroupSize; last += N) {
        const VI vlast = hn::Iota(di, static_cast<int32_t>(last));
        const VF prev_cost = hn::LoadU(df, &costs(cluster_idx - 1, last));
        VF min = prev_cost;
        VI arg = hn::LoadU(di, &argmin(cluster_idx - 1, last));
        // For each `first` (j), which is the start of the rightmost of at least
        // two clusters, hence never zero. `first` also continues past `last`
        // because the last `vlast` lane is `last + N - 1`.
        for (size_t first = 1; first < last + N; ++first) {
          const VI vfirst = hn::Set(di, static_cast<int32_t>(first));
          const MF valid = hn::RebindMask(df, hn::Le(vfirst, vlast));
          const VF c =
              ClusterDynProg(df, costs, cc, cluster_idx, first, last, valid);

          // Retain the min cost and the `first` that caused it.
          const MF less = hn::And(valid, hn::Lt(c, min));
          min = hn::IfThenElse(less, c, min);
          arg = hn::IfThenElse(RebindMask(di, less), vfirst, arg);
        }
        HWY_DASSERT(hn::AllTrue(df, hn::Le(min, prev_cost)));

        hn::Store(min, df, &costs(cluster_idx, last));
        hn::Store(arg, di, &argmin(cluster_idx, last));
      }
    }

    // Backtrack to find centers. Clusters are [argmin(k, last), last].
    size_t last = kGroupSize - 1;
    size_t unused_clusters = 0;
    for (size_t k = kClusters - 1; k < kClusters; --k) {
      const size_t start = static_cast<size_t>(argmin(k, last));
      // Center = mean, O(1) thanks to cumulative sums.
      const double sum = cc.SumOfSorted(start, last);
      const int size = static_cast<int>(last) - static_cast<int>(start) + 1;
      HWY_DASSERT(0 < size && size <= static_cast<int>(kGroupSize));
      centers[k] = static_cast<float>(sum / size);

      // We know the range inside sorted_and_i[]; translate to original indices,
      // which are stored inside each of the sorted_and_i mantissas.
      for (size_t i = start; i <= last; ++i) {
        const size_t idx_x = FloatPayload::Get(sorted_and_i[i]);
        HWY_DASSERT(idx_x < kGroupSize);
        indices[idx_x] = static_cast<uint16_t>(k);
      }

      // Not using all clusters. Avoid out of bounds accesses by stopping early.
      if (start == 0) {
        unused_clusters = k;
        for (size_t cluster = 0; cluster < unused_clusters; ++cluster) {
          centers[cluster] = 0.0f;
        }
        break;
      }

      last = start - 1;
      HWY_DASSERT(last < kGroupSize);
    }

    if (HWY_IS_DEBUG_BUILD) {
      // If centers are not in ascending order, print them.
      for (size_t i = unused_clusters + 1; i < kClusters; ++i) {
        if (centers[i] < centers[i - 1]) {
          for (size_t i = 0; i < kClusters; ++i) {
            fprintf(stderr, "%2zu: %.8f\n", i, centers[i]);
          }
          for (size_t i = 0; i < kGroupSize; ++i) {
            fprintf(stderr, "%3zu: %.8f\n", i,
                    FloatPayload::Clear(sorted_and_i[i]));
          }
          for (size_t i = 0; i < num; ++i) {
            fprintf(stderr, "%3zu: %.8f\n", i, x[i]);
          }
          HWY_ABORT("Centers not in ascending order at %zu; unused %zu\n", i,
                    unused_clusters);
        }
      }
    }

    MaybeCheckInitialized(centers, kClusters * sizeof(centers[0]));
    return unused_clusters;
  }
};  // NuqClustering

// Half-vector of u8 from u16/bf16.
template <class D16>
using D8HFromD16 = hn::Half<hn::Repartition<uint8_t, D16>>;

// Bit-packing 4-bit values is trivial if we have 2 or 4 independent vectors:
// simply shift+OR them together into a full vector of 8 or 16-bit lanes.
// However, the order then depends on the vector length, which is unacceptable
// because we may store the encoding to disk and decode on another CPU.
//
// The dependency on vector length could be removed by introducing fixed-size
// packets and loading the next vector from a fixed offset known to be at
// least the vector length. However, this may require packets that are larger
// than the seek granularity of the application (e.g. matrix rows).
//
// We instead choose a continuous stream layout, which seems to entail the
// nibbles being stored and decoded in-order. This involves nontrivial shuffle
// operations which benefit from special-casing for target and vector length.
class NibbleCodec {
 public:
  // Returns a byte vector whose nibbles are the lanes of four u16 vectors, in
  // the same order.
  template <class D16, class V16 = hn::Vec<D16>,
            class D8 = hn::Repartition<uint8_t, D16>, class V8 = hn::Vec<D8>>
  static HWY_INLINE V8 OrderedPackU16(D16 d16, V16 in0, V16 in1, V16 in2,
                                      V16 in3) {
    const D8 d8;
    const hn::Repartition<uint32_t, D16> d32;
    const hn::Repartition<uint64_t, D16> d64;

    // Pairwise compaction of a single vector so nibbles are packed in-order.
    // v16 lanes hold a 4-bit value; OR together adjacent pairs into the lower
    // byte of *even* u16.
    const auto combine_u16_pair_to_8 = [d16, d32](V16 v16) HWY_ATTR {
      return hn::Xor(
          v16, hn::BitCast(d16, hn::ShiftRight<12>(hn::BitCast(d32, v16))));
    };

    const V16 u8_0 = combine_u16_pair_to_8(in0);
    const V16 u8_1 = combine_u16_pair_to_8(in1);
    const V16 u8_2 = combine_u16_pair_to_8(in2);
    const V16 u8_3 = combine_u16_pair_to_8(in3);
    if constexpr (HWY_TARGET <= HWY_AVX3_DL || !HWY_ARCH_X86) {
      // 8-bit ConcatEven is efficient. Let digits denote eight u8 lanes
      // of u8_1/0: ?d?3 ?c?2 / ?b?1 ?a?0. 8-bit ConcatEven = d3c2 b1a0, and
      // again with the second x2_1 gives 7654 3210.
      const V8 x2_0 = hn::ConcatEven(d8, BitCast(d8, u8_1), BitCast(d8, u8_0));
      const V8 x2_1 = hn::ConcatEven(d8, BitCast(d8, u8_3), BitCast(d8, u8_2));
      return hn::ConcatEven(d8, x2_1, x2_0);
    } else {
      // To avoid expensive 8-bit ConcatEven, compact pairs of u32 into the
      // lower 16 bits in each u64, with other bits undefined.
      const auto combine_u32_pair_to_16 = [d16, d64](V16 v16) HWY_ATTR {
        return hn::Xor(
            v16, hn::BitCast(d16, hn::ShiftRight<24>(hn::BitCast(d64, v16))));
      };
      const V16 u16_0 = combine_u32_pair_to_16(u8_0);
      const V16 u16_1 = combine_u32_pair_to_16(u8_1);
      const V16 u16_2 = combine_u32_pair_to_16(u8_2);
      const V16 u16_3 = combine_u32_pair_to_16(u8_3);
      // In-order compaction of four vectors into one, keeping only the low
      // u16 of every u64. This is the same as above but with 16-bit Concat.
      const V16 x2_0 = hn::ConcatEven(d16, u16_1, u16_0);
      const V16 x2_1 = hn::ConcatEven(d16, u16_3, u16_2);
      return hn::BitCast(d8, hn::ConcatEven(d16, x2_1, x2_0));
    }
  }

  // Unpacks nibbles from the `kHalf` (0 or 1) half of a half-vector of bytes.
  // Thus we use a quarter of a vector of bytes and expand nibbles 4x into u16,
  // which fills a whole vector. Its first lane comes from the low nibble of the
  // first byte, the second from its high nibble, then the next low nibble, etc.
  template <size_t kHalf, class D16, class V16 = hn::Vec<D16>,
            class D8H = D8HFromD16<D16>, class V8H = hn::Vec<D8H>>
  static HWY_INLINE V16 OrderedUnpackU16(D16 d16, const V8H packed) {
    const hn::Twice<D8H> d8;  // full vector
    using V8 = hn::Vec<decltype(d8)>;

    // Replicate each byte 4x, so that its two nibbles propagate to both u16
    // lanes that they will initialize.
    const V8 rep4 = Replicate4x<kHalf>(d8, hn::ResizeBitCast(d8, packed));

    const V16 mask4 = hn::Set(d16, 0xF);
    const V16 u16 = BitCast(d16, rep4);
    // In-order unpack. Right-shift odd u16 by 4. Example with two packed
    // bytes, one digit representing a nibble:
    // 32 32 32 32 | 10 10 10 10  u16
    // z3 23 32 32 | z1 01 10 10  OddEven+ShiftRight
    // zz z3 zz z2 | zz z1 zz z0  And (unpacked result)
    return hn::And(mask4, hn::OddEven(hn::ShiftRight<4>(u16), u16));
  }

 private:
  // Returns `bytes[0 + kHalf * N/2]` in lanes 0..3, `bytes[1 + kHalf * N/2]` in
  // lanes 4..7, etc. We fuse `kHalf` into the tables, which avoids the caller
  // having to pass in `UpperHalf(bytes)`.
  template <size_t kHalf, class D8, class V8 = hn::Vec<D8>>
  static HWY_INLINE V8 Replicate4x(D8 d8, V8 bytes) {
    static_assert(kHalf <= 1);
    const size_t N = hn::Lanes(d8);
    constexpr size_t kMaxN = hn::MaxLanes(d8);
    // For kHalf=1 and 512-bit vectors, kAdd would be 16, which is out of
    // bounds for TableLookupBytes. We instead BroadcastBlock<1> there.
    constexpr uint8_t kAdd = kMaxN < 64 ? kHalf * kMaxN / 4 : 0;
    // The only performance-portable op to replicate bytes is TableLookupBytes,
    // but this only works if vectors are 128-bit or we first BroadcastBlock,
    // which only works for <= 512-bit vectors. For scalable vectors, we
    // instead synthesize this table via Iota+ShiftRight.
    alignas(64) static constexpr uint8_t kRep4[64] = {
        HWY_REP4(kAdd + 0),  HWY_REP4(kAdd + 1),  HWY_REP4(kAdd + 2),
        HWY_REP4(kAdd + 3),  HWY_REP4(kAdd + 4),  HWY_REP4(kAdd + 5),
        HWY_REP4(kAdd + 6),  HWY_REP4(kAdd + 7),  HWY_REP4(kAdd + 8),
        HWY_REP4(kAdd + 9),  HWY_REP4(kAdd + 10), HWY_REP4(kAdd + 11),
        HWY_REP4(kAdd + 12), HWY_REP4(kAdd + 13), HWY_REP4(kAdd + 14),
        HWY_REP4(kAdd + 15)};

    if constexpr (HWY_HAVE_SCALABLE) {
      // Replicate bytes 4x: lowest 4 = 0, next 4 = 1 etc. This works for up to
      // 1024-bit vectors: Iota is [128, 256), and [32, 64) after shifting.
      // For larger vectors, this would overflow and we should instead add kAdd.
      HWY_DASSERT(N <= 128);
      const V8 iota = hn::Iota(d8, static_cast<uint8_t>(kHalf * N));
      const V8 idx = hn::ShiftRight<2>(iota);
      return hn::TableLookupLanes(bytes, hn::IndicesFromVec(d8, idx));
    } else if constexpr (kMaxN <= 16) {  // <= 128-bit
      // No BroadcastBlock, we anyway only have one block.
      return hn::TableLookupBytes(bytes, hn::Load(d8, kRep4));
    } else if constexpr (HWY_TARGET <= HWY_AVX3_DL || !HWY_ARCH_X86) {
      // No BroadcastBlock, can directly permute across blocks.
      return hn::TableLookupLanes(bytes, hn::SetTableIndices(d8, kRep4));
    } else {  // 256..512-bit, no efficient TableLookupLanes
      static_assert(kMaxN <= 64);  // Else BroadcastBlock does not work.
      // See kAdd comment above.
      constexpr size_t kBlock = (kMaxN == 64 && kHalf == 1) ? 1 : 0;
      bytes = hn::BroadcastBlock<kBlock>(bytes);
      return hn::TableLookupBytes(bytes, hn::Load(d8, kRep4));
    }
  }
};

// Encode/decode functions.
class NuqCodec {
  static constexpr size_t kClusters = NuqStream::kClusters;
  static constexpr size_t kGroupSize = NuqStream::kGroupSize;

  // 256-bit vectors can hold 16 bf16, otherwise we require 2x128-bit.
  template <class DU>
  static constexpr size_t NumTables(DU du) {
    return (!HWY_HAVE_SCALABLE && du.MaxBytes() >= 32) ? 1 : 2;
  }

  // Offset (in bytes) of a group's table for packed_ofs (in elements) within a
  // set of groups.
  static constexpr size_t TableByteOffset(size_t packed_ofs) {
    const size_t kBytesPerGroup =
        (kClusters * sizeof(SfpStream)) + kGroupSize / 2;
    return (packed_ofs / kGroupSize) * kBytesPerGroup;
  }

  // Unpacks `centers` from SFP into bf16 and loads them into one or two vectors
  // for use by [Two]TableLookups. Returns as u16 because TableLookupLanes might
  // not be available for bf16.
  template <class DU, HWY_IF_U16_D(DU)>
  static HWY_INLINE hn::Vec<DU> LoadTable(DU du, const uint8_t* centers,
                                          hn::Vec<DU>* HWY_RESTRICT tbl1) {
    // Cap to the table size (kClusters) for decoding SFP - sufficient, and may
    // be faster than a large vector.
    const hn::CappedTag<BF16, kClusters> d_table;
    // We ResizeCast tables to DU: if DU is bigger, table lookups will only
    // access lanes < kClusters. If DU is smaller (128-bit), we have 2 tables.
    HWY_DASSERT(hn::Lanes(du) >= hn::Lanes(d_table) || NumTables(du) == 2);

    HWY_ALIGN BF16 table[kClusters];
    SfpCodec::DecompressAndZeroPad(
        d_table,
        MakeSpan(reinterpret_cast<const SfpStream*>(centers), kClusters), 0,
        table, kClusters);

    // If we assume >= 128-bit vectors, we can use [Two]TableLookupLanes
    // instead of TableLookupBytes, which requires extra interleaving of lo/hi.
    HWY_DASSERT(hn::Lanes(du) >= 8);

    if constexpr (NumTables(du) == 2) {
      // Reduce cap for second half to avoid loading past the end of the table.
      const hn::CappedTag<BF16, kClusters / 2> d_table2;
      *tbl1 = hn::ResizeBitCast(du, hn::LoadU(d_table2, table + kClusters / 2));
    }
    return hn::ResizeBitCast(du, hn::Load(d_table, table));
  }

  // Unpacks a half-vector of nibbles into two vectors of u16 indices and sets
  // c0/c1 to the corresponding bf16 (stored in u16) centers from tbl0/tbl1.
  template <class DU, class VU = hn::Vec<DU>, class D8H = D8HFromD16<DU>,
            class V8H = hn::Vec<D8H>>
  static HWY_INLINE void TableLookups(DU du, VU tbl0, VU tbl1, const V8H packed,
                                      VU& c0, VU& c1) {
    const VU idx0 = NibbleCodec::OrderedUnpackU16<0>(du, packed);
    const VU idx1 = NibbleCodec::OrderedUnpackU16<1>(du, packed);

    const auto indices0 = hn::IndicesFromVec(du, idx0);
    const auto indices1 = hn::IndicesFromVec(du, idx1);

    if constexpr (NumTables(du) == 1) {
      (void)tbl1;
      c0 = hn::TableLookupLanes(tbl0, indices0);
      c1 = hn::TableLookupLanes(tbl0, indices1);
    }
    if constexpr (NumTables(du) == 2) {  // `else` is poorly formatted.
      c0 = hn::TwoTablesLookupLanes(du, tbl0, tbl1, indices0);
      c1 = hn::TwoTablesLookupLanes(du, tbl0, tbl1, indices1);
    }
  }

  // As above, but returns a single 16-bit output vector for f32 Dec2, thus
  // packed is only a quarter-vector.
  template <class DU, class VU = hn::Vec<DU>,
            class D8Q = hn::Half<D8HFromD16<DU>>, class V8Q = hn::Vec<D8Q>>
  static HWY_INLINE VU TableLookups(DU du, VU tbl0, VU tbl1, const V8Q packed) {
    const D8HFromD16<DU> d8h;
    // OrderedUnpackU16 expects a half-vector, but will only use the lower half
    // of it.
    const hn::Vec<decltype(d8h)> packed_h = hn::ZeroExtendVector(d8h, packed);
    const VU idx0 = NibbleCodec::OrderedUnpackU16<0>(du, packed_h);

    const auto indices0 = hn::IndicesFromVec(du, idx0);

    if constexpr (NumTables(du) == 1) {
      (void)tbl1;
      return hn::TableLookupLanes(tbl0, indices0);
    }
    if constexpr (NumTables(du) == 2) {  // `else` is poorly formatted.
      return hn::TwoTablesLookupLanes(du, tbl0, tbl1, indices0);
    }
  }

 public:
  // Encodes `num` floats from `raw` into `packed`. `packed` points to
  // compressed storage and `packed_ofs` indicates the destination offset within
  // it, in number of elements. Tables are interleaved with indices (clustered
  // elements) to allow for easier unpacking. Returns the total number of
  // unused clusters, which is typically zero.
  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE size_t Enc(DF df, const float* HWY_RESTRICT raw,
                               const size_t num, NuqStream::ClusterBuf& buf,
                               const PackedSpan<NuqStream>& packed,
                               size_t packed_ofs) {
    const hn::Repartition<uint16_t, DF> d16;
    const hn::Repartition<uint8_t, DF> d8;
    using V16 = hn::Vec<decltype(d16)>;
    using V8 = hn::Vec<decltype(d8)>;
    const size_t N16 = hn::Lanes(d16);

    HWY_ASSERT(packed_ofs % kGroupSize == 0);
    HWY_ASSERT(kGroupSize % (4 * N16) == 0);

    const size_t num_groups = hwy::DivCeil(num, kGroupSize);
    HWY_ALIGN float g_centers[kClusters];
    // Zero-initialize in case of remainders (g_num != kGroupSize).
    HWY_ALIGN uint16_t g_idx[kGroupSize] = {};

    size_t unused_clusters = 0;
    size_t current_offset = packed_ofs;
    for (size_t g = 0; g < num_groups; ++g) {
      const size_t g_num = HWY_MIN(num - g * kGroupSize, kGroupSize);
      const float* HWY_RESTRICT g_in = raw + g * kGroupSize;

      unused_clusters +=
          NuqClustering::ClusterExactL2(df, g_in, g_num, buf, g_centers, g_idx);

      uint8_t* centers = &packed.ptr->byte + TableByteOffset(current_offset);
      SfpCodec::Enc(df, g_centers, kClusters,
                    reinterpret_cast<SfpStream*>(centers));
      uint8_t* packed_start = centers + kClusters;

      current_offset += g_num;

      size_t i = 0;
      if (g_num >= 4 * N16) {
        HWY_UNROLL(1)
        for (; i <= g_num - 4 * N16; i += 4 * N16) {
          const V16 idx0 = hn::LoadU(d16, g_idx + i + 0 * N16);
          const V16 idx1 = hn::LoadU(d16, g_idx + i + 1 * N16);
          const V16 idx2 = hn::LoadU(d16, g_idx + i + 2 * N16);
          const V16 idx3 = hn::LoadU(d16, g_idx + i + 3 * N16);
          const V8 nibbles =
              NibbleCodec::OrderedPackU16(d16, idx0, idx1, idx2, idx3);
          hn::StoreU(nibbles, d8, packed_start + i / 2);
        }
      }

      const size_t remaining = g_num - i;

      if (HWY_UNLIKELY(remaining != 0)) {
        // Safe to load all 4 vectors: g_idx zero-initialized and its size is
        // a multiple of 4 vectors.
        const V16 idx0 = hn::LoadU(d16, g_idx + i + 0 * N16);
        const V16 idx1 = hn::LoadU(d16, g_idx + i + 1 * N16);
        const V16 idx2 = hn::LoadU(d16, g_idx + i + 2 * N16);
        const V16 idx3 = hn::LoadU(d16, g_idx + i + 3 * N16);
        const V8 nibbles =
            NibbleCodec::OrderedPackU16(d16, idx0, idx1, idx2, idx3);
        // i is even, but remaining might not be.
        hn::StoreN(nibbles, d8, packed_start + i / 2,
                   hwy::DivCeil(remaining, 2));
      }
    }
    return unused_clusters;
  }

  // Decompresses to two bf16 vectors. `packed_ofs` must be a multiple of two
  // vectors so that we only have to load one group's table.
  template <class DBF, HWY_IF_BF16_D(DBF)>
  static HWY_INLINE void Dec2(DBF dbf,
                              const PackedSpan<const NuqStream>& packed,
                              const size_t packed_ofs, hn::Vec<DBF>& raw0,
                              hn::Vec<DBF>& raw1) {
    const hn::RebindToUnsigned<decltype(dbf)> d16;
    const D8HFromD16<DBF> d8h;
    using V16 = hn::Vec<decltype(d16)>;
    using V8H = hn::Vec<decltype(d8h)>;

    const size_t within_group = packed_ofs % kGroupSize;
    HWY_DASSERT(within_group % (2 * hn::Lanes(d16)) == 0);
    const uint8_t* table = &packed.ptr->byte + TableByteOffset(packed_ofs);
    const uint8_t* indices = table + kClusters + hwy::DivCeil(within_group, 2);

    V16 tbl1 = Zero(d16);
    const V16 tbl0 = LoadTable(d16, table, &tbl1);

    const V8H nibbles = hn::LoadU(d8h, indices);

    V16 c0, c1;
    TableLookups(d16, tbl0, tbl1, nibbles, c0, c1);
    raw0 = BitCast(dbf, c0);
    raw1 = BitCast(dbf, c1);
  }

  // Decompresses to two f32 vectors. `packed_ofs` must be a multiple of two
  // vectors so that we only have to load one group's table.
  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void Dec2(DF df, const PackedSpan<const NuqStream>& packed,
                              const size_t packed_ofs, hn::Vec<DF>& raw0,
                              hn::Vec<DF>& raw1) {
    const hn::Repartition<BF16, decltype(df)> dbf;
    const hn::RebindToUnsigned<decltype(dbf)> d16;
    const hn::Half<D8HFromD16<decltype(d16)>> d8q;
    using V8Q = hn::Vec<decltype(d8q)>;
    using V16 = hn::Vec<decltype(d16)>;

    const size_t within_group = packed_ofs % kGroupSize;
    HWY_DASSERT(within_group % (2 * hn::Lanes(df)) == 0);
    const uint8_t* table = &packed.ptr->byte + TableByteOffset(packed_ofs);
    const uint8_t* indices = table + kClusters + hwy::DivCeil(within_group, 2);

    V16 tbl1 = Zero(d16);
    const V16 tbl0 = LoadTable(d16, table, &tbl1);

    // The single-vector TableLookups overload only calls OrderedUnpackU16<0>,
    // which expects a quarter vector of bytes.
    const V8Q nibbles = hn::LoadU(d8q, indices);

    // TODO(janwas): From janwas: on AVX-512 I imagine we can get a
    // bit more speed for this function by changing LoadTable to return floats,
    // then we could have a single lookup here instead of PromoteUpperTo which
    // is not cheap.
    const V16 c0 = TableLookups(d16, tbl0, tbl1, nibbles);
    raw0 = hn::PromoteLowerTo(df, BitCast(dbf, c0));
    raw1 = hn::PromoteUpperTo(df, BitCast(dbf, c0));
  }

  template <class D, typename Raw = hn::TFromD<D>>
  static HWY_INLINE void DecompressAndZeroPad(
      D d, const PackedSpan<const NuqStream>& packed, size_t packed_ofs,
      Raw* HWY_RESTRICT raw, size_t num) {
    // If unaligned, load elements from the first group and update the args,
    // from which we compute new tables/indices below.
    size_t current_offset = packed_ofs;
    if (size_t within_group = packed_ofs % kGroupSize; within_group != 0) {
      const uint8_t* tables =
          &packed.ptr->byte + TableByteOffset(current_offset);
      const uint8_t* indices = tables + kClusters + within_group / 2;
      const size_t remaining = HWY_MIN(num, kGroupSize - within_group);

      DecPartialGroup(d, tables, indices, raw, remaining);
      packed_ofs += remaining;
      current_offset += remaining;
      raw += remaining;
      num -= remaining;
      if (num == 0) return;
    }

    HWY_DASSERT(packed_ofs % kGroupSize == 0);

    const size_t num_groups = hwy::DivCeil(num, kGroupSize);
    HWY_UNROLL(1)
    for (size_t g = 0; g < num_groups - 1; ++g) {
      const uint8_t* tables =
          &packed.ptr->byte + TableByteOffset(current_offset);
      const uint8_t* indices = tables + kClusters;
      DecWholeGroup(d, tables, indices, raw + g * kGroupSize);
      current_offset += kGroupSize;
    }

    const size_t g = num_groups - 1;
    const uint8_t* tables = &packed.ptr->byte + TableByteOffset(current_offset);
    const uint8_t* indices = tables + kClusters;
    DecPartialGroup(d, tables, indices, raw + g * kGroupSize,
                    num - g * kGroupSize);
  }

 private:
  template <class DBF, HWY_IF_BF16_D(DBF)>
  static HWY_INLINE void DecWholeGroup(DBF dbf,
                                       const uint8_t* HWY_RESTRICT table,
                                       const uint8_t* HWY_RESTRICT indices,
                                       BF16* HWY_RESTRICT raw_bf) {
    const hn::RebindToUnsigned<decltype(dbf)> d16;
    const D8HFromD16<DBF> d8h;
    using V16 = hn::Vec<decltype(d16)>;
    using V8H = hn::Vec<decltype(d8h)>;
    const size_t N16 = hn::Lanes(d16);

    V16 tbl1 = Zero(d16);
    const V16 tbl0 = LoadTable(d16, table, &tbl1);

    HWY_UNROLL(1)
    for (size_t i = 0; i < kGroupSize; i += 2 * N16) {
      const V8H nibbles = hn::LoadU(d8h, indices + i / 2);
      V16 c0, c1;
      TableLookups(d16, tbl0, tbl1, nibbles, c0, c1);
      hn::StoreU(BitCast(dbf, c0), dbf, raw_bf + i + 0 * N16);
      hn::StoreU(BitCast(dbf, c1), dbf, raw_bf + i + 1 * N16);
    }
  }

  // Called for first and last group.
  template <class DBF, HWY_IF_BF16_D(DBF)>
  static HWY_INLINE void DecPartialGroup(DBF dbf,
                                         const uint8_t* HWY_RESTRICT table,
                                         const uint8_t* HWY_RESTRICT indices,
                                         BF16* HWY_RESTRICT raw_bf,
                                         size_t num) {
    HWY_DASSERT(num <= kGroupSize);

    const hn::RebindToUnsigned<decltype(dbf)> d16;
    const D8HFromD16<DBF> d8h;
    using V16 = hn::Vec<decltype(d16)>;
    using V8H = hn::Vec<decltype(d8h)>;
    const size_t N16 = hn::Lanes(d16);

    V16 tbl1 = Zero(d16);
    const V16 tbl0 = LoadTable(d16, table, &tbl1);

    size_t i = 0;

    if (num >= 2 * N16) {
      HWY_UNROLL(1)
      for (; i <= num - 2 * N16; i += 2 * N16) {
        const V8H nibbles = hn::LoadU(d8h, indices + i / 2);
        V16 c0, c1;
        TableLookups(d16, tbl0, tbl1, nibbles, c0, c1);
        hn::StoreU(BitCast(dbf, c0), dbf, raw_bf + i + 0 * N16);
        hn::StoreU(BitCast(dbf, c1), dbf, raw_bf + i + 1 * N16);
      }
    }

    const size_t remaining = num - i;
    HWY_DASSERT(remaining < 2 * N16);
    if (HWY_UNLIKELY(remaining != 0)) {
      // i is even, but remaining might not be.
      const V8H nibbles =
          hn::LoadN(d8h, indices + i / 2, hwy::DivCeil(remaining, 2));

      V16 c0, c1;
      TableLookups(d16, tbl0, tbl1, nibbles, c0, c1);
      // Out of bounds `nibbles` are 0, but this does not yet guarantee
      // c0/c1 are, because centers[0] might not be 0.
      c0 = hn::IfThenElseZero(hn::FirstN(d16, remaining), c0);
      hn::StoreU(BitCast(dbf, c0), dbf, raw_bf + i);
      // Callers only pad to one vector, so check before storing the second.
      if (remaining > N16) {
        c1 = hn::IfThenElseZero(hn::FirstN(d16, remaining - N16), c1);
        hn::StoreU(BitCast(dbf, c1), dbf, raw_bf + i + N16);
      }
    }
  }

  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void DecWholeGroup(DF df, const uint8_t* HWY_RESTRICT table,
                                       const uint8_t* HWY_RESTRICT indices,
                                       float* HWY_RESTRICT raw_f) {
    const hn::Repartition<BF16, decltype(df)> dbf;
    const hn::RebindToUnsigned<decltype(dbf)> d16;
    const D8HFromD16<decltype(d16)> d8h;
    using V16 = hn::Vec<decltype(d16)>;
    using V8H = hn::Vec<decltype(d8h)>;
    using VF = hn::Vec<decltype(df)>;
    const size_t NF = hn::Lanes(df);

    V16 tbl1 = Zero(d16);
    const V16 tbl0 = LoadTable(d16, table, &tbl1);

    HWY_UNROLL(1)
    for (size_t i = 0; i < kGroupSize; i += 4 * NF) {
      const V8H nibbles = hn::LoadU(d8h, indices + i / 2);
      V16 c0, c1;
      TableLookups(d16, tbl0, tbl1, nibbles, c0, c1);
      const VF f0 = hn::PromoteLowerTo(df, BitCast(dbf, c0));
      const VF f1 = hn::PromoteUpperTo(df, BitCast(dbf, c0));
      const VF f2 = hn::PromoteLowerTo(df, BitCast(dbf, c1));
      const VF f3 = hn::PromoteUpperTo(df, BitCast(dbf, c1));
      hn::StoreU(f0, df, raw_f + i + 0 * NF);
      hn::StoreU(f1, df, raw_f + i + 1 * NF);
      hn::StoreU(f2, df, raw_f + i + 2 * NF);
      hn::StoreU(f3, df, raw_f + i + 3 * NF);
    }
  }

  // Called for first and last group.
  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void DecPartialGroup(DF df,
                                         const uint8_t* HWY_RESTRICT table,
                                         const uint8_t* HWY_RESTRICT indices,
                                         float* HWY_RESTRICT raw_f,
                                         const size_t num) {
    HWY_DASSERT(num <= kGroupSize);

    const hn::Repartition<BF16, decltype(df)> dbf;
    const hn::RebindToUnsigned<decltype(dbf)> d16;
    const D8HFromD16<decltype(d16)> d8h;
    using V16 = hn::Vec<decltype(d16)>;
    using V8H = hn::Vec<decltype(d8h)>;
    using VF = hn::Vec<decltype(df)>;
    const size_t NF = hn::Lanes(df);

    V16 tbl1 = Zero(d16);
    const V16 tbl0 = LoadTable(d16, table, &tbl1);

    size_t i = 0;

    if (num >= 4 * NF) {
      HWY_UNROLL(1)
      for (; i <= num - 4 * NF; i += 4 * NF) {
        const V8H nibbles = hn::LoadU(d8h, indices + i / 2);
        V16 c0, c1;
        TableLookups(d16, tbl0, tbl1, nibbles, c0, c1);
        const VF f0 = hn::PromoteLowerTo(df, BitCast(dbf, c0));
        const VF f1 = hn::PromoteUpperTo(df, BitCast(dbf, c0));
        const VF f2 = hn::PromoteLowerTo(df, BitCast(dbf, c1));
        const VF f3 = hn::PromoteUpperTo(df, BitCast(dbf, c1));
        hn::StoreU(f0, df, raw_f + i + 0 * NF);
        hn::StoreU(f1, df, raw_f + i + 1 * NF);
        hn::StoreU(f2, df, raw_f + i + 2 * NF);
        hn::StoreU(f3, df, raw_f + i + 3 * NF);
      }
    }

    const size_t remaining = num - i;

    HWY_DASSERT(remaining < 4 * NF);
    if (HWY_UNLIKELY(remaining != 0)) {
      // i is even, but remaining might not be.
      const V8H nibbles =
          hn::LoadN(d8h, indices + i / 2, hwy::DivCeil(remaining, 2));

      V16 c0, c1;
      TableLookups(d16, tbl0, tbl1, nibbles, c0, c1);
      const VF f0 = hn::PromoteLowerTo(df, BitCast(dbf, c0));
      const VF f1 = hn::PromoteUpperTo(df, BitCast(dbf, c0));
      const VF f2 = hn::PromoteLowerTo(df, BitCast(dbf, c1));
      const VF f3 = hn::PromoteUpperTo(df, BitCast(dbf, c1));
      // `raw_f` is only guaranteed to padded to NF, hence we cannot store all
      // four vectors. We could conditionally store vectors either to `raw_f`
      // or a buffer. However, we still have to mask because only `nibbles`
      // are guaranteed to be 0, not c0/c1. Copying also involves branches,
      // so we fully unroll the copy loop to avoid a buffer. We could also
      // change the contract to pad to four vectors, but it would anyway be
      // better to decompress to bf16.
      if (remaining <= 1 * NF) {
        const hn::Mask<DF> mask = hn::FirstN(df, remaining);
        hn::StoreU(hn::IfThenElseZero(mask, f0), df, raw_f + i + 0 * NF);
        return;
      }
      hn::StoreU(f0, df, raw_f + i + 0 * NF);
      if (remaining <= 2 * NF) {
        const hn::Mask<DF> mask = hn::FirstN(df, remaining - NF);
        hn::StoreU(hn::IfThenElseZero(mask, f1), df, raw_f + i + 1 * NF);
        return;
      }
      hn::StoreU(f1, df, raw_f + i + 1 * NF);
      if (remaining <= 3 * NF) {
        const hn::Mask<DF> mask = hn::FirstN(df, remaining - 2 * NF);
        hn::StoreU(hn::IfThenElseZero(mask, f2), df, raw_f + i + 2 * NF);
        return;
      }
      hn::StoreU(f2, df, raw_f + i + 2 * NF);
      {
        const hn::Mask<DF> mask = hn::FirstN(df, remaining - 3 * NF);
        hn::StoreU(hn::IfThenElseZero(mask, f3), df, raw_f + i + 3 * NF);
      }
    }
  }
};  // NuqCodec

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_NUQ_INL_H_
