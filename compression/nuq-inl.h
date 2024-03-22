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

// copybara:import_next_line:gemma_cpp
#include "compression/nuq.h"
// copybara:import_next_line:gemma_cpp
#include "compression/sfp.h"
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

// copybara:import_next_line:gemma_cpp
#include "compression/sfp-inl.h"
#include "hwy/contrib/sort/vqsort-inl.h"
#include "hwy/highway.h"

#ifndef HWY_IF_CONSTEXPR
#define HWY_IF_CONSTEXPR if
#endif

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// For internal use by NuqCodec.
class NuqClustering {
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
   public:
    explicit ClusterCost(const float* sorted) {
      cumsum_[0] = cumsum2_[0] = 0.0;
      for (size_t i = 0; i < kGroupSize; ++i) {
        const float x = FloatPayload::Clear(sorted[i]);
        cumsum_[1 + i] = x + cumsum_[i];
        cumsum2_[1 + i] = x * x + cumsum2_[i];
      }

      inv_len_[0] = 0.0f;  // unused
      for (size_t i = 0; i <= kGroupSize; ++i) {
        inv_len_[i] = 1.0f / static_cast<float>(i);
      }
    }

    float SumOfSorted(size_t first, size_t last) const {
      return cumsum_[last + 1] - cumsum_[first];
    }

    // Returns cost of clustering first..last with their mean, for a vector of
    // last. O(1) thanks to cumulative sums, which works for Lp-norms with p >
    // 1; we choose p=2 for simplicity (fewer terms).
    template <class DF>
    hn::Vec<DF> operator()(DF df, size_t first, size_t last) const {
      // Callers are responsible for ignoring lanes where last < first.
      HWY_DASSERT(first < kGroupSize);
      HWY_DASSERT(last < kGroupSize);
      const size_t len = last - first + 1;
      const hn::Vec<DF> vlen =
          hn::Iota(df, static_cast<float>(static_cast<int>(len)));

      const hn::Vec<DF> u_lo = hn::Set(df, cumsum_[first]);
      const hn::Vec<DF> u_lo2 = hn::Set(df, cumsum2_[first]);
      const hn::Vec<DF> hi = hn::LoadU(df, cumsum_ + last + 1);
      const hn::Vec<DF> hi2 = hn::LoadU(df, cumsum2_ + last + 1);
      const hn::Vec<DF> sum = hn::Sub(hi, u_lo);
      const hn::Vec<DF> sum2 = hn::Sub(hi2, u_lo2);

      // Compute mean: table lookup is faster than division.
      const hn::Vec<DF> mu = hn::Mul(sum, hn::LoadU(df, inv_len_ + len));

      // (x - mu)^2 = sum2 - 2mu*sum + mu^2
      const hn::Vec<DF> mu2 = hn::Mul(mu, mu);
      const hn::Vec<DF> two_mu = hn::Add(mu, mu);
      return hn::NegMulAdd(two_mu, sum, hn::MulAdd(vlen, mu2, sum2));
    }

   private:
    // Float has enough precision for our relatively small kGroupSize (256).
    float cumsum_[kGroupSize + 1];
    float cumsum2_[kGroupSize + 1];
    float inv_len_[kGroupSize + 1];
  };

  // Cost of clustering 0..last, where the rightmost cluster is j..last. This is
  // called in a loop over j, and we return the vector of costs for a batch of
  // last = [last, last + N).
  template <class DF>
  static HWY_INLINE hn::Vec<DF> ClusterDynProg(
      DF df, const AlignedMatrix<float>& D, const ClusterCost& cc,
      const size_t num_clusters, const size_t last, const size_t j) {
    HWY_DASSERT(last < kGroupSize);
    HWY_DASSERT(0 != j && j < kGroupSize);

    const hn::RebindToSigned<decltype(df)> di;
    using VF = hn::Vec<decltype(df)>;
    using VI = hn::Vec<decltype(di)>;
    using MI = hn::Mask<decltype(di)>;

    const VI vlast = hn::Iota(di, static_cast<int32_t>(last));

    // We have a non-empty rightmost cluster if j <= last <==> j-1 < last.
    const MI valid = hn::Lt(hn::Set(di, static_cast<int32_t>(j) - 1), vlast);
    // If not valid, return an arbitrary high cost, which will not be the min.
    const VF max = hn::Set(df, 1E38f);
    // Cost of clustering 0..j-1 with one fewer cluster than now.
    const VF vd = hn::Set(df, D(num_clusters - 1, j - 1));
    // Eq2: add to that the cost of another cluster from j..last.
    return hn::MaskedAddOr(max, RebindMask(df, valid), vd, cc(df, j, last));
  }

 public:
  // Clusters `kGroupSize` values in `x`, which need not be sorted already nor
  // aligned, by choosing and filling `centers` (size `kClusters`, ascending
  // order, not necessarily equal to one of the `x`). Fills `indices` with the
  // index of the cluster to which each `x` belongs (16-bit for bit-packing).
  // `buf` is per-thread.
  //
  // Returns the number of unused clusters, i.e., the starting index within
  // `centers`; prior centers are zero-initialized.
  //
  // O(kClusters * kGroupSize * kGroupSize), but the constant factors are so low
  // that this is about 5 times as fast as the O(kClusters * kGroupSize) SMAWK
  // as implemented in FAISS, for our kGroupSize of 256.
  template <class DF>
  static HWY_NOINLINE size_t ClusterExactL2(DF df, const float* x,
                                            ClusterBuf& buf,
                                            float* HWY_RESTRICT centers,
                                            uint16_t* HWY_RESTRICT indices) {
    const hn::RebindToSigned<decltype(df)> di;
    using VF = hn::Vec<decltype(df)>;
    using VI = hn::Vec<decltype(di)>;
    const VI k1 = hn::Set(di, 1);
    const size_t N = hn::Lanes(df);

    HWY_ALIGN float sorted_and_i[kGroupSize];
    for (size_t i = 0; i < kGroupSize; ++i) {
      sorted_and_i[i] = FloatPayload::Set(x[i], i);
    }
    hn::VQSortStatic(sorted_and_i, kGroupSize, hwy::SortAscending());
    ClusterCost cc(sorted_and_i);

    // Reference: https://arxiv.org/abs/1701.07204
    // D[k-1][m] is the lowest cost of clustering x1..m into k clusters.
    AlignedMatrix<float>& D = buf.d;
    // T[k][m] is the starting index within sorted_and_i[] of the k-th cluster.
    AlignedMatrix<int32_t>& T = buf.t;

    // Initialize the first rows for a single cluster.
    for (size_t last = 0; last < kGroupSize; last += N) {
      hn::Store(cc(df, 0, last), df, &D(0, last));  // Cost of 0..last
      hn::Store(Zero(di), di, &T(0, last));         // Cluster index = 0
    }

    for (size_t num_clusters = 1; num_clusters < kClusters; ++num_clusters) {
      // For each batch starting at `last`, one per lane:
      for (size_t last = 0; last < kGroupSize; last += N) {
        VF min = cc(df, 0, last);
        VI arg = hn::Zero(di);
        // For each j (start of rightmost cluster):
        VI vj = k1;
        for (size_t j = 1; j < last + N; ++j, vj = Add(vj, k1)) {
          const VF c = ClusterDynProg(df, D, cc, num_clusters, last, j);

          // Retain the min cost and the j index that caused it.
          const auto less = hn::Lt(c, min);
          min = hn::IfThenElse(less, c, min);
          arg = hn::IfThenElse(RebindMask(di, less), vj, arg);
        }
        hn::Store(min, df, &D(num_clusters, last));
        hn::Store(arg, di, &T(num_clusters, last));
      }
    }

    // Backtrack to find centers. Clusters are [T(k, last), last].
    size_t last = kGroupSize - 1;
    size_t unused_clusters = 0;
    for (size_t k = kClusters - 1; k < kClusters; --k) {
      const size_t start = static_cast<size_t>(T(k, last));
      // Center = mean, O(1) thanks to cumulative sums.
      const float sum = cc.SumOfSorted(start, last);
      const int size = static_cast<int>(last) - static_cast<int>(start) + 1;
      HWY_DASSERT(0 < size && size <= static_cast<int>(kGroupSize));
      centers[k] = sum / static_cast<float>(size);

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
      // Centers are in ascending order.
      for (size_t i = unused_clusters + 1; i < kClusters; ++i) {
        HWY_DASSERT(centers[i] >= centers[i - 1]);
      }
    }
    return unused_clusters;
  }
};  // NuqClustering

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
  // Packs four u16 vectors' lanes to nibbles within one vector, in order, and
  // stores that vector to `out`.
  template <class D16, class V16 = hn::Vec<D16>>
  static HWY_INLINE void OrderedPackU16(D16 d16, V16 in0, V16 in1, V16 in2,
                                        V16 in3, uint8_t* HWY_RESTRICT out) {
    const hn::Repartition<uint8_t, D16> d8;
    const hn::Repartition<uint32_t, D16> d32;
    const hn::Repartition<uint64_t, D16> d64;
    using V8 = hn::Vec<decltype(d8)>;

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
    V8 packed;
    if (HWY_TARGET <= HWY_AVX3_DL || !HWY_ARCH_X86) {
      // 8-bit ConcatEven is efficient. Let digits denote eight u8 lanes
      // of u8_1/0: ?d?3 ?c?2 / ?b?1 ?a?0. 8-bit ConcatEven = d3c2 b1a0, and
      // again with the second x2_1 gives 7654 3210.
      const V8 x2_0 = hn::ConcatEven(d8, BitCast(d8, u8_1), BitCast(d8, u8_0));
      const V8 x2_1 = hn::ConcatEven(d8, BitCast(d8, u8_3), BitCast(d8, u8_2));
      packed = hn::ConcatEven(d8, x2_1, x2_0);
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
      packed = hn::BitCast(d8, hn::ConcatEven(d16, x2_1, x2_0));
    }
    hn::StoreU(packed, d8, out);
  }

  // Unpacks `Lanes(d16)` nibbles to u16 lanes. The first comes from the low
  // nibble of packed[0], then its high nibble, then the next low nibble, etc.
  template <class D16, class V16 = hn::Vec<D16>>
  static HWY_INLINE V16 OrderedUnpackU16(D16 d16, const uint8_t* packed) {
    const hn::Repartition<uint8_t, D16> d8;
    using V8 = hn::Vec<decltype(d8)>;
    const hn::CappedTag<uint8_t, d16.MaxBytes() / 4> d_load;

    // We replicate each byte 4x, so that its two nibbles propagate to both
    // u16 lanes that they will initialize. The only performance-portable op to
    // replicate bytes is TableLookupBytes, which shuffles 128-bit blocks
    // independently. Thus each block receives 4 packed bytes, replicates them
    // 4x, shifts/masks, and casts to 8 u16 lanes.
    //
    // Loading 16 bytes via LoadDup128 only works on AVX3; for smaller vectors,
    // it may trigger asan errors from overrunning the end. We thus special-case
    // vector lengths, handling any non-constexpr, and constexpr <= 512 bit.
    V8 rep4;
    if (HWY_HAVE_SCALABLE) {
      // Non constexpr length: 4 per whole block equals size/4.
      const size_t num_bytes = HWY_MAX(1, hn::Lanes(d8) / 4);
      const V8 bytes = hn::LoadN(d8, packed, num_bytes);
      // Replicate bytes 4x: lowest 4 = 0, next 4 = 1 etc.
      const V8 idx = hn::And(hn::Iota(d8, 0), hn::Set(d8, 0xFCu));
      rep4 = hn::TableLookupBytes(bytes, idx);
    } else if (hn::MaxLanes(d16) <= 8) {  // <= 128-bit
      const V8 bytes = hn::ResizeBitCast(d8, hn::LoadU(d_load, packed));
      alignas(16) static constexpr uint8_t kRep4[16] = {
          HWY_REP4(0), HWY_REP4(1), HWY_REP4(2), HWY_REP4(3)};
      rep4 = hn::TableLookupBytes(bytes, hn::Load(d8, kRep4));
    } else if (HWY_TARGET <= HWY_AVX3_DL || !HWY_ARCH_X86) {
      // Plain load, can do 256..512-bit permute across blocks.
      const V8 bytes = hn::ResizeBitCast(d8, hn::LoadU(d_load, packed));
      alignas(64) static constexpr uint8_t kRep4[64] = {
          HWY_REP4(0),  HWY_REP4(1),  HWY_REP4(2),  HWY_REP4(3),
          HWY_REP4(4),  HWY_REP4(5),  HWY_REP4(6),  HWY_REP4(7),
          HWY_REP4(8),  HWY_REP4(9),  HWY_REP4(10), HWY_REP4(11),
          HWY_REP4(12), HWY_REP4(13), HWY_REP4(14), HWY_REP4(15)};
      rep4 = hn::TableLookupLanes(bytes, hn::SetTableIndices(d8, kRep4));
    } else if (hn::MaxLanes(d16) == 16) {  // 256-bit
      const V8 bytes = hn::ResizeBitCast(d8, hn::LoadU(d_load, packed));
      // First copy to upper block for TableLookupBytes. This is slightly
      // faster than 64-bit BroadcastLane.
      const V8 bcast = hn::ConcatLowerLower(d8, bytes, bytes);
      alignas(32) static constexpr uint8_t kRep4[32] = {
          HWY_REP4(0), HWY_REP4(1), HWY_REP4(2), HWY_REP4(3),
          HWY_REP4(4), HWY_REP4(5), HWY_REP4(6), HWY_REP4(7)};
      rep4 = hn::TableLookupBytes(bcast, hn::Load(d8, kRep4));
    } else if (hn::MaxLanes(d16) == 32) {  // 512-bit
      const V8 bytes = hn::LoadDup128(d8, packed);
      alignas(64) static constexpr uint8_t kRep4[64] = {
          HWY_REP4(0),  HWY_REP4(1),  HWY_REP4(2),  HWY_REP4(3),
          HWY_REP4(4),  HWY_REP4(5),  HWY_REP4(6),  HWY_REP4(7),
          HWY_REP4(8),  HWY_REP4(9),  HWY_REP4(10), HWY_REP4(11),
          HWY_REP4(12), HWY_REP4(13), HWY_REP4(14), HWY_REP4(15)};
      rep4 = hn::TableLookupBytes(bytes, hn::Load(d8, kRep4));
    } else {
      HWY_DASSERT(false);
    }

    const V16 mask4 = hn::Set(d16, 0xF);
    const V16 u16 = BitCast(d16, rep4);
    // In-order unpack. Right-shift odd u16 by 4. Example with two packed
    // bytes, one digit representing a nibble:
    // 32 32 32 32 | 10 10 10 10  u16
    // z3 23 32 32 | z1 01 10 10  OddEven+ShiftRight
    // zz z3 zz z2 | zz z1 zz z0  And (unpacked result)
    return hn::And(mask4, hn::OddEven(hn::ShiftRight<4>(u16), u16));
  }
};

// Encode/decode functions.
class NuqCodec {
  // 256-bit vectors can hold 16 bf16, otherwise we require 2x128-bit.
  template <class DU>
  static constexpr size_t NumTables(DU du) {
    return (!HWY_HAVE_SCALABLE && du.MaxBytes() >= 32) ? 1 : 2;
  }

  // Unpacks `centers` from SFP into bf16 and loads them into one or two vectors
  // for use by [Two]TableLookups. Returns as u16 because TableLookupLanes might
  // not be available for bf16.
  template <class DU, HWY_IF_U16_D(DU)>
  static HWY_INLINE hn::Vec<DU> LoadTable(DU du, const uint8_t* centers,
                                          hn::Vec<DU>* HWY_RESTRICT tbl1) {
    // Cap to the table size (kClusters) for decoding SFP - sufficient, and may
    // be faster than a large vector.
    const hn::CappedTag<hwy::bfloat16_t, kClusters> d_table;
    // We ResizeCast tables to DU: if DU is bigger, table lookups will only
    // access lanes < kClusters. If DU is smaller (128-bit), we have 2 tables.
    HWY_DASSERT(hn::Lanes(du) >= hn::Lanes(d_table) || NumTables(du) == 2);

    HWY_ALIGN hwy::bfloat16_t table[kClusters];
    SfpCodec::Dec(d_table, reinterpret_cast<const SfpStream*>(centers),
                  kClusters, table);

    // If we assume >= 128-bit vectors, we can use [Two]TableLookupLanes
    // instead of TableLookupBytes, which requires extra interleaving of lo/hi.
    HWY_DASSERT(hn::Lanes(du) >= 8);

    HWY_IF_CONSTEXPR(NumTables(du) == 2) {
      // Reduce cap for second half to avoid loading past the end of the table.
      const hn::CappedTag<hwy::bfloat16_t, kClusters / 2> d_table2;
      *tbl1 = hn::ResizeBitCast(du, hn::LoadU(d_table2, table + kClusters / 2));
    }
    return hn::ResizeBitCast(du, hn::Load(d_table, table));
  }

  // Unpacks per-weight indices and sets c0/c1 to the corresponding centers.
  template <class DU>
  static HWY_INLINE void TableLookups(DU du, hn::Vec<DU> tbl0, hn::Vec<DU> tbl1,
                                      const uint8_t* packed, hn::Vec<DU>& c0,
                                      hn::Vec<DU>& c1) {
    using V16 = hn::Vec<decltype(du)>;
    const size_t N16 = hn::Lanes(du);

    const V16 idx0 = NibbleCodec::OrderedUnpackU16(du, packed);
    const V16 idx1 = NibbleCodec::OrderedUnpackU16(du, packed + N16 / 2);

    const auto indices0 = hn::IndicesFromVec(du, idx0);
    const auto indices1 = hn::IndicesFromVec(du, idx1);

    HWY_IF_CONSTEXPR(NumTables(du) == 1) {
      (void)tbl1;
      c0 = hn::TableLookupLanes(tbl0, indices0);
      c1 = hn::TableLookupLanes(tbl0, indices1);
    }
    HWY_IF_CONSTEXPR(NumTables(du) == 2) {  // `else` is poorly formatted.
      c0 = hn::TwoTablesLookupLanes(du, tbl0, tbl1, indices0);
      c1 = hn::TwoTablesLookupLanes(du, tbl0, tbl1, indices1);
    }
  }

 public:
  // Encodes `num` floats starting from `in`. `out` points to compressed
  // storage for `out_capacity` values and `out_ofs` indicates the destination
  // offset within it, in units of float values, for parallel encoding by
  // multiple threads. `num`, `out_capacity`, and `out_ofs` must all be
  // multiples of `kGroupSize`. Returns the total number of unused clusters,
  // which is expected to be zero.
  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE size_t Enc(DF df, const float* const in, const size_t num,
                               ClusterBuf& buf, const size_t out_capacity,
                               NuqStream* const out, const size_t out_ofs) {
    const hn::Repartition<uint16_t, DF> d16;
    using V16 = hn::Vec<decltype(d16)>;

    const size_t N16 = hn::Lanes(d16);
    HWY_ASSERT(kGroupSize >= 4 * N16);

    HWY_ASSERT(out_ofs + num <= out_capacity);
    buf.Resize(num);
    HWY_ASSERT(num % kGroupSize == 0);
    HWY_ASSERT(out_capacity % kGroupSize == 0);
    HWY_ASSERT(out_ofs % kGroupSize == 0);
    const size_t num_groups = num / kGroupSize;
    const size_t ofs_groups = out_ofs / kGroupSize;

    size_t unused_clusters = 0;
    for (size_t g = 0; g < num_groups; ++g) {
      const float* HWY_RESTRICT g_in = in + g * kGroupSize;
      float* HWY_RESTRICT g_centers = buf.centers.get() + g * kClusters;
      uint16_t* HWY_RESTRICT g_idx = buf.idx.get() + g * kGroupSize;
      unused_clusters +=
          NuqClustering::ClusterExactL2(df, g_in, buf, g_centers, g_idx);
    }

    uint8_t* centers = &out->byte + ofs_groups * kClusters;
    SfpCodec::Enc(df, buf.centers.get(), num_groups * kClusters,
                  reinterpret_cast<SfpStream*>(centers));
    uint8_t* packed_start = &out->byte + NuqStream::PackedStart(out_capacity) +
                            ofs_groups * kGroupSize / 2;

    HWY_UNROLL(1)
    for (size_t g = 0; g < num_groups; ++g) {
      const uint16_t* HWY_RESTRICT g_idx = buf.idx.get() + g * kGroupSize;
      uint8_t* HWY_RESTRICT g_packed = packed_start + g * kGroupSize / 2;

      HWY_UNROLL(1)
      for (size_t i = 0; i < kGroupSize; i += 4 * N16) {
        const V16 idx0 = hn::LoadU(d16, g_idx + i + N16 * 0);
        const V16 idx1 = hn::LoadU(d16, g_idx + i + N16 * 1);
        const V16 idx2 = hn::LoadU(d16, g_idx + i + N16 * 2);
        const V16 idx3 = hn::LoadU(d16, g_idx + i + N16 * 3);
        NibbleCodec::OrderedPackU16(d16, idx0, idx1, idx2, idx3,
                                    g_packed + i / 2);
      }
    }

    return unused_clusters;
  }

  // Decodes `num` values from the stream `in`, starting at the offset `in_ofs`
  // (in units of values), to bf16 in `out`. `in_capacity`, `in_ofs` and `num`
  // must all be multiples of `kGroupSize`.
  template <class DBF, HWY_IF_BF16_D(DBF)>
  static HWY_INLINE void Dec(DBF dbf, const size_t in_capacity,
                             const NuqStream* const in, const size_t in_ofs,
                             hwy::bfloat16_t* const out, const size_t num) {
    const hn::RebindToUnsigned<decltype(dbf)> d16;
    using V16 = hn::Vec<decltype(d16)>;

    const size_t N16 = hn::Lanes(d16);
    HWY_DASSERT(kGroupSize >= 4 * N16);

    HWY_DASSERT(in_ofs + num <= in_capacity);
    HWY_DASSERT(in_capacity % kGroupSize == 0);
    HWY_DASSERT(in_ofs % kGroupSize == 0);
    HWY_DASSERT(num % kGroupSize == 0);
    const size_t num_groups = num / kGroupSize;
    const size_t ofs_groups = in_ofs / kGroupSize;
    const uint8_t* tables = &in->byte + ofs_groups * kClusters;
    const uint8_t* packed_start = &in->byte +
                                  NuqStream::PackedStart(in_capacity) +
                                  ofs_groups * kGroupSize / 2;

    HWY_UNROLL(1)
    for (size_t g = 0; g < num_groups; ++g) {
      const uint8_t* g_centers = tables + g * kClusters;
      const uint8_t* HWY_RESTRICT g_packed = packed_start + g * kGroupSize / 2;
      hwy::bfloat16_t* HWY_RESTRICT g_out = out + g * kGroupSize;

      V16 tbl1 = Zero(d16);
      const V16 tbl0 = LoadTable(d16, g_centers, &tbl1);

      HWY_UNROLL(1)
      for (size_t i = 0; i < kGroupSize; i += 2 * N16) {
        V16 c0, c1;
        TableLookups(d16, tbl0, tbl1, g_packed + i / 2, c0, c1);
        hn::StoreU(BitCast(dbf, c0), dbf, g_out + i + N16 * 0);
        hn::StoreU(BitCast(dbf, c1), dbf, g_out + i + N16 * 1);
      }
    }
  }

  // Decodes `num` values from the stream `in`, starting at the offset
  // `in_ofs` (in units of values), to f32 in `out`. `in_capacity`,
  // `in_ofs` and `num` must all be multiples of `kGroupSize`.
  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void Dec(DF df, const size_t in_capacity,
                             const NuqStream* const in, const size_t in_ofs,
                             float* const out, const size_t num) {
    const hn::Repartition<hwy::bfloat16_t, DF> dbf;
    const hn::RebindToUnsigned<decltype(dbf)> d16;
    using V16 = hn::Vec<decltype(d16)>;
    using VF = hn::Vec<DF>;

    const size_t NF = hn::Lanes(df);
    HWY_DASSERT(kGroupSize >= 4 * NF);

    HWY_DASSERT(in_ofs + num <= in_capacity);
    HWY_DASSERT(in_capacity % kGroupSize == 0);
    HWY_DASSERT(in_ofs % kGroupSize == 0);
    HWY_DASSERT(num % kGroupSize == 0);
    const size_t ofs_groups = in_ofs / kGroupSize;
    const size_t num_groups = num / kGroupSize;
    const uint8_t* tables = &in->byte + ofs_groups * kClusters;
    const uint8_t* packed_start = &in->byte +
                                  NuqStream::PackedStart(in_capacity) +
                                  ofs_groups * kGroupSize / 2;

    HWY_UNROLL(1)
    for (size_t g = 0; g < num_groups; ++g) {
      const uint8_t* g_centers = tables + g * kClusters;
      const uint8_t* HWY_RESTRICT g_packed = packed_start + g * kGroupSize / 2;
      float* HWY_RESTRICT g_out = out + g * kGroupSize;

      V16 tbl1 = Zero(d16);
      const V16 tbl0 = LoadTable(d16, g_centers, &tbl1);

      HWY_UNROLL(1)
      for (size_t i = 0; i < kGroupSize; i += 4 * NF) {
        V16 c0, c1;
        TableLookups(d16, tbl0, tbl1, g_packed + i / 2, c0, c1);
        const VF f0 = hn::PromoteLowerTo(df, BitCast(dbf, c0));
        const VF f1 = hn::PromoteUpperTo(df, BitCast(dbf, c0));
        const VF f2 = hn::PromoteLowerTo(df, BitCast(dbf, c1));
        const VF f3 = hn::PromoteUpperTo(df, BitCast(dbf, c1));
        hn::StoreU(f0, df, g_out + i + NF * 0);
        hn::StoreU(f1, df, g_out + i + NF * 1);
        hn::StoreU(f2, df, g_out + i + NF * 2);
        hn::StoreU(f3, df, g_out + i + NF * 3);
      }
    }
  }

  // Accumulates into `sum0..3` dot products of decoded values with `num` bf16
  // from `vec_aligned`. DF is f32 because sum0..3 are also f32. `in_capacity`,
  // `in_ofs` and `num` must all be multiples of `kGroupSize`.
  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void Dot(DF df, const size_t in_capacity,
                             const NuqStream* const in, const size_t in_ofs,
                             const hwy::bfloat16_t* const vec_aligned,
                             const size_t num, hn::Vec<DF>& sum0,
                             hn::Vec<DF>& sum1, hn::Vec<DF>& sum2,
                             hn::Vec<DF>& sum3) {
    const hn::Repartition<hwy::bfloat16_t, DF> dbf;
    const hn::RebindToUnsigned<decltype(dbf)> d16;
    using VBF = hn::Vec<decltype(dbf)>;
    using V16 = hn::Vec<decltype(d16)>;
    const size_t N16 = hn::Lanes(d16);
    HWY_DASSERT(kGroupSize >= 4 * N16);

    HWY_DASSERT(in_ofs + num <= in_capacity);
    HWY_DASSERT(in_capacity % kGroupSize == 0);
    HWY_DASSERT(in_ofs % kGroupSize == 0);
    HWY_DASSERT(num % kGroupSize == 0);
    const size_t ofs_groups = in_ofs / kGroupSize;
    const size_t num_groups = num / kGroupSize;
    const uint8_t* tables = &in->byte + ofs_groups * kClusters;
    const uint8_t* packed_start = &in->byte +
                                  NuqStream::PackedStart(in_capacity) +
                                  ofs_groups * kGroupSize / 2;

    HWY_UNROLL(1)
    for (size_t g = 0; g < num_groups; ++g) {
      const uint8_t* g_centers = tables + g * kClusters;
      const uint8_t* HWY_RESTRICT g_packed = packed_start + g * kGroupSize / 2;
      const hwy::bfloat16_t* HWY_RESTRICT g_in = vec_aligned + g * kGroupSize;

      V16 tbl1 = Zero(d16);
      const V16 tbl0 = LoadTable(d16, g_centers, &tbl1);

      HWY_UNROLL(1)
      for (size_t i = 0; i < kGroupSize; i += 2 * N16) {
        V16 c0, c1;
        TableLookups(d16, tbl0, tbl1, g_packed + i / 2, c0, c1);
        const VBF in0 = hn::Load(dbf, g_in + i + N16 * 0);
        const VBF in1 = hn::Load(dbf, g_in + i + N16 * 1);
        sum0 = hn::ReorderWidenMulAccumulate(df, in0, BitCast(dbf, c0), sum0,
                                             sum1);
        sum2 = hn::ReorderWidenMulAccumulate(df, in1, BitCast(dbf, c1), sum2,
                                             sum3);
      }
    }
  }

  // Accumulates into `sum0..3` dot products of decoded values with `num` f32
  // from `vec_aligned`. `in_capacity`, `in_ofs` and `num` must all be
  // multiples of `kGroupSize`.
  template <class DF, HWY_IF_F32_D(DF)>
  static HWY_INLINE void Dot(DF df, const size_t in_capacity,
                             const NuqStream* const in, const size_t in_ofs,
                             const float* const vec_aligned, const size_t num,
                             hn::Vec<DF>& sum0, hn::Vec<DF>& sum1,
                             hn::Vec<DF>& sum2, hn::Vec<DF>& sum3) {
    const hn::Repartition<hwy::bfloat16_t, DF> dbf;
    const hn::RebindToUnsigned<decltype(dbf)> d16;
    using VF = hn::Vec<decltype(df)>;
    using V16 = hn::Vec<decltype(d16)>;
    const size_t NF = hn::Lanes(df);
    HWY_DASSERT(kGroupSize >= 4 * NF);

    HWY_DASSERT(in_ofs + num <= in_capacity);
    HWY_DASSERT(in_capacity % kGroupSize == 0);
    HWY_DASSERT(in_ofs % kGroupSize == 0);
    HWY_DASSERT(num % kGroupSize == 0);
    const size_t ofs_groups = in_ofs / kGroupSize;
    const size_t num_groups = num / kGroupSize;
    const uint8_t* tables = &in->byte + ofs_groups * kClusters;
    const uint8_t* packed_start = &in->byte +
                                  NuqStream::PackedStart(in_capacity) +
                                  ofs_groups * kGroupSize / 2;

    HWY_UNROLL(1)
    for (size_t g = 0; g < num_groups; ++g) {
      const uint8_t* g_centers = tables + g * kClusters;
      const uint8_t* HWY_RESTRICT g_packed = packed_start + g * kGroupSize / 2;
      const float* HWY_RESTRICT g_in = vec_aligned + g * kGroupSize;

      V16 tbl1 = Zero(d16);
      const V16 tbl0 = LoadTable(d16, g_centers, &tbl1);

      HWY_UNROLL(1)
      for (size_t i = 0; i < kGroupSize; i += 4 * NF) {
        V16 c0, c1;
        TableLookups(d16, tbl0, tbl1, g_packed + i / 2, c0, c1);
        const VF in0 = hn::LoadU(df, g_in + i + NF * 0);
        const VF in1 = hn::LoadU(df, g_in + i + NF * 1);
        const VF in2 = hn::LoadU(df, g_in + i + NF * 2);
        const VF in3 = hn::LoadU(df, g_in + i + NF * 3);
        const VF f0 = hn::PromoteLowerTo(df, BitCast(dbf, c0));
        const VF f1 = hn::PromoteUpperTo(df, BitCast(dbf, c0));
        const VF f2 = hn::PromoteLowerTo(df, BitCast(dbf, c1));
        const VF f3 = hn::PromoteUpperTo(df, BitCast(dbf, c1));
        sum0 = hn::MulAdd(in0, f0, sum0);
        sum1 = hn::MulAdd(in1, f1, sum1);
        sum2 = hn::MulAdd(in2, f2, sum2);
        sum3 = hn::MulAdd(in3, f3, sum3);
      }
    }
  }
};  // NuqCodec

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_NUQ_INL_H_
