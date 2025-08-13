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

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <vector>

#include "compression/types.h"
#include "ops/matmul.h"  // IWYU pragma: export
#include "util/allocator.h"
#include "util/basics.h"
#include "util/mat.h"
#include "util/threading_context.h"
#include "hwy/base.h"
#include "hwy/profiler.h"
#include "hwy/timer.h"

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_MATMUL_TOGGLE) == defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_MATMUL_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_MATMUL_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_MATMUL_TOGGLE
#endif

#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Like hn::PromoteOddTo, but uses assembly to avoid an extra vector register.
template <class DF, class DBF = hn::Repartition<BF16, DF>>
static hn::VFromD<DF> FastPromoteOddTo(DF df, hn::VFromD<DBF> vbf) {
  // Promoting odd means clearing the lower 16 bits. Doing this via AND
  // requires a second input vector, which we prefer to avoid due to high
  // register pressure. Unfortunately `hn::IfThenElseZero` and
  // `IfThenZeroElse` are 'optimized' back to AND, hence resort to assembly.
  // Note that SVE also has separate mask registers, but it anyway uses the
  // native BF16 dot product code path.
#if HWY_TARGET < HWY_AVX2
  const hn::Repartition<uint16_t, decltype(df)> du16;
  const auto odd = static_cast<__mmask32>(0xAAAAAAAAu);  // 10..10 (32 lanes)
  // In-out because this is called after PromoteEvenTo, when we can clobber
  // the original bf16 input.
  auto u16 = hn::BitCast(du16, vbf).raw;
  // Odd u16 lanes are set to the input and even lanes are zero.
  asm("vmovdqu16 %[U16], %[U16]%{%[ODD]%}%{z%};"
      : [U16] "+v"(u16)    // AVX-512 reg
      : [ODD] "Yk"(odd));  // mask reg except k0 (not writable)
  return hn::BitCast(df, hn::VFromD<decltype(du16)>{u16});
#else
  return hn::PromoteOddTo(df, vbf);
#endif
}

// Converts from float intermediate to MatMul output type `TC`.
template <class DC, class DF = hn::Rebind<float, DC>, HWY_IF_F32_D(DC)>
hn::Vec<DC> TCFromF32(DC /*dc*/, hn::Vec<DF> vf) {
  return vf;
}
template <class DC, class DF = hn::Rebind<float, DC>, HWY_IF_BF16_D(DC)>
hn::Vec<DC> TCFromF32(DC dc, hn::Vec<DF> vf) {
  return hn::DemoteTo(dc, vf);
}

// Tag classes, passed to `MMKernel::A2C0` to choose between writing one
// (all-K) result to C via `MMStoreHorizontalSumsIntoC`, or writing the
// first kc result to partial, or accumulating the next kc result into partial
// via `MMAddHorizontalSumsIntoPartial`.
struct MMSetC {};
struct MMSetPartial {};
struct MMAddPartial {};

// Stores horizontal sums of up to 16 vectors via transpose.
template <size_t kRowsAC, bool kAdd>
class MMStoreHorizontalSumsIntoC {
 public:
  static_assert(kNR == 4);  // for `StoreInterleaved4`

  // Computes horizontal sums of `kRowsAC x kNR` vectors and stores into
  // `C` starting at `(row_c, col_c)`.
  //
  // `Crc` are the 16 combinations of an A row vector indexed by `r`, times a
  // transposed B row vector indexed by `c`. Their elements are thus a subset
  // of the terms of the dot product constituting the final `C[r, c]` result.
  // Thus we compute the horizontal sums of each `Crc`. The elements may be
  // permuted because we multiply bf16 via `ReorderWidenMulAccumulate`, but
  // this does not change their horizontal sum.
  template <class DF, class VF = hn::Vec<DF>, typename TC>
  HWY_INLINE void operator()(DF df,                           //
                             VF C00, VF C01, VF C02, VF C03,  //
                             VF C10, VF C11, VF C12, VF C13,  //
                             VF C20, VF C21, VF C22, VF C23,  //
                             VF C30, VF C31, VF C32, VF C33,  //
                             const size_t row_c, const size_t col_c,
                             const MMArgs& args, RowPtrs<TC> C_rows) const {
    HWY_ALIGN float buf[16 * hn::MaxLanes(df)];
    const size_t N = hn::Lanes(df);
    // Horizontal reductions (`ReduceSum`) are rather expensive, entailing
    // log(N) operations for vectors of length N. Because `kNR` == 4, we
    // instead use `StoreInterleaved4` for a vector length-agnostic
    // 'transpose': `buf[0, 4 * N)` holds `C00[0], C01[0], C02[0], C03[0],
    // C00[1], C01[1], C02[1], C03[1] .. C00[N-1], C01[N-1], C02[N-1],
    // C03[N-1]`.
    MaybeStoreInterleaved4<0>(df, N, C00, C01, C02, C03, buf);
    MaybeStoreInterleaved4<1>(df, N, C10, C11, C12, C13, buf);
    MaybeStoreInterleaved4<2>(df, N, C20, C21, C22, C23, buf);
    MaybeStoreInterleaved4<3>(df, N, C30, C31, C32, C33, buf);
    // Adding N consecutive V4 yields horizontal sums of Cr0, Cr1, Cr2, Cr3 in
    // the elements of one V4. We have four independent rows `r`, hence the
    // code is effectively unrolled, which increases throughput.
    const hn::CappedTag<float, kNR> d4;
    using V4 = hn::Vec<decltype(d4)>;
    // Store to four elements per row of `partial`.
    // No loop is required because vectors are at least 4*32 bits.
    V4 sum0 = MaybeLoad<0>(d4, N, buf);
    V4 sum1 = MaybeLoad<1>(d4, N, buf);
    V4 sum2 = MaybeLoad<2>(d4, N, buf);
    V4 sum3 = MaybeLoad<3>(d4, N, buf);

    for (size_t lane = 1; lane < N; ++lane) {
      sum0 = MaybeAdd<0>(d4, N, sum0, buf + kNR * lane);
      sum1 = MaybeAdd<1>(d4, N, sum1, buf + kNR * lane);
      sum2 = MaybeAdd<2>(d4, N, sum2, buf + kNR * lane);
      sum3 = MaybeAdd<3>(d4, N, sum3, buf + kNR * lane);
    }
    const V4 vscale = hn::Set(d4, args.scale);
    V4 vadd = hn::Zero(d4);
    if constexpr (kAdd) {
      vadd = hn::Load(d4, args.add + col_c);
    }
    MaybeScaleAndStore<0>(d4, sum0, vscale, vadd, C_rows, row_c, col_c);
    MaybeScaleAndStore<1>(d4, sum1, vscale, vadd, C_rows, row_c, col_c);
    MaybeScaleAndStore<2>(d4, sum2, vscale, vadd, C_rows, row_c, col_c);
    MaybeScaleAndStore<3>(d4, sum3, vscale, vadd, C_rows, row_c, col_c);
  }

 private:
  // These helper functions hoist if() out of the main code below. They have
  // no effect if kRow >= kRowsAC.
  template <size_t kRow, class DD, class VD = hn::Vec<DD>>
  static HWY_INLINE void MaybeStoreInterleaved4(DD dd, size_t N, VD Cr0, VD Cr1,
                                                VD Cr2, VD Cr3,
                                                float* HWY_RESTRICT buf) {
    if constexpr (kRow < kRowsAC) {
      hn::StoreInterleaved4(Cr0, Cr1, Cr2, Cr3, dd, buf + 4 * kRow * N);
    }
  }

  // Note: N is the number of lanes in the StoreInterleaved4 vectors, not V4.
  template <size_t kRow, class DF4, class VF4 = hn::Vec<DF4>>
  static HWY_INLINE VF4 MaybeLoad(DF4 df4, size_t N,
                                  const float* HWY_RESTRICT buf) {
    if constexpr (kRow < kRowsAC) {
      return hn::Load(df4, buf + 4 * kRow * N);
    } else {
      return hn::Zero(df4);
    }
  }

  template <size_t kRow, class DF4, class VF4 = hn::Vec<DF4>>
  static HWY_INLINE VF4 MaybeAdd(DF4 df4, size_t N, VF4 sum,
                                 const float* HWY_RESTRICT buf) {
    if constexpr (kRow < kRowsAC) {
      return hn::Add(sum, hn::Load(df4, buf + 4 * kRow * N));
    } else {
      return sum;
    }
  }

  template <size_t kRow, /*deduced:*/ class DF4, class VF4 = hn::Vec<DF4>,
            typename TC>
  static HWY_INLINE void MaybeScaleAndStore(DF4 df4, VF4 sum, VF4 vscale,
                                            VF4 vadd, RowPtrs<TC> C_rows,
                                            const size_t row_c,
                                            const size_t col_c) {
    if constexpr (kRow < kRowsAC) {
      TC* HWY_RESTRICT pos = C_rows[row_c + kRow] + col_c;
      const hn::Rebind<TC, DF4> dc4;
      const VF4 out = hn::MulAdd(sum, vscale, vadd);
      hn::Store(TCFromF32(dc4, out), dc4, pos);
    }
  }
};  // MMStoreHorizontalSumsIntoC

// Accumulates horizontal sums of up to 16 vectors via transpose.
template <size_t kRowsAC, class Tag>
class MMAddHorizontalSumsIntoPartial {
 public:
  static_assert(kNR == 4);  // for `StoreInterleaved4`

  // Computes horizontal sums of `kRowsAC x kNR` vectors and accumulates
  // into `partial` starting at `(row_c, col_c)`.
  //
  // `Crc` are the 16 combinations of an A row vector indexed by `r`, times a
  // transposed B row vector indexed by `c`. Their elements are thus a subset
  // of the terms of the dot product constituting the final `C[r, c]` result.
  // Thus we compute the horizontal sums of each `Crc`. The elements may be
  // permuted because we multiply bf16 via `ReorderWidenMulAccumulate`, but
  // this does not change their horizontal sum.
  template <class DF, class VF = hn::Vec<DF>>
  HWY_INLINE void operator()(DF df,                           //
                             VF F00, VF F01, VF F02, VF F03,  //
                             VF F10, VF F11, VF F12, VF F13,  //
                             VF F20, VF F21, VF F22, VF F23,  //
                             VF F30, VF F31, VF F32, VF F33,  //
                             const size_t row_c, const size_t col_c,
                             const StridedViewD& partial) const {
    // We accumulate in 64-bit to avoid loss of precision.
    static_assert(HWY_HAVE_FLOAT64, "Disable Armv7 NEON: we require fp64");

    const hn::Repartition<double, DF> dd;
    HWY_ALIGN double buf[16 * hn::MaxLanes(dd)];
    using VD = hn::Vec<decltype(dd)>;
    const size_t ND = hn::Lanes(dd);
    VD C00 = SumOfPromotedPairs(dd, F00);
    VD C01 = SumOfPromotedPairs(dd, F01);
    VD C02 = SumOfPromotedPairs(dd, F02);
    VD C03 = SumOfPromotedPairs(dd, F03);
    VD C10 = SumOfPromotedPairs(dd, F10);
    VD C11 = SumOfPromotedPairs(dd, F11);
    VD C12 = SumOfPromotedPairs(dd, F12);
    VD C13 = SumOfPromotedPairs(dd, F13);
    VD C20 = SumOfPromotedPairs(dd, F20);
    VD C21 = SumOfPromotedPairs(dd, F21);
    VD C22 = SumOfPromotedPairs(dd, F22);
    VD C23 = SumOfPromotedPairs(dd, F23);
    VD C30 = SumOfPromotedPairs(dd, F30);
    VD C31 = SumOfPromotedPairs(dd, F31);
    VD C32 = SumOfPromotedPairs(dd, F32);
    VD C33 = SumOfPromotedPairs(dd, F33);

    // Horizontal reductions (`ReduceSum`) are rather expensive, entailing
    // log(N) operations for vectors of length N. Because `kNR` == 4, we
    // instead use `StoreInterleaved4` for a vector length-agnostic
    // 'transpose': `buf[0, 4 * N)` holds `C00[0], C01[0], C02[0], C03[0],
    // C00[1], C01[1], C02[1], C03[1] .. C00[N-1], C01[N-1], C02[N-1],
    // C03[N-1]`.
    MaybeStoreInterleaved4<0>(dd, ND, C00, C01, C02, C03, buf);
    MaybeStoreInterleaved4<1>(dd, ND, C10, C11, C12, C13, buf);
    MaybeStoreInterleaved4<2>(dd, ND, C20, C21, C22, C23, buf);
    MaybeStoreInterleaved4<3>(dd, ND, C30, C31, C32, C33, buf);
    // Adding N consecutive V4 yields horizontal sums of Cr0, Cr1, Cr2, Cr3 in
    // the elements of one V4. We have four independent rows `r`, hence the
    // code is effectively unrolled, which increases throughput.
    const hn::CappedTag<double, kNR> d4;
    using V4 = hn::Vec<decltype(d4)>;
    // Store to four elements per row of `partial`.
    // Loop is required because vectors may be smaller than 4*64 bits.
    for (size_t c = 0; c < kNR; c += hn::Lanes(d4)) {
      V4 sum0 = MaybeLoad<0>(d4, ND, buf + c);
      V4 sum1 = MaybeLoad<1>(d4, ND, buf + c);
      V4 sum2 = MaybeLoad<2>(d4, ND, buf + c);
      V4 sum3 = MaybeLoad<3>(d4, ND, buf + c);

      for (size_t lane = 1; lane < ND; ++lane) {
        sum0 = MaybeAdd<0>(d4, ND, sum0, buf + c + kNR * lane);
        sum1 = MaybeAdd<1>(d4, ND, sum1, buf + c + kNR * lane);
        sum2 = MaybeAdd<2>(d4, ND, sum2, buf + c + kNR * lane);
        sum3 = MaybeAdd<3>(d4, ND, sum3, buf + c + kNR * lane);
      }
      MaybeAddStore<0>(d4, sum0, partial, row_c, col_c + c);
      MaybeAddStore<1>(d4, sum1, partial, row_c, col_c + c);
      MaybeAddStore<2>(d4, sum2, partial, row_c, col_c + c);
      MaybeAddStore<3>(d4, sum3, partial, row_c, col_c + c);
    }
  }

 private:
  // Converts lanes to double and adds pairs of them to obtain a vector with the
  // same horizontal sum, but element type double.
  template <class DD, class VD = hn::Vec<DD>,
            class DF = hn::Repartition<float, DD>, class VF = hn::Vec<DF>>
  static HWY_INLINE VD SumOfPromotedPairs(DD dd, VF f) {
    // TODO: SVE could PromoteEvenTo.
    const VD d0 = hn::PromoteLowerTo(dd, f);
    const VD d1 = hn::PromoteUpperTo(dd, f);
    return hn::Add(d0, d1);
  }

  // These helper functions hoist if() out of the main code below. They have
  // no effect if kRow >= kRowsAC.
  template <size_t kRow, class DD, class VD = hn::Vec<DD>>
  static HWY_INLINE void MaybeStoreInterleaved4(DD dd, size_t N, VD Cr0, VD Cr1,
                                                VD Cr2, VD Cr3,
                                                double* HWY_RESTRICT buf) {
    if constexpr (kRow < kRowsAC) {
      hn::StoreInterleaved4(Cr0, Cr1, Cr2, Cr3, dd, buf + 4 * kRow * N);
    }
  }

  // Note: N is the number of lanes in the StoreInterleaved4 vectors, not V4.
  template <size_t kRow, class D4, class V4 = hn::Vec<D4>>
  static HWY_INLINE V4 MaybeLoad(D4 d4, size_t N,
                                 const double* HWY_RESTRICT buf) {
    if constexpr (kRow < kRowsAC) {
      return hn::Load(d4, buf + 4 * kRow * N);
    } else {
      return hn::Zero(d4);
    }
  }

  template <size_t kRow, class D4, class V4 = hn::Vec<D4>>
  static HWY_INLINE V4 MaybeAdd(D4 d4, size_t N, V4 sum,
                                const double* HWY_RESTRICT buf) {
    if constexpr (kRow < kRowsAC) {
      return hn::Add(sum, hn::Load(d4, buf + 4 * kRow * N));
    } else {
      return sum;
    }
  }

  template <size_t kRow, class D4, class V4 = hn::Vec<D4>>
  static HWY_INLINE void MaybeAddStore(D4 d4, V4 sum,
                                       const StridedViewD& partial,
                                       const size_t row_c, const size_t col_c) {
    if constexpr (kRow < kRowsAC) {
      double* HWY_RESTRICT pos = partial.Row(row_c + kRow) + col_c;
      if constexpr (hwy::IsSame<Tag, MMSetPartial>()) {
        hn::Store(sum, d4, pos);
      } else {
        static_assert(hwy::IsSame<Tag, MMAddPartial>());
        const V4 prev = hn::Load(d4, pos);
        hn::Store(hn::Add(sum, prev), d4, pos);
      }
    }
  }
};  // MMAddHorizontalSumsIntoPartial

// Stateless, wraps member functions.
class MMKernel {
 public:
  // Calls `LoopKC` for each of `mc` rows of A in steps of `mr`. `A_view`
  // is `mc x kc` and `B_view` is `(kNR x kc)`. Both start at row/col 0.
  // A2C0 in MOMMS terminology updates a `mc x kNR` slice of the output.
  template <class Tag, typename TC>
  static HWY_INLINE void A2C0(const StridedViewBF& A_view,
                              const StridedViewBF& B_view, size_t mr,
                              const IndexRange& range_mc, const size_t row_b,
                              size_t kc, Tag tag, const MMArgs& args,
                              RowPtrs<TC> C_rows) {
    HWY_DASSERT(1 <= mr && mr <= kMaxMR);
    const size_t row0 = range_mc.begin();
    const size_t mc = range_mc.Num();
    size_t imc = 0;

    // M == 1, or x86 with 8 SIMD registers:
    if (HWY_UNLIKELY(mr == 1)) {
      for (; imc < mc; ++imc) {
        LoopKC<1>(A_view, B_view, row0 + imc, imc, row_b, kc, tag, args,
                  C_rows);
      }
      return;
    }

    // AVX2 (16 registers)
    if (HWY_UNLIKELY(mr == 2)) {
      if (HWY_LIKELY(mc >= 2)) {
        for (; imc <= mc - 2; imc += 2) {
          LoopKC<2>(A_view, B_view, row0 + imc, imc, row_b, kc, tag, args,
                    C_rows);
        }
      }
      if (HWY_UNLIKELY(imc != mc)) {
        LoopKC<1>(A_view, B_view, row0 + imc, imc, row_b, kc, tag, args,
                  C_rows);
      }
      return;
    }

    HWY_DASSERT(mr == 4);
    if (HWY_LIKELY(mc >= 4)) {
      for (; imc <= mc - 4; imc += 4) {
        LoopKC<4>(A_view, B_view, row0 + imc, imc, row_b, kc, tag, args,
                  C_rows);
      }
    }
    const size_t remainder_mc = mc - imc;
    HWY_DASSERT(remainder_mc < 4);
    if (HWY_UNLIKELY(remainder_mc & 2)) {
      LoopKC<2>(A_view, B_view, row0 + imc, imc, row_b, kc, tag, args, C_rows);
      imc += 2;
    }
    if (HWY_UNLIKELY(remainder_mc & 1)) {
      LoopKC<1>(A_view, B_view, row0 + imc, imc, row_b, kc, tag, args, C_rows);
      imc += 1;
    }
    HWY_DASSERT(imc == mc);
  }

 private:
  // Element-wise multiplies a vector from one row of A with `kNR` vectors,
  // each from a row of transposed B, and adds them to `kNR` fp32 `Cc`
  // vectors. The lanes of `Cc` are thus a subset of the terms of the dot
  // product which is the MatMul result at column `c`.
  //
  // Why elementwise, when most MatMul instead broadcast one element from A and
  // multiply with one element from kr columns in B to obtain kr columns of C?
  // We double the compute throughput on NEON_BF16/SVE/AVX3_ZEN4 by using the
  // bf16 * bf16 + f32 `ReorderWidenMulAccumulate`. However, this involves
  // pairwise adds, whereas the kr-column approach requires that lanes remain
  // separate. Our elementwise approach is fine with pairwise adds because they
  // do not change the horizontal sum. However, horizontal sums can be costly,
  // so we introduce a fast and new(?) vector-length agnostic 'transpose', see
  // `MMAddHorizontalSumsIntoPartial`.
  template <class DBF, class VBF = hn::Vec<DBF>,
            class DF = hn::Repartition<float, DBF>, class VF = hn::Vec<DF>>
  static HWY_INLINE void ElementwiseMulAcc(DBF dbf, VBF a, VBF b0, VBF b1,
                                           VBF b2, VBF b3, VF& C0, VF& C1,
                                           VF& C2, VF& C3) {
    // This handles a single row of A, so the horizontal sums of `C0..3` are the
    // (partial) dot products for 4 consecutive values in one row of C.
    static_assert(kNR == 4);

    HWY_DASSERT(HWY_NATIVE_DOT_BF16);

    const DF df;
    VF unused_sum1 = hn::Zero(df);
    // When implemented natively, this op includes 'free' f32 accumulation.
    C0 = hn::ReorderWidenMulAccumulate(df, a, b0, C0, unused_sum1);
    C1 = hn::ReorderWidenMulAccumulate(df, a, b1, C1, unused_sum1);
    C2 = hn::ReorderWidenMulAccumulate(df, a, b2, C2, unused_sum1);
    C3 = hn::ReorderWidenMulAccumulate(df, a, b3, C3, unused_sum1);
    // Ensure unused_sum1 was indeed unused.
    HWY_DASSERT(hn::AllTrue(df, hn::Eq(unused_sum1, hn::Zero(df))));
  }

  // Like `ElementwiseMulAcc`, but splits BF16 inputs into odd and even f32
  // for use with FMA. Also handles two rows at a time to hide the FMA latency
  // (we assume 4 cycles and dual-issue) before writing `C00` again.
  template <class DBF, class VBF = hn::Vec<DBF>,
            class DF = hn::Repartition<float, DBF>, class VF = hn::Vec<DF>>
  static HWY_INLINE void ElementwiseMulAcc2(DBF dbf, VBF a0, VBF a1, VF b0o,
                                            VF b0e, VF b1o, VF b1e, VF b2o,
                                            VF b2e, VF b3o, VF b3e, VF& C00,
                                            VF& C01, VF& C02, VF& C03, VF& C10,
                                            VF& C11, VF& C12, VF& C13) {
    const DF df;
    HWY_DASSERT(!HWY_NATIVE_DOT_BF16);
    // Avoid `ReorderWidenMulAccumulate` because it requires extra adds for
    // the two outputs, and `WidenMulPairwiseAdd` because it wastes an
    // opportunity for a free f32 add via FMA, and `MulOddAdd` because we want
    // to avoid an extra register for a constant. Use scoping to reduce register
    // pressure and avoid spills on 32-register targets. Register usage:
    // 4 for a0, a1, a0e, a1e; 8 for `b*`, 16 for `C*` = 28.
    {
      const VF a0e = hn::PromoteEvenTo(df, a0);
      C00 = hn::MulAdd(a0e, b0e, C00);
      C01 = hn::MulAdd(a0e, b1e, C01);
      C02 = hn::MulAdd(a0e, b2e, C02);
      C03 = hn::MulAdd(a0e, b3e, C03);
    }
    {
      const VF a1e = hn::PromoteEvenTo(df, a1);
      C10 = hn::MulAdd(a1e, b0e, C10);
      C11 = hn::MulAdd(a1e, b1e, C11);
      C12 = hn::MulAdd(a1e, b2e, C12);
      C13 = hn::MulAdd(a1e, b3e, C13);
    }
    {
      const VF a0o = FastPromoteOddTo(df, a0);
      C00 = hn::MulAdd(a0o, b0o, C00);
      C01 = hn::MulAdd(a0o, b1o, C01);
      C02 = hn::MulAdd(a0o, b2o, C02);
      C03 = hn::MulAdd(a0o, b3o, C03);
    }
    {
      const VF a1o = FastPromoteOddTo(df, a1);
      C10 = hn::MulAdd(a1o, b0o, C10);
      C11 = hn::MulAdd(a1o, b1o, C11);
      C12 = hn::MulAdd(a1o, b2o, C12);
      C13 = hn::MulAdd(a1o, b3o, C13);
    }
  }

  // Innermost loop over `kc` columns (typically 1024-4096) in steps of one
  // vector, for `kRowsAC` rows of `A_view` from range_mc-relative `imc` and
  // `B_view` from row 0 (both at column 0). Updates a `kRowsAC x kNR` tile
  // with top-left corner `partial.Row(row_ac) + col_c`. Both A and B must be
  // BF16 so we can load directly without `Decompress2`, which is expensive for
  // NUQ and requires 2x unrolling, which requires more loads.
  template <size_t kRowsAC, /*deduced:*/ class Tag, typename TC>
  static HWY_INLINE void LoopKC(const StridedViewBF& A_view,
                                const StridedViewBF& B_view, size_t row_ac,
                                size_t imc, size_t col_c, size_t kc, Tag tag,
                                const MMArgs& args, RowPtrs<TC> C_rows) {
    const hn::ScalableTag<BF16> dbf;
    using VBF = hn::Vec<decltype(dbf)>;
    const size_t NBF = hn::Lanes(dbf);

    HWY_DASSERT(kRowsAC <= kMaxMR);
    HWY_DASSERT(col_c % kNR == 0);
    // Rows are aligned to `kMaxMR`, except for the last tile of A.

    // `kRowsAC` rows of A (null for the rest) and `kNR` rows of B.
    static_assert(kNR == 4);
    const BF16* HWY_RESTRICT ar0 = A_view.Row(imc + 0);
    const BF16* HWY_RESTRICT ar1 = kRowsAC > 1 ? A_view.Row(imc + 1) : nullptr;
    const BF16* HWY_RESTRICT ar2 = kRowsAC > 2 ? A_view.Row(imc + 2) : nullptr;
    const BF16* HWY_RESTRICT ar3 = kRowsAC > 3 ? A_view.Row(imc + 3) : nullptr;
    const BF16* HWY_RESTRICT br0 = B_view.Row(0);
    const BF16* HWY_RESTRICT br1 = B_view.Row(1);
    const BF16* HWY_RESTRICT br2 = B_view.Row(2);
    const BF16* HWY_RESTRICT br3 = B_view.Row(3);

    // Ensure `A` and `B` were zero-padded by `DecompressAndZeroPad`.
    if constexpr (HWY_IS_DEBUG_BUILD) {
      for (size_t i = kc; i < hwy::RoundUpTo(kc, NBF); ++i) {
        {
          HWY_DASSERT(hwy::ConvertScalarTo<float>(ar0[i]) == 0.0f);
        }
        if constexpr (kRowsAC > 1) {
          HWY_DASSERT(hwy::ConvertScalarTo<float>(ar1[i]) == 0.0f);
        }
        if constexpr (kRowsAC > 2) {
          HWY_DASSERT(hwy::ConvertScalarTo<float>(ar2[i]) == 0.0f);
        }
        if constexpr (kRowsAC > 3) {
          HWY_DASSERT(hwy::ConvertScalarTo<float>(ar3[i]) == 0.0f);
        }
        HWY_DASSERT(hwy::ConvertScalarTo<float>(br0[i]) == 0.0f);
        HWY_DASSERT(hwy::ConvertScalarTo<float>(br1[i]) == 0.0f);
        HWY_DASSERT(hwy::ConvertScalarTo<float>(br2[i]) == 0.0f);
        HWY_DASSERT(hwy::ConvertScalarTo<float>(br3[i]) == 0.0f);
      }
    }

    // Accumulate into f32.
    const hn::Repartition<float, decltype(dbf)> df;
    using VF = hn::Vec<decltype(df)>;
    VF C00 = hn::Zero(df), C01 = hn::Zero(df), C02 = hn::Zero(df),
       C03 = hn::Zero(df), C10 = hn::Zero(df), C11 = hn::Zero(df),
       C12 = hn::Zero(df), C13 = hn::Zero(df), C20 = hn::Zero(df),
       C21 = hn::Zero(df), C22 = hn::Zero(df), C23 = hn::Zero(df),
       C30 = hn::Zero(df), C31 = hn::Zero(df), C32 = hn::Zero(df),
       C33 = hn::Zero(df);

    HWY_UNROLL(1)
    for (size_t ikc = 0; ikc < kc; ikc += NBF) {
      if constexpr (HWY_NATIVE_DOT_BF16) {
        const VBF b0 = hn::Load(dbf, br0 + ikc);
        const VBF b1 = hn::Load(dbf, br1 + ikc);
        const VBF b2 = hn::Load(dbf, br2 + ikc);
        const VBF b3 = hn::Load(dbf, br3 + ikc);
        {
          const VBF a0 = hn::Load(dbf, ar0 + ikc);
          ElementwiseMulAcc(dbf, a0, b0, b1, b2, b3, C00, C01, C02, C03);
        }
        if constexpr (kRowsAC > 1) {
          const VBF a1 = hn::Load(dbf, ar1 + ikc);
          ElementwiseMulAcc(dbf, a1, b0, b1, b2, b3, C10, C11, C12, C13);
        }
        if constexpr (kRowsAC > 2) {
          const VBF a2 = hn::Load(dbf, ar2 + ikc);
          ElementwiseMulAcc(dbf, a2, b0, b1, b2, b3, C20, C21, C22, C23);
        }
        if constexpr (kRowsAC > 3) {
          const VBF a3 = hn::Load(dbf, ar3 + ikc);
          ElementwiseMulAcc(dbf, a3, b0, b1, b2, b3, C30, C31, C32, C33);
        }
      } else {
        VF b0e, b1e, b2e, b3e, b0o, b1o, b2o, b3o;
        {
          const VBF b0 = hn::Load(dbf, br0 + ikc);
          const VBF b1 = hn::Load(dbf, br1 + ikc);
          const VBF b2 = hn::Load(dbf, br2 + ikc);
          const VBF b3 = hn::Load(dbf, br3 + ikc);
          b0e = hn::PromoteEvenTo(df, b0);
          b1e = hn::PromoteEvenTo(df, b1);
          b2e = hn::PromoteEvenTo(df, b2);
          b3e = hn::PromoteEvenTo(df, b3);
          b0o = FastPromoteOddTo(df, b0);
          b1o = FastPromoteOddTo(df, b1);
          b2o = FastPromoteOddTo(df, b2);
          b3o = FastPromoteOddTo(df, b3);
        }

        {
          const VBF a0 = hn::Load(dbf, ar0 + ikc);
          const VBF a1 = kRowsAC > 1 ? hn::Load(dbf, ar1 + ikc) : a0;
          ElementwiseMulAcc2(dbf, a0, a1, b0o, b0e, b1o, b1e, b2o, b2e, b3o,
                             b3e, C00, C01, C02, C03, C10, C11, C12, C13);
        }
        if constexpr (kRowsAC > 2) {
          const VBF a2 = hn::Load(dbf, ar2 + ikc);
          const VBF a3 = kRowsAC > 3 ? hn::Load(dbf, ar3 + ikc) : a2;
          ElementwiseMulAcc2(dbf, a2, a3, b0o, b0e, b1o, b1e, b2o, b2e, b3o,
                             b3e, C20, C21, C22, C23, C30, C31, C32, C33);
        }
      }
    }

    // This is a substantial fraction (about 1/3) of the total time, but is
    // called frequently, so do not add a profiler zone.

    if constexpr (hwy::IsSame<Tag, MMSetC>()) {
      if (args.add) {
        MMStoreHorizontalSumsIntoC<kRowsAC, /*kAdd=*/true>()(
            df, C00, C01, C02, C03, C10, C11, C12, C13, C20, C21, C22, C23, C30,
            C31, C32, C33, row_ac, col_c, args, C_rows);
      } else {
        MMStoreHorizontalSumsIntoC<kRowsAC, /*kAdd=*/false>()(
            df, C00, C01, C02, C03, C10, C11, C12, C13, C20, C21, C22, C23, C30,
            C31, C32, C33, row_ac, col_c, args, C_rows);
      }
    } else {
      MMAddHorizontalSumsIntoPartial<kRowsAC, Tag>()(
          df, C00, C01, C02, C03, C10, C11, C12, C13, C20, C21, C22, C23, C30,
          C31, C32, C33, row_ac, col_c, args.partial);
    }
  }
};

// Multiply partial by scale, add bias if present, demote and store to f32 `C`.
// Stateless, wraps member functions.
class MMScaleDemoteAdd {
 public:
  // Fills the `range_mc/range_nc` region of `outputs.C` by multiplying the
  // same region of `outputs.partial` by `outputs.scale`, which is the product
  // of the scales of A and B, demoting from f64 to f32, then if `outputs.add`
  // is nonzero, adding it to each row.
  // TODO: fuse with subsequent operations - function pointer?
  // Although this region in `outputs.C` is not touched again, streaming stores
  // do not help on SKX and Zen4. TODO: re-check this.
  template <typename TC>
  static HWY_INLINE void FillC(const IndexRange& range_mc,
                               const IndexRange& range_nc, const MMArgs& args,
                               RowPtrs<TC> C_rows) {
    size_t row_c = range_mc.begin();
    if (args.add) {
      constexpr bool kAdd = true;
      if (range_mc.Num() >= 4) {
        for (; row_c <= range_mc.end() - 4; row_c += 4) {
          Do4Rows<kAdd>(row_c, range_nc, args, C_rows);
        }
      }
      for (; row_c < range_mc.end(); ++row_c) {
        Do1Row<kAdd>(row_c, range_nc, args, C_rows);
      }
    } else {
      constexpr bool kAdd = false;
      if (range_mc.Num() >= 4) {
        for (; row_c <= range_mc.end() - 4; row_c += 4) {
          Do4Rows<kAdd>(row_c, range_nc, args, C_rows);
        }
      }
      for (; row_c < range_mc.end(); ++row_c) {
        Do1Row<kAdd>(row_c, range_nc, args, C_rows);
      }
    }
  }

 private:
  // Unrolled for 4 rows to reduce the number of loads from `add`.
  template <bool kAdd, typename TC>
  static HWY_INLINE void Do4Rows(size_t row_c, const IndexRange& range_nc,
                                 const MMArgs& args, RowPtrs<TC> C_rows) {
    const hn::ScalableTag<double> dd;
    const hn::Rebind<float, decltype(dd)> df;  // result of DemoteTo
    const hn::Rebind<TC, decltype(dd)> dc;
    using VD = hn::Vec<decltype(dd)>;
    using VF = hn::Vec<decltype(df)>;
    const size_t ND = hn::Lanes(dd);
    const VD vscale = hn::Set(dd, args.scale);

    const double* HWY_RESTRICT pr0 = args.partial.Row(row_c + 0);
    const double* HWY_RESTRICT pr1 = args.partial.Row(row_c + 1);
    const double* HWY_RESTRICT pr2 = args.partial.Row(row_c + 2);
    const double* HWY_RESTRICT pr3 = args.partial.Row(row_c + 3);

    TC* HWY_RESTRICT cr0 = C_rows[row_c + 0];
    TC* HWY_RESTRICT cr1 = C_rows[row_c + 1];
    TC* HWY_RESTRICT cr2 = C_rows[row_c + 2];
    TC* HWY_RESTRICT cr3 = C_rows[row_c + 3];

    // We manually unroll 2x for higher IPC in batch=1.
    size_t col_c = range_nc.begin();
    if (HWY_LIKELY(range_nc.Num() >= 2 * ND)) {
      for (; col_c <= range_nc.end() - 2 * ND; col_c += 2 * ND) {
        VD a0, a1;  // unused if !kAdd
        if constexpr (kAdd) {
          // Promoting to double lets us fuse the Add into MulAdd.
          a0 = hn::PromoteTo(dd, hn::Load(df, args.add + col_c));
          a1 = hn::PromoteTo(dd, hn::Load(df, args.add + col_c + ND));
        }
        const VD d00 = hn::Load(dd, pr0 + col_c);
        const VD d01 = hn::Load(dd, pr0 + col_c + ND);
        const VD d10 = hn::Load(dd, pr1 + col_c);
        const VD d11 = hn::Load(dd, pr1 + col_c + ND);
        const VD d20 = hn::Load(dd, pr2 + col_c);
        const VD d21 = hn::Load(dd, pr2 + col_c + ND);
        const VD d30 = hn::Load(dd, pr3 + col_c);
        const VD d31 = hn::Load(dd, pr3 + col_c + ND);
        VD m00, m01, m10, m11, m20, m21, m30, m31;
        if constexpr (kAdd) {
          m00 = hn::MulAdd(d00, vscale, a0);
          m01 = hn::MulAdd(d01, vscale, a1);
          m10 = hn::MulAdd(d10, vscale, a0);
          m11 = hn::MulAdd(d11, vscale, a1);
          m20 = hn::MulAdd(d20, vscale, a0);
          m21 = hn::MulAdd(d21, vscale, a1);
          m30 = hn::MulAdd(d30, vscale, a0);
          m31 = hn::MulAdd(d31, vscale, a1);
        } else {
          m00 = hn::Mul(d00, vscale);
          m01 = hn::Mul(d01, vscale);
          m10 = hn::Mul(d10, vscale);
          m11 = hn::Mul(d11, vscale);
          m20 = hn::Mul(d20, vscale);
          m21 = hn::Mul(d21, vscale);
          m30 = hn::Mul(d30, vscale);
          m31 = hn::Mul(d31, vscale);
        }
        // First convert f64 to f32.
        const VF f00 = hn::DemoteTo(df, m00);
        const VF f01 = hn::DemoteTo(df, m01);
        const VF f10 = hn::DemoteTo(df, m10);
        const VF f11 = hn::DemoteTo(df, m11);
        const VF f20 = hn::DemoteTo(df, m20);
        const VF f21 = hn::DemoteTo(df, m21);
        const VF f30 = hn::DemoteTo(df, m30);
        const VF f31 = hn::DemoteTo(df, m31);
        // Note that Stream is neutral on SKX and harmful on Zen4.
        hn::Store(TCFromF32(dc, f00), dc, cr0 + col_c);
        hn::Store(TCFromF32(dc, f01), dc, cr0 + col_c + ND);
        hn::Store(TCFromF32(dc, f10), dc, cr1 + col_c);
        hn::Store(TCFromF32(dc, f11), dc, cr1 + col_c + ND);
        hn::Store(TCFromF32(dc, f20), dc, cr2 + col_c);
        hn::Store(TCFromF32(dc, f21), dc, cr2 + col_c + ND);
        hn::Store(TCFromF32(dc, f30), dc, cr3 + col_c);
        hn::Store(TCFromF32(dc, f31), dc, cr3 + col_c + ND);
      }
    }

    for (; col_c < range_nc.end(); col_c += ND) {
      const size_t remaining = range_nc.end() - col_c;
      HWY_DASSERT(remaining < 2 * ND);

      VD a0;  // unused if !kAdd
      if constexpr (kAdd) {
        // Promoting to double lets us fuse the Add into MulAdd.
        a0 = hn::PromoteTo(dd, hn::LoadN(df, args.add + col_c, remaining));
      }
      const VD d00 = hn::LoadN(dd, pr0 + col_c, remaining);
      const VD d10 = hn::LoadN(dd, pr1 + col_c, remaining);
      const VD d20 = hn::LoadN(dd, pr2 + col_c, remaining);
      const VD d30 = hn::LoadN(dd, pr3 + col_c, remaining);
      VD m00, m10, m20, m30;
      if constexpr (kAdd) {
        m00 = hn::MulAdd(d00, vscale, a0);
        m10 = hn::MulAdd(d10, vscale, a0);
        m20 = hn::MulAdd(d20, vscale, a0);
        m30 = hn::MulAdd(d30, vscale, a0);
      } else {
        m00 = hn::Mul(d00, vscale);
        m10 = hn::Mul(d10, vscale);
        m20 = hn::Mul(d20, vscale);
        m30 = hn::Mul(d30, vscale);
      }
      // First convert f64 to f32.
      const VF f00 = hn::DemoteTo(df, m00);
      const VF f10 = hn::DemoteTo(df, m10);
      const VF f20 = hn::DemoteTo(df, m20);
      const VF f30 = hn::DemoteTo(df, m30);
      hn::StoreN(TCFromF32(dc, f00), dc, cr0 + col_c, remaining);
      hn::StoreN(TCFromF32(dc, f10), dc, cr1 + col_c, remaining);
      hn::StoreN(TCFromF32(dc, f20), dc, cr2 + col_c, remaining);
      hn::StoreN(TCFromF32(dc, f30), dc, cr3 + col_c, remaining);
    }
  }

  // Same as above but handles a single row (for remainder rows).
  template <bool kAdd, typename TC>
  static HWY_INLINE void Do1Row(size_t row_c, const IndexRange& range_nc,
                                const MMArgs& args, RowPtrs<TC> C_rows) {
    const hn::ScalableTag<double> dd;
    const hn::Rebind<float, decltype(dd)> df;  // result of DemoteTo
    const hn::Rebind<TC, decltype(dd)> dc;
    using VD = hn::Vec<decltype(dd)>;
    using VF = hn::Vec<decltype(df)>;
    const size_t ND = hn::Lanes(dd);
    const VD vscale = hn::Set(dd, args.scale);
    const double* HWY_RESTRICT pr0 = args.partial.Row(row_c + 0);
    TC* HWY_RESTRICT cr0 = C_rows[row_c + 0];

    // We manually unroll 2x for higher IPC in batch=1.
    size_t col_c = range_nc.begin();
    if (HWY_LIKELY(range_nc.Num() >= 2 * ND)) {
      for (; col_c <= range_nc.end() - 2 * ND; col_c += 2 * ND) {
        VD a0, a1;  // unused if !kAdd
        if constexpr (kAdd) {
          // Promoting to double lets us fuse the Add into MulAdd.
          a0 = hn::PromoteTo(dd, hn::Load(df, args.add + col_c));
          a1 = hn::PromoteTo(dd, hn::Load(df, args.add + col_c + ND));
        }
        const VD d00 = hn::Load(dd, pr0 + col_c);
        const VD d01 = hn::Load(dd, pr0 + col_c + ND);
        VD m00, m01;
        if constexpr (kAdd) {
          m00 = hn::MulAdd(d00, vscale, a0);
          m01 = hn::MulAdd(d01, vscale, a1);
        } else {
          m00 = hn::Mul(d00, vscale);
          m01 = hn::Mul(d01, vscale);
        }
        // First convert f64 to f32.
        const VF f00 = hn::DemoteTo(df, m00);
        const VF f01 = hn::DemoteTo(df, m01);
        // Note that Stream is neutral on SKX and harmful on Zen4.
        hn::Store(TCFromF32(dc, f00), dc, cr0 + col_c);
        hn::Store(TCFromF32(dc, f01), dc, cr0 + col_c + ND);
      }
    }

    for (; col_c < range_nc.end(); col_c += ND) {
      const size_t remaining = range_nc.end() - col_c;
      HWY_DASSERT(remaining < 2 * ND);

      VD a0;  // unused if !kAdd
      if constexpr (kAdd) {
        // Promoting to double lets us fuse the Add into MulAdd.
        a0 = hn::PromoteTo(dd, hn::LoadN(df, args.add + col_c, remaining));
      }
      const VD d00 = hn::LoadN(dd, pr0 + col_c, remaining);
      VD m00;
      if constexpr (kAdd) {
        m00 = hn::MulAdd(d00, vscale, a0);
      } else {
        m00 = hn::Mul(d00, vscale);
      }
      // First convert f64 to f32.
      const VF f00 = hn::DemoteTo(df, m00);
      hn::StoreN(TCFromF32(dc, f00), dc, cr0 + col_c, remaining);
    }
  }
};  // MMScaleDemoteAdd

// Called on the main thread with the entire N range, or by each package with
// a static partition of N. This class contains several variants of the
// outer M/N/K loops, and calls `A2C0` which loops over the inner KC and MC.
// Its member variables avoid long argument lists in Do*().
class MMPerPackage {
 public:
  template <typename TA>
  MMPerPackage(const MatPtrT<TA>& A, const MMArgs& args, const MMConfig& config,
               size_t pkg_idx, const IndexRange& range_np)
      : args_(args),
        pkg_idx_(pkg_idx),
        // May be overwritten with a view of A, if already BF16.
        A_(args_.env->storage.A(pkg_idx, A.Extents())),
        range_np_(range_np),
        mr_(config.MR()),
        ranges_mc_(config.RangesOfMC(A.Rows())),
        ranges_kc_(config.RangesOfKC(A.Cols())),
        ranges_nc_(config.RangesOfNC(range_np)),
        order_(config.Order()),
        inner_tasks_(config.InnerTasks()),
        out_(config.Out()),
        line_bytes_(args.env->ctx.allocator.LineBytes()) {
    A_ = DecompressA(A);
  }

  // B is decompressed several call layers lower, but not all member functions
  // depend on TB, so pass it as an argument instead of templating the class.
  template <typename TB, typename TC>
  HWY_NOINLINE void operator()(const MatPtrT<TB>& B, RowPtrs<TC> C_rows) const {
    switch (order_) {
      case MMOrder::kNT:
        return DoNT(B, C_rows);
      case MMOrder::kNT_K:
        return DoNT_K(B, C_rows);
      case MMOrder::kNT_MT:
        return DoNT_MT(B, C_rows);
      case MMOrder::kNT_MT_K:
        return DoNT_MT_K(B, C_rows);
      default:
        HWY_UNREACHABLE;
    }
  }

 private:
  // Compute size of per-worker storage for `kNR` row ranges of B. Stack
  // allocation avoids passing a worker index.
  static constexpr size_t B_stride_max_ =
      MMStorage::kMaxKC + 2 * Allocator::MaxLineBytes() / sizeof(BF16);
  static constexpr size_t B_storage_max_ = kNR * B_stride_max_;

  // Granularity of `ForNP`. B rows produce C columns, so we
  // want a multiple of the line size to prevent false sharing.
  size_t MultipleNP(size_t sizeof_TC) const {
    return HWY_MAX(kNR, line_bytes_ / sizeof_TC);
  }

  // Single M and K ranges, parallel N. Fills all of C directly.
  template <typename TB, typename TC>
  HWY_INLINE void DoNT(const MatPtrT<TB>& B, RowPtrs<TC> C_rows) const {
    static const auto zone = args_.env->ctx.profiler.AddZone("MM.NT");
    HWY_DASSERT(ranges_mc_.NumTasks() == 1);
    HWY_DASSERT(ranges_kc_.NumTasks() == 1);
    const IndexRange& range_M = ranges_mc_.Range(0);
    const IndexRange& range_K = ranges_kc_.Range(0);
    const size_t K = range_K.Num();
    const StridedViewBF& A_view = A_.View(range_M.begin(), 0, K);
    const size_t B_stride =
        Stride(MatPadding::kOdd, K, sizeof(BF16), line_bytes_);

    // Similar to `loop_nc` below, but here we hoisted `A_view`.
    args_.env->parallel.ForNP(
        range_np_, MultipleNP(sizeof(TC)), inner_tasks_, pkg_idx_,
        [&](const IndexRange& range_nc, size_t worker) HWY_ATTR {
          MMZone mm_zone;
          mm_zone.MaybeEnter(worker, zone, args_);

          HWY_ALIGN BF16 B_storage[B_storage_max_];  // TLS
          const StridedViewBF B_storage_view(B_storage, K, B_stride);

          for (size_t row_b = range_nc.begin(); row_b < range_nc.end();
               row_b += kNR) {
            StridedViewBF B_view =
                DecompressB(B, row_b, range_K, B_storage_view);
            MMKernel::A2C0(A_view, B_view, mr_, range_M, row_b, K, MMSetC(),
                           args_, C_rows);
          }
        });

    HWY_DASSERT(out_ == MMOut::kDirect);  // already filled C
  }

  // Single M range, parallel N, sequential K. Fills all of partial.
  template <typename TB, typename TC>
  HWY_INLINE void DoNT_K(const MatPtrT<TB>& B, RowPtrs<TC> C_rows) const {
    static const auto zone = args_.env->ctx.profiler.AddZone("MM.NT_K");
    HWY_DASSERT(ranges_mc_.NumTasks() == 1);
    const IndexRange& range_mc = ranges_mc_.Range(0);

    // Loop over NC/MC/KC, called from the outer loops over K/N.
    // C++14 generic lambda enables hoisting branches via template
    // argument, while also capturing to avoid long argument lists.
    const auto loop_nc = [&](BF16* B_storage, const IndexRange& range_kc,
                             const IndexRange& range_nc,
                             auto out_tag) HWY_ATTR {
      const size_t kc = range_kc.Num();
      const StridedViewBF& A_view =
          A_.View(range_mc.begin(), range_kc.begin(), kc);
      const StridedViewBF B_storage_view(
          B_storage, kc,
          Stride(MatPadding::kOdd, kc, sizeof(BF16), line_bytes_));

      for (size_t row_b = range_nc.begin(); row_b < range_nc.end();
           row_b += kNR) {
        StridedViewBF B_view = DecompressB(B, row_b, range_kc, B_storage_view);
        MMKernel::A2C0(A_view, B_view, mr_, range_mc, row_b, kc, out_tag, args_,
                       C_rows);
      }
    };

    args_.env->parallel.ForNP(
        range_np_, MultipleNP(sizeof(TC)), inner_tasks_, pkg_idx_,
        [&](const IndexRange& range_nc, size_t worker) HWY_ATTR {
          MMZone mm_zone;
          mm_zone.MaybeEnter(worker, zone, args_);

          HWY_ALIGN BF16 B_storage[B_storage_max_];  // TLS

          // Peel off the first iteration of the kc loop: avoid
          // zero-initializing `partial` by writing into it.
          ranges_kc_.VisitFirst([&](const IndexRange& range_kc) {
            loop_nc(B_storage, range_kc, range_nc, MMSetPartial());
          });
          ranges_kc_.VisitRemaining([&](const IndexRange& range_kc) {
            loop_nc(B_storage, range_kc, range_nc, MMAddPartial());
          });
        });

    if (out_ == MMOut::kCopy) {
      static const auto zone =
          args_.env->ctx.profiler.AddZone("MM.NT_K.FillC.Copy");
      MMZone fill_zone;
      fill_zone.MaybeEnter(0, zone, args_);
      MMScaleDemoteAdd::FillC(range_mc, range_np_, args_, C_rows);
    } else if (out_ == MMOut::kParM) {
      static const auto zone =
          args_.env->ctx.profiler.AddZone("MM.NT_K.FillC.ParM");
      args_.env->parallel.ForRangeMC(
          range_mc, pkg_idx_, [&](size_t row_a, size_t worker) HWY_ATTR {
            MMZone fill_zone;
            fill_zone.MaybeEnter(worker, zone, args_);
            MMScaleDemoteAdd::FillC(IndexRange(row_a, row_a + 1), range_np_,
                                    args_, C_rows);
          });
    } else {
      HWY_UNREACHABLE;  // kDirect is only used with kNT.
    }
  }

  // Parallel loops over mc/nc blocks of M/range_np, single K.
  // Fills `mc x nc` sections of C directly, in parallel.
  template <typename TB, typename TC>
  HWY_INLINE void DoNT_MT(const MatPtrT<TB>& B, RowPtrs<TC> C_rows) const {
    static const auto zone = args_.env->ctx.profiler.AddZone("MM.NT_MT");
    HWY_DASSERT(ranges_kc_.NumTasks() == 1);
    const IndexRange& range_K = ranges_kc_.Range(0);
    const size_t K = range_K.Num();
    const size_t B_stride =
        Stride(MatPadding::kOdd, K, sizeof(BF16), line_bytes_);

    // Sequential loop over NC/MC/KC, similar to `loop_nc` below
    // except for the profiler strings and `out_tag`.
    args_.env->parallel.ForRangesMC_NC(
        ranges_mc_, ranges_nc_, pkg_idx_,
        [&](const IndexRange& range_mc, const IndexRange& range_nc,
            size_t worker) HWY_ATTR {
          MMZone mm_zone;
          mm_zone.MaybeEnter(worker, zone, args_);

          const StridedViewBF& A_view = A_.View(range_mc.begin(), 0, K);
          HWY_ALIGN BF16 B_storage[B_storage_max_];  // TLS
          const StridedViewBF B_storage_view(B_storage, K, B_stride);

          for (size_t row_b = range_nc.begin(); row_b < range_nc.end();
               row_b += kNR) {
            StridedViewBF B_view =
                DecompressB(B, row_b, range_K, B_storage_view);
            MMKernel::A2C0(A_view, B_view, mr_, range_mc, row_b, K, MMSetC(),
                           args_, C_rows);
          }
        });

    HWY_DASSERT(out_ == MMOut::kDirect);  // already filled C
  }

  // Parallel loops over mc/nc blocks of M/range_np, sequential K.
  // Fills `mc x nc` sections of `partial`, then `C`, in parallel.
  template <typename TB, typename TC>
  HWY_INLINE void DoNT_MT_K(const MatPtrT<TB>& B, RowPtrs<TC> C_rows) const {
    static const auto zone = args_.env->ctx.profiler.AddZone("MM.NT_MT_K");
    static const auto fill_zone =
        args_.env->ctx.profiler.AddZone("MM.NT_MT_K.FillC");
    const size_t kc_max = ranges_kc_.TaskSize();
    HWY_DASSERT(kc_max <= MMStorage::kMaxKC);
    const size_t B_stride =
        Stride(MatPadding::kOdd, kc_max, sizeof(BF16), line_bytes_);
    // Sequential loop over NC/MC/KC, for when the M/N loops are
    // already parallel. This is B3A2C0 in MOMMS terminology: we read
    // `mc x kc` of A, `nc x kc` of B, update `mc x nc` of `partial`.
    const auto loop_nc = [&](const StridedViewBF& B_storage_view,
                             const IndexRange& range_mc,
                             const IndexRange& range_kc,
                             const IndexRange& range_nc,
                             auto out_tag) HWY_ATTR {
      const size_t kc = range_kc.Num();
      const StridedViewBF& A_view =
          A_.View(range_mc.begin(), range_kc.begin(), kc);

      for (size_t row_b = range_nc.begin(); row_b < range_nc.end();
           row_b += kNR) {
        StridedViewBF B_view = DecompressB(B, row_b, range_kc, B_storage_view);
        MMKernel::A2C0(A_view, B_view, mr_, range_mc, row_b, kc, out_tag, args_,
                       C_rows);
      }
    };  // loop_nc
    args_.env->parallel.ForRangesMC_NC(
        ranges_mc_, ranges_nc_, pkg_idx_,
        [&](const IndexRange& range_mc, const IndexRange& range_nc,
            size_t worker) HWY_ATTR {
          MMZone mm_zone;
          mm_zone.MaybeEnter(worker, zone, args_);

          HWY_ALIGN BF16 B_storage[B_storage_max_];  // TLS
          const StridedViewBF B_storage_view(B_storage, kc_max, B_stride);

          // Peel off the first iteration of the kc loop: avoid
          // zero-initializing `partial` by writing into it.
          ranges_kc_.VisitFirst([&](const IndexRange& range_kc) {
            loop_nc(B_storage_view, range_mc, range_kc, range_nc,
                    MMSetPartial());
          });
          ranges_kc_.VisitRemaining([&](const IndexRange& range_kc) {
            loop_nc(B_storage_view, range_mc, range_kc, range_nc,
                    MMAddPartial());
          });

          // Already in parallel section, hence no `kParM`, and
          // `kDirect` is only used with `kNT_MT`.
          HWY_DASSERT(out_ == MMOut::kCopy);
          MMZone fill_mm_zone;
          fill_mm_zone.MaybeEnter(worker, fill_zone, args_);
          MMScaleDemoteAdd::FillC(range_mc, range_nc, args_, C_rows);
        });
  }

  // Decompresses all `M x K` from `A` into padded BF16 `A_`. Assumes `TA` is a
  // seekable type (i.e., not NUQ) so we can use pointer arithmetic.
  template <typename TA>
  HWY_NOINLINE void DoDecompressA(const MatPtrT<TA>& A, MMParA par_a) const {
    const IndexRange all_M(0, A.Rows());
    const IndexRange all_K(0, A.Cols());
    HWY_DASSERT(all_K.Num() == A_.Cols());

    const hn::ScalableTag<BF16> dbf;
    const size_t NBF = hn::Lanes(dbf);
    static_assert(hwy::IsSameEither<TA, BF16, float>(), "Can seek");

    static const auto zone = args_.env->ctx.profiler.AddZone("MM.DecompressA");

    const auto do_range = [&](const IndexRange& range_M,
                              const IndexRange& range_K,
                              size_t worker) HWY_ATTR {
      MMZone mm_zone;
      mm_zone.MaybeEnter(worker, zone, args_);

      const size_t col0 = range_K.begin();
      const size_t cols = range_K.Num();
      // Must be a vector multiple, or the last range before row padding,
      // otherwise `DecompressAndZeroPad` overwrites neighbors.
      HWY_DASSERT(cols % NBF == 0 || range_K.end() == A.Cols());
      for (size_t row_a : range_M) {
        const PackedSpan<const TA> from = MakeSpan(A.Row(row_a) + col0, cols);
        BF16* HWY_RESTRICT to = A_.Row(row_a) + col0;
        DecompressAndZeroPad(dbf, from, 0, to, cols);
        // Verify that we zero-padded.
        if constexpr (HWY_IS_DEBUG_BUILD) {
          for (size_t i = cols; i < hwy::RoundUpTo(cols, NBF); ++i) {
            HWY_DASSERT(hwy::ConvertScalarTo<float>(to[i]) == 0.0f);
          }
        }
      }
    };

    switch (par_a) {
      case MMParA::kNone:
        do_range(all_M, all_K, /*worker=*/0);
        break;
      case MMParA::kK1:
      case MMParA::kK2:
      case MMParA::kK4: {
        const size_t inner_tasks = static_cast<size_t>(par_a);
        // At least one vector, otherwise DecompressAndZeroPad will add
        // padding, which might overwrite neighboring tasks. Also a whole cache
        // line to avoid false sharing.
        const size_t multiple_K = HWY_MAX(NBF, line_bytes_ / sizeof(BF16));

        args_.env->parallel.ForNP(
            all_K, multiple_K, inner_tasks, pkg_idx_,
            [&](const IndexRange& range_K, size_t worker) {
              do_range(all_M, range_K, worker);
            });
        break;
      }
      case MMParA::kM:
        args_.env->parallel.ForRangeMC(
            all_M, pkg_idx_, [&](size_t row_a, size_t worker) {
              do_range(IndexRange(row_a, row_a + 1), all_K, worker);
            });
        break;
    }
  }

  // Autotuning wrapper for `DoDecompressA`.
  template <typename TA>
  HWY_INLINE StridedViewBF DecompressA(const MatPtrT<TA>& A) const {
    MMAutoTune<MMParA>& autotune = args_.per_key->autotune_par_a[pkg_idx_];
    // If already BF16, maybe return a view:
    if constexpr (hwy::IsSame<TA, BF16>()) {
      // Only if vector multiple and padded (see `DoDecompressA`).
      const size_t NBF = hn::Lanes(hn::ScalableTag<BF16>());
      if (HWY_LIKELY(A.Cols() % NBF == 0 && !A.IsPacked())) {
        // Const, but cast because StridedView is also used for `partial` which
        // is non-const.
        return StridedViewBF(const_cast<TA*>(A.Row(0)), A.Cols(), A.Stride());
      }
    }

    if (HWY_LIKELY(autotune.Best())) {
      DoDecompressA(A, *autotune.Best());
      return A_;
    }

    // First call: generate candidates.
    if (HWY_UNLIKELY(!autotune.HasCandidates())) {
      const MMParA other = (A.Rows() == 1) ? MMParA::kNone : MMParA::kM;
      std::vector<MMParA> candidates = {MMParA::kK1, MMParA::kK2, MMParA::kK4,
                                        other};
      autotune.SetCandidates(candidates);
    }

    const MMParA& par_a = autotune.NextConfig();
    const uint64_t t0 = hwy::timer::Start();
    DoDecompressA(A, par_a);
    const uint64_t t1 =
        args_.env->have_timer_stop ? hwy::timer::Stop() : hwy::timer::Start();
    const uint64_t min_elapsed = autotune.NotifyTicks(t1 - t0);
    if (HWY_UNLIKELY(args_.env->print_measurement && autotune.ShouldPrint())) {
      fprintf(stderr, "%s,%7.3f\n", StringFromParA(par_a),
              static_cast<double>(min_elapsed) /
                  hwy::platform::InvariantTicksPerSecond() * 1E6);
    }
    return A_;
  }

  // Decompresses `kNR x kc` from `B[row_b, range_kc.begin()]` to row 0,
  // col 0 of `B_view`. Decompressing SFP is relatively cheap on `AVX3_DL`
  // thanks to its large table lookups, and less so on other targets.
  template <typename TB>
  HWY_INLINE StridedViewBF DecompressB(const MatPtrT<TB>& B, const size_t row_b,
                                       const IndexRange& range_kc,
                                       const StridedViewBF& B_view) const {
    if constexpr (hwy::IsSame<TB, BF16>()) {
      return StridedViewBF(const_cast<BF16*>(B.Row(row_b)) + range_kc.begin(),
                           range_kc.Num(), B.Stride());
    }

    const hn::ScalableTag<BF16> dbf;
    const PackedSpan<const TB> B_span = B.PaddedSpan();

    const size_t kc = range_kc.Num();
    const size_t col0 = range_kc.begin();

    for (size_t r = 0; r < kNR; ++r) {
      const size_t packed_ofs = (row_b + r) * B.Stride() + col0;
      BF16* HWY_RESTRICT to = B_view.Row(r);
      DecompressAndZeroPad(dbf, B_span, packed_ofs, to, kc);
      // Verify that we zero-padded.
      if constexpr (HWY_IS_DEBUG_BUILD) {
        for (size_t i = kc; i < hwy::RoundUpTo(kc, hn::Lanes(dbf)); ++i) {
          HWY_DASSERT(hwy::ConvertScalarTo<float>(to[i]) == 0.0f);
        }
      }
    }
    return B_view;
  }

  const MMArgs args_;  // copy for locality
  const size_t pkg_idx_;
  StridedViewBF A_;  // view into A or pkg_A_, both of which are padded.

  const IndexRange range_np_;
  // From MMConfig:
  const size_t mr_;
  const IndexRangePartition ranges_mc_;
  const IndexRangePartition ranges_kc_;
  const IndexRangePartition ranges_nc_;
  const MMOrder order_;
  const size_t inner_tasks_;
  const MMOut out_;
  const size_t line_bytes_;
};  // MMPerPackage

// Stateless, wraps member functions.
struct MMImpl {
  // Returns existing entry for the given key or -1.
  static HWY_INLINE intptr_t IndexOfKey(MMKeys::Key key, const MMKeys& keys) {
    const hwy::Span<const uint64_t> all_keys = keys.Keys();
    // TODO: SIMD scan
    for (size_t i = 0; i < all_keys.size(); ++i) {
      if (all_keys[i] == key) return static_cast<intptr_t>(i);
    }
    return -1;
  }

  // Called from `MatMul` from two places: either with the next autotune config,
  // or with the best config.
  template <typename TA, typename TB, typename TC>
  static HWY_NOINLINE void DoMatMul(const MatPtrT<TA>& A, const MatPtrT<TB>& B,
                                    RowPtrs<TC> C_rows, const MMArgs& args,
                                    const MMConfig& config) {
    PROFILER_ZONE("MM.DoMatMul");
    static const auto zone =
        args.env->ctx.profiler.AddZone("MM.DoMatMul.PerPkg");

    if constexpr (kMaxPackages > 1) {
      // Outermost loop: static NUMA-aware partition of B rows across packages.
      args.env->parallel.ForPkg(
          args.per_key->ranges_np.NumTasks(), [&](size_t pkg_idx) {
            MMZone mm_zone;
            mm_zone.MaybeEnter(pkg_idx, zone, args);
            const IndexRange& range_np = args.per_key->ranges_np.Range(pkg_idx);
            MMPerPackage(A, args, config, pkg_idx, range_np)(B, C_rows);
          });
    } else {
      const size_t pkg_idx = 0;
      HWY_DASSERT(args.per_key->ranges_np.NumTasks() == 1);
      const IndexRange& range_np = args.per_key->ranges_np.Range(pkg_idx);
      MMPerPackage(A, args, config, pkg_idx, range_np)(B, C_rows);
    }
  }
};

// Computes the matrix product `A * B * scale [+ add]` and stores it in `C`.
//
// `A` is a row-major matrix with `M` rows and `B` is transposed. The latter's
// `K = B.Cols()`, which must match `A.Cols()`, is the number
// of rows in the original B. `N = C.Cols()` must be a multiple of 4. There
// are no other restrictions on shape, though performance is better when `M % 4
// == 0` or `M <= 4`, and when A is padded (`!A.IsPacked()`).
//
// NOTE: if A and/or B are BF16 and padded, the interval `[Cols(),
// hwy::RoundUpTo(Cols(), hn::Lanes(dbf))` must be zero-initialized to match
// the behavior of `DecompressAndZeroPad`. We check this in debug builds.
//
// If `add` is non-null, the row-vector `add` is added to each of the `M` rows
// of `C`, which is a row-major matrix with arbitrary stride. A scale for
// `add` is not supported, so make sure its scale is 1.
//
// Must not be called concurrently with the same `env`. The first few calls
// for a given shape will try different configs. The best is recorded in `env`
// and will be used for subsequent calls with that shape.
//
// Returns the (autotuning) state for the current shape. This pointer may be
// invalidated by the next call to `MatMul`.
//
// Uses considerable stack space: at least 40 KiB per thread.
template <typename TA, typename TB, typename TC>
HWY_NOINLINE MMPerKey* MatMul(const MatPtrT<TA>& A, const MatPtrT<TB>& B,
                              const float* HWY_RESTRICT add, MatMulEnv& env,
                              MatPtrT<TC>& C) {
  RowPtrs<TC> C_rows = GetOrSetTempRowPtrs(C, env.row_ptrs[2]);

  const Allocator& allocator = env.ctx.allocator;
  const size_t M = A.Rows();
  const size_t K = A.Cols();
  const size_t N = B.Rows();
  const MMKeys::Key key = MMKeys::KeyFromDims(M, K, N);
  intptr_t index = MMImpl::IndexOfKey(key, env.keys);
  // First time we see this shape/key.
  if (HWY_UNLIKELY(index < 0)) {
    env.keys.Append(key, allocator);

    size_t max_packages = kMaxPackages;
    // For low-batch, multiple sockets only help if binding is enabled.
    if (!allocator.ShouldBind() && M <= 4) {
      max_packages = 1;
    }

    // invalidates `MMAutoTune::Best()`
    index = env.per_key.size();
    env.per_key.push_back(
        MMPerKey(max_packages, N, sizeof(TC), kNR, env.parallel));
  }
  MMPerKey& per_key = env.per_key[index];
  MMAutoTune<MMConfig>& tuner = per_key.autotune;

  const MMArgs args(env, per_key, static_cast<double>(A.Scale()) * B.Scale(),
                    add, env.storage.Partial());
  if (HWY_LIKELY(tuner.Best())) {
    MMImpl::DoMatMul(A, B, C_rows, args, *tuner.Best());
    return &per_key;
  }

  // From here, CPU time is negligible except DoMatMul.

  // First call: enumerate all feasible configs.
  if (HWY_UNLIKELY(!tuner.HasCandidates())) {
    // Ensure matrix dimensions match each other.
    HWY_ASSERT(K == B.Cols());
    HWY_ASSERT(M <= MMStorage::kMaxM);
    HWY_ASSERT(K <= MMStorage::kMaxK);
    HWY_ASSERT(N <= MMStorage::kMaxN);
    HWY_ASSERT(N % kNR == 0);

    tuner.SetCandidates(MMCandidates(allocator, M, K, N, sizeof(TC), kMaxMR,
                                     kNR, per_key.ranges_np, env.print_config));
  }

  const MMConfig& cfg = tuner.NextConfig();
  const uint64_t t0 = hwy::timer::Start();
  MMImpl::DoMatMul(A, B, C_rows, args, cfg);
  const uint64_t t1 =
      env.have_timer_stop ? hwy::timer::Stop() : hwy::timer::Start();
  const double min_elapsed = static_cast<double>(tuner.NotifyTicks(t1 - t0)) /
                             hwy::platform::InvariantTicksPerSecond();
  const double flops = 2 * M * K * N / min_elapsed;  // * 2 for FMA
  if (HWY_UNLIKELY(env.print_measurement && tuner.ShouldPrint())) {
    fprintf(stderr, "%7.1f,%.2f,%zu,%4zu,%4zu,%5zu,%s,%zu,%s\n", flops * 1E-9,
            min_elapsed * 1E3, cfg.MR(), cfg.MC(), cfg.KC(), cfg.NC(),
            StringFromOrder(cfg.Order()), cfg.InnerTasks(),
            StringFromOut(cfg.Out()));
  }
  if (HWY_UNLIKELY(env.print_best && tuner.Best())) {
    const auto ratio = [per_key](uint64_t ticks) -> double {
      return static_cast<double>(ticks) /
             static_cast<double>(per_key.autotune.BestTicks());
    };
    const MMConfig& best = *tuner.Best();
    fprintf(stderr,
            "\n%zu,%zu,%zu,%7.1f,%.2f,%zu,%4zu,%4zu,%5zu,%s,%zu,%s,%.2f,%.2f\n",
            M, K, N, flops * 1E-9, min_elapsed * 1E3, best.MR(), best.MC(),
            best.KC(), best.NC(), StringFromOrder(best.Order()),
            best.InnerTasks(), StringFromOut(best.Out()),
            ratio(tuner.WorstMinTicks()), ratio(tuner.FirstConfigTicks()));
  }

  return &per_key;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
