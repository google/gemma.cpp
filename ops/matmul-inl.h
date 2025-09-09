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
#include "util/allocator.h"  // CacheInfo
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

// Converts from float intermediate to/from MatMul output type `TC`.
template <class DC, HWY_IF_F32_D(DC)>
hn::Vec<DC> TCFromF32(DC /*dc*/, hn::Vec<DC> vf) {
  return vf;
}
template <class DC, class DF = hn::Rebind<float, DC>, HWY_IF_BF16_D(DC)>
hn::Vec<DC> TCFromF32(DC dc, hn::Vec<DF> vf) {
  return hn::DemoteTo(dc, vf);
}
template <class DC, HWY_IF_F32_D(DC)>
hn::Vec<DC> F32FromTC(DC /*dc*/, hn::Vec<DC> vc) {
  return vc;
}
template <class DC, class DF = hn::Rebind<float, DC>, HWY_IF_BF16_D(DC)>
hn::Vec<DF> F32FromTC(DC dc, hn::Vec<DC> vc) {
  return hn::PromoteTo(DF(), vc);
}

// Tag classes, passed to `MMKernel::A2C0` to choose between writing one
// (all-K) result to C via `MMStoreHorizontalSumsIntoC`, or accumulating the
// next kc result into `C`.
struct MMSetC {};
struct MMAddC {};

// Stores horizontal sums of up to 16 vectors via transpose.
template <size_t kRowsAC>
class MMStoreHorizontalSumsIntoC {
 public:
  static_assert(kNR == 4);  // for `StoreInterleaved4`

  // Given 16 (`kRowsAC x kNR`) full vectors of 32-bit float, returns four
  // 4-wide float vectors with their horizontal sums.
  // `Crc` are the 16 combinations of an A row vector indexed by `r`, times a
  // transposed B row vector indexed by `c`. Their elements are thus a subset
  // of the terms of the dot product constituting the final `C[r, c]` result.
  // Thus we compute the horizontal sums of each `Crc`. The elements may be
  // permuted because we multiply bf16 via `ReorderWidenMulAccumulate`, but
  // this does not change their horizontal sum.
  template <class DF, class VF = hn::Vec<DF>, class D4 = hn::Full128<float>,
            class V4 = hn::Vec<D4>>
  HWY_INLINE void Reduce4x4(DF df,                           //
                            VF C00, VF C01, VF C02, VF C03,  //
                            VF C10, VF C11, VF C12, VF C13,  //
                            VF C20, VF C21, VF C22, VF C23,  //
                            VF C30, VF C31, VF C32, VF C33,  //
                            V4& sum0, V4& sum1, V4& sum2, V4& sum3) {
    HWY_ALIGN float buf[16 * hn::MaxLanes(df)];
    HWY_LANES_CONSTEXPR const size_t N = hn::Lanes(df);
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
    // Store to four elements per row of `C`.
    // No loop is required because vectors are at least 4*32 bits.
    const D4 d4;
    sum0 = MaybeLoad<0>(d4, N, buf);
    sum1 = MaybeLoad<1>(d4, N, buf);
    sum2 = MaybeLoad<2>(d4, N, buf);
    sum3 = MaybeLoad<3>(d4, N, buf);

    for (size_t lane = 1; lane < N; ++lane) {
      sum0 = MaybeAdd<0>(d4, N, sum0, buf + kNR * lane);
      sum1 = MaybeAdd<1>(d4, N, sum1, buf + kNR * lane);
      sum2 = MaybeAdd<2>(d4, N, sum2, buf + kNR * lane);
      sum3 = MaybeAdd<3>(d4, N, sum3, buf + kNR * lane);
    }
  }

  // Scales the dot-product terms and adds bias (if present) and stores the
  // four 4-wide vectors to `C` starting at `(row_c, col_c)`. If `tag` is
  // `MMSetC`, the vectors are written as-is (first call, or small K).
  // Otherwise, they are partial sums and are accumulated into C.
  template <class D4, class V4 = hn::Vec<D4>, class Tag, typename TC>
  HWY_INLINE void Store(D4 d4, V4 sum0, V4 sum1, V4 sum2, V4 sum3, Tag tag,
                        const size_t row_c, const size_t col_c,
                        const MMArgs& args, RowPtrs<TC> C_rows) const {
    const V4 vscale = hn::Set(d4, args.scale);
    HWY_ALIGN static constexpr float kZero[4] = {};
    const V4 vadd = hn::Load(d4, args.add ? args.add + col_c : kZero);
    MaybeScaleAndStore<0>(d4, sum0, vscale, vadd, tag, C_rows, row_c, col_c);
    MaybeScaleAndStore<1>(d4, sum1, vscale, vadd, tag, C_rows, row_c, col_c);
    MaybeScaleAndStore<2>(d4, sum2, vscale, vadd, tag, C_rows, row_c, col_c);
    MaybeScaleAndStore<3>(d4, sum3, vscale, vadd, tag, C_rows, row_c, col_c);
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
            class Tag, typename TC>
  static HWY_INLINE void MaybeScaleAndStore(DF4 df4, VF4 sum, VF4 vscale,
                                            VF4 vadd, Tag, RowPtrs<TC> C_rows,
                                            const size_t row_c,
                                            const size_t col_c) {
    if constexpr (kRow < kRowsAC) {
      TC* HWY_RESTRICT pos = C_rows[row_c + kRow] + col_c;
      const hn::Rebind<TC, DF4> dc4;
      if constexpr (hwy::IsSame<Tag, MMAddC>()) {
        vadd = F32FromTC(dc4, hn::Load(dc4, pos));  // load prior value
      } else {
        static_assert(hwy::IsSame<Tag, MMSetC>());
        // vadd remains the bias (added once, the first time we store to C)
      }
      const VF4 out = hn::MulAdd(sum, vscale, vadd);
      hn::Store(TCFromF32(dc4, out), dc4, pos);
    }
  }
};  // MMStoreHorizontalSumsIntoC

// Stateless, wraps member functions.
class MMKernel {
 public:
  // Calls `LoopKC` for each of `mc` rows of A in steps of `mr`. `A_view`
  // is `mc x kc` and `B_view` is `(kNR x kc)`. Both start at row/col 0.
  // A2C0 in MOMMS terminology updates a `mc x kNR` slice of the output.
  template <class Tag, typename TC>
  static HWY_INLINE void A2C0(const StridedViewBF A_view,
                              const StridedViewBF B_view, size_t mr,
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
  static HWY_INLINE void ElementwiseMulAccNativeBF(DBF dbf, VBF a, VBF b0,
                                                   VBF b1, VBF b2, VBF b3,
                                                   VF& C0, VF& C1, VF& C2,
                                                   VF& C3) {
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

  // Like `ElementwiseMulAccNativeBF`, but splits BF16 inputs into odd and even
  // f32 for use with FMA. Also handles two rows at a time to hide the FMA
  // latency (we assume 4 cycles and dual-issue) before writing `C00` again.
  template <class DBF, class VBF = hn::Vec<DBF>,
            class DF = hn::Repartition<float, DBF>, class VF = hn::Vec<DF>>
  static HWY_INLINE void ElementwiseMulAccEmuBF(DBF dbf, VBF a0, VBF a1, VF b0o,
                                                VF b0e, VF b1o, VF b1e, VF b2o,
                                                VF b2e, VF b3o, VF b3e, VF& C00,
                                                VF& C01, VF& C02, VF& C03,
                                                VF& C10, VF& C11, VF& C12,
                                                VF& C13) {
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

  // Innermost loop over `kc` columns (typically 1024-4096, not necessarily a
  // multiple of `NBF`) in steps of one vector, for `kRowsAC` rows of `A_view`
  // from range_mc-relative `imc` and `B_view` from row 0 (both at column 0).
  // Updates a `kRowsAC x kNR` tile with top-left `C.Row(row_ac) + col_c`.
  // `A` and `B` are always BF16, `C` can be F32 or BF16.
  template <size_t kRowsAC, /*deduced:*/ class Tag, typename TC>
  static HWY_INLINE void LoopKC(const StridedViewBF A_view,
                                const StridedViewBF B_view, size_t row_ac,
                                size_t imc, size_t col_c, size_t kc, Tag tag,
                                const MMArgs& args, RowPtrs<TC> C_rows) {
    const hn::ScalableTag<BF16> dbf;
    using VBF = hn::Vec<decltype(dbf)>;
    HWY_LANES_CONSTEXPR const size_t NBF = hn::Lanes(dbf);

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

    // Neither A nor B are guaranteed to be zero-padded: they might be a view
    // into the left half.

    // Accumulate into f32.
    const hn::Repartition<float, decltype(dbf)> df;
    using VF = hn::Vec<decltype(df)>;
    VF C00 = hn::Zero(df), C01 = hn::Zero(df), C02 = hn::Zero(df),
       C03 = hn::Zero(df), C10 = hn::Zero(df), C11 = hn::Zero(df),
       C12 = hn::Zero(df), C13 = hn::Zero(df), C20 = hn::Zero(df),
       C21 = hn::Zero(df), C22 = hn::Zero(df), C23 = hn::Zero(df),
       C30 = hn::Zero(df), C31 = hn::Zero(df), C32 = hn::Zero(df),
       C33 = hn::Zero(df);

    size_t ikc = 0;
    const HWY_LANES_CONSTEXPR size_t kc_step = NBF;
    if (kc >= kc_step) {
      HWY_UNROLL(1)
      for (; ikc <= kc - kc_step; ikc += kc_step) {
        if constexpr (HWY_NATIVE_DOT_BF16) {
          // NOTE: matmul_test has packed B so that it can call Span. The test
          // cases with non-vector-multiple K require unaligned loads here.
          // However, in actual usage, we should always have padded and thus
          // aligned A and B.
          const VBF b0 = hn::LoadU(dbf, br0 + ikc);
          const VBF b1 = hn::LoadU(dbf, br1 + ikc);
          const VBF b2 = hn::LoadU(dbf, br2 + ikc);
          const VBF b3 = hn::LoadU(dbf, br3 + ikc);

          {
            const VBF a0 = hn::Load(dbf, ar0 + ikc);
            ElementwiseMulAccNativeBF(dbf, a0, b0, b1, b2, b3, C00, C01, C02,
                                      C03);
          }
          if constexpr (kRowsAC > 1) {
            const VBF a1 = hn::Load(dbf, ar1 + ikc);
            ElementwiseMulAccNativeBF(dbf, a1, b0, b1, b2, b3, C10, C11, C12,
                                      C13);
          }
          if constexpr (kRowsAC > 2) {
            const VBF a2 = hn::Load(dbf, ar2 + ikc);
            ElementwiseMulAccNativeBF(dbf, a2, b0, b1, b2, b3, C20, C21, C22,
                                      C23);
          }
          if constexpr (kRowsAC > 3) {
            const VBF a3 = hn::Load(dbf, ar3 + ikc);
            ElementwiseMulAccNativeBF(dbf, a3, b0, b1, b2, b3, C30, C31, C32,
                                      C33);
          }
        } else {  // !HWY_NATIVE_DOT_BF16
          // When both are BF16, it is better to load promote odd/even,
          // because lane-crossing promotion for both might be bottlenecked on
          // shuffles.
          VF b0e, b1e, b2e, b3e, b0o, b1o, b2o, b3o;
          {
            const VBF b0 = hn::LoadU(dbf, br0 + ikc);
            const VBF b1 = hn::LoadU(dbf, br1 + ikc);
            const VBF b2 = hn::LoadU(dbf, br2 + ikc);
            const VBF b3 = hn::LoadU(dbf, br3 + ikc);
            b0e = hn::PromoteEvenTo(df, b0);
            b1e = hn::PromoteEvenTo(df, b1);
            b2e = hn::PromoteEvenTo(df, b2);
            b3e = hn::PromoteEvenTo(df, b3);
            b0o = FastPromoteOddTo(df, b0);
            b1o = FastPromoteOddTo(df, b1);
            b2o = FastPromoteOddTo(df, b2);
            b3o = FastPromoteOddTo(df, b3);
          }

          // Two rows at a time so we have 8 separate dependency chains,
          // sufficient for IPC=2 and 4-cycle latency.
          {
            const VBF a0 = hn::Load(dbf, ar0 + ikc);
            const VBF a1 = kRowsAC > 1 ? hn::Load(dbf, ar1 + ikc) : a0;
            ElementwiseMulAccEmuBF(dbf, a0, a1, b0o, b0e, b1o, b1e, b2o, b2e,
                                   b3o, b3e, C00, C01, C02, C03, C10, C11, C12,
                                   C13);
          }
          if constexpr (kRowsAC > 2) {
            const VBF a2 = hn::Load(dbf, ar2 + ikc);
            const VBF a3 = kRowsAC > 3 ? hn::Load(dbf, ar3 + ikc) : a2;
            ElementwiseMulAccEmuBF(dbf, a2, a3, b0o, b0e, b1o, b1e, b2o, b2e,
                                   b3o, b3e, C20, C21, C22, C23, C30, C31, C32,
                                   C33);
          }
        }
      }
    }

    // Always handle remainders: even though A and B are generally padded, we
    // might have a view into the left half of A and/or B.
    const size_t remaining_kc = kc - ikc;
    HWY_DASSERT(remaining_kc < kc_step);
    if (HWY_UNLIKELY(remaining_kc != 0)) {
      if constexpr (HWY_NATIVE_DOT_BF16) {
        const VBF b0 = hn::LoadN(dbf, br0 + ikc, remaining_kc);
        const VBF b1 = hn::LoadN(dbf, br1 + ikc, remaining_kc);
        const VBF b2 = hn::LoadN(dbf, br2 + ikc, remaining_kc);
        const VBF b3 = hn::LoadN(dbf, br3 + ikc, remaining_kc);

        {
          const VBF a0 = hn::LoadN(dbf, ar0 + ikc, remaining_kc);
          ElementwiseMulAccNativeBF(dbf, a0, b0, b1, b2, b3, C00, C01, C02,
                                    C03);
        }
        if constexpr (kRowsAC > 1) {
          const VBF a1 = hn::LoadN(dbf, ar1 + ikc, remaining_kc);
          ElementwiseMulAccNativeBF(dbf, a1, b0, b1, b2, b3, C10, C11, C12,
                                    C13);
        }
        if constexpr (kRowsAC > 2) {
          const VBF a2 = hn::LoadN(dbf, ar2 + ikc, remaining_kc);
          ElementwiseMulAccNativeBF(dbf, a2, b0, b1, b2, b3, C20, C21, C22,
                                    C23);
        }
        if constexpr (kRowsAC > 3) {
          const VBF a3 = hn::LoadN(dbf, ar3 + ikc, remaining_kc);
          ElementwiseMulAccNativeBF(dbf, a3, b0, b1, b2, b3, C30, C31, C32,
                                    C33);
        }
      } else {  // !HWY_NATIVE_DOT_BF16
        // When both are BF16, it is better to load promote odd/even, because
        // lane-crossing promotion for both might be bottlenecked on shuffles.
        VF b0e, b1e, b2e, b3e, b0o, b1o, b2o, b3o;
        {
          const VBF b0 = hn::LoadN(dbf, br0 + ikc, remaining_kc);
          const VBF b1 = hn::LoadN(dbf, br1 + ikc, remaining_kc);
          const VBF b2 = hn::LoadN(dbf, br2 + ikc, remaining_kc);
          const VBF b3 = hn::LoadN(dbf, br3 + ikc, remaining_kc);
          b0e = hn::PromoteEvenTo(df, b0);
          b1e = hn::PromoteEvenTo(df, b1);
          b2e = hn::PromoteEvenTo(df, b2);
          b3e = hn::PromoteEvenTo(df, b3);
          b0o = FastPromoteOddTo(df, b0);
          b1o = FastPromoteOddTo(df, b1);
          b2o = FastPromoteOddTo(df, b2);
          b3o = FastPromoteOddTo(df, b3);
        }

        // Two rows at a time so we have 8 separate dependency chains,
        // sufficient for IPC=2 and 4-cycle latency.
        {
          const VBF a0 = hn::LoadN(dbf, ar0 + ikc, remaining_kc);
          const VBF a1 =
              kRowsAC > 1 ? hn::LoadN(dbf, ar1 + ikc, remaining_kc) : a0;
          ElementwiseMulAccEmuBF(dbf, a0, a1, b0o, b0e, b1o, b1e, b2o, b2e, b3o,
                                 b3e, C00, C01, C02, C03, C10, C11, C12, C13);
        }
        if constexpr (kRowsAC > 2) {
          const VBF a2 = hn::LoadN(dbf, ar2 + ikc, remaining_kc);
          const VBF a3 =
              kRowsAC > 3 ? hn::LoadN(dbf, ar3 + ikc, remaining_kc) : a2;
          ElementwiseMulAccEmuBF(dbf, a2, a3, b0o, b0e, b1o, b1e, b2o, b2e, b3o,
                                 b3e, C20, C21, C22, C23, C30, C31, C32, C33);
        }
      }
    }  // remaining_kc != 0

    // This is a substantial fraction (about 1/3) of the total time, but is
    // called frequently, so do not add a profiler zone.

    MMStoreHorizontalSumsIntoC<kRowsAC> horz;
    const hn::Full128<float> d4;
    hn::Vec<decltype(d4)> sum0, sum1, sum2, sum3;
    horz.Reduce4x4(df, C00, C01, C02, C03, C10, C11, C12, C13, C20, C21, C22,
                   C23, C30, C31, C32, C33, sum0, sum1, sum2, sum3);
    horz.Store(d4, sum0, sum1, sum2, sum3, tag, row_ac, col_c, args, C_rows);
  }
};

// Miscellaneous stateless helper functions.
class MMImpl {
  // Returns existing entry for the given key or -1.
  static HWY_INLINE intptr_t IndexOfKey(MMKeys::Key key, const MMKeys& keys) {
    const hwy::Span<const uint64_t> all_keys = keys.Keys();

    const hn::ScalableTag<uint64_t> d;
    using V = hn::Vec<decltype(d)>;
    const V broadcasted = Set(d, key);
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    if (all_keys.size() >= N) {
      for (; i <= all_keys.size() - N; i += N) {
        const intptr_t pos = hn::FindFirstTrue(
            d, hn::Eq(broadcasted, hn::LoadU(d, &all_keys[i])));
        if (pos >= 0) return static_cast<intptr_t>(i) + pos;
      }
    }

    const size_t remaining = all_keys.size() - i;
    if (HWY_LIKELY(remaining > 0)) {
      HWY_DASSERT(remaining < N);
      const V v = hn::LoadN(d, &all_keys[i], remaining);
      const intptr_t pos = hn::FindFirstTrue(d, hn::Eq(broadcasted, v));
      if (pos >= 0) return static_cast<intptr_t>(i) + pos;
    }

    return -1;
  }

 public:
  static MMPerKey& FindOrAddPerKey(size_t M, size_t K, size_t N,
                                   size_t vector_bytes,
                                   MatMulEnv::PerCluster& per_cluster) {
    const MMKeys::Key key = MMKeys::KeyFromDims(M, K, N);
    intptr_t index = MMImpl::IndexOfKey(key, per_cluster.keys);
    // First time we see this shape/key.
    if (HWY_UNLIKELY(index < 0)) {
      per_cluster.keys.Append(key, vector_bytes);

      // Invalidates `MMAutoTune::Best()`.
      std::vector<MMPerKey>& per_keys = per_cluster.per_key;
      index = per_keys.size();
      per_keys.push_back(MMPerKey());
    }
    return per_cluster.per_key[index];
  }

  static void NotifyAutotuneResult(size_t M, size_t K, size_t N, double t0,
                                   const MMConfig& cfg, MatMulEnv& env,
                                   MMAutoTune<MMConfig>& tuner) {
    const uint64_t t1 =
        env.have_timer_stop ? hwy::timer::Stop() : hwy::timer::Start();
    const double min_elapsed = static_cast<double>(tuner.NotifyTicks(t1 - t0)) /
                               hwy::platform::InvariantTicksPerSecond();
    const double flops = 2 * M * K * N / min_elapsed;  // * 2 for FMA
    if (HWY_UNLIKELY(env.print_measurement && tuner.ShouldPrint())) {
      fprintf(stderr, "%7.1f,%.2f,%zu,%4zu,%4zu,%5zu,%s,%zu\n", flops * 1E-9,
              min_elapsed * 1E3, cfg.MR(), cfg.MC(), cfg.KC(), cfg.NC(),
              StringFromOrder(cfg.Order()), cfg.InnerTasks());
    }
    if (HWY_UNLIKELY(env.print_best && tuner.Best())) {
      const auto ratio = [&tuner](uint64_t ticks) -> double {
        return static_cast<double>(ticks) /
               static_cast<double>(tuner.BestTicks());
      };
      const MMConfig& best = *tuner.Best();
      fprintf(stderr,
              "\n%zu,%zu,%zu,%7.1f,%.2f,%zu,%4zu,%4zu,%5zu,%s,%zu,%.2f,%.2f\n",
              M, K, N, flops * 1E-9, min_elapsed * 1E3, best.MR(), best.MC(),
              best.KC(), best.NC(), StringFromOrder(best.Order()),
              best.InnerTasks(), ratio(tuner.WorstMinTicks()),
              ratio(tuner.FirstConfigTicks()));
    }
  }

  static void EnsureAligned(const MatPtr& A, const size_t vector_bytes) {
    // Ensure A rows are vector-aligned. Neither `Stride` nor `IsPacked` are
    // reliable: the latter returns true for single rows, and the former may
    // match `Cols` if the width matches the padding.
    // Note that B is packed in matmul_test, but otherwise generally padded.
    HWY_ASSERT(hwy::IsAligned(A.RowBytes(0), vector_bytes));
    if (A.Rows() > 1) {
      HWY_ASSERT(hwy::IsAligned(A.RowBytes(1), vector_bytes));
    }
  }

  static size_t Worker(const MMArgs& args) {
    return args.options.cluster_idx *
           args.env->ctx.pools.MaxWorkersPerCluster();
  }

  // Returns 2D subrange whose top-left is `r, c` and width is `cols`.
  template <typename T>
  static StridedView<T> View(const MatPtrT<T>& AB, size_t r, size_t c,
                             size_t cols) {
    HWY_DASSERT(c < AB.Cols());
    HWY_DASSERT(cols <= AB.Cols() - c);
    return StridedView<T>(const_cast<T*>(AB.Row(r)) + c, cols, AB.Stride());
  }

  template <class Func>
  static void DispatchParallelism(ParallelismStrategy parallelism,
                                  const Func& func) {
    switch (parallelism) {
      case ParallelismStrategy::kHierarchical:
        return func(MMParallelHierarchical());
      case ParallelismStrategy::kNone:
        return func(MMParallelNone());
      case ParallelismStrategy::kWithinCluster:
        return func(MMParallelWithinCluster());
      default:
        HWY_UNREACHABLE;
    }
  }

  // Decompresses all `M x K` from `A` into padded BF16 `A_view`.
  static HWY_NOINLINE void DoDecompressA(const MatPtrT<float>& A,
                                         const StridedViewBF A_view,
                                         MMParA par_a, const MMArgs& args) {
    const IndexRange all_M(0, A.Rows());
    const IndexRange all_K(0, A.Cols());
    HWY_DASSERT(all_K.Num() == A_view.Cols());

    const hn::ScalableTag<BF16> dbf;
    const size_t NBF = hn::Lanes(dbf);

    static const auto zone = args.env->ctx.profiler.AddZone("MM.DecompressA");

    const auto do_range =
        [&](const IndexRange& range_M, const IndexRange& range_K, size_t worker)
            HWY_ATTR {
              MMZone mm_zone;
              mm_zone.MaybeEnter(worker, zone, args);

              const size_t col0 = range_K.begin();
              const size_t cols = range_K.Num();
              // Must be a vector multiple, or the last range before row
              // padding, otherwise `DecompressAndZeroPad` overwrites neighbors.
              HWY_DASSERT(cols % NBF == 0 || range_K.end() == A.Cols());
              for (size_t row_a : range_M) {
                const PackedSpan<const float> from =
                    MakeSpan(A.Row(row_a) + col0, cols);
                BF16* HWY_RESTRICT to = A_view.Row(row_a) + col0;
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
        do_range(all_M, all_K, MMImpl::Worker(args));
        break;

      case MMParA::kK1:
      case MMParA::kK2:
      case MMParA::kK4: {
        const size_t inner_tasks = static_cast<size_t>(par_a);
        // At least one vector, otherwise DecompressAndZeroPad will add
        // padding, which might overwrite neighboring tasks. Also a whole cache
        // line to avoid false sharing.
        const size_t multiple_K = HWY_MAX(NBF, args.line_bytes / sizeof(BF16));

        DispatchParallelism(
            args.options.parallelism, [&](const auto& parallel) {
              parallel.ForN(args.env->ctx, all_K, multiple_K, inner_tasks,
                            args.options.cluster_idx,
                            [&](const IndexRange& range_K, size_t worker) {
                              do_range(all_M, range_K, worker);
                            });
            });
        break;
      }
      case MMParA::kM:
        DispatchParallelism(
            args.options.parallelism, [&](const auto& parallel) {
              parallel.ForRangeMC(
                  args.env->ctx, all_M, args.options.cluster_idx,
                  [&](size_t row_a, size_t worker) {
                    do_range(IndexRange(row_a, row_a + 1), all_K, worker);
                  });
            });
        break;
    }
  }

  // Autotuning wrapper for `DoDecompressA`.
  static HWY_INLINE void DecompressA(const MatPtrT<float>& A,
                                     const StridedViewBF A_view,
                                     const MMArgs& args) {
    MMAutoTune<MMParA>& autotune = args.per_key->autotune_par_a;

    if (HWY_LIKELY(autotune.Best())) {
      return DoDecompressA(A, A_view, *autotune.Best(), args);
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
    DoDecompressA(A, A_view, par_a, args);
    const uint64_t t1 =
        args.env->have_timer_stop ? hwy::timer::Stop() : hwy::timer::Start();
    const uint64_t min_elapsed = autotune.NotifyTicks(t1 - t0);
    if (HWY_UNLIKELY(args.env->print_measurement && autotune.ShouldPrint())) {
      fprintf(stderr, "%s,%7.3f\n", StringFromParA(par_a),
              static_cast<double>(min_elapsed) /
                  hwy::platform::InvariantTicksPerSecond() * 1E6);
    }
  }

  template <typename TA>
  static HWY_INLINE StridedViewBF MaybeDecompressA(const MatPtrT<TA>& A,
                                                   const MMArgs& args) {
    if constexpr (IsBF16<TA>()) {
      // We can use a view, regardless of columns/padding, because `LoopKC`
      // supports non-vector multiples.
      return View(A, 0, 0, A.Cols());
    } else {
      // Always decompress. To reduce code size/compile time, we no longer
      // support a separate F32 kernel; most A are already BF16. We also only
      // have a single MMStorage.
      HWY_ASSERT(args.options.cluster_idx == 0);
      const StridedViewBF A_view = args.env->storage.A(A.Extents());
      DecompressA(A, A_view, args);
      return A_view;
    }
  }
};

// Contains several variants of the outer M/N/K loops, and calls `A2C0` which
// loops over the inner KC and MC. Member variables avoid long argument lists.
class MMState {
 public:
  MMState(size_t M, size_t K, size_t N, const MMArgs& args,
          const MMConfig& config)
      : args_(args),
        range_n_(0, N),
        mr_(config.MR()),
        ranges_mc_(config.RangesOfMC(M)),
        ranges_kc_(config.RangesOfKC(K)),
        ranges_nc_(config.RangesOfNC(N)),
        order_(config.Order()),
        inner_tasks_(config.InnerTasks()) {}

  // Called from `MatMul` from two places: either with the next autotune config,
  // or with the best config.
  template <typename TB, typename TC>
  HWY_NOINLINE void DispatchParallelism(const StridedViewBF A,
                                        const MatPtrT<TB>& B,
                                        RowPtrs<TC> C_rows) const {
    static const auto zone =
        args_.env->ctx.profiler.AddZone("MM.DispatchParallelism");
    PROFILER_ZONE3(args_.env->ctx.profiler, MMImpl::Worker(args_), zone);

    MMImpl::DispatchParallelism(
        args_.options.parallelism,
        [&](const auto& parallel) { DispatchOrder(parallel, A, B, C_rows); });
  }

 private:
  // Compute size of per-worker storage for `kNR` row ranges of B. Stack
  // allocation avoids passing a worker index.
  static constexpr size_t B_stride_max_ =
      kMaxKC + 2 * CacheInfo::MaxLineBytes() / sizeof(BF16);
  static constexpr size_t B_storage_max_ = kNR * B_stride_max_;

  // Granularity of `ForN`. B rows produce C columns, so we
  // want a multiple of the line size to prevent false sharing.
  size_t MultipleN(size_t sizeof_TC) const {
    return HWY_MAX(kNR, args_.line_bytes / sizeof_TC);
  }

  // B is decompressed several call layers lower, but not all member functions
  // depend on `TB`, so pass it as an argument instead of templating the class.
  template <typename TB, typename TC, class ParallelT>
  HWY_NOINLINE void DispatchOrder(const ParallelT& parallel_policy,
                                  const StridedViewBF A, const MatPtrT<TB>& B,
                                  RowPtrs<TC> C_rows) const {
    switch (order_) {
      case MMOrder::kNT:
        return DoNT(parallel_policy, A, B, C_rows);
      case MMOrder::kNT_K:
        return DoNT_K(parallel_policy, A, B, C_rows);
      case MMOrder::kNT_MT:
        return DoNT_MT(parallel_policy, A, B, C_rows);
      case MMOrder::kNT_MT_K:
        return DoNT_MT_K(parallel_policy, A, B, C_rows);
      default:
        HWY_UNREACHABLE;
    }
  }

  // Single M and K ranges, parallel N. Fills all of C directly.
  template <typename TB, typename TC, class ParallelT>
  HWY_INLINE void DoNT(ParallelT parallel, const StridedViewBF A,
                       const MatPtrT<TB>& B, RowPtrs<TC> C_rows) const {
    static const auto zone = args_.env->ctx.profiler.AddZone("MM.NT");
    HWY_DASSERT(ranges_mc_.NumTasks() == 1);
    HWY_DASSERT(ranges_kc_.NumTasks() == 1);
    const IndexRange& range_M = ranges_mc_.Range(0);
    const IndexRange& range_K = ranges_kc_.Range(0);
    const size_t K = range_K.Num();
    const StridedViewBF A_view = A.View(range_M.begin(), 0, K);
    const size_t B_stride =
        Stride(MatPadding::kOdd, K, sizeof(BF16), args_.line_bytes);

    // Similar to `loop_nc` below, but here we hoisted `A_view`.
    parallel.ForN(
        args_.env->ctx, range_n_, MultipleN(sizeof(TC)), inner_tasks_,
        args_.options.cluster_idx,
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
  }

  // Single M range, parallel N, sequential K. Sets C, then accumulates.
  template <typename TB, typename TC, class ParallelT>
  HWY_INLINE void DoNT_K(ParallelT parallel, const StridedViewBF A,
                         const MatPtrT<TB>& B, RowPtrs<TC> C_rows) const {
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
      const StridedViewBF A_view =
          A.View(range_mc.begin(), range_kc.begin(), kc);
      const StridedViewBF B_storage_view(
          B_storage, kc,
          Stride(MatPadding::kOdd, kc, sizeof(BF16), args_.line_bytes));

      for (size_t row_b = range_nc.begin(); row_b < range_nc.end();
           row_b += kNR) {
        StridedViewBF B_view = DecompressB(B, row_b, range_kc, B_storage_view);
        MMKernel::A2C0(A_view, B_view, mr_, range_mc, row_b, kc, out_tag, args_,
                       C_rows);
      }
    };

    parallel.ForN(
        args_.env->ctx, range_n_, MultipleN(sizeof(TC)), inner_tasks_,
        args_.options.cluster_idx,
        [&](const IndexRange& range_nc, size_t worker) HWY_ATTR {
          MMZone mm_zone;
          mm_zone.MaybeEnter(worker, zone, args_);

          HWY_ALIGN BF16 B_storage[B_storage_max_];  // TLS

          // Peel off the first iteration of the kc loop: avoid
          // zero-initializing `partial` by writing into it.
          ranges_kc_.VisitFirst([&](const IndexRange& range_kc) {
            loop_nc(B_storage, range_kc, range_nc, MMSetC());
          });
          ranges_kc_.VisitRemaining([&](const IndexRange& range_kc) {
            loop_nc(B_storage, range_kc, range_nc, MMAddC());
          });
        });
  }

  // Parallel loops over mc/nc blocks of M/range_n, single K.
  // Fills `mc x nc` sections of C directly, in parallel.
  template <typename TB, typename TC, class ParallelT>
  HWY_INLINE void DoNT_MT(ParallelT parallel, const StridedViewBF A,
                          const MatPtrT<TB>& B, RowPtrs<TC> C_rows) const {
    static const auto zone = args_.env->ctx.profiler.AddZone("MM.NT_MT");
    HWY_DASSERT(ranges_kc_.NumTasks() == 1);
    const IndexRange& range_K = ranges_kc_.Range(0);
    const size_t K = range_K.Num();
    const size_t B_stride =
        Stride(MatPadding::kOdd, K, sizeof(BF16), args_.line_bytes);

    // Similar to `loop_nc` below except for the profiler zone and `MMSetC`.
    parallel.ForRangesMC_NC(
        args_.env->ctx, ranges_mc_, ranges_nc_, args_.options.cluster_idx,
        [&](const IndexRange& range_mc, const IndexRange& range_nc,
            size_t worker) HWY_ATTR {
          MMZone mm_zone;
          mm_zone.MaybeEnter(worker, zone, args_);

          const StridedViewBF A_view = A.View(range_mc.begin(), 0, K);
          HWY_ALIGN BF16 B_storage[B_storage_max_];  // TLS
          const StridedViewBF B_storage_view(B_storage, K, B_stride);

          for (size_t row_b = range_nc.begin(); row_b < range_nc.end();
               row_b += kNR) {
            const StridedViewBF B_view =
                DecompressB(B, row_b, range_K, B_storage_view);
            MMKernel::A2C0(A_view, B_view, mr_, range_mc, row_b, K, MMSetC(),
                           args_, C_rows);
          }
        });
  }

  // Parallel loops over mc/nc blocks of M/range_np, sequential K.
  // Accumulates into `mc x nc` sections of `C`.
  template <typename TB, typename TC, class ParallelT>
  HWY_INLINE void DoNT_MT_K(ParallelT parallel, const StridedViewBF A,
                            const MatPtrT<TB>& B, RowPtrs<TC> C_rows) const {
    static const auto zone = args_.env->ctx.profiler.AddZone("MM.NT_MT_K");
    const size_t kc_max = ranges_kc_.TaskSize();
    HWY_DASSERT(kc_max <= kMaxKC);
    const size_t B_stride =
        Stride(MatPadding::kOdd, kc_max, sizeof(BF16), args_.line_bytes);
    // Sequential loop over NC/MC/KC, for when the M/N loops are
    // already parallel. This is B3A2C0 in MOMMS terminology: we read
    // `mc x kc` of A, `nc x kc` of B, update `mc x nc` of `C`.
    const auto loop_nc = [&](const StridedViewBF B_storage_view,
                             const IndexRange& range_mc,
                             const IndexRange& range_kc,
                             const IndexRange& range_nc,
                             auto out_tag) HWY_ATTR {
      const size_t kc = range_kc.Num();
      const StridedViewBF A_view =
          A.View(range_mc.begin(), range_kc.begin(), kc);

      for (size_t row_b = range_nc.begin(); row_b < range_nc.end();
           row_b += kNR) {
        StridedViewBF B_view = DecompressB(B, row_b, range_kc, B_storage_view);
        MMKernel::A2C0(A_view, B_view, mr_, range_mc, row_b, kc, out_tag, args_,
                       C_rows);
      }
    };  // loop_nc
    parallel.ForRangesMC_NC(
        args_.env->ctx, ranges_mc_, ranges_nc_, args_.options.cluster_idx,
        [&](const IndexRange& range_mc, const IndexRange& range_nc,
            size_t worker) HWY_ATTR {
          MMZone mm_zone;
          mm_zone.MaybeEnter(worker, zone, args_);

          HWY_ALIGN BF16 B_storage[B_storage_max_];  // TLS
          const StridedViewBF B_storage_view(B_storage, kc_max, B_stride);

          // Peel off the first iteration of the kc loop: avoid
          // zero-initializing `C` by writing into it.
          ranges_kc_.VisitFirst([&](const IndexRange& range_kc) {
            loop_nc(B_storage_view, range_mc, range_kc, range_nc, MMSetC());
          });
          ranges_kc_.VisitRemaining([&](const IndexRange& range_kc) {
            loop_nc(B_storage_view, range_mc, range_kc, range_nc, MMAddC());
          });
        });
  }

  // Decompresses `kNR x kc` from `B[row_b, range_kc.begin()]` to row 0,
  // col 0 of `B_view`. Decompressing SFP is relatively cheap on `AVX3_DL`
  // thanks to its large table lookups, and less so on other targets.
  template <typename TB>
  HWY_INLINE StridedViewBF DecompressB(const MatPtrT<TB>& B, const size_t row_b,
                                       const IndexRange& range_kc,
                                       const StridedViewBF B_view) const {
    const hn::ScalableTag<BF16> dbf;
    HWY_LANES_CONSTEXPR const size_t NBF = hn::Lanes(dbf);

    // Neither A nor B require padding because `LoopKC` handles remainders.
    if constexpr (hwy::IsSame<TB, BF16>()) {
      return MMImpl::View(B, row_b, range_kc.begin(), range_kc.Num());
    }

    const PackedSpan<const TB> B_span = B.PaddedSpan();

    const size_t kc = range_kc.Num();
    const size_t col0 = range_kc.begin();

    for (size_t r = 0; r < kNR; ++r) {
      const size_t packed_ofs = (row_b + r) * B.Stride() + col0;
      BF16* HWY_RESTRICT to = B_view.Row(r);
      DecompressAndZeroPad(dbf, B_span, packed_ofs, to, kc);
      // Verify that we zero-padded.
      if constexpr (HWY_IS_DEBUG_BUILD) {
        for (size_t i = kc; i < hwy::RoundUpTo(kc, NBF); ++i) {
          HWY_DASSERT(hwy::ConvertScalarTo<float>(to[i]) == 0.0f);
        }
      }
    }
    return B_view;
  }

  const MMArgs args_;  // copy for locality

  const IndexRange range_n_;
  // From MMConfig:
  const size_t mr_;
  const IndexRangePartition ranges_mc_;
  const IndexRangePartition ranges_kc_;
  const IndexRangePartition ranges_nc_;
  const MMOrder order_;
  const size_t inner_tasks_;
};  // MMState

// Computes the matrix product `A * B * scale [+ add]` and stores it in `C`.
//
// `A` is a row-major matrix with `M` rows and `B` is transposed. The latter's
// `K = B.Cols()`, which must match `A.Cols()`, is the number
// of rows in the original B. `N = C.Cols()` must be a multiple of 4. There
// are no other restrictions on shape, though performance is better when `M % 4
// == 0` or `M <= 4`.
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
                              MatPtrT<TC>& C, MMOptions options = MMOptions()) {
  static const auto zone = env.ctx.profiler.AddZone("MM.MatMul");
  PROFILER_ZONE3(env.ctx.profiler,
                 options.cluster_idx * env.ctx.pools.MaxWorkersPerCluster(),
                 zone);

  HWY_DASSERT(options.cluster_idx < env.row_ptrs.size());
  RowPtrs<TC> C_rows =
      GetOrSetTempRowPtrs(C, env.row_ptrs[options.cluster_idx]);

  const size_t M = A.Rows();
  const size_t K = A.Cols();
  const size_t N = B.Rows();

  const CacheInfo& cache = env.ctx.cache_info;
  MMPerKey& per_key = MMImpl::FindOrAddPerKey(
      M, K, N, cache.VectorBytes(), env.per_cluster[options.cluster_idx]);
  MMAutoTune<MMConfig>& tuner = per_key.autotune;

  const MMArgs args(env, per_key, static_cast<double>(A.Scale()) * B.Scale(),
                    add, options);
  if (HWY_LIKELY(tuner.Best())) {
    const MMState state(M, K, N, args, *tuner.Best());
    const StridedViewBF A_view = MMImpl::MaybeDecompressA(A, args);
    state.DispatchParallelism(A_view, B, C_rows);
    return &per_key;
  }

  // Autotuning, first call: enumerate all feasible configs.
  if (HWY_UNLIKELY(!tuner.HasCandidates())) {
    // Ensure matrix dimensions match each other (off the hot path).
    HWY_ASSERT(K == B.Cols());
    HWY_ASSERT(M <= kMaxBatchSize);
    HWY_ASSERT(K <= MMStorage::kMaxK);
    HWY_ASSERT(N % kNR == 0);
    MMImpl::EnsureAligned(A, cache.VectorBytes());
    tuner.SetCandidates(
        MMCandidates(cache, M, K, N, sizeof(TC), env.print_config));
  }

  // (Also auto-tunes, hence outside the timed section to prevent interference.)
  const StridedViewBF A_view = MMImpl::MaybeDecompressA(A, args);

  const MMConfig& cfg = tuner.NextConfig();
  const uint64_t t0 = hwy::timer::Start();
  MMState state(M, K, N, args, cfg);
  state.DispatchParallelism(A_view, B, C_rows);
  MMImpl::NotifyAutotuneResult(M, K, N, t0, cfg, env, tuner);

  return &per_key;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
