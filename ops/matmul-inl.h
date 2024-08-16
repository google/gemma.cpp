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

#include "compression/compress.h"  // IWYU pragma: keep, b/conditionally used
#include "ops/matmul.h"  // IWYU pragma: export

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
#include "hwy/contrib/math/math-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// The MatMul result C[r,c] is Dot(A.Row(r), B.Col(c)). To reduce the number of
// loads, we reuse the same A row for several B columns, which are also loaded
// once for several rows of C. Thus we produce one 'tile' of C at a time of
// dimensions `kRegRows` x `kRegCols`. The Reg naming is because these are
// limited by the number of registers: 32 for NEON/SVE/AVX-512. `kRegCols` == 4
// enables the `StoreInterleaved4` transpose in `AddHorizontalSums`. We assume
// and verify that `C.cols % kRegCols == 0`.
constexpr size_t kRegCols = 4;

// Choosing `kRegRows == kRegCols` minimizes the ratio of loads to FMA, because
// we load `kRegCols + kRegRows` vectors per `kRegRows * kRegCols` element tile.
// In general, `batch_size` (C rows) is not a multiple of `kRegRows`. Thus
// functions that load or store a tile are parameterized on `kNumRows`, which is
// generally `kRegRows`, but `batch_size % kRegRows` on the last row (if != 0).
constexpr size_t kRegRows = kRegCols;

// NEON_BF16/SVE/AVX3_ZEN4 have instructions for bf16 * bf16 + f32 which are
// more efficient than f32 * f32 + f32 because they process twice as many lanes
// at a time. Any combination of A and B can be bf16: activations may already be
// bf16, and weights can be decompressed to bf16.
//
// The corresponding op is `ReordenWidenMulAccumulate`, and it is always
// supported, but only useful if it returns a single vector of pairwise sums
// `a[0] * b[0] + a[1] * b[1]`. On other targets, `ReordenWidenMulAccumulate`
// insteads return `a[1] * b[1]` in its `sum1` output. We cannot afford to keep
// a `sum1` for each of the `kRegRows * kRegCols` C vectors, and it would be
// expensive to add each `sum0` and `sum1`, hence we only 'decompress' A and B
// to bf16 if the native op is available. This will actually demote f32
// activations to bf16. Otherwise, we decompress to f32 and use normal FMA.
using MulT = hwy::If<HWY_NATIVE_DOT_BF16, BF16, float>;

// Loads two vectors at a time with element type MulT from a row of transposed
// B. Called in a loop over col_ab. No bounds checking because `kRow` is
// actually from B columns, which we checked is a multiple of `kRegCols`.
template <size_t kRow, typename MatTB>
class BRow {
  static_assert(kRow < kRegRows);  // which unrolled instance we are
  using TraitsB = CompressTraits<MatTB>;

 public:
  BRow(const Mat<const MatTB>& B, size_t row_b)
      : B_(B.ptr), B_ofs_(B.Row(row_b + kRow)) {}

  template <class DM, class VM = hn::Vec<DM>>
  HWY_INLINE void Load2(DM d, size_t col_ab, VM& b0, VM& b1) const {
    static_assert(hwy::IsSame<hn::TFromD<DM>, MulT>());
    TraitsB::Decompress2(d, B_, B_ofs_ + col_ab, b0, b1);
  }

 private:
  const MatTB* HWY_RESTRICT B_;
  const size_t B_ofs_;
};

// Loads *two* row vectors from A via `Decompress2`, multiplies element-wise
// with `kRegRows` x 2 row vectors from transposed B, and adds them to
// `kRegRows` x `kRegCols` C vectors. The lanes of `C[r,c]` are thus a subset of
// the terms of the dot products that make up the MatMul result at `r,c`.
// No-op for the bottom-most tile where kRow >= kNumRows.
//
// This approach is atypical because it requires a horizontal sum, for which we
// introduce a fast and new(?) vector-length agnostic 'transpose', see
// `AddHorizontalSums`. Most MatMul instead broadcast one element from A and
// multiply with one element from N columns in B to obtain N columns of C.
// This is a poor fit for our setting:
// - `CompressTraits` decompresses two vectors at a time;
// - B is column-major, so unit-stride SIMD loads return a column, not values
//   from different columns, i.e. a row.
// Both could be fixed in a packing stage, which is not implemented yet, and
// might not be necessary otherwise. However, `ReorderWidenMulAccumulate` is
// important for bf16 performance and incompatible with the conventional
// approach, because its pairwise adds would add together unrelated terms.
// By contrast, pairwise adds are fine when our C lanes are the terms of a
// single dot product, which can be reordered or pre-reduced.
template <size_t kRow, typename MatTA>
class ALoadAccumulate {
  static_assert(kRow < kRegRows);  // which unrolled instance we are
  using TraitsA = CompressTraits<MatTA>;

 public:
  ALoadAccumulate(const Mat<const MatTA>& A, size_t row_ac)
      : A_(A.ptr), A_ofs_(A.Row(row_ac + kRow)) {}

  // First iteration, col_ab = 0: initialize C0..3 instead of updating them.
  template <size_t kNumRows, class DM, class VM = hn::Vec<DM>, HWY_IF_F32_D(DM)>
  HWY_INLINE void First(DM dm,  //
                        const VM b00, const VM b01, const VM b10, const VM b11,
                        const VM b20, const VM b21, const VM b30, const VM b31,
                        VM& C0, VM& C1, VM& C2, VM& C3) const {
    static_assert(kNumRows <= kRegRows);  // How many rows actually present
    if constexpr (kRow < kNumRows) {
      VM a0, a1;
      TraitsA::Decompress2(dm, A_, A_ofs_, a0, a1);

      static_assert(kRegCols == 4);
      C0 = hn::Mul(a0, b00);
      C1 = hn::Mul(a0, b10);
      C2 = hn::Mul(a0, b20);
      C3 = hn::Mul(a0, b30);
      C0 = hn::MulAdd(a1, b01, C0);
      C1 = hn::MulAdd(a1, b11, C1);
      C2 = hn::MulAdd(a1, b21, C2);
      C3 = hn::MulAdd(a1, b31, C3);
    }
  }

  // Same as above, only called if MulT == BF16.
  template <size_t kNumRows, class DM, class VM = hn::Vec<DM>,
            HWY_IF_BF16_D(DM), class DF = hn::Repartition<float, DM>,
            class VF = hn::Vec<DF>>
  HWY_INLINE void First(DM dm,  //
                        const VM b00, const VM b01, const VM b10, const VM b11,
                        const VM b20, const VM b21, const VM b30, const VM b31,
                        VF& C0, VF& C1, VF& C2, VF& C3) const {
    static_assert(kNumRows <= kRegRows);  // How many rows actually present
    if constexpr (kRow < kNumRows) {
      VM a0, a1;
      TraitsA::Decompress2(dm, A_, A_ofs_, a0, a1);

      const DF df;
      VF unused_sum1 = hn::Zero(df);

      static_assert(kRegCols == 4);
      C0 = hn::WidenMulPairwiseAdd(df, a0, b00);
      C1 = hn::WidenMulPairwiseAdd(df, a0, b10);
      C2 = hn::WidenMulPairwiseAdd(df, a0, b20);
      C3 = hn::WidenMulPairwiseAdd(df, a0, b30);
      C0 = hn::ReorderWidenMulAccumulate(df, a1, b01, C0, unused_sum1);
      C1 = hn::ReorderWidenMulAccumulate(df, a1, b11, C1, unused_sum1);
      C2 = hn::ReorderWidenMulAccumulate(df, a1, b21, C2, unused_sum1);
      C3 = hn::ReorderWidenMulAccumulate(df, a1, b31, C3, unused_sum1);

      // Ensure sum1 was indeed unused.
      HWY_DASSERT(hn::AllTrue(df, hn::Eq(unused_sum1, hn::Zero(df))));
    }
  }

  // Non-first iteration: accumulate into C0..3.
  template <size_t kNumRows, class DM, class VM = hn::Vec<DM>, HWY_IF_F32_D(DM)>
  HWY_INLINE void Next(DM dm, size_t col_ab, const VM b00, const VM b01,
                       const VM b10, const VM b11, const VM b20, const VM b21,
                       const VM b30, const VM b31, VM& C0, VM& C1, VM& C2,
                       VM& C3) const {
    static_assert(kNumRows <= kRegRows);       // How many rows actually present
    HWY_DASSERT(col_ab >= 2 * hn::Lanes(dm));  // Should not be first iteration.
    if constexpr (kRow < kNumRows) {
      VM a0, a1;
      TraitsA::Decompress2(dm, A_, A_ofs_ + col_ab, a0, a1);

      static_assert(kRegCols == 4);
      C0 = hn::MulAdd(a0, b00, C0);
      C1 = hn::MulAdd(a0, b10, C1);
      C2 = hn::MulAdd(a0, b20, C2);
      C3 = hn::MulAdd(a0, b30, C3);
      C0 = hn::MulAdd(a1, b01, C0);
      C1 = hn::MulAdd(a1, b11, C1);
      C2 = hn::MulAdd(a1, b21, C2);
      C3 = hn::MulAdd(a1, b31, C3);
    }
  }

  // Same as above, only called if MulT == BF16.
  template <size_t kNumRows, class DM, class VM = hn::Vec<DM>,
            HWY_IF_BF16_D(DM), class DF = hn::Repartition<float, DM>,
            class VF = hn::Vec<DF>>
  HWY_INLINE void Next(DM dm, size_t col_ab, const VM b00, const VM b01,
                       const VM b10, const VM b11, const VM b20, const VM b21,
                       const VM b30, const VM b31, VF& C0, VF& C1, VF& C2,
                       VF& C3) const {
    static_assert(kNumRows <= kRegRows);       // How many rows actually present
    HWY_DASSERT(col_ab >= 2 * hn::Lanes(dm));  // Should not be first iteration.
    if constexpr (kRow < kNumRows) {
      VM a0, a1;
      TraitsA::Decompress2(dm, A_, A_ofs_ + col_ab, a0, a1);

      const DF df;
      hn::Vec<DF> unused_sum1 = hn::Zero(df);

      static_assert(kRegCols == 4);
      C0 = hn::ReorderWidenMulAccumulate(df, a0, b00, C0, unused_sum1);
      C1 = hn::ReorderWidenMulAccumulate(df, a0, b10, C1, unused_sum1);
      C2 = hn::ReorderWidenMulAccumulate(df, a0, b20, C2, unused_sum1);
      C3 = hn::ReorderWidenMulAccumulate(df, a0, b30, C3, unused_sum1);
      C0 = hn::ReorderWidenMulAccumulate(df, a1, b01, C0, unused_sum1);
      C1 = hn::ReorderWidenMulAccumulate(df, a1, b11, C1, unused_sum1);
      C2 = hn::ReorderWidenMulAccumulate(df, a1, b21, C2, unused_sum1);
      C3 = hn::ReorderWidenMulAccumulate(df, a1, b31, C3, unused_sum1);

      // Ensure sum1 was indeed unused.
      HWY_DASSERT(hn::AllTrue(df, hn::Eq(unused_sum1, hn::Zero(df))));
    }
  }

 private:
  const MatTA* HWY_RESTRICT A_;
  const size_t A_ofs_;
};  // ALoadAccumulate

// Sets a `kRegRows` x `kRegCols` tile of C to `add[add_ofs + c]` if kAdd,
// otherwise 0.
// `add` has no scale and is a row vector with A.cols entries if `kAdd`,
// otherwise nullptr. In the latter case, adding `add_ofs` to it would be UB,
// hence we pass it as a separate argument.
template <size_t kNumRows, bool kAdd>
HWY_INLINE void InitC(const float* HWY_RESTRICT add, size_t add_ofs,
                      float* HWY_RESTRICT pos_c, size_t stride_c) {
  const hn::FixedTag<float, kRegCols> d4;
  for (size_t r = 0; r < HWY_MIN(kNumRows, kRegRows); ++r) {
    if constexpr (kAdd) {
      hn::StoreU(hn::LoadU(d4, add + add_ofs), d4, pos_c + r * stride_c);
    } else {
      hn::StoreU(hn::Zero(d4), d4, pos_c + r * stride_c);
    }
  }
}

// Accumulates into a tile of C.
template <size_t kNumRows>
class AddHorizontalSums {
  // These helper functions hoist if() out of the main code below. They have no
  // effect if kRow >= kNumRows.
  template <size_t kRow, class DF, class VF = hn::Vec<DF>>
  static void MaybeStoreInterleaved4(DF df, size_t N, VF Cr0, VF Cr1, VF Cr2,
                                     VF Cr3, float* HWY_RESTRICT buf) {
    if constexpr (kRow < kNumRows) {
      hn::StoreInterleaved4(Cr0, Cr1, Cr2, Cr3, df, buf + 4 * kRow * N);
    }
  }

  // Note: N is the number of lanes in the StoreInterleaved4 vectors, not V4.
  template <size_t kRow, class D4, class V4 = hn::Vec<D4>>
  static V4 MaybeLoad(D4 df, size_t N, const float* HWY_RESTRICT buf) {
    if constexpr (kRow < kNumRows) {
      return hn::Load(df, buf + 4 * kRow * N);
    } else {
      return hn::Zero(df);
    }
  }

  template <size_t kRow, class D4, class V4 = hn::Vec<D4>>
  static V4 MaybeAdd(D4 df, size_t N, V4 sum, const float* HWY_RESTRICT buf) {
    if constexpr (kRow < kNumRows) {
      return hn::Add(sum, hn::Load(df, buf + 4 * kRow * N));
    } else {
      return sum;
    }
  }

  template <size_t kRow, class D4, class V4 = hn::Vec<D4>>
  static void MaybeMulAdd(D4 df, V4 sum, V4 scale, float* HWY_RESTRICT tile_c,
                          const size_t stride_c) {
    if constexpr (kRow < kNumRows) {
      const V4 prev_c = hn::LoadU(df, tile_c + kRow * stride_c);
      hn::StoreU(hn::MulAdd(sum, scale, prev_c), df, tile_c + kRow * stride_c);
    }
  }

 public:
  // Adds the contribution from `Crc` accumulators to the 4x4 tile of C whose
  // top left is `tile_c`, after multiplying by `scale`, which is the product of
  // the scales of A and B. C is always f32 to ensure sufficient precision.
  //
  // `Crc` are the 16 combinations of an A row vector indexed by `r`, times a
  // B column vector indexed by `c`. Their elements are thus a subset of the
  // terms of the dot product constituting the final `C[r, c]` result. Thus we
  // compute the horizontal sums of each `Crc`. The elements may be permuted
  // because we multiply bf16 via `ReorderWidenMulAccumulate`, but this does
  // not change their horizontal sum. `buf` is thread-local space for 16 `VF`.
  template <class DF, class VF = hn::Vec<DF>>
  HWY_INLINE void operator()(DF df, float scale,              //
                             VF C00, VF C01, VF C02, VF C03,  //
                             VF C10, VF C11, VF C12, VF C13,  //
                             VF C20, VF C21, VF C22, VF C23,  //
                             VF C30, VF C31, VF C32, VF C33,  //
                             float* HWY_RESTRICT buf,
                             float* HWY_RESTRICT tile_c,
                             size_t stride_c) const {
    const size_t N = hn::Lanes(df);
    // Horizontal reductions (`ReduceSum`) are rather expensive, entailing
    // log(N) operations for vectors of length N. Because kRegCols == 4, we can
    // instead use `StoreInterleaved4` for a vector length-agnostic 'transpose':
    // `buf[0, 4 * N)` holds C00[0], C01[0], C02[0], C03[0],
    // C00[1], C01[1], C02[1], C03[1] .. C00[N-1], C01[N-1], C02[N-1], C03[N-1].
    MaybeStoreInterleaved4<0>(df, N, C00, C01, C02, C03, buf);
    MaybeStoreInterleaved4<1>(df, N, C10, C11, C12, C13, buf);
    MaybeStoreInterleaved4<2>(df, N, C20, C21, C22, C23, buf);
    MaybeStoreInterleaved4<3>(df, N, C30, C31, C32, C33, buf);
    // Adding N consecutive V4 yields four horizontal sums of Cr0, Cr1, Cr2, Cr3
    // in the elements of one V4. We have four independent rows `r`, hence the
    // code is effectively unrolled, which increases throughput.
    const hn::FixedTag<float, 4> d4;
    using V4 = hn::Vec<decltype(d4)>;
    V4 sum0 = MaybeLoad<0>(d4, N, buf);
    V4 sum1 = MaybeLoad<1>(d4, N, buf);
    V4 sum2 = MaybeLoad<2>(d4, N, buf);
    V4 sum3 = MaybeLoad<3>(d4, N, buf);

    for (size_t i = 1; i < N; ++i) {
      sum0 = MaybeAdd<0>(d4, N, sum0, buf + 4 * i);
      sum1 = MaybeAdd<1>(d4, N, sum1, buf + 4 * i);
      sum2 = MaybeAdd<2>(d4, N, sum2, buf + 4 * i);
      sum3 = MaybeAdd<3>(d4, N, sum3, buf + 4 * i);
    }
    // Scale, then store to four elements per row of `tile_c`.
    const V4 vscale = hn::Set(d4, scale);
    MaybeMulAdd<0>(d4, sum0, vscale, tile_c, stride_c);
    MaybeMulAdd<1>(d4, sum1, vscale, tile_c, stride_c);
    MaybeMulAdd<2>(d4, sum2, vscale, tile_c, stride_c);
    MaybeMulAdd<3>(d4, sum3, vscale, tile_c, stride_c);
  }
};

// Streams a `kNumRows` high strip of `A` and the transposed `B`, then writes a
// *finished* tile of f32 `C` whose top left is (row_ac, row_b_col_c).
// TODO: loop over sections instead of full rows and accumulate into `tile_c`.
template <size_t kNumRows, bool kAdd, typename MatTA, typename MatTB>
HWY_INLINE void MatMulTile(const Mat<const MatTA>& A, const Mat<const MatTB>& B,
                           const size_t row_ac, const size_t row_b_col_c,
                           const float scale, const float* HWY_RESTRICT add,
                           float* HWY_RESTRICT buf, const Mat<float>& C) {
  // For 'decompressing' A and B into BF16 or float.
  const hn::ScalableTag<MulT> dm;
  using VM = hn::Vec<decltype(dm)>;
  const size_t NM = hn::Lanes(dm);

  static_assert(kRegRows == 4);
  const BRow<0, MatTB> b_row0(B, row_b_col_c);
  const BRow<1, MatTB> b_row1(B, row_b_col_c);
  const BRow<2, MatTB> b_row2(B, row_b_col_c);
  const BRow<3, MatTB> b_row3(B, row_b_col_c);

  const ALoadAccumulate<0, MatTA> a_row0(A, row_ac);
  const ALoadAccumulate<1, MatTA> a_row1(A, row_ac);
  const ALoadAccumulate<2, MatTA> a_row2(A, row_ac);
  const ALoadAccumulate<3, MatTA> a_row3(A, row_ac);

  const hn::Repartition<float, decltype(dm)> df;
  using VF = hn::Vec<decltype(df)>;
  VF C00, C01, C02, C03;
  VF C10, C11, C12, C13;
  VF C20, C21, C22, C23;
  VF C30, C31, C32, C33;

  {  // First iteration initializes the `Crc` vectors.
    VM b00, b01, b10, b11, b20, b21, b30, b31;
    b_row0.Load2(dm, /*col_ab=*/0, b00, b01);
    b_row1.Load2(dm, /*col_ab=*/0, b10, b11);
    b_row2.Load2(dm, /*col_ab=*/0, b20, b21);
    b_row3.Load2(dm, /*col_ab=*/0, b30, b31);

    a_row0.template First<kNumRows>(dm, b00, b01, b10, b11, b20, b21, b30, b31,
                                    C00, C01, C02, C03);
    a_row1.template First<kNumRows>(dm, b00, b01, b10, b11, b20, b21, b30, b31,
                                    C10, C11, C12, C13);
    a_row2.template First<kNumRows>(dm, b00, b01, b10, b11, b20, b21, b30, b31,
                                    C20, C21, C22, C23);
    a_row3.template First<kNumRows>(dm, b00, b01, b10, b11, b20, b21, b30, b31,
                                    C30, C31, C32, C33);
  }

  // `2 * NM` per iteration because `Load2` returns two vectors.
  HWY_UNROLL(1)
  for (size_t col_ab = 2 * NM; col_ab <= A.cols - 2 * NM; col_ab += 2 * NM) {
    VM b00, b01, b10, b11, b20, b21, b30, b31;
    b_row0.Load2(dm, col_ab, b00, b01);
    b_row1.Load2(dm, col_ab, b10, b11);
    b_row2.Load2(dm, col_ab, b20, b21);
    b_row3.Load2(dm, col_ab, b30, b31);

    a_row0.template Next<kNumRows>(dm, col_ab, b00, b01, b10, b11, b20, b21,
                                   b30, b31, C00, C01, C02, C03);
    a_row1.template Next<kNumRows>(dm, col_ab, b00, b01, b10, b11, b20, b21,
                                   b30, b31, C10, C11, C12, C13);
    a_row2.template Next<kNumRows>(dm, col_ab, b00, b01, b10, b11, b20, b21,
                                   b30, b31, C20, C21, C22, C23);
    a_row3.template Next<kNumRows>(dm, col_ab, b00, b01, b10, b11, b20, b21,
                                   b30, b31, C30, C31, C32, C33);
  }

  // TODO: hoist into outer loop.
  float* HWY_RESTRICT C_tile = C.ptr + C.Row(row_ac) + row_b_col_c;
  InitC<kNumRows, kAdd>(add, row_b_col_c, C_tile, C.stride);

  AddHorizontalSums<kNumRows>()(df, scale, C00, C01, C02, C03, C10, C11, C12,
                                C13, C20, C21, C22, C23, C30, C31, C32, C33,
                                buf, C_tile, C.stride);
}

// Computes the matrix product `A * B * scale [+ add]` and stores it in `C`.
//
// `A` is a row-major matrix of shape `(batch_size, A.cols)`.
// `B` is transposed; `B.cols`, which must match `A.cols`, denotes the number of
// rows in the original B, and `C.cols` the number of columns in the original B.
//
// `scale` allows expanding the smaller range of `SfpStream` to the original
// values. When `A` and/or `B` are from CompressedArray, `scale` should be the
// product of their `.scale()` values, otherwise 1.0f.
//
// If `kAdd` is true, the row-vector `add` is added to each row of `C`,
// otherwise `add` is ignored and can be nullptr. A scale for `add` is not
// supported, so make sure its scale is 1.
//
// `C` is a row-major matrix of size `(batch_size, C.cols)`.
//
// Updates 4x4 tiles of C in parallel using a work-stealing thread pool.
// Typically `batch_size` is 1..512, `A.cols` and `C.cols` are 3k or 24k.
// Must not be called concurrently with the same `env`.
template <bool kAdd, typename MatTA, typename MatTB>
HWY_NOINLINE void MatMul(const size_t batch_size, const Mat<const MatTA>& A,
                         const Mat<const MatTB>& B, const float scale,
                         const float* HWY_RESTRICT add, MatMulEnv& env,
                         const Mat<float>& C) {
  // PROFILER_ZONE("Matmul");
  HWY_DASSERT(A.NotEmpty() && B.NotEmpty() && C.NotEmpty());
  HWY_DASSERT(A.cols == B.cols);

  // Must be a multiple of two vectors because we Decompress2.
  HWY_DASSERT(A.cols % (2 * hn::Lanes(hn::ScalableTag<MulT>())) == 0);
  HWY_DASSERT(C.cols % kRegCols == 0);

  // We currently write C directly, which touches more memory than fits in L3.
  // TODO: add another level of loops to finish L3-sized pieces of C at a time.
  const size_t tilesY = hwy::DivCeil(batch_size, kRegRows);
  const size_t tilesX = C.cols / kRegCols;

  env.Pool().Run(
      0, tilesX * tilesY, [&](const uint64_t idx_tile, size_t thread) HWY_ATTR {
        // TODO: when using PerClusterPool, compute lp from outer and inner.
        float* HWY_RESTRICT buf = env.Buf(thread);
        const size_t tx = idx_tile % tilesX;
        const size_t ty = idx_tile / tilesX;
        const size_t row_ac = ty * kRegRows;
        const size_t row_b_col_c = tx * kRegCols;
        // How many rows of C are left to compute. If more than 4, this
        // tile still only computes 4 rows.
        const size_t num_rows = batch_size - row_ac;
        HWY_DASSERT(num_rows != 0);
        switch (num_rows) {
          case 1:
            MatMulTile<1, kAdd>(A, B, row_ac, row_b_col_c, scale, add, buf, C);
            break;
          case 2:
            MatMulTile<2, kAdd>(A, B, row_ac, row_b_col_c, scale, add, buf, C);
            break;
          case 3:
            MatMulTile<3, kAdd>(A, B, row_ac, row_b_col_c, scale, add, buf, C);
            break;
          default:
            MatMulTile<4, kAdd>(A, B, row_ac, row_b_col_c, scale, add, buf, C);
        }
      });
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
