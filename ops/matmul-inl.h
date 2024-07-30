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

// Include guard for non-SIMD code.
#ifndef THIRD_PARTY_GEMMA_CPP_OPS_MATMUL_INL_H_
#define THIRD_PARTY_GEMMA_CPP_OPS_MATMUL_INL_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"

#endif  // THIRD_PARTY_GEMMA_CPP_OPS_MATMUL_INL_H_

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_MATMUL_TOGGLE) == defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_MATMUL_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_MATMUL_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_MATMUL_TOGGLE
#endif

#include "compression/compress-inl.h"
#include "hwy/contrib/math/math-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// c## are partial sums of the products of A and B; their horizontal sums are
// the final matmul result, stored in C, which is always f32.
template <size_t kNumRows, class DF, class VF = hn::Vec<DF>>
HWY_INLINE void StoreHorizontalSums(DF df,                           //
                                    VF c00, VF c01, VF c02, VF c03,  //
                                    VF c10, VF c11, VF c12, VF c13,  //
                                    VF c20, VF c21, VF c22, VF c23,  //
                                    VF c30, VF c31, VF c32, VF c33,  //
                                    float scale, float* HWY_RESTRICT tile_c,
                                    size_t stride_c) {
  // We are computing the product of (4, 4N) * (4N, 4) = (4, 4) tiles.
  // Each entry of C[r,c] is a dot product of A.row and B.col, which reside in
  // the lanes of `c$r$c`, so we store their horizontal sum (ReduceSum). This is
  // expensive, but only a fraction of the A.cols/N FMAs.
  tile_c[stride_c * 0 + 0] = scale * hn::ReduceSum(df, c00);
  tile_c[stride_c * 0 + 1] = scale * hn::ReduceSum(df, c01);
  tile_c[stride_c * 0 + 2] = scale * hn::ReduceSum(df, c02);
  tile_c[stride_c * 0 + 3] = scale * hn::ReduceSum(df, c03);
  if (kNumRows == 1) return;

  tile_c[stride_c * 1 + 0] = scale * hn::ReduceSum(df, c10);
  tile_c[stride_c * 1 + 1] = scale * hn::ReduceSum(df, c11);
  tile_c[stride_c * 1 + 2] = scale * hn::ReduceSum(df, c12);
  tile_c[stride_c * 1 + 3] = scale * hn::ReduceSum(df, c13);
  if (kNumRows == 2) return;

  tile_c[stride_c * 2 + 0] = scale * hn::ReduceSum(df, c20);
  tile_c[stride_c * 2 + 1] = scale * hn::ReduceSum(df, c21);
  tile_c[stride_c * 2 + 2] = scale * hn::ReduceSum(df, c22);
  tile_c[stride_c * 2 + 3] = scale * hn::ReduceSum(df, c23);
  if (kNumRows == 3) return;

  tile_c[stride_c * 3 + 0] = scale * hn::ReduceSum(df, c30);
  tile_c[stride_c * 3 + 1] = scale * hn::ReduceSum(df, c31);
  tile_c[stride_c * 3 + 2] = scale * hn::ReduceSum(df, c32);
  tile_c[stride_c * 3 + 3] = scale * hn::ReduceSum(df, c33);
}

// As above, but also adds `add[0..3]` to columns 0..3 of `tile_c`. `add` has no
// scale, and points to a 1D slice of the row vector.
template <size_t kNumRows, class DF, class VF = hn::Vec<DF>>
HWY_INLINE void StoreHorizontalSumsAdd(DF df,                           //
                                       VF c00, VF c01, VF c02, VF c03,  //
                                       VF c10, VF c11, VF c12, VF c13,  //
                                       VF c20, VF c21, VF c22, VF c23,  //
                                       VF c30, VF c31, VF c32, VF c33,
                                       const float scale,
                                       const float* HWY_RESTRICT add,
                                       float* HWY_RESTRICT tile_c,
                                       size_t stride_c) {
  // We are computing the product of (4, 4N) * (4N, 4) = (4, 4) tiles.
  // Each entry of C[r,c] is a dot product of A.row and B.col, which reside in
  // the lanes of `c$r$c`, so we store their horizontal sum (ReduceSum). This is
  // expensive, but only a fraction of the A.cols/N FMAs.
  const float add0 = add[0];
  // TODO: 4x4 transpose, then 128-bit vector FMA?
  tile_c[stride_c * 0 + 0] = scale * hn::ReduceSum(df, c00) + add0;
  const float add1 = add[1];
  tile_c[stride_c * 0 + 1] = scale * hn::ReduceSum(df, c01) + add1;
  const float add2 = add[2];
  tile_c[stride_c * 0 + 2] = scale * hn::ReduceSum(df, c02) + add2;
  const float add3 = add[3];
  tile_c[stride_c * 0 + 3] = scale * hn::ReduceSum(df, c03) + add3;
  if (kNumRows == 1) return;

  tile_c[stride_c * 1 + 0] = scale * hn::ReduceSum(df, c10) + add0;
  tile_c[stride_c * 1 + 1] = scale * hn::ReduceSum(df, c11) + add1;
  tile_c[stride_c * 1 + 2] = scale * hn::ReduceSum(df, c12) + add2;
  tile_c[stride_c * 1 + 3] = scale * hn::ReduceSum(df, c13) + add3;
  if (kNumRows == 2) return;

  tile_c[stride_c * 2 + 0] = scale * hn::ReduceSum(df, c20) + add0;
  tile_c[stride_c * 2 + 1] = scale * hn::ReduceSum(df, c21) + add1;
  tile_c[stride_c * 2 + 2] = scale * hn::ReduceSum(df, c22) + add2;
  tile_c[stride_c * 2 + 3] = scale * hn::ReduceSum(df, c23) + add3;
  if (kNumRows == 3) return;

  tile_c[stride_c * 3 + 0] = scale * hn::ReduceSum(df, c30) + add0;
  tile_c[stride_c * 3 + 1] = scale * hn::ReduceSum(df, c31) + add1;
  tile_c[stride_c * 3 + 2] = scale * hn::ReduceSum(df, c32) + add2;
  tile_c[stride_c * 3 + 3] = scale * hn::ReduceSum(df, c33) + add3;
}

// Wrapper around StoreHorizontalSums and StoreHorizontalSumsAdd to shorten call
// sites. If `!kAdd`, `add` is nullptr, so adding `add_offset` to it would be
// UB, hence we pass it as a separate argument.
template <bool kAdd, size_t kNumRows, class DF, class VF = hn::Vec<DF>>
HWY_INLINE void StoreHorizontalSumsMaybeAdd(
    DF df, VF c00, VF c01, VF c02, VF c03, VF c10, VF c11, VF c12, VF c13,
    VF c20, VF c21, VF c22, VF c23, VF c30, VF c31, VF c32, VF c33,
    const float scale, const float* HWY_RESTRICT add, size_t add_offset,
    float* HWY_RESTRICT tile_c, size_t stride_c) {
  if constexpr (kAdd) {
    StoreHorizontalSumsAdd<kNumRows>(df, c00, c01, c02, c03, c10, c11, c12, c13,
                                     c20, c21, c22, c23, c30, c31, c32, c33,
                                     scale, add + add_offset, tile_c, stride_c);
  } else {
    StoreHorizontalSums<kNumRows>(df, c00, c01, c02, c03, c10, c11, c12, c13,
                                  c20, c21, c22, c23, c30, c31, c32, c33,
                                  scale, tile_c, stride_c);
  }
}

// Wrapper to simplify call sites. T can be const or non-const.
template <typename T>
struct Mat {
  bool NotEmpty() const {
    return ptr != nullptr && cols != 0 && stride >= cols;
  }
  size_t Row(size_t r) const { return ofs + stride * r; }

  T* HWY_RESTRICT ptr;
  size_t cols;

  // elements between rows, which is typically the same as `cols`.
  size_t stride;

  // Offset to add to `ptr`; separate because T=NuqStream does not support
  // pointer arithmetic.
  size_t ofs;
};

template <typename T>
Mat<T> MakeMat(T* HWY_RESTRICT ptr, size_t cols, size_t stride,
               size_t ofs = 0) {
  return Mat<T>{.ptr = ptr, .cols = cols, .stride = stride, .ofs = ofs};
}

template <typename T>
Mat<T> MakeMat(T* HWY_RESTRICT ptr, size_t cols) {
  return MakeMat(ptr, cols, cols);
}

#undef GEMMA_NATIVE_BF16
#if HWY_IDE || (defined(HWY_NATIVE_REORDER_WIDEN_MUL_ACC_BF16) == \
                defined(HWY_TARGET_TOGGLE))
#define GEMMA_NATIVE_BF16 1
#else
#define GEMMA_NATIVE_BF16 0
#endif

#if GEMMA_NATIVE_BF16

// Specialization for f32 += bf16 * bf16 that avoids promoting to f32.
template <size_t kNumRows, bool kAdd>
HWY_INLINE void MatMulTile(const Mat<const hwy::bfloat16_t>& A,
                           const Mat<const hwy::bfloat16_t>& B,
                           const size_t row_a, const size_t row_b_col_c,
                           const float scale, const float* HWY_RESTRICT add,
                           const Mat<float>& C) {
  const hn::ScalableTag<float> df;
  using VF = hn::Vec<decltype(df)>;
  // ReorderWidenMulAccumulate does not use its sum1 arg and we can use full
  // bf16 vectors.
  const hn::Repartition<hwy::bfloat16_t, decltype(df)> d;
  const size_t N = Lanes(d);
  VF unused_sum1 = hn::Zero(df);
  VF c00 = hn::Zero(df);
  VF c01 = hn::Zero(df);
  VF c02 = hn::Zero(df);
  VF c03 = hn::Zero(df);

  VF c10 = hn::Zero(df);
  VF c11 = hn::Zero(df);
  VF c12 = hn::Zero(df);
  VF c13 = hn::Zero(df);

  VF c20 = hn::Zero(df);
  VF c21 = hn::Zero(df);
  VF c22 = hn::Zero(df);
  VF c23 = hn::Zero(df);

  VF c30 = hn::Zero(df);
  VF c31 = hn::Zero(df);
  VF c32 = hn::Zero(df);
  VF c33 = hn::Zero(df);

  const hwy::bfloat16_t* HWY_RESTRICT A_tile = A.ptr + A.Row(row_a);
  const hwy::bfloat16_t* HWY_RESTRICT B_tile = B.ptr + B.Row(row_b_col_c);

  // Loop over columns of A and columns of the transposed B, in steps of N.
  // Accumulates into the c## vectors.
  HWY_UNROLL(1)
  for (size_t col_ab = 0; col_ab < A.cols; col_ab += N) {
    using V = hn::Vec<decltype(d)>;
    const V b0 = hn::LoadU(d, B_tile + B.stride * 0 + col_ab);
    const V b1 = hn::LoadU(d, B_tile + B.stride * 1 + col_ab);
    const V b2 = hn::LoadU(d, B_tile + B.stride * 2 + col_ab);
    const V b3 = hn::LoadU(d, B_tile + B.stride * 3 + col_ab);

    const V a0 = hn::LoadU(d, A_tile + A.stride * 0 + col_ab);
    c00 = hn::ReorderWidenMulAccumulate(df, a0, b0, c00, unused_sum1);
    c01 = hn::ReorderWidenMulAccumulate(df, a0, b1, c01, unused_sum1);
    c02 = hn::ReorderWidenMulAccumulate(df, a0, b2, c02, unused_sum1);
    c03 = hn::ReorderWidenMulAccumulate(df, a0, b3, c03, unused_sum1);
    if constexpr (kNumRows == 1) continue;

    const V a1 = hn::LoadU(d, A_tile + A.stride * 1 + col_ab);
    c10 = hn::ReorderWidenMulAccumulate(df, a1, b0, c10, unused_sum1);
    c11 = hn::ReorderWidenMulAccumulate(df, a1, b1, c11, unused_sum1);
    c12 = hn::ReorderWidenMulAccumulate(df, a1, b2, c12, unused_sum1);
    c13 = hn::ReorderWidenMulAccumulate(df, a1, b3, c13, unused_sum1);
    if constexpr (kNumRows == 2) continue;

    const V a2 = hn::LoadU(d, A_tile + A.stride * 2 + col_ab);
    c20 = hn::ReorderWidenMulAccumulate(df, a2, b0, c20, unused_sum1);
    c21 = hn::ReorderWidenMulAccumulate(df, a2, b1, c21, unused_sum1);
    c22 = hn::ReorderWidenMulAccumulate(df, a2, b2, c22, unused_sum1);
    c23 = hn::ReorderWidenMulAccumulate(df, a2, b3, c23, unused_sum1);
    if constexpr (kNumRows == 3) continue;

    const V a3 = hn::LoadU(d, A_tile + A.stride * 3 + col_ab);
    c30 = hn::ReorderWidenMulAccumulate(df, a3, b0, c30, unused_sum1);
    c31 = hn::ReorderWidenMulAccumulate(df, a3, b1, c31, unused_sum1);
    c32 = hn::ReorderWidenMulAccumulate(df, a3, b2, c32, unused_sum1);
    c33 = hn::ReorderWidenMulAccumulate(df, a3, b3, c33, unused_sum1);
  }

  // Ensure sum1 was indeed unused.
  HWY_DASSERT(hn::AllTrue(df, hn::Eq(unused_sum1, hn::Zero(df))));

  float* HWY_RESTRICT C_tile = C.ptr + C.Row(row_a) + row_b_col_c;
  StoreHorizontalSumsMaybeAdd<kAdd, kNumRows>(
      df, c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31,
      c32, c33, scale, add, row_b_col_c, C_tile, C.stride);
}

#endif  // GEMMA_NATIVE_BF16

// The col_ab loop is unrolled 2x, so we have two consecutive a0/a1 and b00/b01
// etc. Multiplies a[c] with b[r,c] and adds to c[r].
template <class VF>
HWY_INLINE void UpdateTileRow(const VF& a0, const VF& a1, const VF& b00,
                              const VF& b01, const VF& b10, const VF& b11,
                              const VF& b20, const VF& b21, const VF& b30,
                              const VF& b31, VF& c0, VF& c1, VF& c2, VF& c3) {
  c0 = hn::MulAdd(a0, b00, c0);
  c1 = hn::MulAdd(a0, b10, c1);
  c2 = hn::MulAdd(a0, b20, c2);
  c3 = hn::MulAdd(a0, b30, c3);
  c0 = hn::MulAdd(a1, b01, c0);
  c1 = hn::MulAdd(a1, b11, c1);
  c2 = hn::MulAdd(a1, b21, c2);
  c3 = hn::MulAdd(a1, b31, c3);
}

// Streams a `(kNumRows, 4)` strip of `A` and the transposed `B`, then writes a
// finished tile of `C`.
// General case: uses CompressTraits to load from A and B.
template <size_t kNumRows, bool kAdd, typename MatTA, typename MatTB>
HWY_INLINE void MatMulTile(const Mat<MatTA>& A, const Mat<MatTB>& B,
                           const size_t row_a, const size_t row_b_col_c,
                           const float scale, const float* HWY_RESTRICT add,
                           const Mat<float>& C) {
  using TraitsA = CompressTraits<hwy::RemoveConst<MatTA>>;
  using TraitsB = CompressTraits<hwy::RemoveConst<MatTB>>;

  const hn::ScalableTag<float> d32;
  const size_t N = hn::Lanes(d32);
  using V = hn::Vec<decltype(d32)>;
  V c00 = hn::Zero(d32);
  V c01 = hn::Zero(d32);
  V c02 = hn::Zero(d32);
  V c03 = hn::Zero(d32);

  V c10 = hn::Zero(d32);
  V c11 = hn::Zero(d32);
  V c12 = hn::Zero(d32);
  V c13 = hn::Zero(d32);

  V c20 = hn::Zero(d32);
  V c21 = hn::Zero(d32);
  V c22 = hn::Zero(d32);
  V c23 = hn::Zero(d32);

  V c30 = hn::Zero(d32);
  V c31 = hn::Zero(d32);
  V c32 = hn::Zero(d32);
  V c33 = hn::Zero(d32);

  const size_t A_ofs = A.Row(row_a);
  const size_t B_ofs = B.Row(row_b_col_c);

  // Loop over columns of A and columns of the transposed B, in steps of 2*N
  // (since we are decoding consecutive bytes at each iteration).
  // Top-left of tile is (row_a, col_ab) for A, and (row_b_col_c,
  // col_ab) for B. Accumulates into the c## vectors.
  size_t col_ab = 0;

  HWY_UNROLL(1)
  for (; col_ab <= A.cols - 2 * N; col_ab += 2 * N) {
    V b00, b01;
    TraitsB::Decompress2(d32, B.ptr, B_ofs + B.stride * 0 + col_ab, b00, b01);
    V b10, b11;
    TraitsB::Decompress2(d32, B.ptr, B_ofs + B.stride * 1 + col_ab, b10, b11);
    V b20, b21;
    TraitsB::Decompress2(d32, B.ptr, B_ofs + B.stride * 2 + col_ab, b20, b21);
    V b30, b31;
    TraitsB::Decompress2(d32, B.ptr, B_ofs + B.stride * 3 + col_ab, b30, b31);

    V a00, a01;
    TraitsA::Decompress2(d32, A.ptr, A_ofs + A.stride * 0 + col_ab, a00, a01);
    UpdateTileRow(a00, a01, b00, b01, b10, b11, b20, b21, b30, b31, c00, c01,
                  c02, c03);
    if constexpr (kNumRows == 1) continue;

    V a10, a11;
    TraitsA::Decompress2(d32, A.ptr, A_ofs + A.stride * 1 + col_ab, a10, a11);
    UpdateTileRow(a10, a11, b00, b01, b10, b11, b20, b21, b30, b31, c10, c11,
                  c12, c13);
    if constexpr (kNumRows == 2) continue;

    V a20, a21;
    TraitsA::Decompress2(d32, A.ptr, A_ofs + A.stride * 2 + col_ab, a20, a21);
    UpdateTileRow(a20, a21, b00, b01, b10, b11, b20, b21, b30, b31, c20, c21,
                  c22, c23);
    if constexpr (kNumRows == 3) continue;

    V a30, a31;
    TraitsA::Decompress2(d32, A.ptr, A_ofs + A.stride * 3 + col_ab, a30, a31);
    UpdateTileRow(a30, a31, b00, b01, b10, b11, b20, b21, b30, b31, c30, c31,
                  c32, c33);
  }

  float* HWY_RESTRICT C_tile = C.ptr + C.Row(row_a) + row_b_col_c;
  StoreHorizontalSumsMaybeAdd<kAdd, kNumRows>(
      d32, c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31,
      c32, c33, scale, add, row_b_col_c, C_tile, C.stride);
}

// Computes the matrix product `A * B * scale [+ add]` and stores it in `C`.
//
// `A` is a row-major matrix of shape `(batch_size, A.cols)`.
// `B` is transposed; `B.cols`, which must match `A.cols`, denotes the number of
// rows in the original B, and `C.cols` the number of columns in the original B.
//
// `scale` allows expanding the smaller range of `SfpStream` to the original
// values. When `A` and/or `B` are from CompressedArray, `scale` should be the
// product of their `.scale()` values.
//
// If `kAdd` is true, the row-vector `add` is added to each row of `C`,
// otherwise `add` is ignored and can be nullptr. A scale for `add` is not
// supported, so make sure its scale is 1.
//
// `C` is a row-major matrix of size `(batch_size, C.cols)`.
// Writes 4x4 tiles of C in parallel using a work-stealing thread pool.
// Typically batch_size is 1..512, A.cols and C.cols are 3k or 24k.
template <bool kAdd, typename MatTA, typename MatTB>
HWY_NOINLINE void MatMul_4x4(const size_t batch_size, const Mat<MatTA>& A,
                             const Mat<MatTB>& B, const float scale,
                             const float* HWY_RESTRICT add, const Mat<float>& C,
                             hwy::ThreadPool& pool) {
  PROFILER_ZONE("Matmul");
  constexpr size_t kRegRows = 4;  // if changing, also update the switch below.
  constexpr size_t kRegCols = 4;

  HWY_DASSERT(A.NotEmpty() && B.NotEmpty() && C.NotEmpty());
  HWY_DASSERT(A.cols == B.cols);

  // Use float instead of MatTA/MatTB because we decompress to float here.
  const size_t N = hn::Lanes(hn::ScalableTag<float>());
  (void)N;
  HWY_DASSERT(A.cols % (N * 2) == 0);  // For Decompress2.
  HWY_DASSERT(C.cols % kRegCols == 0);

  // We currently write C directly, which touches more memory than fits in L3.
  // TODO: add another level of loops to finish L3-sized pieces of C at a time.
  const size_t tilesY = hwy::DivCeil(batch_size, kRegRows);
  const size_t tilesX = C.cols / kRegCols;

  pool.Run(0, tilesX * tilesY,
           [&](const uint64_t idx_tile, size_t /*thread*/) HWY_ATTR {
             const size_t tx = idx_tile % tilesX;
             const size_t ty = idx_tile / tilesX;
             const size_t row_a = ty * kRegRows;
             const size_t row_b_col_c = tx * kRegCols;
             // How many rows of C are left to compute. If more than 4, this
             // tile still only computes 4 rows.
             const size_t num_rows = batch_size - row_a;
             HWY_DASSERT(num_rows != 0);
             switch (num_rows) {
               case 1:
                 MatMulTile<1, kAdd>(A, B, row_a, row_b_col_c, scale, add, C);
                 break;
               case 2:
                 MatMulTile<2, kAdd>(A, B, row_a, row_b_col_c, scale, add, C);
                 break;
               case 3:
                 MatMulTile<3, kAdd>(A, B, row_a, row_b_col_c, scale, add, C);
                 break;
               default:
                 MatMulTile<4, kAdd>(A, B, row_a, row_b_col_c, scale, add, C);
             }
           });
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
