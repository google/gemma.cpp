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

#include "compression/compress.h"
#include "compression/sfp.h"
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
#include "hwy/contrib/dot/dot-inl.h"
#include "hwy/contrib/math/math-inl.h"
#include "hwy/contrib/matvec/matvec-inl.h"

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
  // expensive, but only a fraction of the kColsA_RowsB/N FMAs.
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
                                       const float* HWY_RESTRICT add,
                                       const float scale,
                                       float* HWY_RESTRICT tile_c,
                                       size_t stride_c) {
  // We are computing the product of (4, 4N) * (4N, 4) = (4, 4) tiles.
  // Each entry of C[r,c] is a dot product of A.row and B.col, which reside in
  // the lanes of `c$r$c`, so we store their horizontal sum (ReduceSum). This is
  // expensive, but only a fraction of the kColsA_RowsB/N FMAs.
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
    const float* HWY_RESTRICT add, size_t add_offset, const float scale,
    float* HWY_RESTRICT tile_c, size_t stride_c) {
  if constexpr (kAdd) {
    StoreHorizontalSumsAdd<kNumRows>(df, c00, c01, c02, c03, c10, c11, c12, c13,
                                     c20, c21, c22, c23, c30, c31, c32, c33,
                                     add + add_offset, scale, tile_c, stride_c);
  } else {
    StoreHorizontalSums<kNumRows>(df, c00, c01, c02, c03, c10, c11, c12, c13,
                                  c20, c21, c22, c23, c30, c31, c32, c33,
                                  scale, tile_c, stride_c);
  }
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
HWY_INLINE void GEMM_4x4_Tile(const hwy::bfloat16_t* HWY_RESTRICT A,
                              const size_t A_ofs,
                              const hwy::bfloat16_t* HWY_RESTRICT B,
                              const size_t B_ofs, float* HWY_RESTRICT C,
                              const float scale, const float* HWY_RESTRICT add,
                              const size_t idx_tile, const size_t xtiles,
                              const size_t cols_a, const size_t stride_a,
                              const size_t stride_b, const size_t stride_c) {
  constexpr size_t kRegRows = 4;
  constexpr size_t kRegCols = 4;
  static_assert(kNumRows <= kRegRows);

  // Top-left of tile is (row_a, col_ab) for A, and (row_b_col_c, col_ab) for B.
  const size_t row_a = idx_tile / xtiles * kRegRows;
  const size_t row_b_col_c = idx_tile % xtiles * kRegCols;

  const hn::ScalableTag<float> df;
  using VF = hn::Vec<decltype(df)>;
  // ReorderWidenMulAccumulate does not use its sum1 arg and we can use full
  // bf16 vectors.
  const hn::Repartition<hwy::bfloat16_t, decltype(df)> d;
  VF unused_sum1 = hn::Zero(df);

  const size_t N = Lanes(d);

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

  const hwy::bfloat16_t* HWY_RESTRICT A_tile = A + A_ofs + stride_a * row_a;
  const hwy::bfloat16_t* HWY_RESTRICT B_tile =
      B + B_ofs + stride_b * row_b_col_c;

  // Loop over columns of A and columns of the transposed B, in steps of N.
  // Accumulates into the c## vectors.
  HWY_UNROLL(1)
  for (size_t col_ab = 0; col_ab < cols_a; col_ab += N) {
    using V = hn::Vec<decltype(d)>;
    const V b0 = hn::LoadU(d, B_tile + stride_b * 0 + col_ab);
    const V b1 = hn::LoadU(d, B_tile + stride_b * 1 + col_ab);
    const V b2 = hn::LoadU(d, B_tile + stride_b * 2 + col_ab);
    const V b3 = hn::LoadU(d, B_tile + stride_b * 3 + col_ab);

    const V a0 = hn::LoadU(d, A_tile + stride_a * 0 + col_ab);
    c00 = hn::ReorderWidenMulAccumulate(df, a0, b0, c00, unused_sum1);
    c01 = hn::ReorderWidenMulAccumulate(df, a0, b1, c01, unused_sum1);
    c02 = hn::ReorderWidenMulAccumulate(df, a0, b2, c02, unused_sum1);
    c03 = hn::ReorderWidenMulAccumulate(df, a0, b3, c03, unused_sum1);
    if constexpr (kNumRows == 1) continue;

    const V a1 = hn::LoadU(d, A_tile + stride_a * 1 + col_ab);
    c10 = hn::ReorderWidenMulAccumulate(df, a1, b0, c10, unused_sum1);
    c11 = hn::ReorderWidenMulAccumulate(df, a1, b1, c11, unused_sum1);
    c12 = hn::ReorderWidenMulAccumulate(df, a1, b2, c12, unused_sum1);
    c13 = hn::ReorderWidenMulAccumulate(df, a1, b3, c13, unused_sum1);
    if constexpr (kNumRows == 2) continue;

    const V a2 = hn::LoadU(d, A_tile + stride_a * 2 + col_ab);
    c20 = hn::ReorderWidenMulAccumulate(df, a2, b0, c20, unused_sum1);
    c21 = hn::ReorderWidenMulAccumulate(df, a2, b1, c21, unused_sum1);
    c22 = hn::ReorderWidenMulAccumulate(df, a2, b2, c22, unused_sum1);
    c23 = hn::ReorderWidenMulAccumulate(df, a2, b3, c23, unused_sum1);
    if constexpr (kNumRows == 3) continue;

    const V a3 = hn::LoadU(d, A_tile + stride_a * 3 + col_ab);
    c30 = hn::ReorderWidenMulAccumulate(df, a3, b0, c30, unused_sum1);
    c31 = hn::ReorderWidenMulAccumulate(df, a3, b1, c31, unused_sum1);
    c32 = hn::ReorderWidenMulAccumulate(df, a3, b2, c32, unused_sum1);
    c33 = hn::ReorderWidenMulAccumulate(df, a3, b3, c33, unused_sum1);
  }

  // Ensure sum1 was indeed unused.
  HWY_DASSERT(hn::AllTrue(df, hn::Eq(unused_sum1, hn::Zero(df))));

  float* HWY_RESTRICT C_tile = C + stride_c * row_a + row_b_col_c;
  StoreHorizontalSumsMaybeAdd<kAdd, kNumRows>(
      df, c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31,
      c32, c33, add, row_b_col_c, scale, C_tile, stride_c);
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

// Accumulates a single kNumRows (<= 4) x 4 tile of A x B into C. B is
// transposed, so we iterate over both A and B with consecutive vector loads.
// General case: uses CompressTraits to load from A and B.
template <size_t kNumRows, bool kAdd, typename MatTA, typename MatTB>
HWY_INLINE void GEMM_4x4_Tile(const MatTA* HWY_RESTRICT A, const size_t A_ofs,
                              const MatTB* HWY_RESTRICT B, const size_t B_ofs,
                              float* HWY_RESTRICT C, const float scale,
                              const float* HWY_RESTRICT add,
                              const size_t idx_tile, const size_t xtiles,
                              const size_t cols_a, const size_t stride_a,
                              const size_t stride_b, const size_t stride_c) {
  constexpr size_t kRegRows = 4;
  constexpr size_t kRegCols = 4;
  static_assert(kNumRows <= kRegRows);

  using TraitsA = CompressTraits<MatTA>;
  using TraitsB = CompressTraits<MatTB>;

  // Top-left of tile is (row_a, col_ab) for A, and (row_b_col_c, col_ab) for B.
  const size_t row_a = idx_tile / xtiles * kRegRows;
  const size_t row_b_col_c = idx_tile % xtiles * kRegCols;

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

  const size_t A_tile_ofs = A_ofs + stride_a * row_a;
  const size_t B_tile_ofs = B_ofs + stride_b * row_b_col_c;

  // Loop over columns of A and columns of the transposed B, in steps of 2*N
  // (since we are decoding consecutive bytes at each iteration).
  // Accumulates into the c## vectors.
  size_t col_ab = 0;

  HWY_UNROLL(1)
  for (; col_ab <= cols_a - 2 * N; col_ab += 2 * N) {
    V b00, b01;
    TraitsB::Decompress2(d32, B, B_tile_ofs + stride_b * 0 + col_ab, b00, b01);
    V b10, b11;
    TraitsB::Decompress2(d32, B, B_tile_ofs + stride_b * 1 + col_ab, b10, b11);
    V b20, b21;
    TraitsB::Decompress2(d32, B, B_tile_ofs + stride_b * 2 + col_ab, b20, b21);
    V b30, b31;
    TraitsB::Decompress2(d32, B, B_tile_ofs + stride_b * 3 + col_ab, b30, b31);

    V a00, a01;
    TraitsA::Decompress2(d32, A, A_tile_ofs + stride_a * 0 + col_ab, a00, a01);
    UpdateTileRow(a00, a01, b00, b01, b10, b11, b20, b21, b30, b31, c00, c01,
                  c02, c03);
    if constexpr (kNumRows == 1) continue;

    V a10, a11;
    TraitsA::Decompress2(d32, A, A_tile_ofs + stride_a * 1 + col_ab, a10, a11);
    UpdateTileRow(a10, a11, b00, b01, b10, b11, b20, b21, b30, b31, c10, c11,
                  c12, c13);
    if constexpr (kNumRows == 2) continue;

    V a20, a21;
    TraitsA::Decompress2(d32, A, A_tile_ofs + stride_a * 2 + col_ab, a20, a21);
    UpdateTileRow(a20, a21, b00, b01, b10, b11, b20, b21, b30, b31, c20, c21,
                  c22, c23);
    if constexpr (kNumRows == 3) continue;

    V a30, a31;
    TraitsA::Decompress2(d32, A, A_tile_ofs + stride_a * 3 + col_ab, a30, a31);
    UpdateTileRow(a30, a31, b00, b01, b10, b11, b20, b21, b30, b31, c30, c31,
                  c32, c33);
  }

  float* HWY_RESTRICT C_tile = C + stride_c * row_a + row_b_col_c;
  StoreHorizontalSumsMaybeAdd<kAdd, kNumRows>(
      d32, c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31,
      c32, c33, add, row_b_col_c, scale, C_tile, stride_c);
}

// Tiled 4x4 GEMM: C = A * B * scale [+ add].
// Computes the matrix product of A and B and stores this in C. Processes tiles
// of 4x4 vectors in parallel with a work-stealing thread pool.
//
// If kAdd is true, the row-vector `add` is added to each row of C, otherwise
// `add` is ignored and can be nullptr.
// A is a row-major matrix of size (batch_size, kColsA_RowsB).
// B is passed transposed (column-major), so a matrix of size
// (kColsBC, kColsA_RowsB), representing a B of size (kColsA_RowsB, kColsBC).
// A_ofs and B_ofs are offsets into A and B, respectively; they remain separate
// from the pointers because some MatTA/B such as NuqStream do not support
// pointer arithmetic.
// C is a matrix of size (batch_size, kColsBC).
// The product is scaled by `scale` to support CompressedArray with scale != 1,
// the caller can pass the product of the scales of A and B.
// A scale for `add` is not supported, so make sure its scale is 1.
// Typically batch_size is 1..512, kColsA_RowsB and kColsBC are 3k or 24k.
template <size_t kColsA_RowsB, size_t kColsBC, bool kAdd, typename MatTA,
          typename MatTB, typename OutT>
HWY_NOINLINE void MatMul_4x4(const size_t batch_size,
                             const MatTA* HWY_RESTRICT A, const size_t A_ofs,
                             const MatTB* HWY_RESTRICT B, const size_t B_ofs,
                             const float scale, OutT* HWY_RESTRICT C,
                             const float* HWY_RESTRICT add,
                             hwy::ThreadPool& pool) {
  PROFILER_ZONE("Matmul");
  // We currently write C directly, which touches more memory than fits in L3.
  // TODO: add another level of loops to finish L3-sized pieces of C at a time.
  const hn::ScalableTag<MatTA> d;
  const size_t N = Lanes(d);
  constexpr size_t kRegRows = 4;
  constexpr size_t kRegCols = 4;  // in vectors

  static_assert(kColsBC % kRegCols == 0);
  HWY_ASSERT(kColsA_RowsB % (N * kRegCols) == 0);
  const size_t kTilesY = (batch_size + kRegRows - 1) / kRegRows;
  const size_t kTilesX = kColsBC / kRegCols;
  const size_t kTiles = kTilesX * kTilesY;

  constexpr size_t kStrideA = kColsA_RowsB;
  constexpr size_t kStrideB = kColsA_RowsB;
  constexpr size_t kStrideC = kColsBC;

  pool.Run(0, kTiles, [&](const uint64_t idx_tile, size_t /*thread*/) HWY_ATTR {
    // Computes the finished product of one 4x4N tile and writes to C.
    const size_t num_rows = batch_size - idx_tile / kTilesX * kRegRows;
    HWY_ASSERT(num_rows > 0);
    switch (num_rows) {
      case 1:
        GEMM_4x4_Tile<1, kAdd>(A, A_ofs, B, B_ofs, C, scale, add, idx_tile,
                               kTilesX, kColsA_RowsB, kStrideA, kStrideB,
                               kStrideC);
        break;
      case 2:
        GEMM_4x4_Tile<2, kAdd>(A, A_ofs, B, B_ofs, C, scale, add, idx_tile,
                               kTilesX, kColsA_RowsB, kStrideA, kStrideB,
                               kStrideC);
        break;
      case 3:
        GEMM_4x4_Tile<3, kAdd>(A, A_ofs, B, B_ofs, C, scale, add, idx_tile,
                               kTilesX, kColsA_RowsB, kStrideA, kStrideB,
                               kStrideC);
        break;
      default:
        GEMM_4x4_Tile<4, kAdd>(A, A_ofs, B, B_ofs, C, scale, add, idx_tile,
                               kTilesX, kColsA_RowsB, kStrideA, kStrideB,
                               kStrideC);
    }
  });
}

//------------------------------------------------------------------------------
HWY_INLINE void ToEvenOddF32(const hwy::bfloat16_t* HWY_RESTRICT vec_aligned,
                             const size_t size, float* HWY_RESTRICT out) {
  const hn::ScalableTag<float> df;
  const hn::Repartition<hwy::bfloat16_t, decltype(df)> dbf16;

  HWY_DASSERT(size % hn::Lanes(dbf16) == 0);
  HWY_DASSERT(hn::IsAligned(df, vec_aligned));

  for (size_t i = 0; i < size; i += hn::Lanes(dbf16)) {
    const auto interleaved = hn::LoadU(dbf16, vec_aligned + i);
    hn::Store(hn::PromoteEvenTo(df, interleaved), df, out + i);
    hn::Store(hn::PromoteOddTo(df, interleaved), df, out + i + hn::Lanes(df));
  }
}

HWY_INLINE void ToEvenOddF32(const float* HWY_RESTRICT vec_aligned,
                             const size_t size, float* HWY_RESTRICT out) {
  const hn::ScalableTag<float> df;
  using VF = hn::Vec<decltype(df)>;

  HWY_DASSERT(size % (hn::Lanes(df) * 2) == 0);
  HWY_DASSERT(hn::IsAligned(df, vec_aligned));

  VF vec0, vec1;
  for (size_t i = 0; i < size; i += hn::Lanes(df) * 2) {
    hn::LoadInterleaved2(df, vec_aligned + i, vec0, vec1);
    hn::Store(vec0, df, out + i);
    hn::Store(vec1, df, out + i + hn::Lanes(df));
  }
}

// Simple version without tiling nor threading, but two offsets/outputs and
// always with addition.
template <size_t kOuter, size_t kInner, typename ArrayT, typename VecT,
          typename AddT>
HWY_INLINE void TwoOfsMatVecAddLoop(const ArrayT& mat, const size_t mat_ofs0,
                                    const size_t mat_ofs1,
                                    const VecT* HWY_RESTRICT vec_aligned,
                                    const AddT* HWY_RESTRICT add0,
                                    const AddT* HWY_RESTRICT add1,
                                    float* HWY_RESTRICT out0,
                                    float* HWY_RESTRICT out1) {
  PROFILER_ZONE("TwoOfsMatVecAddLoop");
  constexpr bool kVecEO = false;
  const hn::ScalableTag<float> df;

  for (size_t idx_row = 0; idx_row < kOuter; ++idx_row) {
    const size_t row_ofs0 = mat_ofs0 + (idx_row)*kInner;
    const size_t row_ofs1 = mat_ofs1 + (idx_row)*kInner;
    out0[idx_row] = hwy::ConvertScalarTo<float>(add0[idx_row]) +
                    Dot<kVecEO>(df, mat, row_ofs0, vec_aligned, kInner);
    out1[idx_row] = hwy::ConvertScalarTo<float>(add1[idx_row]) +
                    Dot<kVecEO>(df, mat, row_ofs1, vec_aligned, kInner);
  }
}

HWY_INLINE constexpr size_t MaxCols() {
  // Vec + mat rows should fit into 32 KiB L1.
  return 2048;
}

template <size_t kOuter>
HWY_INLINE constexpr size_t RowsPerStrip() {
  // Aim for 128 work items to reduce pool overhead. Must be at least one
  // vector; prefer a power of two for faster division.
  constexpr size_t kLanes = hn::ScalableTag<float>().MaxLanes();
  constexpr size_t kRowsPerStrip =
      kOuter < 128 ? kLanes
                   : HWY_MAX(kLanes, 1ULL << hwy::FloorLog2(kOuter / 128));
  return kRowsPerStrip;
}

namespace detail {

// For each i = [0, num_rows), compute partial (length `num_cols`) dot product
// of row i with `vec_aligned` and add into `out[i]`. The upper-left
// coordinate of the tile is r0, c0.
template <bool kVecEO, class DF, typename ArrayT, typename VecT>
HWY_INLINE void AccumulatePartialDotProducts(
    DF df, const ArrayT& mat, size_t mat_ofs, size_t mat_stride, size_t r0,
    size_t c0, size_t num_rows, size_t num_cols,
    const VecT* HWY_RESTRICT vec_aligned, float* HWY_RESTRICT out) {
  for (size_t idx_row = 0; idx_row < num_rows; ++idx_row) {
    const size_t row_ofs = mat_ofs + (r0 + idx_row) * mat_stride;
    out[idx_row] +=
        Dot<kVecEO>(df, mat, row_ofs + c0, vec_aligned + c0, num_cols);
  }
}

// Same as AccumulatePartialDotProducts, but sets out[i] to the first partial
// dot product + init (if kInit), which avoids having to zero-initialize and
// accumulate.
template <bool kVecEO, bool kInit, class DF, typename ArrayT, typename VecT,
          typename InitT>
HWY_INLINE void SetFirstPartialDotProducts(DF df, const ArrayT& mat,
                                           size_t mat_ofs, size_t mat_stride,
                                           size_t r0, size_t c0,
                                           size_t num_rows, size_t num_cols,
                                           const VecT* HWY_RESTRICT vec_aligned,
                                           const InitT* HWY_RESTRICT init,
                                           float* HWY_RESTRICT out) {
  for (size_t idx_row = 0; idx_row < num_rows; ++idx_row) {
    const size_t row_ofs = mat_ofs + (r0 + idx_row) * mat_stride;
    if constexpr (kInit) {
      out[idx_row] =
          hwy::ConvertScalarTo<float>(init[idx_row + r0]) +
          Dot<kVecEO>(df, mat, row_ofs + c0, vec_aligned + c0, num_cols);
    } else {
      out[idx_row] =
          Dot<kVecEO>(df, mat, row_ofs + c0, vec_aligned + c0, num_cols);
    }
  }
}

// Adds together partial dot products for all tiles with the same r0 (a
// horizontal strip of the entire matrix); the result is the full dot product
// for rows r in [r0, r0 + num_rows) + optionally the add vector, which we
// store into in out[r - r0].
template <bool kVecEO, bool kAdd, class DF, typename ArrayT, typename VecT,
          typename AddT>
HWY_INLINE void FullDotProductsForStrip(DF df, const ArrayT& mat,
                                        size_t mat_ofs, size_t mat_stride,
                                        size_t r0, size_t num_rows,
                                        const VecT* HWY_RESTRICT vec_aligned,
                                        const AddT* HWY_RESTRICT add,
                                        float* HWY_RESTRICT out) {
  // Tall and skinny: set `out` to the single dot product.
  if (mat_stride < MaxCols()) {
    SetFirstPartialDotProducts<kVecEO, kAdd>(df, mat, mat_ofs, mat_stride, r0,
                                             0, num_rows, mat_stride,
                                             vec_aligned, add, out);
    return;
  }

  // We have at least MaxCols, so start by setting `out` to that:
  SetFirstPartialDotProducts<kVecEO, kAdd>(df, mat, mat_ofs, mat_stride, r0, 0,
                                           num_rows, MaxCols(), vec_aligned,
                                           add, out);
  // For further multiples of MaxCols, accumulate. Remainders handled below.
  size_t c0 = MaxCols();
  for (; c0 <= mat_stride - MaxCols(); c0 += MaxCols()) {
    AccumulatePartialDotProducts<kVecEO>(df, mat, mat_ofs, mat_stride, r0, c0,
                                         num_rows, MaxCols(), vec_aligned, out);
  }

  if (c0 < mat_stride) {  // Final cols
    AccumulatePartialDotProducts<kVecEO>(df, mat, mat_ofs, mat_stride, r0, c0,
                                         num_rows, mat_stride - c0, vec_aligned,
                                         out);
  }
}

template <bool kVecIsEvenOdd, bool kAdd, size_t kOuter, size_t kInner,
          typename ArrayT, typename VecT, typename AddT>
HWY_INLINE void MatVecAddInner(const ArrayT& mat, const size_t mat_ofs,
                               const VecT* HWY_RESTRICT const vec_aligned,
                               const AddT* HWY_RESTRICT const add,
                               float* HWY_RESTRICT out, hwy::ThreadPool& pool) {
  const hn::ScalableTag<float> df;
  constexpr size_t kRowsPerStrip = RowsPerStrip<kOuter>();
  constexpr size_t kNumStrips = kOuter / kRowsPerStrip;

  // For each entire strip.
  pool.Run(0, kNumStrips, [&](const uint64_t strip, size_t thread) HWY_ATTR {
    PROFILER_ZONE("MatVec.lambda");
    const size_t r0 = strip * kRowsPerStrip;
    detail::FullDotProductsForStrip<kVecIsEvenOdd, kAdd>(
        df, mat, mat_ofs, kInner, r0, kRowsPerStrip, vec_aligned, add,
        out + r0);
  });

  // Remaining rows
  const size_t r0 = kNumStrips * kRowsPerStrip;
  if (r0 < kOuter) {
    PROFILER_ZONE("MatVec remainder");
    const size_t num_rows = kOuter - r0;
    detail::FullDotProductsForStrip<kVecIsEvenOdd, kAdd>(
        df, mat, mat_ofs, kInner, r0, num_rows, vec_aligned, add, out + r0);
  }
}

}  // namespace detail

// Stores dot products of rows with `vec_aligned` + add the values from `add`
// (if kAdd), then stores them to `out`.
template <bool kAdd, size_t kOuter, size_t kInner, typename ArrayT,
          typename VecT, typename AddT>
HWY_INLINE void MatVecT(const ArrayT& mat, const size_t mat_ofs,
                        const VecT* HWY_RESTRICT const vec_aligned,
                        const AddT* HWY_RESTRICT const add,
                        float* HWY_RESTRICT even_odd, float* HWY_RESTRICT out,
                        hwy::ThreadPool& pool) {
  PROFILER_ZONE("MatVecAdd");

#if !defined(HWY_NATIVE_DOT_BF16) || !HWY_NATIVE_DOT_BF16
  using MatT = typename ArrayT::value_type;
  // Sfp -> float does not benefit enough to recoup the cost of ToEvenOddF32.
  if constexpr (CompressTraits<MatT>::kSupportsEvenOdd &&
                hwy::IsSameEither<VecT, float, hwy::bfloat16_t>() &&
                !(hwy::IsSame<MatT, SfpStream>() &&
                  hwy::IsSame<VecT, float>())) {
    ToEvenOddF32(vec_aligned, kInner, even_odd);
    detail::MatVecAddInner</*kVecIsEvenOdd=*/true, kAdd, kOuter, kInner>(
        mat, mat_ofs, even_odd, add, out, pool);
    return;
  }
#else
  (void)even_odd;
#endif

  detail::MatVecAddInner</*kVecIsEvenOdd=*/false, kAdd, kOuter, kInner>(
      mat, mat_ofs, vec_aligned, add, out, pool);
}

// With addition
template <size_t kOuter, size_t kInner, typename ArrayT, typename VecT,
          typename AddT>
HWY_INLINE void MatVecAdd(const ArrayT& mat, const size_t mat_ofs,
                          const VecT* HWY_RESTRICT const vec_aligned,
                          const AddT* HWY_RESTRICT const add,
                          float* HWY_RESTRICT even_odd, float* HWY_RESTRICT out,
                          hwy::ThreadPool& pool) {
  return MatVecT</*kAdd=*/true, kOuter, kInner>(mat, mat_ofs, vec_aligned, add,
                                                even_odd, out, pool);
}

// Without addition
template <size_t kOuter, size_t kInner, typename ArrayT, typename VecT>
HWY_INLINE void MatVec(const ArrayT& mat, const size_t mat_ofs,
                       const VecT* HWY_RESTRICT const vec_aligned,
                       float* HWY_RESTRICT even_odd, float* HWY_RESTRICT out,
                       hwy::ThreadPool& pool) {
  MatVecT</*kAdd=*/false, kOuter, kInner>(mat, mat_ofs, vec_aligned,
                                          /*add=*/static_cast<VecT*>(nullptr),
                                          even_odd, out, pool);
}

// Two matrices, same vector
template <bool kAdd, size_t kOuter, size_t kInner, typename ArrayT,
          typename VecT, typename AddT>
HWY_NOINLINE void TwoMatVecT(const ArrayT& mat0, const ArrayT& mat1,
                             const size_t mat_ofs,
                             const VecT* HWY_RESTRICT vec_aligned,
                             const AddT* HWY_RESTRICT add0,
                             const AddT* HWY_RESTRICT add1,
                             float* HWY_RESTRICT out0, float* HWY_RESTRICT out1,
                             hwy::ThreadPool& pool) {
  PROFILER_ZONE("TwoMatVecAdd");

  const hn::ScalableTag<float> df;
  constexpr size_t kRowsPerStrip = RowsPerStrip<kOuter>();
  constexpr size_t kNumStrips = kOuter / kRowsPerStrip;
  constexpr bool kVecIsEvenOdd = false;

  // For each entire strip.
  pool.Run(0, kNumStrips, [&](const uint64_t strip, size_t thread) HWY_ATTR {
    PROFILER_ZONE("TwoMatVec.lambda");
    const size_t r0 = strip * kRowsPerStrip;
    detail::FullDotProductsForStrip<kVecIsEvenOdd, kAdd>(
        df, mat0, mat_ofs, kInner, r0, kRowsPerStrip, vec_aligned, add0,
        out0 + r0);
    detail::FullDotProductsForStrip<kVecIsEvenOdd, kAdd>(
        df, mat1, mat_ofs, kInner, r0, kRowsPerStrip, vec_aligned, add1,
        out1 + r0);
  });

  // Remaining rows
  const size_t r0 = kNumStrips * kRowsPerStrip;
  if (r0 < kOuter) {
    PROFILER_ZONE("TwoMatVec remainder");
    const size_t num_rows = kOuter - r0;
    detail::FullDotProductsForStrip<kVecIsEvenOdd, kAdd>(
        df, mat0, mat_ofs, kInner, r0, num_rows, vec_aligned, add0, out0 + r0);
    detail::FullDotProductsForStrip<kVecIsEvenOdd, kAdd>(
        df, mat1, mat_ofs, kInner, r0, num_rows, vec_aligned, add1, out1 + r0);
  }
}

// With addition
template <size_t kOuter, size_t kInner, typename ArrayT, typename VecT,
          typename AddT>
HWY_NOINLINE void TwoMatVecAdd(
    const ArrayT& mat0, const ArrayT& mat1, const size_t mat_ofs,
    const VecT* HWY_RESTRICT vec_aligned, const AddT* HWY_RESTRICT add0,
    const AddT* HWY_RESTRICT add1, float* HWY_RESTRICT out0,
    float* HWY_RESTRICT out1, hwy::ThreadPool& pool) {
  return TwoMatVecT</*kAdd=*/true, kOuter, kInner>(
      mat0, mat1, mat_ofs, vec_aligned, add0, add1, out0, out1, pool);
}

// Without addition
template <size_t kOuter, size_t kInner, typename ArrayT, typename VecT>
HWY_NOINLINE void TwoMatVec(const ArrayT& mat0, const ArrayT& mat1,
                            const size_t mat_ofs,
                            const VecT* HWY_RESTRICT vec_aligned,
                            float* HWY_RESTRICT out0, float* HWY_RESTRICT out1,
                            hwy::ThreadPool& pool) {
  TwoMatVecT</*kAdd=*/false, kOuter, kInner, ArrayT, VecT, VecT>(
      mat0, mat1, mat_ofs, vec_aligned, /*add0=*/nullptr, /*add1=*/nullptr,
      out0, out1, pool);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
