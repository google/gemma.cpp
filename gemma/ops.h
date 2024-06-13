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
#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_OPS_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_OPS_H_

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <array>
#include <random>
#include <type_traits>  // std::enable_if_t

#include "compression/sfp.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/detect_targets.h"
#include "hwy/profiler.h"

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_OPS_H_

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_OPS_TOGGLE) == defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_OPS_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_OPS_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_OPS_TOGGLE
#endif

#include "compression/compress-inl.h"
#include "hwy/contrib/algo/transform-inl.h"
#include "hwy/contrib/dot/dot-inl.h"
#include "hwy/contrib/math/math-inl.h"
#include "hwy/contrib/matvec/matvec-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

HWY_INLINE constexpr size_t MaxCols() {
  // Vec + mat rows should fit into 32 KiB L1.
  return 2048;
}

template <typename To, typename From>
HWY_INLINE constexpr std::enable_if_t<
    std::is_arithmetic_v<To> && std::is_arithmetic_v<From>, To>
StaticCast(From from) noexcept {
  if constexpr (std::is_unsigned_v<From> && std::is_floating_point_v<To>)
    return static_cast<To>(
        static_cast<hwy::SignedFromSize<sizeof(From)>>(from));
  else
    return static_cast<To>(from);
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

// Shared between f32 and bf16, which also accumulates into f32 vectors.
template <size_t kNumRows, class DF, class VF = hn::Vec<DF>>
HWY_INLINE void StoreHorizontalSums(DF df, VF c00, VF c01, VF c02, VF c03,
                                    VF c10, VF c11, VF c12, VF c13,  //
                                    VF c20, VF c21, VF c22, VF c23,  //
                                    VF c30, VF c31, VF c32, VF c33,
                                    float* HWY_RESTRICT tile_c,
                                    size_t stride_c) {
  // We are computing the product of (4, 4N) * (4N, 4) = (4, 4) tiles.
  // Each entry of C[r,c] is a dot product of A.row and B.col, which reside in
  // the lanes of `c$r$c`, so we store their horizontal sum (ReduceSum). This is
  // expensive, but only a fraction of the kColsA_RowsB/N FMAs.
  tile_c[stride_c * 0 + 0] = hn::ReduceSum(df, c00);
  tile_c[stride_c * 0 + 1] = hn::ReduceSum(df, c01);
  tile_c[stride_c * 0 + 2] = hn::ReduceSum(df, c02);
  tile_c[stride_c * 0 + 3] = hn::ReduceSum(df, c03);
  if (kNumRows == 1) return;

  tile_c[stride_c * 1 + 0] = hn::ReduceSum(df, c10);
  tile_c[stride_c * 1 + 1] = hn::ReduceSum(df, c11);
  tile_c[stride_c * 1 + 2] = hn::ReduceSum(df, c12);
  tile_c[stride_c * 1 + 3] = hn::ReduceSum(df, c13);
  if (kNumRows == 2) return;

  tile_c[stride_c * 2 + 0] = hn::ReduceSum(df, c20);
  tile_c[stride_c * 2 + 1] = hn::ReduceSum(df, c21);
  tile_c[stride_c * 2 + 2] = hn::ReduceSum(df, c22);
  tile_c[stride_c * 2 + 3] = hn::ReduceSum(df, c23);
  if (kNumRows == 3) return;

  tile_c[stride_c * 3 + 0] = hn::ReduceSum(df, c30);
  tile_c[stride_c * 3 + 1] = hn::ReduceSum(df, c31);
  tile_c[stride_c * 3 + 2] = hn::ReduceSum(df, c32);
  tile_c[stride_c * 3 + 3] = hn::ReduceSum(df, c33);
}

// Accumulates a single kNumRowsx4 tile of A x B into C. B is transposed, so we
// can iterate over both A and B with consecutive vector loads. kNumRows<=4.
// Shared between parallelized and sequential (loop) callers.
template <size_t kNumRows, size_t kColsA_RowsB, typename MatT, HWY_IF_F32(MatT)>
HWY_INLINE void GEMM_4x4_Tile(const MatT* HWY_RESTRICT A,
                              const MatT* HWY_RESTRICT B, MatT* HWY_RESTRICT C,
                              const size_t idx_tile, const size_t xtiles,
                              const size_t stride_a, const size_t stride_b,
                              const size_t stride_c) {
  // Tile size. 4x unrolling makes sense on many platforms because we can fit
  // 4x4 accumulators and 8 temporaries in the 32 vectors; we have more than
  // #FMA units * FMA latency (up to 2*5) independent computations in flight;
  // threads write in units of 4*N elements, which is at least one cache line.
  constexpr size_t kRegRows = 4;
  constexpr size_t kRegCols = 4;
  static_assert(kNumRows <= kRegRows);

  // Top-left of tile is (row_a, col_ab) for A, and (row_b_col_c, col_ab) for B.
  const size_t row_a = idx_tile / xtiles * kRegRows;
  const size_t row_b_col_c = idx_tile % xtiles * kRegCols;

  const hn::ScalableTag<MatT> d;
  const size_t N = Lanes(d);
  using V = hn::Vec<decltype(d)>;

  V c00 = hn::Zero(d);
  V c01 = hn::Zero(d);
  V c02 = hn::Zero(d);
  V c03 = hn::Zero(d);

  V c10 = hn::Zero(d);
  V c11 = hn::Zero(d);
  V c12 = hn::Zero(d);
  V c13 = hn::Zero(d);

  V c20 = hn::Zero(d);
  V c21 = hn::Zero(d);
  V c22 = hn::Zero(d);
  V c23 = hn::Zero(d);

  V c30 = hn::Zero(d);
  V c31 = hn::Zero(d);
  V c32 = hn::Zero(d);
  V c33 = hn::Zero(d);

  const MatT* HWY_RESTRICT tile_a = A + stride_a * row_a;
  const MatT* HWY_RESTRICT tile_b = B + stride_b * row_b_col_c;

  // Loop over columns of A and columns of the transposed B, in steps of N.
  // Accumulates into the c## vectors.
  HWY_UNROLL(1)
  for (size_t col_ab = 0; col_ab < kColsA_RowsB; col_ab += N) {
    const V b0 = hn::LoadU(d, tile_b + stride_b * 0 + col_ab);
    const V b1 = hn::LoadU(d, tile_b + stride_b * 1 + col_ab);
    const V b2 = hn::LoadU(d, tile_b + stride_b * 2 + col_ab);
    const V b3 = hn::LoadU(d, tile_b + stride_b * 3 + col_ab);

    const V a0 = hn::LoadU(d, tile_a + stride_a * 0 + col_ab);
    c00 = hn::MulAdd(a0, b0, c00);
    c01 = hn::MulAdd(a0, b1, c01);
    c02 = hn::MulAdd(a0, b2, c02);
    c03 = hn::MulAdd(a0, b3, c03);
    if (kNumRows == 1) continue;

    const V a1 = hn::LoadU(d, tile_a + stride_a * 1 + col_ab);
    c10 = hn::MulAdd(a1, b0, c10);
    c11 = hn::MulAdd(a1, b1, c11);
    c12 = hn::MulAdd(a1, b2, c12);
    c13 = hn::MulAdd(a1, b3, c13);
    if (kNumRows == 2) continue;

    const V a2 = hn::LoadU(d, tile_a + stride_a * 2 + col_ab);
    c20 = hn::MulAdd(a2, b0, c20);
    c21 = hn::MulAdd(a2, b1, c21);
    c22 = hn::MulAdd(a2, b2, c22);
    c23 = hn::MulAdd(a2, b3, c23);
    if (kNumRows == 3) continue;

    const V a3 = hn::LoadU(d, tile_a + stride_a * 3 + col_ab);
    c30 = hn::MulAdd(a3, b0, c30);
    c31 = hn::MulAdd(a3, b1, c31);
    c32 = hn::MulAdd(a3, b2, c32);
    c33 = hn::MulAdd(a3, b3, c33);
  }

  float* HWY_RESTRICT tile_c = C + stride_c * row_a + row_b_col_c;
  StoreHorizontalSums<kNumRows>(
      d, c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23,
      c30, c31, c32, c33, tile_c, stride_c);
}

#undef GEMMA_NATIVE_BF16
#if HWY_IDE || (defined(HWY_NATIVE_REORDER_WIDEN_MUL_ACC_BF16) == \
                defined(HWY_TARGET_TOGGLE))
#define GEMMA_NATIVE_BF16 1
#else
#define GEMMA_NATIVE_BF16 0
#endif

// As above, for MatT=bf16
template <size_t kNumRows, size_t kColsA_RowsB, typename MatT,
          HWY_IF_BF16(MatT)>
HWY_INLINE void GEMM_4x4_Tile(const MatT* HWY_RESTRICT A,
                              const MatT* HWY_RESTRICT B, float* HWY_RESTRICT C,
                              const size_t idx_tile, const size_t xtiles,
                              const size_t stride_a, const size_t stride_b,
                              const size_t stride_c) {
  constexpr size_t kRegRows = 4;
  constexpr size_t kRegCols = 4;
  static_assert(kNumRows <= kRegRows);

  // Top-left of tile is (row_a, col_ab) for A, and (row_b_col_c, col_ab) for B.
  const size_t row_a = idx_tile / xtiles * kRegRows;
  const size_t row_b_col_c = idx_tile % xtiles * kRegCols;

  const hn::ScalableTag<float> df;
  using VF = hn::Vec<decltype(df)>;
#if GEMMA_NATIVE_BF16
  // ReorderWidenMulAccumulate does not use its sum1 arg and we can use full
  // bf16 vectors.
  const hn::Repartition<MatT, decltype(df)> d;
  VF unused_sum1 = hn::Zero(df);
#else
  // Emulated: use half-vectors of bf16 because we cannot afford two sums for
  // each c##.
  const hn::Rebind<MatT, decltype(df)> d;
  HWY_DASSERT(Lanes(d) == Lanes(df));
#endif

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

  const MatT* HWY_RESTRICT tile_a = A + stride_a * row_a;
  const MatT* HWY_RESTRICT tile_b = B + stride_b * row_b_col_c;

  // Loop over columns of A and columns of the transposed B, in steps of N.
  // Accumulates into the c## vectors.
  HWY_UNROLL(1)
  for (size_t col_ab = 0; col_ab < kColsA_RowsB; col_ab += N) {
#if GEMMA_NATIVE_BF16
    using V = hn::Vec<decltype(d)>;
    const V b0 = hn::LoadU(d, tile_b + stride_b * 0 + col_ab);
    const V b1 = hn::LoadU(d, tile_b + stride_b * 1 + col_ab);
    const V b2 = hn::LoadU(d, tile_b + stride_b * 2 + col_ab);
    const V b3 = hn::LoadU(d, tile_b + stride_b * 3 + col_ab);

    const V a0 = hn::LoadU(d, tile_a + stride_a * 0 + col_ab);
    c00 = hn::ReorderWidenMulAccumulate(df, a0, b0, c00, unused_sum1);
    c01 = hn::ReorderWidenMulAccumulate(df, a0, b1, c01, unused_sum1);
    c02 = hn::ReorderWidenMulAccumulate(df, a0, b2, c02, unused_sum1);
    c03 = hn::ReorderWidenMulAccumulate(df, a0, b3, c03, unused_sum1);
    if (kNumRows == 1) continue;

    const V a1 = hn::LoadU(d, tile_a + stride_a * 1 + col_ab);
    c10 = hn::ReorderWidenMulAccumulate(df, a1, b0, c10, unused_sum1);
    c11 = hn::ReorderWidenMulAccumulate(df, a1, b1, c11, unused_sum1);
    c12 = hn::ReorderWidenMulAccumulate(df, a1, b2, c12, unused_sum1);
    c13 = hn::ReorderWidenMulAccumulate(df, a1, b3, c13, unused_sum1);
    if (kNumRows == 2) continue;

    const V a2 = hn::LoadU(d, tile_a + stride_a * 2 + col_ab);
    c20 = hn::ReorderWidenMulAccumulate(df, a2, b0, c20, unused_sum1);
    c21 = hn::ReorderWidenMulAccumulate(df, a2, b1, c21, unused_sum1);
    c22 = hn::ReorderWidenMulAccumulate(df, a2, b2, c22, unused_sum1);
    c23 = hn::ReorderWidenMulAccumulate(df, a2, b3, c23, unused_sum1);
    if (kNumRows == 3) continue;

    const V a3 = hn::LoadU(d, tile_a + stride_a * 3 + col_ab);
    c30 = hn::ReorderWidenMulAccumulate(df, a3, b0, c30, unused_sum1);
    c31 = hn::ReorderWidenMulAccumulate(df, a3, b1, c31, unused_sum1);
    c32 = hn::ReorderWidenMulAccumulate(df, a3, b2, c32, unused_sum1);
    c33 = hn::ReorderWidenMulAccumulate(df, a3, b3, c33, unused_sum1);
#else   // Emulated: promote bf16 to f32
    const VF b0 =
        hn::PromoteTo(df, hn::LoadU(d, tile_b + stride_b * 0 + col_ab));
    const VF b1 =
        hn::PromoteTo(df, hn::LoadU(d, tile_b + stride_b * 1 + col_ab));
    const VF b2 =
        hn::PromoteTo(df, hn::LoadU(d, tile_b + stride_b * 2 + col_ab));
    const VF b3 =
        hn::PromoteTo(df, hn::LoadU(d, tile_b + stride_b * 3 + col_ab));

    const VF a0 =
        hn::PromoteTo(df, hn::LoadU(d, tile_a + stride_a * 0 + col_ab));
    c00 = hn::MulAdd(a0, b0, c00);
    c01 = hn::MulAdd(a0, b1, c01);
    c02 = hn::MulAdd(a0, b2, c02);
    c03 = hn::MulAdd(a0, b3, c03);
    if (kNumRows == 1) continue;

    const VF a1 =
        hn::PromoteTo(df, hn::LoadU(d, tile_a + stride_a * 1 + col_ab));
    c10 = hn::MulAdd(a1, b0, c10);
    c11 = hn::MulAdd(a1, b1, c11);
    c12 = hn::MulAdd(a1, b2, c12);
    c13 = hn::MulAdd(a1, b3, c13);
    if (kNumRows == 2) continue;

    const VF a2 =
        hn::PromoteTo(df, hn::LoadU(d, tile_a + stride_a * 2 + col_ab));
    c20 = hn::MulAdd(a2, b0, c20);
    c21 = hn::MulAdd(a2, b1, c21);
    c22 = hn::MulAdd(a2, b2, c22);
    c23 = hn::MulAdd(a2, b3, c23);
    if (kNumRows == 3) continue;

    const VF a3 =
        hn::PromoteTo(df, hn::LoadU(d, tile_a + stride_a * 3 + col_ab));
    c30 = hn::MulAdd(a3, b0, c30);
    c31 = hn::MulAdd(a3, b1, c31);
    c32 = hn::MulAdd(a3, b2, c32);
    c33 = hn::MulAdd(a3, b3, c33);
#endif  // !GEMMA_NATIVE_BF16
  }

#if GEMMA_NATIVE_BF16
  // Ensure sum1 was indeed unused.
  HWY_DASSERT(hn::AllTrue(df, hn::Eq(unused_sum1, hn::Zero(df))));
#endif

  float* HWY_RESTRICT tile_c = C + stride_c * row_a + row_b_col_c;
  StoreHorizontalSums<kNumRows>(
      df, c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22,
      c23, c30, c31, c32, c33, tile_c, stride_c);
}

// Same as above, but with mixed Mat types: (f32, sfp)).
template <size_t kNumRows, size_t kColsA_RowsB, typename MatTA,
          HWY_IF_F32(MatTA)>
HWY_INLINE void GEMM_4x4_Tile(const MatTA* HWY_RESTRICT A,
                              const SfpStream* HWY_RESTRICT B,
                              float* HWY_RESTRICT C, const size_t idx_tile,
                              const size_t xtiles, const size_t stride_a,
                              const size_t stride_b, const size_t stride_c) {
  constexpr size_t kRegRows = 4;
  constexpr size_t kRegCols = 4;
  static_assert(kNumRows <= kRegRows);

  // Top-left of tile is (row_a, col_ab) for A, and (row_b_col_c, col_ab) for B.
  const size_t row_a = idx_tile / xtiles * kRegRows;
  const size_t row_b_col_c = idx_tile % xtiles * kRegCols;

  const hn::ScalableTag<MatTA> d;
  const size_t N = Lanes(d);
  using V = hn::Vec<decltype(d)>;

  V c00 = hn::Zero(d);
  V c01 = hn::Zero(d);
  V c02 = hn::Zero(d);
  V c03 = hn::Zero(d);

  V c10 = hn::Zero(d);
  V c11 = hn::Zero(d);
  V c12 = hn::Zero(d);
  V c13 = hn::Zero(d);

  V c20 = hn::Zero(d);
  V c21 = hn::Zero(d);
  V c22 = hn::Zero(d);
  V c23 = hn::Zero(d);

  V c30 = hn::Zero(d);
  V c31 = hn::Zero(d);
  V c32 = hn::Zero(d);
  V c33 = hn::Zero(d);

  const MatTA* HWY_RESTRICT tile_a = A + stride_a * row_a;

  hwy::AlignedFreeUniquePtr<float[]> tile_b_unique_ptr =
      hwy::AllocateAligned<float>(kRegRows * kColsA_RowsB);
  CompressTraits<SfpStream>::Decompress(
      d,
      /*in_capacity=*/0, B, stride_b * row_b_col_c, tile_b_unique_ptr.get(),
      kRegRows * kColsA_RowsB);
  const float* HWY_RESTRICT tile_b = tile_b_unique_ptr.get();

  // Loop over columns of A and columns of the transposed B, in steps of N.
  // Accumulates into the c## vectors.
  HWY_UNROLL(1)
  for (size_t col_ab = 0; col_ab < kColsA_RowsB; col_ab += N) {
    const V b0 = hn::LoadU(d, tile_b + stride_b * 0 + col_ab);
    const V b1 = hn::LoadU(d, tile_b + stride_b * 1 + col_ab);
    const V b2 = hn::LoadU(d, tile_b + stride_b * 2 + col_ab);
    const V b3 = hn::LoadU(d, tile_b + stride_b * 3 + col_ab);

    const V a0 = hn::LoadU(d, tile_a + stride_a * 0 + col_ab);
    c00 = hn::MulAdd(a0, b0, c00);
    c01 = hn::MulAdd(a0, b1, c01);
    c02 = hn::MulAdd(a0, b2, c02);
    c03 = hn::MulAdd(a0, b3, c03);
    if (kNumRows == 1) continue;

    const V a1 = hn::LoadU(d, tile_a + stride_a * 1 + col_ab);
    c10 = hn::MulAdd(a1, b0, c10);
    c11 = hn::MulAdd(a1, b1, c11);
    c12 = hn::MulAdd(a1, b2, c12);
    c13 = hn::MulAdd(a1, b3, c13);
    if (kNumRows == 2) continue;

    const V a2 = hn::LoadU(d, tile_a + stride_a * 2 + col_ab);
    c20 = hn::MulAdd(a2, b0, c20);
    c21 = hn::MulAdd(a2, b1, c21);
    c22 = hn::MulAdd(a2, b2, c22);
    c23 = hn::MulAdd(a2, b3, c23);
    if (kNumRows == 3) continue;

    const V a3 = hn::LoadU(d, tile_a + stride_a * 3 + col_ab);
    c30 = hn::MulAdd(a3, b0, c30);
    c31 = hn::MulAdd(a3, b1, c31);
    c32 = hn::MulAdd(a3, b2, c32);
    c33 = hn::MulAdd(a3, b3, c33);
  }

  float* HWY_RESTRICT tile_c = C + stride_c * row_a + row_b_col_c;
  StoreHorizontalSums<kNumRows>(
      d, c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22,
      c23, c30, c31, c32, c33, tile_c, stride_c);
}

// Same as above, but with mixed Mat types: (bf16, sfp)).
template <size_t kNumRows, size_t kColsA_RowsB, typename MatTA,
          HWY_IF_BF16(MatTA)>
HWY_INLINE void GEMM_4x4_Tile(const MatTA* HWY_RESTRICT A,
                              const SfpStream* HWY_RESTRICT B,
                              float* HWY_RESTRICT C, const size_t idx_tile,
                              const size_t xtiles, const size_t stride_a,
                              const size_t stride_b, const size_t stride_c) {
  constexpr size_t kRegRows = 4;
  constexpr size_t kRegCols = 4;
  static_assert(kNumRows <= kRegRows);

  // Top-left of tile is (row_a, col_ab) for A, and (row_b_col_c, col_ab) for B.
  const size_t row_a = idx_tile / xtiles * kRegRows;
  const size_t row_b_col_c = idx_tile % xtiles * kRegCols;

  const hn::ScalableTag<float> d32;
  const size_t N = Lanes(d32);
  using V = hn::Vec<decltype(d32)>;
  // TODO: Using half-vectors for now, it might be faster to
  // PromoteLower/UpperTo, and more so to PromoteEven/OddTo if we have packed B
  // accordingly.
  const hn::Rebind<MatTA, decltype(d32)> d16;
  HWY_DASSERT(Lanes(d16) == Lanes(d32));

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

  const MatTA* HWY_RESTRICT tile_a = A + stride_a * row_a;

  hwy::AlignedFreeUniquePtr<float[]> tile_b_unique_ptr =
      hwy::AllocateAligned<float>(kRegRows * kColsA_RowsB);
  CompressTraits<SfpStream>::Decompress(
      d32,
      /*in_capacity=*/0, B, stride_b * row_b_col_c, tile_b_unique_ptr.get(),
      kRegRows * kColsA_RowsB);
  const float* HWY_RESTRICT tile_b = tile_b_unique_ptr.get();

  // Loop over columns of A and columns of the transposed B, in steps of N.
  // Accumulates into the c## vectors.
  HWY_UNROLL(1)
  for (size_t col_ab = 0; col_ab < kColsA_RowsB; col_ab += N) {
    const V b0 = hn::LoadU(d32, tile_b + stride_b * 0 + col_ab);
    const V b1 = hn::LoadU(d32, tile_b + stride_b * 1 + col_ab);
    const V b2 = hn::LoadU(d32, tile_b + stride_b * 2 + col_ab);
    const V b3 = hn::LoadU(d32, tile_b + stride_b * 3 + col_ab);

    const V a0 =
        hn::PromoteTo(d32, hn::LoadU(d16, tile_a + stride_a * 0 + col_ab));
    c00 = hn::MulAdd(a0, b0, c00);
    c01 = hn::MulAdd(a0, b1, c01);
    c02 = hn::MulAdd(a0, b2, c02);
    c03 = hn::MulAdd(a0, b3, c03);
    if (kNumRows == 1) continue;

    const V a1 =
        hn::PromoteTo(d32, hn::LoadU(d16, tile_a + stride_a * 1 + col_ab));
    c10 = hn::MulAdd(a1, b0, c10);
    c11 = hn::MulAdd(a1, b1, c11);
    c12 = hn::MulAdd(a1, b2, c12);
    c13 = hn::MulAdd(a1, b3, c13);
    if (kNumRows == 2) continue;

    const V a2 =
        hn::PromoteTo(d32, hn::LoadU(d16, tile_a + stride_a * 2 + col_ab));
    c20 = hn::MulAdd(a2, b0, c20);
    c21 = hn::MulAdd(a2, b1, c21);
    c22 = hn::MulAdd(a2, b2, c22);
    c23 = hn::MulAdd(a2, b3, c23);
    if (kNumRows == 3) continue;

    const V a3 =
        hn::PromoteTo(d32, hn::LoadU(d16, tile_a + stride_a * 3 + col_ab));
    c30 = hn::MulAdd(a3, b0, c30);
    c31 = hn::MulAdd(a3, b1, c31);
    c32 = hn::MulAdd(a3, b2, c32);
    c33 = hn::MulAdd(a3, b3, c33);
  }

  float* HWY_RESTRICT tile_c = C + stride_c * row_a + row_b_col_c;
  StoreHorizontalSums<kNumRows>(
      d32, c00, c01, c02, c03, c10, c11, c12, c13, c20, c21,
      c22, c23, c30, c31, c32, c33, tile_c, stride_c);
}

// Same as above, but with mixed Mat types: (f32, bf16).
template <size_t kNumRows, size_t kColsA_RowsB, typename MatTA,
          HWY_IF_F32(MatTA),
          typename MatTB, HWY_IF_BF16(MatTB)>
HWY_INLINE void GEMM_4x4_Tile(const MatTA* HWY_RESTRICT A,
                              const MatTB* HWY_RESTRICT B,
                              float* HWY_RESTRICT C, const size_t idx_tile,
                              const size_t xtiles, const size_t stride_a,
                              const size_t stride_b, const size_t stride_c) {
  constexpr size_t kRegRows = 4;
  constexpr size_t kRegCols = 4;
  static_assert(kNumRows <= kRegRows);

  // Top-left of tile is (row_a, col_ab) for A, and (row_b_col_c, col_ab) for B.
  const size_t row_a = idx_tile / xtiles * kRegRows;
  const size_t row_b_col_c = idx_tile % xtiles * kRegCols;

  const hn::ScalableTag<float> d32;
  using VF = hn::Vec<decltype(d32)>;

  // TODO: Using half-vectors for now, it might be faster to
  // PromoteLower/UpperTo, and more so to PromoteEven/OddTo if we have packed B
  // accordingly.
  const hn::Rebind<MatTB, decltype(d32)> d16;
  HWY_DASSERT(Lanes(d16) == Lanes(d32));

  const size_t N = Lanes(d16);

  VF c00 = hn::Zero(d32);
  VF c01 = hn::Zero(d32);
  VF c02 = hn::Zero(d32);
  VF c03 = hn::Zero(d32);

  VF c10 = hn::Zero(d32);
  VF c11 = hn::Zero(d32);
  VF c12 = hn::Zero(d32);
  VF c13 = hn::Zero(d32);

  VF c20 = hn::Zero(d32);
  VF c21 = hn::Zero(d32);
  VF c22 = hn::Zero(d32);
  VF c23 = hn::Zero(d32);

  VF c30 = hn::Zero(d32);
  VF c31 = hn::Zero(d32);
  VF c32 = hn::Zero(d32);
  VF c33 = hn::Zero(d32);

  const MatTA* HWY_RESTRICT tile_a = A + stride_a * row_a;
  const MatTB* HWY_RESTRICT tile_b = B + stride_b * row_b_col_c;

  // Loop over columns of A and columns of the transposed B, in steps of N.
  // Accumulates into the c## vectors.
  HWY_UNROLL(1)
  for (size_t col_ab = 0; col_ab < kColsA_RowsB; col_ab += N) {
    // Promote bf16 to f32
    const VF b0 =
        hn::PromoteTo(d32, hn::LoadU(d16, tile_b + stride_b * 0 + col_ab));
    const VF b1 =
        hn::PromoteTo(d32, hn::LoadU(d16, tile_b + stride_b * 1 + col_ab));
    const VF b2 =
        hn::PromoteTo(d32, hn::LoadU(d16, tile_b + stride_b * 2 + col_ab));
    const VF b3 =
        hn::PromoteTo(d32, hn::LoadU(d16, tile_b + stride_b * 3 + col_ab));

    const VF a0 = hn::LoadU(d32, tile_a + stride_a * 0 + col_ab);
    c00 = hn::MulAdd(a0, b0, c00);
    c01 = hn::MulAdd(a0, b1, c01);
    c02 = hn::MulAdd(a0, b2, c02);
    c03 = hn::MulAdd(a0, b3, c03);
    if (kNumRows == 1) continue;

    const VF a1 = hn::LoadU(d32, tile_a + stride_a * 1 + col_ab);
    c10 = hn::MulAdd(a1, b0, c10);
    c11 = hn::MulAdd(a1, b1, c11);
    c12 = hn::MulAdd(a1, b2, c12);
    c13 = hn::MulAdd(a1, b3, c13);
    if (kNumRows == 2) continue;

    const VF a2 = hn::LoadU(d32, tile_a + stride_a * 2 + col_ab);
    c20 = hn::MulAdd(a2, b0, c20);
    c21 = hn::MulAdd(a2, b1, c21);
    c22 = hn::MulAdd(a2, b2, c22);
    c23 = hn::MulAdd(a2, b3, c23);
    if (kNumRows == 3) continue;

    const VF a3 = hn::LoadU(d32, tile_a + stride_a * 3 + col_ab);
    c30 = hn::MulAdd(a3, b0, c30);
    c31 = hn::MulAdd(a3, b1, c31);
    c32 = hn::MulAdd(a3, b2, c32);
    c33 = hn::MulAdd(a3, b3, c33);
  }

  float* HWY_RESTRICT tile_c = C + stride_c * row_a + row_b_col_c;
  StoreHorizontalSums<kNumRows>(
      d32, c00, c01, c02, c03, c10, c11, c12, c13, c20, c21,
      c22, c23, c30, c31, c32, c33, tile_c, stride_c);
}

// Tiled 4x4 GEMM. Typically kRowsAC is 4..512, kColsA_RowsB is 3k or 24k, and
// kColsBC is 24k or 3k. Note: B is transposed (column-major).
// This function loops over all tiles (static scheduling). TODO(janwas): we can
// possibly remove this if ThreadPool(0) is as efficient as the loop.
template <size_t kRowsAC, size_t kColsA_RowsB, size_t kColsBC, typename MatT>
void GEMM_4x4_Static(const MatT* HWY_RESTRICT A, const MatT* HWY_RESTRICT B,
                     MatT* HWY_RESTRICT C) {
  const hn::ScalableTag<MatT> d;
  const size_t N = hn::Lanes(d);  // column step size
  constexpr size_t kRegRows = 4;
  constexpr size_t kRegCols = 4;  // in vectors

  static_assert(kRowsAC % kRegRows == 0);
  static_assert(kColsBC % kRegCols == 0);
  HWY_ASSERT(kColsA_RowsB % (N * kRegCols) == 0);
  constexpr size_t kTilesY = kRowsAC / kRegRows;
  constexpr size_t kTilesX = kColsBC / kRegCols;
  constexpr size_t kTiles = kTilesX * kTilesY;

  constexpr size_t kStrideA = kColsA_RowsB;
  constexpr size_t kStrideB = kColsA_RowsB;  // B is column-major
  constexpr size_t kStrideC = kColsBC;

  HWY_UNROLL(1)
  for (size_t idx_tile = 0; idx_tile < kTiles; ++idx_tile) {
    GEMM_4x4_Tile<kRegRows, kColsA_RowsB>(
        A, B, C, idx_tile, kTilesX, kStrideA, kStrideB, kStrideC);
  }
}

// Tiled 4x4 GEMM. Typically kRowsAC is 4..512, kColsA_RowsB is 3k or 24k, and
// kColsBC is 24k or 3k. Note: B is transposed (column-major).
// This function processes tiles in parallel with a work-stealing thread pool.
template <size_t kRowsAC, size_t kColsA_RowsB, size_t kColsBC, typename MatTA,
          typename MatTB, typename OutT>
HWY_NOINLINE void MatMul_4x4(const MatTA* HWY_RESTRICT A,
                             const MatTB* HWY_RESTRICT B, OutT* HWY_RESTRICT C,
                             hwy::ThreadPool& pool) {
  // Process reg-sized tiles of C in parallel. We currently write C directly,
  // which touches more memory than fits in L3. TODO: add another level of loops
  // so that we finish one L3-sized piece of C at a time.
  const hn::ScalableTag<MatTA> d;
  const size_t N = Lanes(d);
  constexpr size_t kRegRows = 4;
  constexpr size_t kRegCols = 4;  // in vectors

  static_assert(kRowsAC % kRegRows == 0);
  static_assert(kColsBC % kRegCols == 0);
  HWY_ASSERT(kColsA_RowsB % (N * kRegCols) == 0);
  const size_t kTilesY = kRowsAC / kRegRows;
  const size_t kTilesX = kColsBC / kRegCols;
  const size_t kTiles = kTilesX * kTilesY;

  constexpr size_t kStrideA = kColsA_RowsB;
  constexpr size_t kStrideB = kColsA_RowsB;
  constexpr size_t kStrideC = kColsBC;

  pool.Run(0, kTiles, [&](const uint64_t idx_tile, size_t /*thread*/) HWY_ATTR {
    // Computes the finished product of one 4x4N tile and writes to C.
    GEMM_4x4_Tile<kRegRows, kColsA_RowsB>(
        A, B, C, idx_tile, kTilesX, kStrideA, kStrideB, kStrideC);
  });
}

// Tiled 4x4 GEMM. Typically batch_size is 1..512, kColsA_RowsB is 3k or 24k,
// and kColsBC is 24k or 3k. Note: B is transposed (column-major).
// NOTE that batch_size is the number of rows of A and C.
// This function processes tiles in parallel with a work-stealing thread pool.
template <size_t kColsA_RowsB, size_t kColsBC, typename MatTA,
          typename MatTB, typename OutT>
HWY_NOINLINE void MatMul_4x4_Batch(
    size_t batch_size, const MatTA* HWY_RESTRICT A, const MatTB* HWY_RESTRICT B,
    OutT* HWY_RESTRICT C, hwy::ThreadPool& pool) {
  // Process reg-sized tiles of C in parallel. We currently write C directly,
  // which touches more memory than fits in L3. TODO: add another level of loops
  // so that we finish one L3-sized piece of C at a time.
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
        GEMM_4x4_Tile<1, kColsA_RowsB>(
            A, B, C, idx_tile, kTilesX, kStrideA, kStrideB, kStrideC);
        break;
      case 2:
        GEMM_4x4_Tile<2, kColsA_RowsB>(
            A, B, C, idx_tile, kTilesX, kStrideA, kStrideB, kStrideC);
        break;
      case 3:
        GEMM_4x4_Tile<3, kColsA_RowsB>(
            A, B, C, idx_tile, kTilesX, kStrideA, kStrideB, kStrideC);
        break;
      default:
        GEMM_4x4_Tile<4, kColsA_RowsB>(
            A, B, C, idx_tile, kTilesX, kStrideA, kStrideB, kStrideC);
    }
  });
}

// Largely unoptimized; reordered innermost loops nets ~5-10X speedup on
// ops_test across instruction sets.
template <size_t kM, size_t kN, size_t kK, typename MatTA, typename MatTB>
HWY_INLINE void MatMulSlow(const MatTA* HWY_RESTRICT a,
                           const MatTB* HWY_RESTRICT b,
                           float* HWY_RESTRICT out) {
  for (size_t i = 0; i < kM; ++i) {
    for (size_t k = 0; k < kN; ++k) {
      for (size_t j = 0; j < kK; ++j) {
        const float a1 = hwy::ConvertScalarTo<float>(a[i * kN + k]);
        const float b1 = hwy::ConvertScalarTo<float>(b[k * kK + j]);
        out[i * kK + j] += a1 * b1;
      }
    }
  }
}

template <size_t kM, size_t kN, size_t kK, typename MatTA>
HWY_INLINE void MatMulSlow(const MatTA* HWY_RESTRICT a,
                           const SfpStream* HWY_RESTRICT b_sfp_stream,
                           float* HWY_RESTRICT out) {
  const hn::ScalableTag<float> d;
  hwy::AlignedFreeUniquePtr<float[]> b = hwy::AllocateAligned<float>(kK * kN);
  CompressTraits<SfpStream>::Decompress(d,
                                        /*in_capacity=*/0, b_sfp_stream, 0,
                                        b.get(), kK * kN);
  MatMulSlow<kM, kN, kK>(a, b.get(), out);
}

// Largely unoptimized; reordered innermost loops nets ~5-10X speedup on
// ops_test across instruction sets.
template <size_t kN, size_t kK, typename MatTA, typename MatTB>
HWY_INLINE void MatMulSlowBatch(size_t batch_size, const MatTA* HWY_RESTRICT a,
                                const MatTB* HWY_RESTRICT b,
                                float* HWY_RESTRICT out) {
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t k = 0; k < kN; ++k) {
      for (size_t j = 0; j < kK; ++j) {
        const float a1 = hwy::ConvertScalarTo<float>(a[i * kN + k]);
        const float b1 = hwy::ConvertScalarTo<float>(b[k * kK + j]);
        out[i * kK + j] += a1 * b1;
      }
    }
  }
}

template <size_t kN, size_t kK, typename MatTA>
HWY_INLINE void MatMulSlowBatch(size_t batch_size, const MatTA* HWY_RESTRICT a,
                                const SfpStream* HWY_RESTRICT b_sfp_stream,
                                float* HWY_RESTRICT out) {
  const hn::ScalableTag<float> d;
  hwy::AlignedFreeUniquePtr<float[]> b = hwy::AllocateAligned<float>(kK * kN);
  CompressTraits<SfpStream>::Decompress(d,
                                        /*in_capacity=*/0, b_sfp_stream, 0,
                                        b.get(), kK * kN);
  MatMulSlowBatch<kN, kK>(batch_size, a, b.get(), out);
}

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

template <class D, HWY_IF_F32_D(D)>
static HWY_INLINE hn::Vec<D> Gelu(D d, hn::Vec<D> v) {
  const hn::Vec<D> kMul = hn::Set(d, 0.044715f);
  const hn::Vec<D> kSqrt2OverPi = hn::Set(d, 0.797884560804236f);
  const hn::Vec<D> kHalf = hn::Set(d, 0.5f);

  // tanh approximation matches training.
  const hn::Vec<D> v3 = hn::Mul(hn::Mul(v, v), v);
  const hn::Vec<D> arg = hn::Mul(kSqrt2OverPi, hn::MulAdd(kMul, v3, v));
  // 0.5 * (1 + tan) = MulAdd(0.5, tan, 0.5).
  const hn::Vec<D> cdf = hn::MulAdd(kHalf, hn::Tanh(d, arg), kHalf);
  return hn::Mul(v, cdf);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void Gelu(float* HWY_RESTRICT x,
                                               size_t size) {
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  hn::Transform(D(), x, size,
                [](D d, hn::Vec<D> v) HWY_ATTR { return Gelu(d, v); });
}

// out[i] = BF(mul[i] * Gelu(gelu_in[i]))
static HWY_NOINLINE HWY_MAYBE_UNUSED void GeluMulToBF16(
    const float* HWY_RESTRICT gelu_in, const float* HWY_RESTRICT mul,
    hwy::bfloat16_t* HWY_RESTRICT out, size_t size) {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> df;
  const hn::Repartition<hwy::bfloat16_t, decltype(df)> dbf;
  const size_t NF = hn::Lanes(df);
  using VF = hn::Vec<decltype(df)>;

  size_t i = 0;
  if (size >= 2 * NF) {
    for (; i <= size - 2 * NF; i += 2 * NF) {
      const VF mul0 = hn::LoadU(df, mul + i);
      const VF mul1 = hn::LoadU(df, mul + i + NF);
      const VF g0 = hn::Mul(mul0, Gelu(df, hn::LoadU(df, gelu_in + i)));
      const VF g1 = hn::Mul(mul1, Gelu(df, hn::LoadU(df, gelu_in + i + NF)));
      const hn::Vec<decltype(dbf)> bf = hn::OrderedDemote2To(dbf, g0, g1);
      hn::StoreU(bf, dbf, out + i);
    }
  }
  if (i != size) {
    const size_t remaining = size - i;
    const VF mul0 = hn::LoadN(df, mul + i, remaining);
    const VF g0 =
        hn::Mul(mul0, Gelu(df, hn::LoadN(df, gelu_in + i, remaining)));
    const hn::Half<decltype(dbf)> dbfh;
    const hn::Vec<decltype(dbfh)> bfh = hn::DemoteTo(dbfh, g0);
    hn::StoreN(bfh, dbfh, out + i, remaining);
  }
}

template <class D, HWY_IF_F32_D(D)>
static HWY_INLINE hn::Vec<D> Sigmoid(D d, hn::Vec<D> v) {
  using VF = hn::Vec<D>;
  // Chebyshev polynomial coefficients for rational approximation
  const VF c0 = hn::Set(d, 0.00949107017368078f);
  const VF c1 = hn::Set(d, 0.0654858946800232f);
  const VF c2 = hn::Set(d, 0.231547489762306f - 0.00949107017368078f);
  const VF c3 = hn::Set(d, 0.530778527259827f);
  const VF c4 = hn::Set(d, 0.855334937572479f);
  const VF c5 = hn::Set(d, 0.500000894069672f);

  const VF d0 = hn::Set(d, 0.130970627069473f);
  const VF d1 = hn::Set(d, 3.99615288415589e-07f);
  const VF d2 = hn::Set(d, 1.06155431270599f - 0.130970627069473f);
  const VF d3 = hn::Set(d, 1.35144250634767e-06f);
  const VF d4 = hn::Set(d, 1);

  // The approximation works in range -12..12, but the input value is clamped
  // in -11.5..11.5 since the approximation slightly overshoots after that.
  // The function is nearly 0 for input values below -11.5 and nearly 1 for
  // input values above 11.5.
  const VF invtwelve = hn::Set(d, 1.0f / 12.0f);
  const VF lo = hn::Set(d, -11.5f);
  const VF hi = hn::Set(d, 11.5f);

  VF f = hn::Clamp(v, lo, hi);
  f = hn::Mul(f, invtwelve);
  VF f2 = hn::Add(f, f);

  VF a1 = hn::MulAdd(f2, c0, c1);
  VF a2 = hn::MulAdd(f2, a1, c2);
  VF a3 = hn::Sub(hn::MulAdd(f2, a2, c3), a1);
  VF a4 = hn::Sub(hn::MulAdd(f2, a3, c4), a2);
  VF f0 = hn::Sub(hn::MulAdd(f, a4, c5), a3);

  VF b1 = hn::MulAdd(f2, d0, d1);
  VF b2 = hn::MulAdd(f2, b1, d2);
  VF b3 = hn::Sub(hn::MulAdd(f2, b2, d3), b1);
  VF f1 = hn::Sub(hn::MulAdd(f, b3, d4), b2);

  return hn::Div(f0, f1);
}

// Sigmoid using the logistic function 1 / (1 + exp(-x[i]))
static HWY_NOINLINE HWY_MAYBE_UNUSED void Sigmoid(float* HWY_RESTRICT x,
                                                  size_t size) {
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  hn::Transform(D(), x, size,
                [](D d, hn::Vec<D> v) HWY_ATTR { return Sigmoid(d, v); });
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

static HWY_NOINLINE HWY_MAYBE_UNUSED float Dot(const float* HWY_RESTRICT a,
                                               const float* HWY_RESTRICT b,
                                               size_t size) {
  const hn::ScalableTag<float> d;
  HWY_DASSERT(size >= hn::Lanes(d));
  HWY_DASSERT(size % hn::Lanes(d) == 0);
  constexpr int kAssumptions =
      hn::Dot::kAtLeastOneVector | hn::Dot::kMultipleOfVector;
  return hn::Dot::Compute<kAssumptions>(d, a, b, size);
}

// = Dot(a, a, size), but that is not allowed due to HWY_RESTRICT.
static HWY_NOINLINE HWY_MAYBE_UNUSED float SquaredL2(
    const float* HWY_RESTRICT a, size_t size) {
  const hn::ScalableTag<float> d;
  using V = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);
  HWY_DASSERT(size >= 2 * N);
  HWY_DASSERT(size % (2 * N) == 0);

  V sum0 = hn::Zero(d);
  V sum1 = hn::Zero(d);
  for (size_t i = 0; i <= size - 2 * N; i += 2 * N) {
    const V a0 = hn::LoadU(d, a + i);
    sum0 = hn::MulAdd(a0, a0, sum0);
    const V a1 = hn::LoadU(d, a + i + N);
    sum1 = hn::MulAdd(a1, a1, sum1);
  }

  return hn::ReduceSum(d, hn::Add(sum0, sum1));
}

// float, float -> float; simple loop.
static HWY_NOINLINE HWY_MAYBE_UNUSED void RMSNorm(
    const float* HWY_RESTRICT x, const float* HWY_RESTRICT weight,
    float* HWY_RESTRICT out, size_t size) {
  constexpr float kEps = 1e-6f;
  float ss = SquaredL2(x, size);
  ss = 1.0f / sqrtf(ss / StaticCast<float>(size) + kEps);
  for (size_t j = 0; j < size; j++) {
    // Note 1.0f centering here
    out[j] = (1.0f + weight[j]) * (ss * x[j]);
  }
}

// x=f, w=bf16 -> out=f
static HWY_NOINLINE HWY_MAYBE_UNUSED void RMSNorm(
    const float* HWY_RESTRICT x, const hwy::bfloat16_t* HWY_RESTRICT weight,
    float* HWY_RESTRICT out, size_t size) {
  namespace hn = hwy::HWY_NAMESPACE;

  constexpr float kEps = 1e-6f;
  constexpr size_t kUnrollSize = 2;

  const hn::ScalableTag<hwy::bfloat16_t> dbf;
  const hn::Repartition<float, decltype(dbf)> df32;
  const size_t N32 = hn::Lanes(df32);

  const float ss = SquaredL2(x, size);
  const auto vss =
      hn::Set(df32, 1.0f / sqrtf(ss / StaticCast<float>(size) + kEps));

  HWY_DASSERT(size % (kUnrollSize * MaxLanes(df32)) == 0);
  for (size_t i = 0; i < size; i += kUnrollSize * N32) {
    const hn::Vec<decltype(dbf)> w16 = hn::LoadU(dbf, weight + i);
    const auto w0 = hn::PromoteLowerTo(df32, w16);
    const auto w1 = hn::PromoteUpperTo(df32, w16);
    const auto m0 = hn::Mul(vss, hn::LoadU(df32, x + i));
    const auto m1 = hn::Mul(vss, hn::LoadU(df32, x + i + N32));

    // (1+weight) * m = m + weight*m = one FMA.
    hn::StoreU(hn::MulAdd(m0, w0, m0), df32, out + i);
    hn::StoreU(hn::MulAdd(m1, w1, m1), df32, out + i + N32);
  }
}

// float -> float; simple loop.
static HWY_NOINLINE HWY_MAYBE_UNUSED void RMSNormInplace(
    const float* HWY_RESTRICT weight, float* HWY_RESTRICT inout, size_t size) {
  constexpr float kEps = 1e-6f;
  float ss = SquaredL2(inout, size);
  ss = 1.0f / sqrtf(ss / StaticCast<float>(size) + kEps);
  for (size_t j = 0; j < size; j++) {
    // Note 1.0f centering here
    inout[j] = (1.0f + weight[j]) * (ss * inout[j]);
  }
}

// w=bf16 -> f
static HWY_NOINLINE HWY_MAYBE_UNUSED void RMSNormInplace(
    const hwy::bfloat16_t* HWY_RESTRICT weight, float* HWY_RESTRICT inout,
    const size_t size) {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<hwy::bfloat16_t> dbf;
  const hn::Repartition<float, decltype(dbf)> df32;
  using VF = hn::Vec<decltype(df32)>;
  const size_t N32 = hn::Lanes(df32);

  constexpr float kEps = 1e-6f;
  const float ss = SquaredL2(inout, size);
  const VF vss =
      hn::Set(df32, 1.0f / sqrtf(ss / StaticCast<float>(size) + kEps));

  HWY_DASSERT(size % (2 * MaxLanes(df32)) == 0);
  for (size_t i = 0; i < size; i += 2 * N32) {
    const hn::Vec<decltype(dbf)> w16 = hn::LoadU(dbf, weight + i);
    const VF w0 = hn::PromoteLowerTo(df32, w16);
    const VF w1 = hn::PromoteUpperTo(df32, w16);
    const VF m0 = hn::Mul(vss, hn::LoadU(df32, inout + i));
    const VF m1 = hn::Mul(vss, hn::LoadU(df32, inout + i + N32));
    // (1+weight) * m = m + weight*m = one FMA.
    hn::StoreU(hn::MulAdd(m0, w0, m0), df32, inout + i);
    hn::StoreU(hn::MulAdd(m1, w1, m1), df32, inout + i + N32);
  }
}

// f, f -> bf
// TODO(janwas): consider generic function with adapter for loading bf16/f32
static HWY_NOINLINE HWY_MAYBE_UNUSED void RMSNorm(
    const float* HWY_RESTRICT x, const float* HWY_RESTRICT weight,
    hwy::bfloat16_t* HWY_RESTRICT out, const size_t size) {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<hwy::bfloat16_t> dbf;
  const hn::Repartition<float, decltype(dbf)> df32;
  using VF = hn::Vec<decltype(df32)>;
  const size_t N32 = hn::Lanes(df32);

  constexpr float kEps = 1e-6f;
  const float ss = SquaredL2(x, size);
  const VF vss =
      hn::Set(df32, 1.0f / sqrtf(ss / StaticCast<float>(size) + kEps));

  HWY_DASSERT(size % (2 * MaxLanes(df32)) == 0);
  for (size_t i = 0; i < size; i += 2 * N32) {
    const VF w0 = hn::LoadU(df32, weight + i);
    const VF w1 = hn::LoadU(df32, weight + i + N32);
    const VF m0 = hn::Mul(vss, hn::LoadU(df32, x + i));
    const VF m1 = hn::Mul(vss, hn::LoadU(df32, x + i + N32));
    // (1+weight) * m = m + weight*m = one FMA.
    const VF out0 = hn::MulAdd(m0, w0, m0);
    const VF out1 = hn::MulAdd(m1, w1, m1);
    hn::StoreU(hn::OrderedDemote2To(dbf, out0, out1), dbf, out + i);
  }
}

// x=f, w=bf16 -> bf16 to enable W16A16 MatVec.
static HWY_NOINLINE HWY_MAYBE_UNUSED void RMSNorm(
    const float* HWY_RESTRICT x, const hwy::bfloat16_t* HWY_RESTRICT weight,
    hwy::bfloat16_t* HWY_RESTRICT out, const size_t size) {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<hwy::bfloat16_t> dbf;
  const hn::Repartition<float, decltype(dbf)> df32;
  using VF = hn::Vec<decltype(df32)>;
  const size_t N32 = hn::Lanes(df32);

  constexpr float kEps = 1e-6f;
  const float ss = SquaredL2(x, size);
  const VF vss =
      hn::Set(df32, 1.0f / sqrtf(ss / StaticCast<float>(size) + kEps));

  HWY_DASSERT(size % (2 * MaxLanes(df32)) == 0);
  for (size_t i = 0; i < size; i += 2 * N32) {
    const hn::Vec<decltype(dbf)> w16 = hn::LoadU(dbf, weight + i);
    const VF w0 = hn::PromoteLowerTo(df32, w16);
    const VF w1 = hn::PromoteUpperTo(df32, w16);
    const VF m0 = hn::Mul(vss, hn::LoadU(df32, x + i));
    const VF m1 = hn::Mul(vss, hn::LoadU(df32, x + i + N32));
    // (1+weight) * m = m + weight*m = one FMA.
    const VF out0 = hn::MulAdd(m0, w0, m0);
    const VF out1 = hn::MulAdd(m1, w1, m1);
    hn::StoreU(hn::OrderedDemote2To(dbf, out0, out1), dbf, out + i);
  }
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void AddAbsolutePositionalEmbeddings(
    float* HWY_RESTRICT x, size_t dim_model, size_t pos) {
  const size_t num_timescales = dim_model / 2;
  const float log_timescale_increment =
      logf(10000.0f) /
      (num_timescales != 0 ? StaticCast<float>(num_timescales - 1) : 1.0f);
  for (size_t dim = 0; dim < num_timescales; ++dim) {
    const float inv_timescale =
        expf(StaticCast<float>(dim) * -log_timescale_increment);
    x[dim] += sinf(StaticCast<float>(pos) * inv_timescale);
    x[num_timescales + dim] += cosf(StaticCast<float>(pos) * inv_timescale);
  }
}

/* RoPE as in Rotary Position Embeddings from the RoFormer paper
   (https://arxiv.org/abs/2104.09864v5). The query and key vectors are rotated
   as a function of their absolute position using the rotation matrix R before
   the self-attention operation. R is a d x d matrix.

   R = cos(m*theta_1) -sin(m*theta_1) ...  0              0
       sin(m*theta_1)  cos(m*theta_1)
            0               0         ...  0              0
            0               0         ...  0              0
           ...
            0               0         ...  cos(m*theta_{d/2}) sin(m*theta_{d/2})
            0               0         ...  sin(m*theta_{d/2}) cos(m*theta_{d/2})

  Here theta_i = 10000^(-2(i-1)/d), where d is the dimension of the vector and
  i is the ith index of the vector.

  Applying the rotation matrix R to a vector v is equivalent to rotating every
  consecutive pair of dimensions of v i.e. v_{2i} and v_{2i+1} by an angle
  m*theta_i. However in the Gemma implementation we choose to rotate
  the pairs of dimensions v_{i} and v_{i + d//2} instead.

  pos parameter is deliberately an int because in the backward pass we
  call this with negative values (for the VJP calculation we need the transpose
  of this rotation matrix which is simply the same matrix with -pos parameter)
*/

static HWY_NOINLINE HWY_MAYBE_UNUSED void Rope(float* HWY_RESTRICT x,
                                               size_t dim_qkv, int pos) {
  HWY_DASSERT(dim_qkv % 2 == 0);
  const size_t half_dim_qkv = dim_qkv / 2;
  for (size_t dim = 0; dim < half_dim_qkv; ++dim) {
    const float freq_exponents =
        StaticCast<float>(2 * dim) / StaticCast<float>(dim_qkv);
    // Replacing with expf(ln(1E4) * freq_exponents) changes results noticeably.
    const float timescale = powf(10000.0f, freq_exponents);
    const float theta = StaticCast<float>(pos) / timescale;
    const float cos_val = cosf(theta);
    const float sin_val = sinf(theta);
    const float x0 = x[dim];
    const float x1 = x[dim + half_dim_qkv];
    x[dim] = x0 * cos_val - x1 * sin_val;
    x[dim + half_dim_qkv] = x0 * sin_val + x1 * cos_val;
  }
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void RopeAndMulBy(const float mul,
                                                       float* HWY_RESTRICT x,
                                                       size_t dim_qkv,
                                                       int pos) {
  HWY_DASSERT(dim_qkv % 2 == 0);
  const size_t half_dim_qkv = dim_qkv / 2;
  for (size_t dim = 0; dim < half_dim_qkv; ++dim) {
    const float freq_exponents =
        StaticCast<float>(2 * dim) / StaticCast<float>(dim_qkv);
    // Replacing with expf(ln(1E4) * freq_exponents) changes results noticeably.
    const float timescale = powf(10000.0f, freq_exponents);
    const float theta = StaticCast<float>(pos) / timescale;
    const float cos_val = cosf(theta);
    const float sin_val = sinf(theta);
    const float x0 = x[dim];
    const float x1 = x[dim + half_dim_qkv];
    x[dim] = mul * (x0 * cos_val - x1 * sin_val);
    x[dim + half_dim_qkv] = mul * (x0 * sin_val + x1 * cos_val);
  }
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void AddFrom(
    const float* HWY_RESTRICT other, float* HWY_RESTRICT x, const size_t size) {
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  const D d;

  hn::Transform1(d, x, size, other,
                 [](const auto d, const auto x, const auto other)
                     HWY_ATTR { return hn::Add(x, other); });
}

static HWY_NOINLINE void MulBy(const float* HWY_RESTRICT other,
                               float* HWY_RESTRICT x, const size_t size,
                               const size_t max_pos) {
  HWY_DASSERT(max_pos <= size);
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  const D d;

  hn::Transform1(d, x, max_pos, other,
                 [](const auto d, const auto x, const auto other)
                     HWY_ATTR { return hn::Mul(x, other); });
}

static HWY_INLINE HWY_MAYBE_UNUSED void MulBy(const float* HWY_RESTRICT other,
                                              float* HWY_RESTRICT x,
                                              const size_t size) {
  return MulBy(other, x, size, size);
}

static HWY_NOINLINE void MulByConst(const float c, float* HWY_RESTRICT x,
                                    const size_t size, const size_t max_pos) {
  HWY_DASSERT(max_pos <= size);
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  const D d;
  const auto constant = hn::Set(d, c);
  hn::Transform(d, x, max_pos,
                [&constant](const auto d, const auto x)
                    HWY_ATTR { return hn::Mul(x, constant); });
}

static HWY_INLINE HWY_MAYBE_UNUSED void MulByConst(const float c,
                                                   float* HWY_RESTRICT x,
                                                   const size_t size) {
  MulByConst(c, x, size, size);
}

static HWY_NOINLINE void MulByConstAndAdd(const float c,
                                          const float* HWY_RESTRICT x,
                                          float* HWY_RESTRICT out,
                                          const size_t size,
                                          const size_t max_pos) {
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  const D d;
  const auto constant = hn::Set(d, c);
  hn::Transform1(
      d, out, max_pos, x,
      [&constant](const auto d, const auto out_element, const auto x_element)
          HWY_ATTR { return hn::MulAdd(x_element, constant, out_element); });
}

static HWY_INLINE HWY_MAYBE_UNUSED void MulByConstAndAdd(
    float c, const float* HWY_RESTRICT x, float* HWY_RESTRICT out,
    size_t size) {
  MulByConstAndAdd(c, x, out, size, size);
}

static HWY_NOINLINE void Softmax(float* HWY_RESTRICT x, const size_t size,
                                 const size_t mask_pos) {
  HWY_DASSERT(size != 0);
  HWY_DASSERT(mask_pos <= size);

  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  const D d;

  const auto vmin = hn::Set(d, hwy::LowestValue<float>());
  auto vmax = vmin;
  Foreach(d, x, mask_pos, vmin,
          [&vmax](const auto d, const auto value)
              HWY_ATTR { vmax = hn::Max(vmax, value); });
  vmax = hn::MaxOfLanes(d, vmax);

  // Subtract max (avoid precision loss for large exponents) and exponentiate.
  hn::Transform(d, x, mask_pos,
                [&vmax](const auto d, const auto value) HWY_ATTR {
#if HWY_TARGET & HWY_ALL_SVE
                  // Temporary workaround for buggy SVE codegen: avoid inlined
                  // Exp().
                  return hn::CallExp(d, hn::Sub(value, vmax));
#else
                  return hn::Exp(d, hn::Sub(value, vmax));
#endif
                });

  auto sum = hn::Zero(d);
  Foreach(d, x, mask_pos, sum, [&sum](const auto d, const auto value) HWY_ATTR {
    sum = hn::Add(sum, value);
  });

  // Normalize to probability distribution
  const float mul = 1.0f / hn::ReduceSum(d, sum);
  MulByConst(mul, x, size, mask_pos);
}

static HWY_INLINE HWY_MAYBE_UNUSED void Softmax(float* HWY_RESTRICT x,
                                                const size_t size) {
  Softmax(x, size, size);
}

static HWY_NOINLINE void LogitsSoftCap(const float cap, float* HWY_RESTRICT x,
                                       const size_t size,
                                       const size_t max_pos) {
  HWY_DASSERT(max_pos <= size);

  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  const D d;
  using V = hn::Vec<D>;

  const V vcap = hn::Set(d, cap);
  const V vinv_cap = hn::Div(hn::Set(d, 1.0f), vcap);

  // If we do not subtract the max as in softmax, values > 100 (which do occur)
  // will all saturate to the cap, and then this function is no longer
  // monotonic, which would change the results on TopK.
  const V vmin = hn::Set(d, hwy::LowestValue<float>());
  V vmax = vmin;
  Foreach(d, x, max_pos, vmin,
          [&vmax](const auto d, const auto value)
              HWY_ATTR { vmax = hn::Max(vmax, value); });
  vmax = hn::MaxOfLanes(d, vmax);

  // We want (v-vmax) * vinv_cap. To take advantage of FMA, multiply this out to
  // v * vinv_cap + (-vmax*vinv_cap).
  const V add = hn::Neg(hn::Mul(vmax, vinv_cap));

  hn::Transform(
      d, x, size, [&vcap, &vinv_cap, &add](D d, hn::Vec<D> v) HWY_ATTR {
        return hn::Mul(vcap, hn::Tanh(d, hn::MulAdd(v, vinv_cap, add)));
      });
}

static HWY_INLINE HWY_MAYBE_UNUSED void LogitsSoftCap(const float cap,
                                                      float* HWY_RESTRICT x,
                                                      const size_t size) {
  LogitsSoftCap(cap, x, size, size);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED size_t
SampleArgmax(const float* probabilities, size_t vocab_size) {
  size_t max_index = 0;
  float max_prob = probabilities[0];
  for (size_t i = 1; i < vocab_size; ++i) {
    if (probabilities[i] > max_prob) {
      max_index = i;
      max_prob = probabilities[i];
    }
  }
  return max_index;
}

template <size_t k>
static HWY_NOINLINE HWY_MAYBE_UNUSED std::discrete_distribution<int>
create_distribution(std::array<float, k>& top_k, float temperature) {
  // re-normalize distribution
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  const D d;

  const auto temperature_inv =
      hn::Div(hn::Set(d, 1.0f), hn::Set(d, temperature));

  hn::Transform(d, top_k.data(), top_k.size(),
                [&temperature_inv](D d, hn::Vec<D> v) HWY_ATTR {
                  return hn::Exp(d, hn::Mul(hn::Log(d, v), temperature_inv));
                });

  return std::discrete_distribution<int>(std::begin(top_k), std::end(top_k));
}

template <size_t k, typename TAcceptToken>
static HWY_NOINLINE HWY_MAYBE_UNUSED int SampleTopK(
    const float* HWY_RESTRICT probabilities, size_t vocab_size,
    std::mt19937& gen, float temperature, TAcceptToken& accept_token) {
  static_assert(k != 0, "");
  // TODO: Optimize, potentially using new VQSort PartialSort.
  std::array<float, k> top_k{};  // sorted from highest [0], to lowest [k-1]
  std::array<int, k> indices{};
  for (size_t i = 0; i < vocab_size; ++i) {
    if (probabilities[i] < top_k[k - 1] &&
        (!accept_token || accept_token(StaticCast<int>(i)))) {
      continue;
    }
    for (size_t j = 0; j < k; ++j) {
      if (probabilities[i] > top_k[j] &&
          (!accept_token || accept_token(StaticCast<int>(i)))) {
        // shift elements by 1, insert the new value, move on to next value
        for (size_t idx = k - 1; idx > j; --idx) {
          top_k[idx] = top_k[idx - 1];
          indices[idx] = indices[idx - 1];
        }
        top_k[j] = probabilities[i];
        indices[j] = StaticCast<int>(i);
        break;
      }
    }
  }
  return indices[create_distribution<k>(top_k, temperature)(gen)];
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
