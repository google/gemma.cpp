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
#ifndef THIRD_PARTY_GEMMA_CPP_OPS_H_
#define THIRD_PARTY_GEMMA_CPP_OPS_H_
#include <stddef.h>
#include <stdint.h>

#include <array>
#include <cmath>
#include <random>

// copybara:import_next_line:gemma_cpp
#include "compression/compress.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"

#endif  // THIRD_PARTY_GEMMA_CPP_OPS_H_

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_OPS_TOGGLE) == defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_OPS_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_OPS_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_OPS_TOGGLE
#endif

// copybara:import_next_line:gemma_cpp
#include "compression/compress-inl.h"
#include "hwy/cache_control.h"  // FlushStream
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

template <size_t kOuter>
HWY_INLINE constexpr size_t RowsPerStrip() {
  // Aim for 128 work items to reduce pool overhead. Must be at least one
  // vector; prefer a power of two for faster division.
  constexpr size_t kRowsPerStrip =
      HWY_MAX(hn::ScalableTag<float>().MaxLanes(),
              1ULL << hwy::FloorLog2(kOuter / 128));
  return kRowsPerStrip;
}

// Simple version without tiling nor threading.
template <size_t kOuter, size_t kInner, typename MatT, size_t kCapacity,
          typename VecT>
HWY_INLINE void MatVecLoop(const CompressedArray<MatT, kCapacity>& mat,
                           const size_t mat_ofs,
                           const VecT* HWY_RESTRICT vec_aligned,
                           float* HWY_RESTRICT out) {
  PROFILER_ZONE("MatVecLoop");
  const hn::ScalableTag<float> df;

  for (size_t idx_row = 0; idx_row < kOuter; ++idx_row) {
    const size_t row_ofs = mat_ofs + idx_row * kInner;
    out[idx_row] = Dot(df, mat, row_ofs, vec_aligned, kInner);
  }
}

// Simple version without tiling nor threading, but two offsets/outputs.
template <size_t kOuter, size_t kInner, typename MatT, size_t kCapacity,
          typename VecT>
HWY_INLINE void TwoOfsMatVecLoop(const CompressedArray<MatT, kCapacity>& mat,
                                 const size_t mat_ofs0, const size_t mat_ofs1,
                                 const VecT* HWY_RESTRICT vec_aligned,
                                 float* HWY_RESTRICT out0,
                                 float* HWY_RESTRICT out1) {
  PROFILER_ZONE("MatVecLoop");
  const hn::ScalableTag<float> df;

  for (size_t idx_row = 0; idx_row < kOuter; ++idx_row) {
    const size_t row_ofs0 = mat_ofs0 + (idx_row)*kInner;
    const size_t row_ofs1 = mat_ofs1 + (idx_row)*kInner;
    out0[idx_row] = Dot(df, mat, row_ofs0, vec_aligned, kInner);
    out1[idx_row] = Dot(df, mat, row_ofs1, vec_aligned, kInner);
  }
}

namespace detail {

// For each i = [0, num_rows), compute partial (length `num_cols`) dot product
// of row i with `vec_aligned` and add into `out[i]`. The upper-left coordinate
// of the tile is r0, c0.
template <class DF, typename MatT, size_t kCapacity, typename VecT>
HWY_INLINE void AccumulatePartialDotProducts(
    DF df, const CompressedArray<MatT, kCapacity>& mat, size_t mat_ofs,
    size_t mat_stride, size_t r0, size_t c0, size_t num_rows, size_t num_cols,
    const VecT* HWY_RESTRICT vec_aligned, float* HWY_RESTRICT out) {
  for (size_t idx_row = 0; idx_row < num_rows; ++idx_row) {
    const size_t row_ofs = mat_ofs + (r0 + idx_row) * mat_stride;
    out[idx_row] += Dot(df, mat, row_ofs + c0, vec_aligned + c0, num_cols);
  }
}

// Same as above, but sets out[i] to the first partial dot product, which
// avoids having to zero-initialize and accumulate.
template <class DF, typename MatT, size_t kCapacity, typename VecT>
HWY_INLINE void SetFirstPartialDotProducts(
    DF df, const CompressedArray<MatT, kCapacity>& mat, size_t mat_ofs,
    size_t mat_stride, size_t r0, size_t c0, size_t num_rows, size_t num_cols,
    const VecT* HWY_RESTRICT vec_aligned, float* HWY_RESTRICT out) {
  for (size_t idx_row = 0; idx_row < num_rows; ++idx_row) {
    const size_t row_ofs = mat_ofs + (r0 + idx_row) * mat_stride;
    out[idx_row] = Dot(df, mat, row_ofs + c0, vec_aligned + c0, num_cols);
  }
}

// Adds together partial dot products for all tiles with the same r0 (a
// horizontal strip of the entire matrix); the result is the full dot product
// for rows r in [r0, r0 + num_rows), which we store into in out[r - r0].
template <class DF, typename MatT, size_t kCapacity, typename VecT>
HWY_INLINE void FullDotProductsForStrip(
    DF df, const CompressedArray<MatT, kCapacity>& mat, size_t mat_ofs,
    size_t mat_stride, size_t r0, size_t num_rows,
    const VecT* HWY_RESTRICT vec_aligned, float* HWY_RESTRICT out) {
  // Tall and skinny: set `out` to the single dot product.
  if (mat_stride < MaxCols()) {
    SetFirstPartialDotProducts(df, mat, mat_ofs, mat_stride, r0, 0, num_rows,
                               mat_stride, vec_aligned, out);
    return;
  }

  // We have at least MaxCols, so start by setting `out` to that:
  SetFirstPartialDotProducts(df, mat, mat_ofs, mat_stride, r0, 0, num_rows,
                             MaxCols(), vec_aligned, out);
  // For further multiples of MaxCols, accumulate. Remainders handled below.
  size_t c0 = MaxCols();
  for (; c0 <= mat_stride - MaxCols(); c0 += MaxCols()) {
    AccumulatePartialDotProducts(df, mat, mat_ofs, mat_stride, r0, c0, num_rows,
                                 MaxCols(), vec_aligned, out);
  }

  if (c0 < mat_stride) {  // Final cols
    AccumulatePartialDotProducts(df, mat, mat_ofs, mat_stride, r0, c0, num_rows,
                                 mat_stride - c0, vec_aligned, out);
  }
}

}  // namespace detail

// Stores dot products of rows with `vec_aligned` to a buffer, then stores them
// to `out`.
template <size_t kOuter, size_t kInner, typename MatT, size_t kCapacity,
          typename VecT>
HWY_INLINE void MatVec(const CompressedArray<MatT, kCapacity>& mat,
                       const size_t mat_ofs,
                       const VecT* HWY_RESTRICT const vec_aligned,
                       float* HWY_RESTRICT out, hwy::ThreadPool& pool) {
  PROFILER_ZONE("MatVec");

  const hn::ScalableTag<float> df;
  constexpr size_t kRowsPerStrip = RowsPerStrip<kOuter>();
  constexpr size_t kNumStrips = kOuter / kRowsPerStrip;

  // For each entire strip.
  pool.Run(0, kNumStrips, [&](const uint64_t strip, size_t thread) HWY_ATTR {
    PROFILER_ZONE("MatVec.lambda");
    const size_t r0 = strip * kRowsPerStrip;
    detail::FullDotProductsForStrip(df, mat, mat_ofs, kInner, r0, kRowsPerStrip,
                                    vec_aligned, out + r0);
  });

  // Remaining rows
  const size_t r0 = kNumStrips * kRowsPerStrip;
  if (r0 < kOuter) {
    PROFILER_ZONE("MatVec remainder");
    const size_t num_rows = kOuter - r0;
    detail::FullDotProductsForStrip(df, mat, mat_ofs, kInner, r0, num_rows,
                                    vec_aligned, out + r0);
  }
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
    for (; i < size - 2 * NF; i += 2 * NF) {
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

// Two matrices, same vector
// TODO(janwas): apply optimizations from MatVec/replace with above overload
template <size_t kOuter, size_t kInner, typename MatT, size_t kCapacity,
          typename VecT>
HWY_NOINLINE void TwoMatVec(const CompressedArray<MatT, kCapacity>& mat0,
                            const CompressedArray<MatT, kCapacity>& mat1,
                            const size_t mat_ofs,
                            const VecT* HWY_RESTRICT vec_aligned,
                            float* HWY_RESTRICT out0, float* HWY_RESTRICT out1,
                            hwy::ThreadPool& pool) {
  const hn::ScalableTag<float> df;
  const size_t NF = hn::Lanes(df);

  // Process multiple rows at a time so that we write multiples of a cache line
  // to avoid false sharing (>= 64).
  constexpr size_t kRowsPerStrip = 128 / sizeof(float);
  const uint32_t num_strips = kOuter / kRowsPerStrip;

  // No remainder handling after ThreadPool.
  static_assert(kOuter % kRowsPerStrip == 0, "Add remainder handling");

  // Required for Stream loop, otherwise we might have partial vectors.
  HWY_DASSERT(kRowsPerStrip >= NF);
  pool.Run(0, num_strips,
           [&](const uint32_t strip, size_t /*thread*/) HWY_ATTR {
             // MSVC workaround: duplicate to ensure constexpr.
             constexpr size_t kRowsPerStrip = 128 / sizeof(float);
             // Software write-combining to avoid cache pollution from out.
             // Although `out` may be used later, keeping it out of the cache
             // now and avoiding RFOs is a consistent 5% overall win.
             HWY_ALIGN float buf0[kRowsPerStrip];
             HWY_ALIGN float buf1[kRowsPerStrip];

             // Only handle entire strips here because the Stream is not masked.
             const size_t begin = strip * kRowsPerStrip;
             for (size_t idx_row = 0; idx_row < kRowsPerStrip; ++idx_row) {
               const size_t row_ofs = mat_ofs + (begin + idx_row) * kInner;
               buf0[idx_row] = Dot(df, mat0, row_ofs, vec_aligned, kInner);
               buf1[idx_row] = Dot(df, mat1, row_ofs, vec_aligned, kInner);
             }

             HWY_UNROLL(4)
             for (size_t i = 0; i != kRowsPerStrip; i += NF) {
               hn::Stream(hn::Load(df, buf0 + i), df, out0 + begin + i);
             }
             HWY_UNROLL(4)
             for (size_t i = 0; i != kRowsPerStrip; i += NF) {
               hn::Stream(hn::Load(df, buf1 + i), df, out1 + begin + i);
             }
           });
  hwy::FlushStream();
}

// Baseline Naive MatMul
template <size_t kOuter, size_t kInner, size_t kBatchSize, typename MatT,
          size_t kCapacity, typename VecT>
HWY_NOINLINE void MatMul(const CompressedArray<MatT, kCapacity>& mat,
                         const size_t mat_ofs, const VecT* HWY_RESTRICT vec,
                         float* HWY_RESTRICT out, hwy::ThreadPool& pool) {
  for (size_t i = 0; i < kBatchSize; ++i) {
    MatVec<kOuter, kInner, MatT, kCapacity, VecT>(
        mat, mat_ofs, vec + i * kInner, out + i * kOuter, pool);
  }
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
  float total = 0.f;
  for (size_t i = 0; i < size; ++i) {
    total += a[i] * a[i];
  }
  return total;
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void RMSNorm(
    const float* HWY_RESTRICT x, const float* HWY_RESTRICT weight,
    float* HWY_RESTRICT out, size_t size) {
  constexpr float eps = 1e-6f;
  float ss = SquaredL2(x, size);
  ss = 1.0f / sqrtf(ss / static_cast<int>(size) + eps);
  for (size_t j = 0; j < size; j++) {
    // Note 1.0f centering here
    out[j] = (1.0f + weight[j]) * (ss * x[j]);
  }
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void RMSNorm(
    const float* HWY_RESTRICT x, const hwy::bfloat16_t* HWY_RESTRICT weight,
    float* HWY_RESTRICT out, size_t size) {
  constexpr float eps = 1e-6f;
  float ss = SquaredL2(x, size);
  ss = 1.0f / sqrtf(ss / static_cast<int>(size) + eps);
  for (size_t j = 0; j < size; j++) {
    // Note 1.0f centering here
    out[j] = (1.0f + hwy::F32FromBF16(weight[j])) * (ss * x[j]);
  }
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void RMSNormInplace(
    const float* HWY_RESTRICT weight, float* HWY_RESTRICT inout, size_t size) {
  constexpr float eps = 1e-6f;
  float ss = SquaredL2(inout, size);
  ss = 1.0f / sqrtf(ss / static_cast<int>(size) + eps);
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

  constexpr float eps = 1e-6f;
  const float ss = SquaredL2(inout, size);
  const VF vss = hn::Set(df32, 1.0f / sqrtf(ss / static_cast<int>(size) + eps));

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

  constexpr float eps = 1e-6f;
  const float ss = SquaredL2(x, size);
  const VF vss = hn::Set(df32, 1.0f / sqrtf(ss / static_cast<int>(size) + eps));

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

  constexpr float eps = 1e-6f;
  const float ss = SquaredL2(x, size);
  const VF vss = hn::Set(df32, 1.0f / sqrtf(ss / size + eps));

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
      (num_timescales != 0
           ? static_cast<float>(static_cast<int>(num_timescales) - 1)
           : 1.0f);
  for (size_t dim = 0; dim < num_timescales; ++dim) {
    const float inv_timescale =
        expf(static_cast<int>(dim) * -log_timescale_increment);
    x[dim] += sinf(pos * inv_timescale);
    x[num_timescales + dim] += cosf(pos * inv_timescale);
  }
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void Rope(float* HWY_RESTRICT x,
                                               size_t dim_qkv, size_t pos) {
  HWY_DASSERT(dim_qkv % 2 == 0);
  const size_t half_dim_qkv = dim_qkv / 2;
  for (size_t dim = 0; dim < half_dim_qkv; ++dim) {
    const float freq_exponents = static_cast<float>(2 * static_cast<int>(dim)) /
                                 static_cast<float>(dim_qkv);
    // Replacing with expf(ln(1E4) * freq_exponents) changes results noticeably.
    const float timescale = powf(10000.0f, freq_exponents);
    const float theta = pos / timescale;
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
                                                       size_t pos) {
  HWY_DASSERT(dim_qkv % 2 == 0);
  const size_t half_dim_qkv = dim_qkv / 2;
  for (size_t dim = 0; dim < half_dim_qkv; ++dim) {
    const float freq_exponents = static_cast<float>(2 * static_cast<int>(dim)) /
                                 static_cast<float>(dim_qkv);
    // Replacing with expf(ln(1E4) * freq_exponents) changes results noticeably.
    const float timescale = powf(10000.0f, freq_exponents);
    const float theta = pos / timescale;
    const float cos_val = cosf(theta);
    const float sin_val = sinf(theta);
    const float x0 = x[dim];
    const float x1 = x[dim + half_dim_qkv];
    x[dim] = mul * (x0 * cos_val - x1 * sin_val);
    x[dim + half_dim_qkv] = mul * (x0 * sin_val + x1 * cos_val);
  }
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void AddFrom(
    const float* HWY_RESTRICT other, float* HWY_RESTRICT x, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    x[i] += other[i];
  }
}

static HWY_NOINLINE void MulBy(const float* HWY_RESTRICT other,
                               float* HWY_RESTRICT x, size_t size,
                               size_t max_pos) {
  HWY_DASSERT(max_pos <= size);
  for (size_t i = 0; i < max_pos; ++i) {
    x[i] *= other[i];
  }
}

static HWY_INLINE HWY_MAYBE_UNUSED void MulBy(const float* HWY_RESTRICT other,
                                              float* HWY_RESTRICT x,
                                              size_t size) {
  return MulBy(other, x, size, size);
}

static HWY_NOINLINE void MulByConst(float c, float* HWY_RESTRICT x, size_t size,
                                    size_t max_pos) {
  HWY_DASSERT(max_pos <= size);
  for (size_t i = 0; i < max_pos; ++i) {
    x[i] *= c;
  }
}

static HWY_INLINE HWY_MAYBE_UNUSED void MulByConst(float c,
                                                   float* HWY_RESTRICT x,
                                                   size_t size) {
  MulByConst(c, x, size, size);
}

static HWY_NOINLINE void MulByConstAndAdd(float c, const float* HWY_RESTRICT x,
                                          float* HWY_RESTRICT out, size_t size,
                                          size_t max_pos) {
  for (size_t i = 0; i < max_pos; ++i) {
    out[i] += x[i] * c;
  }
}

static HWY_INLINE HWY_MAYBE_UNUSED void MulByConstAndAdd(
    float c, const float* HWY_RESTRICT x, float* HWY_RESTRICT out,
    size_t size) {
  MulByConstAndAdd(c, x, out, size, size);
}

static HWY_NOINLINE void Softmax(float* HWY_RESTRICT x, size_t size,
                                 size_t mask_pos) {
  HWY_DASSERT(size != 0);
  HWY_DASSERT(mask_pos <= size);

  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  const D d;
  const size_t N = hn::Lanes(d);

  // Find max so we can subtract it below. Avoid hn::Foreach because SVE vectors
  // cannot be lambda-captured.
  // TODO(janwas): could be replaced with an hn::Accumulate algo.
  const hn::Vec<D> vmin = hn::Set(d, hwy::LowestValue<float>());
  hn::Vec<D> vmax = vmin;
  size_t idx = 0;
  if (mask_pos >= N) {
    for (; idx <= mask_pos - N; idx += N) {
      vmax = hn::Max(vmax, LoadU(d, x + idx));
    }
  }
  vmax = hn::Max(vmax, LoadNOr(vmin, d, x + idx, mask_pos - idx));
  vmax = hn::MaxOfLanes(d, vmax);  // broadcast

  // Subtract max (avoid precision loss for large exponents) and exponentiate.
  // Also avoid hn::Transform because the additional `sum` output vector cannot
  // be captured by a lambda.
  hn::Vec<D> sum = hn::Zero(d);
  idx = 0;
  if (mask_pos >= N) {
    for (; idx <= mask_pos - N; idx += N) {
      const hn::Vec<D> out = hn::Exp(d, hn::Sub(hn::LoadU(d, x + idx), vmax));
      sum = hn::Add(sum, out);
      hn::StoreU(out, d, x + idx);
    }
  }
  if (mask_pos > idx) {
    const size_t remaining = mask_pos - idx;
    const hn::Vec<D> out =
        hn::Exp(d, hn::Sub(hn::LoadN(d, x + idx, remaining), vmax));
    sum = hn::Add(sum, out);
    hn::StoreN(out, d, x + idx, remaining);
  }

  // Normalize to probability distribution
  const float mul = 1.0f / hn::ReduceSum(d, sum);
  MulByConst(mul, x, size, mask_pos);
}

static HWY_INLINE HWY_MAYBE_UNUSED void Softmax(float* HWY_RESTRICT x,
                                                size_t size) {
  Softmax(x, size, size);
}

static HWY_NOINLINE void LogitsSoftCap(const float cap, float* HWY_RESTRICT x,
                                       size_t size, size_t max_pos) {
  HWY_DASSERT(max_pos <= size);

  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  const D d;

  const float inv_cap = 1.0f / cap;

  hn::Transform(d, x, size, [cap, inv_cap](D d, hn::Vec<D> v) HWY_ATTR {
    return hn::Mul(hn::Set(d, cap),
                   hn::Tanh(d, hn::Mul(v, hn::Set(d, inv_cap))));
  });
}

static HWY_INLINE HWY_MAYBE_UNUSED void LogitsSoftCap(const float cap,
                                                      float* HWY_RESTRICT x,
                                                      size_t size) {
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
  for (size_t i = 0; i < k; ++i) {
    top_k[i] = exp(log(top_k[i]) / temperature);
  }
  float denominator = 0.0f;
  for (size_t i = 0; i < k; ++i) {
    denominator += top_k[i];
  }
  denominator = 1.0f / denominator;
  MulByConst(denominator, top_k.data(), k);
  return std::discrete_distribution<int>(std::begin(top_k), std::end(top_k));
}

template <size_t k, typename TAcceptToken>
static HWY_NOINLINE HWY_MAYBE_UNUSED int SampleTopK(
    const float* HWY_RESTRICT probabilities, size_t vocab_size,
    std::mt19937& gen, float temperature, TAcceptToken& accept_token) {
  static_assert(k != 0, "");
  // TODO(austinvhuang): Optimize this implementation.
  std::array<float, k> top_k{};  // sorted from highest [0], to lowest [k-1]
  std::array<int, k> indices{};
  for (size_t i = 0; i < vocab_size; ++i) {
    if (probabilities[i] < top_k[k - 1] && accept_token(static_cast<int>(i))) {
      continue;
    }
    for (size_t j = 0; j < k; ++j) {
      if (probabilities[i] > top_k[j] && accept_token(static_cast<int>(i))) {
        // shift elements by 1, insert the new value, move on to next value
        for (size_t idx = k - 1; idx > j; --idx) {
          top_k[idx] = top_k[idx - 1];
          indices[idx] = indices[idx - 1];
        }
        top_k[j] = probabilities[i];
        indices[j] = static_cast<int>(i);
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
