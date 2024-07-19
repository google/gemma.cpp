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
#ifndef THIRD_PARTY_GEMMA_CPP_OPS_OPS_INL_H_
#define THIRD_PARTY_GEMMA_CPP_OPS_OPS_INL_H_

#include <math.h>
#include <stddef.h>
#include <stdio.h>

#include <array>
#include <cmath>
#include <random>
#include <type_traits>  // std::enable_if_t

#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/detect_targets.h"
#include "hwy/profiler.h"

#endif  // THIRD_PARTY_GEMMA_CPP_OPS_OPS_INL_H_

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_OPS_TOGGLE) == defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_OPS_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_OPS_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_OPS_TOGGLE
#endif

#include "hwy/contrib/algo/transform-inl.h"
#include "hwy/contrib/dot/dot-inl.h"
#include "hwy/contrib/math/math-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

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
  using V = hn::Vec<D>;

  hn::Transform1(D(), x, size, other,
                 [](const auto d, const V x, const V other)
                     HWY_ATTR { return hn::Add(x, other); });
}

// Simple loops unless/until batch sizes are large enough to parallelize.
template <typename WeightT, typename OutT>
void RMSNormBatched(size_t num_tokens, const float* activations,
                    const WeightT* weights, OutT* out, const size_t model_dim) {
  for (size_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
    RMSNorm(activations + token_idx * model_dim, weights,
            out + token_idx * model_dim, model_dim);
  }
}

// TODO: pass RowVectorBatch argument.
template <typename WeightT, typename InOutT>
void RMSNormInplaceBatched(size_t num_tokens, const WeightT* weights,
                           InOutT* inout, const size_t model_dim) {
  for (size_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
    RMSNormInplace(weights, inout + token_idx * model_dim, model_dim);
  }
}

static HWY_INLINE void AddFromBatched(size_t num_tokens, const float* other,
                                      float* x, const size_t model_dim) {
  for (size_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
    AddFrom(other + token_idx * model_dim, x + token_idx * model_dim,
            model_dim);
  }
}

static HWY_NOINLINE void MulBy(const float* HWY_RESTRICT other,
                               float* HWY_RESTRICT x, const size_t size,
                               const size_t max_pos) {
  HWY_DASSERT(max_pos <= size);
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  using V = hn::Vec<D>;

  hn::Transform1(D(), x, max_pos, other,
                 [](const auto d, const V x, const V other)
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
  using V = hn::Vec<D>;
  hn::Transform(D(), x, max_pos, [c](const auto d, const V x) HWY_ATTR {
    return hn::Mul(x, hn::Set(d, c));
  });
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
  using V = hn::Vec<D>;
  hn::Transform1(D(), out, max_pos, x,
                 [c](const auto d, const V v_out, const V v_x) HWY_ATTR {
                   return hn::MulAdd(v_x, hn::Set(d, c), v_out);
                 });
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
  using V = hn::Vec<D>;
  const D d;

  const V vmin = hn::Set(d, hwy::LowestValue<float>());
  V vmax = vmin;
  V* pmax = &vmax;  // workaround for SVE: cannot capture &vector directly
  Foreach(d, x, mask_pos, vmin, [pmax](const auto d, const V value) HWY_ATTR {
    *pmax = hn::Max(*pmax, value);
  });
  vmax = hn::MaxOfLanes(d, vmax);

  // Subtract max (avoid precision loss for large exponents) and exponentiate.
  hn::Transform(d, x, mask_pos, [pmax](const auto d, const V value) HWY_ATTR {
#if HWY_TARGET & HWY_ALL_SVE
    // Temporary workaround for buggy SVE codegen: avoid inlined
    // Exp().
    return hn::CallExp(d, hn::Sub(value, *pmax));
#else
                  return hn::Exp(d, hn::Sub(value, *pmax));
#endif
  });

  V sum = hn::Zero(d);
  V* psum = &sum;
  Foreach(d, x, mask_pos, sum, [psum](const auto d, const V value) HWY_ATTR {
    *psum = hn::Add(*psum, value);
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
  using V = hn::Vec<D>;

  const float inv_cap = 1.0f / cap;

  hn::Transform(D(), x, max_pos, [cap, inv_cap](D d, V v) HWY_ATTR {
    return hn::Mul(hn::Set(d, cap),
                   hn::Tanh(d, hn::Mul(v, hn::Set(d, inv_cap))));
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
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;

  // re-normalize distribution
  const float temperature_inv = 1.0f / temperature;
  hn::Transform(D(), top_k.data(), top_k.size(),
                [temperature_inv](D d, hn::Vec<D> v) HWY_ATTR {
                  return hn::Exp(
                      d, hn::Mul(hn::Log(d, v), hn::Set(d, temperature_inv)));
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
        (!accept_token || accept_token(StaticCast<int>(i), probabilities[i]))) {
      continue;
    }
    for (size_t j = 0; j < k; ++j) {
      if (probabilities[i] > top_k[j] &&
          (!accept_token ||
           accept_token(StaticCast<int>(i), probabilities[i]))) {
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
