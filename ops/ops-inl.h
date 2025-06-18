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

#include <cmath>
#include <cstdint>
#include <random>
#include <type_traits>  // std::enable_if_t
#include <vector>

#include "ops/matmul.h"
#include "util/allocator.h"
#include "util/basics.h"  // TokenAndProb
#include "util/mat.h"
#include "util/threading_context.h"
#include "hwy/base.h"
#include "hwy/contrib/sort/order.h"
#include "hwy/contrib/sort/vqsort.h"
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

#include "compression/compress-inl.h"
#include "ops/dot-inl.h"
#include "ops/matmul_static.h"  // includes highway.h
#include "ops/sum-inl.h"
#include "hwy/contrib/algo/transform-inl.h"
#include "hwy/contrib/math/math-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <typename TA, typename TC>
MMPerKey* CallMatMul(const MatPtrT<TA>& A, const MatPtr& B,
                     const float* HWY_RESTRICT add, MatMulEnv& env,
                     MatPtrT<TC>& C) {
  return CallUpcasted(
      &B, [&](const auto* B_t) { return MatMulStatic(A, *B_t, add, env, C); });
}

HWY_INLINE double PackTokenAndProb(int32_t token, float prob) {
  // casting prob from float to double just makes some changes to the
  // exponent bias and pads zeros in the mantissa.
  double packed = static_cast<double>(prob);
  int64_t packed_int64;
  hwy::CopySameSize(&packed, &packed_int64);
  // stuff the token into the lower 32 bits of packed_int64. (it is an int32_t
  // anyway)
  packed_int64 &= 0xFFFFFFFF00000000;
  packed_int64 |= token;
  // copy bytes back into packed.
  hwy::CopySameSize(&packed_int64, &packed);
  return packed;
}

HWY_INLINE TokenAndProb UnpackTokenAndProb(double packed) {
  TokenAndProb tp;

  int64_t packed_int64;
  hwy::CopySameSize(&packed, &packed_int64);
  tp.token = static_cast<int>(packed_int64 & 0xFFFFFFFFULL);

  // clear the lower 32 bits of packed_int64 before copying back into packed.
  packed_int64 &= 0xFFFFFFFF00000000ULL;
  hwy::CopySameSize(&packed_int64, &packed);
  tp.prob = static_cast<float>(packed);
  return tp;
}

template <typename To, typename From>
HWY_INLINE constexpr std::enable_if_t<
    std::is_arithmetic_v<To> && std::is_arithmetic_v<From>, To>
StaticCast(From from) noexcept {
  if constexpr (std::is_unsigned_v<From> && std::is_floating_point_v<To>) {
    return static_cast<To>(
        static_cast<hwy::SignedFromSize<sizeof(From)>>(from));
  } else {
    return static_cast<To>(from);
  }
}

// We use the tanh approximation for gelu (also used in training).
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
//         = 0.5 * x * (1 + tanh(x * (sqrt(2/π) + sqrt(2/π) * 0.044715 * x^2)))
//         = 0.5 * x * (1 + tanh(x * (0.79788 + 0.035677 * x^2)))
//         = x * (0.5 + 0.5 * tanh(x * (0.79788 + 0.035677 * x^2))))
template <class D, HWY_IF_F32_D(D)>
HWY_INLINE hn::Vec<D> Gelu(D d, hn::Vec<D> v) {
  const hn::Vec<D> kMul = hn::Set(d, 0.03567740813636141f);
  const hn::Vec<D> kSqrt2OverPi = hn::Set(d, 0.797884560804236f);
  const hn::Vec<D> kHalf = hn::Set(d, 0.5f);

  const hn::Vec<D> v2 = hn::Mul(v, v);
  const hn::Vec<D> arg = hn::Mul(v, hn::MulAdd(kMul, v2, kSqrt2OverPi));
  const hn::Vec<D> cdf = hn::MulAdd(kHalf, hn::Tanh(d, arg), kHalf);
  return hn::Mul(v, cdf);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void Gelu(float* HWY_RESTRICT x,
                                               size_t size) {
  PROFILER_ZONE("ops.Gelu");
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  hn::Transform(D(), x, size,
                [](D d, hn::Vec<D> v) HWY_ATTR { return Gelu(d, v); });
}

template <class D, HWY_IF_F32_D(D)>
HWY_INLINE hn::Vec<D> Sigmoid(D d, hn::Vec<D> v) {
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
  PROFILER_ZONE("ops.Sigmoid");
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  hn::Transform(D(), x, size,
                [](D d, hn::Vec<D> v) HWY_ATTR { return Sigmoid(d, v); });
}

namespace detail {

// Shared by RMSNorm and RMSNormInplace.
template <typename VT>
float RMSNormMul(const VT* HWY_RESTRICT x, size_t size) {
  PROFILER_ZONE("ops.RMSNormMul");

  const hn::ScalableTag<float> d;
  const float l2 = DecompressAndCall(d, MakeSpan(x, size), DotKernelDefault());
  constexpr float kEps = 1e-6f;  // avoid divide by zero
  return 1.0f / sqrtf(l2 / StaticCast<float>(size) + kEps);
}

}  // namespace detail

// `x_ofs` is the offset within `x`, required for NuqStream.
template <typename XT, typename WT, typename OT>
HWY_NOINLINE HWY_MAYBE_UNUSED void RMSNorm(const XT* HWY_RESTRICT x,
                                           const WT* HWY_RESTRICT weight,
                                           size_t w_ofs, OT* HWY_RESTRICT out,
                                           const size_t size) {
  PROFILER_ZONE("ops.RMSNorm");

  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> df;
  using VF = hn::Vec<decltype(df)>;
  const size_t NF = hn::Lanes(df);

  const VF mul = hn::Set(df, detail::RMSNormMul(x, size));

  const auto packed_x = MakeSpan(x, size);
  const auto packed_w = MakeSpan(weight, w_ofs + size);
  const auto packed_out = MakeSpan(out, size);

  HWY_DASSERT(size % (2 * NF) == 0);
  for (size_t i = 0; i < size; i += 2 * NF) {
    VF x0, x1, w0, w1;
    Decompress2(df, packed_x, i, x0, x1);
    Decompress2(df, packed_w, w_ofs + i, w0, w1);
    const VF m0 = hn::Mul(mul, x0);
    const VF m1 = hn::Mul(mul, x1);
    // (1+weight) * m = m + weight*m = one FMA.
    const VF out0 = hn::MulAdd(m0, w0, m0);
    const VF out1 = hn::MulAdd(m1, w1, m1);
    Compress2(df, out0, out1, packed_out, i);
  }
}

// Same as RMSNorm, but its HWY_RESTRICT forbids passing the same pointer.
template <typename WT, typename XT>
HWY_NOINLINE HWY_MAYBE_UNUSED void RMSNormInplace(const WT* HWY_RESTRICT weight,
                                                  size_t w_ofs,
                                                  XT* HWY_RESTRICT inout,
                                                  const size_t size) {
  PROFILER_ZONE("ops.RMSNormInplace");

  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> df;
  using VF = hn::Vec<decltype(df)>;
  const size_t NF = hn::Lanes(df);

  const VF mul = hn::Set(df, detail::RMSNormMul(inout, size));

  const auto packed_w = MakeSpan(weight, w_ofs + size);
  const auto packed_x = MakeSpan(inout, size);

  HWY_DASSERT(size % (2 * NF) == 0);
  for (size_t i = 0; i < size; i += 2 * NF) {
    VF x0, x1, w0, w1;
    Decompress2(df, packed_x, i, x0, x1);
    Decompress2(df, packed_w, w_ofs + i, w0, w1);
    const VF m0 = hn::Mul(mul, x0);
    const VF m1 = hn::Mul(mul, x1);
    // (1+weight) * m = m + weight*m = one FMA.
    const VF out0 = hn::MulAdd(m0, w0, m0);
    const VF out1 = hn::MulAdd(m1, w1, m1);
    Compress2(df, out0, out1, packed_x, i);
  }
}

// Computes mean mu and mean of squares mu2 of a vector. Used in LayerNorm.
template <typename XT>
HWY_NOINLINE void ComputeMoments(const XT* HWY_RESTRICT x, size_t size,
                                 double& mu, double& mu2) {
  HWY_ASSERT(size > 0);
  const hn::ScalableTag<float> df;

  // Use the existing Sum and Dot kernels for simplicity. The second pass
  // is likely not too expensive because it will be in L1.
  const double sum = Sum(df, x, size);
  // We only have one array, so calling `DecompressAndCall` instead of `Dot``
  // avoids loading the 'second' vector again.
  const double sum2 =
      DecompressAndCall(df, MakeSpan(x, size), DotKernelDouble());

  const double inv_size = 1.0 / static_cast<double>(size);
  mu = sum * inv_size;
  mu2 = sum2 * inv_size;
}

// Compare py/flax/linen/normalization.py.
// out = (x - mean) * scale * rsqrt(var + epsilon) + bias
// x and out may be the same.
template <typename XT, typename WT, typename OT>
HWY_NOINLINE void LayerNorm(const XT* x, const WT* HWY_RESTRICT scale,
                            const WT* HWY_RESTRICT bias, OT* out, size_t size) {
  PROFILER_ZONE("ops.LayerNorm");

  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> df;
  using VF = hn::Vec<decltype(df)>;
  const size_t NF = hn::Lanes(df);

  double mu, mu2;
  ComputeMoments(x, size, mu, mu2);
  double var = mu2 - mu * mu;
  var = HWY_MAX(var, 0.0);
  var = 1.0 / sqrt(var + 1E-6);
  const VF vmu = hn::Set(df, static_cast<float>(mu));
  const VF vvar = hn::Set(df, static_cast<float>(var));
  const VF* HWY_RESTRICT pmu = &vmu;
  const VF* HWY_RESTRICT pvar = &vvar;

  const auto packed_x = MakeSpan(x, size);
  const auto packed_scale = MakeSpan(scale, size);
  const auto packed_bias = MakeSpan(bias, size);
  const auto packed_out = MakeSpan(out, size);

  // Loop body for one vector, called from main loop and remainder loop.
  const auto norm = [pmu, pvar](VF x, VF s, VF add) HWY_ATTR -> VF {
    const VF centered = hn::Sub(x, *pmu);
    const VF mul = hn::Mul(s, *pvar);
    return hn::MulAdd(centered, mul, add);
  };

  size_t i = 0;
  if (size >= 2 * NF) {
    for (; i <= size - 2 * NF; i += 2 * NF) {
      VF x0, x1, s0, s1, add0, add1;
      Decompress2(df, packed_x, i, x0, x1);
      Decompress2(df, packed_scale, i, s0, s1);
      Decompress2(df, packed_bias, i, add0, add1);
      const VF n0 = norm(x0, s0, add0);
      const VF n1 = norm(x1, s1, add1);
      Compress2(df, n0, n1, packed_out, i);
    }
  }

  const size_t remaining = size - i;
  HWY_DASSERT(remaining < 2 * NF);
  if (HWY_UNLIKELY(remaining != 0)) {
    HWY_ALIGN float buf_x[2 * hn::MaxLanes(df)];
    HWY_ALIGN float buf_scale[2 * hn::MaxLanes(df)];
    HWY_ALIGN float buf_bias[2 * hn::MaxLanes(df)];
    // Ensure the second vectors are zeroed even if remaining <= NF.
    hn::Store(hn::Zero(df), df, buf_x + NF);
    hn::Store(hn::Zero(df), df, buf_scale + NF);
    hn::Store(hn::Zero(df), df, buf_bias + NF);
    HWY_ALIGN OT buf_out[2 * hn::MaxLanes(df)];
    DecompressAndZeroPad(df, packed_x, i, buf_x, remaining);
    DecompressAndZeroPad(df, packed_scale, i, buf_scale, remaining);
    DecompressAndZeroPad(df, packed_bias, i, buf_bias, remaining);
    const VF x0 = hn::Load(df, buf_x);
    const VF x1 = hn::Load(df, buf_x + NF);
    const VF s0 = hn::Load(df, buf_scale);
    const VF s1 = hn::Load(df, buf_scale + NF);
    const VF add0 = hn::Load(df, buf_bias);
    const VF add1 = hn::Load(df, buf_bias + NF);
    const VF n0 = norm(x0, s0, add0);
    const VF n1 = norm(x1, s1, add1);
    Compress2(df, n0, n1, MakeSpan(buf_out, 2 * NF), 0);
    hwy::CopyBytes(buf_out, out + i, remaining * sizeof(OT));
  }
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void AddAbsolutePositionalEmbeddings(
    float* HWY_RESTRICT x, size_t dim_model, size_t pos) {
  PROFILER_ZONE("ops.AddAbsolutePositionalEmbeddings");
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

/* RoPE as in Rotary Position Embeddings from the `RoFormer` paper
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

// `inv_timescale[dim_qkv / 2]` is precomputed in AttentionActivations.
// This overload is called if `post_qk == PostQKType::HalfRope`.
static HWY_NOINLINE HWY_MAYBE_UNUSED void Rope(
    float* HWY_RESTRICT x, const size_t dim_qkv,
    const float* HWY_RESTRICT inv_timescale, const int pos) {
  PROFILER_ZONE("ops.Rope");
  HWY_DASSERT(dim_qkv % 2 == 0);
  const size_t half_dim_qkv = dim_qkv / 2;

  const hn::ScalableTag<float> df;
  const size_t NF = hn::Lanes(df);
  using VF = hn::Vec<decltype(df)>;
  const VF vpos = hn::Set(df, static_cast<float>(pos));

  // Vectorize computation for half_dim_qkv - (half_dim_qkv % Lanes)
  const size_t vectorizable_dims = hwy::RoundDownTo(half_dim_qkv, NF);
  size_t dim = 0;
  for (; dim < vectorizable_dims; dim += NF) {
    const VF vinv_time_scale = hn::LoadU(df, inv_timescale + dim);
    const VF vtheta = hn::Mul(vpos, vinv_time_scale);

    // Compute rotations.
    VF vcos_theta;
    VF vsin_theta;
    hn::SinCos(df, vtheta, vsin_theta, vcos_theta);

    // Scale input with rotations.
    const VF vx0 = hn::LoadU(df, x + dim);
    const VF vx1 = hn::LoadU(df, x + dim + half_dim_qkv);
    const VF vout0 = hn::MulSub(vx0, vcos_theta, hn::Mul(vx1, vsin_theta));
    const VF vout1 = hn::MulAdd(vx0, vsin_theta, hn::Mul(vx1, vcos_theta));

    hn::StoreU(vout0, df, x + dim);
    hn::StoreU(vout1, df, x + dim + half_dim_qkv);
  }

  // Vectorize computation for remaining dims - same as above, but with LoadN.
  const size_t remaining_dims = half_dim_qkv - dim;
  HWY_DASSERT(remaining_dims < NF);  // at most one iteration
  if (remaining_dims != 0) {
    VF vinv_time_scale = hn::LoadN(df, inv_timescale + dim, remaining_dims);
    VF vtheta = hn::Mul(vpos, vinv_time_scale);

    // Compute rotations.
    VF vcos_theta;
    VF vsin_theta;
    hn::SinCos(df, vtheta, vsin_theta, vcos_theta);

    // Scale input with rotations.
    const VF vx0 = hn::LoadN(df, x + dim, remaining_dims);
    const VF vx1 = hn::LoadN(df, x + dim + half_dim_qkv, remaining_dims);
    const VF vout0 = hn::MulSub(vx0, vcos_theta, hn::Mul(vx1, vsin_theta));
    const VF vout1 = hn::MulAdd(vx0, vsin_theta, hn::Mul(vx1, vcos_theta));

    hn::StoreN(vout0, df, x + dim, remaining_dims);
    hn::StoreN(vout1, df, x + dim + half_dim_qkv, remaining_dims);
  }
}

// `inv_timescale[dim_qkv / 2]` is precomputed in AttentionActivations.
static HWY_NOINLINE HWY_MAYBE_UNUSED void RopeAndMulBy(
    const float mul, float* HWY_RESTRICT x, const size_t dim_qkv,
    const float* HWY_RESTRICT inv_timescale, const int pos) {
  PROFILER_ZONE("ops.RopeAndMulBy");
  HWY_DASSERT(dim_qkv % 2 == 0);
  const size_t half_dim_qkv = dim_qkv / 2;

  const hn::ScalableTag<float> df;
  const size_t NF = hn::Lanes(df);
  using VF = hn::Vec<decltype(df)>;
  const VF vmul = hn::Set(df, mul);
  const VF vpos = hn::Set(df, static_cast<float>(pos));

  // Vectorize computation for half_dim_qkv - (half_dim_qkv % Lanes)
  const size_t vectorizable_dims = hwy::RoundDownTo(half_dim_qkv, NF);
  size_t dim = 0;
  for (; dim < vectorizable_dims; dim += NF) {
    const VF vinv_time_scale = hn::LoadU(df, inv_timescale + dim);
    const VF vtheta = hn::Mul(vpos, vinv_time_scale);

    // Compute rotations.
    VF vcos_theta;
    VF vsin_theta;
    hn::SinCos(df, vtheta, vsin_theta, vcos_theta);

    // Scale input with rotations and multiply with constant.
    const VF vx0 = hn::Mul(vmul, hn::LoadU(df, x + dim));
    const VF vx1 = hn::Mul(vmul, hn::LoadU(df, x + dim + half_dim_qkv));
    const VF vout0 = hn::MulSub(vx0, vcos_theta, hn::Mul(vx1, vsin_theta));
    const VF vout1 = hn::MulAdd(vx0, vsin_theta, hn::Mul(vx1, vcos_theta));

    hn::StoreU(vout0, df, x + dim);
    hn::StoreU(vout1, df, x + dim + half_dim_qkv);
  }

  // Vectorize computation for remaining dims - same as above, but with LoadN.
  const size_t remaining_dims = half_dim_qkv - dim;
  HWY_DASSERT(remaining_dims < NF);  // at most one iteration
  if (remaining_dims != 0) {
    VF vinv_time_scale = hn::LoadN(df, inv_timescale + dim, remaining_dims);
    VF vtheta = hn::Mul(vpos, vinv_time_scale);

    // Compute rotations.
    VF vcos_theta;
    VF vsin_theta;
    hn::SinCos(df, vtheta, vsin_theta, vcos_theta);

    // Scale input with rotations and multiply with constant.
    const VF vx0 = hn::Mul(vmul, hn::LoadN(df, x + dim, remaining_dims));
    const VF vx1 =
        hn::Mul(vmul, hn::LoadN(df, x + dim + half_dim_qkv, remaining_dims));
    const VF vout0 = hn::MulSub(vx0, vcos_theta, hn::Mul(vx1, vsin_theta));
    const VF vout1 = hn::MulAdd(vx0, vsin_theta, hn::Mul(vx1, vcos_theta));

    hn::StoreN(vout0, df, x + dim, remaining_dims);
    hn::StoreN(vout1, df, x + dim + half_dim_qkv, remaining_dims);
  }
}

template <typename XT>
static HWY_NOINLINE HWY_MAYBE_UNUSED void AddFrom(const XT* HWY_RESTRICT x,
                                                  float* HWY_RESTRICT out,
                                                  const size_t size) {
  PROFILER_ZONE("ops.AddFrom");

  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> df;
  const size_t NF = hn::Lanes(df);
  using VF = hn::Vec<decltype(df)>;

  const auto packed_x = MakeSpan(x, size);

  size_t i = 0;
  if (size >= 2 * NF) {
    for (; i <= size - 2 * NF; i += 2 * NF) {
      VF x0, x1;
      Decompress2(df, packed_x, i, x0, x1);
      VF out0 = hn::Load(df, out + i);
      VF out1 = hn::Load(df, out + i + NF);
      hn::Store(hn::Add(x0, out0), df, out + i);
      hn::Store(hn::Add(x1, out1), df, out + i + NF);
    }
  }

  const size_t remaining = size - i;
  const size_t remaining1 = remaining - HWY_MIN(remaining, NF);
  HWY_DASSERT(remaining < 2 * NF);
  HWY_DASSERT(remaining1 < NF);
  if (HWY_UNLIKELY(remaining != 0)) {
    HWY_ALIGN float buf_x[2 * hn::MaxLanes(df)];
    // Ensure the second vector is zeroed even if remaining <= NF.
    hn::Store(hn::Zero(df), df, buf_x + NF);
    DecompressAndZeroPad(df, packed_x, i, buf_x, remaining);
    const VF x0 = hn::Load(df, buf_x);
    const VF x1 = hn::Load(df, buf_x + NF);
    const VF out0 = hn::LoadN(df, out + i, remaining);
    const VF out1 = hn::LoadN(df, out + i + NF, remaining1);
    hn::StoreN(hn::Add(x0, out0), df, out + i, remaining);
    hn::StoreN(hn::Add(x1, out1), df, out + i + NF, remaining1);
  }
}

// Simple loops unless/until batch sizes are large enough to parallelize.
template <typename XT, typename OT>
void RMSNormBatched(const MatPtrT<XT>& activations, const MatPtr& weights,
                    MatPtrT<OT>& out) {
  HWY_DASSERT(weights.Rows() == 1);
  HWY_DASSERT(weights.Cols() == activations.Cols());
  HWY_DASSERT(activations.SameShape(out));

  CallUpcasted(&weights, [&](const auto* weights_t) {
    for (size_t token_idx = 0; token_idx < activations.Rows(); ++token_idx) {
      RMSNorm(activations.Row(token_idx), weights_t->PackedScale1(), 0,
              out.Row(token_idx), activations.Cols());
    }
  });
}

template <typename XT>
void RMSNormInplaceBatched(const MatPtr& weights, MatPtrT<XT>& inout) {
  HWY_DASSERT(weights.Rows() == 1);
  HWY_DASSERT(weights.Cols() == inout.Cols());

  CallUpcasted(&weights, [&](const auto* weights_t) {
    for (size_t token_idx = 0; token_idx < inout.Rows(); ++token_idx) {
      RMSNormInplace(weights_t->PackedScale1(), 0, inout.Row(token_idx),
                     inout.Cols());
    }
  });
}

// x and out may be the same.
template <typename XT, typename OT>
void LayerNormBatched(const MatPtrT<XT>& x, const MatPtr& weight,
                      const MatPtr& bias, MatPtrT<OT>& out) {
  HWY_DASSERT(weight.Cols() == bias.Cols());
  HWY_DASSERT(weight.Cols() == x.Cols());
  HWY_DASSERT(x.SameShape(out));

  CallUpcastedSame(
      &weight, &bias, [&](const auto* weight_t, const auto* bias_t) {
        for (size_t token_idx = 0; token_idx < x.Rows(); ++token_idx) {
          LayerNorm(x.Row(token_idx), weight_t->PackedScale1(),
                    bias_t->PackedScale1(), out.Row(token_idx), x.Cols());
        }
      });
}

template <typename XT>
static HWY_INLINE void AddFromBatched(const MatPtrT<XT>& x,
                                      MatPtrT<float>& out) {
  HWY_DASSERT(out.SameShape(x));
  for (size_t token_idx = 0; token_idx < out.Rows(); ++token_idx) {
    AddFrom(x.Row(token_idx), out.Row(token_idx), x.Cols());
  }
}

template <typename XT>
HWY_NOINLINE HWY_MAYBE_UNUSED void MulByConst(float c, XT* HWY_RESTRICT x,
                                              size_t size) {
  PROFILER_ZONE("ops.MulByConst");
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> df;
  const size_t NF = hn::Lanes(df);
  using VF = hn::Vec<decltype(df)>;

  const VF v_c = hn::Set(df, c);
  const auto packed_x = MakeSpan(x, size);

  size_t i = 0;
  if (size >= 2 * NF) {
    for (; i <= size - 2 * NF; i += 2 * NF) {
      VF x0, x1;
      Decompress2(df, packed_x, i, x0, x1);
      x0 = hn::Mul(x0, v_c);
      x1 = hn::Mul(x1, v_c);
      Compress2(df, x0, x1, packed_x, i);
    }
  }

  const size_t remaining = size - i;
  HWY_DASSERT(remaining < 2 * NF);
  if (HWY_UNLIKELY(remaining != 0)) {
    HWY_ALIGN float buf_x[2 * hn::MaxLanes(df)];
    // Ensure the second vector is zeroed even if remaining <= NF.
    hn::Store(hn::Zero(df), df, buf_x + NF);
    DecompressAndZeroPad(df, packed_x, i, buf_x, remaining);
    VF x0 = hn::Load(df, buf_x);
    VF x1 = hn::Load(df, buf_x + NF);
    x0 = hn::Mul(x0, v_c);
    x1 = hn::Mul(x1, v_c);
    Compress2(df, x0, x1, MakeSpan(buf_x, 2 * NF), 0);
    hwy::CopyBytes(buf_x, x + i, remaining * sizeof(XT));
  }
}

template <typename XT, typename OT>
HWY_NOINLINE HWY_MAYBE_UNUSED void MulByConstAndAdd(float c,
                                                    const XT* HWY_RESTRICT x,
                                                    OT* HWY_RESTRICT out,
                                                    size_t size) {
  PROFILER_ZONE("ops.MulByConstAndAdd");
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> df;
  const size_t NF = hn::Lanes(df);
  using VF = hn::Vec<decltype(df)>;

  const VF v_c = hn::Set(df, c);
  const auto packed_x = MakeSpan(x, size);
  const auto packed_out = MakeSpan(out, size);

  size_t i = 0;
  if (size >= 2 * NF) {
    for (; i <= size - 2 * NF; i += 2 * NF) {
      VF x0, x1, out0, out1;
      Decompress2(df, packed_x, i, x0, x1);
      Decompress2(df, packed_out, i, out0, out1);
      out0 = hn::MulAdd(x0, v_c, out0);
      out1 = hn::MulAdd(x1, v_c, out1);
      Compress2(df, out0, out1, packed_out, i);
    }
  }

  const size_t remaining = size - i;
  HWY_DASSERT(remaining < 2 * NF);
  if (HWY_UNLIKELY(remaining != 0)) {
    HWY_ALIGN float buf_x[2 * hn::MaxLanes(df)];
    HWY_ALIGN float buf_out[2 * hn::MaxLanes(df)];
    // Ensure the second vectors are zeroed even if remaining <= NF.
    hn::Store(hn::Zero(df), df, buf_x + NF);
    hn::Store(hn::Zero(df), df, buf_out + NF);
    DecompressAndZeroPad(df, packed_x, i, buf_x, remaining);
    DecompressAndZeroPad(df, packed_out, i, buf_out, remaining);
    const VF x0 = hn::Load(df, buf_x);
    const VF x1 = hn::Load(df, buf_x + NF);
    VF out0 = hn::Load(df, buf_out);
    VF out1 = hn::Load(df, buf_out + NF);
    out0 = hn::MulAdd(x0, v_c, out0);
    out1 = hn::MulAdd(x1, v_c, out1);
    Compress2(df, out0, out1, MakeSpan(buf_out, 2 * NF), 0);
    hwy::CopyBytes(buf_out, out + i, remaining * sizeof(OT));
  }
}

// See below for a specialized version for top-1 sampling.
static HWY_NOINLINE void Softmax(float* HWY_RESTRICT x, const size_t size,
                                 float temperature = 1.0f) {
  PROFILER_ZONE("ops.Softmax");
  HWY_DASSERT(size != 0);

  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  using V = hn::Vec<D>;
  const D d;

  const V vmin = hn::Set(d, hwy::LowestValue<float>());
  V vmax = vmin;
  V* pmax = &vmax;  // workaround for SVE: cannot capture &vector directly
  hn::Foreach(d, x, size, vmin, [pmax](const auto d, const V value) HWY_ATTR {
    *pmax = hn::Max(*pmax, value);
  });
  vmax = hn::MaxOfLanes(d, vmax);

  // Subtract max (avoid precision loss for large exponents) and exponentiate.
  hn::Transform(d, x, size, [pmax](const auto d, const V value) HWY_ATTR {
    if constexpr (HWY_TARGET & HWY_ALL_SVE) {
      // Temporary workaround for buggy SVE codegen: avoid inlined Exp().
      return hn::CallExp(d, hn::Sub(value, *pmax));
    } else {
      return hn::Exp(d, hn::Sub(value, *pmax));
    }
  });

  if (temperature != 1.0f) {
    const float temperature_inv = 1.0f / temperature;
    hn::Transform(d, x, size,
                  [temperature_inv](const auto d, const V value) HWY_ATTR {
                    return hn::Mul(value, hn::Set(d, temperature_inv));
                  });
  }

  // Normalize to probability distribution. The exact sum seems like it should
  // not make a huge difference. It halves the standard deviation of the sum of
  // the normalized probabilities from 1E-7 to 5E-8, but actually also changes
  // the generated text after a few hundred tokens.
  const float sum_exp = Sum(d, x, size);
  // Double-precision reciprocal does not appear to affect the results.
  const float mul = 1.0f / sum_exp;
  MulByConst(mul, x, size);
}

// Note: https://arxiv.org/pdf/2001.04438 proposes to replace the three max /
// exp / mul passes with two passes, both of which compute Exp. This is
// reportedly only faster for very large arrays, larger even than our 256K
// vocab size. We instead fuse the subsequent sampling pass into the softmax,
// which already knows the max value which top-1 sampling would again seek.

// Returns the argmax and x[argmax].
static HWY_INLINE TokenAndProb ArgmaxAndMax(const float* HWY_RESTRICT x,
                                            const size_t num) {
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  using V = hn::Vec<D>;
  using M = hn::Mask<D>;
  const D d;
  const hn::RebindToSigned<D> di;
  using TI = hn::TFromD<decltype(di)>;
  using VI = hn::Vec<decltype(di)>;
  const size_t N = hn::Lanes(d);
  HWY_ASSERT(num % (2 * N) == 0);

  V max0 = hn::Set(d, hwy::LowestValue<float>());
  V max1 = max0;
  VI argmax0 = hn::Zero(di);
  VI argmax1 = argmax0;

  for (size_t i = 0; i < num; i += 2 * N) {
    const V v0 = hn::LoadU(d, x + i);
    const V v1 = hn::LoadU(d, x + i + N);
    const VI vi0 = hn::Iota(di, static_cast<TI>(i));
    const VI vi1 = hn::Iota(di, static_cast<TI>(i + N));
    const M gt0 = hn::Gt(v0, max0);
    const M gt1 = hn::Gt(v1, max1);
    max0 = hn::IfThenElse(gt0, v0, max0);
    max1 = hn::IfThenElse(gt1, v1, max1);
    argmax0 = hn::IfThenElse(hn::RebindMask(di, gt0), vi0, argmax0);
    argmax1 = hn::IfThenElse(hn::RebindMask(di, gt1), vi1, argmax1);
  }

  // Combine the two vectors
  const M gt0 = hn::Gt(max0, max1);
  max0 = hn::IfThenElse(gt0, max0, max1);
  argmax0 = hn::IfThenElse(hn::RebindMask(di, gt0), argmax0, argmax1);

  // Reduce to the global max
  const V max = hn::MaxOfLanes(d, max0);  // broadcasts

  // Argmax = lowest-indexed lane equal to the global max
  const size_t lane = hn::FindKnownFirstTrue(d, hn::Eq(max, max0));
  const TI argmax = hn::ExtractLane(argmax0, lane);
  return TokenAndProb{.token = argmax, .prob = hn::GetLane(max)};
}

// Returns argmax of softmax and its probability. This overwrites `x`, but not
// with normalized probabilities. Only equivalent to `Softmax` + `sample_func`
// if `kTopK` == 1. This is worthwhile because `num` is typically `kVocabSize`
// == 256K, and this avoids writing and then scanning again for the max.
// However, this is not enough to make parallelization worthwhile.
static HWY_MAYBE_UNUSED TokenAndProb Top1OfSoftmax(float* HWY_RESTRICT x,
                                                   const size_t num) {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> d;
  using V = hn::Vec<decltype(d)>;

  const TokenAndProb argmax = ArgmaxAndMax(x, num);

  // Subtract max (avoid precision loss for large exponents) and exponentiate.
  const V max = hn::Set(d, argmax.prob);
  const V* pmax = &max;
  hn::Transform(d, x, num, [pmax](const auto d, const V value) HWY_ATTR {
    if constexpr (HWY_TARGET & HWY_ALL_SVE) {
      // Temporary workaround for buggy SVE codegen: avoid inlined Exp().
      return hn::CallExp(d, hn::Sub(value, *pmax));
    } else {
      return hn::Exp(d, hn::Sub(value, *pmax));
    }
  });

  // Normalize to a single probability. The exact sum seems like it should not
  // make a huge difference. It halves the standard deviation of the sum of the
  // normalized probabilities from 1E-7 to 5E-8, but actually also changes the
  // generated text after a few hundred tokens.
  const float sum_exp = Sum(d, x, num);
  const float prob = x[argmax.token] / sum_exp;
  return TokenAndProb{.token = argmax.token, .prob = prob};
}

static HWY_NOINLINE void LogitsSoftCap(const float cap, float* HWY_RESTRICT x,
                                       const size_t size,
                                       const size_t max_pos) {
  PROFILER_ZONE("ops.LogitsSoftCap");
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

static HWY_INLINE void LogitsSoftCap(const float cap, float* HWY_RESTRICT x,
                                     const size_t size) {
  LogitsSoftCap(cap, x, size, size);
}

// Calls LogitsSoftCap if cap != 0.0f.
static HWY_INLINE HWY_MAYBE_UNUSED void MaybeLogitsSoftCap(
    const float cap, float* HWY_RESTRICT x, const size_t size) {
  if (cap != 0.0f) {
    LogitsSoftCap(cap, x, size, size);
  }
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

HWY_INLINE HWY_MAYBE_UNUSED std::discrete_distribution<int> create_distribution(
    std::vector<float>& top_k, float temperature) {
  HWY_ASSERT(temperature >= 0.0f);
  if (temperature == 0.0f) {
    // Temperature == 0 is a special case which always returns the argmax (0).
    // We also want to avoid dividing by zero in the code below.
    return std::discrete_distribution<int>();
  }
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

template <typename TAcceptToken>
HWY_NOINLINE HWY_MAYBE_UNUSED std::vector<TokenAndProb> TopK(
    const float* HWY_RESTRICT probabilities, size_t vocab_size, size_t k,
    TAcceptToken& accept_token) {
  HWY_ASSERT(k != 0);
  HWY_ASSERT(k <= vocab_size);
  std::vector<double> packed_token_probs;
  for (int32_t i = 0; i < static_cast<int32_t>(vocab_size); ++i) {
    if (accept_token && !accept_token(i, probabilities[i])) {
      continue;
    }
    packed_token_probs.push_back(PackTokenAndProb(i, probabilities[i]));
  }

  hwy::VQSelect(packed_token_probs.data(), packed_token_probs.size(), k,
                hwy::SortDescending());
  hwy::VQSort(packed_token_probs.data(), k, hwy::SortDescending());

  std::vector<TokenAndProb> token_probs;
  token_probs.reserve(k);
  for (int32_t i = 0; i < static_cast<int32_t>(k); ++i) {
    token_probs.push_back(UnpackTokenAndProb(packed_token_probs[i]));
  }
  return token_probs;
}

template <typename TAcceptToken>
HWY_NOINLINE HWY_MAYBE_UNUSED int SampleTopK(
    const float* HWY_RESTRICT probabilities, size_t k, size_t vocab_size,
    std::mt19937& gen, float temperature, TAcceptToken& accept_token) {
  std::vector<TokenAndProb> token_probs =
      TopK(probabilities, vocab_size, k, accept_token);
  std::vector<int> topk_indices(k);
  std::vector<float> topk_probs(k);
  for (size_t i = 0; i < k; ++i) {
    topk_indices[i] = token_probs[i].token;
    topk_probs[i] = token_probs[i].prob;
  }
  return topk_indices[create_distribution(topk_probs, temperature)(gen)];
}

template <typename TAcceptToken>
HWY_NOINLINE HWY_MAYBE_UNUSED TokenAndProb FusedSoftmaxAndSampleTopK(
    const float* HWY_RESTRICT logits, size_t k, size_t vocab_size,
    std::mt19937& gen, float temperature, TAcceptToken& accept_token) {
  // Softmax and sample top-K is equivalent to taking the top-K logits and
  // sampling from the softmax of the top-K logits. The latter is faster as it
  // avoids computing the softmax of all logits.
  std::vector<TokenAndProb> token_logits =
      TopK(logits, vocab_size, k, accept_token);
  std::vector<int> topk_indices(k);
  std::vector<float> topk_logits(k);
  for (size_t i = 0; i < token_logits.size(); ++i) {
    topk_indices[i] = token_logits[i].token;
    topk_logits[i] = token_logits[i].prob;
  }

  size_t mask = token_logits.size();
  Softmax(topk_logits.data(), mask, temperature);
  auto distribution = std::discrete_distribution<int>(
      std::begin(topk_logits), std::begin(topk_logits) + mask);
  int topk_sampled_index = distribution(gen);
  int sampled_index = topk_indices[topk_sampled_index];
  return TokenAndProb{.token = sampled_index,
                      .prob = topk_logits[topk_sampled_index]};
}

// Performs 4x4 average pooling across row vectors
// Input has 4096 (64*64) rows, output has 256 (16*16) rows
// Each output row is the average of a 4x4 block of input rows
template <typename T>
MatStorageT<T> AvgPool4x4(MatStorageT<T>& input) {
  const Extents2D extents = input.Extents();
  // Input validation
  HWY_DASSERT(extents.rows == 4096);  // 64 * 64 = 4096 input rows
  // Create output with 256 rows and same number of columns
  const size_t out_rows = 256;  // 16 * 16 = 256 output rows
  MatStorageT<T> result("pool4x4", Extents2D(out_rows, extents.cols),
                        MatPadding::kOdd);
  const size_t input_dim = 64;   // Input is 64×64
  const size_t output_dim = 16;  // Output is 16×16
  for (size_t out_row_idx = 0; out_row_idx < output_dim; ++out_row_idx) {
    for (size_t out_col_idx = 0; out_col_idx < output_dim; ++out_col_idx) {
      size_t out_idx = out_row_idx * output_dim + out_col_idx;
      T* output_row = result.Row(out_idx);
      // Initialize output row to zeros
      std::fill(output_row, output_row + extents.cols, 0);
      // Average 16 row vectors from a 4x4 block
      for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
          size_t in_row_idx = out_row_idx * 4 + i;
          size_t in_col_idx = out_col_idx * 4 + j;
          size_t in_idx = in_row_idx * input_dim + in_col_idx;
          const T* input_row = input.Row(in_idx);
          // Add each input row to the output
          // TODO(philculliton): use AddFrom in `ops-inl` for a vectorized loop.
          for (size_t col = 0; col < extents.cols; ++col) {
            output_row[col] += input_row[col];
          }
        }
      }
      // Divide by 16 to get the average
      for (size_t col = 0; col < extents.cols; ++col) {
        output_row[col] *= T{0.0625};
      }
    }
  }
  return result;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
