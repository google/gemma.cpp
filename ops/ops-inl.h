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
#include "util/basics.h"  // TokenAndProb, RngStream
#include "util/mat.h"
#include "util/threading_context.h"
#include "hwy/base.h"
#include "hwy/bit_set.h"
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
                     MatPtrT<TC>& C, const MMOptions& options = MMOptions()) {
  return CallUpcasted(&B, [&](const auto* B_t) {
    return MatMulStatic(A, *B_t, add, env, C, options);
  });
}

static inline void CallTwoMatMul(const MatPtrT<BF16>& A, const MatPtr& B1,
                                 const MatPtr& B2, MatMulEnv& env,
                                 MatPtrT<BF16>& C, const MMOptions& options) {
  return CallUpcastedSame(&B1, &B2, [&](const auto* B1_t, const auto* B2_t) {
    return TwoMatMulStatic(A, *B1_t, *B2_t, env, C, options);
  });
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

// Activation already has a profiler zone.
template <typename T>
static HWY_NOINLINE HWY_MAYBE_UNUSED void Gelu(T* HWY_RESTRICT x, size_t size) {
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  using VF = hn::Vec<DF>;
  DecompressAndCompressInplace(
      DF(), x, size, [](DF d, VF v) HWY_ATTR -> VF { return Gelu(d, v); });
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
template <typename T>
static HWY_NOINLINE HWY_MAYBE_UNUSED void Sigmoid(T* HWY_RESTRICT x,
                                                  size_t size) {
  PROFILER_ZONE("ops.Sigmoid");
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  using VF = hn::Vec<DF>;
  DecompressAndCompressInplace(
      DF(), x, size, [](DF d, VF v) HWY_ATTR -> VF { return Sigmoid(d, v); });
}

namespace detail {

// Shared by RMSNorm and RMSNormInplace.
template <typename VT>
float RMSNormMul(const VT* HWY_RESTRICT x, const size_t size, hwy::Profiler& p,
                 const size_t worker) {
  static const auto zone = p.AddZone("Ops.RMSNormMul");
  PROFILER_ZONE3(p, worker, zone);

  const hn::ScalableTag<float> d;
  const float l2 = DecompressAndCall(d, MakeSpan(x, size), DotKernelDefault());
  constexpr float kEps = 1e-6f;  // avoid divide by zero
  return 1.0f / sqrtf(l2 / StaticCast<float>(size) + kEps);
}

}  // namespace detail

template <typename XT, typename WT, typename OT>
HWY_NOINLINE HWY_MAYBE_UNUSED void RMSNorm(const XT* HWY_RESTRICT x,
                                           const WT* HWY_RESTRICT weight,
                                           OT* HWY_RESTRICT out,
                                           const size_t size, hwy::Profiler& p,
                                           const size_t worker) {
  static const auto zone = p.AddZone("Ops.RMSNorm");
  PROFILER_ZONE3(p, worker, zone);

  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  using VF = hn::Vec<DF>;

  const VF mul = hn::Set(DF(), detail::RMSNormMul(x, size, p, worker));
  const VF* HWY_RESTRICT pmul = &mul;

  Decompress2AndCompressTo(DF(), out, size, x, weight,
                           [pmul](DF /*df*/, VF vx, VF vw) HWY_ATTR -> VF {
                             const VF m = hn::Mul(*pmul, vx);
                             // (1+weight) * m = m + weight*m = one FMA.
                             return hn::MulAdd(m, vw, m);
                           });
}

// Same as RMSNorm, but its HWY_RESTRICT forbids passing the same pointer.
template <typename WT, typename XT>
HWY_NOINLINE HWY_MAYBE_UNUSED void RMSNormInplace(const WT* HWY_RESTRICT weight,
                                                  XT* HWY_RESTRICT inout,
                                                  const size_t size,
                                                  hwy::Profiler& p,
                                                  const size_t worker) {
  static const auto zone = p.AddZone("Ops.RMSNormInplace");
  PROFILER_ZONE3(p, worker, zone);

  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  using VF = hn::Vec<DF>;

  const VF mul = hn::Set(DF(), detail::RMSNormMul(inout, size, p, worker));
  const VF* HWY_RESTRICT pmul = &mul;

  Decompress1AndCompressInplace(DF(), inout, size, weight,
                                [pmul](DF /*df*/, VF vx, VF vw) HWY_ATTR -> VF {
                                  const VF m = hn::Mul(*pmul, vx);
                                  // (1+weight) * m = m + weight*m = one FMA.
                                  return hn::MulAdd(m, vw, m);
                                });
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
  using DF = hn::ScalableTag<float>;
  const DF df;
  using VF = hn::Vec<DF>;

  double mu, mu2;
  ComputeMoments(x, size, mu, mu2);
  double var = mu2 - mu * mu;
  var = HWY_MAX(var, 0.0);
  var = 1.0 / sqrt(var + 1E-6);
  const VF vmu = hn::Set(df, static_cast<float>(mu));
  const VF vvar = hn::Set(df, static_cast<float>(var));
  const VF* HWY_RESTRICT pmu = &vmu;
  const VF* HWY_RESTRICT pvar = &vvar;

  Decompress3AndCompressTo(DF(), out, size, x, scale, bias,
                           [pmu, pvar](DF /*df*/, VF x, VF s, VF add)
                               HWY_ATTR -> VF {
                                 const VF centered = hn::Sub(x, *pmu);
                                 const VF mul = hn::Mul(s, *pvar);
                                 return hn::MulAdd(centered, mul, add);
                               });
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
    const float* HWY_RESTRICT inv_timescale, const int pos, hwy::Profiler& p,
    const size_t worker) {
  static const auto zone = p.AddZone("Ops.Rope");
  PROFILER_ZONE3(p, worker, zone);
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
    const float* HWY_RESTRICT inv_timescale, const int pos, hwy::Profiler& p,
    const size_t worker) {
  static const auto zone = p.AddZone("Ops.RopeAndMulBy");
  PROFILER_ZONE3(p, worker, zone);
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
                                                  const size_t size,
                                                  hwy::Profiler& p,
                                                  const size_t worker) {
  static const auto zone = p.AddZone("Ops.AddFrom");
  PROFILER_ZONE3(p, worker, zone);

  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  using VF = hn::Vec<DF>;
  Decompress1AndCompressInplace(DF(), out, size, x,
                                [&](DF /*df*/, VF out, VF x)
                                    HWY_ATTR -> VF { return hn::Add(x, out); });
}

// Simple loops unless/until batch sizes are large enough to parallelize.
template <typename XT, typename OT>
void RMSNormBatched(const MatPtrT<XT>& activations, const MatPtr& weights,
                    MatPtrT<OT>& out, ThreadingContext& ctx,
                    size_t cluster_idx = 0) {
  HWY_DASSERT(weights.Rows() == 1);
  HWY_DASSERT(weights.Cols() == activations.Cols());
  HWY_DASSERT(activations.SameShape(out));

  CallUpcasted(&weights, [&](const auto* weights_t) {
    ParallelFor(ParallelismStrategy::kFlat, activations.Rows(), ctx,
                cluster_idx, [&](uint64_t token_idx, size_t worker) {
                  RMSNorm(activations.Row(token_idx), weights_t->PackedScale1(),
                          out.Row(token_idx), activations.Cols(), ctx.profiler,
                          worker);
                });
  });
}

template <typename XT>
void RMSNormInplaceBatched(const MatPtr& weights, MatPtrT<XT>& inout,
                           ThreadingContext& ctx, size_t cluster_idx = 0) {
  HWY_DASSERT(weights.Rows() == 1);
  HWY_DASSERT(weights.Cols() == inout.Cols());

  CallUpcasted(&weights, [&](const auto* weights_t) {
    ParallelFor(ParallelismStrategy::kFlat, inout.Rows(), ctx, cluster_idx,
                [&](uint64_t token_idx, size_t worker) {
                  RMSNormInplace(weights_t->PackedScale1(),
                                 inout.Row(token_idx), inout.Cols(),
                                 ctx.profiler, worker);
                });
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
static HWY_INLINE void AddFromBatched(const MatPtrT<XT>& x, MatPtrT<float>& out,
                                      ThreadingContext& ctx,
                                      size_t cluster_idx = 0) {
  HWY_DASSERT(out.SameShape(x));
  ParallelFor(ParallelismStrategy::kFlat, out.Rows(), ctx, cluster_idx,
              [&](uint64_t token_idx, size_t worker) {
                AddFrom(x.Row(token_idx), out.Row(token_idx), x.Cols(),
                        ctx.profiler, worker);
              });
}

template <typename XT>
HWY_NOINLINE HWY_MAYBE_UNUSED void MulByConst(const float c, XT* HWY_RESTRICT x,
                                              const size_t size,
                                              hwy::Profiler& p,
                                              const size_t worker) {
  static const auto zone = p.AddZone("Ops.MulByConst");
  PROFILER_ZONE3(p, worker, zone);
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  using VF = hn::Vec<DF>;

  const VF vc = hn::Set(DF(), c);
  const VF* HWY_RESTRICT pc = &vc;

  DecompressAndCompressInplace(DF(), x, size,
                               [pc](DF /*df*/, VF x)
                                   HWY_ATTR -> VF { return hn::Mul(x, *pc); });
}

// Same as above, but with a separate output. Same as below without the add.
template <typename XT, typename OT>
HWY_NOINLINE HWY_MAYBE_UNUSED void MulByConstTo(
    const float c, const XT* HWY_RESTRICT x, OT* HWY_RESTRICT out,
    const size_t size, hwy::Profiler& p, const size_t worker) {
  static const auto zone = p.AddZone("Ops.MulByConstTo");
  PROFILER_ZONE3(p, worker, zone);
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  using VF = hn::Vec<DF>;

  const VF vc = hn::Set(DF(), c);
  const VF* HWY_RESTRICT pc = &vc;

  Decompress1AndCompressTo(DF(), out, size, x,
                           [pc](DF /*df*/, VF x)
                               HWY_ATTR -> VF { return hn::Mul(x, *pc); });
}

// out[i] += x[i] * c.
template <typename XT, typename OT>
HWY_NOINLINE HWY_MAYBE_UNUSED void MulByConstAndAdd(
    const float c, const XT* HWY_RESTRICT x, OT* HWY_RESTRICT out,
    const size_t size, hwy::Profiler& p, const size_t worker) {
  static const auto zone = p.AddZone("Ops.MulByConstAndAdd");
  PROFILER_ZONE3(p, worker, zone);
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  using VF = hn::Vec<DF>;

  const VF vc = hn::Set(DF(), c);
  const VF* HWY_RESTRICT pc = &vc;

  Decompress1AndCompressInplace(DF(), out, size, x,
                                [&](DF /*df*/, VF out, VF x) HWY_ATTR -> VF {
                                  return hn::MulAdd(x, *pc, out);
                                });
}

template <class DF, class VF = hn::Vec<DF>>
HWY_INLINE HWY_MAYBE_UNUSED void MulAdd16(
    DF df, const VF common, const VF split, VF& sum0, VF& sum1, VF& sum2,
    VF& sum3, VF& sum4, VF& sum5, VF& sum6, VF& sum7, VF& sum8, VF& sum9,
    VF& sum10, VF& sum11, VF& sum12, VF& sum13, VF& sum14, VF& sum15) {
  sum0 = hn::MulAdd(common, hn::Set(df, split.raw[0]), sum0);
  sum1 = hn::MulAdd(common, hn::Set(df, split.raw[1]), sum1);
  sum2 = hn::MulAdd(common, hn::Set(df, split.raw[2]), sum2);
  sum3 = hn::MulAdd(common, hn::Set(df, split.raw[3]), sum3);
  sum4 = hn::MulAdd(common, hn::Set(df, split.raw[4]), sum4);
  sum5 = hn::MulAdd(common, hn::Set(df, split.raw[5]), sum5);
  sum6 = hn::MulAdd(common, hn::Set(df, split.raw[6]), sum6);
  sum7 = hn::MulAdd(common, hn::Set(df, split.raw[7]), sum7);
  sum8 = hn::MulAdd(common, hn::Set(df, split.raw[8]), sum8);
  sum9 = hn::MulAdd(common, hn::Set(df, split.raw[9]), sum9);
  sum10 = hn::MulAdd(common, hn::Set(df, split.raw[10]), sum10);
  sum11 = hn::MulAdd(common, hn::Set(df, split.raw[11]), sum11);
  sum12 = hn::MulAdd(common, hn::Set(df, split.raw[12]), sum12);
  sum13 = hn::MulAdd(common, hn::Set(df, split.raw[13]), sum13);
  sum14 = hn::MulAdd(common, hn::Set(df, split.raw[14]), sum14);
  sum15 = hn::MulAdd(common, hn::Set(df, split.raw[15]), sum15);
}

template <class DF, class VF = hn::Vec<DF>>
HWY_INLINE HWY_MAYBE_UNUSED void MulAdd8(DF df, const VF common, const VF split,
                                         VF& sum0, VF& sum1, VF& sum2, VF& sum3,
                                         VF& sum4, VF& sum5, VF& sum6,
                                         VF& sum7) {
  sum0 = hn::MulAdd(common, hn::Set(df, split.raw[0]), sum0);
  sum1 = hn::MulAdd(common, hn::Set(df, split.raw[1]), sum1);
  sum2 = hn::MulAdd(common, hn::Set(df, split.raw[2]), sum2);
  sum3 = hn::MulAdd(common, hn::Set(df, split.raw[3]), sum3);
  sum4 = hn::MulAdd(common, hn::Set(df, split.raw[4]), sum4);
  sum5 = hn::MulAdd(common, hn::Set(df, split.raw[5]), sum5);
  sum6 = hn::MulAdd(common, hn::Set(df, split.raw[6]), sum6);
  sum7 = hn::MulAdd(common, hn::Set(df, split.raw[7]), sum7);
}

template <class DF, class VF = hn::Vec<DF>>
HWY_INLINE HWY_MAYBE_UNUSED void MulAdd4(DF df, const VF common, const VF split,
                                         VF& sum0, VF& sum1, VF& sum2,
                                         VF& sum3) {
  sum0 = hn::MulAdd(common, hn::Set(df, split.raw[0]), sum0);
  sum1 = hn::MulAdd(common, hn::Set(df, split.raw[1]), sum1);
  sum2 = hn::MulAdd(common, hn::Set(df, split.raw[2]), sum2);
  sum3 = hn::MulAdd(common, hn::Set(df, split.raw[3]), sum3);
}

// For an 8xNF tile of float values in 8xNF-lane registers, multiplies 8 rows
// of V by the corresponding values in c0-c7 and adds them to NF rows of out,
// after first prescaling out by scale.
// The depth (size) must be a multiple of NF.
template <class DF, class VF = hn::Vec<DF>>
HWY_NOINLINE HWY_MAYBE_UNUSED void MulByConstAndAddTile(
    DF df, const VF scale, const VF c0, const VF c1, const VF c2, const VF c3,
    const VF c4, const VF c5, const VF c6, const VF c7, const MatPtrT<float>& v,
    const size_t* HWY_RESTRICT pos, float* HWY_RESTRICT out,
    const uint32_t* HWY_RESTRICT out_offsets, const size_t size,
    hwy::Profiler& p, const size_t worker) {
  static const auto zone = p.AddZone("Ops.MulByConstAndAdd");
  PROFILER_ZONE3(p, worker, zone);
  namespace hn = hwy::HWY_NAMESPACE;
  HWY_LANES_CONSTEXPR size_t NF = hn::MaxLanes(df);

  size_t i = 0;
  while (i + NF <= size) {
    if HWY_LANES_CONSTEXPR (NF == 16) {
      VF out0, out1, out2, out3, out4, out5, out6, out7;
      VF out8, out9, out10, out11, out12, out13, out14, out15;
      out0 = hn::Load(df, out + i + out_offsets[0]);
      out1 = hn::Load(df, out + i + out_offsets[1]);
      out2 = hn::Load(df, out + i + out_offsets[2]);
      out3 = hn::Load(df, out + i + out_offsets[3]);
      out4 = hn::Load(df, out + i + out_offsets[4]);
      out5 = hn::Load(df, out + i + out_offsets[5]);
      out6 = hn::Load(df, out + i + out_offsets[6]);
      out7 = hn::Load(df, out + i + out_offsets[7]);
      out8 = hn::Load(df, out + i + out_offsets[8]);
      out9 = hn::Load(df, out + i + out_offsets[9]);
      out10 = hn::Load(df, out + i + out_offsets[10]);
      out11 = hn::Load(df, out + i + out_offsets[11]);
      out12 = hn::Load(df, out + i + out_offsets[12]);
      out13 = hn::Load(df, out + i + out_offsets[13]);
      out14 = hn::Load(df, out + i + out_offsets[14]);
      out15 = hn::Load(df, out + i + out_offsets[15]);
      out0 = hn::Mul(out0, hn::Set(df, scale.raw[0]));
      out1 = hn::Mul(out1, hn::Set(df, scale.raw[1]));
      out2 = hn::Mul(out2, hn::Set(df, scale.raw[2]));
      out3 = hn::Mul(out3, hn::Set(df, scale.raw[3]));
      out4 = hn::Mul(out4, hn::Set(df, scale.raw[4]));
      out5 = hn::Mul(out5, hn::Set(df, scale.raw[5]));
      out6 = hn::Mul(out6, hn::Set(df, scale.raw[6]));
      out7 = hn::Mul(out7, hn::Set(df, scale.raw[7]));
      out8 = hn::Mul(out8, hn::Set(df, scale.raw[8]));
      out9 = hn::Mul(out9, hn::Set(df, scale.raw[9]));
      out10 = hn::Mul(out10, hn::Set(df, scale.raw[10]));
      out11 = hn::Mul(out11, hn::Set(df, scale.raw[11]));
      out12 = hn::Mul(out12, hn::Set(df, scale.raw[12]));
      out13 = hn::Mul(out13, hn::Set(df, scale.raw[13]));
      out14 = hn::Mul(out14, hn::Set(df, scale.raw[14]));
      out15 = hn::Mul(out15, hn::Set(df, scale.raw[15]));
      VF x0 = hn::Load(df, v.Row(pos[0]) + i);
      MulAdd16(df, x0, c0, out0, out1, out2, out3, out4, out5, out6, out7, out8,
               out9, out10, out11, out12, out13, out14, out15);
      VF x1 = hn::Load(df, v.Row(pos[1]) + i);
      MulAdd16(df, x1, c1, out0, out1, out2, out3, out4, out5, out6, out7, out8,
               out9, out10, out11, out12, out13, out14, out15);
      VF x2 = hn::Load(df, v.Row(pos[2]) + i);
      MulAdd16(df, x2, c2, out0, out1, out2, out3, out4, out5, out6, out7, out8,
               out9, out10, out11, out12, out13, out14, out15);
      VF x3 = hn::Load(df, v.Row(pos[3]) + i);
      MulAdd16(df, x3, c3, out0, out1, out2, out3, out4, out5, out6, out7, out8,
               out9, out10, out11, out12, out13, out14, out15);
      VF x4 = hn::Load(df, v.Row(pos[4]) + i);
      MulAdd16(df, x4, c4, out0, out1, out2, out3, out4, out5, out6, out7, out8,
               out9, out10, out11, out12, out13, out14, out15);
      VF x5 = hn::Load(df, v.Row(pos[5]) + i);
      MulAdd16(df, x5, c5, out0, out1, out2, out3, out4, out5, out6, out7, out8,
               out9, out10, out11, out12, out13, out14, out15);
      VF x6 = hn::Load(df, v.Row(pos[6]) + i);
      MulAdd16(df, x6, c6, out0, out1, out2, out3, out4, out5, out6, out7, out8,
               out9, out10, out11, out12, out13, out14, out15);
      VF x7 = hn::Load(df, v.Row(pos[7]) + i);
      MulAdd16(df, x7, c7, out0, out1, out2, out3, out4, out5, out6, out7, out8,
               out9, out10, out11, out12, out13, out14, out15);
      hn::Store(out0, df, out + i + out_offsets[0]);
      hn::Store(out1, df, out + i + out_offsets[1]);
      hn::Store(out2, df, out + i + out_offsets[2]);
      hn::Store(out3, df, out + i + out_offsets[3]);
      hn::Store(out4, df, out + i + out_offsets[4]);
      hn::Store(out5, df, out + i + out_offsets[5]);
      hn::Store(out6, df, out + i + out_offsets[6]);
      hn::Store(out7, df, out + i + out_offsets[7]);
      hn::Store(out8, df, out + i + out_offsets[8]);
      hn::Store(out9, df, out + i + out_offsets[9]);
      hn::Store(out10, df, out + i + out_offsets[10]);
      hn::Store(out11, df, out + i + out_offsets[11]);
      hn::Store(out12, df, out + i + out_offsets[12]);
      hn::Store(out13, df, out + i + out_offsets[13]);
      hn::Store(out14, df, out + i + out_offsets[14]);
      hn::Store(out15, df, out + i + out_offsets[15]);
    }
    if HWY_LANES_CONSTEXPR (NF == 8) {
      VF out0, out1, out2, out3, out4, out5, out6, out7;
      out0 = hn::Load(df, out + i + out_offsets[0]);
      out1 = hn::Load(df, out + i + out_offsets[1]);
      out2 = hn::Load(df, out + i + out_offsets[2]);
      out3 = hn::Load(df, out + i + out_offsets[3]);
      out4 = hn::Load(df, out + i + out_offsets[4]);
      out5 = hn::Load(df, out + i + out_offsets[5]);
      out6 = hn::Load(df, out + i + out_offsets[6]);
      out7 = hn::Load(df, out + i + out_offsets[7]);
      out0 = hn::Mul(out0, hn::Set(df, scale.raw[0]));
      out1 = hn::Mul(out1, hn::Set(df, scale.raw[1]));
      out2 = hn::Mul(out2, hn::Set(df, scale.raw[2]));
      out3 = hn::Mul(out3, hn::Set(df, scale.raw[3]));
      out4 = hn::Mul(out4, hn::Set(df, scale.raw[4]));
      out5 = hn::Mul(out5, hn::Set(df, scale.raw[5]));
      out6 = hn::Mul(out6, hn::Set(df, scale.raw[6]));
      out7 = hn::Mul(out7, hn::Set(df, scale.raw[7]));
      VF x0 = hn::Load(df, v.Row(pos[0]) + i);
      MulAdd8(df, x0, c0, out0, out1, out2, out3, out4, out5, out6, out7);
      VF x1 = hn::Load(df, v.Row(pos[1]) + i);
      MulAdd8(df, x1, c1, out0, out1, out2, out3, out4, out5, out6, out7);
      VF x2 = hn::Load(df, v.Row(pos[2]) + i);
      MulAdd8(df, x2, c2, out0, out1, out2, out3, out4, out5, out6, out7);
      VF x3 = hn::Load(df, v.Row(pos[3]) + i);
      MulAdd8(df, x3, c3, out0, out1, out2, out3, out4, out5, out6, out7);
      VF x4 = hn::Load(df, v.Row(pos[4]) + i);
      MulAdd8(df, x4, c4, out0, out1, out2, out3, out4, out5, out6, out7);
      VF x5 = hn::Load(df, v.Row(pos[5]) + i);
      MulAdd8(df, x5, c5, out0, out1, out2, out3, out4, out5, out6, out7);
      VF x6 = hn::Load(df, v.Row(pos[6]) + i);
      MulAdd8(df, x6, c6, out0, out1, out2, out3, out4, out5, out6, out7);
      VF x7 = hn::Load(df, v.Row(pos[7]) + i);
      MulAdd8(df, x7, c7, out0, out1, out2, out3, out4, out5, out6, out7);
      hn::Store(out0, df, out + i + out_offsets[0]);
      hn::Store(out1, df, out + i + out_offsets[1]);
      hn::Store(out2, df, out + i + out_offsets[2]);
      hn::Store(out3, df, out + i + out_offsets[3]);
      hn::Store(out4, df, out + i + out_offsets[4]);
      hn::Store(out5, df, out + i + out_offsets[5]);
      hn::Store(out6, df, out + i + out_offsets[6]);
      hn::Store(out7, df, out + i + out_offsets[7]);
    }
    if HWY_LANES_CONSTEXPR (NF == 4) {
      VF out0, out1, out2, out3;
      out0 = hn::Load(df, out + i + out_offsets[0]);
      out1 = hn::Load(df, out + i + out_offsets[1]);
      out2 = hn::Load(df, out + i + out_offsets[2]);
      out3 = hn::Load(df, out + i + out_offsets[3]);
      out0 = hn::Mul(out0, hn::Set(df, scale.raw[0]));
      out1 = hn::Mul(out1, hn::Set(df, scale.raw[1]));
      out2 = hn::Mul(out2, hn::Set(df, scale.raw[2]));
      out3 = hn::Mul(out3, hn::Set(df, scale.raw[3]));
      VF x0 = hn::Load(df, v.Row(pos[0]) + i);
      MulAdd4(df, x0, c0, out0, out1, out2, out3);
      VF x1 = hn::Load(df, v.Row(pos[1]) + i);
      MulAdd4(df, x1, c1, out0, out1, out2, out3);
      VF x2 = hn::Load(df, v.Row(pos[2]) + i);
      MulAdd4(df, x2, c2, out0, out1, out2, out3);
      VF x3 = hn::Load(df, v.Row(pos[3]) + i);
      MulAdd4(df, x3, c3, out0, out1, out2, out3);
      VF x4 = hn::Load(df, v.Row(pos[4]) + i);
      MulAdd4(df, x4, c4, out0, out1, out2, out3);
      VF x5 = hn::Load(df, v.Row(pos[5]) + i);
      MulAdd4(df, x5, c5, out0, out1, out2, out3);
      VF x6 = hn::Load(df, v.Row(pos[6]) + i);
      MulAdd4(df, x6, c6, out0, out1, out2, out3);
      VF x7 = hn::Load(df, v.Row(pos[7]) + i);
      MulAdd4(df, x7, c7, out0, out1, out2, out3);
      hn::Store(out0, df, out + i + out_offsets[0]);
      hn::Store(out1, df, out + i + out_offsets[1]);
      hn::Store(out2, df, out + i + out_offsets[2]);
      hn::Store(out3, df, out + i + out_offsets[3]);
    }
    i += NF;
  }
  const size_t remaining = size - i;
  HWY_DASSERT(remaining == 0);
}

// Prescales NF rows of out by scale, then multiplies 1 row of V by the
// corresponding values in c0 and adds them to the NF rows of out.
// The depth (size) must be a multiple of NF.
template <class DF, class VF = hn::Vec<DF>>
HWY_NOINLINE HWY_MAYBE_UNUSED void MulByConstAndAddVector(
    DF df, const VF scale, const VF c0, const MatPtrT<float>& v,
    const size_t pos, float* HWY_RESTRICT out,
    const uint32_t* HWY_RESTRICT out_offsets, const size_t size,
    hwy::Profiler& p, const size_t worker) {
  static const auto zone = p.AddZone("Ops.MulByConstAndAdd");
  PROFILER_ZONE3(p, worker, zone);
  namespace hn = hwy::HWY_NAMESPACE;
  const size_t NF = hn::MaxLanes(df);

  size_t i = 0;
  while (i + NF <= size) {
    if constexpr (NF == 16) {
      VF out0, out1, out2, out3, out4, out5, out6, out7;
      VF out8, out9, out10, out11, out12, out13, out14, out15;
      out0 = hn::Load(df, out + i + out_offsets[0]);
      out1 = hn::Load(df, out + i + out_offsets[1]);
      out2 = hn::Load(df, out + i + out_offsets[2]);
      out3 = hn::Load(df, out + i + out_offsets[3]);
      out4 = hn::Load(df, out + i + out_offsets[4]);
      out5 = hn::Load(df, out + i + out_offsets[5]);
      out6 = hn::Load(df, out + i + out_offsets[6]);
      out7 = hn::Load(df, out + i + out_offsets[7]);
      out8 = hn::Load(df, out + i + out_offsets[8]);
      out9 = hn::Load(df, out + i + out_offsets[9]);
      out10 = hn::Load(df, out + i + out_offsets[10]);
      out11 = hn::Load(df, out + i + out_offsets[11]);
      out12 = hn::Load(df, out + i + out_offsets[12]);
      out13 = hn::Load(df, out + i + out_offsets[13]);
      out14 = hn::Load(df, out + i + out_offsets[14]);
      out15 = hn::Load(df, out + i + out_offsets[15]);
      out0 = hn::Mul(out0, hn::Set(df, scale.raw[0]));
      out1 = hn::Mul(out1, hn::Set(df, scale.raw[1]));
      out2 = hn::Mul(out2, hn::Set(df, scale.raw[2]));
      out3 = hn::Mul(out3, hn::Set(df, scale.raw[3]));
      out4 = hn::Mul(out4, hn::Set(df, scale.raw[4]));
      out5 = hn::Mul(out5, hn::Set(df, scale.raw[5]));
      out6 = hn::Mul(out6, hn::Set(df, scale.raw[6]));
      out7 = hn::Mul(out7, hn::Set(df, scale.raw[7]));
      out8 = hn::Mul(out8, hn::Set(df, scale.raw[8]));
      out9 = hn::Mul(out9, hn::Set(df, scale.raw[9]));
      out10 = hn::Mul(out10, hn::Set(df, scale.raw[10]));
      out11 = hn::Mul(out11, hn::Set(df, scale.raw[11]));
      out12 = hn::Mul(out12, hn::Set(df, scale.raw[12]));
      out13 = hn::Mul(out13, hn::Set(df, scale.raw[13]));
      out14 = hn::Mul(out14, hn::Set(df, scale.raw[14]));
      out15 = hn::Mul(out15, hn::Set(df, scale.raw[15]));
      VF x0 = hn::Load(df, v.Row(pos) + i);
      MulAdd16(df, x0, c0, out0, out1, out2, out3, out4, out5, out6, out7, out8,
               out9, out10, out11, out12, out13, out14, out15);
      hn::Store(out0, df, out + i + out_offsets[0]);
      hn::Store(out1, df, out + i + out_offsets[1]);
      hn::Store(out2, df, out + i + out_offsets[2]);
      hn::Store(out3, df, out + i + out_offsets[3]);
      hn::Store(out4, df, out + i + out_offsets[4]);
      hn::Store(out5, df, out + i + out_offsets[5]);
      hn::Store(out6, df, out + i + out_offsets[6]);
      hn::Store(out7, df, out + i + out_offsets[7]);
      hn::Store(out8, df, out + i + out_offsets[8]);
      hn::Store(out9, df, out + i + out_offsets[9]);
      hn::Store(out10, df, out + i + out_offsets[10]);
      hn::Store(out11, df, out + i + out_offsets[11]);
      hn::Store(out12, df, out + i + out_offsets[12]);
      hn::Store(out13, df, out + i + out_offsets[13]);
      hn::Store(out14, df, out + i + out_offsets[14]);
      hn::Store(out15, df, out + i + out_offsets[15]);
    } else if constexpr (NF == 8) {
      VF out0, out1, out2, out3, out4, out5, out6, out7;
      out0 = hn::Load(df, out + i + out_offsets[0]);
      out1 = hn::Load(df, out + i + out_offsets[1]);
      out2 = hn::Load(df, out + i + out_offsets[2]);
      out3 = hn::Load(df, out + i + out_offsets[3]);
      out4 = hn::Load(df, out + i + out_offsets[4]);
      out5 = hn::Load(df, out + i + out_offsets[5]);
      out6 = hn::Load(df, out + i + out_offsets[6]);
      out7 = hn::Load(df, out + i + out_offsets[7]);
      out0 = hn::Mul(out0, hn::Set(df, scale.raw[0]));
      out1 = hn::Mul(out1, hn::Set(df, scale.raw[1]));
      out2 = hn::Mul(out2, hn::Set(df, scale.raw[2]));
      out3 = hn::Mul(out3, hn::Set(df, scale.raw[3]));
      out4 = hn::Mul(out4, hn::Set(df, scale.raw[4]));
      out5 = hn::Mul(out5, hn::Set(df, scale.raw[5]));
      out6 = hn::Mul(out6, hn::Set(df, scale.raw[6]));
      out7 = hn::Mul(out7, hn::Set(df, scale.raw[7]));
      VF x0 = hn::Load(df, v.Row(pos) + i);
      MulAdd8(df, x0, c0, out0, out1, out2, out3, out4, out5, out6, out7);
      hn::Store(out0, df, out + i + out_offsets[0]);
      hn::Store(out1, df, out + i + out_offsets[1]);
      hn::Store(out2, df, out + i + out_offsets[2]);
      hn::Store(out3, df, out + i + out_offsets[3]);
      hn::Store(out4, df, out + i + out_offsets[4]);
      hn::Store(out5, df, out + i + out_offsets[5]);
      hn::Store(out6, df, out + i + out_offsets[6]);
      hn::Store(out7, df, out + i + out_offsets[7]);
    } else if constexpr (NF == 4) {
      VF out0, out1, out2, out3;
      out0 = hn::Load(df, out + i + out_offsets[0]);
      out1 = hn::Load(df, out + i + out_offsets[1]);
      out2 = hn::Load(df, out + i + out_offsets[2]);
      out3 = hn::Load(df, out + i + out_offsets[3]);
      out0 = hn::Mul(out0, hn::Set(df, scale.raw[0]));
      out1 = hn::Mul(out1, hn::Set(df, scale.raw[1]));
      out2 = hn::Mul(out2, hn::Set(df, scale.raw[2]));
      out3 = hn::Mul(out3, hn::Set(df, scale.raw[3]));
      VF x0 = hn::Load(df, v.Row(pos) + i);
      MulAdd4(df, x0, c0, out0, out1, out2, out3);
      hn::Store(out0, df, out + i + out_offsets[0]);
      hn::Store(out1, df, out + i + out_offsets[1]);
      hn::Store(out2, df, out + i + out_offsets[2]);
      hn::Store(out3, df, out + i + out_offsets[3]);
    } else {
      HWY_DASSERT(false);
    }
    i += NF;
  }
  const size_t remaining = size - i;
  HWY_DASSERT(remaining == 0);
}

// See below for a specialized version for top-1 sampling.
// TODO: support bf16 logits using Decompress2.
static HWY_NOINLINE void Softmax(Logits logits, hwy::Profiler& p,
                                 const size_t worker,
                                 float temperature = 1.0f) {
  static const auto zone = p.AddZone("Ops.Softmax");
  PROFILER_ZONE3(p, worker, zone);
  HWY_DASSERT(logits.size() != 0);

  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  using V = hn::Vec<D>;
  const D d;

  const V vmin = hn::Set(d, hwy::LowestValue<float>());
  V vmax = vmin;
  V* pmax = &vmax;  // workaround for SVE: cannot capture &vector directly
  hn::Foreach(d, logits.data(), logits.size(), vmin,
              [pmax](const auto d, const V value)
                  HWY_ATTR { *pmax = hn::Max(*pmax, value); });
  vmax = hn::MaxOfLanes(d, vmax);

  // Subtract max (avoid precision loss for large exponents) and exponentiate.
  hn::Transform(d, logits.data(), logits.size(),
                [pmax](const auto d, const V value) HWY_ATTR {
                  if constexpr (HWY_TARGET & HWY_ALL_SVE) {
                    // Workaround for buggy SVE codegen: avoid inlined Exp().
                    return hn::CallExp(d, hn::Sub(value, *pmax));
                  } else {
                    return hn::Exp(d, hn::Sub(value, *pmax));
                  }
                });

  if (temperature != 1.0f) {
    const float temperature_inv = 1.0f / temperature;
    hn::Transform(d, logits.data(), logits.size(),
                  [temperature_inv](const auto d, const V value) HWY_ATTR {
                    return hn::Mul(value, hn::Set(d, temperature_inv));
                  });
  }

  // Normalize to probability distribution. The exact sum seems like it should
  // not make a huge difference. It halves the standard deviation of the sum of
  // the normalized probabilities from 1E-7 to 5E-8, but actually also changes
  // the generated text after a few hundred tokens.
  const float sum_exp = Sum(d, logits.data(), logits.size());
  // Double-precision reciprocal does not appear to affect the results.
  const float mul = 1.0f / sum_exp;
  MulByConst(mul, logits.data(), logits.size(), p, worker);
}

// Note: https://arxiv.org/pdf/2001.04438 proposes to replace the three max /
// exp / mul passes with two passes, both of which compute Exp. This is
// reportedly only faster for very large arrays, larger even than our 256K
// vocab size. We instead fuse the subsequent sampling pass into the softmax,
// which already knows the max value which top-1 sampling would again seek.

// Returns the argmax and x[argmax].
static HWY_INLINE TokenAndProb ArgmaxAndMax(Logits logits) {
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  using V = hn::Vec<D>;
  using M = hn::Mask<D>;
  const D d;
  const hn::RebindToSigned<D> di;
  using TI = hn::TFromD<decltype(di)>;
  using VI = hn::Vec<decltype(di)>;
  const size_t N = hn::Lanes(d);
  HWY_ASSERT(logits.size() % (2 * N) == 0);

  V max0 = hn::Set(d, hwy::LowestValue<float>());
  V max1 = max0;
  VI argmax0 = hn::Zero(di);
  VI argmax1 = argmax0;

  for (size_t i = 0; i < logits.size(); i += 2 * N) {
    const V v0 = hn::LoadU(d, &logits[i]);
    const V v1 = hn::LoadU(d, &logits[i + N]);
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

// Returns argmax of softmax and its probability. This overwrites `logits`, but
// not with normalized probabilities. Only equivalent to `Softmax` +
// `sample_func` if `kTopK` == 1. This is worthwhile because `logits.size()` is
// typically `kVocabSize == 256K`, and this avoids writing and then scanning
// again for the max.
static HWY_MAYBE_UNUSED TokenAndProb Top1OfSoftmax(Logits logits) {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> d;
  using V = hn::Vec<decltype(d)>;

  const TokenAndProb argmax = ArgmaxAndMax(logits);

  // Subtract max (avoid precision loss for large exponents) and exponentiate.
  const V max = hn::Set(d, argmax.prob);
  const V* pmax = &max;
  hn::Transform(d, logits.data(), logits.size(),
                [pmax](const auto d, const V value) HWY_ATTR {
                  if constexpr (HWY_TARGET & HWY_ALL_SVE) {
                    // Temporary workaround for buggy SVE codegen: avoid inlined
                    // Exp().
                    return hn::CallExp(d, hn::Sub(value, *pmax));
                  } else {
                    return hn::Exp(d, hn::Sub(value, *pmax));
                  }
                });

  // Normalize to a single probability. The exact sum seems like it should not
  // make a huge difference. It halves the standard deviation of the sum of the
  // normalized probabilities from 1E-7 to 5E-8, but actually also changes the
  // generated text after a few hundred tokens.
  const float sum_exp = Sum(d, logits.data(), logits.size());
  const float prob = logits[argmax.token] / sum_exp;
  return TokenAndProb{.token = argmax.token, .prob = prob};
}

static HWY_NOINLINE void LogitsSoftCap(const float cap, Logits logits,
                                       hwy::Profiler& p, const size_t worker) {
  static const auto zone = p.AddZone("Ops.LogitsSoftCap");
  PROFILER_ZONE3(p, worker, zone);

  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  using VF = hn::Vec<DF>;

  const VF vcap = hn::Set(DF(), cap);
  const VF vinv_cap = hn::Set(DF(), 1.0f / cap);
  const VF* HWY_RESTRICT pcap = &vcap;
  const VF* HWY_RESTRICT pinv_cap = &vinv_cap;

  DecompressAndCompressInplace(DF(), logits.data(), logits.size(),
                               [pcap, pinv_cap](DF d, VF v) HWY_ATTR -> VF {
                                 return hn::Mul(
                                     *pcap, hn::Tanh(d, hn::Mul(v, *pinv_cap)));
                               });
}

// Calls LogitsSoftCap if cap != 0.0f.
static HWY_INLINE HWY_MAYBE_UNUSED void MaybeLogitsSoftCap(
    const float cap, Logits logits, hwy::Profiler& p, const size_t worker) {
  if (cap != 0.0f) {
    LogitsSoftCap(cap, logits, p, worker);
  }
}

static HWY_INLINE HWY_MAYBE_UNUSED void MaybeLogitsSoftCapBatched(
    const float cap, MatPtrT<float>& x, const hwy::BitSet4096<>& non_eos,
    ThreadingContext& ctx, size_t cluster_idx = 0) {
  if (cap == 0.0f) return;
  ParallelFor(ParallelismStrategy::kFlat, x.Rows(), ctx, cluster_idx,
              [&](uint64_t task, size_t worker) {
                if (non_eos.Get(task)) {
                  LogitsSoftCap(cap, x.RowSpan(task), ctx.profiler, worker);
                }
              });
}

static HWY_NOINLINE HWY_MAYBE_UNUSED size_t SampleArgmax(Logits logits) {
  size_t max_index = 0;
  float max_prob = logits[0];
  for (size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > max_prob) {
      max_index = i;
      max_prob = logits[i];
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
    Logits logits, size_t k, TAcceptToken& accept_token) {
  HWY_ASSERT(k != 0);
  HWY_ASSERT(k <= logits.size());
  std::vector<double> packed_token_probs;
  for (int32_t i = 0; i < static_cast<int32_t>(logits.size()); ++i) {
    if (accept_token && !accept_token(i, logits[i])) {
      continue;
    }
    packed_token_probs.push_back(PackTokenAndProb(i, logits[i]));
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
HWY_NOINLINE HWY_MAYBE_UNUSED int SampleTopK(Logits logits, size_t k,
                                             RngStream& gen, float temperature,
                                             TAcceptToken& accept_token) {
  std::vector<TokenAndProb> token_probs = TopK(logits, k, accept_token);
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
    Logits logits, size_t k, RngStream& gen, float temperature,
    TAcceptToken& accept_token, hwy::Profiler& p, size_t worker) {
  // Softmax and sample top-K is equivalent to taking the top-K logits and
  // sampling from the softmax of the top-K logits. The latter is faster as it
  // avoids computing the softmax of all logits.
  std::vector<TokenAndProb> token_logits = TopK(logits, k, accept_token);
  std::vector<int> topk_indices(k);
  std::vector<float> topk_logits(k);
  for (size_t i = 0; i < token_logits.size(); ++i) {
    topk_indices[i] = token_logits[i].token;
    topk_logits[i] = token_logits[i].prob;
  }

  const size_t mask = token_logits.size();
  Softmax(Logits(topk_logits.data(), mask), p, worker, temperature);
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
MatStorageT<T> AvgPool4x4(MatStorageT<T>& input, const Allocator& allocator) {
  const Extents2D extents = input.Extents();
  // Input validation
  HWY_DASSERT(extents.rows == 4096);  // 64 * 64 = 4096 input rows
  // Create output with 256 rows and same number of columns
  const size_t out_rows = 256;  // 16 * 16 = 256 output rows
  MatStorageT<T> result("pool4x4", Extents2D(out_rows, extents.cols), allocator,
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
