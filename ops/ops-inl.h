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
#include <limits>
#include <random>
#include <type_traits>  // std::enable_if_t
#include <vector>

#include "ops/matmul.h"
#include "ops/ops.h"
#include "util/allocator.h"
#include "util/basics.h"  // TokenAndProb, RngStream
#include "util/mat.h"
#include "util/threading_context.h"
#include "util/zones.h"
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
#include "hwy/contrib/math/fast_math-inl.h"
#include "hwy/contrib/math/math-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Computes C = A * B + add via MatMulStatic.
// This function uses CallUpcasted to dispatch to the correct MatMulStatic
// instantiation based on the runtime type of B.
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
float RMSNormMul(const VT* HWY_RESTRICT x, const size_t size,
                 ThreadingContext& ctx, const size_t worker) {
  GCPP_ZONE(ctx, worker, Zones::kOpsRmsNormMul);

  const hn::ScalableTag<float> d;
  const float l2 = DecompressAndCall(d, MakeSpan(x, size), DotKernelDefault());
  constexpr float kEps = 1e-6f;  // avoid divide by zero
  return 1.0f / sqrtf(l2 / StaticCast<float>(size) + kEps);
}

}  // namespace detail

// `kPlainWeight = false`: gemma weights are exported as scale-1 and scaled by
// (1 + weight). `kPlainWeight = true`: the checkpoint stores the true scale
// (DeepSeek), so the weight multiplies directly.
template <bool kPlainWeight = false, typename XT, typename WT, typename OT>
HWY_NOINLINE HWY_MAYBE_UNUSED void RMSNorm(
    const XT* HWY_RESTRICT x, const WT* HWY_RESTRICT weight, const size_t w_ofs,
    OT* HWY_RESTRICT out, const size_t size, ThreadingContext& ctx,
    const size_t worker) {
  GCPP_ZONE(ctx, worker, Zones::kOpsRmsNorm);

  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  using VF = hn::Vec<DF>;

  const VF mul = hn::Set(DF(), detail::RMSNormMul(x, size, ctx, worker));
  const VF* HWY_RESTRICT pmul = &mul;

  Decompress2AndCompressTo(DF(), out, size, x, weight, w_ofs,
                           [pmul](DF /*df*/, VF vx, VF vw) HWY_ATTR -> VF {
                             const VF m = hn::Mul(*pmul, vx);
                             if constexpr (kPlainWeight) {
                               return hn::Mul(m, vw);
                             } else {
                               // (1+weight) * m = m + weight*m = one FMA.
                               return hn::MulAdd(m, vw, m);
                             }
                           });
}

// Same as RMSNorm, but its HWY_RESTRICT forbids passing the same pointer.
template <bool kPlainWeight = false, typename WT, typename XT>
HWY_NOINLINE HWY_MAYBE_UNUSED void RMSNormInplace(
    const WT* HWY_RESTRICT weight, const size_t w_ofs, XT* HWY_RESTRICT inout,
    const size_t size, ThreadingContext& ctx, const size_t worker) {
  GCPP_ZONE(ctx, worker, Zones::kOpsRmsNormInplace);
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  using VF = hn::Vec<DF>;

  const VF mul = hn::Set(DF(), detail::RMSNormMul(inout, size, ctx, worker));
  const VF* HWY_RESTRICT pmul = &mul;

  Decompress1AndCompressInplace(DF(), inout, size, weight, w_ofs,
                                [pmul](DF /*df*/, VF vx, VF vw) HWY_ATTR -> VF {
                                  const VF m = hn::Mul(*pmul, vx);
                                  if constexpr (kPlainWeight) {
                                    return hn::Mul(m, vw);
                                  } else {
                                    // (1+weight) * m = m + weight*m = one FMA.
                                    return hn::MulAdd(m, vw, m);
                                  }
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
    const float* HWY_RESTRICT inv_timescale, const int pos,
    ThreadingContext& ctx, const size_t worker) {
  GCPP_ZONE(ctx, worker, Zones::kOpsRope);
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
    const float* HWY_RESTRICT inv_timescale, const int pos,
    ThreadingContext& ctx, const size_t worker) {
  GCPP_ZONE(ctx, worker, Zones::kOpsRopeAndMulBy);
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
                                                  ThreadingContext& ctx,
                                                  const size_t worker) {
  GCPP_ZONE(ctx, worker, Zones::kOpsAddFrom);

  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  using VF = hn::Vec<DF>;
  Decompress1AndCompressInplace(DF(), out, size, x, /*p1_ofs=*/0,
                                [&](DF /*df*/, VF out, VF x)
                                    HWY_ATTR -> VF { return hn::Add(x, out); });
}

// Simple loops unless/until batch sizes are large enough to parallelize.
// See RMSNorm for `kPlainWeight`.
template <bool kPlainWeight = false, typename XT, typename OT>
void RMSNormBatched(const MatPtrT<XT>& activations, const MatPtr& weights,
                    MatPtrT<OT>& out, ThreadingContext& ctx,
                    size_t cluster_idx = 0) {
  HWY_DASSERT(weights.Rows() == 1);
  HWY_DASSERT(weights.Cols() == activations.Cols());
  activations.DebugCheckSameShape(out);

  CallUpcasted(&weights, [&](const auto* weights_t) {
    ParallelFor(
        Parallelism::kFlat, activations.Rows(), ctx, cluster_idx,
        Callers::kOpsRMSNormBatched, [&](uint64_t token_idx, size_t worker) {
          RMSNorm<kPlainWeight>(
              activations.Row(token_idx), weights_t->PackedScale1(),
              /*w_ofs=*/0, out.Row(token_idx), activations.Cols(), ctx, worker);
        });
  });
}

template <bool kPlainWeight = false, typename XT>
void RMSNormInplaceBatched(const MatPtr& weights, MatPtrT<XT>& inout,
                           ThreadingContext& ctx, size_t cluster_idx = 0) {
  HWY_DASSERT(weights.Rows() == 1);
  HWY_DASSERT(weights.Cols() == inout.Cols());

  CallUpcasted(&weights, [&](const auto* weights_t) {
    ParallelFor(Parallelism::kFlat, inout.Rows(), ctx, cluster_idx,
                Callers::kOpsRMSNormInplaceBatched,
                [&](uint64_t token_idx, size_t worker) {
                  RMSNormInplace<kPlainWeight>(weights_t->PackedScale1(),
                                               /*w_ofs=*/0,
                                               inout.Row(token_idx),
                                               inout.Cols(), ctx, worker);
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
  ParallelFor(
      Parallelism::kFlat, out.Rows(), ctx, cluster_idx,
      Callers::kOpsAddFromBatched, [&](uint64_t token_idx, size_t worker) {
        AddFrom(x.Row(token_idx), out.Row(token_idx), x.Cols(), ctx, worker);
      });
}

// No profiler zone because this is short and frequently called.
template <typename XT>
HWY_NOINLINE HWY_MAYBE_UNUSED void MulByConst(const float c, XT* HWY_RESTRICT x,
                                              const size_t size) {
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
    const size_t size, ThreadingContext& ctx, const size_t worker) {
  GCPP_ZONE(ctx, worker, Zones::kOpsMulByConstTo);
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
HWY_NOINLINE HWY_MAYBE_UNUSED void MulByConstAndAdd(const float c,
                                                    const XT* HWY_RESTRICT x,
                                                    OT* HWY_RESTRICT out,
                                                    const size_t size) {
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  using VF = hn::Vec<DF>;

  const VF vc = hn::Set(DF(), c);
  const VF* HWY_RESTRICT pc = &vc;

  Decompress1AndCompressInplace(DF(), out, size, x, /*p1_ofs=*/0,
                                [&](DF /*df*/, VF out, VF x) HWY_ATTR -> VF {
                                  return hn::MulAdd(x, *pc, out);
                                });
}

// Same as `RMSNormInplace` without weights (no learned scale).
// Same as above, but in-place (`x == out`).
template <typename VecT>
HWY_NOINLINE void RMSNormNoScaleInplace(VecT* HWY_RESTRICT inout,
                                        const size_t size,
                                        ThreadingContext& ctx,
                                        const size_t worker) {
  GCPP_ZONE(ctx, worker, Zones::kOpsRmsNormNoScaleInplace);

  const float mul = detail::RMSNormMul(inout, size, ctx, worker);
  MulByConst(mul, inout, size);
}

// RMSNormInplace without scale parameters.
template <typename InOutT>
void RMSNormNoScaleInplaceBatched(
    MatPtrT<InOutT>& inout, ThreadingContext& ctx, size_t cluster_idx = 0,
    Parallelism parallelism = Parallelism::kFlat) {
  ParallelFor(parallelism, inout.Rows(), ctx, cluster_idx,
              Callers::kOpsRMSNormNoScaleInplaceBatched,
              [&](uint64_t token_idx, size_t worker) {
                RMSNormNoScaleInplace(inout.Row(token_idx), inout.Cols(), ctx,
                                      worker);
              });
}

// Grouped RMSNorm: applies RMSNorm independently to each group.
template <typename XT, typename WT, typename OT>
static HWY_NOINLINE void GroupedRMSNorm(const XT* HWY_RESTRICT x,
                                        const WT* HWY_RESTRICT weight,
                                        OT* HWY_RESTRICT out, const size_t size,
                                        const size_t num_groups,
                                        ThreadingContext& ctx,
                                        const size_t worker) {
  HWY_DASSERT(reinterpret_cast<uintptr_t>(out) !=
              reinterpret_cast<uintptr_t>(x));
  HWY_DASSERT(size % num_groups == 0);
  const size_t group_size = size / num_groups;
  for (size_t i = 0; i < size; i += group_size) {
    RMSNorm(x + i, weight, i, out + i, group_size, ctx, worker);
  }
}

// Same as above, but in-place (`x == out`).
template <typename WT, typename XT>
static HWY_NOINLINE void GroupedRMSNormInplace(
    const WT* HWY_RESTRICT weight, XT* HWY_RESTRICT inout, const size_t size,
    const size_t num_groups, ThreadingContext& ctx, const size_t worker) {
  HWY_DASSERT(size % num_groups == 0);
  const size_t group_size = size / num_groups;
  for (size_t i = 0; i < size; i += group_size) {
    RMSNormInplace(weight, i, inout + i, group_size, ctx, worker);
  }
}

template <typename XT, typename OT>
void GroupedRMSNormBatched(
    const MatPtrT<XT>& activations, const MatPtr& weights, MatPtrT<OT>& out,
    const size_t num_groups, ThreadingContext& ctx, size_t cluster_idx = 0,
    Parallelism parallelism = Parallelism::kFlat) {
  CallUpcasted(&weights, [&](const auto* weights_t) {
    ParallelFor(parallelism, activations.Rows(), ctx, cluster_idx,
                Callers::kOpsGroupedRMSNormBatched,
                [&](uint64_t token_idx, size_t worker) {
                  GroupedRMSNorm(activations.Row(token_idx),
                                 weights_t->PackedScale1(), out.Row(token_idx),
                                 activations.Cols(), num_groups, ctx, worker);
                });
  });
}

template <typename InOutT>
void GroupedRMSNormInplaceBatched(
    const MatPtr& weights, MatPtrT<InOutT>& inout, const size_t num_groups,
    ThreadingContext& ctx, size_t cluster_idx = 0,
    Parallelism parallelism = Parallelism::kFlat) {
  CallUpcasted(&weights, [&](const auto* weights_t) {
    ParallelFor(parallelism, inout.Rows(), ctx, cluster_idx,
                Callers::kOpsGroupedRMSNormInplaceBatched,
                [&](uint64_t token_idx, size_t worker) {
                  GroupedRMSNormInplace(weights_t->PackedScale1(),
                                        inout.Row(token_idx), inout.Cols(),
                                        num_groups, ctx, worker);
                });
  });
}

// Returns inclusive prefix sum for 4-lane vectors.
template <class D, class V, HWY_IF_LANES_D(D, 4)>
HWY_INLINE V InclusivePrefixSum4(D d, V DCBA) {
  const V CBAz = hn::Slide1Up(d, DCBA);
  const V DC_CB_BA_A = hn::Add(DCBA, CBAz);
  const V BA_A_z_z = hn::ShiftLeftLanes<2>(d, DC_CB_BA_A);
  return hn::Add(BA_A_z_z, DC_CB_BA_A);
}

template <typename T, HWY_IF_T_SIZE(T, 4)>
T ExclusivePrefixSum(const T* HWY_RESTRICT in, size_t size,
                     T* HWY_RESTRICT out) {
  HWY_DASSERT(hwy::IsAligned(in, 16));
  HWY_DASSERT(hwy::IsAligned(out, 16));
  const hn::FixedTag<T, 4> d;
  using V = hn::Vec<decltype(d)>;

  T sum = T{0};

  size_t i = 0;
  if (HWY_LIKELY(size >= 4)) {
    for (; i <= size - 4; i += 4) {
      const V v = hn::Load(d, in + i);
      const V inclusive = InclusivePrefixSum4(d, v);
      const T vec_sum = hn::ExtractLane(inclusive, 3);
      const V exclusive = hn::Add(hn::Slide1Up(d, inclusive), hn::Set(d, sum));
      sum += vec_sum;
      hn::Store(exclusive, d, out + i);
    }
  }

  for (; i < size; ++i) {
    out[i] = sum;
    sum += in[i];
  }
  return sum;  // grand total
}
template <class DF, class VF = hn::Vec<DF>>
HWY_INLINE HWY_MAYBE_UNUSED void MulAdd4(DF df, const VF common, const VF c0,
                                         const VF c1, const VF c2, const VF c3,
                                         VF& sum0, VF& sum1, VF& sum2,
                                         VF& sum3) {
  sum0 = hn::MulAdd(common, c0, sum0);
  sum1 = hn::MulAdd(common, c1, sum1);
  sum2 = hn::MulAdd(common, c2, sum2);
  sum3 = hn::MulAdd(common, c3, sum3);
}

template <class DF, class VF = hn::Vec<DF>>
HWY_INLINE HWY_MAYBE_UNUSED void MulAddNLanesVT4(
    DF df, const BF16* HWY_RESTRICT v, const float* HWY_RESTRICT c,
    const size_t num_lanes, VF& sum0a, VF& sum1a, VF& sum2a, VF& sum3a,
    VF& sum0b, VF& sum1b, VF& sum2b, VF& sum3b) {
  using DBF = hn::ScalableTag<BF16>;
  const DBF dbf;
  using VBF = hn::Vec<DBF>;
  const size_t kNF = hn::Lanes(df);
  for (size_t lane = 0; lane < num_lanes; ++lane, v += 2 * kNF) {
    VBF v0 = hn::Load(dbf, v);
    VF c0 = hn::Set(df, *c++);
    VF c1 = hn::Set(df, *c++);
    VF c2 = hn::Set(df, *c++);
    VF c3 = hn::Set(df, *c++);
    VF v0a = hn::PromoteLowerTo(df, v0);
    VF v0b = hn::PromoteUpperTo(df, v0);
    MulAdd4(df, v0a, c0, c1, c2, c3, sum0a, sum1a, sum2a, sum3a);
    MulAdd4(df, v0b, c0, c1, c2, c3, sum0b, sum1b, sum2b, sum3b);
  }
}

// For a 2NFx4 tile of float values in 8xNF-lane registers, multiplies 2NF rows
// of V by the corresponding values in c00-c31 and adds them to 2NF rows of out,
// after first prescaling out by scale.
// The depth (size) must be a multiple of 2NF.
template <class DF, class VF = hn::Vec<DF>>
HWY_INLINE HWY_MAYBE_UNUSED void MulByConstAndAddVT4Mem(
    DF df, const float* HWY_RESTRICT scales, const VF c00, const VF c01,
    const VF c10, const VF c11, const VF c20, const VF c21, const VF c30,
    const VF c31, const MatPtrT<BF16>& v, const size_t* HWY_RESTRICT pos,
    size_t num_lanes, float* HWY_RESTRICT out,
    const uint32_t* HWY_RESTRICT out_offsets, const size_t size) {
  namespace hn = hwy::HWY_NAMESPACE;
  HWY_LANES_CONSTEXPR size_t NF = hn::Lanes(df);
  constexpr size_t kMaxNF = hn::MaxLanes(df);
  const BF16* HWY_RESTRICT v_bf = v.Row(pos[0] / (2 * NF));
  HWY_DASSERT(pos[0] % (2 * NF) == 0);
  HWY_ALIGN float c_mem[8 * kMaxNF];
  hn::StoreInterleaved4(c00, c10, c20, c30, df, c_mem);
  hn::StoreInterleaved4(c01, c11, c21, c31, df, c_mem + 4 * NF);

  size_t i = 0;
  while (i + NF * 2 <= size) {
    VF out0a, out1a, out2a, out3a, out0b, out1b, out2b, out3b;
    out0a = hn::LoadU(df, out + i + out_offsets[0]);
    out1a = hn::LoadU(df, out + i + out_offsets[1]);
    out2a = hn::LoadU(df, out + i + out_offsets[2]);
    out3a = hn::LoadU(df, out + i + out_offsets[3]);
    VF scale0 = hn::Set(df, scales[0]);
    VF scale1 = hn::Set(df, scales[1]);
    VF scale2 = hn::Set(df, scales[2]);
    VF scale3 = hn::Set(df, scales[3]);
    out0a = hn::Mul(out0a, scale0);
    out1a = hn::Mul(out1a, scale1);
    out2a = hn::Mul(out2a, scale2);
    out3a = hn::Mul(out3a, scale3);
    out0b = hn::LoadU(df, out + i + NF + out_offsets[0]);
    out1b = hn::LoadU(df, out + i + NF + out_offsets[1]);
    out2b = hn::LoadU(df, out + i + NF + out_offsets[2]);
    out3b = hn::LoadU(df, out + i + NF + out_offsets[3]);
    out0b = hn::Mul(out0b, scale0);
    out1b = hn::Mul(out1b, scale1);
    out2b = hn::Mul(out2b, scale2);
    out3b = hn::Mul(out3b, scale3);
    MulAddNLanesVT4(df, v_bf, c_mem, HWY_MIN(num_lanes, 2 * NF), out0a, out1a,
                    out2a, out3a, out0b, out1b, out2b, out3b);
    hn::StoreU(out0a, df, out + i + out_offsets[0]);
    hn::StoreU(out1a, df, out + i + out_offsets[1]);
    hn::StoreU(out2a, df, out + i + out_offsets[2]);
    hn::StoreU(out3a, df, out + i + out_offsets[3]);
    hn::StoreU(out0b, df, out + i + NF + out_offsets[0]);
    hn::StoreU(out1b, df, out + i + NF + out_offsets[1]);
    hn::StoreU(out2b, df, out + i + NF + out_offsets[2]);
    hn::StoreU(out3b, df, out + i + NF + out_offsets[3]);
    i += NF * 2;
    v_bf += 4 * NF * NF;
  }
  if (i < size) {
    VF out0a, out1a, out2a, out3a, out0b, out1b, out2b, out3b;
    out0a = hn::LoadN(df, out + i + out_offsets[0], size - i);
    out1a = hn::LoadN(df, out + i + out_offsets[1], size - i);
    out2a = hn::LoadN(df, out + i + out_offsets[2], size - i);
    out3a = hn::LoadN(df, out + i + out_offsets[3], size - i);
    VF scale0 = hn::Set(df, scales[0]);
    VF scale1 = hn::Set(df, scales[1]);
    VF scale2 = hn::Set(df, scales[2]);
    VF scale3 = hn::Set(df, scales[3]);
    out0a = hn::Mul(out0a, scale0);
    out1a = hn::Mul(out1a, scale1);
    out2a = hn::Mul(out2a, scale2);
    out3a = hn::Mul(out3a, scale3);
    if (i + NF < size) {
      out0b = hn::LoadN(df, out + i + NF + out_offsets[0], size - i - NF);
      out1b = hn::LoadN(df, out + i + NF + out_offsets[1], size - i - NF);
      out2b = hn::LoadN(df, out + i + NF + out_offsets[2], size - i - NF);
      out3b = hn::LoadN(df, out + i + NF + out_offsets[3], size - i - NF);
      out0b = hn::Mul(out0b, scale0);
      out1b = hn::Mul(out1b, scale1);
      out2b = hn::Mul(out2b, scale2);
      out3b = hn::Mul(out3b, scale3);
    } else {
      out0b = hn::Zero(df);
      out1b = hn::Zero(df);
      out2b = hn::Zero(df);
      out3b = hn::Zero(df);
    }
    // Note that v_bf is always padded, so we can always load 2 * NF elements.
    MulAddNLanesVT4(df, v_bf, c_mem, HWY_MIN(num_lanes, 2 * NF), out0a, out1a,
                    out2a, out3a, out0b, out1b, out2b, out3b);
    hn::StoreN(out0a, df, out + i + out_offsets[0], size - i);
    hn::StoreN(out1a, df, out + i + out_offsets[1], size - i);
    hn::StoreN(out2a, df, out + i + out_offsets[2], size - i);
    hn::StoreN(out3a, df, out + i + out_offsets[3], size - i);
    if (i + NF < size) {
      hn::StoreN(out0b, df, out + i + NF + out_offsets[0], size - i - NF);
      hn::StoreN(out1b, df, out + i + NF + out_offsets[1], size - i - NF);
      hn::StoreN(out2b, df, out + i + NF + out_offsets[2], size - i - NF);
      hn::StoreN(out3b, df, out + i + NF + out_offsets[3], size - i - NF);
    }
  }
}

template <class DF, class VF = hn::Vec<DF>>
HWY_INLINE HWY_MAYBE_UNUSED void MulAddNLanesVT1(DF df,
                                                 const BF16* HWY_RESTRICT v,
                                                 const float* HWY_RESTRICT c,
                                                 const size_t num_lanes,
                                                 VF& sum0a, VF& sum0b) {
  using DBF = hn::ScalableTag<BF16>;
  const DBF dbf;
  using VBF = hn::Vec<DBF>;
  const size_t kNF = hn::Lanes(df);
  for (size_t lane = 0; lane < num_lanes; ++lane, v += 2 * kNF) {
    VBF v0 = hn::Load(dbf, v);
    VF c0 = hn::Set(df, *c++);
    VF v0a = hn::PromoteLowerTo(df, v0);
    VF v0b = hn::PromoteUpperTo(df, v0);
    sum0a = hn::MulAdd(v0a, c0, sum0a);
    sum0b = hn::MulAdd(v0b, c0, sum0b);
  }
}

template <class DF, class VF = hn::Vec<DF>>
HWY_INLINE HWY_MAYBE_UNUSED void MulByConstAndAddVT1Mem(
    DF df, const float* HWY_RESTRICT scales, const VF c00, const VF c01,
    const MatPtrT<BF16>& v, const size_t* HWY_RESTRICT pos, size_t num_lanes,
    float* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT out_offsets,
    const size_t size) {
  namespace hn = hwy::HWY_NAMESPACE;
  HWY_LANES_CONSTEXPR size_t NF = hn::Lanes(df);
  constexpr size_t kMaxNF = hn::MaxLanes(df);
  const BF16* HWY_RESTRICT v_bf = v.Row(pos[0] / (2 * NF));
  HWY_DASSERT(pos[0] % (2 * NF) == 0);
  HWY_ALIGN float c_mem[2 * kMaxNF];
  hn::Store(c00, df, c_mem);
  hn::Store(c01, df, c_mem + NF);

  size_t i = 0;
  while (i + NF * 2 <= size) {
    VF out0a, out0b;
    out0a = hn::LoadU(df, out + i + out_offsets[0]);
    VF scale0 = hn::Set(df, scales[0]);
    out0a = hn::Mul(out0a, scale0);
    out0b = hn::LoadU(df, out + i + NF + out_offsets[0]);
    out0b = hn::Mul(out0b, scale0);
    MulAddNLanesVT1(df, v_bf, c_mem, HWY_MIN(num_lanes, 2 * NF), out0a, out0b);
    hn::StoreU(out0a, df, out + i + out_offsets[0]);
    hn::StoreU(out0b, df, out + i + NF + out_offsets[0]);
    i += NF * 2;
    v_bf += 4 * NF * NF;
  }
  if (i < size) {
    VF out0a, out0b;
    out0a = hn::LoadN(df, out + i + out_offsets[0], size - i);
    VF scale0 = hn::Set(df, scales[0]);
    out0a = hn::Mul(out0a, scale0);
    if (i + NF < size) {
      out0b = hn::LoadN(df, out + i + NF + out_offsets[0], size - i - NF);
      out0b = hn::Mul(out0b, scale0);
    } else {
      out0b = hn::Zero(df);
    }
    MulAddNLanesVT1(df, v_bf, c_mem, HWY_MIN(num_lanes, 2 * NF), out0a, out0b);
    hn::StoreN(out0a, df, out + i + out_offsets[0], size - i);
    if (i + NF < size) {
      hn::StoreN(out0b, df, out + i + NF + out_offsets[0], size - i - NF);
    }
  }
}

template <int32_t N, typename DF, class VF = hn::Vec<DF>>
static HWY_INLINE void StoreUpTo8Times2(DF df, MatPtrT<float>& out,
                                        size_t start_col, VF out0_0, VF out0_1,
                                        VF out1_0, VF out1_1, VF out2_0,
                                        VF out2_1, VF out3_0, VF out3_1,
                                        VF out4_0, VF out4_1, VF out5_0,
                                        VF out5_1, VF out6_0, VF out6_1,
                                        VF out7_0, VF out7_1) {
  namespace hn = hwy::HWY_NAMESPACE;
  HWY_LANES_CONSTEXPR size_t NF = hn::Lanes(df);
  hn::Store(out0_0, df, out.Row(0) + start_col);
  hn::Store(out0_1, df, out.Row(0) + start_col + NF);
  if constexpr (N >= 2) {
    hn::Store(out1_0, df, out.Row(1) + start_col);
    hn::Store(out1_1, df, out.Row(1) + start_col + NF);
  }
  if constexpr (N >= 3) {
    hn::Store(out2_0, df, out.Row(2) + start_col);
    hn::Store(out2_1, df, out.Row(2) + start_col + NF);
  }
  if constexpr (N >= 4) {
    hn::Store(out3_0, df, out.Row(3) + start_col);
    hn::Store(out3_1, df, out.Row(3) + start_col + NF);
  }
  if constexpr (N >= 5) {
    hn::Store(out4_0, df, out.Row(4) + start_col);
    hn::Store(out4_1, df, out.Row(4) + start_col + NF);
  }
  if constexpr (N >= 6) {
    hn::Store(out5_0, df, out.Row(5) + start_col);
    hn::Store(out5_1, df, out.Row(5) + start_col + NF);
  }
  if constexpr (N >= 7) {
    hn::Store(out6_0, df, out.Row(6) + start_col);
    hn::Store(out6_1, df, out.Row(6) + start_col + NF);
  }
  if constexpr (N >= 8) {
    hn::Store(out7_0, df, out.Row(7) + start_col);
    hn::Store(out7_1, df, out.Row(7) + start_col + NF);
  }
}

template <int N, typename DF, class VF = hn::Vec<DF>>
static HWY_INLINE void LoadAndMulUpTo8Times2(
    DF df, MatPtrT<float>& out, size_t column, const float* HWY_RESTRICT scales,
    VF& out0_0, VF& out0_1, VF& out1_0, VF& out1_1, VF& out2_0, VF& out2_1,
    VF& out3_0, VF& out3_1, VF& out4_0, VF& out4_1, VF& out5_0, VF& out5_1,
    VF& out6_0, VF& out6_1, VF& out7_0, VF& out7_1) {
  namespace hn = hwy::HWY_NAMESPACE;
  HWY_LANES_CONSTEXPR size_t NF = hn::Lanes(df);
  out0_0 = hn::Load(df, out.Row(0) + column);
  out0_0 = hn::Mul(out0_0, hn::Set(df, scales[0]));
  out0_1 = hn::Load(df, out.Row(0) + column + NF);
  out0_1 = hn::Mul(out0_1, hn::Set(df, scales[0]));
  if constexpr (N >= 2) {
    out1_0 = hn::Load(df, out.Row(1) + column);
    out1_0 = hn::Mul(out1_0, hn::Set(df, scales[1]));
    out1_1 = hn::Load(df, out.Row(1) + column + NF);
    out1_1 = hn::Mul(out1_1, hn::Set(df, scales[1]));
  }
  if constexpr (N >= 3) {
    out2_0 = hn::Load(df, out.Row(2) + column);
    out2_0 = hn::Mul(out2_0, hn::Set(df, scales[2]));
    out2_1 = hn::Load(df, out.Row(2) + column + NF);
    out2_1 = hn::Mul(out2_1, hn::Set(df, scales[2]));
  }
  if constexpr (N >= 4) {
    out3_0 = hn::Load(df, out.Row(3) + column);
    out3_0 = hn::Mul(out3_0, hn::Set(df, scales[3]));
    out3_1 = hn::Load(df, out.Row(3) + column + NF);
    out3_1 = hn::Mul(out3_1, hn::Set(df, scales[3]));
  }
  if constexpr (N >= 5) {
    out4_0 = hn::Load(df, out.Row(4) + column);
    out4_0 = hn::Mul(out4_0, hn::Set(df, scales[4]));
    out4_1 = hn::Load(df, out.Row(4) + column + NF);
    out4_1 = hn::Mul(out4_1, hn::Set(df, scales[4]));
  }
  if constexpr (N >= 6) {
    out5_0 = hn::Load(df, out.Row(5) + column);
    out5_0 = hn::Mul(out5_0, hn::Set(df, scales[5]));
    out5_1 = hn::Load(df, out.Row(5) + column + NF);
    out5_1 = hn::Mul(out5_1, hn::Set(df, scales[5]));
  }
  if constexpr (N >= 7) {
    out6_0 = hn::Load(df, out.Row(6) + column);
    out6_0 = hn::Mul(out6_0, hn::Set(df, scales[6]));
    out6_1 = hn::Load(df, out.Row(6) + column + NF);
    out6_1 = hn::Mul(out6_1, hn::Set(df, scales[6]));
  }
  if constexpr (N >= 8) {
    out7_0 = hn::Load(df, out.Row(7) + column);
    out7_0 = hn::Mul(out7_0, hn::Set(df, scales[7]));
    out7_1 = hn::Load(df, out.Row(7) + column + NF);
    out7_1 = hn::Mul(out7_1, hn::Set(df, scales[7]));
  }
}

template <int32_t N, class DF, class VF = hn::Vec<DF>, typename VType>
HWY_INLINE HWY_MAYBE_UNUSED void MulByConstAndAddTileUpTo8(
    DF df, const float* HWY_RESTRICT scales_old, size_t actual_steps,
    const float* HWY_RESTRICT step_consts,
    const VType* const* HWY_RESTRICT step_v_tiles, MatPtrT<float>& out) {
  static_assert(N <= 8);
  namespace hn = hwy::HWY_NAMESPACE;
  const size_t qkv_dim = out.Cols();
  constexpr size_t kMaxLanes = hn::MaxLanes(df);
  HWY_LANES_CONSTEXPR size_t NF = hn::Lanes(df);

  size_t i = 0;
  HWY_DASSERT(qkv_dim % (NF * 2) == 0);
  while (i + NF * 2 <= qkv_dim) {
    VF out0_0, out1_0, out2_0, out3_0, out4_0, out5_0, out6_0, out7_0;
    VF out0_1, out1_1, out2_1, out3_1, out4_1, out5_1, out6_1, out7_1;
    LoadAndMulUpTo8Times2<N>(df, out, i, scales_old, out0_0, out0_1, out1_0,
                             out1_1, out2_0, out2_1, out3_0, out3_1, out4_0,
                             out4_1, out5_0, out5_1, out6_0, out6_1, out7_0,
                             out7_1);

    for (size_t step_idx = 0; step_idx < actual_steps; ++step_idx) {
      const float* consts_buffer = step_consts + step_idx * (N * 2 * kMaxLanes);
      const VType* v_tile = step_v_tiles[step_idx];
      PackedSpan<const VType> v_span = MakeConstSpan(v_tile, qkv_dim * 2 * NF);

      for (size_t t = 0; t < 2 * NF; ++t) {
        VF xI1, xI2;
        Decompress2(df, v_span, qkv_dim * t + i, xI1, xI2);

        auto mul_add = [&](int j, VF& o0, VF& o1) HWY_ATTR {
          const auto c = hn::Set(df, consts_buffer[t + 2 * j * kMaxLanes]);
          o0 = hn::MulAdd(xI1, c, o0);
          o1 = hn::MulAdd(xI2, c, o1);
        };

        mul_add(0, out0_0, out0_1);
        if constexpr (N >= 2) mul_add(1, out1_0, out1_1);
        if constexpr (N >= 3) mul_add(2, out2_0, out2_1);
        if constexpr (N >= 4) mul_add(3, out3_0, out3_1);
        if constexpr (N >= 5) mul_add(4, out4_0, out4_1);
        if constexpr (N >= 6) mul_add(5, out5_0, out5_1);
        if constexpr (N >= 7) mul_add(6, out6_0, out6_1);
        if constexpr (N >= 8) mul_add(7, out7_0, out7_1);
      }
    }
    StoreUpTo8Times2<N>(df, out, i, out0_0, out0_1, out1_0, out1_1, out2_0,
                        out2_1, out3_0, out3_1, out4_0, out4_1, out5_0, out5_1,
                        out6_0, out6_1, out7_0, out7_1);

    i += 2 * NF;
  }
  HWY_DASSERT(qkv_dim == i);
}

template <int32_t N, class DF, class VF = hn::Vec<DF>>
HWY_INLINE HWY_MAYBE_UNUSED void MulByConstAndAddTileUpTo8_BF16_Int16(
    DF df, const float* HWY_RESTRICT scales_old, size_t actual_steps,
    const int16_t* HWY_RESTRICT step_cs_i16,
    const int8_t* const* HWY_RESTRICT step_v_tiles,
    MatPtrT<float>& out,
    const float* const* HWY_RESTRICT step_q_scales_s) {
  static_assert(N <= 8);
  namespace hn = hwy::HWY_NAMESPACE;
  const size_t qkv_dim = out.Cols();
  constexpr size_t kMaxLanes = hn::MaxLanes(df);
  HWY_LANES_CONSTEXPR size_t NF = hn::Lanes(df);

  using DI16 = hn::Repartition<int16_t, DF>;
  const DI16 di16;
  const auto di16_half = hn::Half<DI16>();
  using DI32 = hn::Repartition<int32_t, DF>;
  const DI32 di32;
  using VI16 = hn::Vec<DI16>;
  using VI32 = hn::Vec<DI32>;
  using DI8 = hn::Repartition<int8_t, DF>;
  const hn::Half<DI8> di8_half;
  HWY_LANES_CONSTEXPR size_t kInt16Lanes = hn::Lanes(di16);

  size_t i = 0;
  HWY_DASSERT(qkv_dim % (NF * 2) == 0);
  while (i + 2 * NF <= qkv_dim) {
    VF out0_0, out1_0, out2_0, out3_0, out4_0, out5_0, out6_0, out7_0;
    VF out0_1, out1_1, out2_1, out3_1, out4_1, out5_1, out6_1, out7_1;
    LoadAndMulUpTo8Times2<N>(df, out, i, scales_old, out0_0, out0_1, out1_0,
                             out1_1, out2_0, out2_1, out3_0, out3_1, out4_0,
                             out4_1, out5_0, out5_1, out6_0, out6_1, out7_0,
                             out7_1);

    for (size_t step_idx = 0; step_idx < actual_steps; ++step_idx) {
      const int16_t* cs_i16 = step_cs_i16 + step_idx * (N * kMaxLanes * 2);
      const int8_t* v_tile = step_v_tiles[step_idx];
      const float* q_scales_s = step_q_scales_s[step_idx];

      VI32 acc0_0 = hn::Zero(di32), acc0_1 = hn::Zero(di32);
      VI32 acc1_0 = hn::Zero(di32), acc1_1 = hn::Zero(di32);
      VI32 acc2_0 = hn::Zero(di32), acc2_1 = hn::Zero(di32);
      VI32 acc3_0 = hn::Zero(di32), acc3_1 = hn::Zero(di32);
      VI32 acc4_0 = hn::Zero(di32), acc4_1 = hn::Zero(di32);
      VI32 acc5_0 = hn::Zero(di32), acc5_1 = hn::Zero(di32);
      VI32 acc6_0 = hn::Zero(di32), acc6_1 = hn::Zero(di32);
      VI32 acc7_0 = hn::Zero(di32), acc7_1 = hn::Zero(di32);

      VI32 acc0_o_0 = hn::Zero(di32), acc0_o_1 = hn::Zero(di32);
      VI32 acc1_o_0 = hn::Zero(di32), acc1_o_1 = hn::Zero(di32);
      VI32 acc2_o_0 = hn::Zero(di32), acc2_o_1 = hn::Zero(di32);
      VI32 acc3_o_0 = hn::Zero(di32), acc3_o_1 = hn::Zero(di32);
      VI32 acc4_o_0 = hn::Zero(di32), acc4_o_1 = hn::Zero(di32);
      VI32 acc5_o_0 = hn::Zero(di32), acc5_o_1 = hn::Zero(di32);
      VI32 acc6_o_0 = hn::Zero(di32), acc6_o_1 = hn::Zero(di32);
      VI32 acc7_o_0 = hn::Zero(di32), acc7_o_1 = hn::Zero(di32);

      for (size_t token_pair_idx = 0; token_pair_idx < NF; ++token_pair_idx) {
        VI16 vi_first8, vi_next8;

        const int8_t* v_ptr = v_tile + 2 * qkv_dim * token_pair_idx + i * 2;

        auto v8_t0 = hn::LoadU(di8_half, v_ptr);
        auto v16_t0 = hn::PromoteTo(di16, v8_t0);

        auto v8_t1 = hn::LoadU(di8_half, v_ptr + kInt16Lanes);
        auto v16_t1 = hn::PromoteTo(di16, v8_t1);

        vi_first8 = v16_t0;
        vi_next8 = v16_t1;

        auto mul_acc = [&](int j, VI32& a0, VI32& a_o0, VI32& a1,
                           VI32& a_o1) HWY_ATTR {
          const int32_t* s_ptr = reinterpret_cast<const int32_t*>(
              &cs_i16[2 * token_pair_idx + j * kMaxLanes * 2]);
          VI16 sj = hn::BitCast(di16, hn::Set(di32, *s_ptr));

          a0 = hn::ReorderWidenMulAccumulate(di32, vi_first8, sj, a0, a_o0);
          a1 = hn::ReorderWidenMulAccumulate(di32, vi_next8, sj, a1, a_o1);
        };

        mul_acc(0, acc0_0, acc0_o_0, acc0_1, acc0_o_1);
        if constexpr (N >= 2) mul_acc(1, acc1_0, acc1_o_0, acc1_1, acc1_o_1);
        if constexpr (N >= 3) mul_acc(2, acc2_0, acc2_o_0, acc2_1, acc2_o_1);
        if constexpr (N >= 4) mul_acc(3, acc3_0, acc3_o_0, acc3_1, acc3_o_1);
        if constexpr (N >= 5) mul_acc(4, acc4_0, acc4_o_0, acc4_1, acc4_o_1);
        if constexpr (N >= 6) mul_acc(5, acc5_0, acc5_o_0, acc5_1, acc5_o_1);
        if constexpr (N >= 7) mul_acc(6, acc6_0, acc6_o_0, acc6_1, acc6_o_1);
        if constexpr (N >= 8) mul_acc(7, acc7_0, acc7_o_0, acc7_1, acc7_o_1);
      }

      auto convert_and_add = [&](int j, VI32& a0, VI32& a_o0, VI32& a1,
                                 VI32& a_o1, VF& o0, VF& o1) HWY_ATTR {
        VF f0 = hn::ConvertTo(df, hn::RearrangeToOddPlusEven(a0, a_o0));
        VF f1 = hn::ConvertTo(df, hn::RearrangeToOddPlusEven(a1, a_o1));

        VF scale_new = hn::Set(df, q_scales_s[j]);
        o0 = hn::MulAdd(f0, scale_new, o0);
        o1 = hn::MulAdd(f1, scale_new, o1);
      };

      convert_and_add(0, acc0_0, acc0_o_0, acc0_1, acc0_o_1, out0_0, out0_1);
      if constexpr (N >= 2)
        convert_and_add(1, acc1_0, acc1_o_0, acc1_1, acc1_o_1, out1_0, out1_1);
      if constexpr (N >= 3)
        convert_and_add(2, acc2_0, acc2_o_0, acc2_1, acc2_o_1, out2_0, out2_1);
      if constexpr (N >= 4)
        convert_and_add(3, acc3_0, acc3_o_0, acc3_1, acc3_o_1, out3_0, out3_1);
      if constexpr (N >= 5)
        convert_and_add(4, acc4_0, acc4_o_0, acc4_1, acc4_o_1, out4_0, out4_1);
      if constexpr (N >= 6)
        convert_and_add(5, acc5_0, acc5_o_0, acc5_1, acc5_o_1, out5_0, out5_1);
      if constexpr (N >= 7)
        convert_and_add(6, acc6_0, acc6_o_0, acc6_1, acc6_o_1, out6_0, out6_1);
      if constexpr (N >= 8)
        convert_and_add(7, acc7_0, acc7_o_0, acc7_1, acc7_o_1, out7_0, out7_1);
    }
    StoreUpTo8Times2<N>(df, out, i, out0_0, out0_1, out1_0, out1_1, out2_0,
                        out2_1, out3_0, out3_1, out4_0, out4_1, out5_0, out5_1,
                        out6_0, out6_1, out7_0, out7_1);

    i += 2 * NF;
  }
}

template <int32_t N, class DF, class VF = hn::Vec<DF>, typename VType>
HWY_INLINE HWY_MAYBE_UNUSED void MulByConstAndAddTileUpTo8_BF16(
    DF df, const float* HWY_RESTRICT scales_old, size_t actual_steps,
    const BF16* HWY_RESTRICT step_cs,
    const VType* const* HWY_RESTRICT step_v_tiles, MatPtrT<float>& out) {
  static_assert(N <= 8);
  namespace hn = hwy::HWY_NAMESPACE;
  const size_t qkv_dim = out.Cols();
  HWY_LANES_CONSTEXPR size_t NF = hn::Lanes(df);
  constexpr size_t kMaxLanes = hn::MaxLanes(df);
  using DBF = hn::ScalableTag<BF16>;
  const DBF dbf;
  using VBF = hn::Vec<DBF>;
  size_t i = 0;
  HWY_DASSERT(qkv_dim % (NF * 2) == 0);
  while (i + NF * 2 <= qkv_dim) {
    VF out0_0, out1_0, out2_0, out3_0;
    VF out0_1, out1_1, out2_1, out3_1;
    VF out4_0, out5_0, out6_0, out7_0;
    VF out4_1, out5_1, out6_1, out7_1;
    LoadAndMulUpTo8Times2<N>(df, out, i, scales_old, out0_0, out0_1, out1_0,
                             out1_1, out2_0, out2_1, out3_0, out3_1, out4_0,
                             out4_1, out5_0, out5_1, out6_0, out6_1, out7_0,
                             out7_1);
    VF helper_out0_0 = hn::Zero(df), helper_out0_1 = hn::Zero(df),
       helper_out1_0 = hn::Zero(df), helper_out1_1 = hn::Zero(df),
       helper_out2_0 = hn::Zero(df), helper_out2_1 = hn::Zero(df),
       helper_out3_0 = hn::Zero(df), helper_out3_1 = hn::Zero(df),
       helper_out4_0 = hn::Zero(df), helper_out4_1 = hn::Zero(df),
       helper_out5_0 = hn::Zero(df), helper_out5_1 = hn::Zero(df),
       helper_out6_0 = hn::Zero(df), helper_out6_1 = hn::Zero(df),
       helper_out7_0 = hn::Zero(df), helper_out7_1 = hn::Zero(df);
    for (size_t step_idx = 0; step_idx < actual_steps; ++step_idx) {
      const BF16* cs = step_cs + step_idx * (N * kMaxLanes * 2);
      const float* cs_as_float = HWY_RCAST_ALIGNED(const float*, cs);
      const VType* v_tile = step_v_tiles[step_idx];
      PackedSpan<const VType> v_span = MakeConstSpan(v_tile, qkv_dim * 2 * NF);

      for (size_t token_pair_idx = 0; token_pair_idx < NF; ++token_pair_idx) {
        VBF xI, xI2;
        Decompress2(dbf, v_span, 2 * qkv_dim * token_pair_idx + i * 2, xI, xI2);

        // Set pair of c scales for 2 value vectors
        auto mul_acc = [&](int j, VF& o0, VF& h0, VF& o1, VF& h1) HWY_ATTR {
          VBF cs_pair = hn::BitCast(
              dbf, hn::Set(df, cs_as_float[token_pair_idx + j * kMaxLanes]));
          o0 = hn::ReorderWidenMulAccumulate(df, xI, cs_pair, o0, h0);
          o1 = hn::ReorderWidenMulAccumulate(df, xI2, cs_pair, o1, h1);
        };

        mul_acc(0, out0_0, helper_out0_0, out0_1, helper_out0_1);
        if constexpr (N >= 2)
          mul_acc(1, out1_0, helper_out1_0, out1_1, helper_out1_1);
        if constexpr (N >= 3)
          mul_acc(2, out2_0, helper_out2_0, out2_1, helper_out2_1);
        if constexpr (N >= 4)
          mul_acc(3, out3_0, helper_out3_0, out3_1, helper_out3_1);
        if constexpr (N >= 5)
          mul_acc(4, out4_0, helper_out4_0, out4_1, helper_out4_1);
        if constexpr (N >= 6)
          mul_acc(5, out5_0, helper_out5_0, out5_1, helper_out5_1);
        if constexpr (N >= 7)
          mul_acc(6, out6_0, helper_out6_0, out6_1, helper_out6_1);
        if constexpr (N >= 8)
          mul_acc(7, out7_0, helper_out7_0, out7_1, helper_out7_1);
      }
    }
#if HWY_NATIVE_DOT_BF16 == 0
    auto add_helper = [&](VF& o0, VF& o1, const VF& h0, const VF& h1) HWY_ATTR {
      o0 = hn::Add(o0, h0);
      o1 = hn::Add(o1, h1);
    };
    add_helper(out0_0, out0_1, helper_out0_0, helper_out0_1);
    if constexpr (N >= 2)
      add_helper(out1_0, out1_1, helper_out1_0, helper_out1_1);
    if constexpr (N >= 3)
      add_helper(out2_0, out2_1, helper_out2_0, helper_out2_1);
    if constexpr (N >= 4)
      add_helper(out3_0, out3_1, helper_out3_0, helper_out3_1);
    if constexpr (N >= 5)
      add_helper(out4_0, out4_1, helper_out4_0, helper_out4_1);
    if constexpr (N >= 6)
      add_helper(out5_0, out5_1, helper_out5_0, helper_out5_1);
    if constexpr (N >= 7)
      add_helper(out6_0, out6_1, helper_out6_0, helper_out6_1);
    if constexpr (N >= 8)
      add_helper(out7_0, out7_1, helper_out7_0, helper_out7_1);
#endif
    StoreUpTo8Times2<N>(df, out, i, out0_0, out0_1, out1_0, out1_1, out2_0,
                        out2_1, out3_0, out3_1, out4_0, out4_1, out5_0, out5_1,
                        out6_0, out6_1, out7_0, out7_1);

    i += 2 * NF;
  }
  HWY_DASSERT(qkv_dim == i);
}

// See below for a specialized version for top-1 sampling.
// TODO: support bf16 logits using Decompress2.
// Computes softmax probabilities for the given logits, normalizing in-place.
// The calculation is numerically stable, using the max-subtraction trick to
// compute exp(logits[i] - max(logits)) before normalizing by the sum.
// Temperature divides the logits before exponentiating; lower values
// concentrate the mass on the max, and zero puts all of it there (ties share).
// @param logits In-out: on input, contains logits; on output, overwritten with
// probabilities.
// @param ctx Input: threading context for parallelism and profiling.
// @param worker Input: worker thread index.
// @param temperature Input: softmax temperature.
// @param softmax_max_out Optional output: if not null, stores the max logit
// value.
// @param softmax_d_out Optional output: if softmax_max is not null, this must
// not be null and stores the sum of exp(logit - max).
static HWY_NOINLINE void Softmax(Logits logits, ThreadingContext& ctx,
                                 const size_t worker, float temperature = 1.0f,
                                 const SMOptions& sm_options = {}) {
  GCPP_ZONE(ctx, worker, Zones::kOpsSoftmax);
  HWY_DASSERT(logits.size() != 0);
  HWY_DASSERT(temperature >= 0.0f);

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

  if (temperature == 0.0f) {
    hn::Transform(d, logits.data(), logits.size(),
                  [pmax](const auto d, const V value) HWY_ATTR {
                    return hn::IfThenElseZero(hn::Eq(value, *pmax),
                                              hn::Set(d, 1.0f));
                  });
  } else {
    // Subtract max to avoid precision loss for large exponents, divide by the
    // temperature, and exponentiate.
    const float temperature_inv = 1.0f / temperature;
    hn::Transform(
        d, logits.data(), logits.size(),
        [pmax, temperature_inv](const auto d, const V value) HWY_ATTR {
          const V scaled =
              hn::Mul(hn::Sub(value, *pmax), hn::Set(d, temperature_inv));
          if constexpr (HWY_TARGET & HWY_ALL_SVE) {
            // Workaround for buggy SVE codegen: avoid inlined
            // FastExpMinusOrZero().
            return hn::CallFastExpMinusOrZero(d, scaled);
          } else {
            return hn::FastExpMinusOrZero(d, scaled);
          }
        });
  }

  // Normalize to probability distribution. The exact sum seems like it should
  // not make a huge difference. It halves the standard deviation of the sum of
  // the normalized probabilities from 1E-7 to 5E-8, but actually also changes
  // the generated text after a few hundred tokens.
  const float sum_exp = Sum(d, logits.data(), logits.size());
  // Double-precision reciprocal does not appear to affect the results.
  const float mul = 1.0f / sum_exp;
  MulByConst(mul, logits.data(), logits.size());
  if (sm_options.max_out) {
    *sm_options.max_out = hn::GetLane(vmax);
    *sm_options.d_out = sum_exp;
  }
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
                                       ThreadingContext& ctx,
                                       const size_t worker) {
  GCPP_ZONE(ctx, worker, Zones::kOpsLogitsSoftCap);

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
static HWY_INLINE HWY_MAYBE_UNUSED float MaybeLogitsSoftCap(const float cap,
                                                            const float logit) {
  if (cap == 0.0f) return logit;
  return cap * tanhf(logit / cap);
}

static HWY_INLINE HWY_MAYBE_UNUSED void MaybeLogitsSoftCap(
    const float cap, Logits logits, ThreadingContext& ctx,
    const size_t worker) {
  if (cap != 0.0f) {
    LogitsSoftCap(cap, logits, ctx, worker);
  }
}

static HWY_INLINE HWY_MAYBE_UNUSED void MaybeLogitsSoftCapBatched(
    const float cap, MatPtrT<float>& x, const hwy::BitSet4096<>& non_eos,
    ThreadingContext& ctx, size_t cluster_idx = 0) {
  if (cap == 0.0f) return;
  ParallelFor(Parallelism::kFlat, x.Rows(), ctx, cluster_idx,
              Callers::kOpsMaybeLogitsSoftCapBatched,
              [&](uint64_t task, size_t worker) {
                if (non_eos.Get(task)) {
                  LogitsSoftCap(cap, x.RowSpan(task), ctx, worker);
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
    TAcceptToken& accept_token, ThreadingContext& ctx, size_t worker) {
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
  Softmax(Logits(topk_logits.data(), mask), ctx, worker, temperature);
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
// This is surprisingly inexpensive for small images (<1 ms).
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

// Reduces each of x and stores in following lanes of max (tested with float32)
template <class DF, typename T = hn::TFromD<DF>,
          class DF4 = hn::CappedTag<T, 4>, class VF4 = hn::Vec<DF4>,
          class VF = hn::Vec<DF>, typename F>
static HWY_INLINE VF4 Reduce4(DF df, VF x_0, VF x_1, VF x_2, VF x_3,
                              F reducer) {
  const DF4 df4;
  constexpr size_t kMaxLanes = hn::MaxLanes(df);
  HWY_LANES_CONSTEXPR size_t kLanes = hn::Lanes(df);
  HWY_ALIGN T x_transposed[4 * kMaxLanes];
  hn::StoreInterleaved4(x_0, x_1, x_2, x_3, df, x_transposed);
  VF x01 =
      reducer(hn::Load(df, x_transposed), hn::Load(df, x_transposed + kLanes));
  VF x23 = reducer(hn::Load(df, x_transposed + 2 * kLanes),
                   hn::Load(df, x_transposed + 3 * kLanes));
  VF x0123 = reducer(x01, x23);
  hn::Store(x0123, df, x_transposed);

  VF4 result = hn::Load(df4, x_transposed);
  for (int i = 1; i < kLanes / 4; ++i) {
    result = reducer(result, hn::Load(df4, x_transposed + i * 4));
  }
  return result;
}

// Returns vector with 8 lanes. Shouldn't be used on architectures
// with less than 8 lanes per vector.
template <class DF, typename T = hn::TFromD<DF>,
          class DF8 = hn::CappedTag<T, 8>, class VF8 = hn::Vec<DF8>,
          class VF = hn::Vec<DF>, typename F>
static HWY_INLINE VF8 Reduce8(DF df, VF x_0, VF x_1, VF x_2, VF x_3, VF x_4,
                              VF x_5, VF x_6, VF x_7, F reducer) {
  auto res0123 = Reduce4(df, x_0, x_1, x_2, x_3, reducer);
  auto res4567 = Reduce4(df, x_4, x_5, x_6, x_7, reducer);

  using DF4 = hn::CappedTag<T, 4>;
  const DF4 df4;
  const DF8 df8;
  HWY_ALIGN T buf[8];
  hn::Store(res0123, df4, buf);
  hn::Store(res4567, df4, buf + 4);
  return hn::Load(df8, buf);
}

template <class DI32, class VI8, class VI32, HWY_IF_I32_D(DI32)>
HWY_API VI32 PerBlock2x2MatMulMaybeEmulate(DI32 di32, VI8 a, VI8 b, VI32 c) {
#if HWY_NATIVE_PER_BLOCK_2X2_MATMUL_INT8
  return hn::PerBlock2x2MatMul(di32, a, b, c);
#else
  const hn::Repartition<int8_t, DI32> di8;
  constexpr size_t kMaxN = hn::MaxLanes(di32);
  constexpr size_t kBufSize = kMaxN * 4;
  HWY_ALIGN int8_t in_a[kBufSize];
  HWY_ALIGN int8_t in_b[kBufSize];
  HWY_ALIGN int32_t expected[kMaxN];

  hn::Store(a, di8, in_a);
  hn::Store(b, di8, in_b);
  hn::Store(c, di32, expected);

  HWY_LANES_CONSTEXPR size_t N = hn::Lanes(di32);

  for (size_t block = 0; block < N; block += 4) {
    const size_t block_i8 = block * 4;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        int32_t sum = 0;
        for (int k = 0; k < 8; ++k) {
          sum += static_cast<int32_t>(in_a[block_i8 + i * 8 + k]) *
                 static_cast<int32_t>(in_b[block_i8 + j * 8 + k]);
        }
        expected[block + i * 2 + j] += sum;
      }
    }
  }
  return hn::Load(di32, expected);
#endif
}

template <class DN, class VBF, class VF, HWY_IF_F32_D(DN)>
HWY_API VF PerBlock2x2MatMulMaybeEmulate(DN dn, VBF a, VBF b, VF c) {
#if HWY_NATIVE_PER_BLOCK_2X2_MATMUL_BF16
  return hn::PerBlock2x2MatMul(dn, a, b, c);
#else
  const hn::Repartition<hwy::bfloat16_t, DN> dbf;
  const auto a_f = hn::BitCast(dn, a);
  const auto a1 = hn::BitCast(dbf, hn::Per4LaneBlockShuffle<2, 2, 0, 0>(a_f));
  const auto a2 = hn::BitCast(dbf, hn::Per4LaneBlockShuffle<3, 3, 1, 1>(a_f));

  const auto b_f = hn::BitCast(dn, b);
  const auto b1 = hn::BitCast(dbf, hn::Per4LaneBlockShuffle<2, 0, 2, 0>(b_f));
  const auto b2 = hn::BitCast(dbf, hn::Per4LaneBlockShuffle<3, 1, 3, 1>(b_f));

  VF sum1 = hn::Zero(dn);
  VF sum0 = hn::ReorderWidenMulAccumulate(dn, a1, b1, hn::Zero(dn), sum1);
  sum0 = hn::ReorderWidenMulAccumulate(dn, a2, b2, sum0, sum1);
  return hn::Add(hn::RearrangeToOddPlusEven(sum0, sum1), c);
#endif
}

static constexpr float kMaskedLogitVal =
    -std::numeric_limits<float>::max() / 64.0f;

template <int kVTileSize, class DF, class VF>
HWY_INLINE void ApplySoftCap(DF df, float att_cap, float one_over_cap, VF& x0,
                             VF& x1, VF& x2, VF& x3, VF& x4, VF& x5, VF& x6,
                             VF& x7) {
  if (att_cap > 0.0f) {
    VF cap = hn::Set(df, att_cap);
    VF one_over_cap_vec = hn::Set(df, one_over_cap);
    x0 = hn::Mul(cap, hn::CallFastTanh(df, hn::Mul(x0, one_over_cap_vec)));
    if constexpr (kVTileSize >= 2) {
      x1 = hn::Mul(cap, hn::CallFastTanh(df, hn::Mul(x1, one_over_cap_vec)));
    }
    if constexpr (kVTileSize >= 3) {
      x2 = hn::Mul(cap, hn::CallFastTanh(df, hn::Mul(x2, one_over_cap_vec)));
    }
    if constexpr (kVTileSize >= 4) {
      x3 = hn::Mul(cap, hn::CallFastTanh(df, hn::Mul(x3, one_over_cap_vec)));
    }
    if constexpr (kVTileSize >= 5) {
      x4 = hn::Mul(cap, hn::CallFastTanh(df, hn::Mul(x4, one_over_cap_vec)));
    }
    if constexpr (kVTileSize >= 6) {
      x5 = hn::Mul(cap, hn::CallFastTanh(df, hn::Mul(x5, one_over_cap_vec)));
    }
    if constexpr (kVTileSize >= 7) {
      x6 = hn::Mul(cap, hn::CallFastTanh(df, hn::Mul(x6, one_over_cap_vec)));
    }
    if constexpr (kVTileSize >= 8) {
      x7 = hn::Mul(cap, hn::CallFastTanh(df, hn::Mul(x7, one_over_cap_vec)));
    }
  }
}

template <int kNumQueries, class DF, class VF = hn::Vec<DF>,
          typename DU = hn::ScalableTag<uint32_t>, class VU = hn::Vec<DU>>
HWY_INLINE void ApplyMasking(DF df, DU du, size_t position,
                               const size_t* HWY_RESTRICT first_pos_per_query,
                               const size_t* HWY_RESTRICT last_pos_per_query,
                               VF& x0_p0, VF& x0_p1, VF& x1_p0, VF& x1_p1,
                               VF& x2_p0, VF& x2_p1, VF& x3_p0, VF& x3_p1,
                               VF& x4_p0, VF& x4_p1, VF& x5_p0, VF& x5_p1,
                               VF& x6_p0, VF& x6_p1, VF& x7_p0, VF& x7_p1) {
  VU lane_indices = hn::Iota(du, 0);
  HWY_LANES_CONSTEXPR size_t kTileSize = hn::Lanes(df);
  auto per_lane_pos_p0 = hn::Add(hn::Set(du, position), lane_indices);
  auto per_lane_pos_p1 =
      hn::Add(hn::Set(du, position + kTileSize), lane_indices);

  VF neg_inf = hn::Set(df, kMaskedLogitVal);

  auto apply_mask_for_query = [&](int query_idx, VF& x_p0, VF& x_p1) HWY_ATTR {
    const size_t first_pos = first_pos_per_query[query_idx];
    const size_t last_pos = last_pos_per_query[query_idx];

    auto valid_tokens_mask_p0 = hn::Ge(per_lane_pos_p0, hn::Set(du, first_pos));
    valid_tokens_mask_p0 = hn::And(
        valid_tokens_mask_p0, hn::Le(per_lane_pos_p0, hn::Set(du, last_pos)));
    x_p0 =
        hn::IfThenElse(hn::RebindMask(df, valid_tokens_mask_p0), x_p0, neg_inf);

    auto valid_tokens_mask_p1 = hn::Ge(per_lane_pos_p1, hn::Set(du, first_pos));
    valid_tokens_mask_p1 = hn::And(
        valid_tokens_mask_p1, hn::Le(per_lane_pos_p1, hn::Set(du, last_pos)));
    x_p1 =
        hn::IfThenElse(hn::RebindMask(df, valid_tokens_mask_p1), x_p1, neg_inf);
  };

  if constexpr (kNumQueries >= 1) {
    apply_mask_for_query(0, x0_p0, x0_p1);
  }
  if constexpr (kNumQueries >= 2) {
    apply_mask_for_query(1, x1_p0, x1_p1);
  }
  if constexpr (kNumQueries >= 3) {
    apply_mask_for_query(2, x2_p0, x2_p1);
  }
  if constexpr (kNumQueries >= 4) {
    apply_mask_for_query(3, x3_p0, x3_p1);
  }
  if constexpr (kNumQueries >= 5) {
    apply_mask_for_query(4, x4_p0, x4_p1);
  }
  if constexpr (kNumQueries >= 6) {
    apply_mask_for_query(5, x5_p0, x5_p1);
  }
  if constexpr (kNumQueries >= 7) {
    apply_mask_for_query(6, x6_p0, x6_p1);
  }
  if constexpr (kNumQueries >= 8) {
    apply_mask_for_query(7, x7_p0, x7_p1);
  }
}

template <int kNumQueries, class DF, class VF>
HWY_INLINE void MultiplyByScale(DF df, const BF16* scales, VF& x0_p0, VF& x0_p1,
                                VF& x1_p0, VF& x1_p1, VF& x2_p0, VF& x2_p1,
                                VF& x3_p0, VF& x3_p1, VF& x4_p0, VF& x4_p1,
                                VF& x5_p0, VF& x5_p1, VF& x6_p0, VF& x6_p1,
                                VF& x7_p0, VF& x7_p1) {
  const size_t kTileSize = hn::Lanes(df);
  const PackedSpan<const BF16> scales_span =
      MakeConstSpan(scales, 2 * kTileSize);
  VF scales_p0, scales_p1;
  Decompress2(df, scales_span, 0, scales_p0, scales_p1);
  if constexpr (kNumQueries >= 1) {
    x0_p0 = hn::Mul(x0_p0, scales_p0);
    x0_p1 = hn::Mul(x0_p1, scales_p1);
  }
  if constexpr (kNumQueries >= 2) {
    x1_p0 = hn::Mul(x1_p0, scales_p0);
    x1_p1 = hn::Mul(x1_p1, scales_p1);
  }
  if constexpr (kNumQueries >= 3) {
    x2_p0 = hn::Mul(x2_p0, scales_p0);
    x2_p1 = hn::Mul(x2_p1, scales_p1);
  }
  if constexpr (kNumQueries >= 4) {
    x3_p0 = hn::Mul(x3_p0, scales_p0);
    x3_p1 = hn::Mul(x3_p1, scales_p1);
  }
  if constexpr (kNumQueries >= 5) {
    x4_p0 = hn::Mul(x4_p0, scales_p0);
    x4_p1 = hn::Mul(x4_p1, scales_p1);
  }
  if constexpr (kNumQueries >= 6) {
    x5_p0 = hn::Mul(x5_p0, scales_p0);
    x5_p1 = hn::Mul(x5_p1, scales_p1);
  }
  if constexpr (kNumQueries >= 7) {
    x6_p0 = hn::Mul(x6_p0, scales_p0);
    x6_p1 = hn::Mul(x6_p1, scales_p1);
  }
  if constexpr (kNumQueries >= 8) {
    x7_p0 = hn::Mul(x7_p0, scales_p0);
    x7_p1 = hn::Mul(x7_p1, scales_p1);
  }
}

template <int kNumQueries, class DF, class VF>
HWY_INLINE void ApplyQuantizationScale(
    DF df, const float* HWY_RESTRICT q_scales, size_t query_idx, VF& x0_p0,
    VF& x0_p1, VF& x1_p0, VF& x1_p1, VF& x2_p0, VF& x2_p1, VF& x3_p0, VF& x3_p1,
    VF& x4_p0, VF& x4_p1, VF& x5_p0, VF& x5_p1, VF& x6_p0, VF& x6_p1, VF& x7_p0,
    VF& x7_p1) {
  auto apply_scale = [&](size_t i, VF& x_p0, VF& x_p1) HWY_ATTR {
    size_t scale_idx = query_idx + i;
    VF s = hn::Set(df, q_scales[scale_idx]);
    x_p0 = hn::Mul(x_p0, s);
    x_p1 = hn::Mul(x_p1, s);
  };

  if constexpr (kNumQueries >= 1) {
    apply_scale(0, x0_p0, x0_p1);
  }
  if constexpr (kNumQueries >= 2) {
    apply_scale(1, x1_p0, x1_p1);
  }
  if constexpr (kNumQueries >= 3) {
    apply_scale(2, x2_p0, x2_p1);
  }
  if constexpr (kNumQueries >= 4) {
    apply_scale(3, x3_p0, x3_p1);
  }
  if constexpr (kNumQueries >= 5) {
    apply_scale(4, x4_p0, x4_p1);
  }
  if constexpr (kNumQueries >= 6) {
    apply_scale(5, x5_p0, x5_p1);
  }
  if constexpr (kNumQueries >= 7) {
    apply_scale(6, x6_p0, x6_p1);
  }
  if constexpr (kNumQueries >= 8) {
    apply_scale(7, x7_p0, x7_p1);
  }
}

template <int kNumQueries, class DF, class VF>
HWY_INLINE void MultiplyByScaleAndQuantScale(
    DF df, const BF16* scales, const float* HWY_RESTRICT q_scales,
    size_t query_idx, VF& x0_p0, VF& x0_p1, VF& x1_p0, VF& x1_p1, VF& x2_p0,
    VF& x2_p1, VF& x3_p0, VF& x3_p1, VF& x4_p0, VF& x4_p1, VF& x5_p0, VF& x5_p1,
    VF& x6_p0, VF& x6_p1, VF& x7_p0, VF& x7_p1) {
  const size_t kTileSize = hn::Lanes(df);
  const PackedSpan<const BF16> scales_span =
      MakeConstSpan(scales, 2 * kTileSize);
  VF scales_p0, scales_p1;
  Decompress2(df, scales_span, 0, scales_p0, scales_p1);

  auto apply_both = [&](size_t i, VF& x_p0, VF& x_p1) HWY_ATTR {
    size_t scale_idx = query_idx + i;
    VF s = hn::Set(df, q_scales[scale_idx]);
    x_p0 = hn::Mul(x_p0, scales_p0);
    x_p1 = hn::Mul(x_p1, scales_p1);
    x_p0 = hn::Mul(x_p0, s);
    x_p1 = hn::Mul(x_p1, s);
  };

  if constexpr (kNumQueries >= 1) apply_both(0, x0_p0, x0_p1);
  if constexpr (kNumQueries >= 2) apply_both(1, x1_p0, x1_p1);
  if constexpr (kNumQueries >= 3) apply_both(2, x2_p0, x2_p1);
  if constexpr (kNumQueries >= 4) apply_both(3, x3_p0, x3_p1);
  if constexpr (kNumQueries >= 5) apply_both(4, x4_p0, x4_p1);
  if constexpr (kNumQueries >= 6) apply_both(5, x5_p0, x5_p1);
  if constexpr (kNumQueries >= 7) apply_both(6, x6_p0, x6_p1);
  if constexpr (kNumQueries >= 8) apply_both(7, x7_p0, x7_p1);
}

template <size_t kTargetLanes = 4, class D, typename V>
HWY_INLINE V SumReduceSegments(D d, V v) {
  constexpr size_t L = HWY_MAX_LANES_D(D);
  if constexpr (L <= kTargetLanes) {
    return v;
  } else {
    using D_half = hn::Half<D>;
    const D_half d_half;
    auto lo = hn::LowerHalf(d_half, v);
    auto hi = hn::UpperHalf(d_half, v);
    auto sum = hn::Add(lo, hi);
    auto reduced_half = SumReduceSegments<kTargetLanes>(d_half, sum);
    return hn::ZeroExtendVector(d, reduced_half);
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
