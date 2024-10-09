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
#include <limits>
#include <random>
#include <type_traits>  // std::enable_if_t

#include "compression/compress.h"
#include "util/allocator.h"  // TokenAndProb
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/detect_targets.h"
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
#include "hwy/contrib/algo/transform-inl.h"
#include "hwy/contrib/math/math-inl.h"
#include "hwy/profiler.h"  // also uses SIMD

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

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

template <class D, HWY_IF_F32_D(D)>
HWY_INLINE hn::Vec<D> Gelu(D d, hn::Vec<D> v) {
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
  const hn::ScalableTag<float> d;
  const float l2 = DecompressAndCall(d, MakeSpan(x, size), DotKernelDefault());
  constexpr float kEps = 1e-6f;  // avoid divide by zero
  return 1.0f / sqrtf(l2 / StaticCast<float>(size) + kEps);
}

}  // namespace detail

template <typename VecT, typename WeightT, typename OutT>
HWY_NOINLINE HWY_MAYBE_UNUSED void RMSNorm(const VecT* HWY_RESTRICT x,
                                           const WeightT* HWY_RESTRICT weight,
                                           OutT* HWY_RESTRICT out,
                                           const size_t size) {
  PROFILER_FUNC;

  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> df;
  using VF = hn::Vec<decltype(df)>;
  const size_t NF = hn::Lanes(df);

  const VF mul = hn::Set(df, detail::RMSNormMul(x, size));

  const auto packed_w = MakeSpan(weight, size);
  const auto packed_v = MakeSpan(x, size);
  const auto packed_out = MakeSpan(out, size);

  HWY_DASSERT(size % (2 * MaxLanes(df)) == 0);
  for (size_t i = 0; i < size; i += 2 * NF) {
    VF v0, v1, w0, w1;
    Decompress2(df, packed_v, i, v0, v1);
    Decompress2(df, packed_w, i, w0, w1);
    const VF m0 = hn::Mul(mul, v0);
    const VF m1 = hn::Mul(mul, v1);
    // (1+weight) * m = m + weight*m = one FMA.
    const VF out0 = hn::MulAdd(m0, w0, m0);
    const VF out1 = hn::MulAdd(m1, w1, m1);
    Compress2(df, out0, out1, packed_out, i);
  }
}

// Same as RMSNorm, but its HWY_RESTRICT forbids passing the same pointer.
template <typename VecT, typename WeightT>
HWY_NOINLINE HWY_MAYBE_UNUSED void RMSNormInplace(
    const WeightT* HWY_RESTRICT weight, VecT* HWY_RESTRICT inout,
    const size_t size) {
  PROFILER_FUNC;

  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> df;
  using VF = hn::Vec<decltype(df)>;
  const size_t NF = hn::Lanes(df);

  const VF mul = hn::Set(df, detail::RMSNormMul(inout, size));

  const auto packed_w = MakeSpan(weight, size);
  const auto packed_v = MakeSpan(inout, size);

  HWY_DASSERT(size % (2 * MaxLanes(df)) == 0);
  for (size_t i = 0; i < size; i += 2 * NF) {
    VF v0, v1, w0, w1;
    Decompress2(df, MakeConst(packed_v), i, v0, v1);
    Decompress2(df, packed_w, i, w0, w1);
    const VF m0 = hn::Mul(mul, v0);
    const VF m1 = hn::Mul(mul, v1);
    // (1+weight) * m = m + weight*m = one FMA.
    const VF out0 = hn::MulAdd(m0, w0, m0);
    const VF out1 = hn::MulAdd(m1, w1, m1);
    Compress2(df, out0, out1, packed_v, i);
  }
}

// Computes mean mu and mean of squares mu2 of a vector. Used in LayerNorm.
template <typename T>
HWY_NOINLINE void ScalarMus(const T* HWY_RESTRICT a, size_t size, T& mu,
                            T& mu2) {
  HWY_ASSERT(size > 0);
  double sum = 0.0;
  double sum2 = 0.0;
  for (size_t i = 0; i < size; ++i) {
    const float f = hwy::ConvertScalarTo<float>(a[i]);
    sum += f;
    sum2 += f * f;
  }
  mu = sum / size;
  mu2 = sum2 / size;
}

// Compare py/flax/linen/normalization.py.
// out = (x - mean) * scale * rsqrt(var + epsilon) + bias
template <typename VecT, typename WeightT, typename OutT>
HWY_NOINLINE void ScalarLayerNorm(const VecT* x,
                                  const WeightT* HWY_RESTRICT scale,
                                  const WeightT* HWY_RESTRICT bias,
                                  OutT* out,
                                  size_t size) {
  constexpr float kEps = 1e-6f;
  VecT mu, mu2;
  ScalarMus(x, size, mu, mu2);
  VecT var = mu2 - mu * mu;
  VecT zero = 0.0f;
  var = HWY_MAX(var, zero);
  var = 1.0f / sqrtf(var + kEps);
  for (size_t j = 0; j < size; j++) {
    const float v = hwy::ConvertScalarTo<float>(x[j]);
    const float s = hwy::ConvertScalarTo<float>(scale[j]);
    const float b = hwy::ConvertScalarTo<float>(bias[j]);
    out[j] = hwy::ConvertScalarTo<OutT>((v - mu) * s * var + b);
  }
}

template <typename VecT, typename WeightT, typename OutT>
HWY_NOINLINE HWY_MAYBE_UNUSED void LayerNorm(const VecT* x,
                                             const WeightT* HWY_RESTRICT weight,
                                             const WeightT* HWY_RESTRICT bias,
                                             OutT* out,
                                             const size_t size) {
  PROFILER_FUNC;
  // For now we only delegate to the scalar version.
  // TODO: implement vectorized version.
  ScalarLayerNorm(x, weight, bias, out, size);
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

// `inv_timescale[dim_qkv / 2]` is precomputed in Activations::Allocate.
// This overload is called from backprop/ and if kUseHalfRope.
static HWY_NOINLINE HWY_MAYBE_UNUSED void Rope(
    float* HWY_RESTRICT x, size_t dim_qkv,
    const float* HWY_RESTRICT inv_timescale, int pos) {
  PROFILER_FUNC;
  HWY_DASSERT(dim_qkv % 2 == 0);
  const size_t half_dim_qkv = dim_qkv / 2;
  for (size_t dim = 0; dim < half_dim_qkv; ++dim) {
    const float theta = StaticCast<float>(pos) * inv_timescale[dim];
    const float cos_val = cosf(theta);
    const float sin_val = sinf(theta);
    const float x0 = x[dim];
    const float x1 = x[dim + half_dim_qkv];
    x[dim] = x0 * cos_val - x1 * sin_val;
    x[dim + half_dim_qkv] = x0 * sin_val + x1 * cos_val;
  }
}

// `inv_timescale[dim_qkv / 2]` is precomputed in Activations::Allocate.
static HWY_NOINLINE HWY_MAYBE_UNUSED void RopeAndMulBy(
    const float mul, const float* HWY_RESTRICT x, size_t dim_qkv,
    const float* HWY_RESTRICT inv_timescale, int pos,
    float* HWY_RESTRICT x_out) {
  PROFILER_FUNC;
  HWY_DASSERT(dim_qkv % 2 == 0);
  const size_t half_dim_qkv = dim_qkv / 2;

  using D = hn::ScalableTag<float>;
  using V = hn::Vec<D>;
  const D d;

  // Vectorize computation for half_dim_qkv - (half_dim_qkv % Lanes)
  const size_t vectorizable_dims = hwy::RoundDownTo(half_dim_qkv, hn::Lanes(d));
  size_t dim = 0;
  for (; dim < vectorizable_dims; dim += hn::Lanes(d)) {
    // Compute thetas
    V pos_vec = hn::Set(d, pos);
    V inv_time_scale_vec = hn::LoadU(d, inv_timescale + dim);
    V theta_vec = hn::Mul(pos_vec, inv_time_scale_vec);

    // Compute rotations.
    V cos_theta_vec;
    V sin_theta_vec;
    hn::SinCos(d, theta_vec, sin_theta_vec, cos_theta_vec);

    // Scale input with rotations and multiply with constant.
    V mul_vec = hn::Set(d, mul);
    V x0_vec = hn::Mul(mul_vec, hn::LoadU(d, x + dim));
    V x1_vec = hn::Mul(mul_vec, hn::LoadU(d, x + dim + half_dim_qkv));

    V xout_0_vec = hn::MulSub(x0_vec, cos_theta_vec,
                                 hn::Mul(x1_vec, sin_theta_vec));
    V xout_1_vec = hn::MulAdd(x0_vec, sin_theta_vec,
                                            hn::Mul(x1_vec, cos_theta_vec));

    // Store
    hn::StoreU(xout_0_vec, d, x_out + dim);
    hn::StoreU(xout_1_vec, d, x_out + dim + half_dim_qkv);
  }

  // Vectorize computation for remaining dims - same as above, but with LoadN.
  const size_t remaining_dims = half_dim_qkv - dim;
  HWY_DASSERT(remaining_dims < hn::Lanes(d));  // at most one iteration
  if (remaining_dims != 0) {
    // Compute thetas
    V pos_vec = hn::Set(d, pos);
    V inv_time_scale_vec = hn::LoadN(d, inv_timescale + dim, remaining_dims);
    V theta_vec = hn::Mul(pos_vec, inv_time_scale_vec);

    // Compute rotations.
    V cos_theta_vec;
    V sin_theta_vec;
    hn::SinCos(d, theta_vec, sin_theta_vec, cos_theta_vec);

    // Scale input with rotations and multiply with constant.
    V mul_vec = hn::Set(d, mul);
    V x0_vec = hn::Mul(mul_vec, hn::LoadN(d, x + dim, remaining_dims));
    V x1_vec =
        hn::Mul(mul_vec, hn::LoadN(d, x + dim + half_dim_qkv, remaining_dims));

    V xout_0_vec =
        hn::MulSub(x0_vec, cos_theta_vec, hn::Mul(x1_vec, sin_theta_vec));
    V xout_1_vec =
        hn::MulAdd(x0_vec, sin_theta_vec, hn::Mul(x1_vec, cos_theta_vec));

    // Store
    hn::StoreN(xout_0_vec, d, x_out + dim, remaining_dims);
    hn::StoreN(xout_1_vec, d, x_out + dim + half_dim_qkv, remaining_dims);
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

template <typename VecT, typename WeightT, typename OutT>
void LayerNormBatched(size_t num_tokens, const VecT* x,
                      const WeightT* HWY_RESTRICT weight,
                      const WeightT* HWY_RESTRICT bias, OutT* out,
                      const size_t size) {
  for (size_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
    LayerNorm(x + token_idx * size, weight, bias, out + token_idx * size, size);
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

// f64 Add, called for f32 inputs promoted to f64. Runs at about half the speed
// of f32 sums.
struct SumKernelDouble {
  // Only `CompressTraits<float>` can `Decompress2` to `double`, so both have
  // to be `float` in order to have `Raw = double`. Note that if either type is
  // smaller than `float`, we may demote the other type from `float` to `BF16`.
  template <typename VT, typename WT>
  using Raw = hwy::If<IsF32<VT>() && IsF32<WT>(), double, BF16>;
  using State = double;

  // Raw = double
  template <class DRaw, class VR = hn::Vec<DRaw>, HWY_IF_F64_D(DRaw)>
  HWY_INLINE void Update4(DRaw /*dd*/, const VR w0, const VR w1, const VR w2,
                          const VR w3, VR, VR, VR, VR, VR& sum0, VR& sum1,
                          VR& sum2, VR& sum3, VR&, VR&, VR&, VR&) const {
    sum0 = hn::Add(sum0, w0);
    sum1 = hn::Add(sum1, w1);
    sum2 = hn::Add(sum2, w2);
    sum3 = hn::Add(sum3, w3);
  }

  // Raw = BF16
  template <class DRaw, class VR = hn::Vec<DRaw>, HWY_IF_BF16_D(DRaw),
            class DS = hn::Repartition<double, DRaw>, class VS = hn::Vec<DS>>
  HWY_INLINE void Update4(DRaw dr, const VR w0, const VR w1, const VR w2,
                          const VR w3, VR, VR, VR, VR, VS& sum0, VS& sum1,
                          VS& sum2, VS& sum3, VS&, VS&, VS&, VS&) const {
    const hn::Repartition<float, DRaw> df;
    using VF = hn::Vec<decltype(df)>;
    // Reduce to two f32 sums so we can promote them to four f64 vectors.
    VF sum02, sum13;
    if constexpr (HWY_NATIVE_DOT_BF16) {
      const VR k1 = hn::Set(dr, hwy::ConvertScalarTo<BF16>(1.0f));
      const VF prod0 = hn::WidenMulPairwiseAdd(df, w0, k1);
      const VF prod1 = hn::WidenMulPairwiseAdd(df, w1, k1);
      // Fuse WidenMulPairwiseAdd plus Add into ReorderWidenMulAccumulate.
      VF unused0 = hn::Zero(df);
      VF unused1 = hn::Zero(df);
      sum02 = hn::ReorderWidenMulAccumulate(df, w2, k1, prod0, unused0);
      sum13 = hn::ReorderWidenMulAccumulate(df, w3, k1, prod1, unused1);
    } else {
      // If not native, the multiplication costs extra, so convert to f32.
      // PromoteEvenTo is cheaper than PromoteUpperTo especially on `SVE`.
      const VF fe0 = hn::PromoteEvenTo(df, w0);
      const VF fe1 = hn::PromoteEvenTo(df, w1);
      const VF fe2 = hn::PromoteEvenTo(df, w2);
      const VF fe3 = hn::PromoteEvenTo(df, w3);
      const VF fo0 = hn::PromoteOddTo(df, w0);
      const VF fo1 = hn::PromoteOddTo(df, w1);
      const VF fo2 = hn::PromoteOddTo(df, w2);
      const VF fo3 = hn::PromoteOddTo(df, w3);
      const VF fe01 = hn::Add(fe0, fe1);
      const VF fe23 = hn::Add(fe2, fe3);
      const VF fo01 = hn::Add(fo0, fo1);
      const VF fo23 = hn::Add(fo2, fo3);
      sum02 = hn::Add(fe01, fe23);
      sum13 = hn::Add(fo01, fo23);
    }

    const DS ds;
    const VS d0 = hn::PromoteLowerTo(ds, sum02);
    const VS d1 = hn::PromoteUpperTo(ds, sum02);
    const VS d2 = hn::PromoteLowerTo(ds, sum13);
    const VS d3 = hn::PromoteUpperTo(ds, sum13);

    sum0 = hn::Add(sum0, d0);
    sum1 = hn::Add(sum1, d1);
    sum2 = hn::Add(sum2, d2);
    sum3 = hn::Add(sum3, d3);
  }

  // Raw = double
  template <class DRaw, class VR = hn::Vec<DRaw>, HWY_IF_F64_D(DRaw)>
  HWY_INLINE void Update1(DRaw /*dd*/, const VR w0, const VR v0, VR& sum0,
                          VR& comp0) const {
    sum0 = hn::Add(sum0, w0);
  }

  // Raw = BF16
  template <class DRaw, class VR = hn::Vec<DRaw>, HWY_IF_BF16_D(DRaw),
            class DS = hn::Repartition<double, DRaw>, class VS = hn::Vec<DS>>
  HWY_INLINE void Update1(DRaw dr, const VR w0, VR, VS& sum0,
                          VS& extra0) const {
    const hn::Repartition<float, DRaw> df;
    using VF = hn::Vec<decltype(df)>;
    VF f0;
    if constexpr (HWY_NATIVE_DOT_BF16) {
      const VR k1 = hn::Set(dr, hwy::ConvertScalarTo<BF16>(1.0f));
      f0 = hn::WidenMulPairwiseAdd(df, w0, k1);
    } else {
      const VF fe0 = hn::PromoteEvenTo(df, w0);
      const VF fo0 = hn::PromoteOddTo(df, w0);
      f0 = hn::Add(fe0, fo0);
    }

    const DS ds;
    const VS d0 = hn::PromoteLowerTo(ds, f0);
    const VS d1 = hn::PromoteUpperTo(ds, f0);

    sum0 = hn::Add(sum0, d0);
    extra0 = hn::Add(extra0, d1);
  }

  template <class DState, class VS = hn::Vec<DState>>
  HWY_INLINE float Reduce(DState dd, VS& sum0, VS& sum1, VS& sum2, VS& sum3,
                          VS& extra0, VS&, VS&, VS&) const {
    // Reduction tree: sum of all accumulators by pairs, then across lanes.
    sum0 = hn::Add(sum0, sum1);
    sum2 = hn::Add(sum2, sum3);
    sum0 = hn::Add(sum0, extra0);  // from Update1
    sum0 = hn::Add(sum0, sum2);
    return static_cast<float>(hn::ReduceSum(dd, sum0));
  }
};

// ORO Cascaded Summation, algorithm 6.11 from Handbook of Floating-Point
// Arithmetic. Note that Algorithm 6.7 (KBN) appears erroneous. We use TwoSums
// instead of FastTwoSums because the magnitude of the initial sum is not
// always greater than the next input, and this does actually change the e2e
// generation results. Note that Kahan summation differs in that it first adds
// comp* to w*, so each operation is serially dependent. By contrast, the sum*
// and comp* here have shorter dependency chains.
//
// This about as accurate as SumKernelDouble but slower, hence we only use this
// if f64 is not supported on this target.
struct SumKernelCascaded {
  template <typename VT, typename WT>
  using Raw = float;
  using State = float;

  template <class DF, class VF = hn::Vec<DF>, HWY_IF_F32_D(DF)>
  HWY_INLINE void Update4(DF df, const VF w0, const VF w1, const VF w2,
                          const VF w3, VF, VF, VF, VF, VF& sum0, VF& sum1,
                          VF& sum2, VF& sum3, VF& comp0, VF& comp1, VF& comp2,
                          VF& comp3) const {
    VF serr0, serr1, serr2, serr3;
    sum0 = TwoSums(df, sum0, w0, serr0);
    sum1 = TwoSums(df, sum1, w1, serr1);
    sum2 = TwoSums(df, sum2, w2, serr2);
    sum3 = TwoSums(df, sum3, w3, serr3);

    comp0 = hn::Add(comp0, serr0);
    comp1 = hn::Add(comp1, serr1);
    comp2 = hn::Add(comp2, serr2);
    comp3 = hn::Add(comp3, serr3);
  }

  template <class DF, class VF = hn::Vec<DF>, HWY_IF_F32_D(DF)>
  HWY_INLINE void Update1(DF df, const VF w0, const VF v0, VF& sum0,
                          VF& comp0) const {
    VF serr0;
    sum0 = TwoSums(df, sum0, w0, serr0);

    comp0 = hn::Add(comp0, serr0);
  }

  template <class DF, class VF = hn::Vec<DF>>
  HWY_INLINE float Reduce(DF df, VF& sum0, VF& sum1, VF& sum2, VF& sum3,
                          VF& comp0, VF& comp1, VF& comp2, VF& comp3) const {
    // Reduction tree: sum of all accumulators by pairs, then across lanes.
    AssimilateCascadedSums(df, sum1, comp1, sum0, comp0);
    AssimilateCascadedSums(df, sum3, comp3, sum2, comp2);
    AssimilateCascadedSums(df, sum2, comp2, sum0, comp0);
    return ReduceCascadedSums(df, sum0, comp0);
  }
};

using SumKernelDefault =
    hwy::If<HWY_HAVE_FLOAT64, SumKernelDouble, SumKernelCascaded>;

template <class D, typename VT>
HWY_INLINE float Sum(D d, const VT* HWY_RESTRICT vec, size_t num) {
  using Raw = hwy::If<HWY_HAVE_FLOAT64, double, float>;
  const hn::Repartition<Raw, D> d_raw;
  return DecompressAndCall(d_raw, MakeSpan(vec, num), SumKernelDefault());
}

// See below for a specialized version for top-1 sampling.
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
  hn::Foreach(d, x, mask_pos, vmin,
              [pmax](const auto d, const V value)
                  HWY_ATTR { *pmax = hn::Max(*pmax, value); });
  vmax = hn::MaxOfLanes(d, vmax);

  // Subtract max (avoid precision loss for large exponents) and exponentiate.
  hn::Transform(d, x, mask_pos, [pmax](const auto d, const V value) HWY_ATTR {
    if constexpr (HWY_TARGET & HWY_ALL_SVE) {
      // Temporary workaround for buggy SVE codegen: avoid inlined Exp().
      return hn::CallExp(d, hn::Sub(value, *pmax));
    } else {
      return hn::Exp(d, hn::Sub(value, *pmax));
    }
  });

  // Normalize to probability distribution. The exact sum seems like it should
  // not make a huge difference. It halves the standard deviation of the sum of
  // the normalized probabilities from 1E-7 to 5E-8, but actually also changes
  // the generated text after a few hundred tokens.
  const float sum_exp = Sum(d, x, mask_pos);
  // Double-precision reciprocal does not appear to affect the results.
  const float mul = 1.0f / sum_exp;
  MulByConst(mul, x, size, mask_pos);
}

static HWY_INLINE HWY_MAYBE_UNUSED void Softmax(float* HWY_RESTRICT x,
                                                const size_t size) {
  Softmax(x, size, size);
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

template <size_t k>
HWY_NOINLINE HWY_MAYBE_UNUSED std::discrete_distribution<int>
create_distribution(std::array<float, k>& top_k, float temperature) {
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

template <size_t k, typename TAcceptToken>
HWY_NOINLINE HWY_MAYBE_UNUSED int SampleTopK(
    const float* HWY_RESTRICT probabilities, size_t vocab_size,
    std::mt19937& gen, float temperature, TAcceptToken& accept_token) {
  static_assert(k != 0, "");
  HWY_ASSERT(k <= vocab_size);
  // TODO: Optimize, potentially using new VQSort PartialSort.
  std::array<float, k> top_k{};  // sorted from highest [0], to lowest [k-1]
  top_k.fill(-std::numeric_limits<float>::infinity());
  std::array<int, k> indices{};
  size_t num_accepted = 0;
  for (size_t i = 0; i < vocab_size; ++i) {
    if (probabilities[i] < top_k[k - 1]) continue;
    bool accepted =
        !accept_token || accept_token(StaticCast<int>(i), probabilities[i]);
    if (!accepted) continue;
    num_accepted++;
    for (size_t j = 0; j < k; ++j) {
      if (probabilities[i] > top_k[j]) {
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
  HWY_ASSERT(k <= num_accepted);
  return indices[create_distribution<k>(top_k, temperature)(gen)];
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
