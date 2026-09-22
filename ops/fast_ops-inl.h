// Copyright 2026 Google LLC
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
#ifndef THIRD_PARTY_GEMMA_CPP_OPS_FAST_OPS_INL_H_
#define THIRD_PARTY_GEMMA_CPP_OPS_FAST_OPS_INL_H_

#include <stddef.h>

#include "ops/ops.h"
#include "util/threading_context.h"
#include "util/zones.h"
#include "hwy/base.h"

#endif  // THIRD_PARTY_GEMMA_CPP_OPS_FAST_OPS_INL_H_

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_OPS_FAST_OPS_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_OPS_FAST_OPS_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_OPS_FAST_OPS_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_OPS_FAST_OPS_TOGGLE
#endif

#include "compression/compress-inl.h"
#include "hwy/contrib/math/fast_math-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// We use the tanh approximation for gelu (also used in training).
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
//         = 0.5 * x * (1 + tanh(x * (sqrt(2/π) + sqrt(2/π) * 0.044715 * x^2)))
//         = 0.5 * x * (1 + tanh(x * (0.79788 + 0.035677 * x^2)))
//         = x * (0.5 + 0.5 * tanh(x * (0.79788 + 0.035677 * x^2))))
//
// This uses hn::FastTanh from
// third_party/highway/hwy/contrib/math/fast_math-inl.h
template <class D, HWY_IF_F32_D(D)>
HWY_INLINE hn::Vec<D> FastGeluCdf(D d, hn::Vec<D> v) {
  const hn::Vec<D> kMul = hn::Set(d, 0.03567740813636141f);
  const hn::Vec<D> kSqrt2OverPi = hn::Set(d, 0.797884560804236f);
  const hn::Vec<D> kHalf = hn::Set(d, 0.5f);

  const hn::Vec<D> v2 = hn::Mul(v, v);
  const hn::Vec<D> v_kMul = hn::Mul(kMul, v);
  const hn::Vec<D> v_kSqrt = hn::Mul(kSqrt2OverPi, v);
  const hn::Vec<D> arg = hn::MulAdd(v_kMul, v2, v_kSqrt);
  return hn::MulAdd(kHalf, hn::FastTanh(d, arg), kHalf);
}

template <class D, HWY_IF_F32_D(D)>
HWY_INLINE hn::Vec<D> FastGelu(D d, hn::Vec<D> v) {
  return hn::Mul(v, FastGeluCdf(d, v));
}

// Fast approximation of sigmoid(x) = 1 / (1 + exp(-x))
// Derived from FastTanh using the identity:
//   sigmoid(x) = 0.5 + 0.5 * tanh(x / 2)
//              = 0.5 + CopySign(0.5 * tanh(|x| / 2), x)
//
// In FastTanh(z) (third_party/highway/hwy/contrib/math/fast_math-inl.h),
// z in [0, 6.65] with u_z = z^2 is approximated by a degree-(7, 6) rational
// function z * P3(u_z) / Q3(u_z):
//   P3(u_z) / Q3(u_z) = (p3*u_z^3 + p2*u_z^2 + p1*u_z + 1) /
//                       (q3*u_z^3 + q2*u_z^2 + q1*u_z + 1)
//
// Substituting z = y / 2 (where y = |x|, clamped to kMax = 2 * 6.65 = 13.30)
// and u = y^2 (so u_z = u / 4), and multiplying by 0.5:
//   0.5 * tanh(y / 2) ~= (y / 4) * P3(u / 4) / Q3(u / 4)
//                      = y * (p3' * u^3 + p2' * u^2 + p1' * u + 0.25) /
//                            (q3' * u^3 + q2' * u^2 + q1' * u + 1)
// where p_k' = p_k / 4^(k+1) and q_k' = q_k / 4^k.
template <class D, HWY_IF_F32_D(D)>
HWY_INLINE hn::Vec<D> FastSigmoid(D d, hn::Vec<D> val) {
  using T = hn::TFromD<D>;

  // Clamp |val| to kMax = 13.30 (= 2 * 6.65) before squaring.
  const auto kMax = hn::Set(d, static_cast<T>(13.30));
  const auto kOne = hn::Set(d, static_cast<T>(1.0));
  const auto kHalf = hn::Set(d, static_cast<T>(0.5));
  const auto kQuarter = hn::Set(d, static_cast<T>(0.25));

  const auto y = hn::Min(hn::Abs(val), kMax);
  const auto u = hn::Mul(y, y);

  const auto p1 = hn::Set(d, static_cast<T>(0.007756593499492869));
  const auto p2 = hn::Set(d, static_cast<T>(3.708395652393883e-05));
  const auto p3 = hn::Set(d, static_cast<T>(1.712842747715196e-08));

  const auto q1 = hn::Set(d, static_cast<T>(0.11435917491679918));
  const auto q2 = hn::Set(d, static_cast<T>(0.0013451487514049556));
  const auto q3 = hn::Set(d, static_cast<T>(2.3876605672811625e-06));

  // Evaluate P3(u) and Q3(u) using Estrin's scheme maximizing ILP:
  const auto u2 = hn::Mul(u, u);

  // p_term0 = p1 * u + 0.25
  const auto p_term0 = hn::MulAdd(p1, u, kQuarter);
  // p_term1 = p3 * u + p2
  const auto p_term1 = hn::MulAdd(p3, u, p2);
  // q_term0 = q1 * u + 1.0
  const auto q_term0 = hn::MulAdd(q1, u, kOne);
  // q_term1 = q3 * u + q2
  const auto q_term1 = hn::MulAdd(q3, u, q2);

  // p3_u = p_term1 * u^2 + p_term0 = p3*u^3 + p2*u^2 + p1*u + 0.25
  const auto p3_u = hn::MulAdd(p_term1, u2, p_term0);
  // q3_u = q_term1 * u^2 + q_term0 = q3*u^3 + q2*u^2 + q1*u + 1.0
  const auto q3_u = hn::MulAdd(q_term1, u2, q_term0);
  const auto num = hn::Mul(y, p3_u);

  // Clamp the approx value to 0.5 for safety in case of rounding differences
  // across architectures.
  const auto approx = hn::Min(hn::Div(num, q3_u), kHalf);

  // sigmoid(x) = 0.5 + sign(x) * (0.5 * tanh(|x|/2))
  return hn::Add(kHalf, hn::CopySign(approx, val));
}

// Activation already has a profiler zone.
template <typename T>
static HWY_NOINLINE HWY_MAYBE_UNUSED void FastGelu(T* HWY_RESTRICT x,
                                                   size_t size) {
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  using VF = hn::Vec<DF>;
  DecompressAndCompressInplace(
      DF(), x, size, [](DF d, VF v) HWY_ATTR -> VF { return FastGelu(d, v); });
}

template <typename T>
static HWY_NOINLINE HWY_MAYBE_UNUSED void FastSigmoid(T* HWY_RESTRICT x,
                                                      size_t size) {
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  using VF = hn::Vec<DF>;
  DecompressAndCompressInplace(DF(), x, size, [](DF d, VF v) HWY_ATTR -> VF {
    return FastSigmoid(d, v);
  });
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
