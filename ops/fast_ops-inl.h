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

// 16-point in-register Fast Walsh-Hadamard Transform on a 16-lane F32 vector.
template <class DF, class VF = hn::Vec<DF>, HWY_IF_F32_D(DF)>
HWY_INLINE VF FWHT16(DF df, VF v) {
  // Stage 1: h = 1
  const auto e1 = hn::DupEven(v);
  const auto o1 = hn::DupOdd(v);
  v = hn::OddEven(hn::Sub(e1, o1), hn::Add(e1, o1));

  // Stage 2: h = 2
  const hn::Repartition<uint64_t, DF> du64;
  const auto vu64 = hn::BitCast(du64, v);
  const auto e2 = hn::BitCast(df, hn::DupEven(vu64));
  const auto o2 = hn::BitCast(df, hn::DupOdd(vu64));
  v = hn::BitCast(
      df, hn::OddEven(hn::BitCast(du64, hn::Sub(e2, o2)),
                      hn::BitCast(du64, hn::Add(e2, o2))));

  // Stage 3: h = 4
  const hn::Half<DF> dfh;
  const hn::Half<decltype(dfh)> dfq;
  auto lo_h = hn::LowerHalf(dfh, v);
  auto hi_h = hn::UpperHalf(dfh, v);
  auto q0 = hn::LowerHalf(dfq, lo_h);
  auto q1 = hn::UpperHalf(dfq, lo_h);
  auto q2 = hn::LowerHalf(dfq, hi_h);
  auto q3 = hn::UpperHalf(dfq, hi_h);
  lo_h = hn::Combine(dfh, hn::Sub(q0, q1), hn::Add(q0, q1));
  hi_h = hn::Combine(dfh, hn::Sub(q2, q3), hn::Add(q2, q3));

  // Stage 4: h = 8
  return hn::Combine(df, hn::Sub(lo_h, hi_h), hn::Add(lo_h, hi_h));
}

template <class D, class T = hn::TFromD<D>>
HWY_INLINE void FastWalshHadamard128(D d, T* HWY_RESTRICT data, size_t length) {
  HWY_DASSERT(length % 128 == 0);
  constexpr float kNorm = 0.08838834764831845f;  // 1.0f / sqrt(128.0f)
  const hn::ScalableTag<float> df;
  const hn::Repartition<BF16, decltype(df)> dbf;
  using VF = hn::Vec<decltype(df)>;
  const auto vnorm = hn::Set(df, kNorm);

  if constexpr (hn::MaxLanes(df) == 16 && !HWY_HAVE_SCALABLE) {
    // Optimal in-register path for AVX-512 (H128 = H8 x H16):
    for (size_t block = 0; block < length; block += 128) {
      VF v0, v1, v2, v3, v4, v5, v6, v7;
      if constexpr (IsBF16<T>()) {
        const auto b0 = hn::Load(dbf, data + block + 0);
        const auto b1 = hn::Load(dbf, data + block + 32);
        const auto b2 = hn::Load(dbf, data + block + 64);
        const auto b3 = hn::Load(dbf, data + block + 96);
        v0 = hn::PromoteLowerTo(df, b0);
        v1 = hn::PromoteUpperTo(df, b0);
        v2 = hn::PromoteLowerTo(df, b1);
        v3 = hn::PromoteUpperTo(df, b1);
        v4 = hn::PromoteLowerTo(df, b2);
        v5 = hn::PromoteUpperTo(df, b2);
        v6 = hn::PromoteLowerTo(df, b3);
        v7 = hn::PromoteUpperTo(df, b3);
      } else {
        v0 = hn::Load(df, data + block + 0);
        v1 = hn::Load(df, data + block + 16);
        v2 = hn::Load(df, data + block + 32);
        v3 = hn::Load(df, data + block + 48);
        v4 = hn::Load(df, data + block + 64);
        v5 = hn::Load(df, data + block + 80);
        v6 = hn::Load(df, data + block + 96);
        v7 = hn::Load(df, data + block + 112);
      }

      // Inter-vector stages (H8):
      // h = 64
      auto t0 = v0; v0 = hn::Add(t0, v4); v4 = hn::Sub(t0, v4);
      auto t1 = v1; v1 = hn::Add(t1, v5); v5 = hn::Sub(t1, v5);
      auto t2 = v2; v2 = hn::Add(t2, v6); v6 = hn::Sub(t2, v6);
      auto t3 = v3; v3 = hn::Add(t3, v7); v7 = hn::Sub(t3, v7);

      // h = 32
      t0 = v0; v0 = hn::Add(t0, v2); v2 = hn::Sub(t0, v2);
      t1 = v1; v1 = hn::Add(t1, v3); v3 = hn::Sub(t1, v3);
      auto t4 = v4; v4 = hn::Add(t4, v6); v6 = hn::Sub(t4, v6);
      auto t5 = v5; v5 = hn::Add(t5, v7); v7 = hn::Sub(t5, v7);

      // h = 16
      t0 = v0; v0 = hn::Add(t0, v1); v1 = hn::Sub(t0, v1);
      auto t2_ = v2; v2 = hn::Add(t2_, v3); v3 = hn::Sub(t2_, v3);
      t4 = v4; v4 = hn::Add(t4, v5); v5 = hn::Sub(t4, v5);
      auto t6 = v6; v6 = hn::Add(t6, v7); v7 = hn::Sub(t6, v7);

      // Intra-vector stages (H16) and normalization:
      v0 = hn::Mul(FWHT16(df, v0), vnorm);
      v1 = hn::Mul(FWHT16(df, v1), vnorm);
      v2 = hn::Mul(FWHT16(df, v2), vnorm);
      v3 = hn::Mul(FWHT16(df, v3), vnorm);
      v4 = hn::Mul(FWHT16(df, v4), vnorm);
      v5 = hn::Mul(FWHT16(df, v5), vnorm);
      v6 = hn::Mul(FWHT16(df, v6), vnorm);
      v7 = hn::Mul(FWHT16(df, v7), vnorm);

      if constexpr (IsBF16<T>()) {
        hn::Store(hn::OrderedDemote2To(dbf, v0, v1), dbf, data + block + 0);
        hn::Store(hn::OrderedDemote2To(dbf, v2, v3), dbf, data + block + 32);
        hn::Store(hn::OrderedDemote2To(dbf, v4, v5), dbf, data + block + 64);
        hn::Store(hn::OrderedDemote2To(dbf, v6, v7), dbf, data + block + 96);
      } else {
        hn::Store(v0, df, data + block + 0);
        hn::Store(v1, df, data + block + 16);
        hn::Store(v2, df, data + block + 32);
        hn::Store(v3, df, data + block + 48);
        hn::Store(v4, df, data + block + 64);
        hn::Store(v5, df, data + block + 80);
        hn::Store(v6, df, data + block + 96);
        hn::Store(v7, df, data + block + 112);
      }
    }
  } else {
    // Portable SIMD fallback for architectures with < 16 F32 lanes:
    const size_t lanes = hn::Lanes(df);
    HWY_ALIGN float buf[128];
    for (size_t block = 0; block < length; block += 128) {
      if constexpr (IsBF16<T>()) {
        for (size_t i = 0; i < 128; i += 2 * lanes) {
          const auto b = hn::Load(dbf, data + block + i);
          hn::Store(hn::PromoteLowerTo(df, b), df, buf + i);
          hn::Store(hn::PromoteUpperTo(df, b), df, buf + i + lanes);
        }
      } else {
        for (size_t i = 0; i < 128; i += lanes) {
          hn::Store(hn::Load(df, data + block + i), df, buf + i);
        }
      }

      for (size_t step = 1; step < 128; step <<= 1) {
        const size_t jump = step << 1;
        if (step >= lanes) {
          for (size_t i = 0; i < 128; i += jump) {
            for (size_t j = 0; j < step; j += lanes) {
              const auto u = hn::Load(df, buf + i + j);
              const auto v = hn::Load(df, buf + i + j + step);
              hn::Store(hn::Add(u, v), df, buf + i + j);
              hn::Store(hn::Sub(u, v), df, buf + i + j + step);
            }
          }
        } else {
          for (size_t i = 0; i < 128; i += jump) {
            for (size_t j = 0; j < step; ++j) {
              const float u = buf[i + j];
              const float v = buf[i + j + step];
              buf[i + j] = u + v;
              buf[i + j + step] = u - v;
            }
          }
        }
      }

      for (size_t i = 0; i < 128; i += lanes) {
        hn::Store(hn::Mul(hn::Load(df, buf + i), vnorm), df, buf + i);
      }

      if constexpr (IsBF16<T>()) {
        for (size_t i = 0; i < 128; i += 2 * lanes) {
          const auto f0 = hn::Load(df, buf + i);
          const auto f1 = hn::Load(df, buf + i + lanes);
          hn::Store(hn::OrderedDemote2To(dbf, f0, f1), dbf, data + block + i);
        }
      } else {
        for (size_t i = 0; i < 128; i += lanes) {
          hn::Store(hn::Load(df, buf + i), df, data + block + i);
        }
      }
    }
  }
}

template <typename T>
static HWY_NOINLINE HWY_MAYBE_UNUSED void FastWalshHadamard128(T* HWY_RESTRICT x,
                                                              size_t size) {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<T> d;
  FastWalshHadamard128(d, x, size);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();


#endif  // NOLINT
