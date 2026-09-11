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
// Derived from FastTanh by substituting x/2.
template <class D, HWY_IF_F32_D(D)>
HWY_INLINE hn::Vec<D> FastSigmoid(D d, hn::Vec<D> val) {
  using T = hn::TFromD<D>;

  // Abs(val) and preserve sign for later for symmetric rational approximation
  auto y = hn::Abs(val);

  constexpr size_t kLanes = HWY_MAX_LANES_D(D);
  hn::Vec<D> b, c, d_coef;

  if constexpr ((kLanes >= 4 && !HWY_HAVE_SCALABLE) ||
                (HWY_HAVE_SCALABLE && sizeof(T) == 4 &&
                 hn::detail::IsFull(d))) {
    // Coefficients for P(y/2) ~ index using CF algo
    const auto k0 = hn::Set(d, static_cast<T>(-0.1145426548151546));
    const auto k1 = hn::Set(d, static_cast<T>(3.4556654973457404));
    const auto k2 = hn::Set(d, static_cast<T>(-0.6278480784875462));
    const auto k3 = hn::Set(d, static_cast<T>(0.04331384030062471));

    // Index calculation: idx = P(y/2)
    // Estrin's scheme
    // k0 + y * k1 + y^2 * (k2 + y * k3)
    const auto y2 = hn::Mul(y, y);
    const auto p01 = hn::MulAdd(k1, y, k0);
    const auto p23 = hn::MulAdd(k3, y, k2);
    auto idx_poly = hn::MulAdd(y2, p23, p01);

    // Convert to integer index
    using DI = hn::RebindToSigned<D>;
    auto idx_i = hn::ConvertTo(DI(), idx_poly);

    // Clamp index to 7
    idx_i = hn::Min(idx_i, hn::Set(DI(), 7));

    HWY_ALIGN static constexpr T arr_b[8] = {
        static_cast<T>(-0.0006967055197996615),
        static_cast<T>(-0.010315055591476996),
        static_cast<T>(-0.05367999021047822),
        static_cast<T>(-0.16943664192343108),
        static_cast<T>(-0.42437007298661206),
        static_cast<T>(-0.9556349519550872),
        static_cast<T>(-2.0824831112860647),
        static_cast<T>(-4.688832585616333)};
    HWY_ALIGN static constexpr T arr_c[8] = {
        static_cast<T>(0.220551955463595),  static_cast<T>(0.5069204289218385),
        static_cast<T>(0.8423809865207907), static_cast<T>(1.1775629610724903),
        static_cast<T>(1.4909222917402543), static_cast<T>(1.757582383623199),
        static_cast<T>(1.9363640518503402), static_cast<T>(1.9985234759675707)};
    HWY_ALIGN static constexpr T arr_d[8] = {
        static_cast<T>(3.9548607753775276), static_cast<T>(3.7450486139396544),
        static_cast<T>(3.253860706225495),  static_cast<T>(2.4240814251983283),
        static_cast<T>(1.1565092321921886), static_cast<T>(-0.7540678688218365),
        static_cast<T>(-3.767209600467866), static_cast<T>(-9.357047249878605)};

    // Since Lookup8 is available for HWY_MIN_BYTES / sizeof(T) >= 4, this
    // condition covers all cases we encounter inside the top level if block
    // inside FastSigmoid
    b = hn::Lookup8(d, arr_b, idx_i);
    c = hn::Lookup8(d, arr_c, idx_i);
    d_coef = hn::Lookup8(d, arr_d, idx_i);
  } else {
    // --- FALLBACK PATH: Blend Chain ---
    // Thresholds for intervals
    const auto t0 = hn::Set(d, static_cast<T>(0.3434497447432422));
    const auto t1 = hn::Set(d, static_cast<T>(0.6955976007186494));
    const auto t2 = hn::Set(d, static_cast<T>(1.1068914127668934));
    const auto t3 = hn::Set(d, static_cast<T>(1.608648163822941));
    const auto t4 = hn::Set(d, static_cast<T>(2.269039121646492));
    const auto t5 = hn::Set(d, static_cast<T>(3.288402547357102));
    const auto t6 = hn::Set(d, static_cast<T>(5.271780018997146));

    if constexpr (HWY_REGISTERS >= 32) {
      // Split into two parallel chains to reduce dependency latency.

      // -- Chain 1: Indices 0 to 3 (Evaluated starting from t3 down to t0)
      auto b_low = hn::Set(d, static_cast<T>(-0.16943664192343108));  // idx 3
      auto c_low = hn::Set(d, static_cast<T>(1.1775629610724903));
      auto d_low = hn::Set(d, static_cast<T>(2.4240814251983283));

      auto mask = hn::Lt(y, t2);
      b_low = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(-0.05367999021047822)), b_low);
      c_low = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(0.8423809865207907)), c_low);
      d_low = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(3.253860706225495)), d_low);

      mask = hn::Lt(y, t1);
      b_low = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(-0.010315055591476996)), b_low);
      c_low = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(0.5069204289218385)), c_low);
      d_low = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(3.7450486139396544)), d_low);

      mask = hn::Lt(y, t0);
      b_low = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(-0.0006967055197996615)), b_low);
      c_low = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(0.220551955463595)), c_low);
      d_low = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(3.9548607753775276)), d_low);

      // -- Chain 2: Indices 4 to 7 (Evaluated starting from t6 down to t4)
      auto b_high = hn::Set(d, static_cast<T>(-4.688832585616333));  // idx 7
      auto c_high = hn::Set(d, static_cast<T>(1.9985234759675707));
      auto d_high = hn::Set(d, static_cast<T>(-9.357047249878605));

      mask = hn::Lt(y, t6);
      b_high = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(-2.0824831112860647)), b_high);
      c_high = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(1.9363640518503402)), c_high);
      d_high = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(-3.767209600467866)), d_high);

      mask = hn::Lt(y, t5);
      b_high = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(-0.9556349519550872)), b_high);
      c_high = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(1.757582383623199)), c_high);
      d_high = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(-0.7540678688218365)), d_high);

      mask = hn::Lt(y, t4);
      b_high = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(-0.42437007298661206)), b_high);
      c_high = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(1.4909222917402543)), c_high);
      d_high = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(1.1565092321921886)), d_high);

      // -- Merge the two chains
      auto merge_mask = hn::Lt(y, t3);
      b = hn::IfThenElse(merge_mask, b_low, b_high);
      c = hn::IfThenElse(merge_mask, c_low, c_high);
      d_coef = hn::IfThenElse(merge_mask, d_low, d_high);
    } else {
      // Start with highest index (7)
      b = hn::Set(d, static_cast<T>(-4.688832585616333));
      c = hn::Set(d, static_cast<T>(1.9985234759675707));
      d_coef = hn::Set(d, static_cast<T>(-9.357047249878605));

      // If y < t6 (idx 6)
      auto mask = hn::Lt(y, t6);
      b = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(-2.0824831112860647)),
                         b);
      c = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(1.9363640518503402)),
                         c);
      d_coef = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(-3.767209600467866)), d_coef);

      // If y < t5 (idx 5)
      mask = hn::Lt(y, t5);
      b = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(-0.9556349519550872)),
                         b);
      c = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(1.757582383623199)),
                         c);
      d_coef = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(-0.7540678688218365)), d_coef);

      // If y < t4 (idx 4)
      mask = hn::Lt(y, t4);
      b = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(-0.42437007298661206)),
                         b);
      c = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(1.4909222917402543)),
                         c);
      d_coef = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(1.1565092321921886)), d_coef);

      // If y < t3 (idx 3)
      mask = hn::Lt(y, t3);
      b = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(-0.16943664192343108)),
                         b);
      c = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(1.1775629610724903)),
                         c);
      d_coef = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(2.4240814251983283)), d_coef);

      // If y < t2 (idx 2)
      mask = hn::Lt(y, t2);
      b = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(-0.05367999021047822)),
                         b);
      c = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(0.8423809865207907)),
                         c);
      d_coef = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(3.253860706225495)), d_coef);

      // If y < t1 (idx 1)
      mask = hn::Lt(y, t1);
      b = hn::IfThenElse(mask,
                         hn::Set(d, static_cast<T>(-0.010315055591476996)), b);
      c = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(0.5069204289218385)),
                         c);
      d_coef = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(3.7450486139396544)), d_coef);

      // If y < t0 (idx 0)
      mask = hn::Lt(y, t0);
      b = hn::IfThenElse(mask,
                         hn::Set(d, static_cast<T>(-0.0006967055197996615)), b);
      c = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(0.220551955463595)),
                         c);
      d_coef = hn::IfThenElse(
          mask, hn::Set(d, static_cast<T>(3.9548607753775276)), d_coef);
    }
  }

  // Math: 0.5 * tanh(y/2) = (y + b)/(cy + d_coef)
  auto num = hn::Add(y, b);
  auto den = hn::MulAdd(c, y, d_coef);

  auto approx = hn::Div(num, den);

  const auto half = hn::Set(d, static_cast<T>(0.5));
  // Clamp the approx value to 0.5
  approx = hn::Min(approx, half);
  // sigmoid(x) = 0.5 + sign(x) * (0.5 * tanh(|x|/2))
  return hn::Add(half, hn::CopySign(approx, val));
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
