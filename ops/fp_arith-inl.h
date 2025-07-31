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

#include <stddef.h>

// Building blocks for floating-point arithmetic.

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_FP_ARITH_TOGGLE) == defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_FP_ARITH_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_FP_ARITH_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_FP_ARITH_TOGGLE
#endif

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

//------------------------------------------------------------------------------
// Exact multiplication

namespace detail {

// Returns non-overlapping `x` and `y` such that `x + y` = `f` and |x| >= |y|.
// Notation from Algorithm 3.1 in Handbook of Floating-Point Arithmetic. 4 ops.
template <class DF, HWY_IF_FLOAT3264_D(DF), class VF = hn::Vec<DF>>
static HWY_INLINE void VeltkampSplit(DF df, VF a, VF& x, VF& y) {
  using TF = hn::TFromD<DF>;
  constexpr int t = hwy::MantissaBits<TF>() + 1;  // = -log2(epsilon)
  constexpr int s = hwy::DivCeil(t, 2);
  const VF factor = hn::Set(df, hwy::ConvertScalarTo<TF>((1ULL << s) + 1));
  const VF c = hn::Mul(factor, a);
  x = hn::Sub(c, hn::Sub(c, a));
  y = hn::Sub(a, x);
}

}  // namespace detail

// Returns `prod` and `err` such that `prod + err` is exactly equal to `a * b`,
// despite floating-point rounding, assuming that `err` is not subnormal, i.e.,
// the sum of exponents >= min exponent + mantissa bits. 2..17 ops. Useful for
// compensated dot products and polynomial evaluation.
template <class DF, HWY_IF_FLOAT3264_D(DF), class VF = hn::Vec<DF>>
static HWY_INLINE VF TwoProducts(DF df, VF a, VF b, VF& err) {
  const VF prod = hn::Mul(a, b);
  if constexpr (HWY_NATIVE_FMA) {
    err = hn::MulSub(a, b, prod);
  } else {
    // Non-FMA fallback: we assume these calculations do not overflow.
    VF a1, a2, b1, b2;
    detail::VeltkampSplit(df, a, a1, a2);
    detail::VeltkampSplit(df, b, b1, b2);
    const VF m = hn::Sub(prod, hn::Mul(a1, b1));
    const VF n = hn::Sub(m, hn::Mul(a2, b1));
    const VF o = hn::Sub(n, hn::Mul(a1, b2));
    err = hn::Sub(hn::Mul(a2, b2), o);
  }
  return prod;
}

//------------------------------------------------------------------------------
// Exact addition

// Returns `sum` and `err` such that `sum + err` is exactly equal to `a + b`,
// despite floating-point rounding. `sum` is already the best estimate for the
// addition, so do not directly add `err` to it. `UpdateCascadedSums` instead
// accumulates multiple `err`, which are then later added to the total `sum`.
//
// Knuth98/Moller65. Unlike FastTwoSums, this does not require any relative
// ordering of the exponents of a and b. 6 ops.
template <class DF, HWY_IF_FLOAT3264_D(DF), class VF = hn::Vec<DF>>
static HWY_INLINE VF TwoSums(DF /*df*/, VF a, VF b, VF& err) {
  const VF sum = hn::Add(a, b);
  const VF a2 = hn::Sub(sum, b);
  const VF b2 = hn::Sub(sum, a2);
  const VF err_a = hn::Sub(a, a2);
  const VF err_b = hn::Sub(b, b2);
  err = hn::Add(err_a, err_b);
  return sum;
}

// As above, but only exact if the exponent of `a` >= that of `b`, which is the
// case if |a| >= |b|. Dekker71, also used in Kahan65 compensated sum. 3 ops.
template <class DF, HWY_IF_FLOAT3264_D(DF), class VF = hn::Vec<DF>>
static HWY_INLINE VF FastTwoSums(DF /*df*/, VF a, VF b, VF& err) {
  const VF sum = hn::Add(a, b);
  const VF b2 = hn::Sub(sum, a);
  err = hn::Sub(b, b2);
  return sum;
}

//------------------------------------------------------------------------------
// Cascaded summation (twice working precision)

// Accumulates numbers with about twice the precision of T using 7 * n FLOPS.
// Rump/Ogita/Oishi08, Algorithm 6.11 in Handbook of Floating-Point Arithmetic.
//
// Because vectors generally cannot be wrapped in a class, we use functions.
// `sum` and `sum_err` must be initially zero. Each lane is an independent sum.
// To reduce them into a single result, use `ReduceCascadedSum`.
template <class DF, HWY_IF_FLOAT3264_D(DF), class VF = hn::Vec<DF>>
void UpdateCascadedSums(DF df, VF v, VF& sum, VF& sum_err) {
  VF err;
  sum = TwoSums(df, sum, v, err);
  sum_err = hn::Add(sum_err, err);
}

// Combines two cascaded sum vectors, typically from unrolling/parallelization.
template <class DF, HWY_IF_FLOAT3264_D(DF), class VF = hn::Vec<DF>>
void AssimilateCascadedSums(DF df, const VF& other_sum, const VF& other_sum_err,
                            VF& sum, VF& sum_err) {
  sum_err = hn::Add(sum_err, other_sum_err);
  UpdateCascadedSums(df, other_sum, sum, sum_err);
}

// Reduces cascaded sums, to a single value. Slow, call outside of loops.
template <class DF, HWY_IF_FLOAT3264_D(DF), class VF = hn::Vec<DF>>
hn::TFromD<DF> ReduceCascadedSums(DF df, const VF sum, VF sum_err) {
  const size_t N = hn::Lanes(df);
  using TF = hn::TFromD<DF>;
  // For non-scalable wide vectors, reduce loop iterations below by recursing
  // once or twice for halves of 256-bit or 512-bit vectors.
  if constexpr (HWY_HAVE_CONSTEXPR_LANES) {
    if constexpr (hn::Lanes(df) > 16 / sizeof(TF)) {
      const hn::Half<DF> dfh;
      using VFH = hn::Vec<decltype(dfh)>;

      VFH sum0 = hn::LowerHalf(dfh, sum);
      VFH sum_err0 = hn::LowerHalf(dfh, sum_err);
      const VFH sum1 = hn::UpperHalf(dfh, sum);
      const VFH sum_err1 = hn::UpperHalf(dfh, sum_err);
      AssimilateCascadedSums(dfh, sum1, sum_err1, sum0, sum_err0);
      return ReduceCascadedSums(dfh, sum0, sum_err0);
    }
  }

  TF total = TF{0.0};
  TF total_err = TF{0.0};
  for (size_t i = 0; i < N; ++i) {
    TF err;
    total_err += hn::ExtractLane(sum_err, i);
    total = TwoSum(total, hn::ExtractLane(sum, i), err);
    total_err += err;
  }
  return total + total_err;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
