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

#include <algorithm>  // std::sort
#include <array>
#include <cstdlib>  // std::abs

#include "compression/compress.h"
#include "compression/distortion.h"  // TwoSum
#include "hwy/base.h"

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_DOT_TOGGLE) == defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_DOT_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_DOT_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_DOT_TOGGLE
#endif

#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"
#include "hwy/contrib/math/math-inl.h"
#include "hwy/profiler.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Returns dot product of `x` and `w`, both length `num`. Uses Decompress2 to
// convert WeightT and VecT to float, then FMA.
// TODO: improve precision?
// TODO: use bf16 products?
template <class DF, typename WeightT, typename VecT>
HWY_INLINE float SimpleDot(DF df, const WeightT* HWY_RESTRICT w, size_t w_ofs,
                           const VecT* HWY_RESTRICT x, size_t num) {
  PROFILER_FUNC;
  const size_t N = hn::Lanes(df);
  HWY_DASSERT(hn::IsAligned(df, x));
  using VF = hn::Vec<DF>;
  using TraitsW = CompressTraits<WeightT>;
  using TraitsV = CompressTraits<VecT>;

  VF sum0 = hn::Zero(df);
  VF sum1 = hn::Zero(df);
  VF sum2 = hn::Zero(df);
  VF sum3 = hn::Zero(df);

  VF w0, w1, w2, w3, v0, v1, v2, v3;  // decompressed inputs

  size_t i = 0;
  if (num >= 4 * N) {
    for (; i <= num - 4 * N; i += 4 * N) {
      TraitsW::Decompress2(df, w, w_ofs + i, w0, w1);
      TraitsW::Decompress2(df, w, w_ofs + i + 2 * N, w2, w3);
      TraitsV::Decompress2(df, x, i, v0, v1);
      TraitsV::Decompress2(df, x, i + 2 * N, v2, v3);

      sum0 = hn::MulAdd(w0, v0, sum0);
      sum1 = hn::MulAdd(w1, v1, sum1);
      sum2 = hn::MulAdd(w2, v2, sum2);
      sum3 = hn::MulAdd(w3, v3, sum3);
    }
  }

  const size_t remaining = num - i;
  if (HWY_UNLIKELY(remaining != 0)) {
    HWY_ALIGN float padded_w[4 * hn::MaxLanes(df)] = {};
    HWY_ALIGN float padded_x[4 * hn::MaxLanes(df)] = {};
    // The actual capacity of w[] is unknown, so pass a lower bound.
    const size_t w_capacity = w_ofs + num;
    TraitsW::Decompress(df, w_capacity, w, w_ofs + i, padded_w, remaining);
    TraitsV::Decompress(df, num, x, i, padded_x, remaining);
    const size_t padding = 4 * N - remaining;
    hwy::ZeroBytes(padded_w + remaining, padding * sizeof(padded_w[0]));
    hwy::ZeroBytes(padded_x + remaining, padding * sizeof(padded_x[0]));
    for (; i < num; i += N) {
      const VF w0 = hn::Load(df, padded_w + i);
      const VF v0 = hn::Load(df, padded_x + i);
      sum0 = hn::MulAdd(w0, v0, sum0);
    }
  }

  // Reduction tree: sum of all accumulators by pairs, then across lanes.
  sum0 = hn::Add(sum0, sum1);
  sum2 = hn::Add(sum2, sum3);
  sum0 = hn::Add(sum0, sum2);
  return hn::ReduceSum(df, sum0);
}

// Adapter for use by matvec-inl.h. TODO: remove when that is no longer used.
template <bool kVecEO, class DF, size_t kCapacity, typename VecT>
HWY_INLINE float Dot(DF df, const std::array<float, kCapacity>& w, size_t ofs,
                     const VecT* vec_aligned, size_t num) {
  PROFILER_ZONE("Dot array");
  HWY_DASSERT(ofs + num <= kCapacity);
  HWY_DASSERT(hn::IsAligned(df, vec_aligned));
  return SimpleDot(df, w.data(), ofs, vec_aligned, num);
}

// Adapter for use by matvec-inl.h. TODO: remove when that is no longer used.
template <bool kVecEO, class DF, typename MatT, size_t kCapacity, typename VecT>
HWY_INLINE float Dot(DF df, const CompressedArray<MatT, kCapacity>& compressed,
                     size_t compressed_ofs, const VecT* vec_aligned,
                     size_t num) {
  PROFILER_ZONE("Dot CompressedArray");
  HWY_DASSERT(compressed_ofs + num <= compressed.size());
  HWY_DASSERT(hn::IsAligned(df, vec_aligned));
  using Traits = CompressTraits<MatT>;
  float dot_result;
  if constexpr (kVecEO) {
    dot_result =
        Traits::DotEO(df, compressed.data(), compressed_ofs, vec_aligned, num);
  } else {
    dot_result =
        SimpleDot(df, compressed.data(), compressed_ofs, vec_aligned, num);
  }
  return compressed.scale() * dot_result;
}

// Returns result accurate to 1.5 ulp, assuming `num` < 2^(52-23), no overflow,
// and round to nearest. See "Accurate and efficient floating point summation".
HWY_INLINE float ExactDot(const float* HWY_RESTRICT a,
                          const float* HWY_RESTRICT b, size_t num,
                          double* HWY_RESTRICT buf) {
  PROFILER_FUNC;
  for (size_t i = 0; i < num; ++i) {
    buf[i] = static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }
  // Sort by decreasing magnitude (not supported by VQSort).
  std::sort(buf, buf + num,
            [](double a, double b) { return std::abs(a) > std::abs(b); });
  double sum = 0.0;
  for (size_t i = 0; i < num; ++i) {
    sum += buf[i];
  }
  return static_cast<float>(sum);
}

//------------------------------------------------------------------------------
// Cascaded summation (twice working precision)

// Returns `sum` and `err` such that `sum + err` is exactly equal to `a + b`,
// despite floating-point rounding. `sum` is already the best estimate for the
// addition, so do not actually add `err` to it. `UpdateCascadedSums` instead
// accumulates multiple `err`, which are then later added to `sum`.
//
// Knuth98/Moller65. Unlike Fast2Sum [Dekker71], this does not require any
// relative ordering of the exponents of a and b.
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

// Adds vectors with about twice the precision of VF using 7 FLOPS.
// Rump/Ogita/Oishi08, Algorithm 6.11 in Handbook of Floating-Point Arithmetic.
// `sum` and `sum_err` must be initially zero.
//
// Each lane is an independent cascaded sum. To obtain a single result, use
// `ReduceCascadedSum`. Vectors generally cannot be wrapped in a class, hence we
// use free functions.
template <class DF, HWY_IF_FLOAT3264_D(DF), class VF = hn::Vec<DF>>
void UpdateCascadedSums(DF df, VF v, VF& sum, VF& sum_err) {
  VF err;
  sum = TwoSums(df, sum, v, err);
  sum_err += err;
}

// Combines two cascaded sum vectors, typically from unrolling/parallelization.
template <class DF, HWY_IF_FLOAT3264_D(DF), class VF = hn::Vec<DF>>
void AssimilateCascadedSums(DF df, const VF& other_sum, const VF& other_sum_err,
                            VF& sum, VF& sum_err) {
  UpdateCascadedSums(df, other_sum, sum, sum_err);
  sum_err += other_sum_err;
}

// Reduces cascaded sums, to a single value. Slow, call outside of loops.
template <class DF, HWY_IF_FLOAT3264_D(DF), class VF = hn::Vec<DF>>
hn::TFromD<DF> ReduceCascadedSums(DF df, const VF sum, VF sum_err) {
  const size_t N = hn::Lanes(df);
  using TF = hn::TFromD<DF>;
  TF total = TF{0.0};
  TF total_err = TF{0.0};
  for (size_t i = 0; i < N; ++i) {
    TF err;
    total = TwoSum(total, hn::ExtractLane(sum, i), err);
    total_err += hn::ExtractLane(sum_err, i);
    total_err += err;
  }
  return total + total_err;
}

//------------------------------------------------------------------------------

// Returns 2 * sum(|f|) / |sum(f)|. This is large when there are many
// similar-magnitude and opposite-sign elements in `f`. See
// https://en.wikipedia.org/wiki/Condition_number.
template <class DF, HWY_IF_FLOAT3264_D(DF), class VF = hn::Vec<DF>>
static inline double ConditionNumber(DF df, const float* HWY_RESTRICT f,
                                     size_t num) {
  PROFILER_FUNC;
  const size_t N = hn::Lanes(df);

  VF sum = hn::Zero(df);
  VF sum_err = hn::Zero(df);
  VF sum_abs = hn::Zero(df);
  VF sum_err_abs = hn::Zero(df);

  size_t i = 0;
  if (num >= N) {
    for (; i <= num - N; i += N) {
      const VF v = hn::Load(df, f + i);
      UpdateCascadedSums(v, sum, sum_err);
      UpdateCascadedSums(hn::Abs(v), sum_abs, sum_err_abs);
    }
  }
  const size_t remaining = num - i;
  if (remaining != 0) {
    const VF v = hn::LoadN(df, f + i, remaining);
    UpdateCascadedSums(v, sum, sum_err);
    UpdateCascadedSums(hn::Abs(v), sum_abs, sum_err_abs);
  }

  const float div = std::abs(ReduceCascadedSums(df, sum, sum_err));
  if (div == 0.0f) return hwy::HighestValue<float>();
  const double cond = 2.0 * ReduceCascadedSums(df, sum_abs, sum_err_abs) /
                      static_cast<double>(div);
  HWY_ASSERT(cond >= 0.0);
  return cond;
}

// Same, but for dot product of two arrays.
// TODO: move into dot_test.
template <class DF, HWY_IF_FLOAT3264_D(DF), class VF = hn::Vec<DF>>
static inline double ConditionNumber(DF df, const float* HWY_RESTRICT a,
                                     const float* HWY_RESTRICT b, size_t num) {
  PROFILER_FUNC;
  const size_t N = hn::Lanes(df);

  VF sum = hn::Zero(df);
  VF sum_err = hn::Zero(df);
  VF sum_abs = hn::Zero(df);
  VF sum_err_abs = hn::Zero(df);

  size_t i = 0;
  if (num >= N) {
    for (; i <= num - N; i += N) {
      const VF va = hn::Load(df, a + i);
      const VF vb = hn::Load(df, b + i);
      const VF mul = hn::Mul(va, vb);
      UpdateCascadedSums(df, mul, sum, sum_err);
      UpdateCascadedSums(df, hn::Abs(mul), sum_abs, sum_err_abs);
    }
  }
  const size_t remaining = num - i;
  if (remaining != 0) {
    const VF va = hn::LoadN(df, a + i, remaining);
    const VF vb = hn::LoadN(df, b + i, remaining);
    const VF mul = hn::Mul(va, vb);
    UpdateCascadedSums(df, mul, sum, sum_err);
    UpdateCascadedSums(df, hn::Abs(mul), sum_abs, sum_err_abs);
  }

  const float div = std::abs(ReduceCascadedSums(df, sum, sum_err));
  if (div == 0.0f) return hn::GetLane(hn::Inf(df));
  const double cond = 2.0 * ReduceCascadedSums(df, sum_abs, sum_err_abs) /
                      static_cast<double>(div);
  HWY_ASSERT(cond >= 0.0);
  return cond;
}

//------------------------------------------------------------------------------
// Compensated dot product

#if !HWY_NATIVE_FMA

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

#endif  // !HWY_NATIVE_FMA

// Returns `prod` and `err` such that `prod + err` is exactly equal to `a * b`,
// despite floating-point rounding, assuming that `err` is not subnormal, i.e.,
// the sum of exponents >= min exponent + mantissa bits. 2..17 ops.
template <class DF, HWY_IF_FLOAT3264_D(DF), class VF = hn::Vec<DF>>
static HWY_INLINE VF TwoProducts(DF df, VF a, VF b, VF& err) {
  const VF prod = hn::Mul(a, b);
#if HWY_NATIVE_FMA
  err = hn::MulSub(a, b, prod);
#else
  VF a1, a2, b1, b2;
  VeltkampSplit(df, a, a1, a2);
  VeltkampSplit(df, b, b1, b2);
  const VF m = hn::Sub(prod, hn::Mul(a1, b1));
  const VF n = hn::Sub(m, hn::Mul(a2, b1));
  const VF o = hn::Sub(n, hn::Mul(a1, b2));
  err = hn::Sub(hn::Mul(a2, b2), o);
#endif
  return prod;
}

// Algorithm 6.15 from Handbook of Floating-Point Arithmetic.
template <class DF, typename WeightT, typename VecT>
HWY_INLINE float CompensatedDot(DF df, const WeightT* HWY_RESTRICT w,
                                size_t w_ofs, const VecT* HWY_RESTRICT x,
                                size_t num) {
  PROFILER_FUNC;
  const size_t N = hn::Lanes(df);
  HWY_ASSERT((num % (2 * N)) == 0);
  HWY_DASSERT(hn::IsAligned(df, x));
  using VF = hn::Vec<DF>;
  using TraitsW = CompressTraits<WeightT>;
  using TraitsV = CompressTraits<VecT>;

  VF sum0 = hn::Zero(df);
  VF sum1 = hn::Zero(df);
  VF sum_err0 = hn::Zero(df);
  VF sum_err1 = hn::Zero(df);

  VF w0, w1, v0, v1;              // decompressed inputs
  VF perr0, perr1, serr0, serr1;  // output arg of TwoProducts/TwoSums

  for (size_t i = 0; i < num; i += 2 * N) {
    TraitsW::Decompress2(df, w, w_ofs + i, w0, w1);
    TraitsV::Decompress2(df, x, i, v0, v1);

    const VF prod0 = TwoProducts(df, w0, v0, perr0);
    const VF prod1 = TwoProducts(df, w1, v1, perr1);

    sum0 = TwoSums(df, prod0, sum0, serr0);
    sum1 = TwoSums(df, prod1, sum1, serr1);

    sum_err0 += perr0 + serr0;
    sum_err1 += perr1 + serr1;
  }

  AssimilateCascadedSums(df, sum1, sum_err1, sum0, sum_err0);
  return ReduceCascadedSums(df, sum0, sum_err0);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
