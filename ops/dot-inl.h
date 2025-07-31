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

#include "hwy/profiler.h"

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
#include "ops/fp_arith-inl.h"
#include "hwy/contrib/math/math-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Our naming convention for dot product arguments is `w` and `v`, in that
// order. This originated in `MatVec`, which computed dot products of a
// compressed "weight" type, and `BF16/float` "vectors". This implementation no
// longer restricts the types of the arguments, but we keep the names for
// consistency, also because there is still a `w_ofs` but not a `v_ofs`.

//------------------------------------------------------------------------------

// Returns 2 * sum(|w.*v|) / |sum(w.*v)|. The log2 of this value
// approximates the number of mantissa bits required for accurate computations.
// See https://en.wikipedia.org/wiki/Condition_number.
template <typename WT, typename VT>
HWY_MAYBE_UNUSED double ConditionNumber(const WT* HWY_RESTRICT w,
                                        const VT* HWY_RESTRICT v, size_t num) {
  PROFILER_FUNC;
  const hn::ScalableTag<float> df;
  using VF = hn::Vec<decltype(df)>;
  const size_t N = hn::Lanes(df);

  VF sum = hn::Zero(df);
  VF sum_err = hn::Zero(df);
  VF sum_abs = hn::Zero(df);
  VF sum_abs_err = hn::Zero(df);

  const auto packed_w = MakeSpan(w, num);
  const auto packed_v = MakeSpan(v, num);

  size_t i = 0;
  if (num >= 2 * N) {
    for (; i <= num - 2 * N; i += 2 * N) {
      VF w0, w1, v0, v1;
      Decompress2(df, packed_w, i, w0, w1);
      Decompress2(df, packed_v, i, v0, v1);
      const VF mul0 = hn::Mul(w0, v0);
      const VF mul1 = hn::Mul(w1, v1);
      UpdateCascadedSums(df, mul0, sum, sum_err);
      UpdateCascadedSums(df, mul1, sum, sum_err);
      UpdateCascadedSums(df, hn::Abs(mul0), sum_abs, sum_abs_err);
      UpdateCascadedSums(df, hn::Abs(mul1), sum_abs, sum_abs_err);
    }
  }

  size_t remaining = num - i;
  HWY_DASSERT(remaining < 2 * N);
  if (HWY_UNLIKELY(remaining != 0)) {
    HWY_ALIGN float padded_w[2 * hn::MaxLanes(df)];
    HWY_ALIGN float padded_v[2 * hn::MaxLanes(df)];
    DecompressAndZeroPad(df, packed_w, i, padded_w, remaining);
    DecompressAndZeroPad(df, packed_v, i, padded_v, remaining);

    // 1..2 whole vectors, possibly zero-padded.
    for (size_t padded_pos = 0; padded_pos < remaining; padded_pos += N) {
      const VF w0 = hn::Load(df, padded_w + padded_pos);
      const VF v0 = hn::Load(df, padded_v + padded_pos);
      const VF mul = hn::Mul(w0, v0);
      UpdateCascadedSums(df, mul, sum, sum_err);
      UpdateCascadedSums(df, hn::Abs(mul), sum_abs, sum_abs_err);
    }
  }

  const float div = hwy::ScalarAbs(ReduceCascadedSums(df, sum, sum_err));
  if (div == 0.0f) return hn::GetLane(hn::Inf(df));
  const double cond = 2.0 * ReduceCascadedSums(df, sum_abs, sum_abs_err) /
                      static_cast<double>(div);
  HWY_ASSERT(cond >= 0.0);
  return cond;
}

// Same, but for a single vector - just skips the product.
template <typename VT>
HWY_MAYBE_UNUSED double ConditionNumber(const VT* HWY_RESTRICT v, size_t num) {
  PROFILER_FUNC;
  const hn::ScalableTag<float> df;
  using VF = hn::Vec<decltype(df)>;
  const size_t N = hn::Lanes(df);

  VF sum = hn::Zero(df);
  VF sum_err = hn::Zero(df);
  VF sum_abs = hn::Zero(df);
  VF sum_abs_err = hn::Zero(df);

  const auto packed_v = MakeSpan(v, num);

  size_t i = 0;
  if (num >= 2 * N) {
    for (; i <= num - 2 * N; i += 2 * N) {
      VF v0, v1;
      Decompress2(df, packed_v, i, v0, v1);
      UpdateCascadedSums(df, v0, sum, sum_err);
      UpdateCascadedSums(df, v1, sum, sum_err);
      UpdateCascadedSums(df, hn::Abs(v0), sum_abs, sum_abs_err);
      UpdateCascadedSums(df, hn::Abs(v1), sum_abs, sum_abs_err);
    }
  }

  size_t remaining = num - i;
  HWY_DASSERT(remaining < 2 * N);
  if (HWY_UNLIKELY(remaining != 0)) {
    HWY_ALIGN float padded_v[2 * hn::MaxLanes(df)];
    DecompressAndZeroPad(df, packed_v, i, padded_v, remaining);

    // 1..2 whole vectors, possibly zero-padded.
    for (size_t padded_pos = 0; padded_pos < remaining; padded_pos += N) {
      const VF v0 = hn::Load(df, padded_v + padded_pos);
      UpdateCascadedSums(df, v0, sum, sum_err);
      UpdateCascadedSums(df, hn::Abs(v0), sum_abs, sum_abs_err);
    }
  }

  const float div = hwy::ScalarAbs(ReduceCascadedSums(df, sum, sum_err));
  if (div == 0.0f) return hn::GetLane(hn::Inf(df));
  const double cond = 2.0 * ReduceCascadedSums(df, sum_abs, sum_abs_err) /
                      static_cast<double>(div);
  HWY_ASSERT(cond >= 0.0);
  return cond;
}

// f64 FMA. Inputs are both f32 promoted to f64, or any types that are either
// promoted or even DEMOTED to bf16. Runs at about half the speed of f32 FMA.
struct DotKernelDouble {
  // Only `CompressTraits<float>` can `Decompress2` to `double`, so both have
  // to be `float` in order to have `Raw = double`. Note that if either type is
  // smaller than `float`, we may demote the other type from `float` to `BF16`.
  template <typename VT, typename WT>
  using Raw = hwy::If<IsF32<VT>() && IsF32<WT>(), double, BF16>;
  using State = double;

  // Raw = double
  template <class DRaw, class VR = hn::Vec<DRaw>, HWY_IF_F64_D(DRaw)>
  HWY_INLINE void Update4(DRaw dd, const VR w0, const VR w1, const VR w2,
                          const VR w3, const VR v0, const VR v1, const VR v2,
                          const VR v3, VR& sum0, VR& sum1, VR& sum2, VR& sum3,
                          VR&, VR&, VR&, VR&) const {
    sum0 = hn::MulAdd(w0, v0, sum0);
    sum1 = hn::MulAdd(w1, v1, sum1);
    sum2 = hn::MulAdd(w2, v2, sum2);
    sum3 = hn::MulAdd(w3, v3, sum3);
  }

  // Raw = BF16
  template <class DRaw, class VR = hn::Vec<DRaw>, HWY_IF_BF16_D(DRaw),
            class DS = hn::Repartition<double, DRaw>, class VS = hn::Vec<DS>>
  HWY_INLINE void Update4(DRaw, const VR w0, const VR w1, const VR w2,
                          const VR w3, const VR v0, const VR v1, const VR v2,
                          const VR v3, VS& sum0, VS& sum1, VS& sum2, VS& sum3,
                          VS&, VS&, VS&, VS&) const {
    const hn::Repartition<float, DRaw> df;
    using VF = hn::Vec<decltype(df)>;
    const VF prod0 = hn::WidenMulPairwiseAdd(df, w0, v0);
    const VF prod1 = hn::WidenMulPairwiseAdd(df, w1, v1);
    // Reduce to two f32 sums so we can promote them to four f64 vectors.
    VF sum02, sum13;
    if constexpr (HWY_NATIVE_DOT_BF16) {
      // Fuse WidenMulPairwiseAdd plus Add into ReorderWidenMulAccumulate.
      VF unused0 = hn::Zero(df);
      VF unused1 = hn::Zero(df);
      sum02 = hn::ReorderWidenMulAccumulate(df, w2, v2, prod0, unused0);
      sum13 = hn::ReorderWidenMulAccumulate(df, w3, v3, prod1, unused1);
    } else {
      // ReorderWidenMulAccumulate does not help because we still end up with
      // four accumulators.
      const VF prod2 = hn::WidenMulPairwiseAdd(df, w2, v2);
      const VF prod3 = hn::WidenMulPairwiseAdd(df, w3, v3);
      sum02 = hn::Add(prod0, prod2);
      sum13 = hn::Add(prod1, prod3);
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
  HWY_INLINE void Update1(DRaw dd, const VR w0, const VR v0, VR& sum0,
                          VR&) const {
    sum0 = hn::MulAdd(w0, v0, sum0);
  }

  // Raw = BF16
  template <class DRaw, class VR = hn::Vec<DRaw>, HWY_IF_BF16_D(DRaw),
            class DS = hn::Repartition<double, DRaw>, class VS = hn::Vec<DS>>
  HWY_INLINE void Update1(DRaw, const VR w0, const VR v0, VS& sum0,
                          VS& extra0) const {
    const hn::Repartition<float, DRaw> df;
    using VF = hn::Vec<decltype(df)>;
    const VF prod0 = hn::WidenMulPairwiseAdd(df, w0, v0);

    const DS ds;
    const VS d0 = hn::PromoteLowerTo(ds, prod0);
    const VS d1 = hn::PromoteUpperTo(ds, prod0);

    sum0 = hn::Add(sum0, d0);
    extra0 = hn::Add(extra0, d1);
  }

  template <class DState, class VS = hn::Vec<DState>, HWY_IF_F64_D(DState)>
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

template <class D, typename WT, typename VT>
HWY_INLINE float DotDouble(D d, const PackedSpan<const WT>& w, size_t w_ofs,
                           const VT* HWY_RESTRICT vec, size_t num) {
  return DecompressAndCall(d, w, w_ofs, MakeSpan(vec, num), DotKernelDouble());
}

// Algorithm 6.15 from Handbook of Floating-Point Arithmetic. This about as
// accurate as DotKernelDouble but slower, hence we only use this if f64 is
// not supported on this target.
struct DotKernelCompensated {
  // The `BF16` overload uses `ReorderWidenMulAccumulate`, which requires both
  // `VT` and `WT` to be `BF16`, or smaller types decompressed to `BF16`.
  // Otherwise, we decompress both inputs to `float`.
  template <typename VT, typename WT>
  using Raw = hwy::If<IsF32<VT>() || IsF32<WT>(), float, BF16>;
  using State = float;

  // Raw = float
  template <class DRaw, class VF = hn::Vec<DRaw>, HWY_IF_F32_D(DRaw)>
  HWY_INLINE void Update4(DRaw df, const VF w0, const VF w1, const VF w2,
                          const VF w3, const VF v0, const VF v1, const VF v2,
                          const VF v3, VF& sum0, VF& sum1, VF& sum2, VF& sum3,
                          VF& comp0, VF& comp1, VF& comp2, VF& comp3) const {
    VF perr0, perr1, perr2, perr3;
    const VF prod0 = TwoProducts(df, w0, v0, perr0);
    const VF prod1 = TwoProducts(df, w1, v1, perr1);
    const VF prod2 = TwoProducts(df, w2, v2, perr2);
    const VF prod3 = TwoProducts(df, w3, v3, perr3);

    VF serr0, serr1, serr2, serr3;
    sum0 = TwoSums(df, prod0, sum0, serr0);
    sum1 = TwoSums(df, prod1, sum1, serr1);
    sum2 = TwoSums(df, prod2, sum2, serr2);
    sum3 = TwoSums(df, prod3, sum3, serr3);

    comp0 = hn::Add(comp0, hn::Add(perr0, serr0));
    comp1 = hn::Add(comp1, hn::Add(perr1, serr1));
    comp2 = hn::Add(comp2, hn::Add(perr2, serr2));
    comp3 = hn::Add(comp3, hn::Add(perr3, serr3));
  }

  // Raw = BF16, State = float
  template <class DRaw, class VR = hn::Vec<DRaw>, HWY_IF_BF16_D(DRaw),
            class DS = hn::Repartition<float, DRaw>, class VS = hn::Vec<DS>>
  HWY_INLINE void Update4(DRaw, const VR w0, const VR w1, const VR w2,
                          const VR w3, const VR v0, const VR v1, const VR v2,
                          const VR v3, VS& sum0, VS& sum1, VS& sum2, VS& sum3,
                          VS& comp0, VS& comp1, VS& comp2, VS& comp3) const {
    const DS df;
    const VS prod1 = hn::WidenMulPairwiseAdd(df, w1, v1);
    const VS prod2 = hn::WidenMulPairwiseAdd(df, w2, v2);
    const VS prod3 = hn::WidenMulPairwiseAdd(df, w3, v3);
    const VS prod0 = hn::WidenMulPairwiseAdd(df, w0, v0);

    VS serr0, serr1, serr2, serr3;
    sum0 = TwoSums(df, prod0, sum0, serr0);
    sum1 = TwoSums(df, prod1, sum1, serr1);
    sum2 = TwoSums(df, prod2, sum2, serr2);
    sum3 = TwoSums(df, prod3, sum3, serr3);

    comp0 = hn::Add(comp0, serr0);
    comp1 = hn::Add(comp1, serr1);
    comp2 = hn::Add(comp2, serr2);
    comp3 = hn::Add(comp3, serr3);
  }

  // Raw = float
  template <class DF, class VF = hn::Vec<DF>, HWY_IF_F32_D(DF)>
  HWY_INLINE void Update1(DF df, const VF w0, const VF v0, VF& sum0,
                          VF& comp0) const {
    VF perr0;
    const VF prod0 = TwoProducts(df, w0, v0, perr0);

    VF serr0;
    sum0 = TwoSums(df, prod0, sum0, serr0);

    comp0 = hn::Add(comp0, hn::Add(perr0, serr0));
  }

  // Raw = BF16, State = float
  template <class DRaw, class VR = hn::Vec<DRaw>, HWY_IF_BF16_D(DRaw),
            class DS = hn::Repartition<float, DRaw>, class VS = hn::Vec<DS>>
  HWY_INLINE void Update1(DRaw, const VR w0, const VR v0, VS& sum0,
                          VS& comp0) const {
    const DS df;
    const VS prod0 = WidenMulPairwiseAdd(df, w0, v0);

    VS serr0;
    sum0 = TwoSums(df, prod0, sum0, serr0);

    comp0 = hn::Add(comp0, serr0);
  }

  template <class DS, class VS = hn::Vec<DS>>
  HWY_INLINE float Reduce(DS df, VS& sum0, VS& sum1, VS& sum2, VS& sum3,
                          VS& comp0, VS& comp1, VS& comp2, VS& comp3) const {
    // Reduction tree: sum of all accumulators by pairs, then across lanes.
    AssimilateCascadedSums(df, sum1, comp1, sum0, comp0);
    AssimilateCascadedSums(df, sum3, comp3, sum2, comp2);
    AssimilateCascadedSums(df, sum2, comp2, sum0, comp0);
    return ReduceCascadedSums(df, sum0, comp0);
  }
};

using DotKernelDefault =
    hwy::If<HWY_HAVE_FLOAT64, DotKernelDouble, DotKernelCompensated>;

// `D` only serves to specify the vector size; its lane type is ignored.
template <class D, typename WT, typename VT>
HWY_INLINE float Dot(D d, const PackedSpan<const WT>& w, size_t w_ofs,
                     const VT* HWY_RESTRICT vec, size_t num) {
  return DecompressAndCall(d, w, w_ofs, MakeSpan(vec, num), DotKernelDefault());
}

// Adapter for two pointers, no bounds checking.
template <typename WT, typename VT>
HWY_INLINE float Dot(const WT* HWY_RESTRICT w, const VT* vec, size_t num) {
  const hn::ScalableTag<VT> d;
  return Dot(d, MakeConstSpan(w, num), /*w_ofs=*/0, vec, num);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
