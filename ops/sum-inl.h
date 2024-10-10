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

#include "hwy/base.h"

// Include guard for SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_SUM_TOGGLE) == defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_SUM_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_SUM_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_SUM_TOGGLE
#endif

#include "compression/compress-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

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

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
