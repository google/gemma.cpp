// Copyright 2023 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "compression/types.h"
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include <stddef.h>
#include <stdio.h>

#include <algorithm>  // std::swap, std::sort
#include <array>
#include <cmath>
#include <random>

#include "compression/compress.h"
#include "util/allocator.h"
#include "util/test_util.h"
#include "util/threading_context.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"
#include "hwy/stats.h"
#include "hwy/timer.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ops/dot_test.cc"
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/test_util-inl.h"
#include "ops/dot-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

enum {  // alphabetical order for consistency and to avoid implying a preference
  kAddTwoProd,
  kAddTwoSum,
  kComp2,
  kCompensated,
  kDouble,
  kKahan,
  kNaive,
  kOnlyTwoProd,
  kPairwise,

  kVariants
};

const char* VariantName(size_t variant) {
  switch (variant) {
    case kAddTwoProd:
      return "add2prod";
    case kAddTwoSum:
      return "add2sum";
    case kComp2:
      return "comp2";
    case kCompensated:
      return "comp";
    case kDouble:
      return "double";
    case kKahan:
      return "kahan";
    case kNaive:
      return "naive";
    case kOnlyTwoProd:
      return "only2prod";
    case kPairwise:
      return "pairwise";
    default:
      HWY_ABORT("Unknown variant %zu", variant);
      return "?";
  }
}

// Wrapper functions allow disabling HWY_ASSERT so that we see all failures in
// one run and can update all thresholds at once.
template <typename T>
void AssertInside(size_t variant, T min, T actual, T max, int line) {
  if (!gcpp::IsInside(min, max, actual)) {
    fprintf(stderr, "!!line %03d, %s actual %E not in [%E, %E]\n", line,
            VariantName(variant), actual, min, max);
    HWY_ASSERT(false);
  }
}

template <typename T>
void AssertLess(size_t variant, T actual, T max, int line) {
  AssertInside(variant, hwy::LowestValue<T>(), actual, max, line);
}

#define ASSERT_LESS(variant, actual, max) \
  AssertLess(variant, actual, max, __LINE__)

#define ASSERT_INSIDE(variant, min, actual, max) \
  AssertInside(variant, min, actual, max, __LINE__)

//------------------------------------------------------------------------------
// Dot product variants

// All combinations of {*, TwoProducts} x {+, FastTwoSums, TwoSums}.

struct DotKernelNaive {
  template <typename VT, typename WT>
  using Raw = float;
  using State = float;

  template <class DF, class VF = hn::Vec<DF>>
  HWY_INLINE void Update4(DF df, const VF w0, const VF w1, const VF w2,
                          const VF w3, const VF v0, const VF v1, const VF v2,
                          const VF v3, VF& sum0, VF& sum1, VF& sum2, VF& sum3,
                          VF& /*comp0*/, VF& /*comp1*/, VF& /*comp2*/,
                          VF& /*comp3*/) const {
    sum0 = hn::MulAdd(w0, v0, sum0);
    sum1 = hn::MulAdd(w1, v1, sum1);
    sum2 = hn::MulAdd(w2, v2, sum2);
    sum3 = hn::MulAdd(w3, v3, sum3);
  }

  template <class DF, class VF = hn::Vec<DF>>
  HWY_INLINE void Update1(DF df, const VF w0, const VF v0, VF& sum0,
                          VF& /*comp0*/) const {
    sum0 = hn::MulAdd(w0, v0, sum0);
  }

  template <class DF, class VF = hn::Vec<DF>>
  HWY_INLINE float Reduce(DF df, VF& sum0, VF& sum1, VF& sum2, VF& sum3,
                          VF& /*comp0*/, VF& /*comp1*/, VF& /*comp2*/,
                          VF& /*comp3*/) const {
    // Reduction tree: sum of all accumulators by pairs, then across lanes.
    sum0 = hn::Add(sum0, sum1);
    sum2 = hn::Add(sum2, sum3);
    sum0 = hn::Add(sum0, sum2);
    return hn::ReduceSum(df, sum0);
  }
};

template <class D, typename WT, typename VT>
HWY_INLINE float DotNaive(D d, const PackedSpan<const WT>& w, size_t w_ofs,
                          const VT* HWY_RESTRICT vec, size_t num) {
  return DecompressAndCall(d, w, w_ofs, MakeSpan(vec, num), DotKernelNaive());
}

// https://en.wikipedia.org/wiki/Kahan_summation_algorithm: FastTwoSum.
struct DotKernelKahan {
  template <typename VT, typename WT>
  using Raw = float;
  using State = float;

  template <class DF, class VF = hn::Vec<DF>>
  HWY_INLINE void Update4(DF df, const VF w0, const VF w1, const VF w2,
                          const VF w3, const VF v0, const VF v1, const VF v2,
                          const VF v3, VF& sum0, VF& sum1, VF& sum2, VF& sum3,
                          VF& comp0, VF& comp1, VF& comp2, VF& comp3) const {
    // Add compensation from last iteration, which is an approximation of the
    // running error.
    const VF prod0 = hn::MulAdd(w0, v0, comp0);
    const VF prod1 = hn::MulAdd(w1, v1, comp1);
    const VF prod2 = hn::MulAdd(w2, v2, comp2);
    const VF prod3 = hn::MulAdd(w3, v3, comp3);

    sum0 = FastTwoSums(df, sum0, prod0, comp0);
    sum1 = FastTwoSums(df, sum1, prod1, comp1);
    sum2 = FastTwoSums(df, sum2, prod2, comp2);
    sum3 = FastTwoSums(df, sum3, prod3, comp3);
  }

  template <class DF, class VF = hn::Vec<DF>>
  HWY_INLINE void Update1(DF df, const VF w0, const VF v0, VF& sum0,
                          VF& comp0) const {
    const VF prod0 = hn::MulAdd(w0, v0, comp0);
    sum0 = FastTwoSums(df, sum0, prod0, comp0);
  }

  template <class DF, class VF = hn::Vec<DF>>
  HWY_INLINE float Reduce(DF df, VF& sum0, VF& sum1, VF& sum2, VF& sum3,
                          VF& comp0, VF& comp1, VF& comp2, VF& comp3) const {
    // Reduction tree: sum of all accumulators by pairs, then across lanes.
    comp0 = hn::Add(comp0, comp1);
    comp2 = hn::Add(comp2, comp3);
    VF sum_err = hn::Add(comp0, comp2);
    UpdateCascadedSums(df, sum1, sum0, sum_err);
    UpdateCascadedSums(df, sum3, sum2, sum_err);
    UpdateCascadedSums(df, sum2, sum0, sum_err);
    return ReduceCascadedSums(df, sum0, sum_err);
  }
};

template <class D, typename WT, typename VT>
HWY_INLINE float DotKahan(D d, const PackedSpan<const WT>& w, size_t w_ofs,
                          const VT* HWY_RESTRICT vec, size_t num) {
  return DecompressAndCall(d, w, w_ofs, MakeSpan(vec, num), DotKernelKahan());
}

template <class D, typename WT, typename VT>
HWY_INLINE float DotCompensated(D d, const PackedSpan<const WT>& w,
                                size_t w_ofs, const VT* HWY_RESTRICT vec,
                                size_t num) {
  return DecompressAndCall(d, w, w_ofs, MakeSpan(vec, num),
                           DotKernelCompensated());
}

// Like Compensated, but FastTwoSum instead of TwoSum.
struct DotKernelTwoProdFast {
  template <typename VT, typename WT>
  using Raw = float;
  using State = float;

  template <class DF, class VF = hn::Vec<DF>>
  HWY_INLINE void Update4(DF df, const VF w0, const VF w1, const VF w2,
                          const VF w3, const VF v0, const VF v1, const VF v2,
                          const VF v3, VF& sum0, VF& sum1, VF& sum2, VF& sum3,
                          VF& comp0, VF& comp1, VF& comp2, VF& comp3) const {
    VF perr0, perr1, perr2, perr3;
    const VF prod0 = TwoProducts(df, w0, v0, perr0);
    const VF prod1 = TwoProducts(df, w1, v1, perr1);
    const VF prod2 = TwoProducts(df, w2, v2, perr2);
    const VF prod3 = TwoProducts(df, w3, v3, perr3);

    VF serr0, serr1, serr2, serr3;
    sum0 = FastTwoSums(df, sum0, prod0, serr0);
    sum1 = FastTwoSums(df, sum1, prod1, serr1);
    sum2 = FastTwoSums(df, sum2, prod2, serr2);
    sum3 = FastTwoSums(df, sum3, prod3, serr3);

    comp0 = hn::Add(comp0, hn::Add(perr0, serr0));
    comp1 = hn::Add(comp1, hn::Add(perr1, serr1));
    comp2 = hn::Add(comp2, hn::Add(perr2, serr2));
    comp3 = hn::Add(comp3, hn::Add(perr3, serr3));
  }

  template <class DF, class VF = hn::Vec<DF>>
  HWY_INLINE void Update1(DF df, const VF w0, const VF v0, VF& sum0,
                          VF& comp0) const {
    VF perr0;
    const VF prod0 = TwoProducts(df, w0, v0, perr0);

    VF serr0;
    sum0 = FastTwoSums(df, sum0, prod0, serr0);

    comp0 = hn::Add(comp0, hn::Add(perr0, serr0));
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

template <class D, typename WT, typename VT>
HWY_INLINE float DotTwoProdFast(D d, const PackedSpan<const WT>& w,
                                size_t w_ofs, const VT* HWY_RESTRICT vec,
                                size_t num) {
  return DecompressAndCall(d, w, w_ofs, MakeSpan(vec, num),
                           DotKernelTwoProdFast());
}

// Like Compensated, but without TwoProducts. Vs Kahan, upgrades FastTwoSums
// to TwoSums.
struct DotKernelMulTwoSum {
  template <typename VT, typename WT>
  using Raw = float;
  using State = float;

  template <class DF, class VF = hn::Vec<DF>>
  HWY_INLINE void Update4(DF df, const VF w0, const VF w1, const VF w2,
                          const VF w3, const VF v0, const VF v1, const VF v2,
                          const VF v3, VF& sum0, VF& sum1, VF& sum2, VF& sum3,
                          VF& comp0, VF& comp1, VF& comp2, VF& comp3) const {
    const VF prod0 = hn::Mul(w0, v0);
    const VF prod1 = hn::Mul(w1, v1);
    const VF prod2 = hn::Mul(w2, v2);
    const VF prod3 = hn::Mul(w3, v3);

    VF serr0, serr1, serr2, serr3;
    sum0 = TwoSums(df, prod0, sum0, serr0);
    sum1 = TwoSums(df, prod1, sum1, serr1);
    sum2 = TwoSums(df, prod2, sum2, serr2);
    sum3 = TwoSums(df, prod3, sum3, serr3);

    comp0 = hn::Add(comp0, serr0);
    comp1 = hn::Add(comp1, serr1);
    comp2 = hn::Add(comp2, serr2);
    comp3 = hn::Add(comp3, serr3);
  }

  template <class DF, class VF = hn::Vec<DF>>
  HWY_INLINE void Update1(DF df, const VF w0, const VF v0, VF& sum0,
                          VF& comp0) const {
    const VF prod0 = hn::Mul(w0, v0);

    VF serr0;
    sum0 = TwoSums(df, prod0, sum0, serr0);

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

template <class D, typename WT, typename VT>
HWY_INLINE float DotMulTwoSum(D d, const PackedSpan<const WT>& w, size_t w_ofs,
                              const VT* HWY_RESTRICT vec, size_t num) {
  return DecompressAndCall(d, w, w_ofs, MakeSpan(vec, num),
                           DotKernelMulTwoSum());
}

// -Like Compensated, but only TwoProducts, no [Fast]TwoSums. This is only 10%
// better (mul) than naive.
struct DotKernelTwoProdAdd {
  template <typename VT, typename WT>
  using Raw = float;
  using State = float;

  template <class DF, class VF = hn::Vec<DF>>
  HWY_INLINE void Update4(DF df, const VF w0, const VF w1, const VF w2,
                          const VF w3, const VF v0, const VF v1, const VF v2,
                          const VF v3, VF& sum0, VF& sum1, VF& sum2, VF& sum3,
                          VF& comp0, VF& comp1, VF& comp2, VF& comp3) const {
    VF perr0, perr1, perr2, perr3;
    const VF prod0 = TwoProducts(df, w0, v0, perr0);
    const VF prod1 = TwoProducts(df, w1, v1, perr1);
    const VF prod2 = TwoProducts(df, w2, v2, perr2);
    const VF prod3 = TwoProducts(df, w3, v3, perr3);

    sum0 = hn::Add(sum0, prod0);
    sum1 = hn::Add(sum1, prod1);
    sum2 = hn::Add(sum2, prod2);
    sum3 = hn::Add(sum3, prod3);

    comp0 = hn::Add(comp0, perr0);
    comp1 = hn::Add(comp1, perr1);
    comp2 = hn::Add(comp2, perr2);
    comp3 = hn::Add(comp3, perr3);
  }

  template <class DF, class VF = hn::Vec<DF>>
  HWY_INLINE void Update1(DF df, const VF w0, const VF v0, VF& sum0,
                          VF& comp0) const {
    VF perr0;
    const VF prod0 = TwoProducts(df, w0, v0, perr0);

    sum0 = hn::Add(sum0, prod0);

    comp0 = hn::Add(comp0, perr0);
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

template <class D, typename WT, typename VT>
HWY_INLINE float DotTwoProdAdd(D d, const PackedSpan<const WT>& w, size_t w_ofs,
                               const VT* HWY_RESTRICT vec, size_t num) {
  return DecompressAndCall(d, w, w_ofs, MakeSpan(vec, num),
                           DotKernelTwoProdAdd());
}

// From "SIMDizing Pairwise Sums". Slower and generally higher error than
// Kahan, but uses fewer regs.
struct DotKernelPairwise {
  template <typename VT, typename WT>
  using Raw = float;
  using State = float;

  template <class DF, class VF = hn::Vec<DF>>
  HWY_INLINE void Update4(DF df, const VF w0, const VF w1, const VF w2,
                          const VF w3, const VF v0, const VF v1, const VF v2,
                          const VF v3, VF& sum0, VF& sum1, VF& sum2, VF& sum3,
                          VF& comp0, VF& comp1, VF& comp2, VF& comp3) const {
    const size_t N = hn::Lanes(df);
    const VF prod0 = hn::Mul(w0, v0);
    const VF prod2 = hn::Mul(w2, v2);
    const VF prod1 = hn::MulAdd(w1, v1, prod0);
    const VF prod3 = hn::MulAdd(w3, v3, prod2);
    VF sum = hn::Add(prod1, prod3);
    for (size_t bit = 4 * N; bit & num_; bit += bit, top_ -= N) {
      HWY_DASSERT(top_ >= N);
      HWY_DASSERT(top_ <= 32 * N);
      sum = hn::Add(sum, hn::LoadU(df, stack_ + top_ - N));
    }
    hn::StoreU(sum, df, stack_ + top_);
    top_ += N;
    HWY_DASSERT(top_ <= 32 * N);
    num_ += 4 * N;
  }

  template <class DF, class VF = hn::Vec<DF>>
  HWY_INLINE void Update1(DF df, const VF w0, const VF v0, VF& sum0,
                          VF& comp0) const {
    const size_t N = hn::Lanes(df);
    VF sum = hn::Mul(w0, v0);
    for (size_t bit = N; bit & num_; bit += bit, top_ -= N) {
      HWY_DASSERT(top_ >= N);
      HWY_DASSERT(top_ <= 32 * N);
      sum = hn::Add(sum, hn::LoadU(df, stack_ + top_ - N));
    }
    hn::StoreU(sum, df, stack_ + top_);
    top_ += N;
    HWY_DASSERT(top_ <= 32 * N);
    num_ += N;
  }

  template <class DF, class VF = hn::Vec<DF>>
  HWY_INLINE float Reduce(DF df, VF& sum0, VF& sum1, VF& sum2, VF& sum3,
                          VF& comp0, VF& comp1, VF& comp2, VF& comp3) const {
    const size_t N = hn::Lanes(df);
    sum0 = hn::Zero(df);
    for (; top_ != 0; top_ -= N) {
      sum0 = hn::Add(sum0, hn::LoadU(df, stack_ + top_ - N));
    }
    return hn::ReduceSum(df, sum0);
  }

 private:
  HWY_ALIGN mutable float stack_[32 * hn::MaxLanes(hn::ScalableTag<float>())];
  mutable size_t top_ = 0;
  mutable size_t num_ = 0;
};

template <class D, typename WT, typename VT>
HWY_INLINE float DotPairwise(D d, const PackedSpan<const WT>& w, size_t w_ofs,
                             const VT* HWY_RESTRICT vec, size_t num) {
  return DecompressAndCall(d, w, w_ofs, MakeSpan(vec, num),
                           DotKernelPairwise());
}

// Hybrid of Pairwise and Compensated. 1.14x time vs. Kahan, but geomean mul
// is 1.02 vs 1.06, mean L1 is 1.21x better, and uses two fewer regs.
struct DotKernelComp2 {
  template <typename VT, typename WT>
  using Raw = float;
  using State = float;

  template <class DF, class VF = hn::Vec<DF>, HWY_IF_F32_D(DF)>
  HWY_INLINE void Update4(DF df, const VF w0, const VF w1, const VF w2,
                          const VF w3, const VF v0, const VF v1, const VF v2,
                          const VF v3, VF& sum0, VF& /*sum1*/, VF& sum2,
                          VF& /*sum3*/, VF& comp0, VF& comp1, VF& comp2,
                          VF& comp3) const {
    VF perr0, perr1, perr2, perr3;
    VF prod0 = TwoProducts(df, w0, v0, perr0);
    VF prod1 = TwoProducts(df, w1, v1, perr1);
    VF prod2 = TwoProducts(df, w2, v2, perr2);
    VF prod3 = TwoProducts(df, w3, v3, perr3);

    // Pairwise sums of prod* and perr*.
    prod0 = hn::Add(prod0, prod1);
    prod2 = hn::Add(prod2, prod3);
    perr0 = hn::Add(perr0, perr1);
    perr2 = hn::Add(perr2, perr3);

    VF serr0, serr2;
    sum0 = TwoSums(df, prod0, sum0, serr0);
    sum2 = TwoSums(df, prod2, sum2, serr2);

    comp0 = hn::Add(comp0, perr0);
    comp1 = hn::Add(comp1, perr2);
    comp2 = hn::Add(comp2, serr0);
    comp3 = hn::Add(comp3, serr2);
  }

  template <class DBF, class VBF = hn::Vec<DBF>, HWY_IF_BF16_D(DBF),
            class DF = hn::Repartition<float, DBF>, class VF = hn::Vec<DF>>
  HWY_INLINE void Update4(DBF /*dbf*/, const VBF w0, const VBF w1, const VBF w2,
                          const VBF w3, const VBF v0, const VBF v1,
                          const VBF v2, const VBF v3, VF& sum0, VF& sum1,
                          VF& sum2, VF& sum3, VF& comp0, VF& comp1, VF& comp2,
                          VF& comp3) const {
    const DF df;
    VF prod0 = hn::WidenMulPairwiseAdd(df, w0, v0);
    VF prod1 = hn::WidenMulPairwiseAdd(df, w1, v1);
    VF prod2 = hn::WidenMulPairwiseAdd(df, w2, v2);
    VF prod3 = hn::WidenMulPairwiseAdd(df, w3, v3);

    // Pairwise sums
    prod0 = hn::Add(prod0, prod1);
    prod2 = hn::Add(prod2, prod3);
    prod0 = hn::Add(prod0, prod2);

    VF serr0;
    sum0 = TwoSums(df, prod0, sum0, serr0);
    comp0 = hn::Add(comp0, serr0);
  }

  template <class DF, class VF = hn::Vec<DF>, HWY_IF_F32_D(DF)>
  HWY_INLINE void Update1(DF df, const VF w0, const VF v0, VF& sum0,
                          VF& comp0) const {
    VF perr0;
    const VF prod0 = TwoProducts(df, w0, v0, perr0);

    VF serr0;
    sum0 = TwoSums(df, prod0, sum0, serr0);

    comp0 = hn::Add(comp0, hn::Add(perr0, serr0));
  }

  template <class DBF, class VBF = hn::Vec<DBF>, HWY_IF_BF16_D(DBF),
            class DF = hn::Repartition<float, DBF>, class VF = hn::Vec<DF>>
  HWY_INLINE void Update1(DBF /*dbf*/, const VBF w0, const VBF v0, VF& sum0,
                          VF& comp0) const {
    const DF df;
    const VF prod0 = WidenMulPairwiseAdd(df, w0, v0);

    VF serr0;
    sum0 = TwoSums(df, prod0, sum0, serr0);
    comp0 = hn::Add(comp0, serr0);
  }

  template <class DF, class VF = hn::Vec<DF>>
  HWY_INLINE float Reduce(DF df, VF& sum0, VF& sum1, VF& sum2, VF& sum3,
                          VF& comp0, VF& comp1, VF& comp2, VF& comp3) const {
    AssimilateCascadedSums(df, sum2, comp2, sum0, comp0);
    comp1 = hn::Add(comp1, comp3);
    return ReduceCascadedSums(df, sum0, hn::Add(comp0, comp1));
  }
};

template <class D, typename WT, typename VT>
HWY_INLINE float DotComp2(D d, const PackedSpan<const WT>& w, size_t w_ofs,
                          const VT* HWY_RESTRICT vec, size_t num) {
  return DecompressAndCall(d, w, w_ofs, MakeSpan(vec, num), DotKernelComp2());
}

template <class D, typename WT, typename VT, HWY_IF_F32_D(D)>
float CallDot(D d, size_t variant, const PackedSpan<const WT>& w, size_t w_ofs,
              const VT* HWY_RESTRICT v, size_t num) {
  switch (variant) {
    case kAddTwoProd:
      return DotTwoProdFast(d, w, 0, v, num);
    case kAddTwoSum:
      return DotMulTwoSum(d, w, 0, v, num);
    case kComp2:
      return DotComp2(d, w, 0, v, num);
    case kCompensated:
      return DotCompensated(d, w, 0, v, num);
    case kDouble:
      if constexpr (HWY_HAVE_FLOAT64) {
        return DotDouble(d, w, 0, v, num);
      } else {
        return DotCompensated(d, w, 0, v, num);
      }
    case kKahan:
      return DotKahan(d, w, 0, v, num);
    case kNaive:
      return DotNaive(d, w, 0, v, num);
    case kOnlyTwoProd:
      return DotTwoProdAdd(d, w, 0, v, num);
    case kPairwise:
      return DotPairwise(d, w, 0, v, num);
    default:
      HWY_ABORT("Unknown variant %zu", variant);
      return 0.0f;
  }
}

// Returns result accurate to 1.5 ulp, assuming `num` < 2^(52-23), no overflow,
// and round to nearest. See "Accurate and efficient floating point summation".
// Much too slow to be useful. Kept separate from the above kernels because it
// is used to compute their error.
template <typename WT, typename VT>
float ExactDot(const WT* HWY_RESTRICT w, const VT* HWY_RESTRICT v, size_t num,
               double* HWY_RESTRICT buf) {
  PROFILER_FUNC;
  for (size_t i = 0; i < num; ++i) {
    buf[i] =
        hwy::ConvertScalarTo<double>(w[i]) * hwy::ConvertScalarTo<double>(v[i]);
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

class DotStats {
  static float Ratio(float a, float b) {
    // If 0, we would return infinity, which messes up the statistics.
    if (a == 0.0f || b == 0.0f) return 1.0f;
    // Absolute value because a sign change and 4x difference would
    // otherwise return the smaller ratio 0.25.
    return HWY_MAX(std::abs(a / b), std::abs(b / a));
  }

 public:
  DotStats() {
    for (size_t i = 0; i < kVariants; ++i) {
      max_muls[i] = 1.0f;
    }
  }

  static void PrintStats(const char* caption, size_t variant,
                         const hwy::Stats& stats) {
    fprintf(stderr, "%s %9s %s\n", caption, VariantName(variant),
            stats.ToString(/*exclude=*/0).c_str());
  }

  // Call once per rep.
  void NotifyRep(size_t num, double cond, float dot_exact,
                 float dots[kVariants]) {
    s_cond.Notify(cond);
    const float mul_tol = cond > 1E8 ? 1.5f : cond > 1E7 ? 1.1f : 1.01f;

    float muls[kVariants];  // ratio
    float l1s[kVariants];   // abs error
    float rels[kVariants];  // relative forward error
    float bwds[kVariants];  // backward error
    int bits[kVariants];    // 'bits correct'
    uint32_t ulps[kVariants];
    for (size_t i = 0; i < kVariants; ++i) {
      muls[i] = Ratio(dots[i], dot_exact);
      max_muls[i] = HWY_MAX(max_muls[i], muls[i]);

      l1s[i] = std::abs(dots[i] - dot_exact);
      const float abs_dot = hwy::ScalarAbs(dots[i]);
      rels[i] = l1s[i] / HWY_MAX(abs_dot, 1E-6f);  // avoid infinity
      bwds[i] = rels[i] / cond;
      bits[i] = HWY_MIN(-std::log2(rels[i]), hwy::MantissaBits<float>());
      s_l1s[i].Notify(l1s[i]);
      s_rels[i].Notify(rels[i]);
      s_bwds[i].Notify(bwds[i]);
      s_bits[i].Notify(bits[i]);

      ulps[i] = hwy::detail::ComputeUlpDelta(dots[i], dot_exact);
      s_ulps[i].Notify(ulps[i]);
    }

    if (muls[kKahan] > mul_tol || l1s[kKahan] > 0.1f ||
        muls[kNaive] + 1E-3f < muls[kKahan] || ulps[kCompensated] > 10) {
      fprintf(stderr, "num %2zu cond %.1E exact %.8f\n", num, cond, dot_exact);
      for (size_t i = 0; i < kVariants; ++i) {
        fprintf(stderr, "  %9s dot %11.8f mul %.8f rel %f bwd %f bits %d\n",
                VariantName(i), dots[i], muls[i], rels[i], bwds[i], bits[i]);
      }
    }
  }

  // Call after all reps.
  void NotifyRatios() {
    for (size_t i = 0; i < kVariants; ++i) {
      s_muls[i].Notify(max_muls[i]);
    }
  }

  void NotifyTimes(double times[kVariants]) {
    for (size_t i = 0; i < kVariants; ++i) {
      s_times[i].Notify(times[i]);
    }
  }

  // Forward to all members' Assimilate().
  void Assimilate(const DotStats& other) {
    s_cond.Assimilate(other.s_cond);
    for (size_t i = 0; i < kVariants; ++i) {
      s_muls[i].Assimilate(other.s_muls[i]);
      s_l1s[i].Assimilate(other.s_l1s[i]);
      s_rels[i].Assimilate(other.s_rels[i]);
      s_bwds[i].Assimilate(other.s_bwds[i]);
      s_bits[i].Assimilate(other.s_bits[i]);
      s_ulps[i].Assimilate(other.s_ulps[i]);
      s_times[i].Assimilate(other.s_times[i]);
    }
  }

  void Print() const {
    PrintStats("cond", 0, s_cond);
    for (size_t variant = 0; variant < kVariants; ++variant) {
      PrintStats("mul", variant, s_muls[variant]);
    }
    for (size_t variant = 0; variant < kVariants; ++variant) {
      PrintStats(" l1", variant, s_l1s[variant]);
    }
    for (size_t variant = 0; variant < kVariants; ++variant) {
      PrintStats("rel", variant, s_rels[variant]);
    }
    for (size_t variant = 0; variant < kVariants; ++variant) {
      PrintStats("bwd", variant, s_bwds[variant]);
    }
    for (size_t variant = 0; variant < kVariants; ++variant) {
      PrintStats("bits", variant, s_bits[variant]);
    }
    for (size_t variant = 0; variant < kVariants; ++variant) {
      PrintStats("ulp", variant, s_ulps[variant]);
    }
    if (s_times[0].Count()) {
      for (size_t variant = 0; variant < kVariants; ++variant) {
        PrintStats("time", variant, s_times[variant]);
      }
    }
  }

  void Check() const {
    CheckMuls();
    CheckL1();
    CheckRel();
    CheckBwd();
    // No need to check bits, it is a monotonic function of rel.
    CheckUlps();

    // We do not check times because they can be noisy/nonportable, but
    // `kAddTwoProd` is only about 10% slower than `kKahan`, and about 1.5 times
    // as fast as `kCompensated`.
  }

 private:
  // Factor by which the approximate result is off; lower is better.
  void CheckMuls() const {
    // Comp2 is between Compensated and Kahan.
    ASSERT_INSIDE(kComp2, 1.001, s_muls[kComp2].Mean(), 1.4);
    ASSERT_INSIDE(kComp2, 1.001f, s_muls[kComp2].Max(), 2.4f);
    ASSERT_INSIDE(kComp2, 1.0, s_muls[kComp2].GeometricMean(), 1.2);

    // Compensated and Double are very accurate.
    ASSERT_LESS(kCompensated, s_muls[kCompensated].Min(), 1.0f + 2E-6f);
    ASSERT_LESS(kCompensated, s_muls[kCompensated].Max(), 1.0f + 1E-4f);
    ASSERT_LESS(kDouble, s_muls[kDouble].Min(), 1.0f + 2E-6f);
    ASSERT_LESS(kDouble, s_muls[kDouble].Max(), 1.0f + 2E-5f);

    // Naive and OnlyTwoProd are considerably worse. >10x is for narrower
    // vectors, compared to AVX-512. GeometricMean overflows, must use Mean.
    ASSERT_INSIDE(kNaive, 1.01, s_muls[kNaive].Mean(), 16.0);
    ASSERT_INSIDE(kOnlyTwoProd, 1.01, s_muls[kOnlyTwoProd].Mean(), 73.0);

    // Kahan (FastTwoSum) is decent:
    ASSERT_INSIDE(kKahan, 1.0005, s_muls[kKahan].Mean(), 4.1);
    ASSERT_INSIDE(kKahan, 1.001f, s_muls[kKahan].Max(), 14.1f);
    ASSERT_INSIDE(kKahan, 1.0, s_muls[kKahan].GeometricMean(), 1.6);

    // But can be considerably improved via TwoProducts:
    ASSERT_INSIDE(kAddTwoProd, 1.0005, s_muls[kAddTwoProd].Mean(), 1.5);
    ASSERT_INSIDE(kAddTwoProd, 1.001f, s_muls[kAddTwoProd].Max(), 2.3f);
    ASSERT_INSIDE(kAddTwoProd, 1.0, s_muls[kAddTwoProd].GeometricMean(), 1.2);
    // Updating Kahan's FastTwoSums to TwoSums is not quite as helpful.
    ASSERT_INSIDE(kAddTwoSum, 1.0005, s_muls[kAddTwoSum].Mean(), 2.2);
    ASSERT_INSIDE(kAddTwoSum, 1.0, s_muls[kAddTwoSum].GeometricMean(), 1.3);

    ASSERT_INSIDE(kPairwise, 1.0, s_muls[kPairwise].GeometricMean(), 1.6);
  }

  // Absolute error; lower is better.
  void CheckL1() const {
    // Comp2 is between Compensated and Kahan.
    ASSERT_INSIDE(kComp2, 1E-5, s_l1s[kComp2].Mean(), 9E-4);
    ASSERT_INSIDE(kComp2, 1E-5f, s_l1s[kComp2].Max(), 2.6E-3f);

    // Compensated and Double are very accurate.
    HWY_ASSERT(s_l1s[kCompensated].Min() == 0.0f);
    ASSERT_LESS(kCompensated, s_l1s[kCompensated].Max(), 3E-7f);
    HWY_ASSERT(s_l1s[kDouble].Min() == 0.0f);
    ASSERT_LESS(kDouble, s_l1s[kDouble].Max(), 3E-7f);

    // Naive and OnlyTwoProd are considerably higher, but not huge.
    ASSERT_INSIDE(kNaive, 1E-3, s_l1s[kNaive].Mean(), 2E-2);
    ASSERT_INSIDE(kOnlyTwoProd, 1E-3, s_l1s[kOnlyTwoProd].Mean(), 2E-2);

    // Kahan (FastTwoSum) is decent:
    ASSERT_INSIDE(kKahan, 3E-4, s_l1s[kKahan].Mean(), 1E-3);
    ASSERT_INSIDE(kKahan, 6E-4f, s_l1s[kKahan].Max(), 3.2E-3f);

    // But can be nearly halved via TwoProducts:
    ASSERT_INSIDE(kAddTwoProd, 2.2E-4, s_l1s[kAddTwoProd].Mean(), 8E-4);
    ASSERT_INSIDE(kAddTwoProd, 4E-4f, s_l1s[kAddTwoProd].Max(), 2.1E-3f);
    // Updating Kahan's FastTwoSums to TwoSums does help a bit.
    ASSERT_INSIDE(kAddTwoSum, 1.5E-4, s_l1s[kAddTwoSum].Mean(), 5.8E-4);

    ASSERT_INSIDE(kPairwise, 4.5E-4, s_l1s[kPairwise].Mean(), 4E-3);
    ASSERT_INSIDE(kPairwise, 1.1E-3f, s_l1s[kPairwise].Max(), 1E-2f);
  }

  // Forward relative error, lower is better.
  void CheckRel() const {
    ASSERT_INSIDE(kComp2, 2E-4, s_rels[kComp2].GeometricMean(), 4E-3);
    ASSERT_INSIDE(kComp2, 1E-5f, s_rels[kComp2].Max(), 1.23f);

    // Compensated and Double are very accurate.
    ASSERT_LESS(kCompensated, s_rels[kCompensated].Min(), 1E-8f);
    ASSERT_LESS(kCompensated, s_rels[kCompensated].Max(), 8E-6f);
    ASSERT_LESS(kDouble, s_rels[kDouble].Min(), 1E-8f);
    ASSERT_LESS(kDouble, s_rels[kDouble].Max(), 8E-6f);

    // Naive and OnlyTwoProd are considerably higher, but not huge.
    ASSERT_INSIDE(kNaive, 1E-3, s_rels[kNaive].GeometricMean(), 8E-2);
    ASSERT_INSIDE(kOnlyTwoProd, 1E-3, s_rels[kOnlyTwoProd].GeometricMean(),
                  0.072);

    // Kahan (FastTwoSum) is decent:
    ASSERT_INSIDE(kKahan, 3E-4, s_rels[kKahan].GeometricMean(), 3.5E-3);
    ASSERT_INSIDE(kKahan, 6E-4f, s_rels[kKahan].Max(), 0.7f);

    // TwoProducts and TwoSums are a bit better.
    ASSERT_INSIDE(kAddTwoProd, 2.2E-4, s_rels[kAddTwoProd].GeometricMean(),
                  3E-3);
    ASSERT_INSIDE(kAddTwoProd, 4E-4f, s_rels[kAddTwoProd].Max(), 0.19f);
    ASSERT_INSIDE(kAddTwoSum, 1.5E-4, s_rels[kAddTwoSum].GeometricMean(),
                  2.6E-3);

    ASSERT_INSIDE(kPairwise, 4.5E-4, s_rels[kPairwise].GeometricMean(), 1.5E-2);
    // Extremely high error on aarch64.
    ASSERT_INSIDE(kPairwise, 1.1E-3f, s_rels[kPairwise].Max(), 2E3f);
  }

  // Backward relative error, lower is better.
  void CheckBwd() const {
    ASSERT_INSIDE(kComp2, 7E-10f, s_rels[kComp2].Max(), 1.3f);

    // Compensated and Double are very accurate.
    ASSERT_LESS(kCompensated, s_rels[kCompensated].Max(), 8E-6f);
    ASSERT_LESS(kDouble, s_rels[kDouble].Max(), 8E-6f);

    // Naive and OnlyTwoProd are considerably higher than others
    ASSERT_INSIDE(kNaive, 1.5E-8f, s_rels[kNaive].Max(), 3080.f);
    ASSERT_INSIDE(kOnlyTwoProd, 1.5E-8f, s_rels[kNaive].Max(), 3080.f);
    // Kahan (FastTwoSum) is not much better here!
    ASSERT_INSIDE(kKahan, 6E-10f, s_rels[kKahan].Max(), 0.7f);

    // But TwoProducts/TwoSums help a bit.
    ASSERT_INSIDE(kAddTwoProd, 9E-10f, s_rels[kAddTwoProd].Max(), 0.19f);
    ASSERT_INSIDE(kAddTwoSum, 5E-10f, s_rels[kAddTwoSum].Max(), 0.34f);

    // Extremely high error on aarch64.
    ASSERT_INSIDE(kPairwise, 7E-10f, s_rels[kPairwise].Max(), 2000.f);
  }

  // Units in the last place; lower is better.
  void CheckUlps() const {
    ASSERT_LESS(kComp2, s_ulps[kCompensated].Max(), 3.6E6f);
    ASSERT_LESS(kCompensated, s_ulps[kCompensated].Max(), 250.0f);
    ASSERT_LESS(kDouble, s_ulps[kDouble].Max(), 250.0f);
    ASSERT_LESS(kNaive, s_ulps[kNaive].Max(), 4E9f);
    ASSERT_LESS(kOnlyTwoProd, s_ulps[kOnlyTwoProd].Max(), 3E9f);
    ASSERT_LESS(kKahan, s_ulps[kKahan].Max(), 4E7f);
    ASSERT_LESS(kAddTwoProd, s_ulps[kAddTwoProd].Max(), 1E7f);
    ASSERT_LESS(kAddTwoSum, s_ulps[kAddTwoSum].Max(), 2.5E7f);
    ASSERT_LESS(kPairwise, s_ulps[kPairwise].Max(), 3.3E9f);
  }

  hwy::Stats s_cond;

  // Relative error
  float max_muls[kVariants];
  hwy::Stats s_muls[kVariants];

  hwy::Stats s_l1s[kVariants];  // Absolute error
  hwy::Stats s_rels[kVariants];  // forward relative
  hwy::Stats s_bwds[kVariants];  // = forward / condition number
  hwy::Stats s_bits[kVariants];  // = -log2(rel), capped to 23

  hwy::Stats s_ulps[kVariants];  // Only relevant for small cond
  hwy::Stats s_times[kVariants];
};

// Returns normalized value in [-1, 1).
float RandomFloat(std::mt19937& rng) {
  const uint32_t exp = hwy::BitCastScalar<uint32_t>(1.0f);
  const uint32_t mantissa_mask = hwy::MantissaMask<float>();
  const uint32_t representation = exp | (rng() & mantissa_mask);
  const float f12 = hwy::BitCastScalar<float>(representation);
  HWY_DASSERT(1.0f <= f12 && f12 < 2.0f);  // exponent is 2^0, only mantissa
  const float f = (2.0f * (f12 - 1.0f)) - 1.0f;
  HWY_DASSERT(-1.0f <= f && f < 1.0f);
  return f;
}

// `raw` holds the decompressed values, so that the test measures only the
// error from the Dot algorithms, not the compression.
template <typename Packed>
void GenerateWellConditionedInputs(const size_t num, float* HWY_RESTRICT raw,
                                   std::mt19937& rng,
                                   const PackedSpan<Packed>& packed,
                                   CompressWorkingSet& work) {
  std::uniform_int_distribution<int> e_dist(0, 6);

  for (size_t i = 0; i < num; ++i) {
    raw[i] = RandomFloat(rng) * (1 << e_dist(rng));
  }

  if (IsCompressed<Packed>()) {
    // Don't care about the original range.
    (void)ScaleWeights(raw, num);
  }

  hwy::ThreadPool pool(0);  // num is too small for parallelization
  const size_t packed_ofs = 0;
  Compress(raw, num, work, packed, packed_ofs, pool);

  const hn::ScalableTag<float> df;
  DecompressAndZeroPad(df, MakeConst(packed), packed_ofs, raw, num);
}

// Returns the actual condition number. Based on Algorithm 6.1 from "Accurate
// Sum and Dot Product". `num` is the (arbitrary) size of w, v, and buf.
template <typename WT, typename VT>
double GenerateIllConditionedInputs(const size_t num, WT* w, VT* HWY_RESTRICT v,
                                    std::mt19937& rng) {
  PROFILER_FUNC;
  const size_t half = HWY_MAX(1, num / 2);  // generate at least one random
  HWY_DASSERT(half != 0);

  const hn::ScalableTag<float> df;
  const PackedSpan<WT> w_span(w, num);

  // Regardless of WT and VT, we will accumulate into float. Multiplying
  // two maximal inputs and accumulating `num` times is enough for some loss of
  // precision and condition numbers between 1E6-1E9, which is what we see for
  // Attention Dot and `RMSNormMul`.
  const int max_exp = 5;
  std::uniform_int_distribution<int> e_dist(0, max_exp);

  // First half: random exponents and mantissas
  for (size_t i = 0; i < half; ++i) {
    // Ensure the min and max exponents are used.
    const int e = i == 0 ? 0 : i == 1 ? max_exp : e_dist(rng);
    w[i] = hwy::ConvertScalarTo<WT>(RandomFloat(rng) * (1 << e));
    v[i] = hwy::ConvertScalarTo<VT>(RandomFloat(rng) * (1 << e));
  }

  const float a_exp_step =
      num == half ? 0.0f : static_cast<float>(max_exp) / (num - half);
  float a_exp = max_exp;  // max_exp downto 0
  for (size_t i = half; i < num; ++i, a_exp -= a_exp_step) {
    const int e = static_cast<int>(a_exp);
    HWY_DASSERT(e >= 0);
    w[i] = hwy::ConvertScalarTo<WT>(RandomFloat(rng) * (1 << e));
    const float r = RandomFloat(rng) * (1 << e);
    if (hwy::ConvertScalarTo<float>(w[i]) == 0.0f) {
      v[i] = hwy::ConvertScalarTo<VT>(0.0f);
    } else {
      // This is called >100K times. DotCompensated is much faster than ExactDot
      // and just about as accurate.
      const float exact =
          DotCompensated(df, MakeConst(w_span), /*w_ofs=*/0, v, i);
      v[i] = hwy::ConvertScalarTo<VT>(
          r - exact / hwy::ConvertScalarTo<float>(w[i]));
    }
  }

  // Fisher-Yates shuffle of both a and b simultaneously - std::shuffle only
  // shuffles one array, and we want the same permutation for both.
  for (size_t i = num - 1; i != 0; --i) {
    std::uniform_int_distribution<size_t> dist(0, i);
    const size_t j = dist(rng);

    std::swap(w[i], w[j]);
    std::swap(v[i], v[j]);
  }

  return ConditionNumber(w, v, num);
}

// Runs all Dot algorithms for all short lengths and all Packed/raw types
// on well-conditioned inputs, and ensures the results are close to exact.
template <typename Packed>
struct TestShortDotsT {
  template <typename T, class D>
  HWY_INLINE void operator()(T /*unused*/, D d) {
    const size_t N = hn::Lanes(d);
    const hn::ScalableTag<float> df;  // for CallDot

    ThreadingArgs threading_args;
    ThreadingContext ctx(threading_args);
    CompressWorkingSet work;
    std::mt19937 rng;
    rng.seed(12345);

    hwy::Stats s_l1[kVariants];

    for (size_t num = 1; num <= 5 * N; ++num) {
      // GenerateWellConditionedInputs calls DecompressAndZeroPad to `raw*`,
      // hence they require padding to one vector.
      const size_t padded_num = hwy::RoundUpTo(num, N);
      MatStorageT<float> raw_w("raw_w", padded_num, ctx.allocator);
      MatStorageT<float> raw_v("raw_v", padded_num, ctx.allocator);
      MatStorageT<Packed> weights("weights", padded_num, ctx.allocator);
      const PackedSpan<Packed> w = weights.Span();
      MatStorageT<T> vectors("vectors", padded_num, ctx.allocator);
      const PackedSpan<T> v = vectors.Span();

      MatStorageT<double> bufs("bufs", padded_num, ctx.allocator);
      double* HWY_RESTRICT buf = bufs.Row(0);

      for (size_t rep = 0; rep < hn::AdjustedReps(20); ++rep) {
        GenerateWellConditionedInputs(num, raw_w.Row(0), rng, w, work);
        GenerateWellConditionedInputs(num, raw_v.Row(0), rng, v, work);

        const float dot_exact =
            ExactDot(raw_w.PackedScale1(), raw_v.PackedScale1(), num, buf);
        float dots[kVariants];
        for (size_t variant = 0; variant < kVariants; ++variant) {
          // Here Packed is not always float, so we must not call kDouble.
          const size_t actual = (variant == kDouble) ? kCompensated : variant;
          dots[variant] = CallDot(df, actual, MakeConst(w), 0, v.ptr, num);

          const float l1 = hwy::ScalarAbs(dots[variant] - dot_exact);
          s_l1[variant].Notify(l1);
        }
      }
    }

    // Avoid extra output for partial vectors.
    if (hn::detail::IsFull(d)) {
      for (size_t variant = 0; variant < kVariants; ++variant) {
        DotStats::PrintStats("l1", variant, s_l1[variant]);
      }
    }

    constexpr bool kCompressed = IsCompressed<Packed>();
    // Verify the dot products are plausible. This is only to verify
    // correctness, not to differentiate between the variants.
    double expected_l1[kVariants];
    // Tolerances are much lower for compressed inputs: the more limited set of
    // values seems to reduce roundoff.
    for (size_t variant = 0; variant < kVariants; ++variant) {
      expected_l1[variant] = kCompressed ? 1.5E-6 : 7E-5;
    }
    expected_l1[kNaive] = kCompressed ? 4E-6 : 2E-4;
    expected_l1[kPairwise] = kCompressed ? 4E-6 : 2E-4;

    for (size_t variant = 0; variant < kVariants; ++variant) {
      HWY_ASSERT(s_l1[variant].Min() >= 0.0f);
      ASSERT_LESS(variant, s_l1[variant].Max(), 1.5E-3f);
      ASSERT_LESS(variant, s_l1[variant].Mean(), expected_l1[variant]);
    }
  }
};

void TestAllShortDots() {
  // Skip EMU128 and old x86, include SSE4 because it tests the non-FMA path.
  if (HWY_TARGET == HWY_EMU128 || HWY_TARGET == HWY_SSSE3 ||
      HWY_TARGET == HWY_SSE2) {
    return;
  }

  ForeachPackedAndRawType<TestShortDotsT>();
}

// Excludes outliers; we might not have enough samples for a reliable mode.
double TrimmedMean(double* seconds, size_t num) {
  std::sort(seconds, seconds + num);
  double sum = 0;
  int count = 0;
  for (size_t i = num / 4; i < num / 2; ++i) {
    sum += seconds[i];
    count += 1;
  }
  return sum / count;
}

// Tests W=float, V=float for one large size and many reps on ill-conditioned
// inputs. Also includes benchmarking.
void TestAllDot() {
  // Skip EMU128 and old x86, include SSE4 because it tests the non-FMA path.
  if (HWY_TARGET == HWY_EMU128 || HWY_TARGET == HWY_SSSE3 ||
      HWY_TARGET == HWY_SSE2) {
    return;
  }

  constexpr size_t kMaxWorkers = 15;

  // Limit workers because we only support `kMaxWorkers`.
  ThreadingArgs threading_args;
  threading_args.max_packages = 1;
  threading_args.max_clusters = 1;
  threading_args.max_lps = kMaxWorkers - 1;
  ThreadingContext ctx(threading_args);

  {  // ensure no profiler zones are active
    const hn::ScalableTag<float> df;

    std::mt19937 rngs[kMaxWorkers];
    for (size_t i = 0; i < kMaxWorkers; ++i) {
      rngs[i].seed(12345 + 65537 * i);
    }

    constexpr size_t kReps = hn::AdjustedReps(40);
    const size_t num = 24 * 1024;
    MatStorageT<float> a("a", Extents2D(kMaxWorkers, num), ctx.allocator,
                         MatPadding::kOdd);
    MatStorageT<float> b("b", Extents2D(kMaxWorkers, num), ctx.allocator,
                         MatPadding::kOdd);
    MatStorageT<double> bufs("bufs", Extents2D(kMaxWorkers, num), ctx.allocator,
                             MatPadding::kOdd);
    std::array<DotStats, kMaxWorkers> all_stats;

    ctx.pools.Cluster(0, 0).Run(
        0, kReps, [&](const uint32_t rep, size_t thread) {
          float* HWY_RESTRICT pa = a.Row(thread);
          float* HWY_RESTRICT pb = b.Row(thread);
          double* HWY_RESTRICT buf = bufs.Row(thread);
          const PackedSpan<const float> a_span(pa, num);
          DotStats& stats = all_stats[thread];
          const double cond =
              GenerateIllConditionedInputs(num, pa, pb, rngs[thread]);

          const float dot_exact = ExactDot(pa, pb, num, buf);

          float dots[kVariants] = {};
          double times[kVariants] = {};
          for (size_t variant = 0; variant < kVariants; ++variant) {
            constexpr size_t kTimeReps = hn::AdjustedReps(10);
            std::array<double, kTimeReps> elapsed;
            for (size_t time_rep = 0; time_rep < kTimeReps; ++time_rep) {
              const double start = hwy::platform::Now();
              dots[variant] +=
                  CallDot(df, variant, a_span, /*w_ofs=*/0, pb, num);
              hwy::PreventElision(*pa);
              elapsed[time_rep] = hwy::platform::Now() - start;
            }
            dots[variant] /= kTimeReps;
            times[variant] = TrimmedMean(elapsed.data(), kTimeReps);
          }

          stats.NotifyTimes(times);
          stats.NotifyRep(num, cond, dot_exact, dots);
          stats.NotifyRatios();
        });

    DotStats& stats = all_stats[0];
    for (size_t i = 1; i < kMaxWorkers; ++i) {
      stats.Assimilate(all_stats[i]);
    }
    static bool once = true;
    if (once) {
      once = false;
      stats.Print();
    }
    stats.Check();
  }
  PROFILER_PRINT_RESULTS();
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gcpp {
HWY_BEFORE_TEST(DotTest);
HWY_EXPORT_AND_TEST_P(DotTest, TestAllShortDots);
HWY_EXPORT_AND_TEST_P(DotTest, TestAllDot);
HWY_AFTER_TEST();

}  // namespace gcpp

#endif
