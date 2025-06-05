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

// OrderedDemote2To is not supported by HWY_SCALAR.
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS HWY_SCALAR
#endif

#include "ops/ops.h"

#include <stddef.h>
#include <stdio.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include "gemma/activations.h"  // ChooseQueryScale
#include "util/allocator.h"
#include "util/basics.h"  // BF16
#include "util/mat.h"     // MatStorageT
#include "util/test_util.h"
#include "util/threading_context.h"
#include "hwy/tests/hwy_gtest.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ops/ops_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "ops/ops-inl.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <class Test>
struct ForeachCountAndMisalign {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) const {
    hwy::RandomState rng;
    const size_t N = Lanes(d);
    const size_t misalignments[3] = {0, N / 4, 3 * N / 5};

    for (size_t count = 0; count < 2 * N; ++count) {
      for (size_t ma : misalignments) {
        for (size_t mb : misalignments) {
          Test()(d, count, ma, mb, rng);
        }
      }
    }
  }
};

template <typename T>
T Random(hwy::RandomState& rng) {
  const int32_t bits = static_cast<int32_t>(Random32(&rng)) & 1023;
  const double val = (bits - 512) / 64.0;
  // Clamp negative to zero for unsigned types.
  return hwy::ConvertScalarTo<T>(
      HWY_MAX(hwy::ConvertScalarTo<double>(hwy::LowestValue<T>()), val));
}

HWY_NOINLINE void SourceAddFrom(const float* HWY_RESTRICT other,
                                float* HWY_RESTRICT x, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    x[i] += other[i];
  }
}

HWY_NOINLINE void SourceMulBy(const float* HWY_RESTRICT other,
                              float* HWY_RESTRICT x, size_t size,
                              size_t max_pos) {
  HWY_DASSERT(max_pos <= size);
  for (size_t i = 0; i < max_pos; ++i) {
    x[i] *= other[i];
  }
}

HWY_NOINLINE void SourceMulByConst(float c, float* HWY_RESTRICT x, size_t size,
                                   size_t max_pos) {
  for (size_t i = 0; i < max_pos; ++i) {
    x[i] *= c;
  }
}

HWY_NOINLINE void SourceMulByConstAndAdd(float c, const float* HWY_RESTRICT x,
                                         float* HWY_RESTRICT out, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    out[i] += x[i] * c;
  }
}

HWY_NOINLINE void SourceSoftmax(float* HWY_RESTRICT x, size_t size,
                                size_t mask_pos) {
  HWY_DASSERT(size != 0);
  HWY_DASSERT(mask_pos <= size);
  float sum = 0.0;
  const float maxval = *std::max_element(x, x + mask_pos);
  for (size_t i = 0; i < mask_pos; ++i) {
    x[i] = std::exp(x[i] - maxval);
    sum += x[i];
  }
  const float scale = 1.0f / sum;
  for (size_t i = 0; i < mask_pos; ++i) {
    x[i] *= scale;
  }
}

template <size_t k>
HWY_NOINLINE std::discrete_distribution<int> SourceCreateDistribution(
    std::array<float, k>& top_k, float temperature) {
  // re-normalize distribution
  for (size_t i = 0; i < k; ++i) {
    top_k[i] = exp(log(top_k[i]) / temperature);
  }
  float denominator = 0.0f;
  for (size_t i = 0; i < k; ++i) {
    denominator += top_k[i];
  }
  denominator = 1.0f / denominator;
  MulByConst(denominator, top_k.data(), k);
  return std::discrete_distribution<int>(std::begin(top_k), std::end(top_k));
}

struct TestAddFrom {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t misalign_b,
                  hwy::RandomState& rng) {
    using T = hn::TFromD<D>;

    hwy::AlignedFreeUniquePtr<T[]> px =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> pe =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> po =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_b + count));
    HWY_ASSERT(px && pe && po);

    T* x = px.get() + misalign_a;
    T* e = pe.get() + misalign_a;
    T* o = po.get() + misalign_b;

    for (size_t i = 0; i < count; ++i) {
      x[i] = Random<T>(rng);
      e[i] = x[i];
      o[i] = Random<T>(rng);
    }

    SourceAddFrom(o, e, count);
    AddFrom(o, x, count);

    hwy::AssertArraySimilar(e, x, count, hwy::TargetName(HWY_TARGET), __FILE__,
                            __LINE__);
  }
};

struct TestMulBy {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t misalign_b,
                  hwy::RandomState& rng) {
    using T = hn::TFromD<D>;

    hwy::AlignedFreeUniquePtr<T[]> px =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> pe =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> po =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_b + count));
    HWY_ASSERT(px && pe && po);

    T* x = px.get() + misalign_a;
    T* e = pe.get() + misalign_a;
    T* o = po.get() + misalign_b;

    for (size_t i = 0; i < count; ++i) {
      x[i] = Random<T>(rng);
      e[i] = x[i];
      o[i] = Random<T>(rng);
    }

    SourceMulBy(o, e, count, count);
    MulBy(o, x, count, count);

    hwy::AssertArraySimilar(e, x, count, hwy::TargetName(HWY_TARGET), __FILE__,
                            __LINE__);
  }
};

struct TestMulByConstAndAdd {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t misalign_b,
                  hwy::RandomState& rng) {
    using T = hn::TFromD<D>;

    hwy::AlignedFreeUniquePtr<T[]> px =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> pe =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> po =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_b + count));
    HWY_ASSERT(px && pe && po);

    T* x = px.get() + misalign_a;
    T* e = pe.get() + misalign_a;
    T* o = po.get() + misalign_b;

    for (size_t i = 0; i < count; ++i) {
      x[i] = Random<T>(rng);
      e[i] = x[i];
      o[i] = Random<T>(rng);
    }
    T constant = Random<T>(rng);

    SourceMulByConstAndAdd(constant, o, e, count);
    MulByConstAndAdd(constant, o, x, count);

    hwy::AssertArraySimilar(e, x, count, hwy::TargetName(HWY_TARGET), __FILE__,
                            __LINE__);
  }
};

struct TestMulByConst {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t misalign_b,
                  hwy::RandomState& rng) {
    if (misalign_b == 0) return;
    using T = hn::TFromD<D>;

    hwy::AlignedFreeUniquePtr<T[]> px =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> pe =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    HWY_ASSERT(px && pe);

    T* x = px.get() + misalign_a;
    T* e = pe.get() + misalign_a;

    for (size_t i = 0; i < count; ++i) {
      x[i] = Random<T>(rng);
      e[i] = x[i];
    }
    T constant = Random<T>(rng);

    SourceMulByConst(constant, e, count, count);
    MulByConst(constant, x, count, count);

    hwy::AssertArraySimilar(e, x, count, hwy::TargetName(HWY_TARGET), __FILE__,
                            __LINE__);
  }
};

struct TestSoftmax {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t misalign_b,
                  hwy::RandomState& rng) {
    if (count == 0) return;  // *Softmax would assert
    if (misalign_b == 0) return;
    using T = hn::TFromD<D>;

    hwy::AlignedFreeUniquePtr<T[]> px =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> pe =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    HWY_ASSERT(px && pe);

    T* x = px.get() + misalign_a;
    T* e = pe.get() + misalign_a;

    for (size_t i = 0; i < count; ++i) {
      x[i] = Random<T>(rng);
      e[i] = x[i];
    }

    SourceSoftmax(e, count, count);
    Softmax(x, count, count);

    T sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
      sum += x[i];
      double rel = std::abs(x[i] - e[i]) / e[i];
      ASSERT_LT(rel, 1e-6) << "Mismatch on coordinate " << i << " out of "
                           << count;
    }
    ASSERT_NEAR(sum, 1.0, 1e-6);
  }
};

template <size_t k>
struct TestCreateDistribution {
  void operator()(hwy::RandomState& rng) {
    std::array<float, k> x;
    std::array<float, k> e;

    for (size_t i = 0; i < k; ++i) {
      x[i] = Random<float>(rng);
      e[i] = x[i];
    }
    const float constant = Random<float>(rng);
    auto expected = SourceCreateDistribution(e, constant);
    auto output = create_distribution(x, constant);

    AssertEqual(expected, output, hwy::TargetName(HWY_TARGET), __FILE__,
                __LINE__);
  }
};

void TestAllAddFrom() {
  hn::ForPartialVectors<ForeachCountAndMisalign<TestAddFrom>>()(float());
}

void TestAllMulBy() {
  hn::ForPartialVectors<ForeachCountAndMisalign<TestMulBy>>()(float());
}

void TestAllMulByConst() {
  hn::ForPartialVectors<ForeachCountAndMisalign<TestMulByConst>>()(float());
}

void TestAllMulByConstAndAdd() {
  hn::ForPartialVectors<ForeachCountAndMisalign<TestMulByConstAndAdd>>()(
      float());
}

void TestAllSoftmax() {
  hn::ForPartialVectors<ForeachCountAndMisalign<TestSoftmax>>()(float());
}

void TestAllCreateDistribution() {
  TestCreateDistribution<2048>();
  TestCreateDistribution<5000>();
}

void TestSigmoid() {
  std::vector<float> values;
  for (int i = -150; i <= 150; ++i) {
    values.push_back(.1f * i);
  }
  std::vector<float> result = values;
  Sigmoid(result.data(), result.size());

  for (size_t i = 0; i < values.size(); i++) {
    const float max_error = 0.00007;
    float value = values[i];
    float approx = result[i];
    float expected = (1 / (1 + std::exp(-values[i])));
    EXPECT_NEAR(approx, expected, max_error) << "Input: " << value;
  }
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void ScalarRopeAndMulBy(
    const float mul, float* HWY_RESTRICT x, size_t dim_qkv,
    const float* HWY_RESTRICT inv_timescale, int pos) {
  HWY_DASSERT(dim_qkv % 2 == 0);
  const size_t half_dim_qkv = dim_qkv / 2;
  for (size_t dim = 0; dim < half_dim_qkv; ++dim) {
    const float theta = StaticCast<float>(pos) * inv_timescale[dim];
    const float cos_val = cosf(theta);
    const float sin_val = sinf(theta);
    const float x0 = x[dim];
    const float x1 = x[dim + half_dim_qkv];
    x[dim] = mul * (x0 * cos_val - x1 * sin_val);
    x[dim + half_dim_qkv] = mul * (x0 * sin_val + x1 * cos_val);
  }
}

void TestRopeAndMulBy() {
  ModelConfig config(Model::GEMMA2_9B, Type::kSFP,
                     ChooseWrapping(Model::GEMMA2_9B));
  int dim_qkv = config.layer_configs[0].qkv_dim;
  MatStorageT<float> x("x", dim_qkv);

  std::mt19937 gen;
  gen.seed(0x12345678);
  std::normal_distribution<float> r{0.0, 5.0};
  auto random_float = [&r, &gen] { return r(gen); };

  for (int i = 0; i < dim_qkv; ++i) {
    x.Row(0)[i] = random_float();
  }

  const float qmul = ChooseQueryScale(config);
  const float kmul = 1.0;

  MatStorageT<float> qexpected("qexpected", dim_qkv);
  MatStorageT<float> qactual("qactual", dim_qkv);
  MatStorageT<float> kexpected("kexpected", dim_qkv);
  MatStorageT<float> kactual("kactual", dim_qkv);
  MatStorageT<float> inv_timescale = CreateInvTimescale(
      config.layer_configs[0].qkv_dim,
      config.layer_configs[0].post_qk == PostQKType::HalfRope);
  // Assert VectorizedRope computation is same as regular rope at different pos.
  for (int pos = 1; pos < 500; pos++) {
    // Rope'd Q embeddings
    CopyMat(x, qactual);
    CopyMat(x, qexpected);
    ScalarRopeAndMulBy(qmul, qexpected.Row(0), dim_qkv, inv_timescale.Row(0),
                       pos);
    RopeAndMulBy(qmul, qactual.Row(0), dim_qkv, inv_timescale.Row(0), pos);

    for (int i = 0; i < dim_qkv; ++i) {
      EXPECT_NEAR(qactual.Row(0)[i], qexpected.Row(0)[i], 1e-4)
          << "qIndex:" << i << "qInput:" << qactual.Row(0)[i];
    }

    // Rope'd K embeddings
    CopyMat(x, kactual);
    CopyMat(x, kexpected);
    ScalarRopeAndMulBy(kmul, kexpected.Row(0), dim_qkv, inv_timescale.Row(0),
                       pos);
    RopeAndMulBy(kmul, kactual.Row(0), dim_qkv, inv_timescale.Row(0), pos);

    for (int i = 0; i < dim_qkv; ++i) {
      EXPECT_NEAR(kactual.Row(0)[i], kexpected.Row(0)[i], 1e-4)
          << "kIndex:" << i << "kInput:" << kactual.Row(0)[i];
    }
  }
}

template <typename T>
HWY_NOINLINE float ScalarSquaredL2(const T* HWY_RESTRICT a, size_t size) {
  double sum = 0.0;
  for (size_t i = 0; i < size; ++i) {
    const float f = hwy::ConvertScalarTo<float>(a[i]);
    sum += f * f;
  }
  return static_cast<float>(sum);
}

// Supports bf16 and f32 inputs/outputs, which can be in-place.
template <typename XT, typename WT, typename OT>
HWY_NOINLINE void ScalarRMSNorm(const XT* x, const WT* HWY_RESTRICT weight,
                                OT* out, size_t size) {
  constexpr float kEps = 1e-6f;
  float ss = ScalarSquaredL2(x, size);
  ss = 1.0f / sqrtf(ss / StaticCast<float>(size) + kEps);
  for (size_t j = 0; j < size; j++) {
    const float v = hwy::ConvertScalarTo<float>(x[j]);
    const float w = hwy::ConvertScalarTo<float>(weight[j]);
    // Note 1.0f centering here
    out[j] = hwy::ConvertScalarTo<OT>((1.0f + w) * (ss * v));
  }
}

template <typename XT, typename WT, typename OT>
void TestRMSNorm(hwy::RandomState& rng) {
  constexpr size_t kSize = 128;
  HWY_ALIGN XT vec[kSize];
  HWY_ALIGN WT weight[kSize];
  HWY_ALIGN OT expected[kSize];
  HWY_ALIGN OT actual[kSize];

  for (size_t i = 0; i < kSize; ++i) {
    vec[i] = hwy::ConvertScalarTo<XT>(RandomGaussian(rng));
    weight[i] = hwy::ConvertScalarTo<WT>(RandomGaussian(rng));
  }

  ScalarRMSNorm(vec, weight, expected, kSize);
  RMSNorm(vec, weight, 0, actual, kSize);

  for (size_t i = 0; i < kSize; i++) {
    const float e = hwy::ConvertScalarTo<float>(expected[i]);
    const float a = hwy::ConvertScalarTo<float>(actual[i]);
    if (!IsNear(e, a, 1e-5f)) {
      HWY_ABORT("RMSNorm %s %s %s mismatch at %zu: %E %E\n", TypeName<XT>(),
                TypeName<WT>(), TypeName<OT>(), i, e, a);
    }
  }
}

void TestAllRMSNorm() {
  hwy::RandomState rng;
  TestRMSNorm<float, float, float>(rng);
  TestRMSNorm<float, float, BF16>(rng);
  TestRMSNorm<float, BF16, float>(rng);
  TestRMSNorm<float, BF16, BF16>(rng);
  TestRMSNorm<BF16, float, float>(rng);
  TestRMSNorm<BF16, float, BF16>(rng);
  TestRMSNorm<BF16, BF16, float>(rng);
  TestRMSNorm<BF16, BF16, BF16>(rng);
}

void TestLayerNormSimple() {
  const size_t kSize = 52;
  std::vector<float> values(kSize);
  // Alternating 1.0/-1.0, so mean=0.0, var=1.0, rsqrt(var+epsilon)=0.9999995
  for (size_t i = 0; i < kSize; ++i) {
    values[i] = (i % 2 == 0) ? 1.0f : -1.0f;
  }
  std::vector<float> scale(kSize, 1.2f);
  std::vector<float> bias(kSize, 0.1f);
  std::vector<float> result(kSize);
  LayerNorm(values.data(), scale.data(), bias.data(), result.data(), kSize);

  for (size_t i = 0; i < kSize; i++) {
    const float max_error = 1e-6f;
    float value = values[i];
    float res = result[i];
    // out = (x - 0.0) * 1.2 * 0.9999995 + 0.1 = 1.2999994 / -1.0999994;
    float expected = (i % 2 == 0) ? 1.2999994f : -1.0999994f;
    EXPECT_NEAR(res, expected, max_error) << "Input: " << value;
  }
}

// Computes mean mu and mean of squares mu2 of a vector. Used in
// ScalarLayerNorm.
template <typename T>
HWY_NOINLINE void ScalarMus(const T* HWY_RESTRICT a, size_t size, double& mu,
                            double& mu2) {
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
template <typename XT, typename WT, typename OT>
HWY_NOINLINE void ScalarLayerNorm(const XT* x, const WT* HWY_RESTRICT scale,
                                  const WT* HWY_RESTRICT bias, OT* out,
                                  size_t size) {
  constexpr double kEps = 1e-6;
  double mu, mu2;
  ScalarMus(x, size, mu, mu2);
  double var = mu2 - mu * mu;
  constexpr double kZero = 0.0;
  var = HWY_MAX(var, kZero);
  var = 1.0 / sqrt(var + kEps);
  for (size_t j = 0; j < size; j++) {
    const float v = hwy::ConvertScalarTo<float>(x[j]);
    const float s = hwy::ConvertScalarTo<float>(scale[j]);
    const float b = hwy::ConvertScalarTo<float>(bias[j]);
    out[j] = hwy::ConvertScalarTo<OT>((v - mu) * s * var + b);
  }
}

template <typename XT, typename WT, typename OT>
void TestLayerNorm(hwy::RandomState& rng) {
  constexpr size_t kSize = 128;
  XT vec[kSize];
  WT weight[kSize];
  WT bias[kSize];
  OT expected[kSize];
  OT actual[kSize];

  for (size_t i = 0; i < kSize; ++i) {
    vec[i] = hwy::ConvertScalarTo<XT>(RandomGaussian(rng));
    weight[i] = hwy::ConvertScalarTo<WT>(RandomGaussian(rng));
    bias[i] = hwy::ConvertScalarTo<WT>(RandomGaussian(rng));
  }

  double expected_mu, expected_mu2;
  ScalarMus(vec, kSize, expected_mu, expected_mu2);
  double actual_mu, actual_mu2;
  ComputeMoments(vec, kSize, actual_mu, actual_mu2);

  ScalarLayerNorm(vec, weight, bias, expected, kSize);
  LayerNorm(vec, weight, bias, actual, kSize);

  for (size_t i = 0; i < kSize; i++) {
    const float e = hwy::ConvertScalarTo<float>(expected[i]);
    const float a = hwy::ConvertScalarTo<float>(actual[i]);
    if (!IsNear(e, a, 1e-5f)) {
      HWY_ABORT("LayerNorm %s %s %s mismatch at %zu: %E %E\n", TypeName<XT>(),
                TypeName<WT>(), TypeName<OT>(), i, e, a);
    }
  }
}

void TestAllLayerNorm() {
  hwy::RandomState rng;
  TestLayerNorm<float, float, float>(rng);
  TestLayerNorm<float, float, BF16>(rng);
  TestLayerNorm<float, BF16, float>(rng);
  TestLayerNorm<float, BF16, BF16>(rng);
}

void TestSampleTopK() {
  const size_t kSize = 52;
  std::vector<float> logits(kSize);
  // Create a vector going from -100 to -100+51=49 and take Softmax.
  std::iota(logits.begin(), logits.end(), -100.0f);
  Softmax(logits.data(), kSize);
  std::mt19937 gen;
  gen.seed(0x12345678);
  float temperature = 1.0f;
  // SampleTopK<1> should return the argmax.
  std::function<bool(int, float)> accept_token;
  int sample =
      SampleTopK(logits.data(), /*k=*/1, kSize, gen, temperature, accept_token);
  EXPECT_EQ(sample, 51);  // Last is largest.
  // Only accept even tokens, expect the last (largest) even index.
  accept_token = [](int i, float) { return i % 2 == 0; };
  sample =
      SampleTopK(logits.data(), /*k=*/1, kSize, gen, temperature, accept_token);
  EXPECT_EQ(sample, 50);  // Last even index.
  // Reset the logits to a positive, increasing sequence and take Softmax.
  std::iota(logits.begin(), logits.end(), 1.0f);
  Softmax(logits.data(), kSize);
  // Sample from the top 3, expect one of the top 3 even indices.
  for (int i = 0; i < 100; ++i) {
    sample = SampleTopK(logits.data(), /*k=*/3, kSize, gen, temperature,
                        accept_token);
    EXPECT_TRUE(sample == 50 || sample == 48 || sample == 46);
  }
  // Now set the temperature to 0.0f, which should always return the argmax,
  // even for k=3.
  temperature = 0.0f;
  for (int i = 0; i < 100; ++i) {
    sample = SampleTopK(logits.data(), /*k=*/3, kSize, gen, temperature,
                        accept_token);
    EXPECT_EQ(sample, 50);
  }
}

void TestPackTokenAndProb() {
  double packed1 = PackTokenAndProb(10, 0.96f);
  TokenAndProb unpacked1 = UnpackTokenAndProb(packed1);
  EXPECT_EQ(unpacked1.token, 10);
  EXPECT_NEAR(unpacked1.prob, 0.96f, 1e-6);

  double packed2 = PackTokenAndProb(1000000000, 0.87f);

  EXPECT_LT(packed2, packed1);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gcpp {
HWY_BEFORE_TEST(OpsTest);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllAddFrom);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllMulBy);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllMulByConst);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllMulByConstAndAdd);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllSoftmax);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllCreateDistribution);
HWY_EXPORT_AND_TEST_P(OpsTest, TestSigmoid);
HWY_EXPORT_AND_TEST_P(OpsTest, TestRopeAndMulBy);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllRMSNorm);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllLayerNorm);
HWY_EXPORT_AND_TEST_P(OpsTest, TestLayerNormSimple);
HWY_EXPORT_AND_TEST_P(OpsTest, TestSampleTopK);
HWY_EXPORT_AND_TEST_P(OpsTest, TestPackTokenAndProb);
HWY_AFTER_TEST();

}  // namespace gcpp

#endif
