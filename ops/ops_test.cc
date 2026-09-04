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
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>  // NOLINT(build/c++11)
#include <cmath>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include "gemma/activations.h"  // ChooseQueryScale
#include "gemma/configs.h"
#include "ops/ops.h"
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
#include "compression/test_util-inl.h"
#include "ops/fast_ops-inl.h"
#include "ops/ops-inl.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

static ThreadingContext& Ctx() {
  static ThreadingContext* ctx = new ThreadingContext(ThreadingArgs());
  return *ctx;
}

static RngStream MakeRng() {
  static AesCtrEngine engine(/*deterministic=*/true);
  static uint64_t stream = 0;
  return RngStream(engine, ++stream);
}

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

class TestAddFrom {
 public:
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

    SimpleAddFrom(o, e, count);
    AddFrom(o, x, count, Ctx(), /*worker=*/0);

    hwy::AssertArraySimilar(e, x, count, hwy::TargetName(HWY_TARGET), __FILE__,
                            __LINE__);
  }

 private:
  template <typename T1, typename T2>
  static HWY_NOINLINE void SimpleAddFrom(const T1* HWY_RESTRICT other,
                                         T2* HWY_RESTRICT x, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      x[i] = hwy::ConvertScalarTo<T2>(hwy::ConvertScalarTo<float>(x[i]) +
                                      hwy::ConvertScalarTo<float>(other[i]));
    }
  }
};

void TestAllAddFrom() {
  hn::ForPartialVectors<ForeachCountAndMisalign<TestAddFrom>>()(float());
}

class TestMulByConstAndAdd {
 public:
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

    SimpleMulByConstAndAdd(constant, o, e, count);
    MulByConstAndAdd(constant, o, x, count);

    hwy::AssertArraySimilar(e, x, count, hwy::TargetName(HWY_TARGET), __FILE__,
                            __LINE__);
  }

 private:
  template <typename T1, typename T2>
  static HWY_NOINLINE void SimpleMulByConstAndAdd(float c,
                                                  const T1* HWY_RESTRICT x,
                                                  T2* HWY_RESTRICT out,
                                                  size_t size) {
    for (size_t i = 0; i < size; ++i) {
      out[i] = hwy::ConvertScalarTo<T2>(hwy::ConvertScalarTo<float>(out[i]) +
                                        hwy::ConvertScalarTo<float>(x[i]) * c);
    }
  }
};

void TestAllMulByConstAndAdd() {
  hn::ForPartialVectors<ForeachCountAndMisalign<TestMulByConstAndAdd>>()(
      float());
}

class TestMulByConst {
 public:
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

    SimpleMulByConst(constant, e, count);
    MulByConst(constant, x, count);

    hwy::AssertArraySimilar(e, x, count, hwy::TargetName(HWY_TARGET), __FILE__,
                            __LINE__);
  }

 private:
  template <typename T1>
  static HWY_NOINLINE void SimpleMulByConst(float c, T1* HWY_RESTRICT x,
                                            size_t size) {
    for (size_t i = 0; i < size; ++i) {
      x[i] = hwy::ConvertScalarTo<T1>(hwy::ConvertScalarTo<float>(x[i]) * c);
    }
  }
};

void TestAllMulByConst() {
  hn::ForPartialVectors<ForeachCountAndMisalign<TestMulByConst>>()(float());
}

struct TestMulByConstTo {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t misalign_b,
                  hwy::RandomState& rng) {
    if (misalign_b == 0) return;
    using T = hn::TFromD<D>;

    hwy::AlignedFreeUniquePtr<T[]> px =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> pe =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> pactual =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    HWY_ASSERT(px && pe && pactual);

    T* x = px.get() + misalign_a;
    T* e = pe.get() + misalign_a;
    T* actual = pe.get() + misalign_a;

    T constant = Random<T>(rng);
    for (size_t i = 0; i < count; ++i) {
      x[i] = Random<T>(rng);
      e[i] = hwy::ConvertScalarTo<T>(hwy::ConvertScalarTo<float>(x[i]) *
                                     hwy::ConvertScalarTo<float>(constant));
    }

    MulByConstTo(constant, x, actual, count, Ctx(),
                 /*worker=*/0);

    hwy::AssertArraySimilar(e, actual, count, hwy::TargetName(HWY_TARGET),
                            __FILE__, __LINE__);
  }
};

void TestAllMulByConstTo() {
  hn::ForPartialVectors<ForeachCountAndMisalign<TestMulByConstTo>>()(float());
}

class TestSoftmax {
 public:
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

    for (const float temperature : {1.0f, 2.0f}) {
      for (size_t i = 0; i < count; ++i) {
        x[i] = Random<T>(rng);
        e[i] = x[i];
      }

      SimpleSoftmax(e, count, temperature);
      Softmax(Logits(x, count), Ctx(), /*worker=*/0, temperature);

      T sum = 0.0f;
      for (size_t i = 0; i < count; ++i) {
        sum += x[i];
        double rel = std::abs(x[i] - e[i]) / e[i];
        ASSERT_LT(rel, 2e-5) << "Mismatch on coordinate " << i << " out of "
                             << count << " at temperature " << temperature;
      }
      ASSERT_NEAR(sum, 1.0, 2e-5);
    }
  }

 private:
  static HWY_NOINLINE void SimpleSoftmax(float* HWY_RESTRICT x, size_t size,
                                         float temperature) {
    HWY_DASSERT(size != 0);
    float sum = 0.0;
    const float maxval = *std::max_element(x, x + size);
    for (size_t i = 0; i < size; ++i) {
      x[i] = std::exp((x[i] - maxval) / temperature);
      sum += x[i];
    }
    const float scale = 1.0f / sum;
    for (size_t i = 0; i < size; ++i) {
      x[i] *= scale;
    }
  }
};

void TestAllSoftmax() {
  hn::ForPartialVectors<ForeachCountAndMisalign<TestSoftmax>>()(float());
}

void TestSoftmaxTemperature() {
  constexpr size_t kNum = 4;
  const float kLogits[kNum] = {2.0f, 1.0f, 0.0f, -1.0f};

  for (const float temperature : {0.5f, 1.0f, 2.0f}) {
    float x[kNum];
    double expected[kNum];
    double sum = 0.0;
    for (size_t i = 0; i < kNum; ++i) {
      x[i] = kLogits[i];
      expected[i] = std::exp((kLogits[i] - kLogits[0]) / temperature);
      sum += expected[i];
    }
    Softmax(Logits(x, kNum), Ctx(), /*worker=*/0, temperature);
    for (size_t i = 0; i < kNum; ++i) {
      EXPECT_NEAR(x[i], expected[i] / sum, 1e-5)
          << "Mismatch on coordinate " << i << " at temperature "
          << temperature;
    }
  }

  // Zero temperature puts all the mass on the max.
  float zero[kNum];
  std::copy(kLogits, kLogits + kNum, zero);
  Softmax(Logits(zero, kNum), Ctx(), /*worker=*/0, /*temperature=*/0.0f);
  EXPECT_FLOAT_EQ(zero[0], 1.0f);
  for (size_t i = 1; i < kNum; ++i) {
    EXPECT_FLOAT_EQ(zero[i], 0.0f) << "Mismatch on coordinate " << i;
  }

  // Ties at zero temperature share the mass evenly.
  float ties[kNum] = {2.0f, 2.0f, 0.0f, -1.0f};
  Softmax(Logits(ties, kNum), Ctx(), /*worker=*/0, /*temperature=*/0.0f);
  EXPECT_FLOAT_EQ(ties[0], 0.5f);
  EXPECT_FLOAT_EQ(ties[1], 0.5f);
  EXPECT_FLOAT_EQ(ties[2], 0.0f);
  EXPECT_FLOAT_EQ(ties[3], 0.0f);
}

class TestSoftmaxState {
 public:
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
    T* initial_logits = pe.get() + misalign_a;

    for (size_t i = 0; i < count; ++i) {
      x[i] = Random<T>(rng);
      initial_logits[i] = x[i];
    }

    float softmax_max;
    float softmax_d;
    Softmax(Logits(x, count), Ctx(), /*worker=*/0, /*temperature=*/1.0f,
            {.max_out = &softmax_max, .d_out = &softmax_d});

    const float maxval =
        *std::max_element(initial_logits, initial_logits + count);

    float sum_exp = 0.0f;
    for (size_t i = 0; i < count; ++i) {
      sum_exp += std::exp(initial_logits[i] - maxval);
    }

    ASSERT_NEAR(softmax_max, maxval, 1e-6);
    ASSERT_NEAR(softmax_d, sum_exp, 2e-5);
  }
};

void TestAllSoftmaxState() {
  hn::ForPartialVectors<ForeachCountAndMisalign<TestSoftmaxState>>()(float());
}

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

void TestAllCreateDistribution() {
  TestCreateDistribution<2048>();
  TestCreateDistribution<5000>();
}

struct TestSigmoid {
  template <typename T, class D>
  void operator()(T, D) const {
    std::vector<T> values;
    for (int i = -150; i <= 150; ++i) {
      values.push_back(hwy::ConvertScalarTo<T>(.1f * i));
    }
    std::vector<T> result = values;
    Sigmoid(result.data(), result.size());

    for (size_t i = 0; i < values.size(); i++) {
      const float max_error = IsBF16<T>() ? 0.2f : 0.00007f;
      const float value = hwy::ConvertScalarTo<float>(values[i]);
      const float actual = hwy::ConvertScalarTo<float>(result[i]);
      const float expected = (1 / (1 + std::exp(-value)));
      EXPECT_NEAR(expected, actual, max_error)
          << (IsBF16<T>() ? "bf16" : "float");
    }
  }
};

static HWY_NOINLINE void TestAllSigmoid() {
  ForeachActivationType1<TestSigmoid>(hn::ScalableTag<float>());
}

struct TestFastSigmoid {
  template <typename T, class D>
  void operator()(T, D) const {
    std::vector<T> values;
    for (int i = -150; i <= 150; ++i) {
      values.push_back(hwy::ConvertScalarTo<T>(.1f * i));
    }
    std::vector<T> result = values;
    gcpp::HWY_NAMESPACE::FastSigmoid(result.data(), result.size());

    for (size_t i = 0; i < values.size(); i++) {
      const float max_error = IsBF16<T>() ? 0.003f : 0.0004f;
      const float value = hwy::ConvertScalarTo<float>(values[i]);
      const float actual = hwy::ConvertScalarTo<float>(result[i]);
      const float expected = (1 / (1 + std::exp(-value)));
      EXPECT_NEAR(expected, actual, max_error)
          << (IsBF16<T>() ? "bf16" : "float");
    }
  }
};

static HWY_NOINLINE void TestAllFastSigmoid() {
  ForeachActivationType1<TestFastSigmoid>(hn::ScalableTag<float>());
}

struct TestGelu {
  template <typename T, class D>
  void operator()(T, D) const {
    std::vector<T> values;
    for (int i = -150; i <= 150; ++i) {
      values.push_back(hwy::ConvertScalarTo<T>(.1f * i));
    }
    std::vector<T> result = values;
    Gelu(result.data(), result.size());

    for (size_t i = 0; i < values.size(); i++) {
      const float max_error = IsBF16<T>() ? 0.2f : 0.00007f;
      const float x = hwy::ConvertScalarTo<float>(values[i]);
      const float actual = hwy::ConvertScalarTo<float>(result[i]);
      const float expected =
          x * (0.5f + 0.5f * tanh(x * (0.79788f + 0.035677f * x * x)));
      EXPECT_NEAR(expected, actual, max_error)
          << (IsBF16<T>() ? "bf16" : "float");
    }
  }
};

static HWY_NOINLINE void TestAllGelu() {
  ForeachActivationType1<TestGelu>(hn::ScalableTag<float>());
}

struct TestFastGelu {
  template <typename T, class D>
  void operator()(T, D) const {
    std::vector<T> values;
    for (int i = -150; i <= 150; ++i) {
      values.push_back(hwy::ConvertScalarTo<T>(.1f * i));
    }
    std::vector<T> result = values;
    gcpp::HWY_NAMESPACE::FastGelu(result.data(), result.size());

    for (size_t i = 0; i < values.size(); i++) {
      const float max_error = IsBF16<T>() ? 0.007f : 1e-5f;
      const float x = hwy::ConvertScalarTo<float>(values[i]);
      const float actual = hwy::ConvertScalarTo<float>(result[i]);
      const float expected =
          x * (0.5f + 0.5f * tanh(x * (0.79788f + 0.035677f * x * x)));
      EXPECT_NEAR(expected, actual, max_error)
          << (IsBF16<T>() ? "bf16" : "float");
    }
  }
};

static HWY_NOINLINE void TestAllFastGelu() {
  ForeachActivationType1<TestFastGelu>(hn::ScalableTag<float>());
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void ScalarRopeAndMulBy(
    const float mul, float* HWY_RESTRICT x, const size_t dim_qkv,
    const float* HWY_RESTRICT inv_timescale, const int pos) {
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
  ThreadingContext& ctx = Ctx();
  const size_t worker = 0;

  const ModelConfig config(Model::GEMMA2_9B, Type::kSFP,
                           ChooseWrapping(Model::GEMMA2_9B));
  const size_t dim_qkv = config.layer_configs[0].qkv_dim;
  MatStorageT<float> x("x", dim_qkv, ctx.allocator);

  RngStream rng = MakeRng();
  std::normal_distribution<float> r{0.0, 5.0};
  auto random_float = [&r, &rng] { return r(rng); };

  for (size_t i = 0; i < dim_qkv; ++i) {
    x.Row(0)[i] = random_float();
  }

  const float qmul = ChooseQueryScale(config);
  constexpr float kmul = 1.0f;

  MatStorageT<float> qexpected("qexpected", dim_qkv, ctx.allocator);
  MatStorageT<float> qactual("qactual", dim_qkv, ctx.allocator);
  MatStorageT<float> kexpected("kexpected", dim_qkv, ctx.allocator);
  MatStorageT<float> kactual("kactual", dim_qkv, ctx.allocator);
  MatStorageT<float> kactual2("kactual2", dim_qkv, ctx.allocator);
  MatStorageT<float> inv_timescale = CreateInvTimescale(
      ctx.allocator, config.layer_configs[0].qkv_dim,
      config.layer_configs[0].post_qk == PostQKType::HalfRope,
      config.rope_theta);
  // Assert VectorizedRope computation is same as regular rope at different pos.
  for (size_t pos = 1; pos < 500; pos++) {
    // Rope'd Q embeddings with query scale
    CopyMat(x, qexpected);
    CopyMat(x, qactual);
    ScalarRopeAndMulBy(qmul, qexpected.Row(0), dim_qkv, inv_timescale.Row(0),
                       pos);
    RopeAndMulBy(qmul, qactual.Row(0), dim_qkv, inv_timescale.Row(0), pos, ctx,
                 worker);
    for (size_t i = 0; i < dim_qkv; ++i) {
      EXPECT_NEAR(qexpected.Row(0)[i], qactual.Row(0)[i], 1e-4) << " " << i;
    }

    // Same without query scale
    CopyMat(x, qexpected);
    CopyMat(x, qactual);
    ScalarRopeAndMulBy(1.0f, qexpected.Row(0), dim_qkv, inv_timescale.Row(0),
                       pos);
    Rope(qactual.Row(0), dim_qkv, inv_timescale.Row(0), pos, ctx, worker);
    for (size_t i = 0; i < dim_qkv; ++i) {
      EXPECT_NEAR(qexpected.Row(0)[i], qactual.Row(0)[i], 1e-4) << " " << i;
    }

    // Rope'd K embeddings
    CopyMat(x, kexpected);
    CopyMat(x, kactual);
    CopyMat(x, kactual2);
    ScalarRopeAndMulBy(kmul, kexpected.Row(0), dim_qkv, inv_timescale.Row(0),
                       pos);
    RopeAndMulBy(kmul, kactual.Row(0), dim_qkv, inv_timescale.Row(0), pos, ctx,
                 worker);
    static_assert(kmul == 1.0f, "");
    Rope(kactual2.Row(0), dim_qkv, inv_timescale.Row(0), pos, ctx, worker);

    for (size_t i = 0; i < dim_qkv; ++i) {
      EXPECT_NEAR(kexpected.Row(0)[i], kactual.Row(0)[i], 1e-4) << " " << i;
    }
    for (size_t i = 0; i < dim_qkv; ++i) {
      EXPECT_NEAR(kexpected.Row(0)[i], kactual2.Row(0)[i], 1e-4) << " " << i;
    }
  }
}

template <typename T>
static HWY_NOINLINE float ScalarSquaredL2(const T* HWY_RESTRICT a,
                                          size_t size) {
  double sum = 0.0;
  for (size_t i = 0; i < size; ++i) {
    const float f = hwy::ConvertScalarTo<float>(a[i]);
    sum += f * f;
  }
  return static_cast<float>(sum);
}

// Supports bf16 and f32 inputs/outputs, which can be in-place.
// Shared between TestRMSNorm and TestRMSNormInplace.
template <typename XT, typename WT, typename OT>
static HWY_NOINLINE void ScalarRMSNorm(const XT* x,
                                       const WT* HWY_RESTRICT weight, OT* out,
                                       size_t size) {
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

struct TestRMSNorm {
  template <typename XT, typename WT, typename OT, class D>
  void operator()(XT, WT, OT, D) const {
    hwy::RandomState rng;

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
    RMSNorm(vec, weight, /*w_ofs=*/0, actual, kSize, Ctx(),
            /*worker=*/0);

    for (size_t i = 0; i < kSize; i++) {
      const float e = hwy::ConvertScalarTo<float>(expected[i]);
      const float a = hwy::ConvertScalarTo<float>(actual[i]);
      if (!IsNear(e, a, 1e-5f)) {
        HWY_ABORT("RMSNorm %s %s %s mismatch at %zu: %E %E\n", TypeName<XT>(),
                  TypeName<WT>(), TypeName<OT>(), i, e, a);
      }
    }
  }
};

void TestAllRMSNorm() {
  ForeachActivationType3<TestRMSNorm>(hn::ScalableTag<float>());
}

struct TestRMSNormInplace {
  template <typename XT, typename WT, class D>
  void operator()(XT, WT, D) const {
    hwy::RandomState rng;

    constexpr size_t kSize = 128;
    HWY_ALIGN XT expected[kSize];
    HWY_ALIGN XT actual[kSize];
    HWY_ALIGN WT weight[kSize];

    for (size_t i = 0; i < kSize; ++i) {
      expected[i] = hwy::ConvertScalarTo<XT>(RandomGaussian(rng));
      actual[i] = expected[i];
      weight[i] = hwy::ConvertScalarTo<WT>(RandomGaussian(rng));
    }

    ScalarRMSNorm(expected, weight, expected, kSize);
    RMSNormInplace(weight, /*w_ofs=*/0, actual, kSize, Ctx(),
                   /*worker=*/0);

    for (size_t i = 0; i < kSize; i++) {
      const float e = hwy::ConvertScalarTo<float>(expected[i]);
      const float a = hwy::ConvertScalarTo<float>(actual[i]);
      if (!IsNear(e, a, 1e-5f)) {
        HWY_ABORT("RMSNormInplace %s %s mismatch at %zu: %E %E\n",
                  TypeName<XT>(), TypeName<WT>(), i, e, a);
      }
    }
  }
};

void TestAllRMSNormInplace() {
  ForeachActivationType2<TestRMSNormInplace>(hn::ScalableTag<float>());
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
    float res = result[i];
    // out = (x - 0.0) * 1.2 * 0.9999995 + 0.1 = 1.2999994 / -1.0999994;
    float expected = (i % 2 == 0) ? 1.2999994f : -1.0999994f;
    EXPECT_NEAR(res, expected, max_error);
  }
}

class TestLayerNorm {
 public:
  template <typename XT, typename WT, typename OT, class D>
  void operator()(XT, WT, OT, D) const {
    hwy::RandomState rng;
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

 private:
  // Computes mean mu and mean of squares mu2 of a vector. Used in
  // ScalarLayerNorm.
  template <typename T>
  static HWY_NOINLINE void ScalarMus(const T* HWY_RESTRICT a, size_t size,
                                     double& mu, double& mu2) {
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
  static HWY_NOINLINE void ScalarLayerNorm(const XT* x,
                                           const WT* HWY_RESTRICT scale,
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
};

void TestAllLayerNorm() {
  ForeachActivationType3<TestLayerNorm>(hn::ScalableTag<float>());
}

void TestSampleTopK() {
  ThreadingContext& ctx = Ctx();
  const size_t worker = 0;
  const size_t kSize = 52;
  std::vector<float> logits_vec(kSize);
  Logits logits(logits_vec.data(), kSize);
  // Create a vector going from -100 to -100+51=49 and take Softmax.
  std::iota(logits.begin(), logits.end(), -100.0f);
  Softmax(logits, ctx, worker);
  RngStream rng = MakeRng();
  float temperature = 1.0f;
  // SampleTopK<1> should return the argmax.
  std::function<bool(int, float)> accept_token;
  int sample = SampleTopK(logits, /*k=*/1, rng, temperature, accept_token);
  EXPECT_EQ(sample, 51);  // Last is largest.
  // Only accept even tokens, expect the last (largest) even index.
  accept_token = [](int i, float) { return i % 2 == 0; };
  sample = SampleTopK(logits, /*k=*/1, rng, temperature, accept_token);
  EXPECT_EQ(sample, 50);  // Last even index.
  // Reset the logits to a positive, increasing sequence and take Softmax.
  std::iota(logits.begin(), logits.end(), 1.0f);
  Softmax(logits, ctx, worker);
  // Sample from the top 3, expect one of the top 3 even indices.
  for (int i = 0; i < 100; ++i) {
    sample = SampleTopK(logits, /*k=*/3, rng, temperature, accept_token);
    EXPECT_TRUE(sample == 50 || sample == 48 || sample == 46);
  }
  // Now set the temperature to 0.0f, which should always return the argmax,
  // even for k=3.
  temperature = 0.0f;
  for (int i = 0; i < 100; ++i) {
    sample = SampleTopK(logits, /*k=*/3, rng, temperature, accept_token);
    EXPECT_EQ(sample, 50);
  }
}

// `TopK` must return the k largest logits in descending order, with and
// without an `accept_token` filter. `kSize` exceeds the initial capacity of the
// vector that `TopK` fills, so this also covers the reserved-capacity path.
void TestTopK() {
  const size_t kSize = 300;
  std::vector<float> logits_vec(kSize);
  std::iota(logits_vec.begin(), logits_vec.end(), 0.0f);
  Logits logits(logits_vec.data(), kSize);

  // Without a filter, the top-k are the last k indices, largest first.
  std::function<bool(int, float)> accept_token;
  std::vector<TokenAndProb> top = TopK(logits, /*k=*/5, accept_token);
  ASSERT_EQ(top.size(), size_t{5});
  for (size_t i = 0; i < top.size(); ++i) {
    const size_t expected = kSize - 1 - i;
    EXPECT_EQ(top[i].token, static_cast<int>(expected));
    EXPECT_EQ(top[i].prob, logits[expected]);
  }

  // With a filter, only even tokens are eligible.
  accept_token = [](int i, float) { return i % 2 == 0; };
  top = TopK(logits, /*k=*/5, accept_token);
  ASSERT_EQ(top.size(), size_t{5});
  for (size_t i = 0; i < top.size(); ++i) {
    const size_t expected = kSize - 2 - 2 * i;
    EXPECT_EQ(top[i].token, static_cast<int>(expected));
    EXPECT_EQ(top[i].prob, logits[expected]);
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

void TestApplyLogitMaskAllAllowed() {
  const std::vector<size_t> vocab_sizes = {
      1,   3,   7,   8,    16,   63,     64,     65,     127,    128,   129,
      255, 256, 257, 1000, 1001, 129280, 129281, 256000, 256001, 256063};

  for (size_t vocab_size : vocab_sizes) {
    const size_t kPad = 128;
    std::vector<float> buffer(vocab_size + 2 * kPad, 777.0f);
    float* logits_ptr = buffer.data() + kPad;

    for (size_t i = 0; i < vocab_size; ++i) {
      logits_ptr[i] = static_cast<float>(i) * 0.05f - 10.0f;
    }
    const std::vector<float> original(logits_ptr, logits_ptr + vocab_size);

    const size_t num_words = (vocab_size + 63) / 64;
    std::vector<uint64_t> mask(num_words, ~0ULL);

    ApplyLogitMaskKernel(hwy::Span<float>(logits_ptr, vocab_size), mask.data(),
                         vocab_size);

    // Verify all allowed tokens remain unchanged (delta == 0.0f).
    for (size_t i = 0; i < vocab_size; ++i) {
      EXPECT_FLOAT_EQ(logits_ptr[i], original[i]);
    }
    // Verify padding canaries before and after are untouched.
    for (size_t i = 0; i < kPad; ++i) {
      EXPECT_FLOAT_EQ(buffer[i], 777.0f);
      EXPECT_FLOAT_EQ(buffer[kPad + vocab_size + i], 777.0f);
    }
  }
}

void TestApplyLogitMaskAllDisallowed() {
  const std::vector<size_t> vocab_sizes = {64,   65,     128,    129,   1000,
                                           1001, 129280, 256000, 256001};

  for (size_t vocab_size : vocab_sizes) {
    const size_t kPad = 128;
    std::vector<float> buffer(vocab_size + 2 * kPad, 777.0f);
    float* logits_ptr = buffer.data() + kPad;

    for (size_t i = 0; i < vocab_size; ++i) {
      logits_ptr[i] = static_cast<float>(i) * 0.1f + 1.0f;
    }
    const std::vector<float> original(logits_ptr, logits_ptr + vocab_size);

    const size_t num_words = (vocab_size + 63) / 64;
    // Word 0 is all-disallowed (0ULL), word 1 has allowed token, remaining
    // words 0ULL.
    std::vector<uint64_t> mask(num_words, 0ULL);
    if (num_words > 1) {
      mask[1] = 1ULL;  // Token 64 allowed, words 0 and 2..end take Fast Path 2.
    } else {
      mask[0] = (1ULL << (vocab_size - 1));  // Only last token allowed.
    }

    ApplyLogitMaskKernel(hwy::Span<float>(logits_ptr, vocab_size), mask.data(),
                         vocab_size);

    for (size_t i = 0; i < vocab_size; ++i) {
      const bool is_allowed = (mask[i / 64] & (1ULL << (i % 64))) != 0;
      if (is_allowed) {
        EXPECT_FLOAT_EQ(logits_ptr[i], original[i]);
      } else {
        EXPECT_TRUE(std::isinf(logits_ptr[i]) && logits_ptr[i] < 0.0f);
      }
    }
    // Verify canaries untouched.
    for (size_t i = 0; i < kPad; ++i) {
      EXPECT_FLOAT_EQ(buffer[i], 777.0f);
      EXPECT_FLOAT_EQ(buffer[kPad + vocab_size + i], 777.0f);
    }
  }
}

void TestApplyLogitMaskEmptyMaskGuardrail() {
  const std::vector<size_t> vocab_sizes = {1,   63,   64,     65,
                                           128, 1000, 129280, 256000};

  for (size_t vocab_size : vocab_sizes) {
    std::vector<float> logits(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
      logits[i] = static_cast<float>(i) * 0.2f - 3.0f;
    }
    const std::vector<float> original = logits;

    const size_t num_words = (vocab_size + 63) / 64;
    std::vector<uint64_t> empty_mask(num_words, 0ULL);

    // Empty mask must NOT zero or -inf out logits; must preserve original
    // finite values.
    ApplyLogitMaskKernel(hwy::Span<float>(logits.data(), vocab_size),
                         empty_mask.data(), vocab_size);

    for (size_t i = 0; i < vocab_size; ++i) {
      EXPECT_FLOAT_EQ(logits[i], original[i]);
    }
  }

  // Null/zero guardrails
  std::vector<float> dummy(10, 1.0f);
  ApplyLogitMaskKernel(hwy::Span<float>(), nullptr, 0);
  ApplyLogitMaskKernel(hwy::Span<float>(dummy.data(), dummy.size()), nullptr,
                       10);
  EXPECT_FLOAT_EQ(dummy[0], 1.0f);
}

void TestApplyLogitMaskMixed() {
  const std::vector<size_t> vocab_sizes = {64,  65,   127,  128,    129,
                                           256, 1000, 1001, 129280, 256000};

  for (size_t vocab_size : vocab_sizes) {
    const size_t num_words = (vocab_size + 63) / 64;
    std::vector<float> logits(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
      logits[i] = static_cast<float>(i) * 0.01f;
    }
    const std::vector<float> original = logits;

    // Pattern 1: Alternating even/odd bits
    std::vector<uint64_t> alt_mask(num_words, 0xAAAAAAAAAAAAAAAAULL);
    ApplyLogitMaskKernel(hwy::Span<float>(logits.data(), vocab_size),
                         alt_mask.data(), vocab_size);

    for (size_t i = 0; i < vocab_size; ++i) {
      const bool is_allowed = (alt_mask[i / 64] & (1ULL << (i % 64))) != 0;
      if (is_allowed) {
        EXPECT_FLOAT_EQ(logits[i], original[i]);
      } else {
        EXPECT_TRUE(std::isinf(logits[i]) && logits[i] < 0.0f);
      }
    }

    // Pattern 2: Custom mask value (-1000.0f) with inverted alternating bits
    std::vector<float> logits_custom = original;
    std::vector<uint64_t> alt_mask2(num_words, 0x5555555555555555ULL);
    ApplyLogitMaskKernel(hwy::Span<float>(logits_custom.data(), vocab_size),
                         alt_mask2.data(), vocab_size, /*mask_value=*/-1000.0f);

    for (size_t i = 0; i < vocab_size; ++i) {
      const bool is_allowed = (alt_mask2[i / 64] & (1ULL << (i % 64))) != 0;
      if (is_allowed) {
        EXPECT_FLOAT_EQ(logits_custom[i], original[i]);
      } else {
        EXPECT_FLOAT_EQ(logits_custom[i], -1000.0f);
      }
    }
  }
}

void TestApplyLogitMaskUnalignedVocabSizes() {
  // Test irregular boundaries and odd non-multiples of 64
  const std::vector<size_t> vocab_sizes = {
      1,   2,   3,    5,    7,      9,      15,     17,     31,
      33,  63,  65,   77,   99,     127,    129,    255,    257,
      500, 501, 1023, 1025, 129279, 129281, 255999, 256001, 256063};

  std::mt19937_64 rng(42);

  for (size_t vocab_size : vocab_sizes) {
    const size_t kPad = 64;
    std::vector<float> buffer(vocab_size + 2 * kPad, 9999.0f);
    float* logits_ptr = buffer.data() + kPad;

    for (size_t i = 0; i < vocab_size; ++i) {
      logits_ptr[i] = static_cast<float>(i % 100);
    }
    const std::vector<float> original(logits_ptr, logits_ptr + vocab_size);

    const size_t num_words = (vocab_size + 63) / 64;
    std::vector<uint64_t> mask(num_words);
    for (size_t w = 0; w < num_words; ++w) {
      mask[w] = rng();
    }
    // Ensure at least one token is allowed so guardrail doesn't trigger
    mask[0] |= 1ULL;

    ApplyLogitMaskKernel(hwy::Span<float>(logits_ptr, vocab_size), mask.data(),
                         vocab_size);

    for (size_t i = 0; i < vocab_size; ++i) {
      const bool is_allowed = (mask[i / 64] & (1ULL << (i % 64))) != 0;
      if (is_allowed) {
        EXPECT_FLOAT_EQ(logits_ptr[i], original[i]);
      } else {
        EXPECT_TRUE(std::isinf(logits_ptr[i]) && logits_ptr[i] < 0.0f);
      }
    }

    // Verify bounds safety: no overrun before or after logits span
    for (size_t i = 0; i < kPad; ++i) {
      EXPECT_FLOAT_EQ(buffer[i], 9999.0f);
      EXPECT_FLOAT_EQ(buffer[kPad + vocab_size + i], 9999.0f);
    }
  }
}

void TestApplyLogitMaskUnalignedPointers() {
  const size_t vocab_size = 500;
  const size_t num_words = (vocab_size + 63) / 64;
  std::mt19937_64 rng(12345);

  std::vector<uint64_t> mask(num_words);
  for (size_t w = 0; w < num_words; ++w) {
    mask[w] = rng();
  }
  mask[0] |= 1ULL;

  // Test various pointer offsets from unaligned base
  for (size_t misalign = 0; misalign < 16; ++misalign) {
    std::vector<float> buffer(vocab_size + misalign + 32, 555.0f);
    float* logits_ptr = buffer.data() + misalign;

    for (size_t i = 0; i < vocab_size; ++i) {
      logits_ptr[i] = static_cast<float>(i + 1);
    }
    const std::vector<float> original(logits_ptr, logits_ptr + vocab_size);

    ApplyLogitMaskKernel(hwy::Span<float>(logits_ptr, vocab_size), mask.data(),
                         vocab_size);

    for (size_t i = 0; i < vocab_size; ++i) {
      const bool is_allowed = (mask[i / 64] & (1ULL << (i % 64))) != 0;
      if (is_allowed) {
        EXPECT_FLOAT_EQ(logits_ptr[i], original[i]);
      } else {
        EXPECT_TRUE(std::isinf(logits_ptr[i]) && logits_ptr[i] < 0.0f);
      }
    }
  }
}

void TestApplyLogitMaskSingleBitSet() {
  const std::vector<size_t> vocab_sizes = {1,   63,  64,   65,    127,
                                           128, 129, 1000, 256000};
  const std::vector<size_t> test_indices = {
      0, 1, 2, 31, 32, 63, 64, 65, 127, 128, 129, 999, 1000, 255998, 255999};

  for (size_t vocab_size : vocab_sizes) {
    const size_t num_words = (vocab_size + 63) / 64;
    const size_t kPad = 128;

    for (size_t target_idx : test_indices) {
      if (target_idx >= vocab_size) continue;

      std::vector<float> buffer(vocab_size + 2 * kPad, 777.0f);
      float* logits_ptr = buffer.data() + kPad;
      for (size_t i = 0; i < vocab_size; ++i) {
        logits_ptr[i] = static_cast<float>(i) * 0.01f + 1.0f;
      }
      const std::vector<float> original(logits_ptr, logits_ptr + vocab_size);

      std::vector<uint64_t> mask(num_words, 0ULL);
      mask[target_idx / 64] |= (1ULL << (target_idx % 64));

      ApplyLogitMaskKernel(hwy::Span<float>(logits_ptr, vocab_size),
                           mask.data(), vocab_size);

      for (size_t i = 0; i < vocab_size; ++i) {
        if (i == target_idx) {
          EXPECT_FLOAT_EQ(logits_ptr[i], original[i]);
        } else {
          EXPECT_TRUE(std::isinf(logits_ptr[i]) && logits_ptr[i] < 0.0f);
        }
      }

      for (size_t i = 0; i < kPad; ++i) {
        EXPECT_FLOAT_EQ(buffer[i], 777.0f);
        EXPECT_FLOAT_EQ(buffer[kPad + vocab_size + i], 777.0f);
      }
    }
  }
}

void TestApplyLogitMaskSingleBitUnset() {
  const std::vector<size_t> vocab_sizes = {1,   63,  64,   65,    127,
                                           128, 129, 1000, 256000};
  const std::vector<size_t> test_indices = {
      0, 1, 2, 31, 32, 63, 64, 65, 127, 128, 129, 999, 1000, 255998, 255999};

  for (size_t vocab_size : vocab_sizes) {
    const size_t num_words = (vocab_size + 63) / 64;
    const size_t kPad = 128;

    for (size_t target_idx : test_indices) {
      if (target_idx >= vocab_size) continue;

      std::vector<float> buffer(vocab_size + 2 * kPad, 777.0f);
      float* logits_ptr = buffer.data() + kPad;
      for (size_t i = 0; i < vocab_size; ++i) {
        logits_ptr[i] = static_cast<float>(i) * 0.01f + 1.0f;
      }
      const std::vector<float> original(logits_ptr, logits_ptr + vocab_size);

      std::vector<uint64_t> mask(num_words, ~0ULL);
      mask[target_idx / 64] &= ~(1ULL << (target_idx % 64));

      ApplyLogitMaskKernel(hwy::Span<float>(logits_ptr, vocab_size),
                           mask.data(), vocab_size);

      if (vocab_size == 1) {
        EXPECT_FLOAT_EQ(logits_ptr[0], original[0]);
      } else {
        for (size_t i = 0; i < vocab_size; ++i) {
          if (i == target_idx) {
            EXPECT_TRUE(std::isinf(logits_ptr[i]) && logits_ptr[i] < 0.0f);
          } else {
            EXPECT_FLOAT_EQ(logits_ptr[i], original[i]);
          }
        }
      }

      for (size_t i = 0; i < kPad; ++i) {
        EXPECT_FLOAT_EQ(buffer[i], 777.0f);
        EXPECT_FLOAT_EQ(buffer[kPad + vocab_size + i], 777.0f);
      }
    }
  }
}

void TestApplyLogitMaskAlternatingPatterns() {
  const std::vector<uint64_t> patterns = {
      0xAAAAAAAAAAAAAAAAULL, 0x5555555555555555ULL, 0x3333333333333333ULL,
      0x0F0F0F0F0F0F0F0FULL, 0x00FF00FF00FF00FFULL,
  };
  const std::vector<size_t> vocab_sizes = {63,  64,  65,  127,  128,   129,
                                           255, 256, 257, 1000, 256000};

  for (size_t vocab_size : vocab_sizes) {
    const size_t num_words = (vocab_size + 63) / 64;
    const size_t kPad = 64;

    for (uint64_t pattern : patterns) {
      std::vector<float> buffer(vocab_size + 2 * kPad, 888.0f);
      float* logits_ptr = buffer.data() + kPad;
      for (size_t i = 0; i < vocab_size; ++i) {
        logits_ptr[i] = static_cast<float>(i % 256) - 128.0f;
      }
      const std::vector<float> original(logits_ptr, logits_ptr + vocab_size);

      std::vector<uint64_t> mask(num_words, pattern);

      ApplyLogitMaskKernel(hwy::Span<float>(logits_ptr, vocab_size),
                           mask.data(), vocab_size);

      for (size_t i = 0; i < vocab_size; ++i) {
        const bool allowed = (mask[i / 64] & (1ULL << (i % 64))) != 0;
        if (allowed) {
          EXPECT_FLOAT_EQ(logits_ptr[i], original[i]);
        } else {
          EXPECT_TRUE(std::isinf(logits_ptr[i]) && logits_ptr[i] < 0.0f);
        }
      }

      for (size_t i = 0; i < kPad; ++i) {
        EXPECT_FLOAT_EQ(buffer[i], 888.0f);
        EXPECT_FLOAT_EQ(buffer[kPad + vocab_size + i], 888.0f);
      }
    }
  }
}

void TestApplyLogitMaskStressAndBenchmark() {
  const size_t vocab_size = 256000;
  const size_t num_words = (vocab_size + 63) / 64;
  std::vector<float> logits(vocab_size, 1.0f);

  struct BenchmarkCase {
    const char* name;
    std::vector<uint64_t> mask;
  };

  std::vector<BenchmarkCase> cases;
  cases.push_back(
      {"All-Allowed (No-op)", std::vector<uint64_t>(num_words, ~0ULL)});
  cases.push_back(
      {"Empty Mask (Guardrail)", std::vector<uint64_t>(num_words, 0ULL)});
  {
    std::vector<uint64_t> m(num_words, 0ULL);
    m[0] = 1ULL;
    cases.push_back({"All-Disallowed (Fast Path 2)", std::move(m)});
  }
  {
    std::vector<uint64_t> m(num_words, 0ULL);
    m[num_words - 1] = (1ULL << 63);
    cases.push_back({"Single Bit Set (Tail)", std::move(m)});
  }
  {
    std::vector<uint64_t> m(num_words, ~0ULL);
    m[num_words - 1] &= ~(1ULL << 63);
    cases.push_back({"Single Bit Unset (Tail)", std::move(m)});
  }
  cases.push_back({"Alternating 0xAAAA (SIMD Worst-Case)",
                   std::vector<uint64_t>(num_words, 0xAAAAAAAAAAAAAAAAULL)});
  cases.push_back({"Alternating 0x5555 (SIMD Worst-Case)",
                   std::vector<uint64_t>(num_words, 0x5555555555555555ULL)});
  {
    std::mt19937_64 rng(42);
    std::vector<uint64_t> m(num_words);
    for (size_t w = 0; w < num_words; ++w) {
      m[w] = rng();
    }
    cases.push_back({"Random 50% Mask", std::move(m)});
  }

  const int kWarmupIterations = 10;
  const int kBenchIterations = 100;

  for (const auto& bc : cases) {
    for (int it = 0; it < kWarmupIterations; ++it) {
      ApplyLogitMaskKernel(hwy::Span<float>(logits.data(), vocab_size),
                           bc.mask.data(), vocab_size);
    }

    const auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < kBenchIterations; ++it) {
      ApplyLogitMaskKernel(hwy::Span<float>(logits.data(), vocab_size),
                           bc.mask.data(), vocab_size);
    }
    const auto end = std::chrono::high_resolution_clock::now();

    const double total_us =
        std::chrono::duration<double, std::micro>(end - start).count();
    const double avg_us = total_us / kBenchIterations;

    printf("[BENCHMARK %s] Pattern '%s': %.2f us per call (vocab_size=%zu)\n",
           hwy::TargetName(HWY_TARGET), bc.name, avg_us, vocab_size);
    // Timing is logged for diagnostic purposes; wall-clock thresholds are
    // omitted to prevent non-deterministic failures across varying hardware and
    // emulation targets.
    (void)avg_us;
  }
}

static void ScalarApplyLogitMaskOracle(std::vector<float>& logits,
                                       const uint64_t* mask_words,
                                       size_t vocab_size, float mask_value) {
  if (vocab_size == 0 || logits.empty() || mask_words == nullptr) return;
  const size_t num_words = (vocab_size + 63) / 64;
  bool has_any_allowed = false;
  for (size_t w = 0; w < num_words; ++w) {
    uint64_t w_val = mask_words[w];
    if (w == num_words - 1 && (vocab_size % 64 != 0)) {
      w_val &= (1ULL << (vocab_size % 64)) - 1;
    }
    if (w_val != 0ULL) {
      has_any_allowed = true;
      break;
    }
  }
  if (!has_any_allowed) return;

  for (size_t i = 0; i < vocab_size; ++i) {
    const bool allowed = (mask_words[i / 64] & (1ULL << (i % 64))) != 0;
    if (!allowed) {
      logits[i] = mask_value;
    }
  }
}

void TestApplyLogitMaskFuzzingDifferentialOracle() {
  std::vector<size_t> vocab_sizes;
  // Sweep all small sizes 1 to 130
  for (size_t s = 1; s <= 130; ++s) {
    vocab_sizes.push_back(s);
  }
  // Sweep around powers of 2 and model vocabulary limits up to 300,000
  const std::vector<size_t> large_sizes = {
      255,    256,    257,    511,    512,    513,    1023,   1024,   1025,
      2047,   2048,   2049,   4095,   4096,   4097,   8191,   8192,   8193,
      16383,  16384,  16385,  32000,  32001,  65535,  65536,  65537,  129279,
      129280, 129281, 256000, 256063, 256064, 256128, 299993, 299999, 300000};
  vocab_sizes.insert(vocab_sizes.end(), large_sizes.begin(), large_sizes.end());

  std::mt19937_64 rng(987654321ULL);
  constexpr size_t kPad = 64;

  for (size_t vocab_size : vocab_sizes) {
    const size_t num_words = (vocab_size + 63) / 64;
    std::vector<float> buffer(vocab_size + 2 * kPad, 12345.0f);
    float* logits_ptr = buffer.data() + kPad;

    for (size_t i = 0; i < vocab_size; ++i) {
      logits_ptr[i] = static_cast<float>(i % 1000) * 0.1f - 50.0f;
    }
    const std::vector<float> original(logits_ptr, logits_ptr + vocab_size);

    for (int pattern_type = 0; pattern_type < 6; ++pattern_type) {
      std::vector<uint64_t> mask(num_words, 0ULL);
      switch (pattern_type) {
        case 0:  // Only first token allowed
          mask[0] = 1ULL;
          break;
        case 1:  // Only last valid token allowed
          mask[(vocab_size - 1) / 64] = (1ULL << ((vocab_size - 1) % 64));
          break;
        case 2:  // Dense allowed (all 1s)
          std::fill(mask.begin(), mask.end(), ~0ULL);
          break;
        case 3:  // Alternating 0xAAAAAAAAAAAAAAAA
          std::fill(mask.begin(), mask.end(), 0xAAAAAAAAAAAAAAAAULL);
          break;
        case 4:  // Sparse random (~2% allowed)
          for (size_t w = 0; w < num_words; ++w) {
            mask[w] = (rng() & rng() & rng() & rng());
          }
          mask[0] |= 1ULL;  // Ensure at least one allowed
          break;
        case 5:  // Uniform random
          for (size_t w = 0; w < num_words; ++w) {
            mask[w] = rng();
          }
          mask[(vocab_size - 1) / 64] |= (1ULL << ((vocab_size - 1) % 64));
          break;
      }

      std::vector<float> expected = original;
      ScalarApplyLogitMaskOracle(expected, mask.data(), vocab_size,
                                 -std::numeric_limits<float>::infinity());

      std::copy(original.begin(), original.end(), logits_ptr);

      ApplyLogitMaskKernel(hwy::Span<float>(logits_ptr, vocab_size),
                           mask.data(), vocab_size);

      for (size_t i = 0; i < vocab_size; ++i) {
        if (std::isinf(expected[i])) {
          ASSERT_TRUE(std::isinf(logits_ptr[i]) && logits_ptr[i] < 0.0f)
              << "Failed at vocab_size=" << vocab_size
              << ", pattern=" << pattern_type << ", index=" << i;
        } else {
          ASSERT_FLOAT_EQ(logits_ptr[i], expected[i])
              << "Failed at vocab_size=" << vocab_size
              << ", pattern=" << pattern_type << ", index=" << i;
        }
      }

      for (size_t i = 0; i < kPad; ++i) {
        ASSERT_FLOAT_EQ(buffer[i], 12345.0f);
        ASSERT_FLOAT_EQ(buffer[kPad + vocab_size + i], 12345.0f);
      }
    }
  }
}

void TestApplyLogitMaskNumericalStabilityAndBitExactness() {
  const size_t vocab_size = 256;
  const size_t num_words = (vocab_size + 63) / 64;

  std::vector<float> logits(vocab_size);
  logits[0] = 0.0f;
  logits[1] = -0.0f;
  logits[2] = std::numeric_limits<float>::infinity();
  logits[3] = -std::numeric_limits<float>::infinity();
  logits[4] = std::numeric_limits<float>::denorm_min();
  logits[5] = -std::numeric_limits<float>::denorm_min();
  logits[6] = std::numeric_limits<float>::max();
  logits[7] = std::numeric_limits<float>::lowest();
  uint32_t nan_bits = 0x7fc01234;
  float custom_nan;
  hwy::CopySameSize(&nan_bits, &custom_nan);
  logits[8] = custom_nan;

  for (size_t i = 9; i < vocab_size; ++i) {
    logits[i] = static_cast<float>(i);
  }

  std::vector<uint64_t> mask(num_words, 0x5555555555555555ULL);

  std::vector<float> test_logits = logits;
  ApplyLogitMaskKernel(hwy::Span<float>(test_logits.data(), vocab_size),
                       mask.data(), vocab_size);

  for (size_t i = 0; i < vocab_size; i += 2) {
    uint32_t orig_bits, masked_bits;
    hwy::CopySameSize(&logits[i], &orig_bits);
    hwy::CopySameSize(&test_logits[i], &masked_bits);
    EXPECT_EQ(orig_bits, masked_bits)
        << "Bit mismatch for allowed token at index " << i;
  }

  const uint32_t kNegInfBits = 0xFF800000;
  for (size_t i = 1; i < vocab_size; i += 2) {
    uint32_t masked_bits;
    hwy::CopySameSize(&test_logits[i], &masked_bits);
    EXPECT_EQ(masked_bits, kNegInfBits)
        << "Exact -inf bit mismatch for disallowed token at index " << i;
  }

  for (float custom_mask_val : {-1000.0f, -0.0f, 0.0f, 42.0f}) {
    test_logits = logits;
    ApplyLogitMaskKernel(hwy::Span<float>(test_logits.data(), vocab_size),
                         mask.data(), vocab_size, custom_mask_val);
    uint32_t expected_val_bits;
    hwy::CopySameSize(&custom_mask_val, &expected_val_bits);
    for (size_t i = 1; i < vocab_size; i += 2) {
      uint32_t actual_val_bits;
      hwy::CopySameSize(&test_logits[i], &actual_val_bits);
      EXPECT_EQ(actual_val_bits, expected_val_bits);
    }
  }
}

void TestApplyLogitMaskPageBoundaryProtection() {
  const size_t page_size = sysconf(_SC_PAGESIZE);
  const size_t total_alloc = 2 * page_size;

  void* addr = mmap(nullptr, total_alloc, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(addr, MAP_FAILED);

  uint8_t* base = static_cast<uint8_t*>(addr);
  ASSERT_EQ(mprotect(base + page_size, page_size, PROT_NONE), 0);

  for (size_t vocab_size : {1, 7, 15, 16, 31, 32, 63, 64, 65, 127, 128}) {
    const size_t num_words = (vocab_size + 63) / 64;
    const size_t bytes_needed = num_words * sizeof(uint64_t);

    uint64_t* mask_words =
        reinterpret_cast<uint64_t*>(base + page_size - bytes_needed);

    for (size_t w = 0; w < num_words; ++w) {
      mask_words[w] = 0xAAAAAAAAAAAAAAAAULL;
    }
    mask_words[0] |= 1ULL;

    std::vector<float> logits(vocab_size, 1.0f);

    ApplyLogitMaskKernel(hwy::Span<float>(logits.data(), vocab_size),
                         mask_words, vocab_size);

    for (size_t i = 0; i < vocab_size; ++i) {
      const bool allowed = (mask_words[i / 64] & (1ULL << (i % 64))) != 0;
      if (allowed) {
        EXPECT_FLOAT_EQ(logits[i], 1.0f);
      } else {
        EXPECT_TRUE(std::isinf(logits[i]) && logits[i] < 0.0f);
      }
    }
  }

  for (size_t vocab_size : {1, 3, 7, 15, 16, 31, 32, 63, 64, 65, 127, 128}) {
    const size_t bytes_needed = vocab_size * sizeof(float);
    float* logits_at_edge =
        reinterpret_cast<float*>(base + page_size - bytes_needed);

    for (size_t i = 0; i < vocab_size; ++i) {
      logits_at_edge[i] = static_cast<float>(i);
    }
    const size_t num_words = (vocab_size + 63) / 64;
    std::vector<uint64_t> mask(num_words, 0xAAAAAAAAAAAAAAAAULL);
    mask[0] |= 1ULL;

    ApplyLogitMaskKernel(hwy::Span<float>(logits_at_edge, vocab_size),
                         mask.data(), vocab_size);

    for (size_t i = 0; i < vocab_size; ++i) {
      const bool allowed = (mask[i / 64] & (1ULL << (i % 64))) != 0;
      if (allowed) {
        EXPECT_FLOAT_EQ(logits_at_edge[i], static_cast<float>(i));
      } else {
        EXPECT_TRUE(std::isinf(logits_at_edge[i]) && logits_at_edge[i] < 0.0f);
      }
    }
  }

  munmap(addr, total_alloc);
}

void TestApplyLogitMaskExtremeMisalignments() {
  const std::vector<size_t> vocab_sizes = {1,  7,   16,  63,  64,
                                           65, 127, 128, 500, 129280};
  std::mt19937_64 rng(54321);

  for (size_t vocab_size : vocab_sizes) {
    const size_t num_words = (vocab_size + 63) / 64;
    std::vector<uint64_t> mask(num_words);
    for (size_t w = 0; w < num_words; ++w) {
      mask[w] = rng();
    }
    mask[0] |= 1ULL;

    for (size_t misalign = 0; misalign < 16; ++misalign) {
      std::vector<float> buffer(vocab_size + misalign + 32, 333.0f);
      float* logits_ptr = buffer.data() + misalign;

      for (size_t i = 0; i < vocab_size; ++i) {
        logits_ptr[i] = static_cast<float>(i + 1);
      }
      const std::vector<float> original(logits_ptr, logits_ptr + vocab_size);

      ApplyLogitMaskKernel(hwy::Span<float>(logits_ptr, vocab_size),
                           mask.data(), vocab_size);

      for (size_t i = 0; i < vocab_size; ++i) {
        const bool is_allowed = (mask[i / 64] & (1ULL << (i % 64))) != 0;
        if (is_allowed) {
          EXPECT_FLOAT_EQ(logits_ptr[i], original[i]);
        } else {
          EXPECT_TRUE(std::isinf(logits_ptr[i]) && logits_ptr[i] < 0.0f);
        }
      }
    }
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gcpp {
HWY_BEFORE_TEST(OpsTest);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllAddFrom);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllMulByConst);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllMulByConstTo);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllMulByConstAndAdd);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllSoftmax);
HWY_EXPORT_AND_TEST_P(OpsTest, TestSoftmaxTemperature);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllSoftmaxState);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllCreateDistribution);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllSigmoid);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllFastSigmoid);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllGelu);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllFastGelu);
HWY_EXPORT_AND_TEST_P(OpsTest, TestRopeAndMulBy);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllRMSNorm);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllRMSNormInplace);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllLayerNorm);
HWY_EXPORT_AND_TEST_P(OpsTest, TestLayerNormSimple);
HWY_EXPORT_AND_TEST_P(OpsTest, TestSampleTopK);
HWY_EXPORT_AND_TEST_P(OpsTest, TestTopK);
HWY_EXPORT_AND_TEST_P(OpsTest, TestPackTokenAndProb);
HWY_EXPORT_AND_TEST_P(OpsTest, TestApplyLogitMaskAllAllowed);
HWY_EXPORT_AND_TEST_P(OpsTest, TestApplyLogitMaskAllDisallowed);
HWY_EXPORT_AND_TEST_P(OpsTest, TestApplyLogitMaskEmptyMaskGuardrail);
HWY_EXPORT_AND_TEST_P(OpsTest, TestApplyLogitMaskMixed);
HWY_EXPORT_AND_TEST_P(OpsTest, TestApplyLogitMaskUnalignedVocabSizes);
HWY_EXPORT_AND_TEST_P(OpsTest, TestApplyLogitMaskUnalignedPointers);
HWY_EXPORT_AND_TEST_P(OpsTest, TestApplyLogitMaskSingleBitSet);
HWY_EXPORT_AND_TEST_P(OpsTest, TestApplyLogitMaskSingleBitUnset);
HWY_EXPORT_AND_TEST_P(OpsTest, TestApplyLogitMaskAlternatingPatterns);
HWY_EXPORT_AND_TEST_P(OpsTest, TestApplyLogitMaskStressAndBenchmark);
HWY_EXPORT_AND_TEST_P(OpsTest, TestApplyLogitMaskFuzzingDifferentialOracle);
HWY_EXPORT_AND_TEST_P(OpsTest,
                      TestApplyLogitMaskNumericalStabilityAndBitExactness);
HWY_EXPORT_AND_TEST_P(OpsTest, TestApplyLogitMaskPageBoundaryProtection);
HWY_EXPORT_AND_TEST_P(OpsTest, TestApplyLogitMaskExtremeMisalignments);
HWY_AFTER_TEST();

TEST(OpsTest, TestApplyLogitMaskDynamicDispatch) {
  const size_t vocab_size = 100;
  std::vector<float> logits(vocab_size, 1.0f);
  const size_t num_words = (vocab_size + 63) / 64;
  std::vector<uint64_t> mask(num_words, 0ULL);
  mask[0] = 1ULL;  // Only token 0 allowed

  gcpp::ApplyLogitMaskKernel(hwy::Span<float>(logits.data(), vocab_size),
                             mask.data(), vocab_size);

  EXPECT_FLOAT_EQ(logits[0], 1.0f);
  for (size_t i = 1; i < vocab_size; ++i) {
    EXPECT_TRUE(std::isinf(logits[i]) && logits[i] < 0.0f);
  }
}

}  // namespace gcpp

#endif
