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

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS HWY_SCALAR
#endif

#include <stddef.h>

#include <array>
#include <complex>
#include <cstdlib>  // std::abs
#include <random>
#include <vector>

#include "backprop/activations.h"
#include "backprop/backward_scalar.h"
#include "backprop/common_scalar.h"
#include "backprop/forward_scalar.h"
#include "backprop/prompt.h"
#include "backprop/sampler.h"
#include "backprop/test_util.h"
#include "gemma/activations.h"
#include "gemma/configs.h"
#include "gemma/weights.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "backprop/backward_test.cc"  //NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"
// After highway.h
#include "backprop/backward-inl.h"
#include "backprop/forward-inl.h"
#include "compression/compress.h"
#include "ops/ops-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

void TestMatMulVJP() {
  static const size_t kRows = 8;
  static const size_t kCols = 64;
  static const size_t kTokens = 5;
  hwy::ThreadPool pool(8);
  std::mt19937 gen(42);
  MatStorageT<float> weights("weights", kRows, kCols);
  MatStorageT<float> x("x", kTokens, kCols);
  MatStorageT<float> dy("dy", kTokens, kRows);
  MatStorageT<float> grad("grad", kRows, kCols);
  MatStorageT<float> dx("dx", kTokens, kCols);
  MatStorageT<float> grad_scalar("grad_scalar", kRows, kCols);
  MatStorageT<float> dx_scalar("dx_scalar", kTokens, kCols);
  using TC = std::complex<double>;
  MatStorageT<TC> c_weights("c_weights", kRows, kCols);
  MatStorageT<TC> c_x("c_x", kTokens, kCols);
  MatStorageT<TC> c_y("c_y", kTokens, kRows);

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(weights, 1.0f * (1 << iter), gen);
    RandInit(x, 1.0f * (1 << iter), gen);
    RandInit(dy, 1.0f, gen);
    Complexify(weights, c_weights);
    Complexify(x, c_x);
    auto func = [&]() {
      MatMulT(c_weights.data(), c_x.data(), c_y.data(), kRows, kCols, kTokens);
      return DotT(dy.data(), c_y.data(), kTokens * kRows);
    };

    grad.ZeroInit();
    MatMulVJP<kCols, kRows>(weights.data(), x.data(), dy.data(), kTokens,
                            grad.data(), dx.data(), pool);
    TestGradient(dx, c_x, func, 5e-5f, 5e-5f, __LINE__);
    TestGradient(grad, c_weights, func, 5e-5f, 5e-5f, __LINE__);

    grad_scalar.ZeroInit();
    MatMulVJPT(weights.data(), x.data(), dy.data(), grad_scalar.data(),
               dx_scalar.data(), kRows, kCols, kTokens);
    TestNear(dx, dx_scalar, 5e-5, 1e-4, __LINE__);
    TestNear(grad, grad_scalar, 5e-5, 5e-5, __LINE__);
  }
}

void TestMultiHeadMatMulVJP() {
  static const size_t kRows = 2;
  static const size_t kCols = 16;
  static const size_t kHeads = 4;
  static const size_t kTokens = 3;
  hwy::ThreadPool pool(8);
  std::mt19937 gen(42);
  MatStorageT<float> weights("weights", kRows, kCols * kHeads);
  MatStorageT<float> x("x", kTokens, kCols * kHeads);
  MatStorageT<float> grad("grad", kRows, kCols * kHeads);
  MatStorageT<float> dx("dx", kTokens, kCols * kHeads);
  MatStorageT<float> dy("dy", kTokens, kRows);
  MatStorageT<float> grad_scalar("grad_scalar", kRows, kCols * kHeads);
  MatStorageT<float> dx_scalar("dx_scalar", kTokens, kCols * kHeads);
  using TC = std::complex<double>;
  MatStorageT<TC> c_weights("c_weights", kRows, kCols * kHeads);
  MatStorageT<TC> c_x("c_x", kTokens, kCols * kHeads);
  MatStorageT<TC> c_y("c_y", kTokens, kRows);

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(weights, 1.0f * (1 << iter), gen);
    RandInit(x, 1.0f * (1 << iter), gen);
    RandInit(dy, 1.0f, gen);
    Complexify(weights, c_weights);
    Complexify(x, c_x);
    auto func = [&]() {
      MultiHeadMatMul(c_weights.data(), c_x.data(), c_y.data(), kHeads, kRows,
                      kCols, kTokens);
      return DotT(dy.data(), c_y.data(), kTokens * kRows);
    };

    grad.ZeroInit();
    MultiHeadMatMulVJP<kHeads, kCols, kRows>(
        weights.data(), x.data(), dy.data(), kTokens, grad.data(), dx.data(),
        pool);
    TestGradient(dx, c_x, func, 5e-5f, 5e-5f, __LINE__);
    TestGradient(grad, c_weights, func, 5e-5f, 5e-5f, __LINE__);

    grad_scalar.ZeroInit();
    MultiHeadMatMulVJPT(weights.data(), x.data(), dy.data(), grad_scalar.data(),
                        dx_scalar.data(), kHeads, kRows, kCols, kTokens);
    TestNear(dx, dx_scalar, 5e-5, 5e-5, __LINE__);
    TestNear(grad, grad_scalar, 5e-5, 5e-5, __LINE__);
  }
}

void TestRMSNormVJP() {
  static const size_t K = 2;
  static const size_t N = 64;
  hwy::ThreadPool pool(8);
  std::mt19937 gen(42);
  MatStorageT<float> weights("weights", N, 1);
  MatStorageT<float> x("x", K, N);
  MatStorageT<float> grad("grad", N, 1);
  MatStorageT<float> dx("dx", K, N);
  MatStorageT<float> dy("dy", K, N);
  MatStorageT<float> grad_scalar("grad_scalar", N, 1);
  MatStorageT<float> dx_scalar("dx_scalar", K, N);
  using TC = std::complex<double>;
  MatStorageT<TC> c_weights("c_weights", N, 1);
  MatStorageT<TC> c_x("c_x", K, N);
  MatStorageT<TC> c_y("c_y", K, N);

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(weights, 1.0f * (1 << iter), gen);
    RandInit(x, 1.0f * (1 << iter), gen);
    RandInit(dy, 1.0f, gen);
    Complexify(weights, c_weights);
    Complexify(x, c_x);
    auto func = [&]() {
      RMSNormT(c_weights.data(), c_x.data(), c_y.data(), N, K);
      return DotT(dy.data(), c_y.data(), K * N);
    };

    grad.ZeroInit();
    RMSNormVJP(weights.data(), x.data(), dy.data(), N, K, grad.data(),
               dx.data(), pool);
    TestGradient(dx, c_x, func, 5e-5f, 5e-5f, __LINE__);
    TestGradient(grad, c_weights, func, 5e-5f, 5e-5f, __LINE__);

    grad_scalar.ZeroInit();
    RMSNormVJPT(weights.data(), x.data(), dy.data(), grad_scalar.data(),
                dx_scalar.data(), N, K);
    TestNear(dx, dx_scalar, 0, 2e-5, __LINE__);
    TestNear(grad, grad_scalar, 0, 2e-5, __LINE__);
  }
}

template <typename T>
struct TestConfig : ConfigBaseGemmaV2 {
  using Weight = T;
  static constexpr int kSeqLen = 24;
  static constexpr int kVocabSize = 16;
  static constexpr int kModelDim = 32;
  static constexpr int kHeads = 3;
  static constexpr int kQKVDim = 16;
  static constexpr int kFFHiddenDim = 64;
  static constexpr std::array<LayerAttentionType, 2> kLayerConfig =
      FixedLayerConfig<2>(LayerAttentionType::kGemma);
  static constexpr int kLayers = kLayerConfig.size();
  static constexpr int kNumTensorScales = 4 * kLayers;
  static constexpr bool kAbsolutePE = false;
  static constexpr PostNormType kPostNorm = PostNormType::None;

  static constexpr int kKVHeads = 1;
  static constexpr int kGemmaLayers = kLayers;
};

void TestEndToEnd() {
  std::mt19937 gen(42);
  hwy::ThreadPool pool(0);
  using WeightsF = CompressedWeights<TestConfig<float>>;
  using LayerF = CompressedLayer<TestConfig<float>>;
  WeightsWrapper<TestConfig<float>> weights;
  WeightsWrapper<TestConfig<float>> grad;
  ActivationsWrapper<float, TestConfig<float>> forward0;
  ActivationsWrapper<float, TestConfig<float>> forward1;
  ActivationsWrapper<float, TestConfig<float>> backward;
  using TC = std::complex<double>;
  WeightsWrapper<TestConfig<TC>> c_weights;
  ForwardPass<TC, TestConfig<TC>> c_forward;

  ReverseSequenceSampler training_task({0, 0, 1, 1});
  std::vector<Prompt> batch = training_task.SampleBatch(3, gen);

  RowVectorBatch<float> inv_timescale =
      Activations::CreateInvTimescale<TestConfig<float>>();
  for (const Prompt& prompt : batch) {
    ReverseSequenceSampler::LogPrompt(prompt);
    RandInit(weights.get(), 1.0f, gen);

    float loss0 = CrossEntropyLossForwardPass(
        prompt, weights.get(), forward0.get());

    float loss1 =
        CrossEntropyLossForwardPass<TestConfig<float>, WeightsF, LayerF>(
            prompt.tokens, prompt.context_size, weights.get(), forward1.get(),
            inv_timescale, pool);

    EXPECT_NEAR(loss1, loss0, std::abs(loss0) * 2e-5);

    grad.ZeroInit();
    CrossEntropyLossBackwardPass<TestConfig<float>, WeightsF, LayerF>(
        prompt, weights.get(), forward1.get(), grad.get(), backward.get(),
        inv_timescale, pool);

    Complexify(weights.get(), c_weights.get());
    auto func = [&]() {
      return CrossEntropyLossForwardPass(prompt, c_weights.get(), c_forward);
    };

    TestGradient(grad.get(), c_weights.get(), func, 2e-3f);
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gcpp {
HWY_BEFORE_TEST(BackwardTest);
HWY_EXPORT_AND_TEST_P(BackwardTest, TestMatMulVJP);
HWY_EXPORT_AND_TEST_P(BackwardTest, TestMultiHeadMatMulVJP);
HWY_EXPORT_AND_TEST_P(BackwardTest, TestRMSNormVJP);
HWY_EXPORT_AND_TEST_P(BackwardTest, TestEndToEnd);
HWY_AFTER_TEST();

}  // namespace gcpp

#endif
