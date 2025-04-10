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
#include "gemma/configs.h"
#include "ops/ops.h"
#include "util/mat.h"
#include "util/threading_context.h"
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
#include "ops/ops-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

hwy::ThreadPool& ThreadHostileGetPool() {
  // Assume this is only called at the top level, i.e. not in a thread. Then we
  // can safely call `SetArgs` only once, because it would assert otherwise.
  // This is preferable to calling `ThreadHostileInvalidate`, because we would
  // repeat the topology initialization for every test.
  if (!ThreadingContext2::IsInitialized()) {
    gcpp::ThreadingArgs threading_args;
    threading_args.max_packages = 1;
    threading_args.max_clusters = 8;
    threading_args.pin = Tristate::kFalse;
    ThreadingContext2::SetArgs(threading_args);
  }
  return ThreadingContext2::Get().pools.Pool();
}

void TestMatMulVJP() {
  static const size_t kRows = 8;
  static const size_t kCols = 64;
  static const size_t kTokens = 5;

  hwy::ThreadPool& pool = ThreadHostileGetPool();
  std::mt19937 gen(42);
  auto weights = MakePacked<float>("weights", kRows, kCols);
  auto x = MakePacked<float>("x", kTokens, kCols);
  auto dy = MakePacked<float>("dy", kTokens, kRows);
  auto grad = MakePacked<float>("grad", kRows, kCols);
  auto dx = MakePacked<float>("dx", kTokens, kCols);
  auto grad_scalar = MakePacked<float>("grad_scalar", kRows, kCols);
  auto dx_scalar = MakePacked<float>("dx_scalar", kTokens, kCols);
  using TC = std::complex<double>;
  auto c_weights = MakePacked<TC>("c_weights", kRows, kCols);
  auto c_x = MakePacked<TC>("c_x", kTokens, kCols);
  auto c_y = MakePacked<TC>("c_y", kTokens, kRows);

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(weights, 1.0f * (1 << iter), gen);
    RandInit(x, 1.0f * (1 << iter), gen);
    RandInit(dy, 1.0f, gen);
    Complexify(weights, c_weights);
    Complexify(x, c_x);
    auto func = [&]() {
      MatMulT(c_weights.Packed(), c_x.Packed(), c_y.Packed(), kRows, kCols,
              kTokens);
      return DotT(dy.Packed(), c_y.Packed(), kTokens * kRows);
    };

    ZeroInit(grad);
    MatMulVJP(weights.Packed(), x.Packed(), dy.Packed(), kCols, kRows, kTokens,
              grad.Packed(), dx.Packed(), pool);
    TestGradient(dx, c_x, func, 5e-5f, 5e-5f, __LINE__);
    TestGradient(grad, c_weights, func, 5e-5f, 5e-5f, __LINE__);

    ZeroInit(grad_scalar);
    MatMulVJPT(weights.Packed(), x.Packed(), dy.Packed(), grad_scalar.Packed(),
               dx_scalar.Packed(), kRows, kCols, kTokens);
    TestNear(dx, dx_scalar, 5e-5, 1e-4, __LINE__);
    TestNear(grad, grad_scalar, 5e-5, 5e-5, __LINE__);
  }
}

void TestMultiHeadMatMulVJP() {
  static const size_t kRows = 2;
  static const size_t kCols = 16;
  static const size_t kHeads = 4;
  static const size_t kTokens = 3;
  hwy::ThreadPool& pool = ThreadHostileGetPool();
  std::mt19937 gen(42);
  auto weights = MakePacked<float>("weights", kRows, kCols * kHeads);
  auto x = MakePacked<float>("x", kTokens, kCols * kHeads);
  auto grad = MakePacked<float>("grad", kRows, kCols * kHeads);
  auto dx = MakePacked<float>("dx", kTokens, kCols * kHeads);
  auto dy = MakePacked<float>("dy", kTokens, kRows);
  auto grad_scalar = MakePacked<float>("grad_scalar", kRows, kCols * kHeads);
  auto dx_scalar = MakePacked<float>("dx_scalar", kTokens, kCols * kHeads);
  using TC = std::complex<double>;
  auto c_weights = MakePacked<TC>("c_weights", kRows, kCols * kHeads);
  auto c_x = MakePacked<TC>("c_x", kTokens, kCols * kHeads);
  auto c_y = MakePacked<TC>("c_y", kTokens, kRows);

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(weights, 1.0f * (1 << iter), gen);
    RandInit(x, 1.0f * (1 << iter), gen);
    RandInit(dy, 1.0f, gen);
    Complexify(weights, c_weights);
    Complexify(x, c_x);
    auto func = [&]() {
      MultiHeadMatMul(c_weights.Packed(), c_x.Packed(), c_y.Packed(), kHeads,
                      kRows, kCols, kTokens);
      return DotT(dy.Packed(), c_y.Packed(), kTokens * kRows);
    };

    ZeroInit(grad);
    MultiHeadMatMulVJP(weights.Packed(), x.Packed(), dy.Packed(), kHeads, kCols,
                       kRows, kTokens, grad.Packed(), dx.Packed(), pool);
    TestGradient(dx, c_x, func, 5e-5f, 5e-5f, __LINE__);
    TestGradient(grad, c_weights, func, 5e-5f, 5e-5f, __LINE__);

    ZeroInit(grad_scalar);
    MultiHeadMatMulVJPT(weights.Packed(), x.Packed(), dy.Packed(),
                        grad_scalar.Packed(), dx_scalar.Packed(), kHeads, kRows,
                        kCols, kTokens);
    TestNear(dx, dx_scalar, 5e-5, 5e-5, __LINE__);
    TestNear(grad, grad_scalar, 5e-5, 5e-5, __LINE__);
  }
}

void TestRMSNormVJP() {
  static const size_t K = 2;
  static const size_t N = 64;
  hwy::ThreadPool& pool = ThreadHostileGetPool();
  std::mt19937 gen(42);
  auto weights = MakePacked<float>("weights", N, 1);
  auto x = MakePacked<float>("x", K, N);
  auto grad = MakePacked<float>("grad", N, 1);
  auto dx = MakePacked<float>("dx", K, N);
  auto dy = MakePacked<float>("dy", K, N);
  auto grad_scalar = MakePacked<float>("grad_scalar", N, 1);
  auto dx_scalar = MakePacked<float>("dx_scalar", K, N);
  using TC = std::complex<double>;
  auto c_weights = MakePacked<TC>("c_weights", N, 1);
  auto c_x = MakePacked<TC>("c_x", K, N);
  auto c_y = MakePacked<TC>("c_y", K, N);

  for (int iter = 0; iter < 10; ++iter) {
    RandInit(weights, 1.0f * (1 << iter), gen);
    RandInit(x, 1.0f * (1 << iter), gen);
    RandInit(dy, 1.0f, gen);
    Complexify(weights, c_weights);
    Complexify(x, c_x);
    auto func = [&]() {
      RMSNormT(c_weights.Packed(), c_x.Packed(), c_y.Packed(), N, K);
      return DotT(dy.Packed(), c_y.Packed(), K * N);
    };

    ZeroInit(grad);
    RMSNormVJP(weights.Packed(), x.Packed(), dy.Packed(), N, K, grad.Packed(),
               dx.Packed(), pool);
    TestGradient(dx, c_x, func, 5e-5f, 5e-5f, __LINE__);
    TestGradient(grad, c_weights, func, 5e-5f, 5e-5f, __LINE__);

    ZeroInit(grad_scalar);
    RMSNormVJPT(weights.Packed(), x.Packed(), dy.Packed(), grad_scalar.Packed(),
                dx_scalar.Packed(), N, K);
    TestNear(dx, dx_scalar, 0, 2e-5, __LINE__);
    TestNear(grad, grad_scalar, 0, 2e-5, __LINE__);
  }
}

static ModelConfig TestConfig() {
  ModelConfig config;
  config.scale_names = {"att_ein",      "qkv_ein",   "gr_lin_x_w", "gr_lin_y_w",
                        "gr_lin_out_w", "gr_gate_w", "gating_ein", "linear_w"};
  config.model_dim = 32;
  config.vocab_size = 16;
  config.seq_len = 24;
  LayerConfig layer_config;
  layer_config.model_dim = config.model_dim;
  layer_config.ff_hidden_dim = 64;
  layer_config.heads = 3;
  layer_config.kv_heads = 1;
  layer_config.qkv_dim = 16;
  config.layer_configs = {2, layer_config};
  config.num_tensor_scales = 4 * config.layer_configs.size();
  config.query_scale = QueryScaleType::SqrtKeySize;
  config.attention_window_sizes = FixedAttentionWindowSizes<2>(32);
  // This is required for optimize_test to pass.
  config.att_cap = 50.0f;
  config.final_cap = 30.0f;
  return config;
}

void TestEndToEnd() {
  std::mt19937 gen(42);
  hwy::ThreadPool& pool = ThreadHostileGetPool();
  ModelConfig config = TestConfig();
  WeightsWrapper<float> weights(config);
  WeightsWrapper<float> grad(config);
  ForwardPass<float> forward0(config);
  ForwardPass<float> forward1(config);
  ForwardPass<float> backward(config);
  using TC = std::complex<double>;
  WeightsWrapper<TC> c_weights(config);
  ForwardPass<TC> c_forward(config);

  ReverseSequenceSampler training_task({0, 0, 1, 1});
  std::vector<Prompt> batch = training_task.SampleBatch(3, gen);

  RowVectorBatch<float> inv_timescale = CreateInvTimescale(
      ThreadingContext2::Get().allocator, config.layer_configs[0].qkv_dim,
      config.layer_configs[0].post_qk == PostQKType::HalfRope);
  for (const Prompt& prompt : batch) {
    ReverseSequenceSampler::LogPrompt(prompt);
    RandInit(weights.get(), 1.0f, gen);

    float loss0 = CrossEntropyLossForwardPass(prompt, weights.get(), forward0);

    float loss1 = CrossEntropyLossForwardPass(
        prompt.tokens, prompt.context_size, weights.get(), forward1,
        inv_timescale, pool);

    EXPECT_NEAR(loss1, loss0, std::abs(loss0) * 2e-5);

    grad.ZeroInit();
    CrossEntropyLossBackwardPassInl(prompt, weights.get(), forward1, grad.get(),
                                    backward, inv_timescale, pool);

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
