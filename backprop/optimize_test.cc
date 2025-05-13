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

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "backprop/activations.h"
#include "backprop/backward.h"
#include "backprop/forward.h"
#include "backprop/optimizer.h"
#include "backprop/prompt.h"
#include "backprop/sampler.h"
#include "compression/types.h"
#include "gemma/configs.h"
#include "gemma/gemma.h"
#include "gemma/tokenizer.h"
#include "gemma/weights.h"
#include "ops/ops.h"
#include "util/allocator.h"
#include "util/basics.h"
#include "util/threading.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

TEST(OptimizeTest, GradientDescent) {
  gcpp::ThreadingArgs threading_args;
  threading_args.max_packages = 1;
  threading_args.max_clusters = 1;
  threading_args.pin = Tristate::kFalse;
  ThreadingContext::SetArgs(threading_args);
  MatMulEnv env(ThreadingContext::Get());
  const Allocator& allocator = env.ctx.allocator;
  hwy::ThreadPool& pool = env.ctx.pools.Pool();
  std::mt19937 gen(42);

  ModelConfig config(Model::GEMMA_TINY, Type::kF32,
                     ChooseWrapping(Model::GEMMA_TINY));
  config.eos_id = ReverseSequenceSampler::kEndToken;

  WeightsOwner grad(Type::kF32), grad_m(Type::kF32), grad_v(Type::kF32);
  grad.AllocateForTest(config, pool);
  grad_m.AllocateForTest(config, pool);
  grad_v.AllocateForTest(config, pool);
  grad_m.ZeroInit();
  grad_v.ZeroInit();
  ForwardPass<float> forward(config), backward(config);
  KVCache kv_cache(config, /*prefill_tbatch_size=*/16);

  MatStorageT<float> inv_timescale = CreateInvTimescale(
      allocator, config.layer_configs[0].qkv_dim,
      config.layer_configs[0].post_qk == PostQKType::HalfRope);

  Gemma gemma(config, GemmaTokenizer(kMockTokenizer), env);

  const auto generate = [&](const std::vector<int>& prompt) {
    std::vector<int> reply;
    auto stream_token = [&reply](int token, float) {
      reply.push_back(token);
      return token != ReverseSequenceSampler::kEndToken;
    };
    RuntimeConfig runtime = {
        .max_generated_tokens = 16,
        .temperature = 1.0f,
        .gen = &gen,
        .verbosity = 0,
        .stream_token = stream_token,
    };
    TimingInfo timing_info;
    gemma.Generate(runtime, prompt, 0, kv_cache, timing_info);
    return reply;
  };

  // Sanity check of reply tokens.
  // 1) Its length should be greater than the prompt.
  // 2) The prompt should be a prefix of the reply.
  auto verify = [&](const Prompt& prompt) {
    const std::vector<int>& context = prompt.context();
    std::vector<int> reply = generate(context);
    if (reply.size() <= context.size()) return false;
    return std::equal(context.begin(), context.end(), reply.begin(),
                      reply.begin() + context.size());
  };

  gemma.MutableWeights().RandInit(1.0f, gen);
  gemma.MutableWeights().Fixup(pool);

  printf("Initial weights:\n");
  gemma.MutableWeights().LogWeightStatsF32();

  constexpr size_t kBatchSize = 8;
  constexpr float kAlpha = 0.001f;
  constexpr float kBeta1 = 0.9f;
  constexpr float kBeta2 = 0.999f;
  constexpr float kEpsilon = 1e-8f;

  constexpr float kMaxLoss = 20.0f;

  ReverseSequenceSampler training_task({
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1});
  size_t steps = 0;
  size_t num_ok;
  for (; steps < 1000; ++steps) {
    std::mt19937 sgen(42);
    grad.ZeroInit();
    float total_loss = 0.0f;
    num_ok = 0;
    for (size_t i = 0; i < kBatchSize; ++i) {
      Prompt prompt = training_task.Sample(sgen);
      total_loss += CrossEntropyLossForwardPass(
          prompt, *gemma.Weights().GetF32(), forward, inv_timescale, pool);
      CrossEntropyLossBackwardPass(prompt, *gemma.Weights().GetF32(), forward,
                                   *grad.GetF32(), backward, inv_timescale,
                                   pool);
      gemma.MutableWeights().Fixup(pool);
      num_ok += verify(prompt) ? 1 : 0;
    }
    total_loss /= kBatchSize;

    AdamUpdate(grad, kAlpha, kBeta1, kBeta2, kEpsilon, steps + 1,
               gemma.Weights(), grad_m, grad_v, pool);
    printf("step: %zu  total_loss: %.15f   num_ok: %zu/%zu\n",
           steps, total_loss, num_ok, kBatchSize);
    if (steps % 100 == 0) {
      printf("Batch gradient:\n");
      grad.LogWeightStatsF32();
    }
    if (total_loss < kMaxLoss) break;  // Done
  }
  printf("Num steps: %zu\n", steps);
  printf("Final weights:\n");
  gemma.MutableWeights().LogWeightStatsF32();
  EXPECT_LT(steps, 80);
  EXPECT_EQ(num_ok, kBatchSize);
}

}  // namespace gcpp
