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
#include "compression/shared.h"
#include "gemma/activations.h"
#include "gemma/common.h"
#include "gemma/configs.h"
#include "gemma/gemma.h"
#include "gemma/weights.h"
#include "util/threading.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

TEST(OptimizeTest, GradientDescent) {
  NestedPools pools(1, /*pin=*/0, BoundedSlice(0, 1), BoundedSlice(0, 1));
  hwy::ThreadPool& pool = pools.Pool();
  std::mt19937 gen(42);

  const ModelInfo info = {
      .model = Model::GEMMA_TINY,
      .training = ModelTraining::GEMMA_IT,
      .weight = Type::kF32,
  };
  ModelConfig config = ConfigFromModel(info.model);
  ModelWeightsStorage grad, grad_m, grad_v;
  grad.Allocate(info.model, info.weight, pool);
  grad_m.Allocate(info.model, info.weight, pool);
  grad_v.Allocate(info.model, info.weight, pool);
  grad_m.ZeroInit();
  grad_v.ZeroInit();
  ForwardPass<float> forward(config), backward(config);
  KVCache kv_cache = KVCache::Create(config, /*prefill_tbatch_size=*/16);

  RowVectorBatch<float> inv_timescale = Activations::CreateInvTimescale(
      config.layer_configs[0].qkv_dim, config.layer_configs[0].post_qk);

  Gemma gemma(GemmaTokenizer(), info, pools);

  const auto generate = [&](const std::vector<int>& prompt) {
    std::vector<int> reply;
    auto stream_token = [&reply](int token, float) {
      reply.push_back(token);
      return token != ReverseSequenceSampler::kEndToken;
    };
    RuntimeConfig runtime = {
        .max_generated_tokens = 16,
        .temperature = 1.0f,
        .verbosity = 0,
        .gen = &gen,
        .stream_token = stream_token,
        .eos_id = ReverseSequenceSampler::kEndToken,
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

  gemma.MutableWeights().RandInit(gen);
  gemma.MutableWeights().AllocAndCopyWithTranspose(pool);

  printf("Initial weights:\n");
  gemma.MutableWeights().LogWeightStats();

  constexpr size_t kBatchSize = 8;
  const float alpha = 0.001f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float epsilon = 1e-8f;

  ReverseSequenceSampler training_task({
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1});
  size_t steps = 0;
  size_t num_ok;
  for (; steps < 1000000; ++steps) {
    std::mt19937 sgen(42);
    grad.ZeroInit();
    float total_loss = 0.0f;
    num_ok = 0;
    for (size_t i = 0; i < kBatchSize; ++i) {
      Prompt prompt = training_task.Sample(sgen);
      total_loss += CrossEntropyLossForwardPass(
          prompt, *gemma.Weights().GetWeightsOfType<float>(), forward,
          inv_timescale, pool);
      CrossEntropyLossBackwardPass(
          prompt, *gemma.Weights().GetWeightsOfType<float>(), forward,
          *grad.GetWeightsOfType<float>(), backward, inv_timescale, pool);
      gemma.MutableWeights().CopyWithTranspose(pool);
      num_ok += verify(prompt) ? 1 : 0;
    }
    total_loss /= kBatchSize;

    AdamUpdate(info.weight, grad, alpha, beta1, beta2, epsilon, steps + 1,
               gemma.Weights(), grad_m, grad_v, pool);
    printf("step: %zu  total_loss: %.15f   num_ok: %zu/%zu\n",
           steps, total_loss, num_ok, kBatchSize);
    if (steps % 100 == 0) {
      printf("Batch gradient:\n");
      grad.LogWeightStats();
    }
    if (total_loss < 0.5f) {
      break;
    }
  }
  printf("Num steps: %zu\n", steps);
  printf("Final weights:\n");
  gemma.MutableWeights().LogWeightStats();
  EXPECT_LT(steps, 300);
  EXPECT_EQ(num_ok, kBatchSize);
}

}  // namespace gcpp
