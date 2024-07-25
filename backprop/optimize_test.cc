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

#include <limits>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "backprop/activations.h"
#include "backprop/backward.h"
#include "backprop/forward.h"
#include "backprop/optimizer.h"
#include "backprop/prompt.h"
#include "backprop/sampler.h"
#include "gemma/common.h"
#include "gemma/gemma.h"
#include "gemma/weights.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

TEST(OptimizeTest, GradientDescent) {
  hwy::ThreadPool pool(0);
  std::mt19937 gen(42);

  const ModelInfo info = {
      .model = Model::GEMMA_TINY,
      .training = ModelTraining::GEMMA_IT,
      .weight = Type::kF32,
  };
  ByteStorageT grad = CallForModelAndWeight<AllocateCompressedWeights>(
      info.model, info.weight, pool);
  ByteStorageT grad_m = CallForModelAndWeight<AllocateCompressedWeights>(
      info.model, info.weight, pool);
  ByteStorageT grad_v = CallForModelAndWeight<AllocateCompressedWeights>(
      info.model, info.weight, pool);
  ByteStorageT forward =
      CallForModelAndWeight<AllocateForwardPass>(info.model, info.weight);
  ByteStorageT backward =
      CallForModelAndWeight<AllocateForwardPass>(info.model, info.weight);
  KVCache kv_cache = KVCache::Create(info.model, /*prefill_tbatch_size=*/16);

  Gemma gemma(GemmaTokenizer(), info, pool);

  const auto generate = [&](const std::vector<int>& prompt) {
    std::vector<int> reply;
    auto stream_token = [&reply](int token, float) {
      reply.push_back(token);
      return token != ReverseSequenceSampler::kEndToken;
    };
    RuntimeConfig runtime = {
        .max_tokens = 32,
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

  auto verify = [&](const Prompt& prompt) {
    auto context = prompt.context();
    std::vector<int> reply = generate(context);
    bool ok = true;
    for (size_t i = 0; ok && i < prompt.tokens.size(); ++i) {
      if (i >= reply.size() || reply[i] != prompt.tokens[i]) {
        ok = false;
      }
    }
    return ok;
  };

  RandInitWeights(info.model, info.weight, gemma.Weights(), pool, gen);
  CallForModelAndWeight<ZeroInitCompressedWeights>(info.model, info.weight,
                                                   grad_m, pool);
  CallForModelAndWeight<ZeroInitCompressedWeights>(info.model, info.weight,
                                                   grad_v, pool);

  printf("Initial weights:\n");
  LogWeightStats(info.model, info.weight, gemma.Weights());

  constexpr size_t kBatchSize = 8;
  const float alpha = 0.001f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float epsilon = 1e-8f;

  ReverseSequenceSampler training_task({
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1});
  size_t steps = 0;
  float prev_loss = std::numeric_limits<float>::max();
  size_t num_ok;
  for (; steps < 1000000; ++steps) {
    std::mt19937 sgen(42);
    CallForModelAndWeight<ZeroInitCompressedWeights>(info.model, info.weight,
                                                     grad, pool);
    float total_loss = 0.0f;
    num_ok = 0;
    for (size_t i = 0; i < kBatchSize; ++i) {
      Prompt prompt = training_task.Sample(sgen);
      total_loss += CrossEntropyLossForwardPass(info.model, prompt,
                                                gemma.Weights(), forward, pool);
      CrossEntropyLossBackwardPass(info.model, prompt, gemma.Weights(), forward,
                                   grad, backward, pool);
      num_ok += verify(prompt) ? 1 : 0;
    }
    total_loss /= kBatchSize;

    AdamUpdate(info.model, info.weight, grad, alpha, beta1, beta2, epsilon,
               steps + 1, gemma.Weights(), grad_m, grad_v, pool);
    printf("step: %zu  total_loss: %.15f   num_ok: %zu/%zu\n",
           steps, total_loss, num_ok, kBatchSize);
    if (steps % 100 == 0) {
      printf("Batch gradient:\n");
      LogWeightStats(info.model, info.weight, grad);
    }
    if (total_loss < 0.5f) {
      break;
    }
    prev_loss = total_loss;
  }
  printf("Num steps: %zu\n", steps);
  printf("Final weights:\n");
  LogWeightStats(info.model, info.weight, gemma.Weights());
  EXPECT_LT(steps, 300);
  EXPECT_EQ(num_ok, kBatchSize);
}

}  // namespace gcpp
