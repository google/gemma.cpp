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

#include <iostream>
#include <string>

#include "backprop/backward.h"
#include "backprop/forward.h"
#include "backprop/optimizer.h"
#include "backprop/sampler.h"
#include "gemma/activations.h"
#include "gemma/common.h"
#include "gemma/gemma.h"
#include "gemma/weights.h"
#include "gtest/gtest.h"

namespace gcpp {

TEST(OptimizeTest, GradientDescent) {
  if (kWeightsAreCompressed) return;

  hwy::ThreadPool pool(0);
  std::mt19937 gen(42);

  Model model_type = Model::GEMMA_TINY;
  ByteStorageT grad = CallFunctorForModel<AllocateWeightsF>(model_type, pool);
  ByteStorageT grad_m = CallFunctorForModel<AllocateWeightsF>(model_type, pool);
  ByteStorageT grad_v = CallFunctorForModel<AllocateWeightsF>(model_type, pool);
  ByteStorageT forward = CallFunctorForModel<AllocateForwardPass>(model_type);
  ByteStorageT backward = CallFunctorForModel<AllocateForwardPass>(model_type);
  KVCache kv_cache = KVCache::Create(model_type);
  size_t max_tokens = 32;
  size_t max_generated_tokens = 16;
  float temperature = 1.0f;
  int verbosity = 0;
  const auto accept_token = [](int) { return true; };

  Gemma gemma(GemmaTokenizer(), model_type, pool);

  const auto generate = [&](const std::vector<int>& prompt) {
    std::vector<int> reply;
    auto stream_token = [&reply](int token, float) {
      reply.push_back(token);
      return token != ReverseSequenceSampler::kEndToken;
    };
    RuntimeConfig runtime = {
      max_tokens, max_generated_tokens, temperature, verbosity, &gen,
      stream_token, accept_token, ReverseSequenceSampler::kEndToken,
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

  RandInitWeights(model_type, gemma.Weights(), pool, gen);
  CallFunctorForModel<ZeroInitWeightsF>(model_type, grad_m, pool);
  CallFunctorForModel<ZeroInitWeightsF>(model_type, grad_v, pool);

  printf("Initial weights:\n");
  LogWeightStats(model_type, gemma.Weights());

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
    CallFunctorForModel<ZeroInitWeightsF>(model_type, grad, pool);
    float total_loss = 0.0f;
    num_ok = 0;
    for (size_t i = 0; i < kBatchSize; ++i) {
      Prompt prompt = training_task.Sample(sgen);
      total_loss += CrossEntropyLossForwardPass(model_type, prompt,
                                                gemma.Weights(), forward, pool);
      CrossEntropyLossBackwardPass(model_type, prompt, gemma.Weights(), forward,
                                   grad, backward, pool);
      num_ok += verify(prompt) ? 1 : 0;
    }
    total_loss /= kBatchSize;

    AdamUpdate(model_type, grad, alpha, beta1, beta2, epsilon, steps + 1,
               gemma.Weights(), grad_m, grad_v, pool);
    printf("step: %zu  total_loss: %.15f   num_ok: %zu/%zu\n",
           steps, total_loss, num_ok, kBatchSize);
    if (steps % 100 == 0) {
      printf("Batch gradient:\n");
      LogWeightStats(model_type, grad);
    }
    if (total_loss < 0.5f) {
      break;
    }
    prev_loss = total_loss;
  }
  printf("Num steps: %zu\n", steps);
  printf("Final weights:\n");
  LogWeightStats(model_type, gemma.Weights());
  EXPECT_LT(steps, 200);
  EXPECT_EQ(num_ok, kBatchSize);
}

}  // namespace gcpp
