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

#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

// Compiles this file for multiple architectures via "foreach_target.h", to
// which we pass the filename via macro 'argument'.
// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "evals/cross_entropy.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "ops/ops-inl.h"  // Softmax

#ifndef GEMMA_CROSS_ENTROPY_ONCE
#define GEMMA_CROSS_ENTROPY_ONCE

#include <stddef.h>
#include <stdio.h>

#include <algorithm>  // std::sort
#include <cmath>
#include <regex>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#include "evals/cross_entropy.h"
#include "gemma/gemma.h"
#include "hwy/base.h"

namespace gcpp {

namespace {

static std::string TokenString(const GemmaTokenizer& tokenizer, int token) {
  std::string token_str;
  tokenizer.Decode({token}, &token_str);
  return "'" + std::regex_replace(token_str, std::regex("\n"), "\\n") + "'";
}

void LogTopK(const GemmaTokenizer& tokenizer, const float* dist, size_t len,
             size_t k) {
  std::vector<std::pair<float, int>> sorted(len);
  for (size_t i = 0; i < len; ++i) {
    sorted[i] = std::make_pair(dist[i], static_cast<int>(i));
  }
  std::sort(sorted.begin(), sorted.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
              if (a.first != b.first) {
                return a.first > b.first;
              }
              return a.second < b.second;
            });
  for (size_t i = 0; i < k; ++i) {
    printf("  [#%-2d token %6d = %-12s  %.2e]\n", static_cast<int>(i + 1),
           sorted[i].second, TokenString(tokenizer, sorted[i].second).c_str(),
           sorted[i].first);
  }
}
}  // namespace
}  // namespace gcpp
#endif  // GEMMA_CROSS_ENTROPY_ONCE

// SIMD code, compiled once per target.
HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

void CallSoftmax(float* HWY_RESTRICT logits, size_t vocab_size,
                 hwy::Profiler& p) {
  Softmax(logits, vocab_size, p, hwy::Profiler::Thread());
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {

HWY_EXPORT(CallSoftmax);

float ComputeCrossEntropy(const Gemma& gemma, size_t max_generated_tokens,
                          const std::vector<int>& prompt, KVCache& kv_cache,
                          MatMulEnv& env, int verbosity) {
  const StreamFunc stream_token = [](int, float) { return true; };

  const int vocab_size = gemma.Config().vocab_size;
  float cross_entropy = std::log(vocab_size);  // first token; == -log(1/v_s)
  size_t pos = 1;

  const SampleFunc sample_token = [&](float* probs,
                                      size_t vocab_size) -> TokenAndProb {
    // input is logits, not yet probabilities
    HWY_DYNAMIC_DISPATCH(CallSoftmax)(probs, vocab_size, env.ctx.profiler);
    // We are called for each token, but pos starts at 1. Clamping
    // max_generated_tokens to prompt.size() should prevent overrun.
    HWY_ASSERT(pos < prompt.size());
    const int token = prompt[pos];
    const float prob = probs[token];
    cross_entropy -= std::max(std::log(prob), -64.0f);

    if (verbosity >= 4) {
      LogTopK(gemma.Tokenizer(), probs, vocab_size, 10);
    }
    if (verbosity >= 3) {
      printf("pos %4zu token %6d = %-12s  %.10e  %14.10f bits\n", pos, token,
             TokenString(gemma.Tokenizer(), token).c_str(), prob,
             -std::log(prob) / std::log(2.0));
    }
    if (verbosity >= 2 && pos % 100 == 99) {
      printf("Processed %zu tokens, cross-entropy per token: %f\n", pos + 1,
             cross_entropy / std::log(2.0) / (pos + 1));
    }
    ++pos;
    return TokenAndProb{.token = token, .prob = prob};
  };

  std::vector<int> prompt0 = { prompt[0] };
  max_generated_tokens = HWY_MIN(max_generated_tokens, prompt.size());
  RuntimeConfig runtime = {
      .max_generated_tokens = max_generated_tokens - 1,
      .temperature = 0.0f,
      .gen = nullptr,
      .verbosity = verbosity,
      .stream_token = stream_token,
      .sample_func = sample_token,
  };
  TimingInfo timing_info;

  gemma.Generate(runtime, prompt0, 0, kv_cache, env, timing_info);

  const float scale = 1.0f / std::log(2.0f);
  return cross_entropy * scale;
}

}  // namespace gcpp
#endif  // HWY_ONCE
