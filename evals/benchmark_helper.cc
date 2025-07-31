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

#include "evals/benchmark_helper.h"

#include <stdio.h>
#include <time.h>

#include <iostream>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#include "compression/types.h"  // TypeName
#include "evals/cross_entropy.h"
#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "ops/matmul.h"  // MatMulEnv
#include "util/threading_context.h"
#include "hwy/highway.h"
#include "hwy/per_target.h"  // DispatchedTarget
#include "hwy/profiler.h"    // PROFILER_ENABLED
#include "hwy/timer.h"

namespace gcpp {

void InitGenerator(const InferenceArgs& inference, std::mt19937& gen) {
  if (inference.deterministic) {
    // Nothing up my sleeve number, at least some upper bits set.
    gen.seed(0x12345678);
  } else {
    // Depending on the library implementation, this may still be deterministic.
    std::random_device rd;  // NOLINT
    gen.seed(rd());
  }
}

GemmaEnv::GemmaEnv(const LoaderArgs& loader, const ThreadingArgs& threading,
                   const InferenceArgs& inference)
    : ctx_(threading), env_(ctx_), gemma_(loader, inference, ctx_) {
  const ModelConfig& config = gemma_.Config();
  // Only allocate one for starters because GenerateBatch might not be called.
  kv_caches_.push_back(KVCache(config, inference, ctx_.allocator));

  if (inference.verbosity >= 2) {
    ShowConfig(loader, threading, inference, config, gemma_.WeightReadMode(),
               ctx_);
  }

  InitGenerator(inference, gen_);

  runtime_config_ = {
      .max_generated_tokens = inference.max_generated_tokens,
      .temperature = inference.temperature,
      .gen = &gen_,
      .verbosity = inference.verbosity,
  };
  inference.CopyTo(runtime_config_);
}

GemmaEnv::GemmaEnv(int argc, char** argv)
    : GemmaEnv(LoaderArgs(argc, argv), ThreadingArgs(argc, argv),
               InferenceArgs(argc, argv)) {}

QueryResult GemmaEnv::QueryModel(const std::vector<int>& tokens) {
  QueryResult result;

  const BatchStreamFunc batch_stream_token =
      [&result, &tokens, this](size_t /*query_index*/, size_t /*pos*/,
                               int token, float /*score*/) {
        ++result.tokens_generated;
        result.response += StringFromTokens(std::vector<int>{token});
        if (result.tokens_generated == tokens.size()) {
          result.response_start_pos = result.response.size();
        }
        return true;
      };
  if (runtime_config_.verbosity >= 2) {
    std::cout << "max generated tokens: "
              << runtime_config_.max_generated_tokens
              << "\ttemperature: " << runtime_config_.temperature << "\n";
  }
  gcpp::TimingInfo timing_info { .verbosity = runtime_config_.verbosity };
  runtime_config_.batch_stream_token = batch_stream_token;
  gemma_.Generate(runtime_config_, tokens, /*start_pos=*/0, kv_caches_[0], env_,
                  timing_info);
  return result;
}

void GemmaEnv::QueryModel(
    const std::vector<int>& tokens, const StreamFunc& stream_token) {
  gcpp::TimingInfo timing_info { .verbosity = runtime_config_.verbosity };
  const StreamFunc previous_stream_token = runtime_config_.stream_token;
  runtime_config_.stream_token = stream_token;
  gemma_.Generate(runtime_config_, tokens, /*start_pos=*/0, kv_caches_[0], env_,
                  timing_info);
  runtime_config_.stream_token = previous_stream_token;
}

std::vector<QueryResult> GemmaEnv::BatchQueryModel(
    const QueriesPromptTokens& queries_prompt,
    const hwy::Span<const size_t>& prefix_end) {
  const size_t num_queries = queries_prompt.size();
  HWY_ASSERT(num_queries != 0);
  std::vector<QueryResult> res(num_queries);
  const BatchStreamFunc batch_stream_token = [&, this](const size_t query_index,
                                                       const size_t pos,
                                                       const int token, float) {
    HWY_ASSERT(query_index < num_queries);
    std::string token_text;
    HWY_ASSERT(gemma_.Tokenizer().Decode(std::vector<int>{token}, &token_text));
    res[query_index].response.append(token_text);
    HWY_ASSERT(pos == res[query_index].tokens_generated);
    res[query_index].tokens_generated += 1;
    if (res[query_index].tokens_generated ==
        queries_prompt[query_index].size()) {
      res[query_index].response_start_pos = res[query_index].response.size();
    }
    return true;
  };
  runtime_config_.batch_stream_token = batch_stream_token;
  if (runtime_config_.verbosity >= 2) {
    fprintf(stderr, "Max gen: %zu temp: %f tbatch: %zu qbatch: %zu\n",
            runtime_config_.max_generated_tokens, runtime_config_.temperature,
            runtime_config_.prefill_tbatch_size,
            runtime_config_.decode_qbatch_size);
  }

  // Ensure we have at least one KVCache per query.
  while (kv_caches_.size() < num_queries) {
    kv_caches_.push_back(
        KVCache(gemma_.Config(), gemma_.Inference(), ctx_.allocator));
  }
  const hwy::Span<KVCache> kv_caches(&kv_caches_[0], num_queries);

  gcpp::AllQueries all_queries(queries_prompt, kv_caches, prefix_end);
  gcpp::TimingInfo timing_info = {.verbosity = runtime_config_.verbosity};
  gemma_.GenerateBatch(runtime_config_, all_queries, env_, timing_info);
  return res;
}

QueryResult GemmaEnv::QueryModel(std::string& input) {
  const std::vector<int> prompt = WrapAndTokenize(input);
  return QueryModel(prompt);
}

std::vector<QueryResult> GemmaEnv::BatchQueryModel(
    const std::vector<std::string>& inputs) {
  std::vector<std::vector<int>> prompts;
  prompts.reserve(inputs.size());
  for (auto& input : inputs) {
    std::string mutable_prompt = input;
    prompts.push_back(WrapAndTokenize(mutable_prompt));
  }
  std::vector<PromptTokens> prompt_vector;
  prompt_vector.reserve(prompts.size());
  for (auto& prompt : prompts) {
    prompt_vector.push_back(PromptTokens(prompt.data(), prompt.size()));
  }
  QueriesPromptTokens prompt_span(prompt_vector.data(), prompt_vector.size());
  return BatchQueryModel(prompt_span);
}

float GemmaEnv::CrossEntropy(const std::string& input) {
  std::vector<int> prompt = Tokenize(input);
  prompt.insert(prompt.begin(), BOS_ID);
  return ComputeCrossEntropy(*GetGemma(), /*max_generated_tokens=*/3072, prompt,
                             MutableKVCache(), env_,
                             /*verbosity=*/0) /
         static_cast<int>(input.size());
}

void LogSpeedStats(double time_start, size_t total_tokens) {
  const double time_end = hwy::platform::Now();
  const double time_elapsed = time_end - time_start;
  const double tok_sec = total_tokens / time_elapsed;
  std::cout << total_tokens << " tokens in " << time_elapsed << " seconds"
            << " [" << tok_sec << " tokens / sec" << "]\n";
}

std::string CacheString() {
  const hwy::Cache* caches = hwy::DataCaches();
  if (caches == nullptr) return "cache unknown";
  char buf[200];
  // Do not print cores_sharing because that is visible from the topology.
  const int len =
      snprintf(buf, sizeof(buf), "L1 %uK=%u*%u@%u, L2 %uK=%u*%u@%u ",
               caches[1].size_kib, caches[1].sets, caches[1].bytes_per_line,
               caches[1].associativity, caches[2].size_kib, caches[2].sets,
               caches[2].bytes_per_line, caches[2].associativity);
  HWY_ASSERT(len >= 24);
  if (caches[3].size_kib != 0) {
    snprintf(buf + len, sizeof(buf) - len, "L3 %uK=%u*%u@%u",
             caches[3].size_kib, caches[3].sets, caches[3].bytes_per_line,
             caches[3].associativity);
  }
  return buf;
}

static constexpr const char* CompiledConfig() {
  if constexpr (HWY_IS_ASAN) {
    return "asan";
  } else if constexpr (HWY_IS_MSAN) {
    return "msan";
  } else if constexpr (HWY_IS_TSAN) {
    return "tsan";
  } else if constexpr (HWY_IS_HWASAN) {
    return "hwasan";
  } else if constexpr (HWY_IS_UBSAN) {
    return "ubsan";
  } else if constexpr (HWY_IS_DEBUG_BUILD) {
    return "dbg";
  } else {
    return "opt";
  }
}

void ShowConfig(const LoaderArgs& loader, const ThreadingArgs& threading,
                const InferenceArgs& inference, const ModelConfig& config,
                const WeightsPtrs::Mode weight_read_mode,
                const ThreadingContext& ctx) {
  threading.Print(inference.verbosity);
  loader.Print(inference.verbosity);
  inference.Print(inference.verbosity);
  fprintf(
      stderr, "Model                         : %s, to_bf16 %d, mmap %d => %s\n",
      config.Specifier().c_str(), static_cast<int>(loader.to_bf16),
      static_cast<int>(loader.map), WeightsPtrs::ToString(weight_read_mode));

  if (inference.verbosity >= 2) {
    time_t now = time(nullptr);
    char* dt = ctime(&now);  // NOLINT
    char cpu100[100] = "unknown";
    (void)hwy::platform::GetCpuString(cpu100);

    fprintf(stderr,
            "Date & Time                   : %s"  // dt includes \n
            "CPU                           : %s, bind %d\n"
            "CPU topology                  : %s, %s, %s\n"
            "Instruction set               : %s (%zu bits)\n"
            "Compiled config               : %s, profiler %d\n"
            "Memory MiB                    : %4zu\n",
            dt, cpu100, static_cast<int>(threading.bind),
            ctx.topology.TopologyString(), ctx.pools.PinString(),
            CacheString().c_str(), hwy::TargetName(hwy::DispatchedTarget()),
            ctx.allocator.VectorBytes() * 8, CompiledConfig(), PROFILER_ENABLED,
            ctx.allocator.TotalMiB());
  }
}

void ShowHelp(const LoaderArgs& loader, const ThreadingArgs& threading,
              const InferenceArgs& inference) {
  std::cerr
      << "\n\ngemma.cpp : a lightweight, standalone C++ inference engine\n"
         "==========================================================\n\n"
         "To run with pre-2025 weights, specify --tokenizer and --weights.\n"
         "With the single-file weights format, specify just --weights.\n";
  std::cerr << "\n*Example Usage*\n\n./gemma --tokenizer tokenizer.spm "
               "--weights gemma2-2b-it-sfp.sbs\n";
  std::cerr << "\n*Model Loading Arguments*\n\n";
  loader.Help();
  std::cerr << "\n*Threading Arguments*\n\n";
  threading.Help();
  std::cerr << "\n*Inference Arguments*\n\n";
  inference.Help();
  std::cerr << "\n";
}

}  // namespace gcpp
