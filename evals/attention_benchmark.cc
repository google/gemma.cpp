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

// Standalone benchmark measuring single-token autoregressive decoding
// generation throughput across simulated KV cache history lengths.

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "gemma/configs.h"
#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "gemma/tokenizer.h"
#include "io/io.h"
#include "util/args.h"
#include "util/mat.h"
#include "util/threading_context.h"
#include "hwy/base.h"
#include "hwy/timer.h"

namespace gcpp {

class AttentionBenchmarkArgs : public ArgsBase<AttentionBenchmarkArgs> {
 public:
  AttentionBenchmarkArgs(int argc, char* argv[], ConsumedArgs& consumed) {
    InitAndParse(argc, argv, consumed);
  }

  AttentionBenchmarkArgs() = default;

  std::string context_lengths;
  std::string generation_context_lengths;
  std::string prompt_lengths;
  std::string prefill_lengths;
  std::string benchmark_mode;
  size_t prefill_length;
  size_t generation_prefill_length;
  size_t benchmark_tokens;
  size_t benchmark_batch_size;

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(benchmark_mode, "benchmark_mode", std::string("both"),
            "Benchmark mode: 'both' (prefill and decode) or 'decode' (decode "
            "with simulated KV cache history)",
            2);
    visitor(context_lengths, "context_lengths",
            std::string("1024,8192,32768"),
            "Comma-separated list of simulated KV cache history lengths to "
            "benchmark",
            2);
    visitor(generation_context_lengths, "generation_context_lengths",
            std::string(""),
            "Alias for context_lengths", 2);
    visitor(prompt_lengths, "prompt_lengths", std::string(""),
            "Alias for context_lengths", 2);
    visitor(prefill_lengths, "prefill_lengths", std::string(""),
            "Alias for context_lengths", 2);
    visitor(prefill_length, "prefill_length", (size_t)1,
            "Number of prompt tokens to prefill before generation (default: 1)",
            2);
    visitor(generation_prefill_length, "generation_prefill_length", (size_t)0,
            "Alias for prefill_length", 2);
    visitor(benchmark_tokens, "benchmark_tokens", (size_t)32,
            "Number of decode tokens to generate", 2);
    visitor(benchmark_batch_size, "benchmark_batch_size", (size_t)1,
            "Batch size (number of queries) to benchmark", 2);
  }
};

static std::vector<std::string> SplitString(const std::string& str,
                                            char delimiter) {
  std::vector<std::string> result;
  size_t start = 0;
  size_t end = str.find(delimiter);
  while (end != std::string::npos) {
    result.push_back(str.substr(start, end - start));
    start = end + 1;
    end = str.find(delimiter, start);
  }
  result.push_back(str.substr(start));
  return result;
}

static bool ParseSizeT(const std::string& str, size_t* out) {
  if (str.empty()) return false;
  char* endptr = nullptr;
  *out = std::strtoul(str.c_str(), &endptr, 10);
  return endptr != str.c_str() && *endptr == '\0';
}

}  // namespace gcpp


namespace {

std::vector<int> GenerateSyntheticPrompt(const gcpp::Gemma& gemma,
                                         size_t length) {
  std::vector<int> tokens;
  tokens.reserve(length);
  tokens.push_back(gcpp::BOS_ID);

  int valid_token = 500;  // Arbitrary non-EOS, non-control token ID fallback.
  std::vector<int> ids;
  if (gemma.Tokenizer().Encode(" the", &ids) && !ids.empty()) {
    valid_token = ids[0];
  }
  for (size_t i = 1; i < length; ++i) {
    tokens.push_back(valid_token);
  }
  return tokens;
}

// Zero out all allocated buffers in the KV cache to ensure clean state.
void ZeroKVCache(gcpp::KVCache& kv_cache) {
  if (kv_cache.compact_local_kv_cache_ptr.HasPtr()) {
    gcpp::ZeroInit(kv_cache.compact_local_kv_cache_ptr);
  }
  if (kv_cache.compact_global_kv_cache_ptr.HasPtr()) {
    gcpp::ZeroInit(kv_cache.compact_global_kv_cache_ptr);
  }
  if (kv_cache.compact_kv_cache_ptr.HasPtr()) {
    gcpp::ZeroInit(kv_cache.compact_kv_cache_ptr);
  }
  if (kv_cache.kv_cache.HasPtr()) {
    gcpp::ZeroInit(kv_cache.kv_cache);
  }
  if (kv_cache.k_cache.HasPtr()) {
    gcpp::ZeroInit(kv_cache.k_cache);
  }
  if (kv_cache.v_cache.HasPtr()) {
    gcpp::ZeroInit(kv_cache.v_cache);
  }
}

}  // namespace

int main(int argc, char** argv) {
  gcpp::InternalInit();
  gcpp::ConsumedArgs consumed(argc, argv);
  gcpp::GemmaArgs args(argc, argv, consumed);
  gcpp::AttentionBenchmarkArgs bench_args(argc, argv, consumed);

  if (gcpp::HasHelp(argc, argv)) {
    args.Help();
    bench_args.Help();
    return 0;
  }

  consumed.AbortIfUnconsumed();

  // Instantiate model
  gcpp::ThreadingContext ctx(args.threading);
  gcpp::MatMulEnv env(ctx);
  gcpp::Gemma gemma(args, ctx);

  gcpp::RuntimeConfig runtime_config{};
  args.inference.CopyTo(runtime_config);
  runtime_config.verbosity = args.inference.verbosity;
  runtime_config.use_spinning = args.threading.spin;
  size_t decode_tokens = bench_args.benchmark_tokens;
  size_t num_queries = bench_args.benchmark_batch_size;
  if (num_queries == 0) {
    std::cerr << "Benchmark batch size must be > 0" << std::endl;
    return 1;
  }

  std::string lengths_str_arg = bench_args.context_lengths;
  if (lengths_str_arg.empty()) {
    lengths_str_arg = bench_args.generation_context_lengths;
  }
  if (lengths_str_arg.empty()) {
    lengths_str_arg = bench_args.prompt_lengths;
  }
  if (lengths_str_arg.empty()) {
    lengths_str_arg = bench_args.prefill_lengths;
  }
  std::vector<std::string> history_lengths_strs =
      gcpp::SplitString(lengths_str_arg, ',');

  size_t prefill_len = bench_args.prefill_length;
  if (prefill_len == 0 && bench_args.generation_prefill_length > 0) {
    prefill_len = bench_args.generation_prefill_length;
  }
  if (prefill_len == 0) prefill_len = 1;

  const bool is_only_decode = (bench_args.benchmark_mode == "decode");

  std::cout << "--- Benchmark: " << (is_only_decode ? "decode" : "both")
            << " ---" << std::endl;
  std::cout << "Batch Size:             " << num_queries << std::endl;
  if (is_only_decode) {
    std::cout << "Prefill Priming Tokens: " << prefill_len << std::endl;
  }
  std::cout << "Decode Tokens:          " << decode_tokens << std::endl;
  std::cout << "---------------------------------------------------------"
            << std::endl;

  for (const auto& len_str : history_lengths_strs) {
    if (len_str.empty()) continue;
    size_t target_len;
    if (!gcpp::ParseSizeT(len_str, &target_len)) {
      std::cerr << "Invalid context length: " << len_str << std::endl;
      continue;
    }

    const size_t priming_tokens = is_only_decode ? prefill_len : 0;
    const size_t total_capacity =
        target_len + priming_tokens + decode_tokens + 32;
    if (total_capacity > gemma.Config().max_seq_len) {
      const_cast<gcpp::ModelConfig&>(gemma.Config())
          .SetMaxSeqLen(total_capacity);
    }

    const size_t history_len = is_only_decode ? target_len : 0;
    const size_t prompt_tokens = is_only_decode ? prefill_len : target_len;

    std::cout << "\n"
              << (is_only_decode ? "Simulated History Length: "
                                 : "Context / Prefill Length: ")
              << target_len << " tokens\n";

    size_t original_seq_len = args.inference.seq_len;
    args.inference.seq_len = total_capacity;

    gcpp::RuntimeConfig gen_config = runtime_config;
    gen_config.max_generated_tokens = decode_tokens;
    gen_config.decode_qbatch_size = num_queries;
    gen_config.sample_func = [](size_t, size_t, gcpp::Logits,
                                size_t) -> gcpp::TokenAndProb {
      // Return an arbitrary non-EOS token ID so benchmarks never
      // terminate early.
      return gcpp::TokenAndProb{500, 1.0f};
    };

    std::vector<gcpp::KVCache> kv_caches;
    kv_caches.reserve(num_queries);
    for (size_t i = 0; i < num_queries; ++i) {
      kv_caches.emplace_back(gemma.Config(), args.inference, gen_config,
                             ctx.allocator);
      ZeroKVCache(kv_caches.back());
    }

    args.inference.seq_len = original_seq_len;

    std::vector<int> tokens = GenerateSyntheticPrompt(gemma, prompt_tokens);

    gcpp::TimingInfo timing_info{.verbosity = args.inference.verbosity};

    size_t generated = 0;
    const size_t total_prefill_tokens = num_queries * prompt_tokens;
    const double start_time = hwy::platform::Now();
    double prefill_end_time = start_time;

    auto on_token = [&]() {
      if (++generated == total_prefill_tokens) {
        prefill_end_time = hwy::platform::Now();
      }
      return true;
    };
    gen_config.stream_token = [&](int, float) { return on_token(); };
    gen_config.batch_stream_token = [&](size_t, size_t, int, float) {
      return on_token();
    };

    gcpp::AllQueries all_queries(
        tokens, /*start_pos=*/history_len, /*prefix_end=*/0,
        hwy::Span<gcpp::KVCache>(kv_caches.data(), kv_caches.size()));

    gemma.GenerateBatch(gen_config, all_queries, env, timing_info);

    const double end_time = hwy::platform::Now();
    const double prefill_seconds = prefill_end_time - start_time;
    const double decode_seconds = end_time - prefill_end_time;
    const size_t actual_decode_tokens =
        generated > total_prefill_tokens ? generated - total_prefill_tokens : 0;

    if (!is_only_decode) {
      std::cout << "Prefill:  " << total_prefill_tokens << " tokens in "
                << prefill_seconds << "s ("
                << (prefill_seconds > 0 ? total_prefill_tokens / prefill_seconds
                                        : 0.0)
                << " tok/s); TTFT: " << static_cast<int>(prefill_seconds * 1E3)
                << " ms\n";
    }
    if (decode_tokens > 0) {
      std::cout << "Decode:   " << actual_decode_tokens << " tokens in "
                << decode_seconds << "s ("
                << (decode_seconds > 0 ? actual_decode_tokens / decode_seconds
                                       : 0.0)
                << " tok/s)\n";
    }
    std::cout << "---------------------------------------------------------\n";
  }

  return 0;
}
