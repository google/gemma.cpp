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

// Argument parsing for Gemma.

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_ARGS_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_ARGS_H_

#include <stdio.h>

#include <string>

#include "compression/shared.h"
#include "gemma/common.h"
#include "gemma/gemma.h"  // For CreateGemma
#include "hwy/base.h"     // HWY_ABORT
#include "ops/matmul.h"
#include "util/args.h"
#include "util/basics.h"  // Tristate

namespace gcpp {

// Arguments related to inference: sampling, text etc.
struct InferenceArgs : public ArgsBase<InferenceArgs> {
  // Arguments for getc-like interfaces
  size_t max_tokens;
  size_t max_generated_tokens;
  float temperature;
  size_t top_k;
  float top_p;
  float min_p;
  int repeat_penalty_power;
  float repeat_penalty_presence;
  float repeat_penalty_decay;
  float repeat_penalty_range;

  // Batch configuration:
  size_t prefill_tbatch_size;
  size_t decode_tbatch_size;

  // Non-interactive mode prompt
  std::string prompt;
  std::string eot_line;

  template <class Visitor>
  void ForEach(Visitor& visitor) {
    // Each line specifies a variable member, its name, default value, and help.
    visitor(max_tokens, "max_tokens", size_t{50},
            "Maximum number of total tokens including prompt (0=no limit).", 1);
    visitor(max_generated_tokens, "max_generated_tokens", size_t{512},
            "Maximum number of generated tokens (not including prompt) (0=no "
            "limit).",
            1);
    visitor(temperature, "temperature", 1.0f,
            "Temperature (randomness) for logits.", 1);
    visitor(top_k, "top_k", size_t{40},
            "Number of highest-probability tokens to consider (0=unlimited).",
            1);
    visitor(top_p, "top_p", 0.9f, "Top-p probability threshold (0.0=disabled).",
            1);
    visitor(min_p, "min_p", 0.0f, "Min-p probability threshold (0.0=disabled).",
            1);
    visitor(
        repeat_penalty_power, "repeat_penalty_power", 1,
        "Penalty power (1=standard frequentist penalty). If 0, skips penalty "
        "computation.",
        1);
    visitor(repeat_penalty_presence, "repeat_penalty_presence", 0.0f,
            "Penalty for token presence regardless of frequency (additive).",
            1);
    visitor(repeat_penalty_decay, "repeat_penalty_decay", 0.0f,
            "Penalty for token n positions ago is decayed by "
            "power(repeat_penalty_decay, n).",
            1);
    visitor(repeat_penalty_range, "repeat_penalty_range", 8.0f,
            "Penalty fades out near the end of range (tokens)", 1);

    // Batch configuration:
    visitor(prefill_tbatch_size, "prefill_tbatch_size", size_t{2},
            "Token batch size for prefill; <= 32", 2);
    visitor(decode_tbatch_size, "decode_tbatch_size", size_t{1},
            "Token batch size for decode (only 1 currently supported)", 2);

    visitor(
        eot_line, "eot_line", std::string(""),
        "End of turn line. "
        "When you specify this, the prompt will be all lines "
        "before the line where only the given string appears.\n    Default = "
        "When a newline is encountered, that signals the end of the turn.",
        1);

    // Non-interactive mode prompt
    visitor(prompt, "prompt", std::string(""),
            "Prompt to use in non-interactive mode", 1);
  }

  const char* Validate() const {
    if (max_generated_tokens == 0 && max_tokens == 0) {
      return "At least one of max_tokens and max_generated_tokens must be > 0";
    }
    if (temperature <= 0.0) {
      return "Temperature must be > 0.0";
    }
    if (prefill_tbatch_size > 32) {
      return "prefill_tbatch_size must be <= 32";
    }
    if (decode_tbatch_size != 1) {
      return "decode_tbatch_size must be 1";
    }
    return nullptr;
  }
};

// Arguments related to model weights.
struct LoaderArgs : public ArgsBase<LoaderArgs> {
  Path model_path;  // Path to directory containing the weights
  Path tokenizer;   // Optional: can be derived from model_path
  bool model_is_gemma2;
  Gemma::Config::WeightFormat weight_format;

  template <class Visitor>
  void ForEach(Visitor& visitor) {
    // Each line specifies a variable member, its name, default value, and help.
    visitor(model_path, "model", Path{},
            "Directory containing weights or config file from `gemma.cpp "
            "convert`.",
            0);
    visitor(tokenizer, "tokenizer", Path{},
            "Optional path to tokenizer.model; if empty, looks in model_path.",
            2);
    visitor(model_is_gemma2, "model_is_gemma2", false,
            "Whether the model is a Gemma 2 model", 1);
    visitor(weight_format, "format", Gemma::Config::kBfloat16,
            "Model weights format: 0=F32, 1=F16, 2=BF16", 2);
  }

  const char* Validate() const {
    if (model_path.path.empty()) {
      return "Empty model path";
    }
    if (weight_format != Gemma::Config::kBfloat16 &&
        weight_format != Gemma::Config::kFloat16 &&
        weight_format != Gemma::Config::kFloat32) {
      return "Invalid weight format";
    }
    return nullptr;
  }
};

// Threading-related arguments.
struct ThreadingArgs : public ArgsBase<ThreadingArgs> {
  size_t num_threads;
  Tristate pin_threads;
  Tristate use_spinning;
  int verbosity;

  template <class Visitor>
  void ForEach(Visitor& visitor) {
    visitor(num_threads, "threads", size_t{0},
            "Number of threads (0=auto, half of logical cores)", 1);
    visitor(pin_threads, "pin_threads", Tristate::kDefault,
            "Set to true/false to force enable/disable thread pinning.", 2);
    visitor(use_spinning, "use_spinning", Tristate::kDefault,
            "Set to true/false to enable/disable thread spinning (typically "
            "improves "
            "performance but increases power usage)",
            2);
    visitor(verbosity, "verbosity", 1,
            "Controls printing of progress messages to stderr", 1);
  }

  // Returns nullptr if OK, otherwise error message.
  const char* Validate() const { return nullptr; }

  // Returns num_threads to use.
  size_t NumThreadsToUse() const {
    return num_threads == 0 ? (size_t{hwy::NumberOfProcessors()} + 1) / 2
                            : num_threads;
  }
};

// Command-line arguments for PeftGemma and Gemma.
struct GemmaArgs : public ArgsBase<GemmaArgs> {
  InferenceArgs inference;
  LoaderArgs loader;
  ThreadingArgs threading;
  // For collect_stats.cc:
  Path output;

  bool trace_outputs;  // For -ftrace and dump_csv.cc
  bool trace_base;     // For -ftrace
  int time_it;         // For time_it.cc

  template <class Visitor>
  void ForEach(Visitor& visitor) {
    inference.ForEach(visitor);
    loader.ForEach(visitor);
    threading.ForEach(visitor);

    visitor(output, "output", Path{}, "Where to write CSV data / stats", 2);
    visitor(trace_outputs, "trace_outputs", false, "For tracing", 2);
    visitor(trace_base, "trace_base", false, "For tracing", 2);
    visitor(time_it, "time_it", 0, "For benchmarks", 2);
  }
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_ARGS_H_