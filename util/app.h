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

// Shared between various frontends.

#ifndef THIRD_PARTY_GEMMA_CPP_UTIL_APP_H_
#define THIRD_PARTY_GEMMA_CPP_UTIL_APP_H_

#include <iterator>
#if HWY_OS_LINUX
#include <sched.h>

#include <cctype>
#include <cerrno>  // IDE does not recognize errno.h as providing errno.
#include <string>
#endif  // HWY_OS_LINUX
#include <stddef.h>
#include <stdio.h>

#include <algorithm>  // std::clamp
#include <thread>     // NOLINT>

// copybara:import_next_line:gemma_cpp
#include "configs.h"
// copybara:import_next_line:gemma_cpp
#include "gemma.h"
#include "hwy/base.h"  // HWY_ASSERT
// copybara:import_next_line:gemma_cpp
#include "util/args.h"

namespace gcpp {

static inline const char* CompiledConfig() {
  if (HWY_IS_ASAN) {
    return "asan";
  } else if (HWY_IS_MSAN) {
    return "msan";
  } else if (HWY_IS_TSAN) {
    return "tsan";
#if defined(HWY_IS_UBSAN)
  } else if (HWY_IS_UBSAN) {
    return "ubsan";
#endif
  } else if (HWY_IS_DEBUG_BUILD) {
    return "dbg";
  } else {
    return "opt";
  }
}

static inline void PinThreadToCore(size_t cpu_index) {
#if HWY_OS_LINUX
  // Forces the thread to run on the logical processor with the same number.
  cpu_set_t cset;             // bit array
  CPU_ZERO(&cset);            // clear all
  CPU_SET(cpu_index, &cset);  // set bit indicating which processor to run on.
  const int err = sched_setaffinity(0, sizeof(cset), &cset);
  if (err != 0) {
    fprintf(stderr,
            "sched_setaffinity returned %d, errno %d. Can happen if running in "
            "a container; this warning is safe to ignore.\n",
            err, errno);
  }
#else
  (void)cpu_index;
#endif
}

class AppArgs : public ArgsBase<AppArgs> {
  static constexpr size_t kDefaultNumThreads = ~size_t{0};

  void ChooseNumThreads() {
    if (num_threads == kDefaultNumThreads) {
      // This is a rough heuristic, replace with something better in the future.
      num_threads = static_cast<size_t>(std::clamp(
          static_cast<int>(std::thread::hardware_concurrency()) - 2, 1, 18));
    }
  }

 public:
  AppArgs(int argc, char* argv[]) {
    InitAndParse(argc, argv);
    ChooseNumThreads();
  }

  Path log;  // output
  int verbosity;
  size_t num_threads;
  std::string eot_line;

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(verbosity, "verbosity", 1,
            "Show verbose developer information\n    0 = only print generation "
            "output\n    1 = standard user-facing terminal ui\n    2 = show "
            "developer/debug info).\n    Default = 1.",
            2);
    visitor(num_threads, "num_threads",
            kDefaultNumThreads,  // see ChooseNumThreads
            "Number of threads to use.\n    Default = Estimate of the "
            "number of suupported concurrent threads.",
            2);
    visitor(
        eot_line, "eot_line", std::string(""),
        "End of turn line. "
        "When you specify this, the prompt will be all lines "
        "before the line where only the given string appears.\n    Default = "
        "When a newline is encountered, that signals the end of the turn.",
        2);
  }
};

struct LoaderArgs : public ArgsBase<LoaderArgs> {
  LoaderArgs(int argc, char* argv[]) { InitAndParse(argc, argv); }

  gcpp::Model ModelType() const { return model_type; }

  gcpp::ModelTraining ModelTraining() const { return model_training; }

  // Returns error string or nullptr if OK.
  const char* Validate() {
    const char* parse_result =
        ParseModelTypeAndTraining(model_type_str, model_type, model_training);
    if (parse_result) return parse_result;
    if (tokenizer.path.empty()) {
      return "Missing --tokenizer flag, a file for the tokenizer is required.";
    }
    if (!tokenizer.exists()) {
      return "Can't open file specified with --tokenizer flag.";
    }
    if (!compressed_weights.path.empty()) {
      if (weights.path.empty()) {
        weights = compressed_weights;
      } else {
        return "Only one of --weights and --compressed_weights can be "
               "specified. To create compressed weights use the "
               "compress_weights tool.";
      }
    }
    if (weights.path.empty()) {
      return "Missing --weights flag, a file for the model weights.";
    }
    if (!weights.exists()) {
      return "Can't open file specified with --weights flag.";
    }
    return nullptr;
  }

  Path tokenizer;
  Path weights;  // weights file location
  Path compressed_weights;
  std::string model_type_str;
  Model model_type;
  enum ModelTraining model_training;

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(tokenizer, "tokenizer", Path(),
            "Path name of tokenizer model file.\n    Required argument.");
    visitor(weights, "weights", Path(),
            "Path name of model weights (.sbs) file.\n    Required argument.");
    visitor(compressed_weights, "compressed_weights", Path(),
            "Alias for --weights.");
    visitor(model_type_str, "model", std::string(),
            "Model type\n    2b-it = 2B parameters, instruction-tuned\n    "
            "2b-pt = 2B parameters, pretrained\n    7b-it = 7B parameters "
            "instruction-tuned\n    7b-pt = 7B parameters, pretrained\n    "
            "gr2b-it = griffin 2B parameters, instruction-tuned\n    "
            "gr2b-pt = griffin 2B parameters, pretrained\n    "
            "    Required argument.");
  }
};

struct InferenceArgs : public ArgsBase<InferenceArgs> {
  InferenceArgs(int argc, char* argv[]) { InitAndParse(argc, argv); }

  size_t max_tokens;
  size_t max_generated_tokens;

  float temperature;
  bool deterministic;
  bool multiturn;

  // Returns error string or nullptr if OK.
  const char* Validate() const {
    if (max_tokens > gcpp::kSeqLen) {
      return "max_tokens is larger than the maximum sequence length (see "
             "configs.h).";
    }
    if (max_generated_tokens > max_tokens) {
      return "Maximum number of generated tokens is larger than the maximum "
             "total tokens.";
    }
    return nullptr;
  }

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(max_tokens, "max_tokens", size_t{3072},
            "Maximum number of tokens in prompt + generation.");
    visitor(max_generated_tokens, "max_generated_tokens", size_t{2048},
            "Maximum number of tokens to generate.");

    visitor(temperature, "temperature", 1.0f, "Temperature for top-K", 2);
    visitor(deterministic, "deterministic", false,
            "Make top-k sampling deterministic", 2);
    visitor(multiturn, "multiturn", false,
            "Multiturn mode\n    0 = clear KV cache after every "
            "interaction\n    1 = continue KV cache after every interaction\n  "
            "  Default : 0 (conversation "
            "resets every turn)");
  }
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_APP_H_
