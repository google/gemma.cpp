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

#include <stddef.h>
#include <stdio.h>

#include <memory>
#include <string>

#include "compression/io.h"  // Path
#include "gemma/common.h"
#include "gemma/gemma.h"  // For CreateGemma
#include "util/args.h"
#include "hwy/base.h"  // HWY_IS_ASAN

namespace gcpp {

static inline const char* CompiledConfig() {
  if (HWY_IS_ASAN) {
    return "asan";
  } else if (HWY_IS_MSAN) {
    return "msan";
  } else if (HWY_IS_TSAN) {
    return "tsan";
  } else if (HWY_IS_HWASAN) {
    return "hwasan";
  } else if (HWY_IS_UBSAN) {
    return "ubsan";
  } else if (HWY_IS_DEBUG_BUILD) {
    return "dbg";
  } else {
    return "opt";
  }
}

class AppArgs : public ArgsBase<AppArgs> {
 public:
  AppArgs(int argc, char* argv[]) { InitAndParse(argc, argv); }
  AppArgs() { Init(); };

  int verbosity;

  size_t num_threads;  // divided among the detected clusters
  size_t max_clusters;
  int pin;  // -1 = auto, 0 = no, 1 = yes

  std::string eot_line;

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(verbosity, "verbosity", 1,
            "Show verbose developer information\n    0 = only print generation "
            "output\n    1 = standard user-facing terminal ui\n    2 = show "
            "developer/debug info).\n    Default = 1.",
            2);

    visitor(num_threads, "num_threads", size_t{0},
            "Maximum number of threads to use; default 0 = unlimited.", 2);
    visitor(max_clusters, "max_clusters", size_t{0},
            "Maximum number of sockets/CCXs to use; default 0 = unlimited.", 2);
    visitor(pin, "pin", -1, "Pin threads? -1 = auto, 0 = no, 1 = yes.", 2);

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
  LoaderArgs(const std::string& tokenizer_path, const std::string& weights_path,
             const std::string& model) {
    Init();  // Init sets to defaults, so assignments must come after Init().
    tokenizer.path = tokenizer_path;
    weights.path = weights_path;
    model_type_str = model;
  };

  // Returns error string or nullptr if OK.
  const char* Validate() {
    if (const char* err = ParseModelTypeAndTraining(model_type_str, info_.model,
                                                    info_.training)) {
      return err;
    }
    if (const char* err = ParseType(weight_type_str, info_.weight)) {
      return err;
    }
    if (tokenizer.path.empty()) {
      return "Missing --tokenizer flag, a file for the tokenizer is required.";
    }
    if (!tokenizer.Exists()) {
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
    if (!weights.Exists()) {
      return "Can't open file specified with --weights flag.";
    }
    return nullptr;
  }

  Path tokenizer;
  Path weights;  // weights file location
  Path compressed_weights;
  std::string model_type_str;
  std::string weight_type_str;

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
    visitor(weight_type_str, "weight_type", std::string("sfp"),
            "Weight type\n    f32 = float, bf16 = bfloat16, sfp = 8-bit FP\n"
            "    Required argument.");
  }

  // Uninitialized before Validate, must call after that.
  const ModelInfo& Info() const { return info_; }

 private:
  ModelInfo info_;
};

static inline Gemma CreateGemma(const LoaderArgs& loader,
                                PerClusterPools& pools) {
  return Gemma(loader.tokenizer, loader.weights, loader.Info(), pools);
}

static inline std::unique_ptr<Gemma> AllocateGemma(const LoaderArgs& loader,
                                                   PerClusterPools& pools) {
  return std::make_unique<Gemma>(loader.tokenizer, loader.weights,
                                 loader.Info(), pools);
}

struct InferenceArgs : public ArgsBase<InferenceArgs> {
  InferenceArgs(int argc, char* argv[]) { InitAndParse(argc, argv); }
  InferenceArgs() { Init(); };

  size_t max_tokens;
  size_t max_generated_tokens;

  size_t prefill_tbatch_size;
  size_t decode_qbatch_size;

  float temperature;
  bool deterministic;
  bool multiturn;
  Path image_file;

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

    visitor(prefill_tbatch_size, "prefill_tbatch", size_t{64},
            "Prefill: max tokens per batch.");
    visitor(decode_qbatch_size, "decode_qbatch", size_t{16},
            "Decode: max queries per batch.");

    visitor(temperature, "temperature", 1.0f, "Temperature for top-K", 2);
    visitor(deterministic, "deterministic", false,
            "Make top-k sampling deterministic", 2);
    visitor(multiturn, "multiturn", false,
            "Multiturn mode\n    0 = clear KV cache after every "
            "interaction\n    1 = continue KV cache after every interaction\n  "
            "  Default : 0 (conversation "
            "resets every turn)");
    visitor(image_file, "image_file", Path(), "Image file to load.");
  }

  void CopyTo(RuntimeConfig& runtime_config) const {
    runtime_config.max_tokens = max_tokens;
    runtime_config.max_generated_tokens = max_generated_tokens;

    runtime_config.prefill_tbatch_size = prefill_tbatch_size;
    runtime_config.decode_qbatch_size = decode_qbatch_size;

    runtime_config.temperature = temperature;
  }
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_APP_H_
