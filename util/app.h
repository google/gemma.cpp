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
#include "compression/shared.h"
#include "gemma/common.h"
#include "gemma/gemma.h"  // For CreateGemma
#include "ops/matmul.h"
#include "util/args.h"
#include "util/basics.h"  // Tristate
#include "util/threading.h"
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

  size_t max_threads;  // divided among the detected clusters
  Tristate pin;        // pin threads?
  Tristate spin;       // use spin waits?

  // For BoundedSlice:
  size_t skip_packages;
  size_t max_packages;
  size_t skip_clusters;
  size_t max_clusters;
  size_t skip_lps;
  size_t max_lps;

  std::string eot_line;

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(verbosity, "verbosity", 1,
            "Show verbose developer information\n    0 = only print generation "
            "output\n    1 = standard user-facing terminal ui\n    2 = show "
            "developer/debug info).\n    Default = 1.",
            2);

    // The exact meaning is more subtle: see the comment at NestedPools ctor.
    visitor(max_threads, "num_threads", size_t{0},
            "Maximum number of threads to use; default 0 = unlimited.", 2);
    visitor(pin, "pin", Tristate::kDefault,
            "Pin threads? -1 = auto, 0 = no, 1 = yes.", 2);
    visitor(spin, "spin", Tristate::kDefault,
            "Use spin waits? -1 = auto, 0 = no, 1 = yes.", 2);
    // These can be used to partition CPU sockets/packages and their
    // clusters/CCXs across several program instances. The default is to use
    // all available resources.
    visitor(skip_packages, "skip_packages", size_t{0},
            "Index of the first socket to use; default 0 = unlimited.", 2);
    visitor(max_packages, "max_packages", size_t{0},
            "Maximum number of sockets to use; default 0 = unlimited.", 2);
    visitor(skip_clusters, "skip_clusters", size_t{0},
            "Index of the first CCX to use; default 0 = unlimited.", 2);
    visitor(max_clusters, "max_clusters", size_t{0},
            "Maximum number of CCXs to use; default 0 = unlimited.", 2);
    // These are only used when CPU topology is unknown.
    visitor(skip_lps, "skip_lps", size_t{0},
            "Index of the first LP to use; default 0 = unlimited.", 2);
    visitor(max_lps, "max_lps", size_t{0},
            "Maximum number of LPs to use; default 0 = unlimited.", 2);

    visitor(
        eot_line, "eot_line", std::string(""),
        "End of turn line. "
        "When you specify this, the prompt will be all lines "
        "before the line where only the given string appears.\n    Default = "
        "When a newline is encountered, that signals the end of the turn.",
        2);
  }
};

static inline BoundedTopology CreateTopology(const AppArgs& app) {
  return BoundedTopology(BoundedSlice(app.skip_packages, app.max_packages),
                         BoundedSlice(app.skip_clusters, app.max_clusters),
                         BoundedSlice(app.skip_lps, app.max_lps));
}
static inline NestedPools CreatePools(const BoundedTopology& topology,
                                      const AppArgs& app) {
  Allocator::Init(topology);
  return NestedPools(topology, app.max_threads, app.pin);
}

struct LoaderArgs : public ArgsBase<LoaderArgs> {
  LoaderArgs(int argc, char* argv[], bool validate = true) {
    InitAndParse(argc, argv);

    if (validate) {
      if (const char* error = Validate()) {
        HWY_ABORT("Invalid args: %s", error);
      }
    }
  }
  LoaderArgs(const std::string& tokenizer_path, const std::string& weights_path,
             const std::string& model, bool validate = true) {
    Init();  // Init sets to defaults, so assignments must come after Init().
    tokenizer.path = tokenizer_path;
    weights.path = weights_path;
    model_type_str = model;

    if (validate) {
      if (const char* error = Validate()) {
        HWY_ABORT("Invalid args: %s", error);
      }
    }
  };

  // Returns error string or nullptr if OK.
  const char* Validate() {
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
    info_.model = Model::UNKNOWN;
    info_.wrapping = PromptWrapping::GEMMA_PT;
    info_.weight = Type::kUnknown;
    if (!model_type_str.empty()) {
      const char* err = ParseModelTypeAndWrapping(model_type_str, info_.model,
                                                  info_.wrapping);
      if (err != nullptr) return err;
    }
    if (!weight_type_str.empty()) {
      const char* err = ParseType(weight_type_str, info_.weight);
      if (err != nullptr) return err;
    }
    if (!tokenizer.path.empty()) {
      if (!tokenizer.Exists()) {
        return "Can't open file specified with --tokenizer flag.";
      }
    }
    // model_type and tokenizer must be either both present or both absent.
    // Further checks happen on weight loading.
    if (model_type_str.empty() != tokenizer.path.empty()) {
      return "Missing or extra flags for model_type or tokenizer.";
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
            "Path name of tokenizer model file.");
    visitor(weights, "weights", Path(),
            "Path name of model weights (.sbs) file.\n    Required argument.");
    visitor(compressed_weights, "compressed_weights", Path(),
            "Alias for --weights.");
    visitor(model_type_str, "model", std::string(),
            "Model type\n    2b-it = 2B parameters, instruction-tuned\n    "
            "2b-pt = 2B parameters, pretrained\n    7b-it = 7B parameters "
            "instruction-tuned\n    7b-pt = 7B parameters, pretrained\n    "
            "gr2b-it = griffin 2B parameters, instruction-tuned\n    "
            "gr2b-pt = griffin 2B parameters, pretrained.");
    visitor(weight_type_str, "weight_type", std::string("sfp"),
            "Weight type\n    f32 = float, bf16 = bfloat16, sfp = 8-bit SFP.");
  }

  // Uninitialized before Validate, must call after that.
  const ModelInfo& Info() const { return info_; }

 private:
  // TODO(rays): remove this. Eventually ModelConfig will be loaded from the
  // weights file, so we can remove the need for this struct entirely.
  ModelInfo info_;
};

// `env` must remain valid for the lifetime of the Gemma.
static inline Gemma CreateGemma(const LoaderArgs& loader, MatMulEnv& env) {
  if (Type::kUnknown == loader.Info().weight ||
      Model::UNKNOWN == loader.Info().model || loader.tokenizer.path.empty()) {
    // New weights file format doesn't need tokenizer path or model/weightinfo.
    return Gemma(loader.weights, env);
  }
  return Gemma(loader.tokenizer, loader.weights, loader.Info(), env);
}

// `env` must remain valid for the lifetime of the Gemma.
static inline std::unique_ptr<Gemma> AllocateGemma(const LoaderArgs& loader,
                                                   MatMulEnv& env) {
  if (Type::kUnknown == loader.Info().weight ||
      Model::UNKNOWN == loader.Info().model || loader.tokenizer.path.empty()) {
    // New weights file format doesn't need tokenizer path or model/weight info.
    return std::make_unique<Gemma>(loader.weights, env);
  }
  return std::make_unique<Gemma>(loader.tokenizer, loader.weights,
                                 loader.Info(), env);
}

struct InferenceArgs : public ArgsBase<InferenceArgs> {
  InferenceArgs(int argc, char* argv[]) { InitAndParse(argc, argv); }
  InferenceArgs() { Init(); };

  size_t max_generated_tokens;

  size_t prefill_tbatch_size;
  size_t decode_qbatch_size;

  float temperature;
  size_t top_k;
  bool deterministic;
  bool multiturn;
  Path image_file;

  // Returns error string or nullptr if OK.
  const char* Validate() const {
    if (max_generated_tokens > gcpp::kSeqLen) {
      return "max_generated_tokens is larger than the maximum sequence length "
             "(see configs.h).";
    }
    return nullptr;
  }

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(max_generated_tokens, "max_generated_tokens", size_t{2048},
            "Maximum number of tokens to generate.");

    visitor(prefill_tbatch_size, "prefill_tbatch", size_t{256},
            "Prefill: max tokens per batch.");
    visitor(decode_qbatch_size, "decode_qbatch", size_t{16},
            "Decode: max queries per batch.");

    visitor(temperature, "temperature", 1.0f, "Temperature for top-K", 2);
    visitor(top_k, "top_k", size_t{1}, "Number of top-K tokens to sample from",
            2);
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
    runtime_config.max_generated_tokens = max_generated_tokens;
    runtime_config.prefill_tbatch_size = prefill_tbatch_size;
    runtime_config.decode_qbatch_size = decode_qbatch_size;
    if (prefill_tbatch_size > MMStorage::kMaxM) {
      HWY_ABORT(
          "prefill_tbatch_size %zu > kMaxM %zu: specify a smaller value, "
          "or increase the constant in MMStorage.\n",
          prefill_tbatch_size, MMStorage::kMaxM);
    }
    if (decode_qbatch_size > MMStorage::kMaxM) {
      HWY_ABORT(
          "decode_qbatch_size %zu > kMaxM %zu: specify a smaller value, "
          "or increase the constant in MMStorage.\n",
          decode_qbatch_size, MMStorage::kMaxM);
    }

    runtime_config.temperature = temperature;
    runtime_config.top_k = top_k;
  }
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_APP_H_
