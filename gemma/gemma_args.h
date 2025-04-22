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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_ARGS_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_ARGS_H_

#include <stddef.h>
#include <stdio.h>

#include <memory>
#include <string>

#include "compression/io.h"  // Path
#include "compression/shared.h"
#include "gemma/configs.h"
#include "gemma/gemma.h"  // For CreateGemma
#include "ops/matmul.h"
#include "util/args.h"
#include "util/basics.h"  // Tristate
#include "hwy/base.h"       // HWY_ABORT

namespace gcpp {

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
            "Path name of model weights (.sbs) file.\n  Required argument.\n");
    visitor(compressed_weights, "compressed_weights", Path(),
            "Deprecated alias for --weights.");
    visitor(model_type_str, "model", std::string(),
            "Model type, see common.cc for valid values.\n");
    visitor(weight_type_str, "weight_type", std::string("sfp"),
            "Weight type\n    f32 = float, bf16 = bfloat16, sfp = 8-bit SFP.");
  }

  // Uninitialized before Validate, must call after that.
  const ModelInfo& Info() const { return info_; }

 private:
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

  int verbosity;

  size_t max_generated_tokens;

  size_t prefill_tbatch_size;
  size_t decode_qbatch_size;

  float temperature;
  size_t top_k;
  bool deterministic;
  bool multiturn;
  Path image_file;

  std::string prompt;  // Added prompt flag for non-interactive mode
  std::string eot_line;

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
    visitor(verbosity, "verbosity", 1,
            "Show verbose developer information\n    0 = only print generation "
            "output\n    1 = standard user-facing terminal ui\n    2 = show "
            "developer/debug info).\n    Default = 1.",
            1);  // Changed verbosity level to 1 since it's user-facing

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

    visitor(prompt, "prompt", std::string(""),
            "Initial prompt for non-interactive mode. When specified, "
            "generates a response"
            " and exits.",
            1);  // Added as user-facing option

    visitor(
        eot_line, "eot_line", std::string(""),
        "End of turn line. "
        "When you specify this, the prompt will be all lines "
        "before the line where only the given string appears.\n    Default = "
        "When a newline is encountered, that signals the end of the turn.",
        2);
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

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_ARGS_H_
