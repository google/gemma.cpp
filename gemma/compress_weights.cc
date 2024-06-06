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

// Command line tool to create compressed weights.

// Compiles this file for multiple architectures via "foreach_target.h", to
// which we pass the filename via macro 'argument'.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE \
  "gemma/compress_weights.cc"  // NOLINT
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// Must come after foreach_target.h to avoid redefinition errors.
#include "compression/compress-inl.h"
#include "hwy/highway.h"

#ifndef GEMMA_COMPRESS_WEIGHTS_ONCE
#define GEMMA_COMPRESS_WEIGHTS_ONCE

#include <stddef.h>
#include <stdio.h>

#include <algorithm>  // std::clamp
#include <iostream>
#include <string>
#include <thread>  // NOLINT

#include "compression/io.h"  // Path
#include "gemma/common.h"    // Model
#include "gemma/weights.h"
#include "util/args.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

struct Args : public ArgsBase<Args> {
  static constexpr size_t kDefaultNumThreads = ~size_t{0};

  void ChooseNumThreads() {
    if (num_threads == kDefaultNumThreads) {
      // This is a rough heuristic, replace with something better in the future.
      num_threads = static_cast<size_t>(std::clamp(
          static_cast<int>(std::thread::hardware_concurrency()) - 2, 1, 18));
    }
  }

 public:
  Args(int argc, char* argv[]) {
    InitAndParse(argc, argv);
    ChooseNumThreads();
  }

  gcpp::Model ModelType() const { return model_type; }

  // Returns error string or nullptr if OK.
  const char* Validate() {
    ModelTraining model_training;
    const char* parse_result =
        ParseModelTypeAndTraining(model_type_str, model_type, model_training);
    if (parse_result) return parse_result;
    if (weights.path.empty()) {
      return "Missing --weights flag, a file for the uncompressed model.";
    }
    if (compressed_weights.path.empty()) {
      return "Missing --compressed_weights flag, a file for the compressed "
             "model.";
    }
    if (!weights.Exists()) {
      return "Can't open file specified with --weights flag.";
    }
    return nullptr;
  }

  Path weights;             // uncompressed weights file location
  Path compressed_weights;  // compressed weights file location
  std::string model_type_str;
  Model model_type;
  size_t num_threads;

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(weights, "weights", Path(),
            "Path to model weights (.bin) file.\n"
            "    Required argument.");
    visitor(model_type_str, "model", std::string(),
            "Model type\n    2b-it = 2B parameters, instruction-tuned\n    "
            "2b-pt = 2B parameters, pretrained\n    7b-it = 7B parameters "
            "instruction-tuned\n    7b-pt = 7B parameters, pretrained\n    "
            "gr2b-it = griffin 2B parameters, instruction-tuned\n    "
            "gr2b-pt = griffin 2B parameters, pretrained\n    "
            "    Required argument.");
    visitor(compressed_weights, "compressed_weights", Path(),
            "Path name where compressed weights (.sbs) file will be written.\n"
            "    Required argument.");
    visitor(num_threads, "num_threads",
            kDefaultNumThreads,  // see ChooseNumThreads
            "Number of threads to use.\n    Default = Estimate of the "
            "number of suupported concurrent threads.",
            2);
  }
};

void ShowHelp(gcpp::Args& args) {
  std::cerr
      << "Usage:\n./compress_weights --weights <path to uncompressed weights> "
         " --model <model type> --compressed_weights <output path>\n";
  std::cerr << "\n*Arguments*\n\n";
  args.Help();
  std::cerr << "\n";
}

}  // namespace gcpp
#endif  // GEMMA_COMPRESS_WEIGHTS_ONCE

// SIMD code, compiled once per target.
HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

template <class TConfig>
void CompressWeights(const Path& weights_path,
                     const Path& compressed_weights_path, Model model_type,
                     hwy::ThreadPool& pool) {
  if (!weights_path.Exists()) {
    HWY_ABORT("The model weights file '%s' does not exist.",
              weights_path.path.c_str());
  }

  // Allocate compressed weights.
  using CWeights = CompressedWeights<TConfig>;
  ByteStorageT c_weights_u8 = AllocateSizeof<CWeights>();
  CWeights* c_weights = reinterpret_cast<CWeights*>(c_weights_u8.get());
  new (&c_weights->c_layer_ptrs) CompressedLayerPointers<TConfig>(pool);

  // Get weights, compress, and store.
  const bool scale_for_compression = TConfig::kNumTensorScales > 0;
  const ByteStorageT weights_u8 = gcpp::LoadRawWeights(
      weights_path, model_type, pool, scale_for_compression);
  WeightsF<TConfig>* weights =
      reinterpret_cast<WeightsF<TConfig>*>(weights_u8.get());
  Compressor compressor(pool);
  ForEachTensor<TConfig>(weights, *c_weights, compressor);
  compressor.AddScales(weights->scales.data(), weights->scales.size());
  compressor.WriteAll(pool, compressed_weights_path);

  weights->layer_ptrs.~LayerPointers<float, TConfig>();
  c_weights->c_layer_ptrs.~CompressedLayerPointers<TConfig>();
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {

void Run(Args& args) {
  hwy::ThreadPool pool(args.num_threads);
  const Model model_type = args.ModelType();
  GEMMA_EXPORT_AND_DISPATCH_MODEL(
      model_type, CompressWeights,
      (args.weights, args.compressed_weights, model_type, pool));
}

}  // namespace gcpp

int main(int argc, char** argv) {
  gcpp::Args args(argc, argv);

  if (gcpp::HasHelp(argc, argv)) {
    gcpp::ShowHelp(args);
    return 0;
  }

  if (const char* error = args.Validate()) {
    gcpp::ShowHelp(args);
    HWY_ABORT("\nInvalid args: %s", error);
  }

  gcpp::Run(args);

  return 0;
}

#endif  // HWY_ONCE
