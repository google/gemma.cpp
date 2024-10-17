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
  "compression/compress_weights.cc"  // NOLINT
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"
#include "gemma/configs.h"

#ifndef GEMMA_COMPRESS_WEIGHTS_ONCE
#define GEMMA_COMPRESS_WEIGHTS_ONCE

#include <stddef.h>
#include <stdio.h>

#include <algorithm>  // std::clamp
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "compression/compress.h"
#include "compression/io.h"  // Path
#include "gemma/common.h"    // Model
#include "gemma/weights.h"
#include "util/allocator.h"
#include "util/args.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

namespace {

}  // namespace

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

  // Returns error string or nullptr if OK.
  const char* Validate() {
    ModelTraining model_training;
    if (const char* err = ParseModelTypeAndTraining(model_type_str, model_type_,
                                                    model_training)) {
      return err;
    }
    if (const char* err = ParseType(weight_type_str, weight_type_)) {
      return err;
    }
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
  std::string weight_type_str;
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
    visitor(weight_type_str, "weight_type", std::string("sfp"),
            "Weight type\n    f32 = float, bf16 = bfloat16, SFP = 8-bit FP\n"
            "    Required argument.");
    visitor(compressed_weights, "compressed_weights", Path(),
            "Path name where compressed weights (.sbs) file will be written.\n"
            "    Required argument.");
    visitor(num_threads, "num_threads",
            kDefaultNumThreads,  // see ChooseNumThreads
            "Number of threads to use.\n    Default = Estimate of the "
            "number of supported concurrent threads.",
            2);
  }

  // Uninitialized before Validate, must call after that.
  gcpp::Model ModelType() const { return model_type_; }
  gcpp::Type WeightType() const { return weight_type_; }

 private:
  Model model_type_;
  Type weight_type_;
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

template <typename T>
void CompressWeights(const Path& weights_path,
                     const Path& compressed_weights_path, Model model_type,
                     hwy::ThreadPool& pool) {
  if (!weights_path.Exists()) {
    HWY_ABORT("The model weights file '%s' does not exist.",
              weights_path.path.c_str());
  }
  printf("Compressing weights from %s to %s\n", weights_path.path.c_str(),
         compressed_weights_path.path.c_str());
  ModelConfig config = ConfigFromModel(model_type);
  std::vector<MatStorage> model_storage;
  ModelWeightsPtrs<T> c_weights(config, pool);
  c_weights.Allocate(model_storage, pool);
  ModelWeightsPtrs<float> uc_weights(config, pool);
  uc_weights.Allocate(model_storage, pool);
  // Get uncompressed weights, compress, and store.
  FILE* fptr = fopen(weights_path.path.c_str(), "rb");
  if (fptr == nullptr) {
    HWY_ABORT("Failed to open model file %s - does it exist?",
              weights_path.path.c_str());
  }
  bool ok = true;
  uint64_t total_size = 0;
  ModelWeightsPtrs<float>::ForEachTensor(
      {&uc_weights}, ForEachType::kLoadNoToc,
      [&](const char* name, hwy::Span<MatPtr*> tensors) {
        fprintf(stderr, "Loading Parameters (size %zu): %s\n",
                tensors[0]->SizeBytes(), name);
        ok &= 1 == fread(tensors[0]->Ptr(), tensors[0]->SizeBytes(), 1, fptr);
        total_size += tensors[0]->SizeBytes();
      });
  const bool scale_for_compression = config.num_tensor_scales > 0;
  std::vector<float> scales;
  if (scale_for_compression) {
    uc_weights.GetOrApplyScales(scales);
  }
  Compressor compressor(pool);
  ModelWeightsPtrs<T>::ForEachTensor(
      {reinterpret_cast<ModelWeightsPtrs<T>*>(&uc_weights), &c_weights},
      ForEachType::kLoadNoToc,
      [&compressor](const char* name, hwy::Span<MatPtr*> tensors) {
        tensors[1]->CallUpcasted(
            compressor, name,
            reinterpret_cast<const float*>(tensors[0]->Ptr()));
      });
  compressor.AddScales(scales.data(), scales.size() * sizeof(scales[0]));
  compressor.WriteAll(pool, compressed_weights_path);
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {

void Run(Args& args) {
  hwy::ThreadPool pool(args.num_threads);
  const Model model_type = args.ModelType();
  if (model_type == Model::PALIGEMMA_224) {
    HWY_ABORT("PaliGemma is not supported in compress_weights.");
  }
  const Type weight_type = args.WeightType();
  switch (weight_type) {
    case Type::kF32:
      HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(CompressWeights<float>)
      (args.weights, args.compressed_weights, model_type, pool);
      break;
    case Type::kBF16:
      HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(CompressWeights<BF16>)
      (args.weights, args.compressed_weights, model_type, pool);
      break;
    case Type::kSFP:
      HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(CompressWeights<SfpStream>)
      (args.weights, args.compressed_weights, model_type, pool);
      break;
    case Type::kNUQ:
      HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(CompressWeights<NuqStream>)
      (args.weights, args.compressed_weights, model_type, pool);
      break;
    default:
      HWY_ABORT("Weight type %d unsupported.", static_cast<int>(weight_type));
  }
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
