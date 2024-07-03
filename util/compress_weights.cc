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
  "util/compress_weights.cc"   // NOLINT
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// Must come after foreach_target.h to avoid redefinition errors.
#include "compression/compress-inl.h"
#include "hwy/highway.h"

#ifndef GEMMA_COMPRESS_WEIGHTS_ONCE
#define GEMMA_COMPRESS_WEIGHTS_ONCE

#include <stddef.h>
#include <stdio.h>

#include <algorithm>  // std::clamp
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>  // NOLINT

#include "compression/io.h"  // Path
#include "gemma/common.h"    // Model
#include "gemma/weights.h"
#include "gemma/weights_raw.h"
#include "util/args.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"

namespace gcpp {

// Setting this to true disables fread() calls that read the model file.
constexpr bool kDryRunFread = false;

namespace {
float ScaleWeights(float* data, size_t len) {
  float maxabs = 0.0;
  for (size_t i = 0; i < len; ++i) {
    maxabs = std::max(maxabs, std::abs(data[i]));
  }
  const float kMaxRange = 1.875f;
  if (maxabs <= kMaxRange) {
    return 1.0f;
  }
  const float scale = maxabs / kMaxRange;
  const float inv_scale = 1.0f / scale;
  for (size_t i = 0; i < len; ++i) {
    data[i] *= inv_scale;
  }
  return scale;
}

#define READ_WEIGHTS(name)                                                 \
  do {                                                                     \
    do_fread(&(layer_view->name), layer, #name, sizeof(layer_view->name)); \
  } while (0)

#define SCALE_WEIGHTS(name)                                               \
  do {                                                                    \
    if (ok && !kDryRunFread && scale_for_compression) {                   \
      weights->scales[scale_pos++] =                                      \
          ScaleWeights(layer_view->name.data(), layer_view->name.size()); \
    }                                                                     \
  } while (0)

template <typename TConfig>
struct LoadRawWeightsT {
  ByteStorageT operator()(const Path& checkpoint, hwy::ThreadPool& pool,
                          bool scale_for_compression) const {
    PROFILER_ZONE("Startup.LoadWeights");
    if (!checkpoint.Exists()) {
      HWY_ABORT("The model weights file '%s' does not exist.",
                checkpoint.path.c_str());
    }

    ByteStorageT weights_u8 = AllocateWeightsF<TConfig>()(pool);
    auto* weights = reinterpret_cast<WeightsF<TConfig>*>(weights_u8.get());

    size_t scale_pos = 0;
    FILE* fptr;
    if constexpr (kDryRunFread) {
      fprintf(stderr, "Dry-Run, not reading model-file.\n");
    } else {
      fptr = fopen(checkpoint.path.c_str(), "rb");
      if (fptr == nullptr) {
        HWY_ABORT("Failed to open model file %s - does it exist?",
                  checkpoint.path.c_str());
      }
    }
    bool ok = true;
    uint64_t total_size = 0;
    auto do_fread = [&](void* var, int layer, const char* name, size_t size) {
      if (layer == -1) {
        fprintf(stderr, "Loading Parameters (size %zu): %s\n", size, name);
      } else {
        fprintf(stderr, "Loading Parameters (layer=%d, size %zu): %s\n", layer,
                size, name);
      }
      if constexpr (!kDryRunFread) {
        ok &= 1 == fread(var, size, 1, fptr);
        total_size += size;
      }
    };
    do_fread(&(weights->embedder_input_embedding), -1,
             "embedder_input_embedding",
             sizeof(weights->embedder_input_embedding));
    do_fread(&(weights->final_norm_scale), -1, "final_norm_scale",
             sizeof(weights->final_norm_scale));
    for (size_t layer = 0; layer < TConfig::kLayers; ++layer) {
      auto type = TConfig::kLayerConfig[layer];
      LayerF<TConfig>* layer_view = weights->GetLayer(layer);

      // Make sure we don't have uninitialized memory.
      hwy::ZeroBytes(layer_view, sizeof(*layer_view));
      if (type == LayerAttentionType::kGemma) {
        READ_WEIGHTS(attn_vec_einsum_w);
        READ_WEIGHTS(qkv_einsum_w);
        SCALE_WEIGHTS(attn_vec_einsum_w);
        SCALE_WEIGHTS(qkv_einsum_w);
      } else {
        READ_WEIGHTS(griffin.linear_x_w);
        READ_WEIGHTS(griffin.linear_x_biases);
        READ_WEIGHTS(griffin.linear_y_w);
        READ_WEIGHTS(griffin.linear_y_biases);
        READ_WEIGHTS(griffin.linear_out_w);
        READ_WEIGHTS(griffin.linear_out_biases);
        READ_WEIGHTS(griffin.conv_w);
        READ_WEIGHTS(griffin.conv_biases);
        READ_WEIGHTS(griffin.gate_w);
        READ_WEIGHTS(griffin.gate_biases);
        READ_WEIGHTS(griffin.a);
        SCALE_WEIGHTS(griffin.linear_x_w);
        SCALE_WEIGHTS(griffin.linear_y_w);
        SCALE_WEIGHTS(griffin.linear_out_w);
        SCALE_WEIGHTS(griffin.gate_w);
      }
      READ_WEIGHTS(gating_einsum_w);
      READ_WEIGHTS(linear_w);
      SCALE_WEIGHTS(gating_einsum_w);
      SCALE_WEIGHTS(linear_w);
      READ_WEIGHTS(pre_attention_norm_scale);
      READ_WEIGHTS(pre_ffw_norm_scale);
      if (TConfig::kPostNorm == PostNormType::Scale) {
        READ_WEIGHTS(post_attention_norm_scale);
        READ_WEIGHTS(post_ffw_norm_scale);
      }
      if (TConfig::kFFBiases) {
        READ_WEIGHTS(ffw_gating_biases);
        READ_WEIGHTS(ffw_output_biases);
      }
      if (TConfig::kSoftmaxAttnOutputBiases &&
          type == LayerAttentionType::kGemma) {
        READ_WEIGHTS(attention_output_biases);
      }
    }
    if (!ok) {
      HWY_ABORT(
          "Failed to read from %s - might be a directory, or too small? "
          "expected size: %d kB",
          checkpoint.path.c_str(), static_cast<uint32_t>(total_size >> 10));
    }
    if (!kDryRunFread) {
      HWY_ASSERT(0 == fclose(fptr));
      if (scale_for_compression) {
        HWY_ASSERT(scale_pos == TConfig::kNumTensorScales);
      }
    }
    return weights_u8;
  }
};

#undef READ_WEIGHTS
#undef SCALE_WEIGHTS
}  // namespace

ByteStorageT LoadRawWeights(const Path& weights, Model model_type,
                            Type weight_type, hwy::ThreadPool& pool,
                            bool scale_for_compression) {
  return CallForModelAndWeight<LoadRawWeightsT>(
      model_type, weight_type, weights, pool, scale_for_compression);
}

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

template <class TConfig>
void CompressWeights(const Path& weights_path,
                     const Path& compressed_weights_path, Model model_type,
                     Type weight_type, hwy::ThreadPool& pool) {
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
      weights_path, model_type, weight_type, pool, scale_for_compression);
  WeightsF<TConfig>* weights =
      reinterpret_cast<WeightsF<TConfig>*>(weights_u8.get());
  Compressor compressor(pool);
  ForEachTensor<TConfig, LayerF<TConfig>>(weights, *c_weights, compressor);
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
  const Type weight_type = args.WeightType();
  GEMMA_EXPORT_AND_DISPATCH(
      model_type, weight_type, CompressWeights,
      (args.weights, args.compressed_weights, model_type, weight_type, pool));
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
