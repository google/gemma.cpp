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

#include "gemma/weights.h"

#include <algorithm>
#include <cstdlib>

#include "compression/compress.h"
#include "compression/io.h"  // Path
#include "gemma/common.h"
#include "gemma/configs.h"
#include "hwy/base.h"  // HWY_ABORT
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"
#include "hwy/stats.h"

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

    ByteStorageT weights_u8 = AllocateWeights<TConfig>()(pool);
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
      if (TConfig::kPostNormScale) {
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

ByteStorageT LoadRawWeights(const Path& weights, Model model,
                            hwy::ThreadPool& pool, bool scale_for_compression) {
  return CallFunctorForModel<LoadRawWeightsT>(model, weights, pool,
                                              scale_for_compression);
}

namespace {
template <class TConfig>
struct LoadCompressedWeightsT {
  ByteStorageT operator()(const Path& weights, hwy::ThreadPool& pool) const {
    PROFILER_ZONE("Startup.LoadCompressedWeights");
    if (!weights.Exists()) {
      HWY_ABORT("The model weights file '%s' does not exist.",
                weights.path.c_str());
    }

    // Allocate compressed weights.
    using CWeights = CompressedWeights<TConfig>;
    ByteStorageT c_weights_u8 = AllocateSizeof<CWeights>();
    CWeights* c_weights = reinterpret_cast<CWeights*>(c_weights_u8.get());
    new (&c_weights->c_layer_ptrs) CompressedLayerPointers<TConfig>(pool);

    std::array<float, TConfig::kNumTensorScales> scales;
    CacheLoader loader(weights);
    ForEachTensor<TConfig>(nullptr, *c_weights, loader);
    loader.LoadScales(scales.data(), scales.size());
    if (!loader.ReadAll(pool)) {
      HWY_ABORT("Failed to load model weights.");
    }
    if (TConfig::kNumTensorScales > 0) {
      size_t scale_pos = 0;
      for (int layer_idx = 0; layer_idx < TConfig::kLayers; ++layer_idx) {
        auto type = TConfig::kLayerConfig[layer_idx];
        const size_t idx = static_cast<size_t>(layer_idx);
        CompressedLayer<TConfig>* layer_weights = c_weights->GetLayer(idx);
        if (type == LayerAttentionType::kGemma) {
          layer_weights->attn_vec_einsum_w.set_scale(scales[scale_pos++]);
          layer_weights->qkv_einsum_w.set_scale(scales[scale_pos++]);
        } else {
          layer_weights->griffin.linear_x_w.set_scale(scales[scale_pos++]);
          layer_weights->griffin.linear_y_w.set_scale(scales[scale_pos++]);
          layer_weights->griffin.linear_out_w.set_scale(scales[scale_pos++]);
          layer_weights->griffin.gate_w.set_scale(scales[scale_pos++]);
        }
        layer_weights->gating_einsum_w.set_scale(scales[scale_pos++]);
        layer_weights->linear_w.set_scale(scales[scale_pos++]);
      }
      HWY_ASSERT(scale_pos == TConfig::kNumTensorScales);
    }
    return c_weights_u8;
  }
};
}  // namespace

ByteStorageT LoadCompressedWeights(const Path& weights, Model model,
                                   hwy::ThreadPool& pool) {
  return CallFunctorForModel<LoadCompressedWeightsT>(model, weights, pool);
}

ByteStorageT LoadWeights(const Path& weights, Model model,
                         hwy::ThreadPool& pool) {
  if constexpr (kWeightsAreCompressed) {
    return LoadCompressedWeights(weights, model, pool);
  } else {
    return LoadRawWeights(weights, model, pool,
                          /*scale_for_compression=*/false);
  }
}

namespace {
void LogVec(const char* name, const float* data, size_t len) {
  hwy::Stats stats;
  for (size_t i = 0; i < len; ++i) {
    stats.Notify(data[i]);
  }
  printf("%-20s  %12zu   %13.10f   %8.5f   %13.10f\n",
         name, len, stats.Min(), stats.Mean(), stats.Max());
}

class WeightLogger {
 public:
  template <size_t N>
  void operator()(const char* name, const std::array<float, N>& tensor) {
    LogVec(name, tensor.data(), N);
    total_weights += N;
  }
  size_t total_weights = 0;
};

template <typename TConfig>
struct LogWeightStatsT {
  void operator()(const ByteStorageT& weights_u8) const {
    const auto& weights =
        *reinterpret_cast<WeightsF<TConfig>*>(weights_u8.get());
    WeightLogger logger;
    ForEachTensor1<float, TConfig>(logger, weights);
    printf("%-20s  %12zu\n", "Total", logger.total_weights);
  }
};
}  // namespace

void LogWeightStats(gcpp::Model model, const ByteStorageT& weights) {
  CallFunctorForModel<LogWeightStatsT>(model, weights);
}

}  // namespace gcpp
