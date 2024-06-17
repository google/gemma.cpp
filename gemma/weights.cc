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
    new (c_weights) CWeights(pool);

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

ByteStorageT LoadCompressedWeights(const Path& weights, Model model_type,
                                   Type weight_type, hwy::ThreadPool& pool) {
  return CallForModelAndWeight<LoadCompressedWeightsT>(model_type, weight_type,
                                                       weights, pool);
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
  void operator()(const char* name, const CompressedArray<float, N>& tensor) {
    LogVec(name, tensor.data(), N);
    total_weights += N;
  }
  size_t total_weights = 0;
};

template <typename TConfig>
struct LogWeightStatsT {
  void operator()(const ByteStorageT& weights_u8) const {
    const auto& weights =
        *reinterpret_cast<CompressedWeights<TConfig>*>(weights_u8.get());
    WeightLogger logger;
    ForEachTensor1<TConfig>(logger, weights);
    printf("%-20s  %12zu\n", "Total", logger.total_weights);
  }
};
}  // namespace

void LogWeightStats(gcpp::Model model_type, Type weight_type,
                    const ByteStorageT& weights) {
  HWY_ASSERT(weight_type == Type::kF32);
  CallForModel<float, LogWeightStatsT>(model_type, weights);
}

}  // namespace gcpp
