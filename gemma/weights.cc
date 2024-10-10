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

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "compression/compress.h"
#include "compression/io.h"  // Path
#include "gemma/common.h"
#include "util/allocator.h"
#include "hwy/aligned_allocator.h"
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

    CacheLoader loader(weights);
    ForEachType fet =
        loader.HaveToc() ? ForEachType::kLoadWithToc : ForEachType::kLoadNoToc;
    CWeights::ForEachTensor(
        {c_weights}, fet,
        [&loader](const char* name, hwy::Span<MatPtr*> tensors) {
          loader(name, tensors);
        });
    std::vector<float> scales(TConfig::kNumTensorScales);
    if (TConfig::kNumTensorScales > 0) {
      loader.LoadScales(scales.data(), scales.size());
    }
    if (!loader.ReadAll(pool, c_weights->model_storage)) {
      HWY_ABORT("Failed to load model weights.");
    }
    if (TConfig::kNumTensorScales > 0) {
      c_weights->GetOrApplyScales(scales);
    }
    {
      PROFILER_ZONE("Startup.Reshape");
      c_weights->Reshape(pool);
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
// For reasons unknown, this is shown as potentially unused in the IDE.
void HWY_MAYBE_UNUSED LogVec(const char* name, const float* data, size_t len) {
  hwy::Stats stats;
  for (size_t i = 0; i < len; ++i) {
    stats.Notify(data[i]);
  }
  printf("%-20s  %12zu   %13.10f   %8.5f   %13.10f\n",
         name, len, stats.Min(), stats.Mean(), stats.Max());
}

class WeightLogger {
 public:
  void operator()(const char* name, hwy::Span<MatPtr*> tensors) {
    const MatPtr& tensor = *tensors[0];
    if (tensor.scale() != 1.0f) {
      printf("[scale=%f] ", tensor.scale());
    }
    LogVec(name, tensor.data<float>(), tensor.NumElements());
    total_weights += tensor.NumElements();
  }
  size_t total_weights = 0;
};

template <typename TConfig>
struct LogWeightStatsT {
  void operator()(const ByteStorageT& weights_u8) const {
    auto& weights =
        *reinterpret_cast<CompressedWeights<TConfig>*>(weights_u8.get());
    WeightLogger logger;
    CompressedWeights<TConfig>::ForEachTensor(
        {&weights}, ForEachType::kIgnoreNulls, logger);
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
