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
#include <memory>
#include <random>
#include <vector>

#include "compression/blob_store.h"
#include "compression/compress.h"
#include "compression/io.h"  // Path
#include "gemma/common.h"
#include "gemma/configs.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"  // HWY_ABORT
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"
#include "hwy/stats.h"

namespace gcpp {

template <typename T>
struct TensorLoader {
  void operator()(ModelWeightsPtrs<T>& weights, ForEachType fet,
                  ReadFromBlobStore& loader) {
    weights.ForEachTensor(
        {&weights}, fet,
        [&loader](const char* name, hwy::Span<MatPtr*> tensors) {
          loader(name, tensors);
        });
  }
};

BlobError ModelWeightsStorage::Load(const Path& weights, Model model_type,
                                    Type weight_type, hwy::ThreadPool& pool) {
  PROFILER_ZONE("Startup.LoadModelWeightsPtrs");
  if (!weights.Exists()) {
    HWY_ABORT("The model weights file '%s' does not exist.",
              weights.path.c_str());
  }
  ReadFromBlobStore loader(weights);
  ForEachType fet =
      loader.HaveToc() ? ForEachType::kLoadWithToc : ForEachType::kLoadNoToc;
  if (fet == ForEachType::kLoadWithToc) {
    // TODO(rays): Load the config from the file.
    HWY_ABORT("TOC not supported yet.");
  } else {
    // No Toc-> no config.
    config_ = ConfigFromModel(model_type);
    config_.weight = weight_type;
  }
  CreateForType(weight_type, pool);
  CallForModelWeightT<TensorLoader>(fet, loader);
  std::vector<float> scales(config_.num_tensor_scales + config_.num_vit_scales);
  if (!scales.empty()) {
    loader.LoadScales(scales.data(), scales.size());
  }
  BlobError err = loader.ReadAll(pool, model_storage_);
  if (err != 0) {
    fprintf(stderr, "Failed to load model weights: %d\n", err);
    return err;
  }
  if (!scales.empty()) {
    GetOrApplyScales(scales);
  }
  if (fet == ForEachType::kLoadNoToc) {
    PROFILER_ZONE("Startup.Reshape");
    AllocAndCopyWithTranspose(pool);
  }
  return 0;
}

void ModelWeightsStorage::Allocate(const ModelConfig& config, Type weight_type,
                                   hwy::ThreadPool& pool) {
  PROFILER_ZONE("Startup.AllocateModelWeightsPtrs");
  config_ = config;
  config_.weight = weight_type;
  CreateForType(weight_type, pool);
  if (float_weights_) float_weights_->Allocate(model_storage_, pool);
  if (bf16_weights_) bf16_weights_->Allocate(model_storage_, pool);
  if (sfp_weights_) sfp_weights_->Allocate(model_storage_, pool);
  if (nuq_weights_) nuq_weights_->Allocate(model_storage_, pool);
}

class WeightInitializer {
 public:
  WeightInitializer(std::mt19937& gen) : dist_(0.0f, 1.0f), gen_(gen) {}

  void operator()(const char* name, hwy::Span<MatPtr*> tensors) {
    float* data = tensors[0]->data<float>();
    for (size_t i = 0; i < tensors[0]->NumElements(); ++i) {
      data[i] = dist_(gen_);
    }
    tensors[0]->set_scale(1.0f);
  }

 private:
  std::normal_distribution<float> dist_;
  std::mt19937& gen_;
};

void ModelWeightsStorage::RandInit(std::mt19937& gen) {
  HWY_ASSERT(float_weights_);
  WeightInitializer init(gen);
  ModelWeightsPtrs<float>::ForEachTensor({float_weights_.get()},
                                         ForEachType::kLoadNoToc, init);
}

void ModelWeightsStorage::ZeroInit() {
  if (float_weights_) float_weights_->ZeroInit();
  if (bf16_weights_) bf16_weights_->ZeroInit();
  if (sfp_weights_) sfp_weights_->ZeroInit();
  if (nuq_weights_) nuq_weights_->ZeroInit();
}

void ModelWeightsStorage::GetOrApplyScales(std::vector<float>& scales) {
  if (float_weights_) float_weights_->GetOrApplyScales(scales);
  if (bf16_weights_) bf16_weights_->GetOrApplyScales(scales);
  if (sfp_weights_) sfp_weights_->GetOrApplyScales(scales);
  if (nuq_weights_) nuq_weights_->GetOrApplyScales(scales);
}

void ModelWeightsStorage::AllocAndCopyWithTranspose(hwy::ThreadPool& pool) {
  if (float_weights_)
    float_weights_->AllocAndCopyWithTranspose(pool, model_storage_);
  if (bf16_weights_)
    bf16_weights_->AllocAndCopyWithTranspose(pool, model_storage_);
  if (sfp_weights_)
    sfp_weights_->AllocAndCopyWithTranspose(pool, model_storage_);
  if (nuq_weights_)
    nuq_weights_->AllocAndCopyWithTranspose(pool, model_storage_);
}

void ModelWeightsStorage::CopyWithTranspose(hwy::ThreadPool& pool) {
  if (float_weights_) float_weights_->CopyWithTranspose(pool);
  if (bf16_weights_) bf16_weights_->CopyWithTranspose(pool);
  if (sfp_weights_) sfp_weights_->CopyWithTranspose(pool);
  if (nuq_weights_) nuq_weights_->CopyWithTranspose(pool);
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

}  // namespace

void ModelWeightsStorage::LogWeightStats() {
  size_t total_weights = 0;
  // Only for float weights.
  ModelWeightsPtrs<float>::ForEachTensor(
      {float_weights_.get()}, ForEachType::kInitNoToc,
      [&total_weights](const char* name, hwy::Span<MatPtr*> tensors) {
        const MatPtr& tensor = *tensors[0];
        if (tensor.scale() != 1.0f) {
          printf("[scale=%f] ", tensor.scale());
        }
        LogVec(name, tensor.data<float>(), tensor.NumElements());
        total_weights += tensor.NumElements();
      });
  printf("%-20s  %12zu\n", "Total", total_weights);
}

void ModelWeightsStorage::CreateForType(Type weight_type,
                                        hwy::ThreadPool& pool) {
  switch (weight_type) {
    case Type::kF32:
      float_weights_ = std::make_unique<ModelWeightsPtrs<float>>(config_, pool);
      break;
    case Type::kBF16:
      bf16_weights_ = std::make_unique<ModelWeightsPtrs<BF16>>(config_, pool);
      break;
    case Type::kSFP:
      sfp_weights_ =
          std::make_unique<ModelWeightsPtrs<SfpStream>>(config_, pool);
      break;
    case Type::kNUQ:
      nuq_weights_ =
          std::make_unique<ModelWeightsPtrs<NuqStream>>(config_, pool);
      break;
    default:
      HWY_ABORT("Weight type %d unsupported.", static_cast<int>(weight_type));
  }
}

}  // namespace gcpp
