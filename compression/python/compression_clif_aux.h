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

#ifndef THIRD_PARTY_GEMMA_CPP_COMPRESSION_PYTHON_COMPRESSION_CLIF_AUX_H_
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_PYTHON_COMPRESSION_CLIF_AUX_H_

#include <stddef.h>

#include <memory>
#include <string>

#include "compression/types.h"  // Type
#include "gemma/configs.h"
#include "gemma/model_store.h"
#include "gemma/tensor_info.h"
#include "io/blob_store.h"
#include "util/mat.h"
#include "hwy/aligned_allocator.h"  // Span

namespace gcpp {

// Can be modified in place by ScaleWeights.
using F32Span = hwy::Span<float>;

// Interface because we compile one derived implementation per SIMD target,
// because Compress() uses SIMD.
class ISbsWriter {
 public:
  virtual ~ISbsWriter() = default;

  virtual void Insert(const char* name, F32Span weights, Type type,
                      const TensorInfo& tensor_info) = 0;

  virtual void Write(const ModelConfig& config,
                     const std::string& tokenizer_path) = 0;
};

// Non-virtual class used by pybind that calls the interface's virtual methods.
// This avoids having to register the derived types with pybind.
class SbsWriter {
 public:
  explicit SbsWriter(const std::string& sbs_path);

  void Insert(const char* name, F32Span weights, Type type,
              const TensorInfo& tensor_info) {
    impl_->Insert(name, weights, type, tensor_info);
  }

  void Write(const ModelConfig& config, const std::string& tokenizer_path) {
    impl_->Write(config, tokenizer_path);
  }

 private:
  std::unique_ptr<ISbsWriter> impl_;
};

// Limited metadata-only reader for tests.
class SbsReader {
 public:
  SbsReader(const std::string& path);

  const ModelConfig& Config() const { return model_.Config(); }
  const MatPtr* FindMat(const char* name) const { return model_.FindMat(name); }

 private:
  gcpp::BlobReader reader_;
  gcpp::ModelStore model_;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_PYTHON_COMPRESSION_CLIF_AUX_H_
