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

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "compression/shared.h"
#include "gemma/configs.h"
#include "gemma/tensor_index.h"

namespace gcpp {

// How to process the data.
enum class CompressorMode {
  // No compression, no write to file, just for testing.
  kTEST_ONLY,
  // Old-style compression, no table of contents.
  kNO_TOC,
  // New-style compression, with table of contents.
  kWITH_TOC,
};

class WriterInterface;

class SbsWriter {
 public:
  explicit SbsWriter(CompressorMode mode);
  ~SbsWriter();

  void Insert(std::string name, absl::Span<const float> weights, Type type,
              const TensorInfo& tensor_info, float scale);
  void InsertSfp(std::string name, absl::Span<const float> weights);
  void InsertNUQ(std::string name, absl::Span<const float> weights);
  void InsertBfloat16(std::string name, absl::Span<const float> weights);
  void InsertFloat(std::string name, absl::Span<const float> weights);
  void AddScales(const std::vector<float>& scales);
  void AddTokenizer(const std::string& tokenizer_path);

  size_t DebugNumBlobsAdded() const;

  int Write(std::string path) { return WriteWithConfig(path, nullptr); }
  int WriteWithConfig(std::string path, const ModelConfig* config);

 private:
  // Isolates Highway-dispatched types and other internals from CLIF.
  std::unique_ptr<WriterInterface> impl_;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_PYTHON_COMPRESSION_CLIF_AUX_H_
