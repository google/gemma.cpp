// Copyright 2025 Google LLC
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

// Parses the safetensors format used by HuggingFace models.
// Format: [8-byte header_size LE][header_size bytes JSON][tensor data ...]
// Multiple sharded files (model-NNNNN-of-MMMMM.safetensors) are supported.

#ifndef THIRD_PARTY_GEMMA_CPP_IO_SAFETENSORS_H_
#define THIRD_PARTY_GEMMA_CPP_IO_SAFETENSORS_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "io/io.h"  // File, Path
#include "hwy/base.h"

namespace gcpp {

// Metadata for a single tensor in a safetensors file.
struct SafetensorEntry {
  std::string dtype;            // "BF16", "F32", "F16", "I8", etc.
  std::vector<uint64_t> shape;  // dimension sizes (may be empty for scalars)
  uint64_t file_offset;         // absolute byte offset in shard file
  uint64_t num_bytes;           // total byte size of tensor data
  size_t shard_idx;             // index into SafetensorsIndex::shards_
};

// Opens one or more *.safetensors files and provides random tensor access.
// Supports sharded models (model-00001-of-00002.safetensors, etc.).
class SafetensorsIndex {
 public:
  // Scans `dir` for all *.safetensors files, parses their headers, and builds
  // a unified tensor index. Aborts if no files are found or parsing fails.
  explicit SafetensorsIndex(const std::string& dir);

  // Returns nullptr if `name` is not found in any shard.
  const SafetensorEntry* Find(const std::string& name) const;

  // All tensor names across all shards.
  const std::vector<std::string>& Names() const { return names_; }

  // Reads `entry.num_bytes` bytes into `out`. Returns true on success.
  bool ReadTensor(const SafetensorEntry& entry, void* out) const;

  // For debugging: prints all indexed tensor names to stderr.
  void PrintNames() const;

  struct Shard {
    std::unique_ptr<File> file;
    uint64_t data_offset;  // = 8 + header_size
  };

 private:
  std::vector<Shard> shards_;
  std::unordered_map<std::string, SafetensorEntry> entries_;
  std::vector<std::string> names_;
};

// Returns total number of elements given a shape vector.
inline uint64_t SafetensorNumElems(const std::vector<uint64_t>& shape) {
  if (shape.empty()) return 1;
  uint64_t n = 1;
  for (uint64_t d : shape) n *= d;
  return n;
}

// Returns bytes per element for a safetensors dtype string.
inline size_t SafetensorDtypeBytes(const std::string& dtype) {
  if (dtype == "BF16" || dtype == "F16") return 2;
  if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
  if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8;
  if (dtype == "I8" || dtype == "U8" || dtype == "BOOL") return 1;
  if (dtype == "I16" || dtype == "U16") return 2;
  return 0;
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_IO_SAFETENSORS_H_
