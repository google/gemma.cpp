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

#include "io/safetensors.h"

#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "io/io.h"
#include "hwy/base.h"
#include <nlohmann/json.hpp>

namespace gcpp {

namespace {

// Reads a uint64_t from a little-endian byte buffer.
inline uint64_t ReadLE64(const uint8_t* p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
  return v;
}

// Returns sorted list of *.safetensors files in a directory.
std::vector<std::string> FindSafetensorsFiles(const std::string& dir) {
  std::vector<std::string> paths;
  namespace fs = std::filesystem;
  if (!fs::is_directory(dir)) {
    HWY_ABORT("safetensors: '%s' is not a directory", dir.c_str());
  }
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (!entry.is_regular_file()) continue;
    const std::string ext = entry.path().extension().string();
    if (ext == ".safetensors") {
      paths.push_back(entry.path().string());
    }
  }
  if (paths.empty()) {
    HWY_ABORT("safetensors: no *.safetensors files found in '%s'", dir.c_str());
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

// Parses a single safetensors header JSON and populates `entries`.
// `shard_idx` is the index of this file in SafetensorsIndex::shards_.
// Returns the data_offset (= 8 + header_size).
uint64_t ParseSafetensorsHeader(
    const File& file, size_t shard_idx,
    std::unordered_map<std::string, SafetensorEntry>* entries,
    std::vector<std::string>* names) {
  // Read 8-byte header size.
  uint8_t size_buf[8];
  if (!file.Read(0, 8, size_buf)) {
    HWY_ABORT("safetensors: failed to read header size");
  }
  const uint64_t header_size = ReadLE64(size_buf);
  if (header_size == 0 || header_size > 100 * 1024 * 1024) {
    HWY_ABORT("safetensors: implausible header_size %" PRIu64, header_size);
  }

  // Read JSON header.
  std::string header_json(static_cast<size_t>(header_size), '\0');
  if (!file.Read(8, header_size, &header_json[0])) {
    HWY_ABORT("safetensors: failed to read header JSON");
  }

  const uint64_t data_offset = 8 + header_size;

  nlohmann::json j = nlohmann::json::parse(header_json);
  for (auto& [name, val] : j.items()) {
    if (name == "__metadata__") continue;
    SafetensorEntry entry;
    entry.dtype = val["dtype"].get<std::string>();
    for (const auto& d : val["shape"]) {
      entry.shape.push_back(d.get<uint64_t>());
    }
    const auto& offsets = val["data_offsets"];
    const uint64_t rel_start = offsets[0].get<uint64_t>();
    const uint64_t rel_end = offsets[1].get<uint64_t>();
    entry.file_offset = data_offset + rel_start;
    entry.num_bytes = rel_end - rel_start;
    entry.shard_idx = shard_idx;
    if (entries->find(name) != entries->end()) {
      HWY_ABORT("safetensors: duplicate tensor name '%s'", name.c_str());
    }
    names->push_back(name);
    (*entries)[name] = std::move(entry);
  }
  return data_offset;
}

}  // namespace

SafetensorsIndex::SafetensorsIndex(const std::string& dir) {
  const std::vector<std::string> paths = FindSafetensorsFiles(dir);
  shards_.resize(paths.size());
  for (size_t i = 0; i < paths.size(); ++i) {
    shards_[i].file = OpenFileOrAbort(Path(paths[i]), "r");
    shards_[i].data_offset = ParseSafetensorsHeader(
        *shards_[i].file, i, &entries_, &names_);
  }
  fprintf(stderr, "[safetensors] indexed %zu tensors from %zu shard(s) in %s\n",
          entries_.size(), shards_.size(), dir.c_str());
}

const SafetensorEntry* SafetensorsIndex::Find(const std::string& name) const {
  const auto it = entries_.find(name);
  return it == entries_.end() ? nullptr : &it->second;
}

bool SafetensorsIndex::ReadTensor(const SafetensorEntry& entry,
                                  void* out) const {
  const Shard& shard = shards_[entry.shard_idx];
  return shard.file->Read(entry.file_offset, entry.num_bytes, out);
}

void SafetensorsIndex::PrintNames() const {
  for (const auto& n : names_) {
    fprintf(stderr, "  %s\n", n.c_str());
  }
}

}  // namespace gcpp
