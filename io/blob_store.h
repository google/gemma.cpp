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

#ifndef THIRD_PARTY_GEMMA_CPP_IO_BLOB_STORE_H_
#define THIRD_PARTY_GEMMA_CPP_IO_BLOB_STORE_H_

// Reads/writes arrays of bytes from/to file.

#include <stddef.h>
#include <stdint.h>

#include <memory>  // std::unique_ptr
#include <string>
#include <unordered_map>
#include <vector>

#include "io/io.h"                // File, Path, MapPtr
#include "util/basics.h"          // Tristate
#include "hwy/aligned_allocator.h"  // Span
#include "hwy/base.h"               // HWY_ASSERT
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

// One blob's extents within the file.
struct BlobRange {
  uint64_t End() const { return offset + bytes; }

  uint64_t offset = 0;
  size_t bytes = 0;  // We check blobs are not zero-sized.
  // Index within `BlobReader::Keys()` for error reporting.
  size_t key_idx;
};

// Reads `BlobStore` header, converts keys to strings and creates a hash map for
// faster lookups.
// TODO(janwas): rename to BlobFinder or similar.
// Thread-safe: it is safe to concurrently call all methods except `CloseFile`.
class BlobReader {
 public:
  // Acquires ownership of `file` (which must be non-null) and reads its header.
  // Aborts on error.
  explicit BlobReader(const Path& blob_path);

  const Path& blob_path() const { return blob_path_; }

  const File& file() const { return *file_; }
  uint64_t file_bytes() const { return file_bytes_; }
  MapPtr Map() { return file_->Map(); }
  // OK to call if Map() was called; the smart pointer keeps the mapping alive.
  void CloseFile() { file_.reset(); }

  const std::vector<std::string>& Keys() const { return keys_; }

  const BlobRange& Range(size_t key_idx) const {
    HWY_ASSERT(key_idx < keys_.size());
    return ranges_[key_idx];
  }

  // Returns nullptr if not found. O(1).
  const BlobRange* Find(const std::string& key) const {
    auto it = key_idx_for_key_.find(key);
    if (it == key_idx_for_key_.end()) return nullptr;
    const BlobRange& range = Range(it->second);
    HWY_ASSERT(range.offset != 0 && range.bytes != 0);
    HWY_ASSERT(range.End() <= file_bytes_);
    return &range;
  }

  // Returns error, or calls `func(span)` with the blob identified by `key`.
  // Allocates unaligned memory for the blob; intended for small metadata blobs.
  template <typename T, class Func>
  bool CallWithSpan(const std::string& key, const Func& func) const {
    const BlobRange* range = Find(key);
    if (!range) {
      HWY_WARN("Blob %s not found, sizeof T=%zu", key.c_str(), sizeof(T));
      return false;
    }

    HWY_ASSERT(range->bytes % sizeof(T) == 0);
    std::vector<T> storage(range->bytes / sizeof(T));
    if (!file_->Read(range->offset, range->bytes, storage.data())) {
      HWY_WARN("Read failed for blob %s from %zu, size %zu; file %zu\n",
               key.c_str(), static_cast<size_t>(range->offset), range->bytes,
               static_cast<size_t>(file_bytes_));
      return false;
    }
    func(hwy::Span<const T>(storage.data(), storage.size()));
    return true;
  }

 private:
  Path blob_path_;
  std::unique_ptr<File> file_;
  uint64_t file_bytes_;  // const after ctor

  std::vector<std::string> keys_;
  std::vector<BlobRange> ranges_;
  std::unordered_map<std::string, size_t> key_idx_for_key_;
};

// Writes blobs immediately using parallel I/O, and collects their metadata for
// writing the file footer.
// Thread-compatible: independent instances can be used concurrently, but it
// does not make sense to call the methods concurrently.
class BlobWriter {
 public:
  BlobWriter(const Path& filename, hwy::ThreadPool& pool);

  // Writes the blob to disk with padding for alignment. Aborts on error.
  void Add(const std::string& key, const void* data, size_t bytes);

  // Appends a footer and closes the file. Must be called once after all `Add`.
  void Finalize();

 private:
  std::unique_ptr<File> file_;
  std::vector<hwy::uint128_t> keys_;
  std::vector<size_t> blob_sizes_;
  hwy::ThreadPool& pool_;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_IO_BLOB_STORE_H_
