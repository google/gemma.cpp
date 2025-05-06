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

#ifndef THIRD_PARTY_GEMMA_CPP_COMPRESSION_BLOB_STORE_H_
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_BLOB_STORE_H_

// Reads/writes arrays of bytes from/to file.

#include <stddef.h>
#include <stdint.h>

#include <memory>  // std::unique_ptr
#include <string>
#include <unordered_map>
#include <vector>

#include "compression/io.h"       // File, Path, MapPtr
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

// A read or write I/O request, each serviced by one thread in a pool.
struct BlobIO2 {
  BlobIO2(BlobRange range, void* data) : range(range), data(data) {}

  BlobRange range;
  void* data;  // Modified only if a read request. Read-only for writes.
};

class BlobStore;

// Reads `BlobStore` header, converts keys to strings and creates a hash map for
// faster lookups, and reads or maps blob data.
// Thread-safe: it is safe to concurrently call all methods except `Enqueue`,
// because they are const.
// TODO(janwas): split into header and reader/mapper classes.
class BlobReader {
 public:
  // Parallel I/O into allocated memory, or mapped view of file. The latter is
  // better when the file is huge, but page faults add noise to measurements.
  enum class Mode { kRead, kMap };

  // Acquires ownership of `file` (which must be non-null) and reads its header.
  // Factory function instead of ctor because this can fail (return null).
  static std::unique_ptr<BlobReader> Make(const Path& blob_path,
                                          Tristate map = Tristate::kDefault);

  ~BlobReader() = default;

  // Returns true if the mode passed to ctor was `kMap` and mapping succeeded.
  bool IsMapped() const { return mode_ == Mode::kMap; }

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

  // Only if `IsMapped()`: returns blob as a read-only span of `T`. Note that
  // everything else except `CallWithSpan` is in units of bytes.
  template <typename T>
  hwy::Span<const T> MappedSpan(const BlobRange& range) const {
    HWY_ASSERT(IsMapped());
    HWY_ASSERT(range.bytes % sizeof(T) == 0);
    return hwy::Span<const T>(
        HWY_RCAST_ALIGNED(const T*, mapped_.get() + range.offset),
        range.bytes / sizeof(T));
  }

  // Returns error, or calls `func(span)` with the blob identified by `key`.
  // This may allocate memory for the blob, and is intended for small blobs for
  // which an aligned allocation is unnecessary.
  template <typename T, class Func>
  bool CallWithSpan(const std::string& key, const Func& func) const {
    const BlobRange* range = Find(key);
    if (!range) {
      HWY_WARN("Blob %s not found, sizeof T=%zu", key.c_str(), sizeof(T));
      return false;
    }

    if (mode_ == Mode::kMap) {
      func(MappedSpan<T>(*range));
      return true;
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

  // The following methods must only be called if `!IsMapped()`.

  // Enqueues a BlobIO2 for `ReadAll` to execute.
  void Enqueue(const BlobRange& range, void* data);

  // Reads in parallel all enqueued requests to the specified destinations.
  // Aborts on error.
  void ReadAll(hwy::ThreadPool& pool) const;

 private:
  // Only for use by `Make`.
  BlobReader(std::unique_ptr<File> file, uint64_t file_bytes,
             const BlobStore& bs, Mode mode);

  const std::unique_ptr<File> file_;
  const uint64_t file_bytes_;
  Mode mode_;

  std::vector<std::string> keys_;
  std::vector<BlobRange> ranges_;
  std::unordered_map<std::string, size_t> key_idx_for_key_;

  MapPtr mapped_;                  // only if `kMap`
  std::vector<BlobIO2> requests_;  // only if `kRead`
};

// Collects references to blobs and writes them all at once with parallel I/O.
// Thread-compatible: independent instances can be used concurrently, but it
// does not make sense to call the methods concurrently.
class BlobWriter {
 public:
  void Add(const std::string& key, const void* data, size_t bytes);

  // For `ModelStore`: this is the `key_idx` of the next blob to be added.
  size_t NumAdded() const { return keys_.size(); }

  // Stores all blobs to disk in the given order with padding for alignment.
  // Aborts on error.
  void WriteAll(hwy::ThreadPool& pool, const Path& filename);

 private:
  std::vector<hwy::uint128_t> keys_;
  std::vector<hwy::Span<const uint8_t>> blobs_;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_BLOB_STORE_H_
