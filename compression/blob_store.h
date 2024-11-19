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

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

#include "compression/io.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"  // hwy::uint128_t
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

// Convenient way to construct a key from a string (<= 16 chars).
hwy::uint128_t MakeKey(const char* string);

// Returns a string from a key.
std::string StringFromKey(hwy::uint128_t key);

// Ordered list of opaque blobs (~hundreds), identified by unique opaque
// 128-bit keys.
class BlobStore;

// Incomplete type, so dtor will not be called.
using BlobStorePtr = hwy::AlignedFreeUniquePtr<BlobStore>;

// 0 if successful, otherwise the line number of the failing check.
using BlobError = int;

// Blob offsets on disk and memory addresses are a multiple of this, because
// we pad the header and each blob's size. This matches CUDA alignment and the
// maximum SVE vector size, and exceeds typical x86 cache line sizes (64 or
// 128), which can help performance.
static constexpr size_t kBlobAlign = 256;

// One I/O request, serviced by threads in a pool.
struct BlobIO {
  BlobIO(uint64_t offset, size_t size, void* data, uint64_t padding)
      : offset(offset), size(size), data(data), padding(padding) {}

  uint64_t offset;
  size_t size;  // bytes
  void* data;
  uint64_t padding;
};

class BlobReader {
 public:
  BlobReader() { requests_.reserve(500); }
  ~BlobReader() = default;

  // Opens `filename` and reads its header.
  BlobError Open(const Path& filename);

  // Returns the size of the blob identified by `key`, or 0 if not found.
  size_t BlobSize(hwy::uint128_t key) const;

  // Enqueues read requests if `key` is found and its size matches `size`, which
  // is in units of bytes.
  BlobError Enqueue(hwy::uint128_t key, void* data, size_t size);

  // Reads all enqueued requests.
  BlobError ReadAll(hwy::ThreadPool& pool);

  // Reads one blob directly.
  BlobError ReadOne(hwy::uint128_t key, void* data, size_t size) const;

  // Returns all available blob keys.
  hwy::Span<const hwy::uint128_t> Keys() const;

 private:
  BlobStorePtr blob_store_;  // holds header, not the entire file
  std::vector<BlobIO> requests_;
  std::unique_ptr<File> file_;
};

class BlobWriter {
 public:
  // `size` is in bytes.
  void Add(hwy::uint128_t key, const void* data, size_t size) {
    keys_.push_back(key);
    blobs_.emplace_back(static_cast<const uint8_t*>(data), size);
  }

  // Stores all blobs to disk in the given order with padding for alignment.
  BlobError WriteAll(hwy::ThreadPool& pool, const Path& filename);

 private:
  std::vector<hwy::uint128_t> keys_;
  std::vector<hwy::Span<const uint8_t>> blobs_;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_BLOB_STORE_H_
