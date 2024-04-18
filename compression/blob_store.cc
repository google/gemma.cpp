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

#include "compression/blob_store.h"

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <vector>

#include "compression/io.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/detect_compiler_arch.h"

namespace gcpp {

hwy::uint128_t MakeKey(const char* string) {
  size_t length = 0;
  for (size_t i = 0; string[i] != '\0'; ++i) {
    ++length;
  }
  if (length > 16) {
    HWY_ABORT("Key %s is too long, please truncate to 16 chars.", string);
  }

  hwy::uint128_t ret;
  hwy::ZeroBytes<sizeof(ret)>(&ret);
  hwy::CopyBytes(string, &ret, length);
  return ret;
}

namespace {
void EnqueueChunkRequests(uint64_t offset, uint64_t size, uint8_t* data,
                          std::vector<BlobIO>& requests) {
  // Split into chunks for load-balancing even if blob sizes vary.
  constexpr size_t kChunkSize = 4 * 1024 * 1024;

  // Split into whole chunks and possibly one remainder.
  uint64_t pos = 0;
  if (size >= kChunkSize) {
    for (; pos <= size - kChunkSize; pos += kChunkSize) {
      requests.emplace_back(offset + pos, kChunkSize, data + pos, 0);
    }
  }
  if (pos != size) {
    requests.emplace_back(offset + pos, size - pos, data + pos, 0);
  }
}
}  // namespace

static_assert(HWY_IS_LITTLE_ENDIAN, "Assumes little endian");

// On-disk representation (little-endian).
//
// Deliberately omits a version number because this file format is unchanging.
// Additional data may be added only inside new blobs. Changes to the blob
// contents or type should be handled by renaming keys.
#pragma pack(push, 1)
class BlobStore {
  static constexpr uint32_t kMagic = 0x0A534253;  // SBS\n

 public:
  // NOT including padding, so that we can also use ZeroFillPadding after
  // copying the header.
  static constexpr size_t HeaderSize(size_t num_blobs) {
    // 16-byte fixed fields plus per-blob: 16-byte key, 16-byte offset/size.
    return 16 + 32 * num_blobs;
  }

  // Returns how many bytes to allocate for the header without the subsequent
  // blobs. Requires num_blobs_ to already be set, typically by reading
  // sizeof(BlobStore) bytes from disk.
  size_t PaddedHeaderSize() const {
    return hwy::RoundUpTo(HeaderSize(num_blobs_), kBlobAlign);
  }

  // Returns aligned offset and zero-fills between that and `offset`.
  uint64_t ZeroFillPadding(uint64_t offset) {
    uint8_t* const bytes = reinterpret_cast<uint8_t*>(this);
    const uint64_t padded = hwy::RoundUpTo(offset, kBlobAlign);
    hwy::ZeroBytes(bytes + offset, padded - offset);
    return padded;
  }

  BlobError CheckValidity(const uint64_t file_size) {
    if (magic_ != kMagic) return __LINE__;
    if (num_blobs_ == 0) return __LINE__;
    if (file_size_ != file_size) return __LINE__;

    // Ensure blobs are back to back, and zero-pad.
    uint64_t offset = ZeroFillPadding(HeaderSize(num_blobs_));
    for (size_t i = 0; i < num_blobs_; ++i) {
      const hwy::uint128_t val = keys_[num_blobs_ + i];
      if (val.lo != offset) return __LINE__;
      offset = hwy::RoundUpTo(offset + val.hi, kBlobAlign);
    }

    if (offset != file_size_) return __LINE__;

    return 0;  // all OK
  }

  static BlobStorePtr Allocate(uint64_t total_size) {
    uint8_t* bytes =
        static_cast<uint8_t*>(hwy::AllocateAlignedBytes(total_size));
    if (!bytes) return BlobStorePtr();
    return BlobStorePtr(new (bytes) BlobStore(), hwy::AlignedFreer());
  }

  static std::vector<BlobIO> PrepareWriteRequests(
      const hwy::uint128_t keys[], const hwy::Span<uint8_t> blobs[],
      size_t num_blobs, BlobStore* bs) {
    // Sanity check and ensure the cast below is safe.
    HWY_ASSERT(num_blobs < (1ULL << 20));

    // Allocate var-length header.
    const size_t header_size = HeaderSize(num_blobs);
    const size_t padded_header_size = hwy::RoundUpTo(header_size, kBlobAlign);
    const uint64_t padded_header_end = bs->ZeroFillPadding(header_size);
    HWY_ASSERT(padded_header_end == padded_header_size);

    // All-zero buffer used to write padding to the file without copying the
    // input blobs.
    static uint8_t zeros[kBlobAlign] = {0};

    // Total file size will be the header plus all padded blobs.
    uint64_t payload = 0;
    for (size_t i = 0; i < num_blobs; ++i) {
      payload += hwy::RoundUpTo(blobs[i].size(), kBlobAlign);
    }
    const size_t total_size = padded_header_size + payload;

    // Fill header.
    bs->magic_ = kMagic;
    bs->num_blobs_ = static_cast<uint32_t>(num_blobs);
    bs->file_size_ = total_size;
    hwy::CopyBytes(keys, bs->keys_, num_blobs * sizeof(keys[0]));

    // First IO request is for the header (not yet filled!).
    std::vector<BlobIO> requests;
    requests.reserve(1 + 2 * num_blobs);
    requests.emplace_back(/*offset=*/0, padded_header_size,
                          reinterpret_cast<uint8_t*>(bs), 0);

    // Fill second half of keys_ with offset/size and prepare IO requests.
    uint64_t offset = padded_header_end;
    for (size_t i = 0; i < num_blobs; ++i) {
      bs->keys_[num_blobs + i].lo = offset;
      bs->keys_[num_blobs + i].hi = blobs[i].size();

      EnqueueChunkRequests(offset, blobs[i].size(), blobs[i].data(), requests);
      offset += blobs[i].size();
      const size_t padded_size = hwy::RoundUpTo(blobs[i].size(), kBlobAlign);
      if (padded_size != blobs[i].size()) {
        const size_t padding = padded_size - blobs[i].size();
        HWY_ASSERT(padding <= kBlobAlign);
        requests.emplace_back(offset, padding, zeros, 0);
        offset += padding;
      }
    }

    HWY_ASSERT(offset == total_size);
    return requests;
  }

  bool FindKey(const hwy::uint128_t key, uint64_t& offset, size_t& size) const {
    for (size_t i = 0; i < num_blobs_; ++i) {
      if (keys_[i] == key) {
        const hwy::uint128_t val = keys_[num_blobs_ + i];
        offset = val.lo;
        size = val.hi;
        return true;
      }
    }
    return false;
  }

 private:
  uint32_t magic_;
  uint32_t num_blobs_;      // never 0
  uint64_t file_size_;      // must match actual size of file
  hwy::uint128_t keys_[1];  // length: 2 * num_blobs
  // Padding, then the blob identified by keys[0], then padding etc.
};
#pragma pack(pop)

BlobError BlobReader::Open(const char* filename) {
  if (!file_.Open(filename, "r")) return __LINE__;

  // Read first part of header to get actual size.
  BlobStore bs;
  if (!file_.Read(0, sizeof(bs), &bs)) return __LINE__;
  const size_t padded_size = bs.PaddedHeaderSize();
  HWY_ASSERT(padded_size >= sizeof(bs));

  // Allocate full header.
  blob_store_ = BlobStore::Allocate(padded_size);
  if (!blob_store_) return __LINE__;

  // Copy what we already read (more efficient than seek + re-read).
  hwy::CopySameSize(&bs, blob_store_.get());
  // Read the rest of the header, but not the full file.
  uint8_t* bytes = reinterpret_cast<uint8_t*>(blob_store_.get());
  if (!file_.Read(sizeof(bs), padded_size - sizeof(bs), bytes + sizeof(bs))) {
    return __LINE__;
  }

  return blob_store_->CheckValidity(file_.FileSize());
}

BlobError BlobReader::Enqueue(hwy::uint128_t key, void* data, size_t size) {
  uint64_t offset;
  size_t actual_size;
  if (!blob_store_->FindKey(key, offset, actual_size)) return __LINE__;
  if (actual_size != size) {
    fprintf(stderr,
            "Mismatch between expected %d and actual %d KiB size. Please see "
            "README.md on how to update the weights.\n",
            static_cast<int>(size >> 10), static_cast<int>(actual_size >> 10));
    return __LINE__;
  }

  EnqueueChunkRequests(offset, actual_size, reinterpret_cast<uint8_t*>(data),
                       requests_);
  return 0;
}

// Parallel synchronous I/O. Alternatives considered:
// - readv is limited to 0x7FFFF000 bytes on Linux (even 64-bit). Note that
//   pread calls preadv with a single iovec.
// - O_DIRECT seems undesirable because we do want to use the OS cache
//   between consecutive runs.
// - memory-mapped I/O is less predictable and adds noise to measurements.
BlobError BlobReader::ReadAll(hwy::ThreadPool& pool) {
  File* pfile = &file_;  // not owned
  const auto& requests = requests_;
  std::atomic_flag err = ATOMIC_FLAG_INIT;
  // >5x speedup from parallel reads when cached.
  pool.Run(0, requests.size(),
           [pfile, &requests, &err](uint64_t i, size_t /*thread*/) {
             if (!pfile->Read(requests[i].offset, requests[i].size,
                              requests[i].data)) {
               err.test_and_set();
             }
           });
  if (err.test_and_set()) return __LINE__;
  return 0;
}

BlobError BlobWriter::WriteAll(hwy::ThreadPool& pool, const char* filename) {
  HWY_ASSERT(keys_.size() == blobs_.size());

  // Concatenate blobs in memory.
  const size_t header_size = BlobStore::HeaderSize(keys_.size());
  const size_t padded_header_size = hwy::RoundUpTo(header_size, kBlobAlign);
  BlobStorePtr bs = BlobStore::Allocate(padded_header_size);
  std::vector<BlobIO> requests = BlobStore::PrepareWriteRequests(
      keys_.data(), blobs_.data(), keys_.size(), bs.get());

  // Create/replace existing file.
  File file;
  if (!file.Open(filename, "w+")) return __LINE__;
  File* pfile = &file;  // not owned

  std::atomic_flag err = ATOMIC_FLAG_INIT;
  pool.Run(0, requests.size(),
           [pfile, &requests, &err](uint64_t i, size_t /*thread*/) {
             if (!pfile->Write(requests[i].data, requests[i].size,
                               requests[i].offset)) {
               err.test_and_set();
             }
           });
  if (err.test_and_set()) return __LINE__;
  return 0;
}

}  // namespace gcpp
