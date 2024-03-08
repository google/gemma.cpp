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

// Request POSIX 2008, including `pread()` and `posix_fadvise()`.
#if !defined(_XOPEN_SOURCE) || _XOPEN_SOURCE < 700
#undef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif
#if !defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 200809
#define _POSIX_C_SOURCE 200809
#endif

// Make `off_t` 64-bit even on 32-bit systems. Works for Android >= r15c.
#undef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64

// copybara:import_next_line:gemma_cpp
#include "compression/blob_store.h"

#include <stdint.h>
#include <stdio.h>     // SEEK_END - unistd isn't enough for IDE.
#include <sys/stat.h>  // O_RDONLY
#include <fcntl.h>  // open
#if HWY_OS_WIN
#include <io.h>  // read, write, close
#include <fileapi.h>
#else
#include <unistd.h>    // read, write, close
#endif

#include <atomic>
#include <vector>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/detect_compiler_arch.h"

namespace {
#if HWY_OS_WIN

// pread is not supported on Windows
static int64_t pread(int fd, void* buf, uint64_t size, uint64_t offset) {
  HANDLE file = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  if (file == INVALID_HANDLE_VALUE) {
    return -1;
  }

  OVERLAPPED overlapped = {0};
  overlapped.Offset = offset & 0xFFFFFFFF;
  overlapped.OffsetHigh = (offset >> 32) & 0xFFFFFFFF;

  DWORD bytes_read;
  if (!ReadFile(file, buf, size, &bytes_read, &overlapped)) {
    if (GetLastError() != ERROR_HANDLE_EOF) {
      return -1;
    }
  }

  return bytes_read;
}

// pwrite is not supported on Windows
static int64_t pwrite(int fd, const void* buf, uint64_t size, uint64_t offset) {
  HANDLE file = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  if (file == INVALID_HANDLE_VALUE) {
    return -1;
  }

  OVERLAPPED overlapped = {0};
  overlapped.Offset = offset & 0xFFFFFFFF;
  overlapped.OffsetHigh = (offset >> 32) & 0xFFFFFFFF;

  DWORD bytes_written;
  if (!WriteFile(file, buf, size, &bytes_written, &overlapped)) {
    if (GetLastError() != ERROR_HANDLE_EOF) {
      return -1;
    }
  }

  return bytes_written;
}

#endif
}  // namespace

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

static void EnqueueChunkRequests(uint64_t offset, uint64_t size, uint8_t* data,
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


struct IO {
  // Returns size in bytes or 0.
  static uint64_t FileSize(const char* filename) {
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
      return 0;
    }

#if HWY_OS_WIN
    const int64_t size = _lseeki64(fd, 0, SEEK_END);
    HWY_ASSERT(close(fd) != -1);
    if (size < 0) {
      return 0;
    }
#else
    static_assert(sizeof(off_t) == 8, "64-bit off_t required");
    const off_t size = lseek(fd, 0, SEEK_END);
    HWY_ASSERT(close(fd) != -1);
    if (size == static_cast<off_t>(-1)) {
      return 0;
    }
#endif

    return static_cast<uint64_t>(size);
  }

  static bool Read(int fd, uint64_t offset, uint64_t size, void* to) {
    uint8_t* bytes = reinterpret_cast<uint8_t*>(to);
    uint64_t pos = 0;
    for (;;) {
      // pread seems to be faster than lseek + read when parallelized.
      const auto bytes_read = pread(fd, bytes + pos, size - pos, offset + pos);
      if (bytes_read <= 0) break;
      pos += bytes_read;
      HWY_ASSERT(pos <= size);
      if (pos == size) break;
    }
    return pos == size;  // success if managed to read desired size
  }

  static bool Write(const void* from, uint64_t size, uint64_t offset, int fd) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(from);
    uint64_t pos = 0;
    for (;;) {
      const auto bytes_written =
          pwrite(fd, bytes + pos, size - pos, offset + pos);
      if (bytes_written <= 0) break;
      pos += bytes_written;
      HWY_ASSERT(pos <= size);
      if (pos == size) break;
    }
    return pos == size;  // success if managed to write desired size
  }
};  // IO

static_assert(HWY_IS_LITTLE_ENDIAN, "Assumes little endian");

// On-disk representation (little-endian).
//
// Deliberately omits a version number because this file format is unchanging.
// Additional data may be added only inside new blobs. Changes to the blob
// contents or type should be handled by renaming keys.
#pragma pack(push, 1)
class BlobStore {
  static constexpr uint32_t kMagic = 0x0A534253;  // SBS\n

  // Blob offsets on disk and memory addresses are a multiple of this, because
  // we pad the header and each blob's size. This matches CUDA alignment and the
  // maximum SVE vector size, and exceeds typical x86 cache line sizes (64 or
  // 128), which can help performance.
  static constexpr size_t kAlign = 256;

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
    return hwy::RoundUpTo(HeaderSize(num_blobs_), kAlign);
  }

  // Returns aligned offset and zero-fills between that and `offset`.
  uint64_t ZeroFillPadding(uint64_t offset) {
    uint8_t* const bytes = reinterpret_cast<uint8_t*>(this);
    const uint64_t padded = hwy::RoundUpTo(offset, kAlign);
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
      offset = ZeroFillPadding(offset + val.hi);
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
      size_t num_blobs) {
    // Sanity check and ensure the cast below is safe.
    HWY_ASSERT(num_blobs < (1ULL << 20));

    // Allocate var-length header.
    const size_t header_size = HeaderSize(num_blobs);
    const size_t padded_header_size = hwy::RoundUpTo(header_size, kAlign);
    BlobStorePtr bs = Allocate(padded_header_size);
    const uint64_t padded_header_end = bs->ZeroFillPadding(header_size);
    HWY_ASSERT(padded_header_end == padded_header_size);

    // All-zero buffer used to write padding to the file without copying the
    // input blobs.
    static uint8_t zeros[kAlign] = {0};

    // Total file size will be the header plus all padded blobs.
    uint64_t payload = 0;
    for (size_t i = 0; i < num_blobs; ++i) {
      payload += hwy::RoundUpTo(blobs[i].size(), kAlign);
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
                          reinterpret_cast<uint8_t*>(bs.get()), 0);

    // Fill second half of keys_ with offset/size and prepare IO requests.
    uint64_t offset = padded_header_end;
    for (size_t i = 0; i < num_blobs; ++i) {
      bs->keys_[num_blobs + i].lo = offset;
      bs->keys_[num_blobs + i].hi = blobs[i].size();

      EnqueueChunkRequests(offset, blobs[i].size(), blobs[i].data(), requests);
      offset += blobs[i].size();
      const size_t padded_size = hwy::RoundUpTo(blobs[i].size(), kAlign);
      if (padded_size != blobs[i].size()) {
        const size_t padding = padded_size - blobs[i].size();
        HWY_ASSERT(padding <= kAlign);
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
#if HWY_OS_WIN
  DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN;
  HANDLE file = CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, flags, nullptr);
  if (file == INVALID_HANDLE_VALUE) return __LINE__;
  fd_ = _open_osfhandle(reinterpret_cast<intptr_t>(file), _O_RDONLY);
#else
  fd_ = open(filename, O_RDONLY);
#endif
  if (fd_ < 0) return __LINE__;

#if HWY_OS_LINUX && (!defined(__ANDROID_API__) || __ANDROID_API__ >= 21)
  // Doubles the readahead window, which seems slightly faster when cached.
  (void)posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif

  // Read first part of header to get actual size.
  BlobStore bs;
  if (!IO::Read(fd_, 0, sizeof(bs), &bs)) return __LINE__;
  const size_t padded_size = bs.PaddedHeaderSize();
  HWY_ASSERT(padded_size >= sizeof(bs));

  // Allocate full header.
  blob_store_ = BlobStore::Allocate(padded_size);
  if (!blob_store_) return __LINE__;

  // Copy what we already read (more efficient than seek + re-read).
  hwy::CopySameSize(&bs, blob_store_.get());
  // Read the rest of the header, but not the full file.
  uint8_t* bytes = reinterpret_cast<uint8_t*>(blob_store_.get());
  if (!IO::Read(fd_, sizeof(bs), padded_size - sizeof(bs),
                bytes + sizeof(bs))) {
    return __LINE__;
  }

  return blob_store_->CheckValidity(IO::FileSize(filename));
}

BlobReader::~BlobReader() {
  if (fd_ >= 0) {
    HWY_ASSERT(close(fd_) != -1);
  }
}

BlobError BlobReader::Enqueue(hwy::uint128_t key, void* data, size_t size) {
  uint64_t offset;
  size_t actual_size;
  if (!blob_store_->FindKey(key, offset, actual_size)) return __LINE__;
  if (actual_size != size) return __LINE__;

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
  const int fd = fd_;
  const auto& requests = requests_;
  std::atomic_flag err = ATOMIC_FLAG_INIT;
  // >5x speedup from parallel reads when cached.
  pool.Run(0, requests.size(),
           [fd, &requests, &err](uint64_t i, size_t /*thread*/) {
             if (!IO::Read(fd, requests[i].offset, requests[i].size,
                           requests[i].data)) {
               err.test_and_set();
             }
           });
  if (err.test_and_set()) return __LINE__;
  return 0;
}

BlobError BlobWriter::WriteAll(hwy::ThreadPool& pool,
                               const char* filename) const {
  HWY_ASSERT(keys_.size() == blobs_.size());

  // Concatenate blobs in memory.
  std::vector<BlobIO> requests = BlobStore::PrepareWriteRequests(
      keys_.data(), blobs_.data(), keys_.size());

  // Create/replace existing file.
#if HWY_OS_WIN
  DWORD flags = FILE_ATTRIBUTE_NORMAL;
  HANDLE file = CreateFileA(filename, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                            flags, nullptr);
  if (file == INVALID_HANDLE_VALUE) return __LINE__;
  const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(file), _O_WRONLY);
#else
  const int fd = open(filename, O_CREAT | O_RDWR | O_TRUNC, 0644);
#endif
  if (fd < 0) return __LINE__;

  std::atomic_flag err = ATOMIC_FLAG_INIT;
  pool.Run(0, requests.size(),
           [fd, &requests, &err](uint64_t i, size_t /*thread*/) {
             if (!IO::Write(requests[i].data, requests[i].size,
                            requests[i].offset, fd)) {
               err.test_and_set();
             }
           });
  HWY_ASSERT(close(fd) != -1);
  if (err.test_and_set()) return __LINE__;
  return 0;
}

}  // namespace gcpp
