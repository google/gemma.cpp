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

// Safe to be first, does not include POSIX headers.
#include "hwy/detect_compiler_arch.h"
// Request POSIX 2008, including `pread()` and `posix_fadvise()`. This also
// implies `_POSIX_C_SOURCE`.
#if !defined(_XOPEN_SOURCE) || _XOPEN_SOURCE < 700
#undef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700  // SUSv4
#endif

// Make `off_t` 64-bit even on 32-bit systems. Works for Android >= r15c.
#undef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64

#include <stddef.h>

#include <memory>
#include <string>

#include "io/io.h"
#include "hwy/base.h"  // HWY_ASSERT

#if (HWY_OS_LINUX || HWY_OS_FREEBSD) && \
    (!defined(__ANDROID_API__) || __ANDROID_API__ >= 24)
#define GEMMA_IO_PREADV 1
#else
#define GEMMA_IO_PREADV 0
#endif

#if (HWY_OS_LINUX || HWY_OS_FREEBSD) && \
    (!defined(__ANDROID_API__) || __ANDROID_API__ >= 21)
#define GEMMA_IO_FADVISE 1
#else
#define GEMMA_IO_FADVISE 0
#endif

// FilePosix should only be compiled on non-Windows. It is easier to
// check this in source code because we support multiple build systems. Note
// that IOBatch at the end of this TU is still compiled on all platforms.
#if !HWY_OS_WIN

#if GEMMA_IO_PREADV
// Replacement for the _BSD_SOURCE specified by preadv documentation.
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#include <errno.h>
#include <sys/uio.h>  // preadv
#endif                // GEMMA_IO_PREADV

#include <fcntl.h>   // open
#include <limits.h>  // IOV_MAX
#include <stdint.h>
#include <stdio.h>  // SEEK_END - unistd isn't enough for IDE.
#include <sys/types.h>
// Old OSX may require sys/types.h before sys/mman.h.
#include <sys/mman.h>  // mmap
#include <sys/stat.h>  // O_RDONLY
#include <unistd.h>    // read, write, close

// Placeholder for internal header, do not modify.
#include "util/allocator.h"

namespace gcpp {

class FilePosix : public File {
  int fd_ = 0;

 public:
  explicit FilePosix(int fd) : fd_(fd) { HWY_ASSERT(fd > 0); }
  ~FilePosix() override {
    if (fd_ != 0) {
      HWY_ASSERT(close(fd_) != -1);
    }
  }

  // pwrite is thread-safe and allows arbitrary offsets.
  bool IsAppendOnly() const override { return false; }

  uint64_t FileSize() const override {
    static_assert(sizeof(off_t) == 8, "64-bit off_t required");
    const off_t size = lseek(fd_, 0, SEEK_END);
    if (size < 0) {
      return 0;
    }
    return static_cast<uint64_t>(size);
  }

  bool Read(uint64_t offset, uint64_t size, void* to) const override {
    uint8_t* bytes = reinterpret_cast<uint8_t*>(to);
    uint64_t pos = 0;
    for (;;) {
      // pread seems to be faster than lseek + read when parallelized.
      const auto bytes_read = pread(fd_, bytes + pos, size - pos, offset + pos);
      if (bytes_read <= 0) break;
      pos += bytes_read;
      HWY_ASSERT(pos <= size);
      if (pos == size) break;
    }
    return pos == size;  // success if managed to read desired size
  }

  bool Write(const void* from, uint64_t size, uint64_t offset) override {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(from);
    uint64_t pos = 0;
    for (;;) {
      const auto bytes_written =
          pwrite(fd_, bytes + pos, size - pos, offset + pos);
      if (bytes_written <= 0) break;
      pos += bytes_written;
      HWY_ASSERT(pos <= size);
      if (pos == size) break;
    }
    return pos == size;  // success if managed to write desired size
  }

  MapPtr Map() override {
    const size_t mapping_size = FileSize();
    // No `MAP_POPULATE` because we do not want to wait for I/O, and
    // `MAP_NONBLOCK` is not guaranteed. `MAP_HUGETLB` fails. `MAP_SHARED` is
    // more efficient than `MAP_PRIVATE`; the main difference is that the former
    // will eventually see subsequent changes to the file.
    const int flags = MAP_SHARED;
    void* mapping =
        mmap(nullptr, mapping_size, PROT_READ, flags, fd_, /*offset=*/0);
    if (mapping == MAP_FAILED) return MapPtr();

#ifdef MADV_WILLNEED  // Missing on some OSX.
    // (Maybe) initiate readahead.
    madvise(mapping, mapping_size, MADV_WILLNEED);
#endif

    return MapPtr(static_cast<const uint8_t*>(mapping),
                  DeleterFunc([mapping_size](void* ptr) {
                    HWY_ASSERT(munmap(ptr, mapping_size) == 0);
                  }));
  }

  int Handle() const override { return fd_; }
};  // FilePosix

HWY_MAYBE_UNUSED extern std::unique_ptr<File> OpenFileGoogle(
    const Path& filename, const char* mode);

std::unique_ptr<File> OpenFileOrNull(const Path& filename, const char* mode) {
  std::unique_ptr<File> file;  // OpenFileGoogle omitted
  if (file) return file;

  const bool is_read = mode[0] != 'w';
  const int flags = is_read ? O_RDONLY : O_CREAT | O_RDWR | O_TRUNC;
  const int fd = open(filename.path.c_str(), flags, 0644);
  if (fd < 0) return file;

#if GEMMA_IO_FADVISE
  if (is_read) {
    // Doubles the readahead window, which seems slightly faster when cached.
    (void)posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
  }
#endif  // GEMMA_IO_FADVISE

  return std::make_unique<FilePosix>(fd);
}

}  // namespace gcpp

#endif  // !HWY_OS_WIN

namespace gcpp {

std::unique_ptr<File> OpenFileOrAbort(const Path& filename, const char* mode) {
  std::unique_ptr<File> file = OpenFileOrNull(filename, "r");
  if (!file) {
    HWY_ABORT("Failed to open %s", filename.path.c_str());
  }
  return file;
}

std::string ReadFileToString(const Path& path) {
  std::unique_ptr<File> file = OpenFileOrAbort(path, "r");
  const size_t size = file->FileSize();
  if (size == 0) {
    HWY_ABORT("Empty file %s", path.path.c_str());
  }
  std::string content(size, ' ');
  if (!file->Read(0, size, content.data())) {
    HWY_ABORT("Failed to read %s", path.path.c_str());
  }
  return content;
}

#ifdef IOV_MAX
constexpr size_t kMaxSpans = IOV_MAX;
#else
constexpr size_t kMaxSpans = 1024;  // Linux limit
#endif

IOBatch::IOBatch(uint64_t offset, size_t key_idx)
    : offset_(offset), key_idx_(key_idx) {
  spans_.reserve(kMaxSpans);
}

// Returns true if the batch was full; if so, call again on the new batch.
bool IOBatch::Add(void* mem, size_t bytes) {
  if (spans_.size() >= kMaxSpans) return false;
  if (total_bytes_ + bytes > 0x7FFFF000) return false;  // Linux limit
  spans_.push_back({.mem = mem, .bytes = bytes});
  total_bytes_ += bytes;
  return true;
}

void InternalInit() {
}

uint64_t IOBatch::Read(const File& file) const {
#if GEMMA_IO_PREADV
  HWY_ASSERT(!spans_.empty());

  ssize_t bytes_read;
  for (;;) {
    bytes_read =
        preadv(file.Handle(), reinterpret_cast<const iovec*>(spans_.data()),
               static_cast<int>(spans_.size()), offset_);
    if (bytes_read >= 0) break;
    if (errno == EINTR) continue;  // signal: retry
    HWY_WARN("preadv failed, errno %d.", errno);
    return 0;
  }
  return static_cast<uint64_t>(bytes_read);
#else
  uint64_t total = 0;
  uint64_t offset = offset_;
  for (const IOSpan& span : spans_) {
    if (!file.Read(offset, span.bytes, span.mem)) return 0;
    total += span.bytes;
    offset += span.bytes;
  }
  return total;
#endif
}

}  // namespace gcpp
