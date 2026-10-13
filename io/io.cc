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

// Feature-test macros must be defined before including any system header,
// including C++ headers such as <new>, which may include <features.h>.
// Replacement for the _BSD_SOURCE specified by preadv documentation (glibc).
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif

// Safe to include here: does not include any system headers.
#include "hwy/detect_compiler_arch.h"

#if HWY_OS_LINUX
// Request POSIX 2008, including `pread()` and `posix_fadvise()`. This also
// implies `_POSIX_C_SOURCE`. Not done on BSD/Apple, where it would hide
// extensions such as `preadv` and `madvise`.
#if !defined(_XOPEN_SOURCE) || _XOPEN_SOURCE < 700
#undef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700  // SUSv4
#endif
// FilePosix should only be compiled on non-Windows. It is easier to
// check this in source code because we support multiple build systems. Note
// that IOBatch at the end of this TU is still compiled on all platforms.
#include <errno.h>

#endif  // HWY_OS_LINUX

// Make `off_t` 64-bit even on 32-bit systems. Works for Android >= r15c.
#undef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64

#include <stddef.h>

#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

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

#if !HWY_OS_WIN

#if GEMMA_IO_PREADV
#include <sys/uio.h>  // preadv
#endif

#include <fcntl.h>   // open
#include <limits.h>  // IOV_MAX
#include <stdint.h>
#include <sys/types.h>
// Old OSX may require sys/types.h before sys/mman.h.
#include <sys/mman.h>  // mmap
#include <sys/stat.h>  // fstat, O_RDONLY
#include <unistd.h>    // pread, pwrite, close

// Placeholder for internal header, do not modify.
#include "util/allocator.h"

namespace gcpp {

HWY_INLINE_VAR constexpr size_t kMaxBytesPerSyscall =
    0x7FFFF000;  // Linux limit

class FilePosix : public File {
  int fd_ = -1;

 public:
  explicit FilePosix(int fd) : fd_(fd) { HWY_ASSERT(fd >= 0); }
  ~FilePosix() override {
    if (fd_ >= 0) {
      // Do not retry on EINTR: Linux releases the fd regardless, and retrying
      // could close an fd that another thread was just given.
      HWY_ASSERT(close(fd_) == 0 || errno == EINTR);
    }
  }

  // pwrite is thread-safe and allows arbitrary offsets.
  bool IsAppendOnly() const override { return false; }

  uint64_t FileSize() const override {
    static_assert(sizeof(off_t) == 8, "64-bit off_t required");
    struct stat st;
    if (fstat(fd_, &st) != 0 || st.st_size < 0) {
      return 0;
    }
    return static_cast<uint64_t>(st.st_size);
  }

  bool Read(uint64_t offset, uint64_t size, void* to) const override {
    uint8_t* bytes = reinterpret_cast<uint8_t*>(to);
    uint64_t pos = 0;
    while (pos < size) {
      // pread seems to be faster than lseek + read when parallelized.
      const size_t want = static_cast<size_t>(
          HWY_MIN(size - pos, uint64_t{kMaxBytesPerSyscall}));
      const ssize_t bytes_read =
          pread(fd_, bytes + pos, want, static_cast<off_t>(offset + pos));
      if (bytes_read < 0) {
        if (errno == EINTR) continue;
        HWY_WARN(
            "Read failure at pos %zu within size %zu with offset %zu and "
            "errno %d\n",
            static_cast<size_t>(pos), static_cast<size_t>(size),
            static_cast<size_t>(offset), errno);
        break;
      }
      if (bytes_read == 0) break;  // EOF before reading `size` bytes.
      pos += static_cast<uint64_t>(bytes_read);
      HWY_ASSERT(pos <= size);
    }
    return pos == size;  // success if managed to read desired size
  }

  bool Write(const void* from, uint64_t size, uint64_t offset) override {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(from);
    uint64_t pos = 0;
    while (pos < size) {
      const size_t want = static_cast<size_t>(
          HWY_MIN(size - pos, uint64_t{kMaxBytesPerSyscall}));
      const ssize_t bytes_written =
          pwrite(fd_, bytes + pos, want, static_cast<off_t>(offset + pos));
      if (bytes_written < 0) {
        if (errno == EINTR) continue;
        HWY_WARN(
            "Write failure at pos %zu within size %zu with offset %zu and "
            "errno %d\n",
            static_cast<size_t>(pos), static_cast<size_t>(size),
            static_cast<size_t>(offset), errno);
        break;
      }
      if (bytes_written == 0) break;
      pos += static_cast<uint64_t>(bytes_written);
      HWY_ASSERT(pos <= size);
    }
    return pos == size;  // success if managed to write desired size
  }

  MapPtr Map() override {
    const uint64_t file_size = FileSize();
    if (file_size == 0 || file_size > static_cast<uint64_t>(~size_t{0})) {
      return MapPtr();
    }
    const size_t mapping_size = static_cast<size_t>(file_size);
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
  const bool is_read = mode[0] != 'w';
  // Tries POSIX first because local paths (including FUSE mounts) are faster.
  const int flags = is_read ? O_RDONLY : O_CREAT | O_RDWR | O_TRUNC;
  int fd;
  do {
    fd = open(filename.path.c_str(), flags, 0644);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) return std::unique_ptr<File>();

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
  std::unique_ptr<File> file = OpenFileOrNull(filename, mode);
  if (!file) {
    HWY_ABORT("Failed to open %s, errno %d", filename.path.c_str(), errno);
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

bool IOBatch::Add(void* mem, size_t bytes) {
  // Always accept the first span, otherwise callers would loop forever.
  if (!spans_.empty()) {
    if (spans_.size() >= kMaxSpans) return false;
#if !HWY_OS_WIN
    if (total_bytes_ + bytes > kMaxBytesPerSyscall) return false;
#endif
  }
  spans_.push_back(IOSpan{mem, bytes});
  total_bytes_ += bytes;
  return true;
}

int InternalInit() {
  return 0;  // currently unused
}

uint64_t IOBatch::Read(const File& file) const {
  HWY_ASSERT(!spans_.empty());

#if GEMMA_IO_PREADV
  const int fd = file.Handle();
  if (fd != -1) {
    static_assert(sizeof(IOSpan) == sizeof(iovec), "IOSpan must match iovec");
    const IOSpan* spans = spans_.data();
    size_t num_spans = spans_.size();
    std::vector<IOSpan> remaining;  // Only used after a short read.
    uint64_t pos = 0;
    while (pos < total_bytes_) {
      const ssize_t bytes_read = preadv(
          fd, reinterpret_cast<const iovec*>(spans),
          static_cast<int>(num_spans), static_cast<off_t>(offset_ + pos));
      if (bytes_read < 0) {
        if (errno == EINTR) continue;  // signal: retry
        HWY_WARN(
            "preadv(%d) for %4zu spans from offset %12zu failed, errno %d.", fd,
            num_spans, static_cast<size_t>(offset_ + pos), errno);
        return 0;
      }
      if (bytes_read == 0) return 0;  // EOF before reading `total_bytes_`.
      pos += static_cast<uint64_t>(bytes_read);
      HWY_ASSERT(pos <= total_bytes_);
      if (pos == total_bytes_) break;

      // Short read: rebuild the list of not yet (fully) read spans.
      remaining.clear();
      uint64_t skip = pos;
      for (const IOSpan& span : spans_) {
        if (skip >= span.bytes) {
          skip -= span.bytes;
          continue;
        }
        const size_t skip_s = static_cast<size_t>(skip);
        remaining.push_back(IOSpan{static_cast<uint8_t*>(span.mem) + skip_s,
                                   span.bytes - skip_s});
        skip = 0;
      }
      spans = remaining.data();
      num_spans = remaining.size();
    }
    return total_bytes_;
  }
#endif  // GEMMA_IO_PREADV

  // 1. Fast Path: Check if destination memory across all spans is contiguous.
  bool is_contiguous = true;
  const uint8_t* next_expected =
      reinterpret_cast<const uint8_t*>(spans_[0].mem);
  for (const IOSpan& span : spans_) {
    if (reinterpret_cast<const uint8_t*>(span.mem) != next_expected) {
      is_contiguous = false;
      break;
    }
    next_expected += span.bytes;
  }

  // Zero-copy direct read: 1 single bulk RPC directly into destination memory.
  if (is_contiguous) {
    if (file.Read(offset_, total_bytes_, spans_[0].mem)) {
      return total_bytes_;
    }
  } else {
    // 2. Slow Path (Strided Padding): Single bulk read + scatter memcpy.
    std::unique_ptr<uint8_t[]> temp_buf(new (std::nothrow)
                                            uint8_t[total_bytes_]);
    if (temp_buf && file.Read(offset_, total_bytes_, temp_buf.get())) {
      const uint8_t* src = temp_buf.get();
      for (const IOSpan& span : spans_) {
        std::memcpy(span.mem, src, span.bytes);
        src += span.bytes;
      }
      return total_bytes_;
    } else {
      HWY_WARN(
          "Failed to read %zu bytes from offset %zu in file with %zu spans",
          total_bytes_, offset_, spans_.size());
    }
  }

  // 3. Fallback if bulk read or temp allocation fails: per-span sequential
  // reads.
  uint64_t total = 0;
  uint64_t offset = offset_;
  for (const IOSpan& span : spans_) {
    if (!file.Read(offset, span.bytes, span.mem)) return 0;
    total += span.bytes;
    offset += span.bytes;
  }
  return total;
}

}  // namespace gcpp
