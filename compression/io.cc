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
#include "compression/io.h"

// 1.5x slowdown vs. POSIX (200 ms longer startup), hence opt-in.
#ifdef GEMMA_IO_GOOGLE
#include "compression/io_google.cc"
#else

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

#include <fcntl.h>  // open
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>     // SEEK_END - unistd isn't enough for IDE.
#include <sys/stat.h>  // O_RDONLY

#include "hwy/base.h"  // HWY_ASSERT
#include "hwy/detect_compiler_arch.h"
#if HWY_OS_WIN
#include <fileapi.h>
#include <io.h>  // read, write, close
#else
#include <unistd.h>  // read, write, close
#endif

namespace gcpp {

// Emulate missing POSIX functions.
#if HWY_OS_WIN
namespace {

static inline int open(const char* filename, int flags, int mode = 0) {
  const bool is_read = (flags & _O_RDONLY) != 0;
  const DWORD win_flags =
      FILE_ATTRIBUTE_NORMAL | (is_read ? FILE_FLAG_SEQUENTIAL_SCAN : 0);
  const DWORD access = is_read ? GENERIC_READ : GENERIC_WRITE;
  const DWORD share = is_read ? FILE_SHARE_READ : 0;
  const DWORD create = is_read ? OPEN_EXISTING : CREATE_ALWAYS;
  const HANDLE file =
      CreateFileA(filename, access, share, nullptr, create, win_flags, nullptr);
  if (file == INVALID_HANDLE_VALUE) return -1;
  return _open_osfhandle(reinterpret_cast<intptr_t>(file), flags);
}

static inline off_t lseek(int fd, off_t offset, int whence) {
  return _lseeki64(fd, offset, whence);
}

static inline int64_t pread(int fd, void* buf, uint64_t size, uint64_t offset) {
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

static inline int64_t pwrite(int fd, const void* buf, uint64_t size,
                             uint64_t offset) {
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

}  // namespace
#endif  // HWY_OS_WIN

bool File::Open(const char* filename, const char* mode) {
  const bool is_read = mode[0] != 'w';
  const int flags = is_read ? O_RDONLY : O_CREAT | O_RDWR | O_TRUNC;
  int fd = open(filename, flags, 0644);
  if (fd < 0) {
    p_ = 0;
    return false;
  }

  if (is_read) {
#if HWY_OS_LINUX && (!defined(__ANDROID_API__) || __ANDROID_API__ >= 21)
    // Doubles the readahead window, which seems slightly faster when cached.
    (void)posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
  }

  p_ = static_cast<intptr_t>(fd);
  return true;
}

void File::Close() {
  const int fd = static_cast<int>(p_);
  if (fd > 0) {
    HWY_ASSERT(close(fd) != -1);
    p_ = 0;
  }
}

uint64_t File::FileSize() const {
  static_assert(sizeof(off_t) == 8, "64-bit off_t required");
  const int fd = static_cast<int>(p_);
  const off_t size = lseek(fd, 0, SEEK_END);
  if (size < 0) {
    return 0;
  }
  return static_cast<uint64_t>(size);
}

bool File::Read(uint64_t offset, uint64_t size, void* to) const {
  const int fd = static_cast<int>(p_);
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

bool File::Write(const void* from, uint64_t size, uint64_t offset) {
  const int fd = static_cast<int>(p_);
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

}  // namespace gcpp
#endif  // GEMMA_IO_GOOGLE
