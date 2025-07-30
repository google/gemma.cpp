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

#include "hwy/detect_compiler_arch.h"
// Only compile this file on Windows; it replaces io.cc. It is easier to check
// this in source code because we support multiple build systems.
#if HWY_OS_WIN

#include <stddef.h>
#include <stdint.h>

#include "io/io.h"
#include "util/allocator.h"
#include "hwy/base.h"  // HWY_ASSERT
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef VC_EXTRALEAN
#define VC_EXTRALEAN
#endif
#include <Windows.h>

namespace gcpp {

class FileWin : public File {
  HANDLE hFile_ = INVALID_HANDLE_VALUE;

 public:
  FileWin(HANDLE hFile) : hFile_(hFile) {
    HWY_ASSERT(hFile != INVALID_HANDLE_VALUE);
  }
  ~FileWin() override {
    if (hFile_ != INVALID_HANDLE_VALUE) {
      HWY_ASSERT(CloseHandle(hFile_) != 0);
    }
  }

  // WriteFile is thread-safe and allows arbitrary offsets.
  bool IsAppendOnly() const override { return false; }

  uint64_t FileSize() const override {
    DWORD hi;
    const DWORD lo = GetFileSize(hFile_, &hi);
    if (lo == INVALID_FILE_SIZE) return 0;
    return (static_cast<uint64_t>(hi) << 32) | lo;
  }

  bool Read(uint64_t offset, uint64_t size, void* to) const override {
    uint8_t* bytes = reinterpret_cast<uint8_t*>(to);
    OVERLAPPED overlapped = {0};
    // Loop is required because ReadFile[Ex] size argument is 32-bit.
    while (size != 0) {
      overlapped.Offset = offset & 0xFFFFFFFF;
      overlapped.OffsetHigh = (offset >> 32) & 0xFFFFFFFF;
      const DWORD want =
          static_cast<DWORD>(HWY_MIN(size, uint64_t{0xFFFFFFFF}));
      DWORD got;
      if (!ReadFile(hFile_, bytes, want, &got, &overlapped)) {
        if (GetLastError() != ERROR_HANDLE_EOF) {
          return false;
        }
      }
      offset += got;
      bytes += got;
      size -= got;
    }
    return true;  // read everything => success
  }

  bool Write(const void* from, uint64_t size, uint64_t offset) override {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(from);
    OVERLAPPED overlapped = {0};
    // Loop is required because WriteFile[Ex] size argument is 32-bit.
    while (size != 0) {
      overlapped.Offset = offset & 0xFFFFFFFF;
      overlapped.OffsetHigh = (offset >> 32) & 0xFFFFFFFF;
      const DWORD want =
          static_cast<DWORD>(HWY_MIN(size, uint64_t{0xFFFFFFFF}));
      DWORD got;
      if (!WriteFile(hFile_, bytes, want, &got, &overlapped)) {
        if (GetLastError() != ERROR_HANDLE_EOF) {
          return false;
        }
      }
      offset += got;
      bytes += got;
      size -= got;
    }
    return true;  // wrote everything => success
  }

  MapPtr Map() override {
    if (hFile_ == INVALID_HANDLE_VALUE) return MapPtr();

    // Size=0 means the entire file.
    HANDLE hMapping =
        CreateFileMappingA(hFile_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    // Offset zero and size=0 means the entire file.
    void* ptr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!ptr) return MapPtr();
    return MapPtr(static_cast<const uint8_t*>(ptr),
                  DeleterFunc([hMapping](void* ptr) {
                    HWY_ASSERT(UnmapViewOfFile(ptr));
                    HWY_ASSERT(CloseHandle(hMapping));
                  }));
  }
};  // FileWin

std::unique_ptr<File> OpenFileOrNull(const Path& filename, const char* mode) {
  const bool is_read = mode[0] != 'w';
  const DWORD flags =
      FILE_ATTRIBUTE_NORMAL | (is_read ? FILE_FLAG_SEQUENTIAL_SCAN : 0);
  const DWORD access = is_read ? GENERIC_READ : GENERIC_WRITE;
  const DWORD share = is_read ? FILE_SHARE_READ : 0;
  const DWORD create = is_read ? OPEN_EXISTING : CREATE_ALWAYS;
  const HANDLE hFile = CreateFileA(filename.path.c_str(), access, share,
                                   nullptr, create, flags, nullptr);
  if (hFile == INVALID_HANDLE_VALUE) return std::unique_ptr<File>();
  return std::make_unique<FileWin>(hFile);
}

}  // namespace gcpp
#endif  // HWY_OS_WIN
