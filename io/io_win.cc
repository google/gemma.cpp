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

#include <string>
#include <vector>

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
namespace {

// Converts UTF-8 to UTF-16 (including the terminating null), or returns an
// empty vector on failure.
std::vector<wchar_t> Utf8ToWide(const std::string& utf8) {
  std::vector<wchar_t> wide;
  if (utf8.size() > static_cast<size_t>(INT32_MAX)) return wide;
  const int len_with_null = static_cast<int>(utf8.size()) + 1;
  const int num_wide = MultiByteToWideChar(
      CP_UTF8, MB_ERR_INVALID_CHARS, utf8.c_str(), len_with_null, nullptr, 0);
  if (num_wide <= 0) return wide;
  wide.resize(static_cast<size_t>(num_wide));
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.c_str(),
                          len_with_null, wide.data(), num_wide) != num_wide) {
    wide.clear();
  }
  return wide;
}

// Manual-reset event for waiting on one overlapped request. Each concurrent
// request requires its own event because waiting on the file handle itself
// cannot distinguish between multiple outstanding requests.
class OverlappedEvent {
 public:
  OverlappedEvent()
      : event_(CreateEventW(/*lpEventAttributes=*/nullptr,
                            /*bManualReset=*/TRUE, /*bInitialState=*/FALSE,
                            /*lpName=*/nullptr)) {}
  ~OverlappedEvent() {
    if (event_ != nullptr) HWY_ASSERT(CloseHandle(event_));
  }
  OverlappedEvent(const OverlappedEvent&) = delete;
  OverlappedEvent& operator=(const OverlappedEvent&) = delete;

  HANDLE Get() const { return event_; }

 private:
  HANDLE event_;
};

// Issues a single ReadFile (`is_write` = false) or WriteFile at `offset` and
// waits for it to complete. Returns the number of bytes transferred, or 0 on
// failure or EOF.
DWORD TransferAt(HANDLE file, HANDLE event, bool is_write, uint64_t offset,
                 void* bytes, DWORD want) {
  OVERLAPPED overlapped = {};
  overlapped.Offset = static_cast<DWORD>(offset & 0xFFFFFFFFu);
  overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
  // ReadFile/WriteFile reset the event to non-signaled when they begin.
  overlapped.hEvent = event;
  // For overlapped handles, the byte count must be obtained from
  // GetOverlappedResult, hence pass nullptr.
  const BOOL ok = is_write ? WriteFile(file, bytes, want, nullptr, &overlapped)
                           : ReadFile(file, bytes, want, nullptr, &overlapped);
  if (!ok && GetLastError() != ERROR_IO_PENDING) return 0;  // incl. EOF
  DWORD got = 0;
  if (!GetOverlappedResult(file, &overlapped, &got, /*bWait=*/TRUE)) return 0;
  return got;
}

}  // namespace

// Opened with FILE_FLAG_OVERLAPPED so that concurrent Read/Write calls on the
// same handle can proceed in parallel. Otherwise, the I/O manager would
// serialize them via a per-file-object lock, even with explicit offsets. Each
// call still waits for its own completion, hence the interface is synchronous.
class FileWin : public File {
  HANDLE hFile_ = INVALID_HANDLE_VALUE;

 public:
  explicit FileWin(HANDLE hFile) : hFile_(hFile) {
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
    LARGE_INTEGER size;
    if (!GetFileSizeEx(hFile_, &size) || size.QuadPart < 0) return 0;
    return static_cast<uint64_t>(size.QuadPart);
  }

  bool Read(uint64_t offset, uint64_t size, void* to) const override {
    return Transfer(/*is_write=*/false, offset, size, to);
  }

  bool Write(const void* from, uint64_t size, uint64_t offset) override {
    // WriteFile does not modify the buffer.
    return Transfer(/*is_write=*/true, offset, size, const_cast<void*>(from));
  }

  MapPtr Map() override {
    const uint64_t file_size = FileSize();
    if (file_size == 0 || file_size > static_cast<uint64_t>(~size_t{0})) {
      return MapPtr();
    }

    // Size=0 means the entire file.
    HANDLE hMapping =
        CreateFileMappingW(hFile_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (hMapping == nullptr) return MapPtr();
    // Offset zero and size=0 means the entire file.
    void* ptr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!ptr) {
      HWY_ASSERT(CloseHandle(hMapping));
      return MapPtr();
    }
    return MapPtr(static_cast<const uint8_t*>(ptr),
                  DeleterFunc([hMapping](void* ptr) {
                    HWY_ASSERT(UnmapViewOfFile(ptr));
                    HWY_ASSERT(CloseHandle(hMapping));
                  }));
  }

 private:
  bool Transfer(bool is_write, uint64_t offset, uint64_t size,
                void* bytes) const {
    if (size == 0) return true;
    // One event per call (not per chunk) suffices because chunks are
    // sequential. Creating it is cheap relative to disk I/O.
    const OverlappedEvent event;
    if (event.Get() == nullptr) return false;
    uint8_t* pos = static_cast<uint8_t*>(bytes);
    // Loop is required because the ReadFile/WriteFile size argument is 32-bit.
    while (size != 0) {
      const DWORD want = static_cast<DWORD>(HWY_MIN(size, uint64_t{1} << 30));
      const DWORD got =
          TransferAt(hFile_, event.Get(), is_write, offset, pos, want);
      if (got == 0) return false;
      pos += got;
      offset += got;
      size -= got;
    }
    return true;
  }
};  // FileWin

std::unique_ptr<File> OpenFileOrNull(const Path& filename, const char* mode) {
  const bool is_read = mode[0] != 'w';
  const DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED |
                      (is_read ? FILE_FLAG_SEQUENTIAL_SCAN : 0);
  const DWORD access = is_read ? GENERIC_READ : (GENERIC_READ | GENERIC_WRITE);
  const DWORD share =
      is_read ? FILE_SHARE_READ
              : (FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE);
  const DWORD create = is_read ? OPEN_EXISTING : CREATE_ALWAYS;
  // Path is UTF-8; CreateFileA would interpret it in the ANSI code page.
  const std::vector<wchar_t> wide = Utf8ToWide(filename.path);
  if (wide.empty()) return std::unique_ptr<File>();
  const HANDLE hFile =
      CreateFileW(wide.data(), access, share, nullptr, create, flags, nullptr);
  if (hFile == INVALID_HANDLE_VALUE) return std::unique_ptr<File>();
  return std::make_unique<FileWin>(hFile);
}

}  // namespace gcpp
#endif  // HWY_OS_WIN
