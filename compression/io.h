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

#ifndef THIRD_PARTY_GEMMA_CPP_COMPRESSION_IO_H_
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_IO_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <string>
#include <utility>  // std::move

#include "util/allocator.h"
#include "hwy/base.h"

namespace gcpp {

// Forward-declare to break the circular dependency: OpenFileOrNull returns
// File and has a Path argument, and Path::Exists calls OpenFileOrNull. We
// prefer to define Exists inline because there are multiple io*.cc files.
struct Path;

using MapPtr = AlignedPtr2<const uint8_t[]>;

// Abstract base class enables multiple I/O backends in the same binary.
class File {
 public:
  File() = default;
  virtual ~File() = default;

  // Noncopyable.
  File(const File& other) = delete;
  const File& operator=(const File& other) = delete;

  // Returns size in bytes or 0.
  virtual uint64_t FileSize() const = 0;

  // Returns true if all the requested bytes were read.
  virtual bool Read(uint64_t offset, uint64_t size, void* to) const = 0;

  // Returns true if all the requested bytes were written.
  virtual bool Write(const void* from, uint64_t size, uint64_t offset) = 0;

  // Maps the entire file into read-only memory or returns nullptr on failure.
  // We do not support offsets because Windows requires them to be a multiple of
  // the allocation granularity, which is 64 KiB. Some implementations may fail
  // if the file is zero-sized and return a nullptr.
  virtual MapPtr Map() = 0;
};

// Returns nullptr on failure. `mode` is either "r" or "w+". This is not just
// named 'OpenFile' to avoid a conflict with Windows.h #define.
std::unique_ptr<File> OpenFileOrNull(const Path& filename, const char* mode);

// Wrapper for strings representing a path name. Differentiates vs. arbitrary
// strings and supports shortening for display purposes.
struct Path {
  Path() {}
  explicit Path(const char* p) : path(p) {}
  explicit Path(std::string p) : path(std::move(p)) {}

  Path& operator=(const char* other) {
    path = other;
    return *this;
  }

  std::string Shortened() const {
    constexpr size_t kMaxLen = 48;
    constexpr size_t kCutPoint = kMaxLen / 2 - 5;
    if (path.size() > kMaxLen) {
      return std::string(begin(path), begin(path) + kCutPoint) + " ... " +
             std::string(end(path) - kCutPoint, end(path));
    }
    if (path.empty()) return "[no path specified]";
    return path;
  }

  bool Empty() const { return path.empty(); }

  // Returns whether the file existed when this was called.
  bool Exists() const { return !!OpenFileOrNull(*this, "r"); }

  std::string path;
};

// Aborts on error.
static inline HWY_MAYBE_UNUSED std::string ReadFileToString(const Path& path) {
  std::unique_ptr<File> file = OpenFileOrNull(path, "r");
  if (!file) {
    HWY_ABORT("Failed to open %s", path.path.c_str());
  }
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

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_IO_H_
