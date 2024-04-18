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

#include <stdint.h>

namespace gcpp {

// unique_ptr-like interface with RAII, but not (yet) moveable.
class File {
 public:
  File() = default;
  ~File() { Close(); }
  File(const File& other) = delete;
  const File& operator=(const File& other) = delete;

  // Returns false on failure. `mode` is either "r" or "w+".
  bool Open(const char* filename, const char* mode);

  // No effect if `Open` returned false or `Close` already called.
  void Close();

  // Returns size in bytes or 0.
  uint64_t FileSize() const;

  // Returns true if all the requested bytes were read.
  bool Read(uint64_t offset, uint64_t size, void* to) const;

  // Returns true if all the requested bytes were written.
  bool Write(const void* from, uint64_t size, uint64_t offset);

 private:
  intptr_t p_ = 0;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_IO_H_
