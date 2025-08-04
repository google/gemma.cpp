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

#ifndef THIRD_PARTY_GEMMA_CPP_UTIL_BASICS_H_
#define THIRD_PARTY_GEMMA_CPP_UTIL_BASICS_H_

// IWYU pragma: begin_exports
#include <stddef.h>
#include <stdint.h>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
// IWYU pragma: end_exports

#if HWY_IS_MSAN
#include <sanitizer/msan_interface.h>
#endif

namespace gcpp {

// Maximum number of packages (CPU sockets) to use. `ThreadingArgs` verifies the
// runtime `max_packages` does not exceed this. MatMul's outer per-package loop
// is disabled if this is 1.
constexpr size_t kMaxPackages = 1;

enum class Tristate : int32_t { kFalse = 0, kTrue = 1, kDefault = -1 };

static inline const char* ToString(Tristate t) {
  switch (t) {
    case Tristate::kFalse:
      return "false";
    case Tristate::kTrue:
      return "true";
    case Tristate::kDefault:
      return "default";
    default:
      HWY_ABORT("Bug: unknown Tristate %d", static_cast<int>(t));
  }
}

using BF16 = hwy::bfloat16_t;

static inline void MaybeCheckInitialized(const void* ptr, size_t size) {
#if HWY_IS_MSAN
  __msan_check_mem_is_initialized(ptr, size);
#else
  (void)ptr;
  (void)size;
#endif
}

// Shared between gemma.h and ops-inl.h.
#pragma pack(push, 1)
struct TokenAndProb {
  int token;
  float prob;
};
#pragma pack(pop)

// Entire size of a 2D array.
struct Extents2D {
  constexpr Extents2D() : rows(0), cols(0) {}
  constexpr Extents2D(size_t rows, size_t cols) : rows(rows), cols(cols) {}

  size_t Area() const { return rows * cols; }

  size_t rows;
  size_t cols;
};

struct IndexRange {
  IndexRange() = default;
  IndexRange(size_t begin, size_t end) : begin_(begin), end_(end) {
    HWY_DASSERT(begin < end);
  }
  IndexRange(const IndexRange& other) = default;
  IndexRange& operator=(const IndexRange& other) = default;

  size_t Num() const { return end_ - begin_; }
  bool Contains(IndexRange other) const {
    return other.begin_ >= begin_ && other.end_ <= end_;
  }

  // Enable range-based for loops.
  class Iterator {
   public:
    Iterator(size_t i) : i_(i) {}

    Iterator& operator++() {
      ++i_;
      return *this;
    }
    bool operator!=(const Iterator& other) const { return i_ != other.i_; }
    size_t operator*() const { return i_; }
    // Enable using begin() directly as a size_t.
    operator size_t() const { return i_; }

   private:
    size_t i_;
  };
  Iterator begin() const { return Iterator(begin_); }
  Iterator end() const { return Iterator(end_); }

  size_t begin_;
  size_t end_;
};

static inline IndexRange MakeIndexRange(size_t begin, size_t end,
                                        size_t max_size) {
  return IndexRange(begin, HWY_MIN(begin + max_size, end));
}
}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_BASICS_H_
