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

#include "hwy/aligned_allocator.h"  // Span
#include "hwy/base.h"
// IWYU pragma: end_exports

#if HWY_IS_MSAN
#include <sanitizer/msan_interface.h>
#endif

namespace gcpp {

// For hwy::BitSet4096. Note that KVs are extremely large for such batches.
HWY_INLINE_VAR constexpr size_t kMaxBatchSize = 4096;

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

static inline void MaybePrintInitialized(const void* ptr, size_t size) {
#if HWY_IS_MSAN
  __msan_print_shadow(ptr, size);
#else
  (void)ptr;
  (void)size;
#endif
}

static inline intptr_t MaybeTestInitialized(const void* ptr, size_t size) {
#if HWY_IS_MSAN
  return __msan_test_shadow(ptr, size);
#else
  (void)ptr;
  (void)size;
  return 0;
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

using Logits = hwy::Span<float>;  // size() is vocab_size.

// Non-cryptographic 64-bit pseudo-random number generator. Supports random or
// deterministic seeding.
//
// Based on 5-round AES-CTR. Supports 2^64 streams, each with period 2^64. This
// is useful for parallel sampling. Each thread can generate the stream for a
// particular task, without caring about prior/subsequent generations.
class alignas(16) AesCtrEngine {
  // "Large-scale randomness study of security margins for 100+ cryptographic
  // functions": at least four.
  // "Parallel Random Numbers: As Easy as 1, 2, 3": four not Crush-resistant.
  static constexpr size_t kRounds = 5;

 public:
  // If `deterministic` is true, uses a fixed seed; otherwise, attempts to
  // grab entropy from the OS.
  explicit AesCtrEngine(bool deterministic);

  // Pure and thread safe; typically called via `RngStream`, which increments
  // `counter`. Throughput is about 100M/s on 3 GHz Skylake. It could be
  // increased 4x via unrolling by the AES latency (4-7 cycles), but because
  // users generally call once at a time, this requires buffering, which is not
  // worth the complexity in this application.
  uint64_t operator()(uint64_t stream, uint64_t counter) const;

 private:
  uint64_t key_[2 * (1 + kRounds)];
};

// Flyweight per-thread adapter that maintains the counter. Conforms to C++
// `UniformRandomBitGenerator`.
class RngStream {
 public:
  RngStream() = default;  // Allow C arrays with subsequent initialization.

  // Binds to an engine, which holds the seed and must outlive this object.
  // Sets the stream; any other `RngStream` with the same `counter_rng` and
  // `stream` will return the same sequence. This is typically the task ID, so
  // that threads can independently generate values for each task.
  RngStream(const AesCtrEngine& counter_rng, uint64_t stream)
      : engine_(&counter_rng), stream_(stream), counter_(0) {}

  using result_type = uint64_t;
  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return ~result_type{0}; }
  result_type operator()() { return (*engine_)(stream_, counter_++); }

 private:
  const AesCtrEngine* engine_ = nullptr;
  uint64_t stream_ = 0;  // immutable after ctor
  uint64_t counter_ = 0;
  // Prevent false sharing if used by multiple threads.
  HWY_MAYBE_UNUSED uint8_t padding_[HWY_ALIGNMENT - 16 - sizeof(engine_)];
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_BASICS_H_
