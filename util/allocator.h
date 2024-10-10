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

#ifndef THIRD_PARTY_GEMMA_CPP_UTIL_ALLOCATOR_H_
#define THIRD_PARTY_GEMMA_CPP_UTIL_ALLOCATOR_H_

#include <stddef.h>
#include <stdint.h>

#include "hwy/aligned_allocator.h"  // IWYU pragma: export
#include "hwy/base.h"

namespace gcpp {

using ByteStorageT = hwy::AlignedFreeUniquePtr<uint8_t[]>;

template <typename T>
ByteStorageT AllocateSizeof() {
  return hwy::AllocateAligned<uint8_t>(sizeof(T));
}

// Owns dynamically-allocated aligned memory for a batch of row vectors.
// This can be seen as a (batch_size x len) matrix.
template <typename T>
class RowVectorBatch {
 public:
  // Default ctor for Activations ctor.
  RowVectorBatch() : batch_size_(0), len_(0) {}
  // Main ctor, called from Activations::Allocate.
  RowVectorBatch(size_t batch_size, size_t len)
      : batch_size_(batch_size), len_(len) {
    mem_ = hwy::AllocateAligned<T>(batch_size * len);
  }

  // Move-only
  RowVectorBatch(RowVectorBatch&) noexcept = delete;
  RowVectorBatch& operator=(RowVectorBatch&) noexcept = delete;
  RowVectorBatch(RowVectorBatch&&) noexcept = default;
  RowVectorBatch& operator=(RowVectorBatch&&) noexcept = default;

  size_t BatchSize() const { return batch_size_; }
  size_t Len() const { return len_; }

  // Returns the given row vector of length `Len()`.
  T* Batch(size_t batch_idx) {
    HWY_DASSERT(batch_idx < batch_size_);
    return mem_.get() + batch_idx * len_;
  }
  const T* Batch(size_t batch_idx) const {
    HWY_DASSERT(batch_idx < batch_size_);
    return mem_.get() + batch_idx * len_;
  }

  // For MatMul or other operations that process the entire batch at once.
  T* All() { return mem_.get(); }
  const T* Const() const { return mem_.get(); }
  size_t NumBytes() const { return batch_size_ * len_ * sizeof(T); }

 private:
  hwy::AlignedFreeUniquePtr<T[]> mem_;
  size_t batch_size_;  // rows in the matrix
  size_t len_;         // columns in the matrix = vector length
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_ALLOCATOR_H_
