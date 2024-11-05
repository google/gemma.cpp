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
#include "hwy/base.h"  // HWY_IS_MSAN
// IWYU pragma: end_exports

#if HWY_IS_MSAN
#include <sanitizer/msan_interface.h>
#endif

namespace gcpp {

enum class Tristate : int32_t { kFalse = 0, kTrue = 1, kDefault = -1 };

static inline const char* ToString(Tristate t) {
  switch (t) {
    case Tristate::kFalse:
      return "false";
    case Tristate::kTrue:
      return "true";
    case Tristate::kDefault:
      return "default";
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
struct TokenAndProb {
  int token;
  float prob;
};

// Entire size of a 2D array. By contrast, Range2D is a subrange.
struct Extents2D {
  Extents2D() : rows(0), cols(0) {}
  Extents2D(size_t rows, size_t cols) : rows(rows), cols(cols) {
    HWY_DASSERT(rows != 0);
    HWY_DASSERT(cols != 0);
  }

  size_t Area() const { return rows * cols; }

  size_t rows;
  size_t cols;
};

// Range2D consists of two Range1D.
struct Range1D {
  Range1D(size_t begin, size_t end) : begin_(begin), end_(end) {
    HWY_DASSERT(begin < end);
  }
  size_t Num() const { return end_ - begin_; }

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

  const size_t begin_;
  const size_t end_;
};

static inline Range1D MakeRange1D(size_t begin, size_t end, size_t max_size) {
  return Range1D(begin, HWY_MIN(begin + max_size, end));
}

// In MatMul, the two axes are used independently, hence we do not define
// Range2D as a top-left and extents.
struct Range2D {
  Range2D(Range1D rows, Range1D cols) : rows(rows), cols(cols) {}
  const Range1D rows;
  const Range1D cols;
};

// Lightweight version of `MatPtr` used for the C argument of `MatMul`, because
// it is always float and does not support compressed T, but does support an
// arbitrary stride >= cols.
template <typename T>
class RowPtr {
 public:
  RowPtr(T* HWY_RESTRICT row0, size_t cols)
      : row0_(row0), cols_(cols), stride_(cols) {}

  T* HWY_RESTRICT Row(size_t r) const { return row0_ + stride_ * r; }
  size_t Cols() const { return cols_; }

  size_t Stride() const { return stride_; }
  void SetStride(size_t stride) {
    HWY_DASSERT(stride >= Cols());
    stride_ = stride;
  }

 private:
  T* HWY_RESTRICT row0_;
  size_t stride_;
  size_t cols_;
};

using RowPtrF = RowPtr<float>;

// Owns dynamically-allocated aligned memory for a batch of row vectors.
// This can be seen as a (batch_size x cols) matrix. Unlike `RowPtr`, this owns
// the memory.
template <typename T>
class RowVectorBatch {
 public:
  // Default ctor for Activations ctor.
  RowVectorBatch() = default;
  // Main ctor, called from Activations::Allocate.
  RowVectorBatch(Extents2D extents) : extents_(extents) {
    mem_ = hwy::AllocateAligned<T>(extents_.rows * extents_.cols);
  }

  // Move-only
  RowVectorBatch(RowVectorBatch&) noexcept = delete;
  RowVectorBatch& operator=(RowVectorBatch&) noexcept = delete;
  RowVectorBatch(RowVectorBatch&&) noexcept = default;
  RowVectorBatch& operator=(RowVectorBatch&&) noexcept = default;

  size_t BatchSize() const { return extents_.rows; }
  size_t Cols() const { return extents_.cols; }
  Extents2D Extents() const { return extents_; }

  // Returns the given row vector of length `Cols()`.
  T* Batch(size_t batch_idx) {
    HWY_DASSERT(batch_idx < BatchSize());
    return mem_.get() + batch_idx * Cols();
  }
  const T* Batch(size_t batch_idx) const {
    HWY_DASSERT(batch_idx < BatchSize());
    return mem_.get() + batch_idx * Cols();
  }

  // For MatMul or other operations that process the entire batch at once.
  // TODO: remove once we only use Mat.
  T* All() { return mem_.get(); }
  const T* Const() const { return mem_.get(); }
  size_t NumBytes() const { return BatchSize() * Cols() * sizeof(T); }

 private:
  hwy::AlignedFreeUniquePtr<T[]> mem_;
  Extents2D extents_;
};

// Used for the A and B arguments of `MatMul`, which are always const.
// Create via MakeConstMat. This differs from `RowPtr` in that it supports the
// `ofs` required for compressed T.
template <typename T>
struct ConstMat {
  ConstMat(const T* ptr, Extents2D extents, size_t ofs = 0)
      : ptr(ptr), extents(extents), ofs(ofs) {
    HWY_DASSERT(ptr != nullptr);
  }
  // TODO: support stride for page alignment.
  size_t Row(size_t r) const {
    if constexpr (HWY_IS_DEBUG_BUILD) {
      if (r >= extents.rows) {
        HWY_ABORT("ConstMat::Row %zu out of bounds %zu", r, extents.rows);
      }
    }
    return ofs + extents.cols * r;
  }

  const Extents2D& Extents() const { return extents; }

  // Shrinks the row-extent of this matrix view, i.e. reduces the view to a
  // subrange of the original rows starting at row 0.
  void ShrinkRows(size_t rows) {
    HWY_ASSERT(rows <= extents.rows);
    extents.rows = rows;
  }

  const T* HWY_RESTRICT ptr;
  Extents2D extents;

  // `scale` allows expanding the smaller range of `SfpStream` to the original
  // values. MatFromWeights sets this from `MatPtr`.
  float scale = 1.0f;

  // Offset to add to `ptr`; separate because T=NuqStream does not support
  // pointer arithmetic.
  size_t ofs;
};

// For deducing T.
template <typename T>
ConstMat<T> MakeConstMat(T* HWY_RESTRICT ptr, Extents2D extents,
                         size_t ofs = 0) {
  return ConstMat<T>(ptr, extents, ofs);
}

// For A argument to MatMul (activations).
template <typename T>
ConstMat<T> ConstMatFromBatch(size_t batch_size,
                              const RowVectorBatch<T>& row_vectors) {
  HWY_DASSERT(batch_size <= row_vectors.BatchSize());
  return MakeConstMat(const_cast<T*>(row_vectors.Const()),
                      Extents2D(batch_size, row_vectors.Cols()));
}

// For C argument to MatMul.
template <typename T>
RowPtr<T> RowPtrFromBatch(RowVectorBatch<T>& row_vectors) {
  return RowPtr<T>(row_vectors.All(), row_vectors.Cols());
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_BASICS_H_
