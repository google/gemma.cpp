// Copyright 2023 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Tensor metadata and in-memory representation.
#ifndef THIRD_PARTY_GEMMA_CPP_UTIL_MAT_H_
#define THIRD_PARTY_GEMMA_CPP_UTIL_MAT_H_

#include <stddef.h>
#include <stdint.h>

#include <random>
#include <string>
#include <vector>

// IWYU pragma: begin_exports
#include "compression/shared.h"  // Type
#include "gemma/tensor_info.h"
#include "io/fields.h"
#include "util/allocator.h"  // AlignedPtr2
#include "util/basics.h"  // Extents2D
// IWYU pragma: end_exports
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

// Type-erased, non-owning pointer and metadata for rank-1 or 2 tensors (vector
// or matrix). Base class of the non-type-erased `MatPtrT`. Use this class
// to store hetereogeneous tensor references in a vector.
//
// Copyable, (de)serializable via `fields.h` for `model_store.h`.
class MatPtr : public IFields {
 public:
  MatPtr() = default;
  // `name`: see `SetName`. Note that `stride` is initially `cols` and only
  // differs after deserializing, or calling `SetPtr`.
  MatPtr(const char* name, Type type, Extents2D extents)
      : private_rows_(static_cast<uint32_t>(extents.rows)),
        cols_(static_cast<uint32_t>(extents.cols)) {
    SetName(name);
    SetType(type);
    SetPtr(nullptr, cols_);
  }

  // Copying allowed because the metadata is small.
  MatPtr(const MatPtr& other) = default;
  MatPtr& operator=(const MatPtr& other) = default;

  virtual ~MatPtr() = default;

  // Only for use by ctor, `AllocateFor` and 'loading' memory-mapped tensors.
  void SetPtr(void* ptr, size_t stride) {
    HWY_ASSERT(stride >= Cols());
    ptr_ = ptr;
    stride_ = static_cast<uint32_t>(stride);

    // NUQ streams must not be padded because that would change the position of
    // the group tables.
    if (type_ == Type::kNUQ) HWY_ASSERT(IsPacked());
  }

  bool HasPtr() const { return ptr_ != nullptr; }

  // A single row counts as packed because there is no padding between rows.
  bool IsPacked() const { return (stride_ == cols_) || (Rows() == 1); }

  const void* Packed() const {
    HWY_DASSERT_M(IsPacked(), name_.c_str());
    return ptr_;
  }
  void* Packed() {
    HWY_DASSERT_M(IsPacked(), name_.c_str());
    return ptr_;
  }

  // Returns size in bytes for purposes of copying/initializing or I/O. Must
  // only be called if `IsPacked`.
  size_t PackedBytes() const {
    HWY_DASSERT_M(IsPacked(), name_.c_str());
    // num_elements_ already includes the NUQ tables.
    return num_elements_ * element_bytes_;
  }

  // Works for any kind of padding.
  template <typename T>
  T* MutableRowT(size_t row) const {
    HWY_DASSERT(row < Rows());
    return HWY_RCAST_ALIGNED(T*, ptr_) + row * stride_;
  }
  template <typename T>
  T* RowT(size_t row) {
    HWY_DASSERT(row < Rows());
    return HWY_RCAST_ALIGNED(T*, ptr_) + row * stride_;
  }
  template <typename T>
  const T* RowT(size_t row) const {
    HWY_DASSERT(row < Rows());
    return HWY_RCAST_ALIGNED(const T*, ptr_) + row * stride_;
  }

  Type GetType() const { return type_; }
  void SetType(Type type) {
    type_ = type;
    element_bytes_ = static_cast<uint32_t>(hwy::DivCeil(TypeBits(type), 8));
    num_elements_ = static_cast<uint32_t>(ComputeNumElements(type, Extents()));
    HWY_DASSERT(0 != element_bytes_ && element_bytes_ <= 16);
  }

  size_t Rows() const {
    return override_rows_ == 0 ? private_rows_ : override_rows_;
  }
  size_t Cols() const { return cols_; }
  Extents2D Extents() const { return Extents2D(Rows(), cols_); }
  bool IsEmpty() const { return Rows() == 0 || cols_ == 0; }
  bool SameShape(const MatPtr& other) const {
    return Rows() == other.Rows() && cols_ == other.cols_;
  }
  // Future calls to `Rows()` during this class' lifetime (not serialized)
  // will return this value. Used to set the actual number of rows for
  // activations preallocated according to the batch size.
  void OverrideRows(size_t rows) {
    HWY_ASSERT(rows <= private_rows_);
    override_rows_ = static_cast<uint32_t>(rows);
  }

  // Offset by which to advance pointers to the next row.
  size_t Stride() const { return stride_; }

  // For use by `BlobStore`, `CopyMat` and `ZeroInit`.
  size_t ElementBytes() const { return element_bytes_; }

  // Decoded elements should be multiplied by this to restore their original
  // range. This is required because `SfpStream` can only encode a limited range
  // of magnitudes.
  float Scale() const { return scale_; }
  void SetScale(float scale) { scale_ = scale; }

  // A terse identifier unique across all tensors of the model.
  const char* Name() const override { return name_.c_str(); }
  // `MakeKey` in `blob_store.cc` requires that this be <= 16 bytes, including
  // the `LayerSuffix` for per-layer tensors.
  void SetName(const char* name) {
    name_ = name;
    HWY_ASSERT_M(name_.size() <= sizeof(hwy::uint128_t), name);
  }

  void VisitFields(IFieldsVisitor& visitor) override {
    // Order determines the order of serialization and must not change.
    visitor(name_);
    visitor(type_);
    visitor(element_bytes_);
    visitor(num_elements_);
    visitor(private_rows_);
    visitor(cols_);
    visitor(scale_);
    visitor(stride_);
  }

 protected:
  // For initializing `num_elements_`: "elements" are how many objects we
  // actually store in order to represent rows * cols values. For NUQ, this is
  // greater because it includes additional per-group tables. This is the only
  // place where we compute this fixup. Note that elements are independent of
  // padding, which is anyway not supported for NUQ because `compress-inl.h`
  // assumes a contiguous stream for its group indexing.
  static size_t ComputeNumElements(Type type, Extents2D extents) {
    size_t num_elements = extents.Area();
    if (type == Type::kNUQ) {
      // `CompressedArrayElements` is a wrapper function that has the same
      // effect, but that requires a template argument, not `type`.
      num_elements = NuqStream::PackedEnd(num_elements);
    }
    return num_elements;
  }

  std::string name_;  // See `SetName`.
  Type type_;

  // Most members are u32 because that is the preferred type of fields.h.

  // Bytes per element. This is fully determined by `type_`, but stored here
  // for convenience and backward compatibility.
  uint32_t element_bytes_ = 0;
  // Number of elements to store (including NUQ tables but not padding).
  // This a function of `type_` and `Extents()` and stored for compatibility.
  uint32_t num_elements_ = 0;
  uint32_t private_rows_ = 0;  // Only access via Rows()! See OverrideRows().
  uint32_t cols_ = 0;

  uint32_t override_rows_ = 0;  // not serialized

  // Non-owning pointer, must not be freed. The underlying memory must outlive
  // this object.
  void* ptr_ = nullptr;  // not serialized

  // Offset by which to advance pointers to the next row, >= `cols_`.
  uint32_t stride_;

  float scale_ = 1.0f;  // multiplier for each value, for MatMul.
};

// Non-type erased version of `MatPtr`. Although `MatPtr` also provides
// type-aware accessors (`RowT`), this class is more convenient when accessing
// elements, and ensures the template argument and `Type` are consistent.
template <typename MatT>
class MatPtrT : public MatPtr {
 public:
  using T = MatT;

  // Called by `MatStorageT`.
  MatPtrT(const char* name, Extents2D extents)
      : MatPtr(name, TypeEnum<MatT>(), extents) {}
  // Retrieves shape by name via `TensorInfo` from `TensorInfoRegistry`. This is
  // not a factory function because `weights.h` initializes members of type
  // `MatPtrT<T>`, and `T` cannot be inferred at compile time from arguments.
  MatPtrT(const std::string& name, const TensorInfoRegistry& info)
      : MatPtrT<MatT>(name.c_str(), ExtentsFromInfo(info.Find(name))) {}

  // Copying allowed because the metadata is small.
  MatPtrT(const MatPtr& other) : MatPtr(other) {}
  MatPtrT& operator=(const MatPtr& other) {
    MatPtr::operator=(other);
    return *this;
  }
  MatPtrT(const MatPtrT& other) = default;
  MatPtrT& operator=(const MatPtrT& other) = default;

  // Returns the entire tensor for use by `backprop/*`. Verifies layout is
  // `kPacked`. Preferably call `Row` instead, which works for either layout.
  MatT* Packed() {
    HWY_DASSERT_M(IsPacked(), name_.c_str());
    return HWY_RCAST_ALIGNED(MatT*, ptr_);
  }
  const MatT* Packed() const {
    HWY_DASSERT_M(IsPacked(), name_.c_str());
    return HWY_RCAST_ALIGNED(const MatT*, ptr_);
  }
  // As `Packed()`, plus checks the scale is 1.0 because callers will ignore it.
  // This is typically used for `MatMul` bias vectors and norm weights.
  const MatT* PackedScale1() const {
    HWY_DASSERT(Scale() == 1.0f);
    return Packed();
  }

  const MatT* Row(size_t row) const { return this->RowT<MatT>(row); }
  MatT* Row(size_t row) { return this->RowT<MatT>(row); }

  // For `compress-inl.h` functions, which assume contiguous streams and thus
  // require packed layout.
  PackedSpan<const MatT> Span() const {
    HWY_ASSERT(IsPacked());
    return MakeConstSpan(Row(0), num_elements_);
  }
  PackedSpan<MatT> Span() {
    HWY_ASSERT(IsPacked());
    return MakeSpan(Row(0), num_elements_);
  }
};

// Calls `func` with a dynamic_cast of `MatPtr` to `MatPtrT<T>`, plus the
// optional `args`. This supports all types used as weights, which excludes
// `kC64` and `kF64` (used only in `backprop/`).
template <class Func, typename... Args>
decltype(auto) CallUpcasted(const MatPtr* base, const Func& func,
                            Args&&... args) {
  if (base->GetType() == Type::kF32) {
    return func(dynamic_cast<const MatPtrT<float>*>(base),
                std::forward<Args>(args)...);
  } else if (base->GetType() == Type::kBF16) {
    return func(dynamic_cast<const MatPtrT<BF16>*>(base),
                std::forward<Args>(args)...);
  } else if (base->GetType() == Type::kSFP) {
    return func(dynamic_cast<const MatPtrT<SfpStream>*>(base),
                std::forward<Args>(args)...);
  } else if (base->GetType() == Type::kNUQ) {
    return func(dynamic_cast<const MatPtrT<NuqStream>*>(base),
                std::forward<Args>(args)...);
  } else {
    HWY_ABORT("Unhandled type %s.", TypeName(base->GetType()));
  }
}

// Calls `func(base1, base2, args...)`.
template <class Func, typename... Args>
decltype(auto) CallUpcastedSame(const MatPtr* base1, const MatPtr* base2,
                                const Func& func, Args&&... args) {
  HWY_ASSERT(base1->GetType() == base2->GetType());
  if (base1->GetType() == Type::kF32) {
    return func(dynamic_cast<const MatPtrT<float>*>(base1),
                dynamic_cast<const MatPtrT<float>*>(base2),
                std::forward<Args>(args)...);
  } else if (base1->GetType() == Type::kBF16) {
    return func(dynamic_cast<const MatPtrT<BF16>*>(base1),
                dynamic_cast<const MatPtrT<BF16>*>(base2),
                std::forward<Args>(args)...);
  } else if (base1->GetType() == Type::kSFP) {
    return func(dynamic_cast<const MatPtrT<SfpStream>*>(base1),
                dynamic_cast<const MatPtrT<SfpStream>*>(base2),
                std::forward<Args>(args)...);
  } else if (base1->GetType() == Type::kNUQ) {
    return func(dynamic_cast<const MatPtrT<NuqStream>*>(base1),
                dynamic_cast<const MatPtrT<NuqStream>*>(base2),
                std::forward<Args>(args)...);
  } else {
    HWY_ABORT("Unhandled type %s.", TypeName(base1->GetType()));
  }
}

// Like CallUpcasted, but only for activation types: kBF16 and kF32.
template <class Func, typename... Args>
decltype(auto) CallUpcastedActivation(const MatPtr* base, const Func& func,
                                      Args&&... args) {
  HWY_ASSERT(base != nullptr);
  if (base->GetType() == Type::kF32) {
    return func(dynamic_cast<const MatPtrT<float>*>(base),
                std::forward<Args>(args)...);
  } else if (base->GetType() == Type::kBF16) {
    return func(dynamic_cast<const MatPtrT<BF16>*>(base),
                std::forward<Args>(args)...);
  } else {
    HWY_ABORT("Unhandled type %s.", TypeName(base->GetType()));
  }
}

void CopyMat(const MatPtr& from, MatPtr& to);
void ZeroInit(MatPtr& mat);
// F32/F64 only.
void RandInit(MatPtr& mat, float stddev, std::mt19937& gen);

// Sufficient value of `stride` to enable the "cyclic offsets" optimization. If
// `Allocator::ShouldBind()`, `Allocator::QuantumBytes()` is typically 4KiB.
// To avoid remote accesses, we would thus pad each row to that, which results
// in 4K aliasing and/or cache conflict misses. `RowPtr` is able to prevent that
// by pulling rows forward by a cyclic offset, which is still a multiple of the
// cache line size. This requires an additional `Allocator::QuantumBytes()` of
// padding after also rounding up to that, which considerably increases size for
// tall and skinny tensors.
static inline size_t StrideForCyclicOffsets(size_t cols, size_t quantum) {
  return hwy::RoundUpTo(cols, quantum) + quantum;
}
// Constexpr version (upper bound) for allocating storage in MatMul.
template <typename T>
constexpr size_t MaxStrideForCyclicOffsets(size_t cols) {
  constexpr size_t quantum = Allocator::MaxQuantum<T>();
  return hwy::RoundUpTo(cols, quantum) + quantum;
}

// Our tensors are always row-major. This enum indicates how much (if any)
// padding comes after each row.
enum class MatPadding {
  // None, stride == cols. `compress-inl.h` requires this layout because its
  // interface assumes a continuous 1D array, without awareness of rows. Note
  // that tensors which were written via `compress-inl.h` (i.e. most in
  // `BlobStore`) are not padded, which also extends to memory-mapped tensors.
  // However, `BlobStore` is able to insert padding via row-wise I/O when
  // reading from disk via `Mode::kRead`.
  //
  // `backprop/*` also requires this layout because it indexes directly into
  // the storage instead of calling `Row()`.
  kPacked,
  // Enough to round up to an odd number of cache lines, which can reduce
  // cache conflict misses or 4K aliasing.
  kOdd,
  // Enough to enable the "cyclic offsets" optimization for `MatMul`.
  kCyclic,
};

// Type-erased, allows storing `AlignedPtr2<T[]>` for various T in the same
// vector.
class MatOwner {
 public:
  MatOwner() = default;
  // Allow move for `MatStorageT`.
  MatOwner(MatOwner&&) = default;
  MatOwner& operator=(MatOwner&&) = default;

  // Allocates the type/extents indicated by `mat` and sets its pointer.
  void AllocateFor(MatPtr& mat, MatPadding padding);

 private:
  AlignedPtr2<uint8_t[]> storage_;
};

// Multiple `MatOwner`, with support for parallel allocation.
class MatOwners {
 public:
  // Ignores `padding` for NUQ tensors, which are always packed.
  void AllocateFor(MatPtr& mat, MatPadding padding) {
    if (mat.GetType() == Type::kNUQ) padding = MatPadding::kPacked;
    owners_.push_back(MatOwner());
    owners_.back().AllocateFor(mat, padding);
  }

  // Allocates multiple in parallel. Ignores `padding` for NUQ tensors,
  // which are always packed.
  void AllocateFor(const std::vector<MatPtr*>& mats, MatPadding padding,
                   hwy::ThreadPool& pool);

 private:
  std::vector<MatOwner> owners_;
};

// `MatStorageT` IS-A `MatPtrT` and HAS-A `MatOwner`. Used by `backprop/` and
// tests to allocate and access tensors of a known type. By contrast, the
// heterogeneous model weights are owned by vectors of `MatOwner`.
template <typename MatT>
class MatStorageT : public MatPtrT<MatT> {
 public:
  MatStorageT(const char* name, Extents2D extents, MatPadding padding)
      : MatPtrT<MatT>(name, extents) {
    if (extents.Area() != 0) owner_.AllocateFor(*this, padding);
  }
  // Shorthand for 1D tensors: packing does not help, hence `kPacked`.
  MatStorageT(const char* name, size_t cols)
      : MatStorageT(name, Extents2D(1, cols), MatPadding::kPacked) {}
  ~MatStorageT() = default;

  // Allow move for backprop/activations.
  MatStorageT(MatStorageT&&) = default;
  MatStorageT& operator=(MatStorageT&&) = default;

 private:
  MatOwner owner_;
};

// Helper factory function for use by `backprop/` to avoid specifying the
// `MatPadding` argument everywhere.
template <typename T>
MatStorageT<T> MakePacked(const char* name, size_t rows, size_t cols) {
  return MatStorageT<T>(name, Extents2D(rows, cols), MatPadding::kPacked);
}

// Lightweight version of `MatPtr` used by matmul-inl.h for padded tensors with
// seekable (non-NUQ) T. This has less metadata, but support for cyclic offsets.
#pragma pack(push, 1)  // power of two size
template <typename T>
class RowPtr {
 public:
  RowPtr(const Allocator& allocator, T* HWY_RESTRICT row0, size_t cols,
         size_t stride)
      : row0_(row0),
        stride_(stride),
        // TODO: disabled because otherwise we see non-deterministic results.
        row_mask_(0),
        // static_cast<uint32_t>(allocator.QuantumStepMask() & 0xFFFFFFFFu)),
        cols_(static_cast<uint32_t>(cols)),
        step_bytes_(static_cast<uint32_t>(allocator.StepBytes())),
        quantum_bytes_(allocator.QuantumBytes()) {
    HWY_DASSERT(stride >= cols);
    HWY_DASSERT(row_mask_ != ~uint32_t{0});
    if (stride < StrideForCyclicOffsets(cols, quantum_bytes_ / sizeof(T))) {
      row_mask_ = 0;
      if constexpr (HWY_IS_DEBUG_BUILD) {
        static bool once;
        if (stride != cols && !once) {
          once = true;
          HWY_WARN(
              "Check why RowPtr stride=%zu < StrideForCyclicOffsets(cols=%zu), "
              "T=%zu; this forces us to disable cyclic offsets.",
              stride, cols, sizeof(T));
        }
      }
    }
  }

  RowPtr(const Allocator& allocator, T* HWY_RESTRICT row0, size_t cols)
      : RowPtr(allocator, row0, cols, cols) {}

  T* HWY_RESTRICT Row(size_t r) const {
    // How much of the previous row's padding to consume.
    const size_t pad_bytes = (r & row_mask_) * step_bytes_;
    HWY_DASSERT(pad_bytes < static_cast<size_t>(quantum_bytes_));
    return row0_ + stride_ * r - pad_bytes;
  }
  size_t Cols() const { return static_cast<size_t>(cols_); }

  size_t Stride() const { return stride_; }
  void SetStride(size_t stride) {
    HWY_DASSERT(stride >= Cols());
    stride_ = stride;
    // The caller might not have padded enough, so disable the padding in Row().
    // Rows will now be exactly `stride` elements apart. This is used when
    // writing to the KV cache via MatMul.
    row_mask_ = 0;
  }

  // Returns 2D subrange whose top-left is `r, c` and width is `cols`.
  RowPtr<T> View(size_t r, size_t c, size_t cols) const {
    HWY_DASSERT(c < Cols());
    HWY_DASSERT(cols <= Cols() - c);
    return RowPtr<T>(Row(r) + c, cols, stride_, row_mask_, step_bytes_,
                     quantum_bytes_);
  }

 private:
  // For `View()`.
  RowPtr(T* new_row0, size_t new_cols, size_t stride, uint32_t row_mask,
         uint32_t step_bytes, uint32_t quantum_bytes)
      : row0_(new_row0),
        stride_(stride),
        row_mask_(row_mask),
        cols_(new_cols),
        step_bytes_(step_bytes),
        quantum_bytes_(quantum_bytes) {}

  T* HWY_RESTRICT row0_;
  size_t stride_;
  uint32_t row_mask_;
  uint32_t cols_;
  uint32_t step_bytes_;
  uint32_t quantum_bytes_;
};
#pragma pack(pop)

using RowPtrBF = RowPtr<BF16>;
using RowPtrF = RowPtr<float>;
using RowPtrD = RowPtr<double>;

// TODO: remove allocator arg once kCyclic is removed.
template <typename T>
RowPtr<T> RowPtrFromMat(const Allocator& allocator,
                        const MatPtrT<T>& row_vectors) {
  // RowPtr is non-const for MatMul C, but is also used for A which is const.
  // Callers are responsible for checking their usage of RowPtr.
  return RowPtr<T>(allocator, const_cast<T*>(row_vectors.Row(0)),
                   row_vectors.Cols(), row_vectors.Stride());
}

}  // namespace gcpp
#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_MAT_H_
