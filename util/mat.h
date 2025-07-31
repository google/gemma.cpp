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

#include <string>

// IWYU pragma: begin_exports
#include "compression/types.h"  // Type
#include "gemma/tensor_info.h"
#include "io/fields.h"
#include "util/allocator.h"  // AlignedPtr
#include "util/basics.h"  // Extents2D
// IWYU pragma: end_exports
#include "hwy/base.h"

namespace gcpp {

// Type-safe wrapper over type-erased uint8_t row pointers from MatPtr. Used
// for C (KV output), in future also for A or even B.
template <typename T>
class RowPtrs {
 public:
  RowPtrs(uint8_t** row_ptrs) : row_ptrs_(row_ptrs) {}

  T* HWY_RESTRICT operator[](size_t row_idx) const {
    return HWY_RCAST_ALIGNED(T*, row_ptrs_[row_idx]);
  }

 private:
  uint8_t** row_ptrs_;
};

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

    // If row pointers were already attached, `SetPtr` would invalidate them.
    HWY_DASSERT_M(row_ptrs_ == nullptr, "Do not call after AttachRowPtrs.");

    // NUQ streams must not be padded because that would change the position of
    // the group tables.
    if (type_ == Type::kNUQ) {
      HWY_ASSERT_M(GEMMA_ENABLE_NUQ, "Set GEMMA_ENABLE_NUQ=1.");
      HWY_ASSERT(IsPacked());
    }
  }

  bool HasPtr() const { return ptr_ != nullptr; }

  // Caller has initialized Rows() pointers in row_ptrs[]. Note that this only
  // changes `GetRowPtrs`, not `Row()`, because that would require branching
  // and only a few call sites, in particular MatMul, use row pointers.
  void AttachRowPtrs(uint8_t** row_ptrs) {
    row_ptrs_ = row_ptrs;
    for (size_t r = 0; r < Rows(); ++r) {
      HWY_DASSERT(row_ptrs[r] != nullptr);
    }
  }

  // Called by Activations to allocate once, rather than have to fill row
  // pointers in each call to MatMul.
  void AllocateAndAttachRowPtrs(
      std::vector<hwy::AlignedFreeUniquePtr<uint8_t*[]>>& row_ptrs) {
    if (!HasPtr()) return;
    row_ptrs.push_back(hwy::AllocateAligned<uint8_t*>(Rows()));
    uint8_t** ptrs = row_ptrs.back().get();
    for (size_t r = 0; r < Rows(); ++r) {
      ptrs[r] = RowBytes(r);
    }
    AttachRowPtrs(ptrs);
  };

  // If non-null, this array should be used instead of `Row()`.
  uint8_t** GetRowPtrs() const { return row_ptrs_; }

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

  // Works for any kind of padding and element type.
  uint8_t* RowBytes(size_t row) {
    HWY_DASSERT(row < Rows());
    return static_cast<uint8_t*>(ptr_) + row * (stride_ * element_bytes_);
  }
  const uint8_t* RowBytes(size_t row) const {
    HWY_DASSERT(row < Rows());
    return static_cast<const uint8_t*>(ptr_) + row * (stride_ * element_bytes_);
  }

  Type GetType() const { return type_; }
  void SetType(Type type) {
    type_ = type;
    if (type == Type::kUnknown) {
      // Temporary invalid state. Happens during weights.h construction, before
      // the ForEachTensor that loads them and sets the type.
      element_bytes_ = 0;
      num_elements_ = 0;
      return;
    }
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

  // Points to an array of pointers, one per row, or nullptr if `AttachRowPtrs`
  // was not called. Only used for MatMul output tensors, hence we
  // minimize the cost for other tensors by only holding a non-owning pointer.
  uint8_t** row_ptrs_ = nullptr;  // not serialized

  // Offset by which to advance pointers to the next row, >= `cols_`.
  uint32_t stride_;

  float scale_ = 1.0f;  // multiplier for each value, for MatMul.
};

// Non-type erased version of `MatPtr`: provides type-safe `Row()` and ensures
// the template argument and `Type` are consistent.
template <typename MatT>
class MatPtrT : public MatPtr {
 public:
  using T = MatT;

  // Called by `MatStorageT`.
  MatPtrT(const char* name, Extents2D extents)
      : MatPtr(name, TypeEnum<MatT>(), extents) {}

  // Copying allowed because the metadata is small.
  MatPtrT(const MatPtr& other) : MatPtr(other) {
    // Happens in weights.h when constructing via MatFinder, which does not
    // know the type. Setting the type here avoids having to keep the
    // initializer list and member type in sync.
    if (GetType() == Type::kUnknown) {
      SetType(TypeEnum<MatT>());
    } else {
      HWY_ASSERT(other.GetType() == TypeEnum<MatT>());
    }
  }
  MatPtrT& operator=(const MatPtr& other) {
    MatPtr::operator=(other);
    return *this;
  }
  MatPtrT(const MatPtrT& other) = default;
  MatPtrT& operator=(const MatPtrT& other) = default;

  // Returns the entire tensor after checking the scale is 1.0 because callers
  // will ignore it. Used for `MatMul` bias vectors and norm weights.
  const MatT* PackedScale1() const {
    HWY_DASSERT(Scale() == 1.0f);
    return HWY_RCAST_ALIGNED(const MatT*, ptr_);
  }

  MatT* Row(size_t row) { return HWY_RCAST_ALIGNED(T*, RowBytes(row)); }
  const MatT* Row(size_t row) const {
    return HWY_RCAST_ALIGNED(const T*, RowBytes(row));
  }

  PackedSpan<const MatT> PaddedSpan() const {
    return MakeConstSpan(HWY_RCAST_ALIGNED(MatT*, ptr_), Rows() * Stride());
  }

  // For `compress-inl.h` functions, which assume contiguous streams and thus
  // require packed layout.
  PackedSpan<MatT> Span() {
    HWY_ASSERT(IsPacked());
    return MakeSpan(HWY_RCAST_ALIGNED(MatT*, ptr_), num_elements_);
  }
  PackedSpan<const MatT> Span() const {
    HWY_ASSERT(IsPacked());
    return MakeConstSpan(HWY_RCAST_ALIGNED(MatT*, ptr_), num_elements_);
  }
};

template <typename T>
RowPtrs<T> GetOrSetTempRowPtrs(
    const MatPtrT<T>& mat,
    const hwy::AlignedFreeUniquePtr<uint8_t*[]>& storage) {
  if (HWY_LIKELY(mat.GetRowPtrs())) return RowPtrs<T>(mat.GetRowPtrs());

  if constexpr (HWY_IS_DEBUG_BUILD) {
    fprintf(stderr,
            "MatMul perf warning: setting row pointers because "
            "%s.AttachRowPtrs() was not called.\n",
            mat.Name());
  }
  HWY_DASSERT(mat.HasPtr());
  for (size_t r = 0; r < mat.Rows(); ++r) {
    storage[r] = reinterpret_cast<uint8_t*>(const_cast<T*>(mat.Row(r)));
  }
  return RowPtrs<T>(storage.get());
}

// Calls `func` with `MatPtrT<T>*` plus the optional `args`. This supports all
// types used as weights.
template <class Func, typename... Args>
decltype(auto) CallUpcasted(const MatPtr* base, const Func& func,
                            Args&&... args) {
#if GEMMA_ENABLE_NUQ
  if (base->GetType() == Type::kNUQ) {
    const MatPtrT<NuqStream> mat(*base);
    return func(&mat, std::forward<Args>(args)...);
  }
#endif  // GEMMA_ENABLE_NUQ

  if (base->GetType() == Type::kF32) {
    const MatPtrT<float> mat(*base);
    return func(&mat, std::forward<Args>(args)...);
  } else if (base->GetType() == Type::kBF16) {
    const MatPtrT<BF16> mat(*base);
    return func(&mat, std::forward<Args>(args)...);
  } else if (base->GetType() == Type::kSFP) {
    const MatPtrT<SfpStream> mat(*base);
    return func(&mat, std::forward<Args>(args)...);
  } else {
    HWY_ABORT("Unhandled type %s.", TypeName(base->GetType()));
  }
}

// Calls `func(base1, base2, args...)`.
template <class Func, typename... Args>
decltype(auto) CallUpcastedSame(const MatPtr* base1, const MatPtr* base2,
                                const Func& func, Args&&... args) {
  HWY_DASSERT(base1->GetType() == base2->GetType());

#if GEMMA_ENABLE_NUQ
  if (base1->GetType() == Type::kNUQ) {
    const MatPtrT<NuqStream> mat1(*base1);
    const MatPtrT<NuqStream> mat2(*base2);
    return func(&mat1, &mat2, std::forward<Args>(args)...);
  }
#endif  // GEMMA_ENABLE_NUQ

  if (base1->GetType() == Type::kF32) {
    const MatPtrT<float> mat1(*base1);
    const MatPtrT<float> mat2(*base2);
    return func(&mat1, &mat2, std::forward<Args>(args)...);
  } else if (base1->GetType() == Type::kBF16) {
    const MatPtrT<BF16> mat1(*base1);
    const MatPtrT<BF16> mat2(*base2);
    return func(&mat1, &mat2, std::forward<Args>(args)...);
  } else if (base1->GetType() == Type::kSFP) {
    const MatPtrT<SfpStream> mat1(*base1);
    const MatPtrT<SfpStream> mat2(*base2);
    return func(&mat1, &mat2, std::forward<Args>(args)...);
  } else {
    HWY_ABORT("Unhandled type %s.", TypeName(base1->GetType()));
  }
}

// Like CallUpcasted, but only for activation types: kBF16 and kF32.
template <class Func, typename... Args>
decltype(auto) CallUpcastedActivation(const MatPtr* base, const Func& func,
                                      Args&&... args) {
  if (base->GetType() == Type::kF32) {
    const MatPtrT<float> mat(*base);
    return func(&mat, std::forward<Args>(args)...);
  } else if (base->GetType() == Type::kBF16) {
    const MatPtrT<BF16> mat(*base);
    return func(&mat, std::forward<Args>(args)...);
  } else {
    HWY_ABORT("Unhandled type %s.", TypeName(base->GetType()));
  }
}

void CopyMat(const MatPtr& from, MatPtr& to);
void ZeroInit(MatPtr& mat);

// Our tensors are always row-major. This enum indicates how much (if any)
// padding comes after each row.
enum class MatPadding {
  // None, stride == cols. `compress-inl.h` requires this layout because its
  // interface assumes a continuous 1D array, without awareness of rows. Note
  // that tensors which were written via `compress-inl.h` (i.e. most in
  // `BlobStore`) are not padded, which also extends to memory-mapped tensors.
  // However, `BlobStore` is able to insert padding via row-wise I/O when
  // reading from disk via `Mode::kRead`.
  kPacked,
  // Enough to round up to an odd number of cache lines, which can reduce
  // cache conflict misses or 4K aliasing.
  kOdd,
};

// The stride (offset in elements between rows) that `MatOwner/MatStorageT`
// will use.
size_t Stride(MatPadding padding, size_t cols, size_t element_bytes,
              size_t line_bytes);

// Type-erased, allows storing `AlignedPtr<T[]>` for various T in the same
// vector.
class MatOwner {
 public:
  MatOwner() = default;
  // Allow move for `MatStorageT`.
  MatOwner(MatOwner&&) = default;
  MatOwner& operator=(MatOwner&&) = default;

  // Allocates the type/extents indicated by `mat` and sets its pointer.
  // Ignores `padding` for NUQ tensors, which are always packed.
  // Thread-compatible, weights are allocated in parallel.
  void AllocateFor(MatPtr& mat, const Allocator& allocator, MatPadding padding);

 private:
  AlignedPtr<uint8_t[]> storage_;
};

// `MatStorageT` IS-A `MatPtrT` and HAS-A `MatOwner`. Used by tests to allocate
// and access tensors of a known type. By contrast, the heterogeneous model
// weights are owned by vectors of `MatOwner`.
template <typename MatT>
class MatStorageT : public MatPtrT<MatT> {
 public:
  MatStorageT(const char* name, Extents2D extents, const Allocator& allocator,
              MatPadding padding)
      : MatPtrT<MatT>(name, extents) {
    if (extents.Area() != 0) owner_.AllocateFor(*this, allocator, padding);
  }
  // Shorthand for 1D tensors: packing does not help, hence `kPacked`.
  MatStorageT(const char* name, size_t cols, const Allocator& allocator)
      : MatStorageT(name, Extents2D(1, cols), allocator, MatPadding::kPacked) {}
  ~MatStorageT() = default;

  // Allow move for KVCache.
  MatStorageT(MatStorageT&&) = default;
  MatStorageT& operator=(MatStorageT&&) = default;

 private:
  MatOwner owner_;
};

// Helper for initializing members which are `MatStorageT<T>`: avoids having to
// specify Extents2D and MatPadding at each call site.
class MatFactory {
 public:
  // The constructor captures all the necessary arguments.
  MatFactory(const char* name, size_t rows, size_t cols,
             const Allocator& allocator, MatPadding padding = MatPadding::kOdd)
      : name_(name),
        extents_(rows, cols),
        allocator_(allocator),
        padding_(padding) {}

  // Templated conversion so we do not have to specify the type in the
  // member initializer.
  template <typename T>
  operator MatStorageT<T>() const {
    return MatStorageT<T>(name_.c_str(), extents_, allocator_, padding_);
  }

 private:
  const std::string name_;
  Extents2D extents_;
  const Allocator& allocator_;
  MatPadding padding_;
};

}  // namespace gcpp
#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_MAT_H_
