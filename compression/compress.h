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

// Target-independent definitions.
#ifndef THIRD_PARTY_GEMMA_CPP_COMPRESSION_COMPRESS_H_
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_COMPRESS_H_

#define COMPRESS_STATS 0

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// IWYU pragma: begin_exports
#include "compression/blob_store.h"
#include "compression/io.h"
#include "compression/shared.h"
// IWYU pragma: end_exports
#include "util/allocator.h"
#if COMPRESS_STATS
#include "compression/distortion.h"
#include "hwy/stats.h"
#endif

namespace gcpp {

// Base class for rank-1 or 2 tensors (vector or matrix).
// Supports both dynamic and compile-time sizing.
// Holds metadata and a non-owning pointer to the data, owned by the derived
// MatStorageT class.
// This class also provides easy conversion from/to a table of contents for a
// BlobStore file, and a templated (compile-time) accessor for a 2-d array of
// fixed inner dimension and type.
// It is designed to be put in a vector, and has default copy and operator=, so
// it is easy to read/write a blob_store file.
class MatPtr {
 public:
  // Full constructor for dynamic sizing.
  MatPtr(const std::string& name, Type type, size_t element_size, size_t rows,
         size_t cols)
      : name_(name),
        type_(type),
        element_size_(element_size),
        num_elements_(rows * cols),
        rows_(rows),
        cols_(cols),
        ptr_(nullptr) {}
  // Default is to leave all fields default-initialized.
  MatPtr() = default;
  virtual ~MatPtr();

  // Number of hwy::uint128_t in a TOC entry.
  // Note that the old-style BlobStore files only have a list of keys and size.
  // The new-style BlobStore files have an entry called "toc" that contains a
  // vector of 4-tuples of
  // (name, type, (num_elements, element_size), (rows, cols)).
  // The listed blobs can be read directly into MatPtr from the BlobStore
  // file, without needing any external knowledge of the number of elements,
  // element size or type of the data.
  static constexpr size_t kNumU128InTocEntry = 4;

  // Construct from a TOC entry.
  MatPtr(const hwy::uint128_t& key0, const hwy::uint128_t& key1,
         const hwy::uint128_t& key2, const hwy::uint128_t& key3)
      : name_(StringFromKey(key0)),
        type_(static_cast<Type>(key1.lo)),
        element_size_(key2.hi),
        num_elements_(key2.lo),
        rows_(key3.lo),
        cols_(key3.hi) {}

  // Adds the contents entry to the table of contents.
  void AddToToc(std::vector<hwy::uint128_t>& toc) const {
    toc.push_back(MakeKey(name_.c_str()));
    toc.push_back({static_cast<uint64_t>(type_), 0});
    toc.push_back({num_elements_, element_size_});
    toc.push_back({rows_, cols_});
  }

  // Compatibility interface for CompressedArray.
  // TODO: remove.
  template <typename T>
  T* data() {
    return HWY_RCAST_ALIGNED(T*, ptr_);
  }
  template <typename T>
  const T* data() const {
    return HWY_RCAST_ALIGNED(const T*, ptr_);
  }

  const void* Ptr() const { return ptr_; }
  void* Ptr() { return ptr_; }
  // Sets the pointer from another MatPtr.
  void SetPtr(const MatPtr& other) { ptr_ = other.ptr_; }

  // Copying allowed as the metadata is small.
  MatPtr(const MatPtr& other) = default;
  MatPtr& operator=(const MatPtr& other) = default;

  // Returns the name of the blob.
  const std::string& Name() const { return name_; }
  void SetName(const std::string& name) { name_ = name; }

  // Returns the type of the blob.
  Type GetType() const { return type_; }

  // Returns the size of each element in bytes.
  size_t ElementSize() const { return element_size_; }

  // Returns the number of elements in the array.
  size_t NumElements() const { return num_elements_; }

  // Returns the number of bytes in the array.
  size_t SizeBytes() const { return num_elements_ * element_size_; }

  // Returns the number of rows in the 2-d array (outer dimension).
  size_t Rows() const { return rows_; }

  // Returns the number of columns in the 2-d array (inner dimension).
  size_t Cols() const { return cols_; }

  // Decoded elements should be multiplied by this to restore their original
  // range. This is required because SfpStream can only encode a limited range
  // of magnitudes.
  float scale() const { return scale_; }
  void set_scale(float scale) { scale_ = scale; }

  std::string LayerName(int layer) const {
    std::string name = name_ + std::to_string(layer);
    HWY_ASSERT(name.size() <= sizeof(hwy::uint128_t));
    return name;
  }

  // Adds the blob to the writer.
  void AddToWriter(BlobWriter& writer) const {
    fprintf(stderr, "Adding %s to writer\n", name_.c_str());
    writer.Add(MakeKey(name_.c_str()), ptr_, SizeBytes());
  }

  // Sets all data to zero.
  void ZeroInit() {
    if (ptr_ == nullptr)
      HWY_ABORT("ptr_ is null on tensor %s\n", name_.c_str());
    hwy::ZeroBytes(ptr_, SizeBytes());
  }

  // Calls func on the upcasted type. Since MatPtr by design is not templated,
  // here we provide a way to get to the derived type, provided that `Type()`
  // is one of the strings returned by `TypeName()`.
  template <class FuncT, typename... TArgs>
  decltype(auto) CallUpcasted(FuncT& func, TArgs&&... args);

 protected:
  // Arbitrary name for the array of preferably <= 16 characters.
  std::string name_;
  // Should be the result of TypeEnum<T> for CallUpcasted() to work.
  Type type_;
  // sizeof(T)
  size_t element_size_ = 0;
  // Number of elements in the array.
  size_t num_elements_ = 0;  // In element_size units.
  // Number of rows in the 2-d array (outer dimension).
  size_t rows_ = 0;
  // Number of columns in the 2-d array (inner dimension).
  size_t cols_ = 0;
  // Scaling to apply to each element.
  float scale_ = 1.0f;
  // Aligned data array. This is always a borrowed pointer. It should never be
  // freed. The underlying memory is owned by a subclass or some external class
  // and must outlive this object.
  void* ptr_ = nullptr;
};

// MatPtrT adds a single template argument to MatPtr for an explicit type.
// Use this class as a function argument where the type needs to be known.
// Use MatPtr where the type does not need to be known.
template <typename MatT>
class MatPtrT : public MatPtr {
 public:
  // Full constructor for dynamic sizing.
  MatPtrT(const std::string& name, size_t rows, size_t cols)
      : MatPtr(name, TypeEnum<MatT>(), sizeof(MatT), rows, cols) {}

  // Copying allowed as the metadata is small.
  MatPtrT(const MatPtr& other) : MatPtr(other) {}
  MatPtrT& operator=(const MatPtr& other) {
    MatPtr::operator=(other);
    return *this;
  }
  MatPtrT(const MatPtrT& other) = default;
  MatPtrT& operator=(const MatPtrT& other) = default;

  std::string CacheName(int layer = -1, char separator = ' ',
                        int index = -1) const {
    // Already used/retired: s, S, n, 1
    const char prefix = hwy::IsSame<MatT, float>()       ? 'F'
                        : hwy::IsSame<MatT, BF16>()      ? 'B'
                        : hwy::IsSame<MatT, SfpStream>() ? '$'
                        : hwy::IsSame<MatT, NuqStream>() ? '2'
                                                         : '?';
    std::string name = std::string(1, prefix) + name_;
    if (layer >= 0 || index >= 0) {
      name += '_';
      if (layer >= 0) name += std::to_string(layer);
      if (index >= 0) {
        name += separator + std::to_string(index);
      }
    }
    return name;
  }

  // Sets the number of elements in the array. For use when the number of
  // elements is != rows * cols ONLY.
  void SetNumElements(size_t num_elements) {
    num_elements_ = CompressedArrayElements<MatT>(num_elements);
  }

  // 2-d Accessor for a specific type but with a dynamic inner dimension.
  template <typename T = MatT>
  const T& At(size_t row, size_t col) const {
    size_t index = row * cols_ + col;
    HWY_DASSERT(index < num_elements_);
    return HWY_RCAST_ALIGNED(const T*, ptr_)[index];
  }

  // 1-d Accessor for a specific type.
  // TODO: replace this with a Foreach(), or at least a ForEachRow().
  const MatT& At(size_t index) const {
    HWY_DASSERT(index < num_elements_);
    return HWY_RCAST_ALIGNED(const MatT*, ptr_)[index];
  }
  MatT& At(size_t index) { return HWY_RCAST_ALIGNED(MatT*, ptr_)[index]; }

  // Compatibility interface for CompressedArray.
  // TODO: remove
  template <typename T = MatT>
  T* data() {
    return HWY_RCAST_ALIGNED(T*, ptr_);
  }
  template <typename T = MatT>
  const T* data() const {
    return HWY_RCAST_ALIGNED(const T*, ptr_);
  }
  // The const accessor data_scale1() asserts (!) that the scale is 1.0f, so
  // calling it means "I am sure the scale is 1 and therefore ignore the scale".
  // A scale of 0 indicates that the scale has likely never been set, so is
  // "implicitly 1".
  const MatT* data_scale1() const {
    HWY_ASSERT(scale() == 1.f);
    return HWY_RCAST_ALIGNED(const MatT*, ptr_);
  }
};

template <class FuncT, typename... TArgs>
decltype(auto) MatPtr::CallUpcasted(FuncT& func, TArgs&&... args) {
  if (type_ == TypeEnum<float>()) {
    return func(dynamic_cast<MatPtrT<float>*>(this),
                std::forward<TArgs>(args)...);
  } else if (type_ == TypeEnum<BF16>()) {
    return func(dynamic_cast<MatPtrT<BF16>*>(this),
                std::forward<TArgs>(args)...);
  } else if (type_ == TypeEnum<SfpStream>()) {
    return func(dynamic_cast<MatPtrT<SfpStream>*>(this),
                std::forward<TArgs>(args)...);
  } else if (type_ == TypeEnum<NuqStream>()) {
    return func(dynamic_cast<MatPtrT<NuqStream>*>(this),
                std::forward<TArgs>(args)...);
  } else {
    HWY_ABORT("Type %d unknown.", type_);
  }
}

// MatStorageT adds the actual data storage to MatPtrT.
template <typename MatT>
class MatStorageT : public MatPtrT<MatT> {
 public:
  // Full constructor for dynamic sizing.
  MatStorageT(const std::string& name, size_t rows, size_t cols)
      : MatPtrT<MatT>(name, rows, cols) {
    Allocate();
  }
  // Can copy the metadata, from a MatPtr, and allocate later.
  MatStorageT(const MatPtr& other) : MatPtrT<MatT>(other) {}
  ~MatStorageT() = default;

  // Move-only because this contains a unique_ptr.
  MatStorageT(const MatStorageT& other) = delete;
  MatStorageT& operator=(const MatStorageT& other) = delete;
  MatStorageT(MatStorageT&& other) = default;
  MatStorageT& operator=(MatStorageT&& other) = default;

  // Allocate the memory and copy the pointer to the MatPtr.
  // num_elements is in elements. In the default (zero) case, it is computed
  // from the current num_elements_ which was set by the constructor from the
  // rows and cols.
  void Allocate(size_t num_elements = 0) {
    if (num_elements == 0) {
      num_elements = hwy::DivCeil(this->SizeBytes(), sizeof(MatT));
    } else {
      this->num_elements_ = num_elements;
    }
    data_ = Allocator::Alloc<MatT>(num_elements);
    this->ptr_ = data_.get();
  }

  // Zeros the content.
  void ZeroInit() {
    HWY_ASSERT(data_ != nullptr);
    hwy::ZeroBytes(data_.get(), this->SizeBytes());
  }

 private:
  hwy::AlignedFreeUniquePtr<MatT[]> data_;
};

// MatStorage allows heterogeneous tensors to be stored in a single vector.
using MatStorage = MatStorageT<hwy::uint128_t>;

// Table of contents for a blob store file. Full metadata, but not actual data.
class BlobToc {
 public:
  BlobToc() = default;

  // Adds all blobs to the blob writer. Note that the blobs must have unique
  // names.
  static void AddAllToBlobWriter(const std::vector<MatStorage>& blobs,
                                 BlobWriter& writer) {
    std::vector<hwy::uint128_t> toc;
    for (const auto& blob : blobs) {
      blob.AddToToc(toc);
      blob.AddToWriter(writer);
    }
    writer.Add(MakeKey(kTocName), toc.data(), toc.size() * sizeof(toc[0]));
  }

  // Loads the table of contents from the given reader.
  BlobError LoadToc(BlobReader& reader) {
    hwy::uint128_t toc_key = MakeKey(kTocName);
    size_t toc_size = reader.BlobSize(toc_key);
    if (toc_size != 0) {
      std::vector<hwy::uint128_t> toc(toc_size / sizeof(hwy::uint128_t));
      BlobError err = reader.ReadOne(toc_key, toc.data(), toc_size);
      if (err != 0) {
        fprintf(stderr, "Failed to read toc (error %d)\n", err);
        return err;
      }
      for (size_t i = 0; i < toc.size(); i += MatPtr::kNumU128InTocEntry) {
        AddToToc(MatPtr(toc[i], toc[i + 1], toc[i + 2], toc[i + 3]));
      }
    }
    return 0;
  }

  bool Empty() const { return toc_map_.empty(); }

  // Returns true if the table of contents contains the given name.
  bool Contains(const std::string& name) const {
    return toc_map_.find(name) != toc_map_.end();
  }

  // Returns the blob with the given name, or nullptr if not found.
  const MatPtr* Get(const std::string& name) const {
    auto it = toc_map_.find(name);
    if (it == toc_map_.end()) return nullptr;
    return &toc_[it->second];
  }

 private:
  // The name of the toc in the blob store file.
  static constexpr char kTocName[] = "toc";

  // Adds the blob to the table of contents.
  void AddToToc(const MatPtr& blob) {
    HWY_ASSERT(!Contains(blob.Name()));
    toc_map_[blob.Name()] = toc_.size();
    toc_.push_back(blob);
  }

  std::unordered_map<std::string, size_t> toc_map_;
  std::vector<MatPtr> toc_;
};

#if COMPRESS_STATS
class CompressStats {
 public:
  void Notify(const DistortionStats& stats) {
    const float pnorm = stats.PNorm();
    const float snr = stats.GeomeanValueDivL1();
    num_exact_ += stats.NumExact();
    s_pnorm_.Notify(pnorm);
    // No loss - skip to avoid dragging down the average.
    if (snr != 0.0f) {
      s_snr_.Notify(snr);
    }
  }

  void NotifyIn(int sfp) { hist_weights_.Notify(sfp); }

  void Assimilate(const CompressStats& other) {
    s_pnorm_.Assimilate(other.s_pnorm_);
    s_snr_.Assimilate(other.s_snr_);
    num_exact_ += other.num_exact_;
    hist_weights_.Assimilate(other.hist_weights_);
  }

  void PrintAll() {
    const int skip = hwy::Stats::kNoGeomean;
    fprintf(stderr, "  pnorm %s\n", s_pnorm_.ToString(skip).c_str());
    fprintf(stderr, "   SNR  %s\n", s_snr_.ToString(skip).c_str());
    fprintf(stderr, "  #exact %.3E\n", static_cast<double>(num_exact_));
    // hist_weights_.Print("indices");
  }

  void Reset() {
    s_pnorm_.Reset();
    s_snr_.Reset();
    num_exact_ = 0;
    hist_weights_.Reset();
  }

 private:
  hwy::Stats s_pnorm_;
  hwy::Stats s_snr_;
  size_t num_exact_ = 0;
  hwy::Bins<1000> hist_weights_;
  char padding_[64];  // prevent false sharing
};
#else
struct CompressStats {
  void Notify(...) {}
  void NotifyIn(int) {}
  void Assimilate(const CompressStats&) {}
  void PrintAll() {}
  void Reset() {}
};
#endif  // COMPRESS_STATS

struct CompressPerThread {
  NuqStream::ClusterBuf buf;
  CompressStats stats;
};

struct CompressWorkingSet {
  std::vector<CompressPerThread> tls;
};

// Functor called for each tensor, which loads them and their scaling factors
// from BlobStore.
class ReadFromBlobStore {
 public:
  explicit ReadFromBlobStore(const Path& blob_filename) {
    err_ = reader_.Open(blob_filename);
    if (HWY_UNLIKELY(err_ != 0)) {
      fprintf(stderr, "Error %d opening BlobStore %s.\n", err_,
              blob_filename.path.c_str());
      return;  // avoid overwriting err_ to ensure ReadAll will fail.
    }
    err_ = file_toc_.LoadToc(reader_);
    if (HWY_UNLIKELY(err_ != 0)) {
      fprintf(stderr, "Found a TOC, but failed to load it (code %d)\n", err_);
    }
  }

  // Returns true if there is a TOC.
  bool HaveToc() const { return !file_toc_.Empty(); }

  // Called for each tensor, enqueues read requests.
  void operator()(const char* name, hwy::Span<MatPtr*> tensors) {
    if (file_toc_.Empty() || file_toc_.Contains(name)) {
      if (tensors[0]->NumElements() == 0)
        fprintf(stderr, "Zero elements for %s\n", name);
      model_toc_.push_back(tensors[0]);
      file_keys_.push_back(name);
    }
  }

  BlobError LoadScales(float* scales, size_t len) {
    for (size_t i = 0; i < len; ++i) {
      scales[i] = 1.0f;
    }
    MatPtrT<float> scales_ptr("scales", 0, 1);
    auto key = MakeKey(scales_ptr.CacheName().c_str());
    if (reader_.BlobSize(key) == 0) return 0;
    return reader_.Enqueue(key, scales, len * sizeof(scales[0]));
  }

  // Returns whether all tensors are successfully loaded from cache.
  BlobError ReadAll(hwy::ThreadPool& pool,
                    std::vector<MatStorage>& model_memory) {
    // reader_ invalid or any Enqueue failed
    if (err_ != 0) return err_;
    // Setup the model_memory.
    for (int b = 0; b < model_toc_.size(); ++b) {
      const std::string& file_key = file_keys_[b];
      MatPtr* blob = model_toc_[b];
      if (!file_toc_.Empty()) {
        const MatPtr* toc_blob = file_toc_.Get(file_key);
        if (toc_blob == nullptr) {
          fprintf(stderr, "Blob %s not found in TOC\n", file_key.c_str());
          return __LINE__;
        }
        if (toc_blob->Rows() != blob->Rows() ||
            toc_blob->Cols() != blob->Cols()) {
          fprintf(stderr, "Blob %s has size mismatch TOC\n", file_key.c_str());
          return __LINE__;
        }
        MatStorage toc_blob_array(*toc_blob);
        model_memory.push_back(std::move(toc_blob_array));
      } else {
        model_memory.emplace_back(*blob);
        model_memory.back().SetName(file_key);
      }
    }
    // Allocate in parallel using the pool.
    pool.Run(0, model_memory.size(),
             [this, &model_memory](uint64_t task, size_t /*thread*/) {
               model_memory[task].Allocate();
               model_toc_[task]->SetPtr(model_memory[task]);
             });
    // Enqueue the read requests.
    for (auto& blob : model_memory) {
      err_ = reader_.Enqueue(MakeKey(blob.Name().c_str()), blob.data(),
                             blob.SizeBytes());
      if (err_ != 0) {
        fprintf(stderr,
                "Failed to read blob %s (error %d) of size %zu x %zu x %zu\n",
                blob.Name().c_str(), err_, blob.Rows(), blob.Cols(),
                blob.ElementSize());
        return err_;
      }
    }
    return reader_.ReadAll(pool);
  }

 private:
  BlobReader reader_;
  BlobError err_ = 0;
  // Table of contents from the file, if present.
  BlobToc file_toc_;
  // Table of contents from the model. Pointers to original MatPtrT so the
  // data pointers can be updated.
  std::vector<MatPtr*> model_toc_;
  // Mangled names of the tensors in model_toc_ for reading from the file.
  std::vector<std::string> file_keys_;
};

}  // namespace gcpp
#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_COMPRESS_H_
