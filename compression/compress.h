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

#include "hwy/base.h"
#define COMPRESS_STATS 0

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// IWYU pragma: begin_exports
#include "compression/blob_store.h"
#include "compression/fields.h"
#include "compression/io.h"
#include "compression/shared.h"
#include "gemma/tensor_index.h"
#include "util/basics.h"
// IWYU pragma: end_exports
#include "gemma/configs.h"
#include "util/allocator.h"
#include "hwy/per_target.h"
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
class MatPtr : public IFields {
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
        ptr_(nullptr) {
    stride_ = cols;
  }
  // Default is to leave all fields default-initialized.
  MatPtr() = default;
  virtual ~MatPtr();

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
  const char* Name() const override { return name_.c_str(); }
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

  Extents2D Extents() const { return Extents2D(rows_, cols_); }

  // Currently same as cols, but may differ in the future. This is the offset by
  // which to advance pointers to the next row.
  size_t Stride() const { return stride_; }

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

  // Sets all data to zero.
  void ZeroInit() {
    if (ptr_ == nullptr)
      HWY_ABORT("ptr_ is null on tensor %s\n", name_.c_str());
    hwy::ZeroBytes(ptr_, SizeBytes());
  }

  void VisitFields(IFieldsVisitor& visitor) override {
    visitor(name_);
    visitor(type_);
    visitor(element_size_);
    visitor(num_elements_);
    visitor(rows_);
    visitor(cols_);
    visitor(scale_);
    visitor(stride_);
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
  uint32_t element_size_ = 0;
  // Number of elements in the array.
  uint32_t num_elements_ = 0;  // In element_size units.
  // Number of rows in the 2-d array (outer dimension).
  uint32_t rows_ = 0;
  // Number of columns in the 2-d array (inner dimension).
  uint32_t cols_ = 0;
  // Scaling to apply to each element.
  float scale_ = 1.0f;
  // Aligned data array. This is always a borrowed pointer. It should never be
  // freed. The underlying memory is owned by a subclass or some external class
  // and must outlive this object.
  void* ptr_ = nullptr;

  uint32_t stride_;
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
  // Construction from TensorIndex entry to remove duplication of sizes.
  MatPtrT(const std::string& name, const TensorIndex& tensor_index)
      : MatPtrT<MatT>(name, tensor_index.FindName(name)) {}
  MatPtrT(const std::string& name, const TensorInfo* tensor)
      : MatPtr(name, TypeEnum<MatT>(), sizeof(MatT), 0, 0) {
    if (tensor == nullptr) {
      cols_ = 0;
      rows_ = 0;
    } else {
      cols_ = tensor->shape.back();
      rows_ = 1;
      if (tensor->cols_take_extra_dims) {
        // The columns eat the extra dimensions.
        rows_ = tensor->shape[0];
        for (size_t i = 1; i < tensor->shape.size() - 1; ++i) {
          cols_ *= tensor->shape[i];
        }
      } else {
        // The rows eat the extra dimensions.
        for (size_t i = 0; i < tensor->shape.size() - 1; ++i) {
          rows_ *= tensor->shape[i];
        }
      }
    }
    stride_ = cols_;
    num_elements_ = rows_ * cols_;
  }

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
// TODO: use Extents2D instead of rows and cols.
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
    // Pad to allow overrunning the last row by 2 BF16 vectors, hence at most
    // `2 * VectorBytes / sizeof(BF16)` elements of MatT.
    const size_t padding = hwy::VectorBytes();
    data_ = Allocator::Alloc<MatT>(num_elements + padding);
    hwy::ZeroBytes(&data_[num_elements], padding * sizeof(MatT));
    this->ptr_ = data_.get();
  }

  // Zeros the content.
  void ZeroInit() {
    HWY_ASSERT(data_ != nullptr);
    hwy::ZeroBytes(data_.get(), this->SizeBytes());
  }

 private:
  AlignedPtr<MatT> data_;
};

// MatStorage allows heterogeneous tensors to be stored in a single vector.
using MatStorage = MatStorageT<hwy::uint128_t>;

// Table of contents for a blob store file. Full metadata, but not actual data.
class BlobToc {
 public:
  BlobToc() = default;

  // Loads the table of contents from the given reader.
  BlobError LoadToc(BlobReader& reader) {
    hwy::uint128_t toc_key = MakeKey(kTocName);
    size_t toc_size = reader.BlobSize(toc_key);
    if (toc_size != 0) {
      std::vector<uint32_t> toc(toc_size / sizeof(uint32_t));
      BlobError err = reader.ReadOne(toc_key, toc.data(), toc_size);
      if (err != 0) {
        fprintf(stderr, "Failed to read toc (error %d)\n", err);
        return err;
      }
      size_t consumed = 0;
      size_t prev_consumed = static_cast<size_t>(-1);
      while (consumed < toc.size() && prev_consumed != consumed) {
        MatPtr blob;
        const IFields::ReadResult result =
            blob.Read(hwy::Span<const uint32_t>(toc), consumed);
        prev_consumed = consumed;
        consumed = result.pos;
        if (blob.NumElements() > 0) {
          AddToToc(blob);
        }
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
  // The name of the toc in the blob store file.
  static constexpr char kTocName[] = "toc";

  // The name of the config in the blob store file.
  static constexpr char kConfigName[] = "config";

  // The name of the tokenizer in the blob store file.
  static constexpr char kTokenizerName[] = "tokenizer";

 private:
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
class DistortionStats;

struct CompressStats {
  void Notify(const DistortionStats&) {}
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

// Class to collect and write a set of tensors to a blob store file.
class WriteToBlobStore {
 public:
  explicit WriteToBlobStore(hwy::ThreadPool& pool) : pool_(pool) {}

  template <typename Packed>
  void operator()(MatPtrT<Packed>* compressed, const char* decorated_name) {
    if (compressed->Ptr() == nullptr) return;
    writer_.Add(MakeKey(decorated_name), compressed->Ptr(),
                compressed->SizeBytes());
    MatPtr renamed_tensor(*compressed);
    renamed_tensor.SetName(decorated_name);
    renamed_tensor.AppendTo(toc_);
  }

  void AddTokenizer(const std::string& tokenizer) {
    writer_.Add(MakeKey(BlobToc::kTokenizerName), tokenizer.data(),
                tokenizer.size() * sizeof(tokenizer[0]));
  }

  void AddScales(const float* scales, size_t len) {
    if (len) {
      MatPtrT<float> scales_ptr("scales", 0, 1);
      writer_.Add(MakeKey(scales_ptr.CacheName().c_str()), scales,
                  len * sizeof(scales[0]));
    }
  }

  // Writes all blobs to disk in the given order. The config is optional and
  // if given, it is written to the file, along with the TOC, making it
  // single-file format. Otherwise, the file is written in the multi-file format
  // without a TOC.
  BlobError WriteAll(const Path& blob_filename, const ModelConfig* config) {
    if (config) {
      writer_.Add(MakeKey(BlobToc::kTocName), toc_.data(),
                  toc_.size() * sizeof(toc_[0]));
      config_buffer_ = config->Write();
      writer_.Add(MakeKey(BlobToc::kConfigName), config_buffer_.data(),
                  config_buffer_.size() * sizeof(config_buffer_[0]));
    }
    const BlobError err = writer_.WriteAll(pool_, blob_filename);
    if (err != 0) {
      fprintf(stderr, "Failed to write blobs to %s (error %d)\n",
              blob_filename.path.c_str(), err);
    }
    return err;
  }

  // Returns the number of blobs added.
  size_t DebugNumBlobsAdded() const { return writer_.DebugNumBlobsAdded(); }

  hwy::ThreadPool& pool() { return pool_; }

 protected:
  hwy::ThreadPool& pool_;

 private:
  std::vector<uint32_t> toc_;
  BlobWriter writer_;
  std::vector<uint32_t> config_buffer_;
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

  // Reads the config from the blob store file.
  BlobError LoadConfig(ModelConfig& config) {
    hwy::uint128_t config_key = MakeKey(BlobToc::kConfigName);
    size_t config_size = reader_.BlobSize(config_key);
    if (config_size == 0) return __LINE__;
    std::vector<uint32_t> config_buffer(config_size / sizeof(uint32_t));
    BlobError err =
        reader_.ReadOne(config_key, config_buffer.data(), config_size);
    if (err != 0) {
      fprintf(stderr, "Failed to read config (error %d)\n", err);
      return err;
    }
    config.Read(hwy::Span<const uint32_t>(config_buffer), 0);
    return 0;
  }

  // Reads the tokenizer from the blob store file.
  BlobError LoadTokenizer(std::string& tokenizer) {
    hwy::uint128_t key = MakeKey(BlobToc::kTokenizerName);
    size_t tokenizer_size = reader_.BlobSize(key);
    if (tokenizer_size == 0) return __LINE__;
    tokenizer.resize(tokenizer_size);
    ;
    BlobError err = reader_.ReadOne(key, tokenizer.data(), tokenizer_size);
    if (err != 0) {
      fprintf(stderr, "Failed to read tokenizer (error %d)\n", err);
      return err;
    }
    return 0;
  }

  // Called for each tensor, enqueues read requests.
  void operator()(const char* name, hwy::Span<MatPtr*> tensors) {
    if (file_toc_.Empty() || file_toc_.Contains(name)) {
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
        std::string name = blob->Name();
        *blob = *toc_blob;
        blob->SetName(name);
      }
      model_memory.emplace_back(*blob);
      model_memory.back().SetName(file_key);
    }
    // Allocate in parallel using the pool.
    pool.Run(0, model_memory.size(),
             [this, &model_memory](uint64_t task, size_t /*thread*/) {
               model_memory[task].Allocate();
               model_toc_[task]->SetPtr(model_memory[task]);
             });
    // Enqueue the read requests.
    for (auto& blob : model_memory) {
      err_ =
          reader_.Enqueue(MakeKey(blob.Name()), blob.data(), blob.SizeBytes());
      if (err_ != 0) {
        fprintf(stderr,
                "Failed to read blob %s (error %d) of size %zu x %zu x %zu\n",
                blob.Name(), err_, blob.Rows(), blob.Cols(),
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
