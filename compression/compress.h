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
#include "util/mat.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#if COMPRESS_STATS
#include "compression/distortion.h"
#include "hwy/stats.h"
#endif

namespace gcpp {

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
        if (!blob.IsEmpty()) {
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
  void operator()(MatPtrT<Packed>* compressed,
                  const char* decorated_name) const {
    if (!compressed->HasPtr()) return;
    writer_.Add(MakeKey(decorated_name), compressed->Packed(),
                compressed->PackedBytes());
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
      MatPtrT<float> scales_ptr("scales", Extents2D(0, 1));
      writer_.Add(MakeKey(scales_ptr.Name()), scales, len * sizeof(scales[0]));
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
  mutable std::vector<uint32_t> toc_;
  mutable BlobWriter writer_;
  mutable std::vector<uint32_t> config_buffer_;
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
      HWY_ASSERT(tensors[0]);
      model_toc_.push_back(tensors[0]);
      file_keys_.push_back(name);
    }
  }

  BlobError LoadScales(float* scales, size_t len) {
    for (size_t i = 0; i < len; ++i) {
      scales[i] = 1.0f;
    }
    MatPtrT<float> scales_ptr("scales", Extents2D(0, 1));
    auto key = MakeKey(scales_ptr.Name());
    if (reader_.BlobSize(key) == 0) return 0;
    return reader_.Enqueue(key, scales, len * sizeof(scales[0]));
  }

  // Returns whether all tensors are successfully loaded from cache.
  BlobError ReadAll(hwy::ThreadPool& pool,
                    std::vector<MatOwner>& model_memory) {
    // reader_ invalid or any Enqueue failed
    if (err_ != 0) return err_;
    // Setup the model_memory.
    for (size_t b = 0; b < model_toc_.size(); ++b) {
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
        blob->SetName(name.c_str());
      }
      model_memory.push_back(MatOwner());
    }
    // Allocate in parallel using the pool.
    pool.Run(0, model_memory.size(),
             [this, &model_memory](uint64_t task, size_t /*thread*/) {
               model_memory[task].AllocateFor(*model_toc_[task],
                                              MatPadding::kPacked);
             });
    // Enqueue the read requests.
    for (size_t b = 0; b < model_toc_.size(); ++b) {
      err_ = reader_.Enqueue(MakeKey(file_keys_[b].c_str()),
                             model_toc_[b]->RowT<uint8_t>(0),
                             model_toc_[b]->PackedBytes());
      if (err_ != 0) {
        fprintf(
            stderr,
            "Failed to read blob %s (error %d) of size %zu x %zu, type %d\n",
            file_keys_[b].c_str(), err_, model_toc_[b]->Rows(),
            model_toc_[b]->Cols(), static_cast<int>(model_toc_[b]->GetType()));
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
