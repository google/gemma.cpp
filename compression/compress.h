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
#include <stdio.h>

#include <array>
#include <string>
#include <vector>

// IWYU pragma: begin_exports
#include "compression/blob_store.h"
#include "compression/io.h"
#include "compression/shared.h"
// IWYU pragma: end_exports
#include "compression/distortion.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#if COMPRESS_STATS
#include "hwy/stats.h"
#endif

namespace gcpp {

// Compressed representation of floating-point elements. The array length may
// differ from the number of elements. Associated operations such as Dot are
// implemented in SIMD code and are thus non-member functions.
template <typename Packed, size_t kCapacity>
class CompressedArray {
 public:
  using value_type = Packed;

  // Note that whenever you access data(), you have to consider a scale() that
  // may be different from 1.0f.
  Packed* data() { return data_.data(); }
  const Packed* data() const { return data_.data(); }
  // The const accessor data_scale1() asserts (!) that the scale is 1.0f, so
  // calling it means "I am sure the scale is 1 and therefore ignore the scale".
  // A scale of 0 indicates that the scale has likely never been set, so is
  // "implicitly 1".
  const Packed* data_scale1() const {
    HWY_ASSERT(scale() == 1.f || scale() == 0.f);
    return data_.data();
  }

  // Decoded elements should be multiplied by this to restore their original
  // range. This is required because SfpStream can only encode a limited range
  // of magnitudes.
  float scale() const { return scale_[0]; }
  void set_scale(float scale) { scale_[0] = scale; }

  constexpr size_t NumElements() const { return kCapacity; }

  // Returns total number of packed elements for `BlobReader::Enqueue` and
  // `Compress`. This differs from `NumElements` for `Packed=NuqStream`.
  PackedSpan<Packed> GetSpan() { return MakeSpan(data(), data_.size()); }
  PackedSpan<const Packed> GetSpan() const {
    return MakeSpan(data(), data_.size());
  }

 private:
  std::array<Packed, CompressedArrayElements<Packed>(kCapacity)> data_;
  // Blobs are at least kBlobAlign bytes anyway.
  float scale_[kBlobAlign / sizeof(float)];
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

// Returns key for the given tensor name. Also encodes the type, so that
// changing the representation automatically invalidates prior cached files
// (the new blob name will not be found).
template <typename Packed>
hwy::uint128_t CacheKey(const char* name) {
  // Already used/retired: s, S, n, 1
  const char prefix = hwy::IsSame<Packed, float>()       ? 'F'
                      : hwy::IsSame<Packed, BF16>()      ? 'B'
                      : hwy::IsSame<Packed, SfpStream>() ? '$'
                      : hwy::IsSame<Packed, NuqStream>() ? '2'
                                                         : '?';

  return MakeKey((std::string(1, prefix) + name).c_str());
}

// Functor called for each tensor, which loads them and their scaling factors
// from BlobStore.
class CacheLoader {
 public:
  explicit CacheLoader(const Path& blob_filename) {
    err_ = reader_.Open(blob_filename);
    if (err_ != 0) {
      fprintf(stderr,
              "Cached compressed weights does not exist yet (code %d), "
              "compressing weights and creating file: %s.\n",
              err_, blob_filename.path.c_str());
    }
  }

  // Called for each tensor, enqueues read requests.
  template <typename Packed, size_t kCapacity>
  void operator()(const char* name, const float* null,
                  CompressedArray<Packed, kCapacity>& compressed) {
    HWY_DASSERT(null == nullptr);

    // Skip if reader_ is invalid or any load failed: we will regenerate
    // everything because it's rare to update only a few tensors.
    if (err_ != 0) return;

    const PackedSpan<Packed> span = compressed.GetSpan();
    const size_t num_bytes = span.num * sizeof(Packed);
    err_ = reader_.Enqueue(CacheKey<Packed>(name), span.ptr, num_bytes);
    compressed.set_scale(1.0f);
    if (err_ != 0) {
      fprintf(stderr, "Failed to read cache %s (error %d)\n", name, err_);
    }
  }

  void LoadScales(float* scales, size_t len) {
    if (0 != reader_.Enqueue(CacheKey<float>("scales"), scales,
                             len * sizeof(scales[0]))) {
      for (size_t i = 0; i < len; ++i) {
        scales[i] = 1.0f;
      }
    }
  }

  // Returns whether all tensors are successfully loaded from cache.
  bool ReadAll(hwy::ThreadPool& pool) {
    // reader_ invalid or any Enqueue failed
    if (err_ != 0) return false;

    err_ = reader_.ReadAll(pool);
    if (err_ != 0) {
      fprintf(stderr, "Failed to read all tensors (error %d)\n", err_);
      return false;
    }

    return true;
  }

 private:
  BlobReader reader_;
  BlobError err_ = 0;
};

}  // namespace gcpp
#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_COMPRESS_H_
