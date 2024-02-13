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
// copybara:import_next_line:gemma_cpp
#include "compression/blob_store.h"
// copybara:import_next_line:gemma_cpp
#include "compression/nuq.h"
// copybara:import_next_line:gemma_cpp
#include "compression/sfp.h"
// IWYU pragma: end_exports
// copybara:import_next_line:gemma_cpp
#include "compression/distortion.h"
#include "hwy/base.h"  // hwy::bfloat16_t
#include "hwy/contrib/thread_pool/thread_pool.h"
#if COMPRESS_STATS
// copybara:import_next_line:gemma_cpp
#include "compression/stats.h"
#endif

namespace gcpp {

static inline const char* TypeName(float) { return "f32"; }
static inline const char* TypeName(hwy::bfloat16_t) { return "b16"; }

namespace detail {
// How many MatT are required to store `capacity` weights. For all but
// NuqStream, this is the same as `capacity`. For use by CompressedArray.
template <typename MatT>
constexpr size_t CompressedArrayLen(size_t capacity) {
  return capacity;
}
template <>
constexpr size_t CompressedArrayLen<NuqStream>(size_t capacity) {
  return NuqStream::PackedEnd(capacity);
}
}  // namespace detail

// Compressed representation of floating-point elements. The array length may
// differ from the number of elements. Associated operations such as Dot are
// implemented in SIMD code and are thus non-member functions.
template <typename MatT, size_t kCapacity>
class CompressedArray {
  static constexpr size_t NumCompressed() {
    return detail::CompressedArrayLen<MatT>(kCapacity);
  }

 public:
  MatT* data() { return data_.data(); }
  const MatT* data() const { return data_.data(); }

  constexpr size_t NumElements() const { return kCapacity; }

  constexpr size_t CompressedSize() const {
    return NumCompressed() * sizeof(MatT);
  }

 private:
  std::array<MatT, NumCompressed()> data_;
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
    const int skip = Stats::kNoGeomean;
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
  Stats s_pnorm_;
  Stats s_snr_;
  size_t num_exact_ = 0;
  Bins<1000> hist_weights_;
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
  CompressStats stats;
  ClusterBuf buf;
};

struct CompressWorkingSet {
  std::vector<CompressPerThread> tls;
};

// Returns key for the given tensor name. Also encodes the type, so that
// changing the representation automatically invalidates prior cached files
// (the new blob name will not be found).
template <typename MatT>
hwy::uint128_t CacheKey(const char* name) {
  // Already used/retired: s, S, n, 1
  const char prefix = hwy::IsSame<MatT, float>()             ? 'F'
                      : hwy::IsSame<MatT, hwy::bfloat16_t>() ? 'B'
                      : hwy::IsSame<MatT, SfpStream>()       ? '$'
                      : hwy::IsSame<MatT, NuqStream>()       ? '2'
                                                             : '?';

  return MakeKey((std::string(1, prefix) + name).c_str());
}

class CacheLoader {
 public:
  explicit CacheLoader(const char* blob_filename) {
    err_ = reader_.Open(blob_filename);
    if (err_ != 0) {
      fprintf(stderr,
              "Cached compressed weights does not exist yet (code %d), "
              "compressing weights and creating file: %s.\n",
              err_, blob_filename);
    }
  }

  // Called for each tensor, enqueues read requests.
  template <typename MatT, size_t kCapacity>
  void operator()(const char* name, const float* null,
                  CompressedArray<MatT, kCapacity>& compressed) {
    HWY_DASSERT(null == nullptr);

    // Skip if reader_ is invalid or any load failed: we will regenerate
    // everything because it's rare to update only a few tensors.
    if (err_ != 0) return;

    err_ = reader_.Enqueue(CacheKey<MatT>(name), compressed.data(),
                           compressed.CompressedSize());
    if (err_ != 0) {
      fprintf(stderr, "Failed to read cache %s (error %d)\n", name, err_);
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
