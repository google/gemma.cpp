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

#ifndef THIRD_PARTY_GEMMA_CPP_COMPRESSION_NUQ_H_
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_NUQ_H_

// Non-uniform quantization: a compressed representation of f32 inputs that
// supports seeking at a granularity of kGroupSize, decoding to bf16/f32, and a
// fused decode/dot product with bf16/f32 vectors.

#include <stddef.h>
#include <stdint.h>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"  // HWY_INLINE

namespace gcpp {

// 4-bit indices are a sweet spot in terms of quality per size.
static constexpr size_t kClusters = 16;

// Number of weights that share a table. Larger = slower encode, higher error,
// smaller size (table amortized over more weights). This is the minimum
// granularity for seeking/decoding in the stream, and must be at least four
// times the number of bf16 elements per vector.
static constexpr size_t kGroupSize = 256;

// Points to the *start* of a NUQ stream. Aligning the allocation (see
// aligned_allocator.h) may be speed up decoding but is not required.
//
// See go/streaming-weight-decode for background and design. Layout: first one
// table of kClusters entries per group, in ascending order of group index,
// then two packed indices per byte.
//
// Indices are stored in-order to enable vector-length agnostic decode, because
// streams may be persisted to disk and used by other CPUs.
//
// To enable parallel encoding and decoding, Enc/Dec have `offset` parameters
// which refer to the stream, NOT the raw from/to pointers, which point directly
// to the source/destination. Offsets are in units of values, NOT compressed
// bytes within the stream.
#pragma pack(push, 1)
struct NuqStream {
  // Returns offset of packed indices from the start of the stream. This matches
  // the (padded) total table size because table entries are bytes. `capacity`
  // is already a multiple of `kGroupSize`.
  static constexpr size_t PackedStart(size_t capacity) {
    // Round up to avoid cache-line splits when loading indices. No effect on
    // size as long as capacity / kGroupSize is a multiple of 4.
    return hwy::RoundUpTo((capacity / kGroupSize) * kClusters, 64);
  }

  // Returns number of NuqStream to allocate for the stream, which matches its
  // size in bytes. `capacity` is already a multiple of `kGroupSize`.
  static constexpr size_t PackedEnd(size_t capacity) {
    return PackedStart(capacity) + capacity / 2;  // two 4-bit indices per byte.
  }

  uint8_t byte;
};
#pragma pack(pop)

static inline const char* TypeName(NuqStream) { return "NUQ"; }

// Storage for dynamic programming. There are two matrices; we use separate
// allocations to avoid type punning.
template <class T>
class AlignedMatrix {
 public:
  AlignedMatrix() : mem_(hwy::AllocateAligned<T>(kClusters * kGroupSize)) {}

  HWY_INLINE const T& operator()(size_t row, size_t col) const {
    return mem_[row * kGroupSize + col];
  }

  HWY_INLINE T& operator()(size_t row, size_t col) {
    return mem_[row * kGroupSize + col];
  }

 private:
  hwy::AlignedFreeUniquePtr<T[]> mem_;
};

// Reuse memory across calls to Enc to avoid per-call allocations.
struct ClusterBuf {
  void Resize(size_t new_num) {
    if (new_num < num) return;

    num = new_num;
    const size_t num_groups = hwy::DivCeil(num, kGroupSize);
    centers = hwy::AllocateAligned<float>(num_groups * kClusters);
    idx = hwy::AllocateAligned<uint16_t>(num);
  }

  AlignedMatrix<float> d;
  AlignedMatrix<int32_t> t;

  size_t num = 0;
  hwy::AlignedFreeUniquePtr<float[]> centers;
  hwy::AlignedFreeUniquePtr<uint16_t[]> idx;
};

}  // namespace gcpp
#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_NUQ_H_
