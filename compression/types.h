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

// Types shared between tensor definitions and `compress-inl.h`.

#ifndef THIRD_PARTY_GEMMA_CPP_COMPRESSION_TYPES_H_
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_TYPES_H_

#include <stddef.h>
#include <stdint.h>

// IWYU pragma: begin_exports
#include "util/basics.h"  // BF16
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"  // HWY_INLINE
// IWYU pragma: end_exports

namespace gcpp {

// EMU128 must not be disabled because we disable SCALAR.
#define HWY_BROKEN_EMU128 0

// Allow user override of disabled targets.
#ifndef GEMMA_DISABLED_TARGETS

// All platforms: exclude SCALAR because we use ReorderWidenMulAccumulate.

#if HWY_ARCH_ARM_V7
// No NEON because we require double-precision support.
#define HWY_DISABLED_TARGETS (HWY_SCALAR | HWY_ALL_NEON)
#elif HWY_ARCH_ARM_A64
// We do not yet use AES (e.g. for random generation), hence NEON is the same
// as NEON_WITHOUT_AES. Also skip SVE because SVE2_128 and SVE_256 cover most.
#define GEMMA_DISABLED_TARGETS (HWY_SCALAR | HWY_NEON | HWY_SVE)
#elif HWY_ARCH_X86
// Skip anything older than Haswell (2013); also use Zen4 for recent CPUs,
// because we do not use anything added by SPR (e.g. FP16) nor AVX 10.2.
#define GEMMA_DISABLED_TARGETS \
  (HWY_SCALAR | HWY_SSE2 | HWY_SSSE3 | HWY_SSE4 | HWY_AVX3_SPR | HWY_AVX10_2)
#endif  // HWY_ARCH_*

#endif  // GEMMA_DISABLED_TARGETS

// Only used in experiments, hence disable in default builds.
#ifndef GEMMA_ENABLE_NUQ
#define GEMMA_ENABLE_NUQ 0
#endif

// Switching Floating Point: a hybrid 8-bit float representation of bf16/f32
// inputs that combines the advantages of e4m3 and e5m2 into a single format.
// It supports seeking at a granularity of 1 and decoding to bf16/f32.
//
// Characteristics:
// - 24-bit dynamic range, with max exponent 2^0.
// - 3 bit mantissa for values >= 2^-7, otherwise 2.
//
// A pointer to this is the *start* of an SFP stream. Aligning the allocation
// (see aligned_allocator.h) may speed up decoding but is not required.
//
// Layout: Values are stored in-order to enable vector-length agnostic seeking,
// because streams may be written to disk for loading on other CPUs.
//
// This is faster to decode than a straightforward implementation of eXmY, in
// part because SFP does not require subnormals. Unlike OCP MX, it also does not
// require side information (shared exponents).
//
// Although the representation could probably be shrunk to 6-7 bits, more
// savings can be had by non-uniform clustering - see NuqStream.
#pragma pack(push, 1)
struct SfpStream {
  // Largest possible input magnitude: 1.111 * 2^0. This could be increased by
  // shifting the value range (exponent bias).
  static constexpr float kMax = 1.875f;

  uint8_t byte;
};
#pragma pack(pop)

// Non-uniform quantization: a compressed representation of f32 inputs that
// supports seeking at a granularity of 1 (for `DecompressAndZeroPad`) or
// two vectors (for `Decompress2`), and decoding to bf16/f32.
//
// A pointer to this is the *start* of a NUQ stream. Aligning the allocation
// (see aligned_allocator.h) may be speed up decoding but is not required.
//
// Layout: first one table of kClusters entries per group, in ascending order
// of group index, then two packed indices per byte. Indices are stored
// in-order to enable vector-length agnostic decode, because streams may be
// persisted to disk and used by other CPUs.
//
// To enable parallel encoding and decoding, Enc/Dec have `offset` parameters
// which refer to the stream, NOT the raw from/to pointers, which point directly
// to the source/destination. Offsets are in units of values, NOT compressed
// bytes within the stream.
#pragma pack(push, 1)
struct NuqStream {
  // 4-bit indices are a sweet spot in terms of quality per size.
  static constexpr size_t kClusters = 16;

  // Number of weights that share a table. Larger = slower encode, higher error,
  // smaller size (table amortized over more weights).
  static constexpr size_t kGroupSize = 256;

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
    // Move-only (stored inside vector in CompressWorkingSet).
    ClusterBuf() = default;
    ClusterBuf(const ClusterBuf&) = delete;
    ClusterBuf& operator=(const ClusterBuf&) = delete;
    ClusterBuf(ClusterBuf&&) = default;
    ClusterBuf& operator=(ClusterBuf&&) = default;

    // Independent of num_groups.
    AlignedMatrix<float> costs;
    AlignedMatrix<int32_t> argmin;
  };

  // Returns offset of packed indices from the start of the stream. This matches
  // the (padded) total table size because table entries are bytes.
  static constexpr size_t PackedStart(size_t capacity) {
    // Round up to avoid cache-line splits when loading indices. No effect on
    // size as long as capacity / kGroupSize is a multiple of 4.
    return hwy::RoundUpTo(hwy::DivCeil(capacity, kGroupSize) * kClusters, 64);
  }

  // Returns number of NuqStream to allocate for the stream, which matches its
  // size in bytes.
  static constexpr size_t PackedEnd(size_t capacity) {
    const size_t num_groups = hwy::DivCeil(capacity, kGroupSize);
    return (kClusters * num_groups) +
           hwy::DivCeil(capacity, 2);  // 2x 4-bit/byte
  }

  uint8_t byte;
};
#pragma pack(pop)

template <typename Packed>
constexpr bool IsF32() {
  return hwy::IsSame<hwy::RemoveCvRef<Packed>, float>();
}

template <typename Packed>
constexpr bool IsBF16() {
  return hwy::IsSame<hwy::RemoveCvRef<Packed>, BF16>();
}

template <typename Packed>
constexpr bool IsSfpStream() {
  return hwy::IsSame<hwy::RemoveCvRef<Packed>, SfpStream>();
}

template <typename Packed>
constexpr bool IsNuqStream() {
  return hwy::IsSame<hwy::RemoveCvRef<Packed>, NuqStream>();
}

// Tensor types for loading weights.
enum class Type { kUnknown, kF32, kBF16, kSFP, kNUQ, kF64 };
// These are used in `ModelConfig.Specifier`, hence the strings will not
// change, though new ones may be added.
static constexpr const char* kTypeStrings[] = {"unknown", "f32", "bf16",
                                               "sfp",     "nuq", "f64"};
static constexpr size_t kNumTypes =
    sizeof(kTypeStrings) / sizeof(kTypeStrings[0]);
static constexpr size_t kTypeBits[] = {
    0,
    8 * sizeof(float),
    8 * sizeof(BF16),
    8 * sizeof(SfpStream),
    4 /* NuqStream, actually 4.5 */,
    8 * sizeof(double),
};

static inline bool EnumValid(Type type) {
  return static_cast<size_t>(type) < kNumTypes;
}

// Returns a Type enum for the type of the template parameter.
template <typename PackedT>
Type TypeEnum() {
  using Packed = hwy::RemoveCvRef<PackedT>;
  if constexpr (hwy::IsSame<Packed, float>()) {
    return Type::kF32;
  } else if constexpr (hwy::IsSame<Packed, BF16>()) {
    return Type::kBF16;
  } else if constexpr (hwy::IsSame<Packed, SfpStream>()) {
    return Type::kSFP;
  } else if constexpr (hwy::IsSame<Packed, NuqStream>()) {
    return Type::kNUQ;
  } else if constexpr (hwy::IsSame<Packed, double>()) {
    return Type::kF64;
  } else {
    HWY_DASSERT(false);
    return Type::kUnknown;
  }
}

static inline size_t TypeBits(Type type) {
  return kTypeBits[static_cast<int>(type)];
}

static inline const char* TypeName(Type type) {
  return kTypeStrings[static_cast<int>(type)];
}
template <typename PackedT>
const char* TypeName() {
  return TypeName(TypeEnum<PackedT>());
}

template <typename Packed>
constexpr bool IsCompressed() {
  return hwy::IsSameEither<hwy::RemoveCvRef<Packed>, SfpStream, NuqStream>();
}

// Returns the number of `MatT` elements required to store `capacity` values,
// which must not be zero. This is only intended to support the extra tables
// required for NUQ. `capacity` includes any padding and is `rows * stride`.
// Deprecated, replaced by fixup within `MatPtr`. Only used by tests.
template <typename Packed>
constexpr size_t CompressedArrayElements(size_t capacity) {
  if constexpr (hwy::IsSame<hwy::RemoveCvRef<Packed>, NuqStream>()) {
    return NuqStream::PackedEnd(capacity);
  } else {
    return capacity;
  }
}

// Non-owning view of packed elements. Shortens argument lists.
//
// Callers typically also pass an `ofs` starting offset. This is not folded
// into `ptr` because NUQ consists of two separate streams. To discourage direct
// use of `ptr` without that offset, we define a separate class instead of
// reusing `hwy::Span`.
template <typename Packed>
struct PackedSpan {
  // Ensures callers can read or write `num_accessible` elements starting at
  // `packed_ofs`.
  void BoundsCheck(size_t packed_ofs, size_t num_accessible) const {
    if constexpr (HWY_IS_DEBUG_BUILD) {
      // For NUQ, there can be fewer Packed than the number of elements, hence
      // check the compressed count and ensure we have that many.
      const size_t required =
          CompressedArrayElements<Packed>(packed_ofs + num_accessible);
      if (num < required) {
        HWY_ABORT("PackedSpan: ofs %zu, want %zu, req %zu > %zu packed",
                  packed_ofs, num_accessible, required, num);
      }
    }
  }

  Packed* HWY_RESTRICT ptr;
  size_t num;  // for BoundsCheck, also required by nuq-inl.h.
};

// Avoids spelling out the template parameter in every call.
template <typename Packed>
HWY_INLINE PackedSpan<Packed> MakeSpan(Packed* ptr, size_t size) {
  return {ptr, size};
}

template <typename Packed>
HWY_INLINE PackedSpan<const Packed> MakeConstSpan(Packed* ptr, size_t size) {
  return {ptr, size};
}

// "Implicit" conversion from a PackedSpan<T> to PackedSpan<const T>, used in
// `RMSNormInplace` and compression tests.
template <typename Packed>
HWY_INLINE PackedSpan<const Packed> MakeConst(PackedSpan<Packed> packed) {
  return {packed.ptr, packed.num};
}

}  // namespace gcpp
#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_TYPES_H_
