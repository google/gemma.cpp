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

// Definitions shared between the public compress-inl.h interface and the
// sfp-inl.h and nuq-inl.h implementation details.

#ifndef THIRD_PARTY_GEMMA_CPP_COMPRESSION_SHARED_H_
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_SHARED_H_

#include <stddef.h>

#include "hwy/base.h"  // hwy::bfloat16_t

namespace gcpp {

using BF16 = hwy::bfloat16_t;

// Switching Floating Point: a hybrid 8-bit float representation of bf16/f32
// inputs that combines the advantages of e4m3 and e5m2 into a single format.
// It supports seeking at a granularity of 1 and decoding to bf16/f32.
//
// Characteristics:
// - 24-bit dynamic range, with max exponent 2^0.
// - 3 bit mantissa for values >= 2^-7, otherwise 2.
//
// A pointer to this is the *start* of an SFP stream. Values are stored
// in-order to enable vector-length agnostic seeking, because streams may be
// written to disk for loading on other CPUs.
//
// This is faster to decode than a straightforward implementation of eXmY, in
// part because SFP does not require subnormals. Unlike OCP MX, it also does not
// require side information (shared exponents).
//
// Although the representation could probably be shrunk to 6-7 bits, more
// savings can be had by non-uniform clustering - see nuq.h.
#pragma pack(push, 1)
struct SfpStream {
  uint8_t byte;
};
#pragma pack(pop)

// Largest possible input magnitude: 1.111 * 2^0. This could be increased by
// shifting the value range (exponent bias).
constexpr float kMaxSFP = 1.875f;

// Non-owning view of packed elements. Shortens argument lists.
//
// Callers typically also pass an `ofs` starting offset. This is not folded
// into `ptr` because NUQ consists of two separate streams. To discourage direct
// use of `ptr` without that offset, we define a separate class instead of
// reusing `hwy::Span`.
template <typename Packed>
struct PackedSpan {
  void BoundsCheck(size_t packed_ofs, size_t num) const {
    HWY_DASSERT(packed_ofs + num <= size);
    (void)size;
  }

  Packed* HWY_RESTRICT ptr;
  size_t size;  // for BoundsCheck and nuq-inl.h HWY_ASSERT.
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
  return {packed.ptr, packed.size};
}

}  // namespace gcpp
#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_SHARED_H_
