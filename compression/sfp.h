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

#ifndef THIRD_PARTY_GEMMA_CPP_COMPRESSION_SFP_H_
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_SFP_H_

// Switching Floating Point: a hybrid 8-bit float representation of bf16/f32
// inputs that combines the advantages of e4m3 and e5m2 into a single format.
// It supports seeking at a granularity of 1, decoding to bf16/f32, and a
// fused decode/dot product with bf16/f32 vectors.

#include <stdint.h>

namespace gcpp {

// Points to the *start* of an SFP stream. Values are stored in-order to enable
// vector-length agnostic seeking, because streams may be written to disk for
// loading on other CPUs.
//
// Characteristics:
// - 24-bit dynamic range, with max exponent 2^0.
// - 3 bit mantissa for values >= 2^-7, otherwise 2.
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

static inline const char* TypeName(SfpStream) { return "SFP"; }

}  // namespace gcpp
#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_SFP_H_
