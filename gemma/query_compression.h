// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_QUERY_COMPRESSION_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_QUERY_COMPRESSION_H_

// Compression of float32 attention queries to BF16, int16, uint8 or int8.
// Kept separate from tiled_attention so that it can be used without depending
// on the full Gemma library.

#include <stddef.h>
#include <stdint.h>

#include "util/basics.h"  // BF16
#include "hwy/base.h"
#include "hwy/highway.h"

namespace gcpp {

// Passed to HWY_VISIT_TARGETS; declares for one target.
#define GEMMA_DECL_QUERY_COMPRESSION(TARGET, NAMESPACE)                       \
  namespace NAMESPACE {                                                       \
  void CompressQueriesBF16(hwy::Span<const float* const> input, int qkv_dim,  \
                           BF16* HWY_RESTRICT output);                        \
  void CompressQueriesBF16Contiguous(const float* HWY_RESTRICT input,         \
                                     int qkv_dim, size_t num_queries,         \
                                     BF16* HWY_RESTRICT output);              \
                                                                              \
  void CompressQueriesInt16(hwy::Span<const float* const> input, int qkv_dim, \
                            int16_t* HWY_RESTRICT output,                     \
                            float* HWY_RESTRICT scale);                       \
                                                                              \
  void CompressQueriesInt16Contiguous(const float* HWY_RESTRICT input,        \
                                      int qkv_dim, size_t num_queries,        \
                                      int16_t* HWY_RESTRICT output,           \
                                      float* HWY_RESTRICT scale);             \
                                                                              \
  void CompressQueriesUint8(hwy::Span<const float* const> input, int qkv_dim, \
                            int8_t* HWY_RESTRICT output,                      \
                            float* HWY_RESTRICT scale);                       \
                                                                              \
  void CompressQueriesUint8Contiguous(const float* HWY_RESTRICT input,        \
                                      int qkv_dim, size_t num_queries,        \
                                      int8_t* HWY_RESTRICT output,            \
                                      float* HWY_RESTRICT scale);             \
                                                                              \
  void CompressQueriesInt8(hwy::Span<const float* const> input, int qkv_dim,  \
                           int8_t* HWY_RESTRICT output,                       \
                           float* HWY_RESTRICT scale);                        \
                                                                              \
  void CompressQueriesInt8Contiguous(const float* HWY_RESTRICT input,         \
                                     int qkv_dim, size_t num_queries,         \
                                     int8_t* HWY_RESTRICT output,             \
                                     float* HWY_RESTRICT scale);              \
  /* NOLINTNEXTLINE(google-readability-namespace-comments) */                 \
  }  // namespace NAMESPACE

// Function declarations for each SIMD target. Allows direct call from the
// per-target namespace.
HWY_VISIT_TARGETS(GEMMA_DECL_QUERY_COMPRESSION)

#undef GEMMA_DECL_QUERY_COMPRESSION

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_QUERY_COMPRESSION_H_
