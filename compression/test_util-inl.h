// Copyright 2024 Google LLC
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

// Include guard for headers.
#ifndef THIRD_PARTY_GEMMA_CPP_COMPRESSION_TEST_UTIL_INL_H_
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_TEST_UTIL_INL_H_

// IWYU pragma: begin_exports
#include "compression/distortion.h"
#include "util/mat.h"
// IWYU pragma: end_exports

#include "compression/compress.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_TEST_UTIL_INL_H_

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_COMPRESS_TEST_UTIL_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef THIRD_PARTY_GEMMA_CPP_COMPRESS_TEST_UTIL_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_COMPRESS_TEST_UTIL_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_COMPRESS_TEST_UTIL_TOGGLE
#endif

#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"
#include "hwy/tests/test_util-inl.h"  // IWYU pragma: export

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// `Packed` is the type passed to `TestT`.
template <typename Packed, template <class> class TestT>
void ForeachRawType() {
  const hn::ForGEVectors<128, TestT<Packed>> test;
  // The argument selects the type to decode to: BF16 or float.
  test(BF16());
  test(float());
  // Do not include double because it is not supported as an input type - we
  // would also have to implement double -> Packed Compress().
}

template <template <class> class TestT>
void ForeachPackedAndRawType() {
  ForeachRawType<BF16, TestT>();
  ForeachRawType<float, TestT>();
  ForeachRawType<SfpStream, TestT>();
  if constexpr (GEMMA_ENABLE_NUQ) {
    ForeachRawType<NuqStream, TestT>();
  }
}

// Generates inputs: deterministic, within max SfpStream range.
template <typename MatT>
MatStorageT<MatT> GenerateMat(const Extents2D& extents,
                              const Allocator& allocator, MatPadding padding,
                              hwy::ThreadPool& pool) {
  gcpp::CompressWorkingSet ws;
  ws.tls.resize(pool.NumWorkers());
  MatStorageT<float> raw("raw", extents, allocator, MatPadding::kPacked);
  MatStorageT<MatT> compressed("mat", extents, allocator, padding);
  const float scale = SfpStream::kMax / extents.Area();
  pool.Run(0, extents.rows, [&](const size_t r, size_t thread) {
    float* HWY_RESTRICT row = raw.Row(r);
    for (size_t c = 0; c < extents.cols; c++) {
      float f = static_cast<float>(r * extents.cols + c) * scale;
      if ((r + c) & 1) f = -f;  // Also generate some negative values.
      row[c] = f;
    }
    Compress(raw.Row(r), raw.Cols(), ws.tls[thread],
             MakeSpan(compressed.Row(r), compressed.Cols()),
             /*packed_ofs=*/0);
  });

  compressed.SetScale(0.6f);  // Arbitrary value, different from 1.
  return compressed;
}

// Same, but `extents` describes the transposed matrix.
template <typename MatT>
MatStorageT<MatT> GenerateTransposedMat(const Extents2D extents,
                                        const Allocator& allocator,
                                        MatPadding padding,
                                        hwy::ThreadPool& pool) {
  gcpp::CompressWorkingSet ws;
  ws.tls.resize(pool.NumWorkers());
  MatStorageT<float> raw("raw", extents, allocator, MatPadding::kPacked);
  MatStorageT<MatT> compressed("trans", extents, allocator, padding);
  const float scale = SfpStream::kMax / extents.Area();
  pool.Run(0, extents.rows, [&](const size_t r, size_t thread) {
    float* HWY_RESTRICT row = raw.Row(r);
    for (size_t c = 0; c < extents.cols; c++) {
      float f = static_cast<float>(c * extents.rows + r) * scale;
      if ((r + c) & 1) f = -f;  // Also generate some negative values.
      row[c] = f;
    }
    Compress(raw.Row(r), raw.Cols(), ws.tls[thread],
             MakeSpan(compressed.Row(r), compressed.Cols()),
             /*packed_ofs=*/0);
  });

  // Arbitrary value, different from 1, must match `GenerateMat`.
  compressed.SetScale(0.6f);
  return compressed;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
