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

#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "util/basics.h"
#include "hwy/base.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/query_compression_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "gemma/query_compression.h"  // includes highway.h
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

void TestCompressQueriesBF16Pointers() {
  constexpr size_t kQkvDim = 64;
  constexpr size_t kNumQueries = 24;
  std::vector<float> input(kQkvDim * kNumQueries);
  for (size_t i = 0; i < kNumQueries; ++i) {
    for (size_t j = 0; j < kQkvDim; ++j) {
      input[i * kQkvDim + j] = 0.01f * (i + 1) / (j + 1);
    }
  }
  std::vector<const float*> queries;
  for (size_t i = 0; i < kNumQueries; ++i) {
    queries.push_back(input.data() + i * kQkvDim);
  }
  std::vector<BF16> output(kQkvDim * kNumQueries);
  CompressQueriesBF16(
      hwy::Span<const float* const>(queries.data(), queries.size()), kQkvDim,
      output.data());
  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_EQ(hwy::ConvertScalarTo<float>(output[i]),
              hwy::ConvertScalarTo<float>(hwy::ConvertScalarTo<BF16>(input[i])))
        << "i=" << i;
  }
}

void TestCompressQueriesContiguousArbitraryDim() {
  constexpr size_t kGuard = 256;  // > 2 * max lanes of any target.
  for (size_t qkv_dim : {1, 7, 24, 40, 63, 64, 80, 129, 256}) {
    for (size_t num_queries : {1, 3}) {
      const size_t n = qkv_dim * num_queries;
      std::vector<float> input(n + kGuard, 123.0f);
      for (size_t i = 0; i < n; ++i) {
        input[i] = 0.01f * static_cast<float>(i % 97) - 0.4f;
      }

      const BF16 bf16_sentinel = hwy::ConvertScalarTo<BF16>(9.5f);
      std::vector<BF16> out_bf16(n + kGuard, bf16_sentinel);
      CompressQueriesBF16Contiguous(input.data(), static_cast<int>(qkv_dim),
                                    num_queries, out_bf16.data());
      for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(
            hwy::ConvertScalarTo<float>(out_bf16[i]),
            hwy::ConvertScalarTo<float>(hwy::ConvertScalarTo<BF16>(input[i])))
            << "qkv_dim=" << qkv_dim << " i=" << i;
      }
      for (size_t i = n; i < n + kGuard; ++i) {
        EXPECT_EQ(hwy::ConvertScalarTo<float>(out_bf16[i]), 9.5f)
            << "BF16 overrun at qkv_dim=" << qkv_dim << " i=" << i;
      }

      constexpr int16_t kI16Sentinel = 0x5A5A;
      std::vector<int16_t> out_i16(n + kGuard, kI16Sentinel);
      std::vector<float> scales(num_queries + 1, -1.0f);
      CompressQueriesInt16Contiguous(input.data(), static_cast<int>(qkv_dim),
                                     num_queries, out_i16.data(),
                                     scales.data());
      for (size_t q = 0; q < num_queries; ++q) {
        for (size_t j = 0; j < qkv_dim; ++j) {
          const size_t i = q * qkv_dim + j;
          EXPECT_NEAR(out_i16[i] * scales[q], input[i], scales[q])
              << "qkv_dim=" << qkv_dim << " i=" << i;
        }
      }
      EXPECT_EQ(scales[num_queries], -1.0f);
      for (size_t i = n; i < n + kGuard; ++i) {
        EXPECT_EQ(out_i16[i], kI16Sentinel)
            << "Int16 overrun at qkv_dim=" << qkv_dim << " i=" << i;
      }
    }
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gcpp {
HWY_BEFORE_TEST(QueryCompressionTest);
HWY_EXPORT_AND_TEST_P(QueryCompressionTest, TestCompressQueriesBF16Pointers);
HWY_EXPORT_AND_TEST_P(QueryCompressionTest,
                      TestCompressQueriesContiguousArbitraryDim);
HWY_AFTER_TEST();

}  // namespace gcpp

#endif
