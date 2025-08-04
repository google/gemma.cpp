// Copyright 2024 Google LLC
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

#include "io/blob_store.h"

#include <stdio.h>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "io/io.h"
#include "util/basics.h"
#include "util/threading_context.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"  // HWY_ASSERT_EQ

namespace gcpp {
namespace {

#if !HWY_TEST_STANDALONE
class BlobStoreTest : public testing::Test {};
#endif

TEST(BlobStoreTest, TestReadWrite) {
  ThreadingArgs threading_args;
  ThreadingContext ctx(threading_args);
  hwy::ThreadPool& pool = ctx.pools.Pool();

  static const std::array<float, 4> kOriginalData = {-1, 0, 3.14159, 2.71828};

  // mkstemp will modify path_str so it holds a newly-created temporary file.
  char path_str[] = "/tmp/blob_store_test.sbs-XXXXXX";
  const int fd = mkstemp(path_str);
  HWY_ASSERT(fd > 0);

  const Path path(path_str);
  std::array<float, 4> buffer = kOriginalData;

  const std::string keyA("0123456789abcdef");  // max 16 characters
  const std::string keyB("q");
  BlobWriter writer(path, pool);
  writer.Add(keyA, "DATA", 5);
  writer.Add(keyB, buffer.data(), sizeof(buffer));
  writer.Finalize();
  HWY_ASSERT_ARRAY_EQ(kOriginalData.data(), buffer.data(), buffer.size());

  std::fill(buffer.begin(), buffer.end(), 0);

  const BlobReader reader(path);

  HWY_ASSERT_EQ(reader.Keys().size(), 2);
  HWY_ASSERT_STRING_EQ(reader.Keys()[0].c_str(), keyA.c_str());
  HWY_ASSERT_STRING_EQ(reader.Keys()[1].c_str(), keyB.c_str());

  const BlobRange* range = reader.Find(keyA);
  HWY_ASSERT(range);
  const uint64_t offsetA = range->offset;
  HWY_ASSERT_EQ(offsetA, 256);
  HWY_ASSERT_EQ(range->bytes, 5);
  range = reader.Find(keyB);
  HWY_ASSERT(range);
  const uint64_t offsetB = range->offset;
  HWY_ASSERT_EQ(offsetB, offsetA + 256);
  HWY_ASSERT_EQ(range->bytes, sizeof(buffer));

  HWY_ASSERT(
      reader.CallWithSpan<char>(keyA, [](const hwy::Span<const char> span) {
        HWY_ASSERT_EQ(span.size(), 5);
        HWY_ASSERT_STRING_EQ("DATA", span.data());
      }));
  HWY_ASSERT(
      reader.CallWithSpan<float>(keyB, [](const hwy::Span<const float> span) {
        HWY_ASSERT_EQ(span.size(), 4);
        HWY_ASSERT_ARRAY_EQ(kOriginalData.data(), span.data(), span.size());
      }));

  close(fd);
  unlink(path_str);
}

// Ensures padding works for any number of random-sized blobs.
TEST(BlobStoreTest, TestNumBlobs) {
  ThreadingArgs threading_args;
  ThreadingContext ctx(threading_args);
  hwy::ThreadPool& pool = ctx.pools.Pool();
  hwy::RandomState rng;

  for (size_t num_blobs = 1; num_blobs <= 512; ++num_blobs) {
    // mkstemp will modify path_str so it holds a newly-created temporary file.
    char path_str[] = "/tmp/blob_store_test2.sbs-XXXXXX";
    const int fd = mkstemp(path_str);
    HWY_ASSERT(fd > 0);
    const Path path(path_str);

    BlobWriter writer(path, pool);
    std::vector<std::string> keys;
    keys.reserve(num_blobs);
    std::vector<std::vector<uint8_t>> blobs;
    blobs.reserve(num_blobs);
    for (size_t i = 0; i < num_blobs; ++i) {
      keys.push_back(std::to_string(i));
      // Smaller blobs when there are many, to speed up the test.
      const size_t mask = num_blobs > 1000 ? 1023 : 8191;
      // Never zero, but may be one byte, which we special-case.
      blobs.emplace_back((size_t{hwy::Random32(&rng)} & mask) + 1);
      std::vector<uint8_t>& blob = blobs.back();
      blob[0] = static_cast<uint8_t>(i & 255);
      if (blob.size() != 1) {
        blob.back() = static_cast<uint8_t>(i >> 8);
      }
      writer.Add(keys.back(), blob.data(), blob.size());
    }
    HWY_ASSERT(keys.size() == num_blobs);
    HWY_ASSERT(blobs.size() == num_blobs);
    writer.Finalize();

    BlobReader reader(path);
    HWY_ASSERT_EQ(reader.Keys().size(), num_blobs);
    pool.Run(0, num_blobs, [&](uint64_t i, size_t /*thread*/) {
      HWY_ASSERT_STRING_EQ(reader.Keys()[i].c_str(), std::to_string(i).c_str());
      const BlobRange* range = reader.Find(keys[i]);
      HWY_ASSERT(range);
      HWY_ASSERT_EQ(blobs[i].size(), range->bytes);
      HWY_ASSERT(reader.CallWithSpan<uint8_t>(
          keys[i], [path_str, num_blobs, i, range,
                    &blobs](const hwy::Span<const uint8_t> span) {
            HWY_ASSERT_EQ(blobs[i].size(), span.size());
            const bool match1 = span[0] == static_cast<uint8_t>(i & 255);
            // If size == 1, we don't have a second byte to check.
            const bool match2 =
                span.size() == 1 ||
                span[span.size() - 1] == static_cast<uint8_t>(i >> 8);
            if (!match1 || !match2) {
              HWY_ABORT("%s num_blobs %zu blob %zu offset %zu is corrupted.",
                        path_str, num_blobs, i, range->offset);
            }
          }));
    });

    close(fd);
    unlink(path_str);
  }
}

}  // namespace
}  // namespace gcpp

HWY_TEST_MAIN();
