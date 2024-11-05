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

#include "compression/blob_store.h"

#include <stdio.h>

#include <algorithm>
#include <array>

#include "compression/io.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"  // HWY_ASSERT_EQ

namespace gcpp {
namespace {

#if !HWY_TEST_STANDALONE
class BlobStoreTest : public testing::Test {};
#endif

#if !HWY_OS_WIN
TEST(BlobStoreTest, TestReadWrite) {
  static const std::array<float, 4> kOriginalData = {-1, 0, 3.14159, 2.71828};

  // mkstemp will modify path_str so it holds a newly-created temporary file.
  char path_str[] = "/tmp/blob_store_test.sbs-XXXXXX";
  const int fd = mkstemp(path_str);
  HWY_ASSERT(fd > 0);

  hwy::ThreadPool pool(4);
  const Path path(path_str);
  std::array<float, 4> buffer = kOriginalData;

  const hwy::uint128_t keyA = MakeKey("0123456789abcdef");
  const hwy::uint128_t keyB = MakeKey("q");
  BlobWriter writer;
  writer.Add(keyA, "DATA", 5);
  writer.Add(keyB, buffer.data(), sizeof(buffer));
  HWY_ASSERT_EQ(writer.WriteAll(pool, path), 0);
  HWY_ASSERT_ARRAY_EQ(kOriginalData.data(), buffer.data(), buffer.size());

  std::fill(buffer.begin(), buffer.end(), 0);
  BlobReader reader;
  HWY_ASSERT_EQ(reader.Open(path), 0);
  HWY_ASSERT_EQ(reader.BlobSize(keyA), 5);
  HWY_ASSERT_EQ(reader.BlobSize(keyB), sizeof(buffer));

  HWY_ASSERT_EQ(reader.Enqueue(keyB, buffer.data(), sizeof(buffer)), 0);
  HWY_ASSERT_EQ(reader.ReadAll(pool), 0);
  HWY_ASSERT_ARRAY_EQ(kOriginalData.data(), buffer.data(), buffer.size());

  {
    std::array<char, 5> buffer;
    HWY_ASSERT(reader.ReadOne(keyA, buffer.data(), 1) != 0);
    HWY_ASSERT_EQ(reader.ReadOne(keyA, buffer.data(), 5), 0);
    HWY_ASSERT_STRING_EQ("DATA", buffer.data());
  }

  close(fd);
  unlink(path_str);
}
#endif

}  // namespace
}  // namespace gcpp

HWY_TEST_MAIN();
