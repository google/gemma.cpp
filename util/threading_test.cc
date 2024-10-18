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

#include "util/threading.h"

#include <stddef.h>
#include <stdio.h>

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "hwy/base.h"  // HWY_ASSERT

namespace gcpp {
namespace {

using ::testing::ElementsAre;

TEST(ThreadingTest, TestBoundedSlice) {
  const char* name = "test";
  // No args = no limit.
  {
    BoundedSlice slice;
    std::vector<size_t> expected;
    const size_t detected = 10;
    slice.Foreach(name, detected, [&](size_t i) { expected.push_back(i); });
    EXPECT_EQ(10, slice.Num(detected));
    EXPECT_THAT(expected, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
    EXPECT_TRUE(slice.Contains(detected, 0));
    EXPECT_TRUE(slice.Contains(detected, 9));
    EXPECT_FALSE(slice.Contains(detected, 10));
  }

  // One arg: skip first N
  {
    BoundedSlice slice(3);
    std::vector<size_t> expected;
    const size_t detected = 9;
    slice.Foreach(name, detected, [&](size_t i) { expected.push_back(i); });
    EXPECT_EQ(6, slice.Num(detected));
    EXPECT_THAT(expected, ElementsAre(3, 4, 5, 6, 7, 8));
    EXPECT_FALSE(slice.Contains(detected, 2));
    EXPECT_TRUE(slice.Contains(detected, 3));
    EXPECT_TRUE(slice.Contains(detected, 8));
    EXPECT_FALSE(slice.Contains(detected, 9));
  }

  // Both args: skip first N, then use at most M
  {
    BoundedSlice slice(3, 2);
    std::vector<size_t> expected;
    const size_t detected = 9;
    slice.Foreach(name, detected, [&](size_t i) { expected.push_back(i); });
    EXPECT_EQ(2, slice.Num(detected));
    EXPECT_THAT(expected, ElementsAre(3, 4));
    EXPECT_FALSE(slice.Contains(detected, 2));
    EXPECT_TRUE(slice.Contains(detected, 3));
    EXPECT_TRUE(slice.Contains(detected, 4));
    EXPECT_FALSE(slice.Contains(detected, 5));
  }

  // Both args, but `max > detected - skip`: fewer than limit. Note that
  // `skip >= detected` is an error.
  {
    BoundedSlice slice(3, 2);
    std::vector<size_t> expected;
    const size_t detected = 4;
    slice.Foreach(name, detected, [&](size_t i) { expected.push_back(i); });
    EXPECT_EQ(1, slice.Num(detected));
    EXPECT_THAT(expected, ElementsAre(3));
    EXPECT_FALSE(slice.Contains(detected, 2));
    EXPECT_TRUE(slice.Contains(detected, 3));
    EXPECT_FALSE(slice.Contains(detected, 4));
  }
}

TEST(ThreadingTest, TestBoundedTopology) {
  const BoundedSlice all;
  const BoundedSlice one(0, 1);
  // All
  {
    BoundedTopology topology(all, all, all);
    fprintf(stderr, "%s\n", topology.TopologyString());
  }

  // Max one package
  {
    BoundedTopology topology(one, all, all);
    fprintf(stderr, "%s\n", topology.TopologyString());
    ASSERT_EQ(1, topology.NumPackages());
  }

  // Max one cluster
  {
    BoundedTopology topology(all, one, all);
    fprintf(stderr, "%s\n", topology.TopologyString());
    ASSERT_EQ(1, topology.NumClusters(0));
  }
}

}  // namespace
}  // namespace gcpp
