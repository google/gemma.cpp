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
    slice.ForEach(name, 10, [&](size_t i) { expected.push_back(i); });
    EXPECT_EQ(10, slice.Num(10));
    EXPECT_THAT(expected, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
  }

  // One arg: skip first N
  {
    BoundedSlice slice(3);
    std::vector<size_t> expected;
    slice.ForEach(name, 9, [&](size_t i) { expected.push_back(i); });
    EXPECT_EQ(6, slice.Num(9));
    EXPECT_THAT(expected, ElementsAre(3, 4, 5, 6, 7, 8));
  }

  // Both args: skip first N, then use at most M
  {
    BoundedSlice slice(3, 2);
    std::vector<size_t> expected;
    slice.ForEach(name, 9, [&](size_t i) { expected.push_back(i); });
    EXPECT_EQ(2, slice.Num(9));
    EXPECT_THAT(expected, ElementsAre(3, 4));
  }

  // Both args, but `max > detected - skip`: fewer than limit. Note that
  // `skip >= detected` is an error.
  {
    BoundedSlice slice(3, 2);
    std::vector<size_t> expected;
    slice.ForEach(name, 4, [&](size_t i) { expected.push_back(i); });
    EXPECT_EQ(1, slice.Num(4));
    EXPECT_THAT(expected, ElementsAre(3));
  }
}

TEST(ThreadingTest, TestBoundedTopology) {
  const BoundedSlice all;
  const BoundedSlice one(0, 1);
  // All
  {
    BoundedTopology topology(all, all, all);
    fprintf(stderr, "%s\n", topology.TopologyString());
    ASSERT_NE(0, topology.NumPackages());
    ASSERT_NE(0, topology.NumClusters(0));
  }

  // Max one package
  {
    BoundedTopology topology(one, all, all);
    fprintf(stderr, "%s\n", topology.TopologyString());
    ASSERT_EQ(1, topology.NumPackages());
    ASSERT_NE(0, topology.NumClusters(0));
  }

  // Max one cluster
  {
    BoundedTopology topology(all, one, all);
    fprintf(stderr, "%s\n", topology.TopologyString());
    ASSERT_NE(0, topology.NumPackages());
    ASSERT_EQ(1, topology.NumClusters(0));
  }
}

}  // namespace
}  // namespace gcpp
