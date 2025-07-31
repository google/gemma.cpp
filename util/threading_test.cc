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

#include <stddef.h>
#include <stdio.h>

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "util/basics.h"
#include "util/threading_context.h"
#include "hwy/aligned_allocator.h"  // Span
#include "hwy/auto_tune.h"
#include "hwy/base.h"  // HWY_ASSERT
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/nanobenchmark.h"  // Unpredictable1
#include "hwy/robust_statistics.h"
#include "hwy/stats.h"
#include "hwy/tests/test_util.h"  // AdjustedReps
#include "hwy/timer.h"

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

TEST(ThreadingTest, TestMaxSizePartition) {
  const IndexRange range(0, 100);
  // Round down
  {
    const IndexRangePartition partition = MaxSizePartition(range, 55, 32);
    HWY_ASSERT(partition.TaskSize() == 32);
    HWY_ASSERT(partition.NumTasks() == 4);
  }
  // Huge `max_size`: single task
  {
    const IndexRangePartition partition = MaxSizePartition(range, 9999, 1);
    HWY_ASSERT(partition.TaskSize() == 100);
    HWY_ASSERT(partition.NumTasks() == 1);
  }
  // Huge `max_size`: `size_multiple` is still respected
  {
    const IndexRangePartition partition = MaxSizePartition(range, 9999, 64);
    HWY_ASSERT(partition.TaskSize() == 64);
    HWY_ASSERT(partition.NumTasks() == 2);
  }
  // `size_multiple` larger than range: ignore multiple
  {
    const IndexRangePartition partition = MaxSizePartition(range, 55, 128);
    HWY_ASSERT(partition.TaskSize() == 55);
    HWY_ASSERT(partition.NumTasks() == 2);
  }
  // `size_multiple` almost as large as range: imbalanced
  {
    const IndexRangePartition partition =
        MaxSizePartition(IndexRange(0, 6), 6, 4);
    HWY_ASSERT(partition.TaskSize() == 4);
    HWY_ASSERT(partition.NumTasks() == 2);
  }
  // Small `max_size`: small tasks
  {
    const IndexRangePartition partition = MaxSizePartition(range, 2, 1);
    HWY_ASSERT(partition.TaskSize() == 2);
    HWY_ASSERT(partition.NumTasks() == 50);
  }
  // Large `max_size`: two tasks with lots of overhang
  {
    const IndexRangePartition partition = MaxSizePartition(range, 98, 1);
    HWY_ASSERT(partition.TaskSize() == 98);
    HWY_ASSERT(partition.NumTasks() == 2);
  }
  // `size_multiple` almost as large as a different, smaller range: imbalanced
  {
    const IndexRangePartition partition =
        MaxSizePartition(IndexRange(0, 6), 6, 4);
    HWY_ASSERT(partition.TaskSize() == 4);
    HWY_ASSERT(partition.NumTasks() == 2);
  }
}

TEST(ThreadingTest, TestStaticPartition) {
  const IndexRange range(0, 100);
  // Round up
  {
    const IndexRangePartition partition = StaticPartition(range, 2, 64);
    HWY_ASSERT(partition.TaskSize() == 64);
    HWY_ASSERT(partition.NumTasks() == 2);
  }
  // No `size_multiple`: division still rounds up
  {
    const IndexRangePartition partition = StaticPartition(range, 3, 1);
    HWY_ASSERT(partition.TaskSize() == 34);
    HWY_ASSERT(partition.NumTasks() == 3);
  }
  // Huge `max_tasks`: one each
  {
    const IndexRangePartition partition = StaticPartition(range, 9999, 1);
    HWY_ASSERT(partition.TaskSize() == 1);
    HWY_ASSERT(partition.NumTasks() == 100);
  }
  // `size_multiple` larger than range: single task
  {
    const IndexRangePartition partition = StaticPartition(range, 2, 128);
    HWY_ASSERT(partition.TaskSize() == 100);
    HWY_ASSERT(partition.NumTasks() == 1);
  }
  // `max_tasks` = 1: single task, even if rounding up would exceed the range
  {
    const IndexRangePartition partition = StaticPartition(range, 1, 8);
    HWY_ASSERT(partition.TaskSize() == 100);
    HWY_ASSERT(partition.NumTasks() == 1);
  }
}

TEST(ThreadingTest, TestParallelizeOneRange) {
  const IndexRange range(0, 10);
  const IndexRangePartition partition = StaticPartition(range, 2, 4);
  hwy::ThreadPool null_pool(0);
  size_t calls = 0;
  ParallelizeOneRange(partition, null_pool,
                      [&](const IndexRange& range, size_t) {
                        if (++calls == 1) {
                          HWY_ASSERT(range.begin() == 0 && range.end() == 8);
                        } else {
                          HWY_ASSERT(range.begin() == 8 && range.end() == 10);
                        }
                      });
  HWY_ASSERT(calls == 2);
}

TEST(ThreadingTest, TestParallelizeTwoRanges) {
  const IndexRangePartition partition1 =
      StaticPartition(IndexRange(0, 10), 2, 4);
  const IndexRangePartition partition2 =
      MaxSizePartition(IndexRange(128, 256), 32, 32);
  HWY_ASSERT(partition2.NumTasks() == 4);
  hwy::ThreadPool null_pool(0);
  {
    size_t calls = 0;
    ParallelizeTwoRanges(
        partition1, partition2, null_pool,
        [&](const IndexRange& range1, const IndexRange& range2, size_t) {
          ++calls;
          HWY_ASSERT(range1.begin() == 0 || range1.begin() == 8);
          HWY_ASSERT(range2.begin() % 32 == 0);
          HWY_ASSERT(range2.Num() % 32 == 0);
        });
    HWY_ASSERT(calls == 2 * 4);
  }

  // Also swap order to test Remainder() logic.
  {
    size_t calls = 0;
    ParallelizeTwoRanges(
        partition2, partition1, null_pool,
        [&](const IndexRange& range2, const IndexRange& range1, size_t) {
          ++calls;
          HWY_ASSERT(range1.begin() == 0 || range1.begin() == 8);
          HWY_ASSERT(range2.begin() % 32 == 0);
          HWY_ASSERT(range2.Num() % 32 == 0);
        });
    HWY_ASSERT(calls == 2 * 4);
  }
}

static constexpr size_t kU64PerThread = HWY_ALIGNMENT / sizeof(size_t);
static uint64_t outputs[hwy::kMaxLogicalProcessors * kU64PerThread];

std::vector<uint64_t> MeasureForkJoin(hwy::ThreadPool& pool) {
  // Governs duration of test; avoid timeout in debug builds.
  const size_t max_reps = hwy::AdjustedReps(50 * 1000);

  std::vector<uint64_t> times;
  times.reserve(max_reps);

  const size_t base = hwy::Unpredictable1();

  const double t0 = hwy::platform::Now();
  for (size_t reps = 0; reps < 1200; ++reps) {
    pool.Run(0, pool.NumWorkers(), [&](uint64_t task, size_t thread) {
      outputs[thread * kU64PerThread] = base + thread;
    });
    hwy::PreventElision(outputs[base]);
    if (pool.AutoTuneComplete()) break;
  }
  const double t1 = hwy::platform::Now();

  if (pool.AutoTuneComplete()) {
    hwy::Span<hwy::CostDistribution> cd = pool.AutoTuneCosts();
    std::vector<double> costs;
    costs.reserve(cd.size());
    double min_cost = hwy::HighestValue<double>();
    for (size_t i = 0; i < cd.size(); ++i) {
      costs.push_back(cd[i].EstimateCost());
      min_cost = HWY_MIN(min_cost, costs.back());
    }
    // Harmonic mean = reciprocal of the arithmetic mean of the reciprocals.
    double sum_recip_speedup = 0.0;
    size_t count_sum = 0;
    for (size_t i = 0; i < costs.size(); ++i) {
      if (costs[i] == min_cost) continue;
      const double speedup = costs[i] / min_cost;
      HWY_ASSERT(speedup > 1.0);
      sum_recip_speedup += 1.0 / speedup;
      ++count_sum;
    }
    const double harm_mean_speedup =
        static_cast<double>(count_sum / sum_recip_speedup);
    fprintf(stderr, "\nAuto-tuning took %f ms; harm. mean speedup: %f.\n",
            (t1 - t0) * 1E3, harm_mean_speedup);
  } else {
    HWY_WARN("Auto-tuning did not complete yet.");
  }

  char cpu100[100];
  static const bool have_stop = hwy::platform::HaveTimerStop(cpu100);
  if (have_stop) {
    for (size_t rep = 0; rep < max_reps; ++rep) {
      const uint64_t t0 = hwy::timer::Start();
      pool.Run(0, pool.NumWorkers(), [&](uint64_t task, size_t thread) {
        outputs[thread * kU64PerThread] = base + thread;
      });
      const uint64_t t1 = hwy::timer::Stop();
      times.push_back(t1 - t0);
    }
  } else {
    for (size_t rep = 0; rep < max_reps; ++rep) {
      const uint64_t t0 = hwy::timer::Start();
      pool.Run(0, pool.NumWorkers(), [&](uint64_t task, size_t thread) {
        outputs[thread * kU64PerThread] = base + thread;
      });
      const uint64_t t1 = hwy::timer::Start();
      times.push_back(t1 - t0);
    }
  }
  return times;
}

TEST(ThreadingTest, BenchJoin) {
  const auto measure = [&](hwy::ThreadPool& pool, bool spin,
                           const char* caption) {
    // Only spin for the duration of the benchmark to avoid wasting energy and
    // interfering with the other pools.
    if (spin) {
      pool.SetWaitMode(hwy::PoolWaitMode::kSpin);
    }
    std::vector<uint64_t> times = MeasureForkJoin(pool);
    // Capture before SetWaitMode changes it.
    const hwy::pool::Config config = pool.config();
    if (spin) {
      pool.SetWaitMode(hwy::PoolWaitMode::kBlock);
    }

    const uint64_t median =
        hwy::robust_statistics::Median(times.data(), times.size());
    const uint64_t mode =
        hwy::robust_statistics::ModeOfSorted(times.data(), times.size());
    // Trim upper half and lower quarter to reduce outliers before Stats.
    const size_t quarter = times.size() / 4;
    times.erase(times.begin() + 2 * quarter, times.end());
    times.erase(times.begin(), times.begin() + quarter);

    hwy::Stats stats;
    for (uint64_t t : times) {
      stats.Notify(static_cast<float>(t));
    }

    fprintf(stderr, "%-20s: %3d: %6.2f %6.2f us %s\n", caption,
            static_cast<int>(pool.NumWorkers()),
            median / hwy::platform::InvariantTicksPerSecond() * 1E6,
            mode / hwy::platform::InvariantTicksPerSecond() * 1E6,
            config.ToString().c_str());
    fprintf(stderr, "%s\n", stats.ToString().c_str());

    // Verify outputs to ensure the measured code is not a no-op.
    for (size_t lp = 0; lp < pool.NumWorkers(); ++lp) {
      HWY_ASSERT(outputs[lp * kU64PerThread] >= 1);
      HWY_ASSERT(outputs[lp * kU64PerThread] <= 1 + pool.NumWorkers());
      for (size_t i = 1; i < kU64PerThread; ++i) {
        HWY_ASSERT(outputs[lp * kU64PerThread + i] == 0);
      }
    }
  };

  ThreadingArgs threading_args;
  ThreadingContext ctx(threading_args);
  NestedPools& pools = ctx.pools;
  // Use last package because the main thread has been pinned to it.
  const size_t pkg_idx = pools.NumPackages() - 1;

  measure(pools.AllPackages(), false, "block packages");
  if (pools.AllClusters(pkg_idx).NumWorkers() > 1) {
    measure(pools.AllClusters(pkg_idx), false, "block clusters");
  }
  measure(pools.Cluster(pkg_idx, 0), false, "block in_cluster");

  if (pools.AllPinned()) {
    const bool kSpin = true;
    measure(pools.AllPackages(), kSpin, "spin packages");
    if (pools.AllClusters(pkg_idx).NumWorkers() > 1) {
      measure(pools.AllClusters(pkg_idx), kSpin, "spin clusters");
    }
    measure(pools.Cluster(pkg_idx, 0), kSpin, "spin in_cluster");
  }
}

}  // namespace
}  // namespace gcpp
