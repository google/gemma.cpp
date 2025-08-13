// Copyright 2025 Google LLC
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

#include "util/threading_context.h"

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "hwy/aligned_allocator.h"
#include "hwy/profiler.h"
#include "hwy/tests/test_util.h"  // RandomState

namespace gcpp {

// Invokes `pool.Run` with varying task counts until auto-tuning completes, or
// an upper bound just in case.
static void TunePool(hwy::ThreadPool& pool) {
  const size_t num_workers = pool.NumWorkers();
  // pool.Run would just be a serial loop without auto-tuning, so skip.
  if (num_workers == 1) return;

  // Random shuffle of task counts to defeat branch prediction.
  const size_t num_tasks[4] = {HWY_MAX(1, num_workers / 2), num_workers * 1,
                               num_workers * 5, num_workers * 20};

  // Count tasks executed to ensure workers aren't optimized out. One per
  // cache line to avoid false sharing.
  const size_t kSizePerLine = HWY_ALIGNMENT / sizeof(size_t);

  std::vector<size_t> counters(num_workers * kSizePerLine);
  size_t prev_total = 0;  // avoids having to reset counters.

  hwy::RandomState rng;
  for (size_t rep = 0; rep < 500; ++rep) {
    if (HWY_UNLIKELY(pool.AutoTuneComplete())) {
      break;
    }

    const uint64_t r = hwy::Random64(&rng);
    const size_t begin = r >> 2;
    const size_t end = begin + num_tasks[r & 3];

    pool.Run(begin, end, [&](uint64_t task, size_t thread) {
      HWY_ASSERT(begin <= task && task < end);
      HWY_ASSERT(thread < num_workers);
      counters[thread * kSizePerLine]++;
    });

    // Reduce count and ensure it matches the expected number of tasks.
    size_t total = 0;
    for (size_t i = 0; i < num_workers; ++i) {
      total += counters[i * kSizePerLine];
    }
    const size_t expected = end - begin;
    HWY_ASSERT(total == prev_total + expected);
    prev_total += expected;
  }
}

ThreadingContext::ThreadingContext(const ThreadingArgs& args)
    : profiler(hwy::Profiler::Get()),
      topology(BoundedSlice(args.skip_packages, args.max_packages),
               BoundedSlice(args.skip_clusters, args.max_clusters),
               BoundedSlice(args.skip_lps, args.max_lps)),
      allocator(topology, args.bind != Tristate::kFalse),
      pools(topology, allocator, args.max_threads, args.pin) {
  PROFILER_ZONE("Startup.ThreadingContext autotune");
  TunePool(pools.AllPackages());
  for (size_t pkg_idx = 0; pkg_idx < pools.NumPackages(); ++pkg_idx) {
    hwy::ThreadPool& clusters = pools.AllClusters(pkg_idx);
    TunePool(clusters);

    // Run in parallel because Turin CPUs have 16, and in real usage, we often
    // run all at the same time.
    clusters.Run(0, clusters.NumWorkers(),
                 [&](uint64_t cluster_idx, size_t /*thread*/) {
                   TunePool(pools.Cluster(pkg_idx, cluster_idx));
                 });
  }
}

}  // namespace gcpp
