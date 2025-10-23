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

#include "util/threading.h"  // NOT threading_context..
// to ensure there is no deadlock.

#include <stdio.h>

#include <memory>
#include <optional>
#include <vector>

// Placeholder for container detection, do not remove
#include "util/basics.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/contrib/thread_pool/topology.h"

namespace gcpp {

static bool InContainer() {
  return false;  // placeholder for container detection, do not remove
}

PinningPolicy::PinningPolicy(Tristate pin) {
  if (pin == Tristate::kDefault) {
    // Pinning is unreliable inside containers because the hypervisor might
    // periodically change our affinity mask, or other processes might also
    // pin themselves to the same LPs.
    pin = InContainer() ? Tristate::kFalse : Tristate::kTrue;
  }
  want_pin_ = (pin == Tristate::kTrue);
}

// If `pinning.Want()`, tries to pin each worker in `pool` to an LP in
// `cluster`, and calls `pinning.NotifyFailed()` if any fails.
static void MaybePin(const BoundedTopology& topology, size_t cluster_idx,
                     const BoundedTopology::Cluster& cluster,
                     PinningPolicy& pinning, hwy::ThreadPool& pool) {
  static hwy::pool::Caller caller = hwy::ThreadPool::AddCaller("MaybePin");
  const std::vector<size_t> lps = cluster.LPVector();
  HWY_ASSERT(pool.NumWorkers() <= lps.size());
  pool.Run(0, pool.NumWorkers(), caller, [&](uint64_t task, size_t thread) {
    HWY_ASSERT(task == thread);  // each worker has one task

    char buf[16];  // Linux limitation
    const int bytes_written = snprintf(
        buf, sizeof(buf), "P%zu X%02zu C%03d", topology.SkippedPackages(),
        topology.SkippedClusters() + cluster_idx, static_cast<int>(task));
    HWY_ASSERT(bytes_written < static_cast<int>(sizeof(buf)));
    hwy::SetThreadName(buf, 0);  // does not support varargs

    if (HWY_LIKELY(pinning.Want())) {
      if (HWY_UNLIKELY(!hwy::PinThreadToLogicalProcessor(lps[task]))) {
        // Apple does not support pinning, hence do not warn there.
        if (!HWY_OS_APPLE) {
          HWY_WARN("Pinning failed for task %d of %zu to %zu (size %zu)\n",
                   static_cast<int>(task), pool.NumWorkers(), lps[task],
                   lps.size());
        }
        pinning.NotifyFailed();
      }
    }
  });
}

static PoolPtr MakePool(const Allocator& allocator, size_t num_workers,
                        hwy::PoolWorkerMapping mapping,
                        std::optional<size_t> node = std::nullopt) {
  // `ThreadPool` expects the number of threads to create, which is one less
  // than the number of workers, but avoid underflow if zero.
  const size_t num_threads = num_workers == 0 ? 0 : num_workers - 1;
  PoolPtr ptr =
      allocator.AllocClasses<hwy::ThreadPool>(1, num_threads, mapping);
  const size_t bytes =
      hwy::RoundUpTo(sizeof(hwy::ThreadPool), allocator.QuantumBytes());
  if (node.has_value() && allocator.ShouldBind()) {
    allocator.BindMemory(ptr.get(), bytes, node.value());
  }
  return ptr;
}

// Used to divide max_threads and max_workers_per_package across packages and
// clusters. Ensures small upper bounds are respected.
static size_t DivideMaxAcross(const size_t max, const size_t instances) {
  // No limit.
  if (max == 0) return 0;
  // We have enough to distribute.
  if (max >= instances) return max / instances;
  // Use max as the upper bound for each instance because division would return
  // zero, which means 'unlimited'.
  return max;
}

// `max_or_zero` == 0 means no limit.
static inline size_t CapIfNonZero(size_t num, size_t max_or_zero) {
  return (max_or_zero == 0) ? num : HWY_MIN(num, max_or_zero);
}

NestedPools::NestedPools(const BoundedTopology& topology,
                         const Allocator& allocator, size_t max_threads,
                         Tristate pin)
    : pinning_(pin) {
  const size_t num_clusters = topology.NumClusters();
  const size_t cluster_workers_cap = DivideMaxAcross(max_threads, num_clusters);

  // Precompute cluster sizes to ensure we pass the same values to `MakePool`.
  // The max is also used for `all_clusters_mapping`, see below.
  size_t workers_per_cluster[hwy::kMaxClusters] = {};
  size_t all_clusters_node = 0;
  for (size_t cluster_idx = 0; cluster_idx < num_clusters; ++cluster_idx) {
    const BoundedTopology::Cluster& tcluster = topology.GetCluster(cluster_idx);
    workers_per_cluster[cluster_idx] =
        CapIfNonZero(tcluster.NumWorkers(), cluster_workers_cap);
    // Cluster sizes can vary because individual LPs may be disabled. Use the
    // max so that `GlobalIdx` is consistent within and across clusters. It is
    // OK to have holes or gaps in the worker index space.
    max_workers_per_cluster_ =
        HWY_MAX(max_workers_per_cluster_, workers_per_cluster[cluster_idx]);
    all_clusters_node = tcluster.Node();  // arbitrarily use the last node seen
  }

  const hwy::PoolWorkerMapping all_clusters_mapping(hwy::kAllClusters,
                                                    max_workers_per_cluster_);
  all_clusters_ = MakePool(allocator, num_clusters, all_clusters_mapping,
                           all_clusters_node);

  // Pre-allocate because elements are set concurrently.
  clusters_.resize(num_clusters);

  // Parallel so we also pin the calling worker in `all_clusters` to
  // `cluster.lps`.
  static hwy::pool::Caller caller = hwy::ThreadPool::AddCaller("NestedPools");
  all_clusters_->Run(
      0, num_clusters, caller, [&](size_t cluster_idx, size_t thread) {
        HWY_ASSERT(cluster_idx == thread);  // each thread has one task
        const BoundedTopology::Cluster& tcluster =
            topology.GetCluster(cluster_idx);
        clusters_[cluster_idx] = MakePool(
            allocator, workers_per_cluster[cluster_idx],
            hwy::PoolWorkerMapping(cluster_idx, max_workers_per_cluster_),
            tcluster.Node());
        // Pin workers AND the calling thread from `all_clusters_`.
        MaybePin(topology, cluster_idx, tcluster, pinning_,
                 *clusters_[cluster_idx]);
      });
  all_pinned_ = pinning_.AllPinned(&pin_string_);
}

}  // namespace gcpp
