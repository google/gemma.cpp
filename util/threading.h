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

// Shared between various frontends.

#ifndef THIRD_PARTY_GEMMA_CPP_UTIL_THREADING_H_
#define THIRD_PARTY_GEMMA_CPP_UTIL_THREADING_H_

#include <stddef.h>
#include <stdio.h>

#include <algorithm>  // std::sort
#include <memory>
#include <vector>

#include "hwy/base.h"  // HWY_ASSERT
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/contrib/thread_pool/topology.h"

namespace gcpp {

// Owns 'inner' thread pools, one per 'cluster' (CCX or socket), plus an
// 'outer' thread pool with one worker per cluster.
//
// Useful for hierarchical parallelism, which makes sense when there are few
// but large tasks which should be parallelized by workers sharing a cache.
// This also implies lower latency for barrier synchronization of those workers.
class PerClusterPools {
  using LPS = hwy::LogicalProcessorSet;

  static inline std::vector<size_t> CoresInLPS(const LPS& cluster) {
    std::vector<size_t> cores;
    cores.reserve(cluster.Count());
    cluster.Foreach([&cores](size_t idx) { cores.push_back(idx); });
    return cores;
  }

  using CoreBitSets = std::vector<LPS>;

  // Returns empty if detection failed.
  CoreBitSets DetectCoresPerCluster() {
    CoreBitSets clusters;
    if (!have_threading_support_) return clusters;

    // Which processors are not disabled via OS, taskset, or numactl.
    LPS enabled;
    // If we don't know, better to abort rather than risk oversubscribing.
    if (!GetThreadAffinity(enabled)) return clusters;

    hwy::Topology topology;
    if (topology.packages.empty()) return clusters;

    // Merge all clusters into one set, as a stopgap to emulate gemma-inl's
    // prior single pool.
    // TODO: remove once MatMul supports hierarchical parallelism.
    LPS all;

    // For each cluster, add its enabled *cores*.
    for (const hwy::Topology::Package& package : topology.packages) {
      for (const hwy::Topology::Cluster& cluster : package.clusters) {
        cluster.lps.Foreach([&](size_t lp) {
          if (enabled.Get(lp) && topology.lps[lp].smt == 0) {
            all.Set(lp);
          }
        });
      }

      /* code to reinstate:
      for (const hwy::Topology::Cluster& cluster : package.clusters) {
        // Only use enabled *cores*, and only add if not empty.
        cluster.lps.Foreach([&](size_t lp) {
          if (enabled.Get(lp) && topology.lps[lp].smt == 0) {
            all.Set(lp);
          }
        });
        if (lps.Any()) clusters.push_back(lps);
      }
      */
    }
    if (all.Any()) clusters.push_back(all);

    // Sort by descending number of enabled cores, so that we preferentially
    // use the largest clusters.
    std::sort(clusters.begin(), clusters.end(),
              [](const LPS& a, const LPS& b) { return a.Count() > b.Count(); });

    return clusters;
  }

  void SetWaitMode(hwy::PoolWaitMode wait_mode) {
    outer_pool_.SetWaitMode(wait_mode);
    for (auto& inner : inner_pools_) {
      inner->SetWaitMode(wait_mode);
    }
  }

  // The defaults for `AppArgs` `max_clusters` and `num_threads` are zero, which
  // means no limit.
  size_t CapIfNonzero(size_t detected, size_t user_max_or_zero) {
    if (!have_threading_support_) return 0;
    return (user_max_or_zero == 0) ? detected
                                   : HWY_MIN(detected, user_max_or_zero);
  }

 public:
  // PerClusterPools supports spin waits (see StartSpinning below). To prevent
  // drastic slowdowns caused by excessive user-specified thread counts, which
  // result in threads not running on their own core, we only allow for
  // *upper bounds* on the number of clusters and threads. The actual number of
  // clusters and threads are still limited by the detected topology.
  PerClusterPools(size_t max_clusters, size_t max_threads,
                  size_t pin_offset = 0)
      : have_threading_support_(hwy::HaveThreadingSupport()),
        cores_per_cluster_(DetectCoresPerCluster()),
        outer_pool_(CapIfNonzero(cores_per_cluster_.size(), max_clusters)) {
    // Topology detection failed - it currently requires Linux.
    if (cores_per_cluster_.empty()) {
      // Create a single inner pool with up to TotalLogicalProcessors() / 2
      // workers, further limited by `max_threads` if nonzero, and then pin to
      // the first N processors, which are typically on the first socket.
      const size_t num_threads =
          CapIfNonzero(hwy::TotalLogicalProcessors() / 2, max_threads);
      fprintf(stderr, "CPU topology unknown, using %zu threads\n", num_threads);
      inner_pools_.push_back(std::make_unique<hwy::ThreadPool>(num_threads));
      if (num_threads > 1) {
        inner_pools_.back()->Run(0, num_threads,
                                 [pin_offset, num_threads](uint64_t /*task*/,
                                                           size_t thread) {
                                   auto lp =
                                     (thread + pin_offset) % num_threads;
                                   hwy::PinThreadToLogicalProcessor(lp);
                                 });
      }
      return;
    }

    const size_t max_per_inner = max_threads / outer_pool_.NumWorkers();
    for (size_t outer = 0; outer < outer_pool_.NumWorkers(); ++outer) {
      const size_t num_threads =
          CapIfNonzero(cores_per_cluster_[outer].Count(), max_per_inner);
      inner_pools_.push_back(std::make_unique<hwy::ThreadPool>(num_threads));
    }

    // For each inner pool, pin their threads AND the associated outer thread
    // (the one calling inner.Run()) to the enabled cores in the cluster.
    outer_pool_.Run(
        0, outer_pool_.NumWorkers(),
        [this, pin_offset](uint64_t outer, size_t outer_thread) {
          HWY_ASSERT(outer == outer_thread);  // each outer has one task
          hwy::ThreadPool& inner = *inner_pools_[outer];

          const std::vector<size_t> cores =
              CoresInLPS(cores_per_cluster_[outer]);
          // May have been capped by max_threads.
          HWY_ASSERT(inner.NumWorkers() <= cores.size());

          inner.Run(0, inner.NumWorkers(),
                    [pin_offset, &cores](uint64_t task, size_t thread) {
                      HWY_ASSERT(task == thread);  // each inner has one task
                      auto lp = cores[(task + pin_offset) % cores.size()];
                      hwy::PinThreadToLogicalProcessor(lp);
                    });
        });
  }

  // Spinning reduces the latency of barrier synchronization, but wastes lots of
  // energy for long waits, so only do it during generation. This might also be
  // unsafe in virtualized environments because we require threads to be running
  // on their own core and thus responsive to the barrier synchronization.
  void StartSpinning() { SetWaitMode(hwy::PoolWaitMode::kSpin); }
  void StopSpinning() { SetWaitMode(hwy::PoolWaitMode::kBlock); }

  // Bitset of cores, one per cluster, or empty if detection failed. Useful for
  // displaying the topology.
  const CoreBitSets& CoresPerCluster() const { return cores_per_cluster_; }

  hwy::ThreadPool& Outer() { return outer_pool_; }
  hwy::ThreadPool& Inner(size_t outer) {
    HWY_ASSERT(outer < Outer().NumWorkers());
    return *inner_pools_[outer];
  }

 private:
  bool have_threading_support_;
  CoreBitSets cores_per_cluster_;
  hwy::ThreadPool outer_pool_;
  // hwy::ThreadPool is unfortunately not marked as movable, so we have to use
  // unique_ptr.
  std::vector<std::unique_ptr<hwy::ThreadPool>> inner_pools_;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_THREADING_H_
