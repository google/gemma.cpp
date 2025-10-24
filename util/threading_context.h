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

#ifndef THIRD_PARTY_GEMMA_CPP_UTIL_THREADING_CONTEXT_H_
#define THIRD_PARTY_GEMMA_CPP_UTIL_THREADING_CONTEXT_H_

// Separate component to ensure `threading.cc` does not have access to
// `ThreadingContext`, because that could deadlock.

#include <stddef.h>
#include <stdint.h>

// IWYU pragma: begin_exports
#include "util/allocator.h"
#include "util/args.h"
#include "util/basics.h"  // Tristate
#include "util/threading.h"
#include "util/topology.h"
#include "util/zones.h"
#include "hwy/profiler.h"
// IWYU pragma: end_exports

namespace gcpp {

// Optional arguments for `ThreadingContext` from the command line.
class ThreadingArgs : public ArgsBase<ThreadingArgs> {
 public:
  ThreadingArgs(int argc, char* argv[]) { InitAndParse(argc, argv); }
  ThreadingArgs() { Init(); };

  // For BoundedTopology:
  size_t skip_packages;
  size_t max_packages = 1;  // some users assign 1 to this, hence non-const.
  size_t skip_clusters;
  size_t max_clusters;
  size_t skip_lps;
  size_t max_lps;

  Tristate bind;

  // For NestedPools:
  size_t max_threads;  // divided among the detected clusters
  Tristate pin;        // pin threads?
  Tristate spin;       // use spin waits?

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    // These can be used to partition CPU packages/sockets and their
    // clusters/CCXs across several program instances. The default is to use
    // all available resources on the first package.
    visitor(skip_packages, "skip_packages", size_t{0},
            "Index of the first socket to use; default 0 = unlimited.", 2);
    visitor(skip_clusters, "skip_clusters", size_t{0},
            "Index of the first CCX to use; default 0 = unlimited.", 2);
    visitor(max_clusters, "max_clusters", size_t{0},
            "Max CCXs to use; default 0 = unlimited.", 2);
    // "Logical processors" (LPs). These are used when CPU topology is unknown.
    visitor(skip_lps, "skip_lps", size_t{0},
            "Index of the first LP to use; default 0 = unlimited.", 2);
    visitor(max_lps, "max_lps", size_t{0},
            "Max LPs to use; default 0 = unlimited.", 2);

    // DEPRECATED: superseded by the above fields. If nonzero, `NestedPools`
    // will attempt to create this many threads distributed over the detected
    // topology.
    visitor(max_threads, "num_threads", size_t{0},
            "Max threads to use; default 0 = unlimited.", 2);

    visitor(pin, "pin", Tristate::kDefault,
            "Pin threads? -1 = auto, 0 = no, 1 = yes.", 2);
    visitor(spin, "spin", Tristate::kDefault,
            "Use spin waits? -1 = auto, 0 = no, 1 = yes.", 2);

    visitor(bind, "bind", Tristate::kDefault,
            "Bind memory to sockets? -1 = auto, 0 = no, 1 = yes.", 2);
  }
};

// Owns threads corresponding to a subset of the system's resources. Because
// this is passed to `Gemma::Generate` (via `MatMulEnv`) rather than defined as
// a singleton, we can have multiple concurrent `Generate` calls within the
// same process, each with their own `ThreadingContext`. Because each context
// may pin its threads, it is important that they use distinct packages,
// clusters, or LPs. For example, to use two packages, the first `args` can have
// `skip_packages` = 0 and the second `skip_packages` = 1.
struct ThreadingContext {
  explicit ThreadingContext(const ThreadingArgs& args);

  // Returns a worker index compatible with those from `ParallelFor`, assuming
  // the current thread is running on one thread per cluster, which happens
  // when `ParallelismStrategy` is `kAcrossClusters`.
  size_t Worker(size_t cluster_idx) const {
    return cluster_idx * pools.MaxWorkersPerCluster();
  }

  // Singleton; pass around a reference to reduce overhead.
  hwy::Profiler& profiler;

  ProfilerZones profiler_zones;
  PoolCallers pool_callers;

  // Detects topology, subject to limits imposed by user-specified `args`.
  // For example, if `args.max_clusters` is 1, then `topology.NumClusters()`
  // will be 1 regardless of the actual system topology.
  BoundedTopology topology;

  // Ctor depends on `topology` for per-cluster cache sizes.
  CacheInfo cache_info;

  // Ctor depends on `topology` (for NUMA) and `cache_info` (for step size).
  Allocator allocator;

  // Per-package/cluster/within cluster pools of threads, matching `topology`.
  NestedPools pools;
};

#define GCPP_ZONE(ctx, global_idx, zone_enum) \
  PROFILER_ZONE3(ctx.profiler, global_idx, ctx.profiler_zones.Get(zone_enum))

// Describes the strategy for distributing parallel work across cores.
enum class ParallelismStrategy : uint8_t {
  // Execute using a single-threaded loop on the calling thread. The `worker`
  // index passed to the user's `Func` is unique across clusters.
  kNone,
  // One thread per cluster within the first package. The `worker` index passed
  // to the user's `Func` is a `cluster_idx <= NumClusters()`. Some CPUs may
  // only have a single cluster, hence `Func` should also contain a nested
  // `ParallelFor` with `kWithinCluster`.
  kAcrossClusters,
  // All cores within the cluster identified by `cluster_idx`. The `worker`
  // index passed to the user's `Func` is unique across clusters. Choose this
  // strategy if already within a `ParallelFor` call with `kAcrossClusters`,
  // or latency is more important than memory bandwidth.
  kWithinCluster,
  // Equivalent to `kAcrossClusters` if there are multiple clusters, otherwise
  // `kWithinCluster`. Use for few or lightweight tasks (this only uses a
  // single pool and barrier), or to maximize memory bandwidth availability.
  kFlat,
  // First statically partitions `kAcrossClusters`, then `kWithinCluster`. This
  // utilizes all cores, but has higher fork-join overhead (two barriers); use
  // if there are many or heavy tasks.
  kHierarchical,
};

// Helper functions used to implement `ParallelFor`, also reused in multiple
// places. User code should call `ParallelFor` instead, which accepts the more
// convenient `Callers` enum.
//
// These call `func(task, worker)` for each task in `[0, num_tasks)`.

// NOTE: the worker argument is actually the `cluster_idx`, so that `Func` can
// pass that to `ParallelForWithinCluster`.
template <class Func>
void ParallelForAcrossClusters(size_t num_tasks, ThreadingContext& ctx,
                               hwy::pool::Caller caller, const Func& func) {
  ctx.pools.AllClusters().Run(
      0, num_tasks, caller,
      [&](uint64_t task, size_t cluster_idx) { func(task, cluster_idx); });
}

template <class Func>
void ParallelForWithinCluster(size_t num_tasks, ThreadingContext& ctx,
                              size_t cluster_idx, hwy::pool::Caller caller,
                              const Func& func) {
  const size_t cluster_base = ctx.Worker(cluster_idx);
  ctx.pools.Cluster(cluster_idx)
      .Run(0, num_tasks, caller, [&](uint64_t task, size_t worker) {
        func(task, cluster_base + worker);
      });
}

// Calls `func(range, cluster_idx)`, for passing to `*WithinCluster`.
template <class Func>
void ParallelPartitionAcrossClusters(const IndexRange range,
                                     size_t task_multiple, size_t inner_tasks,
                                     ThreadingContext& ctx,
                                     hwy::pool::Caller caller,
                                     const Func& func) {
  HWY_DASSERT(1 <= inner_tasks && inner_tasks <= 4);
  const IndexRangePartition ranges = StaticPartition(
      range, ctx.pools.NumClusters() * inner_tasks, task_multiple);
  ParallelForAcrossClusters(ranges.NumTasks(), ctx, caller,
                            [&](uint64_t task, size_t cluster_idx) {
                              func(ranges.Range(task), cluster_idx);
                            });
}

// Calls `func(range, worker)`.
template <class Func>
void ParallelPartitionWithinCluster(const IndexRange range,
                                    size_t task_multiple, size_t inner_tasks,
                                    ThreadingContext& ctx, size_t cluster_idx,
                                    hwy::pool::Caller caller,
                                    const Func& func) {
  HWY_DASSERT(1 <= inner_tasks && inner_tasks <= 4);
  const size_t num_workers = ctx.pools.Cluster(cluster_idx).NumWorkers();
  const IndexRangePartition ranges =
      StaticPartition(range, num_workers * inner_tasks, task_multiple);
  ParallelForWithinCluster(
      ranges.NumTasks(), ctx, cluster_idx, caller,
      [&](uint64_t task, size_t worker) { func(ranges.Range(task), worker); });
}

// Parallelizes across clusters, then within each cluster.
template <class Func>
void HierarchicalParallelFor(size_t num_tasks, ThreadingContext& ctx,
                             Callers callers, const Func& func) {
  const hwy::pool::Caller caller = ctx.pool_callers.Get(callers);

  // If at most one task per cluster worker, run on a single cluster to avoid
  // the expensive cross-cluster barrier.
  {
    const size_t cluster_idx = 0;
    const size_t cluster_workers = ctx.pools.Cluster(cluster_idx).NumWorkers();
    if (HWY_UNLIKELY(num_tasks <= cluster_workers)) {
      return ParallelForWithinCluster(num_tasks, ctx, cluster_idx, caller,
                                      func);
    }
  }

  ParallelPartitionAcrossClusters(
      IndexRange(0, num_tasks), /*task_multiple=*/1, /*inner_tasks=*/1, ctx,
      caller, [&](const IndexRange& cluster_range, size_t cluster_idx) {
        ParallelForWithinCluster(cluster_range.Num(), ctx, cluster_idx, caller,
                                 [&](uint64_t i, size_t worker) {
                                   func(cluster_range.begin() + i, worker);
                                 });
      });
}

// Calls `func(task, worker)` for each `task` in `[0, num_tasks)`, with the
// number/type of workers determined by `parallelism`. NOTE: worker is actually
// `cluster_idx` for `kAcrossClusters`. The `cluster_idx` argument is for
// `parallelism == {kWithinCluster, kNone}`, and should be 0 if unknown.
template <class Func>
void ParallelFor(ParallelismStrategy parallelism, size_t num_tasks,
                 ThreadingContext& ctx, size_t cluster_idx, Callers callers,
                 const Func& func) {
  HWY_DASSERT(cluster_idx < ctx.topology.NumClusters());
  if (cluster_idx != 0) {
    // If already running across clusters, only use within-cluster modes.
    HWY_DASSERT(parallelism == ParallelismStrategy::kNone ||
                parallelism == ParallelismStrategy::kWithinCluster);
  }
  const hwy::pool::Caller caller = ctx.pool_callers.Get(callers);

  switch (parallelism) {
    case ParallelismStrategy::kNone: {
      const size_t worker = ctx.Worker(cluster_idx);
      for (size_t task = 0; task < num_tasks; ++task) {
        func(task, worker);
      }
      return;
    }

    case ParallelismStrategy::kAcrossClusters:
      return ParallelForAcrossClusters(
          num_tasks, ctx, caller,
          [&](uint64_t task, size_t cluster_idx) { func(task, cluster_idx); });

    case ParallelismStrategy::kWithinCluster:
      return ParallelForWithinCluster(num_tasks, ctx, cluster_idx, caller,
                                      func);

    case ParallelismStrategy::kFlat:
      // Choose a single pool: the only cluster, or across all clusters
      // (slower synchronization, but more memory bandwidth)
      if (HWY_UNLIKELY(ctx.pools.NumClusters() == 1)) {
        return ParallelForWithinCluster(num_tasks, ctx, cluster_idx, caller,
                                        func);
      }
      return ParallelForAcrossClusters(num_tasks, ctx, caller,
                                       [&](uint64_t task, size_t cluster_idx) {
                                         func(task, ctx.Worker(cluster_idx));
                                       });

    case ParallelismStrategy::kHierarchical:
      return HierarchicalParallelFor(num_tasks, ctx, callers, func);
  }
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_THREADING_CONTEXT_H_
