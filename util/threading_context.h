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
#include "util/basics.h"  // Tristate, kMaxPackages
#include "util/threading.h"
#include "util/topology.h"
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
  size_t max_packages;
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
    // all available resources on one package. Note that `kMaxPackages` is an
    // upper bound on `max_packages`.
    visitor(skip_packages, "skip_packages", size_t{0},
            "Index of the first socket to use; default 0 = unlimited.", 2);
    visitor(max_packages, "max_packages", size_t{1},
            "Max sockets to use; default = 1, 0 = unlimited.", 2);
    HWY_ASSERT(max_packages <= kMaxPackages);
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

  // Singleton; pass around a reference to reduce overhead.
  hwy::Profiler& profiler;

  // Detects topology, subject to limits imposed by user-specified `args`.
  // For example, if `args.max_packages` is 1, then `topology.NumPackages()`
  // will be 1 regardless of the actual system topology.
  BoundedTopology topology;

  // Ctor depends on `topology` for deciding whether to enable NUMA.
  Allocator allocator;

  // Per-package/cluster/within cluster pools of threads, matching `topology`.
  NestedPools pools;
};

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

// Calls `func(task, worker)` for each `task` in `[0, num_tasks)`, with the
// number/type of workers determined by `parallelism`. `cluster_idx` is for
// `parallelism == kWithinCluster`, and should be 0 if unknown.
template <class Func>
void ParallelFor(ParallelismStrategy parallelism, size_t num_tasks,
                 ThreadingContext& ctx, size_t cluster_idx, const Func& func) {
  HWY_DASSERT(ctx.topology.NumPackages() == 1);
  const size_t pkg_idx = 0;

  HWY_DASSERT(cluster_idx < ctx.topology.NumClusters(pkg_idx));
  if (cluster_idx != 0) {
    // If already running across clusters, only use within-cluster modes.
    HWY_DASSERT(parallelism == ParallelismStrategy::kNone ||
                parallelism == ParallelismStrategy::kWithinCluster);
  }

  switch (parallelism) {
    case ParallelismStrategy::kNone: {
      const size_t worker = cluster_idx * ctx.pools.MaxWorkersPerCluster();
      for (size_t task = 0; task < num_tasks; ++task) {
        func(task, worker);
      }
      return;
    }

    case ParallelismStrategy::kAcrossClusters:
      return ctx.pools.AllClusters(pkg_idx).Run(
          0, num_tasks,
          [&](uint64_t task, size_t cluster_idx) { func(task, cluster_idx); });

    case ParallelismStrategy::kWithinCluster: {
      // Ensure the worker argument is unique across clusters, because it is
      // used for TLS indexing for example in profiler.h.
      const size_t base = cluster_idx * ctx.pools.MaxWorkersPerCluster();
      return ctx.pools.Cluster(pkg_idx, cluster_idx)
          .Run(0, num_tasks, [&](uint64_t task, size_t worker) {
            func(task, base + worker);
          });
    }

    case ParallelismStrategy::kFlat: {
      // Check for single cluster; if not, we must compute `cluster_base` for
      // consistent and non-overlapping worker indices.
      hwy::ThreadPool& all_clusters = ctx.pools.AllClusters(pkg_idx);
      const size_t num_clusters = all_clusters.NumWorkers();
      if (num_clusters == 1) {
        return ctx.pools.Cluster(pkg_idx, cluster_idx)
            .Run(0, num_tasks,
                 [&](uint64_t task, size_t worker) { func(task, worker); });
      }

      return ctx.pools.AllClusters(pkg_idx).Run(
          0, num_tasks, [&](uint64_t task, size_t cluster_idx) {
            const size_t worker =
                cluster_idx * ctx.pools.MaxWorkersPerCluster();
            func(task, worker);
          });
    }

    case ParallelismStrategy::kHierarchical:
      return HierarchicalParallelFor(num_tasks, ctx.pools, func);
  }
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_THREADING_CONTEXT_H_
