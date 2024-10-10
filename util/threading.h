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
#include <memory>     // std::unique_ptr
#include <utility>    // std::move
#include <vector>

#include "hwy/base.h"  // HWY_ASSERT
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/contrib/thread_pool/topology.h"

namespace gcpp {

// DEPRECATED, will be replaced by NestedPools once MatMul is updated.
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

  // `user_max_or_zero` == 0 means no limit, which is the case for the defaults
  // of `AppArgs` `max_clusters` and `num_threads`.
  static inline size_t CapIfNonZero(size_t num_workers,
                                    size_t user_max_or_zero) {
    return (user_max_or_zero == 0) ? num_workers
                                   : HWY_MIN(num_workers, user_max_or_zero);
  }

  // Returns the number of threads for `ThreadPool` to create: zero if there is
  // no threading support, otherwise the capped number of workers minus the
  // caller of `ThreadPool::Run`, which is the outer worker or main thread.
  size_t CappedNumThreads(size_t num_workers, size_t user_max_or_zero) const {
    if (!have_threading_support_) return 0;
    const size_t capped_num_workers =
        CapIfNonZero(num_workers, user_max_or_zero);
    // Avoid underflow if number of workers is zero.
    return capped_num_workers == 0 ? 0 : capped_num_workers - 1;
  }

  // Returns the number of workers for the inner pool whose index is `outer`, or
  // 0 to indicate no limit if `max_threads` is zero.
  size_t MaxInnerWorkers(const size_t max_threads, const size_t outer_workers,
                         const size_t outer) const {
    HWY_DASSERT(outer < outer_workers);
    if (max_threads == 0) return 0;  // no limit
    // Round down so we do not exceed the max.
    const size_t max_threads_per_outer = max_threads / outer_workers;
    // First outer pool gets the remainder.
    const size_t remainder = (outer == 0) ? (max_threads % outer_workers) : 0;
    return 1 + max_threads_per_outer + remainder;
  }

 public:
  // Move-only.
  PerClusterPools() = delete;
  PerClusterPools(const PerClusterPools&) = delete;
  PerClusterPools& operator=(const PerClusterPools&) = delete;
  PerClusterPools(PerClusterPools&&) = delete;
  PerClusterPools& operator=(PerClusterPools&&) = delete;

  // PerClusterPools supports spin waits (see StartSpinning below). To prevent
  // drastic slowdowns caused by excessive user-specified thread counts, which
  // result in threads not running on their own core, we only allow for
  // *upper bounds* on the number of clusters and threads. The actual number of
  // clusters and threads are still limited by the detected topology.
  // `max_threads` is the upper bound on threads to distribute among clusters,
  // not including the one outer thread per cluster.
  //
  // `pin` is 0 or 1 to force enable/disable, or -1 to choose automatically.
  PerClusterPools(size_t max_clusters, size_t max_threads, int pin = -1)
      : have_threading_support_(hwy::HaveThreadingSupport()),
        cores_per_cluster_(DetectCoresPerCluster()),
        outer_pool_(CappedNumThreads(cores_per_cluster_.size(), max_clusters)) {
    // Topology detection failed - it currently requires Linux.
    if (cores_per_cluster_.empty()) {
      // Create a single inner pool with up to TotalLogicalProcessors() / 2
      // workers, further limited by `max_threads` if nonzero, and then pin to
      // the first N processors, which are typically on the first socket.
      const size_t num_threads =
          CappedNumThreads(hwy::TotalLogicalProcessors() / 2, max_threads);
      if (pin == -1) pin = num_threads > 8;
      fprintf(stderr, "CPU topology unknown, using %zu threads, pin %d\n",
              num_threads, pin);
      inner_pools_.push_back(std::make_unique<hwy::ThreadPool>(num_threads));
      if (num_threads > 1 && pin) {
        inner_pools_.back()->Run(0, num_threads,
                                 [](uint64_t /*task*/, size_t thread) {
                                   hwy::PinThreadToLogicalProcessor(thread);
                                 });
      }
      return;
    }

    for (size_t outer = 0; outer < outer_pool_.NumWorkers(); ++outer) {
      const size_t max_inner_workers =
          MaxInnerWorkers(max_threads, outer_pool_.NumWorkers(), outer);
      const size_t num_threads = CappedNumThreads(
          cores_per_cluster_[outer].Count(), max_inner_workers);
      inner_pools_.push_back(std::make_unique<hwy::ThreadPool>(num_threads));
    }

    if (pin == -1) {
      pin = (outer_pool_.NumWorkers() * inner_pools_[0]->NumWorkers()) >= 12;
    }

    if (pin) {
      // For each inner pool, pin their threads AND the associated outer thread
      // (the one calling inner.Run()) to the enabled cores in the cluster.
      outer_pool_.Run(
          0, outer_pool_.NumWorkers(),
          [this](uint64_t outer, size_t outer_thread) {
            HWY_ASSERT(outer == outer_thread);  // each outer has one task
            hwy::ThreadPool& inner = *inner_pools_[outer];

            const std::vector<size_t> cores =
                CoresInLPS(cores_per_cluster_[outer]);
            // May have been capped by max_threads.
            HWY_ASSERT(inner.NumWorkers() <= cores.size());

            inner.Run(0, inner.NumWorkers(),
                      [&cores](uint64_t task, size_t thread) {
                        HWY_ASSERT(task == thread);  // each inner has one task
                        hwy::PinThreadToLogicalProcessor(cores[task]);
                      });
          });
    }
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

  // Returns number of logical processors, for allocating per-thread buffers.
  size_t NumLP() const {
    return outer_pool_.NumWorkers() * inner_pools_[0]->NumWorkers();
  }

 private:
  bool have_threading_support_;
  CoreBitSets cores_per_cluster_;
  hwy::ThreadPool outer_pool_;
  // hwy::ThreadPool is unfortunately not marked as movable, so we have to use
  // unique_ptr.
  std::vector<std::unique_ptr<hwy::ThreadPool>> inner_pools_;
};

// A slice of a 1D integer range such as the indices of packages or clusters.
// This allows assigning them to multiple instances of our binary.
struct BoundedSlice {
  // Defaults to "use all detected".
  BoundedSlice(size_t skip = 0, size_t max = 0) : skip(skip), max(max) {}

  // How many to skip, or equivalently, index of the first to use. It is an
  // error if this is >= `detected`, because that would leave none for this
  // instance to use.
  size_t skip;

  // Upper bound on the number to use, or zero if no limit.
  size_t max;

  // STL-style one past the end.
  size_t End(size_t detected) const {
    return (max == 0) ? detected : HWY_MIN(detected, skip + max);
  }

  // Number of elements in the slice.
  size_t Num(size_t detected) const { return End(detected) - skip; }

  template <class Func>
  void ForEach(const char* name, size_t detected, const Func& func) {
    if (skip >= detected) {
      HWY_ABORT("Invalid skip=%zu for %s, detected=%zu", skip, name, detected);
    }
    for (size_t i = skip; i < End(detected); ++i) {
      func(i);
    }
  }
};

// "LP" is a logical processor, a 0-based index passed to the OS.
using LPS = hwy::LogicalProcessorSet;

// Wraps hwy::Topology and only keeps the subset of packages and clusters
// apportioned by BoundedSlice, further limited by the OS affinity mask.
// NOTE: if topology is unknown or the OS affinity is too restrictive, we fall
// back to a single package and cluster.
class BoundedTopology {
  // Sort packages/clusters by descending size so that users who only use one
  // get the largest.
  template <class Group>
  static void SortByDescendingLPs(std::vector<Group>& groups) {
    std::sort(groups.begin(), groups.end(), [](const Group& a, const Group& b) {
      return a.num_lps > b.num_lps;
    });
  }

 public:
  struct Cluster {
    // Simple version when topology is unknown.
    explicit Cluster(size_t num_workers) : num_lps(num_workers) {
      HWY_ASSERT(num_lps != 0);
    }

    Cluster(const std::vector<hwy::Topology::LP>& all_lps, const LPS& enabled,
            size_t package_lp, const hwy::Topology::Cluster& cluster,
            LPS& package_lps) {
      // All first-hyperthread LPs from the cluster that are enabled and not
      // already in use as the package representative.
      cluster.lps.Foreach([&](size_t lp) {
        if (all_lps[lp].smt == 0 && enabled.Get(lp) && lp != package_lp) {
          HWY_ASSERT(!lps.Get(lp));
          lps.Set(lp);
          HWY_ASSERT(!package_lps.Get(lp));
          package_lps.Set(lp);
        }
      });
      num_lps = lps.Count();  // = 0 if all disabled.
    }

    LPS lps;
    size_t num_lps;
    // Set by caller to the first of `lps` if there are multiple clusters in a
    // package.
    size_t cluster_lp = 0;
  };

  struct Package {
    // Simple version when topology is unknown.
    explicit Package(size_t num_workers) {
      package_lp = 0;
      num_lps = num_workers;
      clusters.push_back(Cluster(num_workers));
    }

    Package(size_t package_idx, const hwy::Topology& topology,
            const LPS& enabled, BoundedSlice cluster_slice) {
      const hwy::Topology::Package& package = topology.packages[package_idx];
      package_lp = package.clusters[0].lps.First();
      cluster_slice.ForEach(
          "cluster", package.clusters.size(), [&](size_t cluster_idx) {
            Cluster cluster(topology.lps, enabled, package_lp,
                            package.clusters[cluster_idx], lps);
            if (HWY_LIKELY(cluster.num_lps != 0)) {
              num_lps += cluster.num_lps;  // before std::move
              clusters.push_back(std::move(cluster));
            }
          });

      // Note that it is possible for `clusters` to be empty if its LPs are all
      // disabled. If so, the caller will ignore topology and create a single
      // package and cluster.

      SortByDescendingLPs(clusters);

      // If there are multiple clusters, set their first LP to represent the
      // cluster and mark them as unavailable for its pool.
      if (clusters.size() > 1) {
        for (Cluster& cluster : clusters) {
          cluster.cluster_lp = cluster.lps.First();
          // Nonzero because if lp == 0 were enabled, it would be used as
          // `package_lp` and excluded from `cluster.lps`.
          HWY_ASSERT(cluster.cluster_lp != 0);
          HWY_ASSERT(cluster.cluster_lp != package_lp);
          cluster.lps.Clear(cluster.cluster_lp);
        }
      }
    }

    size_t package_lp;
    LPS lps;
    size_t num_lps = 0;
    std::vector<Cluster> clusters;
  };

  BoundedTopology(BoundedSlice package_slice, BoundedSlice cluster_slice,
                  BoundedSlice lp_slice) {
    const bool have_threading_support = hwy::HaveThreadingSupport();
    LPS enabled_lps;  // LPs not disabled via OS, taskset, or numactl.
    bool missing_cluster = false;

    if (HWY_LIKELY(have_threading_support && !topology_.packages.empty())) {
      (void)GetThreadAffinity(enabled_lps);  // failure = all disabled

      // No effect if topology is unknown or `enabled_lps` is empty.
      package_slice.ForEach(
          "package", topology_.packages.size(), [&](size_t package_idx) {
            Package package(package_idx, topology_, enabled_lps, cluster_slice);
            // Skip if empty - can happen due to `enabled_lps`.
            if (HWY_LIKELY(!package.clusters.empty())) {
              total_lps_ += package.num_lps;  // before std::move
              packages_.push_back(std::move(package));
            }
          });

      for (Package& package : packages_) {
        missing_cluster = package.clusters.empty();
        if (HWY_UNLIKELY(missing_cluster)) {
          fprintf(
              stderr,
              "Warning, found no clusters for package with %zu LPs.\nWe will "
              "ignore topology and assume a single package/cluster.\n",
              package.num_lps);
          break;
        }
      }
    }

    // Topology unknown or any package ended up empty: create a single package
    // with one cluster.
    if (HWY_UNLIKELY(packages_.empty() || missing_cluster)) {
      // We do not bother to detect hyperthreads. Not all CPUs have two per
      // core, so instead of dividing, rely on the user's `lp_slice.max`. This
      // works because Linux groups LPs by HT.
      const size_t num_lps = have_threading_support
                                 ? lp_slice.Num(hwy::TotalLogicalProcessors())
                                 : 1;
      packages_.clear();
      packages_.push_back(Package(num_lps));
      total_lps_ = num_lps;
      snprintf(topology_string_, sizeof(topology_string_), "LPs=%zu", num_lps);
    } else {
      SortByDescendingLPs(packages_);

      const hwy::Topology::Package& tpackage0 = topology_.packages[0];
      HWY_ASSERT(!tpackage0.clusters.empty());
      const hwy::Topology::Cluster& tcluster0 = tpackage0.clusters[0];
      const Package& package0 = GetPackage(0);
      const Cluster& cluster0 = GetCluster(0, 0);
      snprintf(topology_string_, sizeof(topology_string_),
               "%zux%zux%zu, using %zux%zux%zu", topology_.packages.size(),
               tpackage0.clusters.size(), tcluster0.lps.Count(),
               packages_.size(), package0.clusters.size(), cluster0.num_lps);
    }

    HWY_ASSERT(NumPackages() != 0);
    for (size_t package_idx = 0; package_idx < NumPackages(); ++package_idx) {
      HWY_ASSERT(NumClusters(package_idx) != 0);
    }
  }

  const char* TopologyString() const { return topology_string_; }

  size_t NumPackages() const { return packages_.size(); }
  const Package& GetPackage(size_t package_idx) const {
    HWY_ASSERT(package_idx < NumPackages());
    return packages_[package_idx];
  }
  Package& GetPackage(size_t package_idx) {
    HWY_ASSERT(package_idx < NumPackages());
    return packages_[package_idx];
  }

  size_t NumClusters(size_t package_idx) const {
    return GetPackage(package_idx).clusters.size();
  }
  const Cluster& GetCluster(size_t package_idx, size_t cluster_idx) const {
    const Package& package = GetPackage(package_idx);
    HWY_ASSERT(cluster_idx < package.clusters.size());
    return package.clusters[cluster_idx];
  }
  Cluster& GetCluster(size_t package_idx, size_t cluster_idx) {
    Package& package = GetPackage(package_idx);
    HWY_ASSERT(cluster_idx < package.clusters.size());
    return package.clusters[cluster_idx];
  }

  // Returns number of logical processors, for allocating per-thread buffers.
  size_t NumLP() const { return total_lps_; }

 private:
  hwy::Topology topology_;
  size_t total_lps_ = 0;
  std::vector<Package> packages_;
  char topology_string_[96];
};

// Creates a hierarchy of thread pools according to BoundedTopology: one with a
// thread per enabled package; for each of those, one with a thread per enabled
// cluster (CCX/shared L3), and for each of those, the remaining enabled cores
// in that cluster. The cores representing each package and cluster are not
// included in the per-cluster pool because we support spin-waiting, hence
// there should be at most one thread per HW core.
//
// Useful when there are tasks which should be parallelized by workers sharing a
// cache, or on the same NUMA node. In both cases, individual pools have lower
// barrier synchronization latency than one large pool. However, to utilize all
// cores, call sites will have to use nested parallel-for loops.
class NestedPools {
 public:
  // Neither move nor copy.
  NestedPools() = delete;
  NestedPools(const NestedPools&) = delete;
  NestedPools& operator=(const NestedPools&) = delete;
  NestedPools(NestedPools&&) = delete;
  NestedPools& operator=(NestedPools&&) = delete;

  // `max_threads` is the maximum number of threads to divide among all
  // clusters. It does not include the package and cluster representatives.
  // This is more intuitive than a per-cluster limit for users who may not be
  // aware of the CPU topology.
  //
  // To ensure we do not create more threads than there are HW cores, which
  // would cause huge slowdowns when spinning, `BoundedSlice` imposes upper
  // bounds on the number of detected packages and clusters rather than
  // defining an exact amount.
  //
  // `pin` is 0 or 1 to force enable/disable, or -1 to choose automatically.
  NestedPools(size_t max_threads, int pin = -1,
              BoundedSlice package_slice = BoundedSlice(),
              BoundedSlice cluster_slice = BoundedSlice(),
              BoundedSlice lp_slice = BoundedSlice())
      : topology_(package_slice, cluster_slice, lp_slice) {
    if (pin == -1) pin = topology_.NumLP() >= 12;

    packages_.resize(topology_.NumPackages());
    all_packages_ = MakePool(packages_.size());
    const size_t max_workers_per_package = max_threads / packages_.size();
    // Parallel to ensure we also pin the calling (main) thread.
    all_packages_->Run(
        0, all_packages_->NumWorkers(),
        [&](uint64_t package_idx, size_t thread) {
          HWY_ASSERT(package_idx == thread);  // each thread has one task
          packages_[package_idx] = Package(
              topology_, package_idx, max_workers_per_package, pin, lp_slice);
        });
  }

  // Spinning reduces the latency of barrier synchronization, but wastes lots
  // of energy for long waits, so only do it during generation. This might
  // also be unsafe in virtualized environments because we require threads to
  // be running on their own core and thus responsive to the barrier
  // synchronization.
  void StartSpinning() { SetWaitMode(hwy::PoolWaitMode::kSpin); }
  void StopSpinning() { SetWaitMode(hwy::PoolWaitMode::kBlock); }

  hwy::ThreadPool& AllPackages() { return *all_packages_; }
  hwy::ThreadPool& AllClusters(size_t package_idx) {
    HWY_ASSERT(package_idx < AllPackages().NumWorkers());
    return *packages_[package_idx].all_clusters;
  }
  hwy::ThreadPool& Cluster(size_t package_idx, size_t cluster_idx) {
    HWY_ASSERT(cluster_idx < AllClusters(package_idx).NumWorkers());
    return *packages_[package_idx].clusters[cluster_idx];
  }

  const char* TopologyString() const { return topology_.TopologyString(); }

  // Returns number of logical processors, for allocating per-thread buffers.
  size_t NumLP() const { return topology_.NumLP(); }

 private:
  // `max_or_zero` == 0 means no limit.
  static inline size_t CapIfNonZero(size_t num, size_t max_or_zero) {
    return (max_or_zero == 0) ? num : HWY_MIN(num, max_or_zero);
  }

  // We want vectors of hwy::ThreadPool, which is unfortunately not movable,
  // hence we wrap them in unique_ptr.
  using PoolPtr = std::unique_ptr<hwy::ThreadPool>;

  static PoolPtr MakePool(size_t num_workers) {
    // `ThreadPool` expects the number of threads to create, which is one less
    // than the number of workers, but avoid underflow if zero.
    const size_t num_threads = num_workers == 0 ? 0 : num_workers - 1;
    return std::make_unique<hwy::ThreadPool>(num_threads);
  }

  class Package {
    static PoolPtr CreateClusterPool(const BoundedTopology::Cluster& cluster,
                                     size_t max_cluster_workers, int pin,
                                     BoundedSlice lp_slice) {
      PoolPtr pool =
          MakePool(CapIfNonZero(cluster.num_lps, max_cluster_workers));

      if (!pin) return pool;
      // Else: pin all new threads AND the calling thread from `all_clusters`.

      // We know the topology: pin to this cluster's cores, including the
      // calling thread from `all_clusters`.
      if (cluster.lps.Any()) {
        std::vector<size_t> lps;
        lps.reserve(cluster.num_lps);
        cluster.lps.Foreach([&lps](size_t lp) { lps.push_back(lp); });

        pool->Run(0, pool->NumWorkers(), [&lps](uint64_t task, size_t thread) {
          HWY_ASSERT(task == thread);  // each worker has one task
          hwy::PinThreadToLogicalProcessor(lps[task]);
        });
      } else {
        // Pin to consecutive LPs.
        pool->Run(0, pool->NumWorkers(),
                  [lp_slice](uint64_t task, size_t thread) {
                    HWY_ASSERT(task == thread);  // each worker has one task
                    hwy::PinThreadToLogicalProcessor(lp_slice.skip + thread);
                  });
      }
      return pool;
    }

   public:
    Package() = default;  // for vector
    Package(const BoundedTopology& topology, size_t package_idx,
            size_t max_workers_per_package, int pin, BoundedSlice lp_slice) {
      clusters.resize(topology.NumClusters(package_idx));
      const size_t max_workers_per_cluster =
          max_workers_per_package / clusters.size();

      all_clusters = MakePool(clusters.size());
      // Parallel so we also pin the calling thread from `all_packages_`.
      all_clusters->Run(
          0, all_clusters->NumWorkers(),
          [&](size_t cluster_idx, size_t thread) {
            HWY_ASSERT(cluster_idx == thread);  // each thread has one task
            const BoundedTopology::Cluster& cluster =
                topology.GetCluster(package_idx, cluster_idx);
            clusters[cluster_idx] = CreateClusterPool(
                cluster, max_workers_per_cluster, pin, lp_slice);
          });
    }

    std::vector<PoolPtr> clusters;
    PoolPtr all_clusters;
  };

  void SetWaitMode(hwy::PoolWaitMode wait_mode) {
    all_packages_->SetWaitMode(wait_mode);
    for (Package& package : packages_) {
      package.all_clusters->SetWaitMode(wait_mode);
      for (PoolPtr& cluster : package.clusters) {
        cluster->SetWaitMode(wait_mode);
      }
    }
  }

  BoundedTopology topology_;

  std::vector<Package> packages_;
  PoolPtr all_packages_;
};

static inline NestedPools CreateSinglePool(size_t max_threads, int pin = -1) {
  const BoundedSlice one(0, 1);
  return NestedPools(max_threads, pin, one, one);
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_THREADING_H_
