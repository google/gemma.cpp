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

// A slice of a 1D integer range such as the indices of packages or clusters.
// This allows assigning them to multiple instances of our binary.
class BoundedSlice {
 public:
  // Defaults to "use all detected".
  BoundedSlice(size_t skip = 0, size_t max = 0) : skip_(skip), max_(max) {}

  size_t Begin() const { return skip_; }

  // STL-style one past the end.
  size_t End(size_t detected) const {
    return (max_ == 0) ? detected : HWY_MIN(detected, skip_ + max_);
  }

  // Number of elements in the slice.
  size_t Num(size_t detected) const { return End(detected) - Begin(); }

  bool Contains(size_t detected, size_t idx) const {
    return Begin() <= idx && idx < End(detected);
  }

  template <class Func>
  void Foreach(const char* name, size_t detected, const Func& func) {
    if (Begin() >= detected) {
      HWY_ABORT("Invalid skip=%zu for %s, detected=%zu", skip_, name, detected);
    }
    for (size_t i = Begin(); i < End(detected); ++i) {
      func(i);
    }
  }

 private:
  // How many to skip, or equivalently, index of the first to use. It is an
  // error if this is >= `detected`, because that would leave none for this
  // instance to use.
  size_t skip_;

  // Upper bound on the number to use, or zero if no limit.
  size_t max_;
};

// "LP" is a logical processor, a 0-based index passed to the OS.
using LPS = hwy::LogicalProcessorSet;

// Wraps hwy::Topology and only keeps the subset of packages and clusters
// apportioned by BoundedSlice, further limited by the OS affinity mask.
// NOTE: if topology is unknown or the OS affinity is too restrictive, we fall
// back to a single package and cluster.
class BoundedTopology {
 public:
  BoundedTopology(BoundedSlice package_slice, BoundedSlice cluster_slice,
                  BoundedSlice lp_slice) {
    // Regardless of topology, ignore LPs disabled via OS, taskset, or numactl.
    LPS enabled_lps;
    if (HWY_UNLIKELY(!GetThreadAffinity(enabled_lps))) {
      const size_t num_lps = hwy::TotalLogicalProcessors();
      fprintf(
          stderr,
          "Warning, unknown OS affinity, considering all %zu LPs enabled\n.",
          num_lps);
      for (size_t lp = 0; lp < hwy::TotalLogicalProcessors(); ++lp) {
        enabled_lps.Set(lp);
      }
    }

    // Without threading support, only keep the first enabled LP; it might still
    // make sense to pin the main thread.
    if (HWY_UNLIKELY(!hwy::HaveThreadingSupport())) {
      HWY_ASSERT(enabled_lps.Any());
      const size_t lp = enabled_lps.First();
      enabled_lps = LPS();
      enabled_lps.Set(lp);
    }

    if (HWY_LIKELY(!topology_.packages.empty())) {
      InitFromTopology(enabled_lps, package_slice, cluster_slice);
    }

    // Topology unknown or no packages with enabled LPs: create a single
    // package with one cluster, and one node.
    if (HWY_UNLIKELY(NumPackages() == 0)) {
      InitFromSlice(enabled_lps, lp_slice);
    }

    HWY_ASSERT(NumPackages() != 0 && NumClusters(0) != 0 && NumNodes() != 0);
  }

  size_t NumPackages() const { return packages_.size(); }
  const char* TopologyString() const { return topology_string_; }
  size_t NumNodes() const { return nodes_.Count(); }

  class Cluster {
   public:
    // Topology is unknown, rely on OS affinity and user-specified slice.
    Cluster(const LPS& enabled_lps, BoundedSlice lp_slice) {
      // Interpret `lp_slice` as a slice of the 1-bits of `enabled_lps`, so
      // we honor both the OS affinity and the user-specified slice. Note that
      // this can be used to exclude hyperthreads because Linux groups LPs by
      // sibling index. For example, the first `num_cores` are not siblings.
      const size_t detected = enabled_lps.Count();
      size_t enabled_idx = 0;
      enabled_lps.Foreach([&](size_t lp) {
        if (lp_slice.Contains(detected, enabled_idx++)) {
          AddLP(lp);
        }
      });

      // lp_slice can only reduce the number of `enabled_lps`, and not below 1.
      HWY_ASSERT(num_workers_ != 0);
    }

    Cluster(const LPS& enabled_lps,
            const std::vector<hwy::Topology::LP>& all_lps,
            const hwy::Topology::Cluster& tcluster) {
      bool is_first_lp = true;

      tcluster.lps.Foreach([&](size_t lp) {
        // Skip if not first-hyperthread or disabled.
        if (all_lps[lp].smt != 0 || !enabled_lps.Get(lp)) return;

        AddLP(lp);

        // Set `node` once, and ensure subsequent nodes match - we assume there
        // is only one NUMA node per cluster.
        const size_t lp_node = static_cast<size_t>(all_lps[lp].node);
        if (is_first_lp) {
          is_first_lp = false;
          node_ = lp_node;
        } else {
          static bool warned = false;
          if (lp_node != node_ && !warned) {
            warned = true;
            fprintf(stderr,
                    "WARNING: lp %zu on node %zu != cluster node %zu.\n", lp,
                    lp_node, node_);
          }
        }
      });
    }

    // For SortByDescendingSize.
    size_t Size() const { return num_workers_; }

    // Returns vector with all enabled LPs, used for pinning.
    std::vector<size_t> LPVector() const {
      std::vector<size_t> lps;
      lps.reserve(lps_.Count());
      lps_.Foreach([&lps](size_t lp) { lps.push_back(lp); });
      return lps;
    }

    size_t Node() const { return node_; }

   private:
    void AddLP(size_t lp) {
      HWY_ASSERT(!lps_.Get(lp));  // Foreach ensures uniqueness
      lps_.Set(lp);
      ++num_workers_;
    }

    // Enabled LPs; if topology is known, only the ones in this cluster.
    LPS lps_;
    // How many workers in the per-cluster pool. If 0, this Cluster is removed.
    size_t num_workers_ = 0;
    // NUMA node, set from hwy::Topology::LP::node.
    size_t node_ = 0;
  };  // Cluster

  size_t NumClusters(size_t package_idx) const {
    HWY_ASSERT(package_idx < NumPackages());
    return packages_[package_idx].clusters.size();
  }
  const Cluster& GetCluster(size_t package_idx, size_t cluster_idx) const {
    HWY_ASSERT(package_idx < NumPackages());
    const Package& package = packages_[package_idx];
    HWY_ASSERT(cluster_idx < package.clusters.size());
    return package.clusters[cluster_idx];
  }
  Cluster& GetCluster(size_t package_idx, size_t cluster_idx) {
    HWY_ASSERT(package_idx < NumPackages());
    Package& package = packages_[package_idx];
    HWY_ASSERT(cluster_idx < package.clusters.size());
    return package.clusters[cluster_idx];
  }

  // Returns total number of cluster workers, for deciding whether to pin.
  size_t TotalWorkers() const {
    size_t total_workers = 0;
    for (size_t package_idx = 0; package_idx < NumPackages(); ++package_idx) {
      const size_t num_clusters = NumClusters(package_idx);
      for (size_t cluster_idx = 0; cluster_idx < num_clusters; ++cluster_idx) {
        total_workers += GetCluster(package_idx, cluster_idx).Size();
      }
    }
    return total_workers;
  }

 private:
  // Sort T := packages/clusters by descending 'size' so that users who only use
  // one Group get the largest.
  template <class T>
  static void SortByDescendingSize(std::vector<T>& groups) {
    std::sort(groups.begin(), groups.end(),
              [](const T& a, const T& b) { return a.Size() > b.Size(); });
  }

  struct Package {
    // Topology is unknown, rely on OS affinity and user-specified slice.
    Package(const LPS& enabled_lps, BoundedSlice lp_slice) {
      clusters.push_back(Cluster(enabled_lps, lp_slice));
    }

    // NOTE: caller is responsible for checking whether `clusters` is empty.
    Package(const LPS& enabled_lps, const hwy::Topology& topology,
            size_t package_idx, BoundedSlice cluster_slice) {
      const hwy::Topology::Package& tpackage = topology.packages[package_idx];
      // Populate `clusters` with the subset of clusters in `cluster_slice` that
      // have any enabled LPs. If `clusters` remains empty, the caller will
      // skip this `Package`.
      clusters.reserve(cluster_slice.Num(tpackage.clusters.size()));
      cluster_slice.Foreach(
          "cluster", tpackage.clusters.size(), [&](size_t cluster_idx) {
            const hwy::Topology::Cluster& tcluster =
                tpackage.clusters[cluster_idx];
            Cluster cluster(enabled_lps, topology.lps, tcluster);
            // Skip if empty, i.e. too few `enabled_lps`.
            if (HWY_LIKELY(cluster.Size() != 0)) {
              clusters.push_back(std::move(cluster));
            }
          });
      SortByDescendingSize(clusters);
    }

    // For SortByDescendingSize.
    size_t Size() const { return clusters.size(); }

    std::vector<Cluster> clusters;
  };  // Package

  // Main part of ctor, called when topology is known.
  void InitFromTopology(const LPS& enabled_lps, BoundedSlice package_slice,
                        BoundedSlice cluster_slice) {
    // (Possibly empty) subset of `Topology` packages that have `enabled_lps`.
    package_slice.Foreach(
        "package", topology_.packages.size(), [&](size_t package_idx) {
          Package package(enabled_lps, topology_, package_idx, cluster_slice);
          // Skip if empty, i.e. too few `enabled_lps`.
          if (HWY_LIKELY(!package.clusters.empty())) {
            packages_.push_back(std::move(package));
          }
        });
    if (NumPackages() == 0) return;
    SortByDescendingSize(packages_);

    const hwy::Topology::Package& tpackage0 = topology_.packages[0];
    HWY_ASSERT(!tpackage0.clusters.empty());
    const hwy::Topology::Cluster& tcluster0 = tpackage0.clusters[0];
    // GetCluster(0, 0) is valid because only non-empty Packages were kept.
    snprintf(topology_string_, sizeof(topology_string_),
             "%zux%zux%zu, using %zux%zux%zu", topology_.packages.size(),
             tpackage0.clusters.size(), tcluster0.lps.Count(), packages_.size(),
             NumClusters(0), GetCluster(0, 0).Size());

    // Remember NUMA nodes of *enabled* LPs.
    enabled_lps.Foreach([&](size_t lp) {
      nodes_.Set(static_cast<size_t>(topology_.lps[lp].node));
    });
  }

  void InitFromSlice(const LPS& enabled_lps, BoundedSlice lp_slice) {
    packages_.push_back(Package(enabled_lps, lp_slice));

    snprintf(topology_string_, sizeof(topology_string_), "LPs=%zu",
             GetCluster(0, 0).Size());

    // Assume a single NUMA node.
    nodes_.Set(0);
    HWY_ASSERT(NumNodes() == 1);
  }

  hwy::Topology topology_;
  std::vector<Package> packages_;
  char topology_string_[96];
  LPS nodes_;
};

// Creates a hierarchy of thread pools according to `BoundedTopology`: one with
// a thread per enabled package; for each of those, one with a thread per
// enabled cluster (CCX/shared L3), and for each of those, the remaining
// enabled cores in that cluster.
//
// Note that we support spin waits, thus it is important for each thread to be
// responsive, hence we do not create more than one thread per enabled core.
// For example, when there are two packages with four clusters of 8 cores,
// `AllPackages` has the main thread plus one extra thread, each `AllClusters`
// has one of the `AllPackages` threads plus three extras, each `Cluster` runs
// on one `AllClusters` thread plus seven extra workers, for a total of
// 1 + 2*3 + 2*(4*7) = 63 extras plus the main thread.
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
  // clusters. This is more intuitive than a per-cluster limit for users who
  // may not be aware of the CPU topology.
  //
  // To ensure we do not create more threads than there are HW cores, which
  // would cause huge slowdowns when spinning, the `BoundedSlice` arguments
  // only impose upper bounds on the number of detected packages and clusters
  // rather than defining the actual number of threads.
  //
  // `pin` is 0 or 1 to force enable/disable, or -1 to choose automatically.
  NestedPools(size_t max_threads, int pin = -1,
              BoundedSlice package_slice = BoundedSlice(),
              BoundedSlice cluster_slice = BoundedSlice(),
              BoundedSlice lp_slice = BoundedSlice())
      : topology_(package_slice, cluster_slice, lp_slice) {
    if (pin == -1) pin = topology_.TotalWorkers() >= 12;

    packages_.resize(topology_.NumPackages());
    all_packages_ = MakePool(packages_.size());
    const size_t max_workers_per_package = max_threads / packages_.size();
    // Each worker in all_packages_, including the main thread, will be the
    // calling thread of an all_clusters->Run, and hence pinned to one of the
    // `cluster.lps` if `pin`.
    all_packages_->Run(
        0, all_packages_->NumWorkers(),
        [&](uint64_t package_idx, size_t thread) {
          HWY_ASSERT(package_idx == thread);  // each thread has one task
          packages_[package_idx] = Package(
              topology_, package_idx, max_workers_per_package, pin, lp_slice);
        });

    // For mapping package/cluster/thread to noncontiguous TLS indices, in case
    // cluster/thread counts differ.
    HWY_ASSERT(!packages_.empty() && packages_.size() <= 16);
    for (const Package& p : packages_) {
      max_clusters_per_package_ =
          HWY_MAX(max_clusters_per_package_, p.NumClusters());
      max_workers_per_cluster_ =
          HWY_MAX(max_workers_per_cluster_, p.MaxWorkersPerCluster());
    }
    HWY_ASSERT(max_clusters_per_package_ >= 1);
    HWY_ASSERT(max_clusters_per_package_ <= 64);
    HWY_ASSERT(max_workers_per_cluster_ >= 1);
    HWY_ASSERT(max_workers_per_cluster_ <= 256);
  }

  // Spinning reduces the latency of barrier synchronization, but wastes lots
  // of energy for long waits, so only do it during generation. Spinning might
  // also be unsafe in virtualized environments because we require threads to
  // be running on their own core and thus responsive to the barrier
  // synchronization.
  void StartSpinning() { SetWaitMode(hwy::PoolWaitMode::kSpin); }
  void StopSpinning() { SetWaitMode(hwy::PoolWaitMode::kBlock); }

  hwy::ThreadPool& AllPackages() { return *all_packages_; }
  hwy::ThreadPool& AllClusters(size_t package_idx) {
    HWY_DASSERT(package_idx < packages_.size());
    return packages_[package_idx].AllClusters();
  }
  hwy::ThreadPool& Cluster(size_t package_idx, size_t cluster_idx) {
    HWY_DASSERT(package_idx < packages_.size());
    return packages_[package_idx].Cluster(cluster_idx);
  }

  // For binding to NUMA nodes.
  size_t Node(size_t package_idx, size_t cluster_idx) const {
    return topology_.GetCluster(package_idx, cluster_idx).Node();
  }

  // Reasonably tight upper bound for allocating thread-local storage (TLS).
  size_t MaxWorkers() const {
    return packages_.size() * max_clusters_per_package_ *
           max_workers_per_cluster_;
  }
  // Returns the first of `cluster.NumWorkers()` TLS indices, to which callers
  // add the worker index given by `cluster.Run`.
  size_t WorkerOffset(size_t package_idx, size_t cluster_idx) const {
    return (package_idx * max_clusters_per_package_ + cluster_idx) *
           max_workers_per_cluster_;
  }

  // For Allocator
  const BoundedTopology& Topology() const { return topology_; }
  const char* TopologyString() const { return topology_.TopologyString(); }

  // Returns a single pool on the first package: either one thread per cluster
  // if there is more than one, which maximizes available memory bandwidth, or
  // the first cluster, which is typically the whole package. For use by callers
  // that only parallelize over a 1D range, as opposed to the nested
  // parallelism of `StaticPartitionRowsAndCols`.
  hwy::ThreadPool& Pool() {
    // Only one cluster: use its pool, typically a whole socket.
    if (AllClusters(0).NumWorkers() == 1) return Cluster(0, 0);
    return AllClusters(0);
  }

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
   public:
    Package() = default;  // for vector
    Package(const BoundedTopology& topology, size_t package_idx,
            size_t max_workers_per_package, int pin, BoundedSlice lp_slice) {
      // Pre-allocate because elements are set concurrently.
      clusters_.resize(topology.NumClusters(package_idx));
      const size_t max_workers_per_cluster =
          max_workers_per_package / clusters_.size();

      all_clusters_ = MakePool(clusters_.size());
      // Parallel so we also pin the calling worker in `all_clusters` to
      // `cluster.lps`.
      all_clusters_->Run(
          0, all_clusters_->NumWorkers(),
          [&](size_t cluster_idx, size_t thread) {
            HWY_ASSERT(cluster_idx == thread);  // each thread has one task
            const BoundedTopology::Cluster& cluster =
                topology.GetCluster(package_idx, cluster_idx);
            clusters_[cluster_idx] =
                MakePool(CapIfNonZero(cluster.Size(), max_workers_per_cluster));
            if (HWY_LIKELY(pin)) {
              // Pin threads AND the calling thread from `all_clusters` to lps.
              const std::vector<size_t> lps = cluster.LPVector();
              HWY_ASSERT(clusters_[cluster_idx]->NumWorkers() <= lps.size());
              clusters_[cluster_idx]->Run(
                  0, clusters_[cluster_idx]->NumWorkers(),
                  [&lps](uint64_t task, size_t thread) {
                    HWY_ASSERT(task == thread);  // each worker has one task
                    hwy::PinThreadToLogicalProcessor(lps[task]);
                  });
            }
          });
    }

    size_t NumClusters() const { return clusters_.size(); }
    size_t MaxWorkersPerCluster() const {
      size_t max_workers_per_cluster = 0;
      for (const PoolPtr& cluster : clusters_) {
        max_workers_per_cluster =
            HWY_MAX(max_workers_per_cluster, cluster->NumWorkers());
      }
      return max_workers_per_cluster;
    }

    hwy::ThreadPool& AllClusters() { return *all_clusters_; }
    hwy::ThreadPool& Cluster(size_t cluster_idx) {
      HWY_DASSERT(cluster_idx < clusters_.size());
      return *clusters_[cluster_idx];
    }

    void SetWaitMode(hwy::PoolWaitMode wait_mode) {
      all_clusters_->SetWaitMode(wait_mode);
      for (PoolPtr& cluster : clusters_) {
        cluster->SetWaitMode(wait_mode);
      }
    }

   private:
    std::vector<PoolPtr> clusters_;
    PoolPtr all_clusters_;
  };  // Package

  void SetWaitMode(hwy::PoolWaitMode wait_mode) {
    all_packages_->SetWaitMode(wait_mode);
    for (Package& package : packages_) {
      package.SetWaitMode(wait_mode);
    }
  }

  BoundedTopology topology_;

  std::vector<Package> packages_;
  PoolPtr all_packages_;

  // For TLS indices.
  size_t max_clusters_per_package_ = 0;
  size_t max_workers_per_cluster_ = 0;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_THREADING_H_
