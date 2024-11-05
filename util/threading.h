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

#include <memory>  // std::unique_ptr
#include <vector>

#include "util/basics.h"  // Tristate
#include "hwy/base.h"       // HWY_ASSERT
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/contrib/thread_pool/topology.h"

#ifndef GEMMA_DISABLE_TOPOLOGY
#define GEMMA_DISABLE_TOPOLOGY 0
#endif  // !GEMMA_DISABLE_TOPOLOGY

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

// We want vectors of hwy::ThreadPool, which is unfortunately not movable,
// hence we wrap them in unique_ptr.
using PoolPtr = std::unique_ptr<hwy::ThreadPool>;

// Wraps hwy::Topology and only keeps the subset of packages and clusters
// apportioned by BoundedSlice, further limited by the OS affinity mask.
// NOTE: if topology is unknown or the OS affinity is too restrictive, we fall
// back to a single package and cluster.
class BoundedTopology {
 public:
  BoundedTopology(BoundedSlice package_slice, BoundedSlice cluster_slice,
                  BoundedSlice lp_slice);

  size_t NumPackages() const { return packages_.size(); }
  size_t NumNodes() const { return nodes_.Count(); }
  const char* TopologyString() const { return topology_string_; }

  class Cluster {
   public:
    Cluster(const LPS& enabled_lps, BoundedSlice lp_slice);
    Cluster(const LPS& enabled_lps,
            const std::vector<hwy::Topology::LP>& all_lps,
            const hwy::Topology::Cluster& tcluster);

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

 private:
  struct Package {
    // Topology is unknown, rely on OS affinity and user-specified slice.
    Package(const LPS& enabled_lps, BoundedSlice lp_slice) {
      clusters.push_back(Cluster(enabled_lps, lp_slice));
    }

    Package(const LPS& enabled_lps, const hwy::Topology& topology,
            size_t package_idx, BoundedSlice cluster_slice);

    // For SortByDescendingSize.
    size_t Size() const { return clusters.size(); }

    std::vector<Cluster> clusters;
  };  // Package

  void InitFromTopology(const LPS& enabled_lps, BoundedSlice package_slice,
                        BoundedSlice cluster_slice);
  void InitFromSlice(const LPS& enabled_lps, BoundedSlice lp_slice);

#if !GEMMA_DISABLE_TOPOLOGY
  hwy::Topology topology_;
#endif
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
  NestedPools(size_t max_threads, Tristate pin = Tristate::kDefault,
              BoundedSlice package_slice = BoundedSlice(),
              BoundedSlice cluster_slice = BoundedSlice(),
              BoundedSlice lp_slice = BoundedSlice());

  // Subject to `use_spinning`, enables spin waits with the goal of reducing the
  // latency of barrier synchronization. We only spin during Generate to avoid
  // wasting energy during long waits. If `use_spinning` is kDefault, we first
  // set it to kTrue or kFalse based on a heuristic.
  void MaybeStartSpinning(Tristate& use_spinning) {
    if (HWY_UNLIKELY(use_spinning == Tristate::kDefault)) {
      // The default is to only spin when pinning was enabled and supported by
      // the OS. Unless spin-waits have near-exclusive use of a core, the tail
      // latency can be higher than blocking waits.
      use_spinning = all_pinned_ ? Tristate::kTrue : Tristate::kFalse;
    }
    if (use_spinning == Tristate::kTrue) {
      SetWaitMode(hwy::PoolWaitMode::kSpin);
    }
  }
  void MaybeStopSpinning(const Tristate use_spinning) {
    HWY_DASSERT(use_spinning != Tristate::kDefault);  // see MaybeStartSpinning
    if (use_spinning == Tristate::kTrue) {
      SetWaitMode(hwy::PoolWaitMode::kBlock);
    }
  }

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
  // For ShowConfig
  const char* TopologyString() const { return topology_.TopologyString(); }
  const char* PinString() const { return pin_string_; }

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
  class Pinning;

  class Package {
   public:
    Package() = default;  // for vector
    Package(const BoundedTopology& topology, size_t package_idx,
            size_t max_workers_per_package, Pinning& pinning,
            BoundedSlice lp_slice);

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
  bool all_pinned_;
  const char* pin_string_;

  std::vector<Package> packages_;
  PoolPtr all_packages_;

  // For TLS indices.
  size_t max_clusters_per_package_ = 0;
  size_t max_workers_per_cluster_ = 0;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_THREADING_H_
