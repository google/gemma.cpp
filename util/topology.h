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

#ifndef THIRD_PARTY_GEMMA_CPP_UTIL_TOPOLOGY_H_
#define THIRD_PARTY_GEMMA_CPP_UTIL_TOPOLOGY_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

// IWYU pragma: begin_exports
#include "hwy/base.h"  // HWY_ASSERT
#include "hwy/contrib/thread_pool/topology.h"
// IWYU pragma: end_exports

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

// Wraps hwy::Topology and only keeps the subset of packages and clusters
// apportioned by BoundedSlice, further limited by the OS affinity mask.
// NOTE: if topology is unknown or the OS affinity is too restrictive, we fall
// back to a single package and cluster.
class BoundedTopology {
 public:
  // Defaults to "use all detected".
  BoundedTopology(BoundedSlice package_slice = BoundedSlice(),
                  BoundedSlice cluster_slice = BoundedSlice(),
                  BoundedSlice lp_slice = BoundedSlice());

  size_t NumPackages() const { return packages_.size(); }
  size_t NumNodes() const { return nodes_.Count(); }
  const char* TopologyString() const { return topology_string_; }

  class Cluster {
   public:
    Cluster(const LPS& lps);
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

    const LPS& LPSet() const { return lps_; }
    size_t Node() const { return node_; }
    size_t PrivateKiB() const { return private_kib_; }
    size_t SharedKiB() const { return shared_kib_; }

   private:
    // Enabled LPs; if topology is known, only the ones in this cluster.
    LPS lps_;
    // How many workers in the per-cluster pool. If 0, this Cluster is removed.
    size_t num_workers_ = 0;
    // NUMA node, set from hwy::Topology::LP::node.
    size_t node_ = 0;
    // L2 cache size in KiB, or 0 if unknown.
    size_t private_kib_ = 0;
    // L3 cache size in KiB, or 0 if unknown.
    size_t shared_kib_ = 0;
  };  // Cluster

  size_t NumClusters(size_t pkg_idx) const {
    HWY_ASSERT(pkg_idx < NumPackages());
    return packages_[pkg_idx].clusters.size();
  }
  const Cluster& GetCluster(size_t pkg_idx, size_t cluster_idx) const {
    HWY_ASSERT(pkg_idx < NumPackages());
    const Package& package = packages_[pkg_idx];
    HWY_ASSERT(cluster_idx < package.clusters.size());
    return package.clusters[cluster_idx];
  }
  Cluster& GetCluster(size_t pkg_idx, size_t cluster_idx) {
    HWY_ASSERT(pkg_idx < NumPackages());
    Package& package = packages_[pkg_idx];
    HWY_ASSERT(cluster_idx < package.clusters.size());
    return package.clusters[cluster_idx];
  }

#if !GEMMA_DISABLE_TOPOLOGY
  const hwy::Topology& FullTopology() const { return topology_; }
#endif

  // In case we are running with a subset of packages/clusters, these are added
  // to the package/cluster indices for purposes of the thread name, so that
  // they are distinct.
  size_t SkippedPackages() const { return package_slice_.Begin(); }
  size_t SkippedClusters() const { return cluster_slice_.Begin(); }

 private:
  struct Package {
    explicit Package(const LPS& enabled_lps);
    Package(const LPS& enabled_lps, const hwy::Topology& topology,
            size_t pkg_idx, BoundedSlice cluster_slice);

    // For SortByDescendingSize.
    size_t Size() const { return clusters.size(); }

    std::vector<Cluster> clusters;
  };  // Package

  void InitFromTopology(const LPS& enabled_lps);
  void InitFromLPs(const LPS& enabled_lps);

#if !GEMMA_DISABLE_TOPOLOGY
  hwy::Topology topology_;
#endif
  BoundedSlice package_slice_;
  BoundedSlice cluster_slice_;
  std::vector<Package> packages_;
  char topology_string_[96];
  LPS nodes_;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_TOPOLOGY_H_
