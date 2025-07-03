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

#include "util/topology.h"

#include <stdio.h>

#include <algorithm>  // std::sort
#include <utility>  // std::move
#include <vector>

#include "hwy/base.h"

namespace gcpp {

// Sort T := packages/clusters by descending 'size' so that users who only use
// one Group get the largest.
template <class T>
static void SortByDescendingSize(std::vector<T>& groups) {
  std::sort(groups.begin(), groups.end(),
            [](const T& a, const T& b) { return a.Size() > b.Size(); });
}

// Returns set of LPs available for use.
static LPS EnabledLPs(const BoundedSlice& lp_slice) {
  LPS enabled_lps;

  // Thread-safe caching during the first call because subsequent pinning
  // overwrites the main thread's affinity.
  static const LPS affinity = []() {
    LPS affinity;
    if (!GetThreadAffinity(affinity)) affinity = LPS();
    return affinity;
  }();
  if (HWY_LIKELY(affinity.Any())) {
    // To honor taskset/numactl *and* the users's `lp_slice`, we interpret
    // the latter as a slice of the 1-bits of `enabled_lps`. Note that this
    // can be used to exclude hyperthreads because Linux groups LPs by
    // sibling index. For example, the first `num_cores` are not siblings.
    const size_t detected = affinity.Count();
    size_t enabled_idx = 0;
    affinity.Foreach([&](size_t lp) {
      if (lp_slice.Contains(detected, enabled_idx)) {
        enabled_lps.Set(lp);
      }
      ++enabled_idx;
    });
  } else {
    const size_t num_lps = hwy::TotalLogicalProcessors();
    // Do not warn on Apple, where affinity is not supported.
    if (!HWY_OS_APPLE) {
      HWY_WARN("unknown OS affinity, max %zu LPs and slice %zu.", num_lps,
               lp_slice.Num(num_lps));
    }
    for (size_t lp = 0; lp < num_lps; ++lp) {
      if (lp_slice.Contains(num_lps, lp)) {
        enabled_lps.Set(lp);
      }
    }
  }

  // Without threading support, only keep the first enabled LP; it might still
  // make sense to pin the main thread to avoid migrations.
  if (HWY_UNLIKELY(!hwy::HaveThreadingSupport())) {
    HWY_ASSERT(enabled_lps.Any());
    const size_t lp = enabled_lps.First();
    enabled_lps = LPS();
    enabled_lps.Set(lp);
    HWY_WARN("Warning, threads not supported, using only the main thread.");
  }

  return enabled_lps;
}

BoundedTopology::BoundedTopology(BoundedSlice package_slice,
                                 BoundedSlice cluster_slice,
                                 BoundedSlice lp_slice)
    : package_slice_(package_slice), cluster_slice_(cluster_slice) {
  const LPS enabled_lps = EnabledLPs(lp_slice);

#if !GEMMA_DISABLE_TOPOLOGY
  if (HWY_LIKELY(!topology_.packages.empty())) {
    InitFromTopology(enabled_lps);
  }
#endif

  // Topology unknown or no packages with enabled LPs: create a single
  // package with one cluster, and one node.
  if (HWY_UNLIKELY(NumPackages() == 0)) {
    InitFromLPs(enabled_lps);
  }

  HWY_ASSERT(NumPackages() != 0 && NumClusters(0) != 0 && NumNodes() != 0);
}

// Topology is unknown, take the given set of LPs.
BoundedTopology::Cluster::Cluster(const LPS& lps) {
  lps_ = lps;
  num_workers_ = lps.Count();
}

BoundedTopology::Cluster::Cluster(const LPS& enabled_lps,
                                  const std::vector<hwy::Topology::LP>& all_lps,
                                  const hwy::Topology::Cluster& tcluster) {
  bool is_first_lp = true;

  tcluster.lps.Foreach([&](size_t lp) {
    // Skip if not first-hyperthread or disabled.
    if (all_lps[lp].smt != 0 || !enabled_lps.Get(lp)) return;

    HWY_ASSERT(!lps_.Get(lp));  // Foreach ensures uniqueness
    lps_.Set(lp);
    ++num_workers_;

    // Set fields once, and ensure subsequent LPs match - we assume there
    // is only one NUMA node per cluster, with the same L2/L3 size.
    const size_t lp_node = static_cast<size_t>(all_lps[lp].node);
    if (is_first_lp) {
      is_first_lp = false;
      node_ = lp_node;
      private_kib_ = tcluster.private_kib;
      shared_kib_ = tcluster.shared_kib;
    } else {
      static bool warned = false;
      if (HWY_LIKELY(!warned)) {
        if (HWY_UNLIKELY(lp_node != node_)) {
          warned = true;
          HWY_WARN("lp %zu on node %zu != cluster node %zu.", lp, lp_node,
                   node_);
        }
        if (HWY_UNLIKELY(private_kib_ != tcluster.private_kib)) {
          warned = true;
          HWY_WARN("lp %zu private_kib %zu != cluster %u.", lp, private_kib_,
                   static_cast<unsigned>(tcluster.private_kib));
        }
        if (HWY_UNLIKELY(shared_kib_ != tcluster.shared_kib)) {
          warned = true;
          HWY_WARN("lp %zu shared_kib %zu != cluster %u.", lp, shared_kib_,
                   static_cast<unsigned>(tcluster.shared_kib));
        }
      }  // !warned
    }
  });
}

// CPUs without clusters are rarely more than dozens of cores, and 6 is a
// decent number of threads in a per-cluster pool.
constexpr bool kSplitLargeClusters = false;
constexpr size_t kMaxClusters = 8;
constexpr size_t kMaxLPsPerCluster = 6;

// Topology is unknown, use only the given LPs which derive from OS affinity
// and `lp_slice`.
BoundedTopology::Package::Package(const LPS& enabled_lps) {
  LPS clusters_lps[kMaxClusters];
  const size_t num_clusters =
      kSplitLargeClusters
          ? HWY_MIN(kMaxClusters,
                    hwy::DivCeil(enabled_lps.Count(), kMaxLPsPerCluster))
          : 1;

  size_t enabled_idx = 0;
  enabled_lps.Foreach([&](size_t lp) {
    clusters_lps[enabled_idx % num_clusters].Set(lp);
    ++enabled_idx;
  });

  for (size_t cluster_idx = 0; cluster_idx < num_clusters; ++cluster_idx) {
    clusters.push_back(Cluster(clusters_lps[cluster_idx]));
  }
}

// NOTE: caller is responsible for checking whether `clusters` is empty.
BoundedTopology::Package::Package(const LPS& enabled_lps,
                                  const hwy::Topology& topology, size_t pkg_idx,
                                  BoundedSlice cluster_slice) {
  const hwy::Topology::Package& tpackage = topology.packages[pkg_idx];
  // Populate `clusters` with the subset of clusters in `cluster_slice` that
  // have any enabled LPs. If `clusters` remains empty, the caller will
  // skip this `Package`.
  clusters.reserve(cluster_slice.Num(tpackage.clusters.size()));
  cluster_slice.Foreach(
      "cluster", tpackage.clusters.size(), [&](size_t cluster_idx) {
        const hwy::Topology::Cluster& tcluster = tpackage.clusters[cluster_idx];
        Cluster cluster(enabled_lps, topology.lps, tcluster);

        // Skip if empty, i.e. too few `enabled_lps`.
        if (HWY_LIKELY(cluster.Size() != 0)) {
          clusters.push_back(cluster);
        }
      });
  SortByDescendingSize(clusters);

  // If there is only one large cluster, split it into smaller ones.
  if (kSplitLargeClusters && clusters.size() == 1 &&
      enabled_lps.Count() >= 16) {
    const LPS lps = clusters[0].LPSet();  // copy so we can clear
    clusters.clear();

    // Split `lps` into several clusters.
    LPS clusters_lps[kMaxClusters];
    const size_t num_clusters =
        HWY_MIN(kMaxClusters, hwy::DivCeil(lps.Count(), kMaxLPsPerCluster));
    size_t num_lps = 0;
    lps.Foreach(
        [&](size_t lp) { clusters_lps[num_lps++ % num_clusters].Set(lp); });
    HWY_DASSERT(num_lps == lps.Count());

    // Create new clusters, just inserting the new LPS.
    hwy::Topology::Cluster tcluster = tpackage.clusters[0];  // modifiable copy
    for (size_t cluster_idx = 0; cluster_idx < num_clusters; ++cluster_idx) {
      tcluster.lps = clusters_lps[cluster_idx];
      // Keep same `private_kib` and `shared_kib`.
      clusters.push_back(Cluster(enabled_lps, topology.lps, tcluster));
    }
  }
}

#if !GEMMA_DISABLE_TOPOLOGY

static size_t CoresFromLPs(const LPS& lps, const hwy::Topology& topology) {
  LPS cores;
  lps.Foreach([&](size_t lp) {
    if (topology.lps[lp].smt == 0) cores.Set(lp);
  });
  return cores.Count();
}

// Scans hwy::Topology for clusters and their size, for use by topology_string_.
static void ScanTClusters(hwy::Topology& topology_, size_t& max_tclusters,
                          size_t& max_tcluster_cores,
                          size_t& max_tcluster_lps) {
  max_tclusters = 0;
  max_tcluster_cores = 0;
  max_tcluster_lps = 0;
  for (size_t pkg_idx = 0; pkg_idx < topology_.packages.size(); ++pkg_idx) {
    const std::vector<hwy::Topology::Cluster>& tclusters =
        topology_.packages[pkg_idx].clusters;
    max_tclusters = HWY_MAX(max_tclusters, tclusters.size());
    size_t tcluster_cores = 0;
    size_t tcluster_lps = 0;
    for (size_t cluster_idx = 0; cluster_idx < tclusters.size();
         ++cluster_idx) {
      const size_t cores = CoresFromLPs(tclusters[cluster_idx].lps, topology_);
      const size_t lps = tclusters[cluster_idx].lps.Count();
      tcluster_cores = HWY_MAX(tcluster_cores, cores);
      tcluster_lps = HWY_MAX(tcluster_lps, lps);
    }

    if (tclusters.size() > 1 && tcluster_cores > 8) {
      HWY_WARN(
          "Package %zu: multiple clusters with max size %zu, whereas CCX "
          "only have 8, may indicate a bug in hwy::Topology.",
          pkg_idx, tcluster_cores);
    }
    max_tcluster_cores = HWY_MAX(max_tcluster_cores, tcluster_cores);
    max_tcluster_lps = HWY_MAX(max_tcluster_lps, tcluster_lps);
  }
  HWY_ASSERT(max_tclusters != 0);
  HWY_ASSERT(max_tcluster_cores != 0);
  HWY_ASSERT(max_tcluster_lps >= max_tcluster_cores);
}

// Main part of ctor, called when topology is known.
void BoundedTopology::InitFromTopology(const LPS& enabled_lps) {
  size_t max_tclusters, max_tcluster_cores, max_tcluster_lps;
  ScanTClusters(topology_, max_tclusters, max_tcluster_cores, max_tcluster_lps);

  // (Possibly empty) subset of `Topology` packages that have `enabled_lps`.
  package_slice_.Foreach(
      "package", topology_.packages.size(), [&](size_t pkg_idx) {
        Package package(enabled_lps, topology_, pkg_idx, cluster_slice_);
        // Skip if empty, i.e. too few `enabled_lps`.
        if (HWY_LIKELY(!package.clusters.empty())) {
          packages_.push_back(std::move(package));
        }
      });
  if (NumPackages() == 0) return;
  SortByDescendingSize(packages_);

  // Remember NUMA nodes that we are actually using (not just enabled).
  for (const Package& p : packages_) {
    for (const Cluster& c : p.clusters) {
      nodes_.Set(c.Node());
    }
  }

  // Scan for max BoundedTopology clusters and their size, for topology_string_.
  size_t all_max_cluster_size = 0;
  for (size_t pkg_idx = 0; pkg_idx < NumPackages(); ++pkg_idx) {
    size_t max_cluster_size = 0;
    for (size_t cluster_idx = 0; cluster_idx < NumClusters(pkg_idx);
         ++cluster_idx) {
      max_cluster_size =
          HWY_MAX(max_cluster_size, GetCluster(pkg_idx, cluster_idx).Size());
    }
    if (NumClusters(pkg_idx) > 1 && max_cluster_size > 8) {
      HWY_WARN(
          "Package %zu: multiple clusters with max size %zu, whereas CCX "
          "only have 8, may indicate a bug in BoundedTopology.",
          pkg_idx, max_cluster_size);
    }
    all_max_cluster_size = HWY_MAX(all_max_cluster_size, max_cluster_size);
  }

  snprintf(topology_string_, sizeof(topology_string_),
           "%zuS %zuX %zuC %zuH, using %zuS %zuX %zuC (nodes=%zu)",
           topology_.packages.size(), max_tclusters, max_tcluster_cores,
           max_tcluster_lps / max_tcluster_cores, packages_.size(),
           NumClusters(0), all_max_cluster_size, nodes_.Count());
}

#endif  // !GEMMA_DISABLE_TOPOLOGY

void BoundedTopology::InitFromLPs(const LPS& enabled_lps) {
  packages_.push_back(Package(enabled_lps));

  snprintf(topology_string_, sizeof(topology_string_), "LPs=%zu",
           GetCluster(0, 0).Size());

  // Assume a single NUMA node.
  nodes_.Set(0);
  HWY_ASSERT(NumNodes() == 1);
}

}  // namespace gcpp
