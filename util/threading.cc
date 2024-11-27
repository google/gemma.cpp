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

#include "util/threading.h"

#include <stdio.h>

#include <algorithm>  // std::sort
#include <atomic>
#include <memory>   // std::make_unique
#include <utility>  // std::move
#include <vector>

// Placeholder for container detection, do not remove
#include "util/basics.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/contrib/thread_pool/topology.h"

namespace gcpp {

// Sort T := packages/clusters by descending 'size' so that users who only use
// one Group get the largest.
template <class T>
static void SortByDescendingSize(std::vector<T>& groups) {
  std::sort(groups.begin(), groups.end(),
            [](const T& a, const T& b) { return a.Size() > b.Size(); });
}

// Singleton, holds the original process affinity and the pinning status.
class Pinning {
  static bool InContainer() {
  return false;  }

 public:
  // Returns set of LPs available for use. Subsequent calls return the same
  // set as the first, because pinning overwrites the main thread's affinity.
  // Thread-hostile, not called concurrently.
  LPS EnabledLPs() {
    if (original_affinity_.Any()) return original_affinity_;

    // Regardless of topology, ignore LPs disabled via OS, taskset, or numactl.
    LPS enabled_lps;
    if (HWY_UNLIKELY(!GetThreadAffinity(enabled_lps))) {
      const size_t num_lps = hwy::TotalLogicalProcessors();
      fprintf(
          stderr,
          "Warning, unknown OS affinity, considering all %zu LPs enabled\n.",
          num_lps);
      for (size_t lp = 0; lp < num_lps; ++lp) {
        enabled_lps.Set(lp);
      }
    }

    // Without threading support, only keep the first enabled LP; it might still
    // make sense to pin the main thread to avoid migrations.
    if (HWY_UNLIKELY(!hwy::HaveThreadingSupport())) {
      HWY_ASSERT(enabled_lps.Any());
      const size_t lp = enabled_lps.First();
      enabled_lps = LPS();
      enabled_lps.Set(lp);
      fprintf(stderr,
              "Warning, threads not supported, using only the main thread\n.");
    }

    original_affinity_ = enabled_lps;
    return enabled_lps;
  }

  void SetPolicy(Tristate pin) {
    if (pin == Tristate::kDefault) {
      // Pinning is unreliable inside containers because the hypervisor might
      // periodically change our affinity mask, or other processes might also
      // pin themselves to the same LPs.
      pin = InContainer() ? Tristate::kFalse : Tristate::kTrue;
    }
    want_pin_ = (pin == Tristate::kTrue);
    any_error_.clear();
  }

  // If want_pin_, tries to pin each worker in `pool` to an LP in `cluster`,
  // and sets `any_error_` if any fails.
  void MaybePin(const BoundedTopology::Cluster& cluster, PoolPtr& pool) {
    if (HWY_UNLIKELY(!want_pin_)) return;

    const std::vector<size_t> lps = cluster.LPVector();
    HWY_ASSERT(pool->NumWorkers() <= lps.size());
    pool->Run(
        0, pool->NumWorkers(),
        [this, &pool, &lps](uint64_t task, size_t thread) {
          HWY_ASSERT(task == thread);  // each worker has one task
          if (HWY_UNLIKELY(!hwy::PinThreadToLogicalProcessor(lps[task]))) {
            fprintf(stderr,
                    "Pinning failed for task %zu of %zu to %zu (size %zu)\n",
                    task, pool->NumWorkers(), lps[task], lps.size());
            (void)any_error_.test_and_set();
          }
        });
  }

  // Called ONCE after all MaybePin because it invalidates the error status.
  bool AllPinned(const char** pin_string) {
    // If !want_pin_, MaybePin will return without setting any_error_, but in
    // that case we still want to return false to avoid spinning.
    // .test() was only added in C++20, so we use .test_and_set() instead.
    const bool all_pinned = want_pin_ && !any_error_.test_and_set();
    *pin_string = all_pinned  ? "pinned"
                  : want_pin_ ? "pinning failed"
                              : "pinning skipped";
    return all_pinned;
  }

 private:
  std::atomic_flag any_error_ = ATOMIC_FLAG_INIT;
  bool want_pin_;  // set in SetPolicy
  LPS original_affinity_;
};  // Pinning

// Singleton saves global affinity across all BoundedTopology instances because
// pinning overwrites it.
static Pinning& GetPinning() {
  static Pinning pinning;
  return pinning;
}

BoundedTopology::BoundedTopology(BoundedSlice package_slice,
                                 BoundedSlice cluster_slice,
                                 BoundedSlice lp_slice) {
  const LPS enabled_lps = GetPinning().EnabledLPs();

#if !GEMMA_DISABLE_TOPOLOGY
  if (HWY_LIKELY(!topology_.packages.empty())) {
    InitFromTopology(enabled_lps, package_slice, cluster_slice);
  }
#endif

  // Topology unknown or no packages with enabled LPs: create a single
  // package with one cluster, and one node.
  if (HWY_UNLIKELY(NumPackages() == 0)) {
    InitFromSlice(enabled_lps, lp_slice);
  }

  HWY_ASSERT(NumPackages() != 0 && NumClusters(0) != 0 && NumNodes() != 0);
}

// Topology is unknown, rely on OS affinity and user-specified slice.
BoundedTopology::Cluster::Cluster(const LPS& enabled_lps,
                                  BoundedSlice lp_slice) {
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

BoundedTopology::Cluster::Cluster(const LPS& enabled_lps,
                                  const std::vector<hwy::Topology::LP>& all_lps,
                                  const hwy::Topology::Cluster& tcluster) {
  bool is_first_lp = true;

  tcluster.lps.Foreach([&](size_t lp) {
    // Skip if not first-hyperthread or disabled.
    if (all_lps[lp].smt != 0 || !enabled_lps.Get(lp)) return;

    AddLP(lp);

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
          fprintf(stderr, "WARNING: lp %zu on node %zu != cluster node %zu.\n",
                  lp, lp_node, node_);
        }
        if (HWY_UNLIKELY(private_kib_ != tcluster.private_kib)) {
          warned = true;
          fprintf(stderr, "WARNING: lp %zu private_kib %zu != cluster %zu.\n",
                  lp, private_kib_, tcluster.private_kib);
        }
        if (HWY_UNLIKELY(shared_kib_ != tcluster.shared_kib)) {
          warned = true;
          fprintf(stderr, "WARNING: lp %zu shared_kib %zu != cluster %zu.\n",
                  lp, shared_kib_, tcluster.shared_kib);
        }
      }  // !warned
    }
  });
}

// NOTE: caller is responsible for checking whether `clusters` is empty.
BoundedTopology::Package::Package(const LPS& enabled_lps,
                                  const hwy::Topology& topology,
                                  size_t package_idx,
                                  BoundedSlice cluster_slice) {
  const hwy::Topology::Package& tpackage = topology.packages[package_idx];
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
          clusters.push_back(std::move(cluster));
        }
      });
  SortByDescendingSize(clusters);
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
  for (size_t package_idx = 0; package_idx < topology_.packages.size();
       ++package_idx) {
    const std::vector<hwy::Topology::Cluster>& tclusters =
        topology_.packages[package_idx].clusters;
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
      fprintf(stderr,
              "Package %zu: multiple clusters with max size %zu, whereas CCX "
              "only have 8, may indicate a bug in hwy::Topology.\n",
              package_idx, tcluster_cores);
    }
    max_tcluster_cores = HWY_MAX(max_tcluster_cores, tcluster_cores);
    max_tcluster_lps = HWY_MAX(max_tcluster_lps, tcluster_lps);
  }
  HWY_ASSERT(max_tclusters != 0);
  HWY_ASSERT(max_tcluster_cores != 0);
  HWY_ASSERT(max_tcluster_lps >= max_tcluster_cores);
}

// Main part of ctor, called when topology is known.
void BoundedTopology::InitFromTopology(const LPS& enabled_lps,
                                       BoundedSlice package_slice,
                                       BoundedSlice cluster_slice) {
  size_t max_tclusters, max_tcluster_cores, max_tcluster_lps;
  ScanTClusters(topology_, max_tclusters, max_tcluster_cores, max_tcluster_lps);

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

  // Remember NUMA nodes that we are actually using (not just enabled).
  for (const Package& p : packages_) {
    for (const Cluster& c : p.clusters) {
      nodes_.Set(c.Node());
    }
  }

  // Scan for max BoundedTopology clusters and their size, for topology_string_.
  size_t all_max_cluster_size = 0;
  for (size_t package_idx = 0; package_idx < NumPackages(); ++package_idx) {
    size_t max_cluster_size = 0;
    for (size_t cluster_idx = 0; cluster_idx < NumClusters(package_idx);
         ++cluster_idx) {
      max_cluster_size = HWY_MAX(max_cluster_size,
                                 GetCluster(package_idx, cluster_idx).Size());
    }
    if (NumClusters(package_idx) > 1 && max_cluster_size > 8) {
      fprintf(stderr,
              "Package %zu: multiple clusters with max size %zu, whereas CCX "
              "only have 8, may indicate a bug in BoundedTopology.\n",
              package_idx, max_cluster_size);
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

void BoundedTopology::InitFromSlice(const LPS& enabled_lps,
                                    BoundedSlice lp_slice) {
  packages_.push_back(Package(enabled_lps, lp_slice));

  snprintf(topology_string_, sizeof(topology_string_), "LPs=%zu",
           GetCluster(0, 0).Size());

  // Assume a single NUMA node.
  nodes_.Set(0);
  HWY_ASSERT(NumNodes() == 1);
}

static PoolPtr MakePool(size_t num_workers) {
  // `ThreadPool` expects the number of threads to create, which is one less
  // than the number of workers, but avoid underflow if zero.
  const size_t num_threads = num_workers == 0 ? 0 : num_workers - 1;
  return std::make_unique<hwy::ThreadPool>(num_threads);
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

NestedPools::NestedPools(size_t max_threads, Tristate pin,
                         BoundedSlice package_slice, BoundedSlice cluster_slice,
                         BoundedSlice lp_slice)
    : topology_(package_slice, cluster_slice, lp_slice) {
  GetPinning().SetPolicy(pin);
  packages_.resize(topology_.NumPackages());
  all_packages_ = MakePool(packages_.size());
  const size_t max_workers_per_package =
      DivideMaxAcross(max_threads, packages_.size());
  // Each worker in all_packages_, including the main thread, will be the
  // calling thread of an all_clusters->Run, and hence pinned to one of the
  // `cluster.lps` if `pin`.
  all_packages_->Run(
      0, all_packages_->NumWorkers(), [&](uint64_t package_idx, size_t thread) {
        HWY_ASSERT(package_idx == thread);  // each thread has one task
        packages_[package_idx] =
            Package(topology_, package_idx, max_workers_per_package, lp_slice);
      });

  all_pinned_ = GetPinning().AllPinned(&pin_string_);

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

NestedPools::Package::Package(const BoundedTopology& topology,
                              size_t package_idx,
                              size_t max_workers_per_package,
                              BoundedSlice lp_slice) {
  // Pre-allocate because elements are set concurrently.
  clusters_.resize(topology.NumClusters(package_idx));
  const size_t max_workers_per_cluster =
      DivideMaxAcross(max_workers_per_package, clusters_.size());

  all_clusters_ = MakePool(clusters_.size());
  // Parallel so we also pin the calling worker in `all_clusters` to
  // `cluster.lps`.
  all_clusters_->Run(
      0, all_clusters_->NumWorkers(), [&](size_t cluster_idx, size_t thread) {
        HWY_ASSERT(cluster_idx == thread);  // each thread has one task
        const BoundedTopology::Cluster& cluster =
            topology.GetCluster(package_idx, cluster_idx);
        clusters_[cluster_idx] =
            MakePool(CapIfNonZero(cluster.Size(), max_workers_per_cluster));
        // Pin workers AND the calling thread from `all_clusters`.
        GetPinning().MaybePin(cluster, clusters_[cluster_idx]);
      });
}

}  // namespace gcpp
