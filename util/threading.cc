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

#include <algorithm>  // std::sort
#include <atomic>
#include <memory>
#include <optional>
#include <vector>

// Placeholder for container detection, do not remove
#include "util/basics.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/contrib/thread_pool/topology.h"
#include "hwy/profiler.h"

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
  void MaybePin(const BoundedTopology& topology, size_t pkg_idx,
                size_t cluster_idx, const BoundedTopology::Cluster& cluster,
                hwy::ThreadPool& pool) {
    const std::vector<size_t> lps = cluster.LPVector();
    HWY_ASSERT(pool.NumWorkers() <= lps.size());
    pool.Run(0, pool.NumWorkers(), [&](uint64_t task, size_t thread) {
      HWY_ASSERT(task == thread);  // each worker has one task

      char buf[16];  // Linux limitation
      const int bytes_written = snprintf(
          buf, sizeof(buf), "P%zu X%02zu C%03d",
          topology.SkippedPackages() + pkg_idx,
          topology.SkippedClusters() + cluster_idx, static_cast<int>(task));
      HWY_ASSERT(bytes_written < static_cast<int>(sizeof(buf)));
      hwy::SetThreadName(buf, 0);  // does not support varargs

      if (HWY_LIKELY(want_pin_)) {
        if (HWY_UNLIKELY(!hwy::PinThreadToLogicalProcessor(lps[task]))) {
          // Apple does not support pinning, hence do not warn there.
          if (!HWY_OS_APPLE) {
            HWY_WARN("Pinning failed for task %d of %zu to %zu (size %zu)\n",
                     static_cast<int>(task), pool.NumWorkers(), lps[task],
                     lps.size());
          }
          (void)any_error_.test_and_set();
        }
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
};  // Pinning

// Singleton saves global affinity across all BoundedTopology instances because
// pinning overwrites it.
static Pinning& GetPinning() {
  static Pinning pinning;
  return pinning;
}

static PoolPtr MakePool(const Allocator& allocator, size_t num_workers,
                        std::optional<size_t> node = std::nullopt) {
  // `ThreadPool` expects the number of threads to create, which is one less
  // than the number of workers, but avoid underflow if zero.
  const size_t num_threads = num_workers == 0 ? 0 : num_workers - 1;
  PoolPtr ptr = allocator.AllocClasses<hwy::ThreadPool>(1, num_threads);
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

NestedPools::NestedPools(const BoundedTopology& topology,
                         const Allocator& allocator, size_t max_threads,
                         Tristate pin) {
  GetPinning().SetPolicy(pin);
  packages_.resize(topology.NumPackages());
  all_packages_ = MakePool(allocator, packages_.size());
  const size_t max_workers_per_package =
      DivideMaxAcross(max_threads, packages_.size());
  // Each worker in all_packages_, including the main thread, will be the
  // calling thread of an all_clusters->Run, and hence pinned to one of the
  // `cluster.lps` if `pin`.
  all_packages_->Run(0, packages_.size(), [&](uint64_t pkg_idx, size_t thread) {
    HWY_ASSERT(pkg_idx == thread);  // each thread has one task
    packages_[pkg_idx] =
        Package(topology, allocator, pkg_idx, max_workers_per_package);
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

  hwy::Profiler::Get().SetMaxThreads(MaxWorkers());
}

// `max_or_zero` == 0 means no limit.
static inline size_t CapIfNonZero(size_t num, size_t max_or_zero) {
  return (max_or_zero == 0) ? num : HWY_MIN(num, max_or_zero);
}

NestedPools::Package::Package(const BoundedTopology& topology,
                              const Allocator& allocator, size_t pkg_idx,
                              size_t max_workers_per_package) {
  // Pre-allocate because elements are set concurrently.
  clusters_.resize(topology.NumClusters(pkg_idx));
  const size_t max_workers_per_cluster =
      DivideMaxAcross(max_workers_per_package, clusters_.size());

  all_clusters_ = MakePool(allocator, clusters_.size(),
                           topology.GetCluster(pkg_idx, 0).Node());
  // Parallel so we also pin the calling worker in `all_clusters` to
  // `cluster.lps`.
  all_clusters_->Run(
      0, all_clusters_->NumWorkers(), [&](size_t cluster_idx, size_t thread) {
        HWY_ASSERT(cluster_idx == thread);  // each thread has one task
        const BoundedTopology::Cluster& cluster =
            topology.GetCluster(pkg_idx, cluster_idx);
        clusters_[cluster_idx] = MakePool(
            allocator, CapIfNonZero(cluster.Size(), max_workers_per_cluster),
            cluster.Node());
        // Pin workers AND the calling thread from `all_clusters`.
        GetPinning().MaybePin(topology, pkg_idx, cluster_idx, cluster,
                              *clusters_[cluster_idx]);
      });
}

}  // namespace gcpp
