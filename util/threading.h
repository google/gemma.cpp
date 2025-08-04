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
#include <stdint.h>

#include <vector>

// IWYU pragma: begin_exports
#include "util/allocator.h"
#include "util/args.h"
#include "util/basics.h"  // Tristate
#include "util/topology.h"
#include "hwy/base.h"  // HWY_ASSERT
#include "hwy/contrib/thread_pool/thread_pool.h"
// IWYU pragma: end_exports

#ifndef GEMMA_DISABLE_TOPOLOGY
#define GEMMA_DISABLE_TOPOLOGY 0
#endif  // !GEMMA_DISABLE_TOPOLOGY

namespace gcpp {

// Page-aligned on NUMA systems so we can bind to a NUMA node. This also allows
// moving because it is a typedef to `std::unique_ptr`.
using PoolPtr = AlignedClassPtr<hwy::ThreadPool>;

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
  // may not be aware of the CPU topology. 0 means no limit.
  //
  // To ensure we do not create more threads than there are HW cores, which
  // would cause huge slowdowns when spinning, the `BoundedSlice` arguments
  // only impose upper bounds on the number of detected packages and clusters
  // rather than defining the actual number of threads.
  NestedPools(const BoundedTopology& topology, const Allocator& allocator,
              size_t max_threads = 0, Tristate pin = Tristate::kDefault);

  bool AllPinned() const { return all_pinned_; }

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

  size_t NumPackages() const { return packages_.size(); }
  hwy::ThreadPool& AllPackages() { return *all_packages_; }
  hwy::ThreadPool& AllClusters(size_t pkg_idx) {
    HWY_DASSERT(pkg_idx < NumPackages());
    return packages_[pkg_idx].AllClusters();
  }
  hwy::ThreadPool& Cluster(size_t pkg_idx, size_t cluster_idx) {
    HWY_DASSERT(pkg_idx < NumPackages());
    return packages_[pkg_idx].Cluster(cluster_idx);
  }

  // Reasonably tight upper bounds for allocating thread-local storage (TLS).
  size_t MaxWorkersPerCluster() const { return max_workers_per_cluster_; }
  size_t MaxWorkersPerPackage() const {
    return max_clusters_per_package_ * MaxWorkersPerCluster();
  }
  size_t MaxWorkers() const { return NumPackages() * MaxWorkersPerPackage(); }

  // Actual number of workers.
  size_t TotalWorkers() const {
    size_t total_workers = 0;
    for (size_t pkg_idx = 0; pkg_idx < NumPackages(); ++pkg_idx) {
      total_workers += packages_[pkg_idx].TotalWorkers();
    }
    return total_workers;
  }

  // For ShowConfig
  const char* PinString() const { return pin_string_; }

  // Returns a single pool on the given package: either one thread per cluster
  // if there is more than one, which maximizes available memory bandwidth, or
  // the first cluster, which is typically the whole package. For use by callers
  // that only have a single parallel-for.
  hwy::ThreadPool& Pool(size_t pkg_idx = 0) {
    // Only one cluster: use its pool, typically a whole socket.
    if (AllClusters(pkg_idx).NumWorkers() == 1) {
      return Cluster(pkg_idx, 0);
    }
    // One worker per cluster to maximize bandwidth availability.
    return AllClusters(pkg_idx);
  }

 private:
  class Package {
   public:
    Package() = default;  // for vector
    Package(const BoundedTopology& topology, const Allocator& allocator,
            size_t pkg_idx, size_t max_workers_per_package);

    size_t NumClusters() const { return clusters_.size(); }
    size_t MaxWorkersPerCluster() const {
      size_t max_workers_per_cluster = 0;
      for (const PoolPtr& cluster : clusters_) {
        max_workers_per_cluster =
            HWY_MAX(max_workers_per_cluster, cluster->NumWorkers());
      }
      return max_workers_per_cluster;
    }
    size_t TotalWorkers() const {
      size_t total_workers = 0;
      for (const PoolPtr& cluster : clusters_) {
        total_workers += cluster->NumWorkers();
      }
      return total_workers;
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

  bool all_pinned_;
  const char* pin_string_;

  std::vector<Package> packages_;
  PoolPtr all_packages_;

  // For TLS indices. One might think this belongs in BoundedTopology, but it
  // depends on max_threads, which is passed to the NestedPools constructor.
  size_t max_clusters_per_package_ = 0;
  size_t max_workers_per_cluster_ = 0;
};

// Splits `range` into subranges of size `task_size`, except for the last,
// which receives the remainder. Used with the `ParallelizeOneRange` etc.
// functions below.
class IndexRangePartition {
 public:
  IndexRangePartition() = default;  // for MMPartitions
  IndexRangePartition(const IndexRange& range, const size_t task_size)
      : range_(range), task_size_(static_cast<uint32_t>(task_size)) {
    const uint32_t num = static_cast<uint32_t>(range.Num());
    HWY_DASSERT(task_size_ != 0);
    num_tasks_ = hwy::DivCeil(num, task_size_);
    HWY_DASSERT(num_tasks_ != 0);
    if constexpr (HWY_IS_DEBUG_BUILD) {
      const uint32_t handled = num_tasks_ * task_size_;
      // The last task may extend beyond items, but at most by (task_size_ - 1).
      HWY_DASSERT(num <= handled && handled < num + task_size_);
      (void)handled;
    }
  }

  size_t TaskSize() const { return static_cast<size_t>(task_size_); }
  size_t NumTasks() const { return static_cast<size_t>(num_tasks_); }

  IndexRange Range(size_t task_idx) const {
    HWY_DASSERT(task_idx < NumTasks());
    return MakeIndexRange(range_.begin() + task_idx * TaskSize(), range_.end(),
                          TaskSize());
  }

  template <typename Func>
  void VisitAll(const Func& func) const {
    for (size_t task_idx = 0; task_idx < NumTasks(); ++task_idx) {
      func(Range(task_idx));
    }
  }

  template <typename Func>
  void VisitFirst(const Func& func) const {
    func(Range(0));
  }

  template <typename Func>
  void VisitRemaining(const Func& func) const {
    for (size_t task_idx = 1; task_idx < NumTasks(); ++task_idx) {
      func(Range(task_idx));
    }
  }

 private:
  IndexRange range_;
  uint32_t task_size_;
  uint32_t num_tasks_;
};

// Starts with `max_size` and rounds DOWN to a multiple of `size_multiple`
// unless that would be zero. It is the caller's responsibility to choose
// `size_multiple` to avoid two heavily imbalanced tasks.
// Use when the number of tasks does not matter, but each must fit into caches.
static inline IndexRangePartition MaxSizePartition(const IndexRange& range,
                                                   const size_t max_size,
                                                   const size_t size_multiple) {
  HWY_DASSERT(size_multiple != 0);
  size_t size = HWY_MIN(range.Num(), max_size);
  if (size > size_multiple) size = hwy::RoundDownTo(size, size_multiple);
  return IndexRangePartition(range, size);
}

// Up to `max_tasks` tasks, each rounded UP to `size_multiple`, unless that
// would be more than the range. Use when the number of tasks is known, e.g.
// one per ThreadPool worker.
static inline IndexRangePartition StaticPartition(const IndexRange& range,
                                                  const size_t max_tasks,
                                                  const size_t size_multiple) {
  HWY_DASSERT(max_tasks != 0);
  size_t size =
      hwy::RoundUpTo(hwy::DivCeil(range.Num(), max_tasks), size_multiple);
  size = HWY_MIN(size, range.Num());
  return IndexRangePartition(range, size);
}

// Parallel-for over a single range. This takes care of translating the task
// index to a range.
template <class Func>
void ParallelizeOneRange(const IndexRangePartition& get1, hwy::ThreadPool& pool,
                         const Func& func) {
  const size_t num_tasks = get1.NumTasks();
  pool.Run(0, num_tasks, [&](uint64_t task, size_t thread) {
    const IndexRange range1 = get1.Range(task);
    func(range1, thread);
  });
}

// Parallel-for over the Cartesian product of the two sets of ranges. This
// combines their indices into a single 'task' so they can be executed by one
// `pool.Run`, which increases the amount of work available to workers and
// reduces fork-join overhead vs. nested parallel-for loops. Calls `func` with
// the two ranges and the thread index within `pool`.
template <class Func>
void ParallelizeTwoRanges(const IndexRangePartition& get1,
                          const IndexRangePartition& get2,
                          hwy::ThreadPool& pool, const Func& func) {
  const hwy::Divisor div1(static_cast<uint32_t>(get1.NumTasks()));

  const size_t num_tasks = get1.NumTasks() * get2.NumTasks();
  pool.Run(0, num_tasks, [&](uint64_t task, size_t thread) {
    HWY_DASSERT(task < (uint64_t{1} << 32));
    const size_t idx2 = div1.Divide(static_cast<uint32_t>(task));
    const size_t idx1 = div1.Remainder(static_cast<uint32_t>(task));
    HWY_DASSERT(idx1 < get1.NumTasks());
    HWY_DASSERT(idx2 < get2.NumTasks());
    const IndexRange range1 = get1.Range(idx1);
    const IndexRange range2 = get2.Range(idx2);
    func(range1, range2, thread);
  });
}

// Calls `func(task, worker)` for each task in `[0, num_tasks)`. Parallelizes
// over clusters of ONE package, then within each cluster.
template <class Func>
void ParallelFor(size_t num_tasks, NestedPools& pools, const Func& func) {
  // Even if there are multiple packages, we only use the first.
  const size_t pkg_idx = 0;

  // If few tasks, run on a single cluster. Also avoids a bit of overhead if
  // there is only one cluster.
  hwy::ThreadPool& all_clusters = pools.AllClusters(pkg_idx);
  const size_t num_clusters = all_clusters.NumWorkers();
  hwy::ThreadPool& cluster = pools.Cluster(pkg_idx, 0);
  if (num_clusters == 1 || num_tasks <= cluster.NumWorkers()) {
    return cluster.Run(0, num_tasks, [&](uint64_t task, size_t thread) {
      func(task, thread);
    });
  }

  // Assign each cluster a sub-range.
  const IndexRangePartition ranges =
      StaticPartition(IndexRange(0, num_tasks), num_clusters, 1);
  ParallelizeOneRange(
      ranges, all_clusters,
      [&](const IndexRange& range, const size_t cluster_idx) {
        hwy::ThreadPool& cluster = pools.Cluster(pkg_idx, cluster_idx);
        const size_t cluster_base = cluster_idx * pools.MaxWorkersPerCluster();
        cluster.Run(range.begin(), range.end(),
                    [&](uint64_t task, size_t thread) {
                      func(task, cluster_base + thread);
                    });
      });
}

// As above, but for lightweight tasks. Uses only one pool.
template <class Func>
void SmallParallelFor(size_t num_tasks, NestedPools& pools, const Func& func) {
  // Even if there are multiple packages, we only use the first.
  const size_t pkg_idx = 0;

  pools.Pool(pkg_idx).Run(
      0, num_tasks, [&](uint64_t task, size_t thread) { func(task, thread); });
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_THREADING_H_
