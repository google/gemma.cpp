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

#include <atomic>
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

class PinningPolicy {
 public:
  explicit PinningPolicy(Tristate pin);

  bool Want() const { return want_pin_; }
  void NotifyFailed() { (void)any_error_.test_and_set(); }

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
};  // PinningPolicy

// Creates a hierarchy of thread pools according to `BoundedTopology`: one with
// a thread per enabled cluster (CCX/shared L3), and for each of those, the
// remaining enabled cores in that cluster.
//
// Note that we support spin waits, thus it is important for each thread to be
// responsive, hence we do not create more than one thread per enabled core.
// For example, when there are four clusters of 8 cores, `AllClusters` has the
// main thread plus three extras, each `Cluster` runs on one of `AllClusters`
// plus seven extras, for a total of 3 + (4*7) = 31 extras plus the main thread.
//
// Useful when there are tasks which should be parallelized by workers sharing a
// cache, or on the same NUMA node. In both cases, individual pools have lower
// barrier synchronization latency than one large pool. However, to utilize all
// cores, call sites will have to use nested parallel-for loops as in
// `HierarchicalParallelFor`. To allow switching modes easily, prefer using the
// `ParallelFor` abstraction in threading_context.h).
//
// Note that this was previously intended to use all cores, but we are now
// moving toward also allowing concurrent construction with subsets of cores.
class NestedPools {
 public:
  // Neither move nor copy.
  NestedPools() = delete;
  NestedPools(const NestedPools&) = delete;
  NestedPools& operator=(const NestedPools&) = delete;
  NestedPools(NestedPools&&) = delete;
  NestedPools& operator=(NestedPools&&) = delete;

  // Because cross-package latency is high, this interface assumes only one
  // package is used. The `skip_packages` argument to `BoundedTopology` selects
  // which package that is for this `NestedPools` instance.
  //
  // `max_threads` is the maximum number of threads to divide among all
  // clusters. This is more intuitive than a per-cluster limit for users who
  // may not be aware of the CPU topology. This should be zero (meaning no
  // further limits) if the caller has already set limits via `skip_*` or
  // `max_*` args passed to `ThreadingContext`.
  //
  // To ensure we do not create more threads than there are HW cores, which
  // would cause huge slowdowns when spinning, the `BoundedSlice` arguments
  // only impose upper bounds on the number of detected clusters rather than
  // defining the actual number of threads.
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

  size_t NumClusters() const { return clusters_.size(); }
  hwy::ThreadPool& AllClusters() { return *all_clusters_; }
  hwy::ThreadPool& Cluster(size_t cluster_idx) {
    HWY_DASSERT(cluster_idx < clusters_.size());
    return *clusters_[cluster_idx];
  }

  // Reasonably tight upper bounds for allocating thread-local storage (TLS).
  size_t MaxWorkersPerCluster() const { return max_workers_per_cluster_; }
  size_t MaxWorkers() const { return NumClusters() * MaxWorkersPerCluster(); }

  // For ShowConfig
  const char* PinString() const { return pin_string_; }

  // Returns a single pool on the given package: either one thread per cluster
  // if there is more than one, which maximizes available memory bandwidth, or
  // the first cluster, which is typically the whole package. For use by
  // callers that only have a single parallel-for.
  // DEPRECATED: use ParallelFor instead.
  hwy::ThreadPool& Pool(size_t pkg_idx = 0) {
    // Only one cluster: use its pool, typically a whole socket.
    if (NumClusters() == 1) return Cluster(0);
    // One worker per cluster to maximize bandwidth availability.
    return AllClusters();
  }

 private:
  void SetWaitMode(hwy::PoolWaitMode wait_mode) {
    all_clusters_->SetWaitMode(wait_mode);
    for (PoolPtr& cluster : clusters_) {
      cluster->SetWaitMode(wait_mode);
    }
  }

  PinningPolicy pinning_;
  bool all_pinned_;
  const char* pin_string_;

  // Must be freed after `clusters_` because it reserves threads which are
  // the main threads of `clusters_`.
  PoolPtr all_clusters_;
  std::vector<PoolPtr> clusters_;

  // Used by `PoolWorkerMapping`. This depends on the `max_threads` argument,
  // hence we can only compute this here, not in `BoundedTopology`.
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
                         hwy::pool::Caller caller, const Func& func) {
  const size_t num_tasks = get1.NumTasks();
  pool.Run(0, num_tasks, caller, [&](uint64_t task, size_t thread) {
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
                          hwy::ThreadPool& pool, hwy::pool::Caller caller,
                          const Func& func) {
  const hwy::Divisor div1(static_cast<uint32_t>(get1.NumTasks()));

  const size_t num_tasks = get1.NumTasks() * get2.NumTasks();
  pool.Run(0, num_tasks, caller, [&](uint64_t task, size_t thread) {
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

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_THREADING_H_
