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

#ifndef THIRD_PARTY_GEMMA_CPP_UTIL_ALLOCATOR_H_
#define THIRD_PARTY_GEMMA_CPP_UTIL_ALLOCATOR_H_

#include <stddef.h>
#include <stdint.h>

#include <cstdlib>  // std::aligned_alloc / _aligned_malloc

// IWYU pragma: begin_exports
#include "util/basics.h"
#include "util/threading.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
// IWYU pragma: end_exports

#ifndef GEMMA_NUMA
// The check below requires two #if, hence start with 0 and redefine to 1.
#define GEMMA_NUMA 0

// To avoid a dependency on libnuma, use syscalls directly. We require six
// arguments, which has been supported by glibc since around 2010.
#if defined(__GLIBC__) && defined(__GLIBC_PREREQ)
#if HWY_OS_LINUX && __GLIBC_PREREQ(2, 11)
#undef GEMMA_NUMA
#define GEMMA_NUMA 1
#endif
#endif

#endif  // GEMMA_NUMA

namespace gcpp {

using ByteStorageT = hwy::AlignedFreeUniquePtr<uint8_t[]>;

template <typename T>
ByteStorageT AllocateSizeof() {
  return hwy::AllocateAligned<uint8_t>(sizeof(T));
}

// Stateful in order to know whether to bind to NUMA nodes. `Monostate` for
// convenience - avoids passing around a reference.
class Allocator {
 public:
  static void Init(const BoundedTopology& topology) {
    bytes_per_page_ = DetectPageSize();
    HWY_ASSERT(bytes_per_page_ <= (4 << 20));

    // NUMA only makes sense if:
    // - the page size is known and 'reasonably small', preferably less than
    //   a fraction of MatMul row/col sizes, which for 27B are up to 144 KiB.
    // - we successfully detected topology and there are multiple nodes;
    // - there are multiple packages, because we shard by package_idx.
    use_numa_ = (bytes_per_page_ != 0 && bytes_per_page_ <= 16 * 1024) &&
                topology.NumNodes() > 1 && topology.NumPackages() > 1;
    // TODO: remove once tensors are page-aligned.
    use_numa_ = false;
    fprintf(stderr, "Warning: disabling use_numa_\n");

    alignment_ = use_numa_ ? bytes_per_page_ : HWY_ALIGNMENT;
  }

  static bool UseNUMA() { return use_numa_; }

  // BindTensor requires row pointers and lengths be a multiple of this.
  static size_t Alignment() { return alignment_; }

  template <typename T>
  static hwy::AlignedFreeUniquePtr<T[]> Alloc(size_t num) {
    // For non-NUMA, use the Highway allocator because it defends against 2k
    // aliasing.
    if (!use_numa_) return hwy::AllocateAligned<T>(num);

    constexpr size_t kSize = sizeof(T);
    // Ensure the `bytes = num * kSize` computation did not overflow.
    constexpr bool kIsPow2 = (kSize & (kSize - 1)) == 0;
    constexpr size_t kBits = hwy::detail::ShiftCount(kSize);
    static_assert(!kIsPow2 || (1ull << kBits) == kSize, "ShiftCount has a bug");
    const size_t bytes = kIsPow2 ? num << kBits : num * kSize;
    const size_t check = kIsPow2 ? bytes >> kBits : bytes / kSize;
    if (check != num) {
      return hwy::AlignedFreeUniquePtr<T[]>();  // overflowed
    }

    // AlignedFreeUniquePtr has a deleter that can call an arbitrary `free`, but
    // with an extra opaque pointer, which we discard via `call_free`.
#if defined(__ANDROID_API__) && __ANDROID_API__ < 28
    const auto call_free = [](void* ptr, void*) { std::free(ptr); };
    void* mem = nullptr;
    int err = posix_memalign(&mem, Alignment(), bytes);
    HWY_ASSERT(err == 0);
    T* p = static_cast<T*>(mem);
#elif HWY_OS_WIN
    const auto call_free = [](void* ptr, void*) { _aligned_free(ptr); };
    T* p = static_cast<T*>(_aligned_malloc(bytes, Alignment()));
#else
    const auto call_free = [](void* ptr, void*) { std::free(ptr); };
    T* p = static_cast<T*>(std::aligned_alloc(Alignment(), bytes));
#endif
    return hwy::AlignedFreeUniquePtr<T[]>(
        p, hwy::AlignedFreer(call_free, nullptr));
  }

 private:
  static size_t DetectPageSize();

  // Required for BindMemory. Usually 4K, but can differ on Arm.
  static size_t bytes_per_page_;
  static bool use_numa_;
  static size_t alignment_;
};

// For shorter arguments to the StaticPartitionRowsAndCols functor.
struct TaskLocation {
  TaskLocation(size_t node, size_t package_idx, hwy::ThreadPool& cluster,
               size_t worker_offset)
      : node(node),
        package_idx(package_idx),
        cluster(cluster),
        worker_offset(worker_offset) {}
  size_t node;
  size_t package_idx;
  hwy::ThreadPool& cluster;
  const size_t worker_offset;
};

// Used in MatMul and allocator.h. Defined here because it depends on
// Allocator::Alignment().
template <class Func>
void StaticPartitionRowsAndCols(NestedPools& nested, Extents2D extents,
                                size_t bytes_per_element, const Func& func) {
  // Both rows and cols must be a multiple of the alignment to avoid
  // touching remote pages.
  const size_t multiple = Allocator::Alignment() / bytes_per_element;

  // Static partitioning of columns across packages. We assume that column
  // sharding is more expensive, hence we distribute columns across packages,
  // of which there are usually only one or two. For MatMul, the final result is
  // the sum of each package's partial dot products.
  hwy::ThreadPool& all_packages = nested.AllPackages();
  const size_t num_packages = all_packages.NumWorkers();
  const size_t cols_per_package =
      hwy::RoundUpTo(hwy::DivCeil(extents.cols, num_packages), multiple);
  const size_t col_tasks = hwy::DivCeil(extents.cols, cols_per_package);
  HWY_ASSERT(col_tasks <= num_packages);
  all_packages.Run(
      0, col_tasks, [&](uint64_t package_idx, size_t package_thread) {
        HWY_ASSERT(package_idx == package_thread);  // one task per worker
        const size_t col_begin = package_idx * cols_per_package;
        const Range1D col_range =
            MakeRange1D(col_begin, extents.cols, cols_per_package);

        // Static partitioning of rows across the package's clusters. We assume
        // that row sharding is cheaper. In MatMul, results can indeed be
        // computed independently for each row of B.
        hwy::ThreadPool& all_clusters = nested.AllClusters(package_idx);
        const size_t num_clusters = all_clusters.NumWorkers();
        const size_t rows_per_cluster =
            hwy::RoundUpTo(hwy::DivCeil(extents.rows, num_clusters), multiple);
        const size_t row_tasks = hwy::DivCeil(extents.rows, rows_per_cluster);
        HWY_ASSERT(row_tasks <= num_clusters);
        all_clusters.Run(
            0, row_tasks, [&](uint64_t cluster_idx, size_t cluster_thread) {
              HWY_ASSERT(cluster_idx == cluster_thread);  // one task per worker

              // For binding to NUMA node.
              const size_t node = nested.Node(package_idx, cluster_idx);
              // Older CPUs that predate chiplets typically have only one
              // cluster, so callers should also parallelize using this
              // per-cluster pool.
              hwy::ThreadPool& cluster =
                  nested.Cluster(package_idx, cluster_idx);
              // This plus the worker from `cluster->Run` is the TLS index.
              const size_t worker_offset =
                  nested.WorkerOffset(package_idx, cluster_idx);

              const size_t row_begin = cluster_idx * rows_per_cluster;
              const Range1D row_range =
                  MakeRange1D(row_begin, extents.rows, rows_per_cluster);

              func(Range2D(row_range, col_range),
                   TaskLocation(node, package_idx, cluster, worker_offset));
            });
      });
}

void BindTensor(NestedPools& nested, size_t rows, size_t cols,
                size_t bytes_per_col, void* ptr);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_ALLOCATOR_H_
