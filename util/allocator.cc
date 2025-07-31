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

#include "util/allocator.h"

#include <stdint.h>
#include <stdio.h>

#include "util/basics.h"  // MaybeCheckInitialized
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/topology.h"
#include "hwy/per_target.h"  // VectorBytes

// To avoid a dependency on libnuma, use syscalls directly. We require six
// arguments, which has been supported by glibc since around 2010.
#if defined(__GLIBC__) && defined(__GLIBC_PREREQ)
#if HWY_OS_LINUX && __GLIBC_PREREQ(2, 11)
#define GEMMA_LINUX_SYSCALL6
#endif
#endif

#ifndef GEMMA_BIND  // allow override
// OSes will generally do the right thing when threads allocate their own
// working memory. However, matmul's B and C matrices are preferably sharded
// across NUMA nodes. To simplify the matrix representation, we prefer a
// single allocation. This requires page-level control over the memory layout,
// which Linux provides via `move_pages`, but Windows does not.
#if defined(GEMMA_LINUX_SYSCALL6) && !defined(__ANDROID_API__)
#define GEMMA_BIND 1
#else
#define GEMMA_BIND 0
#endif
#endif  // GEMMA_BIND

#if GEMMA_BIND && HWY_OS_LINUX
#include <atomic>

#include "hwy/contrib/thread_pool/futex.h"
#endif

#if HWY_OS_LINUX
#include <unistd.h>  // sysconf
#if GEMMA_BIND
// `move_pages` requires anonymous/private mappings, hence mmap.
#include <sys/mman.h>
#include <sys/syscall.h>

#include <cerrno>
#include <vector>
#endif  // GEMMA_BIND
#elif HWY_OS_WIN
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef VC_EXTRALEAN
#define VC_EXTRALEAN
#endif
#include <Windows.h>
#elif HWY_OS_APPLE
#include <sys/sysctl.h>
#endif  // HWY_OS_LINUX

namespace gcpp {
namespace {

size_t DetectLineBytes() {
  if (const hwy::Cache* caches = hwy::DataCaches()) {
    // Might not have an L3.
    return HWY_MAX(caches[2].bytes_per_line, caches[3].bytes_per_line);
  } else {
    return HWY_ALIGNMENT;
  }
}

size_t DetectPageSize() {
#if HWY_OS_LINUX
  const long ret = sysconf(_SC_PAGESIZE);  // NOLINT(runtime/int)
  HWY_ASSERT(ret != -1);
  const size_t page_bytes = static_cast<size_t>(ret);
  HWY_ASSERT(page_bytes <= (4 << 20));
  return page_bytes;
#elif HWY_OS_WIN
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return info.dwPageSize;
#elif HWY_OS_APPLE
  uint64_t data = 0;
  size_t len = sizeof(data);
  HWY_ASSERT(sysctlbyname("vm.pagesize", &data, &len, nullptr, 0) == 0);
  return data;
#else
  return 0;
#endif
}

size_t DetectTotalMiB(size_t page_bytes) {
  (void)page_bytes;
#if HWY_OS_LINUX
  const long ret = sysconf(_SC_PHYS_PAGES);  // NOLINT(runtime/int)
  HWY_ASSERT(ret != -1);
  return static_cast<size_t>(ret) * page_bytes >> 20;
#elif HWY_OS_WIN
  MEMORYSTATUSEX ms = {sizeof(MEMORYSTATUSEX)};
  HWY_ASSERT(GlobalMemoryStatusEx(&ms) != 0);
  return ms.ullTotalPhys >> 20;
#elif HWY_OS_APPLE
  int mib[2] = {CTL_HW, HW_MEMSIZE};
  uint64_t data = 0;
  size_t len = sizeof(data);
  HWY_ASSERT(sysctl(mib, sizeof(mib) / sizeof(*mib), &data, &len, nullptr, 0) ==
             0);
  return data >> 20;
#else
#error "Port"
#endif
}

}  // namespace

Allocator::Allocator(const BoundedTopology& topology, bool enable_bind) {
  line_bytes_ = DetectLineBytes();
  // Ensure MaxLineBytes() is an upper bound.
  HWY_ASSERT(MaxLineBytes() >= LineBytes());

  vector_bytes_ = hwy::VectorBytes();

  step_bytes_ = HWY_MAX(line_bytes_, vector_bytes_);
  base_page_bytes_ = DetectPageSize();
  quantum_bytes_ = step_bytes_;  // may overwrite below

  const BoundedTopology::Cluster& cluster = topology.GetCluster(0, 0);
  if (const hwy::Cache* caches = hwy::DataCaches()) {
    l1_bytes_ = caches[1].size_kib << 10;
    l2_bytes_ = caches[2].size_kib << 10;
    l3_bytes_ = (caches[3].size_kib << 10) * caches[3].cores_sharing;
  } else {  // Unknown, make reasonable assumptions.
    l1_bytes_ = 32 << 10;
    l2_bytes_ = (cluster.PrivateKiB() ? cluster.PrivateKiB() : 256) << 10;
  }
  if (l3_bytes_ == 0) {
    l3_bytes_ = (cluster.SharedKiB() ? cluster.SharedKiB() : 1024) << 10;
  }

  total_mib_ = DetectTotalMiB(base_page_bytes_);

  // Prerequisites for binding:
  // - supported by the OS (currently Linux only),
  // - the page size is known and 'reasonably small', preferably less than
  //   a fraction of MatMul row/col sizes, which for 27B are up to 144 KiB.
  // - we successfully detected topology and there are multiple nodes;
  // - there are multiple packages, because we shard by package_idx.
  if constexpr (GEMMA_BIND) {
    if ((base_page_bytes_ != 0 && base_page_bytes_ <= 16 * 1024) &&
        topology.NumNodes() > 1 && topology.NumPackages() > 1) {
      if (enable_bind) {
        // Ensure pages meet the alignment requirements of `AllocBytes`.
        HWY_ASSERT(base_page_bytes_ >= quantum_bytes_);
        quantum_bytes_ = base_page_bytes_;
        should_bind_ = true;
      } else {
        HWY_WARN(
            "Multiple sockets but binding disabled. This reduces speed; "
            "set --bind 1 to avoid this warning.");
      }
    }
  }
}

size_t Allocator::FreeMiB() const {
#if HWY_OS_LINUX
  const long ret = sysconf(_SC_AVPHYS_PAGES);  // NOLINT(runtime/int)
  HWY_ASSERT(ret != -1);
  return static_cast<size_t>(ret) * base_page_bytes_ >> 20;
#elif HWY_OS_WIN
  MEMORYSTATUSEX ms = {sizeof(MEMORYSTATUSEX)};
  HWY_ASSERT(GlobalMemoryStatusEx(&ms) != 0);
  return ms.ullAvailVirtual >> 20;
#elif HWY_OS_APPLE
  uint64_t free = 0, inactive = 0, speculative = 0;
  size_t len = sizeof(free);
  sysctlbyname("vm.page_free_count", &free, &len, nullptr, 0);
  sysctlbyname("vm.page_inactive_count", &inactive, &len, nullptr, 0);
  sysctlbyname("vm.page_speculative_count", &speculative, &len, nullptr, 0);
  return (free + inactive + speculative) * base_page_bytes_ >> 20;
#else
#error "Port"
#endif
}

AlignedPtr<uint8_t[]> Allocator::AllocBytes(size_t bytes) const {
  // If we are not binding, the Highway allocator is cheaper than `mmap`, and
  // defends against 2K aliasing.
  if (!should_bind_) {
    // Perf warning if Highway's alignment is less than we want.
    if (HWY_ALIGNMENT < QuantumBytes()) {
      HWY_WARN(
          "HWY_ALIGNMENT %d < QuantumBytes %zu: either vector or cache lines "
          "are huge, enable GEMMA_BIND and set --bind 1 to avoid this warning.",
          HWY_ALIGNMENT, QuantumBytes());
    }
    auto p = hwy::AllocateAligned<uint8_t>(bytes);
    // The `hwy::AlignedFreeUniquePtr` deleter is unfortunately specific to the
    // alignment scheme in aligned_allocator.cc and does not work for
    // already-aligned pointers as returned by `mmap`, hence we wrap the Highway
    // pointer in our own deleter.
    return AlignedPtr<uint8_t[]>(p.release(), DeleterFunc([](void* ptr) {
                                   hwy::FreeAlignedBytes(ptr, nullptr, nullptr);
                                 }));
  }

  // Binding, or large vector/cache line size: use platform-specific allocator.

#if HWY_OS_LINUX && !defined(__ANDROID_API__)
  // `move_pages` is documented to require an anonymous/private mapping or
  // `MAP_SHARED`. A normal allocation might not suffice, so we use `mmap`.
  // `Init` verified that the page size is a multiple of `QuantumBytes()`.
  const int prot = PROT_READ | PROT_WRITE;
  const int flags = MAP_ANONYMOUS | MAP_PRIVATE;
  const int fd = -1;
  void* p = mmap(0, bytes, prot, flags, fd, off_t{0});
  if (p == MAP_FAILED) p = nullptr;
  return AlignedPtr<uint8_t[]>(
      static_cast<uint8_t*>(p),
      DeleterFunc([bytes](void* ptr) { HWY_ASSERT(munmap(ptr, bytes) == 0); }));
#elif HWY_OS_WIN
  const size_t alignment = HWY_MAX(vector_bytes_, line_bytes_);
  return AlignedPtr<uint8_t[]>(
      static_cast<uint8_t*>(_aligned_malloc(bytes, alignment)),
      DeleterFunc([](void* ptr) { _aligned_free(ptr); }));
#else
  return AlignedPtr<uint8_t[]>(nullptr, DeleterFunc());
#endif
}

#if GEMMA_BIND && HWY_OS_LINUX

using Ret = long;          // NOLINT(runtime/int)
using UL = unsigned long;  // NOLINT(runtime/int)
static constexpr size_t ULBits = sizeof(UL) * 8;

// Calling via syscall avoids a dependency on libnuma.
struct SyscallWrappers {
  static Ret mbind(void* ptr, UL bytes, int mode, const UL* nodes, UL max_nodes,
                   unsigned flags) {
    MaybeCheckInitialized(nodes, hwy::DivCeil(max_nodes, ULBits) * sizeof(UL));
    return syscall(__NR_mbind, ptr, bytes, mode, max_nodes, max_nodes, flags);
  };

  static Ret move_pages(int pid, UL count, void** pages, const int* nodes,
                        int* status, int flags) {
    MaybeCheckInitialized(pages, count * sizeof(void*));
    MaybeCheckInitialized(nodes, count * sizeof(int));
    MaybeCheckInitialized(status, count * sizeof(int));
    return syscall(__NR_move_pages, pid, count, pages, nodes, status, flags);
  }

  static Ret get_mempolicy(int* mode, UL* nodes, UL max_node, void* addr,
                           unsigned flags) {
    return syscall(__NR_get_mempolicy, mode, nodes, max_node, addr, flags);
  }
};

// Returns the number of pages that are currently busy (hence not yet moved),
// and warns if there are any other reasons for not moving a page. Note that
// `move_pages` can return 0 regardless of whether all pages were moved.
size_t CountBusyPages(size_t num_pages, size_t node, void** pages,
                      const int* status) {
  size_t num_busy = 0;
  for (size_t i = 0; i < num_pages; ++i) {
    if (status[i] == -EBUSY) {
      ++num_busy;
    } else if (status[i] != static_cast<int>(node)) {
      static std::atomic_flag first = ATOMIC_FLAG_INIT;
      if (!first.test_and_set()) {
        HWY_WARN("Error %d moving pages[%zu]=%p to node %zu (errno %d).",
                 status[i], i, pages[i], node, errno);
      }
    }
  }
  return num_busy;
}

bool Allocator::BindMemory(void* ptr, size_t bytes, size_t node) const {
  HWY_DASSERT(should_bind_);
  constexpr size_t kMaxNodes = 1024;  // valid for x86/x64, and "enough"

  if constexpr (HWY_IS_DEBUG_BUILD) {
    // Ensure the requested `node` is allowed.
    UL nodes[kMaxNodes / 64] = {0};
    const unsigned flags = 4;  // MPOL_F_MEMS_ALLOWED
    HWY_ASSERT(SyscallWrappers::get_mempolicy(nullptr, nodes, kMaxNodes,
                                              nullptr, flags) == 0);
    HWY_ASSERT(nodes[node / 64] & (1ull << (node % 64)));
  }

  // Avoid mbind because it does not report why it failed, which is most likely
  // because pages are busy, in which case we want to know which.

  // `MPOL_MF_MOVE_ALL` requires cap sys_nice, which is not easy to set.
  const unsigned flags = 2;  // MPOL_MF_MOVE
  HWY_ASSERT(bytes % quantum_bytes_ == 0);
  const size_t num_pages = bytes / quantum_bytes_;
  std::vector<void*> pages;
  pages.reserve(num_pages);
  for (size_t i = 0; i < num_pages; ++i) {
    pages.push_back(static_cast<uint8_t*>(ptr) + i * quantum_bytes_);
    // Ensure the page is faulted in to prevent `move_pages` from failing,
    // because freshly allocated pages may be mapped to a shared 'zero page'.
    hwy::ZeroBytes(pages.back(), 8);
  }
  std::vector<int> nodes(num_pages, node);
  std::vector<int> status(num_pages, static_cast<int>(kMaxNodes));

  Ret ret = SyscallWrappers::move_pages(
      /*pid=*/0, num_pages, pages.data(), nodes.data(), status.data(), flags);
  if (ret < 0) {
    HWY_WARN("Failed to bind %p %zu to node %zu (errno %d) status %d.", ptr,
             bytes, node, errno, status[0]);
    return false;
  }

  const size_t num_busy =
      CountBusyPages(num_pages, node, pages.data(), status.data());
  if (HWY_UNLIKELY(num_busy != 0)) {
    // Trying again is usually enough to succeed.
    hwy::NanoSleep(5000);
    (void)SyscallWrappers::move_pages(
        /*pid=*/0, num_pages, pages.data(), nodes.data(), status.data(), flags);
    const size_t still_busy =
        CountBusyPages(num_pages, node, pages.data(), status.data());
    if (HWY_UNLIKELY(still_busy != 0)) {
      HWY_WARN("BindMemory: %zu pages still busy after retrying %zu.",
               still_busy, num_busy);
    }
  }
  return true;
}

#else
bool Allocator::BindMemory(void*, size_t, size_t) const { return false; }
#endif  // GEMMA_BIND && HWY_OS_LINUX

}  // namespace gcpp
