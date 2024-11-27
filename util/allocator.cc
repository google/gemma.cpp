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

#include <stdio.h>

#include <vector>

#include "util/basics.h"  // MaybeCheckInitialized

#if GEMMA_NUMA
#if HWY_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif HWY_OS_LINUX
#include <sys/syscall.h>

#include <cerrno>
#endif  // HWY_OS_*
#endif  // GEMMA_NUMA

namespace gcpp {

/*static*/ size_t Allocator::bytes_per_page_;
/*static*/ bool Allocator::use_numa_;
/*static*/ size_t Allocator::alignment_;

/*static*/ size_t Allocator::DetectPageSize() {
#if HWY_OS_WIN
  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);
  return sys_info.dwPageSize;
#elif HWY_OS_LINUX
  return sysconf(_SC_PAGESIZE);
#else
  return 0;
#endif
}

#if GEMMA_NUMA && HWY_OS_LINUX

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
};

size_t CountBusyPages(size_t num_pages, size_t node, void** pages,
                      const int* status) {
  // Return value 0 does not actually guarantee all pages were moved.
  size_t num_busy = 0;
  for (size_t i = 0; i < num_pages; ++i) {
    if (status[i] == -EBUSY) {
      ++num_busy;
      // Touch
      hwy::ZeroBytes(pages[i], 8);
    } else if (status[i] != static_cast<int>(node)) {
      fprintf(stderr, "Error %d moving pages[%zu]=%p to node %zu (errno %d)\n",
              status[i], i, pages[i], node, errno);
    }
  }
  return num_busy;
}

// Attempts to move(!) memory to the given NUMA node, typically obtained from
// `BoundedTopology::GetCluster(package_idx, cluster_idx).node`. Using `mbind`
// directly is easier than calling libnuma's `numa_move_pages`, which requires
// an array of pages. Note that `numa_tonode_memory` is insufficient because
// it does not specify the `MPOL_MF_MOVE` flag, so it only sets the policy,
// which means it would have to be called before pages are faulted in, but
// `aligned_allocator.h` modifies the first bytes for its bookkeeping.
// May overwrite some of the memory with zeros.
void BindMemory(void* ptr, size_t bytes, size_t node) {
  constexpr size_t kMaxNodes = 1024;  // valid for x86/x64, and "enough"
  // Avoid mbind because it does not report why it failed, which is most likely
  // because pages are busy, in which case we want to know which.
#if 0
  // nodemask with only the given node set.
  UL nodes[hwy::DivCeil(kMaxNodes, ULBits)] = {};
  nodes[node / ULBits] = 1ULL << (node % ULBits);

  const int mode = 2;        // MPOL_BIND
  const unsigned flags = 3;  // MPOL_MF_MOVE | MPOL_MF_STRICT
  const int ret =
      SyscallWrappers::mbind(ptr, bytes, mode, nodes, kMaxNodes, flags);
  if (ret != 0) {
    fprintf(stderr, "Failed to bind %p %zu to node %zu (errno %d)\n", ptr,
            bytes, node, errno);
  }
#elif 1
  const unsigned flags = 2;  // MPOL_MF_MOVE
  const size_t bytes_per_page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  HWY_ASSERT(bytes % bytes_per_page == 0);
  const size_t num_pages = bytes / bytes_per_page;
  std::vector<void*> pages;
  pages.reserve(num_pages);
  for (size_t i = 0; i < num_pages; ++i) {
    pages.push_back(static_cast<uint8_t*>(ptr) + i * bytes_per_page);
  }
  std::vector<int> nodes(num_pages, node);
  std::vector<int> status(num_pages, static_cast<int>(kMaxNodes));
  Ret ret = SyscallWrappers::move_pages(
      /*pid=*/0, num_pages, pages.data(), nodes.data(), status.data(), flags);
  size_t num_busy =
      CountBusyPages(num_pages, node, pages.data(), status.data());
  if (num_busy != 0) {
    // Try again
    ret = SyscallWrappers::move_pages(
        /*pid=*/0, num_pages, pages.data(), nodes.data(), status.data(), flags);
    const size_t num_busy_before = num_busy;
    num_busy = CountBusyPages(num_pages, node, pages.data(), status.data());
    fprintf(
        stderr,
        "second try still %zu busy, was %zu. 2nd ret %d status %d %d %d %d\n",
        num_busy, num_busy_before, static_cast<int>(ret), status[0], status[1],
        status[2], status[3]);
  }

  if (ret < 0) {
    fprintf(stderr,
            "Failed to bind %p %zu to node %zu (errno %d) status %d %d\n", ptr,
            bytes, node, errno, status[0], status[1]);
  }
#endif
}

#else
// TODO: support other OSes.
void BindMemory(void*, size_t, size_t) {}
#endif  // GEMMA_NUMA && HWY_OS_LINUX

}  // namespace gcpp
