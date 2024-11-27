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

// For future NUMA support. TODO: use.
void BindMemory(void* ptr, size_t bytes, size_t node);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_ALLOCATOR_H_
