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

// Allocator with support for sharding tensors across NUMA nodes.

#include <stddef.h>
#include <stdint.h>

#include <functional>
// IWYU pragma: begin_exports
#include <memory>  // std::unique_ptr

#include "util/basics.h"
#include "util/topology.h"
#include "hwy/base.h"
// IWYU pragma: end_exports

namespace gcpp {

// Custom deleter for types without a dtor, but where the deallocation requires
// state, e.g. a lambda with *by-value* capture.
class DeleterFunc {
 public:
  // `MatOwnerT` requires this to be default-constructible.
  DeleterFunc() = default;

  template <class Closure>
  DeleterFunc(const Closure& free_closure) : free_func_(free_closure) {}

  template <typename T>
  void operator()(T* p) const {
    free_func_(const_cast<hwy::RemoveConst<T>*>(p));
  }

 private:
  std::function<void(void*)> free_func_;
};

// Wrapper that also calls the destructor for each element being deallocated.
class DeleterDtor {
 public:
  DeleterDtor() {}
  DeleterDtor(size_t num, DeleterFunc free) : num_(num), free_(free) {}

  template <typename T>
  void operator()(T* p) const {
    for (size_t i = 0; i < num_; ++i) {
      p[i].~T();
    }
    free_(p);
  }

 private:
  size_t num_;
  DeleterFunc free_;
};

// Unique (move-only) pointer to aligned POD T, which can be an array or class.
template <typename T>
using AlignedPtr = std::unique_ptr<T, DeleterFunc>;
// Unique (move-only) pointer to an aligned array of non-POD T.
template <typename T>
using AlignedClassPtr = std::unique_ptr<T, DeleterDtor>;

// Both allocation, binding, and row accessors depend on the sizes of memory
// pages and cache lines. To avoid having to pass `Allocator&` everywhere, we
// wrap this in a singleton. A monostate requires explicit initialization,
// which we prefer to avoid because there are many main() functions.
class Allocator {
 public:
  // Must be called at least once before any other function. Not thread-safe,
  // hence only call this from the main thread.
  Allocator(const BoundedTopology& topology, bool enable_bind);

  // Bytes per cache line, or a reasonable guess if unknown. Used to choose
  // ranges such that there will be no false sharing.
  size_t LineBytes() const { return line_bytes_; }
  // Upper bound on `LineBytes()`, for stack allocations.
  static constexpr size_t MaxLineBytes() { return 256; }
  // Bytes per full vector. Used to compute loop steps.
  size_t VectorBytes() const { return vector_bytes_; }
  // Work granularity that avoids false sharing and partial vectors.
  // = HWY_MAX(LineBytes(), VectorBytes())
  size_t StepBytes() const { return step_bytes_; }

  // File size multiple required for memory mapping. Also used when binding
  // memory to NUMA nodes (see `BindB/BindC`).
  size_t BasePageBytes() const { return base_page_bytes_; }

  // Desired allocator alignment: Either StepBytes, or BasePageBytes if NUMA.
  size_t QuantumBytes() const { return quantum_bytes_; }

  // L1 and L2 are typically per core.
  size_t L1Bytes() const { return l1_bytes_; }
  size_t L2Bytes() const { return l2_bytes_; }
  // Clusters often share an L3. We return the total size per package.
  size_t L3Bytes() const { return l3_bytes_; }

  size_t TotalMiB() const { return total_mib_; }
  size_t FreeMiB() const;

  // Returns byte pointer aligned to `QuantumBytes()`, without calling
  // constructors nor destructors on deletion. Type-erased so this can be
  // implemented in `allocator.cc` and called by `MatOwner`.
  AlignedPtr<uint8_t[]> AllocBytes(size_t bytes) const;

  // Returns pointer aligned to `QuantumBytes()`, without calling constructors
  // nor destructors on deletion.
  template <typename T>
  AlignedPtr<T[]> Alloc(size_t num) const {
    const size_t bytes = num * sizeof(T);
    // Fail if the `bytes = num * sizeof(T)` computation overflowed.
    HWY_ASSERT(bytes / sizeof(T) == num);

    AlignedPtr<uint8_t[]> p8 = AllocBytes(bytes);
    return AlignedPtr<T[]>(HWY_RCAST_ALIGNED(T*, p8.release()),
                           p8.get_deleter());
  }

  // Same as Alloc, but calls constructor(s) with `args` and the deleter will
  // call destructor(s).
  template <typename T, class... Args>
  AlignedClassPtr<T> AllocClasses(size_t num, Args&&... args) const {
    const size_t bytes = num * sizeof(T);
    // Fail if the `bytes = num * sizeof(T)` computation overflowed.
    HWY_ASSERT(bytes / sizeof(T) == num);

    AlignedPtr<uint8_t[]> p8 = AllocBytes(bytes);
    T* p = HWY_RCAST_ALIGNED(T*, p8.release());
    for (size_t i = 0; i < num; ++i) {
      new (p + i) T(std::forward<Args>(args)...);
    }
    return AlignedClassPtr<T>(p, DeleterDtor(num, p8.get_deleter()));
  }

  // Returns whether `BindMemory` can/should be called, i.e. we have page-level
  // control over memory placement and multiple packages and NUMA nodes.
  bool ShouldBind() const { return should_bind_; }

  // Attempts to move(!) `[p, p + bytes)` to the given NUMA node, which is
  // typically `BoundedTopology::GetCluster(package_idx, cluster_idx).node`.
  // Writes zeros to SOME of the memory. Only call if `ShouldBind()`.
  // `p` and `bytes` must be multiples of `QuantumBytes()`.
  bool BindMemory(void* p, size_t bytes, size_t node) const;

 private:
  size_t line_bytes_;
  size_t vector_bytes_;
  size_t step_bytes_;
  size_t base_page_bytes_;
  size_t quantum_bytes_;

  size_t l1_bytes_ = 0;
  size_t l2_bytes_ = 0;
  size_t l3_bytes_ = 0;

  size_t total_mib_;

  bool should_bind_ = false;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_ALLOCATOR_H_
