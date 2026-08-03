// Copyright 2026 DeepMind Technologies Limited.
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

// OneDNN matmul-primitive integration for MatMul via the threadpool runtime.
// Enabled at compile time via GEMMA_ONEDNN_MATMUL=1 (Bazel: --define
// gemma_onednn_matmul=1).
//
// This is a different integration than ops/brgemm.h (GEMMA_ONEDNN_BRGEMM):
//  - BRGeMM uses oneDNN's low-level ukernel API with oneDNN built SEQ; gemma.cpp
//    drives all parallelism.
//  - Here we use the high-level dnnl::matmul primitive with oneDNN built for the
//    THREADPOOL runtime, so oneDNN picks the kernel and parallelizes *internally*
//    by calling back into gemma.cpp's hwy::ThreadPool through the adapter below.

#ifndef THIRD_PARTY_GEMMA_CPP_OPS_ONEDNN_MATMUL_H_
#define THIRD_PARTY_GEMMA_CPP_OPS_ONEDNN_MATMUL_H_

#include <stddef.h>
#include <stdint.h>

// opt-in
#ifndef GEMMA_ONEDNN_MATMUL
#define GEMMA_ONEDNN_MATMUL 0
#endif  // GEMMA_ONEDNN_MATMUL

#if GEMMA_ONEDNN_MATMUL && GEMMA_ONEDNN_BRGEMM
#error \
    "GEMMA_ONEDNN_MATMUL and GEMMA_ONEDNN_BRGEMM are mutually exclusive: " \
    "oneDNN's CPU runtime (SEQ for BRGeMM vs THREADPOOL for the matmul " \
    "primitive) is a whole-library compile-time choice. Enable only one."
#endif

#if GEMMA_ONEDNN_MATMUL

#include <functional>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_threadpool.hpp"
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"
#include "util/threading_context.h"
#include "util/zones.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

inline dnnl::engine& OneDnnEngine() {
  static dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  return engine;
}

inline void SetOneDnnMaxConcurrency(int num_threads) {
  dnnl_threadpool_interop_set_max_concurrency(num_threads);
}

// Adapts one gemma.cpp cluster ThreadPool to oneDNN's threadpool interface.
class HwyThreadPoolAdapter : public dnnl::threadpool_interop::threadpool_iface {
 public:
  HwyThreadPoolAdapter(ThreadingContext& ctx, size_t cluster_idx,
                       hwy::pool::Caller caller)
      : pool_(ctx.pools.Cluster(cluster_idx)), caller_(caller) {}

  int get_num_threads() const override {
    return static_cast<int>(pool_.NumWorkers());
  }

  // True when the calling thread is already executing inside our parallel_for.
  // oneDNN uses this to avoid nesting parallel regions
  bool get_in_parallel() const override { return InParallel(); }

  // Synchronous flags (0): parallel_for must block.
  uint64_t get_flags() const override { return 0; }

  void parallel_for(int n,
                    const std::function<void(int, int)>& fn) override {
    if (n <= 0) return;
    // Run inline if there is nothing to fan out to, or if we are already inside
    // a parallel region on this pinned pool
    if (n == 1 || pool_.NumWorkers() <= 1 || InParallel()) {
      for (int i = 0; i < n; ++i) fn(i, n);
      return;
    }
    pool_.Run(0, static_cast<uint64_t>(n), caller_,
              [&](uint64_t task, size_t /*worker*/) {
                InParallelGuard guard;
                fn(static_cast<int>(task), n);
              });
  }

  void wait() override {}

 private:
  static bool& InParallel() {
    static thread_local bool in_parallel = false;
    return in_parallel;
  }

  struct InParallelGuard {
    InParallelGuard() { InParallel() = true; }
    ~InParallelGuard() { InParallel() = false; }
  };

  hwy::ThreadPool& pool_;
  hwy::pool::Caller caller_;
};

// ---------------------------------------------------------------------------
// Reordered-weights cache keyed by B pointer. Lives in
// `MatMulEnv::PerCluster::onednn_weights` so that independent instances within
// one binary do not share entries, and so that concurrent per-cluster `MatMul`
// calls do not insert into one map. The caller separately checks the stored
// layout against the one the primitive wants.
struct OneDnnWeightsKey {
  uintptr_t B_ptr;
  bool operator==(const OneDnnWeightsKey& o) const {
    return B_ptr == o.B_ptr;
  }
};

struct OneDnnWeightsKeyHash {
  size_t operator()(const OneDnnWeightsKey& k) const {
    size_t h = 14695981039346656037ULL;
    h = (h ^ k.B_ptr) * 1099511628211ULL;
    return h;
  }
};

struct OneDnnWeightsEntry {
  dnnl::memory packed;
};

using OneDnnWeightsCache =
    std::unordered_map<OneDnnWeightsKey, OneDnnWeightsEntry,
                       OneDnnWeightsKeyHash>;

}  // namespace gcpp

#endif  // GEMMA_ONEDNN_MATMUL

#endif  // THIRD_PARTY_GEMMA_CPP_OPS_ONEDNN_MATMUL_H_
