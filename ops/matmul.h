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

#ifndef THIRD_PARTY_GEMMA_CPP_OPS_MATMUL_H_
#define THIRD_PARTY_GEMMA_CPP_OPS_MATMUL_H_

// Non-SIMD part of MatMul: parallelization, allocation, and autotuning.

#include <stddef.h>
#include <stdint.h>

#include <vector>

// IWYU pragma: begin_exports
#include "compression/compress.h"
#include "util/allocator.h"
#include "util/basics.h"
#include "util/threading.h"
#include "hwy/aligned_allocator.h"  // Span
#include "hwy/base.h"
#include "hwy/bit_set.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"
// IWYU pragma: end_exports

namespace gcpp {

// The MatMul result C[r,c] is Dot(A.Row(r), B.Col(c)). To reduce the number of
// loads, we reuse the same A row for several B columns, which are also loaded
// once for several rows of C. Thus we produce one 'tile' of C at a time of
// dimensions `mr (<= kMaxMR)` x `kNR`. To keep FMA units busy, this should be
// at least the product of the FMA latency (3..5) times the throughput (2).
// This and `mr` are limited by the number of registers, which is generally
// 32 but 16 for AVX2. `kNR` == 4 enables the `StoreInterleaved4` transpose in
// `MMAddHorizontalSumsIntoPartial`. We ensure `C.Cols() % kNR == 0`.
constexpr size_t kNR = 4;

class MMParallel {
 public:
  static constexpr size_t kMaxPackages = 4;

  MMParallel(NestedPools& pools) : pools_(pools) {
    HWY_DASSERT(pools_.NumPackages() <= kMaxPackages);
  }

  // Used by tests.
  NestedPools& Pools() { return pools_; }

  // Initial static partitioning of B rows across packages.
  IndexRangePartition RangesOfNP(size_t max_packages, size_t N,
                                 size_t sizeof_TC, size_t nr) const;

  // For `BindB` and `BindC`.
  size_t Node(size_t pkg_idx) const {
    return pools_.Topology().GetCluster(pkg_idx, 0).Node();
  }

  // Calls `func(pkg_idx)` for each package in parallel.
  template <class Func>
  void ForPkg(const size_t max_packages, const Func& func) {
    pools_.AllPackages().Run(0, HWY_MIN(max_packages, pools_.NumPackages()),
                             [&](uint64_t task, size_t pkg_idx) {
                               HWY_DASSERT(task == pkg_idx);
                               (void)task;
                               func(pkg_idx);
                             });
  }

  // Cluster/CCX-aware parallel-for over B rows in `range_np`. `nx_multiple` is
  // the granularity of per-cluster tasks. Calls `func(worker_range)`.
  template <class Func>
  void ForNP(const IndexRange& range_np, size_t nx_multiple, size_t inner_tasks,
             size_t pkg_idx, const Func& func) {
    HWY_DASSERT(1 <= inner_tasks && inner_tasks <= 4);
    // Single cluster: parallel-for over static partition of `range_np`.
    hwy::ThreadPool& all_clusters = pools_.AllClusters(pkg_idx);
    const size_t num_clusters = all_clusters.NumWorkers();
    if (num_clusters == 1) {
      hwy::ThreadPool& cluster = pools_.Cluster(pkg_idx, 0);
      const IndexRangePartition worker_ranges = StaticPartition(
          range_np, cluster.NumWorkers() * inner_tasks, nx_multiple);
      return ParallelizeOneRange(
          worker_ranges, cluster,
          [&](const IndexRange& worker_range, size_t /*thread*/) {
            func(worker_range);
          });
    }

    // Assign each cluster a sub-range of `range_np` (typically hundreds).
    const IndexRangePartition nx_ranges =
        StaticPartition(range_np, num_clusters, nx_multiple);
    ParallelizeOneRange(
        nx_ranges, all_clusters,
        [&](const IndexRange& nx_range, const size_t cluster_idx) {
          hwy::ThreadPool& cluster = pools_.Cluster(pkg_idx, cluster_idx);
          // Parallel-for over sub-ranges of `cluster_range` within the cluster.
          const IndexRangePartition worker_ranges = StaticPartition(
              nx_range, cluster.NumWorkers() * inner_tasks, nx_multiple);
          ParallelizeOneRange(worker_ranges, cluster,
                              [&](const IndexRange& worker_range,
                                  size_t /*thread*/) { func(worker_range); });
        });
  }

  // Cluster/CCX-aware parallel-for over blocks (separate subranges of A and B
  // rows). Calls `func(range_mc, range_nc)`.
  template <class Func>
  void ForRangesMC_NC(const IndexRangePartition& ranges_mc,
                      const IndexRangePartition& ranges_nc, size_t pkg_idx,
                      const Func& func) {
    hwy::ThreadPool& all_clusters = pools_.AllClusters(pkg_idx);
    // `all_clusters` is a pool with one worker per cluster in a package.
    const size_t num_clusters = all_clusters.NumWorkers();
    // Single (big) cluster: collapse two range indices into one parallel-for
    // to reduce the number of fork-joins.
    if (num_clusters == 1) {
      const size_t cluster_idx = 0;
      hwy::ThreadPool& cluster = pools_.Cluster(pkg_idx, cluster_idx);
      // Low-batch: avoid Divide/Remainder.
      if (HWY_UNLIKELY(ranges_mc.NumTasks() == 1)) {
        return ParallelizeOneRange(
            ranges_nc, cluster,
            [&](const IndexRange& range_nc, size_t /*thread*/) {
              func(ranges_mc.Range(0), range_nc);
            });
      } else {
        return ParallelizeTwoRanges(
            ranges_mc, ranges_nc, cluster,
            [&](const IndexRange& range_mc, const IndexRange& range_nc,
                size_t /*thread*/) { func(range_mc, range_nc); });
      }
    }

    // Multiple clusters: N across clusters (both are usually the larger), and
    // M within each cluster. We assume auto-tuning finds small MC/NC tasks.
    ParallelizeOneRange(
        ranges_nc, all_clusters,
        [&](const IndexRange range_nc, size_t cluster_idx) {
          hwy::ThreadPool& cluster = pools_.Cluster(pkg_idx, cluster_idx);
          ParallelizeOneRange(
              ranges_mc, cluster,
              [&](const IndexRange& range_mc, size_t /*thread*/) {
                func(range_mc, range_nc);
              });
        });
  }

  // Calls `func(row_a)` in parallel.
  template <class Func>
  void ForRangeMC(const IndexRange& range_mc, size_t pkg_idx,
                  const Func& func) {
    pools_.Pool(pkg_idx).Run(
        range_mc.begin(), range_mc.end(),
        [&](uint64_t row_a, size_t /*thread*/) { func(row_a); });
  }

 private:
  NestedPools& pools_;
};

template <typename TC>  // BF16/float for C, double for partial
void BindC(size_t M, const RowPtr<TC>& C, MMParallel& parallel) {
  if (!Allocator::ShouldBind()) return;

  const IndexRangePartition ranges_np =
      parallel.RangesOfNP(MMParallel::kMaxPackages, C.Cols(), sizeof(TC), kNR);
  const size_t quantum = Allocator::QuantumBytes() / sizeof(TC);
  bool ok = true;
  for (size_t pkg_idx = 0; pkg_idx < ranges_np.NumTasks(); ++pkg_idx) {
    const IndexRange& cols_c = ranges_np.Range(pkg_idx);
    const size_t node = parallel.Node(pkg_idx);
    for (size_t im = 0; im < M; ++im) {
      // BindRowsToPackageNodes may not be page-aligned.
      const size_t begin = hwy::RoundUpTo(cols_c.begin(), quantum);
      const size_t end = hwy::RoundDownTo(cols_c.end(), quantum);
      ok &= Allocator::BindMemory(C.Row(im) + begin, (end - begin) * sizeof(TC),
                                  node);
    }
  }
  if (HWY_UNLIKELY(!ok)) {
    HWY_WARN("Failed to bind C (%zux%zu), %zu packages.", M, C.Cols(),
             ranges_np.NumTasks());
  }
}

// Per-package storage for packed A, and one global C-shaped `partial` for
// accumulating partial dot products (sections of K).
class MMStorage {
 public:
  // Compile-time bounds on matrix dimensions to enable pre-allocating storage
  // and reusing it across `MatMul` calls. The resulting allocations are 256 MiB
  // per package and 512 MiB, respectively.
  static constexpr size_t kMaxM = 2048;
  static constexpr size_t kMaxK = 64 * 1024;
  static constexpr size_t kMaxN = 256 * 1024;
  // Upper bound for per-worker B storage on the stack. Chosen such that one row
  // of BF16 A and B fit in 32 KiB L1, but there may be `kMaxMR` and `kNR`.
  static constexpr size_t kMaxKC = 8 * 1024;

  explicit MMStorage(MMParallel& parallel) {
    // Per-package allocation so each can decompress A into its own copy.
    parallel.ForPkg(MMParallel::kMaxPackages, [&](size_t pkg_idx) {
      pkg_A_[pkg_idx] = AllocateAlignedRows<BF16>(Extents2D(kMaxM, kMaxK));

      if (Allocator::ShouldBind()) {
        const size_t node = parallel.Node(pkg_idx);
        if (!Allocator::BindMemory(pkg_A_[pkg_idx].All(),
                                   pkg_A_[pkg_idx].NumBytes(), node)) {
          HWY_WARN("Failed to bind memory for package %zu", pkg_idx);
        }
      }
    });

    // Per-worker copies of `partial` would be wasteful. We instead allocate
    // one instance of the maximum matrix extents because threads write at
    // false-sharing-free granularity.
    partial_storage_ = AllocateAlignedRows<double>(Extents2D(kMaxM, kMaxN));
    // Same stride independent of the actual C.Cols() so we can pre-bind.
    partial_ = RowPtrD(partial_storage_.All(), kMaxN,
                       StrideForCyclicOffsets<double>(kMaxN));
    // Avoid cross-package accesses.
    BindC(kMaxM, partial_, parallel);
  }

  // Returns per-package matrix view. Non-const so that `RowVectorBatch` is
  // non-const, because `RowPtr` requires a non-const pointer.
  RowPtrBF A(size_t pkg_idx, const Extents2D& extents) {
    HWY_DASSERT(extents.rows <= kMaxM);
    HWY_DASSERT(extents.cols <= kMaxK);
    const size_t stride = StrideForCyclicOffsets<BF16>(extents.cols);
    return RowPtrBF(pkg_A_[pkg_idx].All(), extents.cols, stride);
  }

  RowPtrD Partial() const { return partial_; }

 private:
  RowVectorBatch<BF16> pkg_A_[MMParallel::kMaxPackages];
  RowVectorBatch<double> partial_storage_;
  RowPtrD partial_;
};

//------------------------------------------------------------------------------
// Autotuning

// Naming convention: outer loop first, T suffix means threaded. This refers to
// the loops *around* `A2C0`, which contains loops over mc/kc. The outermost
// `ranges_np` loop across packages is implicit and applies to all of these.
//
// Parallelizing across K (A/B columns) is undesirable because the resulting
// partial dot products require synchronization or reduction across threads.
enum class MMOrder : uint8_t {
  // Single M, parallel N, sequential K (inside the parallel section to
  // reduce fork-joins). Similar to GotoBLAS, good for large N vs. M and K.
  kNT_K,
  // Specialization of `kNT_K` for a single K task with `kDirect`.
  kNT,

  // Parallelize over blocks of M and N: good when both are large. We no longer
  // support `kMT_NT_K`: no advantage on Skylake, and `kNT_MT_K` is 1.5x as
  // fast on Zen4.
  kNT_MT_K,
  kNT_MT,  // Specialization of `kNT_MT_K` for a single K task with `kDirect`.

  // Resident C (`kK_M_NT`) should be good for large K relative to M and N.
  // However, it does not (much) outperform `kNT_K` on SKX and Zen4. There are
  // no kN* because we expect M (batch size) to be small relative to K and N.
};

static inline bool IsBlock(MMOrder order) {
  return order == MMOrder::kNT_MT_K || order == MMOrder::kNT_MT;
}

static inline bool IsOneKC(MMOrder order) {
  return order == MMOrder::kNT || order == MMOrder::kNT_MT;
}

static inline const char* StringFromOrder(MMOrder order) {
  switch (order) {
    case MMOrder::kNT_K:
      return "NT_K";
    case MMOrder::kNT:
      return "NT";
    case MMOrder::kNT_MT_K:
      return "NT_MT_K";
    case MMOrder::kNT_MT:
      return "NT_MT";
    default:
      return nullptr;
  }
}

// How/where to write the A2C0 result. This determines the `tag` argument to
// that function, which governs whether we call `MMStoreHorizontalSumsIntoC` or
// `MMAddHorizontalSumsIntoPartial`.
enum class MMOut : uint8_t {
  kCopy,    // accumulate into partial, scale/add to C
  kDirect,  // single kc task, write directly to C
  kParM     // kCopy but parallel over M
  // kParN is not better on SKX/Zen4.
};

static inline const char* StringFromOut(MMOut out) {
  switch (out) {
    case MMOut::kDirect:
      return "Direct";
    case MMOut::kCopy:
      return "Copy";
    case MMOut::kParM:
      return "ParM";
    default:
      return nullptr;
  }
}

// How to parallelize the per-package `DecompressA`. To reduce combinatorial
// explosion, we tune this separately from `MMConfig`.
enum class MMParA : uint8_t { kNone, kK1 = 1, kK2 = 2, kK4 = 4, kM };

static inline const char* StringFromParA(MMParA par_a) {
  switch (par_a) {
    case MMParA::kNone:
      return "ParA0 ";
    case MMParA::kK1:
      return "ParAK1";
    case MMParA::kK2:
      return "ParAK2";
    case MMParA::kK4:
      return "ParAK4";
    case MMParA::kM:
      return "ParAM ";
    default:
      return nullptr;
  }
}

// Possible configurations for the autotuner to choose from:
// `mr` := C rows to write at a time (< #registers / `kNR`),
// `kc` := A / B columns such that `mr` rows fit in L1,
// `mc` := A rows such that `kc` columns fit in L2,
// `nc` := B rows such that `kc` columns fit in L3 alongside `mc x nc` C.
// Also includes loop order and task granularity.
#pragma pack(push, 1)
class MMConfig {
 public:
  MMConfig() = default;  // for std::vector
  // `mr` is the number of A rows per call to `MMKernel::LoopKC`.
  // `MMOrder` is how to parallelize the outer loops.
  // `MMOut` is how/whether to parallelize filling the C result.
  // `inner_tasks` chooses the within-cluster task granularity in `ForNP`.
  MMConfig(size_t K, size_t N, size_t mr, size_t mc, size_t kc, size_t nc,
           size_t kc_multiple, size_t nc_multiple, MMOrder order, MMOut out,
           int inner_tasks)
      : mr_(static_cast<uint32_t>(mr)),
        mc_(static_cast<uint32_t>(mc)),
        kc_(static_cast<uint32_t>(kc)),
        nc_(static_cast<uint32_t>(nc)),
        nc_multiple_(static_cast<uint32_t>(nc_multiple)),
        kc_multiple_(static_cast<uint32_t>(kc_multiple)),
        order_(order),
        out_(out),
        inner_tasks_(static_cast<uint8_t>(inner_tasks)),
        reserved_{} {
    HWY_DASSERT(mr == 1 || mr == 2 || mr == 4);
    if (mc % mr != 0) {
      HWY_WARN("mc %zu not a multiple of mr %zu", mc, mr);
    }
    // Do not warn for single-kc tasks; some models unfortunately have K which
    // are not multiples of `kc_multiple`.
    if (kc != K && (kc % kc_multiple) != 0) {
      HWY_WARN("kc %zu not a multiple of kc_multiple %zu", kc, kc_multiple);
    }
    if (nc != N && (nc % nc_multiple) != 0) {
      HWY_WARN("nc %zu not a multiple of nc_multiple %zu", nc, nc_multiple);
    }
    HWY_DASSERT(StringFromOrder(order_) != nullptr);
    HWY_DASSERT(StringFromOut(out_) != nullptr);
    HWY_DASSERT(1 <= inner_tasks && inner_tasks <= 4);
  }

  // Splits M/N into blocks which are visited sequentially or in parallel.
  // K is always sequential, see `MMOrder`.
  IndexRangePartition RangesOfMC(size_t M) const {
    return MaxSizePartition(IndexRange(0, M), mc_, mr_);
  }
  IndexRangePartition RangesOfKC(size_t K) const {
    return MaxSizePartition(IndexRange(0, K), kc_, kc_multiple_);
  }
  IndexRangePartition RangesOfNC(IndexRange range_np) const {
    return MaxSizePartition(range_np, nc_, nc_multiple_);
  }

  MMOrder Order() const { return order_; }
  MMOut Out() const { return out_; }
  // No `OuterTasks` because static partitioning across clusters is sufficient.
  size_t InnerTasks() const { return static_cast<size_t>(inner_tasks_); }

  // Accessors for printing autotune result.
  size_t MR() const { return static_cast<size_t>(mr_); }
  size_t MC() const { return static_cast<size_t>(mc_); }
  size_t KC() const { return static_cast<size_t>(kc_); }
  size_t NC() const { return static_cast<size_t>(nc_); }

 private:
  // Somewhat-compressed representation because MMCandidates may return dozens.
  uint32_t mr_;
  uint32_t mc_;
  uint32_t kc_;
  uint32_t nc_;
  uint32_t nc_multiple_;
  uint32_t kc_multiple_;
  MMOrder order_;
  MMOut out_;
  uint8_t inner_tasks_;
  HWY_MAYBE_UNUSED uint8_t reserved_[5];
};
static_assert(sizeof(MMConfig) == 32);  // for faster indexing
#pragma pack(pop)

std::vector<MMConfig> MMCandidates(size_t M, size_t K, size_t N,
                                   size_t sizeof_TC, size_t max_mr, size_t nr,
                                   const IndexRangePartition& ranges_np,
                                   bool print_config);

// State machine for choosing the best `TConfig`, which is `MMConfig` for the
// main MatMul autotuner.
template <typename TConfig>
class MMAutoTune {
 public:
  // Returns nullptr if not yet finished, otherwise the best config. Do not
  // store this pointer because it can be invalidated.
  const TConfig* Best() const { return best_; }

  // If false, caller must call `SetCandidates` before `NextConfig`.
  bool HasCandidates() const {
    HWY_DASSERT(!Best());
    return !candidates_.empty();
  }
  void SetCandidates(std::vector<TConfig> candidates) {
    HWY_DASSERT(!HasCandidates());
    candidates_.swap(candidates);
    HWY_DASSERT(HasCandidates());
    min_ticks_.resize(candidates_.size(), ~uint64_t{0});
  }

  // Returns the current `TConfig` to measure.
  const TConfig& NextConfig() const {
    HWY_DASSERT(!Best() && HasCandidates());
    return candidates_[config_idx_];
  }

  // Returns the best ticks so far for this candidate. Negligible CPU time.
  uint64_t NotifyTicks(uint64_t ticks) {
    HWY_DASSERT(HasCandidates());
    HWY_DASSERT(!skipped_.Get(config_idx_));

    best_ticks_ = HWY_MIN(best_ticks_, ticks);
    min_ticks_[config_idx_] = HWY_MIN(min_ticks_[config_idx_], ticks);
    // Best so far. Save because we update `config_idx_` below.
    const size_t my_best_ticks = min_ticks_[config_idx_];
    const size_t my_idx = config_idx_;

    // Advance/wrap around to next non-skipped config. Do this first because it
    // updates `rounds_complete_`. To decorrelate measurements, we do not
    // immediately re-measure the same config.
    for (;;) {
      ++config_idx_;
      if (HWY_UNLIKELY(config_idx_ == candidates_.size())) {
        config_idx_ = 0;
        ++rounds_complete_;
      }
      // Guaranteed to terminate because `best_ticks_` is never worse than any
      // other, hence is not skipped.
      if (!skipped_.Get(config_idx_)) break;
    }

    // Disqualify from future `NextConfig` if the best of two measurements so
    // far is sufficiently worse than `best_ticks_`. This tolerates some noise
    // in the first or second measurement.
    if (rounds_complete_ != 0 && my_best_ticks > 5 * best_ticks_ / 4) {
      skipped_.Set(my_idx);
    }

    // After sufficient rounds, choose the winner.
    if (rounds_complete_ == 4) {
      for (size_t i = 0; i < candidates_.size(); ++i) {
        worst_min_ticks_ = HWY_MAX(worst_min_ticks_, min_ticks_[i]);
        if (min_ticks_[i] == best_ticks_) {
          // Causes `Best()` to be non-null, hence `MatMul` will no longer call
          // `NextConfig` for this shape.
          best_ = &candidates_[i];
          config_idx_ = i;  // just in case callers want to know which index.
        }
      }
      HWY_DASSERT(best_ != nullptr);  // no min_ticks_ matches best_ticks_
    }

    return my_best_ticks;
  }

  // Avoid printing the first two rounds, because those might be noisy and not
  // yet skipped.
  bool ShouldPrint() { return rounds_complete_ > 2; }

  // Only valid after Best() is non-null. Used to compute the autotuning gain.
  uint64_t BestTicks() const { return best_ticks_; }
  uint64_t WorstMinTicks() const { return worst_min_ticks_; }
  uint64_t FirstConfigTicks() const { return min_ticks_[0]; }

 private:
  const TConfig* best_ = nullptr;
  std::vector<TConfig> candidates_;
  // Use Min because threads are pinned, so we only expect additive noise.
  std::vector<uint64_t> min_ticks_;  // one per candidate
  size_t config_idx_ = 0;            // [0, candidates_.size())
  size_t rounds_complete_ = 0;
  uint64_t best_ticks_ = ~uint64_t{0};
  uint64_t worst_min_ticks_ = 0;
  hwy::BitSet4096<> skipped_;
};

//------------------------------------------------------------------------------

// Map of previously seen dimensions to index via linear search.
class MMKeys {
 public:
  using Key = uint64_t;
  // KeyFromDims will only return this if all dims are zero, which is invalid.
  static constexpr Key kPadding = 0;

  // Compresses the dimensions into a single Key for faster comparison.
  static Key KeyFromDims(size_t M, size_t K, size_t N) {
    HWY_DASSERT(M < (Key{1} << 16));  // batch sizes are smaller
    HWY_DASSERT(K < (Key{1} << 24));
    HWY_DASSERT(N < (Key{1} << 24));
    const Key key = static_cast<Key>(M) | (static_cast<Key>(K) << 16) |
                    (static_cast<Key>(N) << 40);
    HWY_DASSERT(key != kPadding);
    return key;
  }

  // We leave the search to callers so they can use dynamic-dispatched SIMD,
  // which is not possible in this header.
  hwy::Span<const Key> Keys() const {
    return hwy::Span<const Key>(keys_.get(), num_unique_);
  }

  // Must only be called if not already present in `Keys()`.
  void Append(Key key) {
    // Dynamic allocation because the test checks many more dimensions than
    // would be reasonable to pre-allocate. DIY for alignment and padding.
    if (HWY_UNLIKELY(num_unique_ >= capacity_)) {
      const size_t NU64 = Allocator::VectorBytes() / sizeof(Key);
      // Start at one vector so the size is always a multiple of N.
      if (HWY_UNLIKELY(capacity_ == 0)) {
        capacity_ = hwy::DivCeil(NU64, 2);  // will be doubled below
      }
      capacity_ *= 2;
      HWY_DASSERT(capacity_ >= num_unique_ + 1);
      hwy::AlignedFreeUniquePtr<Key[]> new_keys =
          hwy::AllocateAligned<Key>(capacity_);
      hwy::CopyBytes(keys_.get(), new_keys.get(), num_unique_ * sizeof(Key));
      // Pad for SIMD.
      for (size_t i = num_unique_; i < hwy::RoundUpTo(num_unique_, NU64); ++i) {
        new_keys[i] = kPadding;
      }
      keys_.swap(new_keys);
    }
    keys_[num_unique_++] = key;
  }

 private:
  size_t capacity_ = 0;
  size_t num_unique_ = 0;
  hwy::AlignedFreeUniquePtr<Key[]> keys_;
};

// Per-MatMul-shape state.
struct MMPerKey {
  MMPerKey(size_t max_packages, size_t N, size_t sizeof_TC, size_t nr,
           MMParallel& parallel)
      : ranges_np(parallel.RangesOfNP(max_packages, N, sizeof_TC, nr)) {}

  // Only profile if enabled and the main autotuner finished (the par_a
  // autotuner is per-package and we want to avoid synchronization).
  bool WantProfile() const { return PROFILER_ENABLED != 0 && autotune.Best(); }

  const IndexRangePartition ranges_np;
  MMAutoTune<MMConfig> autotune;
  MMAutoTune<MMParA> autotune_par_a[MMParallel::kMaxPackages];
};

// Stores state shared across MatMul calls. Non-copyable.
struct MatMulEnv {
  explicit MatMulEnv(NestedPools& pools);

  bool have_timer_stop = false;

  // Enable binding: disabled in Gemma until tensors support it, enabled in
  // bench_matmul.cc.
  bool enable_bind = false;

  // Whether `MMCandidates()` should print the set of parameters.
  bool print_config = false;
  // Whether to print each config's speed during autotuning.
  bool print_measurement = false;
  // Whether to print the best config immediately after autotuning finished.
  bool print_best = false;

  MMParallel parallel;
  MMStorage storage;
  MMKeys keys;
  std::vector<MMPerKey> per_key;
};

// Arguments to MatMul() that are independent of the A/B/C types.
// Reduces register pressure compared to individual values/references.
struct MMArgs {
  MMArgs(MatMulEnv& env, MMPerKey& per_key, double scale,
         const float* HWY_RESTRICT add, const RowPtrD& partial)
      : env(&env),
        per_key(&per_key),
        scale(scale),
        add(add),
        partial(partial) {}

  MatMulEnv* env;
  MMPerKey* per_key;

  double scale;
  const float* HWY_RESTRICT add;
  // Same size as C, threads write at false-sharing-free granularity.
  RowPtrD partial;
};

// Wrapper over hwy::Zone that is only enabled when autotuning finished.
#if PROFILER_ENABLED
class MMZone {
  using Zone = hwy::Zone;
  static_assert(alignof(Zone) <= 8 && sizeof(Zone) <= 8);

 public:
  ~MMZone() {
    if (used_) {
      Zone* zone = reinterpret_cast<Zone*>(&data_);
      zone->~Zone();
    }
  }

  // `name` must be a string literal.
  void MaybeEnter(const char* name, const MMArgs& args) {
    if (args.per_key->WantProfile()) {
      new (&data_) Zone(name);
      used_ = true;
    }
  }

 private:
  uint64_t data_ = 0;
  bool used_ = false;
};
#else
struct MMZone {
  void MaybeEnter(const char*, const MMArgs&) {}
};
#endif  // PROFILER_ENABLED

// Used for the A and B arguments of `MatMul`, which are always const.
// Create via MakeConstMat. This differs from `RowPtr` in that it supports the
// `ofs` required for compressed T.
template <typename T>
struct ConstMat {
  ConstMat(const T* ptr, Extents2D extents, size_t stride, size_t ofs = 0)
      : ptr(ptr), extents(extents), stride(stride), ofs(ofs) {
    HWY_DASSERT(ptr != nullptr);
    HWY_DASSERT(stride >= extents.cols);
  }
  size_t Row(size_t r) const {
    if constexpr (HWY_IS_DEBUG_BUILD) {
      if (r >= extents.rows) {
        HWY_ABORT("ConstMat::Row %zu out of bounds %zu", r, extents.rows);
      }
    }
    return ofs + r * stride;
  }

  const Extents2D& Extents() const { return extents; }
  size_t Stride() const { return stride; }

  // Shrinks the row-extent of this matrix view, i.e. reduces the view to a
  // subrange of the original rows starting at row 0.
  void ShrinkRows(size_t rows) {
    HWY_ASSERT(rows <= extents.rows);
    extents.rows = rows;
  }

  const T* HWY_RESTRICT ptr;
  Extents2D extents;
  size_t stride;

  // `scale` allows expanding the smaller range of `SfpStream` to the original
  // values. MatFromWeights sets this from `MatPtr`.
  float scale = 1.0f;

  // Offset to add to `ptr`; separate because T=NuqStream does not support
  // pointer arithmetic.
  size_t ofs;
};

// For deducing T.
template <typename T>
ConstMat<T> MakeConstMat(T* HWY_RESTRICT ptr, Extents2D extents, size_t stride,
                         size_t ofs = 0) {
  return ConstMat<T>(ptr, extents, stride, ofs);
}

// For A argument to MatMul (activations).
template <typename T>
ConstMat<T> ConstMatFromBatch(size_t batch_size,
                              const RowVectorBatch<T>& row_vectors) {
  HWY_DASSERT(batch_size <= row_vectors.BatchSize());
  return MakeConstMat(const_cast<T*>(row_vectors.Const()),
                      Extents2D(batch_size, row_vectors.Cols()),
                      row_vectors.Stride());
}

template <typename T>
ConstMat<T> ConstMatFromWeights(const MatPtrT<T>& m, size_t ofs = 0) {
  ConstMat<T> mat =
      MakeConstMat(const_cast<T*>(m.data()), m.Extents(), m.Stride(), ofs);
  mat.scale = m.scale();
  return mat;
}

template <typename TB>
void BindB(size_t N, size_t sizeof_TC, const ConstMat<TB>& B,
           MMParallel& parallel) {
  if (!Allocator::ShouldBind()) return;

  const IndexRangePartition ranges_np =
      parallel.RangesOfNP(MMParallel::kMaxPackages, N, sizeof_TC, kNR);
  const size_t quantum = Allocator::QuantumBytes() / sizeof(TB);
  for (size_t pkg_idx = 0; pkg_idx < ranges_np.NumTasks(); ++pkg_idx) {
    const IndexRange& rows_b = ranges_np.Range(pkg_idx);
    const size_t node = parallel.Node(pkg_idx);
    uintptr_t begin =
        reinterpret_cast<uintptr_t>(B.ptr + B.Row(rows_b.begin()));
    uintptr_t end = begin + rows_b.Num() * B.Stride() * sizeof(TB);
    // B is not yet guaranteed to have padded rows, so only bind the
    // subset that is page-aligned.
    begin = hwy::RoundUpTo(begin, quantum);
    end = hwy::RoundDownTo(end, quantum);
    if (HWY_LIKELY(begin != end)) {
      Allocator::BindMemory(reinterpret_cast<void*>(begin), end - begin, node);
    }
  }
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_OPS_MATMUL_H_
