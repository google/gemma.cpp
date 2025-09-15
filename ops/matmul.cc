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

#include "ops/matmul.h"

// Analytical model of cache parameters for generating autotune candidates.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <vector>

#include "util/allocator.h"
#include "util/basics.h"
#include "util/mat.h"
#include "util/threading_context.h"
#include "hwy/base.h"
#include "hwy/detect_targets.h"
#include "hwy/per_target.h"
#include "hwy/profiler.h"
#include "hwy/timer.h"

namespace gcpp {
namespace {

// Rounds down to a multiple of `multiple`, but returns at least `multiple`.
size_t RoundDownWithFloor(size_t value, size_t multiple) {
  HWY_DASSERT(multiple != 0);
  return HWY_MAX(multiple, hwy::RoundDownTo(value, multiple));
}

// Returns the highest number in `[begin, end)` that divides `dim` and is a
// multiple of `multiple`, or 0 if none exists.
size_t PrevDivisor(const size_t begin, const size_t end, const size_t dim,
                   const size_t multiple) {
  HWY_DASSERT(end != 0 && dim != 0 && multiple != 0);
  size_t prev = RoundDownWithFloor(end, multiple);
  // Avoid returning `end` if rounding down had no effect.
  if (prev == end) prev -= multiple;
  for (;;) {
    if (prev == 0) return 0;  // No divisor if large multiple or small end.
    if (dim % prev == 0) return prev;
    if (prev <= begin) return 0;
    prev -= multiple;
  }
}

// Implementation of `MMCandidates`. Class hides the `KC` etc member functions
// and holds most of their arguments in member variables.
class GenerateCandidates {
 public:
  GenerateCandidates(const CacheInfo& cache, size_t M, size_t K, size_t N,
                     size_t num_B, size_t sizeof_TC, bool print_config)
      : cache_(cache),
        M_(M),
        K_(K),
        N_(N),
        num_B_(num_B),
        sizeof_TC_(sizeof_TC),
        // These influence kc/nc, but are also stored in `MMConfig` for
        // `RangesOf*`. Must be a vector multiple. The previous/next cache line
        // is likely still in L1, but we expect K > 1000 and might as well round
        // up to the line size. Both A and B are BF16.
        kc_multiple_(HWY_MIN(K, cache.LineBytes() / sizeof(BF16))),
        nc_multiple_(cache.StepBytes() / sizeof_TC),
        print_config_(print_config) {}

  std::vector<MMConfig> operator()() const {
    std::vector<MMConfig> candidates;
    candidates.reserve(128);

    for (size_t mr : MR()) {
      for (MMOrder order : Orders(mr)) {
        const std::vector<int>& all_inner_tasks = InnerTasks(order);
        for (size_t kc : KC(mr, order)) {
          for (size_t mc : MC(mr, kc, order)) {
            for (size_t nc : NC(mr, mc, kc, order)) {
              for (int inner_tasks : all_inner_tasks) {
                const MMConfig config(K_, N_, mr, mc, kc, nc, kc_multiple_,
                                      nc_multiple_, order, inner_tasks);
                const size_t M_tasks = config.RangesOfMC(M_).NumTasks();
                const size_t K_tasks = config.RangesOfKC(K_).NumTasks();

                // Blocks only make sense when there are multiple M tasks.
                if (IsBlock(order) != (M_tasks > 1)) continue;
                // Single KC only makes sense when there is a single K task.
                if (IsOneKC(order) != (K_tasks == 1)) continue;

                candidates.push_back(config);
              }
            }
          }
        }
      }
    }

    HWY_ASSERT(!candidates.empty());
    return candidates;
  }

 private:
  using SizeVec = std::vector<size_t>;

  // How many rows of A per call to `MMKernel::LoopKC`. Lower values may
  // be better for SIMD targets with fewer registers.
  SizeVec MR() const {
    const int64_t target = hwy::DispatchedTarget();
    const bool is_avx2 = target == HWY_AVX2;
    const bool is_sse = HWY_SSE4 <= target && target <= HWY_SSE2;
    const bool is_wasm = target == HWY_WASM || target == HWY_WASM_EMU256;

    SizeVec all_mr;
    all_mr.reserve(3);
    // AVX2's 16 registers are not enough for four rows, but SSE4 may benefit.
    if (M_ >= kMaxMR && !is_avx2) all_mr.push_back(kMaxMR);
    // Allow for AVX-512 but not SSE4 (for which 4 are usually better). Also
    // enable if not enough rows for 4.
    if (M_ >= 2 && (M_ < kMaxMR || (!is_sse && !is_wasm))) {
      all_mr.push_back(size_t{2});
    }
    // Even SSE4 usually prefers 2 rows; only enable for single rows.
    if (M_ == 1) all_mr.push_back(size_t{1});
    HWY_ASSERT(!all_mr.empty());
    return all_mr;
  }

  // Which loop orders to enable depending on M.
  std::vector<MMOrder> Orders(size_t mr) const {
    std::vector<MMOrder> orders;
    for (size_t order_idx = 0;; ++order_idx) {
      const MMOrder order = static_cast<MMOrder>(order_idx);
      if (StringFromOrder(order) == nullptr) return orders;  // done
      // 2D blocking is useless for a single row of M.
      if (IsBlock(order) && M_ <= mr) continue;
      // Conversely, N-only parallelism is uncompetitive for large M.
      if (!IsBlock(order) && M_ >= kMaxTilesM * mr) continue;
      orders.push_back(order);
    }
  }

  // The number of A and B columns to read between updating `C`.
  SizeVec KC(size_t mr, MMOrder order) const {
    // `LoopKC` handles up to `mr` rows of A.
    const size_t rows_a = HWY_MIN(M_, mr);

    // After looping over `kc` columns, we write `mr x 4` outputs and 16 vector
    // `buf`. To amortize the write cost, we want to maximize `kc`. However, it
    // is important that B fits in L1, because batch=1 only has a single row of
    // A and thus no reuse of the packed B. When L1-resident, we can use the
    // separate `DecompressAndZeroPad` to write `kc` columns, rather than having
    // to integrate `Decompress2` into `LoopKC`, which is less efficient for
    // TB=NUQ due to less amortization of the table loads. Due to the low L1
    // latency, the packing is still effectively fused into `LoopKC`. It may
    // be better to round up and accept a few L2 accesses in exchange for
    // fewer loops over K, and thus fewer writes to `C`. Hence we do not
    // subtract the output and buf, and allow using more than the actual L1
    // size. This results in an overestimate, and the loop below will propose
    // the next few smaller values for the autotuner to evaluate.
    const size_t bytes_ab =
        cache_.L1Bytes() * (sizeof(BF16) + sizeof(SfpStream));
    const size_t col_bytes = rows_a * sizeof(BF16) + kNR * sizeof(BF16);
    size_t kc_max = hwy::DivCeil(bytes_ab, col_bytes);
    kc_max = RoundDownWithFloor(HWY_MIN(kc_max, kMaxKC), kc_multiple_);
    kc_max = HWY_MIN(kc_max, K_);

    SizeVec all_kc(1, kc_max);

    // Avoid proposing kc > K.
    if (K_ > kc_multiple_) {
      // Generally it is best to use the full `kc` (fewer writes to `C`),
      // but a bit less can be better if it evenly divides `K`, or enables an
      // `mc` that evenly divides `M`. Try several smaller values.

      // If we can afford a single K task, that's usually best; only try one
      // more. Otherwise, blocks may require smaller kc (more options).
      const size_t reps = (kc_max == K_) ? 1 : IsBlock(order) ? 3 : 2;

      size_t prev = kc_max;
      for (size_t rep = 0; rep < reps; ++rep) {
        const size_t div = PrevDivisor(kc_multiple_, prev, K_, kc_multiple_);
        prev = div ? div : RoundDownWithFloor(prev / 2, kc_multiple_);
        all_kc.push_back(prev);
      }
    }

    if (print_config_ && all_kc.size() > 1) {
      fprintf(stderr, "num_B %zu: KC: ", num_B_);
      for (size_t kc : all_kc) {
        fprintf(stderr, "%zu ", kc);
      }
      fprintf(stderr, "\n");
    }

    return all_kc;
  }

  // The number of (L2 resident) A rows for `A2C0` to loop over.
  SizeVec MC(size_t mr, size_t kc, MMOrder order) const {
    // Typically 12-24K. The B rows are pinned in L1, but also occupy L2 because
    // it is typically inclusive.
    const size_t bytes_b = kNR * kc * (sizeof(SfpStream) + sizeof(BF16));

    // Choose the largest feasible `mc_max` (A/C rows) to maximize reuse of the
    // packed B. We want `mc * kc` elements of A to fit in L2, alongside
    // `bytes_b` plus `mc` cache lines because resident-A updates `mc` C rows.
    const size_t bytes_per_mc = kc * sizeof(BF16) + cache_.LineBytes();
    size_t mc_max = hwy::DivCeil(cache_.L2Bytes() - bytes_b, bytes_per_mc);
    mc_max = HWY_MIN(mc_max, HWY_MIN(kMaxBatchSize, kMaxMC));
    HWY_DASSERT(mc_max != 0);
    mc_max = HWY_MIN(mc_max, M_);
    mc_max = hwy::RoundDownTo(mc_max, mr);

    SizeVec all_mc(1, mc_max);
    // Larger MC is better for non-blocks, otherwise we want more small options,
    // especially for two B.
    const size_t reps = !IsBlock(order) ? 2 : (2 + num_B_);

    size_t prev = mc_max;
    for (size_t rep = 0; rep < reps; ++rep) {
      prev = PrevDivisor(1, prev, M_, mr);
      if (prev >= mc_max || prev == 0) break;
      all_mc.push_back(prev);
    }

    // Blocks: largest is not useful.
    if (IsBlock(order) && all_mc.size() > 1) {
      all_mc.erase(all_mc.begin(), all_mc.begin() + 1);
    }

    if (print_config_ && all_mc.size() > 1) {
      fprintf(stderr, "num_B %zu: MC: ", num_B_);
      for (size_t mc : all_mc) {
        fprintf(stderr, "%zu ", mc);
      }
      fprintf(stderr, "\n");
    }

    return all_mc;
  }

  // The number of (possibly L3 resident) B rows per `NT_MT` task.
  SizeVec NC(size_t mr, size_t mc, size_t kc, MMOrder order) const {
    size_t nc_max = kMaxNC;
    // Only if there will be reuse of B: choose the largest `nc_max` (C cols)
    // such that `nc x kc` of B and `mc x nc` of `C` fit in L3. Otherwise,
    // leave it unbounded.
    if (M_ > mr) {
      const size_t bytes_per_nc = (kc * sizeof(BF16) + mc * sizeof_TC_);
      nc_max = HWY_MIN(hwy::DivCeil(cache_.L3Bytes(), bytes_per_nc), kMaxNC);
    }
    nc_max = HWY_MIN(nc_max, N_);
    HWY_DASSERT(nc_max != 0);
    nc_max = RoundDownWithFloor(nc_max, nc_multiple_);

    // If there are going to be multiple ranges, anything more than half would
    // be imbalanced and suboptimal.
    if (nc_max < N_ && nc_max >= N_ / 2) {
      nc_max = RoundDownWithFloor(N_ / 2, nc_multiple_);
    }

    // Non-block calls ForNP, which ignores `range_nc` and uses `range_np`.
    if (!IsBlock(order)) return SizeVec(1, N_);

    SizeVec all_nc(1, nc_max);

    // Avoid proposing nc > N.
    if (N_ > nc_multiple_) {
      // Large L3, but its behavior and characteristics varies across platforms,
      // hence autotune a wider range of nc than the other dimensions.
      size_t reps = 9 + num_B_;
      // For small M, we can afford larger NC, hence allow fewer small options.
      if (M_ <= 2 * mr) reps -= 1;

      size_t prev = nc_max;
      for (size_t rep = 0; rep < reps; ++rep) {
        const size_t div = PrevDivisor(nc_multiple_, prev, N_, nc_multiple_);
        prev = div ? div : RoundDownWithFloor(prev / 2, nc_multiple_);
        all_nc.push_back(prev);
        if (prev == nc_multiple_) break;
      }

      // Skip the larger values (unlikely to be chosen), keep about 40%.
      const ptrdiff_t want_delete =
          static_cast<ptrdiff_t>(all_nc.size() * 5 / 9 + 2);
      // Keep at least 2.
      const ptrdiff_t max_delete =
          HWY_MAX(static_cast<ptrdiff_t>(all_nc.size()) - 2, ptrdiff_t{0});
      all_nc.erase(all_nc.begin(),
                   all_nc.begin() + HWY_MIN(want_delete, max_delete));
    }

    if (print_config_ && all_nc.size() > 1) {
      fprintf(stderr, "num_B %zu: NC: ", num_B_);
      for (size_t nc : all_nc) {
        fprintf(stderr, "%zu ", nc);
      }
      fprintf(stderr, "\n");
    }

    return all_nc;
  }

  // How many tasks per cluster worker. More = smaller tasks, which can lead
  // to better load balancing at the cost of higher overhead.
  std::vector<int> InnerTasks(MMOrder order) const {
    std::vector<int> inner_tasks;
    inner_tasks.reserve(3);
    inner_tasks.push_back(1);
    // Blocks have one task per mc/nc range and ignore this parameter.
    if (!IsBlock(order)) {
      inner_tasks.push_back(2);
      inner_tasks.push_back(4);
    }
    return inner_tasks;
  }

  const CacheInfo& cache_;
  const size_t M_;
  const size_t K_;
  const size_t N_;
  const size_t num_B_;
  const size_t sizeof_TC_;

  const size_t kc_multiple_;
  const size_t nc_multiple_;

  const bool print_config_;
};

}  // namespace

// Facade to avoid exposing `GenerateCandidates` in the header.
std::vector<MMConfig> MMCandidates(const CacheInfo& cache, size_t M, size_t K,
                                   size_t N, size_t num_B, size_t sizeof_TC,
                                   bool print_config) {
  return GenerateCandidates(cache, M, K, N, num_B, sizeof_TC, print_config)();
}

MatMulEnv::MatMulEnv(ThreadingContext& ctx)
    : ctx(ctx), A_BF(ctx.allocator), C_tiles(ctx) {
  const size_t num_clusters = ctx.pools.AllClusters(/*pkg_idx=*/0).NumWorkers();
  per_cluster.resize(num_clusters);
  for (size_t cluster_idx = 0; cluster_idx < num_clusters; ++cluster_idx) {
    row_ptrs.push_back(hwy::AllocateAligned<uint8_t*>(kMaxBatchSize));  // C
  }

  char cpu100[100];
  have_timer_stop = hwy::platform::HaveTimerStop(cpu100);
}

void BindB(ThreadingContext& ctx, MatPtr& B, size_t sizeof_TC) {
  Allocator& allocator = ctx.allocator;
  if (!allocator.ShouldBind()) return;
  if (B.Rows() == 1) return;

  PROFILER_ZONE("Startup.BindB");

  const size_t node = ctx.topology.GetCluster(/*pkg_idx=*/0, 0).Node();
  uintptr_t begin = reinterpret_cast<uintptr_t>(B.RowBytes(0));
  uintptr_t end = begin + B.Rows() * B.Stride() * B.ElementBytes();
  // B row padding is less than the page size, so only bind the subset that
  // is page-aligned.
  begin = hwy::RoundUpTo(begin, allocator.BasePageBytes());
  end = hwy::RoundDownTo(end, allocator.BasePageBytes());
  if (HWY_LIKELY(begin != end)) {
    allocator.BindMemory(reinterpret_cast<void*>(begin), end - begin, node);
  }
}

// C is BF16/float
void BindC(ThreadingContext& ctx, MatPtr& C) {
  Allocator& allocator = ctx.allocator;
  if (!allocator.ShouldBind()) return;

  PROFILER_ZONE("Startup.BindC");

  const IndexRange cols_c(0, C.Cols());
  // `BindMemory` requires page alignment. These are in bytes.
  const size_t begin = hwy::RoundUpTo(cols_c.begin() * C.ElementBytes(),
                                      allocator.BasePageBytes());
  const size_t end = hwy::RoundDownTo(cols_c.end() * C.ElementBytes(),
                                      allocator.BasePageBytes());

  const size_t node = ctx.topology.GetCluster(/*pkg_idx=*/0, 0).Node();
  bool ok = true;
  for (size_t im = 0; im < C.Rows(); ++im) {
    ok &= allocator.BindMemory(C.RowBytes(im) + begin, end - begin, node);
  }
  if (HWY_UNLIKELY(!ok)) {
    HWY_WARN("Failed to bind C (%zux%zu).", C.Rows(), C.Cols());
  }
}

}  // namespace gcpp
