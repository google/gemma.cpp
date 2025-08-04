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

#include <atomic>
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
  GenerateCandidates(const Allocator& allocator, size_t M, size_t K, size_t N,
                     size_t sizeof_TC, size_t max_mr, size_t nr,
                     const IndexRangePartition& ranges_np, bool print_config)
      : allocator_(allocator),
        M_(M),
        K_(K),
        N_(N),
        sizeof_TC_(sizeof_TC),
        max_mr_(max_mr),
        nr_(nr),
        // These influence kc/nc, but are also stored in `MMConfig` for
        // `RangesOf*`. Must be a vector multiple. The previous/next cache line
        // is likely still in L1, but we expect K > 1000 and might as well round
        // up to the line size.
        kc_multiple_(HWY_MIN(K, allocator.LineBytes() / sizeof(BF16))),
        nc_multiple_(allocator.StepBytes() / sizeof_TC),
        ranges_np_(ranges_np),
        print_config_(print_config) {}

  std::vector<MMConfig> operator()() const {
    std::vector<MMConfig> candidates;
    candidates.reserve(128);

    for (size_t mr : MR()) {
      for (MMOrder order : Orders(mr)) {
        const std::vector<int>& all_inner_tasks = InnerTasks(order);
        const std::vector<MMOut>& all_outs = Outs(order);
        for (size_t kc : KC(mr, order)) {
          for (size_t mc : MC(mr, kc, order)) {
            for (size_t nc : NC(mr, mc, kc, order)) {
              for (int inner_tasks : all_inner_tasks) {
                for (MMOut out : all_outs) {
                  const MMConfig config(K_, N_, mr, mc, kc, nc, kc_multiple_,
                                        nc_multiple_, order, out, inner_tasks);
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
    if (M_ >= max_mr_ && !is_avx2) all_mr.push_back(max_mr_);
    // Allow for AVX-512 but not SSE4 (for which 4 are usually better). Also
    // enable if not enough rows for 4.
    if (M_ >= 2 && (M_ < max_mr_ || (!is_sse && !is_wasm))) {
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

  // The number of A and B columns to read between updating `partial`.
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
    // fewer loops over K, and thus fewer writes to `partial`. Hence we do not
    // subtract the output and buf, and allow using more than the actual L1
    // size. This results in an overestimate, and the loop below will propose
    // the next few smaller values for the autotuner to evaluate.
    const size_t bytes_ab = allocator_.L1Bytes() * 3;
    const size_t col_bytes = rows_a * sizeof(BF16) + nr_ * sizeof(BF16);
    size_t kc_max = hwy::DivCeil(bytes_ab, col_bytes);
    kc_max =
        RoundDownWithFloor(HWY_MIN(kc_max, MMStorage::kMaxKC), kc_multiple_);
    kc_max = HWY_MIN(kc_max, K_);

    SizeVec all_kc(1, kc_max);

    // Avoid proposing kc > K.
    if (K_ > kc_multiple_) {
      // Generally it is best to use the full `kc` (fewer writes to `partial`),
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
      fprintf(stderr, "KC: ");
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
    const size_t bytes_b = nr_ * kc * (sizeof(SfpStream) + sizeof(BF16));

    // Choose the largest feasible `mc_max` (A/C rows) to maximize reuse of the
    // packed B. We want `mc * kc` elements of A to fit in L2, alongside
    // `bytes_b` plus `mc` cache lines because resident-A updates `mc` rows of
    // partial.
    const size_t bytes_per_mc = kc * sizeof(BF16) + allocator_.LineBytes();
    size_t mc_max = hwy::DivCeil(allocator_.L2Bytes() - bytes_b, bytes_per_mc);
    mc_max = HWY_MIN(mc_max, MMStorage::kMaxM);
    HWY_DASSERT(mc_max != 0);
    mc_max = HWY_MIN(mc_max, M_);
    mc_max = hwy::RoundDownTo(mc_max, mr);

    SizeVec all_mc(1, mc_max);
    // Larger MC is better for non-blocks, otherwise we want more small options.
    const size_t reps = !IsBlock(order) ? 2 : 3;

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
      fprintf(stderr, "MC: ");
      for (size_t mc : all_mc) {
        fprintf(stderr, "%zu ", mc);
      }
      fprintf(stderr, "\n");
    }

    return all_mc;
  }

  // The number of (possibly L3 resident) B rows per `NT_MT` task.
  SizeVec NC(size_t mr, size_t mc, size_t kc, MMOrder order) const {
    const size_t np_max = ranges_np_.TaskSize();
    size_t nc_max = np_max;
    const size_t out_bytes = IsOneKC(order) ? sizeof_TC_ : sizeof(double);
    // Only if there will be reuse of B: choose the largest `nc_max` (C cols)
    // such that `nc x kc` of B and `mc x nc` of `partial` or `C` fit in L3.
    // Otherwise, leave it unbounded.
    if (M_ > mr) {
      const size_t bytes_per_nc = (kc * sizeof(BF16) + mc * out_bytes);
      nc_max = hwy::DivCeil(allocator_.L3Bytes(), bytes_per_nc);
      nc_max = HWY_MIN(HWY_MIN(nc_max, MMStorage::kMaxN), np_max);
    }
    HWY_DASSERT(nc_max != 0);
    nc_max = RoundDownWithFloor(nc_max, nc_multiple_);

    // If there are going to be multiple ranges, anything more than half would
    // be imbalanced and suboptimal.
    if (nc_max < np_max && nc_max >= np_max / 2) {
      nc_max = RoundDownWithFloor(np_max / 2, nc_multiple_);
    }

    // Non-block calls ForNP, which ignores `range_nc` and uses `range_np`.
    if (!IsBlock(order)) return SizeVec(1, np_max);

    SizeVec all_nc(1, nc_max);

    // Avoid proposing nc > N.
    if (np_max > nc_multiple_) {
      // Large L3, but its behavior and characteristics varies across platforms,
      // hence autotune a wider range of nc than the other dimensions.
      size_t reps = 10;
      // For small M, we can afford larger NC, hence allow fewer small options.
      if (M_ <= 2 * mr) reps -= 1;

      size_t prev = nc_max;
      for (size_t rep = 0; rep < reps; ++rep) {
        const size_t div =
            PrevDivisor(nc_multiple_, prev, np_max, nc_multiple_);
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
      fprintf(stderr, "NC: ");
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

  // Whether to parallelize FillC or enable direct writes to C.
  std::vector<MMOut> Outs(MMOrder order) const {
    std::vector<MMOut> outs;
    for (size_t out_idx = 0;; ++out_idx) {
      const MMOut out = static_cast<MMOut>(out_idx);
      if (StringFromOut(out) == nullptr) return outs;  // done
      // kParM only makes sense if we have more than one row of A.
      if (out == MMOut::kParM && M_ == 1) continue;
      // Blocks are already parallelized.
      if (out == MMOut::kParM && IsBlock(order)) continue;
      // Direct only works for a single kc range.
      if ((out == MMOut::kDirect) != IsOneKC(order)) continue;
      // For non-block, kCopy does not beat kDirect.
      if (out == MMOut::kCopy && IsOneKC(order) && !IsBlock(order)) continue;
      outs.push_back(out);
    }
  }

  const Allocator& allocator_;
  const size_t M_;
  const size_t K_;
  const size_t N_;
  const size_t sizeof_TC_;

  const size_t max_mr_;
  const size_t nr_;

  const size_t kc_multiple_;
  const size_t nc_multiple_;

  IndexRangePartition ranges_np_;

  const bool print_config_;
};

}  // namespace

// Facade to avoid exposing `GenerateCandidates` in the header.
std::vector<MMConfig> MMCandidates(const Allocator& allocator, size_t M,
                                   size_t K, size_t N, size_t sizeof_TC,
                                   size_t max_mr, size_t nr,
                                   const IndexRangePartition& ranges_np,
                                   bool print_config) {
  return GenerateCandidates(allocator, M, K, N, sizeof_TC, max_mr, nr,
                            ranges_np, print_config)();
}

// Returns the granularity of B rows for `RangesOfNP`. Aims to avoid remote
// memory accesses or false sharing, unless there are insufficient per-package
// rows for that.
static size_t NPMultiple(const Allocator& allocator, size_t N,
                         size_t sizeof_TC, size_t nr, size_t num_packages) {
  size_t np_multiple = allocator.BasePageBytes() / sizeof_TC;
  // If binding, `np_multiple` is typically 1024 and `num_packages` > 1. For
  // `N` < 4096, this can cause significant load imbalance. If split unevenly,
  // choose a smaller multiple.
  if (N % (np_multiple * num_packages)) {
    const size_t min_multiple = allocator.LineBytes() / sizeof_TC;
    np_multiple =
        PrevDivisor(min_multiple, np_multiple, N / num_packages, min_multiple);
    if (HWY_UNLIKELY(np_multiple == 0)) {
      np_multiple = min_multiple;
    }
    // This happens in tests with small N, hence do not assert.
    if (N % (np_multiple * num_packages) && N >= 128) {
      static std::atomic_flag warned = ATOMIC_FLAG_INIT;
      if (!warned.test_and_set()) {
        HWY_WARN(
            "NPMultiple: N=%zu still not divisible by np_multiple=%zu * "
            "num_packages=%zu\n",
            N, np_multiple, num_packages);
      }
      np_multiple = nr;
    }
  }
  return np_multiple;
}

IndexRangePartition MMParallel::RangesOfNP(size_t max_packages, size_t N,
                                           size_t sizeof_TC, size_t nr) const {
  const size_t num_packages = HWY_MIN(max_packages, ctx_.pools.NumPackages());
  return StaticPartition(
      IndexRange(0, N), num_packages,
      NPMultiple(ctx_.allocator, N, sizeof_TC, nr, num_packages));
}

MatMulEnv::MatMulEnv(ThreadingContext& ctx)
    : ctx(ctx), parallel(ctx), storage(ctx.allocator, parallel) {
  char cpu100[100];
  have_timer_stop = hwy::platform::HaveTimerStop(cpu100);

  row_ptrs.push_back(hwy::AllocateAligned<uint8_t*>(MMStorage::kMaxM));  // A
  row_ptrs.push_back(hwy::AllocateAligned<uint8_t*>(MMStorage::kMaxN));  // B
  row_ptrs.push_back(hwy::AllocateAligned<uint8_t*>(MMStorage::kMaxM));  // C
}

void BindB(MatPtr& B, size_t sizeof_TC, MMParallel& parallel) {
  Allocator& allocator = parallel.allocator();
  if (!allocator.ShouldBind()) return;
  if (B.Rows() == 1) return;

  PROFILER_ZONE("Startup.BindB");

  const IndexRangePartition ranges_np =
      parallel.RangesOfNP(kMaxPackages, B.Rows(), sizeof_TC, kNR);
  for (size_t pkg_idx = 0; pkg_idx < ranges_np.NumTasks(); ++pkg_idx) {
    const IndexRange& rows_b = ranges_np.Range(pkg_idx);
    const size_t node = parallel.Node(pkg_idx);
    uintptr_t begin = reinterpret_cast<uintptr_t>(B.RowBytes(rows_b.begin()));
    uintptr_t end = begin + rows_b.Num() * B.Stride() * B.ElementBytes();
    // B row padding is less than the page size, so only bind the subset that
    // is page-aligned.
    begin = hwy::RoundUpTo(begin, allocator.BasePageBytes());
    end = hwy::RoundDownTo(end, allocator.BasePageBytes());
    if (HWY_LIKELY(begin != end)) {
      allocator.BindMemory(reinterpret_cast<void*>(begin), end - begin, node);
    }
  }
}

// C is BF16/float, or double for partial
void BindC(MatPtr& C, MMParallel& parallel) {
  Allocator& allocator = parallel.allocator();
  if (!allocator.ShouldBind()) return;

  PROFILER_ZONE("Startup.BindC");

  const IndexRangePartition ranges_np =
      parallel.RangesOfNP(kMaxPackages, C.Cols(), C.ElementBytes(), kNR);
  bool ok = true;
  for (size_t pkg_idx = 0; pkg_idx < ranges_np.NumTasks(); ++pkg_idx) {
    const IndexRange& cols_c = ranges_np.Range(pkg_idx);
    // `BindMemory` requires page alignment. These are in bytes.
    const size_t begin = hwy::RoundUpTo(cols_c.begin() * C.ElementBytes(),
                                        allocator.BasePageBytes());
    const size_t end = hwy::RoundDownTo(cols_c.end() * C.ElementBytes(),
                                        allocator.BasePageBytes());

    const size_t node = parallel.Node(pkg_idx);
    for (size_t im = 0; im < C.Rows(); ++im) {
      ok &= allocator.BindMemory(C.RowBytes(im) + begin, end - begin, node);
    }
  }
  if (HWY_UNLIKELY(!ok)) {
    HWY_WARN("Failed to bind C (%zux%zu), %zu packages.", C.Rows(), C.Cols(),
             ranges_np.NumTasks());
  }
}

}  // namespace gcpp
