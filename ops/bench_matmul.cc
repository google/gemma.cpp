// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Benchmark of large MatMul instances for which the MatMulSlow would be too
// slow. This lacks a reference and is only useful for performance measurement.

#include <stddef.h>
#include <stdio.h>

#include <algorithm>
#include <vector>

#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include "ops/matmul.h"
#include "util/basics.h"
#include "util/threading_context.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/nanobenchmark.h"
#include "hwy/profiler.h"
#include "hwy/timer.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ops/bench_matmul.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"
#include "compression/test_util-inl.h"
#include "ops/matmul-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
// For running BenchAllMatMul only once. Defined within HWY_ONCE.
extern int64_t first_target;

namespace HWY_NAMESPACE {

void PrintSpeed(const Extents2D& A_extents, const Extents2D& B_extents,
                std::vector<double>& times, MMPerKey* per_key) {
  std::sort(times.begin(), times.end());
  // bench_dnn reports the best and average, but the median seems more
  // consistent and resistant to outliers.
  const double elapsed = times[times.size() / 2];
  const double vs_best = elapsed / (times[0] + 1E-6);  // avoid / 0

  const size_t num_b = B_extents.Area();
  const double flops = 2 * A_extents.rows * num_b / elapsed;  // FMA = 2 ops

  fprintf(stderr, "\t%.1f GFLOPS %.3f ms %0.2fx\n", flops * 1E-9, elapsed * 1E3,
          vs_best);
}

// Generates inputs and prints observed throughput of MatMul.
// M = A rows, K = A cols, N = C cols.
template <typename TA, typename TB = TA, typename TC = float>
void BenchMatMul(size_t M, size_t K, size_t N, bool add, MatMulEnv& env) {
  hwy::ThreadPool& pool = env.ctx.pools.Pool(0);
  if (env.print_config || env.print_measurement) {
    fprintf(stderr, "\n");
  }
  fprintf(stderr,
          "BenchMatMul %zu, %zu, %zu, add=%d, TA=%s, TB=%s, TC=%s\n",  //
          M, K, N, add, TypeName<TA>(), TypeName<TB>(), TypeName<TC>());

  const Extents2D A_extents(M, K);
  const Extents2D B_extents(N, K);  // already transposed
  const Extents2D C_extents(M, N);

  MatStorageT<TC> C_slow("c_slow_batch", C_extents, env.ctx.allocator,
                         MatPadding::kOdd);
  MatStorageT<TC> C("c_batch", C_extents, env.ctx.allocator, MatPadding::kOdd);

  MatStorageT<float> add_storage("add", Extents2D(), env.ctx.allocator,
                                 MatPadding::kPacked);
  if (add) {
    add_storage = GenerateMat<float>(Extents2D(1, N), env.ctx.allocator,
                                     MatPadding::kPacked, pool);
    add_storage.SetScale(1.0f);
  }

  MatStorageT<TA> a =
      GenerateMat<TA>(A_extents, env.ctx.allocator, MatPadding::kOdd, pool);
  MatStorageT<TB> b_trans = GenerateTransposedMat<TB>(
      B_extents, env.ctx.allocator, MatPadding::kOdd, pool);

  const float* add_row = add ? add_storage.PackedScale1() : nullptr;

  // Fewer reps for large batch sizes, which take longer.
  const size_t num_samples = M < 32 ? 20 : 12;
  std::vector<double> times;
  times.reserve(num_samples);

  // Ensure usage conditions are set before autotuning. Both binding and
  // spinning may materially affect the choice of config. No harm in calling
  // BindB/C if there is a single package: they will be a no-op.
  BindB(b_trans, sizeof(TC), env.parallel);
  BindC(C, env.parallel);
  C.AllocateAndAttachRowPtrs(env.row_ptrs);

  Tristate use_spinning = Tristate::kDefault;
  env.ctx.pools.MaybeStartSpinning(use_spinning);

  // env.print_config = true;
  // env.print_measurement = true;
  env.print_best = true;

  double keep = 0.0;
  MMPerKey* per_key;
  // Until enough samples collected *after* autotuning finished:
  while (times.size() < num_samples) {
    const double t0 = hwy::platform::Now();
    per_key = MatMul(a, b_trans, add_row, env, C);
    const double t1 = hwy::platform::Now();
    double elapsed = t1 - t0;
    keep += hwy::ConvertScalarTo<double>(C.Row(0)[hwy::Unpredictable1()]);

    // Only record times after autotuning finished.
    if (per_key->autotune.Best()) times.push_back(elapsed);
  }
  hwy::PreventElision(keep);
  env.ctx.pools.MaybeStopSpinning(use_spinning);
  PrintSpeed(A_extents, B_extents, times, per_key);
}

using F32 = float;
using SFP = SfpStream;

void BenchAllMatMul() {
  if (first_target == 0) first_target = HWY_TARGET;
  // Comment out to disable the best-target-only limitation.
  if (HWY_TARGET != first_target) return;

  // Skip EMU128 (10x slower than SSE4 for SFP) and older x86.
  if (HWY_TARGET == HWY_EMU128 || HWY_TARGET == HWY_SSSE3 ||
      HWY_TARGET == HWY_SSE2 || HWY_TARGET == HWY_NEON_WITHOUT_AES) {
    return;
  }

  ThreadingArgs threading_args;
  ThreadingContext ctx(threading_args);
  fprintf(stderr, "BenchAllMatMul %s %s\n", ctx.topology.TopologyString(),
          ctx.pools.PinString());
  MatMulEnv env(ctx);

  for (size_t batch_size : {1, 4, 128, 512}) {
    constexpr bool kAdd = false;
    BenchMatMul<BF16, SFP, BF16>(batch_size, 24576, 3072, kAdd, env);
    BenchMatMul<BF16, SFP, BF16>(batch_size, 3072, 24576, kAdd, env);
  }

  PROFILER_PRINT_RESULTS();
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gcpp {
int64_t first_target = 0;  // none run yet
HWY_BEFORE_TEST(BenchMatMul);
HWY_EXPORT_AND_TEST_P(BenchMatMul, BenchAllMatMul);
HWY_AFTER_TEST();

}  // namespace gcpp

#endif
