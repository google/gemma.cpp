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

#include "hwy/base.h"
#ifndef HWY_DISABLED_TARGETS
// Exclude HWY_SCALAR due to 2x bf16 -> f32, and Armv7 NEON because we require
// double-precision support.
#if HWY_ARCH_ARM_V7
#define HWY_DISABLED_TARGETS (HWY_SCALAR | HWY_NEON)
#else
#define HWY_DISABLED_TARGETS HWY_SCALAR
#endif
#endif

#include <stddef.h>
#include <stdio.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "compression/compress.h"
#include "compression/shared.h"
#include "ops/matmul.h"
#include "util/allocator.h"
#include "util/basics.h"
#include "util/threading.h"
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
#include "ops/matmul-inl.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
// For running BenchAllMatMul only once. Defined within HWY_ONCE.
extern int64_t first_target;

namespace HWY_NAMESPACE {

using FloatPtr = hwy::AlignedFreeUniquePtr<float[]>;

template <typename MatT>
using MatStoragePtr = std::unique_ptr<MatStorageT<MatT>>;

// Generates inputs: deterministic, within max SfpStream range.
template <typename MatT>
MatStoragePtr<MatT> GenerateMat(const Extents2D extents,
                                hwy::ThreadPool& pool) {
  gcpp::CompressWorkingSet ws;
  auto mat =
      std::make_unique<MatStorageT<MatT>>("mat", extents.rows, extents.cols);
  FloatPtr content = hwy::AllocateAligned<float>(mat->NumElements());
  HWY_ASSERT(content);
  const float scale =
      SfpStream::kMax / (mat->NumElements() + hwy::Unpredictable1() - 1);
  pool.Run(0, extents.rows, [&](const size_t r, size_t /*thread*/) {
    for (size_t c = 0; c < extents.cols; c++) {
      float f = static_cast<float>(r * extents.cols + c) * scale;
      if ((r + c) & 1) f = -f;  // Also generate some negative values.
      content[r * extents.cols + c] = f;
    }
  });

  CompressScaled(content.get(), mat->NumElements(), ws, *mat, pool);
  mat->set_scale(0.6f);  // Arbitrary value, different from 1.
  return mat;
}

// extents describes the transposed matrix.
template <typename MatT>
MatStoragePtr<MatT> GenerateTransposedMat(const Extents2D extents,
                                          hwy::ThreadPool& pool) {
  gcpp::CompressWorkingSet ws;
  auto mat =
      std::make_unique<MatStorageT<MatT>>("trans", extents.rows, extents.cols);
  FloatPtr content = hwy::AllocateAligned<float>(mat->NumElements());
  const float scale =
      SfpStream::kMax / (mat->NumElements() + hwy::Unpredictable1() - 1);
  pool.Run(0, extents.rows, [&](const size_t r, size_t /*thread*/) {
    for (size_t c = 0; c < extents.cols; c++) {
      float f = static_cast<float>(c * extents.rows + r) * scale;
      if ((r + c) & 1) f = -f;  // Also generate some negative values.
      content[r * extents.cols + c] = f;
    }
  });

  CompressScaled(content.get(), mat->NumElements(), ws, *mat, pool);
  // Arbitrary value, different from 1, must match GenerateMat.
  mat->set_scale(0.6f);
  return mat;
}

void PrintSpeed(const Extents2D& A_extents, const Extents2D& B_extents,
                std::vector<double>& times) {
  std::sort(times.begin(), times.end());
  // bench_dnn reports the best and average, but the median seems more
  // consistent and resistant to outliers.
  const double elapsed = times[times.size() / 2];
  const double ratio = elapsed / (times[0] + 1E-6);  // vs best, avoid / 0

  const size_t num_b = B_extents.Area();
  // FMA counts as two FLOP.
  fprintf(stderr, "%.1f\t(med %.3f ms = %0.2fx min)\n",
          2 * 1E-9 * A_extents.rows * num_b / elapsed, elapsed * 1E3, ratio);
}

// Generates inputs and prints observed throughput of MatMul.
// M = A rows, K = A cols, N = C cols.
template <typename MatTA, typename MatTB = MatTA>
void BenchMatMul(size_t M, size_t K, size_t N, bool add, MatMulEnv& env) {
  hwy::ThreadPool& pool = env.parallel.Pools().Pool(0);
  fprintf(stderr, "\nBenchMatMul %lu, %lu, %lu, add=%d, MatTA=%s, MatTB=%s\n",
          M, K, N, add, TypeName<MatTA>(), TypeName<MatTB>());

  const Extents2D A_extents(M, K);
  const Extents2D B_extents(N, K);  // already transposed
  const Extents2D C_extents(M, N);

  RowVectorBatch<float> c_slow_batch(C_extents);
  RowVectorBatch<float> c_batch(C_extents);

  std::unique_ptr<MatStorageT<float>> add_storage;
  if (add) {
    add_storage = GenerateMat<float>(Extents2D(1, N), pool);
    HWY_ASSERT(add_storage);
    add_storage->set_scale(1.0f);
  }

  MatStoragePtr<MatTA> a = GenerateMat<MatTA>(A_extents, pool);
  MatStoragePtr<MatTB> b_trans = GenerateTransposedMat<MatTB>(B_extents, pool);
  HWY_ASSERT(a && b_trans);
  const auto A = ConstMatFromWeights(*a);
  const auto B = ConstMatFromWeights(*b_trans);

  const float* add_row = add ? add_storage->data_scale1() : nullptr;
  const RowPtrF C = RowPtrFromBatch(c_batch);

  constexpr size_t kSamples = 20;
  std::vector<double> times;
  times.reserve(kSamples);

  Tristate use_spinning = Tristate::kDefault;
  env.parallel.Pools().MaybeStartSpinning(use_spinning);

  double keep = 0.0;
  // Until enough samples collected *after* autotuning finished:
  while (times.size() < kSamples) {
    const double t0 = hwy::platform::Now();
    MatMul(A, B, add_row, env, C);
    const double t1 = hwy::platform::Now();
    double elapsed = t1 - t0;
    keep += C.Row(0)[hwy::Unpredictable1()];

    times.push_back(elapsed);
  }
  hwy::PreventElision(keep);
  env.parallel.Pools().MaybeStopSpinning(use_spinning);
  PrintSpeed(A_extents, B_extents, times);
}

using F32 = float;
using SFP = SfpStream;

void BenchAllMatMul() {
  if (first_target == 0) first_target = HWY_TARGET;
  if (HWY_TARGET != first_target) return;

  for (size_t max_packages : {/*1,*/ 2}) {
    const size_t max_threads = 0;  // no limit
    NestedPools pools(max_threads, Tristate::kDefault,
                      BoundedSlice(0, max_packages));
#if GEMMA_DISABLE_TOPOLOGY
    if (max_packages == 2) break;  // we only have one package
#else
    // If less than the limit, we have already tested all num_packages.
    if (pools.Topology().FullTopology().packages.size() < max_packages) break;
#endif
    fprintf(stderr, "BenchAllMatMul %zu: %s %s\n", max_packages,
            pools.TopologyString(), pools.PinString());

    Allocator::Init(pools.Topology());
    MatMulEnv env(pools);

    for (size_t batch_size : {1, 4, 128, 512}) {
      constexpr bool kAdd = false;
      BenchMatMul<BF16, SFP>(batch_size, 24576, 3072, kAdd, env);
      BenchMatMul<BF16, SFP>(batch_size, 3072, 24576, kAdd, env);
    }
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
