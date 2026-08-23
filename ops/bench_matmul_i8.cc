// Copyright 2025 Google LLC
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

// Compares the BF16 MatMul (`ops/matmul-inl.h`) against the W8A8 int8 kernel
// (`ops/matmul_i8-inl.h`) on Gemma-shaped problems, and reports the accuracy
// of both relative to an F64 reference. Standalone binary (no gtest) so that
// it can be run directly.

#include <math.h>

#include <cmath>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <vector>

#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include "ops/matmul.h"
#include "util/basics.h"
#include "util/mat.h"
#include "util/threading_context.h"
#include "hwy/aligned_allocator.h"
#include "hwy/timer.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ops/bench_matmul_i8.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"
#include "ops/matmul-inl.h"
#include "ops/matmul_i8-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Deterministic, reproducible pseudo-Gaussian. Real activations and weights
// are roughly bell-shaped; the ramp in `compression/test_util-inl.h` would
// flatter or penalize int8 quantization for the wrong reasons.
class Rng {
 public:
  explicit Rng(uint64_t seed) : state_(seed * 6364136223846793005ull + 1) {}

  float Normal() {
    // Sum of 4 uniforms: close enough to Gaussian, and cheap.
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) sum += Uniform();
    return (sum - 2.0f) * 1.732f;  // zero mean, unit-ish variance
  }

 private:
  float Uniform() {
    state_ = state_ * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<float>((state_ >> 40) & 0xFFFFFF) / 16777216.0f;
  }
  uint64_t state_;
};

// Fills `mat` with N(0, stddev), and additionally gives a few columns of each
// row a 10x larger magnitude. Outlier channels are the known hard case for
// per-tensor int8; per-row/per-column scales are supposed to absorb them.
void FillNormal(MatStorageT<float>& mat, uint64_t seed, float stddev,
                bool outliers) {
  Rng rng(seed);
  for (size_t r = 0; r < mat.Rows(); ++r) {
    float* HWY_RESTRICT row = mat.Row(r);
    for (size_t c = 0; c < mat.Cols(); ++c) {
      row[c] = rng.Normal() * stddev;
    }
    if (outliers) {
      for (size_t c = (r * 7) % 64; c < mat.Cols(); c += 512) {
        row[c] *= 10.0f;
      }
    }
    for (size_t c = mat.Cols(); c < mat.Stride(); ++c) row[c] = 0.0f;
  }
}

// Converts F32 `in` to `MatT` (BF16 or a compressed stream), row by row.
template <typename MatT>
void ConvertRows(const MatStorageT<float>& in, MatStorageT<MatT>& out,
                 ThreadingContext& ctx) {
  CompressWorkingSet ws;
  ws.tls.resize(ctx.pools.MaxWorkers());
  const size_t cols = in.Cols();
  ParallelFor(Parallelism::kFlat, in.Rows(), ctx, /*cluster_idx=*/0,
              Callers::kTest, [&](size_t r, size_t thread) HWY_ATTR {
                Compress(in.Row(r), cols, ws.tls[thread],
                         MakeSpan(out.Row(r), cols), /*packed_ofs=*/0);
              });
}

//------------------------------------------------------------------------------
// Reference and error metric

// `B` is transposed: `ref[m, n] = sum_k A[m, k] * B[n, k]`.
void ReferenceMatMul(const MatStorageT<float>& A, const MatStorageT<float>& B,
                     MatStorageT<double>& ref, ThreadingContext& ctx) {
  const size_t K = A.Cols();
  ParallelFor(Parallelism::kFlat, A.Rows(), ctx, /*cluster_idx=*/0,
              Callers::kTest, [&](size_t m, size_t /*thread*/) {
                const float* HWY_RESTRICT a = A.Row(m);
                double* HWY_RESTRICT out = ref.Row(m);
                for (size_t n = 0; n < B.Rows(); ++n) {
                  const float* HWY_RESTRICT b = B.Row(n);
                  double sum = 0.0;
                  for (size_t k = 0; k < K; ++k) {
                    sum += static_cast<double>(a[k]) * static_cast<double>(b[k]);
                  }
                  out[n] = sum;
                }
              });
}

// Relative Frobenius error ||C - ref|| / ||ref||.
template <typename TC>
double RelError(const MatStorageT<TC>& C, const MatStorageT<double>& ref) {
  double num = 0.0, den = 0.0;
  for (size_t r = 0; r < ref.Rows(); ++r) {
    const TC* HWY_RESTRICT c = C.Row(r);
    const double* HWY_RESTRICT e = ref.Row(r);
    for (size_t n = 0; n < ref.Cols(); ++n) {
      const double d = hwy::ConvertScalarTo<double>(c[n]) - e[n];
      num += d * d;
      den += e[n] * e[n];
    }
  }
  return (den == 0.0) ? 0.0 : std::sqrt(num / den);
}

//------------------------------------------------------------------------------
// Timing

struct Result {
  double median_sec = 0.0;
  double gflops = 0.0;
  double rel_error = -1.0;  // < 0 if not measured
};

// Repeats `fn` (which returns the autotune state) until autotuning has settled,
// then collects `num_samples` timings and returns the median.
template <class Fn>
Result TimeMatMul(size_t M, size_t K, size_t N, Fn&& fn) {
  const size_t num_samples = M < 32 ? 40 : 12;
  std::vector<double> times;
  times.reserve(num_samples);

  // Bound the loop: a config that never reports Best() would otherwise hang.
  // Skip a few runs after autotuning settles, so the first timed sample is not
  // the one that still has the autotuner's working set in cache.
  size_t warmup = 3;
  for (size_t iter = 0; times.size() < num_samples && iter < 8192; ++iter) {
    const double t0 = hwy::platform::Now();
    MMPerKey* per_key = fn();
    const double t1 = hwy::platform::Now();
    if (!per_key->autotune.Best()) continue;
    if (warmup != 0) {
      --warmup;
      continue;
    }
    times.push_back(t1 - t0);
  }
  HWY_ASSERT(!times.empty());

  std::sort(times.begin(), times.end());
  Result r;
  r.median_sec = times[times.size() / 2];
  r.gflops = 2.0 * M * K * N / r.median_sec * 1E-9;
  return r;
}

//------------------------------------------------------------------------------
// One shape

// Runs BF16xBF16, BF16xSFP and int8 W8A8 on the same `M x K x N` problem.
// `check_error` also computes the F64 reference, which is O(M*K*N) scalar work
// and thus only affordable for smaller shapes.
void BenchShape(size_t M, size_t K, size_t N, bool check_error,
                ThreadingContext& ctx, MatMulEnv& env_bf, MatMulEnv& env_sfp,
                MatMulEnv& env_i8, MMI8AStorage& a_i8,
                bool a_outliers = true) {
  const Allocator& allocator = ctx.allocator;
  const Extents2D A_extents(M, K);
  const Extents2D B_extents(N, K);  // already transposed
  const Extents2D C_extents(M, N);

  // Sources, in F32.
  MatStorageT<float> A_f32("A_f32", A_extents, allocator, MatPadding::kOdd);
  MatStorageT<float> B_f32("B_f32", B_extents, allocator, MatPadding::kOdd);
  FillNormal(A_f32, /*seed=*/1, /*stddev=*/1.0f, a_outliers);
  FillNormal(B_f32, /*seed=*/2, /*stddev=*/0.02f, /*outliers=*/false);

  // Operands for the BF16 kernel.
  MatStorageT<BF16> A_bf("A_bf", A_extents, allocator, MatPadding::kOdd);
  MatStorageT<BF16> B_bf("B_bf", B_extents, allocator, MatPadding::kOdd);
  MatStorageT<SfpStream> B_sfp("B_sfp", B_extents, allocator, MatPadding::kOdd);
  ConvertRows(A_f32, A_bf, ctx);
  ConvertRows(B_f32, B_bf, ctx);
  ConvertRows(B_f32, B_sfp, ctx);

  // Operands for the int8 kernel. `A` is quantized inside `MatMulI8`.
  MatStorageT<int8_t> B_i8("B_i8", B_extents, allocator, MatPadding::kOdd);
  hwy::AlignedVector<float> b_scale(N);
  const MMI8B B_packed = PackB(B_f32, B_i8, b_scale.data(), ctx);

  MatStorageT<float> C_bf("C_bf", C_extents, allocator, MatPadding::kOdd);
  MatStorageT<float> C_sfp("C_sfp", C_extents, allocator, MatPadding::kOdd);
  MatStorageT<float> C_i8("C_i8", C_extents, allocator, MatPadding::kOdd);
  C_bf.AllocateAndAttachRowPtrs(env_bf.row_ptrs);
  C_sfp.AllocateAndAttachRowPtrs(env_sfp.row_ptrs);
  C_i8.AllocateAndAttachRowPtrs(env_i8.row_ptrs);

  Tristate use_spinning = Tristate::kDefault;
  ctx.pools.MaybeStartSpinning(use_spinning);

  const Result r_bf = TimeMatMul(M, K, N, [&] {
    return MatMul(A_bf, B_bf, /*add=*/nullptr, env_bf, C_bf);
  });
  const Result r_sfp = TimeMatMul(M, K, N, [&] {
    return MatMul(A_bf, B_sfp, /*add=*/nullptr, env_sfp, C_sfp);
  });
  const Result r_i8 = TimeMatMul(M, K, N, [&] {
    return MatMulI8(A_bf, B_packed, /*add=*/nullptr, env_i8, C_i8, a_i8);
  });

  ctx.pools.MaybeStopSpinning(use_spinning);

  double e_bf = -1.0, e_sfp = -1.0, e_i8 = -1.0;
  if (check_error) {
    MatStorageT<double> ref("ref", C_extents, allocator, MatPadding::kOdd);
    ReferenceMatMul(A_f32, B_f32, ref, ctx);
    e_bf = RelError(C_bf, ref);
    e_sfp = RelError(C_sfp, ref);
    e_i8 = RelError(C_i8, ref);
  }

  printf("%5zu %6zu %7zu | %8.1f %8.1f %8.1f | %6.3f %6.3f %6.3f | %5.2fx %5.2fx",
         M, K, N, r_bf.gflops, r_sfp.gflops, r_i8.gflops,
         r_bf.median_sec * 1E3, r_sfp.median_sec * 1E3, r_i8.median_sec * 1E3,
         r_i8.gflops / r_bf.gflops, r_i8.gflops / r_sfp.gflops);
  if (check_error) {
    printf(" | %.2e %.2e %.2e", e_bf, e_sfp, e_i8);
  }
  printf("\n");
  fflush(stdout);
}

// Measures raw instruction throughput of the two dot products with 16
// independent accumulator chains, no memory traffic. The matmul speedups below
// cannot exceed this ratio, and how close they get says how much of the win is
// compute rather than the halved footprint of `B`.
void BenchDotThroughput() {
  const hn::ScalableTag<BF16> dbf;
  const hn::Repartition<float, decltype(dbf)> df;
  const hn::ScalableTag<int8_t> di8;
  const hn::Repartition<int32_t, decltype(di8)> di32;
  constexpr size_t kChains = 16;
  constexpr size_t kReps = 2000000;

  const size_t bf16_macs = hn::Lanes(dbf);  // per instruction
  const size_t i8_macs = hn::Lanes(di8);

  double keep = 0.0;
  double bf_sec = 0.0, i8_sec = 0.0;

  {
    const auto a = hn::Set(dbf, hwy::ConvertScalarTo<BF16>(1.0f));
    hn::Vec<decltype(df)> c[kChains], unused = hn::Zero(df);
    for (size_t i = 0; i < kChains; ++i) c[i] = hn::Zero(df);
    const double t0 = hwy::platform::Now();
    for (size_t r = 0; r < kReps; ++r) {
      for (size_t i = 0; i < kChains; ++i) {
        c[i] = hn::ReorderWidenMulAccumulate(df, a, a, c[i], unused);
      }
    }
    bf_sec = hwy::platform::Now() - t0;
    for (size_t i = 0; i < kChains; ++i) keep += hn::GetLane(c[i]);
  }
  {
    const auto a = hn::Set(di8, int8_t{1});
    hn::Vec<decltype(di32)> c[kChains];
    for (size_t i = 0; i < kChains; ++i) c[i] = hn::Zero(di32);
    const double t0 = hwy::platform::Now();
    for (size_t r = 0; r < kReps; ++r) {
      for (size_t i = 0; i < kChains; ++i) {
        c[i] = hn::SumOfMulQuadAccumulate(di32, a, a, c[i]);
      }
    }
    i8_sec = hwy::platform::Now() - t0;
    for (size_t i = 0; i < kChains; ++i) keep += hn::GetLane(c[i]);
  }
  hwy::PreventElision(keep);

  const double ops = static_cast<double>(kChains) * kReps;
  const double bf_gmac = ops * bf16_macs / bf_sec * 1E-9;
  const double i8_gmac = ops * i8_macs / i8_sec * 1E-9;
  printf(
      "1-core dot product throughput: bf16 %.1f GMAC/s, int8 %.1f GMAC/s "
      "(%.2fx)\n",
      bf_gmac, i8_gmac, i8_gmac / bf_gmac);
}

void BenchAll() {
  ThreadingArgs threading_args;
  ThreadingContext ctx(threading_args);
  printf("target=%s %s %s\n", hwy::TargetName(HWY_TARGET),
         ctx.topology.TopologyString(), ctx.pools.PinString());
  printf("B biased to u8: %d, HWY_NATIVE_DOT_BF16=%d, vector bytes=%zu\n",
         GEMMA_MM_I8_BIASED_B, HWY_NATIVE_DOT_BF16,
         hn::Lanes(hn::ScalableTag<uint8_t>()));

  BenchDotThroughput();

  MatMulEnv env_bf(ctx), env_sfp(ctx), env_i8(ctx);
  // Sized for the largest shape below.
  MMI8AStorage a_i8(/*max_M=*/512, /*max_K=*/8192, ctx.allocator);

  printf(
      "\n    M      K       N |  GFLOPS: bf16   sfp     i8 |     ms: bf16    "
      "sfp     i8 | i8 vs bf16/sfp | rel err: bf16 sfp i8\n");

  // Gemma3-1B decode shapes, as in `ops/bench_matmul.cc`.
  for (size_t M : {size_t{1}, size_t{4}}) {
    BenchShape(M, 1152, 1536, /*check_error=*/false, ctx, env_bf, env_sfp,
               env_i8, a_i8);  // QKV
    BenchShape(M, 1152, 13824, false, ctx, env_bf, env_sfp, env_i8,
               a_i8);  // FFN gate+up
    BenchShape(M, 6912, 1152, false, ctx, env_bf, env_sfp, env_i8,
               a_i8);  // FFN down
    BenchShape(M, 1152, 32768, false, ctx, env_bf, env_sfp, env_i8,
               a_i8);  // logits (N reduced to fit memory)
  }

  // Prefill / batched shapes.
  BenchShape(128, 3072, 3072, false, ctx, env_bf, env_sfp, env_i8, a_i8);
  BenchShape(512, 3072, 3072, false, ctx, env_bf, env_sfp, env_i8, a_i8);
  BenchShape(128, 1152, 13824, false, ctx, env_bf, env_sfp, env_i8, a_i8);

  // B larger than last-level cache in both formats, so neither kernel can
  // hide the streaming cost of B.
  printf("\n(B exceeds LLC in both formats)\n");
  BenchShape(1, 4096, 32768, false, ctx, env_bf, env_sfp, env_i8, a_i8);
  BenchShape(8, 4096, 32768, false, ctx, env_bf, env_sfp, env_i8, a_i8);
  BenchShape(128, 4096, 32768, false, ctx, env_bf, env_sfp, env_i8, a_i8);

  printf(
      "\nAccuracy vs F64 reference. A has outlier channels (10x), which is the"
      "\nknown hard case for per-token int8 activations:\n");
  BenchShape(32, 1152, 512, /*check_error=*/true, ctx, env_bf, env_sfp, env_i8,
             a_i8, /*a_outliers=*/true);
  BenchShape(32, 3072, 512, true, ctx, env_bf, env_sfp, env_i8, a_i8, true);
  BenchShape(32, 6912, 512, true, ctx, env_bf, env_sfp, env_i8, a_i8, true);

  printf("\nSame, but A is plain Gaussian with no outlier channels:\n");
  BenchShape(32, 1152, 512, true, ctx, env_bf, env_sfp, env_i8, a_i8,
             /*a_outliers=*/false);
  BenchShape(32, 3072, 512, true, ctx, env_bf, env_sfp, env_i8, a_i8, false);
  BenchShape(32, 6912, 512, true, ctx, env_bf, env_sfp, env_i8, a_i8, false);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {
HWY_EXPORT(BenchAll);
}  // namespace gcpp

int main(int /*argc*/, char** /*argv*/) {
  // Best available target only; this is a benchmark, not a test.
  HWY_DYNAMIC_DISPATCH(gcpp::BenchAll)();
  return 0;
}
#endif  // HWY_ONCE
