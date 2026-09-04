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

// Correctness of the W8A8 kernel in `ops/matmul_i8-inl.h`. The reference is
// computed in F64 from the *quantized* operands, so this checks the kernel's
// arithmetic (accumulation, remainder handling, the u8 bias correction, and
// the MMSetC/MMAddC split across kc ranges) rather than quantization error.
//
// Built twice, with `GEMMA_MM_I8_FORCE_BIASED_B` 0 and 1, so that the x86
// biased-u8 path is covered on non-x86 hosts too.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <cmath>
#include <vector>

#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include "hwy/aligned_allocator.h"
#include "ops/matmul.h"
#include "util/basics.h"
#include "util/mat.h"
#include "util/threading_context.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ops/matmul_i8_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"
#include "ops/matmul-inl.h"
#include "ops/matmul_i8-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {

// Not in HWY_NAMESPACE: the `HWY_ONCE` section below must read the same
// instance that the dispatched target wrote to.
extern size_t g_failures;

namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

class Rng {
 public:
  explicit Rng(uint64_t seed) : state_(seed * 6364136223846793005ull + 1) {}
  float Normal() {
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) sum += Uniform();
    return (sum - 2.0f) * 1.732f;
  }

 private:
  float Uniform() {
    state_ = state_ * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<float>((state_ >> 40) & 0xFFFFFF) / 16777216.0f;
  }
  uint64_t state_;
};

void TestRotationPreservesDotProducts(size_t block_size, size_t hash_bits) {
  const size_t size = 2 * block_size;
  std::vector<float> a(size);
  std::vector<float> b(size);
  Rng rng(123);
  double expected = 0.0;
  for (size_t i = 0; i < size; ++i) {
    a[i] = rng.Normal();
    b[i] = rng.Normal();
    expected += static_cast<double>(a[i]) * b[i];
  }

  MMI8Rotate(a.data(), a.size(), block_size, hash_bits);
  MMI8Rotate(b.data(), b.size(), block_size, hash_bits);
  double actual = 0.0;
  for (size_t i = 0; i < size; ++i) {
    actual += static_cast<double>(a[i]) * b[i];
  }

  const double relative = hwy::ScalarAbs(actual - expected) /
                          HWY_MAX(1.0, hwy::ScalarAbs(expected));
  if (relative > 1E-6) {
    ++g_failures;
    printf("FAIL rotation block=%zu hash=%zu dot relative error %.3e\n",
           block_size, hash_bits, relative);
  } else {
    printf("  ok rotation block=%zu hash=%zu preserves dot (%.3e)\n",
           block_size, hash_bits, relative);
  }
}

void TestHash16() {
  std::vector<bool> seen(65536, false);
  size_t negatives = 0;
  bool deterministic = true;
  for (size_t i = 0; i < 65536; ++i) {
    const uint16_t hash = MMI8Hash16(static_cast<uint16_t>(i));
    deterministic &= hash == MMI8Hash16(static_cast<uint16_t>(i));
    if (seen[hash]) {
      ++g_failures;
      printf("FAIL 16-bit hash collision at %zu\n", i);
      return;
    }
    seen[hash] = true;
    negatives += MMI8NegativeSign(i, 16) ? 1 : 0;
  }

  bool short_period = false;
  for (size_t period = 1; period <= 256; ++period) {
    bool matches = true;
    for (size_t i = 0; i < 4096; ++i) {
      if (MMI8NegativeSign(i, 16) != MMI8NegativeSign(i + period, 16)) {
        matches = false;
        break;
      }
    }
    short_period |= matches;
  }

  const bool ok = deterministic && negatives == 32768 && !short_period;
  if (!ok) ++g_failures;
  printf("%s 16-bit hash deterministic=%d negatives=%zu short_period=%d\n",
         ok ? "  ok" : "FAIL", deterministic, negatives, short_period);
}

void TestL2Scaling() {
  bool clamped = false;
  const float balanced = MMI8L2Scale(4.0, 1.0, &clamped);
  const bool balanced_ok = hwy::ScalarAbs(balanced - 0.5f) < 1E-7f && !clamped;
  const float both_zero = MMI8L2Scale(0.0, 0.0, &clamped);
  const bool zero_ok = both_zero == 1.0f && !clamped;
  const float low = MMI8L2Scale(1E20, 1E-20, &clamped);
  const bool low_ok = low == kMMI8L2ScaleMin && clamped;
  const float high = MMI8L2Scale(1E-20, 1E20, &clamped);
  const bool high_ok = high == kMMI8L2ScaleMax && clamped;

  constexpr size_t kSize = 256;
  std::vector<float> a(kSize), b(kSize), scale(kSize);
  Rng rng(456);
  double expected = 0.0;
  for (size_t i = 0; i < kSize; ++i) {
    a[i] = rng.Normal();
    b[i] = rng.Normal();
    scale[i] = MMI8L2Scale(0.25 + (i % 13), 0.5 + (i % 17));
    expected += static_cast<double>(a[i]) * b[i];
    a[i] *= scale[i];
    b[i] /= scale[i];
  }

  double scaled_dot = 0.0;
  for (size_t i = 0; i < kSize; ++i) {
    scaled_dot += static_cast<double>(a[i]) * b[i];
  }
  MMI8Rotate(a.data(), a.size(), 64, 16);
  MMI8Rotate(b.data(), b.size(), 64, 16);
  double transformed_dot = 0.0;
  for (size_t i = 0; i < kSize; ++i) {
    transformed_dot += static_cast<double>(a[i]) * b[i];
  }
  const double scaled_relative = hwy::ScalarAbs(scaled_dot - expected) /
                                 HWY_MAX(1.0, hwy::ScalarAbs(expected));
  const double transformed_relative =
      hwy::ScalarAbs(transformed_dot - expected) /
      HWY_MAX(1.0, hwy::ScalarAbs(expected));
  const bool invariant =
      scaled_relative <= 1E-6 && transformed_relative <= 1E-6;
  const bool ok = balanced_ok && zero_ok && low_ok && high_ok && invariant;
  if (!ok) ++g_failures;
  printf(
      "%s L2 scaling calculation/clamping and dot invariance "
      "(scaled %.3e, transformed %.3e)\n",
      ok ? "  ok" : "FAIL", scaled_relative, transformed_relative);
}

// Fills A and B. Row magnitudes deliberately vary by up to 7x, so that a
// mixed-up per-row scale index would show up.
void FillOperands(size_t M, size_t K, size_t N, MatStorageT<float>& A_f32,
                  MatStorageT<float>& B_f32, float b_mean = 0.0f) {
  Rng rng(M * 131 + K * 17 + N);
  for (size_t r = 0; r < M; ++r) {
    float* row = A_f32.Row(r);
    const float row_scale = 0.01f * static_cast<float>(1 + (r % 7));
    for (size_t c = 0; c < K; ++c) row[c] = rng.Normal() * row_scale;
    for (size_t c = K; c < A_f32.Stride(); ++c) row[c] = 0.0f;
  }
  for (size_t r = 0; r < N; ++r) {
    float* row = B_f32.Row(r);
    const float row_scale = 0.5f * static_cast<float>(1 + (r % 5));
    // A nonzero mean makes the per-channel sums of the quantized weights
    // large. Correcting `B`'s bias once over the whole `K` (rather than per
    // `kc` range) would then write intermediates to `C` that are far larger
    // than the result, which is unrecoverable when `C` is BF16.
    for (size_t c = 0; c < K; ++c) {
      row[c] = (rng.Normal() + b_mean) * row_scale;
    }
    for (size_t c = K; c < B_f32.Stride(); ++c) row[c] = 0.0f;
  }
}

// One `M x K x N` case. `TC` is the output type; `add` exercises the bias.
template <typename TC>
void TestCase(size_t M, size_t K, size_t N, bool add, ThreadingContext& ctx,
              MatMulEnv& env, MMI8AStorage& a_i8, float b_mean = 0.0f,
              const float* a_pre_scale = nullptr) {
  const Allocator& allocator = ctx.allocator;
  MatStorageT<float> A_f32("A", Extents2D(M, K), allocator, MatPadding::kOdd);
  MatStorageT<float> B_f32("B", Extents2D(N, K), allocator, MatPadding::kOdd);
  FillOperands(M, K, N, A_f32, B_f32, b_mean);
  // Non-unit tensor scales, which must be folded in by QuantizeA/PackB.
  A_f32.SetScale(0.75f);
  B_f32.SetScale(1.25f);

  MatStorageT<int8_t> B_i8("B_i8", Extents2D(N, K), allocator,
                           MatPadding::kOdd);
  hwy::AlignedVector<float> b_scale(N), add_row(N);
  const MMI8B B_packed = PackB(B_f32, B_i8, b_scale.data(), ctx, a_pre_scale);
  for (size_t n = 0; n < N; ++n)
    add_row[n] = 0.125f * static_cast<float>(n % 9);

  MatStorageT<TC> C("C", Extents2D(M, N), allocator, MatPadding::kOdd);
  C.AllocateAndAttachRowPtrs(env.row_ptrs);
  // Run until autotuning settles, then check the result produced by the best
  // config. Otherwise every call would use a different blocking, and with a
  // BF16 `C` the number of kc ranges changes how much precision is lost.
  MMPerKey* per_key = nullptr;
  for (size_t iter = 0; iter < 4096; ++iter) {
    per_key =
        MatMulI8(A_f32, B_packed, add ? add_row.data() : nullptr, env, C, a_i8);
    if (per_key->autotune.Best()) break;
  }
  HWY_ASSERT(per_key->autotune.Best());
  const size_t kc = per_key->autotune.Best()->KC();
  const size_t k_ranges = per_key->autotune.Best()->RangesOfKC(K).NumTasks();
  MatMulI8(A_f32, B_packed, add ? add_row.data() : nullptr, env, C, a_i8);

  // Reference from the quantized operands. `QuantizeA` has already written
  // them, so read them back rather than re-deriving.
  const MMI8AView A_q = a_i8.View(Extents2D(M, K));
  double max_abs_err = 0.0;
  double sum_sq = 0.0;
  for (size_t m = 0; m < M; ++m) {
    const MMI8AT* qa = A_q.data.Row(m);
    for (size_t n = 0; n < N; ++n) {
      const MMI8BT* qb = HWY_RCAST_ALIGNED(const MMI8BT*, B_i8.Row(n));
      int64_t dot = 0;
      for (size_t k = 0; k < K; ++k) {
        const int32_t b =
            static_cast<int32_t>(qb[k]) - (GEMMA_MM_I8_BIASED_B ? 128 : 0);
        dot += static_cast<int64_t>(qa[k]) * static_cast<int64_t>(b);
      }
      const double expected = static_cast<double>(A_q.scale[m]) * b_scale[n] *
                                  static_cast<double>(dot) +
                              (add ? add_row[n] : 0.0f);
      const double actual = hwy::ConvertScalarTo<double>(C.Row(m)[n]);
      max_abs_err = HWY_MAX(max_abs_err, hwy::ScalarAbs(actual - expected));
      sum_sq += expected * expected;
    }
  }
  // Individual outputs are sums of `K` signed products and can cancel to near
  // zero, where an elementwise relative error is meaningless. Normalize the
  // worst absolute error by the RMS of the expected outputs instead.
  const double rms = std::sqrt(sum_sq / static_cast<double>(M * N));
  const double err = (rms == 0.0) ? 0.0 : max_abs_err / rms;

  // BF16 output has 8 mantissa bits, and `MMAddC` rounds once per kc range.
  const double tolerance = IsBF16<TC>() ? 6E-2 : 1E-5;
  const bool ok = err <= tolerance;
  if (!ok) ++g_failures;
  printf(
      "%s M=%4zu K=%5zu N=%5zu add=%d TC=%-5s biasedB=%d kc=%5zu(x%zu) "
      "err/rms=%.2e\n",
      ok ? "  ok  " : "FAILED", M, K, N, add, TypeName<TC>(),
      GEMMA_MM_I8_BIASED_B, kc, k_ranges, err);
}

// Control: how much precision the *existing* BF16 kernel loses when `TC` is
// BF16 and `K` spans several kc ranges, so that `MMAddC` accumulates through
// BF16. Reported as a reference point for the int8 kernel's BF16-output
// tolerance, since both inherit this from `MMStoreHorizontalSumsIntoC`.
void ControlBF16OutputError(size_t M, size_t K, size_t N, ThreadingContext& ctx,
                            MatMulEnv& env) {
  const Allocator& allocator = ctx.allocator;
  MatStorageT<float> A_f32("A", Extents2D(M, K), allocator, MatPadding::kOdd);
  MatStorageT<float> B_f32("B", Extents2D(N, K), allocator, MatPadding::kOdd);
  FillOperands(M, K, N, A_f32, B_f32);

  MatStorageT<BF16> A_bf("A_bf", Extents2D(M, K), allocator, MatPadding::kOdd);
  MatStorageT<BF16> B_bf("B_bf", Extents2D(N, K), allocator, MatPadding::kOdd);
  CompressWorkingSet ws;
  ws.tls.resize(ctx.pools.MaxWorkers());
  for (size_t r = 0; r < M; ++r) {
    Compress(A_f32.Row(r), K, ws.tls[0], MakeSpan(A_bf.Row(r), K), 0);
  }
  for (size_t r = 0; r < N; ++r) {
    Compress(B_f32.Row(r), K, ws.tls[0], MakeSpan(B_bf.Row(r), K), 0);
  }

  MatStorageT<float> C_f32("Cf", Extents2D(M, N), allocator, MatPadding::kOdd);
  MatStorageT<BF16> C_bf("Cb", Extents2D(M, N), allocator, MatPadding::kOdd);
  C_f32.AllocateAndAttachRowPtrs(env.row_ptrs);
  for (size_t iter = 0; iter < 4096; ++iter) {
    if (MatMul(A_bf, B_bf, nullptr, env, C_f32)->autotune.Best()) break;
  }
  MatMul(A_bf, B_bf, nullptr, env, C_f32);
  C_bf.AllocateAndAttachRowPtrs(env.row_ptrs);
  MMPerKey* per_key = nullptr;
  for (size_t iter = 0; iter < 4096; ++iter) {
    per_key = MatMul(A_bf, B_bf, nullptr, env, C_bf);
    if (per_key->autotune.Best()) break;
  }
  HWY_ASSERT(per_key->autotune.Best());
  const size_t kc = per_key->autotune.Best()->KC();
  const size_t k_ranges = per_key->autotune.Best()->RangesOfKC(K).NumTasks();
  MatMul(A_bf, B_bf, nullptr, env, C_bf);

  double max_abs = 0.0, sum_sq = 0.0;
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      const double f = C_f32.Row(m)[n];
      const double b = hwy::ConvertScalarTo<double>(C_bf.Row(m)[n]);
      max_abs = HWY_MAX(max_abs, hwy::ScalarAbs(f - b));
      sum_sq += f * f;
    }
  }
  const double rms = std::sqrt(sum_sq / static_cast<double>(M * N));
  printf(
      "control  M=%4zu K=%5zu N=%5zu kc=%5zu(x%zu) bf16 kernel, TC=bf16 vs "
      "TC=f32: err/rms=%.2e\n",
      M, K, N, kc, k_ranges, rms == 0.0 ? 0.0 : max_abs / rms);
}

void TestAll() {
  ThreadingArgs threading_args;
  ThreadingContext ctx(threading_args);
  MatMulEnv env(ctx);
  printf("target=%s biasedB=%d block=%zu hash=%zu vector bytes=%zu\n",
         hwy::TargetName(HWY_TARGET), GEMMA_MM_I8_BIASED_B,
         MMI8RotateBlockSize(), MMI8HashBits(),
         hn::Lanes(hn::ScalableTag<uint8_t>()));
  for (size_t block : {size_t{64}, size_t{128}}) {
    for (size_t hash : {size_t{16}, size_t{32}})
      TestRotationPreservesDotProducts(block, hash);
  }
  TestHash16();
  TestL2Scaling();

  // `kMaxKC` is 6 KiB, so K = 20096 forces several kc ranges and thus the
  // MMSetC-then-MMAddC path where the bias correction must be applied once.
  MMI8AStorage a_i8(/*max_M=*/64, /*max_K=*/20096, ctx.allocator);

  // Smallest supported K and multiple Hadamard block counts.
  const size_t block = MMI8RotateBlockSize();
  for (size_t K : {block, 2 * block, 3 * block}) {
    TestCase<float>(4, K, 8, /*add=*/false, ctx, env, a_i8);
  }
  std::vector<float> pre_scale(1152);
  for (size_t i = 0; i < pre_scale.size(); ++i)
    pre_scale[i] = 0.25f + 0.01f * static_cast<float>(i % 100);
  TestCase<float>(4, 1152, 12, false, ctx, env, a_i8, 0.0f, pre_scale.data());

  // `kRowsAC` 1/2/4 and the M remainder handling in `A2C0`.
  for (size_t M : {size_t{1}, size_t{2}, size_t{3}, size_t{4}, size_t{5},
                   size_t{7}, size_t{8}, size_t{13}, size_t{64}}) {
    TestCase<float>(M, 1152, 12, /*add=*/true, ctx, env, a_i8);
  }

  // N is required to be a multiple of kNR.
  for (size_t N :
       {size_t{4}, size_t{8}, size_t{16}, size_t{100}, size_t{1536}}) {
    TestCase<float>(4, 512, N, /*add=*/false, ctx, env, a_i8);
  }

  // Multiple kc ranges: exercises MMAddC accumulation and the once-only
  // application of the u8 bias correction.
  TestCase<float>(1, 20096, 8, false, ctx, env, a_i8);
  TestCase<float>(4, 20096, 64, true, ctx, env, a_i8);
  TestCase<float>(32, 12416, 64, true, ctx, env, a_i8);

  // BF16 output. The tolerance is loose because `MMAddC` accumulates through
  // `C`, so with several kc ranges the intermediate sums are rounded to BF16;
  // the control below shows the existing kernel does the same.
  TestCase<BF16>(4, 1152, 64, false, ctx, env, a_i8);
  TestCase<BF16>(32, 20096, 64, false, ctx, env, a_i8);
  TestCase<BF16>(32, 20096, 64, true, ctx, env, a_i8);
  ControlBF16OutputError(4, 1152, 64, ctx, env);
  ControlBF16OutputError(32, 20096, 64, ctx, env);

  // Weights with a large nonzero channel mean, across several kc ranges. This
  // is the case that a whole-K bias correction gets badly wrong.
  TestCase<float>(32, 20096, 64, true, ctx, env, a_i8, /*b_mean=*/3.0f);
  TestCase<BF16>(32, 20096, 64, true, ctx, env, a_i8, /*b_mean=*/3.0f);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {
size_t g_failures = 0;
HWY_EXPORT(TestAll);
}  // namespace gcpp

int main(int /*argc*/, char** /*argv*/) {
  HWY_DYNAMIC_DISPATCH(gcpp::TestAll)();
  const size_t failures = gcpp::g_failures;
  printf("%s (%zu failures)\n", failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}
#endif  // HWY_ONCE
