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

#include "ops/matmul_i8_model-inl.h"

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

void TestRotationMatchesScalar(size_t block_size, size_t hash_bits) {
  std::vector<float> expected(3 * block_size);
  Rng rng(891);
  for (float& value : expected) value = rng.Normal();
  auto actual = expected;
  MMI8Rotate(actual.data(), actual.size(), block_size, hash_bits);
  for (size_t start = 0; start < expected.size(); start += block_size) {
    for (size_t i = 0; i < block_size; ++i) {
      if (MMI8NegativeSign(start + i, hash_bits))
        expected[start + i] = -expected[start + i];
    }
    for (size_t width = 1; width < block_size; width *= 2) {
      for (size_t group = 0; group < block_size; group += 2 * width) {
        for (size_t i = 0; i < width; ++i) {
          const size_t at = start + group + i;
          const float left = expected[at], right = expected[at + width];
          expected[at] = left + right;
          expected[at + width] = left - right;
        }
      }
    }
    const float norm = block_size == 64 ? 0.125f : 0.08838834764831845f;
    for (size_t i = 0; i < block_size; ++i) expected[start + i] *= norm;
  }
  if (memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) !=
      0) {
    ++g_failures;
    printf("FAIL SIMD/scalar transform mismatch block=%zu hash=%zu\n",
           block_size, hash_bits);
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
              const float* a_pre_scale = nullptr, size_t block_size = 0) {
  const Allocator& allocator = ctx.allocator;
  MatStorageT<float> A_f32("A", Extents2D(M, K), allocator, MatPadding::kOdd);
  MatStorageT<float> B_f32("B", Extents2D(N, K), allocator, MatPadding::kOdd);
  FillOperands(M, K, N, A_f32, B_f32, b_mean);
  // Non-unit tensor scales, which must be folded in by QuantizeA/PackB.
  A_f32.SetScale(0.75f);
  B_f32.SetScale(1.25f);

  MatStorageT<int8_t> B_i8("B_i8", Extents2D(N, K), allocator,
                           MatPadding::kOdd);
  hwy::AlignedVector<float> b_scale(N * (block_size ? K / block_size : 1)),
      add_row(N);
  const MMI8B B_packed =
      PackB(B_f32, B_i8, b_scale.data(), ctx, a_pre_scale, block_size);
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
  if (!env.autotune && MMI8Flag("GEMMA_MM_I8_MIN_K_SPLITS"))
    HWY_ASSERT(K > kMaxKC || k_ranges == 1);
  MatMulI8(A_f32, B_packed, add ? add_row.data() : nullptr, env, C, a_i8);

  // Reference from the quantized operands. `QuantizeA` has already written
  // them, so read them back rather than re-deriving.
  const MMI8AView A_q = a_i8.View(Extents2D(M, K), block_size);
  double max_abs_err = 0.0;
  double sum_sq = 0.0;
  for (size_t m = 0; m < M; ++m) {
    const MMI8AT* qa = A_q.data.Row(m);
    for (size_t n = 0; n < N; ++n) {
      const MMI8BT* qb = HWY_RCAST_ALIGNED(const MMI8BT*, B_i8.Row(n));
      double expected = add ? add_row[n] : 0.0;
      const size_t group_size = block_size ? block_size : K;
      for (size_t begin = 0; begin < K; begin += group_size) {
        int64_t dot = 0;
        for (size_t k = begin; k < begin + group_size; ++k) {
          const int32_t b =
              static_cast<int32_t>(qb[k]) - (GEMMA_MM_I8_BIASED_B ? 128 : 0);
          dot += static_cast<int64_t>(qa[k]) * b;
        }
        const size_t group = begin / group_size;
        expected +=
            static_cast<double>(A_q.scale[group * A_q.scale_stride + m]) *
            b_scale[group * N + n] * static_cast<double>(dot);
      }
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
  if (MMI8Flag("GEMMA_MM_I8_TEST_FIXED")) {
    // Stable digest for comparing fast/reference kernels in separate runs.
    uint64_t digest = 14695981039346656037ull;
    for (size_t r = 0; r < M; ++r) {
      const auto* bytes = reinterpret_cast<const unsigned char*>(C.Row(r));
      for (size_t c = 0; c < N * sizeof(TC); ++c)
        digest = (digest ^ bytes[c]) * 1099511628211ull;
    }
    printf("DIGEST M=%zu K=%zu N=%zu block=%zu TC=%s %016llx\n", M, K, N,
           block_size, TypeName<TC>(), static_cast<unsigned long long>(digest));
  }
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

// Packing must preserve exact values, KC rounding, partial N tiles, row
// offsets, and first-KC bias semantics for both F32 and BF16 outputs.
template <typename TC>
void TestPackedMicro(ThreadingContext& ctx, size_t k, bool dense = false) {
  constexpr size_t m = 4, n = 24;
  MatMulEnv env(ctx);
  MatStorageT<float> a("a", Extents2D(m, k), ctx.allocator, MatPadding::kOdd);
  MatStorageT<float> b("b", Extents2D(n, k), ctx.allocator, MatPadding::kOdd);
  FillOperands(m, k, n, a, b, 0.3f);
  MatStorageT<int8_t> q("q", b.Extents(), ctx.allocator, MatPadding::kOdd);
  MatStorageT<int8_t> packed("packed", b.Extents(), ctx.allocator,
                             dense ? MatPadding::kPacked : MatPadding::kOdd);
  MatStorageT<TC> expected("expected", Extents2D(m, n), ctx.allocator,
                           MatPadding::kOdd);
  MatStorageT<TC> actual("actual", expected.Extents(), ctx.allocator,
                         MatPadding::kOdd);
  MMI8AStorage storage(m, k, ctx.allocator);
  hwy::AlignedVector<float> scales(n * k / 32), add(n), bias(n);
  for (size_t c = 0; c < n; ++c) {
    add[c] = static_cast<float>(c % 5) * 0.0625f;
    bias[c] = -static_cast<float>(c % 7) * 0.09375f;
  }
  const MMConfig cfg(1, k, n, 1, 1, k, n, 1, 4, MMOrder::kNT, 1);
  MMAutoTune<MMConfig> tuner;
  tuner.SetCandidates({cfg}, false);
  size_t cases = 0;
  bool ok = true;
  for (size_t block : {size_t{32}, size_t{64}, size_t{128}}) {
    auto ordinary = PackB(b, q, scales.data(), ctx, nullptr, block);
    for (size_t r = 0; r < n; ++r)
      hwy::CopyBytes(q.Row(r), packed.Row(r), k);
    MMI8PackMicroB(packed);
    auto interleaved = ordinary;
    interleaved.data = &packed;
    interleaved.packed_micro = true;
    const auto av = QuantizeA(a, storage, ctx, 0, nullptr, block);
    for (int bias_mode = 0; bias_mode < 4; ++bias_mode) {
      ordinary.bias = interleaved.bias = bias_mode & 2 ? bias.data() : nullptr;
      const MMArgs args(env, m, k, n, 1.0f,
                        bias_mode & 1 ? add.data() : nullptr, MMOptions(),
                        tuner, cfg);
      for (size_t split : {size_t{0}, size_t{64}, size_t{128}, size_t{71},
                            size_t{76}, size_t{576}}) {
        if (split >= k) continue;
        for (const IndexRange rm : {IndexRange(0, 1), IndexRange(1, m)}) {
          for (const IndexRange rn : {IndexRange(0, n), IndexRange(4, n),
                                      IndexRange(0, n - 4),
                                      IndexRange(4, n - 4)}) {
            for (size_t r = 0; r < m; ++r)
              for (size_t c = 0; c < n; ++c)
                expected.Row(r)[c] = actual.Row(r)[c] =
                    hwy::ConvertScalarTo<TC>(-123.5f);
            const auto run = [&](const MMI8B& weights, MatStorageT<TC>& out) {
              const StridedView<TC> cv(out.Row(rm.begin()) + rn.begin(),
                                       rn.Num(), out.Stride());
              const IndexRange first(0, split ? split : k);
              MMI8Kernel::B3A2C0(av, weights, rm, first, rn, args, MMSetC(), cv);
              if (split)
                MMI8Kernel::B3A2C0(av, weights, rm, IndexRange(split, k), rn,
                                   args, MMAddC(), cv);
            };
            run(ordinary, expected);
            run(interleaved, actual);
            for (size_t r = 0; r < m; ++r)
              ok &= memcmp(expected.Row(r), actual.Row(r), n * sizeof(TC)) == 0;
            ++cases;
          }
        }
      }
    }
  }
  if (!ok) ++g_failures;
  printf("%s packed N8 exact outputs, splits, N tails, M offsets, bias "
         "K=%zu TC=%s dense=%d (%zu cases)\n", ok ? "  ok" : "FAIL", k,
         TypeName<TC>(), dense, cases);
}

// Scalar integer dots independently verify both biased-weight corrections.
// Use the target's F32 multiply-add semantics, then round once per KC range.
template <typename TC>
void DualMicroReference(const MMI8AView& a, const MMI8B& b,
                        const IndexRange& rm, const IndexRange& rk,
                        const IndexRange& rn, bool add_previous,
                        const float* add, bool dual, MatStorageT<TC>& out) {
  const hn::CappedTag<float, 1> df;
  const auto madd = [&](float x, float y, float z) {
    return hn::GetLane(
        hn::MulAdd(hn::Set(df, x), hn::Set(df, y), hn::Set(df, z)));
  };
  for (size_t r : rm) {
    for (size_t n : rn) {
      float sum = 0;
      for (size_t c = rk.begin(); c < rk.end();) {
        const size_t g = c / b.block_size;
        const size_t count = HWY_MIN(static_cast<size_t>(rk.end()), (g + 1) * b.block_size) - c;
        for (size_t stream = 0; stream < (dual ? 2 : 1); ++stream) {
          const auto& av = stream ? *a.residual : a;
          int32_t dot = 0, encoded_dot = 0;
          const auto* q = reinterpret_cast<const MMI8BT*>(b.data->Row(n));
          for (size_t k = c; k < c + count; ++k) {
            dot += av.data.Row(r)[k] * (static_cast<int32_t>(q[k]) -
                                        (GEMMA_MM_I8_BIASED_B ? 128 : 0));
            encoded_dot += av.data.Row(r)[k] * static_cast<int32_t>(q[k]);
          }
          if constexpr (GEMMA_MM_I8_BIASED_B) {
            HWY_ASSERT(dot ==
                       encoded_dot -
                           128 * av.ViewGroup(r, c, count, g).RowSum(0, count));
          }
          sum = madd(
              static_cast<float>(dot),
              av.scale[g * av.scale_stride + r] * b.scale[g * b.Rows() + n],
              sum);
        }
        c += count;
      }
      if (add_previous)
        sum += hwy::ConvertScalarTo<float>(out.Row(r)[n]);
      else if (const float* bias = MMI8Bias(b, add, n))
        sum += *bias;
      out.Row(r)[n] = hwy::ConvertScalarTo<TC>(sum);
    }
  }
}

template <typename TA, typename TC>
void TestDualMicro(ThreadingContext& ctx, size_t k) {
  constexpr size_t m = 4, n = 24;
  MatMulEnv env(ctx);
  env.autotune = false;
  MatStorageT<float> af("af", Extents2D(m, k), ctx.allocator, MatPadding::kOdd);
  MatStorageT<float> b("b", Extents2D(n, k), ctx.allocator, MatPadding::kOdd);
  FillOperands(m, k, n, af, b, 0.7f);
  MatStorageT<TA> a("a", af.Extents(), ctx.allocator, MatPadding::kOdd);
  for (size_t r = 0; r < m; ++r)
    for (size_t c = 0; c < k; ++c)
      a.Row(r)[c] = hwy::ConvertScalarTo<TA>(af.Row(r)[c]);
  a.SetScale(0.75f);
  MatStorageT<int8_t> q("q", b.Extents(), ctx.allocator, MatPadding::kOdd);
  MatStorageT<int8_t> packed("p", b.Extents(), ctx.allocator,
                             MatPadding::kPacked);
  MatStorageT<TC> expected("expected", Extents2D(m, n), ctx.allocator,
                           MatPadding::kOdd);
  MatStorageT<TC> actual("actual", expected.Extents(), ctx.allocator,
                         MatPadding::kOdd);
  // Storage capacity deliberately exceeds the batch size: primary and residual
  // scale strides differ, and both must survive later M=1 calls.
  MMI8AStorage storage(m + 3, k, ctx.allocator),
      primary(m + 3, k, ctx.allocator);
  hwy::AlignedVector<float> scales(n * k / 32), add(n), bias(n), pre(k),
      rotated(k);
  for (size_t c = 0; c < k; ++c) pre[c] = 0.75f + 0.25f * (c % 3);
  for (size_t c = 0; c < n; ++c) {
    add[c] = static_cast<float>(c % 5) * 0.0625f;
    bias[c] = -static_cast<float>(c % 7) * 0.09375f;
  }
  const MMConfig cfg(m, k, n, 1, m, k, n, 1, 4, MMOrder::kNT, 1);
  MMAutoTune<MMConfig> tuner;
  tuner.SetCandidates({cfg}, false);
  bool ok = true;
  size_t cases = 0;
  for (size_t block : {size_t{32}, size_t{64}, size_t{128}}) {
    auto ordinary = PackB(b, q, scales.data(), ctx, pre.data(), block);
    for (size_t r = 0; r < n; ++r) hwy::CopyBytes(q.Row(r), packed.Row(r), k);
    MMI8PackMicroB(packed);
    auto weights = ordinary;
    weights.data = &packed;
    weights.packed_micro = true;
    weights.dual_a = true;
    MMI8AView residual;
    const auto av = QuantizeA(a, storage, ctx, 0, pre.data(), block, &residual);
    const auto original = QuantizeA(a, primary, ctx, 0, pre.data(), block);
    double first_error = 0, residual_error = 0;
    for (size_t r = 0; r < m; ++r) {
      ok &= memcmp(av.data.Row(r), original.data.Row(r), k) == 0;
      if constexpr (GEMMA_MM_I8_BIASED_B)
        ok &= memcmp(storage.prefix(r), primary.prefix(r),
                     (k + 1) * sizeof(int32_t)) == 0;
      MMI8PrepareInputRow(a.Row(r), k, pre.data(),
                          MMI8Flag("GEMMA_MM_I8_MATCH_BF16_A"), rotated.data());
      MMI8Rotate(rotated.data(), k);
      for (size_t c = 0; c < k; ++c) {
        const size_t g = c / block;
        const float s = av.scale[g * av.scale_stride + r];
        ok &= s == original.scale[g * original.scale_stride + r];
        const double exact = rotated[c] * a.Scale();
        const double first = av.data.Row(r)[c] * s;
        const double both =
            first + residual.data.Row(r)[c] *
                        residual.scale[g * residual.scale_stride + r];
        first_error += (exact - first) * (exact - first);
        residual_error += (exact - both) * (exact - both);
      }
    }
    ok &= residual_error < first_error * 0.001;
    for (bool dual : {false, true}) {
      weights.dual_a = dual;
      for (int bias_mode = 0; bias_mode < 4; ++bias_mode) {
        ordinary.bias = weights.bias = bias_mode & 2 ? bias.data() : nullptr;
        const float* add_row = bias_mode & 1 ? add.data() : nullptr;
        const MMArgs args(env, m, k, n, 1.0f, add_row, MMOptions(), tuner, cfg);
        for (size_t split :
             {size_t{0}, size_t{64}, size_t{71}, size_t{76}, size_t{576}}) {
          if (split >= k) continue;
          for (const IndexRange rm : {IndexRange(0, 1), IndexRange(1, m)}) {
            for (const IndexRange rn :
                 {IndexRange(0, n), IndexRange(4, n - 4)}) {
              for (size_t r = 0; r < m; ++r)
                for (size_t c = 0; c < n; ++c)
                  expected.Row(r)[c] = actual.Row(r)[c] =
                      hwy::ConvertScalarTo<TC>(-123.5f);
              const StridedView<TC> cv(actual.Row(rm.begin()) + rn.begin(),
                                       rn.Num(), actual.Stride());
              const IndexRange first(0, split ? split : k);
              DualMicroReference(av, ordinary, rm, first, rn, false, add_row,
                                 dual, expected);
              MMI8Kernel::B3A2C0(av, weights, rm, first, rn, args, MMSetC(),
                                 cv);
              if (split) {
                const IndexRange rest(split, k);
                DualMicroReference(av, ordinary, rm, rest, rn, true, add_row,
                                   dual, expected);
                MMI8Kernel::B3A2C0(av, weights, rm, rest, rn, args, MMAddC(),
                                   cv);
              }
              for (size_t r = 0; r < m; ++r)
                ok &=
                    memcmp(expected.Row(r), actual.Row(r), n * sizeof(TC)) == 0;
              ++cases;
            }
          }
        }
      }
    }
    weights.dual_a = true;
    for (size_t rows : {size_t{1}, m, size_t{1}}) {
      a.OverrideRows(rows);
      actual.OverrideRows(rows);
      MMI8AView rv;
      const auto qa = QuantizeA(a, primary, ctx, 0, pre.data(), block, &rv);
      const auto* key = MatMulI8(a, weights, add.data(), env, actual, storage);
      const auto ranges = key->autotune.Best()->RangesOfKC(k);
      for (size_t i = 0; i < ranges.NumTasks(); ++i)
        DualMicroReference(qa, ordinary, IndexRange(0, rows), ranges.Range(i),
                           IndexRange(0, n), i != 0, add.data(),
                           MMI8UseDualA(weights, rows), expected);
      for (size_t r = 0; r < rows; ++r)
        ok &= memcmp(expected.Row(r), actual.Row(r), n * sizeof(TC)) == 0;
    }
    a.OverrideRows(m);
    actual.OverrideRows(m);
  }
  if (!ok) ++g_failures;
  printf(
      "%s dual A scalar dots, A8 identity, prefixes, KC/N/M tails, bias, reuse "
      "K=%zu TA=%s TC=%s (%zu cases)\n",
      ok ? "  ok" : "FAIL", k, TypeName<TA>(), TypeName<TC>(), cases);
}

void TestDualFused(ThreadingContext& ctx) {
  constexpr size_t m = 3, k = 1152, n = 24, block = 128;
  MatMulEnv env(ctx);
  env.autotune = false;
  MatStorageT<float> af("af", Extents2D(m, k), ctx.allocator, MatPadding::kOdd);
  MatStorageT<float> b("b", Extents2D(n, k), ctx.allocator, MatPadding::kOdd);
  FillOperands(m, k, n, af, b, 0.5f);
  MatStorageT<BF16> a("a", af.Extents(), ctx.allocator, MatPadding::kOdd);
  for (size_t r = 0; r < m; ++r)
    for (size_t c = 0; c < k; ++c)
      a.Row(r)[c] = hwy::ConvertScalarTo<BF16>(af.Row(r)[c]);
  MatStorageT<int8_t> q("q", b.Extents(), ctx.allocator, MatPadding::kOdd);
  MatStorageT<int8_t> p("p", b.Extents(), ctx.allocator, MatPadding::kPacked);
  MatStorageT<BF16> c1("c1", Extents2D(m, n), ctx.allocator, MatPadding::kOdd);
  MatStorageT<BF16> c2("c2", c1.Extents(), ctx.allocator, MatPadding::kOdd);
  MatStorageT<BF16> expected("e", c1.Extents(), ctx.allocator,
                             MatPadding::kOdd);
  hwy::AlignedVector<float> scale(n * k / block), bias(n);
  for (size_t c = 0; c < n; ++c) bias[c] = 0.125f * static_cast<float>(c % 5);
  auto ordinary = PackB(b, q, scale.data(), ctx, nullptr, block);
  for (size_t r = 0; r < n; ++r) hwy::CopyBytes(q.Row(r), p.Row(r), k);
  MMI8PackMicroB(p);
  auto b1 = ordinary, b2 = ordinary;
  b1.data = b2.data = &p;
  b1.packed_micro = b2.packed_micro = true;
  b1.bias = bias.data();
  const auto copy_second = [&](RowPtrsBF, IndexRange rm, IndexRange rn,
                               StridedViewBF tile, size_t) {
    for (size_t r = 0; r < rm.Num(); ++r)
      hwy::CopyBytes(tile.Row(r), c2.Row(rm.begin() + r) + rn.begin(),
                     rn.Num() * sizeof(BF16));
  };
  MMOptions options;
  options.SetFunc(copy_second);
  MMI8AStorage storage(m, k, ctx.allocator), reference(m, k, ctx.allocator);
  MMI8AView residual;
  const auto av = QuantizeA(a, reference, ctx, 0, nullptr, block, &residual);
  bool ok = true;
  for (int flags = 0; flags < 4; ++flags) {
    b1.dual_a = flags & 1;
    b2.dual_a = flags & 2;
    const auto* key = TwoMatMulI8(a, b1, b2, env, c1, storage, options);
    const auto ranges = key->autotune.Best()->RangesOfKC(k);
    for (size_t branch = 0; branch < 2; ++branch) {
      const auto& weights = branch ? b2 : b1;
      ordinary.bias = weights.bias;
      for (size_t i = 0; i < ranges.NumTasks(); ++i)
        DualMicroReference(av, ordinary, IndexRange(0, m), ranges.Range(i),
                           IndexRange(0, n), i != 0, nullptr,
                           MMI8UseDualA(weights, m), expected);
      for (size_t r = 0; r < m; ++r)
        ok &= memcmp(expected.Row(r), (branch ? c2 : c1).Row(r),
                     n * sizeof(BF16)) == 0;
    }
  }
  if (!ok) ++g_failures;
  printf("%s dual A fused shared quantization and per-branch selection/bias\n",
         ok ? "  ok" : "FAIL");
}
void TestPackedHeadScheduling(ThreadingContext& ctx) {
  constexpr size_t k = 1152, n = 262144;
  MatPtrT<int8_t> head("head_shape", Extents2D(n, k));
  MatPtrT<int8_t> small("small_shape", Extents2D(24, k));
  MMI8B weights{&head, nullptr, nullptr, 128};
  weights.packed_micro = true;
  const bool enabled = MMI8Flag("GEMMA_MM_I8_PACKED_HEAD_FULL_K");
  bool ok = MMI8PreferFullHeadK(weights, 1, true) == enabled;
  ok &= !MMI8PreferFullHeadK(weights, 2, true);
  ok &= !MMI8PreferFullHeadK(weights, 1, false);
  weights.packed_micro = false;
  ok &= !MMI8PreferFullHeadK(weights, 1, true);
  weights.packed_micro = true;
  weights.data = &small;
  ok &= !MMI8PreferFullHeadK(weights, 1, true);

  MatMulEnv env(ctx);
  env.autotune = false;
  const auto generic = MMCandidates(ctx.cache_info, 1, k, n, 1, 4, false);
  const auto full = MMI8Candidates(env, 1, k, n, 1, 4, true);
  ok &= full.front().RangesOfKC(k).NumTasks() == 1;
  const auto defaults = MMI8Candidates(env, 1, k, n, 1, 4);
  if (!MMI8Flag("GEMMA_MM_I8_MIN_K_SPLITS"))
    ok &= defaults.front().KC() == generic.front().KC() &&
          defaults.front().Order() == generic.front().Order();
  env.autotune = true;
  const auto tunable = MMI8Candidates(env, 1, k, n, 1, 4, true);
  ok &= tunable.front().KC() == generic.front().KC() &&
        tunable.front().Order() == generic.front().Order();
  if (!ok) ++g_failures;
  printf("%s optional packed F32 M1 head scheduling and unchanged defaults\n",
         ok ? "  ok" : "FAIL");
}

void TestMicroscaleIsolation(ThreadingContext& ctx) {
  MatStorageT<float> a("a", Extents2D(1, 384), ctx.allocator, MatPadding::kOdd);
  for (size_t c = 0; c < 384; ++c)
    a.Row(0)[c] = c < 128 ? 10000.0f : c < 256 ? 0.001f : 0.0f;
  MMI8AStorage storage(1, 384, ctx.allocator);
  const auto q = QuantizeA(a, storage, ctx, 0, nullptr, 128);
  double recovered = 0.0;
  bool zero = q.scale[2 * q.scale_stride] == 1.0f;
  for (size_t c = 128; c < 256; ++c) {
    const double v = q.data.Row(0)[c] * q.scale[q.scale_stride];
    recovered += v * v;
  }
  for (size_t c = 256; c < 384; ++c) zero &= q.data.Row(0)[c] == 0;
  const bool ok = zero && std::abs(recovered / (128.0 * 1E-6) - 1.0) < 0.02;
  if (!ok) ++g_failures;
  printf("%s microscale isolates large outliers and zero blocks\n",
         ok ? "  ok" : "FAIL");
}

void TestF32MatchesBF16Inputs(ThreadingContext& ctx) {
  constexpr size_t k = 384;
  MatStorageT<float> input("f32_input", Extents2D(2, k), ctx.allocator,
                            MatPadding::kOdd);
  MatStorageT<BF16> rounded("bf16_input", input.Extents(), ctx.allocator,
                            MatPadding::kOdd);
  hwy::AlignedVector<float> copied(k), pre_scale(k);
  const hn::ScalableTag<BF16> dbf;
  bool ok = true;
  for (size_t c = 0; c < k; ++c)
    pre_scale[c] = 0.75f + static_cast<float>(c % 9) * 0.0625f;
  for (size_t r = 0; r < input.Rows(); ++r) {
    for (size_t c = 0; c < k; ++c)
      input.Row(r)[c] = static_cast<float>(static_cast<int>(c % 37) - 18) *
                         0.0712345f + static_cast<float>(r) * 0.012345f;
    DecompressAndZeroPad(dbf, MakeConst(MakeSpan(input.Row(r), k)), 0,
                         rounded.Row(r), k);
    MMI8PrepareInputRow(input.Row(r), k, pre_scale.data(), false, copied.data());
    for (size_t c = 0; c < k; ++c)
      ok &= copied[c] == input.Row(r)[c] * pre_scale[c];
    MMI8PrepareInputRow(input.Row(r), k, pre_scale.data(), true, copied.data());
    for (size_t c = 0; c < k; ++c)
      ok &= copied[c] == hwy::ConvertScalarTo<float>(rounded.Row(r)[c]) *
                           pre_scale[c];
  }
  if (MMI8Flag("GEMMA_MM_I8_MATCH_BF16_A")) {
    MMI8AStorage f32_storage(2, k, ctx.allocator);
    MMI8AStorage bf16_storage(2, k, ctx.allocator);
    for (size_t block : {size_t{0}, size_t{64}, size_t{128}}) {
      const auto a =
          QuantizeA(input, f32_storage, ctx, 0, pre_scale.data(), block);
      const auto b =
          QuantizeA(rounded, bf16_storage, ctx, 0, pre_scale.data(), block);
      const size_t groups = block ? k / block : 1;
      for (size_t r = 0; r < input.Rows(); ++r) {
        for (size_t c = 0; c < k; ++c)
          ok &= a.data.Row(r)[c] == b.data.Row(r)[c];
        for (size_t g = 0; g < groups; ++g)
          ok &= a.scale[g * a.scale_stride + r] ==
                b.scale[g * b.scale_stride + r];
        if constexpr (GEMMA_MM_I8_BIASED_B)
          for (size_t c = 0; c <= k; ++c)
            ok &= a.prefix[r * a.prefix_stride + c] ==
                  b.prefix[r * b.prefix_stride + c];
      }
    }
  }
  if (!ok) ++g_failures;
  printf("%s F32 activation rounding matches SFP BF16 preparation\n",
         ok ? "  ok" : "FAIL");
}

void TestQuantizedPrefixes() {
  if constexpr (!GEMMA_MM_I8_BIASED_B) return;
  hwy::AlignedVector<float> values(128);
  hwy::AlignedVector<MMI8AT> quantized(128);
  hwy::AlignedVector<int32_t> prefix(129);
  bool ok = true;
  for (size_t count : {size_t{7}, size_t{71}, size_t{128}}) {
    for (int32_t base : {-4096, 0, 377}) {
      for (size_t c = 0; c < count; ++c)
        values[c] = base == 0 ? 0.0f : (static_cast<int>(c % 17) - 8) * 0.13f;
      QuantizeRowA(values.data(), count, quantized.data(), prefix.data(),
                   quantized.size(), base);
      int32_t expected = base;
      ok &= prefix[0] == expected;
      for (size_t c = 0; c < count; ++c) {
        expected += quantized[c];
        ok &= prefix[c + 1] == expected;
      }
    }
  }
  if (!ok) ++g_failures;
  printf("%s quantized prefixes, incoming bases, zeros and tails\n",
         ok ? "  ok" : "FAIL");
}

void TestFixedTuningAndKeys() {
  const auto bf = MMKeys::KeyFromDims(4, 256, 128, 1);
  const auto i8 = MMKeys::KeyFromDims(4, 256, 128, 1, MMActivation::kI8);
  const auto block =
      MMKeys::KeyFromDims(4, 256, 128, 1, MMActivation::kI8Block);
  MMAutoTune<int> tuner;
  tuner.SetCandidates({7, 11, 19}, false);
  bool ok = bf != i8 && i8 != block && bf != block;
  for (int i = 0; i < 32; ++i) {
    ok &= tuner.Best() && *tuner.Best() == 7 && tuner.NextConfig() == 7;
    tuner.NotifyTicks(32 - i);
  }
  if (!ok) ++g_failures;
  printf("%s separate precision keys and fixed tuning\n", ok ? "  ok" : "FAIL");
}

void TestModelScaling(ThreadingContext& ctx) {
  auto& cache = MMI8WeightCache::Get();
  if (!cache.Enabled() || !MMI8Flag("GEMMA_MM_I8_L2_SCALE")) return;
  const size_t k = 256, n = 256;
  MatMulEnv env(ctx);
  MatStorageT<float> gate("test_gate", Extents2D(n, k), ctx.allocator,
                          MatPadding::kOdd);
  MatStorageT<float> up("test_up", Extents2D(n, k), ctx.allocator,
                        MatPadding::kOdd);
  MatStorageT<float> down("test_down", Extents2D(k, n), ctx.allocator,
                          MatPadding::kOdd);
  MatStorageT<float> norm("test_norm", Extents2D(1, k), ctx.allocator,
                          MatPadding::kOdd);
  Rng rng(315);
  for (size_t r = 0; r < n; ++r)
    for (size_t c = 0; c < k; ++c) {
      gate.Row(r)[c] = rng.Normal() * (0.01f + 0.01f * (r % 7));
      up.Row(r)[c] = rng.Normal() * 0.04f;
      down.Row(r)[c] = rng.Normal() * 0.03f;
    }
  for (size_t c = 0; c < k; ++c) norm.Row(0)[c] = 0.25f;
  const MatPtr& folded = cache.NormWeights(norm, {&gate, &up}, env);
  cache.PrepareFFN(gate, up, down, env);
  const auto* packed_down = cache.Lookup(down, env);
  const auto* packed_up = cache.Lookup(up, env);
  const size_t groups = packed_up->block_size ? k / packed_up->block_size : 1;
  bool ok =
      &folded != &norm && !packed_down->a_pre_scale && !packed_up->a_pre_scale;
  const MatPtrT<float> folded_t(folded);
  MatStorageT<float> compensated("compensated", up.Extents(), ctx.allocator,
                                 MatPadding::kOdd);
  for (size_t c = 0; c < k; ++c) {
    double sum_sq = 0.0;
    for (size_t r = 0; r < n; ++r)
      sum_sq += double(gate.Row(r)[c]) * gate.Row(r)[c] +
                double(up.Row(r)[c]) * up.Row(r)[c];
    const float scale = MMI8L2Scale(1.25, std::sqrt(sum_sq));
    ok &= std::abs((folded_t.Row(0)[c] + 1.0f) - 1.25f * scale) < 1E-6f;
    for (size_t r = 0; r < n; ++r) compensated.Row(r)[c] = up.Row(r)[c] / scale;
  }
  MatStorageT<int8_t> bytes("bytes", up.Extents(), ctx.allocator,
                            MatPadding::kOdd);
  hwy::AlignedVector<float> scales(n * groups);
  PackB(compensated, bytes, scales.data(), ctx, nullptr, packed_up->block_size);
  for (size_t r = 0; r < n; ++r) {
    double gs = 0.0, us = 0.0, ds = 0.0;
    for (size_t c = 0; c < k; ++c) {
      gs += double(gate.Row(r)[c]) * gate.Row(r)[c];
      us += double(up.Row(r)[c]) * up.Row(r)[c];
      ds += double(down.Row(c)[r]) * down.Row(c)[r];
    }
    const float expected =
        MMI8L2Scale(std::sqrt(gs) * std::sqrt(us), std::sqrt(ds));
    for (size_t g = 0; g < groups; ++g)
      ok &= std::abs(packed_up->scale[g * n + r] / scales[g * n + r] -
                     expected) < 2E-5f;
  }
  // A partially ineligible consumer set must leave RMSNorm untouched.
  MatStorageT<float> bad("bad", Extents2D(3, k), ctx.allocator,
                         MatPadding::kOdd);
  ok &= &cache.NormWeights(norm, {&gate, &bad}, env) == &norm;
  const size_t odd_k = 3 * MMI8RotateBlockSize();
  MatStorageT<float> odd_group("test_gate_odd", Extents2D(8, odd_k),
                               ctx.allocator, MatPadding::kOdd);
  const size_t chosen = cache.QuantBlockSize(odd_group);
  const size_t requested = MMI8QuantBlockSize();
  ok &= chosen == (requested && odd_k % requested != 0
                       ? MMI8RotateBlockSize()
                       : requested);
  for (size_t r = 0; r < odd_group.Rows(); ++r)
    for (size_t c = 0; c < odd_k; ++c)
      odd_group.Row(r)[c] = static_cast<float>(static_cast<int>(c % 7) - 3);
  const auto* packed_odd = cache.Lookup(odd_group, env);
  ok &= packed_odd && packed_odd->block_size == chosen;
  ok &= env.weight_prepare_seconds > 0.0;
  if (!ok) ++g_failures;
  printf(
      "%s RMSNorm compensation, two-branch FFN scales, folding and fallback\n",
      ok ? "  ok" : "FAIL");
}

void TestAll() {
  TestFixedTuningAndKeys();
  ThreadingArgs threading_args;
  ThreadingContext ctx(threading_args);
  MatMulEnv env(ctx);
  env.autotune = !MMI8Flag("GEMMA_MM_I8_TEST_FIXED");
  printf("target=%s biasedB=%d block=%zu hash=%zu vector bytes=%zu\n",
         hwy::TargetName(HWY_TARGET), GEMMA_MM_I8_BIASED_B,
         MMI8RotateBlockSize(), MMI8HashBits(),
         hn::Lanes(hn::ScalableTag<uint8_t>()));
  for (size_t block : {size_t{64}, size_t{128}}) {
    for (size_t hash : {size_t{16}, size_t{32}}) {
      TestRotationMatchesScalar(block, hash);
      TestRotationPreservesDotProducts(block, hash);
    }
  }
  TestHash16();
  TestL2Scaling();

  TestMicroscaleIsolation(ctx);
  TestPackedHeadScheduling(ctx);
  TestDualMicro<float, float>(ctx, 1152);
  TestDualMicro<BF16, BF16>(ctx, 1152);
  TestDualFused(ctx);
  for (size_t k : {size_t{384}, size_t{1152}}) {
    TestPackedMicro<float>(ctx, k);
    TestPackedMicro<BF16>(ctx, k);
  }
  TestPackedMicro<float>(ctx, 1152, true);
  TestPackedMicro<BF16>(ctx, 1152, true);
  TestF32MatchesBF16Inputs(ctx);
  TestQuantizedPrefixes();

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

  // Local scales: signed/biased correction, output bias, partial M, and KC
  // boundaries that need not coincide with a quantization group boundary.
  for (size_t block : {size_t{32}, size_t{64}, size_t{128}}) {
    TestCase<float>(5, 1152, 12, true, ctx, env, a_i8, 3.0f, nullptr, block);
    TestCase<float>(4, 20096, 8, true, ctx, env, a_i8, 3.0f, nullptr, block);
    TestCase<BF16>(5, 1152, 12, true, ctx, env, a_i8, 3.0f, nullptr, block);
    TestCase<BF16>(5, 20096, 8, true, ctx, env, a_i8, 3.0f, nullptr, block);
  }
  TestModelScaling(ctx);

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
void RunTests() { HWY_DYNAMIC_DISPATCH(TestAll)(); }
}  // namespace gcpp

int main(int /*argc*/, char** /*argv*/) {
  gcpp::RunTests();
  const size_t failures = gcpp::g_failures;
  printf("%s (%zu failures)\n", failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}
#endif  // HWY_ONCE
