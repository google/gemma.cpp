// Copyright 2025 Google LLC
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

// Experiment harness that routes the model's MatMuls through the W8A8 kernel
// in `ops/matmul_i8-inl.h`, to measure end-to-end quality. Weights are
// quantized lazily on first use and cached, keyed by their data pointer, so
// this needs no changes to the loading path.
//
// NOT a production integration:
//  - quantizing from whatever the file holds (e.g. SFP) stacks a second
//    quantization on top of the first; a real path would quantize the
//    original checkpoint;
//  - the int8 and BF16 kernels share `MatMulEnv`'s autotune keys, which are
//    shape-only, so a model that mixes them will mis-tune;
//  - the cache is a process-wide singleton and never freed.
//
// Enabled by environment variables, so no CLI plumbing is needed:
//   GEMMA_MM_I8=1            route eligible MatMuls through the int8 kernel
//   GEMMA_MM_I8_MIN_K=<n>    leave tensors with K < n in their original format
//   GEMMA_MM_I8_SKIP_ROWS=<n> leave tensors with N >= n alone (e.g. the vocab-
//                            sized logits projection, the usual first thing to
//                            exclude from W8A8)
//   GEMMA_MM_I8_VERBOSE=1    log each tensor as it is quantized

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <memory>
#include <mutex>  // NOLINT
#include <unordered_map>

#include "ops/matmul.h"
#include "util/mat.h"
#include "util/threading_context.h"
#include "hwy/base.h"

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MODEL_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MODEL_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MODEL_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MODEL_TOGGLE
#endif

#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"
#include "ops/matmul_i8-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Reads an integer environment variable, or returns `fallback`.
static inline size_t MMI8EnvSize(const char* name, size_t fallback) {
  const char* s = getenv(name);
  if (s == nullptr || *s == '\0') return fallback;
  const long long v = atoll(s);  // NOLINT
  return v < 0 ? fallback : static_cast<size_t>(v);
}

// Quantizes one row of `k` floats to symmetric int8, biased by 128 if
// `GEMMA_MM_I8_BIASED_B`. Returns the dequantization scale.
static HWY_INLINE float PackBRow(const float* HWY_RESTRICT in, size_t k,
                                 MMI8BT* HWY_RESTRICT out, size_t padded_k) {
  const hn::ScalableTag<float> df;
  const hn::Rebind<int32_t, decltype(df)> di32;
  const hn::Rebind<int8_t, decltype(df)> di8;
  using VF = hn::Vec<decltype(df)>;
  const size_t NF = hn::Lanes(df);

  VF vmax = hn::Zero(df);
  size_t i = 0;
  if (k >= NF) {
    for (; i <= k - NF; i += NF) {
      vmax = hn::Max(vmax, hn::Abs(hn::LoadU(df, in + i)));
    }
  }
  if (i != k) vmax = hn::Max(vmax, hn::Abs(hn::LoadN(df, in + i, k - i)));
  const float amax = hn::ReduceMax(df, vmax);

  const float scale = (amax == 0.0f) ? 1.0f : amax / kMMI8Max;
  const float inv = (amax == 0.0f) ? 0.0f : kMMI8Max / amax;
  const VF vinv = hn::Set(df, inv);
  // Store as int8 and add the bias afterwards: `DemoteTo` to u8 would saturate
  // negative values to zero.
  const auto vbias = hn::Set(di32, GEMMA_MM_I8_BIASED_B ? 128 : 0);

  i = 0;
  if (k >= NF) {
    for (; i <= k - NF; i += NF) {
      const auto q = hn::NearestInt(hn::Mul(hn::LoadU(df, in + i), vinv));
      // Bias in the int32 domain, then narrow; `DemoteTo` saturates, and
      // `q + 128` is within [1, 255] so nothing is clamped.
      if constexpr (GEMMA_MM_I8_BIASED_B) {
        const hn::Rebind<uint8_t, decltype(df)> du8;
        hn::StoreU(hn::DemoteTo(du8, hn::Add(q, vbias)), du8,
                   HWY_RCAST_ALIGNED(uint8_t*, out) + i);
      } else {
        hn::StoreU(hn::DemoteTo(di8, q), di8,
                   HWY_RCAST_ALIGNED(int8_t*, out) + i);
      }
    }
  }
  for (; i < k; ++i) {
    const int32_t q = static_cast<int32_t>(std::lroundf(in[i] * inv));
    out[i] = static_cast<MMI8BT>(q + (GEMMA_MM_I8_BIASED_B ? 128 : 0));
  }
  for (; i < padded_k; ++i) out[i] = static_cast<MMI8BT>(0);
  return scale;
}

// Process-wide cache of int8 weights, keyed by the tensor's data pointer.
class MMI8WeightCache {
 public:
  static MMI8WeightCache& Get() {
    static MMI8WeightCache cache;
    return cache;
  }

  bool Enabled() const { return enabled_; }

  // Returns the int8 form of `B`, quantizing and caching on first use, or
  // nullptr if this tensor is not eligible (see the environment variables).
  template <typename TB>
  const MMI8B* Lookup(const MatPtrT<TB>& B, ThreadingContext& ctx) {
    const size_t N = B.Rows();
    const size_t K = B.Cols();
    if (K < min_k_ || N >= skip_rows_ || (N % kNR) != 0) return nullptr;

    const void* key = B.RowBytes(0);
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = map_.find(key);
    if (it != map_.end()) return it->second ? &it->second->b : nullptr;

    auto entry = std::make_unique<Entry>(B, ctx.allocator);
    Quantize(B, *entry);
    if (verbose_) {
      fprintf(stderr, "MM.I8: quantized %-16s %6zu x %6zu\n", B.Name(), N, K);
    }
    const MMI8B* result = &entry->b;
    map_[key] = std::move(entry);
    return result;
  }

  // Storage for the quantized `A`, grown on demand. `MatMul` for a given
  // `MatMulEnv` is not called concurrently, and this experiment runs a single
  // cluster, so one instance suffices.
  MMI8AStorage& AStorage(size_t M, size_t K, const Allocator& allocator) {
    if (a_ == nullptr || M > a_max_M_ || K > a_max_K_) {
      a_max_M_ = HWY_MAX(M, a_max_M_);
      a_max_K_ = HWY_MAX(K, a_max_K_);
      a_ = std::make_unique<MMI8AStorage>(a_max_M_, a_max_K_, allocator);
    }
    return *a_;
  }

 private:
  struct Entry {
    Entry(const MatPtr& B, const Allocator& allocator)
        : data("B_i8", Extents2D(B.Rows(), B.Cols()), allocator,
               MatPadding::kOdd),
          scale(B.Rows()) {
      b = MMI8B{&data, scale.data()};
    }
    MatStorageT<int8_t> data;
    hwy::AlignedVector<float> scale;
    MMI8B b;
  };

  MMI8WeightCache()
      : enabled_(MMI8EnvSize("GEMMA_MM_I8", 0) != 0),
        verbose_(MMI8EnvSize("GEMMA_MM_I8_VERBOSE", 0) != 0),
        min_k_(MMI8EnvSize("GEMMA_MM_I8_MIN_K", 0)),
        skip_rows_(MMI8EnvSize("GEMMA_MM_I8_SKIP_ROWS", ~size_t{0})) {}

  // Serial (the caller may already be inside a parallel region), but
  // vectorized, so a 2B-parameter model takes a few seconds in total.
  template <typename TB>
  void Quantize(const MatPtrT<TB>& B, Entry& entry) {
    const hn::ScalableTag<float> df;
    const size_t K = B.Cols();
    const size_t padded_k = hwy::RoundUpTo(K, hn::Lanes(df));
    hwy::AlignedVector<float> row(padded_k + hn::Lanes(df));
    const PackedSpan<const TB> span = B.PaddedSpan();
    const float b_scale = B.Scale();

    for (size_t r = 0; r < B.Rows(); ++r) {
      DecompressAndZeroPad(df, span, r * B.Stride(), row.data(), K);
      MMI8BT* HWY_RESTRICT out =
          HWY_RCAST_ALIGNED(MMI8BT*, entry.data.Row(r));
      entry.scale[r] =
          b_scale * PackBRow(row.data(), K, out, entry.data.Stride());
    }
  }

  bool enabled_;
  bool verbose_;
  size_t min_k_;
  size_t skip_rows_;

  std::mutex mutex_;
  std::unordered_map<const void*, std::unique_ptr<Entry>> map_;

  std::unique_ptr<MMI8AStorage> a_;
  size_t a_max_M_ = 0;
  size_t a_max_K_ = 0;
};

// As `MaybeMatMulI8`, for the fused gated-FFN pair. Both operands must be
// eligible, else we fall back so that the pair stays consistent.
static inline MMPerKey* MaybeTwoMatMulI8(const MatPtrT<BF16>& A,
                                         const MatPtr& B1, const MatPtr& B2,
                                         MatMulEnv& env, MatPtrT<BF16>& C,
                                         const MMOptions& options) {
  MMI8WeightCache& cache = MMI8WeightCache::Get();
  if (!cache.Enabled()) return nullptr;
  return CallUpcastedSame(
      &B1, &B2, [&](const auto* B1_t, const auto* B2_t) -> MMPerKey* {
        const MMI8B* i8_1 = cache.Lookup(*B1_t, env.ctx);
        if (i8_1 == nullptr) return nullptr;
        const MMI8B* i8_2 = cache.Lookup(*B2_t, env.ctx);
        if (i8_2 == nullptr) return nullptr;
        MMI8AStorage& a_storage =
            cache.AStorage(A.Rows(), A.Cols(), env.ctx.allocator);
        return TwoMatMulI8(A, *i8_1, *i8_2, env, C, a_storage, options);
      });
}

// If the int8 path is enabled and `B` is eligible, computes `C = A * B + add`
// with the W8A8 kernel and returns its autotune state; else returns nullptr so
// the caller falls back to `MatMulStatic`.
template <typename TA, typename TB, typename TC>
MMPerKey* MaybeMatMulI8(const MatPtrT<TA>& A, const MatPtrT<TB>& B,
                        const float* HWY_RESTRICT add, MatMulEnv& env,
                        MatPtrT<TC>& C, const MMOptions& options) {
  MMI8WeightCache& cache = MMI8WeightCache::Get();
  if (!cache.Enabled()) return nullptr;
  // `TwoMatMul`'s fused second output is not wired up here.
  if (options.func != nullptr) return nullptr;
  const MMI8B* B_i8 = cache.Lookup(B, env.ctx);
  if (B_i8 == nullptr) return nullptr;

  MMI8AStorage& a_storage =
      cache.AStorage(A.Rows(), A.Cols(), env.ctx.allocator);
  return MatMulI8(A, *B_i8, add, env, C, a_storage, options);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
