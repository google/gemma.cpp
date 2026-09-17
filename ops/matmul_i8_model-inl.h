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
//  - the cache is a process-wide singleton and never freed.
//
// Enabled by environment variables, so no CLI plumbing is needed:
//   GEMMA_MM_I8=1            route eligible MatMuls through the int8 kernel
//   GEMMA_MM_I8_MIN_K=<n>    leave tensors with K < n in their original format
//   GEMMA_MM_I8_SKIP_ROWS=<n> leave tensors with N >= n alone (e.g. the vocab-
//                            sized logits projection, the usual first thing to
//                            exclude from W8A8)
//   GEMMA_MM_I8_INCLUDE=<list> only quantize tensor names containing one of the
//                            comma-separated substrings
//   GEMMA_MM_I8_EXCLUDE=<list> leave matching tensor names in their old format
//   GEMMA_MM_I8_VERBOSE=1    log each tensor as it is quantized
//   GEMMA_MM_I8_BLOCK_SIZE=64 select 64-wide instead of 128-wide rotation
//   GEMMA_MM_I8_HASH_BITS=16  select the cheaper 16-bit sign hash
//   GEMMA_MM_I8_L2_SCALE=1    equalize FFN hidden and RMSNorm input channels
//   GEMMA_MM_I8_MICROSCALE=1  use per-rotation-block A/B quantization scales
//   GEMMA_MM_I8_SCALE_FFN=0  disable hidden-channel scaling for ablation
//   GEMMA_MM_I8_SCALE_NORM=0 disable RMSNorm folding for ablation

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>  // NOLINT
#include <unordered_map>
#include <vector>

#include "hwy/base.h"
#include "ops/matmul.h"
#include "util/mat.h"
#include "util/threading_context.h"

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

// True if `name` contains any non-empty comma-separated token in `list`.
static inline bool MMI8NameMatches(const char* name, const char* list) {
  if (list == nullptr || *list == '\0') return false;
  for (const char* begin = list; *begin != '\0';) {
    const char* end = strchr(begin, ',');
    if (end == nullptr) end = begin + strlen(begin);
    const size_t len = static_cast<size_t>(end - begin);
    if (len != 0) {
      for (const char* at = name; *at != '\0'; ++at) {
        if (strncmp(at, begin, len) == 0) return true;
      }
    }
    begin = *end == '\0' ? end : end + 1;
  }
  return false;
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

  bool ScalingEnabled() const { return enabled_ && l2_scaling_; }

  bool Eligible(const MatPtr& B) const {
    if (!enabled_ || !B.HasPtr() || B.Cols() < min_k_ ||
        B.Rows() >= skip_rows_ || B.Rows() % kNR != 0 ||
        B.Cols() % MMI8RotateBlockSize() != 0)
      return false;
    if (include_ && *include_ && !MMI8NameMatches(B.Name(), include_))
      return false;
    return !MMI8NameMatches(B.Name(), exclude_);
  }

  // Explicit pairing avoids stale "previous tensor" state, including when
  // routing filters skip a projection or an unfused FFN is used.
  void PrepareFFN(const MatPtr& gate, const MatPtr& up, const MatPtr& down,
                  MatMulEnv& env) {
    if (!l2_scaling_ || !MMI8Flag("GEMMA_MM_I8_SCALE_FFN", true) ||
        !Eligible(gate) || !Eligible(up) || !Eligible(down))
      return;
    std::lock_guard<std::mutex> lock(mutex_);
    const void* key = down.RowBytes(0);
    if (map_.count(key)) return;
    // Never change the representation of an already packed up projection.
    if (map_.count(up.RowBytes(0))) return;
    if (gate.Rows() != up.Rows() || up.Rows() != down.Cols()) return;
    PrepareTimer timer(env);
    std::vector<double> gate_norms(gate.Rows()), up_norms(up.Rows());
    CallUpcasted(&gate, [&](const auto* typed) {
      ComputeRowNorms(*typed, env.ctx, gate_norms);
    });
    CallUpcasted(&up, [&](const auto* typed) {
      ComputeRowNorms(*typed, env.ctx, up_norms);
    });
    // Data-free proxy for the magnitude of act(W1*x) * (W2*x).
    // This is a heuristic, not an exact GELU/SILU activation statistic.
    for (size_t i = 0; i < up_norms.size(); ++i) up_norms[i] *= gate_norms[i];
    auto entry = std::make_unique<Entry>(down, env.ctx.allocator);
    CallUpcasted(&down, [&](const auto* typed) {
      ConfigureL2Scale(*typed, up_norms, up.RowBytes(0), *entry, env.ctx);
      Quantize(*typed, *entry);
    });
    // Scale the linear branch's output in its existing dequantization multiply.
    // GELU(W1*x) * (s*W2*x), followed by Wdown/s, preserves the real-valued
    // FFN.
    output_scales_[up.RowBytes(0)] = entry->input_scale;
    entry->b.a_pre_scale = nullptr;
    map_[key] = std::move(entry);
  }

  // Fuse input equalization into RMSNorm gamma and compensate every consumer.
  // Check all consumers before changing anything: fallback must stay exact.
  const MatPtr& NormWeights(const MatPtr& norm,
                            const std::vector<const MatPtr*>& consumers,
                            MatMulEnv& env) {
    if (!l2_scaling_ || !MMI8Flag("GEMMA_MM_I8_SCALE_NORM", true) ||
        !norm.HasPtr() || norm.Scale() != 1.0f || consumers.empty())
      return norm;
    for (const MatPtr* b : consumers)
      if (!Eligible(*b) || b->Cols() != norm.Cols()) return norm;
    std::lock_guard<std::mutex> lock(mutex_);
    const void* key = norm.RowBytes(0);
    auto found = norms_.find(key);
    if (found != norms_.end()) return *found->second;
    for (const MatPtr* b : consumers)
      if (map_.count(b->RowBytes(0))) return norm;
    PrepareTimer timer(env);
    const size_t k = norm.Cols();
    const hn::ScalableTag<float> df;
    const size_t padded = hwy::RoundUpTo(k, hn::Lanes(df));
    hwy::AlignedVector<float> gamma(padded), row(padded);
    CallUpcasted(&norm, [&](const auto* typed) {
      DecompressAndZeroPad(df, typed->PaddedSpan(), 0, gamma.data(), k);
    });
    std::vector<double> column_sq(k, 0.0);
    for (const MatPtr* b : consumers)
      CallUpcasted(b, [&](const auto* typed) {
        for (size_t r = 0; r < b->Rows(); ++r) {
          DecompressAndZeroPad(df, typed->PaddedSpan(), r * b->Stride(),
                               row.data(), k);
          for (size_t c = 0; c < k; ++c) {
            const double v = row[c] * static_cast<double>(b->Scale());
            column_sq[c] += v * v;
          }
        }
      });
    auto folded = std::make_unique<MatStorageT<float>>(
        "i8_norm", norm.Extents(), env.ctx.allocator, MatPadding::kOdd);
    hwy::AlignedVector<float> scales(k);
    for (size_t c = 0; c < k; ++c) {
      const float g =
          1.0f + gamma[c];  // Gemma's stored gamma is offset by one.
      scales[c] = MMI8L2Scale(std::abs(g), std::sqrt(column_sq[c]));
      folded->Row(0)[c] = g * scales[c] - 1.0f;
    }

    for (const MatPtr* b : consumers) input_scales_[b->RowBytes(0)] = scales;
    if (verbose_)
      fprintf(stderr, "MM.I8: RMSNorm fold %s, %zu consumers\n", norm.Name(),
              consumers.size());
    const MatPtr& result = *folded;
    norms_[key] = std::move(folded);
    return result;
  }

  template <typename TB>
  const MMI8B* Lookup(const MatPtrT<TB>& B, MatMulEnv& env) {
    if (!Eligible(B)) return nullptr;

    const void* key = B.RowBytes(0);
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = map_.find(key);
    if (it != map_.end()) return &it->second->b;
    PrepareTimer timer(env);
    auto entry = std::make_unique<Entry>(B, env.ctx.allocator);
    const auto input = input_scales_.find(key);
    if (input != input_scales_.end()) entry->input_scale = input->second;
    Quantize(B, *entry);
    const auto output = output_scales_.find(key);
    if (output != output_scales_.end()) {
      const size_t groups =
          entry->b.block_size ? B.Cols() / entry->b.block_size : 1;
      for (size_t g = 0; g < groups; ++g)
        for (size_t r = 0; r < B.Rows(); ++r)
          entry->scale[g * B.Rows() + r] *= output->second[r];
    }
    if (verbose_)
      fprintf(stderr, "MM.I8: quantized %-16s %6zu x %6zu block=%zu\n",
              B.Name(), B.Rows(), B.Cols(), entry->b.block_size);
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
  class PrepareTimer {
   public:
    explicit PrepareTimer(MatMulEnv& env)
        : env_(env), start_(std::chrono::steady_clock::now()) {}
    ~PrepareTimer() {
      env_.weight_prepare_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        start_)
              .count();
    }

   private:
    MatMulEnv& env_;
    std::chrono::steady_clock::time_point start_;
  };

  struct Entry {
    Entry(const MatPtr& B, const Allocator& allocator)
        : data("B_i8", Extents2D(B.Rows(), B.Cols()), allocator,
               MatPadding::kOdd),
          scale(B.Rows() *
                (MMI8QuantBlockSize() ? B.Cols() / MMI8QuantBlockSize() : 1)) {
      b = MMI8B{&data, scale.data(), nullptr, MMI8QuantBlockSize()};
    }
    MatStorageT<int8_t> data;
    hwy::AlignedVector<float> scale;
    hwy::AlignedVector<float> input_scale;
    const void* scale_source = nullptr;
    MMI8B b;
  };

  MMI8WeightCache()
      : enabled_(MMI8EnvSize("GEMMA_MM_I8", 0) != 0),
        verbose_(MMI8EnvSize("GEMMA_MM_I8_VERBOSE", 0) != 0),
        l2_scaling_(MMI8EnvSize("GEMMA_MM_I8_L2_SCALE", 0) != 0),
        min_k_(MMI8EnvSize("GEMMA_MM_I8_MIN_K", 0)),
        skip_rows_(MMI8EnvSize("GEMMA_MM_I8_SKIP_ROWS", ~size_t{0})),
        include_(getenv("GEMMA_MM_I8_INCLUDE")),
        exclude_(getenv("GEMMA_MM_I8_EXCLUDE")) {}

  template <typename TB>
  void ComputeRowNorms(const MatPtrT<TB>& B, ThreadingContext& ctx,
                       std::vector<double>& norms) {
    const hn::ScalableTag<float> df;
    const size_t K = B.Cols();
    const size_t padded_k = hwy::RoundUpTo(K, hn::Lanes(df));
    hwy::AlignedVector<float> row(padded_k + hn::Lanes(df));
    const PackedSpan<const TB> span = B.PaddedSpan();
    const double tensor_scale = hwy::ScalarAbs(B.Scale());
    for (size_t r = 0; r < B.Rows(); ++r) {
      DecompressAndZeroPad(df, span, r * B.Stride(), row.data(), K);
      double sum_sq = 0.0;
      for (size_t c = 0; c < K; ++c) {
        sum_sq += static_cast<double>(row[c]) * row[c];
      }
      norms[r] = tensor_scale * std::sqrt(sum_sq);
    }
    (void)ctx;
  }

  template <typename TB>
  void ConfigureL2Scale(const MatPtrT<TB>& down,
                        const std::vector<double>& up_norms, const void* source,
                        Entry& entry, ThreadingContext& ctx) {
    const hn::ScalableTag<float> df;
    const size_t K = down.Cols();
    const size_t padded_k = hwy::RoundUpTo(K, hn::Lanes(df));
    hwy::AlignedVector<float> row(padded_k + hn::Lanes(df));
    const PackedSpan<const TB> span = down.PaddedSpan();
    std::vector<double> sum_sq(K, 0.0);
    double before_max = 0.0;
    const double tensor_scale = hwy::ScalarAbs(down.Scale());
    for (size_t r = 0; r < down.Rows(); ++r) {
      DecompressAndZeroPad(df, span, r * down.Stride(), row.data(), K);
      for (size_t c = 0; c < K; ++c) {
        const double value = tensor_scale * row[c];
        sum_sq[c] += value * value;
        before_max = HWY_MAX(before_max, hwy::ScalarAbs(value));
      }
    }

    entry.input_scale.resize(K);
    size_t clamped = 0;
    for (size_t c = 0; c < K; ++c) {
      bool was_clamped = false;
      entry.input_scale[c] =
          MMI8L2Scale(up_norms[c], std::sqrt(sum_sq[c]), &was_clamped);
      clamped += was_clamped ? 1 : 0;
    }
    entry.scale_source = source;
    entry.b.a_pre_scale = entry.input_scale.data();

    if (verbose_) {
      std::vector<float> sorted(entry.input_scale.begin(),
                                entry.input_scale.end());
      std::sort(sorted.begin(), sorted.end());
      double after_max = 0.0;
      for (size_t r = 0; r < down.Rows(); ++r) {
        DecompressAndZeroPad(df, span, r * down.Stride(), row.data(), K);
        for (size_t c = 0; c < K; ++c) {
          after_max = HWY_MAX(after_max, hwy::ScalarAbs(tensor_scale * row[c] /
                                                        entry.input_scale[c]));
        }
      }
      const size_t p95 = (95 * (K - 1)) / 100;
      fprintf(stderr,
              "MM.I8: L2 %-16s scale[min/med/p95/max]="
              "%.5g/%.5g/%.5g/%.5g clamped=%zu weight|max|=%.5g->%.5g\n",
              down.Name(), sorted.front(), sorted[K / 2], sorted[p95],
              sorted.back(), clamped, before_max, after_max);
    }
    (void)ctx;
  }
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
      if (!entry.input_scale.empty()) {
        for (size_t c = 0; c < K; ++c) row[c] /= entry.input_scale[c];
      }
      MMI8Rotate(row.data(), K);
      MMI8BT* HWY_RESTRICT out = HWY_RCAST_ALIGNED(MMI8BT*, entry.data.Row(r));
      const size_t group_size = entry.b.block_size ? entry.b.block_size : K;
      for (size_t c = 0; c < K; c += group_size) {
        entry.scale[(c / group_size) * B.Rows() + r] =
            b_scale * PackBRow(row.data() + c, group_size, out + c, group_size);
      }
      for (size_t c = K; c < entry.data.Stride(); ++c) out[c] = 0;
    }
  }

  bool enabled_;
  bool verbose_;
  bool l2_scaling_;
  size_t min_k_;
  size_t skip_rows_;
  const char* include_;
  const char* exclude_;

  std::mutex mutex_;
  std::unordered_map<const void*, std::unique_ptr<Entry>> map_;

  std::unordered_map<const void*, hwy::AlignedVector<float>> input_scales_;
  std::unordered_map<const void*, hwy::AlignedVector<float>> output_scales_;
  std::unordered_map<const void*, std::unique_ptr<MatStorageT<float>>> norms_;
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
        const MMI8B* i8_1 = cache.Lookup(*B1_t, env);
        if (i8_1 == nullptr) return nullptr;
        const MMI8B* i8_2 = cache.Lookup(*B2_t, env);
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
  const MMI8B* B_i8 = cache.Lookup(B, env);
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
