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
//   GEMMA_MM_I8_MICROSCALE=1  use local A/B quantization scales
//   GEMMA_MM_I8_FAST_MICRO=0  use the reference microscaling kernel
//   GEMMA_MM_I8_MIN_K_SPLITS=1 prefer fewer KC ranges when autotuning is off
//   GEMMA_MM_I8_QUANT_BLOCK_SIZE=<n> quantize groups of 32, 64 or 128 values;
//                            fall back to rotation width if K is not divisible
//   GEMMA_MM_I8_SCALE_FFN=0  disable hidden-channel scaling for ablation
//   GEMMA_MM_I8_SCALE_NORM=0 disable RMSNorm folding for ablation
//   GEMMA_MM_I8_SCALE_NORM_INCLUDE=<list> fold only matching norm tensor names
//   GEMMA_MM_I8_SCALE_CALIBRATION_DIR=<dir> measured input RMS for folding
//   GEMMA_MM_I8_SCALE_CALIBRATION_ALPHA=<n> balancing exponent (default 0.5)
//   GEMMA_MM_I8_SCALE_CALIBRATION_POW2=0 disable power-of-two scale rounding
//   GEMMA_MM_I8_CALIBRATION_CAPTURE=<dir> capture rotated SFP activation rows
//   GEMMA_MM_I8_CALIBRATION_SAMPLES=<n> captured rows per tensor (default 2048)
//   GEMMA_MM_I8_CALIBRATION_ROWS_PER_CALL=<n> sample each call (default 32)
//   GEMMA_MM_I8_CALIBRATION_DIR=<dir> refine packed weights using H/G files
//   GEMMA_MM_I8_CALIBRATION_SWEEPS=<n> 0 = scales only, default 1, maximum 8
//   GEMMA_MM_I8_CALIBRATION_INCLUDE/EXCLUDE=<list> select calibration tensors
//   GEMMA_MM_I8_BIAS_CORRECTION=1 use .mean files for output bias correction
//   GEMMA_MM_I8_PACKED_HEAD=1 pack large microscale heads for the N8 kernel
//   GEMMA_MM_I8_PACKED_HEAD_FULL_K=1 prefer one KC for packed F32 M1 heads
//   GEMMA_MM_I8_DUAL_A_DOWN=1 pack FFN-down weights and quantize A residuals
//   GEMMA_MM_I8_DUAL_A_HEAD=1 pack c_embedding logits and quantize A residuals
//   GEMMA_MM_I8_DUAL_A_BODY=1 use A residuals for eligible non-embedding weights
//   GEMMA_MM_I8_DUAL_A_M1_ONLY=1 limit A residual correction to decode
//   GEMMA_MM_I8_EXPORT_DIR=<dir> export rotated unscaled F32 weight rows
//   GEMMA_MM_I8_IMPORT_DIR=<dir> import externally calibrated q/scales

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
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
#include "ops/matmul_i8_calibration-inl.h"

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

  // Keep every eligible tensor in W8A8 even if its K cannot use the requested
  // quantization group. Fused FFN inputs have the same K and choose alike.
  size_t QuantBlockSize(const MatPtr& B) const {
    const size_t requested = MMI8QuantBlockSize();
    if (requested == 0) return 0;
    return B.Cols() % requested == 0 ? requested : MMI8RotateBlockSize();
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
    std::vector<double> up_norms;
    const bool calibrated = LoadInputRMS(down, up_norms);
    if (!calibrated) {
      std::vector<double> gate_norms(gate.Rows());
      up_norms.resize(up.Rows());
      CallUpcasted(&gate, [&](const auto* typed) {
        ComputeRowNorms(*typed, env.ctx, gate_norms);
      });
      CallUpcasted(&up, [&](const auto* typed) {
        ComputeRowNorms(*typed, env.ctx, up_norms);
      });
      // Data-free proxy for the magnitude of act(W1*x) * (W2*x).
      // Calibration instead supplies the actual hidden activation RMS.
      for (size_t i = 0; i < up_norms.size(); ++i) up_norms[i] *= gate_norms[i];
    }
    auto entry = std::make_unique<Entry>(down, env.ctx.allocator,
                                       QuantBlockSize(down));
    CallUpcasted(&down, [&](const auto* typed) {
      ConfigureL2Scale(*typed, up_norms, up.RowBytes(0), *entry, env.ctx,
                       calibrated);
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
    const char* include = getenv("GEMMA_MM_I8_SCALE_NORM_INCLUDE");
    if (include != nullptr && *include != '\0' &&
        !MMI8NameMatches(norm.Name(), include))
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
    std::vector<double> input_rms;
    const bool calibrated = LoadInputRMS(*consumers.front(), input_rms);
    for (size_t i = 1; i < consumers.size(); ++i) {
      std::vector<double> other_rms;
      const bool other_calibrated = LoadInputRMS(*consumers[i], other_rms);
      if (other_calibrated != calibrated ||
          (calibrated && other_rms != input_rms))
        HWY_ABORT("RMSNorm consumers must share identical activation RMS: "
                  "%s and %s", consumers.front()->Name(), consumers[i]->Name());
    }
    if (calibrated) MeasuredInputScales(input_rms, column_sq, scales);
    for (size_t c = 0; c < k; ++c) {
      const float g =
          1.0f + gamma[c];  // Gemma's stored gamma is offset by one.
      if (!calibrated)
        scales[c] = MMI8L2Scale(std::abs(g), std::sqrt(column_sq[c]));
      folded->Row(0)[c] = g * scales[c] - 1.0f;
    }

    for (const MatPtr* b : consumers) input_scales_[b->RowBytes(0)] = scales;
    if (verbose_)
      fprintf(stderr, "MM.I8: RMSNorm fold %s, %zu consumers, %s RMS\n",
              norm.Name(), consumers.size(), calibrated ? "measured" : "proxy");
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
    auto entry = std::make_unique<Entry>(B, env.ctx.allocator,
                                       QuantBlockSize(B));
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
      for (size_t r = 0; r < entry->bias.size(); ++r)
        entry->bias[r] *= output->second[r];
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

  static bool PackedHeadEligible(const MatPtr& B, size_t block_size) {
    return MMI8PackedHead() && B.Rows() >= 65536 && B.Rows() % 8 == 0 &&
           (block_size == 32 || block_size == 64 || block_size == 128);
  }

  static bool DualAEligible(const MatPtr& B, size_t block_size) {
    static const bool down = MMI8Flag("GEMMA_MM_I8_DUAL_A_DOWN");
    static const bool head = MMI8Flag("GEMMA_MM_I8_DUAL_A_HEAD");
    static const bool body = MMI8Flag("GEMMA_MM_I8_DUAL_A_BODY");
    if ((!down && !head && !body) || !HWY_ARCH_X86_64 || !MMI8FastMicro() ||
        !MMI8NativeVNNI() ||
        B.Rows() % 8 != 0 ||
        (block_size != 32 && block_size != 64 && block_size != 128) ||
        B.Cols() % block_size != 0)
      return false;
    const bool is_embedding = strcmp(B.Name(), "c_embedding") == 0;
    if (is_embedding) return head && B.Rows() >= 65536;
    if (body) return true;
    if (!down) return false;
    // Limit the down selection to ordinary FFN-down tensors. A head requires
    // its separate flag; other tensors containing "linear" remain unchanged.
    constexpr char prefix[] = "linear_w_";
    if (strncmp(B.Name(), prefix, sizeof(prefix) - 1) != 0) return false;
    const char* suffix = B.Name() + sizeof(prefix) - 1;
    if (*suffix == '\0') return false;
    for (; *suffix != '\0'; ++suffix)
      if (*suffix < '0' || *suffix > '9') return false;
    return true;
  }

  static bool PackedEligible(const MatPtr& B, size_t block_size) {
    return PackedHeadEligible(B, block_size) ||
           DualAEligible(B, block_size);
  }

  struct Entry {
    Entry(const MatPtr& B, const Allocator& allocator, size_t block_size)
        : data("B_i8", Extents2D(B.Rows(), B.Cols()), allocator,
               PackedEligible(B, block_size) ? MatPadding::kPacked
                                             : MatPadding::kOdd),
          scale(B.Rows() * (block_size ? B.Cols() / block_size : 1)) {
      b = MMI8B{&data, scale.data(), nullptr, block_size};
      b.dual_a = DualAEligible(B, block_size);
    }
    MatStorageT<int8_t> data;
    hwy::AlignedVector<float> scale;
    hwy::AlignedVector<float> bias;
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

  bool LoadInputRMS(const MatPtr& down, std::vector<double>& rms) {
    const char* directory = getenv("GEMMA_MM_I8_SCALE_CALIBRATION_DIR");
    if (directory == nullptr || *directory == '\0') return false;
    const std::string path = std::string(directory) + "/" +
                             MMI8CalibrationName(down.Name()) + ".rms";
    FILE* file = fopen(path.c_str(), "rb");
    if (file == nullptr) {
      if (errno == ENOENT) return false;
      HWY_ABORT("Cannot open activation RMS calibration %s", path.c_str());
    }
    unsigned char header[16];
    if (fread(header, 1, sizeof(header), file) != sizeof(header) ||
        memcmp(header, "MMI8RM01", 8) != 0)
      HWY_ABORT("Invalid activation RMS calibration header %s", path.c_str());
    uint64_t k = 0;
    for (size_t i = 0; i < 8; ++i) k |= uint64_t{header[8 + i]} << (8 * i);
    const uint16_t endian = 1;
    if (k != down.Cols() ||
        *reinterpret_cast<const unsigned char*>(&endian) != 1)
      HWY_ABORT("Activation RMS dimensions/endian mismatch %s", path.c_str());
    std::vector<float> values(static_cast<size_t>(k));
    if (fread(values.data(), sizeof(float), values.size(), file) != values.size() ||
        fgetc(file) != EOF || ferror(file))
      HWY_ABORT("Invalid activation RMS calibration payload %s", path.c_str());
    if (fclose(file) != 0)
      HWY_ABORT("Cannot close activation RMS calibration %s", path.c_str());
    rms.resize(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      if (!std::isfinite(values[i]) || values[i] < 0.0f)
        HWY_ABORT("Invalid activation RMS value %s", path.c_str());
      rms[i] = values[i];
    }
    // H/G and mean files describe a specific input basis. They must be
    // regenerated after input equalization before the two can be combined.
    const char* weight_calibration = getenv("GEMMA_MM_I8_CALIBRATION_DIR");
    if (weight_calibration != nullptr && *weight_calibration != '\0')
      HWY_ABORT("Activation scaling requires its own transformed H/G basis; "
                "unset GEMMA_MM_I8_CALIBRATION_DIR");
    return true;
  }

  // For activation RMS a_i and joint consumer-column L2 b_i, the isotropic
  // quantization-error proxy is (sum a_i^2 s_i^2)(sum b_i^2 / s_i^2).
  // Its minimizer has s_i proportional to sqrt(b_i / a_i). A common factor
  // cancels in the real-valued product; normalize before rounding/clamping.
  size_t MeasuredInputScales(const std::vector<double>& input_rms,
                             const std::vector<double>& column_sq,
                             hwy::AlignedVector<float>& scales) {
    HWY_DASSERT(input_rms.size() == column_sq.size() && !input_rms.empty());
    const size_t k = input_rms.size();
    scales.resize(k);
    double alpha = 0.5;
    const char* value = getenv("GEMMA_MM_I8_SCALE_CALIBRATION_ALPHA");
    if (value != nullptr && *value != '\0') {
      char* end = nullptr;
      alpha = strtod(value, &end);
      if (end == value || *end != '\0' || !std::isfinite(alpha) ||
          alpha < 0.0 || alpha > 1.0)
        HWY_ABORT("GEMMA_MM_I8_SCALE_CALIBRATION_ALPHA must be in [0,1]");
    }
    const bool powers_of_two =
        MMI8Flag("GEMMA_MM_I8_SCALE_CALIBRATION_POW2", true);
    const auto log_scale = [&](size_t c) {
      const double activation = HWY_MAX(input_rms[c], kMMI8L2NormFloor);
      const double weight = HWY_MAX(std::sqrt(column_sq[c]), kMMI8L2NormFloor);
      return alpha * (std::log(weight) - std::log(activation));
    };
    double mean_log_scale = 0.0;
    for (size_t c = 0; c < k; ++c) mean_log_scale += log_scale(c);
    mean_log_scale /= static_cast<double>(k);
    size_t clamped = 0;
    for (size_t c = 0; c < k; ++c) {
      const double centered = log_scale(c) - mean_log_scale;
      const double raw = powers_of_two
                             ? std::exp2(std::round(centered / std::log(2.0)))
                             : std::exp(centered);
      const double bounded = HWY_MIN(double(kMMI8L2ScaleMax),
                                     HWY_MAX(double(kMMI8L2ScaleMin), raw));
      scales[c] = static_cast<float>(bounded);
      clamped += raw != bounded ? 1 : 0;
    }
    return clamped;
  }

  template <typename TB>
  void ConfigureL2Scale(const MatPtrT<TB>& down,
                        const std::vector<double>& up_norms, const void* source,
                        Entry& entry, ThreadingContext& ctx,
                        bool calibrated = false) {
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
    if (calibrated) {
      clamped = MeasuredInputScales(up_norms, sum_sq, entry.input_scale);
    } else {
      for (size_t c = 0; c < K; ++c) {
        bool was_clamped = false;
        entry.input_scale[c] =
            MMI8L2Scale(up_norms[c], std::sqrt(sum_sq[c]), &was_clamped);
        clamped += was_clamped ? 1 : 0;
      }
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
              "MM.I8: %s %-16s scale[min/med/p95/max]="
              "%.5g/%.5g/%.5g/%.5g clamped=%zu weight|max|=%.5g->%.5g\n",
              calibrated ? "RMS" : "L2", down.Name(), sorted.front(),
              sorted[K / 2], sorted[p95],
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
    const char* include = getenv("GEMMA_MM_I8_CALIBRATION_INCLUDE");
    const bool selected =
        (include == nullptr || *include == '\0' ||
         MMI8NameMatches(B.Name(), include)) &&
        !MMI8NameMatches(B.Name(), getenv("GEMMA_MM_I8_CALIBRATION_EXCLUDE"));
    MMI8WeightIO interchange(
        B, entry.b.block_size, selected,
        entry.input_scale.empty() ? nullptr : entry.input_scale.data());
    const MMI8WeightCalibration calibration(
        B, entry.b.block_size, selected && !interchange.Importing());
    const MMI8MeanCalibration means(B, entry.b.block_size, selected);
    if (means.Enabled()) {
      entry.bias.resize(B.Rows());
      entry.b.bias = entry.bias.data();
    }

    const bool needs_original = !interchange.Importing() ||
                                interchange.Exporting() || means.Enabled();
    for (size_t r = 0; r < B.Rows(); ++r) {
      if (needs_original) {
        DecompressAndZeroPad(df, span, r * B.Stride(), row.data(), K);
        if (!entry.input_scale.empty()) {
          for (size_t c = 0; c < K; ++c) row[c] /= entry.input_scale[c];
        }
        MMI8Rotate(row.data(), K);
        interchange.ExportRow(r, row.data());
      }
      MMI8BT* HWY_RESTRICT out = HWY_RCAST_ALIGNED(MMI8BT*, entry.data.Row(r));
      const bool imported = interchange.ImportRow(r, out);
      const size_t group_size = entry.b.block_size ? entry.b.block_size : K;
      double bias = 0.0;
      for (size_t c = 0; c < K; c += group_size) {
        float refined_scale;
        if (imported) {
          refined_scale = interchange.Scale(r, c / group_size);
        } else {
          const float initial_scale =
              PackBRow(row.data() + c, group_size, out + c, group_size);
          refined_scale =
              calibration.Refine(row.data() + c, c, out + c, initial_scale);
        }
        entry.scale[(c / group_size) * B.Rows() + r] = b_scale * refined_scale;
        bias += means.Correction(row.data() + c, c, out + c, refined_scale);
      }
      if (means.Enabled()) {
        entry.bias[r] = static_cast<float>(double(b_scale) * bias);
        if (!std::isfinite(entry.bias[r]))
          HWY_ABORT("Nonfinite calibrated output bias for %s", B.Name());
      }
      for (size_t c = K; c < entry.data.Stride(); ++c) out[c] = 0;
    }
    if (PackedEligible(B, entry.b.block_size)) {
      MMI8PackMicroB(entry.data);
      entry.b.packed_micro = true;
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
  if (MMI8CalibrationCaptureEnabled()) {
    MMI8CalibrationCapture::Get().Capture(A, B1);
    MMI8CalibrationCapture::Get().Capture(A, B2);
  }
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
  if (MMI8CalibrationCaptureEnabled())
    MMI8CalibrationCapture::Get().Capture(A, B);
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
