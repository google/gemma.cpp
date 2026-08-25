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

#include "gemma/tensor_stats.h"

#if GCPP_TENSOR_STATS
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <atomic>
#include <cmath>
#include <memory>

#include "io/io.h"
#include "util/mat.h"
#include "util/threading_context.h"
#include "util/zones.h"
#include "hwy/profiler.h"  // StringTable

// Compiles this file for multiple architectures via "foreach_target.h", to
// which we pass the filename via macro 'argument'.
// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/tensor_stats.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"
#include "ops/dot-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

float Correlation(const float* x, size_t num) {
  double sum = 0.0;
  for (size_t i = 0; i < num; ++i) {
    sum += x[i];
  }
  const double mean = sum / static_cast<double>(num);

  double numerator = 0.0;
  double sum_sq_current = 0.0;
  double sum_sq_next = 0.0;

  for (size_t i = 0; i < num - 1; ++i) {
    const double diff_current = static_cast<double>(x[i]) - mean;
    const double diff_next = static_cast<double>(x[i + 1]) - mean;

    numerator += diff_current * diff_next;
    sum_sq_current += diff_current * diff_current;
    sum_sq_next += diff_next * diff_next;
  }

  if (sum_sq_current == 0.0 || sum_sq_next == 0.0) return 0.0f;
  const double denominator = std::sqrt(sum_sq_current * sum_sq_next);
  const float corr = static_cast<float>(numerator / denominator);
  HWY_DASSERT(-1.0f <= corr && corr <= 1.0f);
  return corr;
}

// Only write tensor data the first time it is encountered per layer. This is
// a concurrent string+layer -> flag map which avoids std::mutex (incompatible
// with fibers). We use a string table to index into per-layer atomic flags.
static bool ShouldWrite(const char* name, size_t layer_idx) {
  constexpr size_t kMaxNames = 128;
  constexpr size_t kMaxLayers = 128;
  HWY_DASSERT(layer_idx < kMaxLayers);
  static hwy::StringTable<kMaxNames> s_table;
  const size_t name_idx = s_table.Add(name);
  static std::atomic_flag flags[kMaxNames * kMaxLayers] = {};
  return !flags[name_idx * kMaxLayers + layer_idx].test_and_set(
      std::memory_order_acq_rel);
}

std::unique_ptr<File> MaybeOpenFile(size_t layer_idx, const MatPtr& type_erased,
                                    const Path& tensor_output) {
  if (tensor_output.Empty()) return nullptr;
  if (!ShouldWrite(type_erased.Name(), layer_idx)) return nullptr;
  char path[1024];
  snprintf(path, sizeof(path), "%s/%s_L%02zu_%zux%zu_%s.bin",
           tensor_output.path.c_str(), type_erased.Name(), layer_idx,
           type_erased.Rows(), type_erased.Cols(),
           TypeName(type_erased.GetType()));
  return OpenFileOrAbort(Path(path), "wb");
}

void MaybeWriteRow(const std::unique_ptr<File>& file, const MatPtr& type_erased,
                   size_t row_idx) {
  if (!file) return;
  const size_t bytes_per_row = type_erased.Cols() * type_erased.ElementBytes();
  file->Write(type_erased.RowBytes(row_idx), bytes_per_row,
              bytes_per_row * row_idx);
}

constexpr size_t kGroupSize = 128;  // subchannel

void QuantizeGroup(const float* HWY_RESTRICT in,
                   TensorStatsAccumulator& my_stats) {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> df;
  using VF = hn::Vec<decltype(df)>;
  using MF = hn::Mask<decltype(df)>;
  const hn::ScalableTag<double> dd;
  using VD = hn::Vec<decltype(dd)>;
  HWY_LANES_CONSTEXPR size_t NF = hn::Lanes(df);
  HWY_ALIGN float enc[kGroupSize];
  HWY_ALIGN float dec[kGroupSize];
  HWY_ALIGN float all_snr[kGroupSize];
  HWY_DASSERT(kGroupSize % NF == 0);  // No remainder handling required.

  const VF k0 = hn::Zero(df);
  const VF k1 = hn::Set(df, 1.0f);

  // Scan for min/max for quantization.
  VF vmin = hn::Set(df, hwy::HighestValue<float>());
  VF vmax = hn::Set(df, hwy::LowestValue<float>());
  for (size_t i = 0; i < kGroupSize; i += NF) {
    const VF v = hn::Load(df, in + i);
    vmin = hn::Min(vmin, v);
    vmax = hn::Max(vmax, v);
  }
  const float min = hn::ReduceMin(df, vmin);
  const float max = hn::ReduceMax(df, vmax);
  // Avoid division by zero during quantization.
  if (max == min) return;

  // Distortion stats.
  VF vsum_err = hn::Zero(df);
  VD sum_log_snr0 = hn::Zero(dd);
  VD sum_log_snr1 = hn::Zero(dd);
  size_t num_snr = 0;

  // Unclipped asymmetric quantization (for activations).
  const VF scale = hn::Set(df, 255.0f / (max - min));
  const VF inv_scale = hn::Div(k1, scale);
  const VF zeropoint = hn::Sub(hn::Round(hn::Mul(hn::Set(df, -min), scale)),
                               hn::Set(df, 128.0f));
  const VF dq_sub = hn::Mul(zeropoint, inv_scale);  // For MulSub.
  for (size_t i = 0; i < kGroupSize; i += NF) {
    const VF v = hn::Load(df, in + i);
    const VF q = hn::Round(hn::MulAdd(v, scale, zeropoint));
    hn::Store(q, df, enc + i);
    // Dequantize.
    const VF d = hn::MulSub(q, inv_scale, dq_sub);
    hn::Store(d, df, dec + i);

    const VF err = hn::AbsDiff(v, d);  // L1
    vsum_err = hn::Add(vsum_err, err);

    // For preventing division by zero. However, we still want to
    // clamp snr because it could be very high (>1E3 when most
    // elements are lossless).
    const MF has_err = hn::Gt(err, k0);
    const VF rel = hn::MaskedDivOr(k0, has_err, hn::Abs(v), err);
    // SNR = 1 + abs/L1, with cap on the latter term.
    const VF snr = hn::Add(k1, hn::Min(rel, hn::Set(df, 300.f)));
    hn::Store(snr, df, all_snr + i);
    // Where `has_err` is false, `snr` elements are 1 and log(1) is zero, hence
    // they do not affect sum_log. However, very high errors also result in
    // snr=1, which drags down the average because `sum_log` is increased.
    num_snr += hn::CountTrue(df, has_err);

    const VD log_snr0 = hn::Log(dd, hn::PromoteLowerTo(dd, snr));
    const VD log_snr1 = hn::Log(dd, hn::PromoteUpperTo(dd, snr));
    sum_log_snr0 = hn::Add(sum_log_snr0, log_snr0);
    sum_log_snr1 = hn::Add(sum_log_snr1, log_snr1);
  }

  const float sum_err = hn::ReduceSum(df, vsum_err);
  const float avg_L1 = sum_err / static_cast<float>(kGroupSize);
  const double sum_log = hn::ReduceSum(dd, hn::Add(sum_log_snr0, sum_log_snr1));
  // SNR >= 1, hence log >= 0.
  HWY_ASSERT(sum_log >= 0.0);
  if (num_snr == 0) {  // Avoid division by zero.
    // It can happen that dequantization is lossless, i.e. SNR is
    // infinite; skip such groups.
    HWY_ASSERT(sum_err == 0.0f);
    return;
  }
  // Signal to noise ratio (Shannon's channel capacity, NOT the
  // L2-based and logarithmic PSNR)
  const float snr = std::exp(sum_log / static_cast<double>(num_snr));

  my_stats.NotifyGroup(avg_L1, snr);
}

// First dispatch to the type, then parallel over rows, then vectorized
// decompress and Notify for each value.
void UpdateStatsT(TensorStats& stats, size_t layer_idx,
                  const MatPtr& type_erased, ThreadingContext& ctx, int flags,
                  size_t cluster_idx, Parallelism parallelism) {
  std::unique_ptr<File> file =
      MaybeOpenFile(layer_idx, type_erased, ctx.tensor_output);

  if ((flags & kTensorStatsIsWeight) && layer_idx != 0) {
    // Still compute stats, but remember not to print them.
    stats.Get(layer_idx, 0).DoNotPrint();
  }

  CallUpcasted(&type_erased, [&](const auto* mat) {
    const size_t cols = mat->Cols();

    ParallelFor(
        parallelism, mat->Rows(), ctx, cluster_idx, Callers::kTensorStats,
        [&](size_t row_idx, size_t global_idx) {
          GCPP_ZONE(ctx, global_idx, Zones::kGenStats);

          auto* HWY_RESTRICT row = mat->Row(row_idx);
          MaybeWriteRow(file, type_erased, row_idx);

          using Packed = hwy::RemoveCvRef<decltype(*row)>;
          PackedSpan<Packed> packed(const_cast<Packed*>(row), cols);

          TensorStatsAccumulator& my_stats = stats.Get(layer_idx, global_idx);
          my_stats.NotifyCond(ConditionNumber(row, cols));

          namespace hn = hwy::HWY_NAMESPACE;
          const hn::ScalableTag<float> df;
          using VF = hn::Vec<decltype(df)>;
          HWY_LANES_CONSTEXPR size_t NF = hn::Lanes(df);
          HWY_ALIGN float buf[kGroupSize];
          size_t buf_filled = 0;

          size_t packed_ofs = 0;
          if (cols >= 2 * NF) {
            for (; packed_ofs <= cols - 2 * NF; packed_ofs += 2 * NF) {
              VF v0, v1;
              Decompress2(df, packed, packed_ofs, v0, v1);
              hn::Store(v0, df, buf + buf_filled);
              hn::Store(v1, df, buf + buf_filled + NF);
              buf_filled += 2 * NF;
              if (buf_filled == kGroupSize) {
                QuantizeGroup(buf, my_stats);

                for (size_t i = 0; i < kGroupSize; ++i) {
                  my_stats.Notify(buf[i], row_idx, packed_ofs + i);
                }
                my_stats.NotifyCorr(Correlation(buf, kGroupSize));

                buf_filled = 0;
              }
            }
          }

          // Zero to two vectors remaining.
          for (; packed_ofs < cols; packed_ofs += NF) {
            const size_t remaining = HWY_MIN(NF, cols - packed_ofs);
            DecompressAndZeroPad(df, packed, packed_ofs, buf, remaining);
            // Skip QuantizeGroup because it requires full groups.
            for (size_t i = 0; i < remaining; ++i) {
              my_stats.Notify(buf[i], row_idx, packed_ofs + i);
            }
            my_stats.NotifyCorr(Correlation(buf, remaining));
          }
        });
  });
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {
HWY_EXPORT(UpdateStatsT);

// Must reside in .cc file so that we can #include compress-inl.h.
void TensorStats::Notify(size_t layer_idx, const MatPtr& type_erased,
                         ThreadingContext& ctx, int flags, size_t cluster_idx,
                         Parallelism parallelism) {
  // Ignore empty tensors.
  if (type_erased.GetType() == Type::kUnknown || type_erased.Cols() == 0) {
    return;
  }
  HWY_DYNAMIC_DISPATCH(UpdateStatsT)(*this, layer_idx, type_erased, ctx, flags,
                                     cluster_idx, parallelism);
}

}  // namespace gcpp
#endif  // HWY_ONCE

#endif  // GCPP_TENSOR_STATS
