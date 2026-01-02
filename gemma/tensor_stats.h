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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_TENSOR_STATS_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_TENSOR_STATS_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "util/basics.h"
#include "hwy/base.h"

#ifndef GCPP_TENSOR_STATS
#define GCPP_TENSOR_STATS 0
#endif

#include "util/mat.h"
#include "util/threading_context.h"

#if GCPP_TENSOR_STATS
#include <cmath>
#include <vector>

#include "hwy/stats.h"
#endif  // GCPP_TENSOR_STATS

namespace gcpp {

// For flags. Used to inhibit printing per-layer stats for weights.
HWY_INLINE_VAR constexpr int kTensorStatsIsWeight = 1;

#if GCPP_TENSOR_STATS

HWY_INLINE_VAR constexpr size_t kStatsMaxCols = 8192;

// Separate summary of the per-layer stats, updated by `TensorStatsAccumulator`.
// We pass per-layer statistics such as the mean value to `hwy::Stats::Notify``
// to see the distribution of per-layer means.
struct TensorStatsAcrossLayers {
  bool IsEmpty() const { return s_frobenius.Count() == 0; }

  void Print() {
    const int skip = hwy::Stats::kNoGeomean;
    fprintf(stderr, "frob    %s\n", s_frobenius.ToString(skip).c_str());
    fprintf(stderr, "cnd.min %s\n", s_cond_min.ToString(skip).c_str());
    fprintf(stderr, "cnd.avg %s\n", s_cond_avg.ToString(skip).c_str());
    fprintf(stderr, "cnd.max %s\n", s_cond_max.ToString(skip).c_str());
    fprintf(stderr, "val.min %s\n", s_val_min.ToString(skip).c_str());
    fprintf(stderr, "val.avg %s\n", s_val_avg.ToString(skip).c_str());
    fprintf(stderr, "val.krt %s\n", s_val_kurt.ToString(skip).c_str());
    fprintf(stderr, "mag.min %s\n", s_mag_min.ToString(skip).c_str());
    fprintf(stderr, "mag.avg %s\n", s_mag_avg.ToString(skip).c_str());
    fprintf(stderr, "mag.max %s\n", s_mag_max.ToString(skip).c_str());
    if (hwy::ScalarAbs(s_corr_avg.Max()) > 0.05f) {
      fprintf(stderr, "cor.avg %s\n", s_corr_avg.ToString(skip).c_str());
    }
    fprintf(stderr, "cor.max %s\n", s_corr_max.ToString(skip).c_str());
    fprintf(stderr, "err_avg %s\n", s_grp_err_avg.ToString(skip).c_str());
    fprintf(stderr, "err_std %s\n", s_grp_err_std.ToString(skip).c_str());
    fprintf(stderr, "err_max %s\n", s_grp_err_max.ToString(skip).c_str());
    fprintf(stderr, "snr_1   %s\n", s_grp_snr1.ToString(skip).c_str());
    fprintf(stderr, "snr_avg %s\n", s_grp_snr_avg.ToString(skip).c_str());
    fprintf(stderr, "snr_std %s\n", s_grp_snr_std.ToString(skip).c_str());
    fprintf(stderr, "exp.min %s\n", s_exp_min.ToString(skip).c_str());
    fprintf(stderr, "exp.max %s\n", s_exp_max.ToString(skip).c_str());
    fprintf(stderr, "exp.mod %s\n", s_exp_mode.ToString(skip).c_str());
    if (s_exp_subnormal.Min() != 0.0f) {
      fprintf(stderr, "exp.sub %s\n", s_exp_subnormal.ToString(skip).c_str());
    }
    if (s_big_cols.Count() != 0) {
      fprintf(stderr, "bigCols %s\n", s_big_cols.ToString(skip).c_str());
      const size_t modal_col = b_big_cols.ModalBinIdx();
      const size_t num_outlier_cols = b_big_cols.NumNonzero();
      if (num_outlier_cols > 256) {
        fprintf(stderr, "bigCols: all up to %zu (max at %zu: %u layers):\n",
                b_big_cols.LastNonzero(), modal_col, b_big_cols.Bin(modal_col));
      } else {
        fprintf(stderr, "bigCols (max at %zu: %u layers):\n", modal_col,
                b_big_cols.Bin(modal_col));
        for (size_t i = 0; i < kStatsMaxCols; ++i) {
          if (b_big_cols.Bin(i) > 2) {
            fprintf(stderr, " %3zu: %u\n", i, b_big_cols.Bin(i));
          }
        }
      }
    }
    fprintf(stderr, "\n");
  }

  hwy::Stats s_frobenius;

  hwy::Stats s_cond_min;
  hwy::Stats s_cond_avg;
  hwy::Stats s_cond_max;

  hwy::Stats s_val_min;
  hwy::Stats s_val_avg;
  hwy::Stats s_val_kurt;

  hwy::Stats s_mag_min;
  hwy::Stats s_mag_avg;
  hwy::Stats s_mag_max;

  hwy::Stats s_corr_avg;
  hwy::Stats s_corr_max;

  hwy::Stats s_grp_err_avg;
  hwy::Stats s_grp_err_std;
  hwy::Stats s_grp_err_max;
  hwy::Stats s_grp_snr1;
  hwy::Stats s_grp_snr_avg;
  hwy::Stats s_grp_snr_std;

  hwy::Stats s_exp_min;
  hwy::Stats s_exp_max;
  hwy::Stats s_exp_mode;
  hwy::Stats s_exp_subnormal;

  hwy::Stats s_big_cols;                // total number of outlier cols
  hwy::Bins<kStatsMaxCols> b_big_cols;  // # layers with outlier per col
};

// Per-thread and layer.
class TensorStatsAccumulator {
 public:
  void Notify(float val, size_t row_idx, size_t col_idx) {
    const double dval = static_cast<double>(val);
    sum_sq_ += dval * dval;

    s_val_.Notify(val);
    const float mag = hwy::ScalarAbs(val);

    if (HWY_UNLIKELY(mag >= 64.0f)) {
      if (row_idx < kMaxBatchSize) b_big_row_.Notify(row_idx);
      if (col_idx < kStatsMaxCols) b_big_col_.Notify(col_idx);
    }

    // Skip zero so we can see the lowest actual magnitude
    if (mag != 0.0f && mag != -0.0f) s_mag_.Notify(mag);

    const uint32_t binary32 = hwy::BitCastScalar<uint32_t>(mag);
    // Use biased exponent because Bins wants unsigned values.
    const uint32_t biased_exp = binary32 >> 23;
    HWY_DASSERT(biased_exp < 256);  // already cleared sign bit
    b_exp256_.Notify(biased_exp);
  }

  void DoNotPrint() { skip_.fetch_or(1); }
  bool ShouldPrint() const { return skip_.load() == 0; }

  // Computed by vector code, much faster than doing it in `Notify`.
  void NotifyGroup(float avg_L1, float snr) {
    s_group_err_.Notify(avg_L1);
    s_group_snr_.Notify(snr);
    num_snr1_ += (snr == 1.0f);
  }

  void NotifyCorr(float corr) { s_corr_.Notify(corr); }
  void NotifyCond(double cond) { s_cond_.Notify(cond); }

  void Assimilate(const TensorStatsAccumulator& other) {
    skip_.fetch_or(other.skip_.load());

    sum_sq_ += other.sum_sq_;
    b_exp256_.Assimilate(other.b_exp256_);
    b_big_row_.Assimilate(other.b_big_row_);
    b_big_col_.Assimilate(other.b_big_col_);
    s_val_.Assimilate(other.s_val_);
    s_mag_.Assimilate(other.s_mag_);
    s_corr_.Assimilate(other.s_corr_);
    s_group_err_.Assimilate(other.s_group_err_);
    s_group_snr_.Assimilate(other.s_group_snr_);
    num_snr1_ += other.num_snr1_;
  }

  // Called on the per-layer representative after reducing across threads.
  void NotifyAcrossLayer(TensorStatsAcrossLayers& s) {
    s.s_frobenius.Notify(std::sqrt(sum_sq_));

    s.s_cond_min.Notify(s_cond_.Min());
    s.s_cond_avg.Notify(s_cond_.Mean());
    s.s_cond_max.Notify(s_cond_.Max());

    s.s_val_min.Notify(s_val_.Min());
    s.s_val_avg.Notify(s_val_.Mean());
    s.s_val_kurt.Notify(s_val_.Kurtosis());

    s.s_mag_min.Notify(s_mag_.Min());
    s.s_mag_avg.Notify(s_mag_.Mean());
    s.s_mag_max.Notify(s_mag_.Max());

    s.s_corr_avg.Notify(s_corr_.Mean());
    s.s_corr_max.Notify(s_corr_.Max());

    s.s_grp_err_avg.Notify(s_group_err_.Mean());
    s.s_grp_err_std.Notify(s_group_err_.StandardDeviation());
    s.s_grp_err_max.Notify(s_group_err_.Max());
    s.s_grp_snr1.Notify(static_cast<float>(num_snr1_));
    s.s_grp_snr_avg.Notify(s_group_snr_.Mean());
    s.s_grp_snr_std.Notify(s_group_snr_.StandardDeviation());

    const uint32_t subnormals = b_exp256_.Bin(0);
    // Prevent subnormals from hiding the min exponent.
    b_exp256_.ResetBin(0);
    s.s_exp_min.Notify(b_exp256_.FirstNonzero());
    s.s_exp_max.Notify(b_exp256_.LastNonzero());
    s.s_exp_mode.Notify(b_exp256_.ModalBinIdx());
    s.s_exp_subnormal.Notify(subnormals);

    const uint32_t num_outliers = b_big_col_.NumNonzero();
    if (num_outliers != 0) {
      s.s_big_cols.Notify(num_outliers);
      // For each col, count the number of layers that have an outlier there.
      for (size_t i = 0; i < kStatsMaxCols; ++i) {
        if (b_big_col_.Bin(i) != 0) s.b_big_cols.Notify(i);
      }
    }
  }

  bool IsEmpty() const { return s_val_.Count() == 0; }

  void PrintAll() {
    fprintf(stderr, "Frob %.2E\n", std::sqrt(sum_sq_));
    const int skip = hwy::Stats::kNoGeomean;
    fprintf(stderr, "cnd %s\n", s_cond_.ToString(skip).c_str());
    fprintf(stderr, "val %s\n", s_val_.ToString(skip).c_str());
    fprintf(stderr, "mag %s\n", s_mag_.ToString(skip).c_str());
    fprintf(stderr, "crr %s\n", s_corr_.ToString(skip).c_str());
    fprintf(stderr, "err %s\n", s_group_err_.ToString(skip).c_str());
    fprintf(stderr, "snr %s\n", s_group_snr_.ToString(skip).c_str());
    b_exp256_.Print("exp");
    PrintBinRanges(b_big_row_, "big row");
    PrintBinRanges(b_big_col_, "big col");
    fprintf(stderr, "\n");
  }

 private:
  template <size_t N>
  void PrintBinRanges(const hwy::Bins<N>& b, const char* name) {
    uint64_t total = 0;
    for (size_t i = 0; i < N; ++i) {
      total += b.Bin(i);
    }
    if (total == 0) return;

    fprintf(stderr, "%s total %zu: \n", name, total);
    // Group together runs to reduce the log size.
    const size_t min = HWY_MAX(1, total / (N * 10));
    for (size_t i = 0; i < N; ++i) {
      if (b.Bin(i) == 0) continue;
      if (b.Bin(i) < min) {
        fprintf(stderr, " %3zu: %zu\n", i, b.Bin(i));
        continue;
      }
      const size_t first = i;
      while (i + 1 < N && b.Bin(i + 1) >= min) {
        i++;
      }
      if (first == i) {
        fprintf(stderr, " %3zu: %zu\n", i, b.Bin(i));
      } else {
        fprintf(stderr, " [%3zu, %3zu]\n", first, i);
      }
    }
  }

  double sum_sq_ = 0.0;      // for Frobenius norm
  hwy::Bins<256> b_exp256_;  // exponent
  hwy::Bins<kMaxBatchSize> b_big_row_;
  hwy::Bins<kStatsMaxCols> b_big_col_;
  hwy::Stats s_val_;
  hwy::Stats s_mag_;
  hwy::Stats s_cond_;  // condition number
  hwy::Stats s_corr_;  // lag-1 autocorrelation
  hwy::Stats s_group_err_;
  hwy::Stats s_group_snr_;
  size_t num_snr1_ = 0;
  std::atomic<int> skip_{0};
};

class TensorStats {
 public:
  TensorStats(size_t num_layers, size_t max_workers)
      : num_layers_(num_layers),
        max_workers_(max_workers),
        acc_(num_layers * max_workers) {}

  // Parallelized across rows. If `ctx.tensor_output` is not empty, writes
  // tensor data to disk for offline analysis, once per tensor and layer.
  void Notify(size_t layer_idx, const MatPtr& type_erased,
              ThreadingContext& ctx, int flags = 0, size_t cluster_idx = 0,
              Parallelism parallelism = Parallelism::kFlat);

  // For use by `UpdateStatsT`.
  TensorStatsAccumulator& Get(size_t layer_idx, size_t global_idx) {
    const size_t idx = layer_idx * max_workers_ + global_idx;
    HWY_DASSERT(idx < acc_.size());
    return acc_[idx];
  }

  void ReduceAndPrint(const char* prefix) {
    for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
      TensorStatsAccumulator& per_layer = Get(layer_idx, 0);
      for (size_t global_idx = 1; global_idx < max_workers_; ++global_idx) {
        per_layer.Assimilate(Get(layer_idx, global_idx));
      }
      if (per_layer.IsEmpty()) continue;
      per_layer.NotifyAcrossLayer(across_layers_);
      if (per_layer.ShouldPrint()) {
        fprintf(stderr, "-------------------- %s %zu\n", prefix, layer_idx);
        per_layer.PrintAll();
      }
    }

    if (!across_layers_.IsEmpty()) {
      fprintf(stderr, "================= across layers %s\n", prefix);
      across_layers_.Print();
    }
  }

 private:
  size_t num_layers_;
  size_t max_workers_;
  std::vector<TensorStatsAccumulator> acc_;
  TensorStatsAcrossLayers across_layers_;
};

#else  // GCPP_TENSOR_STATS

class TensorStats {
 public:
  TensorStats(size_t, size_t) {}
  void Notify(size_t, const MatPtr&, ThreadingContext&, int = 0, size_t = 0,
              Parallelism = Parallelism::kFlat) {}
  void ReduceAndPrint(const char*) {}
};

#endif  // GCPP_TENSOR_STATS
}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_TENSOR_STATS_H_
