// Copyright 2024 Google LLC
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

// Normal include guard to placate lint.
#ifndef THIRD_PARTY_GEMMA_CPP_COMPRESSION_ANALYZE_H_
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_ANALYZE_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>  // memcpy

#include <cmath>    // std::signbit
#include <cstdlib>  // std::abs
#include <vector>

#include "compression/types.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/stats.h"

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_ANALYZE_H_

// Actual per-target include guard.
#if defined(THIRD_PARTY_GEMMA_CPP_ANALYZE_TOGGLE) == defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_ANALYZE_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_ANALYZE_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_ANALYZE_TOGGLE
#endif

#include "compression/nuq-inl.h"
#include "compression/sfp-inl.h"
#include "hwy/contrib/sort/vqsort-inl.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

class PerThread {
 public:
  void NotifyGroup(const float* group) {
    constexpr size_t kGroupSize = NuqStream::kGroupSize;
    hwy::Stats s_group;
    for (size_t i = 0; i < kGroupSize; ++i) {
      // Skip zero so we can see the lowest actual magnitude
      if (group[i] == 0.0f || group[i] == -0.0f) continue;
      s_all_.Notify(group[i]);
      s_group.Notify(group[i]);

      num_tiny_ += std::abs(group[i]) < 1e-3f;

      // b_magn100_.Notify(group[i] * 40.0f + 20.0f);
      const uint32_t binary32 =
          hwy::BitCastScalar<uint32_t>(std::abs(group[i]));

      // const int32_t exp = (binary32 >> 23) - 127;
      b_exp256_.Notify(binary32 >> 23);
      const uint32_t m4 = (binary32 & 0x7FFFFF) >> (23 - 4);
      b_m4_.Notify(m4);
    }
    s_group_ranges_.Notify(s_group.Max() - s_group.Min());
    s_group_mins_.Notify(s_group.Min());
    s_group_maxs_.Notify(s_group.Max());

    float desc[kGroupSize];
    memcpy(desc, group, kGroupSize * sizeof(group[0]));
    hn::VQSortStatic(desc, kGroupSize, hwy::SortDescending());

    // Find largest |max/min| (dynamic range)
    float max_ratio = 0.0f;
    for (size_t i = 0; i < kGroupSize; ++i) {
      if (desc[i] != 0.0f && desc[i] != -0.0f) {
        max_ratio = std::max(max_ratio, std::abs(desc[0] / desc[i]));
      }
    }
    s_group_max_vs_min_.Notify(max_ratio);

    // Relative errors
    float diffs[kGroupSize];
    for (size_t i = 0; i < kGroupSize - 1; ++i) {
      // was in descending order. Avoid div by 0. Ignore sign changes.
      diffs[i] = std::abs(desc[i]) < 1e-5
                     ? 0
                     : std::abs((desc[i] - desc[i + 1]) / desc[i]);
    }
    hn::VQSortStatic(diffs, kGroupSize, hwy::SortDescending());
    s_cut15_.Notify(diffs[15]);
  }

  void Assimilate(const PerThread& other) {
    num_tiny_ += other.num_tiny_;
    s_all_.Assimilate(other.s_all_);
    s_group_ranges_.Assimilate(other.s_group_ranges_);
    s_group_mins_.Assimilate(other.s_group_mins_);
    s_group_maxs_.Assimilate(other.s_group_maxs_);
    s_group_max_vs_min_.Assimilate(other.s_group_max_vs_min_);
    s_erange_.Assimilate(other.s_erange_);
    s_km_1_.Assimilate(other.s_km_1_);
    s_km_2_.Assimilate(other.s_km_2_);
    s_cut15_.Assimilate(other.s_cut15_);
    b_magn100_.Assimilate(other.b_magn100_);
    b_exp256_.Assimilate(other.b_exp256_);
    b_m4_.Assimilate(other.b_m4_);
  }

  void PrintAll() {
    const int skip = hwy::Stats::kNoGeomean;
    fprintf(stderr, "num tiny %zu\n", num_tiny_);
    fprintf(stderr, "weights %s\n", s_all_.ToString(skip).c_str());
    fprintf(stderr, " ranges %s\n", s_group_ranges_.ToString(skip).c_str());
    fprintf(stderr, "   mins %s\n", s_group_mins_.ToString(skip).c_str());
    fprintf(stderr, "   maxs %s\n", s_group_maxs_.ToString(skip).c_str());
    fprintf(stderr, "   Mvm  %s\n", s_group_max_vs_min_.ToString(skip).c_str());
    fprintf(stderr, "  cut15 %s\n", s_cut15_.ToString(skip).c_str());
    fprintf(stderr, " erange %s\n", s_erange_.ToString(skip).c_str());
    fprintf(stderr, "   km1 %s\n", s_km_1_.ToString(skip).c_str());
    fprintf(stderr, "   km2 %s\n", s_km_2_.ToString(skip).c_str());

    // b_magn100_.Print("magn100");
    // b_exp256_.Print("exp");
    // b_m4_.Print("mantissa bits4");

    fprintf(stderr, "\n");
  }

 private:
  size_t num_tiny_ = 0;
  hwy::Stats s_all_;
  hwy::Stats s_group_ranges_;
  hwy::Stats s_group_mins_;
  hwy::Stats s_group_maxs_;
  hwy::Stats s_group_max_vs_min_;
  hwy::Stats s_erange_;
  hwy::Stats s_km_1_;
  hwy::Stats s_km_2_;
  hwy::Stats s_cut15_;
  hwy::Bins<100> b_magn100_;
  hwy::Bins<256> b_exp256_;
  hwy::Bins<16> b_m4_;
  uint8_t padding_[64];  // prevent false sharing
};

class PerLayer {
 public:
  void NotifyGroup(const float* group) {
    for (size_t i = 0; i < NuqStream::kGroupSize; ++i) {
      s_layer_.Notify(group[i]);
    }
  }

  void UpdateOutliers(const float* layer, size_t weights_per_layer) {
    const float layer_mean = s_layer_.Mean();
    const float layer_sd = s_layer_.StandardDeviation();
    for (size_t i = 0; i < weights_per_layer; ++i) {
      num_outliers_ +=
          std::abs(std::abs(layer[i]) - layer_mean) >= 3.0f * layer_sd;
    }
  }

  const hwy::Stats& GetStats() const { return s_layer_; }
  size_t Outliers() const { return num_outliers_; }

 private:
  hwy::Stats s_layer_;
  size_t num_outliers_ = 0;
  uint8_t padding[64];  // prevent false sharing
};

static HWY_NOINLINE void Analyze(const char* caption, float* mat, size_t layers,
                                 size_t weights_per_layer,
                                 hwy::ThreadPool& pool) {
  std::vector<PerThread> tls;
  std::vector<PerLayer> per_layer(layers);
  const auto init = [&](size_t num_threads) {
    tls.resize(num_threads);
    return true;
  };

  pool.Run(0, static_cast<uint32_t>(layers), init,
           [&](uint32_t idx_layer, size_t idx_thread) {
             PerThread& self = tls[idx_thread];
             const float* layer = &mat[idx_layer * weights_per_layer];
             // For each whole group in the layer
             for (size_t group_start = 0;
                  group_start + NuqStream::kGroupSize <= weights_per_layer;
                  group_start += NuqStream::kGroupSize) {
               const float* group = layer + group_start;
               per_layer[idx_layer].NotifyGroup(group);
               self.NotifyGroup(group);
             }

             per_layer[idx_layer].UpdateOutliers(layer, weights_per_layer);
           });

  const int skip = hwy::Stats::kNoGeomean;
  fprintf(stderr, "\n------------%s\n", caption);

  for (size_t i = 1; i < pool.NumWorkers(); ++i) {
    tls[0].Assimilate(tls[i]);
  }
  tls[0].PrintAll();

  hwy::Stats s_layer_ranges;
  hwy::Stats s_layer_outliers;
  for (size_t i = 0; i < layers; ++i) {
    fprintf(stderr, "  %02zu %s\n", i,
            per_layer[i].GetStats().ToString(skip).c_str());
    const float range =
        per_layer[i].GetStats().Max() - per_layer[i].GetStats().Min();
    s_layer_ranges.Notify(range);
    s_layer_outliers.Notify((100.0 * per_layer[i].Outliers()) /
                            weights_per_layer);
  }
  fprintf(stderr, "layer outliers%% %s\n",
          s_layer_outliers.ToString(skip).c_str());
  fprintf(stderr, "layer ranges %s\n", s_layer_ranges.ToString(skip).c_str());
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_ANALYZE_H_
