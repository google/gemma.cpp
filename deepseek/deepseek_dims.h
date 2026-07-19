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

#ifndef THIRD_PARTY_GEMMA_CPP_DEEPSEEK_DEEPSEEK_DIMS_H_
#define THIRD_PARTY_GEMMA_CPP_DEEPSEEK_DEEPSEEK_DIMS_H_

// DeepSeek-specific activation-buffer sizing and RoPE timescales, used by
// `Activations` (gemma/activations.h).

#include <stddef.h>

#include <cmath>

#include "gemma/configs.h"
#include "util/allocator.h"
#include "util/mat.h"
#include "hwy/base.h"  // HWY_MAX

namespace gcpp {

static inline size_t MaxIndexerHeads(const ModelConfig& config) {
  size_t max_heads = 0;
  for (const LayerConfig& lc : config.layer_configs) {
    max_heads = HWY_MAX(max_heads, size_t{lc.indexer_heads});
  }
  return max_heads;
}

static inline size_t MaxOLoraRank(const ModelConfig& config) {
  size_t max_rank = 0;
  for (const LayerConfig& lc : config.layer_configs) {
    max_rank = HWY_MAX(max_rank, size_t{lc.o_lora_rank});
  }
  return max_rank;
}

// Per-batch buffer widths for DeepSeek MLA layers; all zero if none.
struct MLADims {
  explicit MLADims(const ModelConfig& config) {
    for (const LayerConfig& lc : config.layer_configs) {
      if (!lc.IsMLA()) continue;
      q_lora_rank = HWY_MAX(q_lora_rank, size_t{lc.q_lora_rank});
      kv_a_dim =
          HWY_MAX(kv_a_dim, size_t{lc.kv_lora_rank} + lc.rope_head_dim);
      indexer_dim = HWY_MAX(indexer_dim, size_t{lc.indexer_heads} *
                                             lc.indexer_head_dim);
      rope_head_dim = HWY_MAX(rope_head_dim, size_t{lc.rope_head_dim});
      if (lc.HasCompressor()) {
        comp_dim = HWY_MAX(comp_dim, lc.CompressorCoff() * lc.KVLatentDim());
        if (lc.HasIndexer()) {
          idxc_dim = HWY_MAX(
              idxc_dim, lc.CompressorCoff() * size_t{lc.indexer_head_dim});
        }
      }
      if (lc.IsV4MLA()) {
        o_in_dim = HWY_MAX(o_in_dim, size_t{lc.heads} / lc.o_groups *
                                         lc.qkv_dim);
        o_mid_dim =
            HWY_MAX(o_mid_dim, size_t{lc.o_groups} * lc.o_lora_rank);
      }
    }
  }
  bool Any() const { return kv_a_dim > 0; }
  size_t q_lora_rank = 0;
  size_t kv_a_dim = 0;
  size_t indexer_dim = 0;
  size_t rope_head_dim = 0;
  size_t comp_dim = 0;   // compressor output width (coff * latent)
  size_t idxc_dim = 0;   // indexer compressor output width
  size_t o_in_dim = 0;   // per-group o-proj input width
  size_t o_mid_dim = 0;  // grouped o-proj intermediate width
};

// RoPE timescales with YaRN frequency interpolation for DeepSeek V4
// compressed-path layers: freq_i is blended between freq_i / factor and
// freq_i with a linear ramp between the beta_fast and beta_slow correction
// dims. Matches `precompute_freqs_cis` in the reference implementation.
static inline MatStorageT<float> CreateYarnInvTimescale(
    const Allocator& allocator, size_t rope_dim, double base_frequency,
    double factor, size_t orig_seq_len, double beta_fast, double beta_slow) {
  const size_t total_entries = HWY_MAX(rope_dim, size_t{2}) / 2;
  MatStorageT<float> inv_timescale("inv_timescale_c", total_entries,
                                   allocator);
  float* data = inv_timescale.Row(0);
  const double dim = static_cast<double>(rope_dim);
  // find_correction_dim for the low/high rotation counts.
  const auto correction_dim = [&](double num_rotations) {
    return dim *
           std::log(static_cast<double>(orig_seq_len) /
                    (num_rotations * 2.0 * 3.14159265358979323846)) /
           (2.0 * std::log(base_frequency));
  };
  double low = 0.0, high = 0.0;
  if (orig_seq_len > 0) {
    low = std::floor(correction_dim(beta_fast));
    high = std::ceil(correction_dim(beta_slow));
    low = HWY_MAX(low, 0.0);
    high = HWY_MIN(high, dim - 1.0);
    if (low == high) high += 0.001;
  }
  for (size_t i = 0; i < total_entries; ++i) {
    const double freq =
        1.0 / std::pow(base_frequency,
                       static_cast<double>(2 * i) / static_cast<double>(dim));
    if (orig_seq_len > 0) {
      // smooth = 1 - clamp((i - low) / (high - low), 0, 1)
      double ramp = (static_cast<double>(i) - low) / (high - low);
      ramp = HWY_MIN(HWY_MAX(ramp, 0.0), 1.0);
      const double smooth = 1.0 - ramp;
      data[i] =
          static_cast<float>(freq / factor * (1.0 - smooth) + freq * smooth);
    } else {
      data[i] = static_cast<float>(freq);
    }
  }
  return inv_timescale;
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_DEEPSEEK_DEEPSEEK_DIMS_H_
