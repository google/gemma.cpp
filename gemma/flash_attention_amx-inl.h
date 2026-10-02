// Copyright 2026 Google LLC
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

// ============================================================================
// Notice: Intel AMX FlashAttention Implementation
// WARNING: This implementation is highly experimental and subject to change.
// ============================================================================

// Include guard for non-SIMD code.
#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_ATTENTION_AMX_INL_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_ATTENTION_AMX_INL_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <limits>

#include "gemma/flash_attention.h"
#include "gemma/kv_cache.h"
#include "util/basics.h"
#include "hwy/base.h"
#include "hwy/cache_control.h"
#include "hwy/targets.h"

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_ATTENTION_AMX_INL_H_

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_ATTENTION_AMX_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_ATTENTION_AMX_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_ATTENTION_AMX_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_ATTENTION_AMX_TOGGLE
#endif

#include "compression/compress-inl.h"
#include "ops/ops-inl.h"
#include "hwy/contrib/math/fast_math-inl.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// ============================================================================
// Fixed AMX tile geometry (hardware-defined)
// ============================================================================
// These are not tuning knobs: a TMUL tile is 16 rows x 64 bytes, and the
// kernel hardcodes the resulting 2x2 tile grid (32x32 blocks and the `+ 16`
// offsets of each upper half) throughout. Changing them requires rewriting the
// kernel, not just these values.
constexpr size_t kAmxTileTokens = 32;  // Hardware KV tile size (32 tokens)
constexpr size_t kAmxQueriesPerBlock =
    32;                             // Queries processed per AMX block (2x16)
constexpr size_t kAmxDimStep = 32;  // 32 BF16 channels per AMX inner step

// A single TMUL tile is at most 16 rows x 64 bytes. For an f32 accumulator
// that is 16 rows x 16 floats; for a BF16 operand, 16 rows x 32 BF16. Each
// 32x32 block above is therefore covered by a 2x2 grid of tiles, and the
// halves are addressed with these two constants.
constexpr size_t kAmxTileRows = 16;     // Rows in one AMX tile
constexpr size_t kAmxTileColsF32 = 16;  // f32 per accumulator tile row

// The kernel steps `position` by kAmxTileTokens but derives the KV tile row
// index from KVCache::kTileSize; they must agree.
static_assert(kAmxTileTokens == KVCache::kTileSize,
              "AMX flash attention requires a 32-token KV cache tile");

#undef GEMMA_HAVE_AMX
#if HWY_ARCH_X86_64 && (HWY_TARGET <= HWY_AVX3_SPR) && \
    HWY_NATIVE_TILE_64B_MATMUL_BF16
#define GEMMA_HAVE_AMX 1
#else
#define GEMMA_HAVE_AMX 0
#endif

// Returns true if AMX-BF16 is compiled into this target and usable at runtime;
// otherwise logs via HWY_WARN and returns false so callers can skip AMX tests.
HWY_INLINE bool HaveAmxBf16() {
#if GEMMA_HAVE_AMX
  if (!hwy::HaveTile64BMatMulBF16()) {
    HWY_WARN("CPU lacks usable AMX BF16; skipping AMX test.");
    return false;
  }
  return true;
#else
  HWY_WARN("AMX requires an x86-64 AVX3_SPR target; skipping AMX test.");
  return false;
#endif
}

#if HWY_ARCH_X86_64

// Gate on AVX3_SPR: it is the oldest Highway target whose CPUs can have AMX,
// and every target at or below it (AVX10_2, AVX3_SPR) has 512-bit vectors.
// Every helper below decomposes a 32-wide tile row into two 16-lane f32
// vectors and hardcodes the `+ 16` offset of the upper half, so a narrower
// vector would silently process only part of each row.
#if GEMMA_HAVE_AMX

static_assert(HWY_MAX_LANES_D(hn::ScalableTag<float>) == 16,
              "AMX flash attention requires 16-lane f32 vectors");

// Templated on T so that read-only callers can pass a `const float*` without
// casting away constness.
template <typename T>
HWY_INLINE T* GetAccumTilePtr(T* HWY_RESTRICT c_accum_base,
                              size_t num_ch_blocks, size_t q_block,
                              size_t ch_block, size_t row_in_block,
                              size_t col_in_block) {
  // Each block is [kAmxQueriesPerBlock rows][kAmxDimStep channels].
  return c_accum_base +
         ((q_block * num_ch_blocks + ch_block) * kAmxQueriesPerBlock +
          row_in_block) *
             kAmxDimStep +
         col_in_block;
}

HWY_INLINE void ComputeQKTileAMX_BF16(
    const BF16* HWY_RESTRICT q_base, const BF16* HWY_RESTRICT k_base,
    BF16* HWY_RESTRICT q_padded,
    float s_tile[kAmxQueriesPerBlock][kAmxTileTokens], size_t q_base_idx,
    size_t actual_q_in_block, size_t qkv_dim) {
  auto acc00 = hn::MakeTile64B();
  auto acc01 = hn::MakeTile64B();
  auto acc10 = hn::MakeTile64B();
  auto acc11 = hn::MakeTile64B();
  auto q0 = hn::MakeTile64B();
  auto q1 = hn::MakeTile64B();
  auto k0 = hn::MakeTile64B();
  auto k1 = hn::MakeTile64B();
  const hn::ScalableTag<float> df;

  hn::Tile64BZero(&acc00);
  hn::Tile64BZero(&acc01);
  hn::Tile64BZero(&acc10);
  hn::Tile64BZero(&acc11);

  for (size_t ch_base = 0; ch_base < qkv_dim; ch_base += kAmxDimStep) {
    if (actual_q_in_block == kAmxQueriesPerBlock) {
      const BF16* q0_ptr = q_base + (q_base_idx)*qkv_dim + ch_base;
      const BF16* q1_ptr =
          q_base + (q_base_idx + kAmxTileRows) * qkv_dim + ch_base;
      hn::Tile64BLoad(&q0, q0_ptr, qkv_dim * sizeof(BF16));
      hn::Tile64BLoad(&q1, q1_ptr, qkv_dim * sizeof(BF16));
    } else {
      // `q_padded` rows hold one AMX inner step worth of channels.
      constexpr size_t kQRowBytes = kAmxDimStep * sizeof(BF16);
      for (size_t r = 0; r < actual_q_in_block; ++r) {
        hwy::CopyBytes(q_base + (q_base_idx + r) * qkv_dim + ch_base,
                       &q_padded[r * kAmxDimStep], kQRowBytes);
      }
      for (size_t r = actual_q_in_block; r < kAmxQueriesPerBlock; ++r) {
        hwy::ZeroBytes(&q_padded[r * kAmxDimStep], kQRowBytes);
      }
      hn::Tile64BLoad(&q0, &q_padded[0], kQRowBytes);
      hn::Tile64BLoad(&q1, &q_padded[kAmxTileRows * kAmxDimStep], kQRowBytes);
    }

    // K is stored VNNI-interleaved: one B-tile row holds a pair of adjacent
    // channels for all kAmxTileTokens tokens, so tokens 16..31 begin
    // kAmxTileTokens BF16 into the row.
    constexpr size_t kKRowBytes = kAmxTileTokens * 2 * sizeof(BF16);
    const BF16* k0_ptr = k_base + ch_base * kAmxTileTokens;
    const BF16* k1_ptr = k0_ptr + kAmxTileTokens;
    hn::Tile64BLoad(&k0, k0_ptr, kKRowBytes);
    hn::Tile64BLoad(&k1, k1_ptr, kKRowBytes);

    hn::Tile64BMatMul(df, &acc00, &q0, &k0);
    hn::Tile64BMatMul(df, &acc01, &q0, &k1);
    hn::Tile64BMatMul(df, &acc10, &q1, &k0);
    hn::Tile64BMatMul(df, &acc11, &q1, &k1);
  }

  // `s_tile` rows are queries, columns are tokens.
  constexpr size_t kSRowBytes = kAmxTileTokens * sizeof(float);
  hn::Tile64BStore(&acc00, &s_tile[0][0], kSRowBytes);
  hn::Tile64BStore(&acc01, &s_tile[0][kAmxTileColsF32], kSRowBytes);
  hn::Tile64BStore(&acc10, &s_tile[kAmxTileRows][0], kSRowBytes);
  hn::Tile64BStore(&acc11, &s_tile[kAmxTileRows][kAmxTileColsF32], kSRowBytes);
  hn::Tile64BRelease();
}

// Returns true if any query row wrote probabilities into `p_bf16_scratch`.
// When false the P tile is all zeros and the caller can skip the PV multiply.
HWY_INLINE bool SoftmaxAndRescaleAMX_BF16(
    const float s_tile[kAmxQueriesPerBlock][kAmxTileTokens],
    uint16_t p_bf16_scratch[kAmxQueriesPerBlock][kAmxTileTokens],
    float* HWY_RESTRICT c_accum_base, float* HWY_RESTRICT exp_denominator_sums,
    float* HWY_RESTRICT max_logits, hwy::Span<const size_t> start_pos_per_query,
    hwy::Span<const size_t> last_pos_per_query, size_t q_base_idx,
    size_t q_block, size_t actual_q_in_block, size_t num_ch_blocks,
    size_t position, float att_cap, float one_over_cap) {
  hwy::ZeroBytes(p_bf16_scratch,
                 sizeof(uint16_t) * kAmxQueriesPerBlock * kAmxTileTokens);
  bool any_p_written = false;

  const hn::ScalableTag<float> df;
  const hn::ScalableTag<uint32_t> du;
  using DF = decltype(df);
  using VF = hn::Vec<DF>;
  using DU = decltype(du);
  using VU = hn::Vec<DU>;
  using dbf_half_t = hn::Half<hn::ScalableTag<BF16>>;
  const dbf_half_t dbf_half;

  // Loop-invariant across queries. Positions are compared in 32-bit lanes;
  // callers stay far below 2^32 tokens of context. `pos1` covers the upper
  // half of the token row, i.e. the second accumulator tile.
  const VU iota = hn::Iota(du, 0);
  const VU pos0 = hn::Add(hn::Set(du, static_cast<uint32_t>(position)), iota);
  const VU pos1 = hn::Add(
      hn::Set(du, static_cast<uint32_t>(position + kAmxTileColsF32)), iota);
  const VF cap_vec = hn::Set(df, att_cap);
  const VF one_over_cap_vec = hn::Set(df, one_over_cap);

  for (size_t r = 0; r < actual_q_in_block; ++r) {
    const size_t q = q_base_idx + r;
    if (position > last_pos_per_query[q] ||
        position + kAmxTileTokens <= start_pos_per_query[q]) {
      continue;
    }

    VF v0 = hn::LoadU(df, &s_tile[r][0]);
    VF v1 = hn::LoadU(df, &s_tile[r][kAmxTileColsF32]);

    if (att_cap > 0.0f) {
      v0 =
          hn::Mul(cap_vec, hn::CallFastTanh(df, hn::Mul(v0, one_over_cap_vec)));
      v1 =
          hn::Mul(cap_vec, hn::CallFastTanh(df, hn::Mul(v1, one_over_cap_vec)));
    }

    const VU start_vec =
        hn::Set(du, static_cast<uint32_t>(start_pos_per_query[q]));
    const VU last_vec =
        hn::Set(du, static_cast<uint32_t>(last_pos_per_query[q]));
    const hn::Mask<DU> valid0 =
        hn::And(hn::Ge(pos0, start_vec), hn::Le(pos0, last_vec));
    const hn::Mask<DU> valid1 =
        hn::And(hn::Ge(pos1, start_vec), hn::Le(pos1, last_vec));

    v0 = hn::IfThenElse(hn::RebindMask(df, valid0), v0,
                        hn::Set(df, kMaskedLogitVal));
    v1 = hn::IfThenElse(hn::RebindMask(df, valid1), v1,
                        hn::Set(df, kMaskedLogitVal));

    // A fully masked row has every lane at exactly kMaskedLogitVal; real
    // logits are many orders of magnitude above it. This mirrors the
    // `new_m > kMaskedLogitVal` guard in the reference kernel.
    const float block_max = hn::ReduceMax(df, hn::Max(v0, v1));
    if (block_max <= kMaskedLogitVal) {
      continue;
    }
    any_p_written = true;

    const float old_m = max_logits[q];
    const float new_m = std::max(old_m, block_max);
    const float old_sum = exp_denominator_sums[q];

    float exp_diff = 1.0f;
    if (old_m != new_m) {
      const hn::CappedTag<float, 1> d1;
      const hn::Vec<decltype(d1)> v_diff = hn::Set(d1, old_m - new_m);
      exp_diff = hn::GetLane(hn::FastExpMinusOrZero(d1, v_diff));
    }

    const VF exp0 = hn::FastExpMinusOrZero(df, hn::Sub(v0, hn::Set(df, new_m)));
    const VF exp1 = hn::FastExpMinusOrZero(df, hn::Sub(v1, hn::Set(df, new_m)));
    const float block_sum = hn::ReduceSum(df, hn::Add(exp0, exp1));

    const float new_sum = old_sum * exp_diff + block_sum;
    const float scale_old =
        (new_sum > 0.0f) ? (old_sum * exp_diff) / new_sum : 1.0f;
    const float scale_new = (new_sum > 0.0f) ? 1.0f / new_sum : 0.0f;

    max_logits[q] = new_m;
    exp_denominator_sums[q] = new_sum;

    const VF p0 = hn::Mul(exp0, hn::Set(df, scale_new));
    const VF p1 = hn::Mul(exp1, hn::Set(df, scale_new));
    const hn::Vec<dbf_half_t> bf0 = hn::DemoteTo(dbf_half, p0);
    const hn::Vec<dbf_half_t> bf1 = hn::DemoteTo(dbf_half, p1);
    hn::StoreU(bf0, dbf_half, reinterpret_cast<BF16*>(&p_bf16_scratch[r][0]));
    hn::StoreU(bf1, dbf_half,
               reinterpret_cast<BF16*>(&p_bf16_scratch[r][kAmxTileColsF32]));

    if (scale_old != 1.0f) {
      const VF s_old = hn::Set(df, scale_old);
      for (size_t ch_blk = 0; ch_blk < num_ch_blocks; ++ch_blk) {
        float* row_ptr =
            GetAccumTilePtr(c_accum_base, num_ch_blocks, q_block, ch_blk, r, 0);
        hn::StoreU(hn::Mul(hn::LoadU(df, row_ptr), s_old), df, row_ptr);
        hn::StoreU(hn::Mul(hn::LoadU(df, row_ptr + kAmxTileColsF32), s_old), df,
                   row_ptr + kAmxTileColsF32);
      }
    }
  }
  return any_p_written;
}

HWY_INLINE void ComputePVTileAMX_BF16(
    const uint16_t p_bf16_scratch[kAmxQueriesPerBlock][kAmxTileTokens],
    const BF16* HWY_RESTRICT v_base, float* HWY_RESTRICT c_accum_base,
    size_t q_block, size_t num_ch_blocks, size_t qkv_dim) {
  auto p0 = hn::MakeTile64B();
  auto p1 = hn::MakeTile64B();
  auto v0 = hn::MakeTile64B();
  auto v1 = hn::MakeTile64B();
  auto acc00 = hn::MakeTile64B();
  auto acc01 = hn::MakeTile64B();
  auto acc10 = hn::MakeTile64B();
  auto acc11 = hn::MakeTile64B();
  const hn::ScalableTag<float> df;

  // `p_bf16_scratch` rows are queries, columns are tokens.
  constexpr size_t kPRowBytes = kAmxTileTokens * sizeof(uint16_t);
  hn::Tile64BLoad(&p0, &p_bf16_scratch[0][0], kPRowBytes);
  hn::Tile64BLoad(&p1, &p_bf16_scratch[kAmxTileRows][0], kPRowBytes);

  // V is stored VNNI-interleaved over tokens: one B-tile row holds all qkv_dim
  // channels for a pair of adjacent tokens, hence the `* 2` channel offsets.
  const size_t v_row_bytes = 2 * qkv_dim * sizeof(BF16);

  // Accumulator rows span kAmxDimStep channels.
  constexpr size_t kAccumRowBytes = kAmxDimStep * sizeof(float);
  // Offsets of the lower-left / upper-right tiles within the 2x2 grid.
  constexpr size_t kAccumNextRows = kAmxTileRows * kAmxDimStep;

  for (size_t ch_blk = 0; ch_blk < num_ch_blocks; ++ch_blk) {
    const size_t db = ch_blk * kAmxDimStep;

    const BF16* v0_ptr = v_base + db * 2;
    const BF16* v1_ptr = v_base + (db + kAmxTileColsF32) * 2;
    hn::Tile64BLoad(&v0, v0_ptr, v_row_bytes);
    hn::Tile64BLoad(&v1, v1_ptr, v_row_bytes);

    float* accum_ptr =
        GetAccumTilePtr(c_accum_base, num_ch_blocks, q_block, ch_blk, 0, 0);
    hn::Tile64BLoad(&acc00, accum_ptr, kAccumRowBytes);
    hn::Tile64BLoad(&acc01, accum_ptr + kAmxTileColsF32, kAccumRowBytes);
    hn::Tile64BLoad(&acc10, accum_ptr + kAccumNextRows, kAccumRowBytes);
    hn::Tile64BLoad(&acc11, accum_ptr + kAccumNextRows + kAmxTileColsF32,
                    kAccumRowBytes);

    hn::Tile64BMatMul(df, &acc00, &p0, &v0);
    hn::Tile64BMatMul(df, &acc01, &p0, &v1);
    hn::Tile64BMatMul(df, &acc10, &p1, &v0);
    hn::Tile64BMatMul(df, &acc11, &p1, &v1);

    hn::Tile64BStore(&acc00, accum_ptr, kAccumRowBytes);
    hn::Tile64BStore(&acc01, accum_ptr + kAmxTileColsF32, kAccumRowBytes);
    hn::Tile64BStore(&acc10, accum_ptr + kAccumNextRows, kAccumRowBytes);
    hn::Tile64BStore(&acc11, accum_ptr + kAccumNextRows + kAmxTileColsF32,
                     kAccumRowBytes);
  }
  hn::Tile64BRelease();
}

HWY_INLINE void StoreAccumulatorsToOutput(
    const float* HWY_RESTRICT c_accum_base, size_t q_count,
    size_t num_ch_blocks, MatPtrT<float>& att_out) {
  const hn::ScalableTag<float> df;
  for (size_t q = 0; q < q_count; ++q) {
    const size_t q_blk = q / kAmxQueriesPerBlock;
    const size_t r = q % kAmxQueriesPerBlock;
    float* out = att_out.Row(q);
    for (size_t ch_blk = 0; ch_blk < num_ch_blocks; ++ch_blk) {
      const float* accum_chunk =
          GetAccumTilePtr(c_accum_base, num_ch_blocks, q_blk, ch_blk, r, 0);
      hn::StoreU(hn::LoadU(df, accum_chunk), df, out + ch_blk * kAmxDimStep);
      hn::StoreU(hn::LoadU(df, accum_chunk + kAmxTileColsF32), df,
                 out + ch_blk * kAmxDimStep + kAmxTileColsF32);
    }
  }
}

inline void TileFlashAttentionAMX_BF16_Impl(
    hwy::Span<const MatPtr> kvs, size_t q_count,
    const BF16* HWY_RESTRICT q_base,
    hwy::Span<const size_t> start_pos_per_query,
    hwy::Span<const size_t> last_pos_per_query, const float att_cap,
    MatPtrT<float>& att_out, float* HWY_RESTRICT exp_denominator_sums,
    float* HWY_RESTRICT max_logits) {
  if (q_count == 0 || kvs.empty()) return;

  const size_t qkv_dim = att_out.Cols();
  const float one_over_cap = att_cap > 0.0f ? 1.0f / att_cap : 0.0f;

  // The channel loops step by whole AMX tiles. A qkv_dim that is not a
  // multiple of kAmxDimStep would read past the K region in the QK phase and
  // leave the tail channels of `att_out` unwritten.
  HWY_DASSERT(qkv_dim % kAmxDimStep == 0);

  size_t largest_last_pos = 0;
  size_t smallest_start_pos = std::numeric_limits<size_t>::max();
  for (size_t q = 0; q < q_count; ++q) {
    largest_last_pos = std::max(largest_last_pos, last_pos_per_query[q]);
    smallest_start_pos = std::min(smallest_start_pos, start_pos_per_query[q]);
  }

  // Scratchpad buffers for AMX tile operations. Logits and probabilities are
  // [queries x tokens]; padded Q is [queries x channels-per-step].
  HWY_ALIGN float s_tile[kAmxQueriesPerBlock][kAmxTileTokens];
  HWY_ALIGN uint16_t p_bf16_scratch[kAmxQueriesPerBlock][kAmxTileTokens];
  HWY_ALIGN BF16 q_padded[kAmxQueriesPerBlock * kAmxDimStep];

  // Blocked FP32 accumulators: stored in blocks of kAmxQueriesPerBlock queries
  // x qkv_dim, where each block is arranged as (qkv_dim / kAmxDimStep) chunks
  // of [kAmxQueriesPerBlock queries x kAmxDimStep channels]. This layout
  // enables direct Tile64BLoad and Tile64BStore without intermediate
  // transposition.
  const size_t rounded_q = hwy::RoundUpTo(q_count, kAmxQueriesPerBlock);
  const size_t num_q_blocks = rounded_q / kAmxQueriesPerBlock;
  const size_t num_ch_blocks = qkv_dim / kAmxDimStep;
  hwy::AlignedVector<float> C_accumulators(
      num_q_blocks * num_ch_blocks * kAmxQueriesPerBlock * kAmxDimStep, 0.0f);

  size_t current_kv_idx = 0;
  size_t current_kv_start_offset = 0;
  // Skip whole tiles that precede every query's window, as the reference
  // kernel does. If every query is inactive (start_pos == SIZE_MAX sentinel)
  // this lands far above largest_last_pos and the loop below does not run.
  size_t position = smallest_start_pos - (smallest_start_pos % kAmxTileTokens);

  while (position <= largest_last_pos) {
    while (current_kv_idx + 1 < kvs.size() &&
           position - current_kv_start_offset >=
               kvs[current_kv_idx].Rows() * KVCache::kTileSize) {
      current_kv_start_offset +=
          kvs[current_kv_idx].Rows() * KVCache::kTileSize;
      current_kv_idx++;
    }

    const size_t s_idx =
        (position - current_kv_start_offset) / KVCache::kTileSize;
    if (s_idx >= kvs[current_kv_idx].Rows()) {
      // Unreachable: callers clamp last_pos_per_query to the allocated KV
      // extent, and `position` never exceeds largest_last_pos. Stop rather
      // than substitute a wrong tile, which masking would not filter out.
      HWY_DASSERT(false);
      break;
    }

    const BF16* tile_base =
        HWY_RCAST_ALIGNED(const BF16*, kvs[current_kv_idx].RowBytes(s_idx));
    const BF16* k_base = tile_base;
    const BF16* v_base = tile_base + qkv_dim * KVCache::kTileSize;

    // Touch the head of the next KV tile to start the hardware prefetcher on
    // it early; it streams the remainder. Note the stride is in BF16 elements,
    // so 32 elements is one 64-byte cache line.
    if (s_idx + 1 < kvs[current_kv_idx].Rows()) {
      const BF16* next_tile = HWY_RCAST_ALIGNED(
          const BF16*, kvs[current_kv_idx].RowBytes(s_idx + 1));
      hwy::Prefetch(next_tile);
      hwy::Prefetch(next_tile + 32);
    }

    for (size_t q_base_idx = 0; q_base_idx < q_count;
         q_base_idx += kAmxQueriesPerBlock) {
      const size_t q_block = q_base_idx / kAmxQueriesPerBlock;
      const size_t actual_q_in_block =
          std::min(kAmxQueriesPerBlock, q_count - q_base_idx);

      // Check if any query in this block attends to the current KV tile.
      bool any_active = false;
      for (size_t q_off = 0; q_off < actual_q_in_block; ++q_off) {
        const size_t q = q_base_idx + q_off;
        if (position <= last_pos_per_query[q] &&
            position + kAmxTileTokens > start_pos_per_query[q]) {
          any_active = true;
          break;
        }
      }
      if (!any_active) continue;

      // Phase 1: Compute Q * K^T via AMX
      ComputeQKTileAMX_BF16(q_base, k_base, q_padded, s_tile, q_base_idx,
                            actual_q_in_block, qkv_dim);

      // Phase 2: Softmax & Online Rescaling
      const bool any_p_written = SoftmaxAndRescaleAMX_BF16(
          s_tile, p_bf16_scratch, C_accumulators.data(), exp_denominator_sums,
          max_logits, start_pos_per_query, last_pos_per_query, q_base_idx,
          q_block, actual_q_in_block, num_ch_blocks, position, att_cap,
          one_over_cap);

      // Phase 3: P * V via AMX (Direct Tile Accumulation). Skipped when every
      // row was masked out, in which case P is all zeros.
      if (!any_p_written) continue;
      ComputePVTileAMX_BF16(p_bf16_scratch, v_base, C_accumulators.data(),
                            q_block, num_ch_blocks, qkv_dim);
    }

    position += kAmxTileTokens;
  }

  hn::Tile64BRelease();

  // Phase 4: Store final accumulated outputs to att_out
  StoreAccumulatorsToOutput(C_accumulators.data(), q_count, num_ch_blocks,
                            att_out);
}
#endif  // GEMMA_HAVE_AMX
#endif  // HWY_ARCH_X86_64

// Falls back to the generic BF16 implementation when the target was not built
// for AVX3_SPR+ or the CPU lacks AMX. Note this means AMX-specific tests
// silently exercise the BF16 path on hardware without AMX.
inline void TileFlashAttentionReturnExpSumsAndMaxLogitsAMX(
    hwy::Span<const MatPtr> kvs, size_t q_count,
    const BF16* HWY_RESTRICT q_base,
    hwy::Span<const size_t> start_pos_per_query,
    hwy::Span<const size_t> last_pos_per_query, const float att_cap,
    MatPtrT<float>& att_out, float* HWY_RESTRICT exp_denominator_sums,
    float* HWY_RESTRICT max_logits) {
#if GEMMA_HAVE_AMX
  if (hwy::HaveTile64BMatMulBF16()) {
    TileFlashAttentionAMX_BF16_Impl(kvs, q_count, q_base, start_pos_per_query,
                                    last_pos_per_query, att_cap, att_out,
                                    exp_denominator_sums, max_logits);
    return;
  }
#endif  // GEMMA_HAVE_AMX
  DispatchTileFlashAttentionReturnExpSumsAndMaxLogitsBF16(
      kvs, q_count, q_base, start_pos_per_query, last_pos_per_query, att_cap,
      att_out, exp_denominator_sums, max_logits,
      /*worker_workspace=*/nullptr);
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_ATTENTION_AMX_TOGGLE
