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

#undef GEMMA_HAVE_AMX_INT8
#if HWY_ARCH_X86_64 && (HWY_TARGET <= HWY_AVX3_SPR) && \
    HWY_NATIVE_TILE_64B_MATMUL_I8
#define GEMMA_HAVE_AMX_INT8 1
#else
#define GEMMA_HAVE_AMX_INT8 0
#endif

// Returns true if AMX-INT8 is compiled into this target and usable at runtime;
// otherwise logs via HWY_WARN and returns false so callers can skip AMX tests.
HWY_INLINE bool HaveAmxInt8() {
#if GEMMA_HAVE_AMX_INT8
  if (!hwy::HaveTile64BMatMulI8()) {
    HWY_WARN("CPU lacks usable AMX INT8; skipping AMX test.");
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
#if HWY_ARCH_X86_64 && (HWY_TARGET <= HWY_AVX3_SPR)

static_assert(HWY_MAX_LANES_D(hn::ScalableTag<float>) == 16,
              "AMX flash attention requires 16-lane f32 vectors");
// The softmax helpers keep small arrays of vectors (e.g. `v_vec[kNumVecs]`),
// which is only valid for fixed-size vector types, not sizeless SVE/RVV ones.
static_assert(!HWY_HAVE_SCALABLE,
              "AMX flash attention requires compile-time vector sizes");

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

// The BF16 kernel below delegates to Highway's AMX-BF16 wrappers, which are
// only available on a recent clang. The INT8 kernel further below likewise
// uses Highway's AMX-INT8 wrappers and is gated on GEMMA_HAVE_AMX_INT8.
#if GEMMA_HAVE_AMX

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
  // No Tile64BRelease here: this is inlined into the kernel loop, where the
  // compiler only reloads the tile config at function entry and after calls.
  // Releasing here would leave the next iteration's tile ops unconfigured
  // (#UD). The kernel releases once after its loop.
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
  // No Tile64BRelease here: this is inlined into the kernel loop, where the
  // compiler only reloads the tile config at function entry and after calls.
  // Releasing here would leave the next iteration's tile ops unconfigured
  // (#UD). The kernel releases once after its loop.
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

// The INT8 kernel below uses Highway's AMX-INT8 wrappers, whose `da`/`db` tag
// arguments select the signedness of the A and B tiles (and hence the
// instruction): int8 x int8 (TDPBSSD) for Q*K^T and uint8 x int8 (TDPBUSD) for
// P*V.
#if GEMMA_HAVE_AMX_INT8

// 64 INT8 channels per AMX inner step (one 64-byte tile row).
constexpr size_t kAmxInt8DimStep = 64;
// Maximum number of 32-query blocks grouped together to reuse L1D-resident K
// and V slices across queries.
constexpr size_t kMaxAmxInt8QBlocksPerGroup = 4;

// Computes the [kAmxQueriesPerBlock x kAmxTileTokens] int32 logits Q * K^T for
// one 32-query block against one KV sub-tile using a 2x2 outer product (4
// accumulators), writing into `s_out` with stride `s_row_stride_elems`.
// All non-AMX control flow (padding, prefetching) is kept outside the
// `Tile64BZero`..`Tile64BStore` live range so LLVM's `x86_amx` register
// allocator sees a straight-line single-block loop.
// This and the other INT8 tile helpers each use all 8 tile registers. They are
// not inlined because otherwise (e.g. with ThinLTO) the tile register
// allocator may see their live ranges together and run out of registers.
static HWY_NOINLINE void ComputeQKTileAMX_Int8(
    const int8_t* HWY_RESTRICT q_base, const int8_t* HWY_RESTRICT k_base,
    const int8_t* HWY_RESTRICT prefetch_v_ptr, int8_t* HWY_RESTRICT q_padded,
    int32_t* HWY_RESTRICT s_out, size_t s_row_stride_elems, size_t q_base_idx,
    size_t actual_q_in_block, size_t qkv_dim) {
  const int8_t* q_src = q_base + q_base_idx * qkv_dim;
  size_t q_stride = qkv_dim;
  if (actual_q_in_block < kAmxQueriesPerBlock) {
    for (size_t r = 0; r < actual_q_in_block; ++r) {
      hwy::CopyBytes(q_src + r * qkv_dim, &q_padded[r * qkv_dim], qkv_dim);
    }
    hwy::ZeroBytes(&q_padded[actual_q_in_block * qkv_dim],
                   (kAmxQueriesPerBlock - actual_q_in_block) * qkv_dim);
    q_src = q_padded;
  }

  auto acc00 = hn::MakeTile64B();
  auto acc01 = hn::MakeTile64B();
  auto acc10 = hn::MakeTile64B();
  auto acc11 = hn::MakeTile64B();
  auto q0 = hn::MakeTile64B();
  auto q1 = hn::MakeTile64B();
  auto k0 = hn::MakeTile64B();
  auto k1 = hn::MakeTile64B();
  const hn::ScalableTag<int32_t> di32;
  const hn::ScalableTag<int8_t> di8;

  hn::Tile64BZero(&acc00);
  hn::Tile64BZero(&acc01);
  hn::Tile64BZero(&acc10);
  hn::Tile64BZero(&acc11);

  HWY_UNROLL(1)
  for (size_t ch_base = 0; ch_base < qkv_dim; ch_base += kAmxInt8DimStep) {
    const int8_t* q0_ptr = q_src + ch_base;
    const int8_t* q1_ptr = q_src + kAmxTileRows * q_stride + ch_base;
    hn::Tile64BLoad(&q0, q0_ptr, q_stride);
    hn::Tile64BLoad(&q1, q1_ptr, q_stride);

    constexpr size_t kKRowBytes = kAmxTileTokens * 4;
    const int8_t* k0_ptr = k_base + ch_base * kAmxTileTokens;
    const int8_t* k1_ptr = k0_ptr + kKRowBytes / 2;
    hn::Tile64BLoad(&k0, k0_ptr, kKRowBytes);
    hn::Tile64BLoad(&k1, k1_ptr, kKRowBytes);

    hn::Tile64BMatMul(di32, di8, di8, &acc00, &q0, &k0);
    hn::Tile64BMatMul(di32, di8, di8, &acc01, &q0, &k1);
    hn::Tile64BMatMul(di32, di8, di8, &acc10, &q1, &k0);
    hn::Tile64BMatMul(di32, di8, di8, &acc11, &q1, &k1);
  }

  const size_t s_row_bytes = s_row_stride_elems * sizeof(int32_t);
  hn::Tile64BStore(&acc00, s_out, s_row_bytes);
  hn::Tile64BStore(&acc01, s_out + kAmxTileColsF32, s_row_bytes);
  hn::Tile64BStore(&acc10, s_out + kAmxTileRows * s_row_stride_elems,
                   s_row_bytes);
  hn::Tile64BStore(
      &acc11, s_out + kAmxTileRows * s_row_stride_elems + kAmxTileColsF32,
      s_row_bytes);

  if (prefetch_v_ptr != nullptr) {
    const uint8_t* vp = reinterpret_cast<const uint8_t*>(prefetch_v_ptr);
    const size_t v_bytes = qkv_dim * kAmxTileTokens;
    for (size_t off = 0; off < v_bytes; off += 64) {
      hwy::Prefetch(vp + off);
    }
  }
}

// Specialized QK kernel for `qkv_dim == 256`: pins all 256 INT8 channels of up
// to 16 queries into 4 TMM registers (`q_c0..q_c3`) once and streams all
// `n_valid` KV sub-tiles (`j = 0 .. n_valid - 1`) against the pinned Q tiles.
// Eliminates Q reloads inside the sub-tile loop so the 32 KB K working set
// stays 100% resident in the 48 KB L1D cache across multi-block query groups
// (1.21x speedup at M=128) and executes <=16-query half-blocks in half the time
// (1.73x speedup).
template <size_t kTiles>
HWY_NOINLINE void ComputeQKPinQ256AMX_Int8(
    const int8_t* HWY_RESTRICT q_base,
    const int8_t* const HWY_RESTRICT k_base[kTiles], size_t n_valid,
    int8_t* HWY_RESTRICT q_padded256, int32_t* HWY_RESTRICT s_out_half,
    size_t s_row_stride_elems, size_t q_half_idx, size_t actual_q_half) {
  constexpr size_t kFixedDim = 256;
  constexpr size_t kKRowBytes = kAmxTileTokens * 4;
  const size_t s_row_bytes = s_row_stride_elems * sizeof(int32_t);

  const int8_t* q_src = q_base + q_half_idx * kFixedDim;
  if (actual_q_half < kAmxTileRows) {
    for (size_t r = 0; r < actual_q_half; ++r) {
      hwy::CopyBytes(q_src + r * kFixedDim, &q_padded256[r * kFixedDim],
                     kFixedDim);
    }
    hwy::ZeroBytes(&q_padded256[actual_q_half * kFixedDim],
                   (kAmxTileRows - actual_q_half) * kFixedDim);
    q_src = q_padded256;
  }

  auto acc0 = hn::MakeTile64B();
  auto acc1 = hn::MakeTile64B();
  auto k0 = hn::MakeTile64B();
  auto k1 = hn::MakeTile64B();
  auto q_c0 = hn::MakeTile64B();
  auto q_c1 = hn::MakeTile64B();
  auto q_c2 = hn::MakeTile64B();
  auto q_c3 = hn::MakeTile64B();
  const hn::ScalableTag<int32_t> di32;
  const hn::ScalableTag<int8_t> di8;

  hn::Tile64BLoad(&q_c0, q_src + 0, kFixedDim);
  hn::Tile64BLoad(&q_c1, q_src + 64, kFixedDim);
  hn::Tile64BLoad(&q_c2, q_src + 128, kFixedDim);
  hn::Tile64BLoad(&q_c3, q_src + 192, kFixedDim);

  HWY_UNROLL(1)
  for (size_t j = 0; j < n_valid; ++j) {
    const int8_t* kb = k_base[j];
    hn::Tile64BZero(&acc0);
    hn::Tile64BZero(&acc1);

    hn::Tile64BLoad(&k0, kb + 0 * kAmxTileTokens, kKRowBytes);
    hn::Tile64BLoad(&k1, kb + 0 * kAmxTileTokens + kKRowBytes / 2, kKRowBytes);
    hn::Tile64BMatMul(di32, di8, di8, &acc0, &q_c0, &k0);
    hn::Tile64BMatMul(di32, di8, di8, &acc1, &q_c0, &k1);

    hn::Tile64BLoad(&k0, kb + 64 * kAmxTileTokens, kKRowBytes);
    hn::Tile64BLoad(&k1, kb + 64 * kAmxTileTokens + kKRowBytes / 2, kKRowBytes);
    hn::Tile64BMatMul(di32, di8, di8, &acc0, &q_c1, &k0);
    hn::Tile64BMatMul(di32, di8, di8, &acc1, &q_c1, &k1);

    hn::Tile64BLoad(&k0, kb + 128 * kAmxTileTokens, kKRowBytes);
    hn::Tile64BLoad(&k1, kb + 128 * kAmxTileTokens + kKRowBytes / 2,
                    kKRowBytes);
    hn::Tile64BMatMul(di32, di8, di8, &acc0, &q_c2, &k0);
    hn::Tile64BMatMul(di32, di8, di8, &acc1, &q_c2, &k1);

    hn::Tile64BLoad(&k0, kb + 192 * kAmxTileTokens, kKRowBytes);
    hn::Tile64BLoad(&k1, kb + 192 * kAmxTileTokens + kKRowBytes / 2,
                    kKRowBytes);
    hn::Tile64BMatMul(di32, di8, di8, &acc0, &q_c3, &k0);
    hn::Tile64BMatMul(di32, di8, di8, &acc1, &q_c3, &k1);

    int32_t* s_dst = s_out_half + j * kAmxTileTokens;
    hn::Tile64BStore(&acc0, s_dst, s_row_bytes);
    hn::Tile64BStore(&acc1, s_dst + kAmxTileColsF32, s_row_bytes);
  }
}

// Dequantizes the int32 logits, applies soft-capping, masking and the online
// softmax update, then quantizes the (V-scale weighted) probabilities of each
// query row to uint8 in `p_u8_scratch`, with per-row dequantization scale
// `eff_scales[r]` and deferred accumulator rescale factor `scales_old[r]`.
// Returns true if any query row wrote probabilities; when false the P tile is
// all zeros and the caller can skip the PV multiply.
template <size_t kTiles>
inline bool SoftmaxAndQuantizeAMX_Int8(
    const int32_t s_tile[kAmxQueriesPerBlock][kTiles * kAmxTileTokens],
    uint8_t p_u8_scratch[kAmxQueriesPerBlock][kTiles * kAmxTileTokens],
    float eff_scales[kAmxQueriesPerBlock],
    float scales_old[kAmxQueriesPerBlock],
    const BF16* const HWY_RESTRICT k_scales[kTiles],
    const BF16* const HWY_RESTRICT v_scales[kTiles], size_t n_valid,
    hwy::Span<const float> q_scales, float* HWY_RESTRICT c_accum_base,
    float* HWY_RESTRICT exp_denominator_sums, float* HWY_RESTRICT max_logits,
    hwy::Span<const size_t> start_pos_per_query,
    hwy::Span<const size_t> last_pos_per_query, size_t q_base_idx,
    size_t q_block, size_t actual_q_in_block, size_t num_ch_blocks,
    size_t position, float att_cap, float one_over_cap) {
  constexpr size_t kTokensPerStep = kTiles * kAmxTileTokens;
  constexpr size_t kNumVecs = 2 * kTiles;

  const hn::ScalableTag<float> df;
  const hn::ScalableTag<uint32_t> du;
  using DF = decltype(df);
  using DU = decltype(du);
  using VU = hn::Vec<DU>;
  using VF = hn::Vec<DF>;
  using DI32 = hn::Repartition<int32_t, DF>;
  const DI32 di32;
  using VI32 = hn::Vec<DI32>;
  using DI16 = hn::Repartition<int16_t, DF>;
  const DI16 di16;
  using DU8 = hn::Repartition<uint8_t, DI16>;
  using DU8_Half = hn::Half<DU8>;
  const DU8_Half du8_half;
  using dbf_half_t = hn::Half<hn::ScalableTag<BF16>>;
  const dbf_half_t dbf_half;

  VF k_scale_f[kNumVecs];
  VF v_scale_f[kNumVecs];
  for (size_t v = 0; v < kNumVecs; ++v) {
    const size_t j = v / 2;
    const size_t half = v % 2;
    if (j < n_valid) {
      k_scale_f[v] = hn::PromoteTo(
          df, hn::LoadU(dbf_half, k_scales[j] + half * kAmxTileColsF32));
      v_scale_f[v] = hn::PromoteTo(
          df, hn::LoadU(dbf_half, v_scales[j] + half * kAmxTileColsF32));
    } else {
      k_scale_f[v] = hn::Zero(df);
      v_scale_f[v] = hn::Zero(df);
    }
  }

  // Loop-invariant across queries. Token positions are formed per vector at
  // the point of use rather than stored in an array of vectors.
  const VU iota = hn::Iota(du, 0);
  const VF cap_vec = hn::Set(df, att_cap);
  const VF one_over_cap_vec = hn::Set(df, one_over_cap);

  bool any_p_written = false;
  bool any_rescaled = false;

  for (size_t r = 0; r < actual_q_in_block; ++r) {
    const size_t q = q_base_idx + r;
    const size_t start_pos = start_pos_per_query[q];
    const size_t last_pos = last_pos_per_query[q];
    if (position > last_pos || position + kTokensPerStep <= start_pos) {
      eff_scales[r] = 0.0f;
      scales_old[r] = 1.0f;
      continue;
    }

    const float q_s = q_scales.empty() ? 1.0f : q_scales[q];
    const VF q_s_vec = hn::Set(df, q_s);
    const bool full_unmasked =
        (n_valid == kTiles) && (position >= start_pos) &&
        (position + kTokensPerStep - 1 <= last_pos);

    VF v_vec[kNumVecs];
    if (full_unmasked) {
      if (att_cap > 0.0f) {
        const VF q_s_cap = hn::Mul(q_s_vec, one_over_cap_vec);
        for (size_t v = 0; v < kNumVecs; ++v) {
          const VI32 isum = hn::LoadU(di32, &s_tile[r][v * kAmxTileColsF32]);
          const VF scaled =
              hn::Mul(hn::Mul(hn::ConvertTo(df, isum), q_s_cap), k_scale_f[v]);
          v_vec[v] = hn::Mul(cap_vec, hn::CallFastTanh(df, scaled));
        }
      } else {
        for (size_t v = 0; v < kNumVecs; ++v) {
          const VI32 isum = hn::LoadU(di32, &s_tile[r][v * kAmxTileColsF32]);
          v_vec[v] =
              hn::Mul(hn::Mul(hn::ConvertTo(df, isum), q_s_vec), k_scale_f[v]);
        }
      }
    } else {
      const VU start_vec = hn::Set(du, static_cast<uint32_t>(start_pos));
      const VU last_vec = hn::Set(du, static_cast<uint32_t>(last_pos));
      for (size_t v = 0; v < kNumVecs; ++v) {
        if (v / 2 < n_valid) {
          const VI32 isum = hn::LoadU(di32, &s_tile[r][v * kAmxTileColsF32]);
          VF val =
              hn::Mul(hn::Mul(hn::ConvertTo(df, isum), q_s_vec), k_scale_f[v]);
          if (att_cap > 0.0f) {
            val = hn::Mul(cap_vec,
                          hn::CallFastTanh(df, hn::Mul(val, one_over_cap_vec)));
          }
          const uint32_t pos0 =
              static_cast<uint32_t>(position + v * kAmxTileColsF32);
          const VU pos = hn::Add(hn::Set(du, pos0), iota);
          const hn::Mask<DU> valid =
              hn::And(hn::Ge(pos, start_vec), hn::Le(pos, last_vec));
          v_vec[v] = hn::IfThenElse(hn::RebindMask(df, valid), val,
                                    hn::Set(df, kMaskedLogitVal));
        } else {
          v_vec[v] = hn::Set(df, kMaskedLogitVal);
        }
      }
    }

    VF max_val = v_vec[0];
    for (size_t v = 1; v < kNumVecs; ++v) {
      max_val = hn::Max(max_val, v_vec[v]);
    }
    // A fully masked row has every lane at exactly kMaskedLogitVal; real
    // logits are many orders of magnitude above it. This mirrors the
    // `new_m > kMaskedLogitVal` guard in the reference kernel.
    const float block_max = hn::ReduceMax(df, max_val);
    if (block_max <= kMaskedLogitVal) {
      eff_scales[r] = 0.0f;
      scales_old[r] = 1.0f;
      continue;
    }

    const float old_m = max_logits[q];
    const float new_m = std::max(old_m, block_max);
    const float old_sum = exp_denominator_sums[q];

    float exp_diff = 1.0f;
    if (old_m != new_m) {
      const hn::CappedTag<float, 1> d1;
      const hn::Vec<decltype(d1)> v_diff = hn::Set(d1, old_m - new_m);
      exp_diff = hn::GetLane(hn::FastExpMinusOrZero(d1, v_diff));
    }

    // Fuse exp computation, denominator sum, V-scale folding, and max tracking
    // in a single pass. Because `scale_new` is a positive per-row scalar, it
    // cancels out in `p_vec * (255 / max_p)` and only multiplies `eff_scales`.
    VF p_vec[kNumVecs];
    const VF new_m_vec = hn::Set(df, new_m);
    VF sum_vec = hn::Zero(df);
    VF max_p_vec = hn::Zero(df);
    for (size_t v = 0; v < kNumVecs; ++v) {
      const VF e = hn::FastExpMinusOrZero(df, hn::Sub(v_vec[v], new_m_vec));
      sum_vec = hn::Add(sum_vec, e);
      p_vec[v] = hn::Mul(e, v_scale_f[v]);
      max_p_vec = hn::Max(max_p_vec, p_vec[v]);
    }
    const float block_sum = hn::ReduceSum(df, sum_vec);
    const float max_u = hn::ReduceMax(df, max_p_vec);

    const float new_sum = old_sum * exp_diff + block_sum;
    const float scale_old =
        (new_sum > 0.0f) ? (old_sum * exp_diff) / new_sum : 1.0f;
    const float scale_new = (new_sum > 0.0f) ? 1.0f / new_sum : 0.0f;

    max_logits[q] = new_m;
    exp_denominator_sums[q] = new_sum;
    scales_old[r] = scale_old;
    if (scale_old != 1.0f) {
      any_rescaled = true;
    }

    const float max_p = max_u * scale_new;
    if (max_p > 1e-10f) {
      const float scale_to_quant = 255.0f / max_u;
      const VF s_q = hn::Set(df, scale_to_quant);
      for (size_t j = 0; j < n_valid; ++j) {
        VF qp0 = hn::Mul(p_vec[2 * j], s_q);
        VF qp1 = hn::Mul(p_vec[2 * j + 1], s_q);
        qp0 = hn::Min(hn::Max(qp0, hn::Zero(df)), hn::Set(df, 255.0f));
        qp1 = hn::Min(hn::Max(qp1, hn::Zero(df)), hn::Set(df, 255.0f));
        const hn::Vec<DI16> i16 = hn::OrderedDemote2To(
            di16, hn::NearestInt(qp0), hn::NearestInt(qp1));
        const hn::Vec<DU8_Half> u8 = hn::DemoteTo(du8_half, i16);
        hn::StoreU(u8, du8_half, &p_u8_scratch[r][j * kAmxTileTokens]);
      }
      eff_scales[r] = max_p / 255.0f;
      any_p_written = true;
    } else {
      eff_scales[r] = 0.0f;
    }
  }

  // If no row produced non-zero probabilities (so PV will be skipped), apply
  // any pending `scales_old` rescaling here. Otherwise `AccumulatePVTileInt8`
  // fuses `scales_old` with the PV accumulation in a single pass.
  if (!any_p_written && any_rescaled) {
    for (size_t r = 0; r < actual_q_in_block; ++r) {
      if (scales_old[r] != 1.0f) {
        const VF s_old = hn::Set(df, scales_old[r]);
        for (size_t ch_blk = 0; ch_blk < num_ch_blocks; ++ch_blk) {
          float* row_ptr = GetAccumTilePtr(c_accum_base, num_ch_blocks, q_block,
                                           ch_blk, r, 0);
          hn::StoreU(hn::Mul(hn::LoadU(df, row_ptr), s_old), df, row_ptr);
          hn::StoreU(hn::Mul(hn::LoadU(df, row_ptr + kAmxTileColsF32), s_old),
                     df, row_ptr + kAmxTileColsF32);
        }
      }
    }
  }
  return any_p_written;
}

// Dequantizes one channel block of int32 P * V results (buffer `buf_idx`) by
// the per-row `eff_scales` and fuses the `scales_old` accumulator rescaling.
HWY_INLINE void AccumulatePVTileInt8(
    const int32_t pv_tile[2][kAmxQueriesPerBlock][kAmxDimStep],
    const float eff_scales[kAmxQueriesPerBlock],
    const float scales_old[kAmxQueriesPerBlock],
    float* HWY_RESTRICT c_accum_base, size_t q_block, size_t blk_idx,
    int buf_idx, size_t actual_q_in_block, size_t num_ch_blocks) {
  const hn::ScalableTag<float> df;
  using DF = decltype(df);
  using VF = hn::Vec<DF>;
  using DI32 = hn::Repartition<int32_t, DF>;
  const DI32 di32;

  for (size_t r = 0; r < actual_q_in_block; ++r) {
    const float eff_s_val = eff_scales[r];
    const float s_old_val = scales_old[r];
    if (eff_s_val > 0.0f) {
      const VF eff_s = hn::Set(df, eff_s_val);
      const VF s_old = hn::Set(df, s_old_val);
      const VF pv0 =
          hn::ConvertTo(df, hn::LoadU(di32, &pv_tile[buf_idx][r][0]));
      const VF pv1 = hn::ConvertTo(
          df, hn::LoadU(di32, &pv_tile[buf_idx][r][kAmxTileColsF32]));

      float* row_ptr =
          GetAccumTilePtr(c_accum_base, num_ch_blocks, q_block, blk_idx, r, 0);
      VF acc0 = hn::Mul(hn::LoadU(df, row_ptr), s_old);
      VF acc1 = hn::Mul(hn::LoadU(df, row_ptr + kAmxTileColsF32), s_old);

      acc0 = hn::MulAdd(pv0, eff_s, acc0);
      acc1 = hn::MulAdd(pv1, eff_s, acc1);

      hn::StoreU(acc0, df, row_ptr);
      hn::StoreU(acc1, df, row_ptr + kAmxTileColsF32);
    } else if (s_old_val != 1.0f) {
      const VF s_old = hn::Set(df, s_old_val);
      float* row_ptr =
          GetAccumTilePtr(c_accum_base, num_ch_blocks, q_block, blk_idx, r, 0);
      hn::StoreU(hn::Mul(hn::LoadU(df, row_ptr), s_old), df, row_ptr);
      hn::StoreU(hn::Mul(hn::LoadU(df, row_ptr + kAmxTileColsF32), s_old), df,
                 row_ptr + kAmxTileColsF32);
    }
  }
}

// Computes P * V across a group of `q_group_blocks` query blocks and `kTiles`
// KV sub-tiles. Iterates `ch_blk` on the outside and `qb` on the inside so each
// 1 KB * n_valid slice of V stays pinned in the 48 KB L1D cache across all
// query blocks. Keeps the `Tile64BZero`..`Tile64BStore` region strictly
// branch-free so LLVM's `x86_amx` pass sees a single basic block.
template <size_t kTiles>
HWY_NOINLINE void ComputePVGroupAMX_Int8(
    const uint8_t p_u8_scratch[kMaxAmxInt8QBlocksPerGroup][kAmxQueriesPerBlock]
                              [kTiles * kAmxTileTokens],
    const float eff_scales[kMaxAmxInt8QBlocksPerGroup][kAmxQueriesPerBlock],
    const float scales_old[kMaxAmxInt8QBlocksPerGroup][kAmxQueriesPerBlock],
    const bool any_p_written[kMaxAmxInt8QBlocksPerGroup],
    const size_t actual_q_per_qb[kMaxAmxInt8QBlocksPerGroup],
    size_t q_block_base, size_t q_group_blocks,
    const int8_t* const HWY_RESTRICT v_base[kTiles], size_t n_valid,
    int32_t pv_tile[2][kAmxQueriesPerBlock][kAmxDimStep],
    float* HWY_RESTRICT c_accum_base, size_t num_ch_blocks, size_t qkv_dim) {
  auto p0 = hn::MakeTile64B(kAmxTileRows, kAmxTileTokens * sizeof(uint8_t));
  auto p1 = hn::MakeTile64B(kAmxTileRows, kAmxTileTokens * sizeof(uint8_t));
  auto v0 = hn::MakeTile64B(kAmxTileTokens / 4);
  auto v1 = hn::MakeTile64B(kAmxTileTokens / 4);
  auto acc00 = hn::MakeTile64B();
  auto acc01 = hn::MakeTile64B();
  auto acc10 = hn::MakeTile64B();
  auto acc11 = hn::MakeTile64B();
  const hn::ScalableTag<int32_t> di32;
  const hn::ScalableTag<uint8_t> du8;
  const hn::ScalableTag<int8_t> di8;

  constexpr size_t kPRowBytes = kTiles * kAmxTileTokens * sizeof(uint8_t);
  const size_t v_row_bytes = 4 * qkv_dim;
  constexpr size_t kPVRowBytes = kAmxDimStep * sizeof(int32_t);

  int step_count = 0;
  size_t prev_qb = 0;
  size_t prev_ch_blk = 0;
  size_t prev_actual_q = 0;

  HWY_UNROLL(1)
  for (size_t ch_blk = 0; ch_blk < num_ch_blocks; ++ch_blk) {
    const size_t db = ch_blk * kAmxDimStep;

    HWY_UNROLL(1)
    for (size_t qb = 0; qb < q_group_blocks; ++qb) {
      if (!any_p_written[qb]) continue;
      const size_t actual_q_in_block = actual_q_per_qb[qb];
      const int cur_buf = step_count % 2;

      hn::Tile64BZero(&acc00);
      hn::Tile64BZero(&acc01);
      hn::Tile64BZero(&acc10);
      hn::Tile64BZero(&acc11);

      HWY_UNROLL(1)
      for (size_t j = 0; j < n_valid; ++j) {
        hn::Tile64BLoad(&p0, &p_u8_scratch[qb][0][j * kAmxTileTokens],
                        kPRowBytes);
        hn::Tile64BLoad(&p1,
                        &p_u8_scratch[qb][kAmxTileRows][j * kAmxTileTokens],
                        kPRowBytes);
        const int8_t* v0_ptr = v_base[j] + db * 4;
        const int8_t* v1_ptr = v_base[j] + (db + kAmxTileColsF32) * 4;
        hn::Tile64BLoad(&v0, v0_ptr, v_row_bytes);
        hn::Tile64BLoad(&v1, v1_ptr, v_row_bytes);

        hn::Tile64BMatMul(di32, du8, di8, &acc00, &p0, &v0);
        hn::Tile64BMatMul(di32, du8, di8, &acc01, &p0, &v1);
        hn::Tile64BMatMul(di32, du8, di8, &acc10, &p1, &v0);
        hn::Tile64BMatMul(di32, du8, di8, &acc11, &p1, &v1);
      }

      hn::Tile64BStore(&acc00, &pv_tile[cur_buf][0][0], kPVRowBytes);
      hn::Tile64BStore(&acc01, &pv_tile[cur_buf][0][kAmxTileColsF32],
                       kPVRowBytes);
      hn::Tile64BStore(&acc10, &pv_tile[cur_buf][kAmxTileRows][0], kPVRowBytes);
      hn::Tile64BStore(&acc11,
                       &pv_tile[cur_buf][kAmxTileRows][kAmxTileColsF32],
                       kPVRowBytes);

      if (step_count > 0) {
        AccumulatePVTileInt8(pv_tile, eff_scales[prev_qb], scales_old[prev_qb],
                             c_accum_base, q_block_base + prev_qb, prev_ch_blk,
                             1 - cur_buf, prev_actual_q, num_ch_blocks);
      }

      prev_qb = qb;
      prev_ch_blk = ch_blk;
      prev_actual_q = actual_q_in_block;
      step_count++;
    }
  }

  if (step_count > 0) {
    AccumulatePVTileInt8(pv_tile, eff_scales[prev_qb], scales_old[prev_qb],
                         c_accum_base, q_block_base + prev_qb, prev_ch_blk,
                         (step_count - 1) % 2, prev_actual_q, num_ch_blocks);
  }
}

// Byte offsets of the scratch buffers used by
// `TileFlashAttentionAMX_Int8_ImplT`, so they can be carved out of a single
// per-worker allocation rather than placed on the stack or heap-allocated on
// every call. Mirrors `TileFlashAttentionWorkspaceLayout` in
// flash_attention.cc. Each buffer size is defined exactly once, here. The
// array typedefs let the carved pointers be indexed exactly like the
// equivalent multi-dimensional arrays.
template <size_t kTiles>
struct AmxInt8WorkspaceLayout {
  static constexpr size_t kTokensPerStep = kTiles * kAmxTileTokens;
  // Per query block: int32 Q * K^T logits and uint8 quantized probabilities.
  using STile = int32_t[kAmxQueriesPerBlock][kTokensPerStep];
  using PTile = uint8_t[kAmxQueriesPerBlock][kTokensPerStep];
  // Double-buffered int32 P * V results for one channel block.
  using PVTile = int32_t[kAmxQueriesPerBlock][kAmxDimStep];
  // Per query block: one scale per query row.
  using RowScales = float[kAmxQueriesPerBlock];

  size_t num_q_blocks, num_ch_blocks, c_accum_bytes;
  size_t s_tile_offset, p_u8_offset, pv_tile_offset, eff_scales_offset;
  size_t scales_old_offset, q_padded_offset, c_accum_offset;
  size_t total_bytes;

  constexpr AmxInt8WorkspaceLayout(size_t q_count, size_t qkv_dim)
      : num_q_blocks(hwy::DivCeil(q_count, kAmxQueriesPerBlock)),
        num_ch_blocks(qkv_dim / kAmxDimStep),
        // Blocked FP32 accumulators, same layout as the BF16 kernel.
        c_accum_bytes(num_q_blocks * num_ch_blocks * kAmxQueriesPerBlock *
                      kAmxDimStep * sizeof(float)),
        s_tile_offset(0),
        p_u8_offset(s_tile_offset +
                    Aligned(kMaxAmxInt8QBlocksPerGroup * sizeof(STile))),
        pv_tile_offset(p_u8_offset +
                       Aligned(kMaxAmxInt8QBlocksPerGroup * sizeof(PTile))),
        eff_scales_offset(pv_tile_offset + Aligned(2 * sizeof(PVTile))),
        scales_old_offset(
            eff_scales_offset +
            Aligned(kMaxAmxInt8QBlocksPerGroup * sizeof(RowScales))),
        q_padded_offset(
            scales_old_offset +
            Aligned(kMaxAmxInt8QBlocksPerGroup * sizeof(RowScales))),
        c_accum_offset(
            q_padded_offset +
            Aligned(kAmxQueriesPerBlock * qkv_dim * sizeof(int8_t))),
        total_bytes(c_accum_offset + Aligned(c_accum_bytes)) {}

 private:
  static constexpr size_t Aligned(size_t bytes) {
    return hwy::RoundUpTo(bytes, HWY_ALIGNMENT);
  }
};

template <size_t kTiles>
inline void TileFlashAttentionAMX_Int8_ImplT(
    hwy::Span<const MatPtr> kvs, size_t q_count,
    const int8_t* HWY_RESTRICT q_base, const hwy::Span<const float> q_scales,
    hwy::Span<const size_t> start_pos_per_query,
    hwy::Span<const size_t> last_pos_per_query, const float att_cap,
    MatPtrT<float>& att_out, float* HWY_RESTRICT exp_denominator_sums,
    float* HWY_RESTRICT max_logits,
    hwy::AlignedVector<uint8_t>* worker_workspace) {
  if (q_count == 0 || kvs.empty()) return;

  const size_t qkv_dim = att_out.Cols();
  const float one_over_cap = att_cap > 0.0f ? 1.0f / att_cap : 0.0f;

  HWY_DASSERT(qkv_dim % kAmxInt8DimStep == 0);

  size_t largest_last_pos = 0;
  size_t smallest_start_pos = std::numeric_limits<size_t>::max();
  for (size_t q = 0; q < q_count; ++q) {
    largest_last_pos = std::max(largest_last_pos, last_pos_per_query[q]);
    smallest_start_pos = std::min(smallest_start_pos, start_pos_per_query[q]);
  }

  using Layout = AmxInt8WorkspaceLayout<kTiles>;
  constexpr size_t kTokensPerStep = Layout::kTokensPerStep;
  const Layout layout(q_count, qkv_dim);
  const size_t num_q_blocks = layout.num_q_blocks;
  const size_t num_ch_blocks = layout.num_ch_blocks;

  // Use pre-allocated worker_workspace when available, resizing up if needed.
  hwy::AlignedFreeUniquePtr<uint8_t[]> workspace_fallback;
  uint8_t* raw_ptr = nullptr;
  if (worker_workspace != nullptr) {
    auto& ws = *worker_workspace;
    if (ws.size() < layout.total_bytes) {
      ws.resize(layout.total_bytes);
    }
    raw_ptr = ws.data();
  } else {
    workspace_fallback = hwy::AllocateAligned<uint8_t>(layout.total_bytes);
    raw_ptr = workspace_fallback.get();
  }

  // Scratchpad buffers for a group of up to `kMaxAmxInt8QBlocksPerGroup` query
  // blocks (128 queries), enabling L1D reuse of K and V across query blocks.
  // Only `C_accumulators` is read before being written, so it alone is zeroed.
  typename Layout::STile* s_tile = HWY_RCAST_ALIGNED(
      typename Layout::STile*, raw_ptr + layout.s_tile_offset);
  typename Layout::PTile* p_u8_scratch = HWY_RCAST_ALIGNED(
      typename Layout::PTile*, raw_ptr + layout.p_u8_offset);
  typename Layout::PVTile* pv_tile = HWY_RCAST_ALIGNED(
      typename Layout::PVTile*, raw_ptr + layout.pv_tile_offset);
  typename Layout::RowScales* eff_scales = HWY_RCAST_ALIGNED(
      typename Layout::RowScales*, raw_ptr + layout.eff_scales_offset);
  typename Layout::RowScales* scales_old = HWY_RCAST_ALIGNED(
      typename Layout::RowScales*, raw_ptr + layout.scales_old_offset);
  int8_t* q_padded =
      HWY_RCAST_ALIGNED(int8_t*, raw_ptr + layout.q_padded_offset);
  float* C_accumulators =
      HWY_RCAST_ALIGNED(float*, raw_ptr + layout.c_accum_offset);
  hwy::ZeroBytes(C_accumulators, layout.c_accum_bytes);

  size_t current_kv_idx = 0;
  size_t current_kv_start_offset = 0;
  // Skip whole tiles that precede every query's window. If every query is
  // inactive (start_pos == SIZE_MAX sentinel) this lands far above
  // largest_last_pos and the loop below does not run.
  size_t position = smallest_start_pos - (smallest_start_pos % kAmxTileTokens);

  while (position <= largest_last_pos) {
    const int8_t* k_base[kTiles];
    const int8_t* v_base[kTiles];
    const BF16* k_scales[kTiles];
    const BF16* v_scales[kTiles];
    size_t n_valid = 0;

    // Gather pointers to up to kTiles consecutive KV tiles for this step.
    for (size_t j = 0; j < kTiles; ++j) {
      const size_t position_j = position + j * kAmxTileTokens;
      // No query attends to this or any later tile.
      if (position_j > largest_last_pos) {
        break;
      }
      // The KV span may be split across several `kvs` segments; advance to
      // the segment containing position_j. Positions only increase, so the
      // segment cursor persists across tiles and steps.
      while (current_kv_idx + 1 < kvs.size() &&
             position_j - current_kv_start_offset >=
                 kvs[current_kv_idx].Rows() * KVCache::kTileSize) {
        current_kv_start_offset +=
            kvs[current_kv_idx].Rows() * KVCache::kTileSize;
        current_kv_idx++;
      }
      // Tile (row) index within the segment; stop past the end of the cache.
      const size_t s_idx =
          (position_j - current_kv_start_offset) / KVCache::kTileSize;
      if (s_idx >= kvs[current_kv_idx].Rows()) {
        break;
      }
      // Each tile row is laid out as [K int8 VNNI | V int8 VNNI |
      // K scales bf16 | V scales bf16 | ...], one scale per token.
      const int8_t* tile_base = HWY_RCAST_ALIGNED(
          const int8_t*, kvs[current_kv_idx].RowBytes(s_idx));
      k_base[j] = tile_base;
      v_base[j] = tile_base + qkv_dim * KVCache::kTileSize;
      k_scales[j] = reinterpret_cast<const BF16*>(
          tile_base + 2 * qkv_dim * KVCache::kTileSize);
      v_scales[j] = k_scales[j] + KVCache::kTileSize;
      n_valid++;
    }

    // Nothing left to process (past the last position or end of cache).
    if (n_valid == 0) {
      break;
    }

    HWY_UNROLL(1)
    for (size_t q_block_base = 0; q_block_base < num_q_blocks;
         q_block_base += kMaxAmxInt8QBlocksPerGroup) {
      const size_t q_group_blocks = std::min(kMaxAmxInt8QBlocksPerGroup,
                                             num_q_blocks - q_block_base);
      const bool is_last_q_group =
          (q_block_base + q_group_blocks == num_q_blocks);

      bool qb_active[kMaxAmxInt8QBlocksPerGroup] = {};
      bool any_p_written[kMaxAmxInt8QBlocksPerGroup] = {};
      size_t actual_q_per_qb[kMaxAmxInt8QBlocksPerGroup] = {};
      size_t active_qb_count = 0;
      size_t last_active_qb = 0;
      bool group_has_active = false;

      for (size_t qb = 0; qb < q_group_blocks; ++qb) {
        const size_t q_base_idx = (q_block_base + qb) * kAmxQueriesPerBlock;
        const size_t actual_q =
            std::min(kAmxQueriesPerBlock, q_count - q_base_idx);
        actual_q_per_qb[qb] = actual_q;
        for (size_t q_off = 0; q_off < actual_q; ++q_off) {
          const size_t q = q_base_idx + q_off;
          if (position <= last_pos_per_query[q] &&
              position + kTokensPerStep > start_pos_per_query[q]) {
            qb_active[qb] = true;
            last_active_qb = qb;
            active_qb_count++;
            group_has_active = true;
            break;
          }
        }
      }
      if (!group_has_active) continue;

      // Phase 1: Compute Q * K^T via AMX (int8 x int8).
      if (qkv_dim == 256 &&
          (active_qb_count >= 2 ||
           actual_q_per_qb[last_active_qb] <= kAmxTileRows)) {
        HWY_UNROLL(1)
        for (size_t qb_half = 0; qb_half < q_group_blocks * 2; ++qb_half) {
          const size_t qb = qb_half / 2;
          const size_t half = qb_half % 2;
          if (!qb_active[qb]) continue;
          const size_t actual_q = actual_q_per_qb[qb];
          if (half == 1 && actual_q <= kAmxTileRows) continue;
          const size_t actual_q_half =
              (half == 0) ? std::min(kAmxTileRows, actual_q)
                          : (actual_q - kAmxTileRows);
          const size_t q_half_idx =
              (q_block_base + qb) * kAmxQueriesPerBlock + half * kAmxTileRows;
          ComputeQKPinQ256AMX_Int8<kTiles>(
              q_base, k_base, n_valid, q_padded,
              &s_tile[qb][half * kAmxTileRows][0], kTokensPerStep, q_half_idx,
              actual_q_half);
        }
        if (is_last_q_group) {
          for (size_t j = 0; j < n_valid; ++j) {
            const uint8_t* vp = reinterpret_cast<const uint8_t*>(v_base[j]);
            for (size_t off = 0; off < 256 * kAmxTileTokens; off += 64) {
              hwy::Prefetch(vp + off);
            }
          }
        }
      } else {
        HWY_UNROLL(1)
        for (size_t j = 0; j < n_valid; ++j) {
          HWY_UNROLL(1)
          for (size_t qb = 0; qb < q_group_blocks; ++qb) {
            if (!qb_active[qb]) continue;
            const size_t q_base_idx = (q_block_base + qb) * kAmxQueriesPerBlock;
            const int8_t* prefetch_v =
                (is_last_q_group && qb == last_active_qb) ? v_base[j] : nullptr;
            ComputeQKTileAMX_Int8(q_base, k_base[j], prefetch_v, q_padded,
                                  &s_tile[qb][0][j * kAmxTileTokens],
                                  kTokensPerStep, q_base_idx,
                                  actual_q_per_qb[qb], qkv_dim);
          }
        }
      }

      // Phase 2: Softmax & quantize P to uint8 across all N sub-tiles for each
      // active query block in the group.
      bool group_any_p = false;
      for (size_t qb = 0; qb < q_group_blocks; ++qb) {
        if (!qb_active[qb]) continue;
        const size_t q_block = q_block_base + qb;
        const size_t q_base_idx = q_block * kAmxQueriesPerBlock;
        any_p_written[qb] = SoftmaxAndQuantizeAMX_Int8<kTiles>(
            s_tile[qb], p_u8_scratch[qb], eff_scales[qb], scales_old[qb],
            k_scales, v_scales, n_valid, q_scales, C_accumulators,
            exp_denominator_sums, max_logits, start_pos_per_query,
            last_pos_per_query, q_base_idx, q_block, actual_q_per_qb[qb],
            num_ch_blocks, position, att_cap, one_over_cap);
        group_any_p |= any_p_written[qb];
      }

      if (!group_any_p) continue;

      // Phase 3: P * V via AMX with per-j interleaved AVX-512 FP32
      // dequantization and rescaling across all active query blocks in the
      // group, reusing L1D-resident V slices.
      ComputePVGroupAMX_Int8<kTiles>(
          p_u8_scratch, eff_scales, scales_old, any_p_written, actual_q_per_qb,
          q_block_base, q_group_blocks, v_base, n_valid, pv_tile,
          C_accumulators, num_ch_blocks, qkv_dim);
    }

    position += kTokensPerStep;
  }

  hn::Tile64BRelease();

  // Phase 4: Store final accumulated outputs to att_out
  StoreAccumulatorsToOutput(C_accumulators, q_count, num_ch_blocks,
                            att_out);
}

// Default number of 32-token KV tiles processed per AMX INT8 macro-step.
constexpr size_t kDefaultAmxInt8KVTiles = 4;

inline void TileFlashAttentionAMX_Int8_Impl(
    hwy::Span<const MatPtr> kvs, size_t q_count,
    const int8_t* HWY_RESTRICT q_base, const hwy::Span<const float> q_scales,
    hwy::Span<const size_t> start_pos_per_query,
    hwy::Span<const size_t> last_pos_per_query, const float att_cap,
    MatPtrT<float>& att_out, float* HWY_RESTRICT exp_denominator_sums,
    float* HWY_RESTRICT max_logits,
    hwy::AlignedVector<uint8_t>* worker_workspace) {
  TileFlashAttentionAMX_Int8_ImplT<kDefaultAmxInt8KVTiles>(
      kvs, q_count, q_base, q_scales, start_pos_per_query,
      last_pos_per_query, att_cap, att_out, exp_denominator_sums,
      max_logits, worker_workspace);
}
#endif  // GEMMA_HAVE_AMX_INT8
#endif  // HWY_ARCH_X86_64 && (HWY_TARGET <= HWY_AVX3_SPR)
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

inline void TileFlashAttentionReturnExpSumsAndMaxLogitsAMXInt8(
    hwy::Span<const MatPtr> kvs, size_t q_count,
    const int8_t* HWY_RESTRICT q_base, const hwy::Span<const float> q_scales,
    hwy::Span<const size_t> start_pos_per_query,
    hwy::Span<const size_t> last_pos_per_query, const float att_cap,
    MatPtrT<float>& att_out, float* HWY_RESTRICT exp_denominator_sums,
    float* HWY_RESTRICT max_logits,
    hwy::AlignedVector<uint8_t>* worker_workspace) {
#if GEMMA_HAVE_AMX_INT8
  if (hwy::HaveTile64BMatMulI8()) {
    TileFlashAttentionAMX_Int8_Impl(
        kvs, q_count, q_base, q_scales, start_pos_per_query,
        last_pos_per_query, att_cap, att_out, exp_denominator_sums,
        max_logits, worker_workspace);
    return;
  }
#endif  // GEMMA_HAVE_AMX_INT8
  // Slow path for CPUs without AMX-INT8, only reached by direct callers
  // (TiledAttention calls the VNNI kernel itself when AmxInt8Available() is
  // false). The XOR converts signed queries to the +128-biased encoding
  // produced by CompressQueriesUint8 (same scale), and relies on the
  // kFlashAMXInt8 KV tile layout being identical to kFlashTransposedQsInt8
  // (including k_sums).
  const size_t qkv_dim = att_out.Cols();
  const size_t total_elements = q_count * qkv_dim;
  hwy::AlignedFreeUniquePtr<int8_t[]> biased_q =
      hwy::AllocateAligned<int8_t>(total_elements);

  const hn::ScalableTag<int8_t> di8;
  const size_t N = hn::Lanes(di8);
  const auto bias_vec = hn::Set(di8, static_cast<int8_t>(-128));
  size_t i = 0;
  for (; i + N <= total_elements; i += N) {
    const auto v = hn::LoadU(di8, q_base + i);
    hn::StoreU(hn::Xor(v, bias_vec), di8, biased_q.get() + i);
  }
  for (; i < total_elements; ++i) {
    biased_q[i] = static_cast<int8_t>(q_base[i] ^ 0x80);
  }

  DispatchTileFlashAttentionReturnExpSumsAndMaxLogitsInt8(
      kvs, q_count, biased_q.get(), q_scales, start_pos_per_query,
      last_pos_per_query, att_cap, att_out, exp_denominator_sums, max_logits,
      worker_workspace);
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_ATTENTION_AMX_TOGGLE
