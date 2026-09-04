// Include guard for non-SIMD code.
#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_ATTENTION_ARM_INL_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_ATTENTION_ARM_INL_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <vector>

#include "gemma/flash_attention.h"
#include "gemma/kv_cache.h"
#include "util/basics.h"
#include "util/threading_context.h"
#include "hwy/base.h"

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_ATTENTION_ARM_INL_H_

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_ATTENTION_ARM_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_ATTENTION_ARM_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_ATTENTION_ARM_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_ATTENTION_ARM_TOGGLE
#endif

#include "compression/compress-inl.h"
#include "ops/matmul-inl.h"
#include "ops/ops-inl.h"
#include "hwy/contrib/math/fast_math-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

constexpr size_t kArmBlockSizeBf16 = 128;
constexpr size_t kArmBlockSizeInt8 = 512;

struct TileAttentionGroupParams {
  size_t smallest_start_pos;
  size_t largest_last_pos;
  size_t num_loops;
  size_t position;
  float one_over_cap;
  float max_v_scale;
  hwy::AlignedVector<size_t> pos_data;
  hwy::Span<size_t> min_start_pos_per_group;
  hwy::Span<size_t> max_start_pos_per_group;
  hwy::Span<size_t> min_last_pos_per_group;
  hwy::Span<size_t> max_last_pos_per_group;
  std::vector<MatPtrT<float>> att_out_per_query;

  TileAttentionGroupParams(size_t q_count, size_t kNumQueriesPerLoop,
                           size_t step_size,
                           hwy::Span<const size_t> start_pos_per_query,
                           hwy::Span<const size_t> last_pos_per_query,
                           float att_cap, MatPtrT<float>& att_out,
                           bool create_att_out_per_query = false) {
    smallest_start_pos = std::numeric_limits<size_t>::max();
    largest_last_pos = std::numeric_limits<size_t>::min();
    for (size_t i = 0; i < start_pos_per_query.size(); ++i) {
      smallest_start_pos = std::min(smallest_start_pos, start_pos_per_query[i]);
      largest_last_pos = std::max(largest_last_pos, last_pos_per_query[i]);
    }
    num_loops = hwy::DivCeil(q_count, kNumQueriesPerLoop);

    pos_data.resize(num_loops * 4);
    min_start_pos_per_group = hwy::Span<size_t>(pos_data.data(), num_loops);
    max_start_pos_per_group =
        hwy::Span<size_t>(pos_data.data() + num_loops, num_loops);
    min_last_pos_per_group =
        hwy::Span<size_t>(pos_data.data() + 2 * num_loops, num_loops);
    max_last_pos_per_group =
        hwy::Span<size_t>(pos_data.data() + 3 * num_loops, num_loops);

    for (size_t i = 0; i < num_loops; ++i) {
      size_t min_start = std::numeric_limits<size_t>::max();
      size_t max_start = 0;
      size_t min_last = std::numeric_limits<size_t>::max();
      size_t max_last = 0;
      for (size_t j = 0; j < kNumQueriesPerLoop; ++j) {
        if (i * kNumQueriesPerLoop + j < q_count) {
          size_t q_idx = i * kNumQueriesPerLoop + j;
          min_start = std::min(min_start, start_pos_per_query[q_idx]);
          max_start = std::max(max_start, start_pos_per_query[q_idx]);
          min_last = std::min(min_last, last_pos_per_query[q_idx]);
          max_last = std::max(max_last, last_pos_per_query[q_idx]);
        }
      }
      min_start_pos_per_group[i] = min_start;
      max_start_pos_per_group[i] = max_start;
      min_last_pos_per_group[i] = min_last;
      max_last_pos_per_group[i] = max_last;
    }

    constexpr int kTileSize = gcpp::KVCache::kTileSize;
    const size_t base_pos =
        smallest_start_pos - (smallest_start_pos % kTileSize);
    const size_t rem = smallest_start_pos % kTileSize;
    const size_t num_skipped_sub_tiles = rem / step_size;
    position = base_pos + num_skipped_sub_tiles * step_size;
    one_over_cap = att_cap > 0.0f ? 1.0f / att_cap : 0.0f;
    max_v_scale = 1.0f;

    if (create_att_out_per_query) {
      const size_t qkv_dim = att_out.Cols();
      att_out_per_query.reserve(num_loops);
      for (size_t i = 0; i < num_loops; ++i) {
        att_out_per_query.emplace_back("att_out",
                                       Extents2D(kNumQueriesPerLoop, qkv_dim));
        att_out_per_query.back().SetPtr(att_out.Row(i * kNumQueriesPerLoop),
                                        att_out.Stride());
      }
    }
  }
};

template <size_t kRegBytes, class D, typename T>
HWY_INLINE hn::Vec<D> LoadAndDuplicateQueries(D d,
                                              const T* HWY_RESTRICT q_ptr) {
  if constexpr (kRegBytes <= 16) {
    return hn::LoadU(d, q_ptr);
  } else {
    return hn::LoadDup128(d, q_ptr);
  }
}

template <int kNumQueries, class D_ACC, class V_ACC, class V_IN, class LoadA,
          class V_A = decltype(std::declval<LoadA>()(0))>
HWY_INLINE void Accumulate4x4Grid(D_ACC d_acc, LoadA load_A, V_IN B0, V_IN B1,
                                  V_IN B2, V_IN B3, V_ACC& acc00, V_ACC& acc01,
                                  V_ACC& acc02, V_ACC& acc03, V_ACC& acc10,
                                  V_ACC& acc11, V_ACC& acc12, V_ACC& acc13,
                                  V_ACC& acc20, V_ACC& acc21, V_ACC& acc22,
                                  V_ACC& acc23, V_ACC& acc30, V_ACC& acc31,
                                  V_ACC& acc32, V_ACC& acc33) {
  if constexpr (kNumQueries >= 1) {
    const V_A A = load_A(0);
    acc00 = PerBlock2x2MatMulMaybeEmulate(d_acc, A, B0, acc00);
    acc01 = PerBlock2x2MatMulMaybeEmulate(d_acc, A, B1, acc01);
    acc02 = PerBlock2x2MatMulMaybeEmulate(d_acc, A, B2, acc02);
    acc03 = PerBlock2x2MatMulMaybeEmulate(d_acc, A, B3, acc03);
  }
  if constexpr (kNumQueries >= 3) {
    const V_A A = load_A(1);
    acc10 = PerBlock2x2MatMulMaybeEmulate(d_acc, A, B0, acc10);
    acc11 = PerBlock2x2MatMulMaybeEmulate(d_acc, A, B1, acc11);
    acc12 = PerBlock2x2MatMulMaybeEmulate(d_acc, A, B2, acc12);
    acc13 = PerBlock2x2MatMulMaybeEmulate(d_acc, A, B3, acc13);
  }
  if constexpr (kNumQueries >= 5) {
    const V_A A = load_A(2);
    acc20 = PerBlock2x2MatMulMaybeEmulate(d_acc, A, B0, acc20);
    acc21 = PerBlock2x2MatMulMaybeEmulate(d_acc, A, B1, acc21);
    acc22 = PerBlock2x2MatMulMaybeEmulate(d_acc, A, B2, acc22);
    acc23 = PerBlock2x2MatMulMaybeEmulate(d_acc, A, B3, acc23);
  }
  if constexpr (kNumQueries >= 7) {
    const V_A A = load_A(3);
    acc30 = PerBlock2x2MatMulMaybeEmulate(d_acc, A, B0, acc30);
    acc31 = PerBlock2x2MatMulMaybeEmulate(d_acc, A, B1, acc31);
    acc32 = PerBlock2x2MatMulMaybeEmulate(d_acc, A, B2, acc32);
    acc33 = PerBlock2x2MatMulMaybeEmulate(d_acc, A, B3, acc33);
  }
}

template <int kNumQueries, size_t kRegBytes, typename T_IN, typename DF,
          typename VF>
HWY_INLINE void QDotKTilexUpTo8MatrixAccumulation(
    DF df, const T_IN* HWY_RESTRICT q_group, const T_IN* tile_base,
    size_t current_pos, size_t qkv_dim, size_t stride_q, VF& C00, VF& C01,
    VF& C02, VF& C03, VF& C10, VF& C11, VF& C12, VF& C13, VF& C20, VF& C21,
    VF& C22, VF& C23, VF& C30, VF& C31, VF& C32, VF& C33) {
  using D_ACC = hwy::If<IsInt8<T_IN>(), hn::Repartition<int32_t, DF>, DF>;
  const D_ACC d_acc;
  using V_ACC = hn::Vec<D_ACC>;

  V_ACC acc00 = hn::Zero(d_acc), acc01 = hn::Zero(d_acc),
        acc02 = hn::Zero(d_acc), acc03 = hn::Zero(d_acc);
  V_ACC acc10 = hn::Zero(d_acc), acc11 = hn::Zero(d_acc),
        acc12 = hn::Zero(d_acc), acc13 = hn::Zero(d_acc);
  V_ACC acc20 = hn::Zero(d_acc), acc21 = hn::Zero(d_acc),
        acc22 = hn::Zero(d_acc), acc23 = hn::Zero(d_acc);
  V_ACC acc30 = hn::Zero(d_acc), acc31 = hn::Zero(d_acc),
        acc32 = hn::Zero(d_acc), acc33 = hn::Zero(d_acc);

  using D_INPUT = hn::ScalableTag<T_IN>;
  const D_INPUT d_input;
  using VecInput = hn::Vec<D_INPUT>;

  constexpr size_t kStepSize = kRegBytes / sizeof(T_IN);
  constexpr size_t step_size = kStepSize;
  HWY_LANES_CONSTEXPR size_t ch_step = IsInt8<T_IN>() ? 8 : 4;

  // Variables for BF16 path
  const T_IN* k_ptr0_bf16 = nullptr;
  const T_IN* k_ptr1_bf16 = nullptr;
  const T_IN* k_ptr2_bf16 = nullptr;
  const T_IN* k_ptr3_bf16 = nullptr;
  size_t ch_base_k = 0;
  size_t ch_base_q = 0;

  if constexpr (!IsInt8<T_IN>()) {
    if constexpr (kStepSize == 32) {
      k_ptr0_bf16 = tile_base + 0;
      k_ptr1_bf16 = tile_base + 32;
      k_ptr2_bf16 = tile_base + 64;
      k_ptr3_bf16 = tile_base + 96;
    } else if constexpr (kStepSize == 16) {
      const size_t g0 = (current_pos / 8) % 4;
      k_ptr0_bf16 = tile_base + g0 * 32;
      k_ptr1_bf16 = tile_base + (g0 + 1) * 32;
    } else {  // kStepSize == 8
      const size_t g0 = (current_pos / 8) % 4;
      k_ptr0_bf16 = tile_base + g0 * 32;
    }
  }

  for (size_t ch_base = 0; ch_base < qkv_dim; ch_base += ch_step) {
    VecInput B0, B1, B2, B3;

    if constexpr (IsInt8<T_IN>()) {
      // INT8 Path: 8x8 block layout
      const size_t token_in_tile = current_pos % 32;
      const T_IN* k_ptr = tile_base + ch_base * 32 + token_in_tile * 8;

      B0 = hn::LoadU(d_input, k_ptr + 0 * step_size);
      B1 = hn::LoadU(d_input, k_ptr + 1 * step_size);
      B2 = hn::LoadU(d_input, k_ptr + 2 * step_size);
      B3 = hn::LoadU(d_input, k_ptr + 3 * step_size);
    } else {
      // BF16 Path: original loading logic
      if constexpr (kStepSize == 32) {  // 512-bit native path
        B0 = hn::LoadU(d_input, k_ptr0_bf16 + ch_base_k);
        B1 = hn::LoadU(d_input, k_ptr1_bf16 + ch_base_k);
        B2 = hn::LoadU(d_input, k_ptr2_bf16 + ch_base_k);
        B3 = hn::LoadU(d_input, k_ptr3_bf16 + ch_base_k);
      } else if constexpr (kStepSize == 16) {  // 256-bit fallback path
        B0 = hn::LoadU(d_input, k_ptr0_bf16 + ch_base_k);
        B1 = hn::LoadU(d_input, k_ptr0_bf16 + ch_base_k + 16);
        B2 = hn::LoadU(d_input, k_ptr1_bf16 + ch_base_k);
        B3 = hn::LoadU(d_input, k_ptr1_bf16 + ch_base_k + 16);
      } else {  // kStepSize == 8 (128-bit fallback path)
        B0 = hn::LoadU(d_input, k_ptr0_bf16 + ch_base_k);
        B1 = hn::LoadU(d_input, k_ptr0_bf16 + ch_base_k + 8);
        B2 = hn::LoadU(d_input, k_ptr0_bf16 + ch_base_k + 16);
        B3 = hn::LoadU(d_input, k_ptr0_bf16 + ch_base_k + 24);
      }

      ch_base_k += ch_step * 32;
    }

    auto load_A = [&](size_t idx) HWY_ATTR {
      return LoadAndDuplicateQueries<kRegBytes>(
          d_input, q_group + idx * 2 * stride_q + ch_base_q);
    };

    Accumulate4x4Grid<kNumQueries>(
        d_acc, load_A, B0, B1, B2, B3, acc00, acc01, acc02, acc03, acc10, acc11,
        acc12, acc13, acc20, acc21, acc22, acc23, acc30, acc31, acc32, acc33);

    ch_base_q += ch_step * 2;
  }

  auto convert_and_reduce = [&](VF& C, V_ACC acc) HWY_ATTR {
    if constexpr (!IsInt8<T_IN>()) {
      C = acc;
    } else {
      C = hn::ConvertTo(df, acc);
    }
  };

  convert_and_reduce(C00, acc00);
  convert_and_reduce(C01, acc01);
  convert_and_reduce(C02, acc02);
  convert_and_reduce(C03, acc03);
  convert_and_reduce(C10, acc10);
  convert_and_reduce(C11, acc11);
  convert_and_reduce(C12, acc12);
  convert_and_reduce(C13, acc13);
  convert_and_reduce(C20, acc20);
  convert_and_reduce(C21, acc21);
  convert_and_reduce(C22, acc22);
  convert_and_reduce(C23, acc23);
  convert_and_reduce(C30, acc30);
  convert_and_reduce(C31, acc31);
  convert_and_reduce(C32, acc32);
  convert_and_reduce(C33, acc33);
}

// Symmetrical VLA Implementation
template <class DF, class V>
HWY_INLINE V ConcatLowerLower_VLA(DF df, V q1, V q0) {
  using D64 = hn::Repartition<uint64_t, DF>;
  const D64 d64;
  auto q0_64 = hn::BitCast(d64, q0);
  auto q1_64 = hn::BitCast(d64, q1);
  auto interleaved = hn::ConcatEven(d64, q1_64, q0_64);
  return hn::BitCast(df, interleaved);
}

template <class DF, class V>
HWY_INLINE V ConcatUpperUpper_VLA(DF df, V q1, V q0) {
  using D64 = hn::Repartition<uint64_t, DF>;
  const D64 d64;
  auto q0_64 = hn::BitCast(d64, q0);
  auto q1_64 = hn::BitCast(d64, q1);
  auto interleaved = hn::ConcatOdd(d64, q1_64, q0_64);
  return hn::BitCast(df, interleaved);
}

struct GroupInfo {
  const hwy::bfloat16_t* v_scales;
  const int8_t* v_tile_base;
  size_t token_in_tile;
};

template <class DF_T, class DBF_T>
HWY_INLINE void QuantizeAndPackSoftmaxProbs(
    DF_T df, DBF_T dbf, size_t q_base_idx, size_t actual_q_count,
    size_t actual_block_size, size_t qkv_dim,
    const float* HWY_RESTRICT q_scales_new,
    const float* HWY_RESTRICT softmax_buf, const GroupInfo* group_infos,
    uint8_t* HWY_RESTRICT q_weights_buf, float* HWY_RESTRICT w_scales_buf) {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::Full128<float> df_4;
  const hn::Full128<hwy::bfloat16_t> dbf8;
  const hn::Full128<int16_t> di16_8;
  const hn::Full128<uint8_t> du8_16;

  using VF4 = hn::Vec<decltype(df_4)>;
  using VBF8 = hn::Vec<decltype(dbf8)>;
  using VI16_8 = hn::Vec<decltype(di16_8)>;
  using VU8_16 = hn::Vec<decltype(du8_16)>;
  using VI32 = hn::Vec<hn::Repartition<int32_t, decltype(df_4)>>;

  const size_t num_groups = actual_block_size / 8;
  const size_t num_qp = 4;
  constexpr size_t kBlockSize = kArmBlockSizeInt8;

  // Process 2 queries at a time to reduce L1 working set and avoid large macros
  HWY_ALIGN float w_buf[2 * kBlockSize];

  for (size_t qp = 0; qp < num_qp; ++qp) {
    size_t q0 = qp * 2;
    size_t q1 = q0 + 1;

    VF4 max_val0 = hn::Zero(df_4);
    VF4 max_val1 = hn::Zero(df_4);

    float q0_scale = (q0 < actual_q_count) ? q_scales_new[q0] : 0.0f;
    float q1_scale = (q1 < actual_q_count) ? q_scales_new[q1] : 0.0f;
    const VF4 q0_scale_vec = hn::Set(df_4, q0_scale);
    const VF4 q1_scale_vec = hn::Set(df_4, q1_scale);

    // Fallback to safe memory block (multiplied by 0.0 later) to avoid
    // out-of-bounds branches
    const float* softmax_q0 = (q0 < actual_q_count)
                                  ? softmax_buf + (q_base_idx + q0) * kBlockSize
                                  : softmax_buf;
    const float* softmax_q1 = (q1 < actual_q_count)
                                  ? softmax_buf + (q_base_idx + q1) * kBlockSize
                                  : softmax_buf;

    // Pass 1: Compute final unquantized weights w = softmax * q_scale *
    // v_scale, and find max
    for (size_t g = 0; g < num_groups; ++g) {
      const auto& gi = group_infos[g];
      const VBF8 v_scale_bf16 = hn::LoadU(dbf8, gi.v_scales + gi.token_in_tile);
      const VF4 v_scale_f_lo = hn::PromoteLowerTo(df_4, v_scale_bf16);
      const VF4 v_scale_f_hi = hn::PromoteUpperTo(df_4, v_scale_bf16);

      // Query 0
      const VF4 s0_lo = hn::LoadU(df_4, softmax_q0 + g * 8);
      const VF4 s0_hi = hn::LoadU(df_4, softmax_q0 + g * 8 + 4);
      const VF4 qv0_lo = hn::Mul(q0_scale_vec, v_scale_f_lo);
      const VF4 qv0_hi = hn::Mul(q0_scale_vec, v_scale_f_hi);
      const VF4 w0_lo = hn::Mul(s0_lo, qv0_lo);
      const VF4 w0_hi = hn::Mul(s0_hi, qv0_hi);
      max_val0 = hn::Max(max_val0, hn::Max(w0_lo, w0_hi));
      hn::StoreU(w0_lo, df_4, w_buf + g * 8);
      hn::StoreU(w0_hi, df_4, w_buf + g * 8 + 4);

      // Query 1
      const VF4 s1_lo = hn::LoadU(df_4, softmax_q1 + g * 8);
      const VF4 s1_hi = hn::LoadU(df_4, softmax_q1 + g * 8 + 4);
      const VF4 qv1_lo = hn::Mul(q1_scale_vec, v_scale_f_lo);
      const VF4 qv1_hi = hn::Mul(q1_scale_vec, v_scale_f_hi);
      const VF4 w1_lo = hn::Mul(s1_lo, qv1_lo);
      const VF4 w1_hi = hn::Mul(s1_hi, qv1_hi);
      max_val1 = hn::Max(max_val1, hn::Max(w1_lo, w1_hi));
      hn::StoreU(w1_lo, df_4, w_buf + kBlockSize + g * 8);
      hn::StoreU(w1_hi, df_4, w_buf + kBlockSize + g * 8 + 4);
    }

    float global_max0 = hn::ReduceMax(df_4, max_val0);
    float global_max1 = hn::ReduceMax(df_4, max_val1);

    w_scales_buf[q0] = global_max0 / 255.0f;
    w_scales_buf[q1] = global_max1 / 255.0f;

    float inv_scale0 = (global_max0 > 1e-30f) ? 255.0f / global_max0 : 0.0f;
    float inv_scale1 = (global_max1 > 1e-30f) ? 255.0f / global_max1 : 0.0f;

    const VF4 inv_vec0 = hn::Set(df_4, inv_scale0);
    const VF4 inv_vec1 = hn::Set(df_4, inv_scale1);

    // Pass 2: Finalize quantization & packed formatting
    for (size_t g = 0; g < num_groups; ++g) {
      // Query 0
      const VF4 w0_lo = hn::LoadU(df_4, w_buf + g * 8);
      const VF4 w0_hi = hn::LoadU(df_4, w_buf + g * 8 + 4);
      const VI32 w0_lo_i32 = hn::NearestInt(hn::Mul(w0_lo, inv_vec0));
      const VI32 w0_hi_i32 = hn::NearestInt(hn::Mul(w0_hi, inv_vec0));
      const VI16_8 w0_i16 = hn::OrderedDemote2To(di16_8, w0_lo_i32, w0_hi_i32);

      // Query 1
      const VF4 w1_lo = hn::LoadU(df_4, w_buf + kBlockSize + g * 8);
      const VF4 w1_hi = hn::LoadU(df_4, w_buf + kBlockSize + g * 8 + 4);
      const VI32 w1_lo_i32 = hn::NearestInt(hn::Mul(w1_lo, inv_vec1));
      const VI32 w1_hi_i32 = hn::NearestInt(hn::Mul(w1_hi, inv_vec1));
      const VI16_8 w1_i16 = hn::OrderedDemote2To(di16_8, w1_lo_i32, w1_hi_i32);

      // Write tightly grouped chunk for the native HW path to ingest directly
      const VU8_16 w_u8_16 = hn::OrderedDemote2To(du8_16, w0_i16, w1_i16);
      uint8_t* dst = q_weights_buf + g * (num_qp * 16) + qp * 16;
      hn::StoreU(w_u8_16, du8_16, dst);
    }
  }
}

template <int kNumQueries>
HWY_INLINE void TileFlashAttentionSVBlockInt8(
    size_t q_base_idx, size_t qkv_dim, const float* HWY_RESTRICT scales_old,
    float* HWY_RESTRICT C_accumulators, const GroupInfo* group_infos,
    size_t num_groups, const uint8_t* HWY_RESTRICT q_weights_pre,
    const float* HWY_RESTRICT w_scales_pre) {
  namespace hn = hwy::HWY_NAMESPACE;
  using DI8 = hn::Full128<int8_t>;
  const DI8 di8;
  using VI8 = hn::Vec<DI8>;

  using DU8 = hn::Full128<uint8_t>;
  const DU8 du8;

  using DI32 = hn::Full128<int32_t>;
  const DI32 di32;
  using VI32 = hn::Vec<DI32>;

  using DF4 = hn::Full128<float>;
  const DF4 df4;
  using VF4 = hn::Vec<DF4>;

  // 1. Scale the old accumulator values in C_accumulators
  for (size_t q = 0; q < kNumQueries; ++q) {
    float* HWY_RESTRICT out = C_accumulators + (q_base_idx + q) * qkv_dim;
    const VF4 s = hn::Set(df4, scales_old[q]);
    for (size_t d = 0; d < qkv_dim; d += 4) {
      const VF4 old = hn::LoadU(df4, out + d);
      hn::StoreU(hn::Mul(old, s), df4, out + d);
    }
  }

  // 2. Loop over channel groups (each group has 8 channels)
  for (size_t ch_base = 0; ch_base < qkv_dim; ch_base += 8) {
    size_t ch_g = ch_base / 8;

    // Initialize accumulators: up to 4 query pairs x 4 channel pairs
    VI32 acc00 = hn::Zero(di32), acc01 = hn::Zero(di32), acc02 = hn::Zero(di32),
         acc03 = hn::Zero(di32);
    VI32 acc10 = hn::Zero(di32), acc11 = hn::Zero(di32), acc12 = hn::Zero(di32),
         acc13 = hn::Zero(di32);
    VI32 acc20 = hn::Zero(di32), acc21 = hn::Zero(di32), acc22 = hn::Zero(di32),
         acc23 = hn::Zero(di32);
    VI32 acc30 = hn::Zero(di32), acc31 = hn::Zero(di32), acc32 = hn::Zero(di32),
         acc33 = hn::Zero(di32);

    // 3. Accumulate over steps in int32
    for (size_t g = 0; g < num_groups; ++g) {
      const auto& gi = group_infos[g];
      size_t tg8 = gi.token_in_tile / 8;
      const int8_t* v_ptr = gi.v_tile_base + (ch_g * 4 + tg8) * 64;

      // Pre-load V values for the 4 channel pairs
      VI8 B0 = hn::LoadU(di8, v_ptr + 0 * 16);
      VI8 B1 = hn::LoadU(di8, v_ptr + 1 * 16);
      VI8 B2 = hn::LoadU(di8, v_ptr + 2 * 16);
      VI8 B3 = hn::LoadU(di8, v_ptr + 3 * 16);

      // Load Q weights and accumulate (reused across channel groups)
      // The writer (QuantizeAndPackSoftmaxProbs) always uses a stride of 64
      // bytes (4 query pairs) regardless of kNumQueries.
      const uint8_t* q_w_ptr = q_weights_pre + g * 64;

      auto load_A = [&](size_t idx)
                        HWY_ATTR { return hn::LoadU(du8, q_w_ptr + idx * 16); };

      Accumulate4x4Grid<kNumQueries>(di32, load_A, B0, B1, B2, B3, acc00, acc01,
                                     acc02, acc03, acc10, acc11, acc12, acc13,
                                     acc20, acc21, acc22, acc23, acc30, acc31,
                                     acc32, acc33);
    }

    // 4. Dequantize and write back
    auto dequant_and_store = [&](size_t qp, VI32 acc0, VI32 acc1, VI32 acc2,
                                 VI32 acc3) HWY_ATTR {
      size_t q0 = qp * 2;
      size_t q1 = q0 + 1;

      float sq0 = w_scales_pre[q0];
      float sq1 = (q1 < kNumQueries) ? w_scales_pre[q1] : 0.0f;

      alignas(16) float s_arr[4] = {sq0, sq0, sq1, sq1};
      VF4 scale_vec = hn::LoadU(df4, s_arr);

      VF4 C0_f = hn::Mul(hn::ConvertTo(df4, acc0), scale_vec);
      VF4 C1_f = hn::Mul(hn::ConvertTo(df4, acc1), scale_vec);
      VF4 C2_f = hn::Mul(hn::ConvertTo(df4, acc2), scale_vec);
      VF4 C3_f = hn::Mul(hn::ConvertTo(df4, acc3), scale_vec);

      // Reconstruct contiguous float channel vectors
      VF4 Q0_lo = ConcatLowerLower_VLA(df4, C1_f, C0_f);
      VF4 Q0_hi = ConcatLowerLower_VLA(df4, C3_f, C2_f);
      VF4 Q1_lo = ConcatUpperUpper_VLA(df4, C1_f, C0_f);
      VF4 Q1_hi = ConcatUpperUpper_VLA(df4, C3_f, C2_f);

      if (q0 < kNumQueries) {
        float* out = C_accumulators + (q_base_idx + q0) * qkv_dim + ch_base;
        hn::StoreU(hn::Add(hn::LoadU(df4, out + 0), Q0_lo), df4, out + 0);
        hn::StoreU(hn::Add(hn::LoadU(df4, out + 4), Q0_hi), df4, out + 4);
      }
      if (q1 < kNumQueries) {
        float* out = C_accumulators + (q_base_idx + q1) * qkv_dim + ch_base;
        hn::StoreU(hn::Add(hn::LoadU(df4, out + 0), Q1_lo), df4, out + 0);
        hn::StoreU(hn::Add(hn::LoadU(df4, out + 4), Q1_hi), df4, out + 4);
      }
    };

    if constexpr (kNumQueries >= 1) {
      dequant_and_store(0, acc00, acc01, acc02, acc03);
    }
    if constexpr (kNumQueries >= 3) {
      dequant_and_store(1, acc10, acc11, acc12, acc13);
    }
    if constexpr (kNumQueries >= 5) {
      dequant_and_store(2, acc20, acc21, acc22, acc23);
    }
    if constexpr (kNumQueries >= 7) {
      dequant_and_store(3, acc30, acc31, acc32, acc33);
    }
  }
}

template <int kNumQueries, size_t kRegBytes,
          typename DF_T = hn::ScalableTag<float>,
          typename DBF_T = hn::ScalableTag<hwy::bfloat16_t>, typename KV_T>
HWY_INLINE void TileFlashAttentionSVBlockBF16(
    size_t q_base_idx, size_t position, size_t current_kv_start_offset,
    size_t current_kv_idx, size_t actual_steps, size_t qkv_dim,
    const float* HWY_RESTRICT scales_old,
    const float* HWY_RESTRICT q_scales_new,
    const float* HWY_RESTRICT softmax_buf,
    const hwy::Span<const MatPtrT<KV_T>>& kvs,
    float* HWY_RESTRICT C_accumulators) {
  namespace hn = hwy::HWY_NAMESPACE;
  using BF16 = hwy::bfloat16_t;
  const DF_T df;
  const DBF_T dbf;
  const hn::FixedTag<float, 4> df4;
  using VBF = hn::Vec<DBF_T>;
  using VF = hn::Vec<DF_T>;
  using VF4 = hn::Vec<decltype(df4)>;

  constexpr size_t kBf16Lanes = kRegBytes / sizeof(BF16);
  constexpr size_t kBlockSize = kArmBlockSizeBf16;

  using KV_PTR_T = const KV_T*;
  // Dynamically scale the tile pointer buffer with the block size.
  constexpr size_t kMaxTiles =
      (kBlockSize + KVCache::kTileSize - 1) / KVCache::kTileSize;
  KV_PTR_T v_ptrs[kMaxTiles];
  size_t start_tile_idx =
      (position - current_kv_start_offset) / KVCache::kTileSize;

  // Optimized tile pointer loading
  size_t num_tiles_in_block =
      (actual_steps * kBf16Lanes + KVCache::kTileSize - 1) / KVCache::kTileSize;
  for (size_t t = 0; t < num_tiles_in_block; ++t) {
    const KV_T* tile_base = HWY_RCAST_ALIGNED(
        const KV_T*, kvs[current_kv_idx].RowBytes(start_tile_idx + t));
    v_ptrs[t] = tile_base + qkv_dim * 32;
  }

  // Pre-pack Q into BF16 to avoid scaling and demoting in the inner loop.
  constexpr size_t kNumPackedVectors = ((kNumQueries + 1) / 2) * 2;
  // Remove the redundant MaxLanes multiplier to reduce stack usage by up to
  // 32x.
  HWY_ALIGN BF16 q_packed[kBlockSize * kNumPackedVectors];

  for (size_t step_idx = 0; step_idx < actual_steps; ++step_idx) {
    auto load_scaled_q = [&](size_t qp, VF& v0, VF& v1) HWY_ATTR {
      const float* ptr =
          softmax_buf + (q_base_idx + qp) * kBlockSize + step_idx * kBf16Lanes;
      VF qs = hn::Set(df, q_scales_new[qp]);
      v0 = hn::Mul(hn::LoadU(df, ptr + 0), qs);
      v1 = hn::Mul(hn::LoadU(df, ptr + hn::Lanes(df)), qs);
    };

    VF q0_l = hn::Zero(df), q0_h = hn::Zero(df), q1_l = hn::Zero(df),
       q1_h = hn::Zero(df);
    VF q2_l = hn::Zero(df), q2_h = hn::Zero(df), q3_l = hn::Zero(df),
       q3_h = hn::Zero(df);
    VF q4_l = hn::Zero(df), q4_h = hn::Zero(df), q5_l = hn::Zero(df),
       q5_h = hn::Zero(df);
    VF q6_l = hn::Zero(df), q6_h = hn::Zero(df), q7_l = hn::Zero(df),
       q7_h = hn::Zero(df);

    if constexpr (kNumQueries >= 1) load_scaled_q(0, q0_l, q0_h);
    if constexpr (kNumQueries >= 2) load_scaled_q(1, q1_l, q1_h);
    if constexpr (kNumQueries >= 3) load_scaled_q(2, q2_l, q2_h);
    if constexpr (kNumQueries >= 4) load_scaled_q(3, q3_l, q3_h);
    if constexpr (kNumQueries >= 5) load_scaled_q(4, q4_l, q4_h);
    if constexpr (kNumQueries >= 6) load_scaled_q(5, q5_l, q5_h);
    if constexpr (kNumQueries >= 7) load_scaled_q(6, q6_l, q6_h);
    if constexpr (kNumQueries >= 8) load_scaled_q(7, q7_l, q7_h);

    auto pack_and_store_pair = [&](size_t pair_idx, VF ql0, VF qh0, VF ql1,
                                   VF qh1) HWY_ATTR {
      BF16* dst = q_packed + step_idx * kNumPackedVectors * kBf16Lanes +
                  pair_idx * 2 * kBf16Lanes;
      using D64 = hn::Repartition<uint64_t, DBF_T>;
      const D64 d64;
      using D64_half = hn::Half<D64>;
      const D64_half d64_half;
      using dbf_half_t = hn::Half<DBF_T>;
      const dbf_half_t dbf_half;

      auto ql0_bf = hn::DemoteTo(dbf_half, ql0);
      auto ql1_bf = hn::DemoteTo(dbf_half, ql1);

      auto qh0_bf = hn::DemoteTo(dbf_half, qh0);
      auto qh1_bf = hn::DemoteTo(dbf_half, qh1);

      hn::Vec<DBF_T> A0, A1;
      if constexpr (kRegBytes > 16) {
        auto ql0_64 = hn::BitCast(d64_half, ql0_bf);
        auto ql1_64 = hn::BitCast(d64_half, ql1_bf);
        // This interleaves within 128-bit block so it's fast.
        auto lo_l = hn::InterleaveLower(d64_half, ql0_64, ql1_64);
        auto hi_l = hn::InterleaveUpper(d64_half, ql0_64, ql1_64);
        A0 = hn::BitCast(dbf, hn::Combine(d64, hi_l, lo_l));

        auto qh0_64 = hn::BitCast(d64_half, qh0_bf);
        auto qh1_64 = hn::BitCast(d64_half, qh1_bf);
        auto lo_h = hn::InterleaveLower(d64_half, qh0_64, qh1_64);
        auto hi_h = hn::InterleaveUpper(d64_half, qh0_64, qh1_64);
        A1 = hn::BitCast(dbf, hn::Combine(d64, hi_h, lo_h));
      } else {
        A0 = hn::Combine(dbf, ql1_bf, ql0_bf);
        A1 = hn::Combine(dbf, qh1_bf, qh0_bf);
      }

      hn::StoreU(A0, dbf, dst + 0);
      hn::StoreU(A1, dbf, dst + kBf16Lanes);
    };

    if constexpr (kNumQueries >= 1)
      pack_and_store_pair(0, q0_l, q0_h, q1_l, q1_h);
    if constexpr (kNumQueries >= 3)
      pack_and_store_pair(1, q2_l, q2_h, q3_l, q3_h);
    if constexpr (kNumQueries >= 5)
      pack_and_store_pair(2, q4_l, q4_h, q5_l, q5_h);
    if constexpr (kNumQueries >= 7)
      pack_and_store_pair(3, q6_l, q6_h, q7_l, q7_h);
  }

  // Step-Dependent Pre-computation
  const BF16* step_q_ptrs[kBlockSize / 8];
  const BF16* step_v_tiles[kBlockSize / 8];
  size_t step_offsets_even[kBlockSize / 8];
  size_t step_offsets_odd[kBlockSize / 8];

  for (size_t step_idx = 0; step_idx < actual_steps; ++step_idx) {
    step_q_ptrs[step_idx] =
        q_packed + step_idx * kNumPackedVectors * kBf16Lanes;
    size_t step_pos = position + step_idx * kBf16Lanes;
    size_t global_token_pos = step_pos - current_kv_start_offset;
    size_t tile_idx = global_token_pos / 32 - start_tile_idx;
    step_v_tiles[step_idx] = HWY_RCAST_ALIGNED(const BF16*, v_ptrs[tile_idx]);
    size_t t = step_pos % 32;
    size_t t_odd = t + kBf16Lanes / 2;
    step_offsets_even[step_idx] = (t / 16) * 64 + ((t % 16) / 4) * 8;
    step_offsets_odd[step_idx] = (t_odd / 16) * 64 + ((t_odd % 16) / 4) * 8;
  }

  for (size_t ch_base = 0; ch_base < qkv_dim; ch_base += 8) {
    // Initialize accumulators to 0
    VF C00 = hn::Zero(df), C01 = hn::Zero(df), C02 = hn::Zero(df),
       C03 = hn::Zero(df);
    VF C10 = hn::Zero(df), C11 = hn::Zero(df), C12 = hn::Zero(df),
       C13 = hn::Zero(df);
    VF C20 = hn::Zero(df), C21 = hn::Zero(df), C22 = hn::Zero(df),
       C23 = hn::Zero(df);
    VF C30 = hn::Zero(df), C31 = hn::Zero(df), C32 = hn::Zero(df),
       C33 = hn::Zero(df);

    size_t ch_g = ch_base / 4;
    size_t v_offset = ch_g * 128;
    size_t v_next_offset = v_offset + 128;

    for (size_t step_idx = 0; step_idx < actual_steps; ++step_idx) {
      // Retrieve pre-computed values
      const BF16* q_ptr = step_q_ptrs[step_idx];
      const BF16* v_tile_b_curr = step_v_tiles[step_idx];
      size_t offset_even = step_offsets_even[step_idx];
      size_t offset_odd = step_offsets_odd[step_idx];

      const BF16* v_ptr = v_tile_b_curr + v_offset;
      const BF16* v_ptr_next = v_tile_b_curr + v_next_offset;

      VBF B_even0, B_even1, B_even2, B_even3;
      VBF B_odd0, B_odd1, B_odd2, B_odd3;

      B_even0 = hn::LoadU(dbf, v_ptr + offset_even);
      B_odd0 = hn::LoadU(dbf, v_ptr + offset_odd);
      B_even1 = hn::LoadU(dbf, v_ptr + offset_even + 32);
      B_odd1 = hn::LoadU(dbf, v_ptr + offset_odd + 32);

      B_even2 = hn::LoadU(dbf, v_ptr_next + offset_even);
      B_odd2 = hn::LoadU(dbf, v_ptr_next + offset_odd);
      B_even3 = hn::LoadU(dbf, v_ptr_next + offset_even + 32);
      B_odd3 = hn::LoadU(dbf, v_ptr_next + offset_odd + 32);

      // Even halves first (A0, A2, A4, A6)
      auto load_A_even = [&](size_t idx) HWY_ATTR {
        return hn::LoadU(dbf, q_ptr + 2 * idx * kBf16Lanes);
      };
      Accumulate4x4Grid<kNumQueries>(
          df, load_A_even, B_even0, B_even1, B_even2, B_even3, C00, C01, C02,
          C03, C10, C11, C12, C13, C20, C21, C22, C23, C30, C31, C32, C33);

      // Odd halves second (A1, A3, A5, A7)
      auto load_A_odd = [&](size_t idx) HWY_ATTR {
        return hn::LoadU(dbf, q_ptr + (2 * idx + 1) * kBf16Lanes);
      };
      Accumulate4x4Grid<kNumQueries>(
          df, load_A_odd, B_odd0, B_odd1, B_odd2, B_odd3, C00, C01, C02, C03,
          C10, C11, C12, C13, C20, C21, C22, C23, C30, C31, C32, C33);
    }

    // Reduce accumulators to 128-bit and add scaled old values
    auto reduce_and_store_pair = [&](size_t r, size_t col_g, VF acc0,
                                     VF acc1) HWY_ATTR {
      acc0 = SumReduceSegments(df, acc0);
      acc1 = SumReduceSegments(df, acc1);

      // Safe extraction of the lowest 128-bits (4 lanes) via aligned stack
      // spill/load to ensure cross-compilation safety on x86/AVX targets.
      HWY_ALIGN float temp0[hn::MaxLanes(df)];
      HWY_ALIGN float temp1[hn::MaxLanes(df)];
      hn::StoreU(acc0, df, temp0);
      hn::StoreU(acc1, df, temp1);
      VF4 acc0_red = hn::LoadU(df4, temp0);
      VF4 acc1_red = hn::LoadU(df4, temp1);

      // Load and scale old values (128-bit)
      float scale0 = scales_old[2 * r];
      float scale1 = (2 * r + 1 < kNumQueries) ? scales_old[2 * r + 1] : 1.0f;
      VF4 s0 = hn::Set(df4, scale0);
      VF4 s1 = hn::Set(df4, scale1);

      const float* row0 =
          C_accumulators + (q_base_idx + 2 * r) * qkv_dim + ch_base + col_g * 4;
      VF4 old0 = hn::Mul(hn::LoadU(df4, row0), s0);

      VF4 old1;
      if (2 * r + 1 < kNumQueries) {
        const float* row1 = C_accumulators +
                            (q_base_idx + 2 * r + 1) * qkv_dim + ch_base +
                            col_g * 4;
        old1 = hn::Mul(hn::LoadU(df4, row1), s1);
      } else {
        old1 = hn::Zero(df4);
      }

      // Interleave old values to match accumulator layout
      VF4 old_acc0 = hn::ConcatLowerLower(df4, old1, old0);
      VF4 old_acc1 = hn::ConcatUpperUpper(df4, old1, old0);

      // Add to reduced accumulators
      VF4 final_acc0 = hn::Add(acc0_red, old_acc0);
      VF4 final_acc1 = hn::Add(acc1_red, old_acc1);

      // Undo interleaving and store back
      float* out =
          C_accumulators + (q_base_idx + 2 * r) * qkv_dim + ch_base + col_g * 4;
      VF4 q0 = ConcatLowerLower_VLA(df4, final_acc1, final_acc0);
      hn::StoreU(q0, df4, out);

      VF4 q1;
      if (2 * r + 1 < kNumQueries) {
        float* out1 = C_accumulators + (q_base_idx + 2 * r + 1) * qkv_dim +
                      ch_base + col_g * 4;
        q1 = ConcatUpperUpper_VLA(df4, final_acc1, final_acc0);
        hn::StoreU(q1, df4, out1);
      } else {
        q1 = hn::Zero(df4);
      }
    };

    if constexpr (kNumQueries >= 1) {
      reduce_and_store_pair(0, 0, C00, C01);
      reduce_and_store_pair(0, 1, C02, C03);
    }
    if constexpr (kNumQueries >= 3) {
      reduce_and_store_pair(1, 0, C10, C11);
      reduce_and_store_pair(1, 1, C12, C13);
    }
    if constexpr (kNumQueries >= 5) {
      reduce_and_store_pair(2, 0, C20, C21);
      reduce_and_store_pair(2, 1, C22, C23);
    }
    if constexpr (kNumQueries >= 7) {
      reduce_and_store_pair(3, 0, C30, C31);
      reduce_and_store_pair(3, 1, C32, C33);
    }
  }
}

template <int kNumQueries, size_t kRegBytes,
          typename DF_T = hn::ScalableTag<float>,
          typename DBF_T = hn::ScalableTag<hwy::bfloat16_t>, typename KV_T>
HWY_INLINE void TileFlashAttentionSVBlock(
    size_t q_base_idx, size_t position, size_t current_kv_start_offset,
    size_t current_kv_idx, size_t actual_steps, size_t qkv_dim,
    const float* HWY_RESTRICT scales_old,
    const float* HWY_RESTRICT q_scales_new,
    const float* HWY_RESTRICT softmax_buf,
    const hwy::Span<const MatPtrT<KV_T>>& kvs,
    float* HWY_RESTRICT C_accumulators, const GroupInfo* group_infos = nullptr,
    size_t num_groups = 0, const uint8_t* HWY_RESTRICT q_weights_pre = nullptr,
    const float* HWY_RESTRICT w_scales_pre = nullptr) {
  if constexpr (IsInt8<KV_T>()) {
    TileFlashAttentionSVBlockInt8<kNumQueries>(
        q_base_idx, qkv_dim, scales_old, C_accumulators, group_infos,
        num_groups, q_weights_pre, w_scales_pre);
  } else {
    TileFlashAttentionSVBlockBF16<kNumQueries, kRegBytes, DF_T, DBF_T, KV_T>(
        q_base_idx, position, current_kv_start_offset, current_kv_idx,
        actual_steps, qkv_dim, scales_old, q_scales_new, softmax_buf, kvs,
        C_accumulators);
  }
}

template <class DF, class VF = hn::Vec<DF>>
HWY_INLINE void UpdateOnlineSoftmaxSingleQuery(
    DF df, float* HWY_RESTRICT q_logits, size_t actual_block_size,
    size_t q,  // global query index
    float* HWY_RESTRICT max_logits, float* HWY_RESTRICT exp_denominator_sums,
    size_t q_offset,  // local query index in the group (0..7)
    float* HWY_RESTRICT scales_old, float* HWY_RESTRICT q_scales_new) {
  float block_max = kMaskedLogitVal;
  VF v_max0 = hn::Set(df, block_max);
  VF v_max1 = v_max0;
  VF v_max2 = v_max0;
  VF v_max3 = v_max0;

  size_t t = 0;
  const size_t L_f = hn::Lanes(df);
  const size_t unroll_step = 4 * L_f;
  for (; t + unroll_step <= actual_block_size; t += unroll_step) {
    v_max0 = hn::Max(v_max0, hn::LoadU(df, q_logits + t));
    v_max1 = hn::Max(v_max1, hn::LoadU(df, q_logits + t + L_f));
    v_max2 = hn::Max(v_max2, hn::LoadU(df, q_logits + t + 2 * L_f));
    v_max3 = hn::Max(v_max3, hn::LoadU(df, q_logits + t + 3 * L_f));
  }
  for (; t + L_f <= actual_block_size; t += L_f) {
    v_max0 = hn::Max(v_max0, hn::LoadU(df, q_logits + t));
  }
  v_max0 = hn::Max(hn::Max(v_max0, v_max1), hn::Max(v_max2, v_max3));
  if (t < actual_block_size) {
    const size_t remaining = actual_block_size - t;
    VF v_logits = hn::LoadN(df, q_logits + t, remaining);
    auto mask = hn::FirstN(df, remaining);
    VF masked_logits =
        hn::IfThenElse(mask, v_logits, hn::Set(df, kMaskedLogitVal));
    v_max0 = hn::Max(v_max0, masked_logits);
  }
  block_max = std::max(block_max, hn::ReduceMax(df, v_max0));

  float old_m = max_logits[q];
  float old_sum = exp_denominator_sums[q];
  float new_m = std::max(old_m, block_max);

  float block_sum = 0.0f;
  VF v_sum0 = hn::Zero(df);
  VF v_sum1 = hn::Zero(df);
  VF v_sum2 = hn::Zero(df);
  VF v_sum3 = hn::Zero(df);
  VF v_new_m = hn::Set(df, new_m);

  t = 0;
  for (; t + unroll_step <= actual_block_size; t += unroll_step) {
    VF v_logits0 = hn::LoadU(df, q_logits + t);
    VF v_logits1 = hn::LoadU(df, q_logits + t + L_f);
    VF v_logits2 = hn::LoadU(df, q_logits + t + 2 * L_f);
    VF v_logits3 = hn::LoadU(df, q_logits + t + 3 * L_f);

    VF v_exp0 = hn::FastExpMinusOrZero(df, hn::Sub(v_logits0, v_new_m));
    VF v_exp1 = hn::FastExpMinusOrZero(df, hn::Sub(v_logits1, v_new_m));
    VF v_exp2 = hn::FastExpMinusOrZero(df, hn::Sub(v_logits2, v_new_m));
    VF v_exp3 = hn::FastExpMinusOrZero(df, hn::Sub(v_logits3, v_new_m));

    hn::StoreU(v_exp0, df, q_logits + t);
    hn::StoreU(v_exp1, df, q_logits + t + L_f);
    hn::StoreU(v_exp2, df, q_logits + t + 2 * L_f);
    hn::StoreU(v_exp3, df, q_logits + t + 3 * L_f);

    v_sum0 = hn::Add(v_sum0, v_exp0);
    v_sum1 = hn::Add(v_sum1, v_exp1);
    v_sum2 = hn::Add(v_sum2, v_exp2);
    v_sum3 = hn::Add(v_sum3, v_exp3);
  }
  for (; t + L_f <= actual_block_size; t += L_f) {
    VF v_logits = hn::LoadU(df, q_logits + t);
    VF v_exp = hn::FastExpMinusOrZero(df, hn::Sub(v_logits, v_new_m));
    hn::StoreU(v_exp, df, q_logits + t);
    v_sum0 = hn::Add(v_sum0, v_exp);
  }
  v_sum0 = hn::Add(hn::Add(v_sum0, v_sum1), hn::Add(v_sum2, v_sum3));
  if (t < actual_block_size) {
    const size_t remaining = actual_block_size - t;
    auto mask = hn::FirstN(df, remaining);
    VF v_logits = hn::LoadN(df, q_logits + t, remaining);
    VF v_exp = hn::FastExpMinusOrZero(df, hn::Sub(v_logits, v_new_m));
    hn::StoreN(v_exp, df, q_logits + t, remaining);
    v_sum0 = hn::Add(v_sum0, hn::IfThenElseZero(mask, v_exp));
  }
  block_sum = hn::ReduceSum(df, v_sum0);

  float exp_diff = 1.0f;
  if (old_m != new_m) {
    const hn::CappedTag<float, 1> d1;
    auto v_diff = hn::Set(d1, old_m - new_m);
    auto v_exp = hn::FastExpMinusOrZero(d1, v_diff);
    exp_diff = hn::GetLane(v_exp);
  }
  float new_sum = old_sum * exp_diff + block_sum;

  float scale_old = (new_sum > 0.0f) ? (old_sum * exp_diff) / new_sum : 1.0f;
  float q_scale = (new_sum > 0.0f) ? (1.0f / new_sum) : 0.0f;

  scales_old[q_offset] = scale_old;
  q_scales_new[q_offset] = q_scale;
  max_logits[q] = new_m;
  exp_denominator_sums[q] = new_sum;
}

template <size_t kRegBytes, class KV_T, class Q_T>
HWY_ATTR void TileFlashAttentionReturnExpSumsAndMaxLogitsBF16_Impl(
    const hwy::Span<const MatPtrT<KV_T>> kvs, size_t q_count,
    const Q_T* HWY_RESTRICT q_base, hwy::Span<const float> q_scales,
    hwy::Span<const size_t> start_pos_per_query,
    hwy::Span<const size_t> last_pos_per_query, float att_cap,
    MatPtrT<float>& att_out, float* HWY_RESTRICT exp_denominator_sums,
    float* HWY_RESTRICT max_logits) {
  using BF16 = hwy::bfloat16_t;

  const size_t qkv_dim = att_out.Cols();
  using DF = hn::ScalableTag<float>;
  using DBF = hn::ScalableTag<hwy::bfloat16_t>;
  using DU = hn::ScalableTag<uint32_t>;
  const DF df;
  const DBF dbf;
  const DU du;
  using VF = hn::Vec<DF>;
  constexpr size_t kBf16Lanes = kRegBytes / sizeof(BF16);
  const float one_over_cap = 1.0f / att_cap;

  constexpr int kNumQueriesPerLoop = 8;
  constexpr size_t kTileSize = 32;
  constexpr size_t kBlockSize =
      IsInt8<KV_T>() ? kArmBlockSizeInt8 : kArmBlockSizeBf16;
  const size_t kStepsPerTile =
      std::max(size_t(1), KVCache::kTileSize / kBf16Lanes);

  TileAttentionGroupParams preamble(q_count, kNumQueriesPerLoop, kBf16Lanes,
                                    start_pos_per_query, last_pos_per_query,
                                    att_cap, att_out);

  const size_t largest_last_pos = preamble.largest_last_pos;

  const auto min_start_pos_per_group = preamble.min_start_pos_per_group;
  const auto max_start_pos_per_group = preamble.max_start_pos_per_group;
  const auto min_last_pos_per_group = preamble.min_last_pos_per_group;
  const auto max_last_pos_per_group = preamble.max_last_pos_per_group;

  hwy::AlignedVector<float> C_accumulators(hwy::RoundUpTo(q_count, 8) * qkv_dim,
                                           0.0f);
  hwy::AlignedVector<float> softmax_buf(q_count * kBlockSize, kMaskedLogitVal);
  hwy::AlignedVector<uint8_t> q_weights_buf;
  hwy::AlignedVector<float> w_scales_buf;
  if constexpr (IsInt8<KV_T>()) {
    const size_t num_qp = 4;
    q_weights_buf.resize((kBlockSize / 8) * num_qp * 16, 0);
    w_scales_buf.resize(8, 0.0f);
  }

  size_t current_kv_idx = 0;
  size_t current_kv_start_offset = 0;
  size_t position = 0;

  while (position <= largest_last_pos) {
    std::fill(softmax_buf.begin(), softmax_buf.end(), kMaskedLogitVal);
    size_t remaining_tokens = largest_last_pos - position + 1;

    while (position - current_kv_start_offset >=
           kvs[current_kv_idx].Rows() * KVCache::kTileSize) {
      current_kv_start_offset +=
          kvs[current_kv_idx].Rows() * KVCache::kTileSize;
      current_kv_idx++;
    }
    size_t kv_rows = kvs[current_kv_idx].Rows() * KVCache::kTileSize;
    size_t kv_remaining = kv_rows - (position - current_kv_start_offset);

    size_t actual_block_size =
        std::min(kBlockSize, hwy::RoundUpTo(remaining_tokens, kBf16Lanes));
    actual_block_size = std::min(actual_block_size, kv_remaining);

    size_t actual_steps = actual_block_size / kBf16Lanes;
    const size_t num_groups = actual_block_size / 8;
    HWY_ALIGN GroupInfo group_infos[kBlockSize / 8];
    if constexpr (IsInt8<KV_T>()) {
      for (size_t g = 0; g < num_groups; ++g) {
        size_t global_token_pos = position + g * 8;
        size_t kv_ptr_idx = current_kv_idx;
        size_t kv_start_offset = current_kv_start_offset;
        while (global_token_pos - kv_start_offset >=
               kvs[kv_ptr_idx].Rows() * kTileSize) {
          kv_start_offset += kvs[kv_ptr_idx].Rows() * kTileSize;
          kv_ptr_idx++;
        }
        size_t tile_idx = (global_token_pos - kv_start_offset) / kTileSize;
        size_t token_in_tile = (global_token_pos - kv_start_offset) % kTileSize;

        const int8_t* tile_base = HWY_RCAST_ALIGNED(
            const int8_t*, kvs[kv_ptr_idx].RowBytes(tile_idx));
        const int8_t* v_tile_base = tile_base + qkv_dim * kTileSize;
        const BF16* scales =
            HWY_RCAST_ALIGNED(const BF16*, v_tile_base + qkv_dim * kTileSize);
        const BF16* v_scales = scales + kTileSize;

        group_infos[g] = {v_scales, v_tile_base, token_in_tile};
      }
    }

    size_t macro_tile_start_pos = position;

    auto inner_loop_qk = [&]<int kNumQueries>(size_t query_idx,
                                              size_t step_idx) HWY_ATTR {
      size_t loop_idx = query_idx / kNumQueriesPerLoop;
      size_t step_pos = position + step_idx * kBf16Lanes;
      if (step_pos + kBf16Lanes <= min_start_pos_per_group[loop_idx] ||
          step_pos > max_last_pos_per_group[loop_idx]) {
        float* softmax_buf_ptr =
            softmax_buf.data() + query_idx * kBlockSize + step_idx * kBf16Lanes;
        for (size_t q = 0; q < kNumQueries; ++q) {
          for (size_t t = 0; t < kBf16Lanes; ++t) {
            softmax_buf_ptr[q * kBlockSize + t] = kMaskedLogitVal;
          }
        }
        return;
      }

      size_t tile = step_idx / kStepsPerTile;
      size_t s_idx =
          (position - current_kv_start_offset) / KVCache::kTileSize + tile;

      const size_t current_pos = macro_tile_start_pos + step_idx * kBf16Lanes;
      const KV_T* tile_base =
          HWY_RCAST_ALIGNED(const KV_T*, kvs[current_kv_idx].RowBytes(s_idx));

      const Q_T* q_group = q_base + query_idx * qkv_dim;

      VF C00 = hn::Zero(df), C01 = hn::Zero(df), C02 = hn::Zero(df),
         C03 = hn::Zero(df);
      VF C10 = hn::Zero(df), C11 = hn::Zero(df), C12 = hn::Zero(df),
         C13 = hn::Zero(df);
      VF C20 = hn::Zero(df), C21 = hn::Zero(df), C22 = hn::Zero(df),
         C23 = hn::Zero(df);
      VF C30 = hn::Zero(df), C31 = hn::Zero(df), C32 = hn::Zero(df),
         C33 = hn::Zero(df);

      QDotKTilexUpTo8MatrixAccumulation<kNumQueries, kRegBytes>(
          df, q_group, tile_base, current_pos, qkv_dim, qkv_dim, C00, C01, C02,
          C03, C10, C11, C12, C13, C20, C21, C22, C23, C30, C31, C32, C33);

      VF x_0_p_0 = hn::Zero(df), x_0_p_1 = hn::Zero(df), x_1_p_0 = hn::Zero(df),
         x_1_p_1 = hn::Zero(df);
      VF x_2_p_0 = hn::Zero(df), x_2_p_1 = hn::Zero(df), x_3_p_0 = hn::Zero(df),
         x_3_p_1 = hn::Zero(df);
      VF x_4_p_0 = hn::Zero(df), x_4_p_1 = hn::Zero(df), x_5_p_0 = hn::Zero(df),
         x_5_p_1 = hn::Zero(df);
      VF x_6_p_0 = hn::Zero(df), x_6_p_1 = hn::Zero(df), x_7_p_0 = hn::Zero(df),
         x_7_p_1 = hn::Zero(df);

      auto pack_queries = [&](VF c_left, VF c_right, VF& x_even,
                              VF& x_odd) HWY_ATTR {
        using D64 = hn::Repartition<uint64_t, decltype(df)>;
        const D64 d64;

        auto c_left_64 = hn::BitCast(d64, c_left);
        auto c_right_64 = hn::BitCast(d64, c_right);

        auto even_scrambled = hn::ConcatEven(d64, c_right_64, c_left_64);
        auto odd_scrambled = hn::ConcatOdd(d64, c_right_64, c_left_64);

        x_even = hn::BitCast(df, even_scrambled);
        x_odd = hn::BitCast(df, odd_scrambled);
      };

      if constexpr (kNumQueries >= 1) {
        pack_queries(C00, C01, x_0_p_0, x_1_p_0);
        pack_queries(C02, C03, x_0_p_1, x_1_p_1);
      }
      if constexpr (kNumQueries >= 3) {
        pack_queries(C10, C11, x_2_p_0, x_3_p_0);
        pack_queries(C12, C13, x_2_p_1, x_3_p_1);
      }
      if constexpr (kNumQueries >= 5) {
        pack_queries(C20, C21, x_4_p_0, x_5_p_0);
        pack_queries(C22, C23, x_4_p_1, x_5_p_1);
      }
      if constexpr (kNumQueries >= 7) {
        pack_queries(C30, C31, x_6_p_0, x_7_p_0);
        pack_queries(C32, C33, x_6_p_1, x_7_p_1);
      }

      if constexpr (IsInt8<KV_T>()) {
        const size_t token_in_tile = current_pos % 32;
        const BF16* microscaling_scales_k =
            HWY_RCAST_ALIGNED(const BF16*,
                              tile_base + 2 * qkv_dim * kTileSize) +
            token_in_tile;

        const hn::Vec<DBF> v_sk_bf16 = hn::LoadU(dbf, microscaling_scales_k);
        const VF v_sk_f_lo = hn::PromoteLowerTo(df, v_sk_bf16);
        const VF v_sk_f_hi = hn::PromoteUpperTo(df, v_sk_bf16);

        auto apply_scales = [&](size_t r, VF& p_even_lo, VF& p_odd_lo,
                                VF& p_even_hi, VF& p_odd_hi) HWY_ATTR {
          float sq_ev = q_scales[query_idx + 2 * r];
          float sq_od = (2 * r + 1 < kNumQueries)
                            ? q_scales[query_idx + 2 * r + 1]
                            : 0.0f;

          const VF scale_ev_lo = hn::Mul(hn::Set(df, sq_ev), v_sk_f_lo);
          const VF scale_od_lo = hn::Mul(hn::Set(df, sq_od), v_sk_f_lo);
          const VF scale_ev_hi = hn::Mul(hn::Set(df, sq_ev), v_sk_f_hi);
          const VF scale_od_hi = hn::Mul(hn::Set(df, sq_od), v_sk_f_hi);

          p_even_lo = hn::Mul(p_even_lo, scale_ev_lo);
          p_odd_lo = hn::Mul(p_odd_lo, scale_od_lo);
          p_even_hi = hn::Mul(p_even_hi, scale_ev_hi);
          p_odd_hi = hn::Mul(p_odd_hi, scale_od_hi);
        };

        if constexpr (kNumQueries >= 1)
          apply_scales(0, x_0_p_0, x_1_p_0, x_0_p_1, x_1_p_1);
        if constexpr (kNumQueries >= 3)
          apply_scales(1, x_2_p_0, x_3_p_0, x_2_p_1, x_3_p_1);
        if constexpr (kNumQueries >= 5)
          apply_scales(2, x_4_p_0, x_5_p_0, x_4_p_1, x_5_p_1);
        if constexpr (kNumQueries >= 7)
          apply_scales(3, x_6_p_0, x_7_p_0, x_6_p_1, x_7_p_1);
      }

      constexpr int kFirstHalfAmountOfQueries = std::min(kNumQueries, 4);
      constexpr int kSecondHalfAmountOfQueries =
          kNumQueries - kFirstHalfAmountOfQueries;
      ApplySoftCap<kFirstHalfAmountOfQueries * 2>(
          df, att_cap, one_over_cap, x_0_p_0, x_0_p_1, x_1_p_0, x_1_p_1,
          x_2_p_0, x_2_p_1, x_3_p_0, x_3_p_1);
      if constexpr (kNumQueries > 4) {
        ApplySoftCap<kSecondHalfAmountOfQueries * 2>(
            df, att_cap, one_over_cap, x_4_p_0, x_4_p_1, x_5_p_0, x_5_p_1,
            x_6_p_0, x_6_p_1, x_7_p_0, x_7_p_1);
      }

      if (current_pos < max_start_pos_per_group[loop_idx] ||
          current_pos + kBf16Lanes - 1 > min_last_pos_per_group[loop_idx]) {
        ApplyMasking<kNumQueries>(
            df, du, current_pos, start_pos_per_query.data() + query_idx,
            last_pos_per_query.data() + query_idx, x_0_p_0, x_0_p_1, x_1_p_0,
            x_1_p_1, x_2_p_0, x_2_p_1, x_3_p_0, x_3_p_1, x_4_p_0, x_4_p_1,
            x_5_p_0, x_5_p_1, x_6_p_0, x_6_p_1, x_7_p_0, x_7_p_1);
      }

      float* softmax_buf_ptr =
          softmax_buf.data() + query_idx * kBlockSize + step_idx * kBf16Lanes;
      auto store_logits = [&](const VF& x_p0, const VF& x_p1,
                              size_t q) HWY_ATTR {
        hn::StoreU(x_p0, df, softmax_buf_ptr + q * kBlockSize + 0);
        hn::StoreU(x_p1, df, softmax_buf_ptr + q * kBlockSize + hn::Lanes(df));
      };

      if constexpr (kNumQueries >= 1) store_logits(x_0_p_0, x_0_p_1, 0);
      if constexpr (kNumQueries >= 2) store_logits(x_1_p_0, x_1_p_1, 1);
      if constexpr (kNumQueries >= 3) store_logits(x_2_p_0, x_2_p_1, 2);
      if constexpr (kNumQueries >= 4) store_logits(x_3_p_0, x_3_p_1, 3);
      if constexpr (kNumQueries >= 5) store_logits(x_4_p_0, x_4_p_1, 4);
      if constexpr (kNumQueries >= 6) store_logits(x_5_p_0, x_5_p_1, 5);
      if constexpr (kNumQueries >= 7) store_logits(x_6_p_0, x_6_p_1, 6);
      if constexpr (kNumQueries >= 8) store_logits(x_7_p_0, x_7_p_1, 7);
    };

    for (size_t step_idx = 0; step_idx < actual_steps; ++step_idx) {
      size_t query_idx = 0;
      for (; query_idx + kNumQueriesPerLoop <= q_count;
           query_idx += kNumQueriesPerLoop) {
        inner_loop_qk.template operator()<kNumQueriesPerLoop>(query_idx,
                                                              step_idx);
      }
      if (query_idx < q_count) {
        size_t rem = q_count - query_idx;
        if (rem >= 8) {
          inner_loop_qk.template operator()<8>(query_idx, step_idx);
          query_idx += 8;
          rem -= 8;
        }
        if (rem >= 4) {
          inner_loop_qk.template operator()<4>(query_idx, step_idx);
          query_idx += 4;
          rem -= 4;
        }
        switch (rem) {
          case 1:
            inner_loop_qk.template operator()<1>(query_idx, step_idx);
            break;
          case 2:
            inner_loop_qk.template operator()<2>(query_idx, step_idx);
            break;
          case 3:
            inner_loop_qk.template operator()<3>(query_idx, step_idx);
            break;
        }
      }
    }

    for (size_t q_base_idx = 0; q_base_idx < q_count; q_base_idx += 8) {
      size_t actual_q_count = std::min(size_t(8), q_count - q_base_idx);

      HWY_ALIGN float scales_old[8];
      HWY_ALIGN float q_scales_new[8];
      for (int i = 0; i < 8; ++i) {
        scales_old[i] = 1.0f;
        q_scales_new[i] = 0.0f;
      }

      for (size_t q_offset = 0; q_offset < actual_q_count; ++q_offset) {
        size_t q = q_base_idx + q_offset;
        if (position + actual_block_size <= start_pos_per_query[q] ||
            position > last_pos_per_query[q]) {
          continue;
        }
        float* q_logits = softmax_buf.data() + q * kBlockSize;
        UpdateOnlineSoftmaxSingleQuery<decltype(df)>(
            df, q_logits, actual_block_size, q, max_logits,
            exp_denominator_sums, q_offset, scales_old, q_scales_new);
      }
      if constexpr (IsInt8<KV_T>()) {
        QuantizeAndPackSoftmaxProbs(df, dbf, q_base_idx, actual_q_count,
                                    actual_block_size, qkv_dim, q_scales_new,
                                    softmax_buf.data(), group_infos,
                                    q_weights_buf.data(), w_scales_buf.data());
      }

      auto call_sv_block = [&]<int kNumQueries>() HWY_ATTR {
        const GroupInfo* gi_ptr = nullptr;
        size_t n_groups = 0;
        const uint8_t* qw_ptr = nullptr;
        const float* ws_ptr = nullptr;
        if constexpr (IsInt8<KV_T>()) {
          gi_ptr = group_infos;
          n_groups = num_groups;
          qw_ptr = q_weights_buf.data();
          ws_ptr = w_scales_buf.data();
        }
        TileFlashAttentionSVBlock<kNumQueries, kRegBytes, DF, DBF>(
            q_base_idx, position, current_kv_start_offset, current_kv_idx,
            actual_steps, qkv_dim, scales_old, q_scales_new, softmax_buf.data(),
            kvs, C_accumulators.data(), gi_ptr, n_groups, qw_ptr, ws_ptr);
      };

      if (actual_q_count >= 8) {
        call_sv_block.template operator()<8>();
      } else if (actual_q_count >= 7) {
        call_sv_block.template operator()<7>();
      } else if (actual_q_count >= 6) {
        call_sv_block.template operator()<6>();
      } else if (actual_q_count >= 5) {
        call_sv_block.template operator()<5>();
      } else if (actual_q_count >= 4) {
        call_sv_block.template operator()<4>();
      } else if (actual_q_count >= 3) {
        call_sv_block.template operator()<3>();
      } else if (actual_q_count >= 2) {
        call_sv_block.template operator()<2>();
      } else if (actual_q_count >= 1) {
        call_sv_block.template operator()<1>();
      }
    }

    position += actual_block_size;
  }

  for (size_t qi = 0; qi < q_count; ++qi) {
    float* out = att_out.Row(qi);
    const float* accum = C_accumulators.data() + qi * qkv_dim;
    for (size_t d = 0; d < qkv_dim; d += hn::Lanes(df)) {
      VF v = hn::LoadU(df, accum + d);
      hn::StoreU(v, df, out + d);
    }
  }
}

template <class KV_T, class Q_T>
HWY_ATTR void TileFlashAttentionReturnExpSumsAndMaxLogitsBF16(
    const hwy::Span<const MatPtrT<KV_T>> kvs, size_t q_count,
    const Q_T* HWY_RESTRICT q_base, hwy::Span<const float> q_scales,
    hwy::Span<const size_t> start_pos_per_query,
    hwy::Span<const size_t> last_pos_per_query, float att_cap,
    MatPtrT<float>& att_out, float* HWY_RESTRICT exp_denominator_sums,
    float* HWY_RESTRICT max_logits) {
#if HWY_HAVE_CONSTEXPR_LANES
  constexpr size_t kRegBytes = hn::Lanes(hn::ScalableTag<uint8_t>());
  TileFlashAttentionReturnExpSumsAndMaxLogitsBF16_Impl<kRegBytes, KV_T, Q_T>(
      kvs, q_count, q_base, q_scales, start_pos_per_query, last_pos_per_query,
      att_cap, att_out, exp_denominator_sums, max_logits);
#else
  const size_t reg_bytes = hn::Lanes(hn::ScalableTag<uint8_t>());
  if (reg_bytes == 64) {
    TileFlashAttentionReturnExpSumsAndMaxLogitsBF16_Impl<64, KV_T, Q_T>(
        kvs, q_count, q_base, q_scales, start_pos_per_query, last_pos_per_query,
        att_cap, att_out, exp_denominator_sums, max_logits);
  } else if (reg_bytes == 32) {
    TileFlashAttentionReturnExpSumsAndMaxLogitsBF16_Impl<32, KV_T, Q_T>(
        kvs, q_count, q_base, q_scales, start_pos_per_query, last_pos_per_query,
        att_cap, att_out, exp_denominator_sums, max_logits);
  } else if (reg_bytes == 16) {
    TileFlashAttentionReturnExpSumsAndMaxLogitsBF16_Impl<16, KV_T, Q_T>(
        kvs, q_count, q_base, q_scales, start_pos_per_query, last_pos_per_query,
        att_cap, att_out, exp_denominator_sums, max_logits);
  } else {
    HWY_ABORT("Unsupported register size %zu bytes", reg_bytes);
  }
#endif
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_ATTENTION_ARM_TOGGLE
