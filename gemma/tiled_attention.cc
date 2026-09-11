#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "compression/compress.h"
#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#include "gemma/configs.h"
#include "gemma/gemma.h"
#include "gemma/kv_cache.h"
#include "gemma/kv_transcoding.h"
#include "ops/matmul.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"

// Note: HWY_DISABLED_TARGETS needs to be defined the same everywhere.
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include "util/basics.h"
#include "util/mat.h"
#include "util/threading_context.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/tiled_attention.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "gemma/attention.h"
#include "gemma/flash_attention.h"  // includes highway.h
#include "gemma/gemma-inl.h"
#include "ops/ops-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

static HWY_INLINE void MergeOnlineSoftmax(
    const float* HWY_RESTRICT other_att_out, const float other_softmax_max,
    const float other_softmax_d, size_t qkv_dim,
    float* HWY_RESTRICT accumulator_att_out, float& accumulator_softmax_max,
    float& accumulator_softmax_d) {
  if (other_softmax_d == 0.0f) {
    return;
  }
  if (accumulator_softmax_d == 0.0f) {
    memcpy(accumulator_att_out, other_att_out,
           qkv_dim * sizeof(*accumulator_att_out));
    accumulator_softmax_max = other_softmax_max;
    accumulator_softmax_d = other_softmax_d;
    return;
  }
  const float m_new = std::max(accumulator_softmax_max, other_softmax_max);
  const float exp_l = std::exp(accumulator_softmax_max - m_new);
  const float exp_r = std::exp(other_softmax_max - m_new);
  const float d_new = accumulator_softmax_d * exp_l + other_softmax_d * exp_r;
  const float d_new_inv = 1.0f / d_new;
  const float c1 = accumulator_softmax_d * exp_l * d_new_inv;
  const float c2 = other_softmax_d * exp_r * d_new_inv;
  MulByConst(c1, accumulator_att_out, qkv_dim);
  MulByConstAndAdd(c2, other_att_out, accumulator_att_out, qkv_dim);
  accumulator_softmax_max = m_new;
  accumulator_softmax_d = d_new;
}

static constexpr size_t kMergeGroupSize = 32;

static HWY_INLINE void MergeOnlineSoftmaxGroup32(
    const size_t other_task_idx, const size_t accumulator_task_idx,
    const size_t group_start, const size_t group_count, const size_t qkv_dim,
    AttentionActivationsPtrs& activations) {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> df;
  HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(df);
  using VF = hn::Vec<decltype(df)>;

  const MatStorageT<float>& other_att_out =
      activations.sub_task_att_out->at(other_task_idx);
  const float* HWY_RESTRICT other_max_logits =
      activations.sub_task_max_logits->at(other_task_idx).data() + group_start;
  const float* HWY_RESTRICT other_exp_sums =
      activations.sub_task_exp_denominator_sums->at(other_task_idx).data() +
      group_start;

  MatStorageT<float>& acc_att_out =
      activations.sub_task_att_out->at(accumulator_task_idx);
  float* HWY_RESTRICT acc_max_logits =
      activations.sub_task_max_logits->at(accumulator_task_idx).data() +
      group_start;
  float* HWY_RESTRICT acc_exp_sums =
      activations.sub_task_exp_denominator_sums->at(accumulator_task_idx)
          .data() +
      group_start;

  HWY_ALIGN float c1_arr[kMergeGroupSize];
  HWY_ALIGN float c2_arr[kMergeGroupSize];
  const VF neg_inf = hn::Set(df, -std::numeric_limits<float>::max() / 2.0f);

  // Folds one vector of `other` lanes into the accumulator lanes, yielding the
  // updated running statistics and the two blend coefficients later applied to
  // the attention rows. A lane whose denominator is zero contributes nothing:
  // it is forced to -inf so it loses the max, and its exponential is zeroed so
  // it cannot perturb the sum. If both sides are empty the lane collapses to
  // (max = -inf, denominator = 0, c1 = c2 = 0), matching the sentinel that
  // `write_group_output` uses for a group with no context.
  const auto merge_lanes = [&](VF m_acc, VF d_acc, VF m_other, VF d_other,
                               VF& c1, VF& c2, VF& m_new, VF& d_new) HWY_ATTR {
    const auto mask_d_acc_zero = hn::Eq(d_acc, hn::Zero(df));
    const auto mask_d_other_zero = hn::Eq(d_other, hn::Zero(df));

    const VF m_acc_eff = hn::IfThenElse(mask_d_acc_zero, neg_inf, m_acc);
    const VF m_other_eff = hn::IfThenElse(mask_d_other_zero, neg_inf, m_other);
    m_new = hn::Max(m_acc_eff, m_other_eff);

    VF exp_l = hn::FastExpMinusOrZero(df, hn::Sub(m_acc, m_new));
    VF exp_r = hn::FastExpMinusOrZero(df, hn::Sub(m_other, m_new));

    exp_l = hn::IfThenZeroElse(mask_d_acc_zero, exp_l);
    exp_r = hn::IfThenZeroElse(mask_d_other_zero, exp_r);

    const VF num_l = hn::Mul(d_acc, exp_l);
    const VF num_r = hn::Mul(d_other, exp_r);
    d_new = hn::Add(num_l, num_r);

    const VF inv_d_new = hn::IfThenElseZero(hn::Gt(d_new, hn::Zero(df)),
                                            hn::Div(hn::Set(df, 1.0f), d_new));
    c1 = hn::Mul(num_l, inv_d_new);
    c2 = hn::Mul(num_r, inv_d_new);
  };

  size_t q = 0;
  for (; q + lanes <= group_count; q += lanes) {
    VF c1, c2, m_new, d_new;
    merge_lanes(hn::LoadU(df, acc_max_logits + q),
                hn::LoadU(df, acc_exp_sums + q),
                hn::LoadU(df, other_max_logits + q),
                hn::LoadU(df, other_exp_sums + q), c1, c2, m_new, d_new);

    hn::StoreU(c1, df, c1_arr + q);
    hn::StoreU(c2, df, c2_arr + q);
    hn::StoreU(m_new, df, acc_max_logits + q);
    hn::StoreU(d_new, df, acc_exp_sums + q);
  }
  // Remaining lanes, if any. `LoadN` zero-fills past `remaining`, so the unused
  // lanes look like empty accumulators and are discarded by `StoreN`.
  if (q < group_count) {
    const size_t remaining = group_count - q;
    VF c1, c2, m_new, d_new;
    merge_lanes(hn::LoadN(df, acc_max_logits + q, remaining),
                hn::LoadN(df, acc_exp_sums + q, remaining),
                hn::LoadN(df, other_max_logits + q, remaining),
                hn::LoadN(df, other_exp_sums + q, remaining), c1, c2, m_new,
                d_new);

    hn::StoreN(c1, df, c1_arr + q, remaining);
    hn::StoreN(c2, df, c2_arr + q, remaining);
    hn::StoreN(m_new, df, acc_max_logits + q, remaining);
    hn::StoreN(d_new, df, acc_exp_sums + q, remaining);
  }

  for (size_t i = 0; i < group_count; ++i) {
    const float c1 = c1_arr[i];
    const float c2 = c2_arr[i];
    if (c2 == 0.0f) continue;
    float* HWY_RESTRICT acc_row = acc_att_out.Row(group_start + i);
    const float* HWY_RESTRICT other_row = other_att_out.Row(group_start + i);
    if (c1 == 0.0f) {
      hwy::CopyBytes(other_row, acc_row, qkv_dim * sizeof(float));
    } else {
      const VF vc1 = hn::Set(df, c1);
      const VF vc2 = hn::Set(df, c2);
      size_t d = 0;
      for (; d + lanes <= qkv_dim; d += lanes) {
        VF va = hn::LoadU(df, acc_row + d);
        VF vo = hn::LoadU(df, other_row + d);
        hn::StoreU(hn::MulAdd(vo, vc2, hn::Mul(va, vc1)), df, acc_row + d);
      }
      if (d < qkv_dim) {
        const size_t remaining_d = qkv_dim - d;
        VF va = hn::LoadN(df, acc_row + d, remaining_d);
        VF vo = hn::LoadN(df, other_row + d, remaining_d);
        hn::StoreN(hn::MulAdd(vo, vc2, hn::Mul(va, vc1)), df, acc_row + d,
                   remaining_d);
      }
    }
  }
}

template <typename T>
float AbsMaxOfSpan(hwy::Span<const T> span) {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> df;
  using VF = hn::Vec<decltype(df)>;
  VF max_vec = hn::Zero(df);
  HWY_LANES_CONSTEXPR size_t N = hn::Lanes(df);
  HWY_LANES_CONSTEXPR size_t step = 2 * N;
  const PackedSpan<const T> packed_span(span.data(), span.size());
  size_t i = 0;
  for (; i + step <= span.size(); i += step) {
    VF v0, v1;
    Decompress2(df, packed_span, i, v0, v1);
    max_vec = hn::Max(max_vec, hn::Max(hn::Abs(v0), hn::Abs(v1)));
  }
  float max_scalar = 0.0f;
  for (; i < span.size(); ++i) {
    max_scalar = HWY_MAX(max_scalar,
                         hwy::ScalarAbs(hwy::ConvertScalarTo<float>(span[i])));
  }
  return HWY_MAX(hn::ReduceMax(df, max_vec), max_scalar);
}

// Forked from ComputeQKV. But it stores the K/V in the tiled format
// KV_T is type stored in the KV cache (typically float or BF16).
template <typename KV_T>
static HWY_INLINE void ComputeQKVTransposedTile(
    size_t num_tokens, const size_t layer_idx, const LayerWeightsPtrs& layer,
    AttentionImpl attention_impl, AttentionActivationsPtrs& activations,
    const QBatch& qbatch, const int flags, MatMulEnv& env) {
  PROFILER_ZONE("Gen.Attention.QKVTiled");
  const hwy::Divisor div_qbatch(qbatch.Size());
  const size_t num_interleaved = num_tokens * div_qbatch.GetDivisor();
  const LayerConfig& layer_config = layer.layer_config;
  const size_t qkv_dim = layer_config.qkv_dim;
  const size_t kv_heads = layer_config.kv_heads;

  // Resolve KV cache layer index and skip flag
  const size_t kv_layer_idx =
      (layer_config.kv_share_layer_idx >= 0)
          ? static_cast<size_t>(layer_config.kv_share_layer_idx)
          : layer_idx;
  const bool skip_kv =
      (layer_config.kv_share_layer_idx >= 0) || (flags & kSkipKV);

  // The original qkv_einsum_w has shape [(heads + kv_heads * 2), qkv_dim,
  // model_dim], which we reshaped to (heads + kv_heads * 2) * qkv_dim rows.
  // This computes Q and stores it in activations.q.
  CallMatMul(activations.pre_att_rms_out, layer.qkv_einsum_w1,
             /*add=*/nullptr, env, activations.q);

  if (skip_kv) return;

  // Compute the combined KV output from pre_att_rms_out.
  // The output shape is [num_interleaved, kv_heads * 2 * qkv_dim].
  const size_t kv_out_cols = kv_heads * 2 * qkv_dim;
  hwy::AlignedFreeUniquePtr<float[]> kv_out_mem_fallback;
  float* kv_out_data = nullptr;
  if (activations.kv_out_mem != nullptr) {
    if (activations.kv_out_mem->size() < num_interleaved * kv_out_cols) {
      activations.kv_out_mem->resize(num_interleaved * kv_out_cols);
    }
    kv_out_data = activations.kv_out_mem->data();
  } else {
    kv_out_mem_fallback =
        hwy::AllocateAligned<float>(num_interleaved * kv_out_cols);
    kv_out_data = kv_out_mem_fallback.get();
  }
  MatPtrT<float> kv_out_mat("kv_out", Extents2D(num_interleaved, kv_out_cols));
  kv_out_mat.SetPtr(kv_out_data, kv_out_cols);
  CallMatMul(activations.pre_att_rms_out, layer.qkv_einsum_w2,
             /*add=*/nullptr, env, kv_out_mat);

  // Apply positional encodings and store K/V in tiled format.
  hwy::Divisor div_kv_heads(kv_heads);

  bool is_transposed_qs =
      attention_impl == AttentionImpl::kFlashTransposedQsBF16
      || attention_impl == AttentionImpl::kFlashTransposedQsInt16 ||
      attention_impl == AttentionImpl::kFlashTransposedQsInt8;

  hn::ScalableTag<float> df;
  static hwy::Divisor tile_size_divisor(KVCache::kTileSize);
  ParallelFor(
      Parallelism::kFlat, kv_heads * qbatch.Size(), env.ctx,
      /*cluster_idx=*/0, Callers::kAttComputeQKV,
      [&](size_t task, size_t worker) HWY_ATTR {
        const size_t kv_head = div_kv_heads.Remainder(task);
        const size_t query_idx = div_kv_heads.Divide(task);
        CompressPerThread tls;
        size_t current_token_idx = 0;
        float* k_tile_vec = activations.k_tile_vec.Row(task);
        float* v_tile_vec = activations.v_tile_vec.Row(task);
        HWY_ALIGN float k_f32[kMaxQKVDim];
        const size_t start_pos = qbatch.Pos(query_idx);
        const bool is_global_layer =
            activations.config.IsGlobalLayer(layer_idx);
        std::vector<MatPtr> kv_ptrs = qbatch.KV(query_idx).cache->GetPointers(
            kv_layer_idx, kv_head, start_pos, is_global_layer);
        const size_t v_offset = qkv_dim * KVCache::kTileSize;
        const size_t tile_span_size = 2 * qkv_dim * KVCache::kTileSize;
        const size_t k_size = qkv_dim * KVCache::kTileSize;
        size_t tile_offset = 0;
        if (!is_global_layer) {
          tile_offset = start_pos / KVCache::kTileSize;
        }

        while (current_token_idx < num_tokens) {
          const size_t pos = start_pos + current_token_idx;
          const size_t pos_mod = activations.div_seq_len.Remainder(pos);
          const size_t tile_idx = tile_size_divisor.Divide(pos_mod);
          const size_t relative_tile_idx = tile_idx - tile_offset;
          KV_T* tile_ptr;
          int kv_ptr_idx = 0;
          size_t absolute_rows = 0;
          while (absolute_rows + kv_ptrs[kv_ptr_idx].Rows() <=
                 relative_tile_idx) {
            absolute_rows += kv_ptrs[kv_ptr_idx].Rows();
            kv_ptr_idx++;
          }
          tile_ptr = HWY_RCAST_ALIGNED(
              KV_T*,
              kv_ptrs[kv_ptr_idx].RowBytes(relative_tile_idx - absolute_rows));
          PackedSpan<KV_T> tile_packed_span{tile_ptr, tile_span_size};

          DecompressAndZeroPad(df, tile_packed_span, 0, k_tile_vec, k_size);
          DecompressAndZeroPad(df, tile_packed_span, v_offset, v_tile_vec,
                               qkv_dim * KVCache::kTileSize);

          size_t token_in_tile_idx = current_token_idx;
          while (token_in_tile_idx < num_tokens) {
            const size_t current_pos =
                qbatch.Pos(query_idx) + token_in_tile_idx;
            const size_t current_pos_mod =
                activations.div_seq_len.Remainder(current_pos);
            if (tile_size_divisor.Divide(current_pos_mod) != tile_idx) {
              break;  // Moved to next tile
            }

            const float* kv_row =
                kv_out_data +
                (token_in_tile_idx * qbatch.Size() + query_idx) * kv_out_cols;
            const float* k_values = kv_row + kv_head * 2 * qkv_dim;
            const float* v_values = kv_row + kv_head * 2 * qkv_dim + qkv_dim;
            hwy::CopyBytes(k_values, k_f32, qkv_dim * sizeof(float));
            if (layer.key_norm_scale.HasPtr()) {
              CallUpcasted(&layer.key_norm_scale, [&](const auto* weights_t) {
                RMSNormInplace(weights_t->PackedScale1(), /*w_ofs=*/0, k_f32,
                               qkv_dim, env.ctx, worker);
              });
            } else if (layer_config.post_qk == PostQKType::NormLocalRope ||
                       layer_config.use_qk_norm) {
              RMSNormNoScaleInplace(k_f32, qkv_dim, env.ctx, worker);
            }
            PositionalEncodingQK(
                k_f32, layer_idx, activations, env.ctx, worker,
                current_pos ,
                /*mul=*/1.0f);

            const size_t in_tile_idx = current_pos_mod % KVCache::kTileSize;
            const float* v_source = v_values;
            HWY_ALIGN float v_norm_buf[kMaxQKVDim];
            if (layer_config.norm_v) {
              hwy::CopyBytes(v_values, v_norm_buf, qkv_dim * sizeof(float));
              RMSNormNoScaleInplace(v_norm_buf, qkv_dim, env.ctx, worker);
              v_source = v_norm_buf;
            }
            // `v_cache_values` is a pointer to the V data that will be
            // compressed and stored in the KV cache. By default, it points to
            // the raw `v_source`.
            const float* v_cache_values = v_source;
            // `v_buf` is a temporary buffer used only when quantizing V values
            // to int8_t.
            HWY_ALIGN float v_buf[kMaxQKVDim];

            if constexpr (IsInt8<KV_T>()) {
              BF16* scales_ptr = HWY_RCAST_ALIGNED(
                  BF16*, tile_ptr + 2 * qkv_dim * KVCache::kTileSize);

              auto scale_and_store = [&](float* values, int dim,
                                         size_t scale_idx) HWY_ATTR {
                const float max_abs =
                    AbsMaxOfSpan(hwy::Span<const float>(values, dim));
                float scale = max_abs / 127.0f;
                if (scale == 0.0f) scale = 1.0f;
                scales_ptr[scale_idx] = hwy::ConvertScalarTo<BF16>(scale);
                const float inv_scale = 1.0f / scale;
                const hn::Vec<decltype(df)> v_inv_scale =
                    hn::Set(df, inv_scale);
                const size_t lanes = hn::Lanes(df);

                const hn::Rebind<int32_t, decltype(df)> di32;
                auto sum_vec = hn::Zero(di32);
                bool is_k = scale_idx < KVCache::kTileSize;

                size_t i = 0;
                for (; i + lanes <= dim; i += lanes) {
                  auto scaled = hn::Mul(hn::LoadU(df, values + i), v_inv_scale);
                  hn::StoreU(scaled, df, values + i);
                  if (is_k &&
                      attention_impl == AttentionImpl::kFlashTransposedQsInt8) {
                    sum_vec = hn::Add(sum_vec, hn::NearestInt(scaled));
                  }
                }
                if (HWY_UNLIKELY(i < dim)) {
                  auto scaled =
                      hn::Mul(hn::LoadN(df, values + i, dim - i), v_inv_scale);
                  hn::StoreN(scaled, df, values + i, dim - i);
                  if (is_k &&
                      attention_impl == AttentionImpl::kFlashTransposedQsInt8) {
                    sum_vec = hn::Add(sum_vec, hn::NearestInt(scaled));
                  }
                }
                if (is_k &&
                    attention_impl == AttentionImpl::kFlashTransposedQsInt8) {
                  int32_t* k_sums_ptr = reinterpret_cast<int32_t*>(
                      scales_ptr + 2 * KVCache::kTileSize);
                  k_sums_ptr[scale_idx] = hn::ReduceSum(di32, sum_vec);
                }
              };

              // K Scaling
              scale_and_store(k_f32, qkv_dim, in_tile_idx);

              // V Scaling: Copy `v_source` to `v_buf`, scale `v_buf` in-place,
              // and then update `v_cache_values` to point to `v_buf`.
              hwy::CopyBytes(v_source, v_buf, qkv_dim * sizeof(float));
              scale_and_store(v_buf, qkv_dim, KVCache::kTileSize + in_tile_idx);
              v_cache_values = v_buf;
            }

            const MatPtr& compact_kv_cache_ptr =
                qbatch.KV(query_idx).cache->compact_kv_cache_ptr;
            if (compact_kv_cache_ptr.GetType() == Type::kBF16 &&
                compact_kv_cache_ptr.GetLayout() ==
                    MatPtr::Layout::kBF16MatrixAccumulation) {
              for (size_t dim = 0; dim < qkv_dim; ++dim) {
                size_t k_offset = gcpp::KMatrixAccumulationOffset_BF16(
                    qkv_dim, dim, in_tile_idx);
                k_tile_vec[k_offset] = k_f32[dim];

                size_t v_offset = gcpp::VMatrixAccumulationOffset_BF16(
                    qkv_dim, in_tile_idx, dim);
                v_tile_vec[v_offset] = v_cache_values[dim];
              }
            } else if (attention_impl ==
                       AttentionImpl::kInt8MatrixAccumulation) {
              for (size_t dim = 0; dim < qkv_dim; ++dim) {
                size_t k_offset = gcpp::MatrixAccumulationOffset_Int8(
                    qkv_dim, dim, in_tile_idx);
                k_tile_vec[k_offset] = k_f32[dim];

                size_t v_offset_local = gcpp::VMatrixAccumulationOffset_Int8(
                    qkv_dim, in_tile_idx, dim);
                v_tile_vec[v_offset_local] = v_cache_values[dim];
              }
            } else if (attention_impl ==
                       AttentionImpl::kFlashTransposedQsInt8) {
              for (int dim = 0; dim < qkv_dim; ++dim) {
                // K VNNI layout: [qkv_dim/4, kTileSize, 4]
                size_t k_offset = (dim - dim % 4) * KVCache::kTileSize +
                                  in_tile_idx * 4 + (dim % 4);
                k_tile_vec[k_offset] = k_f32[dim];

                // V VNNI layout: [kTileSize/4, qkv_dim, 4]
                size_t v_offset_local =
                    (in_tile_idx - in_tile_idx % 4) * qkv_dim + dim * 4 +
                    (in_tile_idx % 4);
                v_tile_vec[v_offset_local] = v_cache_values[dim];
              }
            } else if (attention_impl ==
                           AttentionImpl::kFlashTransposedQsBF16 &&
                       std::is_same_v<KV_T, int8_t>) {
              for (int dim = 0; dim < qkv_dim; dim += 2) {
                const int dim_mod_2 = dim % 2;
                k_tile_vec[(dim - dim_mod_2) * KVCache::kTileSize +
                           in_tile_idx * 2] = k_f32[dim];
                k_tile_vec[(dim - dim_mod_2) * KVCache::kTileSize +
                           in_tile_idx * 2 + 1] = k_f32[dim + 1];
              }
              for (int dim = 0; dim < qkv_dim; ++dim) {
                size_t v_offset_local =
                    (in_tile_idx - in_tile_idx % 4) * qkv_dim + dim * 4 +
                    (in_tile_idx % 4);
                v_tile_vec[v_offset_local] = v_cache_values[dim];
              }
            } else if (is_transposed_qs) {
              const int in_tile_idx_mod_2 = in_tile_idx % 2;
              for (int dim = 0; dim < qkv_dim; dim += 2) {
                const int dim_mod_2 = dim % 2;
                // Pack k's in pairs in preparation for BF16 dot product.
                // See flash_attention.cc
                // QDotKTilexUpTo4TransposedKDoubleWidthBF16
                k_tile_vec[(dim - dim_mod_2) * KVCache::kTileSize +
                           in_tile_idx * 2] = k_f32[dim];
                k_tile_vec[(dim - dim_mod_2) * KVCache::kTileSize +
                           in_tile_idx * 2 + 1] = k_f32[dim + 1];
                // Pack v's in pairs
                v_tile_vec[(in_tile_idx - in_tile_idx_mod_2) * qkv_dim +
                           dim * 2 + in_tile_idx_mod_2] = v_cache_values[dim];
                v_tile_vec[(in_tile_idx - in_tile_idx_mod_2) * qkv_dim +
                           (dim + 1) * 2 + in_tile_idx_mod_2] =
                    v_cache_values[dim + 1];
              }

            } else {
              for (int i = 0; i < qkv_dim; ++i) {
                k_tile_vec[i * KVCache::kTileSize + in_tile_idx] = k_f32[i];
              }
              Compress(v_cache_values, qkv_dim, tls, tile_packed_span,
                       qkv_dim * (KVCache::kTileSize + in_tile_idx));
            }

            token_in_tile_idx++;
          }
          Compress(k_tile_vec, k_size, tls, tile_packed_span, 0);
          if (is_transposed_qs ||
              attention_impl == AttentionImpl::kFlashMatrixAccumulation ||
              attention_impl == AttentionImpl::kInt8MatrixAccumulation) {
            Compress(v_tile_vec, qkv_dim * KVCache::kTileSize, tls,
                     tile_packed_span, v_offset);
          }
          current_token_idx = token_in_tile_idx;
        }
      });
}

// Note: q_ptr and out_ptr do not use HWY_RESTRICT because this function may be
// called for in-place compression.
template <typename OutT, class DF, class DOut>
static HWY_INLINE void CompressSingleQueryBF16orInt16(
    DF df, DOut d_out, const float* q_ptr, int qkv_dim, OutT* out_ptr,
    // scale_out is required if OutT is int16_t, and unused otherwise.
    float* scale_out = nullptr) {
  namespace hn = hwy::HWY_NAMESPACE;
  const size_t lanes = hn::Lanes(df);
  const hn::ScalableTag<OutT> d_out_full;
  float s = 1.0f;
  if constexpr (IsInt16<OutT>()) {
    HWY_DASSERT(scale_out != nullptr);
    float max_abs = AbsMaxOfSpan(hwy::Span<const float>(q_ptr, qkv_dim));
    s = max_abs == 0.0f ? 1.0f : 32767.0f / max_abs;
    *scale_out = 1.0f / s;
  }
  auto scale_vec = hn::Set(df, s);

  for (size_t i = 0; i < qkv_dim; i += 2 * lanes) {
    auto x0 = hn::LoadU(df, q_ptr + i);
    auto x1 = hn::LoadU(df, q_ptr + i + lanes);
    if constexpr (IsInt16<OutT>()) {
      x0 = hn::Mul(x0, scale_vec);
      x1 = hn::Mul(x1, scale_vec);
      auto demoted = hn::OrderedDemote2To(d_out_full, hn::NearestInt(x0),
                                          hn::NearestInt(x1));
      hn::StoreU(demoted, d_out_full, out_ptr + i);
    } else {
      auto demoted = hn::OrderedDemote2To(d_out_full, x0, x1);
      hn::StoreU(demoted, d_out_full, out_ptr + i);
    }
  }
}

template <typename OutT>
static HWY_INLINE void CompressQueriesBF16orInt16(
    hwy::Span<const float* const> input, int qkv_dim, OutT* HWY_RESTRICT output,
    float* HWY_RESTRICT scale = nullptr) {
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  const DF df;
  auto d_out = hn::Rebind<OutT, decltype(df)>();
  const size_t num_queries = input.size();

  for (size_t q = 0; q < num_queries; ++q) {
    CompressSingleQueryBF16orInt16(df, d_out, input[q], qkv_dim,
                                   output + q * qkv_dim,
                                   scale == nullptr ? nullptr : scale + q);
  }
}

template <typename OutT>
static HWY_INLINE void CompressQueriesBF16orInt16Contiguous(
    const float* HWY_RESTRICT input, int qkv_dim, size_t num_queries,
    OutT* HWY_RESTRICT output, float* HWY_RESTRICT scale = nullptr) {
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  const DF df;
  auto d_out = hn::Rebind<OutT, decltype(df)>();

  for (size_t q = 0; q < num_queries; ++q) {
    CompressSingleQueryBF16orInt16(df, d_out, input + q * qkv_dim, qkv_dim,
                                   output + q * qkv_dim,
                                   scale == nullptr ? nullptr : scale + q);
  }
}

void CompressQueriesBF16(hwy::Span<const float* const> input, int qkv_dim,
                         BF16* HWY_RESTRICT output) {
  CompressQueriesBF16orInt16(input, qkv_dim, output);
}

void CompressQueriesBF16Contiguous(const float* HWY_RESTRICT input, int qkv_dim,
                                   size_t num_queries,
                                   BF16* HWY_RESTRICT output) {
  CompressQueriesBF16orInt16Contiguous(input, qkv_dim, num_queries, output);
}

void CompressQueriesInt16(hwy::Span<const float* const> input, int qkv_dim,
                          int16_t* HWY_RESTRICT output,
                          float* HWY_RESTRICT scale) {
  CompressQueriesBF16orInt16(input, qkv_dim, output, scale);
}

void CompressQueriesInt16Contiguous(const float* HWY_RESTRICT input,
                                    int qkv_dim, size_t num_queries,
                                    int16_t* HWY_RESTRICT output,
                                    float* HWY_RESTRICT scale) {
  CompressQueriesBF16orInt16Contiguous(input, qkv_dim, num_queries, output,
                                       scale);
}

template <class DF>
static HWY_INLINE void CompressSingleQueryInt8(DF df, const float* q_ptr,
                                               int qkv_dim, int8_t* out_ptr,
                                               float* scale_out) {
  namespace hn = hwy::HWY_NAMESPACE;
  const size_t lanes = hn::Lanes(df);
  const hn::ScalableTag<int8_t> d_out_full;
  const hn::ScalableTag<int16_t> d16;

  HWY_DASSERT(scale_out != nullptr);
  float max_abs = AbsMaxOfSpan(hwy::Span<const float>(q_ptr, qkv_dim));
  float s = max_abs == 0.0f ? 1.0f : 127.0f / max_abs;
  *scale_out = 1.0f / s;
  const hn::Vec<DF> scale_vec = hn::Set(df, s);

  HWY_DASSERT(qkv_dim % (4 * lanes) == 0);

  for (size_t i = 0; i < qkv_dim; i += 4 * lanes) {
    hn::Vec<DF> x0 = hn::LoadU(df, q_ptr + i);
    hn::Vec<DF> x1 = hn::LoadU(df, q_ptr + i + lanes);
    hn::Vec<DF> x2 = hn::LoadU(df, q_ptr + i + 2 * lanes);
    hn::Vec<DF> x3 = hn::LoadU(df, q_ptr + i + 3 * lanes);

    x0 = hn::Mul(x0, scale_vec);
    x1 = hn::Mul(x1, scale_vec);
    x2 = hn::Mul(x2, scale_vec);
    x3 = hn::Mul(x3, scale_vec);

    const hn::Vec<decltype(d16)> demoted16_0 =
        hn::OrderedDemote2To(d16, hn::NearestInt(x0), hn::NearestInt(x1));
    const hn::Vec<decltype(d16)> demoted16_1 =
        hn::OrderedDemote2To(d16, hn::NearestInt(x2), hn::NearestInt(x3));
    const hn::Vec<decltype(d_out_full)> demoted8 =
        hn::OrderedDemote2To(d_out_full, demoted16_0, demoted16_1);
    const hn::Vec<decltype(d_out_full)> biased8 =
        hn::Add(demoted8, hn::Set(d_out_full, static_cast<int8_t>(-128)));
    hn::StoreU(biased8, d_out_full, out_ptr + i);
  }
}

void CompressQueriesInt8(hwy::Span<const float* const> input, int qkv_dim,
                         int8_t* HWY_RESTRICT output,
                         float* HWY_RESTRICT scale) {
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  const DF df;
  const size_t num_queries = input.size();

  for (size_t q = 0; q < num_queries; ++q) {
    CompressSingleQueryInt8(df, input[q], qkv_dim, output + q * qkv_dim,
                            scale + q);
  }
}

void CompressQueriesInt8Contiguous(const float* HWY_RESTRICT input, int qkv_dim,
                                   size_t num_queries,
                                   int8_t* HWY_RESTRICT output,
                                   float* HWY_RESTRICT scale) {
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  const DF df;

  for (size_t q = 0; q < num_queries; ++q) {
    CompressSingleQueryInt8(df, input + q * qkv_dim, qkv_dim,
                            output + q * qkv_dim, scale + q);
  }
}

template <typename T>
static HWY_INLINE void MaybeResizeMatStorage(MatStorageT<T>& mat_storage,
                                             int rows, int cols,
                                             const char* name,
                                             const Allocator& allocator) {
  if (mat_storage.Rows() != rows || mat_storage.Cols() != cols) {
    mat_storage = MatStorageT<T>(name, Extents2D(rows, cols), allocator,
                                 MatPadding::kOdd);
  }
}

template <typename QueryProvider>
HWY_INLINE void CompressAndTransposeQueriesMatrixAccumulationImpl(
    QueryProvider query_provider, BF16* packed_queries, size_t num_queries,
    size_t qkv_dim) {
  namespace hn = hwy::HWY_NAMESPACE;
  using InT = hwy::RemoveCvRef<hwy::RemovePtr<decltype(query_provider(0))>>;
  const hn::Full128<BF16> dbf16;
  const hn::Half<decltype(dbf16)> dbf_half;
  const hn::Full128<float> df;
  const size_t kL = hn::Lanes(dbf_half);

  using V_BF16 = hn::Vec<decltype(dbf16)>;
  using V_BF16_Half = hn::Vec<decltype(dbf_half)>;
  using V_F32 = hn::Vec<decltype(df)>;

  HWY_DASSERT(qkv_dim % kL == 0);

  auto pack4x2 = [&](const InT* q0,
                     const InT* q1) HWY_ATTR -> V_BF16 {
    if constexpr (IsBF16<InT>()) {
      const V_BF16_Half v0 = hn::LoadU(dbf_half, q0);
      const V_BF16_Half v1 =
          q1 != nullptr ? hn::LoadU(dbf_half, q1) : hn::Zero(dbf_half);
      return hn::Combine(dbf16, v1, v0);
    } else {
      const V_F32 v0 = hn::LoadU(df, q0);
      const V_F32 v1 = q1 != nullptr ? hn::LoadU(df, q1) : hn::Zero(df);
      return hn::OrderedDemote2To(dbf16, v0, v1);
    }
  };

  auto pack_pair = [&](const InT* q0, const InT* q1, BF16* out) HWY_ATTR {
    if (q1 != nullptr) {
      for (size_t d = 0; d < qkv_dim; d += kL) {
        hn::StoreU(pack4x2(q0 + d, q1 + d), dbf16, out + d * 2);
      }
    } else {
      for (size_t d = 0; d < qkv_dim; d += kL) {
        hn::StoreU(pack4x2(q0 + d, nullptr), dbf16, out + d * 2);
      }
    }
  };

  size_t p = 0;
  for (; p < num_queries / 2; ++p) {
    pack_pair(query_provider(2 * p), query_provider(2 * p + 1),
              packed_queries + 2 * p * qkv_dim);
  }
  if (num_queries % 2 != 0) {
    pack_pair(query_provider(2 * p), nullptr,
              packed_queries + 2 * p * qkv_dim);
  }
}

void CompressAndTransposeQueriesMatrixAccumulation(const float* raw_queries,
                                                   BF16* packed_queries,
                                                   size_t num_queries,
                                                   size_t qkv_dim) {
  CompressAndTransposeQueriesMatrixAccumulationImpl(
      [&](size_t idx) { return raw_queries + idx * qkv_dim; }, packed_queries,
      num_queries, qkv_dim);
}

void CompressAndTransposeQueriesMatrixAccumulationFromBF16(
    const BF16* raw_queries, BF16* packed_queries, size_t num_queries,
    size_t qkv_dim) {
  CompressAndTransposeQueriesMatrixAccumulationImpl(
      [&](size_t idx) { return raw_queries + idx * qkv_dim; }, packed_queries,
      num_queries, qkv_dim);
}

void CompressAndTransposeQueriesMatrixAccumulationNonContiguous(
    hwy::Span<const float* const> input, BF16* packed_queries, size_t qkv_dim) {
  CompressAndTransposeQueriesMatrixAccumulationImpl(
      [&](size_t idx) { return input[idx]; }, packed_queries, input.size(),
      qkv_dim);
}

void CompressAndTransposeQueriesMatrixAccumulationNonContiguousFromBF16(
    hwy::Span<const BF16* const> input, BF16* packed_queries, size_t qkv_dim) {
  CompressAndTransposeQueriesMatrixAccumulationImpl(
      [&](size_t idx) { return input[idx]; }, packed_queries, input.size(),
      qkv_dim);
}

template <typename QueryProvider>
HWY_INLINE void CompressAndQuantizeQueriesMatrixAccumulationInt8Impl(
    QueryProvider query_provider, int8_t* HWY_RESTRICT packed_queries,
    float* HWY_RESTRICT packed_scales, size_t num_queries, size_t qkv_dim) {
  HWY_DASSERT(qkv_dim % 8 == 0);

  namespace hn = hwy::HWY_NAMESPACE;
  using InT = hwy::RemoveCvRef<hwy::RemovePtr<decltype(query_provider(0))>>;
  const hn::Full128<float> df;
  const hn::Full128<int16_t> di16;
  const hn::Full128<int8_t> di8;

  using V_F32 = hn::Vec<decltype(df)>;
  using V_I32 = hn::Vec<hn::Repartition<int32_t, decltype(df)>>;
  using V_I16 = hn::Vec<decltype(di16)>;
  using V_I8 = hn::Vec<decltype(di8)>;

  size_t p = 0;
  for (; p < num_queries / 2; ++p) {
    const InT* q0 = query_provider(2 * p);
    const InT* q1 = query_provider(2 * p + 1);
    int8_t* out = packed_queries + 2 * p * qkv_dim;
    float* out_scale0 = packed_scales + 2 * p;
    float* out_scale1 = packed_scales + (2 * p + 1);

    const PackedSpan<const InT> span_q0(q0, qkv_dim);
    const PackedSpan<const InT> span_q1(q1, qkv_dim);

    // 1. Compute single scale per query over the entire qkv_dim
    float max_abs_q0 = AbsMaxOfSpan(hwy::Span<const InT>(q0, qkv_dim));
    float max_abs_q1 = AbsMaxOfSpan(hwy::Span<const InT>(q1, qkv_dim));

    float scale0_raw = max_abs_q0 == 0.0f ? 1.0f : max_abs_q0 / 127.0f;
    float scale1_raw = max_abs_q1 == 0.0f ? 1.0f : max_abs_q1 / 127.0f;

    gcpp::KV_microscale_t scale0_bf16 =
        hwy::ConvertScalarTo<gcpp::KV_microscale_t>(scale0_raw);
    gcpp::KV_microscale_t scale1_bf16 =
        hwy::ConvertScalarTo<gcpp::KV_microscale_t>(scale1_raw);

    float scale0 = hwy::ConvertScalarTo<float>(scale0_bf16);
    float scale1 = hwy::ConvertScalarTo<float>(scale1_bf16);

    *out_scale0 = scale0;
    *out_scale1 = scale1;

    V_F32 inv_scale0 = hn::Set(df, 1.0f / scale0);
    V_F32 inv_scale1 = hn::Set(df, 1.0f / scale1);

    for (size_t d = 0; d < qkv_dim; d += 8) {
      // 2. Load and quantize Q0 (8 channels)
      V_F32 q0_L, q0_H;
      Decompress2(df, span_q0, d, q0_L, q0_H);
      V_I32 q0_L_scaled = hn::NearestInt(hn::Mul(q0_L, inv_scale0));
      V_I32 q0_H_scaled = hn::NearestInt(hn::Mul(q0_H, inv_scale0));
      V_I16 q0_i16 = hn::OrderedDemote2To(di16, q0_L_scaled, q0_H_scaled);

      // 3. Load and quantize Q1 (8 channels)
      V_F32 q1_L, q1_H;
      Decompress2(df, span_q1, d, q1_L, q1_H);
      V_I32 q1_L_scaled = hn::NearestInt(hn::Mul(q1_L, inv_scale1));
      V_I32 q1_H_scaled = hn::NearestInt(hn::Mul(q1_H, inv_scale1));
      V_I16 q1_i16 = hn::OrderedDemote2To(di16, q1_L_scaled, q1_H_scaled);

      // 4. Pack in pairs at 128-bit boundary: 8 elements of Q0, then 8 elements
      // of Q1
      V_I8 packed = hn::OrderedDemote2To(di8, q0_i16, q1_i16);
      hn::StoreU(packed, di8, out + d * 2);
    }
  }

  if (num_queries % 2 != 0) {
    const InT* q0 = query_provider(2 * p);
    int8_t* out = packed_queries + 2 * p * qkv_dim;
    float* out_scale0 = packed_scales + 2 * p;
    V_I16 zero_i16 = hn::Zero(di16);

    const PackedSpan<const InT> span_q0(q0, qkv_dim);

    float max_abs_q0 = AbsMaxOfSpan(hwy::Span<const InT>(q0, qkv_dim));

    float scale0_raw = max_abs_q0 == 0.0f ? 1.0f : max_abs_q0 / 127.0f;
    gcpp::KV_microscale_t scale0_bf16 =
        hwy::ConvertScalarTo<gcpp::KV_microscale_t>(scale0_raw);
    float scale0 = hwy::ConvertScalarTo<float>(scale0_bf16);

    *out_scale0 = scale0;

    V_F32 inv_scale0 = hn::Set(df, 1.0f / scale0);

    for (size_t d = 0; d < qkv_dim; d += 8) {
      V_F32 q0_L, q0_H;
      Decompress2(df, span_q0, d, q0_L, q0_H);
      V_I32 q0_L_scaled = hn::NearestInt(hn::Mul(q0_L, inv_scale0));
      V_I32 q0_H_scaled = hn::NearestInt(hn::Mul(q0_H, inv_scale0));
      V_I16 q0_i16 = hn::OrderedDemote2To(di16, q0_L_scaled, q0_H_scaled);

      V_I8 packed = hn::OrderedDemote2To(di8, q0_i16, zero_i16);
      hn::StoreU(packed, di8, out + d * 2);
    }
  }
}

void CompressAndQuantizeQueriesMatrixAccumulationInt8(const float* raw_queries,
                                                      int8_t* packed_queries,
                                                      float* packed_scales,
                                                      size_t num_queries,
                                                      size_t qkv_dim) {
  CompressAndQuantizeQueriesMatrixAccumulationInt8Impl(
      [&](size_t idx) { return raw_queries + idx * qkv_dim; }, packed_queries,
      packed_scales, num_queries, qkv_dim);
}

void CompressAndQuantizeQueriesMatrixAccumulationInt8FromBF16(
    const BF16* raw_queries, int8_t* packed_queries, float* packed_scales,
    size_t num_queries, size_t qkv_dim) {
  CompressAndQuantizeQueriesMatrixAccumulationInt8Impl(
      [&](size_t idx) { return raw_queries + idx * qkv_dim; }, packed_queries,
      packed_scales, num_queries, qkv_dim);
}

void CompressAndQuantizeQueriesMatrixAccumulationInt8NonContiguous(
    hwy::Span<const float* const> input, int8_t* packed_queries,
    float* packed_scales, size_t qkv_dim) {
  CompressAndQuantizeQueriesMatrixAccumulationInt8Impl(
      [&](size_t idx) { return input[idx]; }, packed_queries, packed_scales,
      input.size(), qkv_dim);
}

void CompressAndQuantizeQueriesMatrixAccumulationInt8NonContiguousFromBF16(
    hwy::Span<const BF16* const> input, int8_t* packed_queries,
    float* packed_scales, size_t qkv_dim) {
  CompressAndQuantizeQueriesMatrixAccumulationInt8Impl(
      [&](size_t idx) { return input[idx]; }, packed_queries, packed_scales,
      input.size(), qkv_dim);
}

// clang-format off
// Schedules TiledFlashAttention for all heads, tokens and batch.
// Returns partial results in the same order as queries in `activations.q`.
// Might not work yet for prefix lm.
// To help understanding how to use this function below is description of how
// parameters are used:
//
// attention_impl - Used to determine attention kernel to use.
// num_query_tokens - number of tokens/timesteps in processed in a single batch
// it will influence how many queries kvs are evaluated against.
// num_kv_tokens - number of tokens/timesteps in kv cache
// layer_idx - layer index
// layer - used to get kv_heads, heads, qkv_dim
// activations - reads: activations.q queries, att_cap, IsGlobalLayer
// qbatch - kv cache, Pos / EndPrefix
// ctx - threading context
// clang-format on
void LocalAttentionForAllHeadsTokensAndBatch(
    AttentionImpl attention_impl, const size_t num_query_tokens,
    const size_t layer_idx, const LayerWeightsPtrs& layer,
    AttentionActivationsPtrs& activations, QBatch& qbatch,
    ThreadingContext& ctx) {
  constexpr size_t kQueriesPerSubtask = 128;
  const size_t heads_per_kv_head =
      layer.layer_config.heads / layer.layer_config.kv_heads;
  const hwy::Divisor div_heads_per_kv_head(heads_per_kv_head);
  const hwy::Divisor div_kv_heads(layer.layer_config.kv_heads);

  size_t core_count = ctx.pools.MaxWorkers();
  size_t task_multiplier = 1;
  while (qbatch.Size() * layer.layer_config.kv_heads * task_multiplier <
         core_count * 2) {
    task_multiplier++;
  }
  // Finding the smallest context we need to attend to avoid unnecessary
  // overhead when sub-splitting doesn't make sense. This check overestimates
  // context sizes because it ignores [local] layer sizes and explicit
  // qbatch.Prefix settings.
  size_t min_pos = qbatch.Pos(0);
  for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
    min_pos = std::min(min_pos, qbatch.Pos(qi));
  }
  if (min_pos / task_multiplier < num_query_tokens) {
    // In case where min_pos / task_multiplier < num_tokens
    // To make sure we don't over count tokens or read out of bounds code
    // requires quite a bit more involved logic.
    // Also there is not much point to splitting the work into more tasks, when
    // amount of work is small.
    task_multiplier = 1;
  }
  size_t num_queries = num_query_tokens * heads_per_kv_head;
  size_t num_query_tasks = hwy::DivCeil(num_queries, kQueriesPerSubtask);
  [[maybe_unused]] size_t num_tasks =
      qbatch.Size() * layer.layer_config.kv_heads * num_query_tasks;
  [[maybe_unused]] size_t num_sub_tasks = qbatch.Size() *
                                          layer.layer_config.kv_heads *
                                          num_query_tasks * task_multiplier;
  HWY_DASSERT_M(activations.q.Rows() == num_query_tokens * qbatch.Size(),
                "qbatch size mismatch");
  size_t qkv_dim = layer.layer_config.qkv_dim;

  // sizes of all should be in sync
  if (num_sub_tasks > activations.sub_task_att_out->size()) {
    activations.sub_task_att_out->resize(num_sub_tasks);
    activations.sub_task_exp_denominator_sums->resize(num_sub_tasks);
    activations.sub_task_max_logits->resize(num_sub_tasks);
  }
  size_t max_queries_per_subtask = std::min(num_queries, kQueriesPerSubtask);
  if (attention_impl == AttentionImpl::kFlashTransposedQsBF16 ||
      attention_impl == AttentionImpl::kFlashMatrixAccumulation) {
    if (activations.bf16_queries != nullptr &&
        num_sub_tasks * max_queries_per_subtask * qkv_dim >
            activations.bf16_queries->size()) {
      activations.bf16_queries->resize(num_sub_tasks * max_queries_per_subtask *
                                       qkv_dim);
    }
  } else if (attention_impl == AttentionImpl::kFlashTransposedQsInt16) {
    if (activations.int16_queries != nullptr &&
        num_sub_tasks * max_queries_per_subtask * qkv_dim >
            activations.int16_queries->size()) {
      activations.int16_queries->resize(num_sub_tasks *
                                        max_queries_per_subtask * qkv_dim);
    }
    if (activations.q_scales != nullptr &&
        num_sub_tasks * max_queries_per_subtask >
            activations.q_scales->size()) {
      activations.q_scales->resize(num_sub_tasks * max_queries_per_subtask);
    }
  } else if (attention_impl == AttentionImpl::kFlashTransposedQsInt8) {
    if (activations.int8_queries != nullptr &&
        num_sub_tasks * max_queries_per_subtask * qkv_dim >
            activations.int8_queries->size()) {
      activations.int8_queries->resize(num_sub_tasks * max_queries_per_subtask *
                                       qkv_dim);
    }
    if (activations.q_scales != nullptr &&
        num_sub_tasks * max_queries_per_subtask >
            activations.q_scales->size()) {
      activations.q_scales->resize(num_sub_tasks * max_queries_per_subtask);
    }
  } else if (attention_impl == AttentionImpl::kInt8MatrixAccumulation) {
    if (activations.int8_queries != nullptr &&
        num_sub_tasks * max_queries_per_subtask * qkv_dim >
            activations.int8_queries->size()) {
      activations.int8_queries->resize(num_sub_tasks * max_queries_per_subtask *
                                       qkv_dim);
    }
    if (activations.q_scales != nullptr &&
        num_sub_tasks * max_queries_per_subtask >
            activations.q_scales->size()) {
      activations.q_scales->resize(num_sub_tasks * max_queries_per_subtask);
    }
  } else {
    if (activations.float_queries != nullptr &&
        num_sub_tasks * max_queries_per_subtask * qkv_dim >
            activations.float_queries->size()) {
      activations.float_queries->resize(num_sub_tasks *
                                        max_queries_per_subtask * qkv_dim);
    }
  }
  if (activations.worker_workspaces != nullptr &&
      activations.worker_workspaces->size() <= ctx.pools.MaxWorkers()) {
    activations.worker_workspaces->resize(ctx.pools.MaxWorkers() + 1);
  }
  const size_t num_groups_per_task =
      hwy::DivCeil(kQueriesPerSubtask, kMergeGroupSize);
  struct alignas(64) AtomicGroupMergeSlot {
    std::atomic<uint64_t> state;
    static constexpr uint64_t Pack(int32_t active_idx, int32_t remaining) {
      return (static_cast<uint64_t>(static_cast<uint32_t>(active_idx)) << 32) |
             static_cast<uint32_t>(remaining);
    }
    static constexpr int32_t ActiveIdx(uint64_t s) {
      return static_cast<int32_t>(s >> 32);
    }
    static constexpr int32_t Remaining(uint64_t s) {
      return static_cast<int32_t>(s & 0xFFFFFFFFu);
    }
  };
  std::vector<AtomicGroupMergeSlot> merge_slots(num_tasks *
                                                num_groups_per_task);
  const uint64_t initial_slot_state =
      AtomicGroupMergeSlot::Pack(-1, static_cast<int32_t>(task_multiplier));
  for (size_t i = 0; i < merge_slots.size(); ++i) {
    merge_slots[i].state.store(initial_slot_state, std::memory_order_relaxed);
  }

  // This loop parallelizes over qbatch, kv_head and substrings of context
  // tokens. Each parallel invocation handles all query tokens of the given
  // qbatch.
  ParallelFor(
      Parallelism::kHierarchical, num_sub_tasks, ctx,
      /*cluster_idx=*/0, Callers::kFlashAttention,
      [&](size_t task_idx, size_t worker) HWY_ATTR {
        size_t main_task_idx = task_idx / task_multiplier;
        size_t sub_task_idx = task_idx % task_multiplier;
        size_t query_task_idx = main_task_idx % num_query_tasks;
        size_t qbatch_and_kv_head_idx = main_task_idx / num_query_tasks;
        size_t current_qbatch_idx = div_kv_heads.Divide(qbatch_and_kv_head_idx);
        size_t kv_head_idx = div_kv_heads.Remainder(qbatch_and_kv_head_idx);

        size_t query_start_idx = query_task_idx * kQueriesPerSubtask;
        size_t query_end_idx =
            std::min(num_queries, query_start_idx + kQueriesPerSubtask);
        size_t sub_num_queries = query_end_idx - query_start_idx;
        const size_t num_task_groups =
            hwy::DivCeil(sub_num_queries, kMergeGroupSize);

        auto write_group_output = [&](size_t g, int32_t final_idx) {
          const size_t g_start = g * kMergeGroupSize;
          const size_t g_end =
              HWY_MIN(sub_num_queries, g_start + kMergeGroupSize);
          for (size_t sub_q_idx = g_start; sub_q_idx < g_end; ++sub_q_idx) {
            size_t q_idx = query_start_idx + sub_q_idx;
            size_t token_idx = div_heads_per_kv_head.Divide(q_idx);
            size_t head_in_group_idx = div_heads_per_kv_head.Remainder(q_idx);

            const size_t batch_index =
                current_qbatch_idx * num_query_tokens + token_idx;
            const size_t q_head_idx =
                kv_head_idx * heads_per_kv_head + head_in_group_idx;
            const size_t activations_att_out_start_idx = q_head_idx * qkv_dim;

            if (final_idx >= 0) {
              const MatStorageT<float>& final_att_out =
                  activations.sub_task_att_out->at(final_idx);
              const AlignedFloatVector& final_exp_sums =
                  activations.sub_task_exp_denominator_sums->at(final_idx);
              const AlignedFloatVector& final_max_logits =
                  activations.sub_task_max_logits->at(final_idx);
              hwy::CopyBytes(final_att_out.Row(sub_q_idx),
                             activations.att_out.Row(batch_index) +
                                 activations_att_out_start_idx,
                             qkv_dim * sizeof(float));
              activations.softmax_d.Row(batch_index)[q_head_idx] =
                  final_exp_sums[sub_q_idx];
              activations.softmax_max.Row(batch_index)[q_head_idx] =
                  final_max_logits[sub_q_idx];
            } else {
              hwy::ZeroBytes(activations.att_out.Row(batch_index) +
                                 activations_att_out_start_idx,
                             qkv_dim * sizeof(float));
              activations.softmax_d.Row(batch_index)[q_head_idx] = 0.0f;
              activations.softmax_max.Row(batch_index)[q_head_idx] =
                  -std::numeric_limits<float>::max() / 2.0f;
            }
          }
        };

        // Lock-free combining tree per 32-query group: active subtasks park
        // their partial buffer or claim and merge a parked one; skipped
        // subtasks (my_idx < 0) only decrement remaining.
        auto check_in_or_merge_group = [&](size_t g, int32_t my_idx) {
          const size_t g_start = g * kMergeGroupSize;
          const size_t g_count =
              HWY_MIN(kMergeGroupSize, sub_num_queries - g_start);
          auto& slot =
              merge_slots[main_task_idx * num_groups_per_task + g].state;
          uint64_t cur = slot.load(std::memory_order_acquire);
          while (true) {
            const int32_t active_idx = AtomicGroupMergeSlot::ActiveIdx(cur);
            const int32_t rem = AtomicGroupMergeSlot::Remaining(cur);
            if (my_idx >= 0 && active_idx != -1) {
              // Another partial buffer is parked: claim it and merge into
              // my_idx.
              if (slot.compare_exchange_weak(
                      cur, AtomicGroupMergeSlot::Pack(-1, rem),
                      std::memory_order_acq_rel, std::memory_order_acquire)) {
                MergeOnlineSoftmaxGroup32(active_idx, my_idx, g_start, g_count,
                                          qkv_dim, activations);
              }
            } else {
              const int32_t buf_idx = (my_idx < 0) ? active_idx : my_idx;
              if (rem == 1) {
                // All other subtasks have completed: write final merged output.
                if (slot.compare_exchange_weak(
                        cur, AtomicGroupMergeSlot::Pack(-1, 0),
                        std::memory_order_acq_rel, std::memory_order_acquire)) {
                  write_group_output(g, buf_idx);
                  break;
                }
              } else {
                // Parked my_idx (if active) or checked in (if skipped);
                // a subsequent worker will finish the merge and write the
                // output.
                if (slot.compare_exchange_weak(
                        cur, AtomicGroupMergeSlot::Pack(buf_idx, rem - 1),
                        std::memory_order_acq_rel, std::memory_order_acquire)) {
                  break;
                }
              }
            }
          }
        };

        // First and last context token we will attend to.
        size_t global_start_context_pos = StartPos(
            qbatch.Pos(current_qbatch_idx), activations.config, layer_idx);
        // Keep in mind this is overestimation because some timesteps might not
        // need all tokens due to causal mask.
        // We will use it to determine how to divide work between sub tasks
        // and make sure PrefixEnd is taken into account
        size_t start_context_pos = global_start_context_pos;
        size_t last_context_pos =
            qbatch.Pos(current_qbatch_idx) + num_query_tokens - 1;
        // In some models, context is limited to some prefix - make sure we take
        // that into account.
        const size_t prefix_end = qbatch.PrefixEnd(current_qbatch_idx);
        if (prefix_end > 0 && prefix_end - 1 > last_context_pos) {
          const size_t window_size =
              activations.config.attention_window_sizes[layer_idx];
          last_context_pos =
              std::min(prefix_end - 1, last_context_pos + window_size - 1);
        }
        size_t total_num_context_tokens =
            last_context_pos - start_context_pos + 1;
        size_t context_tokens_per_sub_task =
            hwy::DivCeil(total_num_context_tokens, task_multiplier);
        // Restrict tokens to attend to the substring of context tokens that
        // this subtask is responsible for.
        start_context_pos =
            start_context_pos + context_tokens_per_sub_task * sub_task_idx;
        if (start_context_pos > last_context_pos) {
          for (size_t g = 0; g < num_task_groups; ++g) {
            check_in_or_merge_group(g, -1);
          }
          return;
        }
        last_context_pos =
            std::min(last_context_pos,
                     start_context_pos + context_tokens_per_sub_task - 1);
        // pre-initialize memory [to avoid racy resizes laters].
        std::vector<float*> queries_ptrs;
        queries_ptrs.reserve(sub_num_queries);
        for (size_t q_idx = query_start_idx; q_idx < query_end_idx; ++q_idx) {
          size_t token_idx = div_heads_per_kv_head.Divide(q_idx);
          size_t q_head_idx = div_heads_per_kv_head.Remainder(q_idx);
          queries_ptrs.push_back(
              activations.q.Row(token_idx * qbatch.Size() +
                                current_qbatch_idx) +
              (kv_head_idx * heads_per_kv_head + q_head_idx) * qkv_dim);
        }
        hwy::Span<float*> queries_ptrs_span(queries_ptrs.data(),
                                            queries_ptrs.size());

        MatStorageT<float>& att_out =
            activations.sub_task_att_out->at(task_idx);
        AlignedFloatVector& exp_denominator_sums =
            activations.sub_task_exp_denominator_sums->at(task_idx);
        AlignedFloatVector& max_logits =
            activations.sub_task_max_logits->at(task_idx);
        MaybeResizeMatStorage(att_out, sub_num_queries, qkv_dim, "att_out",
                              ctx.allocator);
        for (size_t i = 0; i < sub_num_queries; ++i) {
          hwy::ZeroBytes(att_out.Row(i),
                         att_out.Cols() * sizeof(decltype(att_out.Row(i)[0])));
        }

        size_t num_queries_rounded_to_8 = hwy::RoundUpTo(sub_num_queries, 8);
        exp_denominator_sums.resize(num_queries_rounded_to_8);
        max_logits.resize(num_queries_rounded_to_8);
        for (size_t i = 0; i < num_queries_rounded_to_8; ++i) {
          exp_denominator_sums[i] = 0.0f;
          max_logits[i] = -std::numeric_limits<float>::max() / 2.0f;
        }
        // Get pointers to the KVCache tiles, starting at global_start_pos
        // Returns multiple matrices for non-contiguous memory, for example as a
        // result of the wraparound in local layers.
        std::vector<MatPtr> kv_ptrs =
            qbatch.KV(current_qbatch_idx)
                .cache->GetPointers(
                    layer_idx, kv_head_idx,
                    global_start_context_pos,
                    activations.config.IsGlobalLayer(layer_idx));

        std::vector<size_t, hwy::AlignedAllocator<size_t>> start_pos_per_query;
        std::vector<size_t, hwy::AlignedAllocator<size_t>> last_pos_per_query;
        start_pos_per_query.reserve(sub_num_queries);
        last_pos_per_query.reserve(sub_num_queries);
        // Position of the first token in the first tile whose pointer was
        // returned above. Allows for handling of token positions relative to
        // the KV tiles returned above.
        size_t rounded_down_global_start_pos =
            hwy::RoundDownTo(global_start_context_pos, KVCache::kTileSize);
        for (size_t q_idx = query_start_idx; q_idx < query_end_idx; ++q_idx) {
          size_t token_idx = div_heads_per_kv_head.Divide(q_idx);
          int64_t global_query_pos = qbatch.Pos(current_qbatch_idx) + token_idx;
          // Compute the range of context tokens [query_start_context_pos,
          // query_last_context_pos] that this query token should attend to
          // within the current KV tile/subtask.

          // For standard causal attention, a token cannot attend to future
          // positions (query_last_pos <= global_query_pos). For bidirectional
          // prefix attention (prefix_end > 0), tokens within the prefix can
          // attend forward to subsequent prefix tokens, capped by the local
          // sliding window size.
          int64_t query_last_pos = global_query_pos;
          if (prefix_end > 0 &&
              prefix_end - 1 > static_cast<size_t>(query_last_pos)) {
            const size_t window_size =
                activations.config.attention_window_sizes[layer_idx];
            query_last_pos = std::min(
                static_cast<int64_t>(prefix_end - 1),
                global_query_pos + static_cast<int64_t>(window_size) - 1);
          }
          int64_t query_last_context_pos =
              std::min(static_cast<int64_t>(last_context_pos), query_last_pos);

          // The query cannot attend backward beyond the sliding window
          // (global_query_pos - window_size + 1). Clamp to start_context_pos of
          // the current subtask. Signed int64_t is used to avoid underflow when
          // global_query_pos < window_size.
          int64_t query_start_context_pos = std::max(
              global_query_pos -
                  static_cast<int64_t>(
                      activations.config.attention_window_sizes[layer_idx]) +
                  1,
              static_cast<int64_t>(start_context_pos));

          // If the query's attention window does not overlap with this KV tile,
          // set start_pos > last_pos (SIZE_MAX and 0) so the attention kernel
          // skips this query.
          if (query_last_context_pos < query_start_context_pos) {
            start_pos_per_query.push_back(std::numeric_limits<size_t>::max());
            last_pos_per_query.push_back(0);
          } else {
            // Turn token position into KV-tile relative token positions.
            query_last_context_pos -= rounded_down_global_start_pos;
            query_start_context_pos -= rounded_down_global_start_pos;
            start_pos_per_query.push_back(
                static_cast<size_t>(query_start_context_pos));
            last_pos_per_query.push_back(
                static_cast<size_t>(query_last_context_pos));
          }
        }

        hwy::AlignedVector<uint8_t>* worker_workspace =
            activations.worker_workspaces != nullptr
                ? &(*activations.worker_workspaces)[worker]
                : nullptr;

        if (attention_impl == AttentionImpl::kFlashTransposedQsBF16) {
          HWY_DASSERT(activations.bf16_queries != nullptr);
          BF16* bf16_queries_ptr = activations.bf16_queries->data() +
                                   task_idx * max_queries_per_subtask * qkv_dim;
          CompressQueriesBF16(queries_ptrs_span, qkv_dim, bf16_queries_ptr);
          DispatchTileFlashAttentionReturnExpSumsAndMaxLogitsBF16(
              kv_ptrs, sub_num_queries, bf16_queries_ptr,
              hwy::Span<const size_t>(start_pos_per_query),
              hwy::Span<const size_t>(last_pos_per_query),
              activations.config.att_cap, att_out, exp_denominator_sums.data(),
              max_logits.data(), worker_workspace);

        } else if (attention_impl == AttentionImpl::kFlashTransposedQsInt16) {
          HWY_DASSERT(activations.int16_queries != nullptr);
          HWY_DASSERT(activations.q_scales != nullptr);
          int16_t* int16_queries_ptr =
              activations.int16_queries->data() +
              task_idx * max_queries_per_subtask * qkv_dim;
          float* q_scales_ptr =
              activations.q_scales->data() + task_idx * max_queries_per_subtask;
          CompressQueriesInt16(queries_ptrs_span, qkv_dim, int16_queries_ptr,
                               q_scales_ptr);
          DispatchTileFlashAttentionReturnExpSumsAndMaxLogitsInt16(
              kv_ptrs, sub_num_queries, int16_queries_ptr,
              hwy::Span<const float>(q_scales_ptr, sub_num_queries),
              hwy::Span<const size_t>(start_pos_per_query),
              hwy::Span<const size_t>(last_pos_per_query),
              activations.config.att_cap, att_out, exp_denominator_sums.data(),
              max_logits.data(), worker_workspace);
        } else if (attention_impl == AttentionImpl::kFlashMatrixAccumulation) {
          HWY_DASSERT(activations.bf16_queries != nullptr);
          BF16* bf16_queries_ptr = activations.bf16_queries->data() +
                                   task_idx * max_queries_per_subtask * qkv_dim;
          CompressAndTransposeQueriesMatrixAccumulationNonContiguous(
              queries_ptrs_span, bf16_queries_ptr, qkv_dim);
          DispatchTileFlashAttentionReturnExpSumsAndMaxLogitsMatrixAccumulation(
              kv_ptrs, sub_num_queries, bf16_queries_ptr,
              hwy::Span<const size_t>(start_pos_per_query),
              hwy::Span<const size_t>(last_pos_per_query),
              activations.config.att_cap, att_out, exp_denominator_sums.data(),
              max_logits.data(), worker_workspace);
        } else if (attention_impl == AttentionImpl::kInt8MatrixAccumulation) {
          HWY_DASSERT(activations.int8_queries != nullptr);
          HWY_DASSERT(activations.q_scales != nullptr);
          int8_t* int8_queries_ptr =
              activations.int8_queries->data() +
              task_idx * max_queries_per_subtask * qkv_dim;
          float* q_scales_ptr =
              activations.q_scales->data() + task_idx * max_queries_per_subtask;

          CompressAndQuantizeQueriesMatrixAccumulationInt8NonContiguous(
              queries_ptrs_span, int8_queries_ptr, q_scales_ptr, qkv_dim);

          DispatchTileFlashAttentionReturnExpSumsAndMaxLogitsMatrixAccumulationInt8(
              kv_ptrs, sub_num_queries, int8_queries_ptr,
              hwy::Span<const float>(q_scales_ptr, sub_num_queries),
              hwy::Span<const size_t>(start_pos_per_query),
              hwy::Span<const size_t>(last_pos_per_query),
              activations.config.att_cap, att_out, exp_denominator_sums.data(),
              max_logits.data(), worker_workspace);
        } else if (attention_impl == AttentionImpl::kFlashTransposedQsInt8) {
          HWY_DASSERT(activations.int8_queries != nullptr);
          HWY_DASSERT(activations.q_scales != nullptr);
          int8_t* int8_queries_ptr =
              activations.int8_queries->data() +
              task_idx * max_queries_per_subtask * qkv_dim;
          float* q_scales_ptr =
              activations.q_scales->data() + task_idx * max_queries_per_subtask;
          CompressQueriesInt8(queries_ptrs_span, qkv_dim, int8_queries_ptr,
                              q_scales_ptr);
          DispatchTileFlashAttentionReturnExpSumsAndMaxLogitsInt8(
              kv_ptrs, sub_num_queries, int8_queries_ptr,
              hwy::Span<const float>(q_scales_ptr, sub_num_queries),
              hwy::Span<const size_t>(start_pos_per_query),
              hwy::Span<const size_t>(last_pos_per_query),
              activations.config.att_cap, att_out, exp_denominator_sums.data(),
              max_logits.data(), worker_workspace);
        } else {
          HWY_DASSERT(activations.float_queries != nullptr);
          float* contiguous_queries_ptr =
              activations.float_queries->data() +
              task_idx * max_queries_per_subtask * qkv_dim;
          for (size_t i = 0; i < sub_num_queries; ++i) {
            hwy::CopyBytes(queries_ptrs_span[i],
                           contiguous_queries_ptr + i * qkv_dim,
                           qkv_dim * sizeof(float));
          }
          DispatchTileFlashAttentionReturnExpSumsAndMaxLogits(
              kv_ptrs, sub_num_queries, contiguous_queries_ptr,
              hwy::Span<const size_t>(start_pos_per_query),
              hwy::Span<const size_t>(last_pos_per_query),
              activations.config.att_cap, att_out, exp_denominator_sums.data(),
              max_logits.data(), worker_workspace);
        }

        // Lock-free atomic compare-and-exchange merge per 32-query group:
        // hand off group pointer immediately or claim and merge via SIMD.
        const int32_t my_idx = static_cast<int32_t>(task_idx);
        for (size_t g = 0; g < num_task_groups; ++g) {
          check_in_or_merge_group(g, my_idx);
        }
      });
}

void TiledAttention(AttentionImpl attention_impl, size_t num_tokens,
                    size_t layer_idx, const LayerWeightsPtrs& layer,
                    AttentionActivationsPtrs& activations, QBatch& qbatch,
                    MatMulEnv& env, int flags) {
  static const auto zone = env.ctx.profiler.AddZone(
      "Gen.TiledAttention", hwy::ProfilerFlags::kInclusive);
  PROFILER_ZONE3(env.ctx.profiler, hwy::Profiler::Thread(), zone);

  const LayerConfig& layer_config = layer.layer_config;

  HWY_DASSERT_M((layer_config.heads % layer_config.kv_heads) == 0,
                "query heads must be a multiple of key-value heads");
  (void)layer_config;  // only used in HWY_DASSERT

  const size_t active_qkv_dim = layer_config.heads * layer_config.qkv_dim;
  activations.q.OverrideCols(active_qkv_dim);
  activations.att_out.OverrideCols(active_qkv_dim);

  const Type kv_type = qbatch.KV(0).cache->compact_kv_cache_ptr.GetType();
  if (kv_type == Type::kBF16) {
    ComputeQKVTransposedTile<BF16>(num_tokens, layer_idx, layer, attention_impl,
                                   activations, qbatch, flags, env);
  } else if (kv_type == Type::kF32) {
    ComputeQKVTransposedTile<float>(num_tokens, layer_idx, layer,
                                    attention_impl, activations, qbatch, flags,
                                    env);
  } else if (qbatch.KV(0).cache->compact_kv_cache_ptr.GetType() ==
             Type::kInt8) {
    ComputeQKVTransposedTile<int8_t>(num_tokens, layer_idx, layer,
                                     attention_impl, activations, qbatch, flags,
                                     env);
  } else {
    HWY_ABORT(
        "Unsupported KV cache type: %d",
        static_cast<int>(qbatch.KV(0).cache->compact_kv_cache_ptr.GetType()));
  }
  RMSNormAndPositionalEncoding(num_tokens, qbatch, activations.q,
                               layer.query_norm_scale, layer_idx, activations,
                               env.ctx);
  LocalAttentionForAllHeadsTokensAndBatch(attention_impl, num_tokens, layer_idx,
                                          layer, activations, qbatch, env.ctx);
  SumHeads(layer, activations, env);
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();
