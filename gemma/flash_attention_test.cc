// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include <stddef.h>
#include <stdio.h>

#include <algorithm>  // std::max
#include <cmath>      // std::abs
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include "compression/types.h"
#include "gemma/activations.h"
#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "gemma/kv_cache.h"
#include "gemma/kv_transcoding.h"
#include "gemma/weights.h"
#include "ops/matmul.h"
#include "ops/ops.h"
#include "util/mat.h"
#include "util/test_util.h"
#include "util/threading_context.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/nanobenchmark.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/flash_attention_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"
#include "gemma/attention.h"
#include "gemma/configs.h"
#include "gemma/flash_attention.h"
#include "gemma/tiled_attention.h"
#include "ops/ops-inl.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// Old attention implementation for comparison, formerly in gemma/attention.cc.

// Computes Q.K scores, which are "logits" (or scores) stored to att.
// `k` is a strided view of the kv cache with dimensions [seq_len, qkv_dim].
static HWY_INLINE void QDotK(const size_t start_pos, const size_t last_pos,
                             const hwy::Divisor& div_seq_len,
                             const float* HWY_RESTRICT q,
                             const MatPtrT<KV_t>& k, float* HWY_RESTRICT att,
                             ThreadingContext& ctx, const size_t worker) {
  GCPP_ZONE(ctx, worker, Zones::kGenAttentionQDotK);
  const hn::ScalableTag<BF16> dbf;
  const size_t qkv_dim = k.Cols();
  HWY_ALIGN BF16 q_bf[kMaxQKVDim];

  CompressPerThread tls;
  const hn::ScalableTag<float> df;
  CompressTraits<BF16>::Compress(df, q, qkv_dim, tls, MakeSpan(q_bf, qkv_dim),
                                 0);

  // --seq_len must be large enough to avoid wraparound.
  HWY_DASSERT(last_pos < static_cast<size_t>(div_seq_len.GetDivisor()));
  for (size_t pos = start_pos; pos <= last_pos; ++pos) {
    const float score =
        Dot(dbf, MakeConstSpan(q_bf, qkv_dim), 0, k.Row(pos), qkv_dim);
    att[pos] = score;
  }
}

// Accumulates the sum of v (from `kv_cache`) * probability (`att`) into
// `att_out`. Equivalent in gemma/modules.py:
// encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)
// `v` is a strided view of the kv cache with dimensions [seq_len, qkv_dim].
static HWY_INLINE void WeightedSumV(
    const size_t start_pos, const size_t last_pos,
    const hwy::Divisor& div_seq_len, const float* HWY_RESTRICT att,
    const MatPtrT<KV_t>& v, float* HWY_RESTRICT att_out, ThreadingContext& ctx,
    const size_t worker) {
  // --seq_len must be large enough to avoid wraparound.
  HWY_DASSERT(last_pos < static_cast<size_t>(div_seq_len.GetDivisor()));
  // TODO: replace with MatMul(att, v) after it supports non-transposed B.
  MulByConstTo(att[start_pos], v.Row(start_pos), att_out, v.Cols(), ctx,
               worker);
  for (size_t pos = start_pos + 1; pos <= last_pos; ++pos) {
    MulByConstAndAdd(att[pos], v.Row(pos), att_out, v.Cols());
  }
}

// Calculates the attention outputs for a single q, which may be updated
// in place for RMSNorm.
void SingleDotSoftmaxWeightedSum(
    const size_t q_pos, const size_t kv_start_pos, const size_t kv_last_pos,
    float* HWY_RESTRICT q, const MatPtrT<KV_t>& k, const MatPtrT<KV_t>& v,
    const MatPtr& query_norm_scale, const size_t layer_idx,
    const AttentionActivationsPtrs& activations, float* HWY_RESTRICT att,
    float* HWY_RESTRICT att_out, const SMOptions& sm_options,
    ThreadingContext& ctx, const size_t worker) {
  const float att_cap = activations.config.att_cap;
  const float query_scale = activations.query_scale;
  // --seq_len must be large enough to avoid wraparound.
  HWY_DASSERT(kv_last_pos < activations.SeqLen());
  const LayerConfig& layer_config = activations.config.layer_configs[layer_idx];

  // Apply rope and scaling to Q.
  if (query_norm_scale.HasPtr()) {
    CallUpcasted(&query_norm_scale, [&](const auto* weights_t) {
      RMSNormInplace(weights_t->PackedScale1(), /*w_ofs=*/0, q,
                     layer_config.qkv_dim, ctx, worker);
    });
  }

  PositionalEncodingQK(q, layer_idx, activations, ctx, worker, q_pos,
                       query_scale);

  QDotK(kv_start_pos, kv_last_pos, activations.div_seq_len, q, k, att, ctx,
        worker);

  // SoftMax with optional SoftCap yields "probabilities" in att.
  const Logits logits(att, kv_last_pos + 1);
  MaybeLogitsSoftCap(att_cap, logits, ctx, worker);
  Softmax(logits, ctx, worker, /*temperature=*/1.0f, sm_options);

  WeightedSumV(kv_start_pos, kv_last_pos, activations.div_seq_len, att, v,
               att_out, ctx, worker);
}

void DotSoftmaxWeightedSum(const size_t num_tokens, const size_t layer_idx,
                           const MatPtr& query_norm_scale,
                           AttentionActivationsPtrs& activations,
                           MatPtrT<float> att_storage, QBatch& qbatch,
                           ThreadingContext& ctx) {
  GCPP_ZONE(ctx, 0, Zones::kGenAttentionDotSoftmaxWeightedSumInclusive);

  const hwy::Divisor div_qbatch(qbatch.Size());
  const LayerConfig& layer_config = activations.config.layer_configs[layer_idx];
  const size_t qkv_dim = layer_config.qkv_dim;

  // A "head group" in the context of GQA refers to a collection of query
  // heads that share the same key and value heads.
  const size_t kHeadGroups = layer_config.heads / layer_config.kv_heads;

  const size_t cache_layer_size = layer_config.CacheLayerSize();
  const size_t seq_len = activations.SeqLen();
  // All layers should have the same number of heads.
  HWY_DASSERT(activations.div_heads.GetDivisor() == layer_config.heads);

  // For each head/token/query, compute Q.K, softmax, and weighted V.
  const auto func = [&](const size_t task, size_t worker) HWY_ATTR {
    const size_t tq_idx = activations.div_heads.Divide(task);
    const size_t head = activations.div_heads.Remainder(task);
    GCPP_ZONE(ctx, worker, Zones::kGenAttentionDotSoftmaxWeightedSumPar);

    const size_t qi = div_qbatch.Remainder(tq_idx);
    const size_t token_idx = div_qbatch.Divide(tq_idx);
    auto& kv_cache = qbatch.KV(qi).kv_cache;

    // Find the token position in the query and calculate
    // the range of cache positions to attend to.
    const size_t pos = qbatch.Pos(qi) + token_idx;
    const size_t start_pos = StartPos(pos, activations.config, layer_idx);
    size_t last_pos = pos;
    const size_t prefix_end = qbatch.PrefixEnd(qi);
    if (prefix_end > 0 && prefix_end - 1 > last_pos) {
      // last_pos in QDotK and WeightedSumV is inclusive.
      last_pos = prefix_end - 1;
    }

    float* HWY_RESTRICT q = activations.q.Row(tq_idx) + head * qkv_dim;
    float* HWY_RESTRICT att = att_storage.Row(tq_idx) + head * seq_len;
    float* HWY_RESTRICT att_out =
        activations.att_out.Row(tq_idx) + head * qkv_dim;
    SMOptions sm_options{.max_out = activations.softmax_max.Row(tq_idx) + head,
                         .d_out = activations.softmax_d.Row(tq_idx) + head};

    // Make strided read-only views into the kv cache for
    // this query and head.
    const size_t head_offset = (head / kHeadGroups) * qkv_dim * 2;
    const size_t kv_head_offset = layer_idx * cache_layer_size + head_offset;
    MatPtrT<KV_t> k("k_view", Extents2D(seq_len, qkv_dim));
    k.SetPtr(kv_cache.Row(0) + kv_head_offset, kv_cache.Stride());
    MatPtrT<KV_t> v("v_view", Extents2D(seq_len, qkv_dim));
    v.SetPtr(kv_cache.Row(0) + kv_head_offset + qkv_dim, kv_cache.Stride());

    constexpr size_t offset = 0;  // placeholder, do not remove
    SingleDotSoftmaxWeightedSum(pos + offset, start_pos, last_pos, q, k, v,
                                query_norm_scale, layer_idx, activations, att,
                                att_out, sm_options, ctx, worker);
  };

  {
    PROFILER_ZONE("Gen.Attention.DotSoftmax.ForkJoin");
    // Full parallelism is helpful, kAcrossClusters is insufficient.
    HierarchicalParallelFor(
        num_tokens * div_qbatch.GetDivisor() * layer_config.heads, ctx,
        Callers::kAttDotSoftmaxWeightedSum, func);
  }
}

using FloatPtr = hwy::AlignedFreeUniquePtr<float[]>;

template <typename T>
void SetMat(const size_t offset, MatPtrT<T>& mat) {
  const size_t kOuter = mat.Extents().rows;
  const size_t kInner = mat.Extents().cols;
  const float i_scale = 1.0f / kInner;
  const float j_scale = 1.0f / kOuter;
  for (size_t i = 0; i < kOuter; ++i) {
    T* HWY_RESTRICT row = mat.Row(i);
    for (size_t j = 0; j < kInner; ++j) {
      row[j] = hwy::ConvertScalarTo<T>(
          static_cast<float>((i * kInner * i_scale + (j + offset) * j_scale)));
    }
  }
}

std::unique_ptr<MatStorageT<float>> MakeCopyOfMat(const MatPtrT<float>& mat,
                                                  const Allocator& allocator) {
  auto copy = std::make_unique<MatStorageT<float>>("TestMat", mat.Extents(),
                                                   allocator, MatPadding::kOdd);
  CopyMat(mat, *copy);
  return copy;
}

void AssertClose(const MatPtrT<float>& a, const MatPtrT<float>& b) {
  size_t failures = 0;
  // Avoid comparing the padding bytes, which are uninitialized.
  for (size_t r = 0; r < a.Rows(); ++r) {
    const float* HWY_RESTRICT a_row = a.Row(r);
    const float* HWY_RESTRICT b_row = b.Row(r);
    for (size_t c = 0; c < a.Cols(); ++c) {
      float rel_abs_delta = std::abs(a_row[c] - b_row[c]);
      if (rel_abs_delta > 0.0f) {
        rel_abs_delta /= std::max(std::abs(a_row[c]), std::abs(b_row[c]));
      }
      if (rel_abs_delta >= 1.5e-3) {
        if (failures < 5) {
          EXPECT_LT(rel_abs_delta, 1.5e-3)
              << "a[" << r << "," << c << "]=" << a_row[c] << ", b[" << r << ","
              << c << "]=" << b_row[c];
        }
        failures++;
      }
    }
  }
  if (failures > 5) {
    ADD_FAILURE() << "Truncated " << (failures - 5)
                  << " additional failures in AssertClose.";
  }
}

template <typename T>
void PopulateTestKVCache(MatStorageT<T>& kv, gcpp::KVEncoding encoding,
                         size_t qkv_dim) {
  gcpp::DecodedTile tile(qkv_dim, gcpp::KVCache::kTileSize);

  size_t num_tiles = kv.Rows();
  float unpredictable = hwy::Unpredictable1() * 0.01f;
  for (size_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    for (size_t in_tile = 0; in_tile < gcpp::KVCache::kTileSize; ++in_tile) {
      size_t i = tile_idx * gcpp::KVCache::kTileSize + in_tile;
      for (size_t j = 0; j < qkv_dim; ++j) {
        tile.k_elem(in_tile, j) = unpredictable * (i + 1) / (j + 1);
        tile.v_elem(in_tile, j) = unpredictable * 2 * (i + 1) / (j + 1);
      }
    }
    size_t row_bytes = kv.Cols() * sizeof(T);
    HWY_ASSERT(gcpp::EncodeTile(
        encoding, tile, qkv_dim,
        hwy::Span<char>(reinterpret_cast<char*>(kv.Row(tile_idx)), row_bytes)));
  }
}

AlignedFloatVector PopulateTestQueries(size_t num_queries, size_t qkv_dim) {
  AlignedFloatVector q_all(num_queries * qkv_dim);
  const float unpredictable_factor = 0.01f * hwy::Unpredictable1();
  for (size_t i = 0; i < num_queries; ++i) {
    for (size_t j = 0; j < qkv_dim; ++j) {
      q_all[i * qkv_dim + j] = unpredictable_factor * (i + 1) / (j + 1);
    }
  }
  return q_all;
}

struct AttentionTestEnv {
  AttentionTestEnv(size_t num_queries, size_t kv_seq_len, size_t qkv_dim,
                   AttentionImpl attention_impl);
};

void TestFlashAttention(size_t target_parallelism,
                        AttentionImpl attention_impl) {
  ThreadingArgs threading_args;
  ThreadingContext ctx(threading_args);
  constexpr size_t kSeqLen = 1024;
  ModelConfig config(Model::GEMMA2_2B, Type::kF32, PromptWrapping::GEMMA_PT);
  config.att_cap = 1024.0f;
  TensorInfoRegistry tensor_info_registry(config);
  const LayerConfig& layer_config = config.layer_configs[0];
  const LayerWeightsPtrs layers(0, layer_config, tensor_info_registry);
  InferenceArgs inference_args;
  // attention_impl must be old in order for the att intermediate to be
  // allocated for the old attention.
  inference_args.attention_impl = "old";
  RuntimeConfig runtime_config;
  inference_args.CopyTo(runtime_config);
  KVCache kv_cache(config, inference_args, ctx.allocator);
  MatMulEnv env(ctx);
  Activations activations(runtime_config, config,
                          runtime_config.prefill_tbatch_size, kv_cache.SeqLen(),
                          env.ctx, env.row_ptrs);
  std::vector<int> tokens(kSeqLen);
  std::iota(tokens.begin(), tokens.end(), 1);
  PromptTokens prompt(tokens);
  AllQueries all_queries(hwy::Span<const PromptTokens>(&prompt, 1),
                         hwy::Span<KVCache>(&kv_cache, 1));
  QBatch qbatch(/*start=*/0, /*max_size=*/kSeqLen, all_queries);
  const size_t batch_size = kSeqLen;
  std::vector<hwy::AlignedFreeUniquePtr<uint8_t*[]>> row_ptrs;
  AttentionActivations attention_storage(
      config, layer_config, batch_size, kSeqLen, runtime_config,
      ctx.pools.MaxWorkers(), ctx.allocator, row_ptrs);
  AttentionActivationsPtrs att_activations(config, kSeqLen, attention_storage);
  MatStorageT<float> att("att",
                         Extents2D(batch_size, layer_config.heads * kSeqLen),
                         ctx.allocator, MatPadding::kOdd);
  const size_t qkv_dim = layer_config.qkv_dim;
  ASSERT_EQ(qkv_dim, 256);
  const hwy::Divisor div_qbatch(qbatch.Size());
  // A "head group" in the context of GQA refers to a collection of query
  // heads that share the same key and value heads.
  const size_t kHeadGroups = layer_config.heads / layer_config.kv_heads;
  const size_t seq_len =
      static_cast<size_t>(att_activations.div_seq_len.GetDivisor());
  MaybeReshapeCache(qbatch.KV(0).cache->KOrVDefaultCols(),
                    qbatch.KV(0).k_cache);
  MaybeReshapeCache(qbatch.KV(0).cache->KOrVDefaultCols(),
                    qbatch.KV(0).v_cache);
  auto& kvc = qbatch.KV(0).kv_cache;
  using DF = hn::ScalableTag<float>;
  const DF df;
  const size_t kNF = hn::Lanes(df);
  const size_t kFloatsPerTile = 2 * kNF;
  for (size_t h = 0; h < layer_config.heads; ++h) {
    // Make strided views into the kv cache for
    // this query and head.
    const size_t head_offset = (h / kHeadGroups) * qkv_dim * 2;
    MatPtrT<KV_t> k("k_view", Extents2D(seq_len, qkv_dim));
    k.SetPtr(kvc.Row(0) + head_offset, kvc.Stride());
    MatPtrT<KV_t> v("v_view", Extents2D(seq_len, qkv_dim));
    v.SetPtr(kvc.Row(0) + head_offset + qkv_dim, kvc.Stride());
    SetMat(h + layer_config.heads, k);
    SetMat(h + layer_config.heads * 2, v);
    for (size_t p = 0; p < tokens.size(); ++p) {
      KV_t* HWY_RESTRICT k_src = k.Row(p);
      KV_t* HWY_RESTRICT k_dest =
          qbatch.KV(0).k_cache.Row(p / kFloatsPerTile) +
          qbatch.KV(0).cache->KOffset(0, h / kHeadGroups, kNF, p);
      KV_t* HWY_RESTRICT v_dest =
          qbatch.KV(0).v_cache.Row(p / kFloatsPerTile) +
          qbatch.KV(0).cache->VOffset(0, h / kHeadGroups, kNF, p);

      TransposeKVCacheRow(k_src, k_dest, v_dest, qkv_dim);
    }
  }
  SetMat(1, att_activations.q);
  DotSoftmaxWeightedSum(tokens.size(), 0, layers.query_norm_scale,
                        att_activations, att, qbatch, ctx);
  // Copy the output to saved_att to allow for comparison.
  auto saved_att = MakeCopyOfMat(att_activations.att_out, ctx.allocator);
  SetMat(1, att_activations.q);
  const size_t total_tasks =
      tokens.size() * div_qbatch.GetDivisor() * layer_config.heads;
  const size_t kVTileSize = GetVTileSize(kNF, kHeadGroups, tokens.size(),
                                         total_tasks, target_parallelism);
  printf("FlashAttention: parallelism=%zu, kNF=%zu, kVTileSize=%zu, mode %s\n",
         target_parallelism, kNF, kVTileSize,
         GetAttentionImplName(attention_impl).c_str());
  FlashAttention(tokens.size(), target_parallelism, 0, layers.query_norm_scale,
                 att_activations, qbatch, ctx, attention_impl);
  AssertClose(att_activations.att_out, *saved_att);
  ctx.profiler.PrintResults();
}

void TestAttention() {
  TestFlashAttention(8192, AttentionImpl::kFlash);
  TestFlashAttention(2048, AttentionImpl::kFlash);
  TestFlashAttention(256, AttentionImpl::kFlash);
}

const std::vector<float> exp_denominator_sums_gold = {
    58.722088f, 58.445938f, 58.17153f,  57.89886f,
    58.580994f, 58.302643f, 58.026085f, 57.751308f};
const std::vector<float> max_logits_gold = {
    0.009613638f, 0.019227259f, 0.02884084f, 0.038454376f,
    0.04888253f,  0.058658823f, 0.06843502f, 0.078211054f};
const std::vector<float> att_out_gold = {
    0.600945, 0.300472, 0.200315, 0.150236, 0.120189, 0.100158, 0.085849,
    0.075118, 0.066772, 0.060095, 0.054631, 0.050079, 0.046227, 0.042925,
    0.040063, 0.037559, 0.035350, 0.033386, 0.031629, 0.030047, 0.028616,
    0.027316, 0.026128, 0.025039, 0.024038, 0.023113, 0.022257, 0.021462,
    0.020722, 0.020032, 0.019385, 0.018780, 0.018210, 0.017675, 0.017170,
    0.016693, 0.016242, 0.015814, 0.015409, 0.015024, 0.014657, 0.014308,
    0.013975, 0.013658, 0.013354, 0.013064, 0.012786, 0.012520, 0.012264,
    0.012019, 0.011783, 0.011557, 0.011339, 0.011129, 0.010926, 0.010731,
    0.010543, 0.010361, 0.010186, 0.010016, 0.009852, 0.009693, 0.009539,
    0.009390, 0.601890, 0.300945, 0.200630, 0.150473, 0.120378, 0.100315,
    0.085984, 0.075236, 0.066877, 0.060189, 0.054717, 0.050158, 0.046299,
    0.042992, 0.040126, 0.037618, 0.035405, 0.033438, 0.031678, 0.030095,
    0.028661, 0.027359, 0.026169, 0.025079, 0.024076, 0.023150, 0.022292,
    0.021496, 0.020755, 0.020063, 0.019416, 0.018809, 0.018239, 0.017703,
    0.017197, 0.016719, 0.016267, 0.015839, 0.015433, 0.015047, 0.014680,
    0.014331, 0.013997, 0.013679, 0.013375, 0.013085, 0.012806, 0.012539,
    0.012283, 0.012038, 0.011802, 0.011575, 0.011356, 0.011146, 0.010943,
    0.010748, 0.010559, 0.010377, 0.010202, 0.010032, 0.009867, 0.009708,
    0.009554, 0.009405, 0.602835, 0.301418, 0.200945, 0.150709, 0.120567,
    0.100473, 0.086119, 0.075354, 0.066982, 0.060284, 0.054803, 0.050236,
    0.046372, 0.043060, 0.040189, 0.037677, 0.035461, 0.033491, 0.031728,
    0.030142, 0.028706, 0.027402, 0.026210, 0.025118, 0.024113, 0.023186,
    0.022327, 0.021530, 0.020787, 0.020095, 0.019446, 0.018839, 0.018268,
    0.017730, 0.017224, 0.016745, 0.016293, 0.015864, 0.015457, 0.015071,
    0.014703, 0.014353, 0.014019, 0.013701, 0.013396, 0.013105, 0.012826,
    0.012559, 0.012303, 0.012057, 0.011820, 0.011593, 0.011374, 0.011164,
    0.010961, 0.010765, 0.010576, 0.010394, 0.010218, 0.010047, 0.009883,
    0.009723, 0.009569, 0.009419, 0.603780, 0.301890, 0.201260, 0.150945,
    0.120756, 0.100630, 0.086254, 0.075473, 0.067087, 0.060378, 0.054889,
    0.050315, 0.046445, 0.043127, 0.040252, 0.037736, 0.035516, 0.033543,
    0.031778, 0.030189, 0.028751, 0.027445, 0.026251, 0.025158, 0.024151,
    0.023222, 0.022362, 0.021564, 0.020820, 0.020126, 0.019477, 0.018868,
    0.018296, 0.017758, 0.017251, 0.016772, 0.016318, 0.015889, 0.015482,
    0.015095, 0.014726, 0.014376, 0.014041, 0.013722, 0.013417, 0.013126,
    0.012846, 0.012579, 0.012322, 0.012076, 0.011839, 0.011611, 0.011392,
    0.011181, 0.010978, 0.010782, 0.010593, 0.010410, 0.010234, 0.010063,
    0.009898, 0.009738, 0.009584, 0.009434, 0.614887, 0.307443, 0.204962,
    0.153722, 0.122977, 0.102481, 0.087841, 0.076861, 0.068321, 0.061489,
    0.055899, 0.051241, 0.047299, 0.043920, 0.040992, 0.038430, 0.036170,
    0.034160, 0.032362, 0.030744, 0.029280, 0.027949, 0.026734, 0.025620,
    0.024595, 0.023649, 0.022774, 0.021960, 0.021203, 0.020496, 0.019835,
    0.019215, 0.018633, 0.018085, 0.017568, 0.017080, 0.016619, 0.016181,
    0.015766, 0.015372, 0.014997, 0.014640, 0.014300, 0.013975, 0.013664,
    0.013367, 0.013083, 0.012810, 0.012549, 0.012298, 0.012057, 0.011825,
    0.011602, 0.011387, 0.011180, 0.010980, 0.010787, 0.010601, 0.010422,
    0.010248, 0.010080, 0.009918, 0.009760, 0.009608, 0.615864, 0.307932,
    0.205288, 0.153966, 0.123173, 0.102644, 0.087981, 0.076983, 0.068429,
    0.061586, 0.055988, 0.051322, 0.047374, 0.043990, 0.041058, 0.038491,
    0.036227, 0.034215, 0.032414, 0.030793, 0.029327, 0.027994, 0.026777,
    0.025661, 0.024635, 0.023687, 0.022810, 0.021995, 0.021237, 0.020529,
    0.019867, 0.019246, 0.018663, 0.018114, 0.017596, 0.017107, 0.016645,
    0.016207, 0.015791, 0.015397, 0.015021, 0.014663, 0.014322, 0.013997,
    0.013686, 0.013388, 0.013103, 0.012830, 0.012569, 0.012317, 0.012076,
    0.011844, 0.011620, 0.011405, 0.011198, 0.010998, 0.010805, 0.010618,
    0.010438, 0.010264, 0.010096, 0.009933, 0.009776, 0.009623, 0.616841,
    0.308421, 0.205614, 0.154210, 0.123368, 0.102807, 0.088120, 0.077105,
    0.068538, 0.061684, 0.056076, 0.051403, 0.047449, 0.044060, 0.041123,
    0.038553, 0.036285, 0.034269, 0.032465, 0.030842, 0.029373, 0.028038,
    0.026819, 0.025702, 0.024674, 0.023725, 0.022846, 0.022030, 0.021270,
    0.020561, 0.019898, 0.019276, 0.018692, 0.018142, 0.017624, 0.017134,
    0.016671, 0.016233, 0.015816, 0.015421, 0.015045, 0.014687, 0.014345,
    0.014019, 0.013708, 0.013410, 0.013124, 0.012851, 0.012589, 0.012337,
    0.012095, 0.011862, 0.011639, 0.011423, 0.011215, 0.011015, 0.010822,
    0.010635, 0.010455, 0.010281, 0.010112, 0.009949, 0.009791, 0.009638,
    0.617818, 0.308909, 0.205939, 0.154455, 0.123564, 0.102970, 0.088260,
    0.077227, 0.068646, 0.061782, 0.056165, 0.051485, 0.047524, 0.044130,
    0.041188, 0.038614, 0.036342, 0.034323, 0.032517, 0.030891, 0.029420,
    0.028083, 0.026862, 0.025742, 0.024713, 0.023762, 0.022882, 0.022065,
    0.021304, 0.020594, 0.019930, 0.019307, 0.018722, 0.018171, 0.017652,
    0.017162, 0.016698, 0.016258, 0.015841, 0.015445, 0.015069, 0.014710,
    0.014368, 0.014041, 0.013729, 0.013431, 0.013145, 0.012871, 0.012609,
    0.012356, 0.012114, 0.011881, 0.011657, 0.011441, 0.011233, 0.011032,
    0.010839, 0.010652, 0.010471, 0.010297, 0.010128, 0.009965, 0.009807,
    0.009653};

template <typename KV_T>
void RunTiledFlashAttentionTest(gcpp::KVEncoding kv_encoding,
                                AttentionImpl attention_impl, float tol,
                                float tol_exp, float tol_max,
                                float att_cap = 10.0f) {
  size_t qkv_dim = 64;
  size_t kv_seq_len = 60;  // number of tokens we will attend to.
  // Not divisible by tiles size to test the padding logic.
  size_t padded_kv_seq_len =
      hwy::RoundUpTo(kv_seq_len, gcpp::KVCache::kTileSize);
  size_t num_queries = 8;
  size_t num_queries_per_timestep = 4;
  size_t num_tokens = num_queries / num_queries_per_timestep;
  size_t kv_seq_end =
      kv_seq_len - hwy::DivCeil(num_queries, num_queries_per_timestep);
  ThreadingArgs threading_args;
  ThreadingContext ctx(threading_args);

  size_t num_tiles = padded_kv_seq_len / gcpp::KVCache::kTileSize;
  size_t tile_size_bytes = *gcpp::GetTileSizeBytes(kv_encoding, qkv_dim);
  size_t tile_size_in_elements = tile_size_bytes / sizeof(KV_T);

  MatStorageT<KV_T> kv("kv", Extents2D(num_tiles, tile_size_in_elements),
                       ctx.allocator, MatPadding::kPacked);
  PopulateTestKVCache(kv, kv_encoding, qkv_dim);

  AlignedFloatVector q_all = PopulateTestQueries(num_queries, qkv_dim);

  MatStorageT<float> att_out("att_out", Extents2D(num_queries, qkv_dim),
                             ctx.allocator, MatPadding::kPacked);
  using DF = hn::ScalableTag<float>;
  const DF df;
  HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(df);
  size_t num_queries_rounded_to_lanes = hwy::RoundUpTo(num_queries, lanes);
  std::vector<float> exp_denominator_sums(num_queries_rounded_to_lanes);
  std::vector<float> max_logits(num_queries_rounded_to_lanes);
  for (size_t i = 0; i < num_queries; ++i) {
    hwy::ZeroBytes(att_out.Row(i),
                   qkv_dim * sizeof(decltype(att_out.Row(i)[0])));
    exp_denominator_sums[i] = 0.0f;
    max_logits[i] = -std::numeric_limits<float>::max() / 2.0f;
  }
  std::vector<size_t, hwy::AlignedAllocator<size_t>> start_pos_per_query;
  std::vector<size_t, hwy::AlignedAllocator<size_t>> last_pos_per_query;
  start_pos_per_query.reserve(num_queries);
  last_pos_per_query.reserve(num_queries);
  for (size_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
    ssize_t query_last_pos = kv_seq_end + token_idx;
    ssize_t query_start_pos =
        std::max(query_last_pos - 100000 + 1, static_cast<ssize_t>(0));
    for (size_t q_head_idx = 0; q_head_idx < num_queries_per_timestep;
         ++q_head_idx) {
      start_pos_per_query.push_back(query_start_pos);
      last_pos_per_query.push_back(query_last_pos);
    }
  }

  hwy::Span<const MatPtr> kvs(&kv, 1);
  if (attention_impl == AttentionImpl::kFlashTransposedQsBF16) {
    std::vector<BF16, hwy::AlignedAllocator<BF16>> bf16_queries(num_queries *
                                                                qkv_dim);
    CompressQueriesBF16Contiguous(q_all.data(), qkv_dim, num_queries,
                                  bf16_queries.data());

    DispatchTileFlashAttentionReturnExpSumsAndMaxLogitsBF16(
        kvs, num_queries, bf16_queries.data(),
        hwy::Span<const size_t>(start_pos_per_query),
        hwy::Span<const size_t>(last_pos_per_query), att_cap, att_out,
        exp_denominator_sums.data(), max_logits.data());
  } else if (attention_impl == AttentionImpl::kFlashTransposedQsInt16) {
    std::vector<int16_t, hwy::AlignedAllocator<int16_t>> int16_queries(
        num_queries * qkv_dim);
    AlignedFloatVector q_scales(num_queries);
    CompressQueriesInt16Contiguous(q_all.data(), qkv_dim, num_queries,
                                   int16_queries.data(), q_scales.data());

    DispatchTileFlashAttentionReturnExpSumsAndMaxLogitsInt16(
        kvs, num_queries, int16_queries.data(), q_scales,
        hwy::Span<const size_t>(start_pos_per_query),
        hwy::Span<const size_t>(last_pos_per_query), att_cap, att_out,
        exp_denominator_sums.data(), max_logits.data());
  } else if (attention_impl == AttentionImpl::kInt8MatrixAccumulation) {
    size_t num_queries_rounded = hwy::RoundUpTo(num_queries, 2);
    hwy::AlignedVector<int8_t> int8_queries(num_queries_rounded * qkv_dim);
    hwy::AlignedVector<float> q_scales(num_queries_rounded);
    CompressAndQuantizeQueriesMatrixAccumulationInt8(
        q_all.data(), int8_queries.data(), q_scales.data(), num_queries,
        qkv_dim);

    DispatchTileFlashAttentionReturnExpSumsAndMaxLogitsMatrixAccumulationInt8(
        kvs, num_queries, int8_queries.data(), q_scales,
        hwy::Span<const size_t>(start_pos_per_query),
        hwy::Span<const size_t>(last_pos_per_query), att_cap, att_out,
        exp_denominator_sums.data(), max_logits.data());
  } else if (attention_impl == AttentionImpl::kFlashMatrixAccumulation) {
    size_t num_queries_rounded = hwy::RoundUpTo(num_queries, 2);
    hwy::AlignedVector<BF16> bf16_queries(num_queries_rounded * qkv_dim);
    CompressAndTransposeQueriesMatrixAccumulation(
        q_all.data(), bf16_queries.data(), num_queries, qkv_dim);

    DispatchTileFlashAttentionReturnExpSumsAndMaxLogitsMatrixAccumulation(
        kvs, num_queries, bf16_queries.data(),
        hwy::Span<const size_t>(start_pos_per_query),
        hwy::Span<const size_t>(last_pos_per_query), att_cap, att_out,
        exp_denominator_sums.data(), max_logits.data());
  } else {
    DispatchTileFlashAttentionReturnExpSumsAndMaxLogits(
        kvs, num_queries, q_all.data(),
        hwy::Span<const size_t>(start_pos_per_query),
        hwy::Span<const size_t>(last_pos_per_query), att_cap, att_out,
        exp_denominator_sums.data(), max_logits.data());
  }

  // PrintMatPtr(att_out);
  size_t failures = 0;
  for (size_t i = 0; i < num_queries; ++i) {
    std::cerr << "exp_d: " << exp_denominator_sums[i]
              << " max_logit: " << max_logits[i] << std::endl;

    float diff_exp =
        std::abs(exp_denominator_sums[i] - exp_denominator_sums_gold[i]);
    if (diff_exp >= tol_exp) {
      if (failures < 5) {
        EXPECT_NEAR(exp_denominator_sums[i], exp_denominator_sums_gold[i],
                    tol_exp)
            << "i=" << i;
      }
      failures++;
    }

    float diff_max = std::abs(max_logits[i] - max_logits_gold[i]);
    if (diff_max >= tol_max) {
      if (failures < 5) {
        EXPECT_NEAR(max_logits[i], max_logits_gold[i], tol_max) << "i=" << i;
      }
      failures++;
    }

    for (size_t j = 0; j < qkv_dim; ++j) {
      if (j == 0 && attention_impl == AttentionImpl::kFlashTransposedQsBF16 &&
          kv_encoding == gcpp::KVEncoding::kBF16TwoTranspositions) {
        std::cerr << "att_out[0][" << j << "]=" << att_out.Row(i)[j]
                  << " gold=" << att_out_gold[i * qkv_dim + j] << "\n";
      }

      float diff_out =
          std::abs(att_out.Row(i)[j] - att_out_gold[i * qkv_dim + j]);
      if (diff_out >= tol) {
        if (failures < 5) {
          EXPECT_NEAR(att_out.Row(i)[j], att_out_gold[i * qkv_dim + j], tol)
              << "i=" << i << " j=" << j;
        }
        failures++;
      }
    }
  }
  if (failures > 5) {
    ADD_FAILURE() << "Truncated " << (failures - 5)
                  << " additional failures in RunTiledFlashAttentionTest.";
  }
}

void TestTiledFlashAttention() {
  RunTiledFlashAttentionTest<float>(gcpp::KVEncoding::kF32,
                                    AttentionImpl::kFlash, 1e-5f, 1e-3f, 5e-6f);
}

void TestTiledFlashAttentionBF16() {
  RunTiledFlashAttentionTest<BF16>(gcpp::KVEncoding::kBF16TwoTranspositions,
                                   AttentionImpl::kFlashTransposedQsBF16, 1e-3f,
                                   4e-2f, 1e-3f);
}

void TestTiledFlashAttentionInt8() {
  RunTiledFlashAttentionTest<int8_t>(gcpp::KVEncoding::kInt8,
                                     AttentionImpl::kFlash, 5e-3f, 2e-2f,
                                     1e-3f);
}

void TestTiledFlashAttentionInt8BF16() {
  RunTiledFlashAttentionTest<int8_t>(gcpp::KVEncoding::kInt8TwoTranspositions,
                                     AttentionImpl::kFlashTransposedQsBF16,
                                     5e-3f, 2e-2f, 1e-3f);
}

void TestTiledFlashAttentionInt8Int16() {
  RunTiledFlashAttentionTest<int8_t>(gcpp::KVEncoding::kInt8TwoTranspositions,
                                     AttentionImpl::kFlashTransposedQsInt16,
                                     5e-3f, 2e-2f, 1e-3f);
}

void TestTiledFlashAttentionBF16MatrixAccumulation() {
  const hn::ScalableTag<hwy::bfloat16_t> dbf;
  if (hn::Lanes(dbf) > 32) {
    GTEST_SKIP() << "Skipping MatrixAccumulation test for target with register "
                    "size > 512-bit.";
    return;
  }

  const float tol = 2.0e-3f;
  const float tol_exp = 4e-2f;
  const float tol_max = 1e-3f;

  RunTiledFlashAttentionTest<BF16>(gcpp::KVEncoding::kBF16MatrixAccumulation,
                                   AttentionImpl::kFlashMatrixAccumulation, tol,
                                   tol_exp, tol_max, 0.0f);
}

void TestTiledFlashAttentionInt8MatrixAccumulation() {
  const hn::ScalableTag<hwy::bfloat16_t> dbf;
  if (hn::Lanes(dbf) > 32) {
    GTEST_SKIP() << "Skipping MatrixAccumulation test for target with register "
                    "size > 512-bit.";
    return;
  }

  // INT8 has slightly larger error due to quantization, so we use slightly
  // larger tolerances
  const float tol = 1.5e-1f;
  const float tol_exp = 2e-1f;
  const float tol_max = 5e-2f;

  RunTiledFlashAttentionTest<int8_t>(gcpp::KVEncoding::kInt8MatrixAccumulation,
                                     AttentionImpl::kInt8MatrixAccumulation,
                                     tol, tol_exp, tol_max, 0.0f);
}

template <typename KV_T>
void RunTiledFlashAttentionDifferentialTest(
    size_t kv_seq_len, float tol, float tol_exp, float tol_max,
    gcpp::KVEncoding ref_encoding, gcpp::KVEncoding opt_encoding,
    AttentionImpl opt_impl, const char* type_name) {
  const hn::ScalableTag<hwy::bfloat16_t> dbf;
  if (hn::Lanes(dbf) > 32) {
    GTEST_SKIP() << "Skipping MatrixAccumulation test for target with register "
                    "size > 512-bit.";
    return;
  }

  size_t qkv_dim = 64;
  size_t padded_kv_seq_len =
      hwy::RoundUpTo(kv_seq_len, gcpp::KVCache::kTileSize);
  float att_cap = 0.0f;
  size_t num_queries = 37;
  size_t num_queries_per_timestep = 1;
  size_t num_tokens = num_queries / num_queries_per_timestep;
  size_t kv_seq_end =
      kv_seq_len - hwy::DivCeil(num_queries, num_queries_per_timestep);
  ThreadingArgs threading_args;
  ThreadingContext ctx(threading_args);

  // Set up reference cache
  size_t ref_tile_size_bytes = *gcpp::GetTileSizeBytes(ref_encoding, qkv_dim);
  size_t ref_tile_size_in_elements = ref_tile_size_bytes / sizeof(KV_T);
  MatStorageT<KV_T> kv_ref(
      "kv_ref",
      Extents2D(padded_kv_seq_len / gcpp::KVCache::kTileSize,
                ref_tile_size_in_elements),
      ctx.allocator, MatPadding::kPacked);
  PopulateTestKVCache(kv_ref, ref_encoding, qkv_dim);

  AlignedFloatVector q_all_ref = PopulateTestQueries(num_queries, qkv_dim);
  std::vector<BF16, hwy::AlignedAllocator<BF16>> bf16_queries_ref(num_queries *
                                                                  qkv_dim);
  CompressQueriesBF16Contiguous(q_all_ref.data(), qkv_dim, num_queries,
                                bf16_queries_ref.data());

  MatStorageT<float> att_out_ref("att_out_ref", Extents2D(num_queries, qkv_dim),
                                 ctx.allocator, MatPadding::kPacked);

  HWY_LANES_CONSTEXPR size_t lanes = 4;
  size_t num_queries_rounded_to_lanes = hwy::RoundUpTo(num_queries, lanes);
  std::vector<float> exp_denominator_sums_ref(num_queries_rounded_to_lanes);
  std::vector<float> max_logits_ref(num_queries_rounded_to_lanes);
  for (size_t i = 0; i < num_queries; ++i) {
    hwy::ZeroBytes(att_out_ref.Row(i),
                   qkv_dim * sizeof(decltype(att_out_ref.Row(i)[0])));
    exp_denominator_sums_ref[i] = 0.0f;
    max_logits_ref[i] = -std::numeric_limits<float>::max() / 2.0f;
  }

  // Set up Optimized cache (Matrix Accumulation)
  size_t opt_tile_size_bytes = *gcpp::GetTileSizeBytes(opt_encoding, qkv_dim);
  size_t opt_tile_size_in_elements = opt_tile_size_bytes / sizeof(KV_T);
  MatStorageT<KV_T> kv(
      "kv",
      Extents2D(padded_kv_seq_len / gcpp::KVCache::kTileSize,
                opt_tile_size_in_elements),
      ctx.allocator, MatPadding::kPacked);
  PopulateTestKVCache(kv, opt_encoding, qkv_dim);

  AlignedFloatVector q_all = PopulateTestQueries(num_queries, qkv_dim);

  hwy::AlignedVector<int8_t> int8_queries;
  std::vector<BF16, hwy::AlignedAllocator<BF16>> bf16_queries;
  hwy::AlignedVector<float> q_scales;

  size_t num_queries_rounded = hwy::RoundUpTo(num_queries, 2);

  if (opt_impl == AttentionImpl::kInt8MatrixAccumulation) {
    int8_queries.resize(num_queries_rounded * qkv_dim);
    q_scales.resize(num_queries_rounded);
    CompressAndQuantizeQueriesMatrixAccumulationInt8(
        q_all.data(), int8_queries.data(), q_scales.data(), num_queries,
        qkv_dim);
  } else {
    bf16_queries.resize(num_queries_rounded * qkv_dim);
    CompressAndTransposeQueriesMatrixAccumulation(
        q_all.data(), bf16_queries.data(), num_queries, qkv_dim);
  }

  MatStorageT<float> att_out("att_out", Extents2D(num_queries, qkv_dim),
                             ctx.allocator, MatPadding::kPacked);

  std::vector<float> exp_denominator_sums(num_queries_rounded_to_lanes);
  std::vector<float> max_logits(num_queries_rounded_to_lanes);
  for (size_t i = 0; i < num_queries; ++i) {
    hwy::ZeroBytes(att_out.Row(i),
                   qkv_dim * sizeof(decltype(att_out.Row(i)[0])));
    exp_denominator_sums[i] = 0.0f;
    max_logits[i] = -std::numeric_limits<float>::max() / 2.0f;
  }

  std::vector<size_t, hwy::AlignedAllocator<size_t>> start_pos_per_query;
  std::vector<size_t, hwy::AlignedAllocator<size_t>> last_pos_per_query;
  start_pos_per_query.reserve(num_queries);
  last_pos_per_query.reserve(num_queries);
  for (size_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
    ssize_t query_last_pos = kv_seq_end + token_idx;
    ssize_t query_start_pos =
        std::max(query_last_pos - 100000 + 1, static_cast<ssize_t>(0));
    for (size_t q_head_idx = 0; q_head_idx < num_queries_per_timestep;
         ++q_head_idx) {
      start_pos_per_query.push_back(query_start_pos);
      last_pos_per_query.push_back(query_last_pos);
    }
  }

  // Run reference
  hwy::Span<const MatPtr> kvs_ref(&kv_ref, 1);
  DispatchTileFlashAttentionReturnExpSumsAndMaxLogitsBF16(
      kvs_ref, num_queries, bf16_queries_ref.data(),
      hwy::Span<const size_t>(start_pos_per_query),
      hwy::Span<const size_t>(last_pos_per_query), att_cap, att_out_ref,
      exp_denominator_sums_ref.data(), max_logits_ref.data());

  // Run optimized
  hwy::Span<const MatPtr> kvs(&kv, 1);
  if (opt_impl == AttentionImpl::kInt8MatrixAccumulation) {
    DispatchTileFlashAttentionReturnExpSumsAndMaxLogitsMatrixAccumulationInt8(
        kvs, num_queries, int8_queries.data(), q_scales,
        hwy::Span<const size_t>(start_pos_per_query),
        hwy::Span<const size_t>(last_pos_per_query), att_cap, att_out,
        exp_denominator_sums.data(), max_logits.data());
  } else {
    DispatchTileFlashAttentionReturnExpSumsAndMaxLogitsMatrixAccumulation(
        kvs, num_queries, bf16_queries.data(),
        hwy::Span<const size_t>(start_pos_per_query),
        hwy::Span<const size_t>(last_pos_per_query), att_cap, att_out,
        exp_denominator_sums.data(), max_logits.data());
  }

  // Verification
  float max_abs_err = 0.0f;
  float mse_sum = 0.0f;
  float dot_prod = 0.0f;
  float norm_ref = 0.0f;
  float norm_out = 0.0f;

  size_t failures = 0;
  for (size_t i = 0; i < num_queries; ++i) {
    float diff_exp =
        std::abs(exp_denominator_sums[i] - exp_denominator_sums_ref[i]);
    if (diff_exp >= tol_exp) {
      if (failures < 5) {
        EXPECT_NEAR(exp_denominator_sums[i], exp_denominator_sums_ref[i],
                    tol_exp)
            << "i=" << i << " (Type: " << type_name << ", SeqLen: " << kv_seq_len << ")";
      }
      failures++;
    }

    float diff_max = std::abs(max_logits[i] - max_logits_ref[i]);
    if (diff_max >= tol_max) {
      if (failures < 5) {
        EXPECT_NEAR(max_logits[i], max_logits_ref[i], tol_max)
            << "i=" << i << " (Type: " << type_name << ", SeqLen: " << kv_seq_len << ")";
      }
      failures++;
    }

    for (size_t j = 0; j < qkv_dim; ++j) {
      float v_ref = att_out_ref.Row(i)[j];
      float v_out = att_out.Row(i)[j];
      float diff = std::abs(v_ref - v_out);
      max_abs_err = std::max(max_abs_err, diff);
      mse_sum += diff * diff;
      dot_prod += v_ref * v_out;
      norm_ref += v_ref * v_ref;
      norm_out += v_out * v_out;
      if (diff >= tol) {
        if (failures < 5) {
          EXPECT_NEAR(v_out, v_ref, tol)
              << "i=" << i << " j=" << j << " (Type: " << type_name << ", SeqLen: " << kv_seq_len << ")";
        }
        failures++;
      }
    }
  }
  float cosine_sim = dot_prod / (std::sqrt(norm_ref) * std::sqrt(norm_out));
  float mse = mse_sum / (num_queries * qkv_dim);
  std::cerr << "=== Numerical Verification Results (" << type_name
            << ", SeqLen: " << kv_seq_len << ") ===\n"
            << "  Cosine Similarity:  " << cosine_sim << "\n"
            << "  Max Absolute Error: " << max_abs_err << "\n"
            << "  Mean Squared Error: " << mse << "\n";
}

void TestTiledFlashAttentionBF16MatrixAccumulationLargeVerification() {
  RunTiledFlashAttentionDifferentialTest<BF16>(
      2048, 1.0e-1f, 1.1e-1f, 1e-4f,
      gcpp::KVEncoding::kBF16TwoTranspositions,
      gcpp::KVEncoding::kBF16MatrixAccumulation,
      AttentionImpl::kFlashMatrixAccumulation, "BF16");
}

void TestTiledFlashAttentionInt8MatrixAccumulationLargeVerification() {
  RunTiledFlashAttentionDifferentialTest<int8_t>(
      1024, 1.5e-1f, 5.0, 8.0e-2f,
      gcpp::KVEncoding::kInt8TwoTranspositions,
      gcpp::KVEncoding::kInt8MatrixAccumulation,
      AttentionImpl::kInt8MatrixAccumulation, "Int8");
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gcpp {
HWY_BEFORE_TEST(FlashAttentionTest);
HWY_EXPORT_AND_TEST_P(FlashAttentionTest, TestAttention);
HWY_EXPORT_AND_TEST_P(FlashAttentionTest, TestTiledFlashAttention);

HWY_EXPORT_AND_TEST_P(FlashAttentionTest, TestTiledFlashAttentionBF16);
HWY_EXPORT_AND_TEST_P(FlashAttentionTest,
                      TestTiledFlashAttentionBF16MatrixAccumulation);

HWY_EXPORT_AND_TEST_P(
    FlashAttentionTest,
    TestTiledFlashAttentionBF16MatrixAccumulationLargeVerification);
HWY_EXPORT_AND_TEST_P(
    FlashAttentionTest,
    TestTiledFlashAttentionInt8MatrixAccumulationLargeVerification);

HWY_EXPORT_AND_TEST_P(FlashAttentionTest, TestTiledFlashAttentionInt8);
HWY_EXPORT_AND_TEST_P(FlashAttentionTest, TestTiledFlashAttentionInt8BF16);
HWY_EXPORT_AND_TEST_P(FlashAttentionTest, TestTiledFlashAttentionInt8Int16);
HWY_EXPORT_AND_TEST_P(FlashAttentionTest,
                      TestTiledFlashAttentionInt8MatrixAccumulation);

HWY_AFTER_TEST();

}  // namespace gcpp

#endif
