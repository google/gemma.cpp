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

#include <cstring>
#include <numeric>
#include <vector>

#include "compression/types.h"
#include "gemma/activations.h"
#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "gemma/kv_cache.h"
#include "gemma/weights.h"
#include "ops/matmul.h"
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include <stddef.h>
#include <stdio.h>

#include <algorithm>  // std::max
#include <cmath>      // std::abs
#include <memory>

#include "util/mat.h"
#include "util/threading_context.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"

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
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

using FloatPtr = hwy::AlignedFreeUniquePtr<float[]>;

void SetMat(const size_t offset, MatPtrT<float>& mat) {
  const size_t kOuter = mat.Extents().rows;
  const size_t kInner = mat.Extents().cols;
  const float i_scale = 1.0f / kInner;
  const float j_scale = 1.0f / kOuter;
  for (size_t i = 0; i < kOuter; ++i) {
    float* row = mat.Row(i);
    for (size_t j = 0; j < kInner; ++j) {
      row[j] =
          static_cast<float>((i * kInner * i_scale + (j + offset) * j_scale));
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
  // Avoid comparing the padding bytes, which are uninitialized.
  for (size_t r = 0; r < a.Rows(); ++r) {
    const float* HWY_RESTRICT a_row = a.Row(r);
    const float* HWY_RESTRICT b_row = b.Row(r);
    for (size_t c = 0; c < a.Cols(); ++c) {
      float rel_abs_delta = std::abs(a_row[c] - b_row[c]);
      if (rel_abs_delta > 0.0f) {
        rel_abs_delta /= std::max(std::abs(a_row[c]), std::abs(b_row[c]));
      }
      EXPECT_LT(rel_abs_delta, 1e-5)
          << "a[" << r << "," << c << "]=" << a_row[c] << ", b[" << r << ","
          << c << "]=" << b_row[c];
    }
  }
}

void TestFlashAttention(size_t target_parallelism) {
  ThreadingArgs threading_args;
  ThreadingContext ctx(threading_args);
  constexpr size_t kOuter = 1024;
  constexpr size_t kInner = 256;
  ModelConfig config(Model::GEMMA2_2B, Type::kF32, PromptWrapping::GEMMA_PT);
  config.att_cap = 1024.0f;
  TensorInfoRegistry tensor_info_registry(config);
  const LayerConfig& layer_config = config.layer_configs[0];
  const LayerWeightsPtrs layers(0, layer_config, tensor_info_registry);
  InferenceArgs inference_args;
  RuntimeConfig runtime_config;
  KVCache kv_cache(config, inference_args, ctx.allocator);
  MatMulEnv env(ctx);
  Activations activations(config, runtime_config.prefill_tbatch_size,
                          kv_cache.SeqLen(), env.ctx, env.row_ptrs);
  std::vector<int> tokens(kOuter);
  std::iota(tokens.begin(), tokens.end(), 1);
  PromptTokens prompt(tokens);
  AllQueries all_queries(hwy::Span<const PromptTokens>(&prompt, 1),
                         hwy::Span<KVCache>(&kv_cache, 1));
  QBatch qbatch(/*start=*/0, /*max_size=*/kOuter, all_queries);
  const size_t batch_size = kOuter;
  std::vector<hwy::AlignedFreeUniquePtr<uint8_t*[]>> row_ptrs;
  AttentionActivations attention(config, layer_config, batch_size, kOuter,
                                 ctx.allocator, row_ptrs);
  const size_t qkv_dim = layer_config.qkv_dim;
  ASSERT_EQ(qkv_dim, kInner);
  const hwy::Divisor div_qbatch(qbatch.Size());
  // A "head group" in the context of GQA refers to a collection of query
  // heads that share the same key and value heads.
  const size_t kHeadGroups = layer_config.heads / layer_config.kv_heads;
  const size_t seq_len =
      static_cast<size_t>(attention.div_seq_len.GetDivisor());
  auto& kvc = qbatch.KV(0).LayerCache(0);
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
  }
  SetMat(1, attention.q);
  DotSoftmaxWeightedSum(tokens.size(), 0, layers, attention, qbatch, ctx);
  // Copy the output to saved_att to allow for comparison.
  auto saved_att = MakeCopyOfMat(attention.att_out, ctx.allocator);
  SetMat(1, attention.q);
  using DF = hn::ScalableTag<float>;
  const DF df;
  const size_t kNF = hn::Lanes(df);
  const size_t total_tasks =
      tokens.size() * div_qbatch.GetDivisor() * layer_config.heads;
  const size_t kVTileSize = GetVTileSize(kNF, kHeadGroups, tokens.size(),
                                         total_tasks, target_parallelism);
  printf("FlashAttention: target_parallelism=%zu, kNF=%zu, kVTileSize=%zu\n",
         target_parallelism, kNF, kVTileSize);
  FlashAttention(tokens.size(), target_parallelism, 0, layers, attention,
                 qbatch, ctx);
  AssertClose(attention.att_out, *saved_att);
  ctx.profiler.PrintResults();
}

// Compare compact rings against full-history buffers with exactly the same
// FP32 arithmetic. Exercise local/global wraparound, runtime growth, prefix-LM,
// and all three FlashAttention tile choices as well as the old attention path.
void TestWindowedKVAttention() {
  ThreadingArgs threading;
  threading.max_threads = 2;
  ThreadingContext ctx(threading);
  ModelConfig config(Model::GEMMA3_1B, Type::kF32, PromptWrapping::GEMMA_PT);
  config.max_seq_len = 256;
  config.num_layers = 1;
  config.layer_configs.resize(1);
  config.layer_configs[0].heads = 8;
  config.attention_window_sizes = {17};
  const LayerConfig& layer_config = config.layer_configs[0];
  TensorInfoRegistry registry(config);
  const LayerWeightsPtrs layer(0, layer_config, registry);
  ModelConfig full_config = config;
  full_config.attention_window_sizes = {256};
  InferenceArgs inference;
  inference.seq_len = 256;
  inference.prefill_tbatch_size = 8;

  for (size_t batch : {1u, 8u, 31u}) {
    for (size_t pos : {0u, 23u, 240u, 511u}) {
      for (bool prefix : {false, true}) {
        for (size_t parallelism : {0u, 1u, 16u, 8192u}) {
          SCOPED_TRACE(::testing::Message()
                       << "batch=" << batch << " pos=" << pos << " prefix="
                       << prefix << " parallelism=" << parallelism);
          KVCache compact(config, inference, ctx.allocator);
          KVCache full(full_config, inference, ctx.allocator);
          compact.PrepareLayer(0, batch, 0);
          std::vector<int> tokens(batch, 1);
          const size_t end = prefix ? pos + batch : 0;
          AllQueries cq(PromptTokens(tokens), pos, end,
                        hwy::Span<KVCache>(&compact, 1));
          AllQueries fq(PromptTokens(tokens), pos, end,
                        hwy::Span<KVCache>(&full, 1));
          QBatch cb(0, 1, cq), fb(0, 1, fq);
          std::vector<hwy::AlignedFreeUniquePtr<uint8_t*[]>> cptrs, fptrs;
          AttentionActivations ca(config, layer_config, batch, 256,
                                  ctx.allocator, cptrs);
          AttentionActivations fa(config, layer_config, batch, 256,
                                  ctx.allocator, fptrs);
          for (size_t p = 0; p < pos + batch; ++p) {
            for (size_t col = 0; col < full.LayerCache(0).Cols(); ++col) {
              const float value =
                  0.001f * static_cast<float>(
                               static_cast<int>((p * 13 + col) % 97) - 48);
              compact.Row(0, p)[col] = full.Row(0, p)[col] = value;
            }
          }
          SetMat(3, ca.q);
          CopyMat(ca.q, fa.q);
          for (size_t r = 0; r < batch; ++r) {
            // Identical masked score scratch for the old attention path.
            std::fill(ca.att.Row(r), ca.att.Row(r) + ca.att.Cols(), -1e30f);
            std::fill(fa.att.Row(r), fa.att.Row(r) + fa.att.Cols(), -1e30f);
          }
          if (parallelism == 0) {
            DotSoftmaxWeightedSum(batch, 0, layer, ca, cb, ctx);
            DotSoftmaxWeightedSum(batch, 0, layer, fa, fb, ctx);
          } else {
            FlashAttention(batch, parallelism, 0, layer, ca, cb, ctx);
            FlashAttention(batch, parallelism, 0, layer, fa, fb, ctx);
          }
          for (size_t r = 0; r < batch; ++r) {
            ASSERT_EQ(0, std::memcmp(ca.att_out.Row(r), fa.att_out.Row(r),
                                     ca.att_out.Cols() * sizeof(float)));
          }
        }
      }
    }
  }
}

void TestAttention() {
  TestFlashAttention(8192);
  TestFlashAttention(2048);
  TestFlashAttention(256);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gcpp {
HWY_BEFORE_TEST(FlashAttentionTest);
HWY_EXPORT_AND_TEST_P(FlashAttentionTest, TestAttention);
HWY_EXPORT_AND_TEST_P(FlashAttentionTest, TestWindowedKVAttention);
HWY_AFTER_TEST();

}  // namespace gcpp

#endif
