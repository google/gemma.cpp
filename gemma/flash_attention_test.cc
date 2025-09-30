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
  // hwy::ThreadPool& pool = ctx.pools.Pool();
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
  auto& kvc = qbatch.KV(0).kv_cache;
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
  FlashAttention(tokens.size(), target_parallelism, 0, layers, attention,
                 qbatch, ctx);
  AssertClose(attention.att_out, *saved_att);
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
HWY_AFTER_TEST();

}  // namespace gcpp

#endif
