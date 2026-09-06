// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include "gemma/kv_cache.h"

#include <algorithm>
#include <cstring>

#include "gemma/configs.h"
#include "gemma/gemma_args.h"
#include "gtest/gtest.h"
#include "util/threading_context.h"

namespace gcpp {
namespace {

ModelConfig SmallConfig(size_t window = 17) {
  ModelConfig config(Model::GEMMA3_1B, Type::kF32, PromptWrapping::GEMMA_PT);
  config.max_seq_len = 128;
  config.num_layers = 2;
  config.layer_configs.resize(2);
  config.attention_window_sizes = {static_cast<uint32_t>(window), 128};
  return config;
}

float Value(size_t pos, size_t col) {
  return static_cast<float>(pos * 1024 + col);
}

void Write(KVCache& cache, size_t layer, size_t pos) {
  for (size_t col = 0; col < cache.LayerCache(layer).Cols(); ++col) {
    cache.Row(layer, pos)[col] = Value(pos, col);
  }
}

void Check(const KVCache& cache, size_t layer, size_t pos) {
  for (size_t col = 0; col < cache.LayerCache(layer).Cols(); ++col) {
    ASSERT_EQ(cache.Row(layer, pos)[col], Value(pos, col))
        << "layer=" << layer << "pos=" << pos << " col=" << col;
  }
}

TEST(KVCacheTest, LocalAndGlobalCapacities) {
  ThreadingArgs threading;
  threading.max_threads = 1;
  ThreadingContext ctx(threading);
  auto config = SmallConfig();
  InferenceArgs inference;
  inference.seq_len = 128;
  inference.prefill_tbatch_size = 8;
  KVCache cache(config, inference, ctx.allocator);
  EXPECT_EQ(cache.SeqLen(), 128u);
  EXPECT_EQ(cache.LayerCache(0).Rows(), 24u);
  EXPECT_EQ(cache.LayerCache(1).Rows(), 128u);
  size_t bytes = 0;
  for (size_t i = 0; i < 2; ++i) {
    const auto& layer = cache.LayerCache(i);
    bytes += layer.Rows() * layer.Stride() * sizeof(KV_t);
  }
  EXPECT_EQ(cache.AllocatedBytes(), bytes);
}

TEST(KVCacheTest, BatchedPrefillAndRepeatedWraparound) {
  ThreadingArgs threading;
  threading.max_threads = 1;
  ThreadingContext ctx(threading);
  for (size_t window : {size_t{1}, size_t{17}}) {
    auto config = SmallConfig(window);
    InferenceArgs inference;
    inference.seq_len = 128;
    inference.prefill_tbatch_size = 8;
    KVCache cache(config, inference, ctx.allocator);
    size_t pos = 0;
    for (size_t batch : {8u, 8u, 8u, 1u, 31u, 2u, 31u, 31u, 31u, 31u}) {
      SCOPED_TRACE(::testing::Message() << "window=" << window << " pos=" << pos
                                        << " batch=" << batch);
      cache.PrepareLayer(0, batch, pos);
      // Match ComputeQKV: write ALL new rows before reading any query.
      for (size_t p = pos; p < pos + batch; ++p) Write(cache, 0, p);
      for (size_t q = pos; q < pos + batch; ++q) {
        const size_t first = q - std::min(q, window - 1);
        for (size_t p = first; p <= q; ++p) Check(cache, 0, p);
      }
      pos += batch;
    }
    EXPECT_LT(cache.LayerCache(0).Rows(), cache.SeqLen());
  }
}

TEST(KVCacheTest, CopyIsIndependentAndGrowthRetainsHistory) {
  ThreadingArgs threading;
  threading.max_threads = 1;
  ThreadingContext ctx(threading);
  auto config = SmallConfig();
  InferenceArgs inference;
  inference.seq_len = 128;
  inference.prefill_tbatch_size = 8;
  KVCache cache(config, inference, ctx.allocator);
  for (size_t p = 0; p < 100; ++p) {
    Write(cache, 0, p);
    Write(cache, 1, p);
  }
  auto copy = cache.Copy();
  EXPECT_EQ(copy.AllocatedBytes(), cache.AllocatedBytes());
  EXPECT_NE(copy.LayerCache(0).Row(0), cache.LayerCache(0).Row(0));
  // Grow after wrapping, as when a subsequent turn increases prefill size.
  cache.PrepareLayer(0, 31, 100);
  EXPECT_EQ(cache.LayerCache(0).Rows(), 47u);
  EXPECT_EQ(copy.LayerCache(0).Rows(), 24u);
  for (size_t p = 76; p < 100; ++p) {
    Check(cache, 0, p);
    Check(copy, 0, p);
  }
  for (size_t p = 0; p < 100; ++p) Check(copy, 1, p);
  cache.Row(0, 99)[0] = -1.0f;
  Check(copy, 0, 99);
  // Restarting a conversation and increasing its batch must also be safe.
  cache.PrepareLayer(0, 64, 0);
  Write(cache, 0, 0);
  Check(cache, 0, 0);
}

TEST(KVCacheTest, FullAttentionAndContextCapping) {
  ThreadingArgs threading;
  threading.max_threads = 1;
  ThreadingContext ctx(threading);
  auto config = SmallConfig(128);
  InferenceArgs inference;
  inference.seq_len = 256;
  inference.prefill_tbatch_size = 1;
  KVCache global(config, inference, ctx.allocator);
  EXPECT_EQ(global.SeqLen(), 128u);
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(global.LayerCache(i).Rows(), 128u);
    global.PrepareLayer(i, 64, 120);
    EXPECT_EQ(global.LayerCache(i).Rows(), 128u);
  }
  config = SmallConfig();
  inference.seq_len = 8;
  KVCache short_context(config, inference, ctx.allocator);
  EXPECT_EQ(short_context.LayerCache(0).Rows(), 8u);
  EXPECT_EQ(short_context.LayerCache(1).Rows(), 8u);
}

}  // namespace
}  // namespace gcpp
