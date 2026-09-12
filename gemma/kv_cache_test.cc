#include "gemma/kv_cache.h"

#include <cstddef>
#include <vector>

#include "gtest/gtest.h"
#include "gemma/configs.h"
#include "gemma/gemma_args.h"
#include "util/threading_context.h"
namespace gcpp {
namespace {

TEST(KVCacheTest, ToPtr) {
  ModelConfig model_config;
  model_config.max_seq_len = 1024;
  model_config.num_layers = 2;
  for (int i = 0; i < model_config.num_layers; ++i) {
    model_config.layer_configs.push_back(LayerConfig());
    model_config.layer_configs.back().kv_heads = 4;
    model_config.layer_configs.back().qkv_dim = 256;
    model_config.attention_window_sizes.push_back(1024);
  }
  InferenceArgs inference_args;
  inference_args.seq_len = 1024;
  RuntimeConfig runtime_config;
  runtime_config.attention_impl = AttentionImpl::kFlash;
  ThreadingArgs threading_args;
  ThreadingContext ctx(threading_args);
  std::vector<KVCache> caches;
  caches.emplace_back(model_config, inference_args, runtime_config,
                      ctx.allocator);
  inference_args.seq_len = 512;
  caches.emplace_back(model_config, inference_args, runtime_config,
                      ctx.allocator);

  KVCachePtr ptr0 = caches[0].ToPtr();
  KVCachePtr ptr1 = caches[1].ToPtr();
  if (caches[0].IsTiled()) {
    EXPECT_EQ(ptr0.cache, &caches[0]);
    EXPECT_EQ(ptr1.cache, &caches[1]);
  } else {
    EXPECT_EQ(ptr0.kv_cache.Row(0), caches[0].kv_cache.Row(0));
    EXPECT_EQ(ptr1.kv_cache.Row(0), caches[1].kv_cache.Row(0));
  }
}

TEST(KVCacheTest, EncoderDecoderUsesDecoderLayerConfig) {
  ModelConfig model_config(Model::T5GEMMA_S_S, Type::kSFP,
                           PromptWrapping::GEMMA_PT);
  ASSERT_TRUE(model_config.is_encoder_decoder);
  ASSERT_FALSE(model_config.decoder_layer_configs.empty());
  InferenceArgs inference_args;
  inference_args.seq_len = 128;
  ThreadingArgs threading_args;
  ThreadingContext ctx(threading_args);

  KVCache cache(model_config, inference_args, ctx.allocator);

  EXPECT_EQ(cache.num_layers, model_config.decoder_layer_configs.size());
  EXPECT_EQ(cache.kv_heads, model_config.decoder_layer_configs[0].kv_heads);
  EXPECT_EQ(cache.qkv_dim, model_config.decoder_layer_configs[0].qkv_dim);
  EXPECT_EQ(cache.kv_cache.Cols(), model_config.KVCacheCols());
  RuntimeConfig runtime_config;
  runtime_config.attention_impl = AttentionImpl::kFlash;
  KVCache runtime_cache(model_config, inference_args, runtime_config,
                         ctx.allocator);
  EXPECT_FALSE(runtime_cache.kv_is_scratch);
  EXPECT_EQ(runtime_cache.kv_cache.Rows(), inference_args.seq_len);
  EXPECT_EQ(runtime_cache.kv_cache.Cols(), model_config.KVCacheCols());
}

// Layers that reuse an earlier layer's K/V own no region of the cache.
TEST(KVCacheTest, SharedLayersReserveNoCache) {
  ModelConfig model_config(Model::GEMMA4_2B, Type::kSFP,
                           PromptWrapping::GEMMA_IT);
  InferenceArgs inference_args;
  inference_args.seq_len = 1024;
  RuntimeConfig runtime_config;
  runtime_config.attention_impl = AttentionImpl::kFlash;
  ThreadingArgs threading_args;
  ThreadingContext ctx(threading_args);

  KVCache cache(model_config, inference_args, runtime_config, ctx.allocator);

  // Layer 15 reuses layer 13's K/V, per ConfigGemma4_2B_LM
  EXPECT_EQ(cache.layer_flat_offsets[15], cache.layer_flat_offsets[13]);
  EXPECT_EQ(cache.layer_k_v_offsets[15], cache.layer_k_v_offsets[13]);
  EXPECT_EQ(cache.layer_kv_head_offsets[15], cache.layer_kv_head_offsets[13]);
  EXPECT_TRUE(cache.kv_is_scratch);
  // Global layers project 1024 values; local layers project only 512.
  EXPECT_EQ(cache.kv_cache.Cols(), 1024);
}

TEST(KVCacheTest, FlashScratchPreservesSequenceAndHistory) {
  ModelConfig config(Model::GEMMA3_270M, Type::kSFP, PromptWrapping::GEMMA_IT);
  config.num_layers = 2;
  config.layer_configs.resize(2);
  config.attention_window_sizes.resize(2);
  InferenceArgs args;
  args.seq_len = 32768;
  RuntimeConfig runtime;
  runtime.attention_impl = AttentionImpl::kFlash;
  runtime.prefill_tbatch_size = 16;
  ThreadingArgs threading;
  ThreadingContext ctx(threading);
  KVCache cache(config, args, runtime, ctx.allocator);
  ASSERT_TRUE(cache.kv_is_scratch);
  EXPECT_EQ(cache.SeqLen(), 32768);
  EXPECT_EQ(cache.ToPtr().SeqLen(), 32768);
  EXPECT_FALSE(cache.ToPtr().IsEmpty());
  EXPECT_EQ(cache.kv_cache.Rows(), 16);
  EXPECT_EQ(cache.kv_cache.Cols(), config.layer_configs[0].CacheLayerSize());
  EXPECT_FALSE(cache.compact_kv_cache_ptr.HasPtr());

  ZeroInit(cache.kv_cache);
  ZeroInit(cache.k_cache);
  ZeroInit(cache.v_cache);
  cache.k_cache.Row(32767)[0] = hwy::BF16FromF32(3.0f);
  cache.v_cache.Row(32767)[0] = hwy::BF16FromF32(5.0f);
  cache.EnsureProjectionRows(33, cache.kv_cache.Cols());
  EXPECT_EQ(cache.kv_cache.Rows(), 33);
  cache.EnsureProjectionRows(1, cache.kv_cache.Cols());
  EXPECT_EQ(cache.kv_cache.Rows(), 33);
  const size_t wider_cols = 2 * cache.kv_cache.Cols();
  cache.EnsureProjectionRows(1, wider_cols);
  EXPECT_EQ(cache.kv_cache.Rows(), 33);
  EXPECT_EQ(cache.kv_cache.Cols(), wider_cols);
  ZeroInit(cache.kv_cache);
  KVCache copy = cache.Copy();
  EXPECT_TRUE(copy.kv_is_scratch);
  EXPECT_EQ(copy.SeqLen(), 32768);
  EXPECT_EQ(hwy::F32FromBF16(copy.k_cache.Row(32767)[0]), 3.0f);
  EXPECT_EQ(hwy::F32FromBF16(copy.v_cache.Row(32767)[0]), 5.0f);
  EXPECT_NE(copy.k_cache.Row(0), cache.k_cache.Row(0));
}

}  // namespace
}  // namespace gcpp
