#include "gemma/kv_cache.h"

#include <cstddef>
#include <vector>

#include "gtest/gtest.h"
#include "gemma/configs.h"
#include "gemma/gemma_args.h"
#include "util/threading_context.h"
#include "hwy/aligned_allocator.h"
namespace gcpp {
namespace {

TEST(KVCacheTest, KVCacheToPtrs) {
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

  std::vector<KVCachePtr> ptrs = ToKVCachePtrs({caches.data(), caches.size()});
  ASSERT_EQ(ptrs.size(), 2);
  if (caches[0].IsTiled()) {
    EXPECT_EQ(ptrs[0].cache, &caches[0]);
    EXPECT_EQ(ptrs[1].cache, &caches[1]);
  } else {
    EXPECT_EQ(ptrs[0].kv_cache.Row(0), caches[0].kv_cache.Row(0));
    EXPECT_EQ(ptrs[1].kv_cache.Row(0), caches[1].kv_cache.Row(0));
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
}

}  // namespace
}  // namespace gcpp
