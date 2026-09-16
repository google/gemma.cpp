#include "gemma/kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
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
  EXPECT_EQ(cache.kv_cache.Cols(), model_config.KVCacheCols());
}

TEST(KVCacheTest, CopyTiledCacheAllocatesAndRebases) {
  ModelConfig model_config;
  model_config.max_seq_len = 1024;
  model_config.num_layers = 2;

  // Layer 0: local attention layer
  model_config.layer_configs.push_back(LayerConfig());
  model_config.layer_configs.back().kv_heads = 2;
  model_config.layer_configs.back().qkv_dim = 256;
  model_config.attention_window_sizes.push_back(512);

  // Layer 1: global attention layer
  model_config.layer_configs.push_back(LayerConfig());
  model_config.layer_configs.back().kv_heads = 2;
  model_config.layer_configs.back().qkv_dim = 512;
  model_config.attention_window_sizes.push_back(1024);

  InferenceArgs inference_args;
  inference_args.seq_len = 1024;
  RuntimeConfig runtime_config;
  runtime_config.attention_impl = AttentionImpl::kFlashTransposedQs;
  ThreadingArgs threading_args;
  ThreadingContext ctx(threading_args);

  KVCache orig(model_config, inference_args, runtime_config, ctx.allocator);
  ASSERT_TRUE(orig.compact_local_kv_cache_ptr.HasPtr());
  ASSERT_TRUE(orig.compact_global_kv_cache_ptr.HasPtr());
  ASSERT_EQ(orig.kv_head_ptrs.size(), 4);  // 2 heads * 2 layers

  // Fill buffers with pattern
  std::memset(orig.compact_local_kv_cache_ptr.RowBytes(0), 0xAB,
              orig.compact_local_kv_cache_ptr.Rows() *
                  orig.compact_local_kv_cache_ptr.Stride() *
                  orig.compact_local_kv_cache_ptr.ElementBytes());
  std::memset(orig.compact_global_kv_cache_ptr.RowBytes(0), 0xCD,
              orig.compact_global_kv_cache_ptr.Rows() *
                  orig.compact_global_kv_cache_ptr.Stride() *
                  orig.compact_global_kv_cache_ptr.ElementBytes());

  KVCache copy = orig.Copy();

  // Verify memory allocation in copy
  EXPECT_TRUE(copy.compact_local_kv_cache_ptr.HasPtr());
  EXPECT_TRUE(copy.compact_global_kv_cache_ptr.HasPtr());
  EXPECT_NE(copy.compact_local_kv_cache_ptr.RowBytes(0),
            orig.compact_local_kv_cache_ptr.RowBytes(0));
  EXPECT_NE(copy.compact_global_kv_cache_ptr.RowBytes(0),
            orig.compact_global_kv_cache_ptr.RowBytes(0));

  // Verify data copied
  EXPECT_EQ(0, std::memcmp(copy.compact_local_kv_cache_ptr.RowBytes(0),
                           orig.compact_local_kv_cache_ptr.RowBytes(0),
                           orig.compact_local_kv_cache_ptr.Rows() *
                               orig.compact_local_kv_cache_ptr.Stride() *
                               orig.compact_local_kv_cache_ptr.ElementBytes()));
  EXPECT_EQ(0,
            std::memcmp(copy.compact_global_kv_cache_ptr.RowBytes(0),
                        orig.compact_global_kv_cache_ptr.RowBytes(0),
                        orig.compact_global_kv_cache_ptr.Rows() *
                            orig.compact_global_kv_cache_ptr.Stride() *
                            orig.compact_global_kv_cache_ptr.ElementBytes()));

  // Verify pointer rebasing for all kv_head_ptrs
  EXPECT_EQ(copy.kv_head_ptrs.size(), orig.kv_head_ptrs.size());
  for (size_t i = 0; i < copy.kv_head_ptrs.size(); ++i) {
    const MatPtr& orig_hp = orig.kv_head_ptrs[i];
    const MatPtr& copy_hp = copy.kv_head_ptrs[i];
    EXPECT_NE(copy_hp.RowBytes(0), orig_hp.RowBytes(0));
    EXPECT_EQ(copy_hp.Rows(), orig_hp.Rows());
    EXPECT_EQ(copy_hp.Cols(), orig_hp.Cols());
    EXPECT_EQ(copy_hp.Stride(), orig_hp.Stride());

    if (i < 2) {
      // Local layer heads point into compact_local_kv_cache_ptr
      const uintptr_t orig_offset =
          orig_hp.RowBytes(0) - orig.compact_local_kv_cache_ptr.RowBytes(0);
      const uintptr_t copy_offset =
          copy_hp.RowBytes(0) - copy.compact_local_kv_cache_ptr.RowBytes(0);
      EXPECT_EQ(copy_offset, orig_offset);
    } else {
      // Global layer heads point into compact_global_kv_cache_ptr
      const uintptr_t orig_offset =
          orig_hp.RowBytes(0) - orig.compact_global_kv_cache_ptr.RowBytes(0);
      const uintptr_t copy_offset =
          copy_hp.RowBytes(0) - copy.compact_global_kv_cache_ptr.RowBytes(0);
      EXPECT_EQ(copy_offset, orig_offset);
    }
  }

  // Verify GetPointers works on copy without crash
  std::vector<MatPtr> local_ptrs =
      copy.GetPointers(/*layer_idx=*/0, /*kv_head_idx=*/0, /*start_pos=*/64,
                       /*is_global_layer=*/false);
  EXPECT_FALSE(local_ptrs.empty());
  std::vector<MatPtr> global_ptrs =
      copy.GetPointers(/*layer_idx=*/1, /*kv_head_idx=*/1, /*start_pos=*/64,
                       /*is_global_layer=*/true);
  EXPECT_EQ(global_ptrs.size(), 1);

  // Metadata and byte size
  EXPECT_EQ(copy.k_v_cols, orig.k_v_cols);
  EXPECT_EQ(copy.TotalByteSize(), orig.TotalByteSize());
  EXPECT_GT(copy.TotalByteSize(), 0);
}

TEST(KVCacheTest, ClonePrefixCopiesOnlyPrefix) {
  ModelConfig model_config;
  model_config.max_seq_len = 1024;
  model_config.num_layers = 2;

  // Layer 0: local attention layer
  model_config.layer_configs.push_back(LayerConfig());
  model_config.layer_configs.back().kv_heads = 2;
  model_config.layer_configs.back().qkv_dim = 256;
  model_config.attention_window_sizes.push_back(512);

  // Layer 1: global attention layer
  model_config.layer_configs.push_back(LayerConfig());
  model_config.layer_configs.back().kv_heads = 2;
  model_config.layer_configs.back().qkv_dim = 512;
  model_config.attention_window_sizes.push_back(1024);

  InferenceArgs inference_args;
  inference_args.seq_len = 1024;
  RuntimeConfig runtime_config;
  runtime_config.attention_impl = AttentionImpl::kFlashTransposedQs;
  ThreadingArgs threading_args;
  ThreadingContext ctx(threading_args);

  KVCache orig(model_config, inference_args, runtime_config, ctx.allocator);
  ASSERT_EQ(orig.kv_head_ptrs.size(), 4);

  // Fill head tile rows with distinct byte values per row
  for (size_t h = 0; h < orig.kv_head_ptrs.size(); ++h) {
    MatPtr& hp = orig.kv_head_ptrs[h];
    for (size_t r = 0; r < hp.Rows(); ++r) {
      uint8_t val = static_cast<uint8_t>((h + 1) * 10 + r + 1);
      std::memset(hp.RowBytes(r), val, hp.Stride() * hp.ElementBytes());
    }
  }

  // Clone prefix of length 40. With kTileSize = 32, prefix_tiles = ceil(40/32)
  // = 2.
  const size_t prefix_len = 40;
  const size_t expected_prefix_tiles = 2;
  KVCache clone = orig.ClonePrefix(prefix_len, ctx.allocator);

  EXPECT_EQ(clone.SeqLen(), orig.SeqLen());
  EXPECT_EQ(clone.TotalByteSize(), orig.TotalByteSize());
  EXPECT_EQ(clone.kv_head_ptrs.size(), orig.kv_head_ptrs.size());

  for (size_t h = 0; h < clone.kv_head_ptrs.size(); ++h) {
    const MatPtr& orig_hp = orig.kv_head_ptrs[h];
    const MatPtr& clone_hp = clone.kv_head_ptrs[h];
    EXPECT_NE(clone_hp.RowBytes(0), orig_hp.RowBytes(0));
    EXPECT_EQ(clone_hp.Rows(), orig_hp.Rows());

    // Rows < expected_prefix_tiles must match original
    for (size_t r = 0; r < expected_prefix_tiles; ++r) {
      EXPECT_EQ(0, std::memcmp(clone_hp.RowBytes(r), orig_hp.RowBytes(r),
                               clone_hp.Stride() * clone_hp.ElementBytes()));
    }
    // Rows >= expected_prefix_tiles must be zeroes (not copied)
    for (size_t r = expected_prefix_tiles; r < clone_hp.Rows(); ++r) {
      const uint8_t* row = clone_hp.RowBytes(r);
      const size_t bytes = clone_hp.Stride() * clone_hp.ElementBytes();
      bool all_zero = true;
      for (size_t b = 0; b < bytes; ++b) {
        if (row[b] != 0) {
          all_zero = false;
          break;
        }
      }
      EXPECT_TRUE(all_zero)
          << "Head " << h << " row " << r << " should be zero";
    }
  }
}

TEST(KVCacheTest, DeepSeekCompressorStatePreservation) {
  ModelConfig model_config;
  model_config.max_seq_len = 512;
  model_config.num_layers = 1;
  model_config.layer_configs.push_back(LayerConfig());
  model_config.layer_configs[0].kv_lora_rank = 32;
  model_config.layer_configs[0].o_lora_rank = 32;
  model_config.layer_configs[0].rope_head_dim = 32;
  model_config.layer_configs[0].kv_compression_rate = 16;
  model_config.layer_configs[0].heads = 4;
  model_config.layer_configs[0].kv_heads = 4;

  InferenceArgs inference_args;
  inference_args.seq_len = 512;
  ThreadingArgs threading_args;
  ThreadingContext ctx(threading_args);

  KVCache orig(model_config, inference_args, ctx.allocator);
  ASSERT_GT(orig.ds_state.Rows(), 0);
  ASSERT_GT(orig.ds_state.Cols(), 0);
  ASSERT_EQ(orig.ds_state_snapshot.Rows(), 32);

  // Fill orig.ds_state.Row(0) with unique values
  float* ds_row = orig.ds_state.Row(0);
  for (size_t c = 0; c < orig.ds_state.Cols(); ++c) {
    ds_row[c] = static_cast<float>(c + 1) * 0.25f;
  }

  // Clone prefix
  KVCache clone = orig.ClonePrefix(16, ctx.allocator);
  ASSERT_EQ(clone.ds_state.Rows(), 1);
  ASSERT_EQ(clone.ds_state.Cols(), orig.ds_state.Cols());
  ASSERT_EQ(clone.ds_state_snapshot.Rows(), 32);
  ASSERT_EQ(clone.ds_state_offsets, orig.ds_state_offsets);

  // Verify ds_state.Row(0) and ds_state_snapshot.Row(0) match
  for (size_t c = 0; c < clone.ds_state.Cols(); ++c) {
    EXPECT_FLOAT_EQ(clone.ds_state.Row(0)[c], ds_row[c]);
    EXPECT_FLOAT_EQ(clone.ds_state_snapshot.Row(0)[c], ds_row[c]);
  }

  // Mutate clone.ds_state.Row(0)
  for (size_t c = 0; c < clone.ds_state.Cols(); ++c) {
    clone.ds_state.Row(0)[c] = -999.0f;
  }

  // Restore from snapshot row 0
  clone.RestoreDSStateFromSnapshot(0);
  for (size_t c = 0; c < clone.ds_state.Cols(); ++c) {
    EXPECT_FLOAT_EQ(clone.ds_state.Row(0)[c], ds_row[c]);
  }
}

TEST(KVCacheTest, TotalByteSizeCalculation) {
  ModelConfig model_config;
  model_config.max_seq_len = 256;
  model_config.num_layers = 1;
  model_config.layer_configs.push_back(LayerConfig());
  model_config.layer_configs.back().kv_heads = 2;
  model_config.layer_configs.back().qkv_dim = 128;
  model_config.attention_window_sizes.push_back(256);

  InferenceArgs inference_args;
  inference_args.seq_len = 256;
  RuntimeConfig runtime_config;
  runtime_config.attention_impl = AttentionImpl::kFlashTransposedQs;
  ThreadingArgs threading_args;
  ThreadingContext ctx(threading_args);

  KVCache cache(model_config, inference_args, runtime_config, ctx.allocator);
  const size_t total_bytes = cache.TotalByteSize();
  EXPECT_GT(total_bytes, 0);

  size_t expected_bytes = 0;
  if (cache.kv_cache.HasPtr()) {
    expected_bytes += cache.kv_cache.Rows() * cache.kv_cache.Stride() *
                      cache.kv_cache.ElementBytes();
  }
  if (cache.k_cache.HasPtr()) {
    expected_bytes += cache.k_cache.Rows() * cache.k_cache.Stride() *
                      cache.k_cache.ElementBytes();
  }
  if (cache.v_cache.HasPtr()) {
    expected_bytes += cache.v_cache.Rows() * cache.v_cache.Stride() *
                      cache.v_cache.ElementBytes();
  }
  if (cache.compact_global_kv_cache_ptr.HasPtr()) {
    expected_bytes += cache.compact_global_kv_cache_ptr.Rows() *
                      cache.compact_global_kv_cache_ptr.Stride() *
                      cache.compact_global_kv_cache_ptr.ElementBytes();
  }
  EXPECT_EQ(total_bytes, expected_bytes);
}

}  // namespace
}  // namespace gcpp
