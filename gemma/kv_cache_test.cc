#include "gemma/kv_cache.h"

#include <cstddef>
#include <vector>

#include "gemma/configs.h"
#include "gemma/flash_attention.h"
#include "gemma/gemma_args.h"
#include "gtest/gtest.h"
#include "hwy/targets.h"
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
  caches.emplace_back(model_config, inference_args,
                      runtime_config.attention_impl, ctx.allocator);
  inference_args.seq_len = 512;
  caches.emplace_back(model_config, inference_args,
                      runtime_config.attention_impl, ctx.allocator);

  KVCachePtr ptr0 = caches[0].ToPtr();
  KVCachePtr ptr1 = caches[1].ToPtr();
  EXPECT_EQ(ptr0.cache, &caches[0]);
  EXPECT_EQ(ptr1.cache, &caches[1]);
  EXPECT_FALSE(ptr0.IsEmpty());
  EXPECT_EQ(ptr0.SeqLen(), 1024u);
  EXPECT_EQ(ptr1.SeqLen(), 512u);
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
  EXPECT_FALSE(cache.IsTiled());
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

  KVCache cache(model_config, inference_args, runtime_config.attention_impl,
                ctx.allocator);

  // Layer 15 reuses layer 13's K/V, per ConfigGemma4_2B_LM
  EXPECT_EQ(cache.layer_flat_offsets[15], cache.layer_flat_offsets[13]);
  EXPECT_EQ(cache.layer_kv_head_offsets[15], cache.layer_kv_head_offsets[13]);
  cache.PrepareLayer(13, 1, 0, 16);
  EXPECT_EQ(cache.FlashK(15, 0).Row(0), cache.FlashK(13, 0).Row(0));
  auto copy = cache.Copy();
  EXPECT_EQ(copy.FlashK(15, 0).Row(0), copy.FlashK(13, 0).Row(0));
  EXPECT_NE(copy.FlashK(13, 0).Row(0), cache.FlashK(13, 0).Row(0));
}

ModelConfig RingConfig() {
  ModelConfig config;
  config.max_seq_len = 8192;
  config.num_layers = 2;
  config.layer_configs.resize(2);
  for (auto& layer : config.layer_configs) {
    layer.kv_heads = 1;
    layer.heads = 2;
    layer.qkv_dim = 64;
  }
  config.attention_window_sizes = {512, 8192};
  return config;
}

TEST(KVCacheTest, ConstructorsSelectSameCompactLayout) {
  auto config = RingConfig();
  InferenceArgs inference;
  inference.seq_len = 1031;
  inference.prefill_tbatch_size = 17;
  ThreadingContext ctx{ThreadingArgs{}};
  for (auto impl : {AttentionImpl::kFlash, AttentionImpl::kFlashTransposedQs,
                    AttentionImpl::kFlashTransposedQsBF16,
                    AttentionImpl::kFlashTransposedQsInt8,
                    AttentionImpl::kFlashMatrixAccumulation,
                    AttentionImpl::kInt8MatrixAccumulation}) {
    inference.attention_impl = GetAttentionImplName(impl);
    KVCache implicit(config, inference, ctx.allocator);
    KVCache explicit_cache(config, inference, impl, ctx.allocator);
    EXPECT_EQ(implicit.SeqLen(), 1031u);
    EXPECT_EQ(implicit.AllocatedBytes(), explicit_cache.AllocatedBytes());
    ASSERT_EQ(implicit.kv_head_ptrs.size(), explicit_cache.kv_head_ptrs.size());
    EXPECT_FALSE(implicit.kv_cache.HasPtr());
    for (size_t h = 0; h < implicit.kv_head_ptrs.size(); ++h) {
      const auto& a = implicit.kv_head_ptrs[h];
      const auto& b = explicit_cache.kv_head_ptrs[h];
      EXPECT_EQ(a.Rows(), b.Rows());
      EXPECT_EQ(a.Cols(), b.Cols());
      EXPECT_EQ(a.GetType(), b.GetType());
      EXPECT_EQ(a.GetLayout(), b.GetLayout());
    }
  }
  inference.attention_impl = "flash_transposed_qs";
  inference.kv_cache_type = "bf16";
  KVCache typed(config, inference, ctx.allocator);
  EXPECT_EQ(typed.kv_head_ptrs.front().GetType(), Type::kBF16);
  KVCache explicit_type(config, inference, AttentionImpl::kFlashTransposedQs,
                        ctx.allocator);
  EXPECT_EQ(explicit_type.kv_head_ptrs.front().GetType(), Type::kBF16);
  KVCache overridden(config, inference, AttentionImpl::kFlashTransposedQs,
                     ctx.allocator, Type::kF32);
  EXPECT_EQ(overridden.kv_head_ptrs.front().GetType(), Type::kF32);
}

TEST(KVCacheTest, HeterogeneousSharedHeadsAndTiledSnapshot) {
  auto config = RingConfig();
  config.layer_configs[1].qkv_dim = 128;
  config.layer_configs[1].kv_heads = 2;
  config.layer_configs.push_back(config.layer_configs[0]);
  config.layer_configs.back().kv_share_layer_idx = 0;
  config.attention_window_sizes.push_back(2048);
  config.num_layers = 3;
  InferenceArgs inference;
  inference.seq_len = 4096;
  inference.prefill_tbatch_size = 32;
  ThreadingContext ctx{ThreadingArgs{}};
  for (auto impl :
       {AttentionImpl::kFlash, AttentionImpl::kFlashTransposedQsBF16,
        AttentionImpl::kFlashMatrixAccumulation,
        AttentionImpl::kInt8MatrixAccumulation}) {
    KVCache cache(config, inference, impl, ctx.allocator);
    ASSERT_EQ(cache.kv_head_ptrs.size(), 3u);
    EXPECT_EQ(cache.layer_kv_head_offsets[2], cache.layer_kv_head_offsets[0]);
    EXPECT_GE(cache.LayerCapacity(0), 2048u + 32u);
    EXPECT_EQ(cache.LayerCapacity(1), 4096u);
    EXPECT_GT(cache.kv_head_ptrs[1].Cols(), cache.kv_head_ptrs[0].Cols());
    cache.Clear();
    for (auto& ptr : cache.kv_head_ptrs) ptr.RowBytes(0)[0] = 42;
    auto copy = cache.Copy();
    cache.Clear();
    for (size_t h = 0; h < cache.kv_head_ptrs.size(); ++h) {
      EXPECT_EQ(cache.kv_head_ptrs[h].RowBytes(0)[0], 0);
      EXPECT_EQ(copy.kv_head_ptrs[h].RowBytes(0)[0], 42);
      EXPECT_EQ(copy.kv_head_ptrs[h].GetLayout(),
                cache.kv_head_ptrs[h].GetLayout());
    }
  }
}

TEST(KVCacheTest, LocalCapacityAndContextLimit) {
  auto config = RingConfig();
  InferenceArgs inference;
  inference.seq_len = 8192;
  RuntimeConfig runtime{};
  inference.prefill_tbatch_size = 256;
  ThreadingContext ctx{ThreadingArgs{}};
  KVCache cache(config, inference, runtime.attention_impl, ctx.allocator);
  EXPECT_LT(cache.LayerCapacity(0), 1024u);
  EXPECT_EQ(cache.LayerCapacity(1), 8192u);
  EXPECT_FALSE(cache.kv_cache.HasPtr());
  EXPECT_FALSE(cache.kv_head_ptrs.empty());
  EXPECT_EQ(cache.SeqLen(), 8192u);
  inference.seq_len = 8193;
  KVCache capped(config, inference, runtime.attention_impl, ctx.allocator);
  EXPECT_EQ(capped.SeqLen(), 8192u);
}

TEST(KVCacheTest, WrapGrowthAndIndependentSnapshot) {
  auto config = RingConfig();
  InferenceArgs inference;
  inference.seq_len = 8192;
  RuntimeConfig runtime{};
  inference.prefill_tbatch_size = 1;
  ThreadingContext ctx{ThreadingArgs{}};
  KVCache cache(config, inference, runtime.attention_impl, ctx.allocator);
  // Model the real layout after SIMD-specific transpose.
  constexpr size_t tile = 16;
  cache.PrepareLayer(0, 1, 0, tile);
  const size_t original_rows = cache.LayerCapacity(0);
  constexpr size_t pos = 1607;
  for (size_t p = 0; p < pos; ++p) {
    const auto value = hwy::ConvertScalarTo<KV_t>(float(p % 128));
    auto k = cache.FlashK(0, 0);
    auto v = cache.FlashV(0, 0);
    k.Row((p / tile) % k.Rows())[p % tile] = value;
    v.Row((p / tile) % v.Rows())[p % tile] = value;
  }
  auto snapshot = cache.Copy();
  cache.PrepareLayer(0, 1024, pos, tile);
  EXPECT_GT(cache.LayerCapacity(0), original_rows);
  EXPECT_EQ(snapshot.LayerCapacity(0), original_rows);
  EXPECT_NE(cache.FlashK(0, 0).Row(0), snapshot.FlashK(0, 0).Row(0));
  for (size_t p = pos - 511; p < pos; ++p) {
    const float value = float(p % 128);
    auto saved = snapshot.FlashK(0, 0);
    EXPECT_EQ(hwy::ConvertScalarTo<float>(
                  saved.Row((p / tile) % saved.Rows())[p % tile]),
              value);
    auto k = cache.FlashK(0, 0);
    auto v = cache.FlashV(0, 0);
    EXPECT_EQ(
        hwy::ConvertScalarTo<float>(k.Row((p / tile) % k.Rows())[p % tile]),
        value);
    EXPECT_EQ(
        hwy::ConvertScalarTo<float>(v.Row((p / tile) % v.Rows())[p % tile]),
        value);
  }
  cache.Clear();
  auto cleared = cache.FlashK(0, 0);
  auto saved = snapshot.FlashK(0, 0);
  EXPECT_EQ(hwy::ConvertScalarTo<float>(cleared.Row(0)[0]), 0.0f);
  EXPECT_EQ(hwy::ConvertScalarTo<float>(
                saved.Row(((pos - 1) / tile) % saved.Rows())[(pos - 1) % tile]),
            float((pos - 1) % 128));
}

TEST(KVCacheTest, NonAlignedContextAndBatchPadding) {
  auto config = RingConfig();
  InferenceArgs inference;
  inference.seq_len = 1031;
  RuntimeConfig runtime{};
  inference.prefill_tbatch_size = 256;
  ThreadingContext ctx{ThreadingArgs{}};
  KVCache cache(config, inference, runtime.attention_impl, ctx.allocator);
  EXPECT_EQ(cache.SeqLen(), 1031u);
  EXPECT_GE(cache.LayerCapacity(1), 1031u);
  cache.PrepareLayer(0, 1, 1030, 16);
  const size_t rows = cache.LayerCapacity(0);
  // Even the final tile's padding must not wrap onto the oldest live token.
  EXPECT_GT(rows, 511u + 256u);
}

TEST(KVCacheTest, FlashRingMatchesFullCache) {
  auto config = RingConfig();
  config.att_cap = 10.0f;
  ThreadingArgs threads;
  threads.max_threads = 2;
  ThreadingContext ctx(threads);
  const int64_t native_targets = hwy::SupportedTargets();
  std::vector<int64_t> targets = {native_targets};
#if HWY_IS_TEST
  // CMake builds libgemma for all attainable targets alongside these tests.
  targets.push_back(HWY_EMU128);
#endif
  for (int64_t target : targets) {
    hwy::SetSupportedTargetsForTest(target);
    // hwy itself may be built without its emulated target, whereas libgemma
    // is built for all test targets. EMU128 has four float lanes.
    const size_t tile = target == HWY_EMU128 ? 8 : hwy::VectorBytes() / sizeof(KV_t);
    for (size_t queries : {1, 4, 8}) {
      InferenceArgs inference;
      inference.seq_len = 8192;
      RuntimeConfig runtime{};
      inference.prefill_tbatch_size = 256;
      KVCache cache(config, inference, runtime.attention_impl, ctx.allocator);
      cache.PrepareLayer(0, 256, 0, tile);
      auto ring_k = cache.FlashK(0, 0);
      auto ring_v = cache.FlashV(0, 0);
      const size_t cols = ring_k.Cols();
      MatStorageT<KV_t> full_k("full_k", Extents2D(8192 / tile, cols),
                               ctx.allocator, MatPadding::kPacked);
      MatStorageT<KV_t> full_v("full_v", full_k.Extents(), ctx.allocator,
                               MatPadding::kPacked);
      MatStorageT<BF16> q("q", Extents2D(queries, 64), ctx.allocator,
                          MatPadding::kPacked);
      MatStorageT<float> full_out("out", Extents2D(queries, 64), ctx.allocator,
                                  MatPadding::kPacked);
      MatStorageT<float> ring_out("out", full_out.Extents(), ctx.allocator,
                                  MatPadding::kPacked);
      for (size_t i = 0; i < queries; ++i) {
        for (size_t j = 0; j < 64; ++j)
          q.Row(i)[j] = hwy::ConvertScalarTo<BF16>(float((i + j) % 13) * 0.01f);
      }
      std::vector<Tile148Params> params, split;
      AttentionActivationsPtrs activations(config, 8192, params, split);
      for (size_t end : {size_t{769}, size_t{1607}, size_t{8190}}) {
        for (size_t r = 0; r <= end / tile; ++r) {
          for (size_t c = 0; c < cols; ++c) {
            full_k.Row(r)[c] = hwy::ConvertScalarTo<KV_t>(
                float(int((r * 17 + c) % 31) - 15) * 0.01f);
            full_v.Row(r)[c] = hwy::ConvertScalarTo<KV_t>(
                float(int((r * 7 + c) % 23) - 11) * 0.02f);
          }
          hwy::CopyBytes(full_k.Row(r), ring_k.Row(r % ring_k.Rows()),
                         cols * sizeof(KV_t));
          hwy::CopyBytes(full_v.Row(r), ring_v.Row(r % ring_v.Rows()),
                         cols * sizeof(KV_t));
        }
        Tile148Params tile_params{};
        tile_params.v_tile_size = queries;
        tile_params.min_start_pos = end - 511;
        tile_params.max_last_pos = end;
        for (size_t i = 0; i < queries; ++i) {
          tile_params.start_pos[i] = end - 511 + i;
          tile_params.last_pos[i] = end - (queries - 1 - i);
          tile_params.q_offsets[i] = i * 64;
          tile_params.out_offsets[i] = i * 64;
        }
        auto ring_params = tile_params;
        DispatchDispatchTileFlashAttention148(tile_params, q, full_k, full_v, 0,
                                              activations, full_out, 64, ctx, 0,
                                              AttentionImpl::kFlash);
        DispatchDispatchTileFlashAttention148(ring_params, q, ring_k, ring_v, 0,
                                              activations, ring_out, 64, ctx, 0,
                                              AttentionImpl::kFlash);
        for (size_t i = 0; i < queries; ++i) {
          for (size_t c = 0; c < 64; ++c)
            EXPECT_EQ(full_out.Row(i)[c], ring_out.Row(i)[c]);
          EXPECT_EQ(tile_params.end_state.row_states[i].d,
                    ring_params.end_state.row_states[i].d);
        }
      }
    }
  }
  hwy::SetSupportedTargetsForTest(0);
}

}  // namespace
}  // namespace gcpp
