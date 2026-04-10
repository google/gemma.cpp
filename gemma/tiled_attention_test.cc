#include <stddef.h>

#include <iostream>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include "third_party/absl/types/span.h"
#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#include "gemma/activations.h"
#include "gemma/configs.h"
#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "gemma/kv_cache.h"
#include "gemma/kv_transcoding.h"
#include "gemma/weights.h"
#include "util/mat.h"
#include "util/threading_context.h"
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/tiled_attention_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "gemma/tiled_attention.h"
#include "util/test_util.h"
#include "hwy/aligned_allocator.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

using ::testing::FloatNear;
using ::testing::Pointwise;

struct AttentionTestEnv {
  AttentionTestEnv(
      size_t qkv_dim, size_t kv_seq_len, size_t attention_window_size,
      size_t num_kv_heads, size_t num_heads, size_t num_tokens, size_t last_pos,
      float att_cap, size_t layer_idx, size_t layers_total, size_t qbatch_size,
      AttentionImpl attention_impl,
      std::optional<Type> kv_cache_type = {} )
      : ctx(threading_args), env(ctx) {
    layer_config.heads = num_heads;
    layer_config.kv_heads = num_kv_heads;
    layer_config.qkv_dim = qkv_dim;
    layer_config.model_dim = qkv_dim * num_heads;

    model_config.attention_window_sizes = {
        static_cast<uint32_t>(attention_window_size)};
    model_config.att_cap = att_cap;
    model_config.max_seq_len = kv_seq_len;
    model_config.num_layers = layers_total;
    model_config.model_dim = layer_config.model_dim;
    model_config.vocab_size = 1;  // not vit

    for (size_t i = 0; i < model_config.num_layers; ++i) {
      model_config.layer_configs.push_back(layer_config);
    }
    tensor_info_registry = std::make_unique<TensorInfoRegistry>(model_config);
    layer = std::make_unique<LayerWeightsPtrs>(layer_idx, layer_config,
                                               *tensor_info_registry);

    runtime_config.attention_impl = attention_impl;
    runtime_config.kv_cache_type = kv_cache_type;
    inference_args.seq_len = kv_seq_len;

    all_queries.Reserve(qbatch_size);
    kv_caches.reserve(qbatch_size);
    float unpredictable = hwy::Unpredictable1() * 0.01f;
    for (size_t q = 0; q < qbatch_size; ++q) {
      kv_caches.emplace_back(model_config, inference_args, runtime_config,
                             ctx.allocator);
      if (kv_caches.back().compact_kv_cache_ptr.HasPtr()) {
        const size_t tile_size = gcpp::KVCache::kTileSize;
        gcpp::DecodedTile decoded(qkv_dim, tile_size);
        for (size_t i = 0; i < kv_caches.back().compact_kv_cache_ptr.Rows();
             ++i) {
          for (size_t token = 0; token < tile_size; ++token) {
            for (size_t dim = 0; dim < qkv_dim; ++dim) {
              size_t j_k = dim * tile_size + token;
              decoded.k_elem(token, dim) = unpredictable * (i + j_k + 1);

              size_t j_v = qkv_dim * tile_size + token * qkv_dim + dim;
              decoded.v_elem(token, dim) = unpredictable * (i + j_v + 1);
            }
          }

          bool transposed =
              attention_impl == AttentionImpl::kFlashTransposedQsBF16

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gcpp {
HWY_BEFORE_TEST(TiledAttentionTest);
HWY_EXPORT_AND_TEST_P(TiledAttentionTest, TestTransposeStridedQueries);
// TODO() Fix the goldens for the change in KV_t to BF16
// HWY_EXPORT_AND_TEST_P(TiledAttentionTest,
//                       TestLocalAttentionForAllHeadsTokensAndBatch);
HWY_EXPORT_AND_TEST_P(TiledAttentionTest, TestAttentionMultipleTokens);
HWY_EXPORT_AND_TEST_P(TiledAttentionTest, TestAttentionMultipleTokensBF16);
// HWY_EXPORT_AND_TEST_P(TiledAttentionTest,
//                       TestAttentionMultipleTokensAttentionWindowSizeEdgeCase);

HWY_AFTER_TEST();

}  // namespace gcpp

#endif
