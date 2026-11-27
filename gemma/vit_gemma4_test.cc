#include <stddef.h>
#include <stdint.h>

#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "compression/types.h"
#include "gemma/configs.h"
#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "gemma/weights.h"
#include "ops/matmul.h"
#include "paligemma/image.h"
#include "util/mat.h"
#include "util/threading_context.h"
#include "hwy/base.h"

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/vit_gemma4_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// Defined in gemma.cc.
void GenerateImageTokensT(const ModelConfig& config,
                          const RuntimeConfig& runtime_config, size_t seq_len,
                          const WeightsPtrs& weights, const Image& image,
                          ImageTokens& image_tokens, MatMulEnv& env,
                          TimingInfo& timing_info);

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {
namespace {

HWY_EXPORT(GenerateImageTokensT);

void AllocateNonZero(MatPtr& mat, uint64_t seed, ThreadingContext& ctx,
                     std::vector<MatOwner>& owners) {
  if (mat.IsEmpty()) return;
  mat.SetType(Type::kBF16);
  mat.SetScale(1.0f);
  owners.emplace_back();
  owners.back().AllocateFor(mat, ctx.allocator, MatPadding::kPacked);
  BF16* base = reinterpret_cast<BF16*>(mat.RowBytes(0));
  uint64_t s = seed * 0x9E3779B97F4A7C15ULL + 1;
  for (size_t r = 0; r < mat.Rows(); ++r) {
    BF16* row = base + r * mat.Stride();
    for (size_t c = 0; c < mat.Stride(); ++c) {
      s ^= s >> 12;
      s ^= s << 25;
      s ^= s >> 27;
      const int32_t v = static_cast<int32_t>((s * 0x2545F4914F6CDD1DULL) >> 48);
      row[c] =
          hwy::BF16FromF32(static_cast<float>(v - 32768) * (0.05f / 32768.0f));
    }
  }
}

// Verifies that Gemma 4 ViT trims `image_tokens` rows to the actual number of
// pooled soft tokens rather than leaving trailing zero-padded rows.
TEST(VitGemma4Test, TrimsTrailingZeroPaddedSoftTokens) {
  ThreadingContext ctx{ThreadingArgs()};
  MatMulEnv env(ctx);

  ModelConfig config(Model::GEMMA4_2B, Type::kBF16, PromptWrapping::GEMMA_IT);
  config.model_dim = 128;
  config.num_layers = 0;
  config.layer_configs.clear();
  config.vit_config.model_dim = 128;
  config.vit_config.seq_len = 36;  // Up to 4 soft tokens after 3x3 pooling.
  config.vit_config.layer_configs.resize(1);
  LayerConfig& vlc = config.vit_config.layer_configs[0];
  vlc.model_dim = 128;
  vlc.ff_hidden_dim = 256;
  vlc.heads = 2;
  vlc.kv_heads = 2;
  vlc.qkv_dim = 64;

  auto weights = std::make_unique<WeightsPtrs>(config);
  std::vector<MatOwner> owners;
  uint64_t seed = 1;
  AllocateNonZero(weights->vit_img_embedding_kernel, ++seed, ctx, owners);
  AllocateNonZero(weights->vit_img_pos_embedding, ++seed, ctx, owners);
  AllocateNonZero(weights->vit_img_head_kernel, ++seed, ctx, owners);
  LayerWeightsPtrs* vlayer = weights->VitLayer(0);
  AllocateNonZero(vlayer->qkv_einsum_w1, ++seed, ctx, owners);
  AllocateNonZero(vlayer->qkv_einsum_w2, ++seed, ctx, owners);
  AllocateNonZero(vlayer->attn_vec_einsum_w, ++seed, ctx, owners);
  AllocateNonZero(vlayer->gating_einsum_w1, ++seed, ctx, owners);
  AllocateNonZero(vlayer->gating_einsum_w2, ++seed, ctx, owners);
  AllocateNonZero(vlayer->linear_w, ++seed, ctx, owners);
  AllocateNonZero(vlayer->pre_attention_norm_scale, ++seed, ctx, owners);
  AllocateNonZero(vlayer->post_attention_norm_scale, ++seed, ctx, owners);
  AllocateNonZero(vlayer->pre_ffw_norm_scale, ++seed, ctx, owners);
  AllocateNonZero(vlayer->post_ffw_norm_scale, ++seed, ctx, owners);
  AllocateNonZero(vlayer->query_norm_scale, ++seed, ctx, owners);
  AllocateNonZero(vlayer->key_norm_scale, ++seed, ctx, owners);

  const size_t pool_dim = config.vit_config.pool_dim;
  const size_t max_soft_tokens =
      config.vit_config.seq_len / (pool_dim * pool_dim);
  ASSERT_EQ(max_soft_tokens, 4u);

  // Reuse one ImageTokens buffer across shrinking and growing pooled counts.
  ImageTokens image_tokens("image_tokens",
                           Extents2D(max_soft_tokens, config.model_dim),
                           ctx.allocator, MatPadding::kOdd);
  image_tokens.AllocateAndAttachRowPtrs(env.row_ptrs);

  struct Case {
    int width;
    int height;
    size_t expected_rows;
  };
  const Case kCases[] = {
      {100, 200, 2},  // 48x96 -> 1x2 pooled tokens (would be 4 without trim).
      {96, 96, 4},    // Grows back to full max_soft_tokens on same buffer.
      {200, 100, 2},  // Shrinks again to 2x1 pooled tokens.
      {1, 1, 4},      // Very small image upscaled to 96x96.
      {4096, 16, 4},  // Very wide image clamped to 192x48.
  };

  for (const Case& tc : kCases) {
    std::vector<float> pixels(static_cast<size_t>(tc.width) * tc.height * 3);
    for (size_t i = 0; i < pixels.size(); ++i) {
      pixels[i] = static_cast<float>(i % 251);
    }
    Image image;
    image.Set(tc.width, tc.height, pixels.data());

    TimingInfo timing_info;
    RuntimeConfig runtime_config;
    HWY_DYNAMIC_DISPATCH(GenerateImageTokensT)(config, runtime_config,
                                               /*seq_len=*/64, *weights, image,
                                               image_tokens, env, timing_info);

    ASSERT_EQ(image_tokens.Rows(), tc.expected_rows)
        << tc.width << "x" << tc.height;
    for (size_t r = 0; r < image_tokens.Rows(); ++r) {
      double row_norm_sq = 0.0;
      const float* row = image_tokens.Row(r);
      for (size_t c = 0; c < image_tokens.Cols(); ++c) {
        ASSERT_TRUE(std::isfinite(row[c]));
        row_norm_sq += static_cast<double>(row[c]) * row[c];
      }
      EXPECT_GT(row_norm_sq, 0.0)
          << tc.width << "x" << tc.height << " row " << r;
    }
  }
}

}  // namespace
}  // namespace gcpp
#endif  // HWY_ONCE
