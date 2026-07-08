#include "gemma/configs.h"

#include <stdio.h>

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "compression/types.h"  // Type
#include "io/fields.h"           // Type

namespace gcpp {

TEST(ConfigsTest, TestAll) {
  ForEachModel([&](Model model) {
    ModelConfig config(model, Type::kSFP, ChooseWrapping(model));
    fprintf(stderr, "Testing %s (%s)\n", config.display_name.c_str(),
            config.Specifier().c_str());
    HWY_ASSERT(config.model == model);

    // We can deduce the model/display_name from all other fields.
    config.model = Model::UNKNOWN;
    const std::string saved_display_name = config.display_name;
    config.display_name.clear();
    HWY_ASSERT(config.OverwriteWithCanonical());
    HWY_ASSERT(config.model == model);
    HWY_ASSERT(config.display_name == saved_display_name);

    const std::vector<uint32_t> serialized = config.Write();
    ModelConfig deserialized;
    const IFields::ReadResult result =
        deserialized.Read(hwy::Span<const uint32_t>(serialized), /*pos=*/0);
    HWY_ASSERT(result.pos == serialized.size());
    // We wrote it, so all fields should be known, and no extra.
    HWY_ASSERT(result.extra_u32 == 0);
    HWY_ASSERT(result.missing_fields == 0);
    // All fields should match.
    HWY_ASSERT(deserialized.TestEqual(config, /*print=*/true));
    HWY_ASSERT(deserialized.model == model);
    HWY_ASSERT(deserialized.display_name == saved_display_name);
  });
}

TEST(ConfigsTest, TestAttentionImpl) {
  for (int i = 0; i < static_cast<int>(AttentionImpl::kSentinel); ++i) {
    AttentionImpl impl = static_cast<AttentionImpl>(i);
    std::string name = GetAttentionImplName(impl);
    ASSERT_NE(name, "unknown");
    ASSERT_EQ(GetAttentionImpl(name), impl);
  }
  ASSERT_EQ(GetAttentionImplName(AttentionImpl::kSentinel), "unknown");
  ASSERT_EQ(GetAttentionImpl("unknown"), AttentionImpl::kFlash);
  ASSERT_EQ(GetAttentionImpl("invalid"), AttentionImpl::kFlash);
}

TEST(ConfigsTest, T5GemmaKVCacheUsesDecoderLayers) {
  ModelConfig config(Model::T5GEMMA_S_S, Type::kSFP, PromptWrapping::GEMMA_PT);
  ASSERT_TRUE(config.is_encoder_decoder);
  ASSERT_FALSE(config.decoder_layer_configs.empty());

  const size_t expected_cols = config.decoder_layer_configs.size() *
                               config.decoder_layer_configs[0].CacheLayerSize();
  EXPECT_EQ(config.KVCacheCols(), expected_cols);
}

TEST(ConfigsTest, DeduceT5GemmaSS) {
  EXPECT_EQ(DeduceModel(Path("t5gemma-s-s.sbs"), 8, kDeducedT5Gemma),
            Model::T5GEMMA_S_S);
}

TEST(ConfigsTest, T5GemmaBF16Specifier) {
  ModelConfig config("t5gemma-s-s-bf16-it");
  EXPECT_EQ(config.model, Model::T5GEMMA_S_S);
  EXPECT_EQ(config.weight, Type::kBF16);
  EXPECT_EQ(config.wrapping, PromptWrapping::GEMMA_IT);
  EXPECT_TRUE(config.is_encoder_decoder);
}

}  // namespace gcpp
