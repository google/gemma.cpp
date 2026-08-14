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

TEST(ConfigsTest, DisplayNamesSetByConfig) {
  // Directly test that config construction sets display_name,
  // without relying on OverwriteWithCanonical.
  const ModelConfig lm(Model::GEMMA4_2B_LM, Type::kSFP,
                       PromptWrapping::GEMMA_IT);
  EXPECT_EQ(lm.display_name, "Gemma4_2B_LM");
  EXPECT_FALSE(lm.HasGemma4Vit());

  const ModelConfig vlm(Model::GEMMA4_2B, Type::kSFP,
                        PromptWrapping::GEMMA_IT);
  EXPECT_EQ(vlm.display_name, "Gemma4_2B");
  EXPECT_TRUE(vlm.HasGemma4Vit());
}

TEST(ConfigsTest, WrappingPreservedForVLM) {
  EXPECT_TRUE(IsVlmWrapping(PromptWrapping::GEMMA_VLM));
  EXPECT_TRUE(IsVlmWrapping(PromptWrapping::PALIGEMMA));
  EXPECT_FALSE(IsVlmWrapping(PromptWrapping::GEMMA_IT));
  EXPECT_FALSE(IsVlmWrapping(PromptWrapping::GEMMA_PT));

  // VLM models must keep their config-defined wrapping even when the
  // constructor receives a different wrapping argument.
  // Model::GEMMA3_1B has wrapping = PromptWrapping::GEMMA_VLM.
  const ModelConfig vlm(Model::GEMMA3_1B, Type::kSFP,
                        PromptWrapping::GEMMA_IT);
  EXPECT_EQ(vlm.wrapping, PromptWrapping::GEMMA_VLM);

  // PALIGEMMA2_3B_224 has wrapping = PromptWrapping::PALIGEMMA.
  const ModelConfig pali(Model::PALIGEMMA2_3B_224, Type::kSFP,
                         PromptWrapping::GEMMA_IT);
  EXPECT_EQ(pali.wrapping, PromptWrapping::PALIGEMMA);

  // Non-VLM models should accept the wrapping argument.
  const ModelConfig lm(Model::GEMMA4_2B_LM, Type::kSFP,
                       PromptWrapping::GEMMA_PT);
  EXPECT_EQ(lm.wrapping, PromptWrapping::GEMMA_PT);
}

}  // namespace gcpp
