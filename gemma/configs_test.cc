#include "gemma/configs.h"

#include <stdio.h>

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "compression/types.h"  // Type
#include "io/fields.h"           // Type
#include "io/io.h"               // Path

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

TEST(ConfigsTest, ValidateAcceptsAllCanonicalModels) {
  ForEachModel([](Model m) {
    ModelConfig(m, Type::kBF16, ChooseWrapping(m)).Validate();
  });
}

TEST(ConfigsTest, ValidateRejectsMalformedUntrustedConfigs) {
#ifdef GTEST_HAS_DEATH_TEST
  using M = Model;
  const auto chk = [](const char* re, auto fn, M m = M::GEMMA2_2B) {
    ModelConfig c(m, Type::kBF16, ChooseWrapping(m));
    fn(c, c.layer_configs);
    EXPECT_DEATH(c.Validate(), re);
  };
  const auto ds = M::DEEPSEEK4_FLASH, m4 = M::GEMMA4_26B_MOE;
  const auto v4 = M::GEMMA4_2B, pg = M::PALIGEMMA2_3B_224;
  chk("main layer 0: ff_hidden_dim 0",
      [](auto&, auto& l) { l[0].ff_hidden_dim = 0; });
  chk("main layer 1: heads 4194305 \\* qkv_dim .* > kMaxMatMulK",
      [](auto&, auto& l) { l[1].heads = 0x400001; });
  chk("main layer 1: model_dim .* != expected",
      [](auto&, auto& l) { l[1].model_dim += 64; });
  chk("main layer 0: window_size 0",
      [](auto& c, auto&) { c.attention_window_sizes[0] = 0; });
  chk("main layer 1: ple_dim 256 != model ple_dim 0",
      [](auto&, auto& l) { l[1].ple_dim = 256; });
  chk("vocab_size <= kMaxMatCols",
      [](auto& c, auto&) { c.vocab_size = 0xFFFFFFFFu; });
  chk("max_seq_len <= kMaxSeqLen",
      [](auto& c, auto&) { c.max_seq_len = UINT32_MAX; });
  chk("hc_sinkhorn_iters <= kMaxSinkhornIters",
      [](auto& c, auto&) { c.hc_sinkhorn_iters = 999; });
  chk("main layer 0: window_size .* must be in",
      [](auto& c, auto&) { c.attention_window_sizes[0] = c.max_seq_len + 1; });
  chk("main layer 1: heads .* != layer 0 heads",
      [](auto&, auto& l) { l[1].heads *= 2; });
  chk("main layer 0: non-zero indexer fields",
      [](auto&, auto& l) { l[0].indexer_heads = 16; });
  chk("Invalid decoder stack: window_sizes\\.size\\(\\) 0 != 8",
      [](auto& c, auto&) { c.decoder_attention_window_sizes.clear(); },
      M::T5GEMMA_S_S);
  chk("total_mtp_layers <= kMaxLayers",
      [](auto& c, auto&) { c.num_mtp_layers = 100000; }, ds);
  chk("main layer 0: inconsistent MLA config",
      [](auto&, auto& l) { l[0].kv_lora_rank = 0; }, ds);
  chk("main layer 2: indexer requires compressor",
      [](auto&, auto& l) { l[2].kv_compression_rate = 1; }, ds);
  chk("main layer 0: num_experts 256 > max_experts 128",
      [](auto&, auto& l) { l[0].num_experts = 256; }, m4);
  chk("vit_config\\.seq_len <= kMaxBatchSize",
      [](auto& c, auto&) { c.vit_config.seq_len = 8192; }, v4);
  chk("vit_config\\.pool_dim == kGemma4VitPoolDim",
      [](auto& c, auto&) { c.vit_config.pool_dim = 6; }, v4);
  chk("main layer 15: kv_heads 4 > layer 0 kv_heads",
      [](auto&, auto& l) { l[15].kv_heads = 4; }, M::GEMMA4_2B_LM);
  chk("main layer 15: qkv_dim .* != share source",
      [](auto&, auto& l) { l[15].qkv_dim += 64; }, M::GEMMA4_2B_LM);
  chk("vit_config\\.seq_len == kGemmaVlmSeqLen",
      [](auto& c, auto&) { c.vit_config.seq_len = 16; }, M::GEMMA3_4B);
  chk("patches_per_side \\* patches_per_side == vit_config\\.seq_len",
      [](auto& c, auto&) { c.vit_config.image_size = 14; }, pg);
  ModelConfig bad(M::GEMMA2_2B, Type::kBF16, PromptWrapping::GEMMA_IT);
  EXPECT_DEATH(bad.SetMaxSeqLen(0), "new_max_seq_len > 0");
  EXPECT_DEATH(bad.SetMaxSeqLen(UINT32_MAX), "new_max_seq_len <= kMaxSeqLen");
#endif  // GTEST_HAS_DEATH_TEST
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

  const ModelConfig lm_e4b(Model::GEMMA4_E4B_LM, Type::kSFP,
                           PromptWrapping::GEMMA_IT);
  EXPECT_EQ(lm_e4b.display_name, "Gemma4_E4B_LM");
  EXPECT_FALSE(lm_e4b.HasGemma4Vit());
  EXPECT_EQ(lm_e4b.num_layers, 42u);

  const ModelConfig vlm_e4b(Model::GEMMA4_E4B, Type::kSFP,
                            PromptWrapping::GEMMA_IT);
  EXPECT_EQ(vlm_e4b.display_name, "Gemma4_E4B");
  EXPECT_TRUE(vlm_e4b.HasGemma4Vit());
  EXPECT_EQ(vlm_e4b.num_layers, 42u);
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

TEST(ConfigsTest, Gemma4E4BArchitectureInvariants) {
  constexpr size_t kNumLayers = 42;
  constexpr size_t kModelDim = 2560;
  constexpr size_t kHeads = 8;
  constexpr size_t kKvHeads = 2;
  constexpr size_t kLocalQkvDim = 256;
  constexpr size_t kGlobalQkvDim = 512;
  constexpr size_t kFfHiddenDim = 10240;
  constexpr size_t kLocalWindowSize = 512;
  constexpr size_t kKvSharingStartLayer = 24;
  constexpr size_t kGlobalStride = 6;

  const ModelConfig config(Model::GEMMA4_E4B_LM, Type::kSFP,
                           PromptWrapping::GEMMA_IT);
  EXPECT_EQ(config.num_layers, kNumLayers);
  EXPECT_EQ(config.model_dim, kModelDim);
  EXPECT_EQ(config.layer_configs.size(), kNumLayers);
  EXPECT_EQ(config.attention_window_sizes.size(), kNumLayers);

  for (size_t i = 0; i < kNumLayers; ++i) {
    const LayerConfig& layer = config.layer_configs[i];
    EXPECT_EQ(layer.model_dim, kModelDim);
    EXPECT_EQ(layer.heads, kHeads);
    EXPECT_EQ(layer.kv_heads, kKvHeads);
    EXPECT_TRUE(layer.use_qk_norm);
    EXPECT_TRUE(layer.norm_v);
    EXPECT_EQ(layer.ff_hidden_dim, kFfHiddenDim);

    const bool is_global = (i % kGlobalStride == (kGlobalStride - 1));
    if (is_global) {
      EXPECT_EQ(layer.qkv_dim, kGlobalQkvDim);
      EXPECT_EQ(config.attention_window_sizes[i], config.max_seq_len);
    } else {
      EXPECT_EQ(layer.qkv_dim, kLocalQkvDim);
      EXPECT_EQ(config.attention_window_sizes[i], kLocalWindowSize);
    }

    if (i < kKvSharingStartLayer) {
      EXPECT_EQ(layer.kv_share_layer_idx, -1);
    } else {
      const int expected_shared_idx = is_global ? 23 : 22;
      EXPECT_EQ(layer.kv_share_layer_idx, expected_shared_idx);
    }
  }
  EXPECT_EQ(config.NumPrefillLayers(), kKvSharingStartLayer);
}

TEST(ConfigsTest, NumPrefillLayersStopsAfterLastKVOwningLayer) {
  // Gemma 4 2B: layers 15..34 reuse the KV of layers 13/14.
  const ModelConfig gemma4_2b(Model::GEMMA4_2B_LM, Type::kSFP,
                              PromptWrapping::GEMMA_IT);
  EXPECT_EQ(gemma4_2b.num_layers, 35u);
  EXPECT_EQ(gemma4_2b.NumPrefillLayers(), 15u);
  EXPECT_TRUE(gemma4_2b.layer_configs[14].HasOwnKVCache());
  EXPECT_FALSE(gemma4_2b.layer_configs[15].HasOwnKVCache());

  // Without KV sharing every layer is a prefill layer.
  const ModelConfig gemma3_270m(Model::GEMMA3_270M, Type::kSFP,
                                PromptWrapping::GEMMA_IT);
  EXPECT_EQ(gemma3_270m.NumPrefillLayers(), gemma3_270m.layer_configs.size());

  // Only trailing shared layers are skipped: a shared layer followed by a
  // KV-owning one still has to run so that the later layer sees its output.
  ModelConfig mixed = gemma3_270m;
  mixed.layer_configs[5].kv_share_layer_idx = 4;
  EXPECT_EQ(mixed.NumPrefillLayers(), mixed.layer_configs.size());
  const size_t last = mixed.layer_configs.size() - 1;
  mixed.layer_configs[last].kv_share_layer_idx = static_cast<int>(last - 1);
  EXPECT_EQ(mixed.NumPrefillLayers(), last);
}

TEST(ConfigsTest, Gemma4E4BQuantizerSpecifiers) {
  // SFP quantizer
  const ModelConfig e4b_sfp("gemma4-e4b-sfp-it");
  EXPECT_EQ(e4b_sfp.model, Model::GEMMA4_E4B);
  EXPECT_EQ(e4b_sfp.weight, Type::kSFP);
  EXPECT_EQ(e4b_sfp.wrapping, PromptWrapping::GEMMA_IT);

  const ModelConfig e4b_lm_sfp("gemma4-e4b-lm-sfp-it");
  EXPECT_EQ(e4b_lm_sfp.model, Model::GEMMA4_E4B_LM);
  EXPECT_EQ(e4b_lm_sfp.weight, Type::kSFP);
  EXPECT_EQ(e4b_lm_sfp.wrapping, PromptWrapping::GEMMA_IT);

  // BF16 format
  const ModelConfig e4b_bf16("gemma4-e4b-bf16-it");
  EXPECT_EQ(e4b_bf16.model, Model::GEMMA4_E4B);
  EXPECT_EQ(e4b_bf16.weight, Type::kBF16);
  EXPECT_EQ(e4b_bf16.wrapping, PromptWrapping::GEMMA_IT);

  const ModelConfig e4b_lm_bf16("gemma4-e4b-lm-bf16-it");
  EXPECT_EQ(e4b_lm_bf16.model, Model::GEMMA4_E4B_LM);
  EXPECT_EQ(e4b_lm_bf16.weight, Type::kBF16);
  EXPECT_EQ(e4b_lm_bf16.wrapping, PromptWrapping::GEMMA_IT);

  // NUQ quantizer
  const ModelConfig e4b_nuq("gemma4-e4b-nuq-it");
  EXPECT_EQ(e4b_nuq.model, Model::GEMMA4_E4B);
  EXPECT_EQ(e4b_nuq.weight, Type::kNUQ);
  EXPECT_EQ(e4b_nuq.wrapping, PromptWrapping::GEMMA_IT);

  const ModelConfig e4b_lm_nuq("gemma4-e4b-lm-nuq-it");
  EXPECT_EQ(e4b_lm_nuq.model, Model::GEMMA4_E4B_LM);
  EXPECT_EQ(e4b_lm_nuq.weight, Type::kNUQ);
  EXPECT_EQ(e4b_lm_nuq.wrapping, PromptWrapping::GEMMA_IT);

  // Pre-trained (PT) prompt wrapping
  const ModelConfig e4b_lm_pt("gemma4-e4b-lm-sfp-pt");
  EXPECT_EQ(e4b_lm_pt.model, Model::GEMMA4_E4B_LM);
  EXPECT_EQ(e4b_lm_pt.weight, Type::kSFP);
  EXPECT_EQ(e4b_lm_pt.wrapping, PromptWrapping::GEMMA_PT);
}

TEST(ConfigsTest, EmbeddingGemma2) {
  const ModelConfig config(Model::EMBEDDING_GEMMA2_LM, Type::kBF16,
                           PromptWrapping::GEMMA_IT);
  EXPECT_EQ(config.display_name, "EmbeddingGemma2_LM");
  EXPECT_FALSE(config.HasGemma4Vit());
  EXPECT_TRUE(IsEmbeddingGemma2(config.model));
  EXPECT_TRUE(config.IsEmbedding());
  EXPECT_EQ(config.EmbeddingProjDim(), kEmbeddingGemma2Dim);
  EXPECT_EQ(config.num_layers, 24u);
  EXPECT_EQ(config.model_dim, 512u);
  EXPECT_EQ(config.ple_dim, 512u);
  EXPECT_FALSE(config.HasPLETable());
  EXPECT_EQ(config.vocab_size, 262144u);
  EXPECT_EQ(config.max_seq_len, kMaxBatchSize);
  EXPECT_EQ(config.query_scale, QueryScaleType::One);
  EXPECT_TRUE(config.use_global_timescale);
  EXPECT_FLOAT_EQ(config.partial_rotary_factor, 1.0f);
  EXPECT_EQ(config.NumPrefillLayers(), 24u);

  for (size_t i = 0; i < 24; ++i) {
    const LayerConfig& lc = config.layer_configs[i];
    EXPECT_EQ(lc.ple_dim, 512u) << i;
    EXPECT_EQ(lc.ff_hidden_dim, 2048u) << i;
    EXPECT_EQ(lc.heads, 4u) << i;
    EXPECT_EQ(lc.post_qk, PostQKType::Rope) << i;
    EXPECT_TRUE(lc.use_qk_norm) << i;
    EXPECT_TRUE(lc.norm_v) << i;
    const bool is_global = (i % 6 == 5);
    EXPECT_EQ(lc.kv_heads, is_global ? 1u : 2u) << i;
    EXPECT_EQ(lc.qkv_dim, is_global ? 512u : 256u) << i;
    EXPECT_EQ(config.attention_window_sizes[i],
              is_global ? config.max_seq_len : 513u)
        << i;
  }

  const ModelConfig bf16_it("embeddinggemma2-lm-bf16-it");
  EXPECT_EQ(bf16_it.model, Model::EMBEDDING_GEMMA2_LM);
  EXPECT_EQ(bf16_it.weight, Type::kBF16);
  EXPECT_EQ(bf16_it.wrapping, PromptWrapping::GEMMA_IT);

  const ModelConfig sfp_pt("embeddinggemma2-lm-sfp-pt");
  EXPECT_EQ(sfp_pt.model, Model::EMBEDDING_GEMMA2_LM);
  EXPECT_EQ(sfp_pt.weight, Type::kSFP);
  EXPECT_EQ(sfp_pt.wrapping, PromptWrapping::GEMMA_PT);

  EXPECT_EQ(DeduceModel(Path("embeddinggemma2-lm-bf16-it.sbs"), 24, 0),
            Model::EMBEDDING_GEMMA2_LM);
}

}  // namespace gcpp
