#include "gemma/configs.h"

#include <array>
#include <cstddef>
#include <type_traits>

#include "gtest/gtest.h"

namespace gcpp {

template <size_t kNum>
constexpr std::array<LayerAttentionType, kNum> OldFixedLayerConfig(
    LayerAttentionType type) {
  std::array<LayerAttentionType, kNum> config = {};
  for (LayerAttentionType& l : config) {
    l = type;
  }
  return config;
}

template <size_t kNum>
constexpr std::array<size_t, kNum> OldFixedAttentionWindowSizes(
    size_t window_size) {
  std::array<size_t, kNum> window_size_configs = {};
  for (size_t& l : window_size_configs) {
    l = window_size;
  }
  return window_size_configs;
}

// Repeat window_size_pattern for kNum / kPatternSize times.
template <size_t kNum, size_t kPatternSize>
constexpr std::array<size_t, kNum> OldRepeatedAttentionWindowSizes(
    const std::array<size_t, kPatternSize>& window_size_pattern) {
  static_assert(kNum % kPatternSize == 0,
                "kNum must be a multiple of kPatternSize");
  std::array<size_t, kNum> window_size_configs = {};
  for (size_t i = 0; i < kNum; ++i) {
    window_size_configs[i] = window_size_pattern[i % kPatternSize];
  }
  return window_size_configs;
}

template <size_t kNumLayers>
constexpr size_t OldNumLayersOfTypeBefore(
    const std::array<LayerAttentionType, kNumLayers>& layers,
    LayerAttentionType type, size_t num) {
  size_t count = 0;
  for (size_t i = 0; i < num; i++) {
    if (layers[i] == type) count++;
  }
  return count;
}

template <class TConfig, typename = void>
struct CacheLayerSize {
  constexpr size_t operator()() const {
    return TConfig::kKVHeads * TConfig::kQKVDim * 2;
  }
};

template <class TConfig, typename = void>
struct CachePosSize {
  constexpr size_t operator()() const {
    return TConfig::kGemmaLayers * CacheLayerSize<TConfig>()();
  }
};

struct OldConfigNoVit {
  struct VitConfig {
    // Some of these are needed to make the compiler happy when trying to
    // generate code that will actually never be used.
    using Weight = float;
    static constexpr int kLayers = 0;
    static constexpr std::array<LayerAttentionType, 0> kLayerConfig =
        OldFixedLayerConfig<0>(LayerAttentionType::kVit);
    static constexpr int kModelDim = 0;
    static constexpr int kFFHiddenDim = 0;
    static constexpr int kHeads = 1;  // Avoid division by 0 in griffin gate_w.
    static constexpr int kKVHeads = 0;
    static constexpr int kQKVDim = 0;
    static constexpr int kSeqLen = 0;
    static constexpr ResidualType kResidual = ResidualType::Add;
    static constexpr int kGriffinLayers = 0;
    static constexpr int kConv1dWidth = 0;
    static constexpr bool kFFBiases = false;
    static constexpr bool kSoftmaxAttnOutputBiases = false;
    static constexpr PostNormType kPostNorm = PostNormType::None;
  };
};

struct OldConfigNoSSM : OldConfigNoVit {
  static constexpr int kGriffinLayers = 0;

  static constexpr int kConv1dWidth = 0;
  static constexpr bool kFFBiases = false;
  static constexpr bool kSoftmaxAttnOutputBiases = false;
  static constexpr bool kUseHalfRope = false;
  static constexpr bool kUseLocalAttention = false;
  static constexpr bool kInterleaveQKV = true;
  static constexpr PostQKType kPostQK = PostQKType::Rope;
  static constexpr ActivationType kActivation = ActivationType::Gelu;
  static constexpr ResidualType kResidual = ResidualType::Add;
};

struct OldConfigBaseGemmaV1 : OldConfigNoSSM {
  static constexpr float kAttCap = 0.0f;
  static constexpr float kFinalCap = 0.0f;
  static constexpr PostNormType kPostNorm = PostNormType::None;
  static constexpr QueryScaleType kQueryScale = QueryScaleType::SqrtKeySize;
};

struct OldConfigBaseGemmaV2 : OldConfigNoSSM {
  static constexpr float kAttCap = 50.0f;
  static constexpr float kFinalCap = 30.0f;
  static constexpr PostNormType kPostNorm = PostNormType::Scale;
};

template <typename TWeight>
struct OldConfigGemma2_27B : public OldConfigBaseGemmaV2 {
  using Weight = TWeight;  // make accessible where we only have a TConfig

  static constexpr int kSeqLen = 8192;
  static constexpr int kVocabSize = gcpp::kVocabSize;
  static constexpr std::array<LayerAttentionType, 46> kLayerConfig =
      OldFixedLayerConfig<46>(LayerAttentionType::kGemma);
  static constexpr std::array<size_t, 46> kAttentionWindowSizes =
      OldRepeatedAttentionWindowSizes<46, 2>({4096, kSeqLen});
  static constexpr int kLayers = kLayerConfig.size();
  static constexpr int kNumTensorScales = 4 * kLayers;
  static constexpr int kGemmaLayers = kLayers;
  static constexpr int kModelDim = 4608;
  static constexpr int kFFHiddenDim = 16 * 4608 / 2;  // = 36864
  static constexpr int kHeads = 32;
  static constexpr int kKVHeads = 16;
  static constexpr int kQKVDim = 128;  // query size == key size == value size
  static constexpr int kTopK = gcpp::kTopK;
  static constexpr bool kAbsolutePE = false;
  static constexpr QueryScaleType kQueryScale =
      QueryScaleType::SqrtModelDimDivNumHeads;
};

template <typename TWeight>
struct OldConfigGemma2_9B : public OldConfigBaseGemmaV2 {
  using Weight = TWeight;  // make accessible where we only have a TConfig

  static constexpr int kSeqLen = 8192;
  static constexpr int kVocabSize = gcpp::kVocabSize;
  static constexpr std::array<LayerAttentionType, 42> kLayerConfig =
      OldFixedLayerConfig<42>(LayerAttentionType::kGemma);
  static constexpr std::array<size_t, 42> kAttentionWindowSizes =
      OldRepeatedAttentionWindowSizes<42, 2>({4096, kSeqLen});
  static constexpr int kLayers = kLayerConfig.size();
  static constexpr int kNumTensorScales = 4 * kLayers;
  static constexpr int kGemmaLayers = kLayers;
  static constexpr int kModelDim = 3584;
  static constexpr int kFFHiddenDim = 8 * 3584 / 2;  // = 14336
  static constexpr int kHeads = 16;
  static constexpr int kKVHeads = 8;
  static constexpr int kQKVDim = 256;  // query size == key size == value size
  static constexpr int kTopK = gcpp::kTopK;
  static constexpr bool kAbsolutePE = false;
  static constexpr QueryScaleType kQueryScale = QueryScaleType::SqrtKeySize;
};

template <typename TWeight>
struct OldConfigGemma7B : public OldConfigBaseGemmaV1 {
  using Weight = TWeight;  // make accessible where we only have a TConfig

  static constexpr int kSeqLen = gcpp::kSeqLen;
  static constexpr int kVocabSize = gcpp::kVocabSize;
  static constexpr std::array<LayerAttentionType, 28> kLayerConfig =
      OldFixedLayerConfig<28>(LayerAttentionType::kGemma);
  static constexpr std::array<size_t, 28> kAttentionWindowSizes =
      OldFixedAttentionWindowSizes<28>(kSeqLen);
  static constexpr int kLayers = kLayerConfig.size();
  static constexpr int kNumTensorScales = 4 * kLayers;
  static constexpr int kGemmaLayers = kLayers;
  static constexpr int kModelDim = 3072;
  static constexpr int kFFHiddenDim = 16 * 3072 / 2;  // = 24576
  static constexpr int kHeads = 16;
  static constexpr int kKVHeads = 16;  // standard MHA
  static constexpr int kQKVDim = 256;  // query size == key size == value size
  static constexpr int kTopK = gcpp::kTopK;
  static constexpr bool kAbsolutePE = false;
};

template <typename TWeight>
struct OldConfigGemma2B : public OldConfigBaseGemmaV1 {
  using Weight = TWeight;  // make accessible where we only have a TConfig

  static constexpr int kSeqLen = gcpp::kSeqLen;
  static constexpr int kVocabSize = gcpp::kVocabSize;
  static constexpr std::array<LayerAttentionType, 18> kLayerConfig =
      OldFixedLayerConfig<18>(LayerAttentionType::kGemma);
  static constexpr std::array<size_t, 18> kAttentionWindowSizes =
      OldFixedAttentionWindowSizes<18>(kSeqLen);
  static constexpr int kLayers = kLayerConfig.size();
  static constexpr int kNumTensorScales = 4 * kLayers;
  static constexpr int kGemmaLayers = kLayers;
  static constexpr int kModelDim = 2048;
  static constexpr int kFFHiddenDim = 16 * 2048 / 2;  // = 16384
  static constexpr int kHeads = 8;
  static constexpr int kKVHeads = 1;
  static constexpr int kQKVDim = 256;  // query size == key size == value size
  static constexpr int kTopK = gcpp::kTopK;
  static constexpr bool kAbsolutePE = false;
};

template <typename TWeight>
struct OldConfigPaliGemma_224 : public OldConfigGemma2B<TWeight> {
  // On the LM side, the vocab size is one difference to Gemma1-2B in the
  // architecture. PaliGemma adds 1024 <locNNNN> and 128 <segNNN> tokens.
  static constexpr int kVocabSize = 256000 + 1024 + 128;  // = 257152

  // Sub-config for the Vision-Transformer part.
  struct VitConfig : public OldConfigNoSSM {
    using Weight = TWeight;
    // The ViT parts. https://arxiv.org/abs/2305.13035
    // "SoViT-400m/14 [...] has a width of 1152, depth 27, and MLP dim 4304."
    static constexpr std::array<LayerAttentionType, 27> kLayerConfig =
        OldFixedLayerConfig<27>(LayerAttentionType::kVit);
    static constexpr int kLayers = kLayerConfig.size();
    static constexpr int kNumTensorScales = 4 * kLayers;
    static constexpr int kModelDim = 1152;
    static constexpr int kFFHiddenDim = 4304;
    static constexpr int kHeads = 16;
    static constexpr int kKVHeads = 16;  // standard MHA
    static constexpr int kQKVDim = 72;
    static constexpr int kSeqLen = 16 * 16;  // 256
    static constexpr bool kFFBiases = true;
    // The Vit part does not have a vocabulary, the image patches are embedded.
    static constexpr int kVocabSize = 0;
    // Dimensions related to image processing.
    static constexpr int kPatchWidth = 14;
    static constexpr int kImageSize = 224;
    // Necessary constant for the layer configuration.
    static constexpr PostNormType kPostNorm = PostNormType::None;
  };
};

template <typename TWeight>
struct OldConfigGemma2_2B : public OldConfigBaseGemmaV2 {
  using Weight = TWeight;  // make accessible where we only have a TConfig

  static constexpr int kSeqLen = 8192;
  static constexpr int kVocabSize = gcpp::kVocabSize;
  static constexpr std::array<LayerAttentionType, 26> kLayerConfig =
      OldFixedLayerConfig<26>(LayerAttentionType::kGemma);
  static constexpr std::array<size_t, 26> kAttentionWindowSizes =
      OldRepeatedAttentionWindowSizes<26, 2>({4096, kSeqLen});
  static constexpr int kLayers = kLayerConfig.size();
  static constexpr int kNumTensorScales = 4 * kLayers;
  static constexpr int kGemmaLayers = kLayers;
  static constexpr int kModelDim = 2304;
  static constexpr int kFFHiddenDim = 8 * 2304 / 2;  // = 9216
  static constexpr int kHeads = 8;
  static constexpr int kKVHeads = 4;
  static constexpr int kQKVDim = 256;  // query size == key size == value size
  static constexpr int kTopK = gcpp::kTopK;
  static constexpr bool kAbsolutePE = false;
  static constexpr QueryScaleType kQueryScale = QueryScaleType::SqrtKeySize;
};

template <typename TWeight>
struct OldConfigGemmaTiny : public OldConfigNoSSM {
  using Weight = TWeight;  // make accessible where we only have a TConfig

  static constexpr int kSeqLen = 32;
  static constexpr int kVocabSize = 64;
  static constexpr std::array<LayerAttentionType, 3> kLayerConfig =
      OldFixedLayerConfig<3>(LayerAttentionType::kGemma);
  static constexpr std::array<size_t, 3> kAttentionWindowSizes =
      OldFixedAttentionWindowSizes<3>(kSeqLen);
  static constexpr int kLayers = kLayerConfig.size();
  static constexpr int kNumTensorScales = 4 * kLayers;
  static constexpr int kGemmaLayers = kLayers;
  static constexpr int kModelDim = 128;
  static constexpr int kFFHiddenDim = 256;
  static constexpr int kHeads = 4;
  static constexpr int kKVHeads = 1;
  static constexpr int kQKVDim = 16;  // query size == key size == value size
  static constexpr int kTopK = gcpp::kTopK;
  static constexpr bool kAbsolutePE = false;
  static constexpr PostNormType kPostNorm = PostNormType::None;
  static constexpr QueryScaleType kQueryScale = QueryScaleType::SqrtKeySize;

  static constexpr float kAttCap = 0.0f;
  // This is required for optimize_test to pass.
  static constexpr float kFinalCap = 30.0f;
};

template <typename TWeight>
struct OldConfigGriffin2B : OldConfigNoVit {
  using Weight = TWeight;  // make accessible where we only have a TConfig

  // Griffin uses local attention, so kSeqLen is actually the local attention
  // window.
  static constexpr int kSeqLen = 2048;
  static constexpr int kVocabSize = gcpp::kVocabSize;
  static constexpr std::array<LayerAttentionType, 26> kLayerConfig = {
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGemma,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGemma,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGemma,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGemma,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGemma,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGemma,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGemma,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGemma,
      LayerAttentionType::kGriffinRecurrentBlock,
      LayerAttentionType::kGriffinRecurrentBlock,
  };
  static constexpr std::array<size_t, 26> kAttentionWindowSizes =
      OldFixedAttentionWindowSizes<26>(kSeqLen);
  static constexpr int kLayers = kLayerConfig.size();
  static constexpr int kGemmaLayers = OldNumLayersOfTypeBefore(
      kLayerConfig, LayerAttentionType::kGemma, kLayers);
  static constexpr int kGriffinLayers = OldNumLayersOfTypeBefore(
      kLayerConfig, LayerAttentionType::kGriffinRecurrentBlock, kLayers);
  static constexpr int kModelDim = 2560;
  static constexpr int kFFHiddenDim = 7680;
  static constexpr int kHeads = 10;
  static constexpr int kKVHeads = 1;
  static constexpr int kQKVDim = 256;  // query size == key size == value size
  static constexpr int kTopK = gcpp::kTopK;
  static constexpr bool kAbsolutePE = false;
  static constexpr PostNormType kPostNorm = PostNormType::None;

  // No SoftCap.
  static constexpr float kAttCap = 0.0f;
  static constexpr float kFinalCap = 0.0f;

  // SSM config.
  static constexpr int kConv1dWidth = 4;
  static constexpr bool kFFBiases = true;
  static constexpr bool kSoftmaxAttnOutputBiases = true;
  static constexpr bool kUseHalfRope = true;
  static constexpr bool kUseLocalAttention = true;
  static constexpr bool kInterleaveQKV = false;
  static constexpr int kNumTensorScales = 140;
  static constexpr PostQKType kPostQK = PostQKType::Rope;
  static constexpr ActivationType kActivation = ActivationType::Gelu;
  static constexpr QueryScaleType kQueryScale = QueryScaleType::SqrtKeySize;
  static constexpr ResidualType kResidual = ResidualType::Add;
};

template <class TConfig>
void AssertMatch(const ModelConfig& config) {
  ASSERT_EQ(TConfig::kModelDim, config.model_dim);
  if constexpr (TConfig::VitConfig::kModelDim != 0) {
    ASSERT_EQ(TConfig::VitConfig::kModelDim, config.vit_model_dim);
    ASSERT_EQ(TConfig::VitConfig::kSeqLen, config.vit_seq_len);
    ASSERT_EQ(TConfig::VitConfig::kNumTensorScales, config.num_vit_scales);
    for (size_t i = 0; i < config.vit_layer_configs.size(); ++i) {
      ASSERT_EQ(TConfig::VitConfig::kLayerConfig[i],
                config.vit_layer_configs[i].type);
    }
  }
  ASSERT_EQ(TConfig::kVocabSize, config.vocab_size);
  ASSERT_EQ(TConfig::kSeqLen, config.seq_len);
  // ASSERT_EQ(TConfig::kTopK, config.top_k); - is now a runtime config value.
  ASSERT_EQ(TConfig::kAttCap, config.att_cap);
  ASSERT_EQ(TConfig::kFinalCap, config.final_cap);
  ASSERT_EQ(TConfig::kAbsolutePE, config.absolute_pe);
  ASSERT_EQ(TConfig::kUseLocalAttention, config.use_local_attention);
  ASSERT_EQ(TConfig::kQueryScale, config.query_scale);
  ASSERT_EQ(TConfig::kGemmaLayers,
            config.NumLayersOfType(LayerAttentionType::kGemma));
  ASSERT_EQ(TConfig::kGriffinLayers,
            config.NumLayersOfType(LayerAttentionType::kGriffinRecurrentBlock));
  for (size_t i = 0; i < config.layer_configs.size(); ++i) {
    ASSERT_EQ(TConfig::kModelDim, config.layer_configs[i].model_dim);
    ASSERT_EQ(TConfig::kFFHiddenDim, config.layer_configs[i].ff_hidden_dim);
    ASSERT_EQ(TConfig::kHeads, config.layer_configs[i].heads);
    ASSERT_EQ(TConfig::kKVHeads, config.layer_configs[i].kv_heads);
    ASSERT_EQ(TConfig::kQKVDim, config.layer_configs[i].qkv_dim);
    ASSERT_EQ(TConfig::kConv1dWidth, config.layer_configs[i].conv1d_width);
    ASSERT_EQ(TConfig::kFFBiases, config.layer_configs[i].ff_biases);
    ASSERT_EQ(TConfig::kSoftmaxAttnOutputBiases,
              config.layer_configs[i].softmax_attn_output_biases);
    ASSERT_EQ(TConfig::kPostNorm, config.layer_configs[i].post_norm);
    ASSERT_EQ(TConfig::kLayerConfig[i], config.layer_configs[i].type);
    ASSERT_EQ(TConfig::kActivation, config.layer_configs[i].activation);
    PostQKType post_qk = TConfig::kPostQK;
    if (TConfig::kUseHalfRope) {
      post_qk = PostQKType::HalfRope;
    }
    ASSERT_EQ(post_qk, config.layer_configs[i].post_qk);
  }

  ASSERT_EQ(TConfig::kAttentionWindowSizes.size(),
            config.attention_window_sizes.size());
  for (size_t i = 0; i < config.attention_window_sizes.size(); ++i) {
    ASSERT_EQ(TConfig::kAttentionWindowSizes[i],
              config.attention_window_sizes[i]);
  }
  ASSERT_EQ(TConfig::kNumTensorScales, config.num_tensor_scales);
}

TEST(ConfigsTest, OldConfigGemma2B) {
  AssertMatch<OldConfigGemma2B<float>>(ConfigFromModel(Model::GEMMA_2B));
}

TEST(ConfigsTest, OldConfigGemma7B) {
  AssertMatch<OldConfigGemma7B<float>>(ConfigFromModel(Model::GEMMA_7B));
}

TEST(ConfigsTest, OldConfigGemma2_2B) {
  AssertMatch<OldConfigGemma2_2B<float>>(ConfigFromModel(Model::GEMMA2_2B));
}

TEST(ConfigsTest, OldConfigGemma2_9B) {
  AssertMatch<OldConfigGemma2_9B<float>>(ConfigFromModel(Model::GEMMA2_9B));
}

TEST(ConfigsTest, OldConfigGemma2_27B) {
  AssertMatch<OldConfigGemma2_27B<float>>(ConfigFromModel(Model::GEMMA2_27B));
}

TEST(ConfigsTest, OldConfigGriffin2B) {
  AssertMatch<OldConfigGriffin2B<float>>(ConfigFromModel(Model::GRIFFIN_2B));
}

TEST(ConfigsTest, OldConfigGemmaTiny) {
  AssertMatch<OldConfigGemmaTiny<float>>(ConfigFromModel(Model::GEMMA_TINY));
}

TEST(ConfigsTest, OldConfigPaliGemma_224) {
  AssertMatch<OldConfigPaliGemma_224<float>>(
      ConfigFromModel(Model::PALIGEMMA_224));
}

}  // namespace gcpp
