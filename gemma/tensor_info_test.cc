#include "gemma/tensor_info.h"

#include <stdio.h>

#include "gtest/gtest.h"
#include "compression/types.h"  // SfpStream
#include "gemma/configs.h"
#include "gemma/weights.h"
#include "util/mat.h"
#include "hwy/base.h"  // HWY_ASSERT_M

namespace gcpp {
namespace {

// Tests for all models that each tensor in the model can be found and that the
// TensorInfoRegistry returns the correct shape and name for the tensor.
TEST(TensorInfoRegistryTest, Find) {
  ForEachModel([&](Model model) {
    const ModelConfig config(model, Type::kSFP, ChooseWrapping(model));
    fprintf(stderr, "Testing %s (%s)\n", config.display_name.c_str(),
            config.Specifier().c_str());
    const TensorInfoRegistry tensors(config);
    // Each tensor in the model should be known/found.
    WeightsPtrs weights(config);
    weights.ForEachTensor(nullptr, nullptr, [&tensors](const TensorArgs& t) {
      const TensorInfo* info = tensors.Find(t.mat.Name());
      HWY_ASSERT_M(info, t.mat.Name());
      // Test that the `MatPtr` can be constructed from the TensorInfo,
      // and that the dimensions match.
      const MatPtr mat_ptr(t.mat.Name(), Type::kUnknown,
                           ExtentsFromInfo(tensors.Find(t.mat.Name())));
      EXPECT_STREQ(t.mat.Name(), mat_ptr.Name()) << t.mat.Name();
      EXPECT_EQ(t.mat.Rows(), mat_ptr.Rows()) << t.mat.Name();
      EXPECT_EQ(t.mat.Cols(), mat_ptr.Cols()) << t.mat.Name();
    });
  });
}

// Gemma 3 LM variants must not request any ViT tensors: their `vit_config`
// stays empty so `WeightsPtrs::ForEachTensor` skips the whole block.
TEST(TensorInfoRegistryTest, LmConfigsHaveNoVit) {
  for (Model model :
       {Model::GEMMA3_4B_LM, Model::GEMMA3_12B_LM, Model::GEMMA3_27B_LM}) {
    const ModelConfig config(model, Type::kSFP, ChooseWrapping(model));
    EXPECT_TRUE(config.vit_config.layer_configs.empty())
        << config.display_name;
    EXPECT_EQ(config.wrapping, PromptWrapping::GEMMA_IT) << config.display_name;

    WeightsPtrs weights(config);
    weights.ForEachTensor(nullptr, nullptr, [](const TensorArgs& t) {
      const std::string name = t.mat.Name();
      EXPECT_EQ(name.find("enc_norm_"), std::string::npos) << name;
      EXPECT_EQ(name.find("img_"), std::string::npos) << name;
      EXPECT_EQ(name.find("mm_embed_norm"), std::string::npos) << name;
    });
  }
}

// FindModel must disambiguate `gemma3-4b-...` and `gemma3-4b-lm-...` by
// preferring the longest matching prefix.
TEST(TensorInfoRegistryTest, FindModelLongestMatch) {
  // Construction via the specifier-string ctor goes through `FindModel`.
  const ModelConfig lm("gemma3-4b-lm-sfp-it");
  EXPECT_EQ(lm.model, Model::GEMMA3_4B_LM);

  const ModelConfig vlm("gemma3-4b-sfp");
  EXPECT_EQ(vlm.model, Model::GEMMA3_4B);
}

// Verify computed tensor shapes for Gemma4 VLM model match expected formulas.
// Catches off-by-one mutations in shape computations.
TEST(TensorInfoRegistryTest, VitGemma4Shapes) {
  const ModelConfig config(Model::GEMMA4_2B, Type::kSFP,
                           PromptWrapping::GEMMA_VLM);
  const TensorInfoRegistry tensors(config);

  // Image embedding kernel: shape = {model_dim, 3 * patch_width^2}
  // = {768, 3 * 16 * 16} = {768, 768}
  const TensorInfo* img_emb = tensors.Find("img_emb_kernel");
  ASSERT_NE(img_emb, nullptr);
  ASSERT_EQ(img_emb->shape.size(), 2u);
  EXPECT_EQ(img_emb->shape[0], 768u);  // model_dim
  EXPECT_EQ(img_emb->shape[1], 3u * 16 * 16);  // 768

  // ViT QKV1 (Q projection): shape = {heads * qkv_dim, model_dim}
  // = {12 * 64, 768} = {768, 768}
  const TensorInfo* qkv1 = tensors.Find("vit_qkv1_w_0");
  ASSERT_NE(qkv1, nullptr);
  ASSERT_EQ(qkv1->shape.size(), 2u);
  EXPECT_EQ(qkv1->shape[0], 12u * 64);  // 768
  EXPECT_EQ(qkv1->shape[1], 768u);  // model_dim

  // ViT QKV2 (KV projection): shape = {2 * kv_heads * qkv_dim, model_dim}
  // = {2 * 12 * 64, 768} = {1536, 768}
  const TensorInfo* qkv2 = tensors.Find("vit_qkv2_w_0");
  ASSERT_NE(qkv2, nullptr);
  ASSERT_EQ(qkv2->shape.size(), 2u);
  EXPECT_EQ(qkv2->shape[0], 2u * 12 * 64);  // 1536
  EXPECT_EQ(qkv2->shape[1], 768u);  // model_dim
}

// Verify computed tensor shapes for Gemma4 E4B model match expected formulas.
TEST(TensorInfoRegistryTest, Gemma4E4BShapes) {
  constexpr size_t kModelDim = 2560;
  constexpr size_t kHeads = 8;
  constexpr size_t kKvHeads = 2;
  constexpr size_t kLocalQkvDim = 256;
  constexpr size_t kGlobalQkvDim = 512;
  constexpr size_t kFfHiddenDim = 10240;

  const ModelConfig config(Model::GEMMA4_E4B, Type::kSFP,
                           PromptWrapping::GEMMA_VLM);
  const TensorInfoRegistry tensors(config);

  // Local layer (Layer 0): Q projection shape = {heads * qkv_dim, model_dim}
  const TensorInfo* qkv1_local = tensors.Find("qkv1_w_0");
  ASSERT_NE(qkv1_local, nullptr);
  ASSERT_EQ(qkv1_local->shape.size(), 2u);
  EXPECT_EQ(qkv1_local->shape[0], kHeads * kLocalQkvDim);  // 2048
  EXPECT_EQ(qkv1_local->shape[1], kModelDim);              // 2560

  // Local layer (Layer 0): KV projection shape = {2 * kv_heads * qkv_dim,
  // model_dim}
  const TensorInfo* qkv2_local = tensors.Find("qkv2_w_0");
  ASSERT_NE(qkv2_local, nullptr);
  ASSERT_EQ(qkv2_local->shape.size(), 2u);
  EXPECT_EQ(qkv2_local->shape[0], 2u * kKvHeads * kLocalQkvDim);  // 1024
  EXPECT_EQ(qkv2_local->shape[1], kModelDim);                     // 2560

  // Local layer (Layer 0): Gate/Up projection shape = {2, ff_hidden_dim,
  // model_dim}
  const TensorInfo* gating_local = tensors.Find("gating_ein_0");
  ASSERT_NE(gating_local, nullptr);
  ASSERT_EQ(gating_local->shape.size(), 3u);
  EXPECT_EQ(gating_local->shape[0], 2u);
  EXPECT_EQ(gating_local->shape[1], kFfHiddenDim);  // 10240
  EXPECT_EQ(gating_local->shape[2], kModelDim);     // 2560

  // Global layer (Layer 5): Q projection shape = {heads * global_qkv_dim,
  // model_dim}
  const TensorInfo* qkv1_global = tensors.Find("qkv1_w_5");
  ASSERT_NE(qkv1_global, nullptr);
  ASSERT_EQ(qkv1_global->shape.size(), 2u);
  EXPECT_EQ(qkv1_global->shape[0], kHeads * kGlobalQkvDim);  // 4096
  EXPECT_EQ(qkv1_global->shape[1], kModelDim);               // 2560

  // Global layer (Layer 5): KV projection shape = {2 * kv_heads *
  // global_qkv_dim, model_dim}
  const TensorInfo* qkv2_global = tensors.Find("qkv2_w_5");
  ASSERT_NE(qkv2_global, nullptr);
  ASSERT_EQ(qkv2_global->shape.size(), 2u);
  EXPECT_EQ(qkv2_global->shape[0], 2u * kKvHeads * kGlobalQkvDim);  // 2048
  EXPECT_EQ(qkv2_global->shape[1], kModelDim);                      // 2560

  // KV-shared layer (Layer 24): Gate/Up shape = {2, ff_hidden_dim, model_dim}
  const TensorInfo* gating_shared = tensors.Find("gating_ein_24");
  ASSERT_NE(gating_shared, nullptr);
  ASSERT_EQ(gating_shared->shape.size(), 3u);
  EXPECT_EQ(gating_shared->shape[0], 2u);
  EXPECT_EQ(gating_shared->shape[1], kFfHiddenDim);  // 10240
  EXPECT_EQ(gating_shared->shape[2], kModelDim);     // 2560

  // KV-shared layer (Layer 24): Down projection shape = {model_dim,
  // ff_hidden_dim}
  const TensorInfo* linear_shared = tensors.Find("linear_w_24");
  ASSERT_NE(linear_shared, nullptr);
  ASSERT_EQ(linear_shared->shape.size(), 2u);
  EXPECT_EQ(linear_shared->shape[0], kModelDim);     // 2560
  EXPECT_EQ(linear_shared->shape[1], kFfHiddenDim);  // 10240
}

}  // namespace
}  // namespace gcpp
