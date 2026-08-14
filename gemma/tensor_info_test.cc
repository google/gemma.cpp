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

}  // namespace
}  // namespace gcpp
