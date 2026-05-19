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

}  // namespace
}  // namespace gcpp
