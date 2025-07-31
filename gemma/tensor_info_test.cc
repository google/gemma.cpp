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

}  // namespace
}  // namespace gcpp
