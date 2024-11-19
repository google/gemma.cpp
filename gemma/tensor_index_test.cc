#include "gemma/tensor_index.h"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "compression/compress.h"
#include "compression/shared.h"
#include "gemma/configs.h"
#include "gemma/weights.h"
#include "util/basics.h"
#include "hwy/aligned_allocator.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {
namespace {

// Tests that each tensor in the model can be found by exactly one TensorIndex,
// and that the TensorIndex returns the correct shape and name for the tensor,
// for all models.
TEST(TensorIndexTest, FindName) {
  hwy::ThreadPool pool(4);
  for (Model model : kAllModels) {
    fprintf(stderr, "Testing model %d\n", static_cast<int>(model));
    ModelConfig config = ConfigFromModel(model);
    std::vector<TensorIndex> tensor_indexes;
    tensor_indexes.emplace_back(config, /*llm_layer_idx=*/-1,
                                /*img_layer_idx=*/-1,
                                /*split_and_reshape=*/false);
    for (size_t llm_layer_idx = 0; llm_layer_idx < config.layer_configs.size();
         ++llm_layer_idx) {
      tensor_indexes.emplace_back(config, static_cast<int>(llm_layer_idx),
                                  /*img_layer_idx=*/-1,
                                   /*split_and_reshape=*/false);
    }
    for (size_t img_layer_idx = 0;
         img_layer_idx < config.vit_layer_configs.size();
         ++img_layer_idx) {
      tensor_indexes.emplace_back(config, /*llm_layer_idx=*/-1,
                                  static_cast<int>(img_layer_idx),
                                  /*split_and_reshape=*/false);
    }
    // For each tensor in any model, exactly one TensorIndex should find it.
    ModelWeightsPtrs<SfpStream> weights(config, pool);
    ModelWeightsPtrs<SfpStream>::ForEachTensor(
        {&weights}, ForEachType::kInitNoToc,
        [&tensor_indexes](const char* name, hwy::Span<MatPtr*> tensors) {
          int num_found = 0;
          const MatPtr& tensor = *tensors[0];
          for (const auto& tensor_index : tensor_indexes) {
            // Skip the type marker prefix, but we want the layer index suffix.
            std::string name_to_find(name + 1, strlen(name) - 1);
            const TensorInfo* info = tensor_index.FindName(name_to_find);
            if (info != nullptr) {
              // Test that the MatPtr can be constructed from the TensorInfo,
              // and that the dimensions match.
              MatPtrT<SfpStream> mat_ptr(tensor.Name(), tensor_index);
              EXPECT_EQ(tensor.Name(), mat_ptr.Name()) << "on tensor " << name;
              EXPECT_EQ(tensor.Rows(), mat_ptr.Rows()) << "on tensor " << name;
              EXPECT_EQ(tensor.Cols(), mat_ptr.Cols()) << "on tensor " << name;
              ++num_found;
            }
          }
          EXPECT_EQ(num_found, 1) << " for tensor " << name;
        });
  }
}

}  // namespace
}  // namespace gcpp
