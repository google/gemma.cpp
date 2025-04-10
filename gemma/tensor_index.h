#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_TENSOR_INDEX_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_TENSOR_INDEX_H_

#include <stddef.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "compression/shared.h"
#include "gemma/configs.h"

namespace gcpp {

// Universal tensor information. Holds enough information to construct a
// tensor in LayerWeightsPtrs/ModelWeightsPtrs, as well as to export the
// tensor from the python model with necessary transpose/reshape info.
struct TensorInfo {
  // The name of the tensor in the sbs file
  std::string name;
  // Strings to match to the end of the name of the tensor in the python model.
  std::vector<std::string> source_names;
  // Initial reshape shape. Use only as a last resort when input may have
  // dimensions combined that need to be split before the transpose, as it
  // defeats the post-transpose shape checking. Normally empty.
  std::vector<size_t> preshape;
  // Transpose axes arg. If the input tensor has more dimensions than axes,
  // then leading dimensions are collapsed until the number of axes matches.
  std::vector<size_t> axes;
  // Expected final shape of the tensor after reshape/transpose.
  // Note that this is the shape of the tensor during export,
  // not the shape of the tensor in the sbs file, as the sbs file
  // is restricted to 2D tensors. With few exceptions, the sbs file
  // tensor rows gather all the excess dimensions. See cols_take_extra_dims.
  std::vector<size_t> shape;
  // List of names to concatenate with, used only if multiple tensors are
  // concatenated into one. The first tensor in the concatenation should have
  // concat names thus: The first name is the name of the result, and the
  // tensors with the remaining names are concatenated after this.
  // The remaining tensors to be concatenated should have just a single
  // empty string in concat_names to indicate that they have been consumed.
  std::vector<std::string> concat_names;
  // Axis at which to concatenate.
  size_t concat_axis = 0;
  // The minimum compression weight type for this tensor. The default is
  // kNUQ, which provides maximum compression. Other values such as kBF16
  // or kF32 can be used to limit the compression to a specific type.
  Type min_size = Type::kNUQ;
  // Whether to apply scaled softplus to the data.
  bool scaled_softplus = false;
  // Whether the columns or the rows take any extra dimensions.
  // If false, then [10, 20, 30] -> [10*20, 30] and [30] -> [1, 30].
  // If true, then [10, 20, 30] -> [10, 20*30] and [30] -> [1, 30].
  bool cols_take_extra_dims = false;
};

// Collapses/expands the tensor dims into 2D extents, which may be 0, 0 for
// not-present tensors such as ViT in a text-only model.
static inline Extents2D ExtentsFromInfo(const TensorInfo* tensor) {
  if (tensor == nullptr) return Extents2D(0, 0);

  size_t cols = tensor->shape.back();
  size_t rows = 1;
  if (tensor->cols_take_extra_dims) {
    rows = tensor->shape[0];
    for (size_t i = 1; i < tensor->shape.size() - 1; ++i) {
      cols *= tensor->shape[i];
    }
  } else {  // rows take extra dims
    for (size_t i = 0; i < tensor->shape.size() - 1; ++i) {
      rows *= tensor->shape[i];
    }
  }
  // Sometimes only one of rows or cols is zero; set both for consistency.
  if (rows == 0 || cols == 0) rows = cols = 0;
  return Extents2D(rows, cols);
}

// Universal index of tensor information, which can be built for a specific
// layer_idx.
class TensorIndex {
 public:
  // Builds a list of TensorInfo for the given layer_idx.
  // If reshape_att is true, the attn_vec_einsum tensor is reshaped.
  TensorIndex(const ModelConfig& config, int llm_layer_idx, int img_layer_idx,
              bool reshape_att);
  ~TensorIndex() = default;

  // Returns the TensorInfo whose source_name matches the end of the given path,
  // or an empty TensorInfo if not found.
  // NOTE: that the returned TensorInfo is a copy, so that the source
  // TensorIndex can be destroyed without affecting the returned TensorInfo.
  TensorInfo TensorInfoFromSourcePath(const std::string& path) const;

  // Returns the TensorInfo whose name matches the given name,
  // or an empty TensorInfo if not found.
  // NOTE: that the returned TensorInfo is a copy, so that the source
  // TensorIndex can be destroyed without affecting the returned TensorInfo.
  TensorInfo TensorInfoFromName(const std::string& name) const {
    const TensorInfo* info = FindName(name);
    if (info == nullptr) return TensorInfo();
    return *info;
  }

  // Returns the TensorInfo for the given tensor name, for concise construction
  // of ModelWeightsPtrs/LayerWeightsPtrs.
  const TensorInfo* FindName(const std::string& name) const;

 private:
  // Config that was used to build the tensor index.
  const ModelConfig& config_;
  // Layer that this tensor index is for - either LLM or image.
  int llm_layer_idx_;
  int img_layer_idx_;
  // List of tensor information for this layer.
  std::vector<TensorInfo> tensors_;
  // Map from tensor name to index in tensors_.
  std::unordered_map<std::string, size_t> name_map_;
};

static inline TensorIndex TensorIndexLLM(const ModelConfig& config,
                                         size_t llm_layer_idx) {
  return TensorIndex(config, static_cast<int>(llm_layer_idx), -1, false);
}

static inline TensorIndex TensorIndexImg(const ModelConfig& config,
                                         size_t img_layer_idx) {
  return TensorIndex(config, -1, static_cast<int>(img_layer_idx), false);
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_TENSOR_INDEX_H_
