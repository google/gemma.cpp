#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_TENSOR_INFO_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_TENSOR_INFO_H_

#include <stddef.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "compression/types.h"  // Type
#include "gemma/configs.h"
#include "util/basics.h"  // Extents2D

namespace gcpp {

// Tensor metadata. This is far more than required to construct the `MatPtr` in
// `LayerWeightsPtrs/WeightsPtrs`; they only use `.shape` via `ExtentsFromInfo`.
// This is also bound to Python and filled by the exporter.
struct TensorInfo {
  // The base name of the tensor without a layer suffix.
  std::string base_name;
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
  // The highest permissible compression for this tensor. The default is
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
// not-present tensors such as ViT in a text-only model. Safely handles nullptr
// returned from `TensorInfoRegistry::Find`, hence not a member function.
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

static inline std::string LayerSuffix(size_t layer_idx) {
  return std::string("_") + std::to_string(layer_idx);
}

// Returns tensor base name without any layer suffix.
static inline std::string StripLayerSuffix(const std::string& name) {
  return name.substr(0, name.rfind('_'));
}

// Holds all `TensorInfo` for a model and retrieves them by (unique) name.
class TensorInfoRegistry {
 public:
  explicit TensorInfoRegistry(const ModelConfig& config);
  ~TensorInfoRegistry() = default;

  // Returns nullptr if not found, otherwise the `TensorInfo` for the given
  // `name`, which either lacks a suffix, or is per-layer and ends with
  // `LayerSuffix(layer_idx)`. Used in `WeightsPtrs/LayerWeightsPtrs`.
  const TensorInfo* Find(const std::string& name) const {
    auto it = idx_from_name_.find(name);
    if (it == idx_from_name_.end()) return nullptr;
    return &tensors_[it->second];
  }

  // Returns a copy of the `TensorInfo` whose name matches the given name, or a
  // default-constructed `TensorInfo` if not found. Destroying
  // `TensorInfoRegistry` afterward will not invalidate the returned value.
  TensorInfo TensorInfoFromName(const std::string& name) const {
    const TensorInfo* info = Find(name);
    if (info == nullptr) return TensorInfo();
    return *info;
  }

  // Returns a copy of the `TensorInfo` whose source_name matches the end of the
  // given path, and whose name ends with the given layer_idx, otherwise a
  // default-constructed `TensorInfo`. Destroying `TensorInfoRegistry`
  // afterward will not invalidate the returned value.
  TensorInfo TensorInfoFromSourcePath(const std::string& path,
                                      int layer_idx) const;

 private:
  // `suffix` is empty (only) for per-model tensors, otherwise `LayerSuffix`.
  void Add(const std::string& suffix, const TensorInfo& info);
  void AddModelTensors(const ModelConfig& config);
  void AddLayerTensors(const ModelConfig& config,
                       const LayerConfig& layer_config, size_t layer_idx);
  void AddGriffinLayerTensors(const LayerConfig& layer_config,
                              size_t layer_idx);

  void AddImageLayerTensors(const ModelConfig& config,
                            const LayerConfig& layer_config,
                            size_t img_layer_idx);

  std::vector<TensorInfo> tensors_;
  // Includes entries for base name *and* the suffixed name for each layer.
  std::unordered_map<std::string, size_t> idx_from_name_;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_TENSOR_INFO_H_
