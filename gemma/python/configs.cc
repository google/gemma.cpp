#include "gemma/configs.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "compression/shared.h"
#include "gemma/tensor_index.h"
#include "pybind11/cast.h"

using gcpp::ActivationType;
using gcpp::LayerAttentionType;
using gcpp::LayerConfig;
using gcpp::Model;
using gcpp::ModelConfig;
using gcpp::ModelTraining;
using gcpp::PostNormType;
using gcpp::PostQKType;
using gcpp::QueryScaleType;
using gcpp::ResidualType;
using gcpp::TensorIndex;
using gcpp::TensorInfo;
using gcpp::Type;

namespace pybind11 {

PYBIND11_MODULE(configs, py_module) {
  enum_<ModelTraining>(py_module, "ModelTraining")
      .value("GEMMA_IT", ModelTraining::GEMMA_IT)
      .value("GEMMA_PT", ModelTraining::GEMMA_PT)
      .value("PALIGEMMA", ModelTraining::PALIGEMMA);

  enum_<Type>(py_module, "Type")
      .value("kUnknown", Type::kUnknown)
      .value("kF32", Type::kF32)
      .value("kBF16", Type::kBF16)
      .value("kSFP", Type::kSFP)
      .value("kNUQ", Type::kNUQ)
      .value("kF64", Type::kF64)
      .value("kC64", Type::kC64)
      .value("kU128", Type::kU128);

  enum_<LayerAttentionType>(py_module, "LayerAttentionType")
      .value("kGemma", LayerAttentionType::kGemma)
      .value("kGriffinRecurrentBlock",
             LayerAttentionType::kGriffinRecurrentBlock)
      .value("kVit", LayerAttentionType::kVit);

  enum_<PostNormType>(py_module, "PostNormType")
      .value("NoPostNorm", PostNormType::None)
  .value("Scale", PostNormType::Scale);

  enum_<PostQKType>(py_module, "PostQKType")
      .value("Rope", PostQKType::Rope)
  .value("HalfRope", PostQKType::HalfRope);

  enum_<ActivationType>(py_module, "ActivationType")
  .value("Gelu", ActivationType::Gelu);

  enum_<QueryScaleType>(py_module, "QueryScaleType")
      .value("SqrtKeySize", QueryScaleType::SqrtKeySize)
  .value("SqrtModelDimDivNumHeads",
         QueryScaleType::SqrtModelDimDivNumHeads);

  enum_<ResidualType>(py_module, "ResidualType")
  .value("Add", ResidualType::Add);

  enum_<Model>(py_module, "Model")
      .value("UNKNOWN", Model::UNKNOWN)
      .value("GEMMA_2B", Model::GEMMA_2B)
      .value("GEMMA_7B", Model::GEMMA_7B)
      .value("GEMMA2_9B", Model::GEMMA2_9B)
      .value("GEMMA2_27B", Model::GEMMA2_27B)
      .value("GRIFFIN_2B", Model::GRIFFIN_2B)
      .value("GEMMA_TINY", Model::GEMMA_TINY)
      .value("GEMMA2_2B", Model::GEMMA2_2B)
  .value("PALIGEMMA_224", Model::PALIGEMMA_224);

  class_<TensorInfo>(py_module, "TensorInfo")
      .def(init())
      .def_readwrite("name", &TensorInfo::name)
      .def_readwrite("source_names", &TensorInfo::source_names)
      .def_readwrite("preshape", &TensorInfo::preshape)
      .def_readwrite("axes", &TensorInfo::axes)
      .def_readwrite("shape", &TensorInfo::shape)
      .def_readwrite("concat_names", &TensorInfo::concat_names)
      .def_readwrite("concat_axis", &TensorInfo::concat_axis)
      .def_readwrite("min_size", &TensorInfo::min_size)
      .def_readwrite("scaled_softplus", &TensorInfo::scaled_softplus)
      .def_readwrite("cols_take_extra_dims", &TensorInfo::cols_take_extra_dims);

  class_<TensorIndex>(py_module, "TensorIndex")
      .def(init<const ModelConfig&, int, int, bool>())
      .def("get_tensor_info", &TensorIndex::GetTensorInfo, arg("path"));

  class_<LayerConfig>(py_module, "LayerConfig")
      .def(init())
      .def_readwrite("model_dim", &LayerConfig::model_dim)
      .def_readwrite("griffin_dim", &LayerConfig::griffin_dim)
      .def_readwrite("ff_hidden_dim", &LayerConfig::ff_hidden_dim)
      .def_readwrite("heads", &LayerConfig::heads)
      .def_readwrite("kv_heads", &LayerConfig::kv_heads)
      .def_readwrite("qkv_dim", &LayerConfig::qkv_dim)
      .def_readwrite("conv1d_width", &LayerConfig::conv1d_width)
      .def_readwrite("ff_biases", &LayerConfig::ff_biases)
      .def_readwrite("softmax_attn_output_biases",
                     &LayerConfig::softmax_attn_output_biases)
      .def_readwrite("optimized_gating", &LayerConfig::optimized_gating)
      .def_readwrite("post_norm", &LayerConfig::post_norm)
      .def_readwrite("type", &LayerConfig::type)
      .def_readwrite("activation", &LayerConfig::activation)
  .def_readwrite("post_qk", &LayerConfig::post_qk);

  class_<ModelConfig>(py_module, "ModelConfig")
      .def(init())
      .def_readwrite("model_name", &ModelConfig::model_name)
      .def_readwrite("model", &ModelConfig::model)
      .def_readwrite("training", &ModelConfig::training)
      .def_readwrite("weight", &ModelConfig::weight)
      .def_readwrite("num_layers", &ModelConfig::num_layers)
      .def_readwrite("model_dim", &ModelConfig::model_dim)
      .def_readwrite("vit_model_dim", &ModelConfig::vit_model_dim)
      .def_readwrite("vocab_size", &ModelConfig::vocab_size)
      .def_readwrite("seq_len", &ModelConfig::seq_len)
      .def_readwrite("vit_seq_len", &ModelConfig::vit_seq_len)
      .def_readwrite("num_tensor_scales", &ModelConfig::num_tensor_scales)
      .def_readwrite("num_vit_scales", &ModelConfig::num_vit_scales)
      .def_readwrite("att_cap", &ModelConfig::att_cap)
      .def_readwrite("final_cap", &ModelConfig::final_cap)
      .def_readwrite("absolute_pe", &ModelConfig::absolute_pe)
      .def_readwrite("use_local_attention", &ModelConfig::use_local_attention)
      .def_readwrite("query_scale", &ModelConfig::query_scale)
      .def_readwrite("layer_configs", &ModelConfig::layer_configs)
      .def_readwrite("attention_window_sizes",
                     &ModelConfig::attention_window_sizes)
      .def_readwrite("vit_layer_configs", &ModelConfig::vit_layer_configs)
      .def_readwrite("scale_names", &ModelConfig::scale_names)
      .def_readwrite("norm_num_groups", &ModelConfig::norm_num_groups)
      .def_readwrite("model_family_version", &ModelConfig::model_family_version)
      .def_readwrite("patch_width", &ModelConfig::patch_width)
      .def_readwrite("image_size", &ModelConfig::image_size)
      .def("add_layer_config", &ModelConfig::AddLayerConfig,
           arg("layer_config"))
      .def("test_equal", &ModelConfig::TestEqual, arg("other"), arg("partial"),
           arg("debug"));

  // Returns the config for the given model.
  py_module.def("config_from_model", &gcpp::ConfigFromModel, arg("model"));

  // Returns the model for the given config, if it matches any standard model.
  py_module.def("model_from_config", &gcpp::ModelFromConfig, arg("config"));

  // Returns the sub-config for the ViT model of the PaliGemma model.
  py_module.def("vit_config", &gcpp::VitConfig, arg("config"));
}

}  // namespace pybind11
