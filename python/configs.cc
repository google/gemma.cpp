// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gemma/configs.h"

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "compression/types.h"
#include "gemma/tensor_info.h"

using gcpp::ActivationType;
using gcpp::InternalLayerConfig;
using gcpp::InternalModelConfig;
using gcpp::LayerAttentionType;
using gcpp::LayerConfig;
using gcpp::Model;
using gcpp::ModelConfig;
using gcpp::PostNormType;
using gcpp::PostQKType;
using gcpp::PromptWrapping;
using gcpp::QueryScaleType;
using gcpp::ResidualType;
using gcpp::TensorInfo;
using gcpp::TensorInfoRegistry;
using gcpp::Type;
using gcpp::VitConfig;

namespace pybind11 {

PYBIND11_MODULE(configs, py_module) {
  enum_<PromptWrapping>(py_module, "PromptWrapping")
      .value("GEMMA_IT", PromptWrapping::GEMMA_IT)
      .value("GEMMA_PT", PromptWrapping::GEMMA_PT)
      .value("GEMMA_VLM", PromptWrapping::GEMMA_VLM)
      .value("PALIGEMMA", PromptWrapping::PALIGEMMA);

  enum_<Type>(py_module, "Type")
      .value("kUnknown", Type::kUnknown)
      .value("kF32", Type::kF32)
      .value("kBF16", Type::kBF16)
      .value("kSFP", Type::kSFP)
      .value("kNUQ", Type::kNUQ)
      .value("kF64", Type::kF64)
      .value("kU32", Type::kU32)
      .value("kU64", Type::kU64)
      .value("kI8", Type::kI8);

  enum_<LayerAttentionType>(py_module, "LayerAttentionType")
      .value("kGemma", LayerAttentionType::kGemma)
      .value("kVit", LayerAttentionType::kVit)
      .value("kDeepSeekMLA", LayerAttentionType::kDeepSeekMLA);

  enum_<AttentionVariant>(py_module, "AttentionVariant")
      .value("kDense", AttentionVariant::kDense)
      .value("kCSA", AttentionVariant::kCSA)
      .value("kHCA", AttentionVariant::kHCA);

  enum_<PostNormType>(py_module, "PostNormType")
      .value("NoPostNorm", PostNormType::None)
      .value("Scale", PostNormType::Scale)
  .value("Scale", PostNormType::Scale);

  enum_<PostQKType>(py_module, "PostQKType")
      .value("Rope", PostQKType::Rope)
  .value("HalfRope", PostQKType::HalfRope);

  enum_<ActivationType>(py_module, "ActivationType")
      .value("Gelu", ActivationType::Gelu)
  .value("Silu", ActivationType::Silu);

  enum_<RouterScoreFunc>(py_module, "RouterScoreFunc")
      .value("kSigmoidGatingCompat", RouterScoreFunc::kSigmoidGatingCompat)
      .value("kSoftmax", RouterScoreFunc::kSoftmax)
      .value("kSigmoid", RouterScoreFunc::kSigmoid)
      .value("kSqrtSoftplus", RouterScoreFunc::kSqrtSoftplus);

  enum_<QueryScaleType>(py_module, "QueryScaleType")
      .value("SqrtKeySize", QueryScaleType::SqrtKeySize)
  .value("SqrtModelDimDivNumHeads",
         QueryScaleType::SqrtModelDimDivNumHeads);

  enum_<ResidualType>(py_module, "ResidualType")
  .value("Add", ResidualType::Add);

  enum_<Model>(py_module, "Model")
      .value("UNKNOWN", Model::UNKNOWN)
      .value("CUSTOM", Model::CUSTOM)
      .value("GEMMA2_9B", Model::GEMMA2_9B)
      .value("GEMMA2_27B", Model::GEMMA2_27B)
      .value("GEMMA2_2B", Model::GEMMA2_2B)
      .value("PALIGEMMA2_3B_224", Model::PALIGEMMA2_3B_224)
      .value("PALIGEMMA2_10B_224", Model::PALIGEMMA2_10B_224)
      .value("PALIGEMMA2_3B_448", Model::PALIGEMMA2_3B_448)
      .value("PALIGEMMA2_10B_448", Model::PALIGEMMA2_10B_448)
      .value("GEMMA3_1B", Model::GEMMA3_1B)
      .value("GEMMA3_4B", Model::GEMMA3_4B)
      .value("GEMMA3_12B", Model::GEMMA3_12B)
      .value("GEMMA3_27B", Model::GEMMA3_27B)
      .value("GEMMA3_270M", Model::GEMMA3_270M)
      .value("GEMMA3_4B_LM", Model::GEMMA3_4B_LM)
      .value("GEMMA3_12B_LM", Model::GEMMA3_12B_LM)
      .value("GEMMA3_27B_LM", Model::GEMMA3_27B_LM)
      .value("DEEPSEEK4_FLASH", Model::DEEPSEEK4_FLASH)
      // Insert new models above this line.
  .value("PALIGEMMA_448", Model::PALIGEMMA_448);

  class_<TensorInfo>(py_module, "TensorInfo")
      .def(init())
      .def_readwrite("name", &TensorInfo::base_name)
      .def_readwrite("source_names", &TensorInfo::source_names)
      .def_readwrite("preshape", &TensorInfo::preshape)
      .def_readwrite("axes", &TensorInfo::axes)
      .def_readwrite("shape", &TensorInfo::shape)
      .def_readwrite("concat_names", &TensorInfo::concat_names)
      .def_readwrite("concat_axis", &TensorInfo::concat_axis)
      .def_readwrite("min_size", &TensorInfo::min_size)
      .def_readwrite("scaled_softplus", &TensorInfo::scaled_softplus)
      .def_readwrite("cols_take_extra_dims", &TensorInfo::cols_take_extra_dims);

  class_<TensorInfoRegistry>(py_module, "TensorInfoRegistry")
      .def(init<const ModelConfig&>())
      .def("tensor_info_from_source_path",
           &TensorInfoRegistry::TensorInfoFromSourcePath, arg("path"),
           arg("layer_idx"))
      .def("tensor_info_from_name", &TensorInfoRegistry::TensorInfoFromName,
           arg("name"));

  class_<InternalLayerConfig>(py_module, "InternalLayerConfig")
      .def(init<>());

  class_<LayerConfig>(py_module, "LayerConfig")
      .def(init())
      .def_readwrite("model_dim", &LayerConfig::model_dim)
      .def_readwrite("ff_hidden_dim", &LayerConfig::ff_hidden_dim)
      .def_readwrite("heads", &LayerConfig::heads)
      .def_readwrite("kv_heads", &LayerConfig::kv_heads)
      .def_readwrite("qkv_dim", &LayerConfig::qkv_dim)
      .def_readwrite("ff_biases", &LayerConfig::ff_biases)
      .def_readwrite("optimized_gating", &LayerConfig::optimized_gating)
      .def_readwrite("post_norm", &LayerConfig::post_norm)
      .def_readwrite("type", &LayerConfig::type)
      .def_readwrite("activation", &LayerConfig::activation)
      .def_readwrite("post_qk", &LayerConfig::post_qk)
      .def_readwrite("use_qk_norm", &LayerConfig::use_qk_norm)
      .def_readwrite("kv_share_layer_idx", &LayerConfig::kv_share_layer_idx)
      .def_readwrite("norm_v", &LayerConfig::norm_v)
      .def_readwrite("num_experts", &LayerConfig::num_experts)
      .def_readwrite("num_experts_per_datapoint",
                     &LayerConfig::num_experts_per_datapoint)
      .def_readwrite("kv_lora_rank", &LayerConfig::kv_lora_rank)
      .def_readwrite("q_lora_rank", &LayerConfig::q_lora_rank)
      .def_readwrite("rope_head_dim", &LayerConfig::rope_head_dim)
      .def_readwrite("v_head_dim", &LayerConfig::v_head_dim)
      .def_readwrite("attention_variant", &LayerConfig::attention_variant)
      .def_readwrite("kv_compression_rate", &LayerConfig::kv_compression_rate)
      .def_readwrite("indexer_heads", &LayerConfig::indexer_heads)
      .def_readwrite("indexer_head_dim", &LayerConfig::indexer_head_dim)
      .def_readwrite("indexer_top_k", &LayerConfig::indexer_top_k)
      .def_readwrite("num_shared_experts", &LayerConfig::num_shared_experts)
      .def_readwrite("sigmoid_gating", &LayerConfig::sigmoid_gating)
      .def_readwrite("use_routing_bias", &LayerConfig::use_routing_bias)
      .def_readwrite("o_lora_rank", &LayerConfig::o_lora_rank)
      .def_readwrite("o_groups", &LayerConfig::o_groups)
      .def_readwrite("swiglu_limit", &LayerConfig::swiglu_limit)
      .def_readwrite("route_scale", &LayerConfig::route_scale)
      .def_readwrite("router_score", &LayerConfig::router_score)
      .def_readwrite("hash_routing", &LayerConfig::hash_routing)
      .def_readwrite("internal", &LayerConfig::internal);

  class_<VitConfig>(py_module, "VitConfig")
      .def(init())
      .def_readwrite("model_dim", &VitConfig::model_dim)
      .def_readwrite("seq_len", &VitConfig::seq_len)
      .def_readwrite("num_scales", &VitConfig::num_scales)
      .def_readwrite("patch_width", &VitConfig::patch_width)
      .def_readwrite("image_size", &VitConfig::image_size)
      .def_readwrite("layer_configs", &VitConfig::layer_configs);

  class_<InternalModelConfig>(py_module, "InternalModelConfig").def(init<>());

  class_<ModelConfig>(py_module, "ModelConfig")
      .def(init<>())
      .def(init<Model, Type, PromptWrapping>())
      .def(init<const char*>())
      .def_readwrite("model_family_version", &ModelConfig::model_family_version)
      .def_readwrite("display_name", &ModelConfig::display_name)
      .def_readwrite("model", &ModelConfig::model)
      .def_readwrite("wrapping", &ModelConfig::wrapping)
      .def_readwrite("weight", &ModelConfig::weight)
      .def_readwrite("num_layers", &ModelConfig::num_layers)
      .def_readwrite("model_dim", &ModelConfig::model_dim)
      .def_readwrite("vocab_size", &ModelConfig::vocab_size)
      .def_readwrite("max_seq_len", &ModelConfig::max_seq_len)
      // Skip `unused_num_tensor_scales`.
      .def_readwrite("att_cap", &ModelConfig::att_cap)
      .def_readwrite("final_cap", &ModelConfig::final_cap)
      .def_readwrite("absolute_pe", &ModelConfig::absolute_pe)
      .def_readwrite("query_scale", &ModelConfig::query_scale)
      .def_readwrite("layer_configs", &ModelConfig::layer_configs)
      .def_readwrite("attention_window_sizes",
                     &ModelConfig::attention_window_sizes)
      .def_readwrite("norm_num_groups", &ModelConfig::norm_num_groups)
      .def_readwrite("vit_config", &ModelConfig::vit_config)
      .def_readwrite("pool_dim", &ModelConfig::pool_dim)
      .def_readwrite("eos_id", &ModelConfig::eos_id)
      .def_readwrite("secondary_eos_id", &ModelConfig::secondary_eos_id)
      .def_readwrite("scale_base_names", &ModelConfig::scale_base_names)
      .def_readwrite("internal", &ModelConfig::internal)
      .def_readwrite("use_global_timescale",
                     &ModelConfig::use_global_timescale)
      .def_readwrite("partial_rotary_factor",
                     &ModelConfig::partial_rotary_factor)
      .def_readwrite("hc_mult", &ModelConfig::hc_mult)
      .def_readwrite("rope_theta", &ModelConfig::rope_theta)
      .def_readwrite("compress_rope_theta", &ModelConfig::compress_rope_theta)
      .def_readwrite("yarn_orig_seq_len", &ModelConfig::yarn_orig_seq_len)
      .def_readwrite("yarn_factor", &ModelConfig::yarn_factor)
      .def_readwrite("yarn_beta_fast", &ModelConfig::yarn_beta_fast)
      .def_readwrite("yarn_beta_slow", &ModelConfig::yarn_beta_slow)
      .def_readwrite("hc_sinkhorn_iters", &ModelConfig::hc_sinkhorn_iters)
      .def_readwrite("hc_eps", &ModelConfig::hc_eps)
      .def_readwrite("num_mtp_layers", &ModelConfig::num_mtp_layers)

      .def("add_layer_config", &ModelConfig::AddLayerConfig,
           arg("layer_config"))
      .def("test_equal", &ModelConfig::TestEqual, arg("other"), arg("print"))
      .def("overwrite_with_canonical", &ModelConfig::OverwriteWithCanonical)
      .def("specifier", &ModelConfig::Specifier);

  // Returns the sub-config for the ViT model of the PaliGemma model.
  py_module.def("vit_config", &gcpp::GetVitConfig, arg("config"));

  py_module.def("is_paligemma", &gcpp::IsPaliGemma, arg("model"));
}

}  // namespace pybind11
