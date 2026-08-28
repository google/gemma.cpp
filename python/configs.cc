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

namespace pybind11 {

PYBIND11_MODULE(configs, py_module) {
  enum_<gcpp::PromptWrapping>(py_module, "PromptWrapping")
      .value("GEMMA_IT", gcpp::PromptWrapping::GEMMA_IT)
      .value("GEMMA_PT", gcpp::PromptWrapping::GEMMA_PT)
      .value("GEMMA_VLM", gcpp::PromptWrapping::GEMMA_VLM)
      .value("PALIGEMMA", gcpp::PromptWrapping::PALIGEMMA);

  enum_<gcpp::Type>(py_module, "Type")
      .value("kUnknown", gcpp::Type::kUnknown)
      .value("kF32", gcpp::Type::kF32)
      .value("kBF16", gcpp::Type::kBF16)
      .value("kSFP", gcpp::Type::kSFP)
      .value("kNUQ", gcpp::Type::kNUQ)
      .value("kF64", gcpp::Type::kF64)
      .value("kU32", gcpp::Type::kU32)
      .value("kU64", gcpp::Type::kU64)
      .value("kI8", gcpp::Type::kI8)
      .value("kQ4_0", gcpp::Type::kQ4_0)
      .value("kMXFP4", gcpp::Type::kMXFP4);

  enum_<gcpp::LayerAttentionType>(py_module, "LayerAttentionType")
      .value("kGemma", gcpp::LayerAttentionType::kGemma)
      .value("kVit", gcpp::LayerAttentionType::kVit)
      .value("kVitGemma4", gcpp::LayerAttentionType::kVitGemma4)
      .value("kDeepSeekMLA", gcpp::LayerAttentionType::kDeepSeekMLA);

  enum_<gcpp::AttentionVariant>(py_module, "AttentionVariant")
      .value("kDense", gcpp::AttentionVariant::kDense)
      .value("kCSA", gcpp::AttentionVariant::kCSA)
      .value("kHCA", gcpp::AttentionVariant::kHCA);

  enum_<gcpp::PostNormType>(py_module, "PostNormType")
      .value("Scale", gcpp::PostNormType::Scale)
      .value("NoPostNorm", gcpp::PostNormType::None);

  enum_<gcpp::PostQKType>(py_module, "PostQKType")
      .value("Rope", gcpp::PostQKType::Rope)
      .value("HalfRope", gcpp::PostQKType::HalfRope)
      .value("NormLocalRope", gcpp::PostQKType::NormLocalRope);

  enum_<gcpp::ActivationType>(py_module, "ActivationType")
      .value("Gelu", gcpp::ActivationType::Gelu)
      .value("Silu", gcpp::ActivationType::Silu);

  enum_<gcpp::RouterScoreFunc>(py_module, "RouterScoreFunc")
      .value("kSigmoidGatingCompat",
             gcpp::RouterScoreFunc::kSigmoidGatingCompat)
      .value("kSoftmax", gcpp::RouterScoreFunc::kSoftmax)
      .value("kSigmoid", gcpp::RouterScoreFunc::kSigmoid)
      .value("kSqrtSoftplus", gcpp::RouterScoreFunc::kSqrtSoftplus);

  enum_<gcpp::QueryScaleType>(py_module, "QueryScaleType")
      .value("SqrtModelDimDivNumHeads",
             gcpp::QueryScaleType::SqrtModelDimDivNumHeads)
      .value("SqrtKeySize", gcpp::QueryScaleType::SqrtKeySize)
      .value("One", gcpp::QueryScaleType::One);

  enum_<gcpp::ResidualType>(py_module, "ResidualType")
  .value("Add", gcpp::ResidualType::Add);

  enum_<gcpp::TokenizerKind>(py_module, "TokenizerKind")
      .value("kSentencePiece", gcpp::TokenizerKind::kSentencePiece)
      .value("kHfBpe", gcpp::TokenizerKind::kHfBpe);

  enum_<gcpp::Model>(py_module, "Model")
      .value("UNKNOWN", gcpp::Model::UNKNOWN)
      .value("CUSTOM", gcpp::Model::CUSTOM)
      .value("GEMMA2_9B", gcpp::Model::GEMMA2_9B)
      .value("GEMMA2_27B", gcpp::Model::GEMMA2_27B)
      .value("GEMMA2_2B", gcpp::Model::GEMMA2_2B)
      .value("PALIGEMMA2_3B_224", gcpp::Model::PALIGEMMA2_3B_224)
      .value("PALIGEMMA2_10B_224", gcpp::Model::PALIGEMMA2_10B_224)
      .value("PALIGEMMA2_3B_448", gcpp::Model::PALIGEMMA2_3B_448)
      .value("PALIGEMMA2_10B_448", gcpp::Model::PALIGEMMA2_10B_448)
      .value("GEMMA3_1B", gcpp::Model::GEMMA3_1B)
      .value("GEMMA3_4B", gcpp::Model::GEMMA3_4B)
      .value("GEMMA3_12B", gcpp::Model::GEMMA3_12B)
      .value("GEMMA3_27B", gcpp::Model::GEMMA3_27B)
      .value("GEMMA3_270M", gcpp::Model::GEMMA3_270M)
      .value("GEMMA3_4B_LM", gcpp::Model::GEMMA3_4B_LM)
      .value("GEMMA3_12B_LM", gcpp::Model::GEMMA3_12B_LM)
      .value("GEMMA3_27B_LM", gcpp::Model::GEMMA3_27B_LM)
      .value("GEMMA4_26B_MOE", gcpp::Model::GEMMA4_26B_MOE)
      .value("GEMMA4_2B", gcpp::Model::GEMMA4_2B)
      .value("DEEPSEEK4_FLASH", gcpp::Model::DEEPSEEK4_FLASH)
      .value("T5GEMMA_S_S", gcpp::Model::T5GEMMA_S_S)
      .value("QWEN3_600M", gcpp::Model::QWEN3_600M)
      .value("QWEN3_2B", gcpp::Model::QWEN3_2B)
      .value("QWEN3_4B", gcpp::Model::QWEN3_4B)
      .value("GEMMA4_2B_LM", gcpp::Model::GEMMA4_2B_LM)
      .value("GEMMA4_E4B", gcpp::Model::GEMMA4_E4B)
      .value("GEMMA4_E4B_LM", gcpp::Model::GEMMA4_E4B_LM)
      // Insert new models above this line.
      ;  // NOLINT

  class_<gcpp::TensorInfo>(py_module, "TensorInfo")
      .def(init())
      .def_readwrite("name", &gcpp::TensorInfo::base_name)
      .def_readwrite("source_names", &gcpp::TensorInfo::source_names)
      .def_readwrite("preshape", &gcpp::TensorInfo::preshape)
      .def_readwrite("axes", &gcpp::TensorInfo::axes)
      .def_readwrite("shape", &gcpp::TensorInfo::shape)
      .def_readwrite("concat_names", &gcpp::TensorInfo::concat_names)
      .def_readwrite("concat_axis", &gcpp::TensorInfo::concat_axis)
      .def_readwrite("min_size", &gcpp::TensorInfo::min_size)
      .def_readwrite("scaled_softplus", &gcpp::TensorInfo::scaled_softplus)
      .def_readwrite("cols_take_extra_dims",
                     &gcpp::TensorInfo::cols_take_extra_dims);

  class_<gcpp::TensorInfoRegistry>(py_module, "TensorInfoRegistry")
      .def(init<const gcpp::ModelConfig&>())
      .def("tensor_info_from_source_path",
           &gcpp::TensorInfoRegistry::TensorInfoFromSourcePath, arg("path"),
           arg("layer_idx"))
      .def("tensor_info_from_name",
           &gcpp::TensorInfoRegistry::TensorInfoFromName, arg("name"));

  class_<gcpp::InternalLayerConfig>(py_module, "InternalLayerConfig")
      .def(init<>());

  class_<gcpp::LayerConfig>(py_module, "LayerConfig")
      .def(init())
      .def_readwrite("model_dim", &gcpp::LayerConfig::model_dim)
      .def_readwrite("ff_hidden_dim", &gcpp::LayerConfig::ff_hidden_dim)
      .def_readwrite("heads", &gcpp::LayerConfig::heads)
      .def_readwrite("kv_heads", &gcpp::LayerConfig::kv_heads)
      .def_readwrite("qkv_dim", &gcpp::LayerConfig::qkv_dim)
      .def_readwrite("ff_biases", &gcpp::LayerConfig::ff_biases)
      .def_readwrite("optimized_gating", &gcpp::LayerConfig::optimized_gating)
      .def_readwrite("post_norm", &gcpp::LayerConfig::post_norm)
      .def_readwrite("type", &gcpp::LayerConfig::type)
      .def_readwrite("activation", &gcpp::LayerConfig::activation)
      .def_readwrite("post_qk", &gcpp::LayerConfig::post_qk)
      .def_readwrite("use_qk_norm", &gcpp::LayerConfig::use_qk_norm)
      .def_readwrite("kv_share_layer_idx",
                     &gcpp::LayerConfig::kv_share_layer_idx)
      .def_readwrite("norm_v", &gcpp::LayerConfig::norm_v)
      .def_readwrite("num_experts", &gcpp::LayerConfig::num_experts)
      .def_readwrite("num_experts_per_datapoint",
                     &gcpp::LayerConfig::num_experts_per_datapoint)
      .def_readwrite("kv_lora_rank", &gcpp::LayerConfig::kv_lora_rank)
      .def_readwrite("q_lora_rank", &gcpp::LayerConfig::q_lora_rank)
      .def_readwrite("rope_head_dim", &gcpp::LayerConfig::rope_head_dim)
      .def_readwrite("v_head_dim", &gcpp::LayerConfig::v_head_dim)
      .def_readwrite("attention_variant", &gcpp::LayerConfig::attention_variant)
      .def_readwrite("kv_compression_rate",
                     &gcpp::LayerConfig::kv_compression_rate)
      .def_readwrite("indexer_heads", &gcpp::LayerConfig::indexer_heads)
      .def_readwrite("indexer_head_dim", &gcpp::LayerConfig::indexer_head_dim)
      .def_readwrite("indexer_top_k", &gcpp::LayerConfig::indexer_top_k)
      .def_readwrite("num_shared_experts",
                     &gcpp::LayerConfig::num_shared_experts)
      .def_readwrite("sigmoid_gating", &gcpp::LayerConfig::sigmoid_gating)
      .def_readwrite("use_routing_bias", &gcpp::LayerConfig::use_routing_bias)
      .def_readwrite("o_lora_rank", &gcpp::LayerConfig::o_lora_rank)
      .def_readwrite("o_groups", &gcpp::LayerConfig::o_groups)
      .def_readwrite("swiglu_limit", &gcpp::LayerConfig::swiglu_limit)
      .def_readwrite("route_scale", &gcpp::LayerConfig::route_scale)
      .def_readwrite("router_score", &gcpp::LayerConfig::router_score)
      .def_readwrite("hash_routing", &gcpp::LayerConfig::hash_routing)
      .def_readwrite("internal", &gcpp::LayerConfig::internal);

  class_<gcpp::VitConfig>(py_module, "VitConfig")
      .def(init())
      .def_readwrite("model_dim", &gcpp::VitConfig::model_dim)
      .def_readwrite("seq_len", &gcpp::VitConfig::seq_len)
      .def_readwrite("num_scales", &gcpp::VitConfig::num_scales)
      .def_readwrite("patch_width", &gcpp::VitConfig::patch_width)
      .def_readwrite("image_size", &gcpp::VitConfig::image_size)
      .def_readwrite("layer_configs", &gcpp::VitConfig::layer_configs);

  class_<gcpp::InternalModelConfig>(py_module, "InternalModelConfig")
      .def(init<>());

  class_<gcpp::ModelConfig>(py_module, "ModelConfig")
      .def(init<>())
      .def(init<gcpp::Model, gcpp::Type, gcpp::PromptWrapping>())
      .def(init<const char*>())
      .def_readwrite("model_family_version",
                     &gcpp::ModelConfig::model_family_version)
      .def_readwrite("display_name", &gcpp::ModelConfig::display_name)
      .def_readwrite("model", &gcpp::ModelConfig::model)
      .def_readwrite("wrapping", &gcpp::ModelConfig::wrapping)
      .def_readwrite("weight", &gcpp::ModelConfig::weight)
      .def_readwrite("num_layers", &gcpp::ModelConfig::num_layers)
      .def_readwrite("model_dim", &gcpp::ModelConfig::model_dim)
      .def_readwrite("vocab_size", &gcpp::ModelConfig::vocab_size)
      .def_readwrite("max_seq_len", &gcpp::ModelConfig::max_seq_len)
      // Skip `unused_num_tensor_scales`.
      .def_readwrite("att_cap", &gcpp::ModelConfig::att_cap)
      .def_readwrite("final_cap", &gcpp::ModelConfig::final_cap)
      .def_readwrite("absolute_pe", &gcpp::ModelConfig::absolute_pe)
      .def_readwrite("query_scale", &gcpp::ModelConfig::query_scale)
      .def_readwrite("layer_configs", &gcpp::ModelConfig::layer_configs)
      .def_readwrite("attention_window_sizes",
                     &gcpp::ModelConfig::attention_window_sizes)
      .def_readwrite("norm_num_groups", &gcpp::ModelConfig::norm_num_groups)
      .def_readwrite("vit_config", &gcpp::ModelConfig::vit_config)
      .def_readwrite("pool_dim", &gcpp::ModelConfig::pool_dim)
      .def_readwrite("eos_id", &gcpp::ModelConfig::eos_id)
      .def_readwrite("secondary_eos_id", &gcpp::ModelConfig::secondary_eos_id)
      .def_readwrite("scale_base_names", &gcpp::ModelConfig::scale_base_names)
      .def_readwrite("internal", &gcpp::ModelConfig::internal)
      .def_readwrite("use_global_timescale",
                     &gcpp::ModelConfig::use_global_timescale)
      .def_readwrite("partial_rotary_factor",
                     &gcpp::ModelConfig::partial_rotary_factor)
      .def_readwrite("hc_mult", &gcpp::ModelConfig::hc_mult)
      .def_readwrite("rope_theta", &gcpp::ModelConfig::rope_theta)
      .def_readwrite("global_rope_theta", &gcpp::ModelConfig::global_rope_theta)
      .def_readwrite("compress_rope_theta",
                     &gcpp::ModelConfig::compress_rope_theta)
      .def_readwrite("yarn_orig_seq_len", &gcpp::ModelConfig::yarn_orig_seq_len)
      .def_readwrite("yarn_factor", &gcpp::ModelConfig::yarn_factor)
      .def_readwrite("yarn_beta_fast", &gcpp::ModelConfig::yarn_beta_fast)
      .def_readwrite("yarn_beta_slow", &gcpp::ModelConfig::yarn_beta_slow)
      .def_readwrite("hc_sinkhorn_iters", &gcpp::ModelConfig::hc_sinkhorn_iters)
      .def_readwrite("hc_eps", &gcpp::ModelConfig::hc_eps)
      .def_readwrite("num_mtp_layers", &gcpp::ModelConfig::num_mtp_layers)
      .def_readwrite("tokenizer_kind", &gcpp::ModelConfig::tokenizer_kind)
      .def_readwrite("is_encoder_decoder",
                     &gcpp::ModelConfig::is_encoder_decoder)
      .def_readwrite("encoder_num_layers",
                     &gcpp::ModelConfig::encoder_num_layers)
      .def_readwrite("encoder_layer_configs",
                     &gcpp::ModelConfig::encoder_layer_configs)
      .def_readwrite("encoder_attention_window_sizes",
                     &gcpp::ModelConfig::encoder_attention_window_sizes)
      .def_readwrite("decoder_num_layers",
                     &gcpp::ModelConfig::decoder_num_layers)
      .def_readwrite("decoder_layer_configs",
                     &gcpp::ModelConfig::decoder_layer_configs)
      .def_readwrite("decoder_attention_window_sizes",
                     &gcpp::ModelConfig::decoder_attention_window_sizes)

      .def("add_layer_config", &gcpp::ModelConfig::AddLayerConfig,
           arg("layer_config"))
      .def("test_equal", &gcpp::ModelConfig::TestEqual, arg("other"),
           arg("print"))
      .def("overwrite_with_canonical",
           &gcpp::ModelConfig::OverwriteWithCanonical)
      .def("specifier", &gcpp::ModelConfig::Specifier);

  // Returns the sub-config for the ViT model of the PaliGemma model.
  py_module.def("vit_config", &gcpp::GetVitConfig, arg("config"));

  py_module.def("is_paligemma", &gcpp::IsPaliGemma, arg("model"));
}

}  // namespace pybind11
