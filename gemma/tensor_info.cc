#include "gemma/tensor_info.h"

#include <stddef.h>
#include <stdint.h>

#include <string>

#include "compression/types.h"
#include "gemma/configs.h"

namespace gcpp {

void TensorInfoRegistry::Add(const std::string& suffix,
                             const TensorInfo& info) {
  const size_t idx = tensors_.size();
  tensors_.push_back(info);
  // Also add suffix to `concat_names`.
  for (std::string& name : tensors_.back().concat_names) {
    name += suffix;
  }

  const std::string name = info.base_name + suffix;
  // Ensure successful insertion because `suffix` ensures uniqueness for
  // per-layer tensors, and per-model should only be inserted once.
  HWY_ASSERT_M(idx_from_name_.insert({name, idx}).second, name.c_str());
}

// Non-layer tensors.
void TensorInfoRegistry::AddModelTensors(const ModelConfig& config) {
  const std::string no_suffix;
  Add(no_suffix, {
                     .base_name = "c_embedding",
                     .source_names = {"embedder/input_embedding"},
                     .axes = {0, 1},
                     .shape = {config.vocab_size, config.model_dim},
                     .min_size = Type::kBF16,
                 });
  Add(no_suffix, {
                     .base_name = "c_final_norm",
                     .source_names = {"final_norm/scale"},
                     .axes = {0},
                     .shape = {config.model_dim},
                     .min_size = Type::kBF16,
                 });
  Add(no_suffix, {
                     .base_name = "enc_norm_bias",
                     .source_names = {"img/Transformer/encoder_norm/bias"},
                     .axes = {0},
                     .shape = {config.vit_config.model_dim},
                     .min_size = Type::kBF16,
                 });
  Add(no_suffix, {
                     .base_name = "enc_norm_scale",
                     .source_names = {"img/Transformer/encoder_norm/scale"},
                     .axes = {0},
                     .shape = {config.vit_config.model_dim},
                     .min_size = Type::kBF16,
                 });
  Add(no_suffix, {
                     .base_name = "img_emb_bias",
                     .source_names = {"img/embedding/bias"},
                     .axes = {0},
                     .shape = {config.vit_config.model_dim},
                     .min_size = Type::kF32,
                 });
  Add(no_suffix,
      {
          .base_name = "img_emb_kernel",
          .source_names = {"img/embedding/kernel"},
          .axes = {3, 0, 1, 2},
          .shape = {config.vit_config.model_dim, config.vit_config.patch_width,
                    config.vit_config.patch_width, 3},
          .min_size = Type::kBF16,
          .cols_take_extra_dims = true,
      });
  Add(no_suffix,
      {
          .base_name = "img_head_bias",
          .source_names = {"img/head/bias", "embedder/mm_input_projection/b"},
          .axes = {0},
          .shape = {config.model_dim},
          .min_size = Type::kF32,
      });
  Add(no_suffix,
      {
          .base_name = "img_head_kernel",
          .source_names = {"img/head/kernel", "embedder/mm_input_projection/w"},
          .axes = {1, 0},
          .shape = {config.model_dim, config.vit_config.model_dim},
          .min_size = Type::kBF16,
      });
  Add(no_suffix, {
                     .base_name = "img_pos_emb",
                     .source_names = {"img/pos_embedding"},
                     .axes = {0, 1},
                     .shape = {/*1,*/ config.vit_config.seq_len,
                               config.vit_config.model_dim},
                     .min_size = Type::kF32,
                 });
  // RMS norm applied to soft tokens prior to pos embedding.
  Add(no_suffix, {
                     .base_name = "mm_embed_norm",
                     .source_names = {"embedder/mm_soft_embedding_norm/scale"},
                     .axes = {0},
                     .shape = {config.vit_config.model_dim},
                     .min_size = Type::kBF16,
                 });
}

// Returns the tensors for the given image layer config.
void TensorInfoRegistry::AddImageLayerTensors(const ModelConfig& config,
                                              const LayerConfig& layer_config,
                                              const size_t img_layer_idx) {
  const std::string suffix = LayerSuffix(img_layer_idx);

  // Vit layers.
  Add(suffix, {
                  .base_name = "attn_out_w",
                  .source_names = {"MultiHeadDotProductAttention_0/out/kernel"},
                  .axes = {2, 0, 1},
                  .shape = {config.vit_config.model_dim, layer_config.heads,
                            layer_config.qkv_dim},
                  .min_size = Type::kBF16,
                  .cols_take_extra_dims = true,
              });
  Add(suffix, {
                  .base_name = "attn_out_b",
                  .source_names = {"MultiHeadDotProductAttention_0/out/bias"},
                  .axes = {0},
                  .shape = {config.vit_config.model_dim},
                  .min_size = Type::kF32,
              });
  Add(suffix,
      {
          .base_name = "q_ein_w",
          .source_names = {"MultiHeadDotProductAttention_0/query/kernel"},
          .axes = {1, 2, 0},
          .shape = {layer_config.heads, layer_config.qkv_dim,
                    config.vit_config.model_dim},
          .concat_names = {"qkv_ein_w", "k_ein_w", "v_ein_w"},
          .concat_axis = 1,
          .min_size = Type::kBF16,
      });
  Add(suffix, {
                  .base_name = "k_ein_w",
                  .source_names = {"MultiHeadDotProductAttention_0/key/kernel"},
                  .axes = {1, 2, 0},
                  .shape = {layer_config.heads, layer_config.qkv_dim,
                            config.vit_config.model_dim},
                  .concat_names = {""},
                  .min_size = Type::kBF16,
              });
  Add(suffix,
      {
          .base_name = "v_ein_w",
          .source_names = {"MultiHeadDotProductAttention_0/value/kernel"},
          .axes = {1, 2, 0},
          .shape = {layer_config.heads, layer_config.qkv_dim,
                    config.vit_config.model_dim},
          .concat_names = {""},
          .min_size = Type::kBF16,
      });
  Add(suffix, {
                  .base_name = "qkv_ein_w",
                  .source_names = {"MultiHeadDotProductAttention_0/qkv/kernel"},
                  .axes = {1, 2, 0},
                  .shape = {layer_config.heads, 3 * layer_config.qkv_dim,
                            config.vit_config.model_dim},
                  .min_size = Type::kBF16,
              });
  Add(suffix, {
                  .base_name = "q_ein_b",
                  .source_names = {"MultiHeadDotProductAttention_0/query/bias"},
                  .axes = {0, 1},
                  .shape = {layer_config.heads, layer_config.qkv_dim},
                  .concat_names = {"qkv_ein_b", "k_ein_b", "v_ein_b"},
                  .concat_axis = 1,
                  .min_size = Type::kF32,
              });
  Add(suffix, {
                  .base_name = "k_ein_b",
                  .source_names = {"MultiHeadDotProductAttention_0/key/bias"},
                  .axes = {0, 1},
                  .shape = {layer_config.kv_heads, layer_config.qkv_dim},
                  .concat_names = {""},
                  .min_size = Type::kF32,
              });
  Add(suffix, {
                  .base_name = "v_ein_b",
                  .source_names = {"MultiHeadDotProductAttention_0/value/bias"},
                  .axes = {0, 1},
                  .shape = {layer_config.kv_heads, layer_config.qkv_dim},
                  .concat_names = {""},
                  .min_size = Type::kF32,
              });
  Add(suffix, {
                  .base_name = "qkv_ein_b",
                  .source_names = {"MultiHeadDotProductAttention_0/qkv/bias"},
                  .axes = {0, 1},
                  .shape = {layer_config.heads + layer_config.kv_heads * 2,
                            layer_config.qkv_dim},
                  .min_size = Type::kF32,
              });
  Add(suffix,
      {
          .base_name = "linear_0_w",
          .source_names = {"MlpBlock_0/Dense_0/kernel"},
          .axes = {1, 0},
          .shape = {layer_config.ff_hidden_dim, config.vit_config.model_dim},
          .min_size = Type::kBF16,
      });
  Add(suffix, {
                  .base_name = "linear_0_b",
                  .source_names = {"MlpBlock_0/Dense_0/bias"},
                  .axes = {0},
                  .shape = {layer_config.ff_hidden_dim},
                  .min_size = Type::kF32,
              });
  Add(suffix,
      {
          .base_name = "linear_1_w",
          .source_names = {"MlpBlock_0/Dense_1/kernel"},
          .axes = {1, 0},
          .shape = {config.vit_config.model_dim, layer_config.ff_hidden_dim},
          .min_size = Type::kBF16,
      });
  Add(suffix, {
                  .base_name = "linear_1_b",
                  .source_names = {"MlpBlock_0/Dense_1/bias"},
                  .axes = {0},
                  .shape = {config.vit_config.model_dim},
                  .min_size = Type::kF32,
              });
  Add(suffix,
      {
          .base_name = "ln_0_bias",
          .source_names = {"img/Transformer/encoderblock/LayerNorm_0/bias",
                           "img/Transformer/encoderblock_" +
                               std::to_string(img_layer_idx) +
                               "/LayerNorm_0/bias"},
          .axes = {0},
          .shape = {config.vit_config.model_dim},
          .min_size = Type::kBF16,
      });
  Add(suffix,
      {
          .base_name = "ln_0_scale",
          .source_names = {"img/Transformer/encoderblock/LayerNorm_0/scale",
                           "img/Transformer/encoderblock_" +
                               std::to_string(img_layer_idx) +
                               "/LayerNorm_0/scale"},
          .axes = {0},
          .shape = {config.vit_config.model_dim},
          .min_size = Type::kBF16,
      });
  Add(suffix,
      {
          .base_name = "ln_1_bias",
          .source_names = {"img/Transformer/encoderblock/LayerNorm_1/bias",
                           "img/Transformer/encoderblock_" +
                               std::to_string(img_layer_idx) +
                               "/LayerNorm_1/bias"},
          .axes = {0},
          .shape = {config.vit_config.model_dim},
          .min_size = Type::kBF16,
      });
  Add(suffix,
      {
          .base_name = "ln_1_scale",
          .source_names = {"img/Transformer/encoderblock/LayerNorm_1/scale",
                           "img/Transformer/encoderblock_" +
                               std::to_string(img_layer_idx) +
                               "/LayerNorm_1/scale"},
          .axes = {0},
          .shape = {config.vit_config.model_dim},
          .min_size = Type::kBF16,
      });
}

void TensorInfoRegistry::AddGriffinLayerTensors(const LayerConfig& layer_config,
                                                const size_t layer_idx) {
  const std::string suffix = LayerSuffix(layer_idx);
  Add(suffix, {
                  .base_name = "gr_lin_x_w",
                  .source_names = {"recurrent_block/linear_x/kernel"},
                  .axes = {1, 0},
                  .shape = {layer_config.griffin_dim, layer_config.griffin_dim},
              });
  Add(suffix, {
                  .base_name = "gr_lin_x_b",
                  .source_names = {"recurrent_block/linear_x/bias"},
                  .axes = {0},
                  .shape = {layer_config.griffin_dim},
                  .min_size = Type::kF32,
              });
  Add(suffix, {
                  .base_name = "gr_lin_y_w",
                  .source_names = {"recurrent_block/linear_y/kernel"},
                  .axes = {1, 0},
                  .shape = {layer_config.griffin_dim, layer_config.griffin_dim},
              });
  Add(suffix, {
                  .base_name = "gr_lin_y_b",
                  .source_names = {"recurrent_block/linear_y/bias"},
                  .axes = {0},
                  .shape = {layer_config.griffin_dim},
                  .min_size = Type::kF32,
              });
  Add(suffix, {
                  .base_name = "gr_lin_out_w",
                  .source_names = {"recurrent_block/linear_out/kernel"},
                  .axes = {1, 0},
                  .shape = {layer_config.griffin_dim, layer_config.griffin_dim},
              });
  Add(suffix, {
                  .base_name = "gr_lin_out_b",
                  .source_names = {"recurrent_block/linear_out/bias"},
                  .axes = {0},
                  .shape = {layer_config.griffin_dim},
                  .min_size = Type::kF32,
              });
  Add(suffix,
      {
          .base_name = "gr_conv_w",
          .source_names = {"recurrent_block/conv_1d/w"},
          .axes = {0, 1},
          .shape = {layer_config.conv1d_width, layer_config.griffin_dim},
          .min_size = Type::kF32,
      });
  Add(suffix, {
                  .base_name = "gr_conv_b",
                  .source_names = {"recurrent_block/conv_1d/b"},
                  .axes = {0},
                  .shape = {layer_config.griffin_dim},
                  .min_size = Type::kF32,
              });
  Add(suffix, {
                  .base_name = "gr1_gate_w",
                  .source_names = {"recurrent_block/rg_lru/input_gate/w"},
                  .axes = {0, 2, 1},
                  .shape = {layer_config.heads,
                            layer_config.griffin_dim / layer_config.heads,
                            layer_config.griffin_dim / layer_config.heads},
                  .concat_names = {"gr_gate_w", "gr2_gate_w"},
              });
  Add(suffix, {
                  .base_name = "gr2_gate_w",
                  .source_names = {"recurrent_block/rg_lru/a_gate/w"},
                  .axes = {0, 2, 1},
                  .shape = {layer_config.heads,
                            layer_config.griffin_dim / layer_config.heads,
                            layer_config.griffin_dim / layer_config.heads},
                  .concat_names = {""},
              });
  Add(suffix, {
                  .base_name = "gr_gate_w",
                  .source_names = {"recurrent_block/rg_lru/gate/w"},
                  .axes = {0, 2, 1},
                  .shape = {2 * layer_config.heads,
                            layer_config.griffin_dim / layer_config.heads,
                            layer_config.griffin_dim / layer_config.heads},
              });
  Add(suffix, {
                  .base_name = "gr1_gate_b",
                  .source_names = {"recurrent_block/rg_lru/input_gate/b"},
                  .axes = {0},
                  .shape = {layer_config.griffin_dim},
                  .concat_names = {"gr_gate_b", "gr2_gate_b"},
                  .min_size = Type::kF32,
              });
  Add(suffix, {
                  .base_name = "gr2_gate_b",
                  .source_names = {"recurrent_block/rg_lru/a_gate/b"},
                  .axes = {0},
                  .shape = {layer_config.griffin_dim},
                  .concat_names = {""},
                  .min_size = Type::kF32,
              });
  Add(suffix, {
                  .base_name = "gr_gate_b",
                  .source_names = {"recurrent_block/rg_lru/input_gate/b"},
                  .axes = {0, 1},
                  .shape = {2 * layer_config.griffin_dim},
                  .min_size = Type::kF32,
              });
  Add(suffix, {
                  .base_name = "gr_a",
                  .source_names = {"recurrent_block/rg_lru/a_param"},
                  .axes = {0},
                  .shape = {layer_config.griffin_dim},
                  .min_size = Type::kF32,
                  .scaled_softplus = true,
              });
}

void TensorInfoRegistry::AddLayerTensors(const ModelConfig& config,
                                         const LayerConfig& layer_config,
                                         const size_t layer_idx) {
  const std::string suffix = LayerSuffix(layer_idx);
  Add(suffix, {
                  .base_name = "key_norm",
                  .source_names = {"attn/_key_norm/scale"},
                  .axes = {0},
                  .shape = {layer_config.qkv_dim},
                  .min_size = Type::kBF16,
              });
  Add(suffix, {
                  .base_name = "query_norm",
                  .source_names = {"attn/_query_norm/scale"},
                  .axes = {0},
                  .shape = {layer_config.qkv_dim},
                  .min_size = Type::kBF16,
              });
  Add(suffix, {
                  .base_name = "qkv1_w",
                  .source_names = {"attn/q_einsum/w"},
                  .axes = {0, 2, 1},
                  .shape = {layer_config.heads * layer_config.qkv_dim,
                            config.model_dim},
                  .concat_names = {"qkv_ein", "qkv2_w"},
              });
  Add(suffix, {
                  .base_name = "qkv2_w",
                  .source_names = {"attn/kv_einsum/w"},
                  .axes = {1, 0, 3, 2},
                  .shape = {2 * layer_config.kv_heads * layer_config.qkv_dim,
                            config.model_dim},
                  .concat_names = {""},
              });
  Add(suffix, {
                  .base_name = "q_ein",
                  .source_names = {"attention_block/proj_q/kernel"},
                  .axes = {1, 0},
                  .shape = {layer_config.model_dim, layer_config.model_dim},
                  .concat_names = {"qkv_ein", "k_ein", "v_ein"},
              });
  Add(suffix, {
                  .base_name = "k_ein",
                  .source_names = {"attention_block/proj_k/kernel"},
                  .axes = {1, 0},
                  .shape = {layer_config.qkv_dim, layer_config.model_dim},
                  .concat_names = {""},
              });
  Add(suffix, {
                  .base_name = "v_ein",
                  .source_names = {"attention_block/proj_v/kernel"},
                  .axes = {1, 0},
                  .shape = {layer_config.qkv_dim, layer_config.model_dim},
                  .concat_names = {""},
              });
  Add(suffix, {
                  .base_name = "qkv_ein",
                  .source_names = {"attn/qkv_einsum/w"},
                  .axes = {1, 0, 3, 2},
                  .shape = {(layer_config.heads + 2 * layer_config.kv_heads) *
                                layer_config.qkv_dim,
                            config.model_dim},
              });
  Add(suffix, {
                  .base_name = "attn_ob",
                  .source_names = {"attention_block/proj_final/bias"},
                  .axes = {0},
                  .shape = {config.model_dim},
                  .min_size = Type::kF32,
              });

  Add(suffix, {
                  .base_name = "gating_ein",
                  .source_names = {"mlp/gating_einsum/w", "mlp/gating_einsum",
                                   "mlp_block/ffw_up/w"},
                  .axes = {0, layer_config.optimized_gating ? 1u : 2u,
                           layer_config.optimized_gating ? 2u : 1u},
                  .shape = {2, layer_config.ff_hidden_dim, config.model_dim},
              });
  Add(suffix, {
                  .base_name = "gating1_w",
                  .source_names = {"none"},
                  .axes = {0, layer_config.optimized_gating ? 1u : 2u,
                           layer_config.optimized_gating ? 2u : 1u},
                  .shape = {layer_config.ff_hidden_dim, config.model_dim},
              });
  Add(suffix, {
                  .base_name = "gating2_w",
                  .source_names = {"none"},
                  .axes = {0, layer_config.optimized_gating ? 1u : 2u,
                           layer_config.optimized_gating ? 2u : 1u},
                  .shape = {layer_config.ff_hidden_dim, config.model_dim},
              });
  Add(suffix, {
                  .base_name = "linear_w",
                  .source_names = {"mlp/linear/w", "mlp/linear",
                                   "mlp_block/ffw_down/kernel"},
                  .axes = {1, 0},
                  .shape = {config.model_dim, layer_config.ff_hidden_dim},
              });
  Add(suffix, {
                  .base_name = "pre_att_ns",
                  .source_names = {"pre_attention_norm/scale",
                                   "temporal_pre_norm/scale"},
                  .axes = {0},
                  .shape = {config.model_dim},
                  .min_size = Type::kBF16,
              });
  Add(suffix,
      {
          .base_name = "pre_ff_ns",
          .source_names = {"pre_ffw_norm/scale", "channel_pre_norm/scale"},
          .axes = {0},
          .shape = {config.model_dim},
          .min_size = Type::kBF16,
      });
  Add(suffix, {
                  .base_name = "post_att_ns",
                  .source_names = {"post_attention_norm/scale"},
                  .axes = {0},
                  .shape = {config.model_dim},
                  .min_size = Type::kBF16,
              });
  Add(suffix, {
                  .base_name = "post_ff_ns",
                  .source_names = {"post_ffw_norm/scale"},
                  .axes = {0},
                  .shape = {config.model_dim},
                  .min_size = Type::kBF16,
              });
  Add(suffix, {
                  .base_name = "ffw_gat_b",
                  .source_names = {"mlp_block/ffw_up/b"},
                  .axes = {0},
                  .shape = {2 * layer_config.ff_hidden_dim},
                  .min_size = Type::kF32,
              });
  Add(suffix, {
                  .base_name = "ffw_out_b",
                  .source_names = {"mlp_block/ffw_down/bias"},
                  .axes = {0},
                  .shape = {config.model_dim},
                  .min_size = Type::kF32,
              });
  Add(suffix,
      {
          .base_name = "att_ein",
          .source_names = {"attn/attn_vec_einsum/w",
                           "attention_block/proj_final/kernel"},
          .preshape = {layer_config.heads, layer_config.qkv_dim,
                       config.model_dim},
          .axes = {0, 2, 1},
          .shape = {layer_config.heads, config.model_dim, layer_config.qkv_dim},
      });
  Add(suffix,
      {
          .base_name = "att_w",
          .shape = {config.model_dim, layer_config.heads, layer_config.qkv_dim},
          .cols_take_extra_dims = true,
      });

  if (config.model == Model::GRIFFIN_2B) {
    AddGriffinLayerTensors(layer_config, layer_idx);
  }
}

TensorInfoRegistry::TensorInfoRegistry(const ModelConfig& config) {
  // Upper bound on the number of `Add()` calls in `Add*Tensors()`. Loose bound
  // in case those are changed without updating this. Better to allocate a bit
  // more than to 1.5-2x the size if too little.
  tensors_.reserve(10 + 32 * config.layer_configs.size() +
                   24 * config.vit_config.layer_configs.size());
  AddModelTensors(config);
  for (size_t i = 0; i < config.layer_configs.size(); ++i) {
    AddLayerTensors(config, config.layer_configs[i], i);
  }
  for (size_t i = 0; i < config.vit_config.layer_configs.size(); ++i) {
    AddImageLayerTensors(config, config.vit_config.layer_configs[i], i);
  }
}

TensorInfo TensorInfoRegistry::TensorInfoFromSourcePath(const std::string& path,
                                                        int layer_idx) const {
  for (const TensorInfo& tensor : tensors_) {
    for (const std::string& source_name : tensor.source_names) {
      // path ends with source_name?
      const size_t pos = path.rfind(source_name);
      if (pos != std::string::npos && path.size() == pos + source_name.size()) {
        std::string name = tensor.base_name;
        if (layer_idx >= 0) name += LayerSuffix(static_cast<size_t>(layer_idx));
        return TensorInfoFromName(name);
      }
    }
  }
  return TensorInfo();
}

}  // namespace gcpp
