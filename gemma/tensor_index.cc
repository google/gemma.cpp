#include "gemma/tensor_index.h"

#include <stddef.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include "compression/shared.h"
#include "gemma/configs.h"

namespace gcpp {
namespace {

// Returns the non-layer tensors for the model.
std::vector<TensorInfo> ModelTensors(const ModelConfig& config) {
  return {
      TensorInfo{
          .name = "c_embedding",
          .source_names = {"embedder/input_embedding"},
          .axes = {0, 1},
          .shape = {config.vocab_size, config.model_dim},
          .min_size = Type::kBF16,
      },
      TensorInfo{
          .name = "c_final_norm",
          .source_names = {"final_norm/scale"},
          .axes = {0},
          .shape = {config.model_dim},
          .min_size = Type::kBF16,
      },
      TensorInfo{
          .name = "enc_norm_bias",
          .source_names = {"img/Transformer/encoder_norm/bias"},
          .axes = {0},
          .shape = {config.vit_config.model_dim},
          .min_size = Type::kBF16,
      },
      TensorInfo{
          .name = "enc_norm_scale",
          .source_names = {"img/Transformer/encoder_norm/scale"},
          .axes = {0},
          .shape = {config.vit_config.model_dim},
          .min_size = Type::kBF16,
      },
      TensorInfo{
          .name = "img_emb_bias",
          .source_names = {"img/embedding/bias"},
          .axes = {0},
          .shape = {config.vit_config.model_dim},
          .min_size = Type::kF32,
      },
      TensorInfo{
          .name = "img_emb_kernel",
          .source_names = {"img/embedding/kernel"},
          .axes = {3, 0, 1, 2},
          .shape = {config.vit_config.model_dim, config.vit_config.patch_width,
                    config.vit_config.patch_width, 3},
          .min_size = Type::kBF16,
          .cols_take_extra_dims = true,
      },
      TensorInfo{
          .name = "img_head_bias",
          .source_names = {"img/head/bias", "embedder/mm_input_projection/b"},
          .axes = {0},
          .shape = {config.model_dim},
          .min_size = Type::kF32,
      },
      TensorInfo{
          .name = "img_head_kernel",
          .source_names = {"img/head/kernel", "embedder/mm_input_projection/w"},
          .axes = {1, 0},
          .shape = {config.model_dim, config.vit_config.model_dim},
          .min_size = Type::kBF16,
      },
      TensorInfo{
          .name = "img_pos_emb",
          .source_names = {"img/pos_embedding"},
          .axes = {0, 1},
          .shape = {/*1,*/ config.vit_config.seq_len,
                    config.vit_config.model_dim},
          .min_size = Type::kF32,
      },
      // RMS norm applied to soft tokens prior to pos embedding.
      TensorInfo{
          .name = "mm_embed_norm",
          .source_names = {"embedder/mm_soft_embedding_norm/scale"},
          .axes = {0},
          .shape = {config.vit_config.model_dim},
          .min_size = Type::kBF16,
      },
  };
}

// Returns the tensors for the given image layer config.
std::vector<TensorInfo> ImageLayerTensors(const ModelConfig& config,
                                          const LayerConfig& layer_config,
                                          const int img_layer_idx) {
  return {
      // Vit layers.
      TensorInfo{
          .name = "attn_out_w",
          .source_names = {"MultiHeadDotProductAttention_0/out/kernel"},
          .axes = {2, 0, 1},
          .shape = {config.vit_config.model_dim, layer_config.heads,
                    layer_config.qkv_dim},
          .min_size = Type::kBF16,
          .cols_take_extra_dims = true,
      },
      TensorInfo{
          .name = "attn_out_b",
          .source_names = {"MultiHeadDotProductAttention_0/out/bias"},
          .axes = {0},
          .shape = {config.vit_config.model_dim},
          .min_size = Type::kF32,
      },
      TensorInfo{
          .name = "q_ein_w",
          .source_names = {"MultiHeadDotProductAttention_0/query/kernel"},
          .axes = {1, 2, 0},
          .shape = {layer_config.heads, layer_config.qkv_dim,
                    config.vit_config.model_dim},
          .concat_names = {"qkv_ein_w", "k_ein_w", "v_ein_w"},
          .concat_axis = 1,
          .min_size = Type::kBF16,
      },
      TensorInfo{
          .name = "k_ein_w",
          .source_names = {"MultiHeadDotProductAttention_0/key/kernel"},
          .axes = {1, 2, 0},
          .shape = {layer_config.heads, layer_config.qkv_dim,
                    config.vit_config.model_dim},
          .concat_names = {""},
          .min_size = Type::kBF16,
      },
      TensorInfo{
          .name = "v_ein_w",
          .source_names = {"MultiHeadDotProductAttention_0/value/kernel"},
          .axes = {1, 2, 0},
          .shape = {layer_config.heads, layer_config.qkv_dim,
                    config.vit_config.model_dim},
          .concat_names = {""},
          .min_size = Type::kBF16,
      },
      TensorInfo{
          .name = "qkv_ein_w",
          .source_names = {"MultiHeadDotProductAttention_0/qkv/kernel"},
          .axes = {1, 2, 0},
          .shape = {layer_config.heads, 3 * layer_config.qkv_dim,
                    config.vit_config.model_dim},
          .min_size = Type::kBF16,
      },
      TensorInfo{
          .name = "q_ein_b",
          .source_names = {"MultiHeadDotProductAttention_0/query/bias"},
          .axes = {0, 1},
          .shape = {layer_config.heads, layer_config.qkv_dim},
          .concat_names = {"qkv_ein_b", "k_ein_b", "v_ein_b"},
          .concat_axis = 1,
          .min_size = Type::kF32,
      },
      TensorInfo{
          .name = "k_ein_b",
          .source_names = {"MultiHeadDotProductAttention_0/key/bias"},
          .axes = {0, 1},
          .shape = {layer_config.kv_heads, layer_config.qkv_dim},
          .concat_names = {""},
          .min_size = Type::kF32,
      },
      TensorInfo{
          .name = "v_ein_b",
          .source_names = {"MultiHeadDotProductAttention_0/value/bias"},
          .axes = {0, 1},
          .shape = {layer_config.kv_heads, layer_config.qkv_dim},
          .concat_names = {""},
          .min_size = Type::kF32,
      },
      TensorInfo{
          .name = "qkv_ein_b",
          .source_names = {"MultiHeadDotProductAttention_0/qkv/bias"},
          .axes = {0, 1},
          .shape = {layer_config.heads + layer_config.kv_heads * 2,
                    layer_config.qkv_dim},
          .min_size = Type::kF32,
      },
      TensorInfo{
          .name = "linear_0_w",
          .source_names = {"MlpBlock_0/Dense_0/kernel"},
          .axes = {1, 0},
          .shape = {layer_config.ff_hidden_dim, config.vit_config.model_dim},
          .min_size = Type::kBF16,
      },
      TensorInfo{
          .name = "linear_0_b",
          .source_names = {"MlpBlock_0/Dense_0/bias"},
          .axes = {0},
          .shape = {layer_config.ff_hidden_dim},
          .min_size = Type::kF32,
      },
      TensorInfo{
          .name = "linear_1_w",
          .source_names = {"MlpBlock_0/Dense_1/kernel"},
          .axes = {1, 0},
          .shape = {config.vit_config.model_dim, layer_config.ff_hidden_dim},
          .min_size = Type::kBF16,
      },
      TensorInfo{
          .name = "linear_1_b",
          .source_names = {"MlpBlock_0/Dense_1/bias"},
          .axes = {0},
          .shape = {config.vit_config.model_dim},
          .min_size = Type::kF32,
      },
      TensorInfo{
          .name = "ln_0_bias",
          .source_names = {"img/Transformer/encoderblock/LayerNorm_0/bias",
                           "img/Transformer/encoderblock_" +
                               std::to_string(img_layer_idx) +
                               "/LayerNorm_0/bias"},
          .axes = {0},
          .shape = {config.vit_config.model_dim},
          .min_size = Type::kBF16,
      },
      TensorInfo{
          .name = "ln_0_scale",
          .source_names = {"img/Transformer/encoderblock/LayerNorm_0/scale",
                           "img/Transformer/encoderblock_" +
                               std::to_string(img_layer_idx) +
                               "/LayerNorm_0/scale"},
          .axes = {0},
          .shape = {config.vit_config.model_dim},
          .min_size = Type::kBF16,
      },
      TensorInfo{
          .name = "ln_1_bias",
          .source_names = {"img/Transformer/encoderblock/LayerNorm_1/bias",
                           "img/Transformer/encoderblock_" +
                               std::to_string(img_layer_idx) +
                               "/LayerNorm_1/bias"},
          .axes = {0},
          .shape = {config.vit_config.model_dim},
          .min_size = Type::kBF16,
      },
      TensorInfo{
          .name = "ln_1_scale",
          .source_names = {"img/Transformer/encoderblock/LayerNorm_1/scale",
                           "img/Transformer/encoderblock_" +
                               std::to_string(img_layer_idx) +
                               "/LayerNorm_1/scale"},
          .axes = {0},
          .shape = {config.vit_config.model_dim},
          .min_size = Type::kBF16,
      },
  };
}

// Returns the tensors for the given LLM layer config.
std::vector<TensorInfo> LLMLayerTensors(const ModelConfig& config,
                                        const LayerConfig& layer_config,
                                        bool reshape_att) {
  std::vector<TensorInfo> tensors = {
      TensorInfo{
          .name = "key_norm",
          .source_names = {"attn/_key_norm/scale"},
          .axes = {0},
          .shape = {layer_config.qkv_dim},
          .min_size = Type::kBF16,
      },
      TensorInfo{
          .name = "query_norm",
          .source_names = {"attn/_query_norm/scale"},
          .axes = {0},
          .shape = {layer_config.qkv_dim},
          .min_size = Type::kBF16,
      },
      TensorInfo{
          .name = "qkv1_w",
          .source_names = {"attn/q_einsum/w"},
          .axes = {0, 2, 1},
          .shape = {layer_config.heads * layer_config.qkv_dim,
                    config.model_dim},
          .concat_names = {"qkv_ein", "qkv2_w"},
      },
      TensorInfo{
          .name = "qkv2_w",
          .source_names = {"attn/kv_einsum/w"},
          .axes = {1, 0, 3, 2},
          .shape = {2 * layer_config.kv_heads * layer_config.qkv_dim,
                    config.model_dim},
          .concat_names = {""},
      },
      TensorInfo{
          .name = "q_ein",
          .source_names = {"attention_block/proj_q/kernel"},
          .axes = {1, 0},
          .shape = {layer_config.model_dim, layer_config.model_dim},
          .concat_names = {"qkv_ein", "k_ein", "v_ein"},
      },
      TensorInfo{
          .name = "k_ein",
          .source_names = {"attention_block/proj_k/kernel"},
          .axes = {1, 0},
          .shape = {layer_config.qkv_dim, layer_config.model_dim},
          .concat_names = {""},
      },
      TensorInfo{
          .name = "v_ein",
          .source_names = {"attention_block/proj_v/kernel"},
          .axes = {1, 0},
          .shape = {layer_config.qkv_dim, layer_config.model_dim},
          .concat_names = {""},
      },
      TensorInfo{
          .name = "qkv_ein",
          .source_names = {"attn/qkv_einsum/w"},
          .axes = {1, 0, 3, 2},
          .shape = {(layer_config.heads + 2 * layer_config.kv_heads) *
                        layer_config.qkv_dim,
                    config.model_dim},
      },
      TensorInfo{
          .name = "attn_ob",
          .source_names = {"attention_block/proj_final/bias"},
          .axes = {0},
          .shape = {config.model_dim},
          .min_size = Type::kF32,
      },
      // Griffin layers.
      TensorInfo{
          .name = "gr_lin_x_w",
          .source_names = {"recurrent_block/linear_x/kernel"},
          .axes = {1, 0},
          .shape = {layer_config.griffin_dim, layer_config.griffin_dim},
      },
      TensorInfo{
          .name = "gr_lin_x_b",
          .source_names = {"recurrent_block/linear_x/bias"},
          .axes = {0},
          .shape = {layer_config.griffin_dim},
          .min_size = Type::kF32,
      },
      TensorInfo{
          .name = "gr_lin_y_w",
          .source_names = {"recurrent_block/linear_y/kernel"},
          .axes = {1, 0},
          .shape = {layer_config.griffin_dim, layer_config.griffin_dim},
      },
      TensorInfo{
          .name = "gr_lin_y_b",
          .source_names = {"recurrent_block/linear_y/bias"},
          .axes = {0},
          .shape = {layer_config.griffin_dim},
          .min_size = Type::kF32,
      },
      TensorInfo{
          .name = "gr_lin_out_w",
          .source_names = {"recurrent_block/linear_out/kernel"},
          .axes = {1, 0},
          .shape = {layer_config.griffin_dim, layer_config.griffin_dim},
      },
      TensorInfo{
          .name = "gr_lin_out_b",
          .source_names = {"recurrent_block/linear_out/bias"},
          .axes = {0},
          .shape = {layer_config.griffin_dim},
          .min_size = Type::kF32,
      },
      TensorInfo{
          .name = "gr_conv_w",
          .source_names = {"recurrent_block/conv_1d/w"},
          .axes = {0, 1},
          .shape = {layer_config.conv1d_width, layer_config.griffin_dim},
          .min_size = Type::kF32,
      },
      TensorInfo{
          .name = "gr_conv_b",
          .source_names = {"recurrent_block/conv_1d/b"},
          .axes = {0},
          .shape = {layer_config.griffin_dim},
          .min_size = Type::kF32,
      },
      TensorInfo{
          .name = "gr1_gate_w",
          .source_names = {"recurrent_block/rg_lru/input_gate/w"},
          .axes = {0, 2, 1},
          .shape = {layer_config.heads,
                    layer_config.griffin_dim / layer_config.heads,
                    layer_config.griffin_dim / layer_config.heads},
          .concat_names = {"gr_gate_w", "gr2_gate_w"},
      },
      TensorInfo{
          .name = "gr2_gate_w",
          .source_names = {"recurrent_block/rg_lru/a_gate/w"},
          .axes = {0, 2, 1},
          .shape = {layer_config.heads,
                    layer_config.griffin_dim / layer_config.heads,
                    layer_config.griffin_dim / layer_config.heads},
          .concat_names = {""},
      },
      TensorInfo{
          .name = "gr_gate_w",
          .source_names = {"recurrent_block/rg_lru/gate/w"},
          .axes = {0, 2, 1},
          .shape = {2 * layer_config.heads,
                    layer_config.griffin_dim / layer_config.heads,
                    layer_config.griffin_dim / layer_config.heads},
      },
      TensorInfo{
          .name = "gr1_gate_b",
          .source_names = {"recurrent_block/rg_lru/input_gate/b"},
          .axes = {0},
          .shape = {layer_config.griffin_dim},
          .concat_names = {"gr_gate_b", "gr2_gate_b"},
          .min_size = Type::kF32,
      },
      TensorInfo{
          .name = "gr2_gate_b",
          .source_names = {"recurrent_block/rg_lru/a_gate/b"},
          .axes = {0},
          .shape = {layer_config.griffin_dim},
          .concat_names = {""},
          .min_size = Type::kF32,
      },
      TensorInfo{
          .name = "gr_gate_b",
          .source_names = {"recurrent_block/rg_lru/input_gate/b"},
          .axes = {0, 1},
          .shape = {2 * layer_config.griffin_dim},
          .min_size = Type::kF32,
      },
      TensorInfo{
          .name = "gr_a",
          .source_names = {"recurrent_block/rg_lru/a_param"},
          .axes = {0},
          .shape = {layer_config.griffin_dim},
          .min_size = Type::kF32,
          .scaled_softplus = true,
      },

      TensorInfo{
          .name = "gating_ein",
          .source_names = {"mlp/gating_einsum/w", "mlp/gating_einsum",
                           "mlp_block/ffw_up/w"},
          .axes = {0, layer_config.optimized_gating ? 1u : 2u,
                   layer_config.optimized_gating ? 2u : 1u},
          .shape = {2, layer_config.ff_hidden_dim, config.model_dim},
      },
      TensorInfo{
          .name = "gating1_w",
          .source_names = {"none"},
          .axes = {0, layer_config.optimized_gating ? 1u : 2u,
                   layer_config.optimized_gating ? 2u : 1u},
          .shape = {layer_config.ff_hidden_dim, config.model_dim},
      },
      TensorInfo{
          .name = "gating2_w",
          .source_names = {"none"},
          .axes = {0, layer_config.optimized_gating ? 1u : 2u,
                   layer_config.optimized_gating ? 2u : 1u},
          .shape = {layer_config.ff_hidden_dim, config.model_dim},
      },
      TensorInfo{
          .name = "linear_w",
          .source_names = {"mlp/linear/w", "mlp/linear",
                           "mlp_block/ffw_down/kernel"},
          .axes = {1, 0},
          .shape = {config.model_dim, layer_config.ff_hidden_dim},
      },
      TensorInfo{
          .name = "pre_att_ns",
          .source_names = {"pre_attention_norm/scale",
                           "temporal_pre_norm/scale"},
          .axes = {0},
          .shape = {config.model_dim},
          .min_size = Type::kBF16,
      },
      TensorInfo{
          .name = "pre_ff_ns",
          .source_names = {"pre_ffw_norm/scale", "channel_pre_norm/scale"},
          .axes = {0},
          .shape = {config.model_dim},
          .min_size = Type::kBF16,
      },
      TensorInfo{
          .name = "post_att_ns",
          .source_names = {"post_attention_norm/scale"},
          .axes = {0},
          .shape = {config.model_dim},
          .min_size = Type::kBF16,
      },
      TensorInfo{
          .name = "post_ff_ns",
          .source_names = {"post_ffw_norm/scale"},
          .axes = {0},
          .shape = {config.model_dim},
          .min_size = Type::kBF16,
      },
      TensorInfo{
          .name = "ffw_gat_b",
          .source_names = {"mlp_block/ffw_up/b"},
          .axes = {0},
          .shape = {2 * layer_config.ff_hidden_dim},
          .min_size = Type::kF32,
      },
      TensorInfo{
          .name = "ffw_out_b",
          .source_names = {"mlp_block/ffw_down/bias"},
          .axes = {0},
          .shape = {config.model_dim},
          .min_size = Type::kF32,
      },
  };
  if (reshape_att) {
    tensors.push_back(TensorInfo{
        .name = "att_w",
        .source_names = {"attn/attn_vec_einsum/w",
                         "attention_block/proj_final/kernel"},
        .preshape = {layer_config.heads, layer_config.qkv_dim,
                     config.model_dim},
        .axes = {2, 0, 1},
        .shape = {config.model_dim, layer_config.heads, layer_config.qkv_dim},
        .cols_take_extra_dims = true,
    });
    tensors.push_back(TensorInfo{
        .name = "att_ein",
        .shape = {layer_config.heads, config.model_dim, layer_config.qkv_dim},
    });
  } else {
    tensors.push_back(TensorInfo{
        .name = "att_ein",
        .source_names = {"attn/attn_vec_einsum/w",
                         "attention_block/proj_final/kernel"},
        .preshape = {layer_config.heads, layer_config.qkv_dim,
                     config.model_dim},
        .axes = {0, 2, 1},
        .shape = {layer_config.heads, config.model_dim, layer_config.qkv_dim},
    });
    tensors.push_back(TensorInfo{
        .name = "att_w",
        .shape = {config.model_dim, layer_config.heads, layer_config.qkv_dim},
        .cols_take_extra_dims = true,
    });
  }
  return tensors;
}

}  // namespace

TensorIndex::TensorIndex(const ModelConfig& config, int llm_layer_idx,
                         int img_layer_idx, bool reshape_att)
    : config_(config),
      llm_layer_idx_(llm_layer_idx),
      img_layer_idx_(img_layer_idx) {
  int layer_idx = std::max(llm_layer_idx_, img_layer_idx_);
  std::string suffix;
  if (layer_idx >= 0) {
    suffix = "_" + std::to_string(layer_idx);
  }
  if (llm_layer_idx < 0 && img_layer_idx < 0) {
    tensors_ = ModelTensors(config);
  } else if (llm_layer_idx_ < 0 && 0 <= img_layer_idx &&
             img_layer_idx < config.vit_config.layer_configs.size()) {
    const auto& layer_config = config.vit_config.layer_configs[img_layer_idx];
    tensors_ = ImageLayerTensors(config, layer_config, img_layer_idx);
  } else if (0 <= llm_layer_idx &&
             llm_layer_idx < config.layer_configs.size()) {
    const auto& layer_config = config.layer_configs[llm_layer_idx];
    tensors_ = LLMLayerTensors(config, layer_config, reshape_att);
  }
  for (size_t i = 0; i < tensors_.size(); ++i) {
    std::string key = tensors_[i].name + suffix;
    name_map_.insert({key, i});
  }
}

TensorInfo TensorIndex::TensorInfoFromSourcePath(
    const std::string& path) const {
  for (const auto& tensor : tensors_) {
    for (const auto& source_name : tensor.source_names) {
      auto pos = path.rfind(source_name);
      if (pos != std::string::npos && path.size() == pos + source_name.size())
        return tensor;
    }
  }
  return TensorInfo();
}

const TensorInfo* TensorIndex::FindName(const std::string& name) const {
  std::string name_to_find = name;
  if (!std::isdigit(name[name.size() - 1])) {
    if (img_layer_idx_ >= 0 && llm_layer_idx_ < 0) {
      name_to_find = name + "_" + std::to_string(img_layer_idx_);
    } else if (llm_layer_idx_ >= 0) {
      name_to_find = name + "_" + std::to_string(llm_layer_idx_);
    }
  }
  auto it = name_map_.find(name_to_find);
  if (it == name_map_.end()) {
    return nullptr;
  }
  return &tensors_[it->second];
}

}  // namespace gcpp