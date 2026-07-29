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

// DeepSeek tensor registrations, split out of TensorInfoRegistry
// (gemma/tensor_info.cc) to keep the DeepSeek additions in one place.

#include <stddef.h>

#include <string>

#include "compression/types.h"
#include "gemma/configs.h"
#include "gemma/tensor_info.h"

namespace gcpp {

// Model-level tensors: untied output head, mHC head collapse and the
// multi-token-prediction extras.
void TensorInfoRegistry::AddDeepSeekModelTensors(const ModelConfig& config) {
  const std::string no_suffix;
  // mHC head collapse: per-token sigmoid read weights over the residual
  // streams before the final norm (DeepSeek V4 `hc_head_*`); the MTP block
  // has its own identically-shaped triple under `mtp.0.`.
  const auto add_hc_collapse = [&](const std::string& base_prefix,
                                   const std::string& source_prefix) {
    Add(no_suffix,
        {
            .base_name = base_prefix + "_fn",
            .source_names = {source_prefix + "hc_head_fn"},
            .axes = {0, 1},
            .shape = {config.hc_mult, config.hc_mult * config.model_dim},
            .min_size = Type::kF32,
        });
    Add(no_suffix, {
                       .base_name = base_prefix + "_base",
                       .source_names = {source_prefix + "hc_head_base"},
                       .axes = {0},
                       .shape = {config.hc_mult},
                       .min_size = Type::kF32,
                   });
    Add(no_suffix, {
                       .base_name = base_prefix + "_scale",
                       .source_names = {source_prefix + "hc_head_scale"},
                       .axes = {0},
                       .shape = {1},
                       .min_size = Type::kF32,
                   });
  };

  // DeepSeek models have an untied output head.
  // Note: This is now moved to gemma/tensor_info.cc.

  if (config.hc_mult > 1) {
    add_hc_collapse("hc_head", "");
  }
  if (config.num_mtp_layers > 0) {
    // Multi-token-prediction block extras (DeepSeek V4 `mtp.0.*`). The block
    // itself is registered as an extra layer, see the ctor.
    Add(no_suffix, {
                       .base_name = "mtp_e_proj",
                       .source_names = {"mtp.0.e_proj.weight"},
                       .axes = {0, 1},
                       .shape = {config.model_dim, config.model_dim},
                   });
    Add(no_suffix, {
                       .base_name = "mtp_h_proj",
                       .source_names = {"mtp.0.h_proj.weight"},
                       .axes = {0, 1},
                       .shape = {config.model_dim, config.model_dim},
                   });
    Add(no_suffix, {
                       .base_name = "mtp_enorm",
                       .source_names = {"mtp.0.enorm.weight"},
                       .axes = {0},
                       .shape = {config.model_dim},
                       .min_size = Type::kBF16,
                   });
    Add(no_suffix, {
                       .base_name = "mtp_hnorm",
                       .source_names = {"mtp.0.hnorm.weight"},
                       .axes = {0},
                       .shape = {config.model_dim},
                       .min_size = Type::kBF16,
                   });
    Add(no_suffix, {
                       .base_name = "mtp_norm",
                       .source_names = {"mtp.0.norm.weight"},
                       .axes = {0},
                       .shape = {config.model_dim},
                       .min_size = Type::kBF16,
                   });
    add_hc_collapse("mtp_hc", "mtp.0.");
  }
}

// Per-layer tensors: MLA projections, compressors, lightning indexer,
// routing bias / hash tables and the per-layer mHC weights.
void TensorInfoRegistry::AddDeepSeekLayerTensors(
    const ModelConfig& config, const LayerConfig& layer_config,
    const std::string& suffix) {
  // DeepSeek Multi-head Latent Attention tensors.
  if (layer_config.IsMLA()) {
    const size_t kv_a_rows = layer_config.KVLatentDim();
    if (layer_config.q_lora_rank > 0) {
      Add(suffix, {
                      .base_name = "mla_q_a",
                      .source_names = {"attn.wq_a.weight"},
                      .axes = {0, 1},
                      .shape = {layer_config.q_lora_rank, config.model_dim},
                  });
      Add(suffix, {
                      .base_name = "mla_q_a_ns",
                      .source_names = {"attn.q_norm.weight"},
                      .axes = {0},
                      .shape = {layer_config.q_lora_rank},
                      .min_size = Type::kBF16,
                  });
      Add(suffix, {
                      .base_name = "mla_q_b",
                      .source_names = {"attn.wq_b.weight"},
                      .axes = {0, 1},
                      .shape = {layer_config.heads * layer_config.qkv_dim,
                                layer_config.q_lora_rank},
                  });
    } else {
      // Direct query projection from the model dim.
      Add(suffix, {
                      .base_name = "mla_q_b",
                      .source_names = {"attn.wq.weight"},
                      .axes = {0, 1},
                      .shape = {layer_config.heads * layer_config.qkv_dim,
                                config.model_dim},
                  });
    }
    Add(suffix, {
                    .base_name = "mla_kv_a",
                    .source_names = {"attn.wkv.weight"},
                    .axes = {0, 1},
                    .shape = {kv_a_rows, config.model_dim},
                });
    // V4 normalizes the full latent (RoPE applied after the norm); V3 only
    // normalizes the c_kv part.
    Add(suffix, {
                    .base_name = "mla_kv_a_ns",
                    .source_names = {"attn.kv_norm.weight"},
                    .axes = {0},
                    .shape = {layer_config.IsV4MLA()
                                  ? kv_a_rows
                                  : size_t{layer_config.kv_lora_rank}},
                    .min_size = Type::kBF16,
                });
    if (layer_config.IsV4MLA()) {
      // Grouped low-rank output projection: per-group
      // [o_lora_rank, heads/o_groups * qkv_dim] wo_a slices stacked on rows,
      // then wo_b over the concatenated group outputs.
      Add(suffix,
          {
              .base_name = "mla_o_a",
              .source_names = {"attn.wo_a.weight"},
              .axes = {0, 1},
              .shape = {layer_config.o_groups * layer_config.o_lora_rank,
                        layer_config.heads / layer_config.o_groups *
                            layer_config.qkv_dim},
          });
      Add(suffix, {
                      .base_name = "mla_o_b",
                      .source_names = {"attn.wo_b.weight"},
                      .axes = {0, 1},
                      .shape = {config.model_dim, layer_config.o_groups *
                                                      layer_config.o_lora_rank},
                  });
      // Learned attention sink: per-head logit added to the softmax
      // denominator only.
      Add(suffix, {
                      .base_name = "attn_sink",
                      .source_names = {"attn.attn_sink"},
                      .axes = {0},
                      .shape = {layer_config.heads},
                      .min_size = Type::kF32,
                  });
    } else {
      Add(suffix,
          {
              .base_name = "mla_kv_b",
              .source_names = {"self_attn.kv_b_proj.weight"},
              .axes = {0, 1},
              .shape = {layer_config.heads * (layer_config.NopeHeadDim() +
                                              layer_config.v_head_dim),
                        layer_config.kv_lora_rank},
          });
    }
    // Sequence-axis compression.
    if (layer_config.kv_compression_rate > 1) {
      if (layer_config.IsV4MLA()) {
        // V4 learned gated compressor: wkv/wgate project the layer input x;
        // ape is a learned per-position-in-block bias; norm is applied to the
        // sealed entry. With overlap (rate 4), dims are doubled (coff=2).
        const size_t coff = layer_config.CompressorCoff();
        Add(suffix, {
                        .base_name = "comp_wkv",
                        .source_names = {"attn.compressor.wkv.weight"},
                        .axes = {0, 1},
                        .shape = {coff * kv_a_rows, config.model_dim},
                    });
        Add(suffix, {
                        .base_name = "comp_wgate",
                        .source_names = {"attn.compressor.wgate.weight"},
                        .axes = {0, 1},
                        .shape = {coff * kv_a_rows, config.model_dim},
                    });
        Add(suffix,
            {
                .base_name = "comp_ape",
                .source_names = {"attn.compressor.ape"},
                .axes = {0, 1},
                .shape = {layer_config.kv_compression_rate, coff * kv_a_rows},
                .min_size = Type::kF32,
            });
        Add(suffix, {
                        .base_name = "comp_ns",
                        .source_names = {"attn.compressor.norm.weight"},
                        .axes = {0},
                        .shape = {kv_a_rows},
                        .min_size = Type::kBF16,
                    });
      } else {
        // V3-style softmax-gated pooling of cached latents.
        Add(suffix, {
                        .base_name = "mla_pool_w",
                        .source_names = {"self_attn.pool_gate.weight"},
                        .axes = {0},
                        .shape = {kv_a_rows},
                        .min_size = Type::kF32,
                    });
        Add(suffix, {
                        .base_name = "mla_pool_b",
                        .source_names = {"self_attn.pool_gate.bias"},
                        .axes = {0},
                        .shape = {layer_config.kv_compression_rate},
                        .min_size = Type::kF32,
                    });
      }
    }
    // Lightning indexer (CSA only).
    if (layer_config.indexer_heads > 0) {
      const size_t indexer_dim =
          layer_config.indexer_heads * layer_config.indexer_head_dim;
      if (layer_config.IsV4MLA()) {
        // V4: indexer queries come from the normed q-LoRA latent (qr), keys
        // from the indexer's own compressor, and per-head weights from x.
        Add(suffix, {
                        .base_name = "idx_q_w",
                        .source_names = {"attn.indexer.wq_b.weight"},
                        .axes = {0, 1},
                        .shape = {indexer_dim, layer_config.q_lora_rank},
                    });
        Add(suffix, {
                        .base_name = "idx_w_proj",
                        .source_names = {"attn.indexer.weights_proj.weight"},
                        .axes = {0, 1},
                        .shape = {layer_config.indexer_heads, config.model_dim},
                        .min_size = Type::kBF16,
                    });
        const size_t coff = layer_config.CompressorCoff();
        const size_t idx_dim = layer_config.indexer_head_dim;
        Add(suffix, {
                        .base_name = "idxc_wkv",
                        .source_names = {"attn.indexer.compressor.wkv.weight"},
                        .axes = {0, 1},
                        .shape = {coff * idx_dim, config.model_dim},
                    });
        Add(suffix,
            {
                .base_name = "idxc_wgate",
                .source_names = {"attn.indexer.compressor.wgate.weight"},
                .axes = {0, 1},
                .shape = {coff * idx_dim, config.model_dim},
            });
        Add(suffix,
            {
                .base_name = "idxc_ape",
                .source_names = {"attn.indexer.compressor.ape"},
                .axes = {0, 1},
                .shape = {layer_config.kv_compression_rate, coff * idx_dim},
                .min_size = Type::kF32,
            });
        Add(suffix, {
                        .base_name = "idxc_ns",
                        .source_names = {"attn.indexer.compressor.norm.weight"},
                        .axes = {0},
                        .shape = {idx_dim},
                        .min_size = Type::kBF16,
                    });
      } else {
        Add(suffix, {
                        .base_name = "idx_q_w",
                        .source_names = {"self_attn.indexer.wq.weight"},
                        .axes = {0, 1},
                        .shape = {indexer_dim, config.model_dim},
                    });
        Add(suffix, {
                        .base_name = "idx_k_w",
                        .source_names = {"self_attn.indexer.wk.weight"},
                        .axes = {0, 1},
                        .shape = {indexer_dim, layer_config.kv_lora_rank},
                    });
      }
    }
    // Aux-loss-free routing bias (selection only, not combine weights).
    if (layer_config.IsMoE() && layer_config.use_routing_bias) {
      Add(suffix, {
                      .base_name = "moe_r_bias",
                      .source_names = {"ffn.gate.bias"},
                      .axes = {0},
                      .shape = {layer_config.NumExperts()},
                      .min_size = Type::kF32,
                  });
    }
    // Hash routing (first n_hash_layers): per-token-id expert indices,
    // stored as f32 (values are small integers, exactly representable).
    if (layer_config.IsMoE() && layer_config.hash_routing) {
      Add(suffix, {
                      .base_name = "hash_tid2eid",
                      .source_names = {"ffn.gate.tid2eid"},
                      .axes = {0, 1},
                      .shape = {config.vocab_size,
                                layer_config.NumExpertsPerDatapoint()},
                      .min_size = Type::kF32,
                  });
    }
  }

  // Manifold-constrained hyper-connections (DeepSeek V4): per-token dynamic
  // read/write/mixing weights derived from `hc_fn @ flatten(streams)` via a
  // Sinkhorn-Knopp projection at inference time (see `hc_split_sinkhorn` in
  // the reference implementation).
  if (config.hc_mult > 1) {
    const size_t mix_hc = (2 + config.hc_mult) * config.hc_mult;
    const size_t hc_dim = config.hc_mult * config.model_dim;
    for (const char* block : {"att", "ffw"}) {
      const std::string b(block);
      Add(suffix, {
                      .base_name = "hc_" + b + "_fn",
                      .source_names = {"hc_" + b + "n_fn"},
                      .axes = {0, 1},
                      .shape = {mix_hc, hc_dim},
                      .min_size = Type::kF32,
                  });
      Add(suffix, {
                      .base_name = "hc_" + b + "_base",
                      .source_names = {"hc_" + b + "n_base"},
                      .axes = {0},
                      .shape = {mix_hc},
                      .min_size = Type::kF32,
                  });
      Add(suffix, {
                      .base_name = "hc_" + b + "_scale",
                      .source_names = {"hc_" + b + "n_scale"},
                      .axes = {0},
                      .shape = {3},
                      .min_size = Type::kF32,
                  });
    }
  }
}

}  // namespace gcpp
