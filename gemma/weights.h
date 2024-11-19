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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_WEIGHTS_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_WEIGHTS_H_

#include <stddef.h>

#include <complex>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "compression/compress.h"
#include "compression/shared.h"
#include "gemma/common.h"
#include "gemma/configs.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

// Different tensors need to appear in a ForEachTensor, according to what is
// happening.
enum class ForEachType {
  // Under normal circumstances, when not initializing or loading, we can
  // include all tensors and ignore the null ones.
  kIgnoreNulls,
  // If there is a table of contents, we can include all tensors.
  kLoadWithToc,
  // There is no table of contents, so we have to be careful to only include
  // tensors that are actually present.
  kLoadNoToc,
  // We need to initialize all tensors needed when there is no table of
  // contents. This differs from kLoadNoToc in that we need to include any
  // tensor that is allocated but not loaded directly from file.
  kInitNoToc,
};

template <class Weight>
struct LayerWeightsPtrs {
  // Large data is constructed separately.
  explicit LayerWeightsPtrs(const LayerConfig& config)
      : attn_vec_einsum_w("att_ein", config.model_dim,
                          config.heads * config.qkv_dim),
        qkv_einsum_w("qkv_ein",
                     (config.heads + 2 * config.kv_heads) * config.qkv_dim,
                     config.model_dim),
        qkv_einsum_w1("qkv1_w", config.heads * config.qkv_dim,
                      config.model_dim),
        qkv_einsum_w2("qkv2_w", 2 * config.kv_heads * config.qkv_dim,
                      config.model_dim),
        attention_output_biases(
            "attn_ob", 1,
            config.softmax_attn_output_biases ? config.model_dim : 0),
        griffin(
            {.linear_x_w = {"gr_lin_x_w", config.griffin_dim,
                            config.griffin_dim},
             .linear_x_biases = {"gr_lin_x_b", 1, config.griffin_dim},
             .linear_y_w = {"gr_lin_y_w", config.griffin_dim,
                            config.griffin_dim},
             .linear_y_biases = {"gr_lin_y_b", 1, config.griffin_dim},
             .linear_out_w = {"gr_lin_out_w", config.griffin_dim,
                              config.griffin_dim},
             .linear_out_biases = {"gr_lin_out_b", 1, config.griffin_dim},
             .conv_w = {"gr_conv_w", config.conv1d_width, config.griffin_dim},
             .conv_biases = {"gr_conv_b", 1, config.griffin_dim},
             .gate_w = {"gr_gate_w", 2 * config.griffin_dim,
                        config.griffin_dim / config.heads},
             .gate_biases = {"gr_gate_b", 1, config.griffin_dim * 2},
             .a = {"gr_a", 1, config.griffin_dim}}),
        // MultiHeadDotProductAttention.
        vit({.attn_out_w = {"attn_out_w", config.heads * config.qkv_dim,
                            config.model_dim},
             .attn_out_b = {"attn_out_b", 1, config.model_dim},
             .qkv_einsum_w = {"qkv_ein_w",
                              (config.heads + 2 * config.kv_heads) *
                                  config.qkv_dim,
                              config.model_dim},
             .qkv_einsum_b = {"qkv_ein_b", (config.heads + 2 * config.kv_heads),
                              config.qkv_dim},
             .linear_0_w = {"linear_0_w", config.ff_hidden_dim,
                            config.model_dim},
             .linear_0_b = {"linear_0_b", 1, config.ff_hidden_dim},
             .linear_1_w = {"linear_1_w", config.model_dim,
                            config.ff_hidden_dim},
             .linear_1_b = {"linear_1_b", 1, config.model_dim},
             .layer_norm_0_bias = {"ln_0_bias", 1, config.model_dim},
             .layer_norm_0_scale = {"ln_0_scale", 1, config.model_dim},
             .layer_norm_1_bias = {"ln_1_bias", 1, config.model_dim},
             .layer_norm_1_scale = {"ln_1_scale", 1, config.model_dim}}),
        gating_einsum_w("gating_ein", 2 * config.ff_hidden_dim,
                        config.model_dim),
        gating_einsum_w1("gating1_w", config.ff_hidden_dim, config.model_dim),
        gating_einsum_w2("gating2_w", config.ff_hidden_dim, config.model_dim),
        linear_w("linear_w", config.model_dim, config.ff_hidden_dim),
        pre_attention_norm_scale("pre_att_ns", 1, config.model_dim),
        pre_ffw_norm_scale("pre_ff_ns", 1, config.model_dim),
        post_attention_norm_scale(
            "post_att_ns", 1,
            config.post_norm == PostNormType::Scale ? config.model_dim : 0),
        post_ffw_norm_scale(
            "post_ff_ns", 1,
            config.post_norm == PostNormType::Scale ? config.model_dim : 0),
        ffw_gating_biases("ffw_gat_b", 1,
                          config.ff_biases ? 2 * config.ff_hidden_dim : 0),
        ffw_output_biases("ffw_out_b", 1,
                          config.ff_biases ? config.model_dim : 0),
        att_weights("att_w", config.model_dim, config.heads * config.qkv_dim),
        layer_config(config) {}
  ~LayerWeightsPtrs() = default;

  // If weights are f32, also f32; otherwise at least bf16. Useful for ops that
  // do not yet support smaller compressed types, or require at least bf16. When
  // weights are f32, we also want such tensors to be f32.
  // If weights are complex, this is also complex.
  using WeightF32OrBF16 =
      hwy::If<hwy::IsSame<Weight, std::complex<double>>(), std::complex<double>,
              hwy::If<hwy::IsSame<Weight, double>(), double,
                      hwy::If<IsF32<Weight>(), float, BF16>>>;

  template <class T>
  using ArrayT = MatPtrT<T>;

  ArrayT<Weight> attn_vec_einsum_w;
  // qkv_einsum_w holds 2 different matrices, which may be separated out.
  // On loading, which is used depends on what is in the file.
  // At inference, the one with a non-null ptr is used.
  ArrayT<Weight> qkv_einsum_w;
  ArrayT<Weight> qkv_einsum_w1;
  ArrayT<Weight> qkv_einsum_w2;
  ArrayT<float> attention_output_biases;

  struct {
    ArrayT<Weight> linear_x_w;
    ArrayT<float> linear_x_biases;
    ArrayT<Weight> linear_y_w;
    ArrayT<float> linear_y_biases;
    ArrayT<Weight> linear_out_w;
    ArrayT<float> linear_out_biases;
    ArrayT<float> conv_w;
    ArrayT<float> conv_biases;
    ArrayT<Weight> gate_w;
    ArrayT<float> gate_biases;
    ArrayT<float> a;
  } griffin;

  struct {
    // MultiHeadDotProductAttention.
    ArrayT<WeightF32OrBF16> attn_out_w;
    ArrayT<float> attn_out_b;
    ArrayT<WeightF32OrBF16> qkv_einsum_w;
    ArrayT<float> qkv_einsum_b;
    // MlpBlock.
    ArrayT<WeightF32OrBF16> linear_0_w;
    ArrayT<float> linear_0_b;
    ArrayT<WeightF32OrBF16> linear_1_w;
    ArrayT<float> linear_1_b;
    // LayerNorm.
    ArrayT<WeightF32OrBF16> layer_norm_0_bias;
    ArrayT<WeightF32OrBF16> layer_norm_0_scale;
    ArrayT<WeightF32OrBF16> layer_norm_1_bias;
    ArrayT<WeightF32OrBF16> layer_norm_1_scale;
  } vit;

  // gating_einsum_w holds 2 different matrices, which may be separated out.
  // On loading, which is used depends on what is in the file.
  // At inference, the one with a non-null ptr is used.
  ArrayT<Weight> gating_einsum_w;
  ArrayT<Weight> gating_einsum_w1;
  ArrayT<Weight> gating_einsum_w2;
  ArrayT<Weight> linear_w;
  // We don't yet have an RMSNorm that accepts all Weight.
  ArrayT<WeightF32OrBF16> pre_attention_norm_scale;
  ArrayT<WeightF32OrBF16> pre_ffw_norm_scale;
  ArrayT<WeightF32OrBF16> post_attention_norm_scale;
  ArrayT<WeightF32OrBF16> post_ffw_norm_scale;

  ArrayT<float> ffw_gating_biases;
  ArrayT<float> ffw_output_biases;

  // Reshaped attention; not loaded from disk via ForEachTensor.
  ArrayT<Weight> att_weights;

  const LayerConfig& layer_config;

  // Initializes att_weights from attn_vec_einsum_w, hence this must be called
  // after loading weights via ForEachTensor.
  // TODO: update compression/convert_weights to bake this in.
  void Reshape(MatStorage* storage) {
    if (attn_vec_einsum_w.data() == nullptr) return;

    const size_t model_dim = layer_config.model_dim;
    const size_t heads = layer_config.heads;
    const size_t qkv_dim = layer_config.qkv_dim;

    // TODO: implement a CompressTraits::Copy for NUQ.
    // static_assert(!hwy::IsSame<Weight, NuqStream>());

    // Reshape [kHeads, kModelDim, kQKVDim] to [kModelDim, kHeads * kQKVDim].
    if (storage != nullptr) {
      storage->Allocate();
      att_weights.SetPtr(*storage);
    }
    for (size_t m = 0; m < model_dim; ++m) {
      Weight* HWY_RESTRICT out_row = att_weights.data() + m * heads * qkv_dim;
      for (size_t h = 0; h < heads; ++h) {
        hwy::CopyBytes(
            attn_vec_einsum_w.data() + h * model_dim * qkv_dim + m * qkv_dim,
            out_row + h * qkv_dim, qkv_dim * sizeof(Weight));
      }
    }
    att_weights.set_scale(attn_vec_einsum_w.scale());
  }

// Used by ForEachTensor for per-layer tensors.
#define GEMMA_CALL_FUNC(member)                                             \
  {                                                                         \
    for (int i = 0; i < ptrs.size(); ++i) {                                 \
      tensors[i] = &ptrs[i]->member;                                        \
    }                                                                       \
    if (tensors[0]->Ptr() != nullptr || fet != ForEachType::kIgnoreNulls) { \
      func(ptrs[0]->member.CacheName(layer_idx, sep, sep_index).c_str(),    \
           hwy::Span<MatPtr*>(tensors, ptrs.size()));                       \
    }                                                                       \
  }

  template <class Func>
  static void ForEachTensor(const std::vector<LayerWeightsPtrs<Weight>*>& ptrs,
                            int layer_idx, ForEachType fet, Func func,
                            char sep = ' ', int sep_index = -1) {
    MatPtr* tensors[ptrs.size()];
    auto type = ptrs[0]->layer_config.type;
    if (type == LayerAttentionType::kVit) {
      // MHA.
      GEMMA_CALL_FUNC(vit.attn_out_w);
      GEMMA_CALL_FUNC(vit.attn_out_b);
      GEMMA_CALL_FUNC(vit.qkv_einsum_w);
      GEMMA_CALL_FUNC(vit.qkv_einsum_b);
      // MlpBlock.
      GEMMA_CALL_FUNC(vit.linear_0_w);
      GEMMA_CALL_FUNC(vit.linear_0_b);
      GEMMA_CALL_FUNC(vit.linear_1_w);
      GEMMA_CALL_FUNC(vit.linear_1_b);
      // LayerNorm.
      GEMMA_CALL_FUNC(vit.layer_norm_0_bias);
      GEMMA_CALL_FUNC(vit.layer_norm_0_scale);
      GEMMA_CALL_FUNC(vit.layer_norm_1_bias);
      GEMMA_CALL_FUNC(vit.layer_norm_1_scale);
      return;
    }
    if (type == LayerAttentionType::kGemma) {
      if (fet != ForEachType::kLoadNoToc) {
        GEMMA_CALL_FUNC(att_weights);
      }
      if (fet == ForEachType::kInitNoToc || fet == ForEachType::kLoadNoToc ||
          fet == ForEachType::kIgnoreNulls) {
        GEMMA_CALL_FUNC(attn_vec_einsum_w);
      }
      GEMMA_CALL_FUNC(qkv_einsum_w);
      if (fet == ForEachType::kIgnoreNulls ||
          fet == ForEachType::kLoadWithToc) {
        // The unwanted ones will be null or not in the toc.
        GEMMA_CALL_FUNC(qkv_einsum_w1);
        GEMMA_CALL_FUNC(qkv_einsum_w2);
      }
    } else {
      GEMMA_CALL_FUNC(griffin.linear_x_w);
      GEMMA_CALL_FUNC(griffin.linear_x_biases);
      GEMMA_CALL_FUNC(griffin.linear_y_w);
      GEMMA_CALL_FUNC(griffin.linear_y_biases);
      GEMMA_CALL_FUNC(griffin.linear_out_w);
      GEMMA_CALL_FUNC(griffin.linear_out_biases);
      GEMMA_CALL_FUNC(griffin.conv_w);
      GEMMA_CALL_FUNC(griffin.conv_biases);
      GEMMA_CALL_FUNC(griffin.gate_w);
      GEMMA_CALL_FUNC(griffin.gate_biases);
      GEMMA_CALL_FUNC(griffin.a);
    }
    GEMMA_CALL_FUNC(gating_einsum_w);
    if (fet == ForEachType::kIgnoreNulls || fet == ForEachType::kLoadWithToc) {
      // The unwanted ones will be null or not in the toc.
      GEMMA_CALL_FUNC(gating_einsum_w1);
      GEMMA_CALL_FUNC(gating_einsum_w2);
    }
    GEMMA_CALL_FUNC(linear_w);
    GEMMA_CALL_FUNC(pre_attention_norm_scale);
    GEMMA_CALL_FUNC(pre_ffw_norm_scale);

    if (ptrs[0]->layer_config.post_norm == PostNormType::Scale) {
      GEMMA_CALL_FUNC(post_attention_norm_scale);
      GEMMA_CALL_FUNC(post_ffw_norm_scale);
    }

    if (ptrs[0]->layer_config.ff_biases) {
      GEMMA_CALL_FUNC(ffw_gating_biases);
      GEMMA_CALL_FUNC(ffw_output_biases);
    }

    if (ptrs[0]->layer_config.softmax_attn_output_biases &&
        type == LayerAttentionType::kGemma) {
      GEMMA_CALL_FUNC(attention_output_biases);
    }
  }

  // Sets all the tensors in the layer to zero. Memory must have been allocated.
  void ZeroInit(int layer_idx) {
    ForEachTensor({this}, layer_idx, ForEachType::kIgnoreNulls,
                  [](const char*, hwy::Span<MatPtr*> tensors) {
                    tensors[0]->ZeroInit();
                  });
  }

  // Allocates memory for all the tensors in the layer.
  // Note that this is slow and only used for a stand-alone layer.
  void Allocate(std::vector<MatStorage>& layer_storage) {
    ForEachTensor(
        {this}, /*layer_idx=*/0, ForEachType::kInitNoToc,
        [&layer_storage](const char* name, hwy::Span<MatPtr*> tensors) {
          layer_storage.emplace_back(*tensors[0]);
          layer_storage.back().Allocate();
          tensors[0]->SetPtr(layer_storage.back());
        });
  }
};

template <class Weight>
struct ModelWeightsPtrs {
  ModelWeightsPtrs(const ModelConfig& config, hwy::ThreadPool& pool)
      : embedder_input_embedding("c_embedding", config.vocab_size,
                                 config.model_dim),
        final_norm_scale("c_final_norm", 1, config.model_dim),
        vit_encoder_norm_bias("enc_norm_bias", 1, config.vit_model_dim),
        vit_encoder_norm_scale("enc_norm_scale", 1, config.vit_model_dim),
        vit_img_embedding_bias("img_emb_bias", 1, config.vit_model_dim),
        vit_img_embedding_kernel("img_emb_kernel",
                                 config.patch_width * config.patch_width * 3,
                                 config.vit_model_dim),
        vit_img_pos_embedding("img_pos_emb", 256, config.vit_model_dim),
        vit_img_head_bias("img_head_bias", 1, config.model_dim),
        vit_img_head_kernel("img_head_kernel", config.model_dim,
                            config.vit_model_dim),
        scale_names(config.scale_names),
        weights_config(config) {
    c_layers.reserve(config.layer_configs.size());
    for (const auto& layer_config : config.layer_configs) {
      c_layers.push_back(LayerWeightsPtrs<Weight>(layer_config));
    }
    for (const auto& layer_config : config.vit_layer_configs) {
      vit_layers.push_back(LayerWeightsPtrs<Weight>(layer_config));
    }
  }

  ~ModelWeightsPtrs() = default;
  using WeightF32OrBF16 = typename LayerWeightsPtrs<Weight>::WeightF32OrBF16;
  using WeightF32OrInputT = hwy::If<hwy::IsSame<WeightF32OrBF16, BF16>(),
                                    EmbedderInputT, WeightF32OrBF16>;

  MatPtrT<WeightF32OrInputT> embedder_input_embedding;
  MatPtrT<WeightF32OrBF16> final_norm_scale;

  // Vit parts.
  MatPtrT<WeightF32OrBF16> vit_encoder_norm_bias;
  MatPtrT<WeightF32OrBF16> vit_encoder_norm_scale;
  MatPtrT<float> vit_img_embedding_bias;
  MatPtrT<WeightF32OrBF16> vit_img_embedding_kernel;
  MatPtrT<float> vit_img_pos_embedding;
  // The head maps from VitConfig::kModelDim (Vit final layer) to
  // kModelDim (LLM input).
  MatPtrT<float> vit_img_head_bias;
  MatPtrT<WeightF32OrBF16> vit_img_head_kernel;

  std::unordered_set<std::string> scale_names;

  const ModelConfig& weights_config;

  std::vector<LayerWeightsPtrs<Weight>> c_layers;
  std::vector<LayerWeightsPtrs<Weight>> vit_layers;

  // Called by weights.cc after Loading, before att_w has been allocated.
  void AllocAndCopyWithTranspose(hwy::ThreadPool& pool,
                                 std::vector<MatStorage>& model_storage) {
    size_t storage_index = model_storage.size();
    for (auto& layer : c_layers) {
      model_storage.emplace_back(layer.att_weights);
    }
    pool.Run(0, c_layers.size(),
             [this, &model_storage, storage_index](uint64_t layer,
                                                   size_t /*thread*/) {
               GetLayer(layer)->Reshape(&model_storage[storage_index + layer]);
             });
  }
  // For when the storage has already been allocated.
  void CopyWithTranspose(hwy::ThreadPool& pool) {
    pool.Run(0, c_layers.size(), [this](uint64_t layer, size_t /*thread*/) {
      GetLayer(layer)->Reshape(nullptr);
    });
  }

  void ZeroInit() {
    embedder_input_embedding.ZeroInit();
    final_norm_scale.ZeroInit();
    for (size_t i = 0; i < c_layers.size(); ++i) {
      c_layers[i].ZeroInit(i);
    }
  }

  const LayerWeightsPtrs<Weight>* GetLayer(size_t layer) const {
    return &c_layers[layer];
  }
  LayerWeightsPtrs<Weight>* GetLayer(size_t layer) { return &c_layers[layer]; }
  const LayerWeightsPtrs<Weight>* GetVitLayer(size_t layer) const {
    return &vit_layers[layer];
  }
  LayerWeightsPtrs<Weight>* GetVitLayer(size_t layer) {
    return &vit_layers[layer];
  }

  void Allocate(std::vector<MatStorage>& model_storage, hwy::ThreadPool& pool) {
    std::vector<MatPtr*> model_toc;
    ForEachTensor(
        {this}, ForEachType::kInitNoToc,
        [&model_toc, &model_storage](const char*, hwy::Span<MatPtr*> tensors) {
          model_toc.push_back(tensors[0]);
          model_storage.emplace_back(*tensors[0]);
        });
    // Allocate in parallel using the pool.
    pool.Run(0, model_toc.size(),
             [&model_toc, &model_storage](uint64_t task, size_t /*thread*/) {
               // model_storage may have had content before we started.
               size_t idx = task + model_storage.size() - model_toc.size();
               model_storage[idx].Allocate();
               model_toc[task]->SetPtr(model_storage[idx]);
             });
  }

  // Copies the data from other to *this.
  void CopyFrom(const ModelWeightsPtrs<Weight>& other) {
    ForEachTensor({this, const_cast<ModelWeightsPtrs<Weight>*>(&other)},
                  ForEachType::kIgnoreNulls,
                  [](const char*, hwy::Span<MatPtr*> tensors) {
                    hwy::CopyBytes(tensors[1]->Ptr(), tensors[0]->Ptr(),
                                   tensors[1]->SizeBytes());
                  });
  }

  // If scales is empty, computes and returns the scale factors for the tensors,
  // otherwise applies the scale factors to the tensors.
  void GetOrApplyScales(std::vector<float>& scales) {
    int scale_pos = 0;
    ForEachTensor(
        {this}, ForEachType::kIgnoreNulls,
        [&scales, &scale_pos, this](const char*, hwy::Span<MatPtr*> tensors) {
          if (this->scale_names.count(tensors[0]->Name())) {
            if (scale_pos < scales.size()) {
              tensors[0]->set_scale(scales[scale_pos]);
            } else {
              float scale = ScaleWeights(tensors[0]->data<float>(),
                                         tensors[0]->NumElements());
              scales.push_back(scale);
            }
            ++scale_pos;
          }
        });
    HWY_ASSERT(scale_pos == weights_config.num_tensor_scales);
  }

  template <class Func>
  static void ForEachTensor(const std::vector<ModelWeightsPtrs<Weight>*>& ptrs,
                            ForEachType fet, Func func) {
    std::vector<LayerWeightsPtrs<Weight>*> layers(ptrs.size());
    std::vector<LayerWeightsPtrs<Weight>*> vit_layers(ptrs.size());
    MatPtr* tensors[ptrs.size()];
    // Variables used by GEMMA_CALL_FUNC.
    int layer_idx = -1;
    char sep = ' ';
    int sep_index = -1;
    GEMMA_CALL_FUNC(embedder_input_embedding);
    GEMMA_CALL_FUNC(final_norm_scale);
    if (ptrs[0]->weights_config.vit_layer_configs.size() > 0) {
      // Vit parts.
      GEMMA_CALL_FUNC(vit_encoder_norm_bias);
      GEMMA_CALL_FUNC(vit_encoder_norm_scale);
      GEMMA_CALL_FUNC(vit_img_embedding_bias);
      GEMMA_CALL_FUNC(vit_img_embedding_kernel);
      GEMMA_CALL_FUNC(vit_img_pos_embedding);
      GEMMA_CALL_FUNC(vit_img_head_bias);
      GEMMA_CALL_FUNC(vit_img_head_kernel);
    }

    for (int layer_idx = 0; layer_idx < ptrs[0]->c_layers.size(); ++layer_idx) {
      for (int i = 0; i < ptrs.size(); ++i) {
        layers[i] = ptrs[i]->GetLayer(layer_idx);
      }
      LayerWeightsPtrs<Weight>::ForEachTensor(layers, layer_idx, fet, func);
    }

    // Vit layers. Not supported for compress_weights.
    if (ptrs[0]->weights_config.vit_layer_configs.size() > 0) {
      for (int layer_idx = 0; layer_idx < ptrs[0]->vit_layers.size();
           ++layer_idx) {
        auto type = ptrs[0]->vit_layers[layer_idx].layer_config.type;
        HWY_ASSERT(type == LayerAttentionType::kVit);
        for (int i = 0; i < ptrs.size(); ++i) {
          vit_layers[i] = ptrs[i]->GetVitLayer(layer_idx);
        }
        LayerWeightsPtrs<Weight>::ForEachTensor(vit_layers, layer_idx, fet,
                                                func);
      }
    }
  }
};
#undef GEMMA_CALL_FUNC

// ----------------------------------------------------------------------------
// Interface

class ModelWeightsStorage {
 public:
  ModelWeightsStorage() = default;
  ~ModelWeightsStorage() = default;

  BlobError Load(const Path& weights, Model model_type, Type weight_type,
                 hwy::ThreadPool& pool);
  void Allocate(Model model_type, Type weight_type, hwy::ThreadPool& pool) {
    Allocate(ConfigFromModel(model_type), weight_type, pool);
  }
  void Allocate(const ModelConfig& config, Type weight_type,
                hwy::ThreadPool& pool);
  void RandInit(std::mt19937& gen);
  void ZeroInit();
  void GetOrApplyScales(std::vector<float>& scales);
  void AllocAndCopyWithTranspose(hwy::ThreadPool& pool);
  void CopyWithTranspose(hwy::ThreadPool& pool);
  void LogWeightStats();
  const ModelConfig& Config() const { return config_; }
  ModelConfig& MutableConfig() { return config_; }

  template <typename T>
  ModelWeightsPtrs<T>* GetWeightsOfType() const {
    if constexpr (IsSfpStream<T>()) {
      return sfp_weights_.get();
    } else if constexpr (IsF32<T>()) {
      return float_weights_.get();
    } else if constexpr (IsBF16<T>()) {
      return bf16_weights_.get();
    } else if constexpr (IsNuqStream<T>()) {
      return nuq_weights_.get();
    } else {
      return HWY_ABORT("Unsupported type.");
    }
  }

  template <template <typename T> class FuncT, typename... TArgs>
  decltype(auto) CallForModelWeightT(TArgs&&... args) {
    if (HWY_LIKELY(sfp_weights_))
      return FuncT<SfpStream>()(*sfp_weights_, std::forward<TArgs>(args)...);
    if (bf16_weights_)
      return FuncT<BF16>()(*bf16_weights_, std::forward<TArgs>(args)...);
    if (nuq_weights_)
      return FuncT<NuqStream>()(*nuq_weights_, std::forward<TArgs>(args)...);
    if (float_weights_)
      return FuncT<float>()(*float_weights_, std::forward<TArgs>(args)...);
    return HWY_ABORT("No weights loaded.");
  }

  template <template <typename T> class FuncT, typename... TArgs>
  decltype(auto) CallForModelWeight(TArgs&&... args) {
    if (HWY_LIKELY(sfp_weights_))
      return FuncT<SfpStream>()(*this, std::forward<TArgs>(args)...);
    if (bf16_weights_)
      return FuncT<BF16>()(*this, std::forward<TArgs>(args)...);
    if (nuq_weights_)
      return FuncT<NuqStream>()(*this, std::forward<TArgs>(args)...);
    if (float_weights_)
      return FuncT<float>()(*this, std::forward<TArgs>(args)...);
    return HWY_ABORT("No weights loaded.");
  }

 private:
  void CreateForType(Type weight_type, hwy::ThreadPool& pool);

  ModelConfig config_;
  // To eliminate type templates, we hold a pointer to one of each weight type
  // and dispatch to whichever is non-null.
  std::unique_ptr<ModelWeightsPtrs<float>> float_weights_;
  std::unique_ptr<ModelWeightsPtrs<BF16>> bf16_weights_;
  std::unique_ptr<ModelWeightsPtrs<SfpStream>> sfp_weights_;
  std::unique_ptr<ModelWeightsPtrs<NuqStream>> nuq_weights_;
  // Storage for all the matrices and vectors.
  std::vector<MatStorage> model_storage_;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_WEIGHTS_H_
