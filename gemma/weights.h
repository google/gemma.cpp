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

#include <array>
#include <complex>
#include <cstdio>
#include <string>
#include <unordered_set>
#include <vector>

#include "compression/compress.h"
#include "compression/shared.h"
#include "gemma/common.h"
#include "gemma/configs.h"
#include "util/allocator.h"
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

template <class TConfig>
struct CompressedLayer {
  // Large data is constructed separately.
  CompressedLayer()
      : attn_vec_einsum_w("att_ein", kModelDim, kHeads * kQKVDim),
        qkv_einsum_w("qkv_ein", (kHeads + 2 * kKVHeads) * kQKVDim, kModelDim),
        qkv_einsum_w1("qkv1_w", kHeads * kQKVDim, kModelDim),
        qkv_einsum_w2("qkv2_w", 2 * kKVHeads * kQKVDim, kModelDim),
        attention_output_biases("attn_ob", 1, kAOBiasDim),
        griffin({.linear_x_w = {"gr_lin_x_w", kGriffinDim, kGriffinDim},
                 .linear_x_biases = {"gr_lin_x_b", 1, kGriffinDim},
                 .linear_y_w = {"gr_lin_y_w", kGriffinDim, kGriffinDim},
                 .linear_y_biases = {"gr_lin_y_b", 1, kGriffinDim},
                 .linear_out_w = {"gr_lin_out_w", kGriffinDim, kGriffinDim},
                 .linear_out_biases = {"gr_lin_out_b", 1, kGriffinDim},
                 .conv_w = {"gr_conv_w", kConv1dWidth, kGriffinDim},
                 .conv_biases = {"gr_conv_b", 1, kGriffinDim},
                 .gate_w = {"gr_gate_w", 2 * kGriffinDim, kGriffinDim / kHeads},
                 .gate_biases = {"gr_gate_b", 1, kGriffinDim * 2},
                 .a = {"gr_a", 1, kGriffinDim}}),
        // MultiHeadDotProductAttention.
        vit({.attn_out_w = {"attn_out_w", kHeads * kQKVDim, kModelDim},
             .attn_out_b = {"attn_out_b", 1, kModelDim},
             .qkv_einsum_w = {"qkv_ein_w", (kHeads + 2 * kKVHeads) * kQKVDim,
                              kModelDim},
             .qkv_einsum_b = {"qkv_ein_b", (kHeads + 2 * kKVHeads), kQKVDim},
             .linear_0_w = {"linear_0_w", kModelDim, kFFHiddenDim},
             .linear_0_b = {"linear_0_b", 1, kFFHiddenDim},
             .linear_1_w = {"linear_1_w", kFFHiddenDim, kModelDim},
             .linear_1_b = {"linear_1_b", 1, kModelDim},
             .layer_norm_0_bias = {"ln_0_bias", 1, kModelDim},
             .layer_norm_0_scale = {"ln_0_scale", 1, kModelDim},
             .layer_norm_1_bias = {"ln_1_bias", 1, kModelDim},
             .layer_norm_1_scale = {"ln_1_scale", 1, kModelDim}}),
        gating_einsum_w("gating_ein", 2 * kFFHiddenDim, kModelDim),
        gating_einsum_w1("gating1_w", kFFHiddenDim, kModelDim),
        gating_einsum_w2("gating2_w", kFFHiddenDim, kModelDim),
        linear_w("linear_w", kModelDim, kFFHiddenDim),
        pre_attention_norm_scale("pre_att_ns", 1, kModelDim),
        pre_ffw_norm_scale("pre_ff_ns", 1, kModelDim),
        post_attention_norm_scale(
            "post_att_ns", 1, kPostNorm == PostNormType::Scale ? kModelDim : 0),
        post_ffw_norm_scale("post_ff_ns", 1,
                            kPostNorm == PostNormType::Scale ? kModelDim : 0),
        ffw_gating_biases("ffw_gat_b", 1, kFFBiases ? 2 * kFFHiddenDim : 0),
        ffw_output_biases("ffw_out_b", 1, kFFBiases ? kModelDim : 0),
        att_weights("att_w", kModelDim, kHeads * kQKVDim)
  {}
  ~CompressedLayer() = default;

  using Weight = typename TConfig::Weight;
  // If weights are f32, also f32; otherwise at least bf16. Useful for ops that
  // do not yet support smaller compressed types, or require at least bf16. When
  // weights are f32, we also want such tensors to be f32.
  // If weights are complex, this is also complex.
  using WeightF32OrBF16 =
      hwy::If<hwy::IsSame<Weight, std::complex<double>>(), std::complex<double>,
              hwy::If<hwy::IsSame<Weight, double>(), double,
                      hwy::If<IsF32<Weight>(), float, BF16>>>;

  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kKVHeads = TConfig::kKVHeads;
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;
  static constexpr size_t kAttVecEinsumWSize = kHeads * kQKVDim * kModelDim;
  static constexpr size_t kQKVEinsumWSize =
      (kHeads + 2 * kKVHeads) * kQKVDim * kModelDim;
  static constexpr size_t kQKVEinsumBSize = (kHeads + 2 * kKVHeads) * kQKVDim;
  // 2x for (gelu gating vector, gated vector)
  static constexpr size_t kGatingEinsumWSize = 2 * kFFHiddenDim * kModelDim;
  static constexpr size_t kConv1dWidth = TConfig::kConv1dWidth;
  static constexpr bool kFFBiases = TConfig::kFFBiases;
  static constexpr PostNormType kPostNorm = TConfig::kPostNorm;
  static constexpr size_t kAOBiasDim =
      TConfig::kSoftmaxAttnOutputBiases ? kModelDim : 0;
  static constexpr size_t kGriffinDim =
      TConfig::kGriffinLayers > 0 ? kModelDim : 0;

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

  // Initializes att_weights from attn_vec_einsum_w, hence this must be called
  // after loading weights via ForEachTensor.
  // TODO: update compression/convert_weights to bake this in.
  void Reshape(MatStorage& storage) {
    if (attn_vec_einsum_w.data() == nullptr) return;

    constexpr size_t kModelDim = TConfig::kModelDim;
    constexpr size_t kHeads = TConfig::kHeads;
    constexpr size_t kQKVDim = TConfig::kQKVDim;

    // Would have to implement a CompressTraits::Copy for NUQ.
    static_assert(!hwy::IsSame<Weight, NuqStream>());

    // Reshape [kHeads, kModelDim, kQKVDim] to [kModelDim, kHeads * kQKVDim].
    storage.Allocate();
    att_weights.SetPtr(storage);
    for (size_t m = 0; m < kModelDim; ++m) {
      Weight* HWY_RESTRICT out_row = att_weights.data() + m * kHeads * kQKVDim;
      for (size_t h = 0; h < kHeads; ++h) {
        hwy::CopyBytes(
            attn_vec_einsum_w.data() + h * kModelDim * kQKVDim + m * kQKVDim,
            out_row + h * kQKVDim, kQKVDim * sizeof(Weight));
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
  static void ForEachTensor(const std::vector<CompressedLayer<TConfig>*>& ptrs,
                            int layer_idx, ForEachType fet, Func func,
                            char sep = ' ', int sep_index = -1) {
    MatPtr* tensors[ptrs.size()];
    auto type = TConfig::kLayerConfig[layer_idx];
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

    if (TConfig::kPostNorm == PostNormType::Scale) {
      GEMMA_CALL_FUNC(post_attention_norm_scale);
      GEMMA_CALL_FUNC(post_ffw_norm_scale);
    }

    if (TConfig::kFFBiases) {
      GEMMA_CALL_FUNC(ffw_gating_biases);
      GEMMA_CALL_FUNC(ffw_output_biases);
    }

    if (TConfig::kSoftmaxAttnOutputBiases &&
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
  void Allocate() {
    layer_storage.clear();
    ForEachTensor({this}, /*layer_idx=*/0, ForEachType::kInitNoToc,
                  [this](const char* name, hwy::Span<MatPtr*> tensors) {
                    this->layer_storage.emplace_back(*tensors[0]);
                    layer_storage.back().Allocate();
                    tensors[0]->SetPtr(layer_storage.back());
                  });
  }

  // Storage for all the matrices and vectors. Only used for a stand-alone
  // layer. For a model, the CompressedWeights::model_storage is used instead.
  std::vector<MatStorage> layer_storage;
};

template <class TConfig>
struct CompressedWeights {
  explicit CompressedWeights(hwy::ThreadPool& pool)
      : embedder_input_embedding("c_embedding", TConfig::kVocabSize,
                                 TConfig::kModelDim),
        final_norm_scale("c_final_norm", 1, TConfig::kModelDim),
        vit_encoder_norm_bias("c_vit_encoder_norm_bias", 1,
                              TConfig::VitConfig::kModelDim),
        vit_encoder_norm_scale("c_vit_encoder_norm_scale", 1,
                               TConfig::VitConfig::kModelDim),
        vit_img_embedding_bias("c_vit_img_embedding_bias", 1,
                               TConfig::VitConfig::kModelDim),
        vit_img_embedding_kernel("c_vit_img_embedding_kernel", 14 * 14 * 3,
                                 TConfig::VitConfig::kModelDim),
        vit_img_pos_embedding("c_vit_img_pos_embedding", 256,
                              TConfig::VitConfig::kModelDim),
        vit_img_head_bias("c_vit_img_head_bias", 1, TConfig::kModelDim),
        vit_img_head_kernel("c_vit_img_head_kernel",
                            TConfig::VitConfig::kModelDim, TConfig::kModelDim),
        scale_names({"att_ein", "qkv_ein", "gr_lin_x_w", "gr_lin_y_w",
                     "gr_lin_out_w", "gr_gate_w", "gating_ein", "linear_w"}) {}

  ~CompressedWeights() = default;

  using Weight = typename TConfig::Weight;
  using WeightF32OrBF16 = typename CompressedLayer<TConfig>::WeightF32OrBF16;
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

  // Storage for all the matrices and vectors.
  std::vector<MatStorage> model_storage;
  std::unordered_set<std::string> scale_names;

  CompressedLayer<TConfig> c_layers[TConfig::kLayers];
  CompressedLayer<typename TConfig::VitConfig>
      vit_layers[TConfig::VitConfig::kLayers];

  // Called by weights.cc after ForEachTensor.
  void Reshape(hwy::ThreadPool& pool) {
    size_t storage_index = model_storage.size();
    for (size_t layer = 0; layer < TConfig::kLayers; ++layer) {
      model_storage.emplace_back(GetLayer(layer)->att_weights);
    }
    pool.Run(0, TConfig::kLayers,
             [this, storage_index](uint64_t layer, size_t /*thread*/) {
               GetLayer(layer)->Reshape(model_storage[storage_index + layer]);
             });
  }

  void ZeroInit() {
    embedder_input_embedding.ZeroInit();
    final_norm_scale.ZeroInit();
    for (int i = 0; i < TConfig::kLayers; ++i) {
      c_layers[i].ZeroInit(i);
    }
  }

  const CompressedLayer<TConfig>* GetLayer(size_t layer) const {
    return &c_layers[layer];
  }
  CompressedLayer<TConfig>* GetLayer(size_t layer) { return &c_layers[layer]; }
  const CompressedLayer<typename TConfig::VitConfig>* GetVitLayer(
      size_t layer) const {
    return &vit_layers[layer];
  }
  CompressedLayer<typename TConfig::VitConfig>* GetVitLayer(size_t layer) {
    return &vit_layers[layer];
  }

  // Copies the data from other to *this.
  void CopyFrom(const CompressedWeights<TConfig>& other) {
    ForEachTensor({this, const_cast<CompressedWeights<TConfig>*>(&other)},
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
    HWY_ASSERT(scale_pos == TConfig::kNumTensorScales);
  }

  template <class Func>
  static void ForEachTensor(
      const std::vector<CompressedWeights<TConfig>*>& ptrs, ForEachType fet,
      Func func) {
    std::vector<CompressedLayer<TConfig>*> layers(ptrs.size());
    std::vector<CompressedLayer<typename TConfig::VitConfig>*> vit_layers(
        ptrs.size());
    MatPtr* tensors[ptrs.size()];
    // Variables used by GEMMA_CALL_FUNC.
    int layer_idx = -1;
    char sep = ' ';
    int sep_index = -1;
    GEMMA_CALL_FUNC(embedder_input_embedding);
    GEMMA_CALL_FUNC(final_norm_scale);
    if constexpr (TConfig::VitConfig::kLayers > 0) {
      // Vit parts.
      GEMMA_CALL_FUNC(vit_encoder_norm_bias);
      GEMMA_CALL_FUNC(vit_encoder_norm_scale);
      GEMMA_CALL_FUNC(vit_img_embedding_bias);
      GEMMA_CALL_FUNC(vit_img_embedding_kernel);
      GEMMA_CALL_FUNC(vit_img_pos_embedding);
      GEMMA_CALL_FUNC(vit_img_head_bias);
      GEMMA_CALL_FUNC(vit_img_head_kernel);
    }

    for (int layer_idx = 0; layer_idx < TConfig::kLayers; ++layer_idx) {
      for (int i = 0; i < ptrs.size(); ++i) {
        layers[i] = ptrs[i]->GetLayer(layer_idx);
      }
      CompressedLayer<TConfig>::ForEachTensor(layers, layer_idx, fet, func);
    }

    // Vit layers. Not supported for compress_weights.
    if constexpr (TConfig::VitConfig::kLayers > 0) {
      for (int layer_idx = 0; layer_idx < TConfig::VitConfig::kLayers;
           ++layer_idx) {
        auto type = TConfig::VitConfig::kLayerConfig[layer_idx];
        HWY_ASSERT(type == LayerAttentionType::kVit);
        for (int i = 0; i < ptrs.size(); ++i) {
          vit_layers[i] = ptrs[i]->GetVitLayer(layer_idx);
        }
        CompressedLayer<typename TConfig::VitConfig>::ForEachTensor(
            vit_layers, layer_idx, fet, func);
      }
    }
  }
};
#undef GEMMA_CALL_FUNC

// Pair of configs for the compressed and uncompressed weights.
template <class CConfig, class UCConfig>
struct ConfigPair {
  using uc = UCConfig;
  using c = CConfig;
};

// ----------------------------------------------------------------------------
// Interface

template <typename TConfig>
struct AllocateCompressedWeights {
  ByteStorageT operator()(hwy::ThreadPool& pool) const {
    using TWeights = CompressedWeights<TConfig>;
    ByteStorageT weights_u8 = AllocateSizeof<TWeights>();
    TWeights* weights = reinterpret_cast<TWeights*>(weights_u8.get());
    new (weights) TWeights(pool);
    std::vector<MatPtr*> model_toc;
    auto& model_storage = weights->model_storage;
    TWeights::ForEachTensor(
        {weights}, ForEachType::kInitNoToc,
        [&model_toc, &model_storage](const char*, hwy::Span<MatPtr*> tensors) {
          model_toc.push_back(tensors[0]);
          model_storage.emplace_back(*tensors[0]);
        });
    // Allocate in parallel using the pool.
    pool.Run(0, model_storage.size(),
             [&model_toc, &model_storage](uint64_t task, size_t /*thread*/) {
               model_storage[task].Allocate();
               model_toc[task]->SetPtr(model_storage[task]);
             });
    return weights_u8;
  }
};

template <typename TConfig>
struct ZeroInitCompressedWeights {
  void operator()(ByteStorageT& weights_u8, hwy::ThreadPool& pool) const {
    CompressedWeights<TConfig>& weights =
        *reinterpret_cast<CompressedWeights<TConfig>*>(weights_u8.get());
    weights.ZeroInit();
  }
};

template <typename TConfig>
struct ReshapeCompressedWeights {
  void operator()(ByteStorageT& weights_u8, hwy::ThreadPool& pool) const {
    CompressedWeights<TConfig>& weights =
        *reinterpret_cast<CompressedWeights<TConfig>*>(weights_u8.get());
    weights.Reshape(pool);
  }
};

// TODO: also add RandInitCompressedWeights

ByteStorageT LoadCompressedWeights(const Path& weights, Model model_type,
                                   Type weight_type, hwy::ThreadPool& pool);

void LogWeightStats(Model model, Type weight_type, const ByteStorageT& weights);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_WEIGHTS_H_
