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

#include "compression/compress.h"
#include "gemma/common.h"
#include "gemma/configs.h"
#include "util/allocator.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"

namespace gcpp {

template <class TConfig>
struct CompressedLayer {
  // No ctor/dtor, allocated via AllocateAligned.

  using Weight = typename TConfig::Weight;
  // If weights are f32, also f32; otherwise at least bf16. Useful for ops that
  // do not yet support smaller compressed types, or require at least bf16. When
  // weights are f32, we also want such tensors to be f32.
  using WeightF32OrBF16 =
      hwy::If<hwy::IsSame<Weight, float>(), float, hwy::bfloat16_t>;

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

  template <class T, size_t N>
  using ArrayT = CompressedArray<T, N>;

  union {
    struct {
      ArrayT<Weight, kAttVecEinsumWSize> attn_vec_einsum_w;
      ArrayT<Weight, kQKVEinsumWSize> qkv_einsum_w;
      ArrayT<float, kAOBiasDim> attention_output_biases;
    };

    struct {
      ArrayT<Weight, kGriffinDim * kGriffinDim> linear_x_w;
      ArrayT<float, kGriffinDim> linear_x_biases;
      ArrayT<Weight, kGriffinDim * kGriffinDim> linear_y_w;
      ArrayT<float, kGriffinDim> linear_y_biases;
      ArrayT<Weight, kGriffinDim * kGriffinDim> linear_out_w;
      ArrayT<float, kGriffinDim> linear_out_biases;
      ArrayT<float, kConv1dWidth * kGriffinDim> conv_w;
      ArrayT<float, kGriffinDim> conv_biases;
      ArrayT<Weight, kGriffinDim * kGriffinDim / kHeads * 2> gate_w;
      ArrayT<float, kGriffinDim * 2> gate_biases;
      ArrayT<float, kGriffinDim> a;
    } griffin;

    struct {
      // MultiHeadDotProductAttention.
      ArrayT<WeightF32OrBF16, kAttVecEinsumWSize> attn_out_w;
      ArrayT<float, kModelDim> attn_out_b;
      ArrayT<WeightF32OrBF16, kQKVEinsumWSize> qkv_einsum_w;
      ArrayT<float, kQKVEinsumBSize> qkv_einsum_b;
      // MlpBlock.
      ArrayT<WeightF32OrBF16, kModelDim * kFFHiddenDim> linear_0_w;
      ArrayT<float, kFFHiddenDim> linear_0_b;
      ArrayT<WeightF32OrBF16, kFFHiddenDim * kModelDim> linear_1_w;
      ArrayT<float, kModelDim> linear_1_b;
      // LayerNorm.
      ArrayT<WeightF32OrBF16, kModelDim> layer_norm_0_bias;
      ArrayT<WeightF32OrBF16, kModelDim> layer_norm_0_scale;
      ArrayT<WeightF32OrBF16, kModelDim> layer_norm_1_bias;
      ArrayT<WeightF32OrBF16, kModelDim> layer_norm_1_scale;
    } vit;
  };

  ArrayT<Weight, kGatingEinsumWSize> gating_einsum_w;
  ArrayT<Weight, kModelDim * kFFHiddenDim> linear_w;
  // We don't yet have an RMSNorm that accepts all Weight.
  ArrayT<WeightF32OrBF16, kModelDim> pre_attention_norm_scale;
  ArrayT<WeightF32OrBF16, kModelDim> pre_ffw_norm_scale;
  ArrayT<WeightF32OrBF16, kPostNorm == PostNormType::Scale ? kModelDim : 0>
      post_attention_norm_scale;
  ArrayT<WeightF32OrBF16, kPostNorm == PostNormType::Scale ? kModelDim : 0>
      post_ffw_norm_scale;

  ArrayT<float, kFFBiases ? 2 * kFFHiddenDim : 0> ffw_gating_biases;
  ArrayT<float, kFFBiases ? kModelDim : 0> ffw_output_biases;

  // Reshaped attention; not loaded from disk via ForEachTensor.
  ArrayT<Weight, kModelDim * kHeads * kQKVDim> att_weights;

  // Initializes att_weights from attn_vec_einsum_w, hence this must be called
  // after loading weights via ForEachTensor.
  // TODO: update compression/convert_weights to bake this in.
  void Reshape() {
    PROFILER_ZONE("Startup.Reshape");

    constexpr size_t kModelDim = TConfig::kModelDim;
    constexpr size_t kHeads = TConfig::kHeads;
    constexpr size_t kQKVDim = TConfig::kQKVDim;

    // Would have to implement a CompressTraits::Copy for NUQ.
    static_assert(!hwy::IsSame<Weight, NuqStream>());

    // Reshape [kHeads, kModelDim, kQKVDim] to [kModelDim, kHeads * kQKVDim].
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
};

// Array instead of single large allocation for parallel mem init. Split out
// of CompressedWeights so that only these pointers are initialized, not the
// CompressedArray.
template <class TConfig>
struct CompressedLayerPointers {
  explicit CompressedLayerPointers(hwy::ThreadPool& pool) {
    pool.Run(0, TConfig::kLayers, [this](uint64_t task, size_t /*thread*/) {
      this->c_layers[task] = hwy::AllocateAligned<CompressedLayer<TConfig>>(1);
    });
    if constexpr (TConfig::VitConfig::kLayers > 0) {
      pool.Run(0, TConfig::VitConfig::kLayers,
               [this](uint64_t task, size_t /*thread*/) {
                 this->c_vit_layers[task] = hwy::AllocateAligned<
                     CompressedLayer<typename TConfig::VitConfig>>(1);
               });
    }
  }

  using CLayer = CompressedLayer<TConfig>;
  std::array<hwy::AlignedFreeUniquePtr<CLayer[]>, TConfig::kLayers> c_layers;
  using CVitLayer = CompressedLayer<typename TConfig::VitConfig>;
  std::array<hwy::AlignedFreeUniquePtr<CVitLayer[]>,
             TConfig::VitConfig::kLayers>
      c_vit_layers;
};

template <class TConfig, typename = void>
struct CompressedWeights {
  // Must be allocated via AllocateAligned and initialized with placement new.
  void* operator new(size_t, void* addr) { return addr; }
  void* operator new(size_t) = delete;
  void* operator new[](size_t) = delete;
  void operator delete(void*) = delete;
  void operator delete[](void*) = delete;

  using Weight = typename TConfig::Weight;

  using WeightF32OrInputT =
      hwy::If<hwy::IsSame<Weight, float>(), float, EmbedderInputT>;
  CompressedArray<WeightF32OrInputT, TConfig::kVocabSize * TConfig::kModelDim>
      embedder_input_embedding;

  using WeightF32OrBF16 =
      hwy::If<hwy::IsSame<Weight, float>(), float, hwy::bfloat16_t>;
  CompressedArray<WeightF32OrBF16, TConfig::kModelDim> final_norm_scale;

  // Vit parts.
  CompressedArray<WeightF32OrBF16, TConfig::VitConfig::kModelDim>
      vit_encoder_norm_bias;
  CompressedArray<WeightF32OrBF16, TConfig::VitConfig::kModelDim>
      vit_encoder_norm_scale;
  CompressedArray<float, TConfig::VitConfig::kModelDim> vit_img_embedding_bias;
  CompressedArray<WeightF32OrBF16, TConfig::VitConfig::kModelDim * 14 * 14 * 3>
      vit_img_embedding_kernel;
  CompressedArray<float, 256 * TConfig::VitConfig::kModelDim>
      vit_img_pos_embedding;
  // The head maps from VitConfig::kModelDim (Vit final layer) to
  // kModelDim (LLM input).
  CompressedArray<float, TConfig::kModelDim> vit_img_head_bias;
  CompressedArray<WeightF32OrBF16,
                  TConfig::VitConfig::kModelDim * TConfig::kModelDim>
      vit_img_head_kernel;

  // Must be last so that the other arrays remain aligned.
  CompressedLayerPointers<TConfig> c_layer_ptrs;

  explicit CompressedWeights(hwy::ThreadPool& pool) : c_layer_ptrs(pool) {}

  // Called by weights.cc after ForEachTensor.
  void Reshape() {
    for (size_t layer = 0; layer < TConfig::kLayers; ++layer) {
      GetLayer(layer)->Reshape();
    }
  }

  void ZeroInit() {
    hwy::ZeroBytes(&embedder_input_embedding, sizeof(embedder_input_embedding));
    hwy::ZeroBytes(&final_norm_scale, sizeof(final_norm_scale));
    hwy::ZeroBytes(&vit_encoder_norm_bias, sizeof(vit_encoder_norm_bias));
    hwy::ZeroBytes(&vit_encoder_norm_scale, sizeof(vit_encoder_norm_scale));
    hwy::ZeroBytes(&vit_img_embedding_bias, sizeof(vit_img_embedding_bias));
    hwy::ZeroBytes(&vit_img_embedding_kernel, sizeof(vit_img_embedding_kernel));
    hwy::ZeroBytes(&vit_img_head_bias, sizeof(vit_img_head_bias));
    hwy::ZeroBytes(&vit_img_head_kernel, sizeof(vit_img_head_kernel));
    hwy::ZeroBytes(&vit_img_pos_embedding, sizeof(vit_img_pos_embedding));
    for (int i = 0; i < TConfig::kLayers; ++i) {
      hwy::ZeroBytes(GetLayer(i), sizeof(*GetLayer(i)));
    }
    if constexpr (TConfig::VitConfig::kLayers > 0) {
      for (int i = 0; i < TConfig::VitConfig::kLayers; ++i) {
        hwy::ZeroBytes(GetVitLayer(i), sizeof(*GetVitLayer(i)));
      }
    }
  }

  const CompressedLayer<TConfig>* GetLayer(size_t layer) const {
    return c_layer_ptrs.c_layers[layer].get();
  }
  CompressedLayer<TConfig>* GetLayer(size_t layer) {
    return c_layer_ptrs.c_layers[layer].get();
  }
  const CompressedLayer<typename TConfig::VitConfig>* GetVitLayer(
      size_t layer) const {
    return c_layer_ptrs.c_vit_layers[layer].get();
  }
  CompressedLayer<typename TConfig::VitConfig>* GetVitLayer(size_t layer) {
    return c_layer_ptrs.c_vit_layers[layer].get();
  }
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
    weights.Reshape();
  }
};

// TODO: also add RandInitCompressedWeights

template <class TConfig>
struct DeleteCompressedWeights {
  void operator()(ByteStorageT& weights_u8) const {
    CompressedWeights<TConfig>& weights =
        *reinterpret_cast<CompressedWeights<TConfig>*>(weights_u8.get());
    weights.~CompressedWeights<TConfig>();
  }
};

ByteStorageT LoadCompressedWeights(const Path& weights, Model model_type,
                                   Type weight_type, hwy::ThreadPool& pool);

void LogWeightStats(Model model, Type weight_type, const ByteStorageT& weights);

// ----------------------------------------------------------------------------
// Iterators

// We rely on `if constexpr` to ensure raw_weights->member is only compiled
// when valid, i.e., kHaveRaw == true, but the IDE analysis does not understand
// this, hence hide the member access from it.
#if HWY_IDE
#define GEMMA_MEMBER(aggregate, member) nullptr
#else
#define GEMMA_MEMBER(aggregate, member) aggregate->member
#endif

// Used by ForEachTensor for tensors that are not in a layer.
#define GEMMA_CALL_TOP_FUNC(name, member)                    \
  {                                                          \
    const float* raw_tensor = nullptr;                       \
    if constexpr (kHaveRaw) {                                \
      raw_tensor = GEMMA_MEMBER(raw_weights, member.data()); \
    }                                                        \
    func(name, raw_tensor, c_weights.member);                \
  }

// Used by ForEachTensor for per-layer tensors. Writes into name_buf.
#define GEMMA_CALL_FUNC(name, member)                          \
  snprintf(name_buf, sizeof(name_buf), name "_%d", layer_idx); \
  {                                                            \
    const float* raw_tensor = nullptr;                         \
    if constexpr (kHaveRaw) {                                  \
      raw_tensor = GEMMA_MEMBER(raw_layer, member.data());     \
    }                                                          \
    func(name_buf, raw_tensor, c_layer->member);               \
  }

// Calls func(name, float*, CompressedArray&) for each tensor. float* is
// null if raw_weights is nullptr, e.g., when loading weights from BlobStore.
// Otherwise, RawLayer must be specified and we pass a float* pointing to the
// raw float weights for that tensor for use by compress_weights.cc.
//
// This avoids repeating the list of tensors between loading and compressing,
// while also avoiding dependency on raw_weights.h.
//
// This only calls Func for tensors that TConfig requests/specifies, which means
// scale() is uninitialized for the other tensors, so their data_scale1() must
// not be called. (In other words, if the config doesn't specify a tensor, it
// shouldn't be used.)
template <class TConfig, class RawLayer = void, class RawWeightsPtr, class Func>
void ForEachTensor(RawWeightsPtr raw_weights,
                   CompressedWeights<TConfig>& c_weights, Func& func) {
  constexpr bool kHaveRaw = !hwy::IsSame<RawWeightsPtr, std::nullptr_t>();

  GEMMA_CALL_TOP_FUNC("c_embedding", embedder_input_embedding);
  GEMMA_CALL_TOP_FUNC("c_final_norm", final_norm_scale);

  if constexpr (TConfig::VitConfig::kLayers > 0 && !kHaveRaw) {
    GEMMA_CALL_TOP_FUNC("enc_norm_bias", vit_encoder_norm_bias);
    GEMMA_CALL_TOP_FUNC("enc_norm_scale", vit_encoder_norm_scale);
    GEMMA_CALL_TOP_FUNC("img_emb_bias", vit_img_embedding_bias);
    GEMMA_CALL_TOP_FUNC("img_emb_kernel", vit_img_embedding_kernel);
    GEMMA_CALL_TOP_FUNC("img_head_bias", vit_img_head_bias);
    GEMMA_CALL_TOP_FUNC("img_head_kernel", vit_img_head_kernel);
    GEMMA_CALL_TOP_FUNC("img_pos_emb", vit_img_pos_embedding);
  }

  char name_buf[16];
  for (int layer_idx = 0; layer_idx < TConfig::kLayers; ++layer_idx) {
    auto type = TConfig::kLayerConfig[layer_idx];
    const size_t idx = static_cast<size_t>(layer_idx);
    const RawLayer* raw_layer = nullptr;
    if constexpr (kHaveRaw) {
      raw_layer = raw_weights->GetLayer(idx);
    }
    CompressedLayer<TConfig>* c_layer = c_weights.GetLayer(idx);

    GEMMA_CALL_FUNC("pre_ff_ns", pre_ffw_norm_scale);
    GEMMA_CALL_FUNC("gating_ein", gating_einsum_w);
    GEMMA_CALL_FUNC("linear_w", linear_w);
    if (type == LayerAttentionType::kGemma) {
      GEMMA_CALL_FUNC("qkv_ein", qkv_einsum_w);
      GEMMA_CALL_FUNC("att_ein", attn_vec_einsum_w);
    } else {
      GEMMA_CALL_FUNC("gr_lin_x_w", griffin.linear_x_w);
      GEMMA_CALL_FUNC("gr_lin_x_b", griffin.linear_x_biases);
      GEMMA_CALL_FUNC("gr_lin_y_w", griffin.linear_y_w);
      GEMMA_CALL_FUNC("gr_lin_y_b", griffin.linear_y_biases);
      GEMMA_CALL_FUNC("gr_lin_out_w", griffin.linear_out_w);
      GEMMA_CALL_FUNC("gr_lin_out_b", griffin.linear_out_biases);
      GEMMA_CALL_FUNC("gr_conv_w", griffin.conv_w);
      GEMMA_CALL_FUNC("gr_conv_b", griffin.conv_biases);
      GEMMA_CALL_FUNC("gr_gate_w", griffin.gate_w);
      GEMMA_CALL_FUNC("gr_gate_b", griffin.gate_biases);
      GEMMA_CALL_FUNC("gr_a", griffin.a);
    }
    GEMMA_CALL_FUNC("pre_att_ns", pre_attention_norm_scale);

    if (TConfig::kPostNorm == PostNormType::Scale) {
      GEMMA_CALL_FUNC("post_att_ns", post_attention_norm_scale);
      GEMMA_CALL_FUNC("post_ff_ns", post_ffw_norm_scale);
    }

    if (TConfig::kFFBiases) {
      GEMMA_CALL_FUNC("ffw_gat_b", ffw_gating_biases);
      GEMMA_CALL_FUNC("ffw_out_b", ffw_output_biases);
    }

    if (TConfig::kSoftmaxAttnOutputBiases &&
        type == LayerAttentionType::kGemma) {
      GEMMA_CALL_FUNC("attn_ob", attention_output_biases);
    }
  }

  // Vit layers. Not supported for compress_weights.
  if constexpr (TConfig::VitConfig::kLayers > 0 && !kHaveRaw) {
    for (int layer_idx = 0; layer_idx < TConfig::VitConfig::kLayers;
         ++layer_idx) {
      auto type = TConfig::VitConfig::kLayerConfig[layer_idx];
      HWY_ASSERT(type == LayerAttentionType::kVit);
      const size_t idx = static_cast<size_t>(layer_idx);
      const RawLayer* raw_layer = nullptr;
      CompressedLayer<typename TConfig::VitConfig>* c_layer =
          c_weights.GetVitLayer(idx);

      // MHA.
      GEMMA_CALL_FUNC("attn_out_w", vit.attn_out_w);
      GEMMA_CALL_FUNC("attn_out_b", vit.attn_out_b);
      GEMMA_CALL_FUNC("qkv_ein_w", vit.qkv_einsum_w);
      GEMMA_CALL_FUNC("qkv_ein_b", vit.qkv_einsum_b);
      // MlpBlock.
      GEMMA_CALL_FUNC("linear_0_w", vit.linear_0_w);
      GEMMA_CALL_FUNC("linear_0_b", vit.linear_0_b);
      GEMMA_CALL_FUNC("linear_1_w", vit.linear_1_w);
      GEMMA_CALL_FUNC("linear_1_b", vit.linear_1_b);
      // LayerNorm.
      GEMMA_CALL_FUNC("ln_0_bias", vit.layer_norm_0_bias);
      GEMMA_CALL_FUNC("ln_0_scale", vit.layer_norm_0_scale);
      GEMMA_CALL_FUNC("ln_1_bias", vit.layer_norm_1_bias);
      GEMMA_CALL_FUNC("ln_1_scale", vit.layer_norm_1_scale);
    }
  }
#undef GEMMA_CALL_FUNC
#undef GEMMA_CALL_TOP_FUNC
}  // ForEachTensor

#define GEMMA_CALL_TOP_FUNC1(name, member) func(name, weights1.member)
#define GEMMA_CALL_TOP_FUNC2(name, member)      \
  func(name, weights1.member, weights2.member)
#define GEMMA_CALL_TOP_FUNC3(name, member)      \
  func(name, weights1.member, weights2.member, weights3.member)
#define GEMMA_CALL_TOP_FUNC4(name, member)       \
  func(name, weights1.member, weights2.member,   \
       weights3.member, weights4.member)

#define GEMMA_CALL_LAYER_FUNC1(name, member)                          \
  snprintf(name_buf, sizeof(name_buf), name "_%d", layer_idx);        \
  func(name_buf, layer1.member)

#define GEMMA_CALL_LAYER_FUNC2(name, member)                          \
  snprintf(name_buf, sizeof(name_buf), name "_%d", layer_idx);        \
  func(name_buf, layer1.member, layer2.member)

#define GEMMA_CALL_LAYER_FUNC3(name, member)                          \
  snprintf(name_buf, sizeof(name_buf), name "_%d", layer_idx);        \
  func(name_buf, layer1.member, layer2.member, layer3.member)

#define GEMMA_CALL_LAYER_FUNC4(name, member)                          \
  snprintf(name_buf, sizeof(name_buf), name "_%d", layer_idx);        \
  func(name_buf, layer1.member, layer2.member, layer3.member, layer4.member)

#define GEMMA_CALL_ALL_LAYER_FUNC(N)                                          \
  if (type == LayerAttentionType::kGemma) {                                   \
    GEMMA_CALL_LAYER_FUNC ## N("att_ein", attn_vec_einsum_w);                 \
    GEMMA_CALL_LAYER_FUNC ## N("qkv_ein", qkv_einsum_w);                      \
  } else {                                                                    \
    GEMMA_CALL_LAYER_FUNC ## N("gr_lin_x_w", griffin.linear_x_w);             \
    GEMMA_CALL_LAYER_FUNC ## N("gr_lin_x_b", griffin.linear_x_biases);        \
    GEMMA_CALL_LAYER_FUNC ## N("gr_lin_y_w", griffin.linear_y_w);             \
    GEMMA_CALL_LAYER_FUNC ## N("gr_lin_y_b", griffin.linear_y_biases);        \
    GEMMA_CALL_LAYER_FUNC ## N("gr_lin_out_w", griffin.linear_out_w);         \
    GEMMA_CALL_LAYER_FUNC ## N("gr_lin_out_b", griffin.linear_out_biases);    \
    GEMMA_CALL_LAYER_FUNC ## N("gr_conv_w", griffin.conv_w);                  \
    GEMMA_CALL_LAYER_FUNC ## N("gr_conv_b", griffin.conv_biases);             \
    GEMMA_CALL_LAYER_FUNC ## N("gr_gate_w", griffin.gate_w);                  \
    GEMMA_CALL_LAYER_FUNC ## N("gr_gate_b", griffin.gate_biases);             \
    GEMMA_CALL_LAYER_FUNC ## N("gr_a", griffin.a);                            \
  }                                                                           \
  GEMMA_CALL_LAYER_FUNC ## N("gating_ein", gating_einsum_w);                  \
  GEMMA_CALL_LAYER_FUNC ## N("linear_w", linear_w);                           \
  GEMMA_CALL_LAYER_FUNC ## N("pre_att_ns", pre_attention_norm_scale);         \
  if (TConfig::kPostNorm == PostNormType::Scale) {                            \
    GEMMA_CALL_LAYER_FUNC ## N("post_att_ns", post_attention_norm_scale);     \
    GEMMA_CALL_LAYER_FUNC ## N("post_ff_ns", post_ffw_norm_scale);            \
  }                                                                           \
  GEMMA_CALL_LAYER_FUNC ## N("pre_ff_ns", pre_ffw_norm_scale);                \
  if (TConfig::kFFBiases) {                                                   \
    GEMMA_CALL_LAYER_FUNC ## N("ffw_gat_b", ffw_gating_biases);               \
    GEMMA_CALL_LAYER_FUNC ## N("ffw_out_b", ffw_output_biases);               \
  }                                                                           \
  if (TConfig::kSoftmaxAttnOutputBiases &&                                    \
    type == LayerAttentionType::kGemma) {                                     \
    GEMMA_CALL_LAYER_FUNC ## N("attn_ob", attention_output_biases);           \
  }

template <typename TConfig, class Func>
void ForEachTensor1(Func& func, const CompressedWeights<TConfig>& weights1) {
  GEMMA_CALL_TOP_FUNC1("embedding", embedder_input_embedding);
  GEMMA_CALL_TOP_FUNC1("final_norm", final_norm_scale);
  char name_buf[16];
  for (int layer_idx = 0; layer_idx < TConfig::kLayers; ++layer_idx) {
    auto type = TConfig::kLayerConfig[layer_idx];
    const size_t idx = static_cast<size_t>(layer_idx);
    const CompressedLayer<TConfig>& layer1 = *weights1.GetLayer(idx);
    GEMMA_CALL_ALL_LAYER_FUNC(1)
  }
}

template <typename TConfig, class Func>
void ForEachTensor1(Func& func, CompressedWeights<TConfig>& weights1) {
  GEMMA_CALL_TOP_FUNC1("embedding", embedder_input_embedding);
  GEMMA_CALL_TOP_FUNC1("final_norm", final_norm_scale);
  char name_buf[16];
  for (int layer_idx = 0; layer_idx < TConfig::kLayers; ++layer_idx) {
    auto type = TConfig::kLayerConfig[layer_idx];
    const size_t idx = static_cast<size_t>(layer_idx);
    CompressedLayer<TConfig>& layer1 = *weights1.GetLayer(idx);
    GEMMA_CALL_ALL_LAYER_FUNC(1)
  }
}

template <typename TConfig, class Func>
void ForEachTensor2(Func& func, const CompressedWeights<TConfig>& weights1,
                    CompressedWeights<TConfig>& weights2) {
  GEMMA_CALL_TOP_FUNC2("embedding", embedder_input_embedding);
  GEMMA_CALL_TOP_FUNC2("final_norm", final_norm_scale);
  char name_buf[16];
  for (int layer_idx = 0; layer_idx < TConfig::kLayers; ++layer_idx) {
    auto type = TConfig::kLayerConfig[layer_idx];
    const size_t idx = static_cast<size_t>(layer_idx);
    const CompressedLayer<TConfig>& layer1 = *weights1.GetLayer(idx);
    CompressedLayer<TConfig>& layer2 = *weights2.GetLayer(idx);
    GEMMA_CALL_ALL_LAYER_FUNC(2)
  }
}

template <typename TConfig, class Func>
void ForEachTensor4(Func& func, const CompressedWeights<TConfig>& weights1,
                    CompressedWeights<TConfig>& weights2,
                    CompressedWeights<TConfig>& weights3,
                    CompressedWeights<TConfig>& weights4) {
  GEMMA_CALL_TOP_FUNC4("embedding", embedder_input_embedding);
  GEMMA_CALL_TOP_FUNC4("final_norm", final_norm_scale);
  char name_buf[16];
  for (int layer_idx = 0; layer_idx < TConfig::kLayers; ++layer_idx) {
    auto type = TConfig::kLayerConfig[layer_idx];
    const size_t idx = static_cast<size_t>(layer_idx);
    const CompressedLayer<TConfig>& layer1 = *weights1.GetLayer(idx);
    CompressedLayer<TConfig>& layer2 = *weights2.GetLayer(idx);
    CompressedLayer<TConfig>& layer3 = *weights3.GetLayer(idx);
    CompressedLayer<TConfig>& layer4 = *weights4.GetLayer(idx);
    GEMMA_CALL_ALL_LAYER_FUNC(4)
  }
}

#undef GEMMA_CALL_TOP_FUNC1
#undef GEMMA_CALL_TOP_FUNC2
#undef GEMMA_CALL_TOP_FUNC3
#undef GEMMA_CALL_TOP_FUNC4
#undef GEMMA_CALL_LAYER_FUNC1
#undef GEMMA_CALL_LAYER_FUNC2
#undef GEMMA_CALL_LAYER_FUNC3
#undef GEMMA_CALL_LAYER_FUNC4
#undef GEMMA_CALL_ALL_LAYER_FUNC

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_WEIGHTS_H_
