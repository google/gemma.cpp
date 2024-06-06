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

#include "compression/compress.h"
#include "gemma/common.h"
#include "gemma/configs.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

// Setting this to false will load and use uncompressed weights.
constexpr bool kWeightsAreCompressed = true;

// ----------------------------------------------------------------------------
// Uncompressed

template <typename T, class TConfig>
struct Layer {
  Layer() {}
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kKVHeads = TConfig::kKVHeads;
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;
  static constexpr size_t kAttVecEinsumWSize = kHeads * kQKVDim * kModelDim;
  static constexpr size_t kQKVEinsumWSize =
      (kHeads + 2 * kKVHeads) * kQKVDim * kModelDim;
  // 2x for (gelu gating vector, gated vector)
  static constexpr size_t kGatingEinsumWSize = 2 * kFFHiddenDim * kModelDim;
  static constexpr size_t kConv1dWidth = TConfig::kConv1dWidth;
  static constexpr bool kFFBiases = TConfig::kFFBiases;
  static constexpr bool kPostNormScale = TConfig::kPostNormScale;
  static constexpr size_t kAOBiasDim =
      TConfig::kSoftmaxAttnOutputBiases ? kModelDim : 0;
  static constexpr size_t kGriffinDim =
      TConfig::kGriffinLayers > 0 ? kModelDim : 0;

  union {
    struct {
      std::array<T, kAttVecEinsumWSize> attn_vec_einsum_w;
      std::array<T, kQKVEinsumWSize> qkv_einsum_w;
      std::array<T, kAOBiasDim> attention_output_biases;
    };

    struct {
      std::array<T, kGriffinDim * kGriffinDim> linear_x_w;
      std::array<T, kGriffinDim> linear_x_biases;
      std::array<T, kGriffinDim * kGriffinDim> linear_y_w;
      std::array<T, kGriffinDim> linear_y_biases;
      std::array<T, kGriffinDim * kGriffinDim> linear_out_w;
      std::array<T, kGriffinDim> linear_out_biases;
      std::array<T, kConv1dWidth * kGriffinDim> conv_w;
      std::array<T, kGriffinDim> conv_biases;
      std::array<T, kGriffinDim * kGriffinDim / kHeads * 2> gate_w;
      std::array<T, kGriffinDim * 2> gate_biases;
      std::array<T, kGriffinDim> a;
    } griffin;
  };

  std::array<T, kGatingEinsumWSize> gating_einsum_w;
  std::array<T, kModelDim * kFFHiddenDim> linear_w;
  std::array<T, kModelDim> pre_attention_norm_scale;
  std::array<T, kModelDim> pre_ffw_norm_scale;
  std::array<T, kPostNormScale ? kModelDim : 0> post_attention_norm_scale;
  std::array<T, kPostNormScale ? kModelDim : 0> post_ffw_norm_scale;

  std::array<T, kFFBiases ? 2 * kFFHiddenDim : 0> ffw_gating_biases;
  std::array<T, kFFBiases ? kModelDim : 0> ffw_output_biases;
};

template <class TConfig>
using LayerF = Layer<float, TConfig>;

// Array instead of single large allocation for parallel mem init. Split out of
// Weights so that only these pointers are initialized.
template <typename T, class TConfig>
struct LayerPointers {
  explicit LayerPointers(hwy::ThreadPool& pool) {
    pool.Run(0, TConfig::kLayers, [this](uint64_t task, size_t /*thread*/) {
      this->layers[task] = hwy::AllocateAligned<Layer<T, TConfig>>(1);
    });
  }

  using TLayer = Layer<T, TConfig>;
  std::array<hwy::AlignedFreeUniquePtr<TLayer[]>, TConfig::kLayers> layers;
};

template <typename T, class TConfig>
struct Weights {
  // No ctor/dtor, allocated via AllocateAligned.

  std::array<T, TConfig::kVocabSize * TConfig::kModelDim>
      embedder_input_embedding;

  std::array<T, TConfig::kModelDim> final_norm_scale;

  LayerPointers<T, TConfig> layer_ptrs;

  std::array<T, TConfig::kNumTensorScales> scales;

  const Layer<T, TConfig>* GetLayer(size_t layer) const {
    return layer_ptrs.layers[layer].get();
  }
  Layer<T, TConfig>* GetLayer(size_t layer) {
    return layer_ptrs.layers[layer].get();
  }
};

template <class TConfig>
using WeightsF = Weights<float, TConfig>;

// ----------------------------------------------------------------------------
// Compressed

// If weights are f32, also f32; otherwise at least bf16. Useful for ops that do
// not yet support smaller compressed types, or require at least bf16. When
// weights are f32, we also want such tensors to be f32.
template <class TConfig>
using WeightF32OrBF16T =
    hwy::If<hwy::IsSame<typename TConfig::WeightT, float>(), float,
            hwy::bfloat16_t>;

template <class TConfig>
struct CompressedLayer {
  // No ctor/dtor, allocated via AllocateAligned.

  using TLayer = gcpp::LayerF<TConfig>;
  using WeightT = typename TConfig::WeightT;
  using WeightF32OrBF16 = WeightF32OrBF16T<TConfig>;

  static constexpr size_t kHeads = TLayer::kHeads;
  static constexpr size_t kKVHeads = TLayer::kKVHeads;
  static constexpr size_t kModelDim = TLayer::kModelDim;
  static constexpr size_t kQKVDim = TLayer::kQKVDim;
  static constexpr size_t kFFHiddenDim = TLayer::kFFHiddenDim;
  static constexpr size_t kAttVecEinsumWSize = TLayer::kAttVecEinsumWSize;
  static constexpr size_t kQKVEinsumWSize = TLayer::kQKVEinsumWSize;
  static constexpr size_t kGatingEinsumWSize = TLayer::kGatingEinsumWSize;
  static constexpr size_t kConv1dWidth = TLayer::kConv1dWidth;
  static constexpr bool kFFBiases = TLayer::kFFBiases;
  static constexpr bool kPostNormScale = TConfig::kPostNormScale;
  static constexpr size_t kAOBiasDim = TLayer::kAOBiasDim;
  static constexpr size_t kGriffinDim = TLayer::kGriffinDim;

  // Compressed Parameters

  template <class T, size_t N>
  using ArrayT = CompressedArray<T, N>;

  union {
    struct {
      ArrayT<WeightT, kAttVecEinsumWSize> attn_vec_einsum_w;
      ArrayT<WeightT, kQKVEinsumWSize> qkv_einsum_w;
      ArrayT<float, kAOBiasDim> attention_output_biases;
    };

    struct {
      ArrayT<WeightT, kGriffinDim * kGriffinDim> linear_x_w;
      ArrayT<float, kGriffinDim> linear_x_biases;
      ArrayT<WeightT, kGriffinDim * kGriffinDim> linear_y_w;
      ArrayT<float, kGriffinDim> linear_y_biases;
      ArrayT<WeightT, kGriffinDim * kGriffinDim> linear_out_w;
      ArrayT<float, kGriffinDim> linear_out_biases;
      ArrayT<float, TConfig::kConv1dWidth * kGriffinDim> conv_w;
      ArrayT<float, kGriffinDim> conv_biases;
      ArrayT<WeightT, kGriffinDim * kGriffinDim / kHeads * 2> gate_w;
      ArrayT<float, kGriffinDim * 2> gate_biases;
      ArrayT<float, kGriffinDim> a;
    } griffin;
  };

  ArrayT<WeightT, TLayer::kGatingEinsumWSize> gating_einsum_w;
  ArrayT<WeightT, kModelDim * kFFHiddenDim> linear_w;
  // We don't yet have an RMSNorm that accepts all WeightT.
  ArrayT<WeightF32OrBF16, kModelDim> pre_attention_norm_scale;
  ArrayT<WeightF32OrBF16, kModelDim> pre_ffw_norm_scale;
  ArrayT<WeightF32OrBF16, kPostNormScale ? kModelDim : 0>
      post_attention_norm_scale;
  ArrayT<WeightF32OrBF16, kPostNormScale ? kModelDim : 0> post_ffw_norm_scale;

  ArrayT<float, kFFBiases ? 2 * kFFHiddenDim : 0> ffw_gating_biases;
  ArrayT<float, kFFBiases ? kModelDim : 0> ffw_output_biases;
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
  }

  using CLayer = CompressedLayer<TConfig>;
  std::array<hwy::AlignedFreeUniquePtr<CLayer[]>, TConfig::kLayers> c_layers;
};

template <class TConfig>
struct CompressedWeights {
  // No ctor/dtor, allocated via AllocateAligned.

  CompressedArray<EmbedderInputT, TConfig::kVocabSize * TConfig::kModelDim>
      embedder_input_embedding;

  CompressedArray<hwy::bfloat16_t, TConfig::kModelDim> final_norm_scale;

  // Must be last so that the other arrays remain aligned.
  CompressedLayerPointers<TConfig> c_layer_ptrs;

  const CompressedLayer<TConfig>* GetLayer(size_t layer) const {
    return c_layer_ptrs.c_layers[layer].get();
  }
  CompressedLayer<TConfig>* GetLayer(size_t layer) {
    return c_layer_ptrs.c_layers[layer].get();
  }
};

// ----------------------------------------------------------------------------
// Interface

template <class TConfig>
using WeightsT = hwy::If<kWeightsAreCompressed, CompressedWeights<TConfig>,
                         WeightsF<TConfig>>;

// Call via CallFunctorForModel.
template <typename T, typename TConfig>
struct AllocateWeights {
  ByteStorageT operator()(hwy::ThreadPool& pool) const {
    using TWeights = Weights<T, TConfig>;
    ByteStorageT weights_u8 = AllocateSizeof<TWeights>();
    TWeights* weights = reinterpret_cast<TWeights*>(weights_u8.get());
    new (&weights->layer_ptrs) LayerPointers<T, TConfig>(pool);
    return weights_u8;
  }
};

template <typename TConfig>
struct AllocateWeightsF {
  ByteStorageT operator()(hwy::ThreadPool& pool) const {
    return AllocateWeights<float, TConfig>()(pool);
  }
};

template <typename T, typename TConfig>
struct ZeroInitWeights {
  void operator()(ByteStorageT& weights, hwy::ThreadPool& pool) const {
    Weights<T, TConfig>& w =
        *reinterpret_cast<Weights<T, TConfig>*>(weights.get());
    hwy::ZeroBytes(&w.embedder_input_embedding,
                   sizeof(w.embedder_input_embedding));
    hwy::ZeroBytes(&w.final_norm_scale, sizeof(w.final_norm_scale));
    for (int i = 0; i < TConfig::kLayers; ++i) {
      hwy::ZeroBytes(w.GetLayer(i), sizeof(*w.GetLayer(i)));
    }
  }
};

template <typename TConfig>
struct ZeroInitWeightsF {
  void operator()(ByteStorageT& weights, hwy::ThreadPool& pool) const {
    ZeroInitWeights<float, TConfig>()(weights, pool);
  }
};

template <typename T, typename TConfig>
struct CopyWeights {
void operator()(Weights<T, TConfig>& dst,
                const Weights<T, TConfig>& src) const {
    hwy::CopyBytes(&src.embedder_input_embedding, &dst.embedder_input_embedding,
                   sizeof(src.embedder_input_embedding));
    hwy::CopyBytes(&src.final_norm_scale, &dst.final_norm_scale,
                   sizeof(src.final_norm_scale));
    for (int i = 0; i < TConfig::kLayers; ++i) {
      hwy::CopyBytes(src.GetLayer(i), dst.GetLayer(i),
                     sizeof(*dst.GetLayer(i)));
    }
  }
};

template <class TConfig>
struct DeleteLayersPtrs {
  void operator()(ByteStorageT& weights_u8) const {
    auto* weights = reinterpret_cast<WeightsT<TConfig>*>(weights_u8.get());
    if constexpr (kWeightsAreCompressed) {
      weights->c_layer_ptrs.~CompressedLayerPointers<TConfig>();
    } else {
      weights->layer_ptrs.~LayerPointers<float, TConfig>();
    }
  }
};

// Owns weights and provides access to TConfig.
template <typename T, typename TConfig>
class WeightsWrapper {
 public:
  WeightsWrapper()
      : pool_(0),
        data_(AllocateWeights<T, TConfig>()(pool_)),
        weights_(reinterpret_cast<Weights<T, TConfig>*>(data_.get())) {}

  const Weights<T, TConfig>& get() const { return *weights_; }
  Weights<T, TConfig>& get() { return *weights_; }
  void clear() { ZeroInitWeights<T, TConfig>()(data_, pool_); }
  void copy(const WeightsWrapper<T, TConfig>& other) {
    CopyWeights<T, TConfig>()(get(), other.get());
  }

 private:
  hwy::ThreadPool pool_;
  ByteStorageT data_;
  Weights<T, TConfig>* weights_;
};

// For use by compress_weights.cc.
ByteStorageT LoadRawWeights(const Path& weights, Model model,
                            hwy::ThreadPool& pool, bool scale_for_compression);

// For gemma.cc; calls LoadRawWeights if !kWeightsAreCompressed.
ByteStorageT LoadWeights(const Path& weights, Model model,
                         hwy::ThreadPool& pool);

void LogWeightStats(Model model, const ByteStorageT& weights);

// ----------------------------------------------------------------------------
// Iterators

#define GEMMA_CALL_FUNC(name, member)                          \
  snprintf(name_buf, sizeof(name_buf), name "_%d", layer_idx); \
  func(name_buf, layer ? layer->member.data() : nullptr, layer_weights->member)

// Calls func(name, float*, CompressedArray&) for each tensor. float* is null
// if weights = null, which happens during the first call where we attempt to
// load from cache.
//
// This avoids repeating the list of tensors between loading and compressing.
template <class TConfig, class Func>
void ForEachTensor(const WeightsF<TConfig>* weights,
                   CompressedWeights<TConfig>& c_weights, Func& func) {
  func("c_embedding",
       weights ? weights->embedder_input_embedding.data() : nullptr,
       c_weights.embedder_input_embedding);
  func("c_final_norm", weights ? weights->final_norm_scale.data() : nullptr,
       c_weights.final_norm_scale);

  char name_buf[16];
  for (int layer_idx = 0; layer_idx < TConfig::kLayers; ++layer_idx) {
    auto type = TConfig::kLayerConfig[layer_idx];
    const size_t idx = static_cast<size_t>(layer_idx);
    const LayerF<TConfig>* layer = weights ? weights->GetLayer(idx) : nullptr;
    CompressedLayer<TConfig>* layer_weights = c_weights.GetLayer(idx);

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
    if (TConfig::kPostNormScale) {
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
}

#undef GEMMA_CALL_FUNC

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
  if (TConfig::kPostNormScale) {                                              \
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

template <typename T, typename TConfig, class Func>
void ForEachTensor1(Func& func, const Weights<T, TConfig>& weights1) {
  GEMMA_CALL_TOP_FUNC1("embedding", embedder_input_embedding);
  GEMMA_CALL_TOP_FUNC1("final_norm", final_norm_scale);
  char name_buf[16];
  for (int layer_idx = 0; layer_idx < TConfig::kLayers; ++layer_idx) {
    auto type = TConfig::kLayerConfig[layer_idx];
    const size_t idx = static_cast<size_t>(layer_idx);
    const LayerF<TConfig>& layer1 = *weights1.GetLayer(idx);
    GEMMA_CALL_ALL_LAYER_FUNC(1)
  }
}

template <typename T, typename TConfig, class Func>
void ForEachTensor1(Func& func, Weights<T, TConfig>& weights1) {
  GEMMA_CALL_TOP_FUNC1("embedding", embedder_input_embedding);
  GEMMA_CALL_TOP_FUNC1("final_norm", final_norm_scale);
  char name_buf[16];
  for (int layer_idx = 0; layer_idx < TConfig::kLayers; ++layer_idx) {
    auto type = TConfig::kLayerConfig[layer_idx];
    const size_t idx = static_cast<size_t>(layer_idx);
    LayerF<TConfig>& layer1 = *weights1.GetLayer(idx);
    GEMMA_CALL_ALL_LAYER_FUNC(1)
  }
}

template <typename T, typename TConfig, class Func>
void ForEachTensor2(Func& func, const Weights<T, TConfig>& weights1,
                    Weights<T, TConfig>& weights2) {
  GEMMA_CALL_TOP_FUNC2("embedding", embedder_input_embedding);
  GEMMA_CALL_TOP_FUNC2("final_norm", final_norm_scale);
  char name_buf[16];
  for (int layer_idx = 0; layer_idx < TConfig::kLayers; ++layer_idx) {
    auto type = TConfig::kLayerConfig[layer_idx];
    const size_t idx = static_cast<size_t>(layer_idx);
    const LayerF<TConfig>& layer1 = *weights1.GetLayer(idx);
    LayerF<TConfig>& layer2 = *weights2.GetLayer(idx);
    GEMMA_CALL_ALL_LAYER_FUNC(2)
  }
}

template <typename T, typename TConfig, class Func>
void ForEachTensor4(Func& func, const Weights<T, TConfig>& weights1,
                    Weights<T, TConfig>& weights2,
                    Weights<T, TConfig>& weights3,
                    Weights<T, TConfig>& weights4) {
  GEMMA_CALL_TOP_FUNC4("embedding", embedder_input_embedding);
  GEMMA_CALL_TOP_FUNC4("final_norm", final_norm_scale);
  char name_buf[16];
  for (int layer_idx = 0; layer_idx < TConfig::kLayers; ++layer_idx) {
    auto type = TConfig::kLayerConfig[layer_idx];
    const size_t idx = static_cast<size_t>(layer_idx);
    const LayerF<TConfig>& layer1 = *weights1.GetLayer(idx);
    LayerF<TConfig>& layer2 = *weights2.GetLayer(idx);
    LayerF<TConfig>& layer3 = *weights3.GetLayer(idx);
    LayerF<TConfig>& layer4 = *weights4.GetLayer(idx);
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
