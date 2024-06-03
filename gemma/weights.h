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

#include "gemma/common.h"
#include "gemma/configs.h"
#include "hwy/aligned_allocator.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

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

template <typename T, typename TConfig>
ByteStorageT AllocateWeights(hwy::ThreadPool& pool) {
  using TWeights = Weights<T, TConfig>;
  ByteStorageT weights_u8 = hwy::AllocateAligned<uint8_t>(sizeof(TWeights));
  TWeights* weights = reinterpret_cast<TWeights*>(weights_u8.get());
  new (&weights->layer_ptrs) LayerPointers<T, TConfig>(pool);
  return weights_u8;
}

#define CALL_TOP_FUNC1(name, member) func(name, weights1.member)
#define CALL_TOP_FUNC2(name, member)             \
  func(name, weights1.member, weights2.member)
#define CALL_TOP_FUNC3(name, member)             \
  func(name, weights1.member, weights2.member, weights3.member)
#define CALL_TOP_FUNC4(name, member)             \
  func(name, weights1.member, weights2.memeber,  \
       weights3.member, weights4.member)

#define CALL_LAYER_FUNC1(name, member)                                \
  snprintf(name_buf, sizeof(name_buf), name "_%d", layer_idx);        \
  func(name_buf, layer1.member)

#define CALL_LAYER_FUNC2(name, member)                                \
  snprintf(name_buf, sizeof(name_buf), name "_%d", layer_idx);        \
  func(name_buf, layer1.member, layer2.member)

#define CALL_LAYER_FUNC3(name, member)                                \
  snprintf(name_buf, sizeof(name_buf), name "_%d", layer_idx);        \
  func(name_buf, layer1.member, layer2.member, layer3.member)

#define CALL_LAYER_FUNC4(name, member)                                \
  snprintf(name_buf, sizeof(name_buf), name "_%d", layer_idx);        \
  func(name_buf, layer1.member, layer2.member, layer4.member)

#define CALL_ALL_LAYER_FUNC(N) \
  if (type == LayerAttentionType::kGemma) {                             \
    CALL_LAYER_FUNC ## N("att_ein", attn_vec_einsum_w);                 \
    CALL_LAYER_FUNC ## N("qkv_ein", qkv_einsum_w);                      \
  } else {                                                              \
    CALL_LAYER_FUNC ## N("gr_lin_x_w", griffin.linear_x_w);             \
    CALL_LAYER_FUNC ## N("gr_lin_x_b", griffin.linear_x_biases);        \
    CALL_LAYER_FUNC ## N("gr_lin_y_w", griffin.linear_y_w);             \
    CALL_LAYER_FUNC ## N("gr_lin_y_b", griffin.linear_y_biases);        \
    CALL_LAYER_FUNC ## N("gr_lin_out_w", griffin.linear_out_w);         \
    CALL_LAYER_FUNC ## N("gr_lin_out_b", griffin.linear_out_biases);    \
    CALL_LAYER_FUNC ## N("gr_conv_w", griffin.conv_w);                  \
    CALL_LAYER_FUNC ## N("gr_conv_b", griffin.conv_biases);             \
    CALL_LAYER_FUNC ## N("gr_gate_w", griffin.gate_w);                  \
    CALL_LAYER_FUNC ## N("gr_gate_b", griffin.gate_biases);             \
    CALL_LAYER_FUNC ## N("gr_a", griffin.a);                            \
  }                                                                     \
  CALL_LAYER_FUNC ## N("gating_ein", gating_einsum_w);                  \
  CALL_LAYER_FUNC ## N("linear_w", linear_w);                           \
  CALL_LAYER_FUNC ## N("pre_att_ns", pre_attention_norm_scale);         \
  if (TConfig::kPostNormScale) {                                        \
    CALL_LAYER_FUNC ## N("post_att_ns", post_attention_norm_scale);     \
    CALL_LAYER_FUNC ## N("post_ff_ns", post_ffw_norm_scale);            \
  }                                                                     \
  CALL_LAYER_FUNC ## N("pre_ff_ns", pre_ffw_norm_scale);                \
  if (TConfig::kFFBiases) {                                             \
    CALL_LAYER_FUNC ## N("ffw_gat_b", ffw_gating_biases);               \
    CALL_LAYER_FUNC ## N("ffw_out_b", ffw_output_biases);               \
  }                                                                     \
  if (TConfig::kSoftmaxAttnOutputBiases &&                              \
    type == LayerAttentionType::kGemma) {                               \
    CALL_LAYER_FUNC ## N("attn_ob", attention_output_biases);           \
  }

template <typename T, typename TConfig, class Func>
void ForEachTensor1(Func& func, const Weights<T, TConfig>& weights1) {
  CALL_TOP_FUNC1("embedding", embedder_input_embedding);
  CALL_TOP_FUNC1("final_norm", final_norm_scale);
  char name_buf[16];
  for (int layer_idx = 0; layer_idx < TConfig::kLayers; ++layer_idx) {
    auto type = TConfig::kLayerConfig[layer_idx];
    const size_t idx = static_cast<size_t>(layer_idx);
    const LayerF<TConfig>& layer1 = *weights1.GetLayer(idx);
    CALL_ALL_LAYER_FUNC(1)
  }
}

template <typename T, typename TConfig, class Func>
void ForEachTensor1(Func& func, Weights<T, TConfig>& weights1) {
  CALL_TOP_FUNC1("embedding", embedder_input_embedding);
  CALL_TOP_FUNC1("final_norm", final_norm_scale);
  char name_buf[16];
  for (int layer_idx = 0; layer_idx < TConfig::kLayers; ++layer_idx) {
    auto type = TConfig::kLayerConfig[layer_idx];
    const size_t idx = static_cast<size_t>(layer_idx);
    LayerF<TConfig>& layer1 = *weights1.GetLayer(idx);
    CALL_ALL_LAYER_FUNC(1)
  }
}

template <typename T, typename TConfig, class Func>
void ForEachTensor2(Func& func, const Weights<T, TConfig>& weights1,
                    Weights<T, TConfig>& weights2) {
  CALL_TOP_FUNC2("embedding", embedder_input_embedding);
  CALL_TOP_FUNC2("final_norm", final_norm_scale);
  char name_buf[16];
  for (int layer_idx = 0; layer_idx < TConfig::kLayers; ++layer_idx) {
    auto type = TConfig::kLayerConfig[layer_idx];
    const size_t idx = static_cast<size_t>(layer_idx);
    const LayerF<TConfig>& layer1 = *weights1.GetLayer(idx);
    LayerF<TConfig>& layer2 = *weights2.GetLayer(idx);
    CALL_ALL_LAYER_FUNC(2)
  }
}

#undef CALL_TOP_FUNC1
#undef CALL_TOP_FUNC2
#undef CALL_TOP_FUNC3
#undef CALL_TOP_FUNC4
#undef CALL_LAYER_FUNC1
#undef CALL_LAYER_FUNC2
#undef CALL_LAYER_FUNC3
#undef CALL_LAYER_FUNC4
#undef CALL_ALL_LAYER_FUNC

template<typename T, typename TConfig>
void ZeroInit(Weights<T, TConfig>& w) {
  memset(&w.embedder_input_embedding, 0, sizeof(w.embedder_input_embedding));
  memset(&w.final_norm_scale, 0, sizeof(w.final_norm_scale));
  for (int i = 0; i < TConfig::kLayers; ++i) {
    memset(w.GetLayer(i), 0, sizeof(*w.GetLayer(i)));
  }
}

template<typename T, typename TConfig>
void Copy(Weights<T, TConfig>& dst, const Weights<T, TConfig>& src) {
  memcpy(&dst.embedder_input_embedding, &src.embedder_input_embedding,
         sizeof(src.embedder_input_embedding));
  memcpy(&dst.final_norm_scale, &src.final_norm_scale,
         sizeof(src.final_norm_scale));
  for (int i = 0; i < TConfig::kLayers; ++i) {
    memcpy(dst.GetLayer(i), src.GetLayer(i), sizeof(*dst.GetLayer(i)));
  }
}

template<typename T, typename TConfig>
class WeightsWrapper {
 public:
  WeightsWrapper()
      : pool_(0), data_(AllocateWeights<T, TConfig>(pool_)),
        weights_(reinterpret_cast<Weights<T, TConfig>*>(data_.get())) {}

  const Weights<T, TConfig>& get() const { return *weights_; }
  Weights<T, TConfig>& get() { return *weights_; }
  void clear() { ZeroInit(get()); }
  void copy(const WeightsWrapper<T, TConfig>& other) {
    Copy(get(), other.get());
  }

 private:
  hwy::ThreadPool pool_;
  ByteStorageT data_;
  Weights<T, TConfig>* weights_;
};

ByteStorageT AllocateWeights(Model model, hwy::ThreadPool& pool);

void ZeroInitWeights(Model model, ByteStorageT& weights, hwy::ThreadPool& pool);

void LogWeightStats(Model model, const ByteStorageT& weights);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_WEIGHTS_H_
