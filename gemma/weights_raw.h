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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_WEIGHTS_RAW_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_WEIGHTS_RAW_H_

// NOTE: this file should only be used by compress_weights; it is currently
// also referenced by backprop, but we plan to remove that. Historical note:
// this was the original f32-only simple on-disk format created by a Python
// export script. BlobStore is now the preferred on-disk format, and we load
// that into CompressedWeights.

#include <random>

#include "gemma/common.h"
#include "gemma/configs.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
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
  static constexpr PostNormType kPostNorm = TConfig::kPostNorm;
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
  std::array<T, kPostNorm == PostNormType::Scale ? kModelDim : 0>
      post_attention_norm_scale;
  std::array<T, kPostNorm == PostNormType::Scale ? kModelDim : 0>
      post_ffw_norm_scale;

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

// TODO: can we use TConfig::Weight instead of T?
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

// TODO: make a member of Weights<T>.
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

template <typename T, size_t kLen>
void RandInit(std::array<T, kLen>& x, T stddev, std::mt19937& gen) {
  std::normal_distribution<T> dist(0.0, stddev);
  for (size_t i = 0; i < kLen; ++i) {
    x[i] = dist(gen);
  }
}

// TODO: make a member of Layer<T>.
template <typename T, typename TConfig>
void RandInit(Layer<T, TConfig>& w, T stddev, std::mt19937& gen) {
  RandInit(w.pre_attention_norm_scale, stddev, gen);
  RandInit(w.attn_vec_einsum_w, stddev, gen);
  RandInit(w.qkv_einsum_w, stddev, gen);
  RandInit(w.pre_ffw_norm_scale, stddev, gen);
  RandInit(w.gating_einsum_w, stddev, gen);
  RandInit(w.linear_w, stddev, gen);
}

template <typename T, typename TConfig>
void RandInit(Weights<T, TConfig>& w, T stddev, std::mt19937& gen) {
  static constexpr size_t kLayers = TConfig::kLayers;
  RandInit(w.embedder_input_embedding, stddev, gen);
  RandInit(w.final_norm_scale, stddev, gen);
  for (size_t i = 0; i < kLayers; ++i) {
    RandInit(*w.GetLayer(i), stddev, gen);
  }
}

// Owns weights and provides access to TConfig.
template <typename T, typename TConfig>
class WeightsWrapper {
 public:
  WeightsWrapper()
      : pool_(0),
        data_(AllocateWeights<T, TConfig>()(pool_)),
        weights_(reinterpret_cast<Weights<T, TConfig>*>(data_.get())) {}

  ~WeightsWrapper() {
    get().layer_ptrs.~LayerPointers<T, TConfig>();
  }

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

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_WEIGHTS_RAW_H_
