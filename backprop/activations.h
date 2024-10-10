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

#ifndef THIRD_PARTY_GEMMA_CPP_BACKPROP_ACTIVATIONS_H_
#define THIRD_PARTY_GEMMA_CPP_BACKPROP_ACTIVATIONS_H_

#include <stddef.h>

#include <array>

#include "compression/compress.h"  // MatStorageT
#include "util/allocator.h"  // ByteStorageT

namespace gcpp {

template <typename T, typename TConfig>
struct ForwardLayer {
  ForwardLayer()
      : input("input", kSeqLen, kModelDim),
        pre_att_rms_out("pre_att_rms_out", kSeqLen, kModelDim),
        qkv("qkv", kSeqLen * (kHeads + 2), kQKVDim),
        att("att", kSeqLen * kHeads, kSeqLen),
        att_out("att_out", kSeqLen * kHeads, kQKVDim),
        att_post1("att_post1", kSeqLen, kModelDim),
        attention_out("attention_out", kSeqLen, kModelDim),
        bf_pre_ffw_rms_out("bf_pre_ffw_rms_out", kSeqLen, kModelDim),
        ffw_hidden("ffw_hidden", kSeqLen, kFFHiddenDim * 2),
        ffw_hidden_gated("ffw_hidden_gated", kSeqLen, kFFHiddenDim) {}

  static constexpr size_t kSeqLen = TConfig::kSeqLen;
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;

  MatStorageT<T> input;
  MatStorageT<T> pre_att_rms_out;
  MatStorageT<T> qkv;
  MatStorageT<T> att;
  MatStorageT<T> att_out;
  MatStorageT<T> att_post1;
  MatStorageT<T> attention_out;
  MatStorageT<T> bf_pre_ffw_rms_out;
  MatStorageT<T> ffw_hidden;
  MatStorageT<T> ffw_hidden_gated;
};

template <typename T, typename TConfig>
struct ForwardPass {
  ForwardPass()
      : final_layer_output("final_layer_output", kSeqLen, kModelDim),
        final_norm_output("final_norm_output", kSeqLen, kModelDim),
        logits("logits", kSeqLen, kVocabSize),
        probs("probs", kSeqLen, kVocabSize) {
  }  // prevents placement-new calling memset

  static constexpr size_t kSeqLen = TConfig::kSeqLen;
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kVocabSize = TConfig::kVocabSize;
  static constexpr size_t kLayers = TConfig::kLayers;

  std::array<ForwardLayer<T, TConfig>, kLayers> layers;
  MatStorageT<T> final_layer_output;
  MatStorageT<T> final_norm_output;
  MatStorageT<T> logits;
  MatStorageT<T> probs;
};

template <typename TConfig>
struct AllocateForwardPass {
  ByteStorageT operator()() const {
    ByteStorageT c_weights_u8 = AllocateSizeof<ForwardPass<float, TConfig>>();
    auto* c_weights =
        reinterpret_cast<ForwardPass<float, TConfig>*>(c_weights_u8.get());
    new (c_weights) ForwardPass<float, TConfig>();
    return c_weights_u8;
  }
};

// Owns activations and undoes the type erasure of AllocateAligned.
template<typename T, typename TConfig>
class ActivationsWrapper {
  using WrappedT = ForwardPass<T, TConfig>;

 public:
  ActivationsWrapper()
      : data_(AllocateSizeof<WrappedT>()),
        activations_(*(new(data_.get()) WrappedT())) {}

  const WrappedT& get() const { return activations_; }
  WrappedT& get() { return activations_; }

 private:
  ByteStorageT data_;
  WrappedT& activations_;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_BACKPROP_ACTIVATIONS_H_
