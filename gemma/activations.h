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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_ACTIVATIONS_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_ACTIVATIONS_H_

#include "gemma/common.h"
#include "hwy/aligned_allocator.h"

namespace gcpp {

template <typename T, typename TConfig>
struct ForwardLayer {
  static constexpr size_t kSeqLen = TConfig::kSeqLen;
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;
  std::array<T, kSeqLen * kModelDim> input;
  std::array<T, kSeqLen * kModelDim> pre_att_rms_out;
  std::array<T, kSeqLen * (kHeads + 2) * kQKVDim> qkv;
  std::array<T, kSeqLen * kHeads * kSeqLen> att;
  std::array<T, kSeqLen * kHeads * kQKVDim> att_out;
  std::array<T, kSeqLen * kModelDim> att_post1;
  std::array<T, kSeqLen * kModelDim> attention_out;
  std::array<T, kSeqLen * kModelDim> bf_pre_ffw_rms_out;
  std::array<T, kSeqLen * kFFHiddenDim * 2> ffw_hidden;
  std::array<T, kSeqLen * kFFHiddenDim> ffw_hidden_gated;
};

template <typename T, typename TConfig>
struct ForwardPass {
  ForwardPass() {}  // prevents placement-new calling memset

  static constexpr size_t kSeqLen = TConfig::kSeqLen;
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kVocabSize = TConfig::kVocabSize;
  static constexpr size_t kLayers = TConfig::kLayers;

  std::array<ForwardLayer<T, TConfig>, kLayers> layers;
  std::array<T, kSeqLen * kModelDim> final_layer_output;
  std::array<T, kSeqLen * kModelDim> final_norm_output;
  std::array<T, kSeqLen * kVocabSize> logits;
  std::array<T, kSeqLen * kVocabSize> probs;

  static ByteStorageT Allocate() {
    return hwy::AllocateAligned<uint8_t>(sizeof(ForwardPass<T, TConfig>));
  }
};

template<typename T, typename TConfig>
class ActivationsWrapper {
  using WrappedT = ForwardPass<T, TConfig>;

 public:
  ActivationsWrapper()
      : data_(WrappedT::Allocate()),
        activations_(*reinterpret_cast<WrappedT*>(data_.get())) {}

  const WrappedT& get() const { return activations_; }
  WrappedT& get() { return activations_; }

 private:
  ByteStorageT data_;
  WrappedT& activations_;
};

ByteStorageT AllocateForwardPass(Model model);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_ACTIVATIONS_H_
