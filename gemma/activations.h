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

#include <stddef.h>

#include <array>

#include "gemma/common.h"  // AllocateSizeof
#include "hwy/base.h"        // hwy::bfloat16_t

namespace gcpp {

// Must be aligned.
template <class TConfig, size_t kBatchSize>
struct Activations {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kKVHeads = TConfig::kKVHeads;
  static constexpr bool kIsMHA = kHeads == kKVHeads;  // Multi-Head Attention
  // Stride between subsequent queries. Each of Q, K, V are of length kQKVDim,
  // but for MHA we store them as Q,K,V, Q,K,V, .. instead of Q..Q, K..K, V..V.
  static constexpr size_t kQStride = kQKVDim * (kIsMHA ? 3 : 1);

  std::array<float, kBatchSize * kModelDim> x;  // input
  std::array<float, kBatchSize * kModelDim> pre_att_rms_out;
  std::array<float, kBatchSize * kHeads * kQStride> q;  // query vector
  std::array<float, kBatchSize * kHeads * TConfig::kSeqLen>
      att;                                                   // attention vector
  std::array<float, kBatchSize * kHeads * kQKVDim> att_out;  // attention output
  std::array<float, kHeads * kBatchSize * kModelDim>
      att_post1;  // attention output after linear transformation, per head
  std::array<float, kBatchSize * kModelDim>
      att_post2;  // accumulation of attention outputs over heads
  std::array<hwy::bfloat16_t, kBatchSize * kModelDim> bf_pre_ffw_rms_out;
  std::array<float, kBatchSize * TConfig::kFFHiddenDim * 2> ffw_hidden;

  // For FFW MatMul.
  std::array<float, kBatchSize * TConfig::kFFHiddenDim> C1;
  std::array<float, kBatchSize * TConfig::kFFHiddenDim> C2;

  std::array<float, kBatchSize * kModelDim> ffw_out;
  std::array<float, kBatchSize * TConfig::kVocabSize> logits;

  // For bf16/f32 vectors * bf16 matrix: faster to unpack once beforehand, into
  // per-thread storage.
  std::array<float, kModelDim * kMaxThreads> even_odd;

  // Griffin layer internal activations
  static constexpr size_t kGriffinDim =
      TConfig::kGriffinLayers > 0 ? kModelDim : 0;
  std::array<float, kBatchSize * kGriffinDim> griffin_x;
  std::array<float, kBatchSize * kGriffinDim> griffin_y;
  std::array<float, kBatchSize * kGriffinDim> griffin_gate_x;
  std::array<float, kBatchSize * kGriffinDim> griffin_multiplier;
};

template <typename TConfig>
struct AllocateState {
  void operator()(ByteStorageT& prefill, ByteStorageT& decode) const {
    // When batching queries, the prefill batch size is reduced by a factor
    // of kBatchedQueryBatchSize
    prefill =
        AllocateSizeof<Activations<TConfig, kMinAdjustedPrefillBatchSize *
                                                kBatchedQueryBatchSize>>();
    decode = AllocateSizeof<
        Activations<TConfig, kDecodeBatchSize * kBatchedQueryBatchSize>>();
  }
};

template <class TConfig, size_t kBatchSize>
Activations<TConfig, kBatchSize>& GetActivations(const ByteStorageT& state_u8) {
  return *reinterpret_cast<Activations<TConfig, kBatchSize>*>(state_u8.get());
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_ACTIVATIONS_H_
