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

#include "gemma/common.h"  // kMaxThreads - TODO: remove
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"  // HWY_DASSERT

namespace gcpp {

// Owns dynamically-allocated aligned memory for a batch of row vectors.
// This can be seen as a (batch_size x len) matrix.
template <typename T>
class RowVectorBatch {
 public:
  // Default ctor for Activations ctor.
  RowVectorBatch() : batch_size_(0), len_(0) {}
  // Main ctor, called from Activations::Allocate.
  RowVectorBatch(size_t batch_size, size_t len)
      : batch_size_(batch_size), len_(len) {
    mem_ = hwy::AllocateAligned<T>(batch_size * len);
  }

  // Move-only
  RowVectorBatch(RowVectorBatch&) noexcept = delete;
  RowVectorBatch& operator=(RowVectorBatch&) noexcept = delete;
  RowVectorBatch(RowVectorBatch&&) noexcept = default;
  RowVectorBatch& operator=(RowVectorBatch&&) noexcept = default;

  size_t BatchSize() const { return batch_size_; }
  size_t Len() const { return len_; }

  // Returns the given row vector of length `Len()`.
  T* Batch(size_t batch_idx) {
    HWY_DASSERT(batch_idx < batch_size_);
    return mem_.get() + batch_idx * len_;
  }

  // For MatMul or other operations that process the entire batch at once.
  T* All() { return mem_.get(); }
  size_t NumBytes() const { return batch_size_ * len_ * sizeof(T); }

 private:
  hwy::AlignedFreeUniquePtr<T[]> mem_;
  size_t batch_size_;  // rows in the matrix
  size_t len_;         // columns in the matrix = vector length
};

struct Activations {
  RowVectorBatch<float> x;  // input
  RowVectorBatch<float> q;  // query, also KV if MHA.
  RowVectorBatch<float> logits;

  // Attention
  RowVectorBatch<float> pre_att_rms_out;
  RowVectorBatch<float> att;      // attention vector
  RowVectorBatch<float> att_out;  // attention output
  // After linear transformation, shared by all heads
  RowVectorBatch<float> att_post1;
  // Accumulation of attention outputs over heads
  RowVectorBatch<float> att_post2;

  // Gated FFW
  RowVectorBatch<hwy::bfloat16_t> bf_pre_ffw_rms_out;
  RowVectorBatch<float> C1;
  RowVectorBatch<float> C2;
  RowVectorBatch<float> ffw_out;

  // Griffin
  RowVectorBatch<float> griffin_x;
  RowVectorBatch<float> griffin_y;
  RowVectorBatch<float> griffin_gate_x;
  RowVectorBatch<float> griffin_multiplier;

  // For bf16/f32 vectors * bf16 matrix: faster to unpack once beforehand, into
  // per-thread storage.
  // TODO: remove once MatVec is gone.
  RowVectorBatch<float> even_odd;

  // Multi-Head Attention?
  template <class TConfig>
  static constexpr bool IsMHA() {
    return TConfig::kHeads == TConfig::kKVHeads;
  }

  // Stride between subsequent queries. Each of Q, K, V are of length kQKVDim,
  // but for MHA we store them as Q,K,V, Q,K,V, .. instead of Q..Q, K..K, V..V.
  template <class TConfig>
  static constexpr size_t QStride() {
    return TConfig::kQKVDim * (IsMHA<TConfig>() ? 3 : 1);
  }

  template <class TConfig>
  void Allocate(size_t batch_size) {
    constexpr size_t kModelDim = TConfig::kModelDim;
    constexpr size_t kQKVDim = TConfig::kQKVDim;
    constexpr size_t kHeads = TConfig::kHeads;
    constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;
    constexpr size_t kVocabSize = TConfig::kVocabSize;
    constexpr size_t kSeqLen = TConfig::kSeqLen;
    constexpr size_t kGriffinLayers = TConfig::kGriffinLayers;

    x = RowVectorBatch<float>(batch_size, kModelDim);
    q = RowVectorBatch<float>(batch_size, kHeads * QStride<TConfig>());
    logits = RowVectorBatch<float>(batch_size, kVocabSize);

    pre_att_rms_out = RowVectorBatch<float>(batch_size, kModelDim);
    att = RowVectorBatch<float>(batch_size, kHeads * kSeqLen);
    att_out = RowVectorBatch<float>(batch_size, kHeads * kQKVDim);
    att_post1 = RowVectorBatch<float>(1, kModelDim);
    att_post2 = RowVectorBatch<float>(batch_size, kModelDim);

    bf_pre_ffw_rms_out = RowVectorBatch<hwy::bfloat16_t>(batch_size, kModelDim);
    C1 = RowVectorBatch<float>(batch_size, kFFHiddenDim);
    C2 = RowVectorBatch<float>(batch_size, kFFHiddenDim);
    ffw_out = RowVectorBatch<float>(batch_size, kModelDim);

    if (kGriffinLayers > 0) {
      griffin_x = RowVectorBatch<float>(batch_size, kModelDim);
      griffin_y = RowVectorBatch<float>(batch_size, kModelDim);
      griffin_gate_x = RowVectorBatch<float>(batch_size, kModelDim);
      griffin_multiplier = RowVectorBatch<float>(batch_size, kModelDim);
    }

    even_odd = RowVectorBatch<float>(1, kModelDim * kMaxThreads);
  }
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_ACTIVATIONS_H_
