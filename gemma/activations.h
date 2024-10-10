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

#include <cmath>

#include "compression/shared.h"  // BF16
#include "ops/matmul.h"          // MatMulEnv
#include "util/allocator.h"      // RowVectorBatch
#include "util/threading.h"
#include "hwy/base.h"  // HWY_DASSERT
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

struct Activations {
  RowVectorBatch<float> x;  // input
  RowVectorBatch<float> q;  // query, also KV if MHA.
  RowVectorBatch<float> logits;

  // Attention
  RowVectorBatch<float> pre_att_rms_out;
  RowVectorBatch<float> att;      // attention vector
  RowVectorBatch<float> att_out;  // attention output
  // Accumulation of attention outputs over heads
  RowVectorBatch<float> att_sums;

  // Gated FFW
  RowVectorBatch<BF16> bf_pre_ffw_rms_out;
  RowVectorBatch<float> C1;
  RowVectorBatch<float> C2;
  RowVectorBatch<float> ffw_out;

  // Griffin
  RowVectorBatch<float> griffin_x;
  RowVectorBatch<float> griffin_y;
  RowVectorBatch<float> griffin_gate_x;
  RowVectorBatch<float> griffin_multiplier;

  // Rope
  RowVectorBatch<float> inv_timescale;

  MatMulEnv env;

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
  static RowVectorBatch<float> CreateInvTimescale() {
    constexpr size_t kQKVDim = TConfig::kQKVDim;
    const size_t rope_dim = TConfig::kUseHalfRope ? kQKVDim / 2 : kQKVDim;
    RowVectorBatch<float> inv_timescale(1, rope_dim / 2);
    for (size_t dim = 0; dim < rope_dim / 2; ++dim) {
      const float freq_exponents =
          static_cast<float>(2 * dim) / static_cast<float>(rope_dim);
      // Replacing with expf(ln(1E4) * freq_exponents) changes results
      // noticeably.
      inv_timescale.Batch(0)[dim] = 1.0f / std::pow(10000.0f, freq_exponents);
    }
    return inv_timescale;
  }

  template <class TConfig>
  void Allocate(size_t batch_size, PerClusterPools& pools) {
    constexpr size_t kModelDim = TConfig::kModelDim;
    constexpr size_t kQKVDim = TConfig::kQKVDim;
    constexpr size_t kHeads = TConfig::kHeads;
    constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;
    constexpr size_t kVocabSize = TConfig::kVocabSize;
    constexpr size_t kSeqLen = TConfig::kSeqLen;
    constexpr size_t kGriffinLayers = TConfig::kGriffinLayers;

    x = RowVectorBatch<float>(batch_size, kModelDim);
    q = RowVectorBatch<float>(batch_size, kHeads * QStride<TConfig>());
    if constexpr (kVocabSize > 0) {
     logits = RowVectorBatch<float>(batch_size, kVocabSize);
    }

    pre_att_rms_out = RowVectorBatch<float>(batch_size, kModelDim);
    att = RowVectorBatch<float>(batch_size, kHeads * kSeqLen);
    att_out = RowVectorBatch<float>(batch_size, kHeads * kQKVDim);
    att_sums = RowVectorBatch<float>(batch_size, kModelDim);

    bf_pre_ffw_rms_out = RowVectorBatch<BF16>(batch_size, kModelDim);
    C1 = RowVectorBatch<float>(batch_size, kFFHiddenDim);
    C2 = RowVectorBatch<float>(batch_size, kFFHiddenDim);
    ffw_out = RowVectorBatch<float>(batch_size, kModelDim);

    if constexpr (kGriffinLayers > 0) {
      griffin_x = RowVectorBatch<float>(batch_size, kModelDim);
      griffin_y = RowVectorBatch<float>(batch_size, kModelDim);
      griffin_gate_x = RowVectorBatch<float>(batch_size, kModelDim);
      griffin_multiplier = RowVectorBatch<float>(batch_size, kModelDim);
    }

    inv_timescale = CreateInvTimescale<TConfig>();

    env = MatMulEnv(pools);
  }
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_ACTIVATIONS_H_
