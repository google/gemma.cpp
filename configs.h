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

// Model configurations

#ifndef THIRD_PARTY_GEMMA_CPP_CONFIGS_H_
#define THIRD_PARTY_GEMMA_CPP_CONFIGS_H_

#include <cstddef>

namespace gcpp {

static constexpr size_t kSeqLen = 7168;

struct ConfigGemma7B {
  // NOLINTBEGIN(google3-readability-class-member-naming)
  static constexpr int seq_len = kSeqLen;
  static constexpr int vocab_size = 256128;
  static constexpr int n_layers = 28;
  static constexpr int dim_model = 3072;
  static constexpr int dim_ffw_hidden = 16 * 3072 / 2;  // = 24576
  static constexpr int n_heads = 16;
  static constexpr int n_kv_heads = 16;  // standard MHA, no GQA or MQA
  static constexpr int dim_qkv = 256;    // query size == key size == value size
  static constexpr int top_k = 1;
  // NOLINTEND(google3-readability-class-member-naming)
};

struct ConfigGemma2B {
  // NOLINTBEGIN(google3-readability-class-member-naming)
  static constexpr int seq_len = kSeqLen;
  static constexpr int vocab_size = 256128;
  static constexpr int n_layers = 18;
  static constexpr int dim_model = 2048;
  static constexpr int dim_ffw_hidden = 16 * 2048 / 2;  // = 16384
  static constexpr int n_heads = 8;
  static constexpr int n_kv_heads = 8;  // TODO(austinvhuang): add MQA support
  static constexpr int dim_qkv = 256;   // query size == key size == value size
  static constexpr int top_k = 1;
  // NOLINTEND(google3-readability-class-member-naming)
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_CONFIGS_H_
