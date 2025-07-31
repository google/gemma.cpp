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

#ifndef THIRD_PARTY_GEMMA_CPP_OPS_OPS_H_
#define THIRD_PARTY_GEMMA_CPP_OPS_OPS_H_

#include <stddef.h>

#include <cmath>

#include "util/mat.h"
#include "hwy/base.h"

namespace gcpp {

static inline HWY_MAYBE_UNUSED MatStorageT<float> CreateInvTimescale(
    const Allocator& allocator, size_t qkv_dim, bool half_rope,
    double base_frequency = 10000.0) {
  const size_t rope_dim = half_rope ? qkv_dim / 2 : qkv_dim;
  MatStorageT<float> inv_timescale("inv_timescale", rope_dim / 2, allocator);
  for (size_t dim = 0; dim < rope_dim / 2; ++dim) {
    const double freq_exponents =
        static_cast<double>(2 * dim) / static_cast<double>(rope_dim);
    // Replacing with expf(ln(1E4) * freq_exponents) changes results
    // noticeably.
    inv_timescale.Row(0)[dim] =
        static_cast<float>(1.0 / std::pow(base_frequency, freq_exponents));
  }
  return inv_timescale;
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_OPS_OPS_H_
