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

#ifndef THIRD_PARTY_GEMMA_CPP_OPS_MATMUL_H_
#define THIRD_PARTY_GEMMA_CPP_OPS_MATMUL_H_

#include <stddef.h>

// IWYU pragma: begin_exports
#include "util/basics.h"
#include "util/threading.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
// IWYU pragma: end_exports

#include "hwy/per_target.h"    // VectorBytes

namespace gcpp {

// Allocations and threads, shared across MatMul calls.
class MatMulEnv {
 public:
  MatMulEnv() : pools_(nullptr) {}
  explicit MatMulEnv(NestedPools& pools) : pools_(&pools) {
    const size_t N = hwy::VectorBytes() / sizeof(float);
    buf_ = RowVectorBatch<float>(Extents2D(pools.MaxWorkers(), 16 * N));
  }

  RowVectorBatch<float>& Buf() { return buf_; }
  NestedPools& Pools() const { return *pools_; }
  hwy::ThreadPool& Pool() const { return pools_->Pool(); }

 private:
  RowVectorBatch<float> buf_;
  NestedPools* pools_;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_OPS_MATMUL_H_
