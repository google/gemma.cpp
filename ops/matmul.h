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
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
// IWYU pragma: end_exports

#include "hwy/per_target.h"    // VectorBytes

namespace gcpp {

// TODO: remove deprecated typedef.
using Range1D = IndexRange;

// The MatMul result C[r,c] is Dot(A.Row(r), B.Col(c)). To reduce the number of
// loads, we reuse the same A row for several B columns, which are also loaded
// once for several rows of C. Thus we produce one 'tile' of C at a time of
// dimensions `kRegRows` x `kRegCols`. The Reg naming is because these are
// limited by the number of registers: 32 for NEON/SVE/AVX-512. `kRegCols` == 4
// enables the `StoreInterleaved4` transpose in `StoreHorizontalSums`. We assume
// and verify that `C.Cols() % kRegCols == 0`.
constexpr size_t kRegCols = 4;

// Choosing `kRegRows == kRegCols` minimizes the ratio of loads to FMA, because
// we load `kRegCols + kRegRows` vectors per `kRegRows * kRegCols` element tile.
// In general, `batch_size` (A/C rows) is not a multiple of `kRegRows`. Thus
// functions that load or store a tile are parameterized on `kRowsPerTile`:
// usually `kRegRows`, but `batch_size % kRegRows` on the last row (if != 0).
constexpr size_t kRegRows = kRegCols;

struct CacheSizes {
  CacheSizes() = default;
  CacheSizes(const BoundedTopology::Cluster& cluster) {
    // Assumes each package and cluster has the same cache sizes, and uses
    // reasonable defaults if unknown.
    l1_bytes = 32 * 1024;  // typical size, rarely changes
    l2_bytes = (cluster.PrivateKiB() ? cluster.PrivateKiB() : 256) * 1024;
    l3_bytes = (cluster.SharedKiB() ? cluster.SharedKiB() : 1024) * 1024;
  }

  size_t l1_bytes;
  size_t l2_bytes;
  size_t l3_bytes;
};

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
