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
#include "compression/compress.h"
#include "util/allocator.h"
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

class MMParallel {
 public:
  MMParallel() : pools_(nullptr) {}
  explicit MMParallel(NestedPools& pools) : pools_(&pools) {}

  NestedPools& Pools() const { return *pools_; }
  hwy::ThreadPool& Pool() const { return pools_->Pool(); }

 private:
  NestedPools* pools_;
};

// Allocations and threads, shared across MatMul calls.
class MatMulEnv {
 public:
  explicit MatMulEnv(NestedPools& pools) : parallel(pools) {
    const size_t N = hwy::VectorBytes() / sizeof(float);
    buf_ = RowVectorBatch<float>(Extents2D(pools.MaxWorkers(), 16 * N));
  }

  RowVectorBatch<float>& Buf() { return buf_; }

  MMParallel parallel;

  // TODO: remove once no longer used.
  NestedPools& Pools() const { return parallel.Pools(); }
  hwy::ThreadPool& Pool() const { return parallel.Pool(); }

 private:
  RowVectorBatch<float> buf_;
};

// Used for the A and B arguments of `MatMul`, which are always const.
// Create via MakeConstMat. This differs from `RowPtr` in that it supports the
// `ofs` required for compressed T.
template <typename T>
struct ConstMat {
  ConstMat(const T* ptr, Extents2D extents, size_t ofs = 0)
      : ptr(ptr), extents(extents), ofs(ofs) {
    HWY_DASSERT(ptr != nullptr);
  }
  // TODO: support stride for page alignment.
  size_t Row(size_t r) const {
    if constexpr (HWY_IS_DEBUG_BUILD) {
      if (r >= extents.rows) {
        HWY_ABORT("ConstMat::Row %zu out of bounds %zu", r, extents.rows);
      }
    }
    return ofs + extents.cols * r;
  }

  const Extents2D& Extents() const { return extents; }
  size_t Stride() const { return extents.cols; }

  // Shrinks the row-extent of this matrix view, i.e. reduces the view to a
  // subrange of the original rows starting at row 0.
  void ShrinkRows(size_t rows) {
    HWY_ASSERT(rows <= extents.rows);
    extents.rows = rows;
  }

  const T* HWY_RESTRICT ptr;
  Extents2D extents;

  // `scale` allows expanding the smaller range of `SfpStream` to the original
  // values. MatFromWeights sets this from `MatPtr`.
  float scale = 1.0f;

  // Offset to add to `ptr`; separate because T=NuqStream does not support
  // pointer arithmetic.
  size_t ofs;
};

// For deducing T.
template <typename T>
ConstMat<T> MakeConstMat(T* HWY_RESTRICT ptr, Extents2D extents,
                         size_t ofs = 0) {
  return ConstMat<T>(ptr, extents, ofs);
}

// For A argument to MatMul (activations).
template <typename T>
ConstMat<T> ConstMatFromBatch(size_t batch_size,
                              const RowVectorBatch<T>& row_vectors) {
  HWY_DASSERT(batch_size <= row_vectors.BatchSize());
  return MakeConstMat(const_cast<T*>(row_vectors.Const()),
                      Extents2D(batch_size, row_vectors.Cols()));
}

template <typename T>
ConstMat<T> ConstMatFromWeights(const MatPtrT<T>& m, size_t ofs = 0) {
  ConstMat<T> mat = MakeConstMat(const_cast<T*>(m.data()), m.Extents(), ofs);
  mat.scale = m.scale();
  return mat;
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_OPS_MATMUL_H_
