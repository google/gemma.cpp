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
#include "util/threading.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
// IWYU pragma: end_exports

#include "util/allocator.h"  // RowVectorBatch
#include "hwy/per_target.h"    // VectorBytes

namespace gcpp {

// Bundles ptr/size/stride arguments to simplify MatMul call sites. T can be
// const or non-const. Create via ConstMat/MutableMat.
// TODO(rays): Replace with MatPtr and get rid of stride, which is only != cols
// in one place.
template <typename T>
struct Mat {
  bool NotEmpty() const {
    return ptr != nullptr && cols != 0 && stride >= cols;
  }
  size_t Row(size_t r) const { return ofs + stride * r; }

  T* HWY_RESTRICT ptr;
  size_t cols;

  // elements between rows, which is typically the same as `cols`.
  size_t stride;

  // Offset to add to `ptr`; separate because T=NuqStream does not support
  // pointer arithmetic.
  size_t ofs;
};

template <typename T>
Mat<T> MutableMat(T* HWY_RESTRICT ptr, size_t cols, size_t stride,
                  size_t ofs = 0) {
  return Mat<T>{.ptr = ptr, .cols = cols, .stride = stride, .ofs = ofs};
}

template <typename T>
Mat<const T> ConstMat(const T* HWY_RESTRICT ptr, size_t cols, size_t stride,
                      size_t ofs = 0) {
  return Mat<const T>{.ptr = ptr, .cols = cols, .stride = stride, .ofs = ofs};
}

template <typename T>
Mat<const T> ConstMat(Mat<T> mat) {
  return ConstMat(mat.ptr, mat.cols, mat.stride, mat.ofs);
}

template <typename T>
Mat<T> MutableMat(T* HWY_RESTRICT ptr, size_t cols) {
  return MutableMat(ptr, cols, cols);
}

template <typename T>
Mat<const T> ConstMat(const T* HWY_RESTRICT ptr, size_t cols) {
  return ConstMat(ptr, cols, cols);
}

// Allocations and threads, shared across MatMul calls.
class MatMulEnv {
 public:
  MatMulEnv() : pools_(nullptr) {}
  explicit MatMulEnv(NestedPools& pools) : pools_(&pools) {
    const size_t N = hwy::VectorBytes() / sizeof(float);
    buf_ = RowVectorBatch<float>(pools.MaxWorkers(), 16 * N);
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
