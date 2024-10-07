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

#include "tbb/global_control.h"
#include "util/allocator.h"  // RowVectorBatch
#include "util/threading.h"  // PerClusterPools
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/per_target.h"
#include "third_party/intel_dnnl/include/oneapi/dnnl/dnnl.hpp"

using namespace dnnl;
using namespace tbb;

namespace gcpp {

// Bundles ptr/size/stride arguments to simplify MatMul call sites. T can be
// const or non-const. Create via ConstMat/MutableMat.
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
  explicit MatMulEnv(PerClusterPools& pools) : pools_(&pools) {
    const size_t num_lp = pools.NumLP();
    const size_t NF = hwy::VectorBytes() / sizeof(float);
    buf_ = RowVectorBatch<float>(num_lp, 16 * NF);
    setenv("ONEDNN_MAX_CPU_ISA", "AVX512_CORE_AMX", 1);
    // Enable verbose logging for dnnl when we need to debug.
    // setenv("DNNL_VERBOSE", "2", 2);
    tbb::global_control global_limit(
        tbb::global_control::max_allowed_parallelism, 128);
    // Create execution dnnl::engine.
    engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    // Create dnnl::stream.
    engine_stream = dnnl::stream(engine);
  }
  dnnl::stream engine_stream;
  dnnl::engine engine;
  float* HWY_RESTRICT Buf(size_t lp) { return buf_.Batch(lp); }
  PerClusterPools& Pools() const { return *pools_; }
  hwy::ThreadPool& Pool() const { return pools_->Inner(0); }

 private:
  RowVectorBatch<float> buf_;
  PerClusterPools* pools_;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_OPS_MATMUL_H_
