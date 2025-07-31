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

#include "util/mat.h"

#include <stddef.h>
#include <stdint.h>

#include "util/threading_context.h"
#include "hwy/base.h"
#include "hwy/per_target.h"  // VectorBytes
#include "hwy/profiler.h"

namespace gcpp {

void CopyMat(const MatPtr& from, MatPtr& to) {
  PROFILER_FUNC;
  HWY_ASSERT_M(from.HasPtr() && to.HasPtr(), to.Name());
  HWY_ASSERT(to.Rows() == from.Rows() && to.Cols() == from.Cols());
  HWY_ASSERT(to.GetType() == from.GetType());
  to.SetScale(from.Scale());

  if (to.IsPacked() && from.IsPacked()) {
    HWY_ASSERT(to.PackedBytes() == from.PackedBytes());
    hwy::CopyBytes(from.Packed(), to.Packed(), to.PackedBytes());
    return;
  }
  const size_t row_bytes = to.Cols() * to.ElementBytes();
  for (size_t r = 0; r < to.Rows(); ++r) {
    const uint8_t* from_row = from.RowBytes(r);
    uint8_t* to_row = to.RowBytes(r);
    hwy::CopyBytes(from_row, to_row, row_bytes);
  }
}

void ZeroInit(MatPtr& mat) {
  PROFILER_FUNC;
  HWY_ASSERT_M(mat.HasPtr(), mat.Name());
  mat.SetScale(1.0f);

  if (mat.IsPacked()) {
    hwy::ZeroBytes(mat.Packed(), mat.PackedBytes());
    return;
  }
  // Also zero-initialize padding (required by MatMul).
  hwy::ZeroBytes(mat.RowBytes(0),
                 mat.Stride() * mat.ElementBytes() * mat.Rows());
}

size_t Stride(MatPadding padding, size_t cols, size_t element_bytes,
              size_t line_bytes) {
  switch (padding) {
    case MatPadding::kPacked:
    default:
      return cols;
    case MatPadding::kOdd: {
      // Round up to an odd number of cache lines to prevent 4K aliasing and
      // reduce conflict misses (coprime with the cache associativity).
      HWY_DASSERT(line_bytes >= 32);
      HWY_DASSERT(line_bytes % element_bytes == 0);
      const size_t lines = hwy::DivCeil(cols * element_bytes, line_bytes);
      const size_t padded_cols = (lines | 1) * line_bytes / element_bytes;
      HWY_DASSERT(padded_cols >= cols);
      return padded_cols;
    }
  }
}

void MatOwner::AllocateFor(MatPtr& mat, const Allocator& allocator,
                           MatPadding padding) {
  const bool is_nuq = mat.GetType() == Type::kNUQ;
  if (is_nuq) padding = MatPadding::kPacked;
  const size_t stride =
      Stride(padding, mat.Cols(), mat.ElementBytes(), allocator.LineBytes());
  const size_t num = is_nuq ? mat.PackedBytes() : mat.Rows() * stride;
  // `compress-inl` requires up to 2 BF16 vectors of padding. `MatPadding`
  // might not be enough, hence add extra. `MatT` is at least one byte, which
  // is half of BF16, hence adding `VectorBytes` *elements* is enough.
  const size_t bytes = (num + hwy::VectorBytes()) * mat.ElementBytes();
  // Allow binding the entire matrix.
  const size_t padded_bytes =
      hwy::RoundUpTo(bytes, allocator.QuantumBytes() / mat.ElementBytes());
  storage_ = allocator.AllocBytes(padded_bytes);
  mat.SetPtr(storage_.get(), stride);
}

}  // namespace gcpp
