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

#include <random>
#include <vector>

#include "util/threading_context.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
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
    const uint8_t* from_row = from.RowT<uint8_t>(r);
    uint8_t* to_row = to.RowT<uint8_t>(r);
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
  const size_t row_bytes = mat.Cols() * mat.ElementBytes();
  for (size_t r = 0; r < mat.Rows(); ++r) {
    hwy::ZeroBytes(mat.RowT<uint8_t>(r), row_bytes);
  }
}

void RandInit(MatPtr& mat, float stddev, std::mt19937& gen) {
  PROFILER_FUNC;
  HWY_ASSERT_M(mat.HasPtr(), mat.Name());
  // Only generates float/double for use by backprop/.
  HWY_ASSERT(mat.GetType() == Type::kF32 || mat.GetType() == Type::kF64);
  mat.SetScale(1.0f);

  std::normal_distribution<float> dist(0.0, stddev);
  if (mat.GetType() == Type::kF32) {
    for (size_t r = 0; r < mat.Rows(); ++r) {
      float* HWY_RESTRICT row = mat.RowT<float>(r);
      for (size_t c = 0; c < mat.Cols(); ++c) {
        row[c] = dist(gen);
      }
    }
  } else {
    for (size_t r = 0; r < mat.Rows(); ++r) {
      double* HWY_RESTRICT row = mat.RowT<double>(r);
      for (size_t c = 0; c < mat.Cols(); ++c) {
        row[c] = dist(gen);
      }
    }
  }
}

// Returns `num` rounded up to an odd number of cache lines. This would also
// prevent 4K aliasing and is coprime with the cache associativity, which
// might reduce conflict misses, but we instead use `StrideForCyclicOffsets`.
static size_t RoundUpToOddLines(size_t num, size_t line_bytes,
                                size_t element_bytes) {
  HWY_DASSERT(line_bytes >= 32);
  HWY_DASSERT(line_bytes % element_bytes == 0);
  const size_t lines = hwy::DivCeil(num * element_bytes, line_bytes);
  const size_t padded_num = (lines | 1) * line_bytes / element_bytes;
  HWY_DASSERT(padded_num >= num);
  return padded_num;
}

static size_t Stride(const Allocator2& allocator, const MatPtr& mat,
                     MatPadding padding) {
  switch (padding) {
    case MatPadding::kPacked:
    default:
      return mat.Cols();
    case MatPadding::kOdd:
      return RoundUpToOddLines(mat.Cols(), allocator.LineBytes(),
                               mat.ElementBytes());
    case MatPadding::kCyclic:
      return StrideForCyclicOffsets(
          mat.Cols(), allocator.QuantumBytes() / mat.ElementBytes());
  }
}

void MatOwner::AllocateFor(MatPtr& mat, MatPadding padding) {
  if (mat.GetType() == Type::kNUQ) padding = MatPadding::kPacked;
  const Allocator2& allocator = ThreadingContext2::Get().allocator;
  const size_t stride = Stride(allocator, mat, padding);
  const size_t num = mat.Rows() * stride;
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

void MatOwners::AllocateFor(const std::vector<MatPtr*>& mats,
                            MatPadding padding, hwy::ThreadPool& pool) {
  const size_t start = owners_.size();
  owners_.resize(start + mats.size());

  // Allocate in parallel because faulting in large tensors is slow.
  pool.Run(0, mats.size(), [&](uint64_t task, size_t /*thread*/) {
    owners_[start + task].AllocateFor(*mats[task], padding);
  });
}

}  // namespace gcpp
