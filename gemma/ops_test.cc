// Copyright 2023 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS HWY_SCALAR
#endif

#include <stddef.h>
#include <stdio.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <vector>

#include "compression/compress.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/ops_test.cc"  //NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"
// After highway.h
#include "gemma/ops.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <class Test>
struct ForeachCountAndMisalign {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) const {
    hwy::RandomState rng;
    const size_t N = Lanes(d);
    const size_t misalignments[3] = {0, N / 4, 3 * N / 5};

    for (size_t count = 0; count < 2 * N; ++count) {
      for (size_t ma : misalignments) {
        for (size_t mb : misalignments) {
          Test()(d, count, ma, mb, rng);
        }
      }
    }
  }
};

template <typename T>
T Random(hwy::RandomState& rng) {
  const int32_t bits = static_cast<int32_t>(Random32(&rng)) & 1023;
  const double val = (bits - 512) / 64.0;
  // Clamp negative to zero for unsigned types.
  return hwy::ConvertScalarTo<T>(
      HWY_MAX(hwy::ConvertScalarTo<double>(hwy::LowestValue<T>()), val));
}

HWY_NOINLINE void SourceAddFrom(const float* HWY_RESTRICT other,
                                float* HWY_RESTRICT x, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    x[i] += other[i];
  }
}

HWY_NOINLINE void SourceMulBy(const float* HWY_RESTRICT other,
                              float* HWY_RESTRICT x, size_t size,
                              size_t max_pos) {
  HWY_DASSERT(max_pos <= size);
  for (size_t i = 0; i < max_pos; ++i) {
    x[i] *= other[i];
  }
}

HWY_NOINLINE void SourceMulByConst(float c, float* HWY_RESTRICT x, size_t size,
                                   size_t max_pos) {
  for (size_t i = 0; i < max_pos; ++i) {
    x[i] *= c;
  }
}

HWY_NOINLINE void SourceMulByConstAndAdd(float c, const float* HWY_RESTRICT x,
                                         float* HWY_RESTRICT out, size_t size,
                                         size_t max_pos) {
  for (size_t i = 0; i < max_pos; ++i) {
    out[i] += x[i] * c;
  }
}

HWY_NOINLINE void SourceSoftmax(float* HWY_RESTRICT x, size_t size,
                                size_t mask_pos) {
  HWY_DASSERT(size != 0);
  HWY_DASSERT(mask_pos <= size);
  float sum = 0.0;
  const float maxval = *std::max_element(x, x + mask_pos);
  for (size_t i = 0; i < mask_pos; ++i) {
    x[i] = std::exp(x[i] - maxval);
    sum += x[i];
  }
  const float scale = 1.0f / sum;
  for (size_t i = 0; i < mask_pos; ++i) {
    x[i] *= scale;
  }
}

template <size_t k>
HWY_NOINLINE std::discrete_distribution<int> SourceCreateDistribution(
    std::array<float, k>& top_k, float temperature) {
  // re-normalize distribution
  for (size_t i = 0; i < k; ++i) {
    top_k[i] = exp(log(top_k[i]) / temperature);
  }
  float denominator = 0.0f;
  for (size_t i = 0; i < k; ++i) {
    denominator += top_k[i];
  }
  denominator = 1.0f / denominator;
  MulByConst(denominator, top_k.data(), k);
  return std::discrete_distribution<int>(std::begin(top_k), std::end(top_k));
}

struct TestAddFrom {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t misalign_b,
                  hwy::RandomState& rng) {
    using T = hn::TFromD<D>;

    hwy::AlignedFreeUniquePtr<T[]> px =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> pe =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> po =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_b + count));
    HWY_ASSERT(px && pe && po);

    T* x = px.get() + misalign_a;
    T* e = pe.get() + misalign_a;
    T* o = po.get() + misalign_b;

    for (size_t i = 0; i < count; ++i) {
      x[i] = Random<T>(rng);
      e[i] = x[i];
      o[i] = Random<T>(rng);
    }

    SourceAddFrom(o, e, count);
    AddFrom(o, x, count);

    hwy::AssertArraySimilar(e, x, count, hwy::TargetName(HWY_TARGET), __FILE__,
                            __LINE__);
  }
};

struct TestMulBy {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t misalign_b,
                  hwy::RandomState& rng) {
    using T = hn::TFromD<D>;

    hwy::AlignedFreeUniquePtr<T[]> px =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> pe =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> po =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_b + count));
    HWY_ASSERT(px && pe && po);

    T* x = px.get() + misalign_a;
    T* e = pe.get() + misalign_a;
    T* o = po.get() + misalign_b;

    for (size_t i = 0; i < count; ++i) {
      x[i] = Random<T>(rng);
      e[i] = x[i];
      o[i] = Random<T>(rng);
    }

    SourceMulBy(o, e, count, count);
    MulBy(o, x, count, count);

    hwy::AssertArraySimilar(e, x, count, hwy::TargetName(HWY_TARGET), __FILE__,
                            __LINE__);
  }
};

struct TestMulByConstAndAdd {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t misalign_b,
                  hwy::RandomState& rng) {
    using T = hn::TFromD<D>;

    hwy::AlignedFreeUniquePtr<T[]> px =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> pe =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> po =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_b + count));
    HWY_ASSERT(px && pe && po);

    T* x = px.get() + misalign_a;
    T* e = pe.get() + misalign_a;
    T* o = po.get() + misalign_b;

    for (size_t i = 0; i < count; ++i) {
      x[i] = Random<T>(rng);
      e[i] = x[i];
      o[i] = Random<T>(rng);
    }
    T constant = Random<T>(rng);

    SourceMulByConstAndAdd(constant, o, e, count, count);
    MulByConstAndAdd(constant, o, x, count, count);

    hwy::AssertArraySimilar(e, x, count, hwy::TargetName(HWY_TARGET), __FILE__,
                            __LINE__);
  }
};

struct TestMulByConst {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t misalign_b,
                  hwy::RandomState& rng) {
    if (misalign_b == 0) return;
    using T = hn::TFromD<D>;

    hwy::AlignedFreeUniquePtr<T[]> px =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> pe =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    HWY_ASSERT(px && pe);

    T* x = px.get() + misalign_a;
    T* e = pe.get() + misalign_a;

    for (size_t i = 0; i < count; ++i) {
      x[i] = Random<T>(rng);
      e[i] = x[i];
    }
    T constant = Random<T>(rng);

    SourceMulByConst(constant, e, count, count);
    MulByConst(constant, x, count, count);

    hwy::AssertArraySimilar(e, x, count, hwy::TargetName(HWY_TARGET), __FILE__,
                            __LINE__);
  }
};

struct TestSoftmax {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t misalign_b,
                  hwy::RandomState& rng) {
    if (count == 0) return;  // *Softmax would assert
    if (misalign_b == 0) return;
    using T = hn::TFromD<D>;

    hwy::AlignedFreeUniquePtr<T[]> px =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    hwy::AlignedFreeUniquePtr<T[]> pe =
        hwy::AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    HWY_ASSERT(px && pe);

    T* x = px.get() + misalign_a;
    T* e = pe.get() + misalign_a;

    for (size_t i = 0; i < count; ++i) {
      x[i] = Random<T>(rng);
      e[i] = x[i];
    }

    SourceSoftmax(e, count, count);
    Softmax(x, count, count);

    T sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
      sum += x[i];
      double rel = std::abs(x[i] - e[i]) / e[i];
      ASSERT_LT(rel, 1e-6) << "Mismatch on coordinate " << i << " out of "
                           << count;
    }
    ASSERT_NEAR(sum, 1.0, 1e-6);
  }
};

template <size_t k>
struct TestCreateDistribution {
  void operator()(hwy::RandomState& rng) {
    std::array<float, k> x;
    std::array<float, k> e;

    for (size_t i = 0; i < k; ++i) {
      x[i] = Random<float>(rng);
      e[i] = x[i];
    }
    const float constant = Random<float>(rng);
    auto expected = SourceCreateDistribution(e, constant);
    auto output = create_distribution(x, constant);

    AssertEqual(expected, output, hwy::TargetName(HWY_TARGET), __FILE__,
                __LINE__);
  }
};

void TestAllAddFrom() {
  hn::ForPartialVectors<ForeachCountAndMisalign<TestAddFrom>>()(float());
}

void TestAllMulBy() {
  hn::ForPartialVectors<ForeachCountAndMisalign<TestMulBy>>()(float());
}

void TestAllMulByConst() {
  hn::ForPartialVectors<ForeachCountAndMisalign<TestMulByConst>>()(float());
}

void TestAllMulByConstAndAdd() {
  hn::ForPartialVectors<ForeachCountAndMisalign<TestMulByConstAndAdd>>()(
      float());
}

void TestAllSoftmax() {
  hn::ForPartialVectors<ForeachCountAndMisalign<TestSoftmax>>()(float());
}

void TestAllCreateDistribution() {
  TestCreateDistribution<2048>();
  TestCreateDistribution<5000>();
}

template <typename MatT, size_t kOuter, size_t kInner>
CompressedArray<MatT, kOuter * kInner> GenerateMat(size_t offset,
                                                   hwy::ThreadPool& pool) {
  gcpp::CompressWorkingSet ws;
  CompressedArray<MatT, kOuter * kInner> mat;
  std::array<float, kOuter * kInner> content;
  const float scale = 1.0f / kInner;
  pool.Run(0, kOuter, [&](const size_t i, size_t /*thread*/) {
    for (size_t j = 0; j < kInner; j++) {
      content[i * kInner + j] =
          static_cast<float>((i * kInner + j + offset) * scale);
    }
  });

  Compress(content, ws, mat, pool);
  mat.set_scale(1.0f);
  return mat;
}

template <typename MatT, size_t kOuter, size_t kInner>
CompressedArray<MatT, kOuter * kInner> GenerateZeroMat(hwy::ThreadPool& pool) {
  gcpp::CompressWorkingSet ws;
  CompressedArray<MatT, kOuter * kInner> mat;
  std::array<MatT, kOuter * kInner> content;

  pool.Run(0, kOuter, [&](const size_t i, size_t thread) {
    hwy::ZeroBytes(&content[i * kInner], kInner * sizeof(content[0]));
  });

  Compress(content, ws, mat, pool);
  mat.set_scale(1.0f);
  return mat;
}

template <typename MatT, size_t kOuter, size_t kInner>
std::unique_ptr<CompressedArray<MatT, kOuter * kInner>> GenerateMatHeap(
    size_t offset, hwy::ThreadPool& pool) {
  gcpp::CompressWorkingSet ws;
  std::unique_ptr<CompressedArray<MatT, kOuter * kInner>> mat =
      std::unique_ptr<CompressedArray<MatT, kOuter * kInner>>(
          new CompressedArray<MatT, kOuter * kInner>);
  hwy::AlignedFreeUniquePtr<float[]> content =
      hwy::AllocateAligned<float>(kOuter * kInner);
  const float scale = 1.875f / (kInner * kOuter + offset);
  pool.Run(0, kOuter, [&](const size_t i, size_t /*thread*/) {
    for (size_t j = 0; j < kInner; j++) {
      content[i * kInner + j] =
          static_cast<float>((i * kInner + j + offset) * scale);
    }
  });

  Compress(content.get(), kOuter * kInner, ws, kOuter * kInner, mat->data(), 0,
           pool);
  mat->set_scale(1.0f);
  return mat;
}

template <typename MatT, size_t kOuter, size_t kInner>
std::unique_ptr<CompressedArray<MatT, kOuter * kInner>>
GenerateTransposeMatHeap(size_t offset, hwy::ThreadPool& pool) {
  gcpp::CompressWorkingSet ws;
  std::unique_ptr<CompressedArray<MatT, kOuter * kInner>> mat =
      std::unique_ptr<CompressedArray<MatT, kOuter * kInner>>(
          new CompressedArray<MatT, kOuter * kInner>);
  hwy::AlignedFreeUniquePtr<float[]> content =
      hwy::AllocateAligned<float>(kOuter * kInner);
  const float scale = 1.875f / (kInner * kOuter + offset);
  pool.Run(0, kOuter, [&](const size_t i, size_t /*thread*/) {
    for (size_t j = 0; j < kInner; j++) {
      content[j * kOuter + i] =
          static_cast<float>((i * kInner + j + offset) * scale);
    }
  });

  Compress(content.get(), kOuter * kInner, ws, kOuter * kInner, mat->data(), 0,
           pool);
  mat->set_scale(1.0f);
  return mat;
}

template <typename MatT, size_t kOuter, size_t kInner>
std::unique_ptr<CompressedArray<MatT, kOuter * kInner>> GenerateZeroMatHeap(
    hwy::ThreadPool& pool) {
  gcpp::CompressWorkingSet ws;
  std::unique_ptr<CompressedArray<MatT, kOuter * kInner>> mat =
      std::unique_ptr<CompressedArray<MatT, kOuter * kInner>>(
          new CompressedArray<MatT, kOuter * kInner>);
  hwy::AlignedFreeUniquePtr<float[]> content =
      hwy::AllocateAligned<float>(kOuter * kInner);

  pool.Run(0, kOuter, [&](const size_t i, size_t thread) {
    hwy::ZeroBytes(&content[i * kInner], kInner * sizeof(content[0]));
  });

  Compress(content.get(), kOuter * kInner, ws, kOuter * kInner, mat->data(), 0,
           pool);
  mat->set_scale(1.0f);
  return mat;
}

template <size_t length>
hwy::AlignedFreeUniquePtr<float[]> GenerateVec(size_t offset) {
  hwy::AlignedFreeUniquePtr<float[]> vec = hwy::AllocateAligned<float>(length);
  HWY_ASSERT(vec);
  for (size_t idx = 0; idx < length; idx++) {
    vec[idx] = static_cast<float>(idx + offset);
  }
  return vec;
}

// A simple matrix multiplication. No optimization / tiling.
template <size_t kM, size_t kN, size_t kK>
hwy::AlignedFreeUniquePtr<float[]> SimpleMatMul(
    const hwy::AlignedFreeUniquePtr<float[]>& a,
    const hwy::AlignedFreeUniquePtr<float[]>& b) {
  hwy::AlignedFreeUniquePtr<float[]> out = hwy::AllocateAligned<float>(kM * kK);
  hwy::ZeroBytes(out.get(), kM * kK * sizeof(float));

  int i, j, k;
  for (i = 0; i < kM; ++i) {
    for (j = 0; j < kK; ++j) {
      for (k = 0; k < kN; ++k) {
        out[i * kK + j] += a[i * kN + k] * b[k * kK + j];
      }
    }
  }

  return out;
}

template <size_t kOuter, size_t kInner>
hwy::AlignedFreeUniquePtr<float[]> SimpleMatVecAdd(
    const CompressedArray<float, kOuter * kInner>& mat,
    const hwy::AlignedFreeUniquePtr<float[]>& vec,
    const hwy::AlignedFreeUniquePtr<float[]>& add) {
  hwy::AlignedFreeUniquePtr<float[]> uncompressed_mat =
      hwy::AllocateAligned<float>(kOuter * kInner);
  hwy::AlignedFreeUniquePtr<float[]> out = hwy::AllocateAligned<float>(kOuter);
  HWY_ASSERT(uncompressed_mat && out);
  Decompress(mat, 0, uncompressed_mat.get(), kOuter * kInner);
  for (size_t idx_row = 0; idx_row < kOuter; idx_row++) {
    out[idx_row] = add[idx_row];
    for (size_t idx_col = 0; idx_col < kInner; idx_col++) {
      out[idx_row] +=
          uncompressed_mat[kInner * idx_row + idx_col] * vec[idx_col];
    }
  }
  return out;
}

template <size_t length>
void AssertClose(const hwy::AlignedFreeUniquePtr<float[]>& a,
                 const hwy::AlignedFreeUniquePtr<float[]>& b) {
  for (size_t idx = 0; idx < length; idx++) {
    const float rel_abs_delta = std::abs(a[idx] - b[idx]) /
                                std::max(std::abs(a[idx]), std::abs(b[idx]));
    EXPECT_LT(rel_abs_delta, 2e-6)
        << "a[" << idx << "]=" << a[idx] << ", b[" << idx << "]=" << b[idx];
  }
}

// Largely unoptimized; reordered innermost loops nets ~5-10X speedup on
// ops_test across instruction sets.
template <size_t kN, size_t kK, typename MatTA, typename MatTB,
          HWY_IF_T_SIZE_GT(MatTB, 1)>
HWY_INLINE void MatMulSlowBatch(size_t batch_size, const MatTA* HWY_RESTRICT a,
                                const MatTB* HWY_RESTRICT b, const float* add,
                                float* HWY_RESTRICT out) {
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t k = 0; k < kN; ++k) {
      for (size_t j = 0; j < kK; ++j) {
        const float a1 = hwy::ConvertScalarTo<float>(a[i * kN + k]);
        const float b1 = hwy::ConvertScalarTo<float>(b[k * kK + j]);
        out[i * kK + j] += a1 * b1;
      }
    }
    if (add != nullptr) {
      for (size_t j = 0; j < kK; ++j) {
        out[i * kK + j] += add[j];
      }
    }
  }
}

// The above overload can handle combinations of f32 and bf16, but this one
// is required for MatTB = {SFP, NUQ}.
template <size_t kN, size_t kK, typename MatTA, typename MatTB,
          HWY_IF_T_SIZE(MatTB, 1)>
HWY_INLINE void MatMulSlowBatch(size_t batch_size, const MatTA* HWY_RESTRICT a,
                                const MatTB* HWY_RESTRICT b_compr,
                                const float* add, float* HWY_RESTRICT out) {
  const hn::ScalableTag<float> d;
  hwy::AlignedFreeUniquePtr<float[]> b = hwy::AllocateAligned<float>(kK * kN);
  CompressTraits<MatTB>::Decompress(d, /*in_capacity=*/0, b_compr, 0, b.get(),
                                    kK * kN);
  MatMulSlowBatch<kN, kK>(batch_size, a, b.get(), add, out);
}

template <size_t kM, size_t kN, size_t kK, bool kAdd, typename MatTA,
          typename MatTB = MatTA>
void TestTiledBatchMatMul() {
  fprintf(stderr,
          "TestTiledBatchMatMul %lu, %lu, %lu, add=%d, MatTA=%s, MatTB=%s\n",
          kM, kN, kK, kAdd, typeid(MatTA).name(), typeid(MatTB).name());
  hwy::ThreadPool pool(3);
  std::unique_ptr<CompressedArray<MatTA, kM * kN>> a =
      GenerateMatHeap<MatTA, kM, kN>(0, pool);
  std::unique_ptr<CompressedArray<MatTB, kN * kK>> b =
      GenerateMatHeap<MatTB, kN, kK>(0, pool);
  std::unique_ptr<CompressedArray<float, kK>> add =
      GenerateMatHeap<float, 1, kK>(0, pool);
  std::unique_ptr<CompressedArray<float, kM * kK>> c_slow =
      GenerateZeroMatHeap<float, kM, kK>(pool);

  const double start_slow = hwy::platform::Now();
  MatMulSlowBatch<kN, kK>(kM, a->data(), b->data(),
                          kAdd ? add->data() : nullptr, c_slow->data());
  const double slow_matmul_seconds = hwy::platform::Now() - start_slow;
  fprintf(stderr, "MatMulSlowBatch took %f seconds.\n", slow_matmul_seconds);

  hwy::AlignedFreeUniquePtr<float[]> c = hwy::AllocateAligned<float>(kM * kK);
  std::unique_ptr<CompressedArray<MatTB, kN * kK>> b_trans =
      GenerateTransposeMatHeap<MatTB, kN, kK>(0, pool);

  const double start_tiled = hwy::platform::Now();
  if (kAdd) {
    MatMul_4x4_Batch_Add<kN, kK, kAdd>(kM, a->data(), b_trans->data(), c.get(),
                                       add->data(), pool);
  } else {
    MatMul_4x4_Batch<kN, kK>(kM, a->data(), b_trans->data(), c.get(), pool);
  }
  const double tiled_matmul_seconds = hwy::platform::Now() - start_tiled;
  fprintf(stderr, "MatMul_4x4_Batch took %f seconds.\n", tiled_matmul_seconds);

  AssertClose(c_slow->data(), c.get(), kM * kK);
}

void TestAllTiledBatchMatMul() {
  using BF16 = hwy::bfloat16_t;
  using F32 = float;
  using SFP = SfpStream;
  // medium-sized square test
  TestTiledBatchMatMul<512, 512, 512, /*kAdd=*/false, F32>();
  TestTiledBatchMatMul<512, 512, 512, /*kAdd=*/true, BF16>();
  TestTiledBatchMatMul<512, 512, 512, /*kAdd=*/false, F32, BF16>();
  TestTiledBatchMatMul<512, 512, 512, /*kAdd=*/true, BF16, F32>();
  TestTiledBatchMatMul<512, 512, 512, /*kAdd=*/false, F32, SFP>();
  TestTiledBatchMatMul<512, 512, 512, /*kAdd=*/true, BF16, SFP>();

  // minimal non-square test. kK must be at least 2 vectors.
  TestTiledBatchMatMul<35, 128, 32, /*kAdd=*/false, F32>();
  TestTiledBatchMatMul<34, 128, 32, /*kAdd=*/true, BF16>();
  TestTiledBatchMatMul<33, 128, 32, /*kAdd=*/false, F32, BF16>();
  TestTiledBatchMatMul<33, 128, 32, /*kAdd=*/true, BF16, F32>();
  TestTiledBatchMatMul<31, 128, 32, /*kAdd=*/false, F32, SFP>();
  TestTiledBatchMatMul<29, 128, 32, /*kAdd=*/true, BF16, SFP>();
  TestTiledBatchMatMul<4, 128, 32, /*kAdd=*/true, F32>();
  TestTiledBatchMatMul<4, 128, 32, /*kAdd=*/false, BF16>();
  TestTiledBatchMatMul<4, 128, 32, /*kAdd=*/true, F32, BF16>();
  TestTiledBatchMatMul<4, 128, 32, /*kAdd=*/false, BF16, F32>();
  TestTiledBatchMatMul<4, 128, 32, /*kAdd=*/true, F32, SFP>();
  TestTiledBatchMatMul<4, 128, 32, /*kAdd=*/false, BF16, SFP>();
  TestTiledBatchMatMul<3, 128, 32, /*kAdd=*/false, F32>();
  TestTiledBatchMatMul<3, 128, 32, /*kAdd=*/true, BF16>();
  TestTiledBatchMatMul<3, 128, 32, /*kAdd=*/false, F32, BF16>();
  TestTiledBatchMatMul<3, 128, 32, /*kAdd=*/true, BF16, F32>();
  TestTiledBatchMatMul<3, 128, 32, /*kAdd=*/false, F32, SFP>();
  TestTiledBatchMatMul<3, 128, 32, /*kAdd=*/true, BF16, SFP>();
  TestTiledBatchMatMul<2, 128, 64, /*kAdd=*/true, F32>();
  TestTiledBatchMatMul<2, 128, 64, /*kAdd=*/false, BF16>();
  TestTiledBatchMatMul<2, 128, 64, /*kAdd=*/true, F32, BF16>();
  TestTiledBatchMatMul<2, 128, 64, /*kAdd=*/false, BF16, F32>();
  TestTiledBatchMatMul<2, 128, 64, /*kAdd=*/true, F32, SFP>();
  TestTiledBatchMatMul<2, 128, 64, /*kAdd=*/false, BF16, SFP>();
  TestTiledBatchMatMul<1, 128, 32, /*kAdd=*/false, F32>();
  TestTiledBatchMatMul<1, 128, 32, /*kAdd=*/true, BF16>();
  TestTiledBatchMatMul<1, 128, 32, /*kAdd=*/false, F32, BF16>();
  TestTiledBatchMatMul<1, 128, 32, /*kAdd=*/true, BF16, F32>();
  TestTiledBatchMatMul<1, 128, 32, /*kAdd=*/false, F32, SFP>();
  TestTiledBatchMatMul<1, 128, 32, /*kAdd=*/true, BF16, SFP>();

  // large-scale test
  // TODO(philculliton): investigate rounding issues with large matrices.
  // Causes test timeout.
  // TestTiledBatchMatMul<512, 24576, 3072, float>();
}

void TestMatVecAdd() {
  hwy::ThreadPool pool(hwy::ThreadPool::MaxThreads());
  constexpr size_t kOuter = 128 * 3;
  constexpr size_t kInner = 128 * 5;
  CompressedArray<float, kOuter * kInner> mat =
      GenerateMat<float, kOuter, kInner>(0, pool);
  hwy::AlignedFreeUniquePtr<float[]> vec = GenerateVec<kInner>(0);
  hwy::AlignedFreeUniquePtr<float[]> add = GenerateVec<kOuter>(0);
  hwy::AlignedFreeUniquePtr<float[]> even_odd =
      hwy::AllocateAligned<float>(kInner * pool.NumWorkers());
  hwy::AlignedFreeUniquePtr<float[]> expected_out =
      SimpleMatVecAdd<kOuter, kInner>(mat, vec, add);
  hwy::AlignedFreeUniquePtr<float[]> actual_out =
      hwy::AllocateAligned<float>(kOuter);
  HWY_ASSERT(vec && add && even_odd && expected_out && actual_out);
  MatVecAdd<kOuter, kInner>(mat, 0, vec.get(), add.get(), even_odd.get(),
                            actual_out.get(), pool);
  AssertClose<kOuter>(actual_out, expected_out);
}

void TestTwoMatVecAdd() {
  hwy::ThreadPool pool(hwy::ThreadPool::MaxThreads());
  constexpr size_t kOuter = 128 * 3;
  constexpr size_t kInner = 128 * 5;
  CompressedArray<float, kOuter * kInner> mat0 =
      GenerateMat<float, kOuter, kInner>(0, pool);
  CompressedArray<float, kOuter * kInner> mat1 =
      GenerateMat<float, kOuter, kInner>(1, pool);
  hwy::AlignedFreeUniquePtr<float[]> vec = GenerateVec<kInner>(0);
  hwy::AlignedFreeUniquePtr<float[]> add0 = GenerateVec<kOuter>(0);
  hwy::AlignedFreeUniquePtr<float[]> add1 = GenerateVec<kOuter>(1);
  hwy::AlignedFreeUniquePtr<float[]> expected_out0 =
      SimpleMatVecAdd<kOuter, kInner>(mat0, vec, add0);
  hwy::AlignedFreeUniquePtr<float[]> expected_out1 =
      SimpleMatVecAdd<kOuter, kInner>(mat1, vec, add1);
  hwy::AlignedFreeUniquePtr<float[]> actual_out0 =
      hwy::AllocateAligned<float>(kOuter);
  hwy::AlignedFreeUniquePtr<float[]> actual_out1 =
      hwy::AllocateAligned<float>(kOuter);
  HWY_ASSERT(vec && add0 && add1 && expected_out0 && actual_out0 &&
             expected_out1 && actual_out1);
  TwoMatVecAdd<kOuter, kInner>(mat0, mat1, 0, vec.get(), add0.get(), add1.get(),
                               actual_out0.get(), actual_out1.get(), pool);
  AssertClose<kOuter>(actual_out0, expected_out0);
  AssertClose<kOuter>(actual_out1, expected_out1);
}

void TestTwoOfsMatVecAddLoop() {
  hwy::ThreadPool pool(hwy::ThreadPool::MaxThreads());
  constexpr size_t kOuter = 128 * 3;
  constexpr size_t kInner = 128 * 5;
  CompressedArray<float, kOuter * kInner> mat =
      GenerateMat<float, kOuter, kInner>(0, pool);
  hwy::AlignedFreeUniquePtr<float[]> vec = GenerateVec<kInner>(0);
  hwy::AlignedFreeUniquePtr<float[]> add0 = GenerateVec<kOuter>(0);
  hwy::AlignedFreeUniquePtr<float[]> add1 = GenerateVec<kOuter>(1);
  hwy::AlignedFreeUniquePtr<float[]> expected_out0 =
      SimpleMatVecAdd<kOuter, kInner>(mat, vec, add0);
  hwy::AlignedFreeUniquePtr<float[]> expected_out1 =
      SimpleMatVecAdd<kOuter, kInner>(mat, vec, add1);
  hwy::AlignedFreeUniquePtr<float[]> actual_out0 =
      hwy::AllocateAligned<float>(kOuter);
  hwy::AlignedFreeUniquePtr<float[]> actual_out1 =
      hwy::AllocateAligned<float>(kOuter);
  HWY_ASSERT(vec && add0 && add1 && expected_out0 && actual_out0 &&
             expected_out1 && actual_out1);
  TwoOfsMatVecAddLoop<kOuter, kInner>(mat, 0, 0, vec.get(), add0.get(),
                                      add1.get(), actual_out0.get(),
                                      actual_out1.get());
  AssertClose<kOuter>(actual_out0, expected_out0);
  AssertClose<kOuter>(actual_out1, expected_out1);
}

void TestSigmoid() {
  std::vector<float> values;
  for (int i = -150; i <= 150; ++i) {
    values.push_back(.1f * i);
  }
  std::vector<float> result = values;
  Sigmoid(result.data(), result.size());

  for (size_t i = 0; i < values.size(); i++) {
    const float max_error = 0.00007;
    float value = values[i];
    float approx = result[i];
    float expected = (1 / (1 + std::exp(-values[i])));
    EXPECT_NEAR(approx, expected, max_error) << "Input: " << value;
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gcpp {
HWY_BEFORE_TEST(OpsTest);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllAddFrom);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllMulBy);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllMulByConst);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllMulByConstAndAdd);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllSoftmax);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllCreateDistribution);
HWY_EXPORT_AND_TEST_P(OpsTest, TestAllTiledBatchMatMul);
HWY_EXPORT_AND_TEST_P(OpsTest, TestMatVecAdd);
HWY_EXPORT_AND_TEST_P(OpsTest, TestTwoMatVecAdd);
HWY_EXPORT_AND_TEST_P(OpsTest, TestTwoOfsMatVecAddLoop);
HWY_EXPORT_AND_TEST_P(OpsTest, TestSigmoid);
#ifdef HWY_AFTER_TEST
HWY_AFTER_TEST();
#endif

}  // namespace gcpp

#endif
