// Copyright 2025 Google LLC
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

// W8A8 MatMul: symmetric int8 weights times symmetric int8 activations,
// accumulating in int32 via the 4-way dot product (`vpdpbusd` on x86 VNNI,
// `sdot`/`usdot` on NEON, `svdot` on SVE). Unlike `MatMul`, which dequantizes
// `B` to BF16 for every tile (see `MMDecompress::DecompressB`), `B` is
// consumed as-is.
//
// Quantization scheme (see `#560` discussion): per-row (per-token) scales for
// `A`, computed on the fly, and per-row-of-transposed-B (per output channel)
// scales baked in at pack time. Both are symmetric, i.e. no zero point, so
// `C[r, c] = a_scale[r] * b_scale[c] * dot(qa[r], qb[c])` and the int32
// accumulation can run over an entire `kc` range before a single scaling step.
//
// On x86 the 4-way dot product requires one unsigned operand, so `B` is biased
// by 128 and the `128 * sum_k(qa)` term is subtracted per `kc` range, using
// prefix sums of the quantized `A`. Biasing `B` rather than `A` is what makes
// that per-range correction cheap: the correction then depends on `A`, which is
// small and quantized per call anyway, instead of on `B`. It also keeps the
// values written to `C` close to the true partial sums; correcting once over
// the whole `K` would inflate the intermediates that `MMAddC` accumulates
// through `C`, which loses a lot of precision when `C` is BF16 and the weight
// channels are not zero-mean.

#include <stddef.h>
#include <stdint.h>

#include <cmath>
#include <cstdlib>

#include "ops/matmul.h"  // IWYU pragma: export
#include "util/basics.h"
#include "util/mat.h"
#include "hwy/base.h"

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_MATMUL_I8_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_MATMUL_I8_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_MATMUL_I8_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_MATMUL_I8_TOGGLE
#endif

#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"
#include "ops/matmul-inl.h"

// `SumOfMulQuadAccumulate` is native for i8*i8 on NEON with `FEAT_DotProd`
// and on SVE, but on x86 only for u8*i8 (`vpdpbusd`); there, i8*i8 costs two
// VNNI ops plus a shift and subtract, which would give up most of the win.
// Hence bias `B` by 128 into u8 on x86, and correct for it via `A`.
// Define `GEMMA_MM_I8_FORCE_BIASED_B` to 0 or 1 to exercise either encoding
// regardless of target; both are correct everywhere, only the speed differs.
// `ops/matmul_i8_test.cc` is built twice, once each way.
#undef GEMMA_MM_I8_BIASED_B
#ifdef GEMMA_MM_I8_FORCE_BIASED_B
#define GEMMA_MM_I8_BIASED_B GEMMA_MM_I8_FORCE_BIASED_B
#elif HWY_TARGET <= HWY_AVX3_DL
#define GEMMA_MM_I8_BIASED_B 1
#else
#define GEMMA_MM_I8_BIASED_B 0
#endif

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// `A` is always symmetric int8; only `B`'s encoding varies by target, see
// `GEMMA_MM_I8_BIASED_B`.
using MMI8AT = int8_t;
#if GEMMA_MM_I8_BIASED_B
using MMI8BT = uint8_t;
#else
using MMI8BT = int8_t;
#endif

// Largest quantized magnitude. 127 rather than 128 keeps the scheme symmetric,
// which is what lets us skip the zero-point correction terms.
HWY_INLINE_VAR constexpr float kMMI8Max = 127.0f;

// Experimental QuaRot-style preprocessing. Applying the same orthonormal
// transform to A and each row of transposed B leaves their dot product
// unchanged, while spreading isolated activation outliers over a block. The
// fixed signs avoid always applying the same Hadamard basis to every block.
// 128 divides all Gemma 3 1B MatMul K dimensions.
HWY_INLINE_VAR constexpr size_t kMMI8RotateBlock = 128;

static inline bool MMI8RotateEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("GEMMA_MM_I8_ROTATE");
    return value != nullptr && value[0] != '\0' &&
           !(value[0] == '0' && value[1] == '\0');
  }();
  return enabled;
}

static inline bool MMI8CanRotate(size_t k) {
  return MMI8RotateEnabled() && (k % kMMI8RotateBlock) == 0;
}

static HWY_NOINLINE void MMI8Rotate(float* HWY_RESTRICT row, size_t k) {
  HWY_DASSERT((k % kMMI8RotateBlock) == 0);
  constexpr float kNormalize = 0.08838834764831845f;  // 1 / sqrt(128)
  for (size_t block = 0; block < k; block += kMMI8RotateBlock) {
    float* HWY_RESTRICT x = row + block;
    for (size_t i = 0; i < kMMI8RotateBlock; ++i) {
      // Deterministic Rademacher diagonal, shared by A and B.
      const uint32_t hash =
          static_cast<uint32_t>(block + i) * 0x9E3779B9u + 0x7F4A7C15u;
      if ((hash >> 31) != 0) x[i] = -x[i];
    }
    for (size_t width = 1; width < kMMI8RotateBlock; width *= 2) {
      for (size_t start = 0; start < kMMI8RotateBlock; start += 2 * width) {
        for (size_t i = 0; i < width; ++i) {
          const float left = x[start + i];
          const float right = x[start + width + i];
          x[start + i] = left + right;
          x[start + width + i] = left - right;
        }
      }
    }
    for (size_t i = 0; i < kMMI8RotateBlock; ++i) x[i] *= kNormalize;
  }
}

//------------------------------------------------------------------------------
// Quantized operands

// View into quantized `A`, analogous to `StridedViewBF` but carrying the
// per-row scales (and, when `B` is biased, prefix sums along `K`) alongside,
// because `MMLoops` passes only this one object down to the kernel.
struct MMI8AView {
  // Returns 2D subrange whose top-left is `r, c`, as `StridedView::View`.
  // Only called on the whole-matrix view, hence the offsets do not compound.
  MMI8AView View(size_t r, size_t c, size_t cols) const {
    return MMI8AView{data.View(r, c, cols), scale + r,
                     prefix + r * prefix_stride + c, prefix_stride};
  }

  // Sum of the quantized values of row `r` over the `cols` columns of this
  // view. `prefix` has `K + 1` entries per row, so this is exact for any range.
  int32_t RowSum(size_t r, size_t cols) const {
    const int32_t* HWY_RESTRICT p = prefix + r * prefix_stride;
    return p[cols] - p[0];
  }

  StridedView<MMI8AT> data;
  const float* HWY_RESTRICT scale;   // one per row of `data`
  const int32_t* HWY_RESTRICT prefix;  // null unless `GEMMA_MM_I8_BIASED_B`
  size_t prefix_stride;
};

// Transposed, symmetric-int8 `B`: `N` rows of `K` values each, so that a
// row of `B` is contiguous along `K` and thus already in the layout the 4-way
// dot product wants. The stored bytes are `q + 128` if `GEMMA_MM_I8_BIASED_B`,
// else `q`; the buffer is typed `int8_t` either way and reinterpreted in the
// kernel. Production would pick one encoding for the on-disk format rather
// than deriving it from the target.
struct MMI8B {
  size_t Rows() const { return data->Rows(); }
  size_t Cols() const { return data->Cols(); }

  const MatPtrT<int8_t>* data;
  const float* HWY_RESTRICT scale;  // [N] dequantization scale
};

//------------------------------------------------------------------------------
// Reduction and store

// Like `MMStoreHorizontalSumsIntoC`, but the tile accumulators are int32 and
// the scale is a per-row times per-column outer product rather than a scalar.
template <size_t kRowsAC>
class MMI8StoreHorizontalSumsIntoC {
 public:
  static_assert(kNR == 4);  // for `StoreInterleaved4`

  // Horizontal sums of the 16 (`kRowsAC x kNR`) int32 accumulators, using the
  // same vector-length-agnostic transpose as the BF16 kernel. Valid because
  // the 4-way dot product, like BF16's pairwise add, only permutes the terms
  // of each dot product and thus preserves the horizontal sum.
  template <class DI32, class VI32 = hn::Vec<DI32>,
            class D4 = hn::Full128<int32_t>, class V4 = hn::Vec<D4>>
  HWY_INLINE void Reduce4x4(DI32 di32,                              //
                            VI32 C00, VI32 C01, VI32 C02, VI32 C03,  //
                            VI32 C10, VI32 C11, VI32 C12, VI32 C13,  //
                            VI32 C20, VI32 C21, VI32 C22, VI32 C23,  //
                            VI32 C30, VI32 C31, VI32 C32, VI32 C33,  //
                            V4& sum0, V4& sum1, V4& sum2, V4& sum3) {
    HWY_ALIGN int32_t buf[16 * hn::MaxLanes(di32)];
    HWY_LANES_CONSTEXPR const size_t N = hn::Lanes(di32);

    MaybeStoreInterleaved4<0>(di32, N, C00, C01, C02, C03, buf);
    MaybeStoreInterleaved4<1>(di32, N, C10, C11, C12, C13, buf);
    MaybeStoreInterleaved4<2>(di32, N, C20, C21, C22, C23, buf);
    MaybeStoreInterleaved4<3>(di32, N, C30, C31, C32, C33, buf);

    const D4 d4;
    sum0 = MaybeLoad<0>(d4, N, buf);
    sum1 = MaybeLoad<1>(d4, N, buf);
    sum2 = MaybeLoad<2>(d4, N, buf);
    sum3 = MaybeLoad<3>(d4, N, buf);

    for (size_t lane = 1; lane < N; ++lane) {
      sum0 = MaybeAdd<0>(d4, N, sum0, buf + kNR * lane);
      sum1 = MaybeAdd<1>(d4, N, sum1, buf + kNR * lane);
      sum2 = MaybeAdd<2>(d4, N, sum2, buf + kNR * lane);
      sum3 = MaybeAdd<3>(d4, N, sum3, buf + kNR * lane);
    }
  }

  // Dequantizes the four 4-wide int32 dot products and stores them to `C`.
  // `b_scale` points to the `kNR` current columns and `a_scale` to the current
  // `range_mc` (hence indexed by `imc + kRow`), whereas `a_rowsum` holds just
  // this tile's `kRowsAC` values and is indexed by `kRow` alone. It is the sum
  // of the quantized `A` values over this `kc` range, which undoes `B`'s 128
  // bias, and is unused when `B` is not biased.
  template <class D4I, class V4I = hn::Vec<D4I>, class Tag, class CView>
  HWY_INLINE void Store(D4I d4i, V4I sum0, V4I sum1, V4I sum2, V4I sum3,
                        const float* HWY_RESTRICT a_scale,
                        const int32_t* HWY_RESTRICT a_rowsum,
                        const float* HWY_RESTRICT b_scale,
                        const float* HWY_RESTRICT add, const size_t imc,
                        Tag tag, CView C_MC_NR) const {
    const hn::Full128<float> d4;
    using V4F = hn::Vec<decltype(d4)>;

    const V4F vb_scale = hn::LoadU(d4, b_scale);
    HWY_ALIGN static constexpr float kZero[4] = {};
    const V4F vadd = hn::Load(d4, add ? add : kZero);

    // Each term is `(qb + 128) * qa` instead of `qb * qa`, hence subtract
    // `128 * sum_k(qa)` over this `kc` range. Applied on every visit, so the
    // values written to `C` stay close to the true partial sums.
    MaybeScaleAndStore<0>(d4i, d4, sum0, a_rowsum, vb_scale, vadd, a_scale, tag,
                          imc, C_MC_NR);
    MaybeScaleAndStore<1>(d4i, d4, sum1, a_rowsum, vb_scale, vadd, a_scale, tag,
                          imc, C_MC_NR);
    MaybeScaleAndStore<2>(d4i, d4, sum2, a_rowsum, vb_scale, vadd, a_scale, tag,
                          imc, C_MC_NR);
    MaybeScaleAndStore<3>(d4i, d4, sum3, a_rowsum, vb_scale, vadd, a_scale, tag,
                          imc, C_MC_NR);
  }

 private:
  template <size_t kRow, class DI32, class VI32 = hn::Vec<DI32>>
  static HWY_INLINE void MaybeStoreInterleaved4(DI32 di32, size_t N, VI32 Cr0,
                                                VI32 Cr1, VI32 Cr2, VI32 Cr3,
                                                int32_t* HWY_RESTRICT buf) {
    if constexpr (kRow < kRowsAC) {
      hn::StoreInterleaved4(Cr0, Cr1, Cr2, Cr3, di32, buf + 4 * kRow * N);
    }
  }

  template <size_t kRow, class D4I, class V4I = hn::Vec<D4I>>
  static HWY_INLINE V4I MaybeLoad(D4I d4i, size_t N,
                                  const int32_t* HWY_RESTRICT buf) {
    if constexpr (kRow < kRowsAC) {
      return hn::Load(d4i, buf + 4 * kRow * N);
    } else {
      return hn::Zero(d4i);
    }
  }

  template <size_t kRow, class D4I, class V4I = hn::Vec<D4I>>
  static HWY_INLINE V4I MaybeAdd(D4I d4i, size_t N, V4I sum,
                                 const int32_t* HWY_RESTRICT buf) {
    if constexpr (kRow < kRowsAC) {
      return hn::Add(sum, hn::Load(d4i, buf + 4 * kRow * N));
    } else {
      return sum;
    }
  }

  template <size_t kRow, /*deduced:*/ class D4I, class V4I = hn::Vec<D4I>,
            class D4F, class V4F = hn::Vec<D4F>, class Tag, class CView>
  static HWY_INLINE void MaybeScaleAndStore(
      D4I d4i, D4F d4, V4I sum, const int32_t* HWY_RESTRICT a_rowsum,
      V4F vb_scale, V4F vadd, const float* HWY_RESTRICT a_scale, Tag,
      const size_t imc, CView C_MC_NR) {
    if constexpr (kRow < kRowsAC) {
      using TC = hwy::RemoveCvRef<decltype(C_MC_NR.Row(0)[0])>;
      TC* HWY_RESTRICT pos = C_MC_NR.Row(imc + kRow);
      const hn::Rebind<TC, D4F> dc4;

      const V4F vscale = hn::Mul(vb_scale, hn::Set(d4, a_scale[imc + kRow]));
      if constexpr (GEMMA_MM_I8_BIASED_B) {
        sum = hn::Sub(sum, hn::Set(d4i, static_cast<int32_t>(
                                         a_rowsum[kRow] * 128)));
      }
      const V4F dot = hn::ConvertTo(d4, sum);

      if constexpr (hwy::IsSame<Tag, MMAddC>()) {
        vadd = F32FromTC(dc4, hn::Load(dc4, pos));  // load prior value
      } else {
        static_assert(hwy::IsSame<Tag, MMSetC>());
        // vadd remains the bias (added once, the first time we store to C)
      }
      const V4F out = hn::MulAdd(dot, vscale, vadd);
      hn::Store(TCFromF32(dc4, out), dc4, pos);
    }
  }
};  // MMI8StoreHorizontalSumsIntoC

//------------------------------------------------------------------------------
// Kernel

// Drop-in replacement for `MMKernel` (same `B3A2C0`/`ForeachKC` interface, so
// that `MMLoops` can drive either), but with int8 operands.
class MMI8Kernel {
 public:
  using AView = MMI8AView;

  template <typename BT, typename Tag, class CView>
  static void B3A2C0(const AView A, const BT& B, const IndexRange& range_mc,
                     const IndexRange& range_kc, const IndexRange& range_nc,
                     const MMArgs& args, Tag out_tag, CView C_MC_NC) {
    const size_t kc = range_kc.Num();
    const AView A_view = A.View(range_mc.begin(), range_kc.begin(), kc);

    for (size_t inc = 0; inc < range_nc.Num(); inc += kNR) {
      // For `add` and `B`, which are global, unlike `C_MC_NC`.
      const size_t row_b = range_nc.begin() + inc;
      // No decompression: `B` is already in the layout the kernel wants.
      const StridedView<int8_t> B_view(*B.data, row_b, range_kc.begin(), kc);
      const CView C_MC_NR = C_MC_NC.View(0, inc, kNR);
      const float* HWY_RESTRICT add = args.add ? args.add + row_b : nullptr;
      A2C0(A_view, B_view, B.scale + row_b, args.mr, range_mc, kc, add, out_tag,
           C_MC_NR);
    }
  }

  template <typename BT, class CView>
  static void ForeachKC(const AView A, const BT& B, const IndexRange& range_mc,
                        const IndexRangePartition& ranges_kc,
                        const IndexRange& range_nc, const MMArgs& args,
                        CView C_MC_NC) {
    ranges_kc.VisitFirst([&](const IndexRange& range_kc) {
      B3A2C0(A, B, range_mc, range_kc, range_nc, args, MMSetC(), C_MC_NC);
    });
    ranges_kc.VisitRemaining([&](const IndexRange& range_kc) {
      B3A2C0(A, B, range_mc, range_kc, range_nc, args, MMAddC(), C_MC_NC);
    });
  }

 private:
  // Innermost loop over `kc` columns in steps of one int8 vector, for
  // `kRowsAC` rows of `A_view` and `kNR` rows of `B_view`. Mirrors
  // `MMKernel::LoopKC`: elementwise along `K` with 16 accumulators whose
  // horizontal sums are the `kRowsAC x kNR` results.
  template <size_t kRowsAC, /*deduced:*/ class Tag, class CView>
  static HWY_INLINE void LoopKC(const AView A_view,
                                const StridedView<int8_t> B_view,
                                const float* HWY_RESTRICT b_scale, size_t imc,
                                size_t kc, const float* HWY_RESTRICT add,
                                Tag tag, CView C_MC_NR) {
    const hn::ScalableTag<MMI8AT> da8;  // A: always i8
    const hn::ScalableTag<MMI8BT> db8;  // B: u8 or i8, same lane count
    const hn::Repartition<int32_t, decltype(da8)> di32;
    using VA8 = hn::Vec<decltype(da8)>;
    using VB8 = hn::Vec<decltype(db8)>;
    using VI32 = hn::Vec<decltype(di32)>;
    HWY_LANES_CONSTEXPR const size_t N8 = hn::Lanes(da8);

    HWY_DASSERT(kRowsAC <= kMaxMR);
    static_assert(kNR == 4);

    const MMI8AT* HWY_RESTRICT ar0 = A_view.data.Row(imc + 0);
    const MMI8AT* HWY_RESTRICT ar1 =
        kRowsAC > 1 ? A_view.data.Row(imc + 1) : nullptr;
    const MMI8AT* HWY_RESTRICT ar2 =
        kRowsAC > 2 ? A_view.data.Row(imc + 2) : nullptr;
    const MMI8AT* HWY_RESTRICT ar3 =
        kRowsAC > 3 ? A_view.data.Row(imc + 3) : nullptr;
    const MMI8BT* HWY_RESTRICT br0 =
        HWY_RCAST_ALIGNED(const MMI8BT*, B_view.Row(0));
    const MMI8BT* HWY_RESTRICT br1 =
        HWY_RCAST_ALIGNED(const MMI8BT*, B_view.Row(1));
    const MMI8BT* HWY_RESTRICT br2 =
        HWY_RCAST_ALIGNED(const MMI8BT*, B_view.Row(2));
    const MMI8BT* HWY_RESTRICT br3 =
        HWY_RCAST_ALIGNED(const MMI8BT*, B_view.Row(3));

    VI32 C00 = hn::Zero(di32), C01 = hn::Zero(di32), C02 = hn::Zero(di32),
         C03 = hn::Zero(di32), C10 = hn::Zero(di32), C11 = hn::Zero(di32),
         C12 = hn::Zero(di32), C13 = hn::Zero(di32), C20 = hn::Zero(di32),
         C21 = hn::Zero(di32), C22 = hn::Zero(di32), C23 = hn::Zero(di32),
         C30 = hn::Zero(di32), C31 = hn::Zero(di32), C32 = hn::Zero(di32),
         C33 = hn::Zero(di32);

    size_t ikc = 0;
    if (kc >= N8) {
      HWY_UNROLL(1)
      for (; ikc <= kc - N8; ikc += N8) {
        const VB8 b0 = hn::LoadU(db8, br0 + ikc);
        const VB8 b1 = hn::LoadU(db8, br1 + ikc);
        const VB8 b2 = hn::LoadU(db8, br2 + ikc);
        const VB8 b3 = hn::LoadU(db8, br3 + ikc);

        {
          const VA8 a0 = hn::LoadU(da8, ar0 + ikc);
          MMQuantizedDot4Accumulate<GEMMA_MM_I8_BIASED_B>(
              di32, a0, b0, b1, b2, b3, C00, C01, C02, C03);
        }
        if constexpr (kRowsAC > 1) {
          const VA8 a1 = hn::LoadU(da8, ar1 + ikc);
          MMQuantizedDot4Accumulate<GEMMA_MM_I8_BIASED_B>(
              di32, a1, b0, b1, b2, b3, C10, C11, C12, C13);
        }
        if constexpr (kRowsAC > 2) {
          const VA8 a2 = hn::LoadU(da8, ar2 + ikc);
          MMQuantizedDot4Accumulate<GEMMA_MM_I8_BIASED_B>(
              di32, a2, b0, b1, b2, b3, C20, C21, C22, C23);
        }
        if constexpr (kRowsAC > 3) {
          const VA8 a3 = hn::LoadU(da8, ar3 + ikc);
          MMQuantizedDot4Accumulate<GEMMA_MM_I8_BIASED_B>(
              di32, a3, b0, b1, b2, b3, C30, C31, C32, C33);
        }
      }
    }

    // Remainder. `LoadN` zeroes the upper lanes of both operands, so their
    // products are zero. Zeroing `A` is what makes this safe: a zero `B` lane
    // does not mean zero in the biased-u8 encoding.
    const size_t remaining_kc = kc - ikc;
    HWY_DASSERT(remaining_kc < N8);
    if (HWY_UNLIKELY(remaining_kc != 0)) {
      const VB8 b0 = hn::LoadN(db8, br0 + ikc, remaining_kc);
      const VB8 b1 = hn::LoadN(db8, br1 + ikc, remaining_kc);
      const VB8 b2 = hn::LoadN(db8, br2 + ikc, remaining_kc);
      const VB8 b3 = hn::LoadN(db8, br3 + ikc, remaining_kc);

      {
        const VA8 a0 = hn::LoadN(da8, ar0 + ikc, remaining_kc);
        MMQuantizedDot4Accumulate<GEMMA_MM_I8_BIASED_B>(
            di32, a0, b0, b1, b2, b3, C00, C01, C02, C03);
      }
      if constexpr (kRowsAC > 1) {
        const VA8 a1 = hn::LoadN(da8, ar1 + ikc, remaining_kc);
        MMQuantizedDot4Accumulate<GEMMA_MM_I8_BIASED_B>(
            di32, a1, b0, b1, b2, b3, C10, C11, C12, C13);
      }
      if constexpr (kRowsAC > 2) {
        const VA8 a2 = hn::LoadN(da8, ar2 + ikc, remaining_kc);
        MMQuantizedDot4Accumulate<GEMMA_MM_I8_BIASED_B>(
            di32, a2, b0, b1, b2, b3, C20, C21, C22, C23);
      }
      if constexpr (kRowsAC > 3) {
        const VA8 a3 = hn::LoadN(da8, ar3 + ikc, remaining_kc);
        MMQuantizedDot4Accumulate<GEMMA_MM_I8_BIASED_B>(
            di32, a3, b0, b1, b2, b3, C30, C31, C32, C33);
      }
    }

    // Sums of the quantized `A` values over this `kc` range, for undoing `B`'s
    // bias. `A_view` is already restricted to the range, so `kc` is its width.
    int32_t a_rowsum[kNR] = {};
    if constexpr (GEMMA_MM_I8_BIASED_B) {
      a_rowsum[0] = A_view.RowSum(imc + 0, kc);
      if constexpr (kRowsAC > 1) a_rowsum[1] = A_view.RowSum(imc + 1, kc);
      if constexpr (kRowsAC > 2) a_rowsum[2] = A_view.RowSum(imc + 2, kc);
      if constexpr (kRowsAC > 3) a_rowsum[3] = A_view.RowSum(imc + 3, kc);
    }

    MMI8StoreHorizontalSumsIntoC<kRowsAC> horz;
    const hn::Full128<int32_t> d4i;
    hn::Vec<decltype(d4i)> sum0, sum1, sum2, sum3;
    horz.Reduce4x4(di32, C00, C01, C02, C03, C10, C11, C12, C13, C20, C21, C22,
                   C23, C30, C31, C32, C33, sum0, sum1, sum2, sum3);
    horz.Store(d4i, sum0, sum1, sum2, sum3, A_view.scale, a_rowsum, b_scale,
               add, imc, tag, C_MC_NR);
  }

  // As `MMKernel::A2C0`.
  template <class Tag, class CView>
  static HWY_INLINE void A2C0(const AView A_view,
                              const StridedView<int8_t> B_view,
                              const float* HWY_RESTRICT b_scale, size_t mr,
                              const IndexRange& range_mc, size_t kc,
                              const float* HWY_RESTRICT add, Tag tag,
                              CView C_MC_NR) {
    HWY_DASSERT(1 <= mr && mr <= kMaxMR);
    const size_t mc = range_mc.Num();
    size_t imc = 0;

    if (HWY_UNLIKELY(mr == 1)) {
      for (; imc < mc; ++imc) {
        LoopKC<1>(A_view, B_view, b_scale, imc, kc, add, tag, C_MC_NR);
      }
      return;
    }

    if (HWY_UNLIKELY(mr == 2)) {
      if (HWY_LIKELY(mc >= 2)) {
        for (; imc <= mc - 2; imc += 2) {
          LoopKC<2>(A_view, B_view, b_scale, imc, kc, add, tag, C_MC_NR);
        }
      }
      if (HWY_UNLIKELY(imc != mc)) {
        LoopKC<1>(A_view, B_view, b_scale, imc, kc, add, tag, C_MC_NR);
      }
      return;
    }

    HWY_DASSERT(mr == 4);
    if (HWY_LIKELY(mc >= 4)) {
      for (; imc <= mc - 4; imc += 4) {
        LoopKC<4>(A_view, B_view, b_scale, imc, kc, add, tag, C_MC_NR);
      }
    }
    const size_t remainder_mc = mc - imc;
    HWY_DASSERT(remainder_mc < 4);
    if (HWY_UNLIKELY(remainder_mc & 2)) {
      LoopKC<2>(A_view, B_view, b_scale, imc, kc, add, tag, C_MC_NR);
      imc += 2;
    }
    if (HWY_UNLIKELY(remainder_mc & 1)) {
      LoopKC<1>(A_view, B_view, b_scale, imc, kc, add, tag, C_MC_NR);
      imc += 1;
    }
    HWY_DASSERT(imc == mc);
  }
};  // MMI8Kernel

//------------------------------------------------------------------------------
// Quantization

// Loads one vector of F32 from F32 or BF16 `A`, so that quantization can read
// activations in whichever format the caller already has.
template <class DF, typename TA, class VF = hn::Vec<DF>>
static HWY_INLINE VF LoadF32(DF df, const TA* HWY_RESTRICT p) {
  if constexpr (IsF32<TA>()) {
    return hn::LoadU(df, p);
  } else {
    static_assert(IsBF16<TA>());
    return hn::PromoteTo(df, hn::LoadU(hn::Rebind<BF16, DF>(), p));
  }
}

template <class DF, typename TA, class VF = hn::Vec<DF>>
static HWY_INLINE VF LoadNF32(DF df, const TA* HWY_RESTRICT p, size_t n) {
  if constexpr (IsF32<TA>()) {
    return hn::LoadN(df, p, n);
  } else {
    static_assert(IsBF16<TA>());
    return hn::PromoteTo(df, hn::LoadN(hn::Rebind<BF16, DF>(), p, n));
  }
}

// Quantizes one row of `k` activations to symmetric int8, returning the
// dequantization scale. Also writes `k + 1` prefix sums of the quantized
// values (when `B` is biased), which the kernel uses to undo that bias for
// whichever `kc` range it is working on. `out` is zero-padded to `padded_k`.
template <typename TA>
static HWY_INLINE float QuantizeRowA(const TA* HWY_RESTRICT in, size_t k,
                                     MMI8AT* HWY_RESTRICT out,
                                     int32_t* HWY_RESTRICT prefix,
                                     size_t padded_k) {
  const hn::ScalableTag<float> df;
  const hn::Rebind<int32_t, decltype(df)> di32;
  const hn::Rebind<MMI8AT, decltype(df)> d8;
  using VF = hn::Vec<decltype(df)>;
  const size_t NF = hn::Lanes(df);

  VF vmax = hn::Zero(df);
  size_t i = 0;
  if (k >= NF) {
    for (; i <= k - NF; i += NF) {
      vmax = hn::Max(vmax, hn::Abs(LoadF32(df, in + i)));
    }
  }
  if (i != k) {
    vmax = hn::Max(vmax, hn::Abs(LoadNF32(df, in + i, k - i)));
  }
  const float amax = hn::ReduceMax(df, vmax);

  const float scale = (amax == 0.0f) ? 1.0f : amax / kMMI8Max;
  const float inv_scale = (amax == 0.0f) ? 0.0f : kMMI8Max / amax;
  const VF vinv = hn::Set(df, inv_scale);

  i = 0;
  if (k >= NF) {
    for (; i <= k - NF; i += NF) {
      const auto q = hn::NearestInt(hn::Mul(LoadF32(df, in + i), vinv));
      hn::StoreU(hn::DemoteTo(d8, q), d8, out + i);
    }
  }
  for (; i < k; ++i) {
    const float in_f = hwy::ConvertScalarTo<float>(in[i]);
    out[i] = static_cast<MMI8AT>(std::lroundf(in_f * inv_scale));
  }
  for (; i < padded_k; ++i) {
    out[i] = static_cast<MMI8AT>(0);
  }

  if constexpr (GEMMA_MM_I8_BIASED_B) {
    // Scalar, but only `M * K` additions per MatMul, i.e. the same order as
    // the quantization itself and negligible next to `M * K * N` products.
    int32_t sum = 0;
    prefix[0] = 0;
    for (size_t j = 0; j < k; ++j) {
      sum += out[j];
      prefix[j + 1] = sum;
    }
  }
  return scale;
}

// Storage for quantized `A`, reused across `MatMulI8` calls. Analogous to
// `MMEntireA`, but sized by the caller because this is a prototype and
// `MatMulEnv` does not know about int8 yet.
class MMI8AStorage {
 public:
  // `prefix_` is `K + 1` per row, which is simple but the largest cost here.
  // Production would instead compute one sum per (row, kc range) once the
  // config is known, which is `NumTasks()` rather than `K` per row.
  MMI8AStorage(size_t max_M, size_t max_K, const Allocator& allocator)
      : data_("A_i8", Extents2D(max_M, max_K), allocator, MatPadding::kOdd),
        prefix_stride_(hwy::RoundUpTo(max_K + 1, HWY_ALIGNMENT / 4)),
        prefix_((GEMMA_MM_I8_BIASED_B ? max_M : 1) * prefix_stride_),
        scale_(max_M) {}

  MMI8AView View(const Extents2D& extents) {
    HWY_DASSERT(extents.rows <= data_.Rows());
    HWY_DASSERT(extents.cols <= data_.Cols());
    return MMI8AView{
        StridedView<MMI8AT>(HWY_RCAST_ALIGNED(MMI8AT*, data_.Row(0)),
                            extents.cols, data_.Stride()),
        scale_.data(), prefix_.data(), prefix_stride_};
  }

  float* HWY_RESTRICT scale() { return scale_.data(); }
  int32_t* HWY_RESTRICT prefix(size_t row) {
    return prefix_.data() + (GEMMA_MM_I8_BIASED_B ? row : 0) * prefix_stride_;
  }
  size_t Stride() const { return data_.Stride(); }

 private:
  MatStorageT<uint8_t> data_;
  size_t prefix_stride_;
  hwy::AlignedVector<int32_t> prefix_;
  hwy::AlignedVector<float> scale_;
};

// Quantizes all `M x K` of `A` into `storage`, in parallel over rows.
// This replaces `MMDecompress::DecompressA` and is the same order of cost:
// one pass over `A`, once per `MatMul` rather than per B tile.
template <typename TA>
static HWY_NOINLINE MMI8AView QuantizeA(const MatPtrT<TA>& A,
                                        MMI8AStorage& storage,
                                        ThreadingContext& ctx,
                                        size_t cluster_idx) {
  const MMI8AView view = storage.View(A.Extents());
  const size_t k = A.Cols();
  const size_t padded_k = hwy::RoundUpTo(k, hn::Lanes(hn::ScalableTag<int8_t>()));
  float* HWY_RESTRICT scale = storage.scale();
  const float a_scale = A.Scale();

  ParallelFor(Parallelism::kFlat, A.Rows(), ctx, cluster_idx,
              Callers::kMMQuantizeA,
              [&](size_t r, size_t /*worker*/) HWY_ATTR {
                if (MMI8CanRotate(k)) {
                  hwy::AlignedVector<float> rotated(padded_k);
                  for (size_t c = 0; c < k; ++c) {
                    rotated[c] = hwy::ConvertScalarTo<float>(A.Row(r)[c]);
                  }
                  MMI8Rotate(rotated.data(), k);
                  scale[r] =
                      a_scale * QuantizeRowA(rotated.data(), k,
                                             view.data.Row(r), storage.prefix(r),
                                             padded_k);
                } else {
                  scale[r] =
                      a_scale * QuantizeRowA(A.Row(r), k, view.data.Row(r),
                                             storage.prefix(r), padded_k);
                }
              });
  return view;
}

// Symmetric int8 quantization of already-transposed `B`, i.e. `N` rows of `K`.
// Fills `data` (biased by 128 if `GEMMA_MM_I8_BIASED_B`, zero-padded to its
// stride) and `scale`. Called once per weight matrix, so not performance-
// critical.
static HWY_NOINLINE MMI8B PackB(const MatPtrT<float>& B_f32,
                                MatPtrT<int8_t>& data,
                                float* HWY_RESTRICT scale,
                                ThreadingContext& ctx) {
  const size_t k = B_f32.Cols();
  const float b_scale = B_f32.Scale();

  ParallelFor(Parallelism::kFlat, B_f32.Rows(), ctx, /*cluster_idx=*/0,
              Callers::kTest, [&](size_t r, size_t /*worker*/) HWY_ATTR {
                const float* HWY_RESTRICT in = B_f32.Row(r);
                hwy::AlignedVector<float> rotated;
                if (MMI8CanRotate(k)) {
                  rotated.resize(k);
                  hwy::CopyBytes(in, rotated.data(), k * sizeof(float));
                  MMI8Rotate(rotated.data(), k);
                  in = rotated.data();
                }
                float amax = 0.0f;
                for (size_t c = 0; c < k; ++c) {
                  amax = HWY_MAX(amax, hwy::ScalarAbs(in[c]));
                }
                const float s = (amax == 0.0f) ? 1.0f : amax / kMMI8Max;
                const float inv = (amax == 0.0f) ? 0.0f : kMMI8Max / amax;
                MMI8BT* HWY_RESTRICT out =
                    HWY_RCAST_ALIGNED(MMI8BT*, data.Row(r));
                for (size_t c = 0; c < k; ++c) {
                  const int32_t q =
                      static_cast<int32_t>(std::lroundf(in[c] * inv));
                  HWY_DASSERT(-127 <= q && q <= 127);
                  out[c] = static_cast<MMI8BT>(
                      q + (GEMMA_MM_I8_BIASED_B ? 128 : 0));
                }
                for (size_t c = k; c < data.Stride(); ++c) {
                  out[c] = static_cast<MMI8BT>(0);
                }
                scale[r] = b_scale * s;
              });

  return MMI8B{&data, scale};
}

//------------------------------------------------------------------------------
// Entry point

// As `MatMul`, but `A` is quantized on the fly and `B` was packed by `PackB`.
// Reuses the same blocking, parallelization and autotuning as `MatMul`; only
// the kernel and operand types differ. `env` must not be shared with
// (BF16) `MatMul` calls of the same shape, because the autotuner is keyed on
// shape alone and the two kernels prefer different configs.
template <typename TA, typename TC>
HWY_NOINLINE MMPerKey* MatMulI8(const MatPtrT<TA>& A, const MMI8B& B,
                                const float* HWY_RESTRICT add, MatMulEnv& env,
                                MatPtrT<TC>& C, MMI8AStorage& a_storage,
                                MMOptions options = MMOptions()) {
  const size_t cluster_idx = options.cluster_idx;
  HWY_DASSERT(cluster_idx < env.row_ptrs.size());
  GCPP_ZONE(env.ctx, env.ctx.Worker(cluster_idx), Zones::kMMMatMul);

  RowPtrs<TC> C_rows = GetOrSetTempRowPtrs(C, env.row_ptrs[cluster_idx]);

  const size_t M = A.Rows();
  const size_t K = A.Cols();
  const size_t N = B.Rows();
  const size_t num_B = 1;

  const CacheInfo& cache = env.ctx.cache_info;
  MMPerKey& per_key = MMImpl::FindOrAddPerKey(
      M, K, N, num_B, cache.VectorBytes(), env.per_cluster[cluster_idx]);

  // Outside the timed section, as `MMDecompress::MaybeDecompressA`.
  const MMI8AView A_view = QuantizeA(A, a_storage, env.ctx, cluster_idx);

  const MMI8B* B2 = nullptr;  // required for type matching

  // Scales are per row/column, hence folded into `A_view.scale` and
  // `B.scale`; the scalar `MMArgs::scale_A` is unused.
  MMAutoTune<MMConfig>& tuner = per_key.autotune;
  if (HWY_LIKELY(tuner.Best())) {
    const MMArgs args(env, M, K, N, /*scale_A=*/1.0f, add, options, tuner,
                      *tuner.Best());
    MMLoops::Dispatch<MMI8Kernel>(A_view, B, B2, C_rows, args);
    return &per_key;
  }

  if (HWY_UNLIKELY(!tuner.HasCandidates())) {
    HWY_ASSERT(K == B.Cols());
    HWY_ASSERT(M <= kMaxBatchSize);
    HWY_ASSERT(N % kNR == 0);
    tuner.SetCandidates(
        MMCandidates(cache, M, K, N, num_B, sizeof(TC), env.print_config));
  }

  const MMConfig& cfg = tuner.NextConfig();
  const MMArgs args(env, M, K, N, /*scale_A=*/1.0f, add, options, tuner, cfg);

  const uint64_t t0 = hwy::timer::Start();
  MMLoops::Dispatch<MMI8Kernel>(A_view, B, B2, C_rows, args);
  MMImpl::NotifyAutotuneResult(env, M, K, N, num_B, t0, tuner, cfg);

  return &per_key;
}

// As `TwoMatMul`: computes `A * B1` into `C` and `A * B2` into a per-worker
// tile, passing both to `options.func`. Used by gated FFNs.
static HWY_NOINLINE MMPerKey* TwoMatMulI8(const MatPtrT<BF16>& A,
                                          const MMI8B& B1,
                                          const MMI8B& B2, MatMulEnv& env,
                                          MatPtrT<BF16>& C,
                                          MMI8AStorage& a_storage,
                                          MMOptions options) {
  const size_t cluster_idx = options.cluster_idx;
  HWY_DASSERT(cluster_idx < env.row_ptrs.size());
  GCPP_ZONE(env.ctx, env.ctx.Worker(cluster_idx), Zones::kMMTwoMatMul);
  HWY_DASSERT(options.func != nullptr);  // no other way to get access to C2.

  RowPtrs<BF16> C_rows = GetOrSetTempRowPtrs(C, env.row_ptrs[cluster_idx]);

  const size_t M = A.Rows();
  const size_t K = A.Cols();
  const size_t N = B1.Rows();
  const size_t num_B = 2;

  const CacheInfo& cache = env.ctx.cache_info;
  MMPerKey& per_key = MMImpl::FindOrAddPerKey(
      M, K, N, num_B, cache.VectorBytes(), env.per_cluster[cluster_idx]);

  const MMI8AView A_view = QuantizeA(A, a_storage, env.ctx, cluster_idx);

  MMAutoTune<MMConfig>& tuner = per_key.autotune;
  if (HWY_LIKELY(tuner.Best())) {
    const MMArgs args(env, M, K, N, /*scale_A=*/1.0f, /*add=*/nullptr, options,
                      tuner, *tuner.Best());
    MMLoops::Dispatch<MMI8Kernel>(A_view, B1, &B2, C_rows, args);
    return &per_key;
  }

  if (HWY_UNLIKELY(!tuner.HasCandidates())) {
    HWY_ASSERT(K == B1.Cols());
    HWY_ASSERT(K == B2.Cols());
    HWY_ASSERT(M <= kMaxBatchSize);
    HWY_ASSERT(N % kNR == 0);
    const size_t max_M = MMKeys::BucketM(M);
    tuner.SetCandidates(MMCandidates(cache, max_M, K, N, num_B, sizeof(BF16),
                                     env.print_config));
  }

  const MMConfig& cfg = tuner.NextConfig();
  const MMArgs args(env, M, K, N, /*scale_A=*/1.0f, /*add=*/nullptr, options,
                    tuner, cfg);
  const uint64_t t0 = hwy::timer::Start();
  MMLoops::Dispatch<MMI8Kernel>(A_view, B1, &B2, C_rows, args);
  MMImpl::NotifyAutotuneResult(env, M, K, N, num_B, t0, tuner, cfg);

  return &per_key;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
