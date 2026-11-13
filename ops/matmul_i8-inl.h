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
// Optional microscaling instead stores one A/B scale per rotation block and
// accumulates dequantized block dot products in F32.
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
#include <stdlib.h>

#include <cmath>

#include "hwy/base.h"
#include "ops/matmul.h"  // IWYU pragma: export
#include "util/basics.h"
#include "util/mat.h"

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
#elif HWY_TARGET <= HWY_AVX2
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
// The defaults preserve the issue #1002 reference configuration; the
// alternatives are selected per process for isolated ablations.
HWY_INLINE_VAR constexpr size_t kMMI8DefaultRotateBlock = 128;
HWY_INLINE_VAR constexpr size_t kMMI8DefaultHashBits = 32;

static inline bool MMI8Flag(const char* name, bool fallback = false) {
  const char* value = getenv(name);
  return value == nullptr ? fallback : atoi(value) != 0;
}

static inline bool MMI8FastRotate() {
  static const bool enabled = MMI8Flag("GEMMA_MM_I8_FAST_ROTATE", true);
  return enabled;
}

static inline bool MMI8NativeVNNI() {
#if HWY_TARGET == HWY_AVX2 && GEMMA_MM_I8_BIASED_B && defined(__GNUC__) && \
    !defined(__clang__)
  static const bool enabled =
      MMI8Flag("GEMMA_MM_I8_VNNI", true) && __builtin_cpu_supports("avxvnni");
  return enabled;
#else
  return false;
#endif
}

template <bool kNative, bool kCompact = false, class DI32, class VA, class VB,
          class VI>
static HWY_INLINE void MMI8Dot4(DI32 di32, VA a, VB b0, VB b1, VB b2, VB b3,
                                VI& c0, VI& c1, VI& c2, VI& c3) {
#if HWY_TARGET == HWY_AVX2 && GEMMA_MM_I8_BIASED_B && defined(__GNUC__) && \
    !defined(__clang__)
  if constexpr (kNative) {
    if constexpr (kCompact && HWY_ARCH_X86_64) {
      // Nine distinct registers are required; x86-32 has only eight.
      // Keep all four B vectors live across the outputs. Separate asm blocks
      // let the allocator repeatedly recycle accumulators for B loads when
      // the microscale F32 accumulators also occupy registers.
      asm("%{vex%} vpdpbusd %[a], %[b0], %[c0]\n\t"
          "%{vex%} vpdpbusd %[a], %[b1], %[c1]\n\t"
          "%{vex%} vpdpbusd %[a], %[b2], %[c2]\n\t"
          "%{vex%} vpdpbusd %[a], %[b3], %[c3]"
          : [c0] "+&x"(c0.raw), [c1] "+&x"(c1.raw), [c2] "+&x"(c2.raw),
            [c3] "+&x"(c3.raw)
          : [a] "x"(a.raw), [b0] "x"(b0.raw), [b1] "x"(b1.raw),
            [b2] "x"(b2.raw), [b3] "x"(b3.raw));
      return;
    }
    // VEX encoding: AVX-VNNI is available without AVX-512. Dispatch per tile.
    asm("%{vex%} vpdpbusd %[a], %[b], %[c]"
        : [c] "+x"(c0.raw)
        : [a] "x"(a.raw), [b] "x"(b0.raw));
    asm("%{vex%} vpdpbusd %[a], %[b], %[c]"
        : [c] "+x"(c1.raw)
        : [a] "x"(a.raw), [b] "x"(b1.raw));
    asm("%{vex%} vpdpbusd %[a], %[b], %[c]"
        : [c] "+x"(c2.raw)
        : [a] "x"(a.raw), [b] "x"(b2.raw));
    asm("%{vex%} vpdpbusd %[a], %[b], %[c]"
        : [c] "+x"(c3.raw)
        : [a] "x"(a.raw), [b] "x"(b3.raw));
    return;
  }
#endif
  MMQuantizedDot4Accumulate<GEMMA_MM_I8_BIASED_B>(di32, a, b0, b1, b2, b3, c0,
                                                  c1, c2, c3);
}

static inline size_t MMI8EnvChoice(const char* name, size_t fallback,
                                   size_t alternative) {
  const char* value = getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  const size_t parsed = static_cast<size_t>(strtoull(value, nullptr, 10));
  return parsed == alternative ? alternative : fallback;
}

static inline size_t MMI8RotateBlockSize() {
  static const size_t block =
      MMI8EnvChoice("GEMMA_MM_I8_BLOCK_SIZE", kMMI8DefaultRotateBlock, 64);
  return block;
}

static inline size_t MMI8HashBits() {
  static const size_t bits =
      MMI8EnvChoice("GEMMA_MM_I8_HASH_BITS", kMMI8DefaultHashBits, 16);
  return bits;
}

// Optional local quantization scales, independent of the rotation block.
// Smaller groups limit outlier influence; larger groups reduce kernel overhead.
static inline size_t MMI8QuantBlockSize() {
  static const bool enabled = MMI8Flag("GEMMA_MM_I8_MICROSCALE");
  static const size_t block = []() {
    const char* value = getenv("GEMMA_MM_I8_QUANT_BLOCK_SIZE");
    const size_t requested = value ? strtoull(value, nullptr, 10) : 0;
    return requested == 32 || requested == 64 || requested == 128
               ? requested
               : MMI8RotateBlockSize();
  }();
  return enabled ? block : 0;
}

static inline bool MMI8FastMicro() {
  static const bool enabled = MMI8Flag("GEMMA_MM_I8_FAST_MICRO", true);
  return enabled;
}

// A bijective mixer over uint16_t. Thus the full 65536-value sequence has no
// collisions and its high bit is exactly balanced. Only that high bit is used
// as the Rademacher sign.
static HWY_INLINE uint16_t MMI8Hash16(uint16_t value) {
  value = static_cast<uint16_t>(value + 0x9E37u);
  value ^= static_cast<uint16_t>(value >> 7);
  value = static_cast<uint16_t>(value * 0x85EBu);
  value ^= static_cast<uint16_t>(value >> 9);
  value = static_cast<uint16_t>(value * 0xC2B3u);
  value ^= static_cast<uint16_t>(value >> 8);
  return value;
}

static HWY_INLINE bool MMI8NegativeSign(size_t position, size_t hash_bits) {
  if (hash_bits == 16) {
    return (MMI8Hash16(static_cast<uint16_t>(position)) >> 15) != 0;
  }
  const uint32_t hash =
      static_cast<uint32_t>(position) * 0x9E3779B9u + 0x7F4A7C15u;
  return (hash >> 31) != 0;
}

template <size_t block_size>
static HWY_NOINLINE void MMI8RotateFixed(float* HWY_RESTRICT row, size_t k,
                                         size_t hash_bits) {
  HWY_DASSERT(block_size == 64 || block_size == 128);
  HWY_DASSERT(hash_bits == 16 || hash_bits == 32);
  HWY_DASSERT((k % block_size) == 0);
  const float normalize = block_size == 64 ? 0.125f : 0.08838834764831845f;
  const hn::CappedTag<float, 8> df;
  const hn::Rebind<uint32_t, decltype(df)> du;
  const size_t lanes = hn::Lanes(df);
  const bool fast = MMI8FastRotate();
  thread_local hwy::AlignedVector<uint32_t> signs;
  thread_local size_t cached_hash = 0;
  if (fast && (signs.size() < k || cached_hash != hash_bits)) {
    signs.resize(k);
    for (size_t i = 0; i < k; ++i) {
      signs[i] = MMI8NegativeSign(i, hash_bits) ? 0x80000000u : 0u;
    }
    cached_hash = hash_bits;
  }
  for (size_t block = 0; block < k; block += block_size) {
    float* HWY_RESTRICT x = row + block;
    for (size_t i = 0; !fast && i < block_size; ++i) {
      // Deterministic Rademacher diagonal, shared by A and B.
      if (MMI8NegativeSign(block + i, hash_bits)) x[i] = -x[i];
    }
    // Complete stages smaller than a SIMD register with lane butterflies.
    // Select left/right first so subtraction has exactly the scalar order.
    if (fast) {
      const auto lane = hn::Iota(du, 0);
      for (size_t i = 0; i < block_size; i += lanes) {
        // Apply the signs while loading the first butterfly stage.
        auto v = hn::BitCast(
            df, hn::Xor(hn::BitCast(du, hn::LoadU(df, x + i)),
                        hn::LoadU(du, signs.data() + block + i)));
        for (size_t width = 1; width < lanes; width *= 2) {
          const auto bit = hn::Set(du, static_cast<uint32_t>(width));
          const auto perm = hn::IndicesFromVec(df, hn::Xor(lane, bit));
          const auto other = hn::TableLookupLanes(v, perm);
          const auto upper =
              hn::RebindMask(df, hn::Ne(hn::And(lane, bit), hn::Zero(du)));
          const auto left = hn::IfThenElse(upper, other, v);
          const auto right = hn::IfThenElse(upper, v, other);
          v = hn::IfThenElse(upper, hn::Sub(left, right), hn::Add(left, right));
        }
        hn::StoreU(v, df, x + i);
      }
    }
    const size_t end_width = fast ? block_size / 2 : block_size;
    for (size_t width = fast ? lanes : 1; width < end_width; width *= 2) {
      for (size_t start = 0; start < block_size; start += 2 * width) {
        size_t i = 0;
        for (; fast && i + lanes <= width; i += lanes) {
          const auto left = hn::LoadU(df, x + start + i);
          const auto right = hn::LoadU(df, x + start + width + i);
          hn::StoreU(hn::Add(left, right), df, x + start + i);
          hn::StoreU(hn::Sub(left, right), df, x + start + width + i);
        }
        for (; i < width; ++i) {
          const float left = x[start + i];
          const float right = x[start + width + i];
          x[start + i] = left + right;
          x[start + width + i] = left - right;
        }
      }
    }
    if (fast) {
      // Normalize the final butterfly outputs before storing them, preserving
      // the original add/subtract-then-multiply order without another pass.
      constexpr size_t half = block_size / 2;
      const auto vnormalize = hn::Set(df, normalize);
      for (size_t i = 0; i < half; i += lanes) {
        const auto left = hn::LoadU(df, x + i);
        const auto right = hn::LoadU(df, x + half + i);
        hn::StoreU(hn::Mul(hn::Add(left, right), vnormalize), df, x + i);
        hn::StoreU(hn::Mul(hn::Sub(left, right), vnormalize), df,
                   x + half + i);
      }
    } else {
      for (size_t i = 0; i < block_size; ++i) x[i] *= normalize;
    }
  }
}

static HWY_NOINLINE void MMI8Rotate(float* HWY_RESTRICT row, size_t k,
                                    size_t block_size, size_t hash_bits) {
  HWY_ASSERT(block_size == 64 || block_size == 128);
  if (block_size == 64)
    MMI8RotateFixed<64>(row, k, hash_bits);
  else
    MMI8RotateFixed<128>(row, k, hash_bits);
}

static HWY_NOINLINE void MMI8Rotate(float* HWY_RESTRICT row, size_t k) {
  MMI8Rotate(row, k, MMI8RotateBlockSize(), MMI8HashBits());
}

// Data-free L2 equalization for an FFN's multiplicative up projection and
// down projection. If hidden activations are multiplied by this value and the
// corresponding down-projection input column is divided by it, the
// full-precision computation is unchanged. Bounds prevent zero/tiny norms or
// unusually imbalanced channels from creating extreme values.
HWY_INLINE_VAR constexpr double kMMI8L2NormFloor = 1E-12;
HWY_INLINE_VAR constexpr float kMMI8L2ScaleMin = 1.0f / 16.0f;
HWY_INLINE_VAR constexpr float kMMI8L2ScaleMax = 16.0f;

static HWY_INLINE float MMI8L2Scale(double up_l2, double down_l2,
                                    bool* clamped = nullptr) {
  const double safe_up = HWY_MAX(up_l2, kMMI8L2NormFloor);
  const double safe_down = HWY_MAX(down_l2, kMMI8L2NormFloor);
  const float raw = static_cast<float>(std::sqrt(safe_down / safe_up));
  const float scale = HWY_MIN(kMMI8L2ScaleMax, HWY_MAX(kMMI8L2ScaleMin, raw));
  if (clamped != nullptr) *clamped = scale != raw;
  return scale;
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
    return ViewGroup(r, c, cols, block_size ? c / block_size : 0);
  }

  // As View, with an already-known quantization group to avoid division in
  // the microscaling kernel's group loop.
  MMI8AView ViewGroup(size_t r, size_t c, size_t cols, size_t group) const {
    return MMI8AView{
        data.View(r, c, cols),
        scale + r + group * scale_stride,
        prefix + r * prefix_stride + c,
        prefix_stride,
        scale_stride,
        block_size};
  }

  // Sum of the quantized values of row `r` over the `cols` columns of this
  // view. `prefix` has `K + 1` entries per row, so this is exact for any range.
  int32_t RowSum(size_t r, size_t cols) const {
    const int32_t* HWY_RESTRICT p = prefix + r * prefix_stride;
    return p[cols] - p[0];
  }

  StridedView<MMI8AT> data{nullptr, 0, 0};
  const float* HWY_RESTRICT scale;     // one per row of `data`
  const int32_t* HWY_RESTRICT prefix;  // null unless `GEMMA_MM_I8_BIASED_B`
  size_t prefix_stride;
  size_t scale_stride = 0;  // group-major scales, one column per activation row
  size_t block_size = 0;
  // Optional second quantization of the first stream's reconstruction error.
  // Set only on the whole-matrix view; its lifetime covers MMLoops::Dispatch.
  const MMI8AView* residual = nullptr;
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
  // Optional per-K multiplier applied to A before rotation. The packed B has
  // already been divided by the same values, preserving the dot product.
  const float* HWY_RESTRICT a_pre_scale = nullptr;
  size_t block_size = 0;  // scale[g * Rows() + row], or one scale per row
  const float* HWY_RESTRICT bias = nullptr;  // optional calibrated output bias
  // Eight output channels interleaved in four-K chunks. Each packed tile
  // starts at data->Row(tile_row), preserving the allocation and its padding.
  // Individual data->Row(r) values are no longer ordinary weight rows.
  bool packed_micro = false;
  bool dual_a = false;  // two activation streams with shared weights
};

static inline bool MMI8UseDualA(const MMI8B& B, size_t m) {
  static const bool m1_only = MMI8Flag("GEMMA_MM_I8_DUAL_A_M1_ONLY");
  return B.dual_a && B.packed_micro && B.block_size != 0 &&
         (!m1_only || m == 1);
}

// Existing model bias takes precedence. This experimental correction currently
// applies only to otherwise bias-free MatMuls, including each fused FFN branch.
static HWY_INLINE const float* MMI8Bias(const MMI8B& B, const float* add,
                                        size_t row_b) {
  const float* base = add != nullptr ? add : B.bias;
  return base != nullptr ? base + row_b : nullptr;
}

// Restrict automatic packing to the measured AVX2/VNNI path. A packed B is
// always interpreted by its layout flag, even if a kernel control is disabled.
static inline bool MMI8PackedHead() {
  static const bool enabled = MMI8Flag("GEMMA_MM_I8_PACKED_HEAD");
  return enabled && MMI8FastMicro() && MMI8NativeVNNI();
}

// A continuous head scan can improve bandwidth. Keep this independent from
// transformer scheduling: only a single-token, packed head with F32 output is
// eligible, because a changed KC boundary changes floating-point sum order.
static inline bool MMI8PreferFullHeadK(const MMI8B& B, size_t m,
                                      bool f32_output) {
  static const bool enabled = MMI8Flag("GEMMA_MM_I8_PACKED_HEAD_FULL_K");
  return enabled && B.packed_micro && B.Rows() >= 65536 && m == 1 && f32_output;
}

// In-place N8 x K4 transpose, using only one tile of temporary storage. The
// original eight-row padding follows the packed bytes at the end of the tile.
// Call only after every quantized weight and calibration update is final.
static HWY_NOINLINE void MMI8PackMicroB(MatPtrT<int8_t>& data) {
  const size_t k = data.Cols();
  HWY_ASSERT(data.Rows() % 8 == 0 && k % 4 == 0);
  hwy::AlignedVector<int8_t> tile(8 * k);
  for (size_t r = 0; r < data.Rows(); r += 8) {
    for (size_t c = 0; c < k; c += 4) {
      for (size_t n = 0; n < 8; ++n) {
        hwy::CopyBytes<4>(data.Row(r + n) + c, tile.data() + 8 * c + 4 * n);
      }
    }
    hwy::CopyBytes(tile.data(), data.Row(r), tile.size());
  }
}

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
  template <bool kFast = false, class DI32, class VI32 = hn::Vec<DI32>,
            class D4 = hn::Full128<int32_t>, class V4 = hn::Vec<D4>>
  HWY_INLINE void Reduce4x4(DI32 di32,                               //
                            VI32 C00, VI32 C01, VI32 C02, VI32 C03,  //
                            VI32 C10, VI32 C11, VI32 C12, VI32 C13,  //
                            VI32 C20, VI32 C21, VI32 C22, VI32 C23,  //
                            VI32 C30, VI32 C31, VI32 C32, VI32 C33,  //
                            V4& sum0, V4& sum1, V4& sum2, V4& sum3) {
#if HWY_TARGET == HWY_AVX2
    if constexpr (kFast) {
      // Two pairwise additions produce the four column sums within each
      // 128-bit half; one final add combines halves, without a stack transpose.
      const D4 d4;
      const auto reduce = [&](VI32 c0, VI32 c1, VI32 c2, VI32 c3) HWY_ATTR {
        const auto pairs =
            hn::PairwiseAdd128(di32, hn::PairwiseAdd128(di32, c0, c1),
                               hn::PairwiseAdd128(di32, c2, c3));
        return hn::Add(hn::LowerHalf(d4, pairs), hn::UpperHalf(d4, pairs));
      };
      sum0 = reduce(C00, C01, C02, C03);
      if constexpr (kRowsAC > 1) sum1 = reduce(C10, C11, C12, C13);
      if constexpr (kRowsAC > 2) sum2 = reduce(C20, C21, C22, C23);
      if constexpr (kRowsAC > 3) sum3 = reduce(C30, C31, C32, C33);
      return;
    }
#endif
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
        sum = hn::Sub(sum,
                      hn::Set(d4i, static_cast<int32_t>(a_rowsum[kRow] * 128)));
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
    if (B.packed_micro) {
      PackedMicroB3A2C0(A, B, range_mc, range_kc, range_nc, args, out_tag,
                       C_MC_NC);
      return;
    }
    if (B.block_size != 0) {
      if (MMI8FastMicro()) {
        MicroB3A2C0(A, B, range_mc, range_kc, range_nc, args, out_tag,
                   C_MC_NC);
        return;
      }
      // Accumulate group results in F32, rounding to BF16 only at the KC
      // boundary. Adding BF16 partials per tiny group loses too much accuracy.
      thread_local hwy::AlignedVector<float> sums;
      sums.resize(range_mc.Num() * kNR);
      const StridedView<float> tmp(sums.data(), kNR, kNR);
      for (size_t inc = 0; inc < range_nc.Num(); inc += kNR) {
        const size_t row_b = range_nc.begin() + inc;
        const float* add = MMI8Bias(B, args.add, row_b);
        bool first = true;
        for (size_t c = range_kc.begin(); c < range_kc.end();) {
          const size_t group = c / B.block_size;
          const size_t count = HWY_MIN(static_cast<size_t>(range_kc.end()),
                                       (group + 1) * B.block_size) -
                               c;
          const auto av = A.View(range_mc.begin(), c, count);
          const StridedView<int8_t> bv(*B.data, row_b, c, count);
          const float* scales = B.scale + group * B.Rows() + row_b;
          if (first)
            A2C0(av, bv, scales, args.mr, range_mc, count, nullptr, MMSetC(),
                 tmp);
          else
            A2C0(av, bv, scales, args.mr, range_mc, count, nullptr, MMAddC(),
                 tmp);
          first = false;
          c += count;
        }
        using TC = hwy::RemoveCvRef<decltype(C_MC_NC.Row(0)[0])>;
        for (size_t r = 0; r < range_mc.Num(); ++r) {
          for (size_t j = 0; j < kNR; ++j) {
            float value = sums[r * kNR + j];
            if constexpr (hwy::IsSame<Tag, MMAddC>()) {
              value += hwy::ConvertScalarTo<float>(C_MC_NC.Row(r)[inc + j]);
            } else if (add != nullptr) {
              value += add[j];
            }
            C_MC_NC.Row(r)[inc + j] = hwy::ConvertScalarTo<TC>(value);
          }
        }
      }
      return;
    }
    const size_t kc = range_kc.Num();
    const AView A_view = A.View(range_mc.begin(), range_kc.begin(), kc);

    for (size_t inc = 0; inc < range_nc.Num(); inc += kNR) {
      // For `add` and `B`, which are global, unlike `C_MC_NC`.
      const size_t row_b = range_nc.begin() + inc;
      // No decompression: `B` is already in the layout the kernel wants.
      const StridedView<int8_t> B_view(*B.data, row_b, range_kc.begin(), kc);
      const CView C_MC_NR = C_MC_NC.View(0, inc, kNR);
      const float* HWY_RESTRICT add = MMI8Bias(B, args.add, row_b);
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
  template <size_t kRowsAC, bool kNative, bool kFastReduce, class V4>
  static HWY_INLINE void DotProducts(const AView& A_view,
                                     const StridedView<int8_t>& B_view,
                                     size_t imc, size_t kc, V4& sum0, V4& sum1,
                                     V4& sum2, V4& sum3) {
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
          MMI8Dot4<kNative, kFastReduce>(di32, a0, b0, b1, b2, b3, C00, C01,
                                         C02, C03);
        }
        if constexpr (kRowsAC > 1) {
          const VA8 a1 = hn::LoadU(da8, ar1 + ikc);
          MMI8Dot4<kNative, kFastReduce>(di32, a1, b0, b1, b2, b3, C10, C11,
                                         C12, C13);
        }
        if constexpr (kRowsAC > 2) {
          const VA8 a2 = hn::LoadU(da8, ar2 + ikc);
          MMI8Dot4<kNative, kFastReduce>(di32, a2, b0, b1, b2, b3, C20, C21,
                                         C22, C23);
        }
        if constexpr (kRowsAC > 3) {
          const VA8 a3 = hn::LoadU(da8, ar3 + ikc);
          MMI8Dot4<kNative, kFastReduce>(di32, a3, b0, b1, b2, b3, C30, C31,
                                         C32, C33);
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
        MMI8Dot4<kNative, kFastReduce>(di32, a0, b0, b1, b2, b3, C00, C01, C02,
                                       C03);
      }
      if constexpr (kRowsAC > 1) {
        const VA8 a1 = hn::LoadN(da8, ar1 + ikc, remaining_kc);
        MMI8Dot4<kNative, kFastReduce>(di32, a1, b0, b1, b2, b3, C10, C11, C12,
                                       C13);
      }
      if constexpr (kRowsAC > 2) {
        const VA8 a2 = hn::LoadN(da8, ar2 + ikc, remaining_kc);
        MMI8Dot4<kNative, kFastReduce>(di32, a2, b0, b1, b2, b3, C20, C21, C22,
                                       C23);
      }
      if constexpr (kRowsAC > 3) {
        const VA8 a3 = hn::LoadN(da8, ar3 + ikc, remaining_kc);
        MMI8Dot4<kNative, kFastReduce>(di32, a3, b0, b1, b2, b3, C30, C31, C32,
                                       C33);
      }
    }

    MMI8StoreHorizontalSumsIntoC<kRowsAC> horz;
    horz.template Reduce4x4<kFastReduce>(di32, C00, C01, C02, C03, C10, C11,
                                         C12, C13, C20, C21, C22, C23, C30, C31,
                                         C32, C33, sum0, sum1, sum2, sum3);
  }

  template <size_t kRowsAC, bool kNative, class Tag, class CView>
  static HWY_INLINE void LoopKCImpl(const AView A_view,
                                    const StridedView<int8_t> B_view,
                                    const float* HWY_RESTRICT b_scale,
                                    size_t imc, size_t kc,
                                    const float* HWY_RESTRICT add, Tag tag,
                                    CView C_MC_NR) {
    const hn::Full128<int32_t> d4i;
    hn::Vec<decltype(d4i)> sum0, sum1, sum2, sum3;
    DotProducts<kRowsAC, kNative, false>(A_view, B_view, imc, kc, sum0, sum1,
                                       sum2, sum3);

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
    horz.Store(d4i, sum0, sum1, sum2, sum3, A_view.scale, a_rowsum, b_scale,
               add, imc, tag, C_MC_NR);
  }

  template <size_t kRow, size_t kRowsAC, class V4I, class V4F>
  static HWY_INLINE void AccumulateMicro(const AView A, size_t kc, V4I sum,
                                         V4F b_scale, V4F& accum) {
    if constexpr (kRow < kRowsAC) {
      const hn::Full128<int32_t> di;
      const hn::Full128<float> df;
      if constexpr (GEMMA_MM_I8_BIASED_B) {
        sum = hn::Sub(sum, hn::Set(di, A.RowSum(kRow, kc) * 128));
      }
      const auto scale = hn::Mul(b_scale, hn::Set(df, A.scale[kRow]));
      // Match the reference's group order and one FMA per group exactly.
      accum = hn::MulAdd(hn::ConvertTo(df, sum), scale, accum);
    }
  }

  template <size_t kRow, size_t kRowsAC, class Tag, class CView, class V4F>
  static HWY_INLINE void StoreMicro(V4F accum, size_t imc, const float* add,
                                    Tag, CView C) {
    if constexpr (kRow < kRowsAC) {
      const hn::Full128<float> df;
      using TC = hwy::RemoveCvRef<decltype(C.Row(0)[0])>;
      const hn::Rebind<TC, decltype(df)> dc;
      TC* HWY_RESTRICT pos = C.Row(imc + kRow);
      if constexpr (hwy::IsSame<Tag, MMAddC>()) {
        accum = hn::Add(accum, F32FromTC(dc, hn::LoadU(dc, pos)));
      } else if (add != nullptr) {
        accum = hn::Add(accum, hn::LoadU(df, add));
      }
      hn::StoreU(TCFromF32(dc, accum), dc, pos);
    }
  }

  template <size_t kRowsAC, bool kNative, size_t kBlock, typename BT, class Tag,
            class CView>
  static HWY_NOINLINE void MicroTile(const AView& A, const BT& B,
                                     const IndexRange& range_mc,
                                     const IndexRange& range_kc, size_t row_b,
                                     size_t imc, const float* add, Tag tag,
                                     CView C) {
    const hn::Full128<int32_t> di;
    const hn::Full128<float> df;
    auto accum0 = hn::Zero(df), accum1 = hn::Zero(df);
    auto accum2 = hn::Zero(df), accum3 = hn::Zero(df);
    // Keep one F32 vector per output row across all quantization groups;
    // only the completed KC tile is written to C (and rounded if BF16).
    const size_t block = kBlock ? kBlock : B.block_size;
    size_t group = range_kc.begin() / block;
    for (size_t c = range_kc.begin(); c < range_kc.end(); ++group) {
      const size_t count = kBlock ? kBlock
                                  : HWY_MIN(static_cast<size_t>(range_kc.end()),
                                            (group + 1) * block) -
                                        c;
      const auto av = A.ViewGroup(range_mc.begin() + imc, c, count, group);
      const StridedView<int8_t> bv(*B.data, row_b, c, count);
      auto sum0 = hn::Zero(di), sum1 = hn::Zero(di);
      auto sum2 = hn::Zero(di), sum3 = hn::Zero(di);
      DotProducts<kRowsAC, kNative, true>(av, bv, 0, count, sum0, sum1, sum2,
                                          sum3);
      const auto scale = hn::LoadU(df, B.scale + group * B.Rows() + row_b);
      AccumulateMicro<0, kRowsAC>(av, count, sum0, scale, accum0);
      AccumulateMicro<1, kRowsAC>(av, count, sum1, scale, accum1);
      AccumulateMicro<2, kRowsAC>(av, count, sum2, scale, accum2);
      AccumulateMicro<3, kRowsAC>(av, count, sum3, scale, accum3);
      c += count;
    }
    StoreMicro<0, kRowsAC>(accum0, imc, add, tag, C);
    StoreMicro<1, kRowsAC>(accum1, imc, add, tag, C);
    StoreMicro<2, kRowsAC>(accum2, imc, add, tag, C);
    StoreMicro<3, kRowsAC>(accum3, imc, add, tag, C);
  }

  template <bool kNative, size_t kBlock, typename BT, class Tag, class CView>
  static HWY_INLINE void MicroB3A2C0Impl(const AView A, const BT& B,
                                         const IndexRange& range_mc,
                                         const IndexRange& range_kc,
                                         const IndexRange& range_nc,
                                         const MMArgs& args, Tag tag, CView C) {
    for (size_t inc = 0; inc < range_nc.Num(); inc += kNR) {
      const size_t row_b = range_nc.begin() + inc;
      const auto tile = C.View(0, inc, kNR);
      const float* add = MMI8Bias(B, args.add, row_b);
      const size_t mc = range_mc.Num();
      size_t r = 0;
      if (args.mr == 4) {
        for (; r + 4 <= mc; r += 4) {
          MicroTile<4, kNative, kBlock>(A, B, range_mc, range_kc, row_b, r, add,
                                        tag, tile);
        }
      }
      if (args.mr >= 2) {
        for (; r + 2 <= mc; r += 2) {
          MicroTile<2, kNative, kBlock>(A, B, range_mc, range_kc, row_b, r, add,
                                        tag, tile);
        }
      }
      for (; r < mc; ++r) {
        MicroTile<1, kNative, kBlock>(A, B, range_mc, range_kc, row_b, r, add,
                                      tag, tile);
      }
    }
  }

  // Covers arbitrary KC boundaries and targets without the optimized packed
  // kernel. Keeping this path makes the layout independent of runtime flags.
  template <typename BT, class Tag, class CView>
  static HWY_NOINLINE void PackedMicroReference(const AView& A, const BT& B,
                                                const IndexRange& range_mc,
                                                const IndexRange& range_kc,
                                                const IndexRange& range_nc,
                                                const MMArgs& args, Tag tag,
                                                CView C) {
    const hn::Full128<int32_t> di;
    const hn::Full128<float> df;
    for (size_t inc = 0; inc < range_nc.Num(); inc += 4) {
      const size_t row_b = range_nc.begin() + inc;
      const size_t packed_row = row_b & ~size_t{7};
      const size_t lane = row_b % 8;
      const auto* packed =
          reinterpret_cast<const MMI8BT*>(B.data->Row(packed_row));
      const float* add = MMI8Bias(B, args.add, row_b);
      const auto out = C.View(0, inc, 4);
      for (size_t r = 0; r < range_mc.Num(); ++r) {
        const auto* ar = A.data.Row(range_mc.begin() + r);
        auto accum = hn::Zero(df);
        for (size_t c = range_kc.begin(); c < range_kc.end();) {
          const size_t group = c / B.block_size;
          const size_t count = HWY_MIN(static_cast<size_t>(range_kc.end()),
                                       (group + 1) * B.block_size) -
                               c;
          HWY_ALIGN int32_t sums[4] = {};
          HWY_ALIGN int32_t residual_sums[4] = {};
          for (size_t k = c; k < c + count; ++k) {
            const auto* bp = packed + (k / 4) * 32 + lane * 4 + k % 4;
            for (size_t n = 0; n < 4; ++n) {
              sums[n] += static_cast<int32_t>(ar[k]) * bp[4 * n];
              if (A.residual != nullptr) {
                residual_sums[n] += static_cast<int32_t>(A.residual->data.Row(
                                        range_mc.begin() + r)[k]) *
                                    bp[4 * n];
              }
            }
          }
          const auto av = A.ViewGroup(range_mc.begin() + r, c, count, group);
          const auto scale = hn::LoadU(df, B.scale + group * B.Rows() + row_b);
          AccumulateMicro<0, 1>(av, count, hn::LoadU(di, sums), scale, accum);
          if (A.residual != nullptr) {
            const auto rv =
                A.residual->ViewGroup(range_mc.begin() + r, c, count, group);
            AccumulateMicro<0, 1>(rv, count, hn::LoadU(di, residual_sums),
                                  scale, accum);
          }
          c += count;
        }
        StoreMicro<0, 1>(accum, r, add, tag, out);
      }
    }
  }

#if HWY_TARGET == HWY_AVX2 && GEMMA_MM_I8_BIASED_B && defined(__GNUC__) && \
    !defined(__clang__)
  // Each dot lane directly produces one of eight output channels. Four
  // independent chains hide VNNI latency; no horizontal reduction is needed.
  template <size_t kBlock, bool kAlignedGroups = true, bool kDual = false,
            typename BT, class Tag, class CView>
  static HWY_NOINLINE void PackedMicroNative(const AView& A, const BT& B,
                                             const IndexRange& range_mc,
                                             const IndexRange& range_kc,
                                             const IndexRange& range_nc,
                                             const MMArgs& args, Tag tag,
                                             CView C) {
    const hn::ScalableTag<int8_t> da;
    const hn::ScalableTag<uint8_t> db;
    const hn::Repartition<int32_t, decltype(da)> di;
    const hn::Repartition<uint32_t, decltype(da)> du;
    const hn::ScalableTag<float> df;
    const hn::Full128<float> d4f;
    for (size_t nc = range_nc.begin(); nc < range_nc.end();) {
      const size_t row_b = nc & ~size_t{7};
      const size_t lane = nc % 8;
      const size_t count = HWY_MIN(size_t{8} - lane, range_nc.end() - nc);
      const size_t inc = nc - range_nc.begin();
      const auto* packed = reinterpret_cast<const uint8_t*>(B.data->Row(row_b));
      const float* add = MMI8Bias(B, args.add, nc);
      for (size_t r = 0; r < range_mc.Num(); ++r) {
        const auto* ar = A.data.Row(range_mc.begin() + r);
        const auto* ar1 =
            kDual ? A.residual->data.Row(range_mc.begin() + r) : nullptr;
        auto accum = hn::Zero(df);
        size_t group = range_kc.begin() / kBlock;
        for (size_t c = range_kc.begin(); c < range_kc.end(); ++group) {
          const size_t num_k =
              kAlignedGroups ? kBlock
                             : HWY_MIN(kBlock - c % kBlock, range_kc.end() - c);
          const auto* br = packed + c * 8;
          auto d0 = hn::Zero(di), d1 = hn::Zero(di);
          auto d2 = hn::Zero(di), d3 = hn::Zero(di);
          auto e0 = hn::Zero(di), e1 = hn::Zero(di);
          auto e2 = hn::Zero(di), e3 = hn::Zero(di);
          const auto dot = [&](size_t offset, auto& sum,
                               auto& residual_sum) HWY_ATTR {
            uint32_t bits;
            hwy::CopyBytes<4>(ar + c + offset, &bits);
            const auto a = hn::BitCast(da, hn::Set(du, bits));
            const auto b = hn::LoadU(db, br + offset * 8);
            if constexpr (kDual) {
              uint32_t residual_bits;
              hwy::CopyBytes<4>(ar1 + c + offset, &residual_bits);
              const auto a1 = hn::BitCast(da, hn::Set(du, residual_bits));
              // Keep B in a register for both dots. The dual specialization
              // requires more than the eight SIMD registers of x86-32.
              asm("%{vex%} vpdpbusd %[a], %[b], %[sum]\n\t"
                  "%{vex%} vpdpbusd %[a1], %[b], %[residual_sum]"
                  : [sum] "+&x"(sum.raw), [residual_sum] "+&x"(residual_sum.raw)
                  : [a] "x"(a.raw), [a1] "x"(a1.raw), [b] "x"(b.raw));
            } else {
              asm("%{vex%} vpdpbusd %[a], %[b], %[sum]"
                  : [sum] "+x"(sum.raw)
                  : [a] "x"(a.raw), [b] "x"(b.raw));
            }
          };
          size_t k = 0;
          for (; k + 16 <= num_k; k += 16) {
            dot(k, d0, e0);
            dot(k + 4, d1, e1);
            dot(k + 8, d2, e2);
            dot(k + 12, d3, e3);
          }
          if constexpr (!kAlignedGroups) {
            for (; k < num_k; k += 4) dot(k, d0, e0);
          }
          auto sum = hn::Add(hn::Add(d0, d1), hn::Add(d2, d3));
          const auto av = A.ViewGroup(range_mc.begin() + r, c, num_k, group);
          sum = hn::Sub(sum, hn::Set(di, av.RowSum(0, num_k) * 128));
          const auto bs = hn::LoadU(df, B.scale + group * B.Rows() + row_b);
          const auto scale = hn::Mul(bs, hn::Set(df, av.scale[0]));
          accum = hn::MulAdd(hn::ConvertTo(df, sum), scale, accum);
          if constexpr (kDual) {
            auto residual_sum = hn::Add(hn::Add(e0, e1), hn::Add(e2, e3));
            const auto rv =
                A.residual->ViewGroup(range_mc.begin() + r, c, num_k, group);
            residual_sum =
                hn::Sub(residual_sum, hn::Set(di, rv.RowSum(0, num_k) * 128));
            const auto residual_scale = hn::Mul(bs, hn::Set(df, rv.scale[0]));
            accum = hn::MulAdd(hn::ConvertTo(df, residual_sum), residual_scale,
                               accum);
          }
          c += num_k;
        }
        if (count == 8) {
          using TC = hwy::RemoveCvRef<decltype(C.Row(0)[0])>;
          const hn::Rebind<TC, decltype(df)> dc;
          TC* pos = C.Row(r) + inc;
          if constexpr (hwy::IsSame<Tag, MMAddC>()) {
            accum = hn::Add(accum, F32FromTC(dc, hn::LoadU(dc, pos)));
          } else if (add != nullptr) {
            accum = hn::Add(accum, hn::LoadU(df, add));
          }
          hn::StoreU(TCFromF32(dc, accum), dc, pos);
        } else {
          // Generic scheduling may split N at four-channel boundaries.
          HWY_DASSERT(count == 4);
          const auto half =
              lane ? hn::UpperHalf(d4f, accum) : hn::LowerHalf(d4f, accum);
          StoreMicro<0, 1>(half, r, add, tag, C.View(0, inc, 4));
        }
      }
      nc += count;
    }
  }
#endif

  template <bool kDual, typename BT, class Tag, class CView>
  static HWY_INLINE void DispatchPackedMicro(const AView& A, const BT& B,
                                             const IndexRange& range_mc,
                                             const IndexRange& range_kc,
                                             const IndexRange& range_nc,
                                             const MMArgs& args, Tag tag,
                                             CView C) {
    HWY_DASSERT(B.block_size == 32 || B.block_size == 64 ||
                B.block_size == 128);
    HWY_DASSERT(B.Rows() % 8 == 0 && range_nc.begin() % 4 == 0 &&
                range_nc.Num() % 4 == 0);
#if HWY_TARGET == HWY_AVX2 && GEMMA_MM_I8_BIASED_B && defined(__GNUC__) && \
    !defined(__clang__)
    if constexpr (HWY_ARCH_X86_64 || !kDual) {
      if (MMI8FastMicro() && MMI8NativeVNNI() && range_kc.begin() % 4 == 0 &&
          range_kc.end() % 4 == 0) {
        const bool aligned = range_kc.begin() % B.block_size == 0 &&
                             range_kc.end() % B.block_size == 0;
        if (B.block_size == 32) {
          if (aligned)
            PackedMicroNative<32, true, kDual>(A, B, range_mc, range_kc,
                                               range_nc, args, tag, C);
          else
            PackedMicroNative<32, false, kDual>(A, B, range_mc, range_kc,
                                                range_nc, args, tag, C);
        } else if (B.block_size == 64) {
          if (aligned)
            PackedMicroNative<64, true, kDual>(A, B, range_mc, range_kc,
                                               range_nc, args, tag, C);
          else
            PackedMicroNative<64, false, kDual>(A, B, range_mc, range_kc,
                                                range_nc, args, tag, C);
        } else {
          if (aligned)
            PackedMicroNative<128, true, kDual>(A, B, range_mc, range_kc,
                                                range_nc, args, tag, C);
          else
            PackedMicroNative<128, false, kDual>(A, B, range_mc, range_kc,
                                                 range_nc, args, tag, C);
        }
        return;
      }
    }
#endif
    PackedMicroReference(A, B, range_mc, range_kc, range_nc, args, tag, C);
  }

  template <typename BT, class Tag, class CView>
  static HWY_INLINE void PackedMicroB3A2C0(const AView& A, const BT& B,
                                           const IndexRange& range_mc,
                                           const IndexRange& range_kc,
                                           const IndexRange& range_nc,
                                           const MMArgs& args, Tag tag,
                                           CView C) {
    if (B.dual_a && A.residual != nullptr) {
      DispatchPackedMicro<true>(A, B, range_mc, range_kc, range_nc, args, tag,
                                C);
    } else {
      AView primary = A;
      primary.residual = nullptr;
      DispatchPackedMicro<false>(primary, B, range_mc, range_kc, range_nc, args,
                                 tag, C);
    }
  }
  template <bool kNative, typename BT, class Tag, class CView>
  static HWY_INLINE void DispatchMicroBlock(const AView& A, const BT& B,
                                            const IndexRange& range_mc,
                                            const IndexRange& range_kc,
                                            const IndexRange& range_nc,
                                            const MMArgs& args, Tag tag,
                                            CView C) {
    // Full groups dominate model inference. Fixed widths let the compiler
    // unroll their dot loops and remove all masked loads and view copies.
    const size_t block = B.block_size;
    if (range_kc.begin() % block != 0 || range_kc.end() % block != 0) {
      MicroB3A2C0Impl<kNative, 0>(A, B, range_mc, range_kc, range_nc, args, tag,
                                  C);
      return;
    }
    switch (block) {
      case 32:
        MicroB3A2C0Impl<kNative, 32>(A, B, range_mc, range_kc, range_nc, args,
                                     tag, C);
        return;
      case 64:
        MicroB3A2C0Impl<kNative, 64>(A, B, range_mc, range_kc, range_nc, args,
                                     tag, C);
        return;
      case 128:
        MicroB3A2C0Impl<kNative, 128>(A, B, range_mc, range_kc, range_nc, args,
                                      tag, C);
        return;
      default:
        HWY_ABORT("Invalid microscaling group size %zu", block);
    }
  }

  template <typename BT, class Tag, class CView>
  static HWY_INLINE void MicroB3A2C0(const AView A, const BT& B,
                                     const IndexRange& range_mc,
                                     const IndexRange& range_kc,
                                     const IndexRange& range_nc,
                                     const MMArgs& args, Tag tag, CView C) {
#if HWY_TARGET == HWY_AVX2 && GEMMA_MM_I8_BIASED_B && defined(__GNUC__) && \
    !defined(__clang__)
    if (MMI8NativeVNNI()) {
      DispatchMicroBlock<true>(A, B, range_mc, range_kc, range_nc, args, tag,
                               C);
      return;
    }
#endif
    DispatchMicroBlock<false>(A, B, range_mc, range_kc, range_nc, args, tag, C);
  }

  template <size_t kRowsAC, class Tag, class CView>
  static HWY_INLINE void LoopKC(const AView A, const StridedView<int8_t> B,
                                const float* scale, size_t imc, size_t kc,
                                const float* add, Tag tag, CView C) {
#if HWY_TARGET == HWY_AVX2 && GEMMA_MM_I8_BIASED_B && defined(__GNUC__) && \
    !defined(__clang__)
    if (MMI8NativeVNNI()) {
      LoopKCImpl<kRowsAC, true>(A, B, scale, imc, kc, add, tag, C);
      return;
    }
#endif
    LoopKCImpl<kRowsAC, false>(A, B, scale, imc, kc, add, tag, C);
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
                                     size_t padded_k,
                                     int32_t prefix_base = 0) {
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
    int32_t sum = prefix_base;
    prefix[0] = prefix_base;
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

  MMI8AView View(const Extents2D& extents, size_t block_size = 0) {
    const size_t groups = block_size ? extents.cols / block_size : 1;
    if (scale_.size() < groups * data_.Rows())
      scale_.resize(groups * data_.Rows());
    HWY_DASSERT(extents.rows <= data_.Rows());
    HWY_DASSERT(extents.cols <= data_.Cols());
    return MMI8AView{
        StridedView<MMI8AT>(HWY_RCAST_ALIGNED(MMI8AT*, data_.Row(0)),
                            extents.cols, data_.Stride()),
        scale_.data(),
        prefix_.data(),
        prefix_stride_,
        data_.Rows(),
        block_size};
  }

  float* HWY_RESTRICT scale() { return scale_.data(); }
  int32_t* HWY_RESTRICT prefix(size_t row) {
    return prefix_.data() + (GEMMA_MM_I8_BIASED_B ? row : 0) * prefix_stride_;
  }
  size_t Stride() const { return data_.Stride(); }

  // Allocate only when requested; additional storage scales with the current
  // activation batch. Both streams use the same quantized weights.
  MMI8AView ResidualView(const Extents2D& extents, size_t block_size) {
    HWY_ASSERT(block_size != 0);
    const size_t groups = extents.cols / block_size;
    residual_data_.resize(extents.rows * data_.Stride());
    residual_prefix_.resize((GEMMA_MM_I8_BIASED_B ? extents.rows : 1) *
                            prefix_stride_);
    residual_scale_.resize(extents.rows * groups);
    return MMI8AView{StridedView<MMI8AT>(residual_data_.data(), extents.cols,
                                         data_.Stride()),
                     residual_scale_.data(),
                     residual_prefix_.data(),
                     prefix_stride_,
                     extents.rows,
                     block_size};
  }

  float* residual_scale() { return residual_scale_.data(); }
  int32_t* residual_prefix(size_t row) {
    return residual_prefix_.data() +
           (GEMMA_MM_I8_BIASED_B ? row : 0) * prefix_stride_;
  }

 private:
  MatStorageT<uint8_t> data_;
  size_t prefix_stride_;
  hwy::AlignedVector<int32_t> prefix_;
  hwy::AlignedVector<float> scale_;
  hwy::AlignedVector<int8_t> residual_data_;
  hwy::AlignedVector<int32_t> residual_prefix_;
  hwy::AlignedVector<float> residual_scale_;
};

// Copies an activation row before rotation. The optional F32 roundtrip uses
// exactly the same decompressor as MMDecompress::DecompressA, so comparisons
// with the SFP path begin with the same BF16-rounded activation values.
template <typename TA>
static HWY_INLINE void MMI8PrepareInputRow(
    const TA* HWY_RESTRICT in, size_t k, const float* HWY_RESTRICT a_pre_scale,
    bool match_bf16, float* HWY_RESTRICT out) {
  if constexpr (IsF32<TA>()) {
    if (match_bf16) {
      const hn::ScalableTag<BF16> dbf;
      const size_t padded = hwy::RoundUpTo(k, hn::Lanes(dbf));
      thread_local hwy::AlignedVector<BF16> rounded;
      if (rounded.size() < padded) rounded.resize(padded);
      DecompressAndZeroPad(dbf, MakeSpan(in, k), 0, rounded.data(), k);
      for (size_t c = 0; c < k; ++c) {
        const float value = hwy::ConvertScalarTo<float>(rounded[c]);
        out[c] = a_pre_scale == nullptr ? value : value * a_pre_scale[c];
      }
      return;
    }
  }
  for (size_t c = 0; c < k; ++c) {
    const float value = hwy::ConvertScalarTo<float>(in[c]);
    out[c] = a_pre_scale == nullptr ? value : value * a_pre_scale[c];
  }
}

// Quantizes all `M x K` of `A` into `storage`, in parallel over rows.
// This replaces `MMDecompress::DecompressA` and is the same order of cost:
// one pass over `A`, once per `MatMul` rather than per B tile.
template <typename TA>
static HWY_NOINLINE MMI8AView
QuantizeA(const MatPtrT<TA>& A, MMI8AStorage& storage, ThreadingContext& ctx,
          size_t cluster_idx, const float* a_pre_scale = nullptr,
          size_t block_size = 0, MMI8AView* residual = nullptr) {
  MMI8AView view = storage.View(A.Extents(), block_size);
  if (residual != nullptr) {
    *residual = storage.ResidualView(A.Extents(), block_size);
    view.residual = residual;
  }
  const size_t k = A.Cols();
  HWY_ASSERT(block_size == 0 ||
             ((block_size == 32 || block_size == 64 || block_size == 128) &&
              k % block_size == 0));
  const size_t padded_k =
      hwy::RoundUpTo(k, hn::Lanes(hn::ScalableTag<int8_t>()));
  float* HWY_RESTRICT scale = storage.scale();
  const float a_scale = A.Scale();
  static const bool match_bf16 = MMI8Flag("GEMMA_MM_I8_MATCH_BF16_A");
  HWY_DASSERT((k % MMI8RotateBlockSize()) == 0);

  ParallelFor(
      Parallelism::kFlat, A.Rows(), ctx, cluster_idx, Callers::kMMQuantizeA,
      [&](size_t r, size_t /*worker*/) HWY_ATTR {
        thread_local hwy::AlignedVector<float> rotated;
        if (rotated.size() < padded_k) rotated.resize(padded_k);
        MMI8PrepareInputRow(A.Row(r), k, a_pre_scale, match_bf16,
                            rotated.data());
        MMI8Rotate(rotated.data(), k);
        const size_t group_size = block_size ? block_size : k;
        int32_t* prefix = storage.prefix(r);
        for (size_t c = 0; c < k; c += group_size) {
          const int32_t base = GEMMA_MM_I8_BIASED_B && c ? prefix[c] : 0;
          const float raw_scale =
              QuantizeRowA(rotated.data() + c, group_size, view.data.Row(r) + c,
                           prefix + c, group_size, base);
          scale[(c / group_size) * view.scale_stride + r] = a_scale * raw_scale;
          if (residual != nullptr) {
            const hn::CappedTag<float, 32> df;
            const hn::Rebind<int32_t, decltype(df)> di;
            const hn::Rebind<int8_t, decltype(df)> d8;
            for (size_t j = 0; j < group_size; j += hn::Lanes(df)) {
              const auto q = hn::ConvertTo(
                  df,
                  hn::PromoteTo(di, hn::LoadU(d8, view.data.Row(r) + c + j)));
              const auto error =
                  hn::NegMulAdd(q, hn::Set(df, raw_scale),
                                hn::LoadU(df, rotated.data() + c + j));
              hn::StoreU(error, df, rotated.data() + c + j);
            }
            int32_t* rp = storage.residual_prefix(r);
            const int32_t residual_base = GEMMA_MM_I8_BIASED_B && c ? rp[c] : 0;
            storage.residual_scale()[(c / group_size) * residual->scale_stride +
                                     r] =
                a_scale * QuantizeRowA(rotated.data() + c, group_size,
                                       residual->data.Row(r) + c, rp + c,
                                       group_size, residual_base);
          }
        }
        for (size_t c = k; c < padded_k; ++c) {
          view.data.Row(r)[c] = 0;
          if (residual != nullptr) residual->data.Row(r)[c] = 0;
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
                                ThreadingContext& ctx,
                                const float* a_pre_scale = nullptr,
                                size_t block_size = 0) {
  const size_t k = B_f32.Cols();
  HWY_ASSERT(block_size == 0 ||
             ((block_size == 32 || block_size == 64 || block_size == 128) &&
              k % block_size == 0));
  HWY_DASSERT((k % MMI8RotateBlockSize()) == 0);
  const float b_scale = B_f32.Scale();

  ParallelFor(
      Parallelism::kFlat, B_f32.Rows(), ctx, /*cluster_idx=*/0, Callers::kTest,
      [&](size_t r, size_t /*worker*/) HWY_ATTR {
        hwy::AlignedVector<float> rotated(k);
        hwy::CopyBytes(B_f32.Row(r), rotated.data(), k * sizeof(float));
        if (a_pre_scale != nullptr) {
          for (size_t c = 0; c < k; ++c) rotated[c] /= a_pre_scale[c];
        }
        MMI8Rotate(rotated.data(), k);
        const float* HWY_RESTRICT in = rotated.data();
        const size_t group_size = block_size ? block_size : k;
        MMI8BT* HWY_RESTRICT out = HWY_RCAST_ALIGNED(MMI8BT*, data.Row(r));
        for (size_t begin = 0; begin < k; begin += group_size) {
          float amax = 0.0f;
          for (size_t c = begin; c < begin + group_size; ++c)
            amax = HWY_MAX(amax, hwy::ScalarAbs(in[c]));
          const float qs = amax == 0.0f ? 1.0f : amax / kMMI8Max;
          const float inv = amax == 0.0f ? 0.0f : kMMI8Max / amax;
          for (size_t c = begin; c < begin + group_size; ++c) {
            const int32_t q = static_cast<int32_t>(std::lroundf(in[c] * inv));
            out[c] = static_cast<MMI8BT>(q + (GEMMA_MM_I8_BIASED_B ? 128 : 0));
          }
          scale[(begin / group_size) * B_f32.Rows() + r] = b_scale * qs;
        }
        for (size_t c = k; c < data.Stride(); ++c) out[c] = 0;
      });

  return MMI8B{&data, scale, a_pre_scale, block_size};
}

//------------------------------------------------------------------------------
// Entry point

static inline std::vector<MMConfig> MMI8Candidates(
    MatMulEnv& env, size_t M, size_t K, size_t N, size_t num_B,
    size_t sizeof_TC, bool prefer_full_k = false) {
  auto candidates = MMCandidates(env.ctx.cache_info, M, K, N, num_B,
                                  sizeof_TC, env.print_config);
  if (!env.autotune &&
      (prefer_full_k || MMI8Flag("GEMMA_MM_I8_MIN_K_SPLITS"))) {
    // Generic candidates enumerate split-K loop orders first. Prefer fewer
    // intermediate output rounds in fixed W8A8 evaluation while retaining the
    // generator's legal cache/thread partitions and its order among ties.
    const auto best = std::min_element(
        candidates.begin(), candidates.end(), [&](const auto& a, const auto& b) {
          return a.RangesOfKC(K).NumTasks() < b.RangesOfKC(K).NumTasks();
        });
    if (best != candidates.end()) std::iter_swap(candidates.begin(), best);
  }
  return candidates;
}

// As `MatMul`, but `A` is quantized on the fly and `B` was packed by `PackB`.
// Reuses the same blocking, parallelization and autotuning as `MatMul`; only
// the kernel and operand types differ. Tuning keys distinguish A8 from BF16.
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
      M, K, N, num_B, cache.VectorBytes(), env.per_cluster[cluster_idx],
      B.block_size ? MMActivation::kI8Block : MMActivation::kI8);

  // Outside the timed section, as `MMDecompress::MaybeDecompressA`.
  MMI8AView residual;
  const MMI8AView A_view =
      QuantizeA(A, a_storage, env.ctx, cluster_idx, B.a_pre_scale, B.block_size,
                MMI8UseDualA(B, M) ? &residual : nullptr);

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
        MMI8Candidates(env, M, K, N, num_B, sizeof(TC),
                       MMI8PreferFullHeadK(B, M, hwy::IsSame<TC, float>())),
        env.autotune);
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
                                          const MMI8B& B1, const MMI8B& B2,
                                          MatMulEnv& env, MatPtrT<BF16>& C,
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
      M, K, N, num_B, cache.VectorBytes(), env.per_cluster[cluster_idx],
      B1.block_size ? MMActivation::kI8Block : MMActivation::kI8);

  HWY_DASSERT(B1.a_pre_scale == nullptr && B2.a_pre_scale == nullptr);
  HWY_ASSERT(B1.block_size == B2.block_size);
  MMI8AView residual;
  const bool dual = MMI8UseDualA(B1, M) || MMI8UseDualA(B2, M);
  const MMI8AView A_view =
      QuantizeA(A, a_storage, env.ctx, cluster_idx, nullptr, B1.block_size,
                dual ? &residual : nullptr);

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
    tuner.SetCandidates(
        MMI8Candidates(env, max_M, K, N, num_B, sizeof(BF16)),
        env.autotune);
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
