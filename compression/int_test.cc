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

// SFP uses ConcatEven/Odd which are not supported; skip SVE for faster tests.
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_SCALAR | HWY_SVE)
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "util/test_util.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "compression/int_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/int-inl.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

static constexpr size_t kGroupSize = I8Stream::kGroupSize;
static constexpr float kTolerance = 50000.0f;

// Can encode and decode sub-regions.
// Quantizes and de-quantizes a single (potentially partial) group to check
// that the quantizer is working correctly.
struct TestQuantize {
  template <typename T, class D>
  HWY_INLINE void operator()(T /*unused*/, D d) {
    const size_t total = kGroupSize / 2;  // already padded
    const hn::ScalableTag<float> df;

    auto in = hwy::AllocateAligned<float>(total);  // Enc() requires f32
    auto dec1 = hwy::AllocateAligned<T>(total);
    auto dec2 = hwy::AllocateAligned<T>(total);
    auto dec3 = hwy::AllocateAligned<T>(total);
    auto i8_stream = hwy::AllocateAligned<I8Stream>(I8Stream::PackedEnd(total));
    HWY_ASSERT(in && dec1 && dec2 && dec3 && i8_stream);
    const auto int_span = MakeSpan(i8_stream.get(), total);

    hwy::RandomState rng;
    for (size_t i = 0; i < total; ++i) {
      in[i] = static_cast<float>(RandomGaussian(rng));
    }

    IntCodec::QuantizeGroup(df, in.get(), total, int_span, 0);

    IntCodec::DequantizeGroup(d, MakeConst(int_span), 0, dec1.get(), total);

    const float epsilon =
        hwy::ConvertScalarTo<float>(hwy::Epsilon<hwy::bfloat16_t>());
    const float tolerance = kTolerance * epsilon;

    for (size_t i = 0; i < total; ++i) {
      const float expected_value = static_cast<float>(in[i]);
      const float actual_value = hwy::ConvertScalarTo<float>(dec1[i]);

      if (!(expected_value - tolerance <= actual_value &&
            actual_value <= expected_value + tolerance)) {
        fprintf(stderr,
                "in[%zu] = %f, dec1[%zu] = %f, tolerance = %f, epsilon = %f\n",
                i, expected_value, i, actual_value, tolerance, epsilon);
      }
    }

    // Check that ::Enc works correctly as well.
    IntCodec::Enc(df, in.get(), total, int_span, 0);

    IntCodec::DequantizeGroup(d, MakeConst(int_span), 0, dec2.get(), total);

    for (size_t i = 0; i < total; ++i) {
      const float expected_value = static_cast<float>(in[i]);
      const float actual_value = hwy::ConvertScalarTo<float>(dec2[i]);

      if (!(expected_value - tolerance <= actual_value &&
            actual_value <= expected_value + tolerance)) {
        fprintf(stderr,
                "in[%zu] = %f, dec2[%zu] = %f, tolerance = %f, epsilon = %f\n",
                i, expected_value, i, actual_value, tolerance, epsilon);
      }
    }

    // Check that ::DecompressAndZeroPad works correctly for one group as well.
    IntCodec::Enc(df, in.get(), total, int_span, 0);

    IntCodec::DecompressAndZeroPad(d, MakeConst(int_span), 0, dec3.get(),
                                   total);

    for (size_t i = 0; i < total; ++i) {
      const float expected_value = static_cast<float>(in[i]);
      const float actual_value = hwy::ConvertScalarTo<float>(dec3[i]);

      if (!(expected_value - tolerance <= actual_value &&
            actual_value <= expected_value + tolerance)) {
        fprintf(stderr,
                "in[%zu] = %f, dec3[%zu] = %f, tolerance = %f, epsilon = %f\n",
                i, expected_value, i, actual_value, tolerance, epsilon);
        HWY_ASSERT(false);
      }
    }
  }
};

void TestQuantizeBF16() { hn::ForGEVectors<128, TestQuantize>()(BF16()); }
void TestQuantizeF32() { hn::ForGEVectors<128, TestQuantize>()(float()); }

// Can encode and decode sub-regions.
// Quantizes and de-quantizes multiple (potentially partial) groups to check
// that DecompressAndZeroPad is working correctly.
struct TestMultiGroup {
  template <typename T, class D>
  HWY_INLINE void operator()(T /*unused*/, D d) {
    const hn::Repartition<float, D> df;
    const size_t total = kGroupSize * 2 + kGroupSize / 4;  // already padded

    auto in = hwy::AllocateAligned<float>(total);  // Enc() requires f32
    auto dec1 = hwy::AllocateAligned<T>(total);
    auto dec2 = hwy::AllocateAligned<T>(total);
    auto i8_stream = hwy::AllocateAligned<I8Stream>(I8Stream::PackedEnd(total));
    HWY_ASSERT(in && dec1 && i8_stream);
    const auto int_span = MakeSpan(i8_stream.get(), total);

    hwy::RandomState rng;
    for (size_t i = 0; i < total; ++i) {
      in[i] = static_cast<float>(RandomGaussian(rng));
    }

    const float epsilon =
        hwy::ConvertScalarTo<float>(hwy::Epsilon<hwy::bfloat16_t>());
    const float tolerance = kTolerance * epsilon;

    // Check that ::DecompressAndZeroPad works correctly for one group as well.
    IntCodec::Enc(df, in.get(), total, int_span, 0);

    IntCodec::DecompressAndZeroPad(d, MakeConst(int_span), 0, dec2.get(),
                                   total);

    for (size_t i = 0; i < total; ++i) {
      const float expected_value = static_cast<float>(in[i]);
      const float actual_value = hwy::ConvertScalarTo<float>(dec2[i]);

      if (!(expected_value - tolerance <= actual_value &&
            actual_value <= expected_value + tolerance)) {
        fprintf(stderr,
                "in[%zu] = %f, dec2[%zu] = %f, tolerance = %f, epsilon = %f\n",
                i, expected_value, i, actual_value, tolerance, epsilon);
        HWY_ASSERT(false);
      }
    }
  }
};

void TestMultiGroupBF16() { hn::ForGEVectors<128, TestMultiGroup>()(BF16()); }
void TestMultiGroupF32() { hn::ForGEVectors<128, TestMultiGroup>()(float()); }

// Can encode and decode sub-regions.
struct TestOffset {
  template <typename T, class D>
  HWY_INLINE void operator()(T /*unused*/, D d) {
    const hn::Repartition<float, D> df;
    const size_t total = 10 * kGroupSize;   // already padded
    const size_t kMidLen = 2 * kGroupSize;  // length of middle piece

    auto in = hwy::AllocateAligned<float>(total);  // Enc() requires f32
    auto dec1 = hwy::AllocateAligned<T>(total);
    auto dec2 = hwy::AllocateAligned<T>(kMidLen);
    auto i8_stream = hwy::AllocateAligned<I8Stream>(I8Stream::PackedEnd(total));
    HWY_ASSERT(in && dec1 && dec2 && i8_stream);
    const auto int_span = MakeSpan(i8_stream.get(), total);

    hwy::RandomState rng;
    for (size_t i = 0; i < total; ++i) {
      in[i] = static_cast<float>(RandomGaussian(rng));
    }

    // Encode + decode everything
    (void)IntCodec::Enc(df, in.get(), total, int_span, 0);
    IntCodec::DecompressAndZeroPad(d, MakeConst(int_span), 0, dec1.get(),
                                   total);

    MaybeCheckInitialized(dec1.get(), total * sizeof(T));

    // Overwrite middle with first inputs
    const size_t offset = 5 * kGroupSize;
    (void)IntCodec::Enc(df, in.get(), kMidLen, int_span, offset);

    // Decoded middle now matches previously decoded first
    IntCodec::DecompressAndZeroPad(d, MakeConst(int_span), offset, dec2.get(),
                                   kMidLen);
    MaybeCheckInitialized(dec2.get(), kMidLen * sizeof(T));

    for (size_t i = 0; i < kMidLen; ++i) {
      HWY_ASSERT(dec1[i] == dec2[i]);
    }
  }
};

void TestOffsetBF16() { hn::ForGEVectors<128, TestOffset>()(BF16()); }
void TestOffsetF32() { hn::ForGEVectors<128, TestOffset>()(float()); }

// Can encode and decode sub-regions.
struct TestUnalignedOffset {
  template <typename T, class D>
  HWY_INLINE void operator()(T /*unused*/, D d) {
    const hn::Repartition<float, D> df;
    const size_t total = 10 * kGroupSize;  // already padded

    const int num_unaligned_offsets = 4;
    const std::array<size_t, num_unaligned_offsets> unaligned_offsets = {
        4, kGroupSize + 100, 2 * kGroupSize + 100, 3 * kGroupSize + 100};
    const std::array<size_t, num_unaligned_offsets> num = {4, 16, 32, 64};

    for (int i = 0; i < num_unaligned_offsets; ++i) {
      const size_t unaligned_offset = unaligned_offsets[i];
      const size_t num_decompressed = num[i];

      auto in = hwy::AllocateAligned<float>(total);  // Enc() requires f32
      auto dec1 = hwy::AllocateAligned<T>(total);
      auto i8_stream =
          hwy::AllocateAligned<I8Stream>(I8Stream::PackedEnd(total));
      auto dec2 = hwy::AllocateAligned<T>(num_decompressed);
      HWY_ASSERT(in && dec1 && dec2 && i8_stream);
      const auto int_span = MakeSpan(i8_stream.get(), total);

      hwy::RandomState rng;
      for (size_t i = 0; i < total; ++i) {
        in[i] = static_cast<float>(RandomGaussian(rng));
      }

      // // Encode + decode everything
      (void)IntCodec::Enc(df, in.get(), total, int_span, 0);
      IntCodec::DecompressAndZeroPad(d, MakeConst(int_span), 0, dec1.get(),
                                     total);

      IntCodec::DecompressAndZeroPad(d, MakeConst(int_span), unaligned_offset,
                                     dec2.get(), num_decompressed);

      for (size_t i = 0; i < num_decompressed; ++i) {
        T expected = hwy::ConvertScalarTo<T>(dec1[unaligned_offset + i]);
        T actual = hwy::ConvertScalarTo<T>(dec2[i]);

        HWY_ASSERT_EQ(expected, actual);
      }
    }
  }
};

void TestUnalignedOffsetBF16() {
  hn::ForGEVectors<128, TestUnalignedOffset>()(BF16());
}
void TestUnalignedOffsetF32() {
  hn::ForGEVectors<128, TestUnalignedOffset>()(float());
}

// Can encode and decode sub-regions.
// Uses Dec2 to decode all elements in the packed buffer, then
// compares against DecompressAndZeroPad.
struct TestDec2 {
  template <typename T, class D>
  HWY_INLINE void operator()(T /*unused*/, D d) {
    const hn::Repartition<float, D> df;
    // incl. partial group to test partial group handling
    const size_t total = kGroupSize * 10 + kGroupSize / 2;

    auto in = hwy::AllocateAligned<float>(total);  // Enc() requires f32
    auto dec0 = hwy::AllocateAligned<T>(total);
    auto dec1 = hwy::AllocateAligned<T>(total);
    auto i8_stream = hwy::AllocateAligned<I8Stream>(I8Stream::PackedEnd(total));
    HWY_ASSERT(in && dec0 && dec1 && i8_stream);
    const auto int_span = MakeSpan(i8_stream.get(), total);

    hwy::RandomState rng;
    for (size_t i = 0; i < total; ++i) {
      in[i] = static_cast<float>(RandomGaussian(rng));
    }

    // Non-interleaved encode + decode for comparison
    (void)IntCodec::Enc(df, in.get(), total, int_span, 0);
    IntCodec::DecompressAndZeroPad(d, MakeConst(int_span), 0, dec0.get(),
                                   total);

    // Encode + decode everything
    (void)IntCodec::Enc(df, in.get(), total, int_span, 0);

    using V = hn::Vec<decltype(d)>;
    const size_t N = Lanes(d);

    for (size_t i = 0; i < total; i += 2 * N) {
      V f0, f1;
      IntCodec::Dec2(d, MakeConst(int_span), i, f0, f1);

      hn::StoreU(f0, d, dec1.get() + i + 0 * N);
      hn::StoreU(f1, d, dec1.get() + i + 1 * N);
    }

    for (size_t i = 0; i < total; ++i) {
      if (dec0[i] != dec1[i]) {
        fprintf(stderr, "dec0[%zu] = %g, dec1[%zu] = %g\n", i,
                hwy::ConvertScalarTo<float>(dec0[i]), i,
                hwy::ConvertScalarTo<float>(dec1[i]));
      }

      HWY_ASSERT(dec0[i] == dec1[i]);
    }
  }
};

void TestDec2BF16() { hn::ForGEVectors<128, TestDec2>()(BF16()); }
void TestDec2F32() { hn::ForGEVectors<128, TestDec2>()(float()); }

// Tests that DecompressAndZeroPad fully populates the output array.
// This is intended to catch uninitialized value errors.
struct TestDequantizeAndZeroPad {
  template <typename T, class D>
  HWY_INLINE void operator()(T /*unused*/, D d) {
    const hn::ScalableTag<float> df;
    constexpr size_t kSize = 4096;
    auto in = hwy::AllocateAligned<float>(kSize);
    auto actual_dec = hwy::AllocateAligned<T>(kSize);
    auto i8_stream = hwy::AllocateAligned<I8Stream>(I8Stream::PackedEnd(kSize));
    HWY_ASSERT(in && actual_dec && i8_stream);
    const auto int_span = MakeSpan(i8_stream.get(), kSize);

    // Fill with a known pattern.
    for (size_t i = 0; i < kSize; ++i) {
      in[i] = static_cast<float>(i) - 128.0f;
    }

    IntCodec::Enc(df, in.get(), kSize, int_span, 0);

    // Initialize with a sentinel value to detect if it's overwritten.
    const T sentinel = hwy::ConvertScalarTo<T>(-999.0f);
    std::fill(actual_dec.get(), actual_dec.get() + kSize, sentinel);

    IntCodec::DecompressAndZeroPad(d, MakeConst(int_span), 0, actual_dec.get(),
                                   kSize);

    MaybeCheckInitialized(actual_dec.get(), kSize * sizeof(T));

    // Check that all sentinels were overwritten.
    for (size_t i = 0; i < kSize; ++i) {
      EXPECT_NE(hwy::ConvertScalarTo<float>(actual_dec[i]),
                hwy::ConvertScalarTo<float>(sentinel))
          << " at index " << i;
    }
  }
};

void TestAllDequantizeAndZeroPad() {
  hn::ForGEVectors<128, TestDequantizeAndZeroPad>()(BF16());
  hn::ForGEVectors<128, TestDequantizeAndZeroPad>()(float());
}

// Tests that DecompressAndZeroPad works correctly for small and unaligned
// inputs. This is intended to catch uninitialized value errors in remainder
// handling.
struct TestSmallDequantize {
  template <typename T, class D>
  HWY_INLINE void operator()(T /*unused*/, D d) {
    const hn::ScalableTag<float> df;
    constexpr size_t kGroupSize = I8Stream::kGroupSize;
    constexpr size_t kMaxNum = kGroupSize * 3;
    auto in = hwy::AllocateAligned<float>(kMaxNum);
    auto actual_dec = hwy::AllocateAligned<T>(kMaxNum);
    auto i8_stream =
        hwy::AllocateAligned<I8Stream>(I8Stream::PackedEnd(kMaxNum));
    HWY_ASSERT(in && actual_dec && i8_stream);
    const auto int_span =
        MakeSpan(i8_stream.get(), I8Stream::PackedEnd(kMaxNum));

    // Fill with a known pattern.
    for (size_t i = 0; i < kMaxNum; ++i) {
      in[i] = static_cast<float>(i) - 128.0f;
    }

    IntCodec::Enc(df, in.get(), kMaxNum, int_span, 0);

    for (size_t num = 1; num < kGroupSize * 2; ++num) {
      for (size_t offset = 0; offset < kGroupSize; offset += 16) {
        const T sentinel = hwy::ConvertScalarTo<T>(-999.0f);
        std::fill(actual_dec.get(), actual_dec.get() + num, sentinel);

        IntCodec::DecompressAndZeroPad(d, MakeConst(int_span), offset,
                                       actual_dec.get(), num);

        MaybeCheckInitialized(actual_dec.get(), num);

        // Check that all sentinels were overwritten.
        for (size_t i = 0; i < num; ++i) {
          EXPECT_NE(hwy::ConvertScalarTo<float>(actual_dec[i]),
                    hwy::ConvertScalarTo<float>(sentinel))
              << " at index " << i << " for num=" << num
              << " offset=" << offset;
        }
      }
    }
  }
};

void TestAllSmallDequantize() {
  hn::ForGEVectors<128, TestSmallDequantize>()(BF16());
  hn::ForGEVectors<128, TestSmallDequantize>()(float());
}

// Tests that DecompressAndZeroPad works correctly for a specific failing input.
struct TestSpecificDequantize {
  template <typename T, class D>
  HWY_INLINE void operator()(T /*unused*/, D d) {
    const hn::ScalableTag<float> df;
    constexpr size_t kSize = 737280;
    auto in = hwy::AllocateAligned<float>(kSize);
    auto actual_dec = hwy::AllocateAligned<T>(kSize);
    auto i8_stream = hwy::AllocateAligned<I8Stream>(I8Stream::PackedEnd(kSize));
    HWY_ASSERT(in && actual_dec && i8_stream);
    const auto int_span = MakeSpan(i8_stream.get(), kSize);

    // Fill with a known pattern.
    for (size_t i = 0; i < kSize; ++i) {
      in[i] = static_cast<float>(i) - 128.0f;
    }

    IntCodec::Enc(df, in.get(), kSize, int_span, 0);

    const size_t num = 64;
    const size_t offset = 392704;
    const T sentinel = hwy::ConvertScalarTo<T>(-999.0f);
    std::fill(actual_dec.get(), actual_dec.get() + num, sentinel);

    IntCodec::DecompressAndZeroPad(d, MakeConst(int_span), offset,
                                   actual_dec.get(), num);

    // Check that all sentinels were overwritten.
    for (size_t i = 0; i < num; ++i) {
      EXPECT_NE(hwy::ConvertScalarTo<float>(actual_dec[i]),
                hwy::ConvertScalarTo<float>(sentinel))
          << " at index " << i << " for num=" << num << " offset=" << offset;
    }
  }
};

void TestAllSpecificDequantize() {
  hn::ForGEVectors<128, TestSpecificDequantize>()(BF16());
  hn::ForGEVectors<128, TestSpecificDequantize>()(float());
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {
HWY_BEFORE_TEST(IntTest);
HWY_EXPORT_AND_TEST_P(IntTest, TestOffsetF32);
HWY_EXPORT_AND_TEST_P(IntTest, TestOffsetBF16);
HWY_EXPORT_AND_TEST_P(IntTest, TestQuantizeF32);
HWY_EXPORT_AND_TEST_P(IntTest, TestQuantizeBF16);
HWY_EXPORT_AND_TEST_P(IntTest, TestDec2BF16);
HWY_EXPORT_AND_TEST_P(IntTest, TestDec2F32);
HWY_EXPORT_AND_TEST_P(IntTest, TestMultiGroupF32);
HWY_EXPORT_AND_TEST_P(IntTest, TestMultiGroupBF16);
HWY_EXPORT_AND_TEST_P(IntTest, TestUnalignedOffsetBF16);
HWY_EXPORT_AND_TEST_P(IntTest, TestUnalignedOffsetF32);
HWY_EXPORT_AND_TEST_P(IntTest, TestAllDequantizeAndZeroPad);
HWY_EXPORT_AND_TEST_P(IntTest, TestAllSmallDequantize);
HWY_EXPORT_AND_TEST_P(IntTest, TestAllSpecificDequantize);
HWY_AFTER_TEST();
}  // namespace gcpp
#endif  // HWY_ONCE
