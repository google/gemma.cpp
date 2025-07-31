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

#include "compression/types.h"
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include "compression/compress.h"

#include <stddef.h>
#include <stdio.h>

#include "compression/distortion.h"
#include "util/test_util.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/tests/hwy_gtest.h"
// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "compression/compress_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Calls Compress and Decompress2 and verifies the distortion/error.
template <typename Packed>
struct TestDecompress2T {
  template <typename T, class D>
  HWY_INLINE void operator()(T /*unused*/, D d) {
    const size_t N = hn::Lanes(d);
    CompressWorkingSet work;
    hwy::ThreadPool pool(0);
    hwy::RandomState rng;

    const size_t num = 2 * N;
    const size_t packed_num = CompressedArrayElements<Packed>(num);
    auto raw = hwy::AllocateAligned<float>(num);  // Compress requires f32
    auto packed = hwy::AllocateAligned<Packed>(packed_num);
    auto dec = hwy::AllocateAligned<T>(num);
    HWY_ASSERT(raw && packed && dec);
    const auto packed_span = MakeSpan(packed.get(), packed_num);

    hwy::Stats in_stats;
    for (size_t i = 0; i < num; ++i) {
      raw[i] = static_cast<float>(RandomGaussian(rng));
      in_stats.Notify(raw[i]);
    }
    // Short inputs fail VerifyGaussian.

    const size_t packed_ofs = 0;
    Compress(raw.get(), num, work, packed_span, packed_ofs, pool);
    hn::Vec<D> raw0, raw1;
    Decompress2(d, MakeConst(packed_span), packed_ofs, raw0, raw1);
    hn::Store(raw0, d, dec.get());
    hn::Store(raw1, d, dec.get() + N);

    DistortionStats stats;
    for (size_t i = 0; i < num; ++i) {
      stats.Notify(raw[i], hwy::ConvertScalarTo<float>(dec[i]));
    }

    if constexpr (true) {  // leave enabled due to sporadic failures
      fprintf(stderr,
              "TypeName<Packed>() %s TypeName<T>() %s: num %zu: stats.SumL1() "
              "%f stats.GeomeanValueDivL1() %f stats.WeightedAverageL1() %f "
              "stats.L1().Max() %f\n",
              TypeName<Packed>(), TypeName<T>(), num, stats.SumL1(),
              stats.GeomeanValueDivL1(), stats.WeightedAverageL1(),
              stats.L1().Max());
    }

    constexpr bool kFromFloat = hwy::IsSame<Packed, float>();
    constexpr bool kToFloat = hwy::IsSame<T, float>();
    if constexpr (kFromFloat && kToFloat) {  // Lossless
      HWY_ASSERT(stats.NumExact() == num);
      HWY_ASSERT(stats.SumL1() == 0.0f);
      HWY_ASSERT(stats.L1().Max() == 0.0f);
    } else if constexpr (hwy::IsSame<Packed, BF16>() ||
                         (kFromFloat && hwy::IsSame<T, BF16>())) {
      // Small roundoff error. BF16 to float is not lossless because the
      // comparison is with float `raw`, prior to the Compress to BF16.
      HWY_ASSERT(stats.L1().Max() <= 2E-3f);
      HWY_ASSERT(IsInside(3E-4, 2E-3, stats.WeightedAverageL1()));
      HWY_ASSERT(IsInside(600.0, 900.0, stats.GeomeanValueDivL1()));
    } else if constexpr (hwy::IsSame<Packed, SfpStream>()) {
      HWY_ASSERT(stats.SumL1() <= 0.4f);
      HWY_ASSERT(stats.L1().Max() <= 0.04f);
      HWY_ASSERT(IsInside(0.01, 0.03, stats.WeightedAverageL1()));
      HWY_ASSERT(IsInside(48.0, 72.0, stats.GeomeanValueDivL1()));
    } else if constexpr (hwy::IsSame<Packed, NuqStream>()) {
      static_assert(NuqStream::kGroupSize == 256, "Update expected");
      HWY_ASSERT(stats.SumL1() <= 1.2f);
      HWY_ASSERT(stats.L1().Max() <= 0.08f);
      HWY_ASSERT(IsInside(0.02, 0.05, stats.WeightedAverageL1()));
      HWY_ASSERT(IsInside(18.0, 62.0, stats.GeomeanValueDivL1()));
    } else {
      HWY_ABORT("Unhandled type requested by ForeachPackedAndRawType");
    }
  }
};

void TestAllDecompress2() { ForeachPackedAndRawType<TestDecompress2T>(); }

// Calls Compress and DecompressAndZeroPad for all short lengths and verifies
// the distortion/error.
template <typename Packed>
struct TestShortLengthsT {
  template <typename T, class D>
  HWY_INLINE void operator()(T /*unused*/, D d) {
    const size_t N = hn::Lanes(d);
    CompressWorkingSet work;
    hwy::ThreadPool pool(0);
    hwy::RandomState rng;

    for (size_t num = 1; num < 5 * hn::Lanes(d); ++num) {
      const size_t packed_num = CompressedArrayElements<Packed>(num);

      auto raw = hwy::AllocateAligned<float>(num);  // Compress requires f32
      auto packed = hwy::AllocateAligned<Packed>(packed_num);
      auto dec = hwy::AllocateAligned<T>(hwy::RoundUpTo(num, N));
      HWY_ASSERT(raw && packed && dec);
      const auto packed_span = MakeSpan(packed.get(), packed_num);

      hwy::Stats in_stats;
      for (size_t i = 0; i < num; ++i) {
        raw[i] = static_cast<float>(RandomGaussian(rng));
        in_stats.Notify(raw[i]);
      }
      // Short inputs fail VerifyGaussian.

      const size_t packed_ofs = 0;
      Compress(raw.get(), num, work, packed_span, packed_ofs, pool);
      DecompressAndZeroPad(d, MakeConst(packed_span), packed_ofs, dec.get(),
                           num);

      DistortionStats stats;
      for (size_t i = 0; i < num; ++i) {
        stats.Notify(raw[i], hwy::ConvertScalarTo<float>(dec[i]));
      }

      if constexpr (true) {
        fprintf(stderr, "%s %s: %zu: %f %f %f %f\n", TypeName<Packed>(),
                TypeName<T>(), num, stats.SumL1(), stats.GeomeanValueDivL1(),
                stats.WeightedAverageL1(), stats.L1().Max());
      }

      constexpr bool kFromFloat = hwy::IsSame<Packed, float>();
      constexpr bool kToFloat = hwy::IsSame<T, float>();
      if constexpr (kFromFloat && kToFloat) {  // Lossless
        HWY_ASSERT(stats.NumExact() == num);
        HWY_ASSERT(stats.SumL1() == 0.0f);
        HWY_ASSERT(stats.L1().Max() == 0.0f);
      } else if (hwy::IsSame<Packed, BF16>() ||
                 (kFromFloat && hwy::IsSame<T, BF16>())) {
        // Small roundoff error. BF16 to float is not lossless because the
        // comparison is with float `raw`, prior to the Compress to BF16.
        HWY_ASSERT(stats.L1().Max() <= 4E-3f);
        HWY_ASSERT(IsInside(1E-5, 3E-3, stats.WeightedAverageL1()));
        HWY_ASSERT(IsInside(300.0, 2200.0, stats.GeomeanValueDivL1()));
      } else if (hwy::IsSame<Packed, SfpStream>()) {
        HWY_ASSERT(stats.SumL1() <= 1.3f);
        HWY_ASSERT(stats.L1().Max() <= 0.08f);
        HWY_ASSERT(IsInside(7E-5, 0.05, stats.WeightedAverageL1()));
        HWY_ASSERT(IsInside(28.0, 200.0, stats.GeomeanValueDivL1()));
      } else if (hwy::IsSame<Packed, NuqStream>()) {
        static_assert(NuqStream::kGroupSize == 256, "Update expected");
        HWY_ASSERT(stats.SumL1() <= 4.6f);
        HWY_ASSERT(stats.L1().Max() <= 0.14f);
        HWY_ASSERT(IsInside(7E-5, 0.06, stats.WeightedAverageL1()));
        HWY_ASSERT(IsInside(11.0, 180.0, stats.GeomeanValueDivL1()));
      } else {
        HWY_ABORT("Unhandled type requested by ForeachPackedAndRawType");
      }
    }
  }
};

void TestAllShortLengths() { ForeachPackedAndRawType<TestShortLengthsT>(); }

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {
HWY_BEFORE_TEST(CompressTest);
HWY_EXPORT_AND_TEST_P(CompressTest, TestAllDecompress2);
HWY_EXPORT_AND_TEST_P(CompressTest, TestAllShortLengths);
HWY_AFTER_TEST();
}  // namespace gcpp
#endif  // HWY_ONCE
