#include "gemma/kv_transcoding.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "gemma/configs.h"
#include "util/basics.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"  // For hwy::Span

namespace gcpp {
namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::TestWithParam;
using ::testing::Values;

struct EncodingTestCase {
  gcpp::KVEncoding encoding;
  float tolerance;
};

class KVEncodingTest : public TestWithParam<EncodingTestCase> {};

int8_t Quantize(float v, float inv_scale) {
  float scaled = std::nearbyint(v * inv_scale);
  if (scaled > 127.0f) return 127;
  if (scaled < -127.0f) return -127;
  return hwy::ConvertScalarTo<int8_t>(scaled);
}

TEST_P(KVEncodingTest, EncodeDecodeRoundTrip) {
  const auto& param = GetParam();
  constexpr size_t kTileSize = 32;
  constexpr size_t qkv_dim = 256;

  DecodedTile original(qkv_dim, kTileSize);
  // Fill with dummy data within
  // a reasonable float range to avoid saturation for INT8
  const float pattern[] = {0.5f, 1.0f, 1.5f};
  for (size_t token = 0; token < kTileSize; ++token) {
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      size_t i = dim * kTileSize + token;
      original.k_elem(token, dim) = pattern[i % 3];
      original.v_elem(token, dim) = pattern[i % 3];
    }
  }

  std::optional<size_t> tile_size_bytes =
      GetTileSizeBytes(param.encoding, qkv_dim);
  HWY_ASSERT(tile_size_bytes.has_value());

  hwy::AlignedVector<char> encoded(*tile_size_bytes, 0);
  EXPECT_TRUE(EncodeTile(param.encoding, original, qkv_dim,
                         hwy::Span<char>(encoded.data(), encoded.size())));

  DecodedTile decoded(qkv_dim, kTileSize);
  EXPECT_TRUE(DecodeTile(param.encoding,
                         hwy::Span<const char>(encoded.data(), encoded.size()),
                         qkv_dim, &decoded));

  EXPECT_THAT(decoded.k, Pointwise(FloatNear(param.tolerance), original.k));
  EXPECT_THAT(decoded.v, Pointwise(FloatNear(param.tolerance), original.v));
}

TEST_P(KVEncodingTest, SizeChecks) {
  const auto& param = GetParam();
  constexpr size_t kTileSize = 32;
  constexpr size_t qkv_dim = 256;

  DecodedTile decoded(qkv_dim, kTileSize);
  std::optional<size_t> required_size_or =
      GetTileSizeBytes(param.encoding, qkv_dim);
  HWY_ASSERT(required_size_or.has_value());
  size_t required_size = *required_size_or;

  if (required_size > 0) {
    hwy::AlignedVector<char> too_small_encoded(required_size - 1, 0);
    EXPECT_FALSE(EncodeTile(
        param.encoding, decoded, qkv_dim,
        hwy::Span<char>(too_small_encoded.data(), too_small_encoded.size())));
    EXPECT_FALSE(DecodeTile(param.encoding,
                            hwy::Span<const char>(too_small_encoded.data(),
                                                  too_small_encoded.size()),
                            qkv_dim, &decoded));
  }
}

INSTANTIATE_TEST_SUITE_P(
    AllEncodings, KVEncodingTest,
    Values(EncodingTestCase{gcpp::KVEncoding::kF32, 1e-6f},
           EncodingTestCase{gcpp::KVEncoding::kF32TwoTranspositions, 1e-6f},
           EncodingTestCase{gcpp::KVEncoding::kBF16, 0.05f},
           EncodingTestCase{gcpp::KVEncoding::kBF16TwoTranspositions, 0.05f},
           EncodingTestCase{gcpp::KVEncoding::kBF16MatrixAccumulation, 0.05f},
           EncodingTestCase{gcpp::KVEncoding::kInt8, 0.1f},
           EncodingTestCase{gcpp::KVEncoding::kInt8TwoTranspositions, 0.1f},
           EncodingTestCase{gcpp::KVEncoding::kInt8VNNITwoTranspositions, 0.1f},
           EncodingTestCase{gcpp::KVEncoding::kInt8MatrixAccumulation, 0.02f},
           // Half a step is 1.5 / 7 / 2 ~= 0.107 for this pattern's absmax.
           EncodingTestCase{gcpp::KVEncoding::kInt4, 0.11f}));

class Int8EncodingTest : public TestWithParam<gcpp::KVEncoding> {};

// Decoding `q * scale` must be within half a quantization step of the input,
// as in tiled_attention.cc, which quantizes with the stored bf16 scale.
TEST_P(Int8EncodingTest, RoundTripIsWithinHalfAStep) {
  const gcpp::KVEncoding encoding = GetParam();
  constexpr size_t kTileSize = 32;
  constexpr size_t qkv_dim = 256;

  DecodedTile original(qkv_dim, kTileSize);
  for (size_t token = 0; token < kTileSize; ++token) {
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      // Per-token magnitudes vary so that absmax / 127 is rarely exact in bf16.
      const float i = static_cast<float>(dim * kTileSize + token);
      original.k_elem(token, dim) = std::sin(i * 0.37f) * (1.0f + token);
      original.v_elem(token, dim) = std::cos(i * 0.53f) * (3.0f + 0.7f * token);
    }
  }

  hwy::AlignedVector<char> encoded(GetTileSizeBytes(encoding, qkv_dim).value());
  ASSERT_TRUE(EncodeTile(encoding, original, qkv_dim,
                         hwy::Span<char>(encoded.data(), encoded.size())));
  DecodedTile decoded(qkv_dim, kTileSize);
  ASSERT_TRUE(DecodeTile(encoding,
                         hwy::Span<const char>(encoded.data(), encoded.size()),
                         qkv_dim, &decoded));

  for (size_t token = 0; token < kTileSize; ++token) {
    float k_max = 0.0f, v_max = 0.0f;
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      k_max = std::max(k_max, std::abs(original.k_elem(token, dim)));
      v_max = std::max(v_max, std::abs(original.v_elem(token, dim)));
    }
    // Half a step, plus slack for rounding the scale to bf16.
    const float k_bound = 0.51f * k_max / 127.0f;
    const float v_bound = 0.51f * v_max / 127.0f;
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      ASSERT_LE(
          std::abs(decoded.k_elem(token, dim) - original.k_elem(token, dim)),
          k_bound)
          << "k token=" << token << " dim=" << dim;
      ASSERT_LE(
          std::abs(decoded.v_elem(token, dim) - original.v_elem(token, dim)),
          v_bound)
          << "v token=" << token << " dim=" << dim;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(AllInt8Encodings, Int8EncodingTest,
                         Values(gcpp::KVEncoding::kInt8,
                                gcpp::KVEncoding::kInt8TwoTranspositions,
                                gcpp::KVEncoding::kInt8VNNITwoTranspositions,
                                gcpp::KVEncoding::kInt8MatrixAccumulation));

// Fills a tile with per-token magnitudes that vary, so that absmax / qmax is
// rarely exact in bf16.
void FillVaryingTile(DecodedTile& tile) {
  for (size_t token = 0; token < tile.tile_size; ++token) {
    for (size_t dim = 0; dim < tile.qkv_dim; ++dim) {
      const float i = static_cast<float>(dim * tile.tile_size + token);
      tile.k_elem(token, dim) = std::sin(i * 0.37f) * (1.0f + token);
      tile.v_elem(token, dim) = std::cos(i * 0.53f) * (3.0f + 0.7f * token);
    }
  }
}

// Reference per-token symmetric 4-bit quant-dequant of one row: bf16 scale
// absmax / 7, quantized with the stored scale, values in [-7, 7].
void RefQuantDequantInt4Row(float* row, size_t d) {
  float max_abs = 0.0f;
  for (size_t j = 0; j < d; ++j) max_abs = std::max(max_abs, std::abs(row[j]));
  const float scale = hwy::ConvertScalarTo<float>(
      hwy::ConvertScalarTo<hwy::bfloat16_t>(max_abs == 0.0f ? 1.0f
                                                            : max_abs / 7.0f));
  const float inv = scale == 0.0f ? 0.0f : 1.0f / scale;
  for (size_t j = 0; j < d; ++j) {
    const float q = std::clamp(std::nearbyint(row[j] * inv), -7.0f, 7.0f);
    row[j] = q * scale;
  }
}

TEST(Int4EncodingTest, RoundTripIsWithinHalfAStep) {
  constexpr size_t kTileSize = 32;
  constexpr size_t qkv_dim = 256;
  const KVEncoding encoding = KVEncoding::kInt4;

  DecodedTile original(qkv_dim, kTileSize);
  FillVaryingTile(original);

  hwy::AlignedVector<char> encoded(GetTileSizeBytes(encoding, qkv_dim).value());
  ASSERT_TRUE(EncodeTile(encoding, original, qkv_dim,
                         hwy::Span<char>(encoded.data(), encoded.size())));
  DecodedTile decoded(qkv_dim, kTileSize);
  ASSERT_TRUE(DecodeTile(encoding,
                         hwy::Span<const char>(encoded.data(), encoded.size()),
                         qkv_dim, &decoded));

  for (size_t token = 0; token < kTileSize; ++token) {
    float k_max = 0.0f, v_max = 0.0f;
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      k_max = std::max(k_max, std::abs(original.k_elem(token, dim)));
      v_max = std::max(v_max, std::abs(original.v_elem(token, dim)));
    }
    // Half a step, plus slack for rounding the scale to bf16.
    const float k_bound = 0.51f * k_max / 7.0f;
    const float v_bound = 0.51f * v_max / 7.0f;
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      ASSERT_LE(
          std::abs(decoded.k_elem(token, dim) - original.k_elem(token, dim)),
          k_bound)
          << "k token=" << token << " dim=" << dim;
      ASSERT_LE(
          std::abs(decoded.v_elem(token, dim) - original.v_elem(token, dim)),
          v_bound)
          << "v token=" << token << " dim=" << dim;
    }
  }
}

// The packed encoding must reproduce the simulated (quant-dequant) INT4 that
// the accuracy evaluation used, bit for bit.
TEST(Int4EncodingTest, MatchesReferenceQuantDequantExactly) {
  constexpr size_t kTileSize = 32;
  constexpr size_t qkv_dim = 512;
  const KVEncoding encoding = KVEncoding::kInt4;

  DecodedTile original(qkv_dim, kTileSize);
  FillVaryingTile(original);
  DecodedTile expected = original;
  for (size_t token = 0; token < kTileSize; ++token) {
    RefQuantDequantInt4Row(&expected.k_elem(token, 0), qkv_dim);
    RefQuantDequantInt4Row(&expected.v_elem(token, 0), qkv_dim);
  }

  hwy::AlignedVector<char> encoded(GetTileSizeBytes(encoding, qkv_dim).value());
  ASSERT_TRUE(EncodeTile(encoding, original, qkv_dim,
                         hwy::Span<char>(encoded.data(), encoded.size())));
  DecodedTile decoded(qkv_dim, kTileSize);
  ASSERT_TRUE(DecodeTile(encoding,
                         hwy::Span<const char>(encoded.data(), encoded.size()),
                         qkv_dim, &decoded));

  EXPECT_THAT(decoded.k, Pointwise(::testing::FloatEq(), expected.k));
  EXPECT_THAT(decoded.v, Pointwise(::testing::FloatEq(), expected.v));
}

TEST(Int4EncodingTest, TileIsHalfOfInt8PlusScales) {
  for (size_t qkv_dim : {256, 512}) {
    const size_t scales = 32 * 2 * sizeof(hwy::bfloat16_t);
    const size_t int8 = GetTileSizeBytes(KVEncoding::kInt8, qkv_dim).value();
    const size_t int4 = GetTileSizeBytes(KVEncoding::kInt4, qkv_dim).value();
    EXPECT_EQ(int4, (int8 - scales) / 2 + scales) << "qkv_dim=" << qkv_dim;
  }
  EXPECT_EQ(KVEncodingToString(KVEncoding::kInt4), "Int4");
}

TEST(Int4EncodingTest, LayoutValidation) {
  constexpr size_t kTileSize = 32;
  constexpr size_t qkv_dim = 4;
  const KVEncoding encoding = KVEncoding::kInt4;

  // Token 0 has K = {7, -7, 0, 3.5} and V = {-14, 14, 2, 0}; other tokens are
  // zero. Scales: K 7/7 = 1, V 14/7 = 2.
  DecodedTile original(qkv_dim, kTileSize);
  for (size_t token = 0; token < kTileSize; ++token) {
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      original.k_elem(token, dim) = 0.0f;
      original.v_elem(token, dim) = 0.0f;
    }
  }
  const float k0[] = {7.0f, -7.0f, 0.0f, 3.25f};
  const float v0[] = {-14.0f, 14.0f, 2.0f, 0.0f};
  for (size_t dim = 0; dim < qkv_dim; ++dim) {
    original.k_elem(0, dim) = k0[dim];
    original.v_elem(0, dim) = v0[dim];
  }
  // Token 1, dim 0: K = 5 (alone in its row, so q = 7, scale 5/7).
  original.k_elem(1, 0) = 5.0f;

  hwy::AlignedVector<char> encoded(GetTileSizeBytes(encoding, qkv_dim).value());
  ASSERT_TRUE(EncodeTile(encoding, original, qkv_dim,
                         hwy::Span<char>(encoded.data(), encoded.size())));
  const uint8_t* data = HWY_RCAST_ALIGNED(const uint8_t*, encoded.data());

  // K is [qkv_dim, tile_size] nibbles: byte (dim * 32 + token) / 2, low nibble
  // for even tokens. Stored value is q + 8.
  EXPECT_EQ(data[0], (8 + 7) | ((8 + 7) << 4));   // dim 0: t0 q=7, t1 q=7
  EXPECT_EQ(data[16], (8 - 7) | (8 << 4));        // dim 1: t0 q=-7, t1 q=0
  EXPECT_EQ(data[32], 8 | (8 << 4));              // dim 2: t0 q=0
  EXPECT_EQ(data[48], (8 + 3) | (8 << 4));        // dim 3: t0 3.25 -> q=3
  // V is [tile_size, qkv_dim] nibbles starting at qkv_dim * 32 / 2 = 64; low
  // nibble for even dims.
  EXPECT_EQ(data[64], (8 - 7) | ((8 + 7) << 4));  // t0: d0 q=-7, d1 q=7
  EXPECT_EQ(data[65], (8 + 1) | (8 << 4));        // t0: d2 q=1, d3 q=0
  // Scales follow at 128: K scales then V scales, bf16.
  const hwy::bfloat16_t* scales =
      HWY_RCAST_ALIGNED(const hwy::bfloat16_t*, data + 128);
  EXPECT_EQ(hwy::ConvertScalarTo<float>(scales[0]), 1.0f);
  EXPECT_EQ(hwy::ConvertScalarTo<float>(scales[32]), 2.0f);
}

TEST(Int4EncodingTest, TranscodesToAndFromInt8) {
  constexpr size_t kTileSize = 32;
  constexpr size_t qkv_dim = 256;
  DecodedTile original(qkv_dim, kTileSize);
  FillVaryingTile(original);

  hwy::AlignedVector<char> int4(
      GetTileSizeBytes(KVEncoding::kInt4, qkv_dim).value());
  hwy::AlignedVector<char> int8(
      GetTileSizeBytes(KVEncoding::kInt8, qkv_dim).value());
  hwy::AlignedVector<char> int4_again(int4.size());
  ASSERT_TRUE(EncodeTile(KVEncoding::kInt4, original, qkv_dim,
                         hwy::Span<char>(int4.data(), int4.size())));
  ASSERT_TRUE(TranscodeTile(KVEncoding::kInt4,
                            hwy::Span<const char>(int4.data(), int4.size()),
                            KVEncoding::kInt8,
                            hwy::Span<char>(int8.data(), int8.size()),
                            qkv_dim));
  ASSERT_TRUE(TranscodeTile(
      KVEncoding::kInt8, hwy::Span<const char>(int8.data(), int8.size()),
      KVEncoding::kInt4, hwy::Span<char>(int4_again.data(), int4_again.size()),
      qkv_dim));

  // The INT8 hop moves each value by at most half an INT8 step (~absmax / 254),
  // far below half an INT4 step (~absmax / 14), so re-encoding to INT4 must
  // give the same values up to rounding of the scale.
  DecodedTile a(qkv_dim, kTileSize), b(qkv_dim, kTileSize);
  ASSERT_TRUE(DecodeTile(KVEncoding::kInt4,
                         hwy::Span<const char>(int4.data(), int4.size()),
                         qkv_dim, &a));
  ASSERT_TRUE(DecodeTile(
      KVEncoding::kInt4,
      hwy::Span<const char>(int4_again.data(), int4_again.size()), qkv_dim,
      &b));
  for (size_t token = 0; token < kTileSize; ++token) {
    float k_max = 0.0f;
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      k_max = std::max(k_max, std::abs(a.k_elem(token, dim)));
    }
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      ASSERT_NEAR(a.k_elem(token, dim), b.k_elem(token, dim), 0.02f * k_max)
          << "k token=" << token << " dim=" << dim;
    }
    float v_max = 0.0f;
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      v_max = std::max(v_max, std::abs(a.v_elem(token, dim)));
    }
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      ASSERT_NEAR(a.v_elem(token, dim), b.v_elem(token, dim), 0.02f * v_max)
          << "v token=" << token << " dim=" << dim;
    }
  }
}

TEST(KVEncodingTest, ConvertTileFloat32ToBfloat16) {
  constexpr size_t kTileSize = 32;
  constexpr size_t qkv_dim = 256;
  gcpp::KVEncoding src_encoding = gcpp::KVEncoding::kF32;
  gcpp::KVEncoding dst_encoding = gcpp::KVEncoding::kBF16;

  DecodedTile original(qkv_dim, kTileSize);
  for (size_t token = 0; token < kTileSize; ++token) {
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      size_t i = dim * kTileSize + token;
      original.k_elem(token, dim) = std::sin(i) * 5.0f;
      original.v_elem(token, dim) = std::cos(i) * 5.0f;
    }
  }

  size_t src_size = GetTileSizeBytes(src_encoding, qkv_dim).value();
  size_t dst_size = GetTileSizeBytes(dst_encoding, qkv_dim).value();

  hwy::AlignedVector<char> src_data(src_size);
  hwy::AlignedVector<char> dst_data(dst_size);

  EXPECT_TRUE(EncodeTile(src_encoding, original, qkv_dim,
                         hwy::Span<char>(src_data.data(), src_data.size())));

  EXPECT_TRUE(TranscodeTile(
      src_encoding, hwy::Span<const char>(src_data.data(), src_data.size()),
      dst_encoding, hwy::Span<char>(dst_data.data(), dst_data.size()),
      qkv_dim));

  DecodedTile decoded(qkv_dim, kTileSize);
  EXPECT_TRUE(DecodeTile(
      dst_encoding, hwy::Span<const char>(dst_data.data(), dst_data.size()),
      qkv_dim, &decoded));

  EXPECT_THAT(decoded.k, Pointwise(FloatNear(0.05f), original.k));
}

TEST(KVEncodingTest, PairwiseConversion) {
  constexpr size_t kTileSize = 32;
  constexpr size_t qkv_dim = 256;

  std::vector<gcpp::KVEncoding> encodings = {
      gcpp::KVEncoding::kF32,
      gcpp::KVEncoding::kF32TwoTranspositions,
      gcpp::KVEncoding::kBF16,
      gcpp::KVEncoding::kBF16TwoTranspositions,
      gcpp::KVEncoding::kInt8,
      gcpp::KVEncoding::kInt8TwoTranspositions,
      gcpp::KVEncoding::kInt8VNNITwoTranspositions};

  for (auto src : encodings) {
    for (auto dst : encodings) {
      if (src == dst) continue;

      DecodedTile original(qkv_dim, kTileSize);
      const float pattern[] = {0.5f, 1.0f, 1.5f};
      for (size_t token = 0; token < kTileSize; ++token) {
        for (size_t dim = 0; dim < qkv_dim; ++dim) {
          size_t i = dim * kTileSize + token;
          original.k_elem(token, dim) = pattern[i % 3];
          original.v_elem(token, dim) = pattern[i % 3];
        }
      }

      size_t src_size = GetTileSizeBytes(src, qkv_dim).value();
      size_t dst_size = GetTileSizeBytes(dst, qkv_dim).value();

      hwy::AlignedVector<char> src_data(src_size);
      hwy::AlignedVector<char> dst_data(dst_size);

      ASSERT_TRUE(EncodeTile(src, original, qkv_dim,
                             hwy::Span<char>(src_data.data(), src_data.size())))
          << "src=" << static_cast<int>(src);

      ASSERT_TRUE(TranscodeTile(
          src, hwy::Span<const char>(src_data.data(), src_data.size()), dst,
          hwy::Span<char>(dst_data.data(), dst_data.size()), qkv_dim))
          << "src=" << static_cast<int>(src)
          << " dst=" << static_cast<int>(dst);

      DecodedTile decoded(qkv_dim, kTileSize);
      ASSERT_TRUE(DecodeTile(
          dst, hwy::Span<const char>(dst_data.data(), dst_data.size()), qkv_dim,
          &decoded))
          << "dst=" << static_cast<int>(dst);

      float tolerance = 0.1f;  // Max tolerance for Int8
      EXPECT_THAT(decoded.k, Pointwise(FloatNear(tolerance), original.k))
          << "src=" << static_cast<int>(src)
          << " dst=" << static_cast<int>(dst);
      EXPECT_THAT(decoded.v, Pointwise(FloatNear(tolerance), original.v))
          << "src=" << static_cast<int>(src)
          << " dst=" << static_cast<int>(dst);
    }
  }
}

TEST(KVEncodingTest, LayoutValidationF32) {
  constexpr size_t kTileSize = 32;
  constexpr size_t qkv_dim = 4;
  gcpp::KVEncoding encoding = gcpp::KVEncoding::kF32;

  DecodedTile original(qkv_dim, kTileSize);
  for (size_t token = 0; token < kTileSize; ++token) {
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      original.k_elem(token, dim) = dim * kTileSize + token + 1;
    }
  }
  for (size_t token = 0; token < kTileSize; ++token) {
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      original.v_elem(token, dim) =
          token * qkv_dim + dim + 1 + qkv_dim * kTileSize;
    }
  }

  size_t size = GetTileSizeBytes(encoding, qkv_dim).value();
  hwy::AlignedVector<char> encoded(size);

  ASSERT_TRUE(EncodeTile(encoding, original, qkv_dim,
                         hwy::Span<char>(encoded.data(), encoded.size())));

  const float* data = HWY_RCAST_ALIGNED(const float*, encoded.data());

  // K should be row-major [qkv_dim, tile_size]
  EXPECT_EQ(data[0], 1.0f);    // d=0, t=0
  EXPECT_EQ(data[1], 2.0f);    // d=0, t=1
  EXPECT_EQ(data[32], 33.0f);  // d=1, t=0

  // V should be row-major [tile_size, qkv_dim]
  size_t v_start = qkv_dim * kTileSize;
  EXPECT_EQ(data[v_start], 129.0f);      // t=0, d=0
  EXPECT_EQ(data[v_start + 1], 130.0f);  // t=0, d=1
  EXPECT_EQ(data[v_start + 4], 133.0f);  // t=1, d=0
}

TEST(KVEncodingTest, LayoutValidationF32TwoTranspositions) {
  constexpr size_t kTileSize = 32;
  constexpr size_t qkv_dim = 4;
  gcpp::KVEncoding encoding = gcpp::KVEncoding::kF32TwoTranspositions;

  DecodedTile original(qkv_dim, kTileSize);
  for (size_t token = 0; token < kTileSize; ++token) {
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      original.k_elem(token, dim) = dim * kTileSize + token + 1;
    }
  }
  for (size_t token = 0; token < kTileSize; ++token) {
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      original.v_elem(token, dim) =
          token * qkv_dim + dim + 1 + qkv_dim * kTileSize;
    }
  }

  size_t size = GetTileSizeBytes(encoding, qkv_dim).value();
  hwy::AlignedVector<char> encoded(size);

  ASSERT_TRUE(EncodeTile(encoding, original, qkv_dim,
                         hwy::Span<char>(encoded.data(), encoded.size())));

  const float* data = HWY_RCAST_ALIGNED(const float*, encoded.data());

  // K transposed: [qkv_dim/2, tile_size, 2]
  EXPECT_EQ(data[0], 1.0f);    // d=0, t=0
  EXPECT_EQ(data[1], 33.0f);   // d=1, t=0
  EXPECT_EQ(data[2], 2.0f);    // d=0, t=1
  EXPECT_EQ(data[3], 34.0f);   // d=1, t=1
  EXPECT_EQ(data[64], 65.0f);  // d=2, t=0
  EXPECT_EQ(data[65], 97.0f);  // d=3, t=0

  // V transposed: [tile_size/2, qkv_dim, 2]
  size_t v_start = qkv_dim * kTileSize;
  EXPECT_EQ(data[v_start], 129.0f);      // t=0, d=0
  EXPECT_EQ(data[v_start + 1], 133.0f);  // t=1, d=0
  EXPECT_EQ(data[v_start + 2], 130.0f);  // t=0, d=1
  EXPECT_EQ(data[v_start + 3], 134.0f);  // t=1, d=1
}

TEST(KVEncodingTest, LayoutValidationInt8) {
  constexpr size_t kTileSize = 32;
  constexpr size_t qkv_dim = 4;
  gcpp::KVEncoding encoding = gcpp::KVEncoding::kInt8;

  DecodedTile original(qkv_dim, kTileSize);
  for (size_t token = 0; token < kTileSize; ++token) {
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      original.k_elem(token, dim) = dim * kTileSize + token + 1;
    }
  }
  for (size_t token = 0; token < kTileSize; ++token) {
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      original.v_elem(token, dim) =
          token * qkv_dim + dim + 1 + qkv_dim * kTileSize;
    }
  }

  size_t size = GetTileSizeBytes(encoding, qkv_dim).value();
  hwy::AlignedVector<char> encoded(size);

  ASSERT_TRUE(EncodeTile(encoding, original, qkv_dim,
                         hwy::Span<char>(encoded.data(), encoded.size())));

  const int8_t* data = HWY_RCAST_ALIGNED(const int8_t*, encoded.data());

  // K should be row-major [qkv_dim, tile_size]
  // K[3,0] = 97. Max for t=0 is 97. Scale = 97/127.
  // Quantized K[3,0] = 127.
  // K[3,0] is at offset 3 * 32 + 0 = 96.
  EXPECT_EQ(data[96], 127);

  // V should be row-major [tile_size, qkv_dim]
  size_t v_start = qkv_dim * kTileSize;
  // V[0,3] = 132. Max for t=0 is 132. Scale = 132/127.
  // Quantized V[0,3] = 127.
  // V[0,3] is at offset v_start + 0 * 4 + 3 = v_start + 3.
  EXPECT_EQ(data[v_start + 3], 127);
}

TEST(KVEncodingTest, LayoutValidationInt8TwoTranspositions) {
  constexpr size_t kTileSize = 32;
  constexpr size_t qkv_dim = 4;
  gcpp::KVEncoding encoding = gcpp::KVEncoding::kInt8TwoTranspositions;

  DecodedTile original(qkv_dim, kTileSize);
  for (size_t token = 0; token < kTileSize; ++token) {
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      original.k_elem(token, dim) = dim * kTileSize + token + 1;
    }
  }
  for (size_t token = 0; token < kTileSize; ++token) {
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      original.v_elem(token, dim) =
          token * qkv_dim + dim + 1 + qkv_dim * kTileSize;
    }
  }

  size_t size = GetTileSizeBytes(encoding, qkv_dim).value();
  hwy::AlignedVector<char> encoded(size);

  ASSERT_TRUE(EncodeTile(encoding, original, qkv_dim,
                         hwy::Span<char>(encoded.data(), encoded.size())));

  const int8_t* data = HWY_RCAST_ALIGNED(const int8_t*, encoded.data());

  // K transposed: [qkv_dim/2, tile_size, 2]
  // K[0,0] = 1. Max for t=0 is 97. Scale = 97/127.
  // Quantized K[0,0] = 1.
  // K[1,0] = 33. Quantized K[1,0] = 33 / (97/127) = 43.14 -> 43.
  // K[1,0] is at offset 1.
  EXPECT_EQ(data[0], 1);
  EXPECT_EQ(data[1], 43);

  // V transposed: [tile_size/2, qkv_dim, 2]
  size_t v_start = qkv_dim * kTileSize;
  // V[0,0] = 129. Max for t=0 is 132. Scale = 132/127.
  // Quantized V[0,0] = round(129 * 127 / 132) = 124.
  // V[1,0] = 133. Max for t=1 is 136. Scale = 136/127.
  // Quantized V[1,0] = round(133 * 127 / 136) = 124.
  // In transposed layout, V[0,0] is at v_start. V[1,0] is at v_start + 1.
  EXPECT_EQ(data[v_start], 124);
  EXPECT_EQ(data[v_start + 1], 124);

  // V[1,3] = 136. Max for t=1 is 136. Quantized = 127.
  // Offset in transposed V: t/2*8 + d*2 + t%2.
  // For t=1, d=3: 0*8 + 3*2 + 1 = 7.
  EXPECT_EQ(data[v_start + 7], 127);
}

TEST(KVEncodingTest, LayoutValidationBF16MatrixAccumulation) {
  constexpr size_t kTileSize = 32;
  constexpr size_t qkv_dim = 4;
  gcpp::KVEncoding encoding = gcpp::KVEncoding::kBF16MatrixAccumulation;

  DecodedTile original(qkv_dim, kTileSize);
  for (size_t token = 0; token < kTileSize; ++token) {
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      original.k_elem(token, dim) = dim * kTileSize + token + 1;
      original.v_elem(token, dim) =
          token * qkv_dim + dim + 1 + qkv_dim * kTileSize;
    }
  }

  size_t size = GetTileSizeBytes(encoding, qkv_dim).value();
  hwy::AlignedVector<char> encoded(size);

  ASSERT_TRUE(EncodeTile(encoding, original, qkv_dim,
                         hwy::Span<char>(encoded.data(), encoded.size())));

  const gcpp::BF16* data = HWY_RCAST_ALIGNED(const gcpp::BF16*, encoded.data());

  // K Layout (8x4 block, token-major)
  // base_offset = ch_g * 128 + g * 32.
  // For qkv_dim = 4, ch_g = 0 is the only channel group.
  // For g = 0 (tokens 0-7), base_offset = 0.
  // K[t, c] is at offset t * 4 + c.
  // original.k_elem(t, c) = c * 32 + t + 1.
  // For t=0, c=0: original.k_elem(0,0) = 1. Offset = 0.
  // For t=0, c=1: original.k_elem(0,1) = 33. Offset = 1.
  // For t=1, c=0: original.k_elem(1,0) = 2. Offset = 4.
  // For t=7, c=3: original.k_elem(7,3) = 3*32 + 7 + 1 = 104. Offset = 7 * 4 + 3
  // = 31.
  EXPECT_NEAR(hwy::ConvertScalarTo<float>(data[0]), 1.0f, 0.05f);
  EXPECT_NEAR(hwy::ConvertScalarTo<float>(data[1]), 33.0f, 0.05f);
  EXPECT_NEAR(hwy::ConvertScalarTo<float>(data[4]), 2.0f, 0.05f);
  EXPECT_NEAR(hwy::ConvertScalarTo<float>(data[31]), 104.0f, 0.05f);

  // For g = 1 (tokens 8-15), base_offset = 32.
  // K[t_in_g, c] is at 32 + t_in_g * 4 + c.
  // t=8 (t_in_g=0), c=0: original.k_elem(8,0) = 9. Offset = 32.
  EXPECT_NEAR(hwy::ConvertScalarTo<float>(data[32]), 9.0f, 0.05f);

  // V Layout (Contiguous SV-Blocked Layout)
  // For qkv_dim = 4, ch_g = 0.
  // V[t, c] is at v_start + sub_block * 32 + block_offset
  // where:
  //   g_t = t / 16, g_c = c / 2
  //   sub_block = g_t * 2 + g_c
  //   t' = t % 16, c' = c % 2
  //   g_t4 = t' / 4, t'' = t' % 4
  //   block_offset = g_t4 * 8 + c' * 4 + t''
  // original.v_elem(t, c) = t * 4 + c + 1 + 128.
  // v_start = 4 * 32 = 128 elements of BF16.
  size_t v_start = qkv_dim * kTileSize;
  // For t=0, c=0: original.v_elem(0,0) = 129. Offset = v_start + 0.
  // For t=1, c=0: original.v_elem(1,0) = 133. Offset = v_start + 1.
  // For t=0, c=1: original.v_elem(0,1) = 130. Offset = v_start + 4.
  // For t=1, c=1: original.v_elem(1,1) = 134. Offset = v_start + 5.
  // For t=2, c=0: original.v_elem(2,0) = 137. Offset = v_start + 2.
  // For t=7, c=3: original.v_elem(7,3) = 7*4 + 3 + 1 + 128 = 160.
  //               Offset = v_start + 47.
  EXPECT_NEAR(hwy::ConvertScalarTo<float>(data[v_start + 0]), 129.0f, 0.05f);
  EXPECT_NEAR(hwy::ConvertScalarTo<float>(data[v_start + 1]), 133.0f, 0.05f);
  EXPECT_NEAR(hwy::ConvertScalarTo<float>(data[v_start + 4]), 130.0f, 0.05f);
  EXPECT_NEAR(hwy::ConvertScalarTo<float>(data[v_start + 5]), 134.0f, 0.05f);
  EXPECT_NEAR(hwy::ConvertScalarTo<float>(data[v_start + 2]), 137.0f, 0.05f);
  EXPECT_NEAR(hwy::ConvertScalarTo<float>(data[v_start + 47]), 160.0f, 0.05f);
}

TEST(KVEncodingTest, LayoutValidationInt8MatrixAccumulation) {
  constexpr size_t kTileSize = 32;
  constexpr size_t qkv_dim = 16;
  gcpp::KVEncoding encoding = gcpp::KVEncoding::kInt8MatrixAccumulation;

  DecodedTile original(qkv_dim, kTileSize);
  for (size_t token = 0; token < kTileSize; ++token) {
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      original.k_elem(token, dim) = (dim + 1) * (token + 1) * 0.1f;
      original.v_elem(token, dim) = (dim + 1) * (token + 1) * 0.2f;
    }
  }

  size_t size = GetTileSizeBytes(encoding, qkv_dim).value();
  hwy::AlignedVector<char> encoded(size);

  ASSERT_TRUE(EncodeTile(encoding, original, qkv_dim,
                         hwy::Span<char>(encoded.data(), encoded.size())));

  const int8_t* k_data = HWY_RCAST_ALIGNED(const int8_t*, encoded.data());
  const int8_t* v_data = k_data + qkv_dim * kTileSize;
  const gcpp::BF16* scales =
      HWY_RCAST_ALIGNED(const gcpp::BF16*, v_data + kTileSize * qkv_dim);
  const gcpp::BF16* k_scales = scales;
  const gcpp::BF16* v_scales = scales + kTileSize;

  // 1. Verify quantized values and layout offsets
  for (size_t token = 0; token < kTileSize; ++token) {
    // Compute expected scale for K (across all channels)
    float max_abs_k = 0.0f;
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      max_abs_k = std::max(max_abs_k, std::abs(original.k_elem(token, dim)));
    }
    float scale_k_raw = max_abs_k == 0.0f ? 1.0f : max_abs_k / 127.0f;
    gcpp::BF16 scale_k_bf16 = hwy::ConvertScalarTo<gcpp::BF16>(scale_k_raw);
    float scale_k = hwy::ConvertScalarTo<float>(scale_k_bf16);
    float inv_scale_k = scale_k == 0.0f ? 0.0f : 1.0f / scale_k;

    // Compute expected scale for V (across all channels)
    float max_abs_v = 0.0f;
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      max_abs_v = std::max(max_abs_v, std::abs(original.v_elem(token, dim)));
    }
    float scale_v_raw = max_abs_v == 0.0f ? 1.0f : max_abs_v / 127.0f;
    gcpp::BF16 scale_v_bf16 = hwy::ConvertScalarTo<gcpp::BF16>(scale_v_raw);
    float scale_v = hwy::ConvertScalarTo<float>(scale_v_bf16);
    float inv_scale_v = scale_v == 0.0f ? 0.0f : 1.0f / scale_v;

    // Verify scale storage (flat token-major)
    EXPECT_NEAR(hwy::ConvertScalarTo<float>(k_scales[token]), scale_k, 1e-5f)
        << "K scale mismatch at token=" << token;
    EXPECT_NEAR(hwy::ConvertScalarTo<float>(v_scales[token]), scale_v, 1e-5f)
        << "V scale mismatch at token=" << token;

    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      size_t expected_k_offset =
          MatrixAccumulationOffset_Int8(qkv_dim, dim, token);
      size_t expected_v_offset =
          VMatrixAccumulationOffset_Int8(qkv_dim, token, dim);

      int8_t expected_k = Quantize(original.k_elem(token, dim), inv_scale_k);
      int8_t expected_v = Quantize(original.v_elem(token, dim), inv_scale_v);

      EXPECT_EQ(k_data[expected_k_offset], expected_k)
          << "K quantized value mismatch at token=" << token << ", dim=" << dim
          << ", expected_k_offset=" << expected_k_offset;
      EXPECT_EQ(v_data[expected_v_offset], expected_v)
          << "V quantized value mismatch at token=" << token << ", dim=" << dim
          << ", expected_v_offset=" << expected_v_offset;
    }
  }

  // 2. Verify round-trip decoding
  DecodedTile decoded(qkv_dim, kTileSize);
  ASSERT_TRUE(DecodeTile(encoding,
                         hwy::Span<const char>(encoded.data(), encoded.size()),
                         qkv_dim, &decoded));

  for (size_t token = 0; token < kTileSize; ++token) {
    float scale_k = hwy::ConvertScalarTo<float>(k_scales[token]);
    float scale_v = hwy::ConvertScalarTo<float>(v_scales[token]);

    // Max absolute quantization error is scale * 0.5 (plus epsilon for float
    // precision)
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      EXPECT_NEAR(decoded.k_elem(token, dim), original.k_elem(token, dim),
                  scale_k * 0.501f)
          << "Decoded K mismatch at token=" << token << ", dim=" << dim;
      EXPECT_NEAR(decoded.v_elem(token, dim), original.v_elem(token, dim),
                  scale_v * 0.501f)
          << "Decoded V mismatch at token=" << token << ", dim=" << dim;
    }
  }
}

}  // namespace
}  // namespace gcpp
