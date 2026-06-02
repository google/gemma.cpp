#include "gemma/kv_transcoding.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "gemma/configs.h"
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
  ASSERT_TRUE(tile_size_bytes.has_value());

  std::vector<char> encoded(*tile_size_bytes, 0);
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
  ASSERT_TRUE(required_size_or.has_value());
  size_t required_size = *required_size_or;

  if (required_size > 0) {
    std::vector<char> too_small_encoded(required_size - 1, 0);
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
           EncodingTestCase{gcpp::KVEncoding::kInt8, 0.1f},
           EncodingTestCase{gcpp::KVEncoding::kInt8TwoTranspositions, 0.1f}));

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

  std::vector<char> src_data(src_size);
  std::vector<char> dst_data(dst_size);

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
      gcpp::KVEncoding::kF32,  gcpp::KVEncoding::kF32TwoTranspositions,
      gcpp::KVEncoding::kBF16, gcpp::KVEncoding::kBF16TwoTranspositions,
      gcpp::KVEncoding::kInt8, gcpp::KVEncoding::kInt8TwoTranspositions};

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

      std::vector<char> src_data(src_size);
      std::vector<char> dst_data(dst_size);

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
  std::vector<char> encoded(size);

  ASSERT_TRUE(EncodeTile(encoding, original, qkv_dim,
                         hwy::Span<char>(encoded.data(), encoded.size())));

  const float* data = reinterpret_cast<const float*>(encoded.data());

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
  std::vector<char> encoded(size);

  ASSERT_TRUE(EncodeTile(encoding, original, qkv_dim,
                         hwy::Span<char>(encoded.data(), encoded.size())));

  const float* data = reinterpret_cast<const float*>(encoded.data());

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
  std::vector<char> encoded(size);

  ASSERT_TRUE(EncodeTile(encoding, original, qkv_dim,
                         hwy::Span<char>(encoded.data(), encoded.size())));

  const int8_t* data = reinterpret_cast<const int8_t*>(encoded.data());

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
  std::vector<char> encoded(size);

  ASSERT_TRUE(EncodeTile(encoding, original, qkv_dim,
                         hwy::Span<char>(encoded.data(), encoded.size())));

  const int8_t* data = reinterpret_cast<const int8_t*>(encoded.data());

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

}  // namespace
}  // namespace gcpp
