#include "gemma/kv_transcoding.h"

#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <optional>

#include "compression/types.h"
#include "gemma/activations.h"
#include "gemma/configs.h"
#include "gemma/kv_cache.h"
#include "util/basics.h"
#include "hwy/base.h"
#include "hwy/highway.h"

namespace gcpp {

std::optional<size_t> GetTileSizeBytes(gcpp::KVEncoding encoding,
                                       size_t qkv_dim) {
  constexpr size_t kTileSize = gcpp::KVCache::kTileSize;
  size_t tileElements = qkv_dim * kTileSize * 2;
  switch (encoding) {
    case gcpp::KVEncoding::kInt8:
    case gcpp::KVEncoding::kInt8TwoTranspositions:
    case gcpp::KVEncoding::kInt8MatrixAccumulation:
      return tileElements * sizeof(int8_t) +
             kTileSize * 2 * sizeof(gcpp::KV_microscale_t);
    case gcpp::KVEncoding::kInt8VNNITwoTranspositions:
      return tileElements * sizeof(int8_t) +
             kTileSize * 2 * sizeof(gcpp::KV_microscale_t) +
             kTileSize * sizeof(int32_t);
    case gcpp::KVEncoding::kBF16:
    case gcpp::KVEncoding::kBF16TwoTranspositions:
    case gcpp::KVEncoding::kBF16MatrixAccumulation:
      return tileElements * sizeof(gcpp::BF16);
    case gcpp::KVEncoding::kF32:
    case gcpp::KVEncoding::kF32TwoTranspositions:
      return tileElements * sizeof(float);
    default:
      return std::nullopt;
  }
}

namespace {
constexpr size_t kTileSize = gcpp::KVCache::kTileSize;

inline size_t KOffset(bool transposed, size_t qkv_dim, size_t dim,
                      size_t token) {
  HWY_DASSERT(dim < qkv_dim && token < kTileSize);
  return transposed ? ((dim / 2) * kTileSize * 2 + token * 2 + (dim % 2))
                    : (dim * kTileSize + token);
}

inline size_t VOffset(bool transposed, size_t qkv_dim, size_t dim,
                      size_t token) {
  return transposed ? ((token / 2) * qkv_dim * 2 + dim * 2 + (token % 2))
                    : (token * qkv_dim + dim);
}

int8_t Quantize(float v, float inv_scale) {
  float scaled = std::nearbyint(v * inv_scale);
  if (scaled > 127.0f) return 127;
  if (scaled < -127.0f) return -127;
  return hwy::ConvertScalarTo<int8_t>(scaled);
}

template <typename DecodeKFn, typename DecodeVFn>
inline void DecodeTileWithFn(size_t qkv_dim, DecodedTile* out,
                             const DecodeKFn& decode_k,
                             const DecodeVFn& decode_v) {
  for (size_t token = 0; token < kTileSize; ++token) {
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      out->k_elem(token, dim) = decode_k(dim, token);
    }
  }
  for (size_t token = 0; token < kTileSize; ++token) {
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      out->v_elem(token, dim) = decode_v(dim, token);
    }
  }
}

template <typename EncodeKFn, typename EncodeVFn>
inline void EncodeTileWithFn(size_t qkv_dim, const DecodedTile& decoded,
                             const EncodeKFn& encode_k,
                             const EncodeVFn& encode_v) {
  for (size_t token = 0; token < kTileSize; ++token) {
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      encode_k(dim, token, decoded.k_elem(token, dim));
    }
  }
  for (size_t token = 0; token < kTileSize; ++token) {
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      encode_v(dim, token, decoded.v_elem(token, dim));
    }
  }
}

void EncodeTileF32(bool transposed, size_t qkv_dim, const DecodedTile& decoded,
                   hwy::Span<char> out_encoded_tile_data) {
  float* data = HWY_RCAST_ALIGNED(float*, out_encoded_tile_data.data());
  const size_t v_start = qkv_dim * kTileSize;
  EncodeTileWithFn(
      qkv_dim, decoded,
      [&](size_t dim, size_t token, float val)
          HWY_ATTR { data[KOffset(transposed, qkv_dim, dim, token)] = val; },
      [&](size_t dim, size_t token, float val) HWY_ATTR {
        data[v_start + VOffset(transposed, qkv_dim, dim, token)] = val;
      });
}

void EncodeTileBF16(bool transposed, size_t qkv_dim, const DecodedTile& decoded,
                    hwy::Span<char> out_encoded_tile_data) {
  gcpp::BF16* data =
      HWY_RCAST_ALIGNED(gcpp::BF16*, out_encoded_tile_data.data());
  const size_t v_start = qkv_dim * kTileSize;
  EncodeTileWithFn(
      qkv_dim, decoded,
      [&](size_t dim, size_t token, float val) HWY_ATTR {
        data[KOffset(transposed, qkv_dim, dim, token)] =
            hwy::ConvertScalarTo<hwy::bfloat16_t>(val);
      },
      [&](size_t dim, size_t token, float val) HWY_ATTR {
        data[v_start + VOffset(transposed, qkv_dim, dim, token)] =
            hwy::ConvertScalarTo<hwy::bfloat16_t>(val);
      });
}

void EncodeTileBF16MatrixAccumulation(size_t qkv_dim,
                                      const DecodedTile& decoded,
                                      hwy::Span<char> out_encoded_tile_data) {
  gcpp::BF16* data =
      HWY_RCAST_ALIGNED(gcpp::BF16*, out_encoded_tile_data.data());
  const size_t tile_size = decoded.tile_size;
  const size_t v_start = qkv_dim * tile_size;
  const size_t num_groups = tile_size / 8;
  const size_t num_ch_groups = qkv_dim / 4;

  for (size_t ch_g = 0; ch_g < num_ch_groups; ++ch_g) {
    for (size_t g = 0; g < num_groups; ++g) {
      size_t base_offset = ch_g * 128 + g * 32;
      // Pack K (8x4 block, token-major)
      for (size_t t_in_g = 0; t_in_g < 8; ++t_in_g) {
        size_t token = g * 8 + t_in_g;
        for (size_t ch_in_g = 0; ch_in_g < 4; ++ch_in_g) {
          size_t dim = ch_g * 4 + ch_in_g;
          float val = decoded.k_elem(token, dim);
          data[base_offset + t_in_g * 4 + ch_in_g] =
              hwy::ConvertScalarTo<hwy::bfloat16_t>(val);
        }
      }
    }
  }

  // Pack V (Contiguous SV-Blocked Layout)
  for (size_t t = 0; t < tile_size; ++t) {
    for (size_t c = 0; c < qkv_dim; ++c) {
      float val = decoded.v_elem(t, c);
      size_t g_t = t / 16;
      size_t g_c = (c % 4) / 2;
      size_t sub_block = (c / 4) * 4 + g_t * 2 + g_c;

      size_t t_prime = t % 16;
      size_t c_prime = c % 2;
      size_t g_t4 = t_prime / 4;
      size_t t_double_prime = t_prime % 4;
      size_t block_offset = g_t4 * 8 + c_prime * 4 + t_double_prime;

      size_t v_offset = sub_block * 32 + block_offset;
      data[v_start + v_offset] = hwy::ConvertScalarTo<hwy::bfloat16_t>(val);
    }
  }
}

void EncodeTileInt8MatrixAccumulation(size_t qkv_dim,
                                      const DecodedTile& decoded,
                                      hwy::Span<char> out_encoded_tile_data) {
  HWY_DASSERT(qkv_dim % 8 == 0);
  int8_t* k_data = HWY_RCAST_ALIGNED(int8_t*, out_encoded_tile_data.data());
  int8_t* v_data = k_data + qkv_dim * kTileSize;
  gcpp::KV_microscale_t* scales =
      HWY_RCAST_ALIGNED(gcpp::KV_microscale_t*, v_data + kTileSize * qkv_dim);
  gcpp::KV_microscale_t* k_scales = scales;
  gcpp::KV_microscale_t* v_scales = scales + kTileSize;

  AlignedFloatVector k_max_abs(kTileSize, 0.0f);
  AlignedFloatVector v_max_abs(kTileSize, 0.0f);

  for (size_t token = 0; token < kTileSize; ++token) {
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      k_max_abs[token] =
          std::max(k_max_abs[token], std::abs(decoded.k_elem(token, dim)));
      v_max_abs[token] =
          std::max(v_max_abs[token], std::abs(decoded.v_elem(token, dim)));
    }
  }

  AlignedFloatVector inv_scales_k(kTileSize);
  AlignedFloatVector inv_scales_v(kTileSize);
  for (size_t token = 0; token < kTileSize; ++token) {
    float scale_k = k_max_abs[token] == 0.0f ? 1.0f : k_max_abs[token] / 127.0f;
    k_scales[token] = hwy::ConvertScalarTo<gcpp::KV_microscale_t>(scale_k);
    float decoded_k = hwy::ConvertScalarTo<float>(k_scales[token]);
    inv_scales_k[token] = decoded_k == 0.0f ? 0.0f : 1.0f / decoded_k;

    float scale_v = v_max_abs[token] == 0.0f ? 1.0f : v_max_abs[token] / 127.0f;
    v_scales[token] = hwy::ConvertScalarTo<gcpp::KV_microscale_t>(scale_v);
    float decoded_v = hwy::ConvertScalarTo<float>(v_scales[token]);
    inv_scales_v[token] = decoded_v == 0.0f ? 0.0f : 1.0f / decoded_v;
  }

  // 2. Quantize and pack K and V
  for (size_t token = 0; token < kTileSize; ++token) {
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      size_t k_offset = MatrixAccumulationOffset_Int8(qkv_dim, dim, token);
      k_data[k_offset] =
          Quantize(decoded.k_elem(token, dim), inv_scales_k[token]);

      // V transposed layout (channel-major within 2-token blocks)
      size_t v_offset = VMatrixAccumulationOffset_Int8(qkv_dim, token, dim);
      v_data[v_offset] =
          Quantize(decoded.v_elem(token, dim), inv_scales_v[token]);
    }
  }
}

void EncodeTileInt8(bool transposed, size_t qkv_dim, const DecodedTile& decoded,
                    hwy::Span<char> out_encoded_tile_data) {
  int8_t* k_data = HWY_RCAST_ALIGNED(int8_t*, out_encoded_tile_data.data());
  int8_t* v_data = k_data + qkv_dim * kTileSize;
  gcpp::KV_microscale_t* scales =
      HWY_RCAST_ALIGNED(gcpp::KV_microscale_t*, v_data + kTileSize * qkv_dim);
  gcpp::KV_microscale_t* k_scales = scales;
  gcpp::KV_microscale_t* v_scales = scales + kTileSize;

  AlignedFloatVector k_max_abs(kTileSize, 0.0f);
  AlignedFloatVector v_max_abs(kTileSize, 0.0f);

  for (size_t token = 0; token < kTileSize; ++token) {
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      k_max_abs[token] =
          std::max(k_max_abs[token], std::abs(decoded.k_elem(token, dim)));
    }
  }
  for (size_t token = 0; token < kTileSize; ++token) {
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      v_max_abs[token] =
          std::max(v_max_abs[token], std::abs(decoded.v_elem(token, dim)));
    }
  }

  AlignedFloatVector inv_scales_k(kTileSize);
  AlignedFloatVector inv_scales_v(kTileSize);
  for (size_t token = 0; token < kTileSize; ++token) {
    float scale_k = k_max_abs[token] == 0.0f ? 1.0f : k_max_abs[token] / 127.0f;
    k_scales[token] = hwy::ConvertScalarTo<gcpp::KV_microscale_t>(scale_k);
    inv_scales_k[token] = 1.0f / scale_k;

    float scale_v = v_max_abs[token] == 0.0f ? 1.0f : v_max_abs[token] / 127.0f;
    v_scales[token] = hwy::ConvertScalarTo<gcpp::KV_microscale_t>(scale_v);
    inv_scales_v[token] = 1.0f / scale_v;
  }

  EncodeTileWithFn(
      qkv_dim, decoded,
      [&](size_t dim, size_t token, float val) HWY_ATTR {
        k_data[KOffset(transposed, qkv_dim, dim, token)] =
            Quantize(val, inv_scales_k[token]);
      },
      [&](size_t dim, size_t token, float val) HWY_ATTR {
        v_data[VOffset(transposed, qkv_dim, dim, token)] =
            Quantize(val, inv_scales_v[token]);
      });
}

void EncodeTileInt8VNNI(size_t qkv_dim, const DecodedTile& decoded,
                        hwy::Span<char> out_encoded_tile_data) {
  int8_t* k_data = HWY_RCAST_ALIGNED(int8_t*, out_encoded_tile_data.data());
  int8_t* v_data = k_data + qkv_dim * kTileSize;
  gcpp::KV_microscale_t* scales =
      HWY_RCAST_ALIGNED(gcpp::KV_microscale_t*, v_data + kTileSize * qkv_dim);
  gcpp::KV_microscale_t* k_scales = scales;
  gcpp::KV_microscale_t* v_scales = scales + kTileSize;
  int32_t* k_sums = HWY_RCAST_ALIGNED(int32_t*, v_scales + kTileSize);

  AlignedFloatVector k_max_abs(kTileSize, 0.0f);
  AlignedFloatVector v_max_abs(kTileSize, 0.0f);

  for (size_t token = 0; token < kTileSize; ++token) {
    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      k_max_abs[token] =
          std::max(k_max_abs[token], std::abs(decoded.k_elem(token, dim)));
      v_max_abs[token] =
          std::max(v_max_abs[token], std::abs(decoded.v_elem(token, dim)));
    }
  }

  AlignedFloatVector inv_scales_k(kTileSize);
  AlignedFloatVector inv_scales_v(kTileSize);
  for (size_t token = 0; token < kTileSize; ++token) {
    float scale_k = k_max_abs[token] == 0.0f ? 1.0f : k_max_abs[token] / 127.0f;
    k_scales[token] = hwy::ConvertScalarTo<gcpp::KV_microscale_t>(scale_k);
    inv_scales_k[token] = 1.0f / scale_k;

    float scale_v = v_max_abs[token] == 0.0f ? 1.0f : v_max_abs[token] / 127.0f;
    v_scales[token] = hwy::ConvertScalarTo<gcpp::KV_microscale_t>(scale_v);
    inv_scales_v[token] = 1.0f / scale_v;
  }

  for (size_t token = 0; token < kTileSize; ++token) {
    k_sums[token] = 0;
  }

  auto KOffset_VNNI = [&](size_t dim, size_t token) {
    return (dim / 4) * kTileSize * 4 + token * 4 + (dim % 4);
  };
  auto VOffset_VNNI = [&](size_t dim, size_t token) {
    return (token / 4) * qkv_dim * 4 + dim * 4 + (token % 4);
  };

  EncodeTileWithFn(
      qkv_dim, decoded,
      [&](size_t dim, size_t token, float val) HWY_ATTR {
        int8_t quantized = Quantize(val, inv_scales_k[token]);
        k_data[KOffset_VNNI(dim, token)] = quantized;
        k_sums[token] += quantized;
      },
      [&](size_t dim, size_t token, float val) HWY_ATTR {
        v_data[VOffset_VNNI(dim, token)] = Quantize(val, inv_scales_v[token]);
      });
}

void DecodeTileInt8VNNI(size_t qkv_dim, hwy::Span<const char> encoded_tile_data,
                        DecodedTile* out) {
  const int8_t* k_data =
      HWY_RCAST_ALIGNED(const int8_t*, encoded_tile_data.data());
  const int8_t* v_data = k_data + qkv_dim * kTileSize;
  const gcpp::KV_microscale_t* scales = HWY_RCAST_ALIGNED(
      const gcpp::KV_microscale_t*, v_data + kTileSize * qkv_dim);
  const gcpp::KV_microscale_t* k_scales = scales;
  const gcpp::KV_microscale_t* v_scales = scales + kTileSize;

  auto KOffset_VNNI = [&](size_t dim, size_t token) {
    return (dim / 4) * kTileSize * 4 + token * 4 + (dim % 4);
  };
  auto VOffset_VNNI = [&](size_t dim, size_t token) {
    return (token / 4) * qkv_dim * 4 + dim * 4 + (token % 4);
  };

  DecodeTileWithFn(
      qkv_dim, out,
      [&](size_t dim, size_t token) HWY_ATTR {
        float scale = hwy::ConvertScalarTo<float>(k_scales[token]);
        return k_data[KOffset_VNNI(dim, token)] * scale;
      },
      [&](size_t dim, size_t token) HWY_ATTR {
        float scale = hwy::ConvertScalarTo<float>(v_scales[token]);
        return v_data[VOffset_VNNI(dim, token)] * scale;
      });
}

void DecodeTileF32(bool transposed, size_t qkv_dim,
                   hwy::Span<const char> encoded_tile_data, DecodedTile* out) {
  const float* data = HWY_RCAST_ALIGNED(const float*, encoded_tile_data.data());
  const size_t v_start = qkv_dim * kTileSize;
  DecodeTileWithFn(
      qkv_dim, out,
      [&](size_t dim, size_t token)
          HWY_ATTR { return data[KOffset(transposed, qkv_dim, dim, token)]; },
      [&](size_t dim, size_t token) HWY_ATTR {
        return data[v_start + VOffset(transposed, qkv_dim, dim, token)];
      });
}

void DecodeTileBF16(bool transposed, size_t qkv_dim,
                    hwy::Span<const char> encoded_tile_data, DecodedTile* out) {
  const gcpp::BF16* data =
      HWY_RCAST_ALIGNED(const gcpp::BF16*, encoded_tile_data.data());
  const size_t v_start = qkv_dim * kTileSize;
  DecodeTileWithFn(
      qkv_dim, out,
      [&](size_t dim, size_t token) HWY_ATTR {
        return hwy::ConvertScalarTo<float>(
            data[KOffset(transposed, qkv_dim, dim, token)]);
      },
      [&](size_t dim, size_t token) HWY_ATTR {
        return hwy::ConvertScalarTo<float>(
            data[v_start + VOffset(transposed, qkv_dim, dim, token)]);
      });
}

void DecodeTileBF16MatrixAccumulation(size_t qkv_dim,
                                      hwy::Span<const char> encoded_tile_data,
                                      DecodedTile* out) {
  const gcpp::BF16* data =
      HWY_RCAST_ALIGNED(const gcpp::BF16*, encoded_tile_data.data());
  const size_t tile_size = out->tile_size;
  const size_t v_start = qkv_dim * tile_size;
  const size_t num_groups = tile_size / 8;
  const size_t num_ch_groups = qkv_dim / 4;

  for (size_t ch_g = 0; ch_g < num_ch_groups; ++ch_g) {
    for (size_t g = 0; g < num_groups; ++g) {
      size_t base_offset = ch_g * 128 + g * 32;
      // Unpack K (8x4 block, token-major)
      for (size_t t_in_g = 0; t_in_g < 8; ++t_in_g) {
        size_t token = g * 8 + t_in_g;
        for (size_t ch_in_g = 0; ch_in_g < 4; ++ch_in_g) {
          size_t dim = ch_g * 4 + ch_in_g;
          out->k_elem(token, dim) = hwy::ConvertScalarTo<float>(
              data[base_offset + t_in_g * 4 + ch_in_g]);
        }
      }
    }
  }

  // Unpack V (Contiguous SV-Blocked Layout)
  for (size_t t = 0; t < tile_size; ++t) {
    for (size_t c = 0; c < qkv_dim; ++c) {
      size_t g_t = t / 16;
      size_t g_c = (c % 4) / 2;
      size_t sub_block = (c / 4) * 4 + g_t * 2 + g_c;

      size_t t_prime = t % 16;
      size_t c_prime = c % 2;
      size_t g_t4 = t_prime / 4;
      size_t t_double_prime = t_prime % 4;
      size_t block_offset = g_t4 * 8 + c_prime * 4 + t_double_prime;

      size_t v_offset = sub_block * 32 + block_offset;
      out->v_elem(t, c) = hwy::ConvertScalarTo<float>(data[v_start + v_offset]);
    }
  }
}

void DecodeTileInt8MatrixAccumulation(size_t qkv_dim,
                                      hwy::Span<const char> encoded_tile_data,
                                      DecodedTile* out) {
  HWY_DASSERT(qkv_dim % 8 == 0);
  const int8_t* k_data =
      HWY_RCAST_ALIGNED(const int8_t*, encoded_tile_data.data());
  const int8_t* v_data = k_data + qkv_dim * kTileSize;
  const gcpp::KV_microscale_t* scales = HWY_RCAST_ALIGNED(
      const gcpp::KV_microscale_t*, v_data + kTileSize * qkv_dim);
  const gcpp::KV_microscale_t* k_scales = scales;
  const gcpp::KV_microscale_t* v_scales = scales + kTileSize;

  for (size_t token = 0; token < kTileSize; ++token) {
    float scale_k = hwy::ConvertScalarTo<float>(k_scales[token]);
    float scale_v = hwy::ConvertScalarTo<float>(v_scales[token]);

    for (size_t dim = 0; dim < qkv_dim; ++dim) {
      size_t k_offset = MatrixAccumulationOffset_Int8(qkv_dim, dim, token);
      out->k_elem(token, dim) = k_data[k_offset] * scale_k;

      // V transposed layout (channel-major within 2-token blocks)
      size_t v_offset = VMatrixAccumulationOffset_Int8(qkv_dim, token, dim);
      out->v_elem(token, dim) = v_data[v_offset] * scale_v;
    }
  }
}

void DecodeTileInt8(bool transposed, size_t qkv_dim,
                    hwy::Span<const char> encoded_tile_data, DecodedTile* out) {
  const int8_t* k_data =
      HWY_RCAST_ALIGNED(const int8_t*, encoded_tile_data.data());
  const int8_t* v_data = k_data + qkv_dim * kTileSize;
  const gcpp::KV_microscale_t* scales = HWY_RCAST_ALIGNED(
      const gcpp::KV_microscale_t*, v_data + kTileSize * qkv_dim);
  const gcpp::KV_microscale_t* k_scales = scales;
  const gcpp::KV_microscale_t* v_scales = scales + kTileSize;

  DecodeTileWithFn(
      qkv_dim, out,
      [&](size_t dim, size_t token) HWY_ATTR {
        float scale = hwy::ConvertScalarTo<float>(k_scales[token]);
        return k_data[KOffset(transposed, qkv_dim, dim, token)] * scale;
      },
      [&](size_t dim, size_t token) HWY_ATTR {
        float scale = hwy::ConvertScalarTo<float>(v_scales[token]);
        return v_data[VOffset(transposed, qkv_dim, dim, token)] * scale;
      });
}

}  // namespace

bool IsTransposed(KVEncoding encoding) {
  switch (encoding) {
    case KVEncoding::kF32TwoTranspositions:
    case KVEncoding::kBF16TwoTranspositions:
    case KVEncoding::kInt8TwoTranspositions:
    case KVEncoding::kInt8VNNITwoTranspositions:
      return true;
    default:
      return false;
  }
}

hwy::AlignedUniquePtr<char[]> AllocateEncodedTile(KVEncoding encoding,
                                                  size_t qkv_dim) {
  std::optional<size_t> size = GetTileSizeBytes(encoding, qkv_dim);
  if (!size.has_value()) return hwy::AlignedUniquePtr<char[]>();
  return hwy::MakeUniqueAlignedArray<char>(*size);
}

bool DecodeTile(KVEncoding encoding, hwy::Span<const char> encoded_tile_data,
                size_t qkv_dim, DecodedTile* out) {
  std::optional<size_t> required_size_or = GetTileSizeBytes(encoding, qkv_dim);
  if (!required_size_or.has_value()) return false;
  size_t required_size = *required_size_or;
  if (encoded_tile_data.size() < required_size) {
    return false;
  }
  bool transposed = IsTransposed(encoding);
  switch (encoding) {
    case gcpp::KVEncoding::kF32:
    case gcpp::KVEncoding::kF32TwoTranspositions: {
      DecodeTileF32(transposed, qkv_dim, encoded_tile_data, out);
      return true;
    }
    case gcpp::KVEncoding::kBF16:
    case gcpp::KVEncoding::kBF16TwoTranspositions: {
      DecodeTileBF16(transposed, qkv_dim, encoded_tile_data, out);
      return true;
    }
    case gcpp::KVEncoding::kBF16MatrixAccumulation: {
      DecodeTileBF16MatrixAccumulation(qkv_dim, encoded_tile_data, out);
      return true;
    }
    case gcpp::KVEncoding::kInt8:
    case gcpp::KVEncoding::kInt8TwoTranspositions: {
      DecodeTileInt8(transposed, qkv_dim, encoded_tile_data, out);
      return true;
    }
    case gcpp::KVEncoding::kInt8VNNITwoTranspositions: {
      DecodeTileInt8VNNI(qkv_dim, encoded_tile_data, out);
      return true;
    }
    case gcpp::KVEncoding::kInt8MatrixAccumulation: {
      DecodeTileInt8MatrixAccumulation(qkv_dim, encoded_tile_data, out);
      return true;
    }
    default:
      return false;
  }
}

bool EncodeTile(gcpp::KVEncoding encoding, const DecodedTile& decoded,
                size_t qkv_dim, hwy::Span<char> out_encoded_tile_data) {
  std::optional<size_t> required_size_or = GetTileSizeBytes(encoding, qkv_dim);
  if (!required_size_or.has_value()) return false;
  size_t required_size = *required_size_or;
  if (out_encoded_tile_data.size() < required_size) {
    return false;
  }
  bool transposed = IsTransposed(encoding);
  switch (encoding) {
    case gcpp::KVEncoding::kF32:
    case gcpp::KVEncoding::kF32TwoTranspositions: {
      EncodeTileF32(transposed, qkv_dim, decoded, out_encoded_tile_data);
      return true;
    }
    case gcpp::KVEncoding::kBF16:
    case gcpp::KVEncoding::kBF16TwoTranspositions: {
      EncodeTileBF16(transposed, qkv_dim, decoded, out_encoded_tile_data);
      return true;
    }
    case gcpp::KVEncoding::kBF16MatrixAccumulation: {
      EncodeTileBF16MatrixAccumulation(qkv_dim, decoded, out_encoded_tile_data);
      return true;
    }
    case gcpp::KVEncoding::kInt8:
    case gcpp::KVEncoding::kInt8TwoTranspositions: {
      EncodeTileInt8(transposed, qkv_dim, decoded, out_encoded_tile_data);
      return true;
    }
    case gcpp::KVEncoding::kInt8VNNITwoTranspositions: {
      EncodeTileInt8VNNI(qkv_dim, decoded, out_encoded_tile_data);
      return true;
    }
    case gcpp::KVEncoding::kInt8MatrixAccumulation: {
      EncodeTileInt8MatrixAccumulation(qkv_dim, decoded, out_encoded_tile_data);
      return true;
    }
    default:
      return false;
  }
}

bool TranscodeTile(gcpp::KVEncoding src_encoding,
                   hwy::Span<const char> src_data,
                   gcpp::KVEncoding dst_encoding, hwy::Span<char> dst_data,
                   size_t qkv_dim) {
  DecodedTile decoded(qkv_dim, kTileSize);
  if (!DecodeTile(src_encoding, src_data, qkv_dim, &decoded)) return false;

  return EncodeTile(dst_encoding, decoded, qkv_dim, dst_data);
}

}  // namespace gcpp
