#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_KV_TRANSCODING_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_KV_TRANSCODING_H_

#include <cstddef>
#include <optional>
#include <vector>

#include "gemma/configs.h"
#include "hwy/aligned_allocator.h"

namespace gcpp {

// Returns the size in bytes of a single KV cache tile for a given encoding.
// Returns std::nullopt if the encoding is unsupported.
std::optional<size_t> GetTileSizeBytes(gcpp::KVEncoding encoding,
                                       size_t qkv_dim);

// Canonical representation of a single tile of K and V data decoded to float32.
// Layout: K is [tile_size, qkv_dim] contiguous, V is [tile_size, qkv_dim]
// contiguous.
struct DecodedTile {
  std::vector<float, hwy::AlignedAllocator<float>> k;
  std::vector<float, hwy::AlignedAllocator<float>> v;
  size_t qkv_dim = 0;
  size_t tile_size = 0;

  DecodedTile() = default;
  DecodedTile(size_t qkv_dim, size_t tile_size)
      : k(qkv_dim * tile_size),
        v(tile_size * qkv_dim),
        qkv_dim(qkv_dim),
        tile_size(tile_size) {}

  float& k_elem(size_t token, size_t dim) { return k[token * qkv_dim + dim]; }
  const float& k_elem(size_t token, size_t dim) const {
    return k[token * qkv_dim + dim];
  }

  float& v_elem(size_t token, size_t dim) { return v[token * qkv_dim + dim]; }
  const float& v_elem(size_t token, size_t dim) const {
    return v[token * qkv_dim + dim];
  }
};

// Allocates an aligned buffer for storing
// an encoded tile of the given encoding.
hwy::AlignedUniquePtr<char[]> AllocateEncodedTile(gcpp::KVEncoding encoding,
                                                  size_t qkv_dim);

// Decodes a single tile's K and V data from its encoded byte buffer into
// float32 using the specified encoding.
bool DecodeTile(gcpp::KVEncoding encoding,
                hwy::Span<const char> encoded_tile_data, size_t qkv_dim,
                DecodedTile* out);

// Encodes a single tile's K and V data from standard float32 into the target
// encoding. Returns false if the encoding is unsupported.
bool EncodeTile(gcpp::KVEncoding encoding, const DecodedTile& decoded,
                size_t qkv_dim, hwy::Span<char> out_encoded_tile_data);

// Convenience utility to convert a tile directly from one encoding to another.
// Return false if either encoding is unsupported or passed data is too small.
bool TranscodeTile(gcpp::KVEncoding src_encoding,
                   hwy::Span<const char> src_data,
                   gcpp::KVEncoding dst_encoding, hwy::Span<char> dst_data,
                   size_t qkv_dim);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_KV_TRANSCODING_H_
