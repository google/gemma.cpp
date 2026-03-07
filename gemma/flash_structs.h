#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_STRUCTS_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_STRUCTS_H_

#include <stddef.h>
#include <stdint.h>

#include <limits>

namespace gcpp {

// The vertical tile size in flash attention when register lanes correspond to
// K-timesteps, and the number of registers is 4 for 4 Q-rows.
static constexpr size_t k4xNFVTileSize = 4;
// The vertical tile size in flash attention when register lanes correspond to
// K-timesteps, and the number of registers is 8 for 8 Q-rows.
static constexpr size_t k8xNFVTileSize = 8;

// State for computing softmax in a streaming ("online") manner,
// avoiding large intermediate values by subtracting the running maximum.
// For a sequence x_1, ..., x_n:
// m_i = max(m_{i-1}, x_i)
// d_i = d_{i-1} * exp(m_{i-1} - m_i) + exp(x_i - m_i)
// softmax_i = exp(x_i - m_i) / d_i
struct OnlineSoftmaxState {
  // Maximum logit value encountered so far.
  float max = -std::numeric_limits<float>::max() / 2.0f;
  // Sum of exponentials scaled by exp(-max).
  float d = 0.0f;
};

struct Tile4FlashState {
  OnlineSoftmaxState row_states[k8xNFVTileSize];
};

// Parameters for a strip of tiles of flash attention. For processing a strip
// of tiles, each of 1, k4xNFVTileSize, or k8xNFVTileSize Q-rows, by NF
// k-positions. The total width of the strip might cover the entire sequence,
// or a part of it, depending on whether the strip has been split.
struct Tile148Params {
  // Vertical tile size gives the number used in the k8xNFVTileSize arrays.
  // It is the number of Q rows in the tile.
  uint32_t v_tile_size = 0;
  // min start position across all rows in the tile determines the
  // mask used for the tile.
  uint32_t min_start_pos = std::numeric_limits<uint32_t>::max();
  // max last position across all rows in the tile determines the mask
  // used for the tile.
  uint32_t max_last_pos = 0;
  // Index into the qbatch.KV is the same for each row in the tile.
  uint32_t qi_index;
  // Index into the kv_cache is the same for each row in the tile.
  uint32_t kv_offset;
  // In the original task, the index to the split tasks of the first split task.
  uint32_t split_index = 0;
  // The index of the split for running split attention.
  uint32_t i_of_n = 0;
  // The number of splits for running split attention.
  uint32_t n_of_n = 0;
  // Offsets into original Q for each row in the tile.
  uint32_t q_offsets[k8xNFVTileSize];
  // Offsets into att_out for each row in the tile.
  uint32_t out_offsets[k8xNFVTileSize];
  // Start k-positions for each row in the tile.
  uint32_t start_pos[k8xNFVTileSize];
  // Last k-positions for each row in the tile. Inclusive.
  uint32_t last_pos[k8xNFVTileSize];
  // Row index to att_out.
  uint32_t tq_idx[k8xNFVTileSize];
  // Flash attention state for the tile.
  Tile4FlashState end_state;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_STRUCTS_H_
