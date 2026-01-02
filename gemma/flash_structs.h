#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_STRUCTS_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_STRUCTS_H_

#include <stddef.h>

#include <limits>

namespace gcpp {

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

static constexpr size_t kVTileSize4 = 4;

struct Tile4FlashState {
  OnlineSoftmaxState row_states[kVTileSize4];
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_FLASH_STRUCTS_H_
