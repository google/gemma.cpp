#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_TILED_ATTENTION_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_TILED_ATTENTION_H_

#include <stddef.h>

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "gemma/gemma.h"
#include "util/allocator.h"
#include "hwy/aligned_allocator.h"
#include "hwy/highway.h"

namespace gcpp {

// Passed to HWY_VISIT_TARGETS; declares for one target.
#define GEMMA_DECL_TILED_ATTENTION(TARGET, NAMESPACE)                          \
  namespace NAMESPACE {                                                        \
  void TiledAttention(AttentionImpl attention_impl, size_t num_tokens,         \
                      size_t layer_idx, const LayerWeightsPtrs& layer,         \
                      AttentionActivationsPtrs& activations, QBatch& qbatch,   \
                      MatMulEnv& env, int flags);                              \
  void LocalAttentionForAllHeadsTokensAndBatch(                                \
      AttentionImpl attention_impl, const size_t num_tokens,                   \
      const size_t layer_idx, const LayerWeightsPtrs& layer,                   \
      AttentionActivationsPtrs& activations, QBatch& qbatch,                   \
      ThreadingContext& ctx);                                                  \
                                                                               \
  void CompressQueriesBF16(hwy::Span<const float* const> input, int qkv_dim,   \
                           BF16* HWY_RESTRICT output);                         \
  void CompressQueriesBF16Contiguous(const float* HWY_RESTRICT input,          \
                                     int qkv_dim, size_t num_queries,          \
                                     BF16* HWY_RESTRICT output);               \
                                                                               \
  void CompressQueriesInt16(hwy::Span<const float* const> input, int qkv_dim,  \
                            int16_t* HWY_RESTRICT output,                      \
                            float* HWY_RESTRICT scale);                        \
                                                                               \
  void CompressQueriesInt16Contiguous(const float* HWY_RESTRICT input,         \
                                      int qkv_dim, size_t num_queries,         \
                                      int16_t* HWY_RESTRICT output,            \
                                      float* HWY_RESTRICT scale);              \
                                                                               \
  void CompressAndTransposeQueriesMatrixAccumulation(const float* raw_queries, \
                                                     BF16* packed_queries,     \
                                                     size_t num_queries,       \
                                                     size_t qkv_dim);          \
  void CompressAndQuantizeQueriesMatrixAccumulationInt8(                       \
      const float* raw_queries, int8_t* packed_queries, float* packed_scales,  \
      size_t num_queries, size_t qkv_dim);                                     \
  /* NOLINTNEXTLINE(google-readability-namespace-comments) */                  \
  }  // namespace NAMESPACE

// Function declarations for each SIMD target. Allows direct call from the
// per-target namespace. We may later replace this with dynamic dispatch if
// the overhead is acceptable.
HWY_VISIT_TARGETS(GEMMA_DECL_TILED_ATTENTION)

#undef GEMMA_DECL_TILED_ATTENTION
}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_TILED_ATTENTION_H_
