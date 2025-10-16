#ifndef THIRD_PARTY_GEMMA_CPP_UTIL_ZONES_H_
#define THIRD_PARTY_GEMMA_CPP_UTIL_ZONES_H_

#include "hwy/profiler.h"

namespace gcpp {

// Zones for the profiler.
enum class Zones {
  kOpsRmsNormMul,
  kOpsRmsNorm,
  kOpsRmsNormInplace,
  kOpsRope,
  kOpsRopeAndMulBy,
  kOpsAddFrom,
  kOpsMulByConst,
  kOpsMulByConstTo,
  kOpsMulByConstAndAdd,
  kOpsMulByConstAndAddTile,
  kOpsMulByConstAndAddTile4,
  kOpsMulByConstAndAddVector,
  kOpsSoftmax,
  kOpsLogitsSoftCap,
  kFlashAttentionTransposeQ,
  kFlashAttentionRmsNormAndPositionalEncoding,
  kFlashAttentionSingleFlashAttention,
  kFlashAttentionTileFlashAttention,
  kFlashAttentionTileFlashAttention4,
  kFlashAttentionFlashAttention,
  kGenActivation,
  kGenActivationFused,
  kGenSampleTop1,
  kGenSampleTopK,
  kGenAttentionQDotK,
  kGenAttentionDotSoftmaxWeightedSumPar,
  kStartupWeightsReadAllToBF16,
  kStartupWeightsReadBatches,
  kMMDispatch,
  kMMMatMul,
  kMMTwoMatMul,
  kMMDecompressA,
  kMMNT,
  kMMNT_K,
  kMMNT_MT,
  kMMNT_MT_K,
  kNumZones
};

// Initializes the profiler zones. Must be called before any other profiler
// functions.
void InitProfilerZones(hwy::Profiler& profiler);

// Returns the zone handle for the given zone enum value.
hwy::profiler::ZoneHandle GetProfilerZone(Zones zone);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_ZONES_H_
