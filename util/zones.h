#ifndef THIRD_PARTY_GEMMA_CPP_UTIL_ZONES_H_
#define THIRD_PARTY_GEMMA_CPP_UTIL_ZONES_H_

#include <stddef.h>

#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"

namespace gcpp {

// Zones for the profiler.
enum class Zones {  // Keep sorted
  kFlashAttentionFlashAttention,
  kFlashAttentionInclusive,
  kFlashAttentionRmsNormAndPositionalEncoding,
  kFlashAttentionSingleFlashAttention,
  kFlashAttentionTileFlashAttention,
  kFlashAttentionTileFlashAttention4,
  kFlashAttentionTransposeQ,
  kGenActivation,
  kGenActivationFused,
  kGenAttention,
  kGenAttentionComputeQKV,
  kGenAttentionDotSoftmaxWeightedSumInclusive,
  kGenAttentionDotSoftmaxWeightedSumPar,
  kGenAttentionQDotK,
  kGenAttentionSumHeads,
  kGenEmbed,
  kGenEmbeddingMatmul,
  kGenFFW,
  kGenSampleTop1,
  kGenSampleTopK,
  kMMDecompressA,
  kMMDispatch,
  kMMMatMul,
  kMMNT_K,
  kMMNT_MT_K,
  kMMNT_MT,
  kMMNT,
  kMMTwoMatMul,
  kOpsAddFrom,
  kOpsLogitsSoftCap,
  // kOpsMulByConst,  // removed due to overhead
  // kOpsMulByConstAndAdd,        // removed due to overhead
  // kOpsMulByConstAndAddTile,    // removed due to overhead
  // kOpsMulByConstAndAddTile4,   // removed due to overhead
  // kOpsMulByConstAndAddVector,  // removed due to overhead
  kOpsMulByConstTo,
  kOpsRmsNorm,
  kOpsRmsNormInplace,
  kOpsRmsNormMul,
  kOpsRope,
  kOpsRopeAndMulBy,
  kOpsSoftmax,
  kStartupWeightsReadAllToBF16,
  kStartupWeightsReadBatches,
  kNumZones  // must be last
};

// Owned by ThreadingContext.
class ProfilerZones {
 public:
  ProfilerZones(hwy::Profiler& profiler);

  hwy::profiler::ZoneHandle Get(Zones zone) {
    HWY_DASSERT(zone != Zones::kNumZones);
    return handles_[static_cast<size_t>(zone)];
  }

 private:
  hwy::profiler::ZoneHandle handles_[static_cast<size_t>(Zones::kNumZones)];
};

enum class Callers {  // Keep sorted
  kActivationBatched,
  kAllocateAndBindAll,
  kAttComputeQKV,
  kAttDotSoftmaxWeightedSum,
  kBlobWriter,
  kCompress,
  kFixupWeights,
  kFlashAttention,
  kFlashRMSNormAndPositionalEncoding,
  kFlashTransposeQ,
  kMMClusterForMC,
  kMMClusterForMCNC,
  kMMClusterForN,
  kMMHierForMC,
  kMMHierForMCNC,
  kMMHierForN,
  kOpsAddFromBatched,
  kOpsMaybeLogitsSoftCapBatched,
  kOpsRMSNormBatched,
  kOpsRMSNormInplaceBatched,
  kReadAllToBF16,
  kReadBatches,
  kSampleAndStream,
  kTest,  // only for unit tests.
  kTunePool,
  kVitDotSoftmax1,
  kVitDotSoftmax2,
  kVitDotSoftmax3,
  kVitDotSoftmax4,
  kNumCallers  // must be last
};

// Owned by ThreadingContext.
class PoolCallers {
 public:
  PoolCallers();

  hwy::pool::Caller Get(Callers caller) {
    HWY_DASSERT(caller != Callers::kNumCallers);
    return callers_[static_cast<size_t>(caller)];
  }

 private:
  hwy::pool::Caller callers_[static_cast<size_t>(Callers::kNumCallers)];
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_ZONES_H_
