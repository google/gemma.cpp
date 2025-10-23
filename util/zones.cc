#include "util/zones.h"

#include <stddef.h>

#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"

namespace gcpp {
namespace {

const char* ZoneName(Zones zone) {
  switch (zone) {
    case Zones::kFlashAttentionFlashAttention:
      return "FlashAttention.FlashAttention";
    case Zones::kFlashAttentionInclusive:
      return "FlashAttention.Inclusive";
    case Zones::kFlashAttentionRmsNormAndPositionalEncoding:
      return "FlashAttention.RMSNormAndPositionalEncoding";
    case Zones::kFlashAttentionSingleFlashAttention:
      return "FlashAttention.SingleFlashAttention";
    case Zones::kFlashAttentionTileFlashAttention:
      return "FlashAttention.TileFlashAttention";
    case Zones::kFlashAttentionTileFlashAttention4:
      return "FlashAttention.TileFlashAttention4";
    case Zones::kFlashAttentionTransposeQ:
      return "FlashAttention.TransposeQ";
    case Zones::kGenActivation:
      return "Gen.Activation";
    case Zones::kGenActivationFused:
      return "Gen.ActivationFused";
    case Zones::kGenAttention:
      return "Gen.Attention";
    case Zones::kGenAttentionComputeQKV:
      return "Gen.Attention.ComputeQKV";
    case Zones::kGenAttentionDotSoftmaxWeightedSumInclusive:
      return "Gen.Attention.DotSoftmaxWeightedSumInclusive";
    case Zones::kGenAttentionDotSoftmaxWeightedSumPar:
      return "Gen.Attention.DotSoftmaxWeightedSum.par";
    case Zones::kGenAttentionQDotK:
      return "Gen.Attention.QDotK";
    case Zones::kGenAttentionSumHeads:
      return "Gen.Attention.SumHeads";
    case Zones::kGenEmbed:
      return "Gen.Embed";
    case Zones::kGenEmbeddingMatmul:
      return "Gen.EmbeddingMatmul";
    case Zones::kGenFFW:
      return "Gen.FFW";
    case Zones::kGenSampleTop1:
      return "Gen.SampleTop1";
    case Zones::kGenSampleTopK:
      return "Gen.SampleTopK";
    case Zones::kMMDecompressA:
      return "MM.DecompressA";
    case Zones::kMMDispatch:
      return "MM.Dispatch";
    case Zones::kMMMatMul:
      return "MM.MatMul";
    case Zones::kMMNT_K:
      return "MM.NT_K";
    case Zones::kMMNT_MT_K:
      return "MM.NT_MT_K";
    case Zones::kMMNT_MT:
      return "MM.NT_MT";
    case Zones::kMMNT:
      return "MM.NT";
    case Zones::kMMTwoMatMul:
      return "MM.TwoMatMul";
    case Zones::kOpsAddFrom:
      return "Ops.AddFrom";
    case Zones::kOpsLogitsSoftCap:
      return "Ops.LogitsSoftCap";
    // case Zones::kOpsMulByConst:  // removed due to overhead
    // case Zones::kOpsMulByConstAndAdd:        // removed due to overhead
    // case Zones::kOpsMulByConstAndAddTile:    // removed due to overhead
    // case Zones::kOpsMulByConstAndAddTile4:   // removed due to overhead
    // case Zones::kOpsMulByConstAndAddVector:  // removed due to overhead
    case Zones::kOpsMulByConstTo:
      return "Ops.MulByConstTo";
    case Zones::kOpsRmsNorm:
      return "Ops.RMSNorm";
    case Zones::kOpsRmsNormInplace:
      return "Ops.RMSNormInplace";
    case Zones::kOpsRmsNormMul:
      return "Ops.RMSNormMul";
    case Zones::kOpsRope:
      return "Ops.Rope";
    case Zones::kOpsRopeAndMulBy:
      return "Ops.RopeAndMulBy";
    case Zones::kOpsSoftmax:
      return "Ops.Softmax";
    case Zones::kStartupWeightsReadAllToBF16:
      return "Startup.Weights.ReadAllToBF16";
    case Zones::kStartupWeightsReadBatches:
      return "Startup.Weights.ReadBatches";
    default:
      HWY_ABORT("Invalid zone %d.", static_cast<int>(zone));
  }
}

hwy::ProfilerFlags ZoneFlags(Zones zone) {
  switch (zone) {
    case Zones::kFlashAttentionInclusive:
    case Zones::kGenAttention:
    case Zones::kGenAttentionComputeQKV:
    case Zones::kGenAttentionDotSoftmaxWeightedSumInclusive:
    case Zones::kGenAttentionSumHeads:
    case Zones::kGenEmbed:
    case Zones::kGenEmbeddingMatmul:
    case Zones::kGenFFW:
      return hwy::ProfilerFlags::kInclusive;
    default:
      return hwy::ProfilerFlags::kDefault;
  }
}

const char* CallerName(Callers caller) {
  switch (caller) {
    case Callers::kActivationBatched:
      return "ActivationBatched";
    case Callers::kAllocateAndBindAll:
      return "AllocateAndBindAll";
    case Callers::kAttComputeQKV:
      return "Att.ComputeQKV";
    case Callers::kAttDotSoftmaxWeightedSum:
      return "Att.DotSoftmaxWeightedSum";
    case Callers::kBlobWriter:
      return "BlobWriter";
    case Callers::kCompress:
      return "Compress";
    case Callers::kFixupWeights:
      return "FixupWeights";
    case Callers::kFlashAttention:
      return "FlashAttention";
    case Callers::kFlashRMSNormAndPositionalEncoding:
      return "Flash.RMSNormAndPositionalEncoding";
    case Callers::kFlashTransposeQ:
      return "Flash.TransposeQ";
    case Callers::kMMClusterForMC:
      return "MM.ClusterForMC";
    case Callers::kMMClusterForMCNC:
      return "MM.ClusterForMCNC";
    case Callers::kMMClusterForN:
      return "MM.ClusterForN";
    case Callers::kMMHierForMC:
      return "MM.HierForMC";
    case Callers::kMMHierForMCNC:
      return "MM.HierForMCNC";
    case Callers::kMMHierForN:
      return "MM.HierForN";
    case Callers::kOpsAddFromBatched:
      return "Ops.AddFromBatched";
    case Callers::kOpsMaybeLogitsSoftCapBatched:
      return "Ops.MaybeLogitsSoftCapBatched";
    case Callers::kOpsRMSNormBatched:
      return "Ops.RMSNormBatched";
    case Callers::kOpsRMSNormInplaceBatched:
      return "Ops.RMSNormInplaceBatched";
    case Callers::kReadAllToBF16:
      return "ReadAllToBF16";
    case Callers::kReadBatches:
      return "ReadBatches";
    case Callers::kSampleAndStream:
      return "SampleAndStream";
    case Callers::kTest:  // only for unit tests.
      return "Test-only!";
    case Callers::kTunePool:
      return "TunePool";
    case Callers::kVitDotSoftmax1:
      return "Vit.DotSoftmax1";
    case Callers::kVitDotSoftmax2:
      return "Vit.DotSoftmax2";
    case Callers::kVitDotSoftmax3:
      return "Vit.DotSoftmax3";
    case Callers::kVitDotSoftmax4:
      return "Vit.DotSoftmax4";
    default:
      HWY_ABORT("Invalid caller %d.", static_cast<int>(caller));
  }
}

}  // namespace

ProfilerZones::ProfilerZones(hwy::Profiler& profiler) {
  for (size_t i = 0;; ++i) {
    const Zones zone = static_cast<Zones>(i);
    if (zone == Zones::kNumZones) break;
    handles_[i] = profiler.AddZone(ZoneName(zone), ZoneFlags(zone));
  }
}

PoolCallers::PoolCallers() {
  for (size_t i = 0;; ++i) {
    const Callers caller = static_cast<Callers>(i);
    if (caller == Callers::kNumCallers) break;
    callers_[i] = hwy::ThreadPool::AddCaller(CallerName(caller));
  }
}

}  // namespace gcpp
