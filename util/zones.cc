#include "util/zones.h"

#include "hwy/profiler.h"

namespace gcpp {

#if PROFILER_ENABLED
static constexpr size_t kNumZones = static_cast<size_t>(Zones::kNumZones);

static const char* kProfilerZoneNames[kNumZones] = {
    // Keep in sync with Zones enum.
    "Ops.RMSNormMul",
    "Ops.RMSNorm",
    "Ops.RMSNormInplace",
    "Ops.Rope",
    "Ops.RopeAndMulBy",
    "Ops.AddFrom",
    "Ops.MulByConst",
    "Ops.MulByConstTo",
    "Ops.MulByConstAndAdd",
    "Ops.MulByConstAndAddTile",
    "Ops.MulByConstAndAddTile4",
    "Ops.MulByConstAndAddVector",
    "Ops.Softmax",
    "Ops.LogitsSoftCap",
    "FlashAttention.TransposeQ",
    "FlashAttention.RMSNormAndPositionalEncoding",
    "FlashAttention.SingleFlashAttention",
    "FlashAttention.TileFlashAttention",
    "FlashAttention.TileFlashAttention4",
    "FlashAttention.FlashAttention",
    "Gen.Activation",
    "Gen.ActivationFused",
    "Gen.SampleTop1",
    "Gen.SampleTopK",
    "Gen.Attention.QDotK",
    "Gen.Attention.DotSoftmaxWeightedSum.par",
    "Startup.Weights.ReadAllToBF16",
    "Startup.Weights.ReadBatches",
    "MM.Dispatch",
    "MM.MatMul",
    "MM.TwoMatMul",
    "MM.DecompressA",
    "MM.NT",
    "MM.NT_K",
    "MM.NT_MT",
    "MM.NT_MT_K",
};

static hwy::profiler::ZoneHandle profiler_zone_handles[kNumZones];
#endif

void InitProfilerZones(hwy::Profiler& profiler) {
#if PROFILER_ENABLED
  // Initialize the zone handles. This is done once at startup.
  for (size_t i = 0; i < kNumZones; ++i) {
    profiler_zone_handles[i] = profiler.AddZone(kProfilerZoneNames[i]);
  }
#endif
}

hwy::profiler::ZoneHandle GetProfilerZone(Zones zone) {
#if PROFILER_ENABLED
  return profiler_zone_handles[static_cast<size_t>(zone)];
#else
  return hwy::profiler::ZoneHandle();
#endif
}

}  // namespace gcpp
