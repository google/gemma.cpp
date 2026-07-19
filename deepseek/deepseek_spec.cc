// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Greedy self-speculative decoding for DeepSeek V4 (multi-token prediction).
// Split out of gemma.cc; uses its generation internals via
// gemma/generate_internal.h.

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>  // getenv

#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include "gemma/gemma.h"
#include "gemma/kv_cache.h"
#include "gemma/weights.h"
#include "hwy/timer.h"

// Compiles this file for multiple architectures via "foreach_target.h", to
// which we pass the filename via macro 'argument'.
// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "deepseek/deepseek_spec.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "deepseek/deepseek.h"        // includes highway.h
#include "gemma/generate_internal.h"  // includes highway.h
#include "ops/ops-inl.h"              // Top1OfSoftmax

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// Computes logits for all batch rows via the same FinalNormBatched /
// FinalLogits helpers as `SampleAndStream`, without sampling. The final
// softcap is skipped: it is monotonic, so it cannot change the argmax, and
// DeepSeek V4 does not use one.
static void ComputeLogits(const ModelConfig& config, const WeightsPtrs& weights,
                          Activations& activations, MatMulEnv& env) {
  FinalNormBatched(config, weights, activations, env);
  FinalLogits(weights, activations, env);
}

// Greedy self-speculative decoding with the DeepSeek V4 MTP block: each
// iteration feeds [committed, draft] through the main model in one 2-token
// pass and reads logits at both positions. An accepted draft yields two
// tokens per pass. Output is token-for-token identical to normal greedy
// decoding: every emitted token is the argmax of the main model's logits at
// its position; a wrong draft costs only the wasted second row, whose
// compressor state is rolled back from the boundary snapshot and whose KV
// rows are overwritten by the next pass.
void GenerateSpecV4(const ModelConfig& config,
                           const RuntimeConfig& runtime_config,
                           const WeightsPtrs& weights,
                           Activations& activations, QBatch& qbatch,
                           MatMulEnv& env, TimingInfo& timing_info) {
  HWY_ASSERT(qbatch.Size() == 1);
  const size_t max_gen_steps = PrefillTBatchOrQBatch(
      config, runtime_config, weights, activations, qbatch, env, timing_info);
  env.ctx.profiler.PrintResults();

  hwy::BitSet4096<> non_eos;
  non_eos.Set(0);
  StreamAndUpdateEOSAfterPrefill(config, runtime_config, qbatch, non_eos, 0);
  if (!non_eos.Any() || max_gen_steps == 0) return;

  timing_info.generate_start = hwy::platform::Now();

  // Number of StreamToken calls so far, matching the normal path's `pos`.
  size_t stream_pos = qbatch.Pos(0) + 1;
  size_t gen = 0;
  size_t accepted = 0, rejected = 0;

  // Streams `token`; returns false if generation should stop.
  const auto emit = [&](int token) -> bool {
    timing_info.NotifyGenerated(1);
    const bool ok = runtime_config.StreamToken(qbatch.QueryIdx(0), stream_pos,
                                               token, 0.0f);
    ++stream_pos;
    ++gen;
    return ok && !config.IsEOS(token) && gen < max_gen_steps;
  };

  // Bootstrap: one normal decode step on the last prompt token, then a first
  // draft from (streams, sampled token) at the same position.
  Transformer(config, runtime_config, weights, activations, qbatch, env);
  ComputeLogits(config, weights, activations, env);
  int pending = Top1OfSoftmax(activations.logits.RowSpan(0)).token;
  bool more = emit(pending);
  int draft = -1;
  if (more) {
    DeepSeekMTPStep(1, &pending, /*compute_logits=*/true, weights, activations,
                    qbatch, env);
    draft = Top1OfSoftmax(activations.logits.RowSpan(0)).token;
  }
  qbatch.MutablePos(0) += 1;

  while (more) {
    // Verify step: [pending at pos, draft at pos + 1] in one 2-token pass.
    const size_t pos = qbatch.Pos(0);
    activations.SetBatchSize(2);
    activations.token_ids.resize(2);
    activations.token_ids[0] = pending;
    activations.token_ids[1] = draft;
    EmbedMMToken(pending, 0, pos, /*pos_in_prompt=*/0, config, weights,
                 activations.x, env.ctx, /*image_tokens=*/nullptr,
                 /*image_token_position=*/0);
    EmbedMMToken(draft, 1, pos + 1, /*pos_in_prompt=*/0, config, weights,
                 activations.x, env.ctx, /*image_tokens=*/nullptr,
                 /*image_token_position=*/0);
    DeepSeekMaybeInitHCStreams(activations, env);
    activations.ds_snapshot_after = 0;  // boundary between committed & draft
    for (size_t layer_idx = 0; layer_idx < weights.c_layers.size();
         ++layer_idx) {
      TransformerLayer(/*num_tokens=*/2, layer_idx,
                       *weights.GetLayer(layer_idx), activations, qbatch, env);
    }
    activations.ds_snapshot_after = -1;
    DeepSeekMaybeFinalizeHCStreams(weights, activations, env);
    ComputeLogits(config, weights, activations, env);
    const int true1 = Top1OfSoftmax(activations.logits.RowSpan(0)).token;

    if (true1 == draft) {
      ++accepted;
      const int true2 = Top1OfSoftmax(activations.logits.RowSpan(1)).token;
      if (getenv("GCPP_SPEC_DEBUG")) {
        fprintf(stderr, "[spec] pos=%zu ACCEPT pending=%d draft=%d true2=%d\n",
                pos, pending, draft, true2);
      }
      more = emit(true1);
      if (more) more = emit(true2);
      if (more) {
        // MTP catch-up on both committed rows; row 1 yields the next draft.
        const int next2[2] = {true1, true2};
        DeepSeekMTPStep(2, next2, /*compute_logits=*/true, weights,
                        activations, qbatch, env);
        draft = Top1OfSoftmax(activations.logits.RowSpan(1)).token;
        pending = true2;
      }
      qbatch.MutablePos(0) += 2;
    } else {
      ++rejected;
      if (getenv("GCPP_SPEC_DEBUG")) {
        fprintf(stderr, "[spec] pos=%zu REJECT pending=%d draft=%d true1=%d\n",
                pos, pending, draft, true1);
      }
      more = emit(true1);
      // Roll back the draft row's compressor state to the boundary snapshot.
      KVCache* cache = qbatch.KV(0).cache;
      if (cache != nullptr && cache->ds_state.Rows() > 0) {
        hwy::CopyBytes(cache->ds_state_snapshot.Row(0), cache->ds_state.Row(0),
                       cache->ds_state.Cols() * sizeof(float));
      }
      if (more) {
        DeepSeekMTPStep(1, &true1, /*compute_logits=*/true, weights,
                        activations, qbatch, env);
        draft = Top1OfSoftmax(activations.logits.RowSpan(0)).token;
        pending = true1;
      }
      qbatch.MutablePos(0) += 1;
    }
  }
  timing_info.NotifyGenerateDone();
  const size_t drafts = accepted + rejected;
  fprintf(stderr, "MTP: accepted %zu / %zu drafts (%.1f%%)\n", accepted,
          drafts,
          drafts ? 100.0 * static_cast<double>(accepted) /
                       static_cast<double>(drafts)
                 : 0.0);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();
