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

// Smoke test for the DeepSeek V4 layer: builds a tiny synthetic model
// (V4 MLA with grouped o-proj + sink, sliding window, compressor, indexer,
// sqrtsoftplus/hash MoE with SwiGLU clamp, dynamic Sinkhorn mHC) and checks
// that prefill and decode produce finite activations.

#include <math.h>  // isfinite
#include <stddef.h>
#include <stdint.h>

#include <cstring>  // strncmp
#include <numeric>
#include <optional>
#include <vector>

#include "gtest/gtest.h"
#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#include "gemma/activations.h"
#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "gemma/kv_cache.h"
#include "gemma/weights.h"
#include "ops/matmul.h"
#include "util/mat.h"
#include "util/threading_context.h"
#include "hwy/base.h"
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "deepseek/deepseek_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "gemma/configs.h"
#include "deepseek/deepseek.h"
#include "util/test_util.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// Defined per-target in gemma.cc (not in a public header): drives the full
// prefill + decode loop, including the MTP speculative path when
// runtime_config.use_mtp is set.
void GenerateSingleT(const PromptTokens& prompt, size_t pos, size_t prefix_end,
                     const ModelConfig& config,
                     const RuntimeConfig& runtime_config,
                     const AesCtrEngine& engine, const WeightsPtrs& weights,
                     KVCache& kv_cache, MatMulEnv& env,
                     TimingInfo& timing_info);

static void FillRandom(MatPtrT<float>& mat, uint64_t seed, float scale) {
  hwy::RandomState rng(seed);
  for (size_t r = 0; r < mat.Rows(); ++r) {
    float* row = mat.Row(r);
    for (size_t c = 0; c < mat.Cols(); ++c) {
      row[c] = scale * static_cast<float>(RandomGaussian(rng));
    }
  }
}

static void AllocateAndFillRandom(MatPtr& mat, const Allocator& allocator,
                                  std::vector<MatOwner>& mat_owners,
                                  uint64_t seed, float scale = 0.25f) {
  if (mat.IsEmpty()) return;
  if (mat.GetType() == Type::kUnknown) {
    mat.SetType(Type::kF32);
  }
  mat_owners.emplace_back();
  mat_owners.back().AllocateFor(mat, allocator, MatPadding::kPacked);
  MatPtrT<float> mat_f32(mat);
  FillRandom(mat_f32, seed, scale);
}

// Overwrites a small f32 tensor with the given per-element generator.
template <class Gen>
static void SetF32(MatPtr& mat, const Gen& gen) {
  HWY_ASSERT(mat.HasPtr() && mat.GetType() == Type::kF32);
  MatPtrT<float> t(mat);
  for (size_t r = 0; r < t.Rows(); ++r) {
    for (size_t c = 0; c < t.Cols(); ++c) t.Row(r)[c] = gen(r, c);
  }
}

static ModelConfig TinyDeepSeekConfig() {
  ModelConfig config;
  config.model_family_version = 4;
  config.display_name = "DeepSeekTiny";
  config.model = Model::DEEPSEEK4_FLASH;
  config.wrapping = PromptWrapping::GEMMA_PT;
  config.weight = Type::kF32;
  config.model_dim = 64;
  config.vocab_size = 128;
  config.max_seq_len = 128;
  config.num_layers = 4;
  config.hc_mult = 4;
  config.hc_sinkhorn_iters = 20;
  config.hc_eps = 1e-6f;
  config.eos_id = 1;
  config.secondary_eos_id = 1;
  config.query_scale = QueryScaleType::SqrtKeySize;

  LayerConfig lc;
  lc.model_dim = config.model_dim;
  lc.ff_hidden_dim = 96;
  lc.heads = 2;
  lc.kv_heads = 1;
  lc.qkv_dim = 24;  // = latent: nope 16 + rope 8
  lc.optimized_gating = false;
  lc.type = LayerAttentionType::kDeepSeekMLA;
  lc.activation = ActivationType::Silu;
  lc.kv_lora_rank = 16;
  lc.q_lora_rank = 24;
  lc.rope_head_dim = 8;
  lc.v_head_dim = 24;  // V4: values are the full latent
  lc.o_lora_rank = 8;
  lc.o_groups = 2;
  lc.attention_variant = AttentionVariant::kCSA;
  lc.kv_compression_rate = 4;
  // MatMul outputs require N % 4 == 0, hence >= 4 indexer heads.
  lc.indexer_heads = 4;
  lc.indexer_head_dim = 8;
  lc.indexer_top_k = 2;
  lc.num_experts = 0;

  config.layer_configs = {config.num_layers, lc};
  // Layer 0: dense FFN, pure sliding-window attention (no compressor).
  config.layer_configs[0].SetDenseAttention();
  // Layers 1..3: MoE with a shared expert, sqrtsoftplus scoring, clamp.
  for (size_t i = 1; i < config.num_layers; ++i) {
    LayerConfig& l = config.layer_configs[i];
    l.ff_hidden_dim = 48;
    l.num_experts = 4;
    l.num_experts_per_datapoint = 2;
    l.num_shared_experts = 1;
    l.router_score = RouterScoreFunc::kSqrtSoftplus;
    l.route_scale = 1.5f;
    l.swiglu_limit = 10.0f;
    l.use_routing_bias = true;
  }
  // Layer 1: HCA (heavy pooling, no indexer).
  config.layer_configs[1].attention_variant = AttentionVariant::kHCA;
  config.layer_configs[1].kv_compression_rate = 8;
  config.layer_configs[1].indexer_heads = 0;
  config.layer_configs[1].indexer_head_dim = 0;
  config.layer_configs[1].indexer_top_k = 0;
  // Layer 3: hash routing (expert indices from the token-id table).
  config.layer_configs[3].hash_routing = true;
  config.layer_configs[3].use_routing_bias = false;

  // Small sliding window to exercise the window boundary.
  config.attention_window_sizes = FixedAttentionWindowSizes<4>(8);
  return config;
}

static bool AllFinite(const MatPtrT<float>& mat) {
  for (size_t r = 0; r < mat.Rows(); ++r) {
    const float* row = mat.Row(r);
    for (size_t c = 0; c < mat.Cols(); ++c) {
      if (!isfinite(row[c])) return false;
    }
  }
  return true;
}

void TestDeepSeekTiny() {
  ThreadingContext ctx({});
  MatMulEnv env(ctx);
  std::vector<MatOwner> mat_owners;

  const ModelConfig config = TinyDeepSeekConfig();
  WeightsPtrs weights(config);

  // Allocate and randomize all enumerated tensors, except the stacked
  // "gating_ein" alias (mutually exclusive with the split w1/w2 tensors) and
  // the optional skip_scale.
  uint64_t seed = 1;
  weights.ForEachTensor(nullptr, nullptr, [&](const TensorArgs& t) {
    const char* name = t.mat.Name();
    if (strncmp(name, "gating_ein", 10) == 0) return;
    if (strncmp(name, "skip_scale", 10) == 0) return;
    AllocateAndFillRandom(t.mat, ctx.allocator, mat_owners, ++seed);
  });

  // Hash routing table must contain valid expert indices.
  {
    LayerWeightsPtrs* layer = weights.GetLayer(3);
    const uint32_t num_experts = layer->layer_config.NumExperts();
    SetF32(layer->hash_tid2eid, [&](size_t r, size_t c) {
      return static_cast<float>((r + c) % num_experts);
    });
  }
  // Keep hc scales small so sigmoid/softmax inputs stay tame with random
  // weights.
  for (size_t i = 0; i < config.num_layers; ++i) {
    LayerWeightsPtrs* layer = weights.GetLayer(i);
    SetF32(layer->hc_att_scale, [](size_t, size_t) { return 0.1f; });
    SetF32(layer->hc_ffw_scale, [](size_t, size_t) { return 0.1f; });
  }
  SetF32(weights.hc_head_scale, [](size_t, size_t) { return 0.1f; });

  InferenceArgs inference_args;
  RuntimeConfig runtime_config;
  KVCache kv_cache(config, inference_args, ctx.allocator);
  ASSERT_EQ(kv_cache.SeqLen(), config.max_seq_len);
  ASSERT_GT(kv_cache.ds_state.Rows(), size_t{0});

  const size_t kPrefillTokens = 12;  // covers >1 window/CSA block
  const size_t kDecodeSteps = 6;     // crosses a HCA block boundary at 16
  const size_t batch_size = kPrefillTokens;

  Activations activations(runtime_config, config, batch_size,
                          kv_cache.SeqLen(), ctx, env.row_ptrs);

  std::vector<int> tokens(kPrefillTokens + kDecodeSteps + 1);
  std::iota(tokens.begin(), tokens.end(), 1);
  std::vector<PromptTokens> prompts;
  prompts.emplace_back(tokens);
  std::optional<AllQueries> all_queries;
  all_queries.emplace(prompts, hwy::Span<KVCache>(&kv_cache, 1));
  QBatch qbatch(/*start=*/0, /*max_size=*/1, *all_queries);

  // ---- Prefill-style batch of tokens.
  activations.SetBatchSize(kPrefillTokens);
  activations.token_ids.resize(kPrefillTokens);
  for (size_t i = 0; i < kPrefillTokens; ++i) {
    activations.token_ids[i] = tokens[i];
  }
  {
    MatPtrT<float> x(activations.x);
    FillRandom(x, 12345, 1.0f);
  }
  DeepSeekMaybeInitHCStreams(activations, env);
  for (size_t layer_idx = 0; layer_idx < config.num_layers; ++layer_idx) {
    DeepSeekTransformerLayer(kPrefillTokens, layer_idx,
                             *weights.GetLayer(layer_idx), activations, qbatch,
                             env);
  }
  DeepSeekMaybeFinalizeHCStreams(weights, activations, env);
  EXPECT_TRUE(AllFinite(activations.x)) << "prefill produced non-finite x";
  qbatch.MutablePos(0) += kPrefillTokens;

  // ---- Decode steps (one token at a time).
  for (size_t step = 0; step < kDecodeSteps; ++step) {
    activations.SetBatchSize(1);
    activations.token_ids.resize(1);
    activations.token_ids[0] = tokens[kPrefillTokens + step];
    {
      MatPtrT<float> x(activations.x);
      FillRandom(x, 777 + step, 1.0f);
    }
    DeepSeekMaybeInitHCStreams(activations, env);
    for (size_t layer_idx = 0; layer_idx < config.num_layers; ++layer_idx) {
      DeepSeekTransformerLayer(/*num_tokens=*/1, layer_idx,
                               *weights.GetLayer(layer_idx), activations,
                               qbatch, env);
    }
    DeepSeekMaybeFinalizeHCStreams(weights, activations, env);
    EXPECT_TRUE(AllFinite(activations.x))
        << "decode step " << step << " produced non-finite x";
    qbatch.MutablePos(0) += 1;
  }
}

// Greedily generates `max_tokens` and returns the generated (non-prompt)
// token ids, using the full production decode loop.
static std::vector<int> GreedyGenerate(const ModelConfig& config,
                                       const WeightsPtrs& weights,
                                       const std::vector<int>& prompt_vec,
                                       size_t max_tokens, bool use_mtp,
                                       ThreadingContext& ctx, MatMulEnv& env) {
  InferenceArgs inference_args;
  KVCache kv_cache(config, inference_args, ctx.allocator);
  std::vector<int> generated;
  const size_t prompt_size = prompt_vec.size();
  size_t seen = 0;
  RuntimeConfig runtime_config;
  runtime_config.max_generated_tokens = max_tokens;
  runtime_config.top_k = 1;  // greedy
  runtime_config.verbosity = 0;
  runtime_config.use_mtp = use_mtp;
  runtime_config.batch_stream_token = [&](size_t, size_t, int token, float) {
    if (++seen <= prompt_size) return true;
    generated.push_back(token);
    return true;
  };
  const AesCtrEngine engine(/*deterministic=*/true);
  TimingInfo timing_info = {.verbosity = 0};
  const PromptTokens prompt(prompt_vec);
  GenerateSingleT(prompt, /*pos=*/0, /*prefix_end=*/0, config, runtime_config,
                  engine, weights, kv_cache, env, timing_info);
  return generated;
}

// End-to-end checks of the speculative decoding driver. Every emitted token
// is the argmax of main-model logits, so drafts can only affect speed; but
// MatMul results are not bitwise batch-invariant, and on this tiny random
// model the Sinkhorn + BF16 chain amplifies that to ~0.1 at the logits, so
// argmax can legitimately flip at near-ties and full-sequence equality with
// the 1-token path is not guaranteed (state equivalence is covered by
// TestDeepSeekVerifyStepState). Guaranteed invariants checked here, over a
// sparse output head (only tokens 3/7/11 can win, or 0 when all are
// negative): the bootstrap token matches the normal path bitwise, the run
// yields exactly max_tokens (accept-path double-emission bookkeeping), and
// every token is from the reachable set. Two regimes: healthy MTP (drafts
// share the head: mixed accepts) and corrupted MTP (mostly rejects).
void TestDeepSeekMTPEquivalence() {
  ThreadingContext ctx({});
  MatMulEnv env(ctx);
  std::vector<MatOwner> mat_owners;

  ModelConfig config = TinyDeepSeekConfig();
  config.num_mtp_layers = 1;
  // Out of argmax range so generation always runs to max_tokens.
  config.eos_id = static_cast<int>(config.vocab_size);
  config.secondary_eos_id = static_cast<int>(config.vocab_size);

  WeightsPtrs weights(config);
  ASSERT_EQ(weights.mtp_layers.size(), size_t{1});
  uint64_t seed = 100;
  weights.ForEachTensor(nullptr, nullptr, [&](const TensorArgs& t) {
    const char* name = t.mat.Name();
    if (strncmp(name, "gating_ein", 10) == 0) return;
    if (strncmp(name, "skip_scale", 10) == 0) return;
    AllocateAndFillRandom(t.mat, ctx.allocator, mat_owners, ++seed);
  });
  {
    LayerWeightsPtrs* layer = weights.GetLayer(3);
    const uint32_t num_experts = layer->layer_config.NumExperts();
    SetF32(layer->hash_tid2eid, [&](size_t r, size_t c) {
      return static_cast<float>((r + c) % num_experts);
    });
  }
  for (size_t i = 0; i < config.num_layers; ++i) {
    LayerWeightsPtrs* layer = weights.GetLayer(i);
    SetF32(layer->hc_att_scale, [](size_t, size_t) { return 0.1f; });
    SetF32(layer->hc_ffw_scale, [](size_t, size_t) { return 0.1f; });
  }
  SetF32(weights.hc_head_scale, [](size_t, size_t) { return 0.1f; });
  SetF32(weights.mtp_layers[0].hc_att_scale,
         [](size_t, size_t) { return 0.1f; });
  SetF32(weights.mtp_layers[0].hc_ffw_scale,
         [](size_t, size_t) { return 0.1f; });
  SetF32(weights.mtp_hc_scale, [](size_t, size_t) { return 0.1f; });

  // Sparse output head: only tokens 3/7/11 can win, with O(1) margins; the
  // remaining rows are exactly zero, so their logits tie at 0.0 bitwise.
  SetF32(weights.lm_head, [](size_t r, size_t c) {
    if (r != 3 && r != 7 && r != 11) return 0.0f;
    return 2.0f * sinf(131.3f * static_cast<float>(r) +
                       0.71f * static_cast<float>(c));
  });

  std::vector<int> prompt_vec(12);
  std::iota(prompt_vec.begin(), prompt_vec.end(), 2);
  const size_t kMaxTokens = 40;

  const std::vector<int> ref = GreedyGenerate(
      config, weights, prompt_vec, kMaxTokens, /*use_mtp=*/false, ctx, env);
  EXPECT_EQ(ref.size(), kMaxTokens);

  for (int regime = 0; regime < 2; ++regime) {
    if (regime == 1) {
      // Corrupt the MTP input projections: drafts become unrelated to the
      // main model's output, so most verify steps reject and roll back.
      SetF32(weights.mtp_e_proj, [](size_t r, size_t c) {
        return 0.5f * cosf(17.7f * static_cast<float>(r) -
                           1.3f * static_cast<float>(c));
      });
      SetF32(weights.mtp_h_proj, [](size_t r, size_t c) {
        return 0.5f * sinf(3.9f * static_cast<float>(r) +
                           11.1f * static_cast<float>(c));
      });
    }
    const std::vector<int> spec = GreedyGenerate(
        config, weights, prompt_vec, kMaxTokens, /*use_mtp=*/true, ctx, env);
    ASSERT_EQ(spec.size(), kMaxTokens) << "regime " << regime;
    // The bootstrap step is the same 1-token code path in both runs.
    EXPECT_EQ(ref[0], spec[0]) << "regime " << regime;
    for (size_t i = 0; i < spec.size(); ++i) {
      const int t = spec[i];
      ASSERT_TRUE(t == 0 || t == 3 || t == 7 || t == 11)
          << "regime " << regime << ": unreachable token " << t << " at " << i;
    }
  }
}

// Diagnostic: a 2-row verify-style step (committed token + forced-wrong
// draft, then ds_state rollback) must leave the KV cache and compressor state
// identical to plain 1-token decoding. No MTP involvement; isolates the
// speculative main-path semantics.
void TestDeepSeekVerifyStepState() {
  ThreadingContext ctx({});
  MatMulEnv env(ctx);
  std::vector<MatOwner> mat_owners;

  const ModelConfig config = TinyDeepSeekConfig();
  WeightsPtrs weights(config);
  uint64_t seed = 500;
  weights.ForEachTensor(nullptr, nullptr, [&](const TensorArgs& t) {
    const char* name = t.mat.Name();
    if (strncmp(name, "gating_ein", 10) == 0) return;
    if (strncmp(name, "skip_scale", 10) == 0) return;
    AllocateAndFillRandom(t.mat, ctx.allocator, mat_owners, ++seed);
  });
  {
    LayerWeightsPtrs* layer = weights.GetLayer(3);
    const uint32_t num_experts = layer->layer_config.NumExperts();
    SetF32(layer->hash_tid2eid, [&](size_t r, size_t c) {
      return static_cast<float>((r + c) % num_experts);
    });
  }
  for (size_t i = 0; i < config.num_layers; ++i) {
    LayerWeightsPtrs* layer = weights.GetLayer(i);
    SetF32(layer->hc_att_scale, [](size_t, size_t) { return 0.1f; });
    SetF32(layer->hc_ffw_scale, [](size_t, size_t) { return 0.1f; });
  }
  SetF32(weights.hc_head_scale, [](size_t, size_t) { return 0.1f; });

  InferenceArgs inference_args;
  RuntimeConfig runtime_config;
  runtime_config.verbosity = 0;

  // NOTE on tolerances: MatMul row results are not bitwise batch-invariant
  // (reduction splitting depends on M), and the Sinkhorn + BF16 rounding
  // chain amplifies ulp-level differences to ~1e-2. Structural bugs (wrong
  // cache offsets, missing rollback, wrong positions) produce O(1)+ errors,
  // which these tolerances still catch.
  const float kTol = 0.1f;

  const size_t kPrompt = 12;
  const size_t kRejectSteps = 3;   // positions 12..14
  const size_t kAcceptPairs = 2;   // positions 15..18
  const size_t kTotal = kPrompt + kRejectSteps + 2 * kAcceptPairs;
  std::vector<int> tokens(kTotal);
  std::iota(tokens.begin(), tokens.end(), 2);

  const auto embed = [&](int token, size_t row, Activations& acts) {
    MatPtrT<float> emb(weights.embedder_input_embedding);
    memcpy(acts.x.Row(row), emb.Row(static_cast<size_t>(token)),
           config.model_dim * sizeof(float));
  };

  KVCache ref_kv(config, inference_args, ctx.allocator);
  KVCache spec_kv(config, inference_args, ctx.allocator);
  ZeroInit(ref_kv.kv_cache);
  ZeroInit(spec_kv.kv_cache);

  std::vector<PromptTokens> promptsA, promptsB;
  promptsA.emplace_back(tokens);
  promptsB.emplace_back(tokens);
  AllQueries aqA(promptsA, hwy::Span<KVCache>(&ref_kv, 1));
  AllQueries aqB(promptsB, hwy::Span<KVCache>(&spec_kv, 1));
  QBatch qA(0, 1, aqA), qB(0, 1, aqB);
  Activations aA(runtime_config, config, kPrompt, ref_kv.SeqLen(), ctx,
                 env.row_ptrs);
  Activations aB(runtime_config, config, kPrompt, spec_kv.SeqLen(), ctx,
                 env.row_ptrs);

  const auto layers = [&](Activations& a, QBatch& q, size_t num_tokens) {
    for (size_t l = 0; l < config.num_layers; ++l) {
      DeepSeekTransformerLayer(num_tokens, l, *weights.GetLayer(l), a, q, env);
    }
  };
  const auto step1 = [&](Activations& a, QBatch& q, int tok) {
    a.SetBatchSize(1);
    a.token_ids.assign(1, tok);
    embed(tok, 0, a);
    DeepSeekMaybeInitHCStreams(a, env);
    layers(a, q, 1);
    q.MutablePos(0) += 1;
  };
  const auto step2 = [&](Activations& a, QBatch& q, int tok0, int tok1) {
    a.SetBatchSize(2);
    a.token_ids.assign(2, tok0);
    a.token_ids[1] = tok1;
    embed(tok0, 0, a);
    embed(tok1, 1, a);
    DeepSeekMaybeInitHCStreams(a, env);
    a.ds_snapshot_after = 0;
    layers(a, q, 2);
    a.ds_snapshot_after = -1;
  };

  const auto prefill = [&](Activations& a, QBatch& q) {
    a.SetBatchSize(kPrompt);
    a.token_ids.resize(kPrompt);
    for (size_t i = 0; i < kPrompt; ++i) {
      a.token_ids[i] = tokens[i];
      embed(tokens[i], i, a);
    }
    DeepSeekMaybeInitHCStreams(a, env);
    layers(a, q, kPrompt);
    q.MutablePos(0) += kPrompt;
  };
  prefill(aA, qA);
  prefill(aB, qB);

  // Phase A (reject path): spec processes [committed, wrong draft], rolls
  // back the state, and commits only row 0.
  for (size_t s = 0; s < kRejectSteps; ++s) {
    const int tok = tokens[kPrompt + s];
    step1(aA, qA, tok);
    step2(aB, qB, tok, /*wrong draft=*/99);
    hwy::CopyBytes(spec_kv.ds_state_snapshot.Row(0), spec_kv.ds_state.Row(0),
                   spec_kv.ds_state.Cols() * sizeof(float));
    qB.MutablePos(0) += 1;
  }

  // Phase B (accept path): spec commits both rows of a 2-token pass.
  for (size_t p = 0; p < kAcceptPairs; ++p) {
    const int tok0 = tokens[kPrompt + kRejectSteps + 2 * p];
    const int tok1 = tokens[kPrompt + kRejectSteps + 2 * p + 1];
    step1(aA, qA, tok0);
    step1(aA, qA, tok1);
    step2(aB, qB, tok0, tok1);
    qB.MutablePos(0) += 2;
  }

  ASSERT_EQ(qA.Pos(0), qB.Pos(0));

  // Compressor state and all committed KV rows must match within tolerance.
  size_t state_mismatches = 0;
  for (size_t c = 0; c < ref_kv.ds_state.Cols(); ++c) {
    const float r = ref_kv.ds_state.Row(0)[c];
    const float s = spec_kv.ds_state.Row(0)[c];
    if (fabsf(r - s) > kTol * (1.0f + fabsf(r))) {
      if (++state_mismatches <= 5) {
        ADD_FAILURE() << "ds_state[" << c << "]: ref " << r << " spec " << s;
      }
    }
  }
  EXPECT_EQ(state_mismatches, size_t{0});

  size_t kv_mismatches = 0;
  for (size_t row = 0; row < kTotal; ++row) {
    const KV_t* r = ref_kv.kv_cache.Row(row);
    const KV_t* s = spec_kv.kv_cache.Row(row);
    for (size_t c = 0; c < config.KVCacheCols(); ++c) {
      const float rf = hwy::ConvertScalarTo<float>(r[c]);
      const float sf = hwy::ConvertScalarTo<float>(s[c]);
      if (fabsf(rf - sf) > kTol * (1.0f + fabsf(rf))) {
        if (++kv_mismatches <= 8) {
          ADD_FAILURE() << "kv[" << row << "," << c << "]: ref " << rf
                        << " spec " << sf;
        }
      }
    }
  }
  EXPECT_EQ(kv_mismatches, size_t{0});
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gcpp {
HWY_BEFORE_TEST(DeepSeekTest);
HWY_EXPORT_AND_TEST_P(DeepSeekTest, TestDeepSeekTiny);
HWY_EXPORT_AND_TEST_P(DeepSeekTest, TestDeepSeekVerifyStepState);
HWY_EXPORT_AND_TEST_P(DeepSeekTest, TestDeepSeekMTPEquivalence);
HWY_AFTER_TEST();

}  // namespace gcpp

#endif
