// Copyright 2024 Google LLC
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

// Defines Gemma member functions which dynamic-dispatch into the SIMD
// implementations in gemma-inl.h.

#include "gemma/gemma.h"

#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

// Compiles this file for multiple architectures via "foreach_target.h", to
// which we pass the filename via macro 'argument'.
// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/gemma.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "gemma/attention.h"  // includes highway.h
#include "gemma/gemma-inl.h"
#include "gemma/griffin.h"  // includes highway.h
#include "gemma/vit.h"      // includes highway.h

#ifndef GEMMA_CC_ONCE
#define GEMMA_CC_ONCE

#include <math.h>  // sqrtf
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "gemma/configs.h"
#include "gemma/model_store.h"
#include "gemma/weights.h"
#include "io/blob_store.h"
#include "io/io.h"  // Path
#include "ops/matmul.h"
#include "paligemma/image.h"
#include "util/basics.h"  // PROFILER_ZONE3
#include "util/threading_context.h"
#include "hwy/aligned_allocator.h"  // Span
#include "hwy/base.h"
#include "hwy/timer.h"

// Require opt-in to debug/introspection functions to eliminate their overhead.
HWY_INLINE_VAR constexpr bool kObserver = false;

#endif  // GEMMA_CC_ONCE

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

void Attention(LayerAttentionType type, const size_t num_tokens,
               const size_t layer_idx, const LayerWeightsPtrs& layer,
               Activations& activations, QBatch& qbatch, MatMulEnv& env) {
  if (type == LayerAttentionType::kGemma) {
    GemmaAttention(num_tokens, layer_idx, layer, activations.attention, qbatch,
                   env,
                   /*flags=*/0);
  } else {
    HWY_DASSERT(type == LayerAttentionType::kGriffinRecurrentBlock);
    // KVCache conv1d_cache and rglru_cache have one row per *Griffin* layer,
    // so map `layer` to the Griffin layer index.
    const size_t griffin_layer =
        activations.attention.config.NumLayersOfTypeBefore(type, layer_idx);
    GriffinRecurrent(num_tokens, griffin_layer, &layer, activations, qbatch,
                     env);
  }
}

static HWY_NOINLINE void TransformerLayer(const size_t num_tokens,
                                          const size_t layer_idx,
                                          const LayerWeightsPtrs& layer,
                                          Activations& activations,
                                          QBatch& qbatch, MatMulEnv& env) {
  const LayerConfig& layer_config = layer.layer_config;

  RMSNormBatched(activations.x, layer.pre_attention_norm_scale,
                 activations.attention.pre_att_rms_out, env.ctx);

  Attention(layer_config.type, num_tokens, layer_idx, layer, activations,
            qbatch, env);

  PostNorm(layer_config.post_norm, layer.post_attention_norm_scale,
           activations.attention.att_sums, env.ctx);

  ResidualConnection(activations.attention.att_sums, activations.x, layer,
                     /*is_attention=*/true, env.ctx);

  RMSNormBatched(activations.x, layer.pre_ffw_norm_scale,
                 activations.pre_ffw_rms_out, env.ctx);

  if (layer_config.type == LayerAttentionType::kVit) {
    FFWVit(layer, activations, env);
  } else {
    FFWNoVit(layer, activations, env);
  }

  PostNorm(layer_config.post_norm, layer.post_ffw_norm_scale,
           activations.ffw_out, env.ctx);

  ResidualConnection(activations.ffw_out, activations.x, layer,
                     /*is_attention=*/false, env.ctx);
}

// Returns the scale value to use for the embedding (basically sqrt model_dim).
static float EmbeddingScaling(size_t model_dim) {
  // Round to bf16 to match Gemma's Embedder, which casts before mul.
  return hwy::ConvertScalarTo<float>(
      hwy::ConvertScalarTo<BF16>(sqrtf(static_cast<float>(model_dim))));
}

// `x_row` indicates which row of `x` to write to.
// `pos` is the *token*'s position for `AddAbsolutePositionalEmbeddings`, not
// the start of the batch, because this is called for batches of tokens in
// prefill, but batches of queries in decode.
//
// For GEMMA_VLM, image tokens are copied into -2 locations (per the Gemma 3
// spec) until we run out of image tokens. This allows for a multi-image prompt
// if -2 locations with appropriate begin/end image tokens are created by the
// calling application.
// Returns new image_token_position.
static HWY_NOINLINE size_t
EmbedMMToken(int token, size_t x_row, size_t pos, size_t pos_in_prompt,
             const ModelConfig& model_config, const WeightsPtrs& weights,
             MatStorageT<float>& x, ThreadingContext& ctx,
             const ImageTokens* image_tokens = nullptr,
             size_t image_token_position = 0) {
  static const auto zone =
      ctx.profiler.AddZone("Gen.Embed", hwy::ProfilerFlags::kInclusive);
  PROFILER_ZONE3(ctx.profiler, hwy::Profiler::Thread(), zone);

  // Image tokens just need to be copied.
  if (model_config.wrapping == PromptWrapping::GEMMA_VLM &&
      image_tokens != nullptr && token == -2 &&
      image_token_position < image_tokens->Rows()) {
    hwy::CopyBytes(image_tokens->Row(image_token_position), x.Row(x_row),
                   x.Cols() * x.ElementBytes());
    return image_token_position + 1;
  }

  if (model_config.wrapping == PromptWrapping::PALIGEMMA &&
      image_tokens != nullptr && pos_in_prompt < image_tokens->Rows()) {
    hwy::CopyBytes(image_tokens->Row(pos_in_prompt), x.Row(x_row),
                   x.Cols() * x.ElementBytes());
    return image_token_position;
  }

  const size_t model_dim = model_config.model_dim;
  const float emb_scaling = EmbeddingScaling(model_dim);
  const size_t worker = 0;  // Not yet parallelized.

  HWY_DASSERT(token >= 0);
  HWY_DASSERT(token < static_cast<int>(model_config.vocab_size));

  CallUpcasted(&weights.embedder_input_embedding, [&](const auto* weights_t) {
    // Using `Stride` to compute the offset works for both NUQ (because we use
    // an offset and NUQ is never padded) and padded, because non-NUQ types are
    // seekable, hence the offset can also skip any padding.
    const size_t embedding_ofs = token * weights_t->Stride();
    HWY_ASSERT(weights_t->Cols() == model_dim);
    const auto embedding_span =
        MakeSpan(weights_t->Row(0), embedding_ofs + model_dim);
    const hn::ScalableTag<float> df;
    DecompressAndZeroPad(df, embedding_span, embedding_ofs, x.Row(x_row),
                         model_dim);
    MulByConst(emb_scaling * weights_t->Scale(), x.Row(x_row), model_dim,
               ctx.profiler, worker);
  });

  if (model_config.absolute_pe) {
    AddAbsolutePositionalEmbeddings(x.Row(x_row), model_dim, pos);
  }
  return image_token_position;
}

// Populates KV cache for batches of tokens from one query at a time. This is
// called if prompts are longer than the query batch size, and also in
// prefix-LM mode (end > 0), which must see all tokens in one batch.
static HWY_NOINLINE void PrefillTBatch(const ModelConfig& config,
                                       const RuntimeConfig& runtime_config,
                                       const WeightsPtrs& weights,
                                       Activations& activations, QBatch& qbatch,
                                       MatMulEnv& env,
                                       hwy::BitSet4096<>& non_eos) {
  PROFILER_ZONE("Gen.PrefillT");

  // Batches are important for amortizing loading weights over multiple tokens.
  // This is possible in prefill because we know all tokens beforehand, whereas
  // decode depends on the previous output token. However, each prefill batch of
  // a query requires that preceding batches already wrote to the KV cache,
  // hence we sequentially loop over token batches. We can reduce the number of
  // iterations by increasing the batch size, but this also increases arithmetic
  // intensity, and so we are eventually compute-limited. TransformerLayer uses
  // all available threads, so we do not also parallelize over queries, but note
  // that PrefillQBatch uses queries as the batch dimension.
  const size_t max_tbatch_size = runtime_config.prefill_tbatch_size;

  // For each query. `qi` is within the batch, not the global query index.
  for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
    non_eos.Set(qi);

    // One query at a time, batching will be the query's prompt tokens.
    QBatch qbatch_1 = qbatch.Single(qi);

    const size_t prompt_size = qbatch_1.Prompt(0).size();
    // In autoregressive mode, we don't need to prefill the last token, so - 1.
    size_t prefill_this_query = prompt_size - 1;
    const size_t prefix_end_this_query = qbatch_1.PrefixEnd(0);
    // We can't attend beyond the prompt_size.
    HWY_ASSERT(prefix_end_this_query <= prompt_size);
    // Special case: if the prefix includes the last token, we need to prefill
    // the last token, too. However, we need to rewind this for the generation
    // of the first token. So we need to keep track of this.
    // TODO: consider implementing masking instead of this logic?
    const bool attend_to_last_token =
        (prefill_this_query < prefix_end_this_query);
    if (attend_to_last_token) {
      // The difference can be at most 1.
      prefill_this_query += 1;
      HWY_ASSERT(prefill_this_query == prefix_end_this_query);
    }
    // In prefix-LM mode, we need to look at all the tokens for the prefix in
    // one iteration through the layers, so we need a large enough batch size.
    HWY_ASSERT(prefix_end_this_query == 0 ||
               max_tbatch_size >= prefill_this_query);

    // For each batch of tokens in the query:
    for (size_t tbatch_start = 0; tbatch_start < prefill_this_query;
         tbatch_start += max_tbatch_size) {
      const size_t tbatch_size =
          HWY_MIN(max_tbatch_size, prefill_this_query - tbatch_start);
      activations.SetBatchSize(tbatch_size);

      // Fill activations.x (much faster than TransformerLayer).
      size_t image_token_position = 0;
      for (size_t ti = 0; ti < tbatch_size; ++ti) {
        const size_t pos = qbatch_1.Pos(0) + ti;
        const size_t pos_in_prompt = tbatch_start + ti;
        HWY_DASSERT(pos_in_prompt < prompt_size);
        const int token = qbatch_1.Prompt(0)[pos_in_prompt];
        image_token_position = EmbedMMToken(
            token, ti, pos, pos_in_prompt, config, weights, activations.x,
            env.ctx, runtime_config.image_tokens, image_token_position);
        // NOTE: we unconditionally call StreamToken, even if EOS.
        if (pos_in_prompt < prompt_size - 1) {
          runtime_config.StreamToken(qbatch_1.QueryIdx(0), pos, token, 0.0f);
        } else {
          // The last token will be streamed later and we should only get here
          // if we need to attend to the last token because it is in the prefix.
          HWY_ASSERT(attend_to_last_token);
        }
      }

      // Transformer with one batch of tokens from a single query. No need to
      // set `PrevToken` because we already did the embedding above.
      for (size_t layer_idx = 0; layer_idx < config.layer_configs.size();
           ++layer_idx) {
        TransformerLayer(tbatch_size, layer_idx, *weights.GetLayer(layer_idx),
                         activations, qbatch_1, env);
      }

      qbatch_1.MutablePos(0) += tbatch_size;
    }  // for tbatch_start
    if (attend_to_last_token) {
      // We need to rewind the position for the last token that we only
      // attended to to make sure the prefix LM sees everything.
      // This means we duplicate work on the last prompt token in autoregressive
      // decoding. Alternatives: (1) real masking; (2) always prefill the last
      // token and only generate the next one from the already prefilled
      // activations.
      qbatch_1.MutablePos(0) -= 1;
    }
  }
}

static void MaybeObserve(const RuntimeConfig& runtime_config,
                         Activations& activations, QBatch& qbatch,
                         int layer_idx) {
  if constexpr (kObserver) {
    if (HWY_UNLIKELY(runtime_config.activations_observer)) {
      runtime_config.activations_observer(
          QueriesPos(&qbatch.MutablePos(0), qbatch.Size()), layer_idx,
          activations);
    }
  }
}

// Embeds PrevToken (one from each query) and calls each TransformerLayer.
// Called by query-batched `PrefillQBatch` and `GenerateT`, but not the
// token-batched `PrefillTBatch`, which supports image embedding.
static HWY_NOINLINE void Transformer(const ModelConfig& config,
                                     const RuntimeConfig& runtime_config,
                                     const WeightsPtrs& weights,
                                     Activations& activations, QBatch& qbatch,
                                     MatMulEnv& env) {
  if constexpr (kObserver) {
    if (HWY_UNLIKELY(runtime_config.layers_output)) {
      for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
        const float token_f = qbatch.PrevToken(qi);
        runtime_config.layers_output(qbatch.QueryIdx(qi), qbatch.Pos(qi),
                                     "tokens", -1, &token_f, 1);
      }
    }
  }

  // TODO: parallelize?
  for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
    EmbedMMToken(qbatch.PrevToken(qi), qi, qbatch.Pos(qi),
                 /*pos_in_prompt=*/0, config, weights, activations.x, env.ctx);
  }

  for (size_t layer_idx = 0; layer_idx < weights.c_layers.size(); ++layer_idx) {
    TransformerLayer(/*num_tokens=*/1, layer_idx, *weights.GetLayer(layer_idx),
                     activations, qbatch, env);

    MaybeObserve(runtime_config, activations, qbatch, layer_idx);
  }
}

// Populates KV cache for the batch queries, one token at a time.
static HWY_NOINLINE void PrefillQBatch(const size_t max_prompt_size,
                                       const ModelConfig& config,
                                       const RuntimeConfig& runtime_config,
                                       const WeightsPtrs& weights,
                                       Activations& activations, QBatch& qbatch,
                                       MatMulEnv& env,
                                       hwy::BitSet4096<>& non_eos) {
  PROFILER_ZONE("Gen.PrefillQ");

  for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
    non_eos.Set(qi);

    // Should only be called for autoregressive (non-prefix-LM) prefill.
    HWY_DASSERT(qbatch.PrefixEnd(qi) == 0);
  }

  // In autoregressive mode, we don't prefill the last token, hence - 1.
  for (size_t pos_in_prompt = 0; pos_in_prompt < max_prompt_size - 1;
       ++pos_in_prompt) {
    for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
      int token = config.eos_id;
      if (pos_in_prompt < qbatch.Prompt(qi).size() - 1) {
        token = qbatch.Prompt(qi)[pos_in_prompt];
        // Ignore StreamToken return value because requesting to stop does not
        // make sense during prefill.
        (void)runtime_config.StreamToken(qbatch.QueryIdx(qi), pos_in_prompt,
                                         token, 0.0f);
        qbatch.MutablePos(qi) = pos_in_prompt;
      }

      qbatch.PrevToken(qi) = token;
    }

    // The input (PrevToken) is one token from each query in the batch.
    // Do not call `SampleAndStream` because it computes logits for token
    // probabilities, which are not required for the prompt tokens.
    Transformer(config, runtime_config, weights, activations, qbatch, env);
  }

  for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
    qbatch.MutablePos(qi) = qbatch.Prompt(qi).size() - 1;
  }
}

// Calls `StreamToken`, writes the token to `PrevToken` for use by subsequent
// `Transformer`, and increments `MutablePos`. Also updates `non_eos` if the
// query is at the end of its sequence.
static void StreamAndUpdateEOS(const size_t qi, int token, const float prob,
                               const ModelConfig& config,
                               const RuntimeConfig& runtime_config,
                               QBatch& qbatch, bool pos_plus_1, bool update_pos,
                               hwy::BitSet4096<>& non_eos) {
  HWY_DASSERT(non_eos.Get(qi));  // otherwise, should not be called.

  const size_t pos = qbatch.Pos(qi) + (pos_plus_1 ? 1 : 0);
  if (HWY_UNLIKELY(
          !runtime_config.StreamToken(qbatch.QueryIdx(qi), pos, token, prob))) {
    // User decided to stop: set token to primary EOS to trigger IsEOS below.
    token = config.eos_id;
    HWY_DASSERT(config.IsEOS(token));
  }

  qbatch.PrevToken(qi) = token;
  qbatch.MutablePos(qi) += update_pos ? 1 : 0;

  // Primary or secondary EOS: mark query as EOS, but still increment (for
  // multi-turn, we should still keep the prior EOS).
  if (HWY_UNLIKELY(config.IsEOS(token))) non_eos.Clear(qi);
}

// Must be called after Transformer: either after prefill, or during decode.
// Computes logits, samples and streams the token.
static void SampleAndStream(
    const ModelConfig& config, const RuntimeConfig& runtime_config,
    const WeightsPtrs& weights, const SampleFunc& sample_token,
    Activations& activations, QBatch& qbatch, bool update_pos, MatMulEnv& env,
    hwy::BitSet4096<>& non_eos, TimingInfo& timing_info) {
  HWY_DASSERT(qbatch.Size() == activations.x.Rows());

  RMSNormBatched(activations.x, weights.final_norm_scale, activations.x_bf,
                 env.ctx);

  MaybeObserve(runtime_config, activations, qbatch, -1);

  {
    static const auto zone = env.ctx.profiler.AddZone(
        "Gen.EmbeddingMatmul", hwy::ProfilerFlags::kInclusive);
    PROFILER_ZONE3(env.ctx.profiler, /*worker=*/0, zone);
    // Compute logits from last layer activations.
    CallMatMul(activations.x_bf, weights.embedder_input_embedding,
               /*add=*/nullptr, env, activations.logits);
  }
  PROFILER_ZONE("Gen.Softcap+Sample+Stream");

  MaybeLogitsSoftCapBatched(config.final_cap, activations.logits, non_eos,
                            env.ctx);

  timing_info.NotifyGenerated(non_eos.Count());

  // TODO: parallelize
  non_eos.Foreach([&](size_t qi) {
    float* HWY_RESTRICT logits = activations.logits.Row(qi);
    const TokenAndProb tp = sample_token(logits, config.vocab_size);

    // We streamed all prefill tokens, but pos is still one behind because we
    // started generation at pos = prompt.size() - 1. We want the pos argument
    // to match the number of calls to `StreamToken`, as expected by the caller.
    const bool pos_plus_1 = true;
    StreamAndUpdateEOS(qi, tp.token, tp.prob, config, runtime_config, qbatch,
                       pos_plus_1, update_pos, non_eos);
  });
}

static HWY_INLINE SampleFunc
ChooseSampleFunc(const RuntimeConfig& runtime_config, ThreadingContext& ctx) {
  // If user provided a sample_func, use it.
  if (runtime_config.sample_func) return runtime_config.sample_func;

  static const auto zone = ctx.profiler.AddZone("Gen.Sample Top1");
  const size_t worker = 0;  // TODO: parallelize

  // Fast path for top-1 with no accept_token.
  if (runtime_config.top_k == 1 && !runtime_config.accept_token) {
    return [&](float* logits, size_t vocab_size) HWY_ATTR -> TokenAndProb {
      PROFILER_ZONE3(ctx.profiler, worker, zone);
      return Top1OfSoftmax(logits, vocab_size);
    };
  }

  // General case: Softmax with top-k sampling.
  return [&](float* logits, size_t vocab_size) HWY_ATTR -> TokenAndProb {
    PROFILER_ZONE("Gen.Sample general");
    return FusedSoftmaxAndSampleTopK(
        logits, runtime_config.top_k, vocab_size, *runtime_config.gen,
        runtime_config.temperature, runtime_config.accept_token, ctx.profiler,
        worker);
  };
}

// Decode: generates one continuation token for each query in `qbatch`.
static void GenerateT(const ModelConfig& config,
                      const RuntimeConfig& runtime_config,
                      const WeightsPtrs& weights, Activations& activations,
                      QBatch& qbatch, MatMulEnv& env, TimingInfo& timing_info) {
  // Griffin assumes that the recurrent block cache is zero-initialized.
  for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
    if (qbatch.MutablePos(qi) == 0) {
      qbatch.KV(qi).ZeroGriffinCache();  // No-op for non-Griffin models.
    }
  }

  size_t max_prompt_size = 0;
  bool all_prefix_end_are_zero = true;
  size_t total_prefill_tokens = 0;  // only for throughput stats.
  const size_t seq_len = qbatch.KV(0).SeqLen();
  for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
    const PromptTokens& prompt = qbatch.Prompt(qi);
    // Sanity check: prompts should not be empty. Note that multi-turn prompts
    // start with <end_of_turn>.
    HWY_ASSERT(prompt.size() != 0);

    max_prompt_size = HWY_MAX(max_prompt_size, prompt.size());

    // Prefill stops before size - 1 because the last prompt token is the
    // first input token for generation.
    total_prefill_tokens += prompt.size() - 1;

    all_prefix_end_are_zero &= qbatch.PrefixEnd(qi) == 0;

    // We use a single divisor, so all sequence lengths must be the same.
    HWY_ASSERT(qbatch.KV(qi).SeqLen() == seq_len);
  }
  if (max_prompt_size >= seq_len) {
    HWY_ABORT("max_prompt_size = %zu, increase --seq_len to at least that.",
              max_prompt_size);
  }
  HWY_ASSERT(activations.attention.div_seq_len.GetDivisor() == seq_len);

  // Lacks a constructor to bulk-set, hence initialized by Prefill* which have
  // qi loops anyway.
  hwy::BitSet4096<> non_eos;  // indexed by qi

  timing_info.prefill_start = hwy::platform::Now();
  // Batch over the larger of prompt length, or queries.
  if ((qbatch.Size() > max_prompt_size) && all_prefix_end_are_zero) {
    activations.SetBatchSize(qbatch.Size());  // required before PrefillQBatch
    PrefillQBatch(max_prompt_size, config, runtime_config, weights, activations,
                  qbatch, env, non_eos);
  } else {
    PrefillTBatch(config, runtime_config, weights, activations, qbatch, env,
                  non_eos);
    activations.SetBatchSize(qbatch.Size());  // Restore after PrefillTBatch.
  }
  HWY_DASSERT(non_eos.Count() == qbatch.Size());
  timing_info.NotifyPrefill(total_prefill_tokens);
  // queries_pos have been incremented by Prefill.

  // Stream the last prompt token from each query, fill activations.gen_tokens.
  for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
    const size_t last_pos_in_prompt = qbatch.Pos(qi) - qbatch.InitialPos(qi);
    const bool pos_plus_1 = false;  // during prefill, pos is still correct.
    // In autoregressive mode, we have not prefilled the last token, so do
    // not advance.
    const bool update_pos = (qbatch.Pos(qi) < qbatch.PrefixEnd(qi));
    StreamAndUpdateEOS(qi, qbatch.Prompt(qi)[last_pos_in_prompt], 0.0f, config,
                       runtime_config, qbatch, pos_plus_1, update_pos, non_eos);
  }

  size_t max_gen_steps = runtime_config.max_generated_tokens;
  if (max_prompt_size + max_gen_steps > seq_len) {
    HWY_WARN("prefill %zu + max_gen_steps %zu > seq_len %zu, truncating.",
             max_prompt_size, max_gen_steps, seq_len);
    max_gen_steps = seq_len - max_prompt_size;
  }

  const SampleFunc sample_token = ChooseSampleFunc(runtime_config, env.ctx);

  {
    timing_info.generate_start = hwy::platform::Now();
    for (size_t gen = 0; gen < max_gen_steps && non_eos.Any(); ++gen) {
      Transformer(config, runtime_config, weights, activations, qbatch, env);
      SampleAndStream(config, runtime_config, weights, sample_token,
                      activations, qbatch, /*update_pos=*/true, env, non_eos,
                      timing_info);
    }
    timing_info.NotifyGenerateDone();
  }
}

void GenerateSingleT(const PromptTokens& prompt, size_t pos, size_t prefix_end,
                     const ModelConfig& config,
                     const RuntimeConfig& runtime_config,
                     const WeightsPtrs& weights, KVCache& kv_cache,
                     MatMulEnv& env, TimingInfo& timing_info) {
  Activations activations(config, runtime_config.prefill_tbatch_size,
                          kv_cache.SeqLen(), env.ctx.allocator, env.row_ptrs);

  AllQueries all_queries(prompt, pos, prefix_end,
                         hwy::Span<KVCache>(&kv_cache, 1));
  QBatch qbatch(/*start=*/0, /*max_size=*/1, all_queries);
  GenerateT(config, runtime_config, weights, activations, qbatch, env,
            timing_info);
}

// Splits the input into batches of at most `runtime_config.decode_qbatch_size`
// queries, and calls `GenerateT` on each batch.
void GenerateBatchT(const ModelConfig& config,
                    const RuntimeConfig& runtime_config,
                    const WeightsPtrs& weights, AllQueries& all_queries,
                    MatMulEnv& env, TimingInfo& timing_info) {
  const size_t max_batch_size = HWY_MAX(runtime_config.decode_qbatch_size,
                                        runtime_config.prefill_tbatch_size);
  Activations activations(config, max_batch_size,
                          all_queries[0].kv_cache.SeqLen(), env.ctx.allocator,
                          env.row_ptrs);

  for (size_t start = 0; start < all_queries.NumQueries();
       start += runtime_config.decode_qbatch_size) {
    QBatch qbatch(start, runtime_config.decode_qbatch_size, all_queries);
    // Generate a batch of one token for each of `qbatch.Size()` queries.
    GenerateT(config, runtime_config, weights, activations, qbatch, env,
              timing_info);
  }
}

void GenerateImageTokensT(const ModelConfig& config,
                          const RuntimeConfig& runtime_config, size_t seq_len,
                          const WeightsPtrs& weights, const Image& image,
                          ImageTokens& image_tokens, MatMulEnv& env) {
  if (config.vit_config.layer_configs.empty()) {
    HWY_ABORT("Model does not support generating image tokens.");
  }
  RuntimeConfig prefill_runtime_config = runtime_config;
  const ModelConfig vit_config = GetVitConfig(config);
  const size_t num_tokens = vit_config.max_seq_len;
  prefill_runtime_config.prefill_tbatch_size =
      num_tokens / (vit_config.pool_dim * vit_config.pool_dim);
  Activations prefill_activations(vit_config, num_tokens, num_tokens,
                                  env.ctx.allocator, env.row_ptrs);
  // Weights are for the full PaliGemma model, not just the ViT part.
  PrefillVit(config, weights, prefill_runtime_config, image, image_tokens,
             prefill_activations, env);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {
HWY_EXPORT(GenerateSingleT);
HWY_EXPORT(GenerateBatchT);
HWY_EXPORT(GenerateImageTokensT);

Gemma::Gemma(const LoaderArgs& loader, const InferenceArgs& inference,
             ThreadingContext& ctx)
    : reader_(loader.weights),
      model_(reader_, loader.tokenizer, loader.wrapping),
      weights_(model_.Config()),
      chat_template_(model_.Tokenizer(), model_.Config().model),
      inference_(inference) {
  // Negligible CPU time in the ctor body (except ReadFromBlobs).
  weight_read_mode_ = weights_.ReadFromBlobs(model_, reader_, loader, inference,
                                             mat_owners_, ctx);
  // Read everything into memory, or `weights_.mapped_` keeps the mapping alive.
  reader_.CloseFile();
}

Gemma::~Gemma() = default;

void Gemma::Save(const Path& weights_path, NestedPools& pools) const {
  BlobWriter writer(weights_path, pools.Pool());
  const std::vector<uint32_t> serialized_mat_ptrs =
      weights_.AddTensorDataToWriter(writer);
  WriteSingleFile(model_.Config(), model_.Tokenizer(), serialized_mat_ptrs,
                  writer);
}

void Gemma::Generate(const RuntimeConfig& runtime_config,
                     const PromptTokens& prompt, size_t pos, size_t prefix_end,
                     KVCache& kv_cache, MatMulEnv& env,
                     TimingInfo& timing_info) const {
  env.ctx.pools.MaybeStartSpinning(runtime_config.use_spinning);

  HWY_DYNAMIC_DISPATCH(GenerateSingleT)(prompt, pos, prefix_end,
                                        model_.Config(), runtime_config,
                                        weights_, kv_cache, env, timing_info);

  env.ctx.pools.MaybeStopSpinning(runtime_config.use_spinning);
}

void Gemma::GenerateBatch(const RuntimeConfig& runtime_config,
                          AllQueries& all_queries, MatMulEnv& env,
                          TimingInfo& timing_info) const {
  env.ctx.pools.MaybeStartSpinning(runtime_config.use_spinning);

  HWY_DYNAMIC_DISPATCH(GenerateBatchT)(model_.Config(), runtime_config,
                                       weights_, all_queries, env, timing_info);

  env.ctx.pools.MaybeStopSpinning(runtime_config.use_spinning);
}

void Gemma::GenerateImageTokens(const RuntimeConfig& runtime_config,
                                size_t seq_len, const Image& image,
                                ImageTokens& image_tokens,
                                MatMulEnv& env) const {
  env.ctx.pools.MaybeStartSpinning(runtime_config.use_spinning);

  HWY_DYNAMIC_DISPATCH(GenerateImageTokensT)(model_.Config(), runtime_config,
                                             seq_len, weights_, image,
                                             image_tokens, env);

  env.ctx.pools.MaybeStopSpinning(runtime_config.use_spinning);
}

}  // namespace gcpp
#endif  // HWY_ONCE
