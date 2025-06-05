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

// Placeholder for internal header, do not modify.
#include "gemma/configs.h"
#include "gemma/model_store.h"
#include "gemma/tokenizer.h"
#include "gemma/weights.h"
#include "io/blob_store.h"
#include "io/io.h"  // Path
#include "ops/matmul.h"
#include "paligemma/image.h"
#include "util/threading_context.h"
#include "hwy/aligned_allocator.h"  // Span
#include "hwy/base.h"
#include "hwy/timer.h"

#endif  // GEMMA_CC_ONCE

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

void Attention(LayerAttentionType type, size_t num_tokens,
               const QueriesPos& queries_pos,
               const QueriesPos& queries_prefix_end,
               const hwy::Divisor& div_seq_len, const size_t layer_idx,
               const LayerWeightsPtrs& layer, Activations& activations,
               const KVCaches& kv_caches) {
  if (type == LayerAttentionType::kGemma) {
    GemmaAttention(num_tokens, queries_pos, &queries_prefix_end, div_seq_len,
                   layer_idx, layer, activations, kv_caches,
                   /*flags=*/0);
  } else {
    HWY_DASSERT(type == LayerAttentionType::kGriffinRecurrentBlock);
    // KVCache conv1d_cache and rglru_cache have one row per *Griffin* layer,
    // so map `layer` to the Griffin layer index.
    const size_t griffin_layer =
        activations.weights_config.NumLayersOfTypeBefore(type, layer_idx);
    GriffinRecurrent(queries_pos, num_tokens, griffin_layer, activations,
                     &layer, kv_caches);
  }
}

static HWY_NOINLINE void TransformerLayer(
    const size_t num_tokens, const QueriesPos& queries_pos,
    const QueriesPos& queries_prefix_end, const hwy::Divisor& div_seq_len,
    const size_t layer_idx, const LayerWeightsPtrs& layer,
    Activations& activations, const KVCaches& kv_caches) {
  const LayerConfig& layer_config = layer.layer_config;

  RMSNormBatched(activations.x, layer.pre_attention_norm_scale,
                 activations.pre_att_rms_out);

  Attention(layer_config.type, num_tokens, queries_pos, queries_prefix_end,
            div_seq_len, layer_idx, layer, activations, kv_caches);

  PostNorm(layer_config.post_norm, layer.post_attention_norm_scale,
           activations.att_sums);

  ResidualConnection(activations.att_sums, activations.x, layer,
                     /*is_attention=*/true);

  RMSNormBatched(activations.x, layer.pre_ffw_norm_scale,
                 activations.pre_ffw_rms_out);

  if (layer_config.type == LayerAttentionType::kVit) {
    FFWVit(activations, layer);
  } else {
    FFWNoVit(activations, layer);
  }

  PostNorm(layer_config.post_norm, layer.post_ffw_norm_scale,
           activations.ffw_out);

  ResidualConnection(activations.ffw_out, activations.x, layer,
                     /*is_attention=*/false);
}

// Returns the scale value to use for the embedding (basically sqrt model_dim).
static float EmbeddingScaling(size_t model_dim) {
  // Round to bf16 to match Gemma's Embedder, which casts before mul.
  return hwy::ConvertScalarTo<float>(
      hwy::ConvertScalarTo<BF16>(sqrtf(static_cast<float>(model_dim))));
}

// `batch_idx` indicates which row of `x` to write to.
// `pos` is the *token*'s position, not the start of the batch, because this is
// called for batches of tokens in prefill, but batches of queries in decode.
//
// For GEMMA_VLM, image tokens are copied into -2 locations (per the Gemma 3
// spec) until we run out of image tokens. This allows for a multi-image prompt
// if -2 locations with appropriate begin/end image tokens are created by the
// calling application.
// Returns new image_token_position.
static HWY_NOINLINE size_t
EmbedMMToken(int token, size_t batch_idx, size_t pos, size_t pos_in_prompt,
             const ModelConfig& model_config, const ModelWeightsPtrs& weights,
             MatStorageT<float>& x, const ImageTokens* image_tokens = nullptr,
             size_t image_token_position = 0) {
  // Image tokens just need to be copied.
  if (model_config.wrapping == PromptWrapping::GEMMA_VLM &&
      image_tokens != nullptr && token == -2 &&
      image_token_position < image_tokens->Rows()) {
    hwy::CopyBytes(image_tokens->Row(image_token_position), x.Row(batch_idx),
                   x.Cols() * x.ElementBytes());
    return image_token_position + 1;
  }

  if (model_config.wrapping == PromptWrapping::PALIGEMMA &&
      image_tokens != nullptr && pos_in_prompt < image_tokens->Rows()) {
    hwy::CopyBytes(image_tokens->Row(pos_in_prompt), x.Row(batch_idx),
                   x.Cols() * x.ElementBytes());
    return image_token_position;
  }

  const size_t model_dim = model_config.model_dim;
  const float emb_scaling = EmbeddingScaling(model_dim);

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
    DecompressAndZeroPad(df, embedding_span, embedding_ofs, x.Row(batch_idx),
                         model_dim);
    MulByConst(emb_scaling * weights_t->Scale(), x.Row(batch_idx), model_dim);
  });

  if (model_config.absolute_pe) {
    AddAbsolutePositionalEmbeddings(x.Row(batch_idx), model_dim, pos);
  }
  return image_token_position;
}

// Prefill() and Transformer() increment positions in-place.
using QueriesMutablePos = hwy::Span<size_t>;

// Populates KV cache for batches of tokens from one query at a time.
static HWY_NOINLINE void Prefill(
    const QueriesPromptTokens& queries_prompt,
    const QueriesMutablePos& queries_pos, const QueriesPos& queries_prefix_end,
    const size_t query_idx_start, const ModelConfig& config,
    const ModelWeightsPtrs& weights, Activations& activations,
    const RuntimeConfig& runtime_config, const hwy::Divisor& div_seq_len,
    const KVCaches& kv_caches) {
  PROFILER_ZONE("Gen.Prefill");
  const size_t num_queries = queries_prompt.size();
  HWY_DASSERT(queries_pos.size() == num_queries);
  HWY_DASSERT(queries_prefix_end.size() == num_queries);
  HWY_DASSERT(kv_caches.size() == num_queries);

  // Batches are important for amortizing loading weights over multiple tokens.
  // This is possible in prefill because we know all tokens beforehand, whereas
  // decode depends on the previous output token. However, each prefill batch of
  // a query requires that preceding batches already wrote to the KV cache,
  // hence we sequentially loop over token batches. We can reduce the number of
  // iterations by increasing the batch size, but this also increases arithmetic
  // intensity, and so we are eventually compute-limited. We could devote some
  // threads to parallelizing over queries, but for simplicity we assign them
  // all to MatMul.
  const size_t max_tbatch_size = runtime_config.prefill_tbatch_size;

  // For each query. `qi` is within the batch, not the global query index.
  for (size_t qi = 0; qi < num_queries; ++qi) {
    // Single query at a time, so pass slices of the spans because
    // GemmaAttention will only access the first KV cache and position.
    QueriesPos single_query_pos(&queries_pos[qi], 1);
    QueriesPos single_query_prefix_end(&queries_prefix_end[qi], 1);
    KVCaches single_kv_cache(&kv_caches[qi], 1);

    const size_t prompt_size = queries_prompt[qi].size();
    // In autoregressive mode, we don't need to prefill the last token, so - 1.
    size_t prefill_this_query = prompt_size - 1;
    const size_t prefix_end_this_query = queries_prefix_end[qi];
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
        const size_t pos = queries_pos[qi] + ti;
        const size_t pos_in_prompt = tbatch_start + ti;
        const int token = queries_prompt[qi][pos_in_prompt];
        image_token_position = EmbedMMToken(
            token, ti, pos, pos_in_prompt, config, weights, activations.x,
            runtime_config.image_tokens, image_token_position);
      }

      // Transformer with one batch of tokens from a single query.
      for (size_t layer_idx = 0; layer_idx < config.layer_configs.size();
           ++layer_idx) {
        TransformerLayer(tbatch_size, single_query_pos, single_query_prefix_end,
                         div_seq_len, layer_idx, *weights.GetLayer(layer_idx),
                         activations, single_kv_cache);
      }

      // NOTE: we unconditionally call StreamToken, even if EOS.
      for (size_t ti = 0; ti < tbatch_size; ++ti) {
        const size_t pos = queries_pos[qi] + ti;
        const size_t pos_in_prompt = tbatch_start + ti;
        const int token = queries_prompt[qi][pos_in_prompt];
        if (pos_in_prompt < prompt_size - 1) {
          runtime_config.StreamToken(query_idx_start + qi, pos, token, 0.0f);
        } else {
          // The last token will be streamed later and we should only get here
          // if we need to attend to the last token because it is in the prefix.
          HWY_ASSERT(attend_to_last_token);
        }
      }

      queries_pos[qi] += tbatch_size;
    }  // for tbatch_start
    if (attend_to_last_token) {
      // We need to rewind the position for the last token that we only
      // attended to to make sure the prefix LM sees everything.
      // This means we duplicate work on the last prompt token in autoregressive
      // decoding. Alternatives: (1) real masking; (2) always prefill the last
      // token and only generate the next one from the already prefilled
      // activations.
      queries_pos[qi] -= 1;
    }
  }
}

// Generates one token for each query. `queries_token` is the previous token
// from each query, and `queries_pos` are their position in the sequence.
static HWY_NOINLINE void Transformer(
    const QueriesToken& queries_token, const QueriesMutablePos& queries_pos,
    const QueriesPos& queries_prefix_end, const ModelConfig& config,
    const ModelWeightsPtrs& weights, Activations& activations,
    const hwy::Divisor& div_seq_len, const KVCaches& kv_caches,
    const LayersOutputFunc& layers_output,
    const ActivationsObserverFunc& activations_observer) {
  const size_t num_queries = queries_token.size();
  HWY_DASSERT(queries_pos.size() == num_queries);
  HWY_DASSERT(queries_prefix_end.size() == num_queries);

  if (layers_output) {
    for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
      const float token_f = queries_token[query_idx];
      layers_output(query_idx, queries_pos[query_idx], "tokens", -1, &token_f,
                    1);
    }
  }

  for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
    EmbedMMToken(queries_token[query_idx], query_idx, queries_pos[query_idx],
                 /*pos_in_prompt=*/0, config, weights, activations.x);
  }

  for (size_t layer_idx = 0; layer_idx < weights.c_layers.size(); ++layer_idx) {
    TransformerLayer(/*num_tokens=*/1, queries_pos, queries_prefix_end,
                     div_seq_len, layer_idx, *weights.GetLayer(layer_idx),
                     activations, kv_caches);

    if (activations_observer) {
      activations_observer(queries_pos, layer_idx, activations);
    }
  }

  RMSNormInplaceBatched(weights.final_norm_scale, activations.x);

  if (activations_observer) {
    activations_observer(queries_pos, -1, activations);
  }
  for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
    queries_pos[query_idx] += 1;
  }
}

void RangeChecks(const ModelConfig& weights_config,
                 size_t& max_generated_tokens, const size_t prompt_size) {
  if (!weights_config.use_local_attention) {
    if (max_generated_tokens > weights_config.seq_len) {
      HWY_WARN("max_generated_tokens %zu > kSeqLen %u, truncating.",
               max_generated_tokens, weights_config.seq_len);
      max_generated_tokens = weights_config.seq_len;
    }
  }
  HWY_ASSERT(prompt_size > 0);
}

// Holds "is at end of stream" state for each query.
class TokenStreamer {
 public:
  TokenStreamer(const RuntimeConfig& runtime_config,
                const ModelConfig& model_config)
      : runtime_config_(runtime_config), model_config_(model_config) {}

  // Returns whether the query was already at, or has just reached, the end of
  // the stream: either via token == eos_id, or StreamToken returning false.
  bool operator()(size_t query_idx, size_t pos, int token, float prob) {
    if (HWY_UNLIKELY(is_eos_.Get(query_idx))) return true;

    if (!runtime_config_.StreamToken(query_idx, pos, token, prob) ||
        model_config_.IsEOS(token)) {
      is_eos_.Set(query_idx);
      return true;
    }

    return false;
  }

 private:
  const RuntimeConfig& runtime_config_;
  const ModelConfig& model_config_;
  hwy::BitSet4096<> is_eos_;
};

// Runs one decode step for all the queries in the batch. Returns true if all
// queries are at <end_of_sentence>.
static bool DecodeStepT(const ModelConfig& config,
                        const ModelWeightsPtrs& weights,
                        const RuntimeConfig& runtime_config,
                        const QueriesPromptTokens& queries_prompt,
                        const size_t query_idx_start, const KVCaches& kv_caches,
                        const QueriesPos& queries_prefix_end,
                        const hwy::Divisor div_seq_len, const size_t vocab_size,
                        const SampleFunc& sample_token,
                        Activations& activations, TokenStreamer& token_streamer,
                        std::vector<int>& gen_tokens, TimingInfo& timing_info,
                        const QueriesMutablePos& queries_mutable_pos) {
  const size_t num_queries = queries_prompt.size();
  // Decode generates one token per query and increments
  // queries_mutable_pos.
  Transformer(QueriesToken(gen_tokens.data(), num_queries), queries_mutable_pos,
              queries_prefix_end, config, weights, activations, div_seq_len,
              kv_caches, runtime_config.layers_output,
              runtime_config.activations_observer);
  // queries_pos are incremented by Transformer.

  HWY_DASSERT(num_queries == activations.x.Rows());
  bool all_queries_eos = true;
  {
    PROFILER_ZONE("Gen.EmbeddingMatmul");
    // Compute logits from last layer activations.
    CallMatMul(activations.x, weights.embedder_input_embedding,
               /*add=*/nullptr, *activations.env, activations.logits);
  }
  PROFILER_ZONE("Gen.Softcap+Sample+Stream");
  for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
    float* HWY_RESTRICT logits = activations.logits.Row(query_idx);
    MaybeLogitsSoftCap(config.final_cap, logits, vocab_size);
    const TokenAndProb tp = sample_token(logits, vocab_size);
    timing_info.NotifyGenerated();

    const bool is_eos =
        token_streamer(query_idx_start + query_idx,
                       queries_mutable_pos[query_idx], tp.token, tp.prob);
    all_queries_eos &= is_eos;
    gen_tokens[query_idx] = is_eos ? config.eos_id : tp.token;
  }
  return all_queries_eos;
}

static HWY_INLINE SampleFunc
ChooseSampleFunc(const RuntimeConfig& runtime_config) {
  // If user provided a sample_func, use it.
  if (runtime_config.sample_func) return runtime_config.sample_func;

  // Fast path for top-1 with no accept_token.
  if (runtime_config.top_k == 1 && !runtime_config.accept_token) {
    return [](float* logits, size_t vocab_size) HWY_ATTR -> TokenAndProb {
      PROFILER_ZONE("Gen.Sample Top1");
      return Top1OfSoftmax(logits, vocab_size);
    };
  }

  // General case: Softmax with top-k sampling.
  return [&runtime_config](float* logits,
                           size_t vocab_size) HWY_ATTR -> TokenAndProb {
    PROFILER_ZONE("Gen.Sample general");
    return FusedSoftmaxAndSampleTopK(
        logits, runtime_config.top_k, vocab_size, *runtime_config.gen,
        runtime_config.temperature, runtime_config.accept_token);
  };
}

// Returns the min and max number of tokens for all queries.
static size_t MaxQueryLength(const QueriesPromptTokens& queries_prompt) {
  size_t max_prompt_size = 0;
  for (size_t i = 0; i < queries_prompt.size(); ++i) {
    max_prompt_size = HWY_MAX(max_prompt_size, queries_prompt[i].size());
  }
  return max_prompt_size;
}

// Generates one continuation for each query in `queries_prompt`, which is one
// qbatch whose size is at most the `batch_size` passed to
// `activations.Allocate`.
//
// `queries_pos` stores the KV cache position for each query. In the first turn
// of a chat, pos = 0; we increment each query's position after each token.
//
// `query_idx_start` is the query_idx of the first query in the batch, so that
// `StreamFunc` gets the global query index, not relative to the batch.
//
// `kv_caches` is for the batch, size must match `queries_prompt`.
static void GenerateT(const ModelConfig& config,
                      const ModelWeightsPtrs& weights, Activations& activations,
                      const RuntimeConfig& runtime_config,
                      const QueriesPromptTokens& queries_prompt,
                      const QueriesPos& queries_pos_in,
                      const QueriesPos& queries_prefix_end,
                      const size_t query_idx_start, const KVCaches& kv_caches,
                      TimingInfo& timing_info) {
  HWY_ASSERT(queries_pos_in.size() == kv_caches.size());

  // Griffin assumes that the recurrent block cache is zero-initialized.
  for (size_t i = 0; i < kv_caches.size(); ++i) {
    if (queries_pos_in[i] == 0) {
      kv_caches[i].ZeroGriffinCache();  // No-op for non-Griffin models.
    }
  }

  // Copy so we can increment without requiring users to pass in a mutable span.
  std::vector<size_t> queries_pos_copy(queries_pos_in.cbegin(),
                                       queries_pos_in.cend());
  const QueriesMutablePos queries_mutable_pos(queries_pos_copy.data(),
                                              queries_pos_copy.size());

  // Sanity check: prompts should not be empty, nor start with EOS.
  for (size_t query_idx = 0; query_idx < queries_prompt.size(); ++query_idx) {
    const PromptTokens& prompt = queries_prompt[query_idx];
    HWY_ASSERT(prompt.size() != 0 && prompt[0] != config.eos_id);
  }

  const size_t num_queries = queries_prompt.size();
  HWY_ASSERT(num_queries <= 4096);  // TokenStreamer uses BitSet4096.
  HWY_ASSERT(num_queries <= activations.x.Rows());
  HWY_ASSERT(queries_pos_in.size() == num_queries);
  HWY_ASSERT(kv_caches.size() == num_queries);
  const hwy::Divisor div_seq_len(static_cast<uint32_t>(kv_caches[0].seq_len));
  size_t max_prompt_size = MaxQueryLength(queries_prompt);
  size_t max_generated_tokens = runtime_config.max_generated_tokens;
  RangeChecks(config, max_generated_tokens, max_prompt_size);
  const SampleFunc sample_token = ChooseSampleFunc(runtime_config);

  // Prefill stops before min_prompt_size - 1 because the last prompt
  // token is the first input token for generation.
  timing_info.prefill_start = hwy::platform::Now();
  // Note that Prefill calls activations.SetBatchSize, so we reset it below.
  Prefill(queries_prompt, queries_mutable_pos, queries_prefix_end,
          query_idx_start, config, weights, activations, runtime_config,
          div_seq_len, kv_caches);
  // Compute the number of tokens that were prefilled and notify timing_info.
  size_t prefilled_tokens = 0;
  for (size_t qi = 0; qi < num_queries; ++qi) {
    prefilled_tokens += queries_prompt[qi].size() - 1;
  }
  timing_info.NotifyPrefill(prefilled_tokens);
  // queries_pos are incremented by Prefill.
  activations.SetBatchSize(num_queries);

  // Storage for the last generated token from each query, passed to the next
  // Transformer() call.
  std::vector<int> gen_tokens(num_queries);

  // Stream the last prompt token from each query and fill gen_tokens.
  TokenStreamer token_streamer(runtime_config, config);
  for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
    size_t last_token_pos_in_prompt =
        queries_mutable_pos[query_idx] - queries_pos_in[query_idx];
    gen_tokens[query_idx] = queries_prompt[query_idx][last_token_pos_in_prompt];
    (void)token_streamer(query_idx_start + query_idx,
                         queries_mutable_pos[query_idx], gen_tokens[query_idx],
                         0.0f);
  }

  {
    const size_t vocab_size = config.vocab_size;
    timing_info.generate_start = hwy::platform::Now();
    for (size_t gen = 0; gen < max_generated_tokens; ++gen) {
      bool all_queries_eos = DecodeStepT(
          config, weights, runtime_config, queries_prompt, query_idx_start,
          kv_caches, queries_prefix_end, div_seq_len, vocab_size, sample_token,
          activations, token_streamer, gen_tokens, timing_info,
          queries_mutable_pos);
      if (all_queries_eos) break;
    }  // foreach token to generate
    timing_info.NotifyGenerateDone();
  }
}

void GenerateSingleT(const ModelConfig& config, const ModelWeightsPtrs& weights,
                     const RuntimeConfig& runtime_config,
                     const PromptTokens& prompt, size_t pos, size_t prefix_end,
                     KVCache& kv_cache, MatMulEnv* env,
                     TimingInfo& timing_info) {
  constexpr size_t kNumQueries = 1;
  const size_t qbatch_start = 0;

  const size_t max_batch_size =
      HWY_MAX(kNumQueries, runtime_config.prefill_tbatch_size);
  // TODO: move into Gemma?
  Activations activations(config, max_batch_size, env);

  const QueriesPromptTokens queries_prompt(&prompt, kNumQueries);
  QueriesPos queries_pos(&pos, kNumQueries);
  const QueriesPos queries_prefix_end(&prefix_end, kNumQueries);
  const KVCaches kv_caches{&kv_cache, kNumQueries};

  GenerateT(config, weights, activations, runtime_config, queries_prompt,
            queries_pos, queries_prefix_end, qbatch_start, kv_caches,
            timing_info);
}

void GenerateBatchT(const ModelConfig& config, const ModelWeightsPtrs& weights,
                    const RuntimeConfig& runtime_config,
                    const QueriesPromptTokens& queries_prompt,
                    const QueriesPos& queries_pos,
                    const QueriesPos& queries_prefix_end,
                    const KVCaches& kv_caches, MatMulEnv* env,
                    TimingInfo& timing_info) {
  const size_t num_queries = queries_prompt.size();
  HWY_ASSERT(queries_pos.size() == num_queries);
  HWY_ASSERT(kv_caches.size() >= num_queries);
  const size_t max_qbatch_size = runtime_config.decode_qbatch_size;
  const size_t max_batch_size =
      HWY_MAX(max_qbatch_size, runtime_config.prefill_tbatch_size);

  Activations activations(config, max_batch_size, env);

  for (size_t qbatch_start = 0; qbatch_start < num_queries;
       qbatch_start += max_qbatch_size) {
    // Generate one batch of tokens from `qbatch_size` queries.
    const size_t qbatch_size =
        HWY_MIN(num_queries - qbatch_start, max_qbatch_size);
    const QueriesPromptTokens qbatch_prompts(&queries_prompt[qbatch_start],
                                             qbatch_size);
    QueriesPos qbatch_pos(&queries_pos[qbatch_start], qbatch_size);
    const QueriesPos qbatch_prefix_end(&queries_prefix_end[qbatch_start],
                                       qbatch_size);
    const KVCaches qbatch_kv(&kv_caches[qbatch_start], qbatch_size);
    GenerateT(config, weights, activations, runtime_config, qbatch_prompts,
              qbatch_pos, qbatch_prefix_end, qbatch_start, qbatch_kv,
              timing_info);
  }
}

void GenerateImageTokensT(const ModelConfig& config,
                          const ModelWeightsPtrs& weights,
                          const RuntimeConfig& runtime_config,
                          const Image& image, ImageTokens& image_tokens,
                          MatMulEnv* env) {
  if (config.vit_config.layer_configs.empty()) {
    HWY_ABORT("Model does not support generating image tokens.");
  }
  RuntimeConfig prefill_runtime_config = runtime_config;
  ModelConfig vit_config = GetVitConfig(config);
  prefill_runtime_config.prefill_tbatch_size =
      vit_config.seq_len / (vit_config.pool_dim * vit_config.pool_dim);
  Activations prefill_activations(vit_config, vit_config.seq_len, env);
  // Weights are for the full PaliGemma model, not just the ViT part.
  PrefillVit(config, weights, prefill_runtime_config, image, image_tokens,
             prefill_activations);
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

// Internal init must run before I/O. This helper function takes care of that,
// plus calling `SetArgs`.
MatMulEnv MakeMatMulEnv(const ThreadingArgs& threading_args) {
  // Placeholder for internal init, do not modify.

  ThreadingContext::SetArgs(threading_args);
  return MatMulEnv(ThreadingContext::Get());
}

Gemma::Gemma(const LoaderArgs& loader, const InferenceArgs& inference,
             MatMulEnv& env)
    : env_(env),
      reader_(loader.weights),
      model_(reader_, loader.tokenizer, loader.wrapping),
      weights_(model_.Config()),
      chat_template_(model_.Tokenizer(), model_.Config().model) {
  weights_.ReadFromBlobs(model_, reader_, loader, inference, mat_owners_,
                         env.ctx.pools.Pool());
  reader_.CloseFile();
}

Gemma::~Gemma() = default;

void Gemma::Save(const Path& weights_path, hwy::ThreadPool& pool) const {
  BlobWriter writer;
  const std::vector<uint32_t> serialized_mat_ptrs =
      weights_.AddTensorDataToWriter(writer);
  WriteSingleFile(model_.Config(), model_.Tokenizer(), serialized_mat_ptrs,
                  writer, env_.ctx.pools.Pool(), weights_path);
}

void Gemma::Generate(const RuntimeConfig& runtime_config,
                     const PromptTokens& prompt, size_t pos, size_t prefix_end,
                     KVCache& kv_cache, TimingInfo& timing_info) const {
  env_.ctx.pools.MaybeStartSpinning(runtime_config.use_spinning);

  HWY_DYNAMIC_DISPATCH(GenerateSingleT)(model_.Config(), weights_,
                                        runtime_config, prompt, pos, prefix_end,
                                        kv_cache, &env_, timing_info);

  env_.ctx.pools.MaybeStopSpinning(runtime_config.use_spinning);
}

void Gemma::GenerateBatch(const RuntimeConfig& runtime_config,
                          const QueriesPromptTokens& queries_prompt,
                          const QueriesPos& queries_pos,
                          const QueriesPos& queries_prefix_end,
                          const KVCaches& kv_caches,
                          TimingInfo& timing_info) const {
  // If we did not get passed prefix ends (size 0), assume 0 and pass that on.
  QueriesPos mutable_queries_prefix_end = queries_prefix_end;
  std::vector<size_t> prefix_end_vec;
  if (queries_prefix_end.size() == 0) {  // hwy::Span lacks empty()
    prefix_end_vec.resize(queries_prompt.size(), 0);
    mutable_queries_prefix_end =
        QueriesPos(prefix_end_vec.data(), prefix_end_vec.size());
  }

  env_.ctx.pools.MaybeStartSpinning(runtime_config.use_spinning);

  HWY_DYNAMIC_DISPATCH(GenerateBatchT)(
      model_.Config(), weights_, runtime_config, queries_prompt, queries_pos,
      mutable_queries_prefix_end, kv_caches, &env_, timing_info);

  env_.ctx.pools.MaybeStopSpinning(runtime_config.use_spinning);
}

void Gemma::GenerateImageTokens(const RuntimeConfig& runtime_config,
                                const Image& image,
                                ImageTokens& image_tokens) const {
  env_.ctx.pools.MaybeStartSpinning(runtime_config.use_spinning);

  HWY_DYNAMIC_DISPATCH(GenerateImageTokensT)(
      model_.Config(), weights_, runtime_config, image, image_tokens, &env_);

  env_.ctx.pools.MaybeStopSpinning(runtime_config.use_spinning);
}

}  // namespace gcpp
#endif  // HWY_ONCE
