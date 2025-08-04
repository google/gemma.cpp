// Copyright 2025 Google LLC
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

#include "gemma/bindings/context.h"

#include <stddef.h>
#include <string.h>  // strncpy

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "evals/benchmark_helper.h"  // InitGenerator
#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "gemma/tokenizer.h"  // WrapAndTokenize
#include "util/threading.h"
#include "util/threading_context.h"
#include "hwy/profiler.h"
#include "hwy/timer.h"

#ifdef _WIN32
#include <Windows.h>
#endif

#include "gemma/kv_cache.h"
#include "paligemma/image.h"

namespace gcpp {

// ConversationData constructor implementation
ConversationData::ConversationData(const ModelConfig& model_config,
                                   const InferenceArgs& inference_args,
                                   const Allocator& allocator)
    : kv_cache(
          std::make_unique<KVCache>(model_config, inference_args, allocator)),
      abs_pos(0) {}

// ConversationData copy constructor implementation
ConversationData::ConversationData(const ConversationData& other)
    : kv_cache(nullptr), abs_pos(other.abs_pos) {
  if (other.kv_cache) {
    kv_cache = std::make_unique<KVCache>(other.kv_cache->Copy());
  }
}

// Initialize static members
GemmaLogCallback GemmaContext::s_log_callback = nullptr;
void* GemmaContext::s_log_user_data = nullptr;

GemmaContext* GemmaContext::Create(const char* tokenizer_path,
                                   const char* weights_path,
                                   int max_generated_tokens) {
  std::stringstream ss;
  ss << "Creating GemmaContext with tokenizer_path: "
     << (tokenizer_path ? tokenizer_path : "null")
     << ", weights_path: " << (weights_path ? weights_path : "null")
     << ", max_generated_tokens: " << max_generated_tokens;
  LogDebug(ss.str().c_str());

  ThreadingArgs threading_args;
  threading_args.spin = gcpp::Tristate::kFalse;

  LoaderArgs loader(tokenizer_path, weights_path);
  LogDebug("LoaderArgs created");

  // Initialize cached args
  LogDebug("Initializing inference args");
  InferenceArgs inference_args;
  inference_args.Init();
  inference_args.max_generated_tokens = max_generated_tokens;
  inference_args.temperature = 0.7f;
  inference_args.top_k = 1;
  inference_args.deterministic = false;

  ss.str("");
  ss << "Inference args initialized with max_tokens: " << max_generated_tokens
     << ", temperature: " << inference_args.temperature
     << ", top_k: " << inference_args.top_k << ", deterministic: "
     << (inference_args.deterministic ? "true" : "false");
  LogDebug(ss.str().c_str());

  return new GemmaContext(loader, inference_args, threading_args,
                          max_generated_tokens);
}

GemmaContext::GemmaContext(const LoaderArgs& loader,
                           const InferenceArgs& inference_args,
                           const ThreadingArgs& threading_args,
                           int max_generated_tokens)
    : inference_args(inference_args),
      threading_args(threading_args),
      ctx(threading_args),
      matmul_env(ctx),
      active_conversation_name("default"),
      model(loader, inference_args, matmul_env.ctx) {
  std::stringstream ss;

  LogDebug("Creating initial ConversationData");
  // Create the initial ConversationData object using make_shared
  active_conversation = std::make_shared<ConversationData>(
      model.Config(), inference_args, ctx.allocator);

  LogDebug(
      "Storing initial ConversationData in conversation_cache[\"default\"]");
  // Store the shared_ptr in the map under the "default" key
  conversation_cache["default"] = active_conversation;

  LogDebug("GemmaContext constructor completed");
}

// Internal implementation shared by Generate and GenerateMultimodal
int GemmaContext::GenerateInternal(const char* prompt_string,
                                   const void* image_data, int image_width,
                                   int image_height, char* output,
                                   int max_output_chars,
                                   GemmaTokenCallback callback,
                                   void* user_data) {
  PROFILER_ZONE("Gen.Internal");
  size_t tokens_generated_this_turn = 0;  // differentiates prefill from reply
  size_t prompt_size = 0;
  std::stringstream ss;
  result_buffer.clear();

  InitGenerator(inference_args, gen);

  // Ensure we have an active conversation
  if (!active_conversation || !active_conversation->kv_cache) {
    LogDebug("Generate called with null active_conversation or kv_cache");
    return -1;
  }

  // callback function invoked for each generated token.
  auto stream_token = [&, callback, user_data](int token, float) {
    // Use abs_pos from the active conversation
    ++(active_conversation->abs_pos);
    const bool in_prompt = tokens_generated_this_turn < prompt_size;
    const bool first_response_token = tokens_generated_this_turn == prompt_size;
    ++tokens_generated_this_turn;
    if (in_prompt || model.Config().IsEOS(token)) {
      return true;
    }

    std::string token_text;
    HWY_ASSERT(model.Tokenizer().Decode(std::vector<int>{token}, &token_text));
    if (first_response_token) {
      token_text.erase(0, token_text.find_first_not_of(" \t\n"));
    }

    // if we have a managed callback, pass it the token text
    if (callback) {
      if (!callback(token_text.c_str(), user_data)) {
        LogDebug("Callback returned false, stopping generation");
        return false;
      }
    }

    result_buffer.append(token_text);
    return true;
  };

  // set up runtime config
  TimingInfo timing_info = {};
  RuntimeConfig runtime_config = {.gen = &gen,
                                  .stream_token = stream_token,
                                  .use_spinning = threading_args.spin};
  inference_args.CopyTo(runtime_config);
  size_t prefix_end = 0;

  const ModelConfig& model_config = model.Config();

  // generate
  std::vector<int> prompt;
  const size_t pool_dim = model_config.vit_config.pool_dim;
  ImageTokens image_tokens(
      "image_tokens",
      image_data
          ? Extents2D(model_config.vit_config.seq_len / (pool_dim * pool_dim),
                      model_config.model_dim)
          : Extents2D(0, 0),
      ctx.allocator, MatPadding::kOdd);
  if (image_data != nullptr) {
    HWY_ASSERT(model_config.wrapping == PromptWrapping::PALIGEMMA ||
               model_config.wrapping == PromptWrapping::GEMMA_VLM);

    Image image;
    image.Set(image_width, image_height, static_cast<const float*>(image_data));

    // We may need to resize the supplied image depending on whether we're using
    // PaliGemma or Gemma 3.
    const size_t image_size = model_config.vit_config.image_size;
    image.Resize(image_size, image_size);

    // Use the existing runtime_config defined earlier in the function.
    // RuntimeConfig runtime_config = { ... }; // This was already defined
    double image_tokens_start = hwy::platform::Now();
    // Pass the populated image object to GenerateImageTokens
    model.GenerateImageTokens(runtime_config,
                              active_conversation->kv_cache->SeqLen(), image,
                              image_tokens, matmul_env);
    double image_tokens_duration = hwy::platform::Now() - image_tokens_start;

    ss.str("");
    ss << "\n\n[ Timing info ] Image token generation took: ";
    ss << static_cast<int>(image_tokens_duration * 1000) << " ms\n",
        LogDebug(ss.str().c_str());

    prompt = WrapAndTokenize(
        model.Tokenizer(), model.ChatTemplate(), model_config.wrapping,
        active_conversation->abs_pos, prompt_string, image_tokens.Rows());
    runtime_config.image_tokens = &image_tokens;
    prompt_size = prompt.size();
    // The end of the prefix for prefix-LM style attention in Paligemma.
    // See Figure 2 of https://arxiv.org/abs/2407.07726.
    prefix_end = prompt_size;
  } else {
    // Text-only case (original logic)
    // Use abs_pos from the active conversation
    prompt = WrapAndTokenize(model.Tokenizer(), model.ChatTemplate(),
                             model_config.wrapping,
                             active_conversation->abs_pos, prompt_string);
    prompt_size = prompt.size();
  }

  // Check if prompt generation failed (e.g., multimodal not implemented yet)
  if (prompt.empty() && image_data != nullptr) {
    // Already logged the error, just ensure we don't proceed.
    return -1;
  }

  // Create a span from the prompt vector - Generate() expects a hwy::Span,
  // which has a different memory footprint to that of a std::vector.
  hwy::Span<const int> prompt_span(prompt.data(), prompt.size());

  // Pass the KVCache object by reference from the active conversation
  model.Generate(runtime_config, prompt_span, active_conversation->abs_pos,
                 prefix_end, *active_conversation->kv_cache, matmul_env,
                 timing_info);

  // prepare for next turn
  if (!inference_args.multiturn ||
      model_config.wrapping == PromptWrapping::PALIGEMMA) {
    // If not multiturn, or Paligemma (which handles turns differently),
    // reset the *active* conversation's position.
    active_conversation->abs_pos = 0;
    InitGenerator(inference_args, gen);
  } else {
    // Multi-turn Gemma: Rewind position in the active conversation
    // The last token was either EOS, then it should be ignored because it is
    // never part of the dialog, see Table 5 in the Gemma-2 paper:
    // https://arxiv.org/pdf/2408.00118
    // Or we have hit max_generated_tokens, then the last token will be lost.
    // (We could store it in stream_token, and then prepend to the next turn,
    // but it's not worth the complexity, as multi-turn with max_generated is
    // not a common use case.)
    // In either case, we need to rewind the active conversation's abs_pos by
    // one.
    HWY_ASSERT(active_conversation->abs_pos > 0);
    active_conversation->abs_pos--;
  }

  // Copy result buffer to output C-string (ensure null termination)
  strncpy(output, result_buffer.c_str(), max_output_chars - 1);
  output[max_output_chars - 1] = '\0';

  return static_cast<int>(strlen(output));
}

// Public Generate method (wrapper for text-only)
int GemmaContext::Generate(const char* prompt_string, char* output,
                           int max_output_chars, GemmaTokenCallback callback,
                           void* user_data) {
  // Call the internal implementation with null image_data and 0 dimensions
  return GenerateInternal(prompt_string, nullptr, 0, 0, output,
                          max_output_chars, callback, user_data);
}

// Public GenerateMultimodal method (wrapper)
int GemmaContext::GenerateMultimodal(const char* prompt_string,
                                     const void* image_data, int image_width,
                                     int image_height, char* output,
                                     int max_output_chars,
                                     GemmaTokenCallback callback,
                                     void* user_data) {
  if (image_data == nullptr) {
    LogDebug(
        "GenerateMultimodal called with null image_data. Use Generate for "
        "text-only.");
    // Or potentially call GenerateInternal with null image_data anyway?
    // Returning error seems safer.
    return -1;
  }

  return GenerateInternal(prompt_string, image_data, image_width, image_height,
                          output, max_output_chars, callback, user_data);
}

int GemmaContext::CountTokens(const char* text) {
  LogDebug("CountTokens method started");
  std::stringstream ss;
  ss << "CountTokens called with text: '" << (text ? text : "null") << "'";
  LogDebug(ss.str().c_str());

  if (!text) {
    LogDebug("CountTokens failed: Invalid parameters");
    if (!text) LogDebug("  text is null");
    return -1;
  }

  try {
    LogDebug("Creating text string");
    std::string text_str(text);

    LogDebug("Creating tokens vector");
    std::vector<int> tokens;

    LogDebug("Encoding text to tokens");
    HWY_ASSERT(model.Tokenizer().Encode(text_str, &tokens));

    ss.str("");
    ss << "Text tokenized into " << tokens.size() << " tokens";
    LogDebug(ss.str().c_str());

    LogDebug("CountTokens completed successfully");
    return static_cast<int>(tokens.size());
  } catch (...) {
    LogDebug("Unknown exception in CountTokens");
    return -1;
  }
}

// Get the name of the currently active conversation
const char* GemmaContext::GetCurrentConversation() {
  return active_conversation_name.c_str();
}

}  // namespace gcpp
