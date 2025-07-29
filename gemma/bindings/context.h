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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_CONTEXT_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_CONTEXT_H_

#include <memory>  // For std::shared_ptr, std::make_shared
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// Logging
#ifdef _WIN32
#include <windows.h>
#else
#include <stdio.h>
#endif

#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "gemma/kv_cache.h"
#include "ops/matmul.h"  // MatMulEnv
#include "hwy/base.h"
#include "hwy/highway.h"

namespace gcpp {

// Struct to hold data for a single conversation thread
struct ConversationData {
  ConversationData(const ModelConfig& model_config,
                   const InferenceArgs& inference_args,
                   const Allocator& allocator);
  ConversationData(const ConversationData& other);

  std::unique_ptr<KVCache> kv_cache;
  size_t abs_pos = 0;
};

typedef bool (*GemmaTokenCallback)(const char* text, void* user_data);
typedef void (*GemmaLogCallback)(const char* message, void* user_data);

class GemmaContext {
 private:
  GemmaContext(const LoaderArgs& loader, const InferenceArgs& inference_args,
               const ThreadingArgs& threading_args, int max_generated_tokens);

 public:
  static GemmaContext* Create(const char* tokenizer_path,
                              const char* weights_path,
                              int max_generated_tokens);

  // Returns length of generated text, or -1 on error
  int Generate(const char* prompt_string, char* output, int max_output_chars,
               GemmaTokenCallback callback, void* user_data);
  // Returns length of generated text, or -1 on error
  int GenerateMultimodal(const char* prompt_string, const void* image_data,
                         int image_width, int image_height, char* output,
                         int max_output_chars, GemmaTokenCallback callback,
                         void* user_data);

  // Returns number of tokens in text, or -1 on error
  int CountTokens(const char* text);

  // Add new method to set logger
  static void SetLogCallback(GemmaLogCallback callback, void* user_data) {
    s_log_callback = callback;
    s_log_user_data = user_data;
  }

  // Set max generated tokens
  void SetMaxGeneratedTokens(size_t value) {
    inference_args.max_generated_tokens = value;
    LogDebug("Setting max_generated_tokens to configured value");
  }

  // Set multiturn flag (0 = disabled, 1 = enabled)
  void SetMultiturn(int value) {
    inference_args.multiturn = value;
    LogDebug("Setting multiturn to configured value");
  }

  // Set temperature for token generation
  void SetTemperature(float value) {
    inference_args.temperature = value;
    LogDebug("Setting temperature to configured value");
  }

  // Set top_k parameter for sampling
  void SetTopK(int value) {
    inference_args.top_k = value;
    LogDebug("Setting top_k to configured value");
  }

  // Set deterministic flag
  void SetDeterministic(bool value) {
    inference_args.deterministic = value;
    // Reset the random number generator for deterministic generation
    if (value) {
      gen.seed(0x87654321);
    }
    LogDebug("Setting deterministic flag to configured value");
  }

  // Set prefill_tbatch_size
  void SetPrefillTbatchSize(size_t value) {
    inference_args.prefill_tbatch_size = value;
    LogDebug("Setting prefill_tbatch_size to configured value");
  }

  void SaveConversation() {
    if (!active_conversation || active_conversation_name.empty()) {
      if (!active_conversation) {
        LogDebug("SaveConversation: No active conversation to save.");
      } else {  // active_conversation_name must be empty
        LogDebug(
            "SaveConversation: Active conversation name is empty. Cannot "
            "save.");
      }
      return;
    }
    std::string log_msg = "SaveConversation: Attempting to save '";
    log_msg += active_conversation_name;
    log_msg += "' to prewarmed_cache.";
    LogDebug(log_msg.c_str());

    // Create a deep copy of the active_conversation via copy ctor.
    auto conversation_copy =
        std::make_shared<ConversationData>(*active_conversation);

    // Store the deep copy in prewarmed_cache.
    // If a conversation with the same name already exists, it will be
    // overwritten. std::shared_ptr will handle the destruction of the old
    // object if it's being replaced.
    prewarmed_cache[active_conversation_name] = conversation_copy;

    log_msg = "SaveConversation: Successfully saved '";
    log_msg += active_conversation_name;
    log_msg += "' to prewarmed_cache.";
    LogDebug(log_msg.c_str());
  }

  // Reset the currently active conversation
  void ResetConversation() {
    if (active_conversation) {
      std::string log_prefix = "ResetConversation ('";
      log_prefix += active_conversation_name.empty() ? "[unnamed]"
                                                     : active_conversation_name;
      log_prefix += "'): ";
      LogDebug((log_prefix + "Attempting to reset.").c_str());
      // Attempt to restore from prewarmed_cache first, regardless of name.
      auto it = prewarmed_cache.find(active_conversation_name);
      if (it != prewarmed_cache.end() && it->second && it->second->kv_cache) {
        // Found in prewarmed_cache and the cached entry is valid.
        LogDebug((log_prefix + "Found in prewarmed_cache. Restoring state.")
                     .c_str());
        active_conversation->abs_pos = it->second->abs_pos;
        // Perform a deep copy of the KVCache from the prewarmed version.
        active_conversation->kv_cache =
            std::make_unique<KVCache>(it->second->kv_cache->Copy());
        LogDebug((log_prefix + "Successfully restored from prewarmed_cache.")
                     .c_str());
        return;
      }

      // If not found in prewarmed_cache or prewarmed_cache entry is invalid,
      // rewind to initial state.
      active_conversation->abs_pos = 0;
      // Replace the cache within the current ConversationData object
      active_conversation->kv_cache = std::make_unique<KVCache>(
          model.Config(), inference_args, ctx.allocator);

      LogDebug((log_prefix + "Successfully rewound to initial state.").c_str());
    } else {
      LogDebug("Cannot reset conversation: active_conversation is null");
    }
  }

  // Create a new named conversation
  bool CreateConversation(const char* conversation_name) {
    std::string name(conversation_name);
    if (conversation_cache.count(name)) {
      LogDebug("Conversation already exists");
      return false;
    }
    LogDebug("Creating new conversation");
    // Create a new ConversationData object using make_shared
    conversation_cache[name] = std::make_shared<ConversationData>(
        model.Config(), inference_args, ctx.allocator);
    return true;
  }

  // Switch to a named conversation
  bool SwitchConversation(const char* conversation_name) {
    std::string name(conversation_name);
    auto it = conversation_cache.find(name);
    if (it == conversation_cache.end()) {
      LogDebug("Conversation not found");
      return false;
    }
    LogDebug("Switching active conversation");
    active_conversation = it->second;
    active_conversation_name = conversation_name;
    return true;
  }

  // Delete a named conversation
  bool DeleteConversation(const char* conversation_name) {
    std::string name(conversation_name);
    auto it = conversation_cache.find(name);

    if (it == conversation_cache.end()) {
      LogDebug("Conversation not found for deletion");
      return false;
    }
    if (name == "default") {
      LogDebug("Cannot delete the default conversation");
      return false;
    }
    if (it->second == active_conversation) {
      LogDebug("Cannot delete the currently active conversation");
      return false;
    }

    LogDebug("Deleting conversation");
    conversation_cache.erase(it);

    auto it2 = prewarmed_cache.find(name);
    if (it2 != prewarmed_cache.end()) {
      prewarmed_cache.erase(it2);
    }

    return true;
  }

  // Check if a named conversation exists
  bool HasConversation(const char* conversation_name) {
    std::string name(conversation_name);
    return conversation_cache.count(name);
  }

  // Get the name of the currently active conversation
  const char* GetCurrentConversation();

 private:
  // Internal implementation shared by Generate and GenerateMultimodal
  int GenerateInternal(const char* prompt_string,
                       const void* image_data,  // Null for text-only generation
                       int image_width,
                       int image_height,
                       char* output, int max_output_chars,
                       GemmaTokenCallback callback, void* user_data);

  // Pointer to the currently active conversation's data
  std::shared_ptr<ConversationData> active_conversation;

  // Cache of all named conversations
  std::unordered_map<std::string, std::shared_ptr<ConversationData>>
      conversation_cache;
  std::unordered_map<std::string, std::shared_ptr<ConversationData>>
      prewarmed_cache;

  // Buffers (potentially could be moved into ConversationData if needed
  // per-conversation)
  std::string prompt_buffer;
  std::string result_buffer;
  std::vector<int> token_buffer;

  // Cached args (remain global for the context)
  InferenceArgs inference_args;
  ThreadingArgs threading_args;
  ThreadingContext ctx;
  MatMulEnv matmul_env;

  std::string active_conversation_name;

  // Model itself (don't move this, needs to be below the args above)
  Gemma model;

  // Random generator (remains global for the context)
  std::mt19937 gen;

  // Static members for logging
  static GemmaLogCallback s_log_callback;
  static void* s_log_user_data;

  // Use logging helper method to print messages into a managed callback if
  // necessary
  static void LogDebug(const char* message) {
    if (s_log_callback != nullptr) {
      s_log_callback(message, s_log_user_data);
    } else {
#ifdef _WIN32
      OutputDebugStringA(message);
#else
      printf("%s", message);
#endif
    }
  }
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_CONTEXT_H_
