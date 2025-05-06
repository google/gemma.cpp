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

#include "gemma/common.h"
#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "ops/matmul.h"  // MatMulEnv
#include "hwy/base.h"
#include "hwy/highway.h"

namespace gcpp {

// Forward declaration - use 'struct' to match definition tag
struct KVCache;

// Struct to hold data for a single conversation thread
struct ConversationData {
  std::unique_ptr<KVCache> kv_cache;
  size_t abs_pos = 0;

  // Constructor to initialize kv_cache (requires KVCache definition or forward
  // declaration)
  ConversationData(const ModelConfig& model_config, size_t prefill_tbatch_size);
};

typedef bool (*GemmaTokenCallback)(const char* text, void* user_data);
typedef void (*GemmaLogCallback)(const char* message, void* user_data);

class GemmaContext {
 private:
  GemmaContext(const LoaderArgs& loader, const InferenceArgs& inference_args,
               const ThreadingArgs& threading_args, int max_length);

 public:
  static GemmaContext* Create(const char* tokenizer_path, const char* ignored1,
                              const char* weights_path, const char* ignored2,
                              int max_length);

  // Returns length of generated text, or -1 on error
  int Generate(const char* prompt_string, char* output, int max_length,
               GemmaTokenCallback callback, void* user_data);
  // Returns length of generated text, or -1 on error
  int GenerateMultimodal(const char* prompt_string, const void* image_data,
                         int image_width, int image_height, char* output,
                         int max_length, GemmaTokenCallback callback,
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

  // Reset the currently active conversation
  void ResetConversation() {
    if (active_conversation) {
      LogDebug("Resetting active conversation");
      active_conversation->abs_pos = 0;
      // Replace the cache within the current ConversationData object
      active_conversation->kv_cache = std::make_unique<KVCache>(KVCache::Create(
          model.GetModelConfig(), inference_args.prefill_tbatch_size));
      LogDebug("Active conversation reset");
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
        model.GetModelConfig(), inference_args.prefill_tbatch_size);
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
    return true;
  }

  // Check if a named conversation exists
  bool HasConversation(const char* conversation_name) {
    std::string name(conversation_name);
    return conversation_cache.count(name);
  }

 private:
  // Internal implementation shared by Generate and GenerateMultimodal
  int GenerateInternal(const char* prompt_string,
                       const void* image_data,  // Null for text-only generation
                       int image_width,   // Added dimension (0 if no image)
                       int image_height,  // Added dimension (0 if no image)
                       char* output, int max_length,
                       GemmaTokenCallback callback, void* user_data);

  // Pointer to the currently active conversation's data
  std::shared_ptr<ConversationData> active_conversation;

  // Cache of all named conversations
  std::unordered_map<std::string, std::shared_ptr<ConversationData>>
      conversation_cache;

  // Buffers (potentially could be moved into ConversationData if needed
  // per-conversation)
  std::string prompt_buffer;
  std::string result_buffer;
  std::vector<int> token_buffer;

  // Cached args (remain global for the context)
  InferenceArgs inference_args;
  ThreadingArgs threading_args;
  MatMulEnv matmul_env;

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
    if (s_log_callback) {
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
