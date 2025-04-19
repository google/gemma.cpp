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

#include <memory>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#endif

#include "gemma/gemma.h"
#include "util/app.h"
#include "util/threading.h"

namespace gcpp {

// Initialize global state needed by the library.
// Must be called before creating any Gemma instances.
void InitializeGemmaLibrary();

typedef bool (*GemmaTokenCallback)(const char* text, void* user_data);
typedef void (*GemmaLogCallback)(const char* message, void* user_data);

class GemmaContext {
 public:
  GemmaContext(const char* tokenizer_path, const char* model_type,
               const char* weights_path, const char* weight_type,
               const AppArgs& app_args, int max_length = 2048);

  // Returns length of generated text, or -1 on error
  int Generate(const char* prompt, char* output, int max_length,
               GemmaTokenCallback callback, void* user_data);

  // Returns number of tokens in text, or -1 on error
  int CountTokens(const char* text);

  // Add new method to set logger
  static void SetLogCallback(GemmaLogCallback callback, void* user_data) {
    s_log_callback = callback;
    s_log_user_data = user_data;
  }

 private:
  NestedPools pools;
  std::unique_ptr<Gemma> model;
  std::unique_ptr<KVCache> kv_cache;
  std::string prompt_buffer;
  std::string result_buffer;
  std::vector<int> token_buffer;

  // Cached args
  InferenceArgs inference_args;
  AppArgs app_args;
  std::mt19937 gen;

  // Add static members for logging
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
#endif
    }
  }
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_CONTEXT_H_