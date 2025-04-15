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

#ifndef GEMMA_EXPORTS
#define GEMMA_EXPORTS
#endif

#include "gemma/c_api.h"
#include "gemma/context.h"

extern "C" {

GEMMA_API GemmaContext* GemmaCreate(const char* tokenizer_path,
                                    const char* model_type,
                                    const char* weights_path,
                                    const char* weight_type, int max_length) {
  try {
    GemmaContext* ctx = GemmaContext::Create(tokenizer_path, model_type, weights_path,
                                         weight_type, max_length);
    return ctx;
  } catch (...) {
    return nullptr;
  }
}

GEMMA_API void GemmaDestroy(GemmaContext* ctx) {
  delete static_cast<gcpp::GemmaContext*>(ctx);
}

GEMMA_API int GemmaGenerate(GemmaContext* ctx, const char* prompt, char* output,
                            int max_length, GemmaTokenCallback callback,
                            void* user_data) {
  if (!ctx) return -1;
  return static_cast<gcpp::GemmaContext*>(ctx)->Generate(
      prompt, output, max_length, callback, user_data);
}

GEMMA_API int GemmaGenerateMultimodal(GemmaContext* ctx, const char* prompt,
                                      const void* image_data, // Renamed param
                                      int image_width,        // Added dimension
                                      int image_height,       // Added dimension
                                      char* output,
                                      int max_length,
                                      GemmaTokenCallback callback,
                                      void* user_data) {
  if (!ctx) return -1;
  // Pass dimensions to the C++ method
  return static_cast<gcpp::GemmaContext*>(ctx)->GenerateMultimodal(
      prompt, image_data, image_width, image_height, output, max_length, callback, user_data);
}

GEMMA_API int GemmaCountTokens(GemmaContext* ctx, const char* text) {
  if (!ctx || !text) return -1;
  return static_cast<gcpp::GemmaContext*>(ctx)->CountTokens(text);
}

GEMMA_API void GemmaSetLogCallback(GemmaContext* ctx, GemmaLogCallback callback,
                                   void* user_data) {
  if (!ctx) return;
  static_cast<gcpp::GemmaContext*>(ctx)->SetLogCallback(callback, user_data);
}

// Configuration functions implementation
GEMMA_API void GemmaSetMultiturn(GemmaContext* ctx, int value) {
  if (!ctx) return;
  static_cast<gcpp::GemmaContext*>(ctx)->SetMultiturn(value);
}

GEMMA_API void GemmaSetTemperature(GemmaContext* ctx, float value) {
  if (!ctx) return;
  static_cast<gcpp::GemmaContext*>(ctx)->SetTemperature(value);
}

GEMMA_API void GemmaSetTopK(GemmaContext* ctx, int value) {
  if (!ctx) return;
  static_cast<gcpp::GemmaContext*>(ctx)->SetTopK(value);
}

GEMMA_API void GemmaSetDeterministic(GemmaContext* ctx, int value) {
  if (!ctx) return;
  static_cast<gcpp::GemmaContext*>(ctx)->SetDeterministic(value != 0);
}

GEMMA_API void GemmaResetConversation(GemmaContext* ctx) { // Renamed function
  if (!ctx) return;
  static_cast<gcpp::GemmaContext*>(ctx)->ResetConversation(); // Call renamed method
}

// Conversation management functions implementation (renamed)
GEMMA_API int GemmaCreateConversation(GemmaContext* ctx, const char* conversation_name) { // Renamed function and parameter
  if (!ctx || !conversation_name) return 0;
  return static_cast<gcpp::GemmaContext*>(ctx)->CreateConversation(conversation_name) ? 1 // Call renamed method
                                                                            : 0;
}

GEMMA_API int GemmaSwitchConversation(GemmaContext* ctx, const char* conversation_name) { // Renamed function and parameter
  if (!ctx || !conversation_name) return 0;
  return static_cast<gcpp::GemmaContext*>(ctx)->SwitchConversation(conversation_name) ? 1 // Call renamed method
                                                                            : 0;
}

GEMMA_API int GemmaDeleteConversation(GemmaContext* ctx, const char* conversation_name) { // Renamed function and parameter
  if (!ctx || !conversation_name) return 0;
  return static_cast<gcpp::GemmaContext*>(ctx)->DeleteConversation(conversation_name) ? 1 // Call renamed method
                                                                            : 0;
}

GEMMA_API int GemmaHasConversation(GemmaContext* ctx, const char* conversation_name) { // Renamed function and parameter
  if (!ctx || !conversation_name) return 0;
  return static_cast<gcpp::GemmaContext*>(ctx)->HasConversation(conversation_name) ? 1 // Call renamed method
                                                                         : 0;
}
}
