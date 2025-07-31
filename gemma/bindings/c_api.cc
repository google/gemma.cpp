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

#include "gemma/bindings/c_api.h"

extern "C" {

GEMMA_API GemmaContext* GemmaCreate(const char* tokenizer_path,
                                    const char* weights_path,
                                    int max_generated_tokens) {
  try {
    GemmaContext* ctx = GemmaContext::Create(tokenizer_path, weights_path,
                                             max_generated_tokens);
    return ctx;
  } catch (...) {
    return nullptr;
  }
}

GEMMA_API void GemmaDestroy(GemmaContext* ctx) {
  delete ctx;
}

GEMMA_API int GemmaGenerate(GemmaContext* ctx, const char* prompt, char* output,
                            int max_output_chars, GemmaTokenCallback callback,
                            void* user_data) {
  if (!ctx) return -1;
  return ctx->Generate(prompt, output, max_output_chars, callback, user_data);
}

GEMMA_API int GemmaGenerateMultimodal(GemmaContext* ctx, const char* prompt,
                                      const void* image_data, int image_width,
                                      int image_height, char* output,
                                      int max_output_chars,
                                      GemmaTokenCallback callback,
                                      void* user_data) {
  if (!ctx) return -1;

  return ctx->GenerateMultimodal(prompt, image_data, image_width, image_height,
                                 output, max_output_chars, callback, user_data);
}

GEMMA_API int GemmaCountTokens(GemmaContext* ctx, const char* text) {
  if (!ctx || !text) return -1;
  return ctx->CountTokens(text);
}

GEMMA_API void GemmaSetLogCallback(GemmaContext* ctx, GemmaLogCallback callback,
                                   void* user_data) {
  if (!ctx) return;
  ctx->SetLogCallback(callback, user_data);
}

// Configuration functions implementation
GEMMA_API void GemmaSetMaxGeneratedTokens(GemmaContext* ctx, int value) {
  if (!ctx) return;
  ctx->SetMaxGeneratedTokens(value);
}

GEMMA_API void GemmaSetMultiturn(GemmaContext* ctx, int value) {
  if (!ctx) return;
  ctx->SetMultiturn(value);
}

GEMMA_API void GemmaSetTemperature(GemmaContext* ctx, float value) {
  if (!ctx) return;
  ctx->SetTemperature(value);
}

GEMMA_API void GemmaSetTopK(GemmaContext* ctx, int value) {
  if (!ctx) return;
  ctx->SetTopK(value);
}

GEMMA_API void GemmaSetDeterministic(GemmaContext* ctx, int value) {
  if (!ctx) return;
  ctx->SetDeterministic(value != 0);
}

GEMMA_API void GemmaSetPrefillTbatchSize(GemmaContext* ctx, int value) {
  if (!ctx) return;
  ctx->SetPrefillTbatchSize(value);
}

GEMMA_API void GemmaResetConversation(GemmaContext* ctx) {  // Renamed function
  if (!ctx) return;
  ctx->ResetConversation();
}

GEMMA_API int GemmaCreateConversation(GemmaContext* ctx,
                                      const char* conversation_name) {
  if (!ctx || !conversation_name) return 0;
  return ctx->CreateConversation(conversation_name) ? 1 : 0;
}

GEMMA_API int GemmaSwitchConversation(GemmaContext* ctx,
                                      const char* conversation_name) {
  if (!ctx || !conversation_name) return 0;
  return ctx->SwitchConversation(conversation_name) ? 1 : 0;
}

GEMMA_API int GemmaDeleteConversation(GemmaContext* ctx,
                                      const char* conversation_name) {
  if (!ctx || !conversation_name) return 0;
  return ctx->DeleteConversation(conversation_name) ? 1 : 0;
}

GEMMA_API int GemmaHasConversation(GemmaContext* ctx,
                                   const char* conversation_name) {
  if (!ctx || !conversation_name) return 0;
  return ctx->HasConversation(conversation_name) ? 1 : 0;
}

GEMMA_API const char* GemmaGetCurrentConversation(GemmaContext* ctx) {
  if (!ctx) return nullptr;
  return ctx->GetCurrentConversation();
}

GEMMA_API void GemmaSaveConversation(GemmaContext* ctx) {
  if (!ctx) return;
  ctx->SaveConversation();
}
}
