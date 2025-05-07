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

#ifndef THIRD_PARTY_GEMMA_C_API_H_
#define THIRD_PARTY_GEMMA_C_API_H_

#include "gemma/bindings/context.h"

#ifdef _WIN32
#ifdef GEMMA_EXPORTS
#define GEMMA_API __declspec(dllexport)
#else
#define GEMMA_API __declspec(dllimport)
#endif
#else
#define GEMMA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
typedef gcpp::GemmaContext GemmaContext;
#else
typedef struct GemmaContext GemmaContext;
#endif

typedef bool (*GemmaTokenCallback)(const char* text, void* user_data);
typedef void (*GemmaLogCallback)(const char* message, void* user_data);

GEMMA_API GemmaContext* GemmaCreate(const char* tokenizer_path,
                                    const char* weights_path,
                                    int max_generated_tokens);
GEMMA_API void GemmaDestroy(GemmaContext* ctx);
GEMMA_API int GemmaGenerate(GemmaContext* ctx, const char* prompt, char* output,
                            int max_output_chars, GemmaTokenCallback callback,
                            void* user_data);
GEMMA_API int GemmaGenerateMultimodal(GemmaContext* ctx, const char* prompt,
                                      const void* image_data, int image_width,
                                      int image_height, char* output,
                                      int max_output_chars,
                                      GemmaTokenCallback callback,
                                      void* user_data);

GEMMA_API int GemmaCountTokens(GemmaContext* ctx, const char* text);

GEMMA_API void GemmaSetLogCallback(GemmaContext* ctx, GemmaLogCallback callback,
                                   void* user_data);

// Configuration functions
GEMMA_API void GemmaSetMultiturn(GemmaContext* ctx, int value);
GEMMA_API void GemmaSetTemperature(GemmaContext* ctx, float value);
GEMMA_API void GemmaSetTopK(GemmaContext* ctx, int value);
GEMMA_API void GemmaSetDeterministic(GemmaContext* ctx, int value);
GEMMA_API void GemmaResetConversation(GemmaContext* ctx);

// Conversation management functions (renamed)
GEMMA_API int GemmaCreateConversation(GemmaContext* ctx,
                                      const char* conversation_name);
GEMMA_API int GemmaSwitchConversation(GemmaContext* ctx,
                                      const char* conversation_name);
GEMMA_API int GemmaDeleteConversation(GemmaContext* ctx,
                                      const char* conversation_name);
GEMMA_API int GemmaHasConversation(GemmaContext* ctx,
                                   const char* conversation_name);
GEMMA_API const char* GemmaGetCurrentConversation(GemmaContext* ctx);
GEMMA_API void GemmaSaveConversation(GemmaContext* ctx);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_GEMMA_C_API_H_
