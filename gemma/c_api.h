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

#ifndef THIRD_PARTY_GEMMA_C_API_H_
#define THIRD_PARTY_GEMMA_C_API_H_

#include "gemma/context.h"

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
                                    const char* model_type,
                                    const char* weights_path,
                                    const char* weight_type, int max_length);
GEMMA_API void GemmaDestroy(GemmaContext* ctx);
GEMMA_API int GemmaGenerate(GemmaContext* ctx, const char* prompt, char* output,
                            int max_length, GemmaTokenCallback callback,
                            void* user_data);

GEMMA_API int GemmaCountTokens(GemmaContext* ctx, const char* text);

GEMMA_API void GemmaSetLogCallback(GemmaContext* ctx, GemmaLogCallback callback,
                                   void* user_data);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_GEMMA_C_API_H_