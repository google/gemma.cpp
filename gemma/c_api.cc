#ifndef GEMMA_EXPORTS
#define GEMMA_EXPORTS
#endif

#include "gemma/c_api.h"

// necessary as the C API and GemmaContext effectively wrap up and re-use the
// code for the Gemma executable
#include "util/app.h"

extern "C" {

GEMMA_API GemmaContext* GemmaCreate(const char* tokenizer_path,
                                    const char* model_type,
                                    const char* weights_path,
                                    const char* weight_type, int max_length) {
  try {
    // kludge
    gcpp::AppArgs app_args;
    app_args.Init();
    app_args.max_packages = 1;
    app_args.verbosity = 0;
    app_args.spin = gcpp::Tristate::kFalse;

    return new GemmaContext(tokenizer_path, model_type, weights_path,
                            weight_type, app_args, max_length);
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

GEMMA_API int GemmaCountTokens(GemmaContext* ctx, const char* text) {
  if (!ctx || !text) return -1;
  return static_cast<gcpp::GemmaContext*>(ctx)->CountTokens(text);
}

GEMMA_API void GemmaSetLogCallback(GemmaContext* ctx, GemmaLogCallback callback,
                                   void* user_data) {
  if (!ctx) return;
  static_cast<gcpp::GemmaContext*>(ctx)->SetLogCallback(callback, user_data);
}
}