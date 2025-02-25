#include "gemma/context.h"

namespace gcpp {

void InitializeGemmaLibrary() {
    AppArgs app;
    app.Init();
    app.max_packages = 1;
    NestedPools pools = CreatePools(app);
    Allocator::Init(pools.Topology());
}

// Initialize static members
GemmaLogCallback GemmaContext::s_log_callback = nullptr;
void* GemmaContext::s_log_user_data = nullptr;

GemmaContext::GemmaContext(const char* tokenizer_path, const char* model_type,
                           const char* weights_path, const char* weight_type,
                           const AppArgs& app_args, int max_length)
    : pools(CreatePools(app_args)) {
  LoaderArgs loader(tokenizer_path, weights_path, model_type);
  loader.weight_type_str = weight_type;

  if (const char* error = loader.Validate()) {
    HWY_ABORT("Invalid loader configuration: %s", error);
  }

  // Initialize cached args
  inference_args.Init();
  inference_args.max_generated_tokens = max_length;
  inference_args.temperature = 0.7f;
  inference_args.top_k = 1;
  inference_args.deterministic = false;

  Allocator::Init(pools.Topology());
  model = AllocateGemma(loader, pools);
  kv_cache =
      std::make_unique<KVCache>(KVCache::Create(model->GetModelConfig(), 2048));
}

int GemmaContext::Generate(const char* prompt, char* output, int max_length,
                           GemmaTokenCallback callback, void* user_data) {
  if (!model || !kv_cache || !prompt || !output || max_length <= 0) {
    return -1;
  }

  try {
    // Clear and reuse buffers
    result_buffer.clear();
    prompt_buffer.assign(prompt);
    token_buffer.clear();

    // The prompt is assumed to be already wrapped in the appropriate control
    // tokens if necessary for an instruction tuned model, so we don't use
    // WrapAndTokenize here
    HWY_ASSERT(model->Tokenizer().Encode(prompt, &token_buffer));

    // Both pre-trained and instruction-tuned require BOS as first token
    if (token_buffer.at(0) != BOS_ID) {
      token_buffer.insert(token_buffer.begin(), BOS_ID);
    }

    // Pass prompt_tokens to properly utilize KV cache for subsequent tokens
    const size_t prompt_tokens = token_buffer.size();
    size_t tokens_generated_this_turn = 0;

    auto stream_token = [this, callback, user_data, prompt_tokens,
                         &tokens_generated_this_turn](int token, float) {
      std::string token_text;
      if (model->Tokenizer().Decode(std::vector<int>{token}, &token_text)) {
        // don't re-output the prompt tokens
        if (tokens_generated_this_turn < prompt_tokens) {
          ++tokens_generated_this_turn;
          return true;
        }
        // skip the end of turn token, this way we don't have to do string
        // comparisons at the application level (is this a good idea?)
        if (token == END_OF_TURN_ID) {
          return false;
        }

        if (callback) {
          if (!callback(token_text.c_str(), user_data)) {
            return false;
          }
        }
        result_buffer.append(token_text);
        ++tokens_generated_this_turn;
        return true;
      }
      return false;
    };

    RuntimeConfig runtime_config = {.gen = &gen,
                                    .verbosity = 0,
                                    .stream_token = stream_token,
                                    .use_spinning = Tristate::kFalse};
    inference_args.max_generated_tokens = max_length;
    inference_args.CopyTo(runtime_config);

    TimingInfo timing_info = {.verbosity = 0};
    hwy::Span<const int> testspan(token_buffer.data(), token_buffer.size());

    // Pass prompt_tokens to properly utilize KV cache for subsequent tokens
    model->Generate(runtime_config, testspan, prompt_tokens, 0, *kv_cache,
                    timing_info);

    if (result_buffer.length() >= static_cast<size_t>(max_length)) {
      return -1;
    }
    strcpy(output, result_buffer.c_str());
    return static_cast<int>(result_buffer.length());
  } catch (...) {
    return -1;
  }
}

int GemmaContext::CountTokens(const char* text) {
  if (!model || !text) return -1;
  try {
    std::string text_str(text);
    std::vector<int> tokens;
    HWY_ASSERT(model->Tokenizer().Encode(text_str, &tokens));
    return static_cast<int>(tokens.size());
  } catch (...) {
    return -1;
  }
}

}  // namespace gcpp