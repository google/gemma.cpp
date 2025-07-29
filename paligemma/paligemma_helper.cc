#include "paligemma/paligemma_helper.h"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include "compression/types.h"
#include "evals/benchmark_helper.h"
#include "gemma/configs.h"
#include "gemma/gemma.h"
#include "util/allocator.h"
#include "hwy/base.h"


namespace gcpp {

void PaliGemmaHelper::InitVit(const std::string& path) {
  HWY_ASSERT(env_->GetGemma() != nullptr);
  const Gemma& gemma = *(env_->GetGemma());
  const ModelConfig& config = gemma.Config();
  HWY_ASSERT(config.wrapping == PromptWrapping::PALIGEMMA);

  image_tokens_ = std::make_unique<ImageTokens>(
      "image", Extents2D(config.vit_config.seq_len, config.model_dim),
      env_->Env().ctx.allocator, MatPadding::kPacked);
  image_tokens_->AllocateAndAttachRowPtrs(env_->Env().row_ptrs);
  Image image;
  HWY_ASSERT(image.ReadPPM(path));
  const size_t image_size = config.vit_config.image_size;
  image.Resize(image_size, image_size);
  RuntimeConfig runtime_config = {.gen = &env_->MutableGen(),
                                  .verbosity = 0};
  gemma.GenerateImageTokens(runtime_config, env_->MutableKVCache().SeqLen(),
                            image, *image_tokens_, env_->MutableEnv());
}

std::string PaliGemmaHelper::GemmaReply(const std::string& prompt_text) const {
  const Gemma& model = *(env_->GetGemma());
    env_->MutableGen().seed(0x12345678);

    std::string response;
    auto stream_token = [&](int token, float) {
      std::string token_text;
      HWY_ASSERT(
          model.Tokenizer().Decode(std::vector<int>{token}, &token_text));
      response += token_text;
      return true;
    };

    std::string mutable_prompt = prompt_text;
    std::vector<int> tokens = env_->WrapAndTokenize(mutable_prompt);
    tokens.insert(tokens.begin(), image_tokens_->Rows(), 0);

    RuntimeConfig runtime_config = {.max_generated_tokens = 512,
                                    // PrefixLM sees/attends to all tokens.
                                    .prefill_tbatch_size = tokens.size(),
                                    .gen = &env_->MutableGen(),
                                    .verbosity = 0,
                                    .stream_token = stream_token,
                                    .image_tokens = image_tokens_.get()};

    const size_t prefix_end = tokens.size();
    TimingInfo timing_info = {.verbosity = 0};
    model.Generate(runtime_config, tokens, /*pos=*/0, prefix_end,
                   env_->MutableKVCache(), env_->MutableEnv(), timing_info);
    return response;
}

}  // namespace gcpp
