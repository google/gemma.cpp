#ifndef THIRD_PARTY_GEMMA_CPP_PALIGEMMA_PALIGEMMA_HELPER_H_
#define THIRD_PARTY_GEMMA_CPP_PALIGEMMA_PALIGEMMA_HELPER_H_

#include <memory>
#include <string>
#include "evals/benchmark_helper.h"
#include "gemma/gemma_args.h"

namespace gcpp {

class PaliGemmaHelper {
 public:
  explicit PaliGemmaHelper(GemmaEnv* env) : env_(env) {};

  void InitVit(const std::string& path);
  std::string GemmaReply(const std::string& prompt_text) const;

 private:
  std::unique_ptr<ImageTokens> image_tokens_;
  GemmaEnv* env_;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_PALIGEMMA_PALIGEMMA_HELPER_H_
