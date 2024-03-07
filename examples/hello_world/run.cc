#include <iostream>

// copybara:import_next_line:gemma_cpp
#include "gemma.h"
// copybara:end
// copybara:import_next_line:gemma_cpp
#include "util/args.h"
// copybara:end
#include "hwy/contrib/thread_pool/thread_pool.h"

std::vector<int> tokenize(
    std::string prompt_string,
    const sentencepiece::SentencePieceProcessor* tokenizer) {
  prompt_string = "<start_of_turn>user\n" + prompt_string +
                  "<end_of_turn>\n<start_of_turn>model\n";
  std::vector<int> tokens;
  HWY_ASSERT(tokenizer->Encode(prompt_string, &tokens).ok());
  tokens.insert(tokens.begin(), 2);  // BOS token
  return tokens;
}

int main(int argc, char** argv) {
  gcpp::LoaderArgs loader(argc, argv);
  // A rough heuristic for a reasonable number of threads given hardware
  // concurrency estimate
  size_t num_threads = static_cast<size_t>(std::clamp(
      static_cast<int>(std::thread::hardware_concurrency()) - 2, 1, 18));
  hwy::ThreadPool pool(num_threads);
  hwy::ThreadPool inner_pool(0);
  gcpp::Gemma model(loader, pool);
  std::mt19937 gen;
  std::random_device rd;
  gen.seed(rd());
  std::vector<int> tokens =
      tokenize("Write a greeting to the world.", model.Tokenizer());
  size_t ntokens = tokens.size();
  size_t pos = 0;
  auto stream_token = [&pos, &gen, &ntokens, tokenizer = model.Tokenizer()](
                          int token, float) {
    ++pos;
    if (pos < ntokens) {
      // print feedback
    } else if (token != gcpp::EOS_ID) {
      std::string token_text;
      HWY_ASSERT(tokenizer->Decode(std::vector<int>{token}, &token_text).ok());
      if (pos == ntokens + 1) {
        // first token of response
        token_text.erase(0, token_text.find_first_not_of(" \t\n\n"));
      }
      std::cout << token_text << std::flush;
    }
    return true;
  };
  GenerateGemma(
      model, /*max_tokens=*/2048, /*max_generated_tokens=*/1024,
      /*temperature=*/1.0, tokens, 0, pool, inner_pool, stream_token,
      [](int) { return true; }, gen, 0);
  std::cout << std::endl;
}
