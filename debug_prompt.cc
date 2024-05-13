
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "compression/io.h"
#include "gemma/gemma.h"
#include "nlohmann/json.hpp"
#include "util/app.h"
#include "util/args.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

using json = nlohmann::json;

class PromptArgs : public gcpp::ArgsBase<PromptArgs> {
 public:
  PromptArgs(int argc, char* argv[]) { InitAndParse(argc, argv); }

  gcpp::Path layers_output;
  std::string prompt;

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(layers_output.path, "layers_output", std::string(""),
            "Path to store layers output", 2);
    visitor(prompt, "prompt", std::string(""), "Prompt to the model", 2);
  }
};

std::pair<std::string, int> QueryModel(
    gcpp::Gemma& model, gcpp::InferenceArgs& args, gcpp::AppArgs& app,
    gcpp::KVCache& kv_cache, hwy::ThreadPool& pool, const std::string& input,
    gcpp::LayersOutputT* layers_output) {
  std::vector<int> prompt;
  HWY_ASSERT(model.Tokenizer()->Encode(input, &prompt));

  // For both pre-trained and instruction-tuned models: prepend "<bos>" token
  // if needed.
  prompt.insert(prompt.begin(), 2);
  std::string res;
  size_t total_tokens = 0;
  auto accept_token = [](int) { return true; };
  std::mt19937 gen;
  gen.seed(42);

  auto stream_token = [&res, &total_tokens, &app,
                       tokenizer = model.Tokenizer()](int token, float) {
    ++total_tokens;
    std::string token_text;
    HWY_ASSERT(tokenizer->Decode(std::vector<int>{token}, &token_text));
    res += token_text;
    return true;
  };
  if (app.verbosity >= 2) {
    std::cout << args.max_tokens << " " << args.max_generated_tokens << " "
              << args.temperature;
  }
  gcpp::TimingInfo timing_info;
  GenerateGemma(model, args.max_tokens, args.max_generated_tokens,
                args.temperature, prompt, /*abs_pos=*/0, kv_cache, pool,
                stream_token, accept_token, gen, app.verbosity, timing_info,
                layers_output);
  return {res, total_tokens};
}

class OutputJsonLogger {
 public:
  json json_output;

  gcpp::LayersOutputT layers_output_log_f =
      [this](int pos, const std::string& key, const float* values,
             size_t values_len) {
        std::vector<float> v{values, values + values_len};
        json_output[std::to_string(pos)][key] = v;
      };
};

/* Run this in the same way as gemma, p.ex.:
 ./debug_prompt --tokenizer tokenizer.spm --model 2b-it --weights \
 2b-it-sfp.sbs --prompt "..." --layers_output [path]
*/
int main(int argc, char** argv) {
  gcpp::LoaderArgs loader(argc, argv);
  gcpp::InferenceArgs args(argc, argv);  // inference
  gcpp::AppArgs app(argc, argv);
  PromptArgs prompt_args(argc, argv);

  if (const char* error = loader.Validate()) {
    HWY_ABORT("\nInvalid loader args: %s", error);
  }
  if (const char* error = args.Validate()) {
    HWY_ABORT("\nInvalid inference args: %s", error);
  }
  const bool log_layers_output = !prompt_args.layers_output.path.empty();
  OutputJsonLogger json_logger;
  gcpp::LayersOutputT* layers_output =
      log_layers_output ? &json_logger.layers_output_log_f : nullptr;

  hwy::ThreadPool pool(app.num_threads);
  // For many-core, pinning threads to cores helps.
  if (app.num_threads > 10) {
    gcpp::PinThreadToCore(app.num_threads - 1);  // Main thread

    pool.Run(0, pool.NumThreads(), [](uint64_t /*task*/, size_t thread) {
      gcpp::PinThreadToCore(thread);
    });
  }

  gcpp::Gemma model(loader.tokenizer, loader.weights, loader.ModelType(), pool);
  auto kv_cache = CreateKVCache(loader.ModelType());

  const std::string& prompt = prompt_args.prompt;
  if (prompt.empty()) {
    std::cout << "Please specify --prompt" << std::endl;
    return EXIT_FAILURE;
  }
  const auto [answer, token_count] = QueryModel(
      model, args, app, kv_cache, pool, prompt, layers_output);
  std::cout << answer.substr(prompt.size()) << "\n" << std::flush;

  if (log_layers_output) {
    std::ofstream output_f(prompt_args.layers_output.path, std::ofstream::out);
    if (!output_f) {
      std::cout << "Opening file failed" << std::endl;
      return EXIT_FAILURE;
    }
    output_f << json_logger.json_output.dump();
    if (!output_f) {
      std::cout << "Writing to file failed" << std::endl;
      return EXIT_FAILURE;
    }
    output_f.close();
  }

  return EXIT_SUCCESS;
}
