#include <algorithm>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>  // std::pair
#include <vector>

#include "nlohmann/json.hpp"
// copybara:import_next_line:gemma_cpp
#include "gemma.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/highway.h"
#include "hwy/timer.h"
// copybara:import_next_line:gemma_cpp
#include "util/app.h"
// copybara:import_next_line:gemma_cpp
#include "util/args.h"

using json = nlohmann::json;

class BenchmarkArgs : public gcpp::ArgsBase<BenchmarkArgs> {
 public:
  BenchmarkArgs(int argc, char* argv[]) { InitAndParse(argc, argv); }

  gcpp::Path goldens;
  gcpp::Path summarize_text;
  gcpp::Path cross_entropy;
  gcpp::Path trivia_qa;
  size_t max_questions;
  size_t batch_tokens;

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(goldens.path, "goldens_dir", std::string(""),
            "Directory containing golden files", 2);
    visitor(summarize_text.path, "summarize_text", std::string(""),
            "Path to text file to summarize", 2);
    visitor(cross_entropy.path, "cross_entropy", std::string(""),
            "Path to text file to compute cross entropy on", 2);
    visitor(trivia_qa.path, "trivia_qa", std::string(""),
            "Path to json file containing TriviaQA entries", 2);
    visitor(max_questions, "max_questions", (size_t)20,
            "Maximum number of questions to ask from --trivial_qa input", 2);
    visitor(batch_tokens, "batch_tokens", (size_t)0,
            "If not zero, break prompt into batches of this size and compute "
            "cross entropy on them independently.",
            2);
  }
};

void LogSpeedStats(const double time_start, size_t total_tokens) {
  const double time_end = hwy::platform::Now();
  const double time_elapsed = time_end - time_start;
  const double tok_sec = total_tokens / time_elapsed;
  std::cout << total_tokens << " tokens in " << time_elapsed << " seconds"
            << " [" << tok_sec << " tokens / sec" << "]\n";
}

std::pair<std::string, int> QueryModel(
    gcpp::Gemma& model, gcpp::InferenceArgs& args, gcpp::AppArgs& app,
    gcpp::KVCache& kv_cache, hwy::ThreadPool& inner_pool, hwy::ThreadPool& pool,
    const std::string& input) {
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

  const double time_start = hwy::platform::Now();
  auto stream_token = [&res, &total_tokens, &time_start, &app,
                       tokenizer = model.Tokenizer()](int token, float) {
    ++total_tokens;
    std::string token_text;
    HWY_ASSERT(tokenizer->Decode(std::vector<int>{token}, &token_text));
    res += token_text;
    if (app.verbosity >= 1 && total_tokens % 100 == 0) {
      LogSpeedStats(time_start, total_tokens);
    }
    return true;
  };
  if (app.verbosity >= 2) {
    std::cout << args.max_tokens << " " << args.max_generated_tokens << " "
              << args.temperature;
  }
  GenerateGemma(model, args.max_tokens, args.max_generated_tokens,
                args.temperature, prompt, /*abs_pos=*/0, kv_cache, pool,
                inner_pool, stream_token, accept_token, gen, app.verbosity);
  if (app.verbosity >= 1) {
    LogSpeedStats(time_start, total_tokens);
  }
  return {res, total_tokens};
}

std::vector<std::pair<std::string, std::string>> load_goldens(
    const std::string& path) {
  std::ifstream goldens_file(path);
  if (!goldens_file) {
    std::cout << "Could not load goldens file: " << path << "\n" << std::flush;
    return {};
  }
  std::vector<std::pair<std::string, std::string>> res;
  std::string query_separator;
  std::string query;
  std::string answer_separator;
  std::string answer;
  while (std::getline(goldens_file, query_separator) &&
         std::getline(goldens_file, query) &&
         std::getline(goldens_file, answer_separator) &&
         std::getline(goldens_file, answer)) {
    res.push_back({query, answer});
  }
  return res;
}

std::string ReadFile(const gcpp::Path& path) {
  std::ifstream text_file(path.path);
  if (!text_file) {
    std::cout << "Could not open file: " << path.path << "\n" << std::flush;
    return {};
  }
  std::stringstream buffer;
  buffer << text_file.rdbuf();
  return buffer.str();
}

int BenchmarkGoldens(gcpp::Gemma& model, gcpp::InferenceArgs& args,
                     gcpp::AppArgs& app, gcpp::KVCache& kv_cache,
                     hwy::ThreadPool& inner_pool, hwy::ThreadPool& pool,
                     const std::string& golden_path) {
  const std::vector<std::pair<std::string, std::string>> queries_answers =
      load_goldens(golden_path);
  int correct_answers = 0;
  int total_tokens = 0;
  const double time_start = hwy::platform::Now();
  for (const auto& [question, expected_answer] : queries_answers) {
    const auto [answer, token_count] =
        QueryModel(model, args, app, kv_cache, inner_pool, pool, question);
    total_tokens += token_count;
    if (answer.find(expected_answer) != std::string::npos) {
      correct_answers++;
    } else {
      std::cout << "Wrong!\n";
      std::cout << "Input: " << question << "\n";
      std::cout << "Expected: " << expected_answer << "\n";
      std::cout << "Output: " << answer << "\n\n" << std::flush;
    }
  }
  LogSpeedStats(time_start, total_tokens);

  std::cout << "Correct: " << correct_answers << " out of "
            << queries_answers.size() << "\n"
            << std::flush;
  if (correct_answers != queries_answers.size()) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int BenchmarkSummary(gcpp::Gemma& model, gcpp::InferenceArgs& args,
                     gcpp::AppArgs& app, gcpp::KVCache& kv_cache,
                     hwy::ThreadPool& inner_pool, hwy::ThreadPool& pool,
                     const gcpp::Path& text) {
  std::string prompt("Here is some text to summarize:\n");
  prompt.append(ReadFile(text));
  prompt.append("\nSummarize this text.\n");
  const double time_start = hwy::platform::Now();
  const auto [answer, token_count] =
      QueryModel(model, args, app, kv_cache, inner_pool, pool, prompt);
  std::cout << answer.substr(prompt.size()) << "\n" << std::flush;
  LogSpeedStats(time_start, token_count);
  return EXIT_SUCCESS;
}

int BenchmarkCrossEntropy(gcpp::Gemma& model, gcpp::Model model_type,
                          gcpp::InferenceArgs& args, gcpp::AppArgs& app,
                          hwy::ThreadPool& inner_pool, hwy::ThreadPool& pool,
                          const gcpp::Path& text, size_t batch_tokens) {
  std::string input = ReadFile(text);
  std::vector<int> prompt;
  HWY_ASSERT(model.Tokenizer()->Encode(input, &prompt));
  prompt.resize(std::min<size_t>(args.max_tokens, prompt.size()));
  std::cout << "Number of input tokens: " << prompt.size() << "\n";
  const double time_start = hwy::platform::Now();
  float total_entropy = 0.0f;
  size_t total_input_len = 0;
  if (batch_tokens == 0) batch_tokens = prompt.size();
  for (size_t pos = 0; pos < prompt.size(); pos += batch_tokens) {
    size_t num_tokens = std::min<size_t>(prompt.size() - pos, batch_tokens);
    std::vector<int> prompt_slice(prompt.begin() + pos,
                                  prompt.begin() + pos + num_tokens);
    auto kv_cache = CreateKVCache(model_type);
    float entropy =
        ComputeCrossEntropy(model, num_tokens, prompt_slice, kv_cache, pool,
                            inner_pool, app.verbosity);
    total_entropy += entropy;
    LogSpeedStats(time_start, pos + num_tokens);
    std::string text_slice;
    HWY_ASSERT(model.Tokenizer()->Decode(prompt_slice, &text_slice));
    total_input_len += text_slice.size();
    printf("Cross entropy per byte: %f [cumulative: %f]\n",
           entropy / text_slice.size(), total_entropy / total_input_len);
  }
  return EXIT_SUCCESS;
}

int BenchmarkTriviaQA(gcpp::Gemma& model, gcpp::InferenceArgs& args,
                      gcpp::AppArgs& app, gcpp::KVCache& kv_cache,
                      hwy::ThreadPool& inner_pool, hwy::ThreadPool& pool,
                      const gcpp::Path& json_file, size_t max_questions) {
  std::ifstream trivia_file(json_file.path);
  if (!trivia_file) {
    std::cout << "Could not load file: " << json_file.path << "\n"
              << std::flush;
    return EXIT_FAILURE;
  }
  std::string line;
  size_t correct_answers = 0;
  size_t i = 0;
  while (std::getline(trivia_file, line)) {
    json data = json::parse(line);
    const auto [answer, token_count] = QueryModel(
        model, args, app, kv_cache, inner_pool, pool, data["question"]);
    std::cout << answer << "\n";
    bool correct = false;
    for (const std::string expected : data["answer"]["aliases"]) {
      if (answer.find(expected) != std::string::npos) {
        correct = true;
        break;
      }
    }
    if (correct) {
      ++correct_answers;
      std::cout << "CORRECT\n\n";
    } else {
      std::cout << "WRONG\n\n";
    }
    if (++i >= max_questions) break;
  }
  printf("Correct answers: %zu / %zu\n", correct_answers, i);
  return EXIT_SUCCESS;
}

/* Run this in the same way as gemma, p.ex.:
 ./benchmark --tokenizer tokenizer.spm --model 2b-it --weights \
 2b-it-sfp.sbs --goldens_dir "../goldens"
*/
int main(int argc, char** argv) {
  gcpp::LoaderArgs loader(argc, argv);
  gcpp::InferenceArgs args(argc, argv);  // inference
  gcpp::AppArgs app(argc, argv);
  BenchmarkArgs benchmark_args(argc, argv);

  if (const char* error = loader.Validate()) {
    HWY_ABORT("\nInvalid loader args: %s", error);
  }
  if (const char* error = args.Validate()) {
    HWY_ABORT("\nInvalid inference args: %s", error);
  }

  hwy::ThreadPool inner_pool(0);
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

  if (!benchmark_args.goldens.path.empty()) {
    const std::string golden_path =
        benchmark_args.goldens.path + "/" + loader.model_type_str + ".txt";
    return BenchmarkGoldens(model, args, app, kv_cache, inner_pool, pool,
                            golden_path);
  } else if (!benchmark_args.summarize_text.path.empty()) {
    return BenchmarkSummary(model, args, app, kv_cache, inner_pool, pool,
                            benchmark_args.summarize_text);
  } else if (!benchmark_args.cross_entropy.path.empty()) {
    return BenchmarkCrossEntropy(model, loader.ModelType(), args, app,
                                 inner_pool, pool, benchmark_args.cross_entropy,
                                 benchmark_args.batch_tokens);
  } else if (!benchmark_args.trivia_qa.path.empty()) {
    return BenchmarkTriviaQA(model, args, app, kv_cache, inner_pool, pool,
                             benchmark_args.trivia_qa,
                             benchmark_args.max_questions);
  }
  std::cout << "No benchmark command given." << "\n" << std::flush;
  return EXIT_FAILURE;
}
