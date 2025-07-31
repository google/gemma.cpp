#include <stdio.h>

#include <algorithm>
#include <cstdlib>  // EXIT_FAILURE
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "evals/benchmark_helper.h"
#include "evals/cross_entropy.h"
#include "gemma/gemma.h"
#include "io/io.h"  // Path
#include "util/args.h"
#include "hwy/base.h"
#include "hwy/timer.h"
#include "nlohmann/json.hpp"

namespace gcpp {

using json = nlohmann::json;

class BenchmarkArgs : public ArgsBase<BenchmarkArgs> {
 public:
  BenchmarkArgs(int argc, char* argv[]) { InitAndParse(argc, argv); }

  Path summarize_text;
  Path cross_entropy;
  Path trivia_qa;
  size_t max_questions;
  size_t batch_tokens;

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
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

int BenchmarkSummary(GemmaEnv& env, const Path& text) {
  std::string prompt("Here is some text to summarize:\n");
  prompt.append(ReadFileToString(text));
  prompt.append("\nSummarize this text.\n");
  const double time_start = hwy::platform::Now();
  QueryResult result = env.QueryModel(prompt);
  std::cout << result.response.substr(result.response_start_pos) << "\n"
            << std::flush;
  LogSpeedStats(time_start, result.tokens_generated);
  return EXIT_SUCCESS;
}

int BenchmarkCrossEntropy(GemmaEnv& env, const Path& text,
                          size_t batch_tokens) {
  const Gemma& gemma = *env.GetGemma();
  std::string input = ReadFileToString(text);
  std::vector<int> prompt = env.Tokenize(input);
  std::cout << "Number of input tokens: " << prompt.size() << "\n";
  const double time_start = hwy::platform::Now();
  float total_entropy = 0.0f;
  size_t total_input_len = 0;
  if (batch_tokens == 0) batch_tokens = prompt.size();
  for (size_t pos = 0; pos < prompt.size(); pos += batch_tokens) {
    size_t num_tokens = std::min<size_t>(prompt.size() - pos, batch_tokens);
    std::vector<int> prompt_slice(prompt.begin() + pos,
                                  prompt.begin() + pos + num_tokens);
    KVCache kv_cache(gemma.Config(), gemma.Inference(),
                     env.MutableEnv().ctx.allocator);
    float entropy =
        ComputeCrossEntropy(*env.GetGemma(), num_tokens, prompt_slice, kv_cache,
                            env.MutableEnv(), env.Verbosity());
    total_entropy += entropy;
    LogSpeedStats(time_start, pos + num_tokens);
    std::string text_slice = env.StringFromTokens(prompt_slice);
    total_input_len += text_slice.size();
    printf("Total cross entropy: %f [cumulative: %f]\n",
           entropy, total_entropy);
    printf("Cross entropy per byte: %f [cumulative: %f]\n",
           entropy / text_slice.size(), total_entropy / total_input_len);
  }
  return EXIT_SUCCESS;
}

int BenchmarkTriviaQA(GemmaEnv& env, const Path& json_file,
                      size_t max_questions) {
  std::ifstream trivia_file(json_file.path);
  if (!trivia_file) {
    HWY_ABORT("Could not load file %s\n", json_file.path.c_str());
  }
  std::string line;
  size_t correct_answers = 0;
  size_t i = 0;
  while (std::getline(trivia_file, line)) {
    json data = json::parse(line);
    std::string q(data["question"]);
    QueryResult result = env.QueryModel(q);
    std::cout << result.response << "\n";
    bool correct = false;
    for (const std::string expected : data["answer"]["aliases"]) {
      if (result.response.find(expected) != std::string::npos) {
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

}  // namespace gcpp

int main(int argc, char** argv) {
  gcpp::GemmaEnv env(argc, argv);
  gcpp::BenchmarkArgs benchmark_args(argc, argv);

  if (!benchmark_args.summarize_text.Empty()) {
    return BenchmarkSummary(env, benchmark_args.summarize_text);
  } else if (!benchmark_args.cross_entropy.Empty()) {
    return BenchmarkCrossEntropy(env, benchmark_args.cross_entropy,
                                 benchmark_args.batch_tokens);
  } else if (!benchmark_args.trivia_qa.Empty()) {
    return BenchmarkTriviaQA(env, benchmark_args.trivia_qa,
                             benchmark_args.max_questions);
  }
  HWY_ABORT("No benchmark command given.");
}
