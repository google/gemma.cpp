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

#include <stdio.h>

#include <algorithm>
#include <string>
#include <vector>

#include "evals/benchmark_helper.h"
#include "gemma/gemma.h"  // Gemma
#include "io/io.h"        // Path
#include "util/args.h"
#include "hwy/base.h"
#include "hwy/highway.h"
#include "hwy/profiler.h"
#include "nlohmann/json.hpp"

namespace gcpp {

struct JsonArgs : public ArgsBase<JsonArgs> {
  JsonArgs(int argc, char* argv[]) { InitAndParse(argc, argv); }

  Path input;

  // Returns error string or nullptr if OK.
  const char* Validate() const {
    if (input.Empty()) return "Must specify --input";
    if (!input.Exists()) return "--input file does not exist";
    return nullptr;
  }

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(input, "input", Path(), "Full pathname of mmlu.json.");
  };
};

// Linear search for a few tokens is faster than std::set.
// TODO: instead of accepting for each vocab entry, filter the logits once.
class TokenSet {
 public:
  TokenSet(const GemmaTokenizer& tokenizer,
           const std::vector<std::string>& strings) {
    all_tokens_.reserve(strings.size());
    for (const std::string& str : strings) {
      std::vector<int> tokens;
      fprintf(stderr, "%s -> ", str.c_str());
      HWY_ASSERT(tokenizer.Encode(str, &tokens));
      for (int token : tokens) {
        fprintf(stderr, "%d, ", token);
        all_tokens_.push_back(token);
      }
      fprintf(stderr, "\n");
    }
  }

  bool Contains(int token) const {
    return std::find(all_tokens_.begin(), all_tokens_.end(), token) !=
           all_tokens_.end();
  }

 private:
  std::vector<int> all_tokens_;
};

void Run(GemmaEnv& env, JsonArgs& json) {
  PROFILER_ZONE("Run.all");

  float answers = 0.0f;
  float correct_answers = 0.0f;

  auto json_data = nlohmann::json::parse(ReadFileToString(json.input));

  const std::vector<std::string> accept_strings = {
      "A",  "B",   "C",   "D",   //
      " A", " B",  " C",  " D",  //
      "**", "**:", ":**", "The", "Answer", "is", ":", "."};
  const TokenSet accept_set(env.GetGemma()->Tokenizer(), accept_strings);

  for (auto sample : json_data["samples"]) {
    const int id = sample["i"];
    fprintf(stderr, "Processing question %d\n", id);
    const std::string& correct_answer = accept_strings[sample["input_label"]];
    std::string prompt_string = sample["prompt"];
    // AcceptFunc restricts the output to one of these four tokens, so make an
    // effort to steer the model towards that. See
    // https://huggingface.co/blog/open-llm-leaderboard-mmlu
    prompt_string +=
        "What is start of the line with the correct answer? "
        "Do not include any justifications or explanations. Reply only with a "
        "letter.";
    const std::vector<int> prompt = env.WrapAndTokenize(prompt_string);
    const size_t prompt_size = prompt.size();

    std::vector<int> predicted_token_ids;
    predicted_token_ids.reserve(4096);
    size_t generated = 0;
    const StreamFunc stream_token = [&generated, prompt_size,
                                     &predicted_token_ids](int token,
                                                           float proba) {
      PROFILER_ZONE("Stream");
      ++generated;
      if (generated > prompt_size) {
        predicted_token_ids.push_back(token);
      }
      return true;
    };

    // Although " A" is a token, it is difficult to associate that with the
    // correct answer. Only accepting certain tokens is risky: (A) is easily
    // confused with the word "A".
    gcpp::TimingInfo timing_info;
    gcpp::RuntimeConfig runtime_config = {
        .max_generated_tokens = 30,
        .temperature = 0.0f,
        .gen = &env.MutableGen(),
        .verbosity = env.Verbosity(),
        .stream_token = stream_token,
    };
    env.GetGemma()->Generate(runtime_config, prompt, /*pos=*/0,
                             env.MutableKVCache(), env.MutableEnv(),
                             timing_info);

    std::string output_string = env.StringFromTokens(predicted_token_ids);
    fprintf(stderr, "Correct %s, model '%s'\n", correct_answer.c_str(),
            output_string.c_str());

    answers += 1.0f;
    if (output_string == correct_answer) {
      correct_answers += 1.0f;
    }
    fprintf(stderr, "%.0f/%.0f = %.2f%%\n", correct_answers, answers,
            correct_answers / answers);
  }
}

}  // namespace gcpp

int main(int argc, char** argv) {
  {
    PROFILER_ZONE("Startup.all");
    gcpp::GemmaEnv env(argc, argv);
    gcpp::JsonArgs json(argc, argv);
    gcpp::AbortIfInvalidArgs(json);
    gcpp::Run(env, json);
  }
  PROFILER_PRINT_RESULTS();  // Must call outside the zone above.
  return 0;
}
