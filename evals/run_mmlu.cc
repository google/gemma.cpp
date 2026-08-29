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
#include <array>
#include <cmath>
#include <limits>
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
  JsonArgs(int argc, char* argv[], ConsumedArgs& consumed) {
    InitAndParse(argc, argv, consumed);
  }

  Path input;
  size_t max_questions;

  // Returns error string or nullptr if OK.
  const char* Validate() const {
    if (input.Empty()) return "Must specify --input";
    if (!input.Exists()) return "--input file does not exist";
    return nullptr;
  }

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(input, "input", Path(), "Full pathname of mmlu.json.");
    visitor(max_questions, "max_questions", size_t{0},
            "Maximum questions to run; zero runs the full dataset.");
  };
};

// Maps both "A" and " A" tokenizer variants to answer labels 0..3. Linear
// search is faster than a map for eight tokens.
class AnswerTokens {
 public:
  explicit AnswerTokens(const GemmaTokenizer& tokenizer) {
    for (int label = 0; label < 4; ++label) {
      for (const std::string& prefix : {std::string(), std::string(" ")}) {
        const std::string str = prefix + static_cast<char>('A' + label);
        std::vector<int> tokens;
        HWY_ASSERT(tokenizer.Encode(str, &tokens));
        HWY_ASSERT(tokens.size() == 1);
        fprintf(stderr, "%s -> %d\n", str.c_str(), tokens[0]);
        tokens_.push_back({tokens[0], label});
      }
    }
  }

  int Label(int token) const {
    const auto it = std::find_if(tokens_.begin(), tokens_.end(),
                                 [token](const auto& item) {
                                   return item.first == token;
                                 });
    return it == tokens_.end() ? -1 : it->second;
  }

  const std::vector<std::pair<int, int>>& All() const { return tokens_; }

 private:
  std::vector<std::pair<int, int>> tokens_;
};

void Run(GemmaEnv& env, JsonArgs& json) {
  PROFILER_ZONE("Run.all");

  float answers = 0.0f;
  float correct_answers = 0.0f;

  auto json_data = nlohmann::json::parse(ReadFileToString(json.input));

  const AnswerTokens answer_tokens(env.GetGemma()->Tokenizer());

  for (auto sample : json_data["samples"]) {
    if (json.max_questions != 0 && answers >= json.max_questions) break;
    const int id = sample["i"];
    fprintf(stderr, "Processing question %d\n", id);
    const int correct_label = sample["input_label"];
    const std::string correct_answer(1,
                                     static_cast<char>('A' + correct_label));
    std::string prompt_string = sample["prompt"];
    // The custom sampler restricts the output to one of these four labels, so
    // make an effort to steer the model towards that. See
    // https://huggingface.co/blog/open-llm-leaderboard-mmlu
    prompt_string +=
        "What is start of the line with the correct answer? "
        "Do not include any justifications or explanations. Reply only with a "
        "letter.";
    const std::vector<int> prompt = env.WrapAndTokenize(prompt_string);
    const size_t prompt_size = prompt.size();

    int predicted_token = -1;
    std::array<float, 4> answer_logits;
    std::array<float, 4> answer_probs;
    answer_logits.fill(-std::numeric_limits<float>::infinity());
    answer_probs.fill(0.0f);
    size_t generated = 0;
    const StreamFunc stream_token = [&generated, prompt_size,
                                     &predicted_token](int token,
                                                       float /*proba*/) {
      PROFILER_ZONE("Stream");
      ++generated;
      if (generated > prompt_size) {
        predicted_token = token;
        return false;
      }
      return true;
    };

    gcpp::TimingInfo timing_info;
    gcpp::RuntimeConfig runtime_config = {
        .max_generated_tokens = 1,
        .temperature = 0.0f,
        .verbosity = env.Verbosity(),
        .attention_impl = env.MutableConfig().attention_impl,
        .stream_token = stream_token,
        .sample_func = [&answer_tokens, &answer_logits, &answer_probs](
                           size_t /*query_idx*/, size_t /*pos*/, Logits logits,
                           size_t /*worker*/) -> TokenAndProb {
          int best_token = -1;
          int best_label = -1;
          float best_logit = -std::numeric_limits<float>::infinity();
          for (const auto& [token, label] : answer_tokens.All()) {
            if (logits[token] > answer_logits[label]) {
              answer_logits[label] = logits[token];
            }
            if (logits[token] > best_logit) {
              best_logit = logits[token];
              best_token = token;
              best_label = label;
            }
          }

          float sum = 0.0f;
          for (int label = 0; label < 4; ++label) {
            answer_probs[label] =
                std::exp(answer_logits[label] - best_logit);
            sum += answer_probs[label];
          }
          for (float& prob : answer_probs) prob /= sum;
          return TokenAndProb{.token = best_token,
                              .prob = answer_probs[best_label]};
        },
    };
    env.GetGemma()->Generate(runtime_config, prompt, /*pos=*/0,
                             env.MutableKVCache(), env.MutableEnv(),
                             timing_info);

    const int predicted_label = answer_tokens.Label(predicted_token);
    const std::string output_string =
        predicted_label == -1
            ? std::string("?")
            : std::string(1, static_cast<char>('A' + predicted_label));
    fprintf(stderr, "Correct %s, model '%s'\n", correct_answer.c_str(),
            output_string.c_str());

    const bool is_correct = predicted_label == correct_label;
    float second_logit = -std::numeric_limits<float>::infinity();
    for (int label = 0; label < 4; ++label) {
      if (label != predicted_label) {
        second_logit = std::max(second_logit, answer_logits[label]);
      }
    }
    answers += 1.0f;
    if (is_correct) {
      correct_answers += 1.0f;
    }
    const nlohmann::json result = {
        {"id", id},
        {"expected", correct_answer},
        {"predicted", output_string},
        {"correct", is_correct},
        {"logits", answer_logits},
        {"probabilities", answer_probs},
        {"margin", predicted_label == -1
                       ? 0.0f
                       : answer_logits[predicted_label] - second_logit},
    };
    printf("MMLU_RESULT %s\n", result.dump().c_str());
    fflush(stdout);
    fprintf(stderr, "%.0f/%.0f = %.2f%%\n", correct_answers, answers,
            100.0f * correct_answers / answers);
  }

  const nlohmann::json summary = {
      {"answers", static_cast<size_t>(answers)},
      {"correct", static_cast<size_t>(correct_answers)},
      {"accuracy", answers == 0.0f ? 0.0f : correct_answers / answers},
  };
  printf("MMLU_SUMMARY %s\n", summary.dump().c_str());
}

}  // namespace gcpp

int main(int argc, char** argv) {
  gcpp::InternalInit();

  {
    PROFILER_ZONE("Startup.all");
    gcpp::ConsumedArgs consumed(argc, argv);
    gcpp::GemmaArgs args(argc, argv, consumed);
    gcpp::JsonArgs json_args(argc, argv, consumed);
    gcpp::AbortIfInvalidArgs(json_args);
    consumed.AbortIfUnconsumed();

    gcpp::GemmaEnv env(args);
    gcpp::Run(env, json_args);
  }
  PROFILER_PRINT_RESULTS();  // Must call outside the zone above.
  return 0;
}
