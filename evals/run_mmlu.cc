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
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "evals/benchmark_helper.h"
#include "evals/model_comparison.h"
#include "gemma/gemma.h"  // Gemma
#include "hwy/base.h"
#include "hwy/highway.h"
#include "hwy/profiler.h"
#include "io/io.h"  // Path
#include "nlohmann/json.hpp"
#include "util/args.h"

namespace gcpp {

struct JsonArgs : public ArgsBase<JsonArgs> {
  JsonArgs(int argc, char* argv[], ConsumedArgs& consumed) {
    InitAndParse(argc, argv, consumed);
  }

  Path input;
  Path reference_out;
  Path reference_in;
  size_t max_questions;

  const char* Validate() const {
    if (input.Empty()) return "Must specify --input";
    if (!input.Exists()) return "--input file does not exist";
    if (!reference_out.Empty() && !reference_in.Empty()) {
      return "Specify only one of --reference_out and --reference_in";
    }
    if (!reference_in.Empty() && !reference_in.Exists()) {
      return "--reference_in file does not exist";
    }
    return nullptr;
  }

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(input, "input", Path(), "Full pathname of mmlu.json.");
    visitor(reference_out, "reference_out", Path(),
            "Write root-model full-vocabulary logits to this binary file.");
    visitor(reference_in, "reference_in", Path(),
            "Compare this target model against a root reference file.");
    visitor(max_questions, "max_questions", size_t{0},
            "Maximum questions to run; zero runs the full dataset.");
  }
};

// Maps both "A" and " A" tokenizer variants to answer labels 0..3.
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
    const auto it =
        std::find_if(tokens_.begin(), tokens_.end(),
                     [token](const auto& item) { return item.first == token; });
    return it == tokens_.end() ? -1 : it->second;
  }

  const std::vector<std::pair<int, int>>& All() const { return tokens_; }

 private:
  std::vector<std::pair<int, int>> tokens_;
};

double Percentile(const std::vector<double>& sorted, double quantile) {
  if (sorted.empty()) return 0.0;
  const double position = quantile * static_cast<double>(sorted.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
}

void Run(GemmaEnv& env, JsonArgs& args) {
  PROFILER_ZONE("Run.all");

  size_t answers = 0;
  size_t correct_answers = 0;
  std::vector<double> kl_values;

  const std::string json_text = ReadFileToString(args.input);
  const auto json_data = nlohmann::json::parse(json_text);
  const auto& samples = json_data["samples"];
  const size_t sample_count =
      args.max_questions == 0
          ? samples.size()
          : std::min<size_t>(args.max_questions, samples.size());

  const Gemma& gemma = *env.GetGemma();
  const GemmaTokenizer& tokenizer = gemma.Tokenizer();
  const ModelComparisonMetadata metadata = {
      static_cast<uint32_t>(gemma.Config().vocab_size), sample_count,
      ModelComparisonFingerprint(json_text),
      ModelComparisonFingerprint(tokenizer.Serialize())};

  std::unique_ptr<ModelComparisonWriter> reference_writer;
  std::unique_ptr<ModelComparisonReader> reference_reader;
  if (!args.reference_out.Empty()) {
    reference_writer = std::make_unique<ModelComparisonWriter>(
        args.reference_out.path, metadata);
  } else if (!args.reference_in.Empty()) {
    reference_reader =
        std::make_unique<ModelComparisonReader>(args.reference_in.path);
    reference_reader->Validate(metadata);
    kl_values.reserve(sample_count);
  }

  const AnswerTokens answer_tokens(tokenizer);

  for (const auto& sample : samples) {
    if (answers >= sample_count) break;
    const int64_t id = sample["i"];
    fprintf(stderr, "Processing question %lld\n", static_cast<long long>(id));
    const int correct_label = sample["input_label"];
    const std::string correct_answer(1, static_cast<char>('A' + correct_label));

    ModelComparisonRecord root_record;
    if (reference_reader) {
      root_record = reference_reader->Read();
      if (root_record.sample_id != id ||
          root_record.expected_label != correct_label) {
        throw std::runtime_error("KL reference sample identity mismatch");
      }
    }

    std::string prompt_string = sample["prompt"];
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
    std::vector<float> captured_logits;
    double full_vocab_kl = 0.0;
    size_t generated = 0;
    const StreamFunc stream_token = [&generated, prompt_size, &predicted_token](
                                        int token, float /*proba*/) {
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
        .sample_func = [&answer_tokens, &answer_logits, &answer_probs,
                        &captured_logits, &full_vocab_kl, &reference_writer,
                        &reference_reader, &root_record](
                           size_t /*query_idx*/, size_t /*pos*/, Logits logits,
                           size_t /*worker*/) -> TokenAndProb {
          if (reference_writer) {
            captured_logits.assign(logits.data(),
                                   logits.data() + logits.size());
          } else if (reference_reader) {
            full_vocab_kl = FullVocabKLDivergence(root_record.logits,
                                                  root_record.log_sum_exp,
                                                  logits.data(), logits.size());
          }

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
            answer_probs[label] = std::exp(answer_logits[label] - best_logit);
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

    if (reference_writer) {
      if (captured_logits.size() != metadata.vocab_size) {
        throw std::runtime_error(
            "failed to capture root full-vocabulary logits");
      }
      reference_writer->Write(id, correct_label, captured_logits.data(),
                              captured_logits.size());
    } else if (reference_reader) {
      if (!std::isfinite(full_vocab_kl) || full_vocab_kl < 0.0) {
        throw std::runtime_error("invalid full-vocabulary KL divergence");
      }
      kl_values.push_back(full_vocab_kl);
    }

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
    ++answers;
    correct_answers += static_cast<size_t>(is_correct);
    nlohmann::json result = {
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
    if (reference_reader) result["full_vocab_kl"] = full_vocab_kl;
    printf("MMLU_RESULT %s\n", result.dump().c_str());
    fflush(stdout);
    fprintf(stderr, "%zu/%zu = %.2f%%\n", correct_answers, answers,
            100.0 * static_cast<double>(correct_answers) / answers);
  }

  if (reference_writer) reference_writer->Finish();
  if (reference_reader) reference_reader->Finish();

  const nlohmann::json summary = {
      {"answers", answers},
      {"correct", correct_answers},
      {"accuracy",
       answers == 0 ? 0.0 : static_cast<double>(correct_answers) / answers},
  };
  printf("MMLU_SUMMARY %s\n", summary.dump().c_str());

  if (!kl_values.empty()) {
    std::sort(kl_values.begin(), kl_values.end());
    const double mean =
        std::accumulate(kl_values.begin(), kl_values.end(), 0.0) /
        kl_values.size();
    const nlohmann::json kl_summary = {
        {"samples", kl_values.size()},
        {"mean", mean},
        {"median", Percentile(kl_values, 0.5)},
        {"p95", Percentile(kl_values, 0.95)},
        {"max", kl_values.back()},
        {"unit", "nats"},
        {"direction", "root||target"},
    };
    printf("MMLU_KL_SUMMARY %s\n", kl_summary.dump().c_str());
  }
}

}  // namespace gcpp

int main(int argc, char** argv) {
  gcpp::InternalInit();

  try {
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
    PROFILER_PRINT_RESULTS();
    return 0;
  } catch (const std::exception& error) {
    fprintf(stderr, "model comparison failed: %s\n", error.what());
    return 1;
  }
}
