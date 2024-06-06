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

// Command line text interface to gemma.

#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// Placeholder for internal header, do not modify.
#include "gemma/gemma.h"  // Gemma
#include "util/app.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/highway.h"
#include "hwy/profiler.h"
#include "nlohmann/json.hpp"

namespace gcpp {

void JsonGemma(gcpp::Gemma& model, gcpp::KVCache& kv_cache,
               hwy::ThreadPool& pool,
               const InferenceArgs& args, int verbosity,
               std::string& eot_line) {
  PROFILER_ZONE("Gen.misc");
  // token index within the current turn
  int max_tokens = 4096;

  std::mt19937 gen;
  if (args.deterministic) {
    gen.seed(42);
  } else {
    std::random_device rd;
    gen.seed(rd());
  }

  float answers = 0.0;
  float correct_answers = 0.0;

  std::ifstream fJson("/tmp/mmlu.json");
  std::stringstream buffer;
  buffer << fJson.rdbuf();
  auto json = nlohmann::json::parse(buffer.str());

  std::vector<std::string> accept_tokens = {"A", "B", "C", "D"};
  std::set<int> accept_token_set{};
  for (const std::string& accept_token : accept_tokens) {
    std::vector<int> accept_token_ids;
    HWY_ASSERT(model.Tokenizer().Encode(accept_token, &accept_token_ids));
    accept_token_set.insert(accept_token_ids.begin(), accept_token_ids.end());
  }

  for (auto sample : json["samples"]) {
    int abs_pos = 0;  // absolute token index over all turns
    int current_pos = 0;
    int prompt_size{};

    // cout << "prompt:" << sample["prompt"] << endl;
    const std::string& prompt_string = sample["prompt"];
    std::vector<int> prompt;

    HWY_ASSERT(model.Tokenizer().Encode(prompt_string, &prompt));
    prompt_size = prompt.size();

    const std::string& correct_answer = accept_tokens[sample["input_label"]];

    // max_tokens = prompt_size + max_tokens;

    std::vector<int> predicted_token_ids;
    predicted_token_ids.reserve(max_tokens);
    auto stream_token = [&current_pos, &prompt_size, &predicted_token_ids,
                         &accept_token_set](int token, float proba) {
      ++current_pos;
      if (current_pos > prompt_size) {
        predicted_token_ids.push_back(token);

        // If the generated token is in the accepted token set, return False.
        // This will stop further generation.
        return accept_token_set.find(token) == accept_token_set.end();
      }

      return true;
    };

    auto accept_token = [&current_pos, &prompt_size,
                         &accept_token_set](int token) {
      // i.e. we have no constraints on accepted tokens
      if (accept_token_set.empty()) {
        return true;
      }

      if (current_pos >= prompt_size) {
        return accept_token_set.find(token) != accept_token_set.end();
      } else {
        // auto-accept early tokens
        return true;
      }
    };

    gcpp::TimingInfo timing_info;
    gcpp::RuntimeConfig runtime_config = {
        .max_tokens = args.max_tokens,
        .max_generated_tokens = args.max_generated_tokens,
        .temperature = args.temperature,
        .verbosity = verbosity,
        .gen = &gen,
        .stream_token = stream_token,
        .accept_token = accept_token,
    };
    model.Generate(runtime_config, prompt, abs_pos, kv_cache, timing_info);

    std::string output_string;
    HWY_ASSERT(model.Tokenizer().Decode(predicted_token_ids, &output_string));
    std::cout << "QuestionId: " << sample["i"] << "; "
              << "Predicted Answer: " << output_string << "; "
              << "Correct Answer: " << correct_answer << std::endl;

    answers += 1.0;
    if (output_string == correct_answer) {
      correct_answers += 1.0;
    }
    std::cout << "Running accuracy = " << "["
              << static_cast<int>(correct_answers) << "/"
              << static_cast<int>(answers) << "]" << " = "
              << correct_answers / answers << std::endl;
  }
}

void ShowConfig(LoaderArgs& loader, InferenceArgs& inference, AppArgs& app) {
  loader.Print(app.verbosity);
  inference.Print(app.verbosity);
  app.Print(app.verbosity);
}

void Run(LoaderArgs& loader, InferenceArgs& inference, AppArgs& app) {
  PROFILER_ZONE("Run.misc");

  hwy::ThreadPool pool(app.num_threads);
  // For many-core, pinning workers to cores helps.
  if (app.num_threads > 10) {
    PinWorkersToCores(pool);
  }

  gcpp::Gemma model(loader.tokenizer, loader.weights, loader.ModelType(), pool);

  gcpp::KVCache kv_cache = gcpp::KVCache::Create(loader.ModelType());

  JsonGemma(model, kv_cache, pool, inference, app.verbosity, app.eot_line);
}

}  // namespace gcpp

int main(int argc, char** argv) {
  {
    PROFILER_ZONE("Startup.misc");

    // Placeholder for internal init, do not modify.

    gcpp::LoaderArgs loader(argc, argv);
    gcpp::InferenceArgs inference(argc, argv);
    gcpp::AppArgs app(argc, argv);

    if (const char* error = loader.Validate()) {
      fprintf(stderr,
              "\ngemma.cpp\n---------\n\nTo run gemma.cpp, you need to "
              "specify 3 required model loading arguments: --tokenizer, "
              "--compressed_weights, "
              "and --model.\n\nModel Loading Arguments\n\n");

      loader.Help();
      fprintf(stderr, "\nInference Arguments\n\n");
      inference.Help();
      fprintf(stderr, "\nApplication Arguments\n\n");
      app.Help();
      fprintf(stderr, "\n\n");
      HWY_ABORT("\nInvalid args: %s", error);
    }

    gcpp::Run(loader, inference, app);
  }
  PROFILER_PRINT_RESULTS();  // Must call outside the zone above.
  return 0;
}
