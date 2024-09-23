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

#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

// Placeholder for internal header, do not modify.
#include "evals/benchmark_helper.h"
#include "gemma/common.h"
#include "gemma/gemma.h"  // Gemma
#include "paligemma/image.h"
#include "util/app.h"
#include "util/args.h"  // HasHelp
#include "util/threading.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/highway.h"
#include "hwy/profiler.h"

#if (!defined(HWY_VERSION_LT) || HWY_VERSION_LT(1, 2)) && !HWY_IDE
#error "Please update to version 1.2 of github.com/google/highway."
#endif
#if HWY_CXX_LANG < 201703L
#error "Gemma.cpp requires C++17, please pass -std=c++17."
#endif

static constexpr bool kVerboseLogTokens = false;

namespace gcpp {

static constexpr std::string_view kAsciiArtBanner = R""(
  __ _  ___ _ __ ___  _ __ ___   __ _   ___ _ __  _ __
 / _` |/ _ \ '_ ` _ \| '_ ` _ \ / _` | / __| '_ \| '_ \
| (_| |  __/ | | | | | | | | | | (_| || (__| |_) | |_) |
 \__, |\___|_| |_| |_|_| |_| |_|\__,_(_)___| .__/| .__/
  __/ |                                    | |   | |
 |___/                                     |_|   |_|
)"";

std::string GetPrompt(std::istream& input, int verbosity,
                      std::string_view eot_line) {
  PROFILER_ZONE("Gen.input");
  if (verbosity >= 1) {
    std::cout << "> " << std::flush;
  }

  std::string prompt_string;
  if (eot_line.empty()) {
    std::getline(input, prompt_string);
  } else {
    std::string line;
    while (std::getline(input, line)) {
      if (line == eot_line) {
        break;
      }
      prompt_string += line + "\n";
    }
  }
  return prompt_string;
}

// The main Read-Eval-Print Loop.
void ReplGemma(Gemma& model, KVCache& kv_cache, const InferenceArgs& args,
               int verbosity, const AcceptFunc& accept_token,
               std::string& eot_line) {
  PROFILER_ZONE("Gen.misc");
  size_t abs_pos = 0;                     // across turns
  size_t tokens_generated_this_turn = 0;  // differentiates prefill from reply
  size_t prompt_size = 0;

  std::mt19937 gen;
  InitGenerator(args, gen);

  const bool have_image = !args.image_file.path.empty();
  Image image;
  ImageTokens image_tokens(256, 2048);
  if (have_image) {
    HWY_ASSERT(model.Info().model == Model::PALIGEMMA_224);
    HWY_ASSERT(image.ReadPPM(args.image_file.path));
    image.Resize();
    RuntimeConfig runtime_config = {.verbosity = verbosity, .gen = &gen};
    model.GenerateImageTokens(runtime_config, image, image_tokens);
  }

  // callback function invoked for each generated token.
  auto stream_token = [&](int token, float) {
    ++abs_pos;
    ++tokens_generated_this_turn;
    // <= since position is incremented before
    if (tokens_generated_this_turn <= prompt_size) {
      std::cerr << "." << std::flush;
    } else if (token == EOS_ID) {
      if (!args.multiturn) {
        abs_pos = 0;
        InitGenerator(args, gen);
      }
      if (verbosity >= 2) {
        std::cout << "\n[ End ]\n";
      }
    } else {
      std::string token_text;
      HWY_ASSERT(
          model.Tokenizer().Decode(std::vector<int>{token}, &token_text));
      // +1 since position is incremented above
      if (tokens_generated_this_turn == prompt_size + 1) {
        // first token of response
        token_text.erase(0, token_text.find_first_not_of(" \t\n"));
        if (verbosity >= 1) {
          std::cout << "\n\n";
        }
      }
      std::cout << token_text << std::flush;
    }
    return true;
  };

  while (abs_pos < args.max_tokens) {
    tokens_generated_this_turn = 0;
    std::string prompt_string = GetPrompt(std::cin, verbosity, eot_line);
    if (!std::cin) return;
    // If !eot_line.empty(), we append \n, so only look at the first 2 chars.
    if (prompt_string.size() >= 2 && prompt_string[0] == '%') {
      if (prompt_string[1] == 'q' || prompt_string[1] == 'Q') return;
      if (prompt_string[1] == 'c' || prompt_string[1] == 'C') {
        abs_pos = 0;
        continue;
      }
    }

    if (have_image && abs_pos != 0) {
      // This occurs when we have hit max_generated.
      abs_pos = 0;
    }

    std::vector<int> prompt = WrapAndTokenize(
        model.Tokenizer(), model.Info(), abs_pos, prompt_string);
    prompt_size = prompt.size();
    std::cerr << "\n"
              << "[ Reading prompt ] " << std::flush;
    if constexpr (kVerboseLogTokens) {
      for (int i = 0; i < prompt_size; ++i) {
        fprintf(stderr, "DDD TOKEN %3d: %6d\n", i, prompt[i]);
      }
    }

    TimingInfo timing_info = {.verbosity = verbosity};
    RuntimeConfig runtime_config = {
        .verbosity = verbosity,
        .gen = &gen,
        .stream_token = stream_token,
        .accept_token = accept_token,
    };
    args.CopyTo(runtime_config);
    size_t prefix_end = 0;
    if (have_image) {
      runtime_config.image_tokens = &image_tokens;
      prompt.insert(prompt.begin(), image_tokens.BatchSize(), 0);
      prompt_size = prompt.size();
      // The end of the prefix for prefix-LM style attention in Paligemma.
      // See Figure 2 of https://arxiv.org/abs/2407.07726.
      prefix_end = prompt_size;
      // We need to look at all the tokens for the prefix.
      runtime_config.prefill_tbatch_size = prompt_size;
    }
    model.Generate(runtime_config, prompt, abs_pos, prefix_end, kv_cache,
                   timing_info);
    std::cout << "\n\n";
  }
  std::cout
      << "max_tokens (" << args.max_tokens
      << ") exceeded. Use a larger value if desired using the --max_tokens "
      << "command line flag.\n";
}

void Run(LoaderArgs& loader, InferenceArgs& inference, AppArgs& app) {
  PROFILER_ZONE("Run.misc");

  // Note that num_threads is an upper bound; we also limit to the number of
  // detected and enabled cores.
  PerClusterPools pools(app.max_clusters, app.num_threads, app.pin);

  Gemma model = CreateGemma(loader, pools);
  KVCache kv_cache =
      KVCache::Create(model.Info().model, inference.prefill_tbatch_size);

  if (app.verbosity >= 1) {
    std::string instructions =
        "*Usage*\n"
        "  Enter an instruction and press enter (%C resets conversation, "
        "%Q quits).\n";
    const std::string multiturn =
        inference.multiturn == 0
            ? std::string(
                  "  Since multiturn is set to 0, conversation will "
                  "automatically reset every turn.\n\n")
            : "\n";
    const std::string examples =
        "*Examples*\n"
        "  - Write an email to grandma thanking her for the cookies.\n"
        "  - What are some historical attractions to visit around "
        "Massachusetts?\n"
        "  - Compute the nth fibonacci number in javascript.\n"
        "  - Write a standup comedy bit about GPU programming.\n";
    instructions += multiturn;
    instructions += examples;

    std::cout << "\033[2J\033[1;1H"  // clear screen
              << kAsciiArtBanner << "\n\n";
    ShowConfig(loader, inference, app, pools);
    std::cout << "\n" << instructions << "\n";
  }

  ReplGemma(model, kv_cache, inference, app.verbosity, AcceptFunc(),
            app.eot_line);
}

}  // namespace gcpp

int main(int argc, char** argv) {
  {
    PROFILER_ZONE("Startup.misc");

    // Placeholder for internal init, do not modify.

    gcpp::LoaderArgs loader(argc, argv);
    gcpp::InferenceArgs inference(argc, argv);
    gcpp::AppArgs app(argc, argv);

    if (gcpp::HasHelp(argc, argv)) {
      std::cerr << gcpp::kAsciiArtBanner;
      gcpp::ShowHelp(loader, inference, app);
      return 0;
    }

    if (const char* error = loader.Validate()) {
      std::cerr << gcpp::kAsciiArtBanner;
      gcpp::ShowHelp(loader, inference, app);
      HWY_ABORT("\nInvalid args: %s", error);
    }

    if (const char* error = inference.Validate()) {
      std::cerr << gcpp::kAsciiArtBanner;
      gcpp::ShowHelp(loader, inference, app);
      HWY_ABORT("\nInvalid args: %s", error);
    }

    gcpp::Run(loader, inference, app);
  }
  PROFILER_PRINT_RESULTS();  // Must call outside the zone above.
  return 0;
}
