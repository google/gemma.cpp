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

#include <stdio.h>

#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "compression/shared.h"  // PromptWrapping
#include "evals/benchmark_helper.h"
#include "gemma/common.h"
#include "gemma/gemma.h"  // Gemma
#include "gemma/gemma_args.h"  // LoaderArgs
#include "ops/matmul.h"        // MatMulEnv
#include "paligemma/image.h"
#include "util/args.h"  // HasHelp
#include "util/threading_context.h"
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
void ReplGemma(const ThreadingArgs& threading, const InferenceArgs& inference,
               Gemma& model, KVCache& kv_cache) {
  PROFILER_ZONE("Gen.misc");
  size_t abs_pos = 0;                     // across turns
  size_t tokens_generated_this_turn = 0;  // differentiates prefill from reply
  size_t prompt_size = 0;

  std::mt19937 gen;
  InitGenerator(inference, gen);

  const bool have_image = !inference.image_file.path.empty();
  Image image;
  ImageTokens image_tokens;
  if (have_image) {
    size_t pool_dim = model.GetModelConfig().vit_config.pool_dim;
    image_tokens =
        ImageTokens(model.Env().ctx.allocator,
                    Extents2D(model.GetModelConfig().vit_config.seq_len /
                                  (pool_dim * pool_dim),
                              model.GetModelConfig().model_dim));
    HWY_ASSERT(model.Info().wrapping == PromptWrapping::PALIGEMMA ||
               model.Info().wrapping == PromptWrapping::GEMMA_VLM);
    HWY_ASSERT(image.ReadPPM(inference.image_file.path));
    const size_t image_size = model.GetModelConfig().vit_config.image_size;
    image.Resize(image_size, image_size);
    RuntimeConfig runtime_config = {.gen = &gen,
                                    .verbosity = inference.verbosity,
                                    .use_spinning = threading.spin};
    double image_tokens_start = hwy::platform::Now();
    model.GenerateImageTokens(runtime_config, image, image_tokens);
    if (inference.verbosity >= 1) {
      double image_tokens_duration = hwy::platform::Now() - image_tokens_start;
      fprintf(stderr,
              "\n\n[ Timing info ] Image token generation took: %d ms\n",
              static_cast<int>(image_tokens_duration * 1000));
    }
  }

  // callback function invoked for each generated token.
  auto stream_token = [&](int token, float) {
    ++abs_pos;
    const bool in_prompt = tokens_generated_this_turn < prompt_size;
    const bool first_response_token = tokens_generated_this_turn == prompt_size;
    ++tokens_generated_this_turn;
    if (in_prompt) {
      if (inference.verbosity >= 1) {
        std::cerr << "." << std::flush;
      }
      return true;
    } else if (model.GetModelConfig().IsEOS(token)) {
      if (inference.verbosity >= 2) {
        std::cout << "\n[ End ]\n";
      }
      return true;
    }
    std::string token_text;
    HWY_ASSERT(model.Tokenizer().Decode(std::vector<int>{token}, &token_text));
    if (first_response_token) {
      token_text.erase(0, token_text.find_first_not_of(" \t\n"));
      if (inference.verbosity >= 1) {
        std::cout << "\n\n";
      }
    }
    std::cout << token_text << std::flush;
    return true;
  };

  while (true) {  // Loop until user quits.
    tokens_generated_this_turn = 0;

    // Read prompt and handle special commands.
    std::string prompt_string =
        GetPrompt(std::cin, inference.verbosity, inference.eot_line);
    if (!std::cin) return;
    // If !eot_line.empty(), we append \n, so only look at the first 2 chars.
    if (prompt_string.size() >= 2 && prompt_string[0] == '%') {
      if (prompt_string[1] == 'q' || prompt_string[1] == 'Q') return;
      if (prompt_string[1] == 'c' || prompt_string[1] == 'C') {
        abs_pos = 0;
        continue;
      }
    }
    if (prompt_string.empty()) {
      std::cout << "Use '%q' to quit.\n";
      continue;
    }

    // Set up runtime config.
    TimingInfo timing_info = {.verbosity = inference.verbosity};
    RuntimeConfig runtime_config = {.gen = &gen,
                                    .verbosity = inference.verbosity,
                                    .stream_token = stream_token,
                                    .use_spinning = threading.spin};
    inference.CopyTo(runtime_config);
    size_t prefix_end = 0;

    std::vector<int> prompt;
    if (have_image) {
      prompt =
          WrapAndTokenize(model.Tokenizer(), model.ChatTemplate(), model.Info(),
                          abs_pos, prompt_string, image_tokens.BatchSize());
      runtime_config.image_tokens = &image_tokens;
      prompt_size = prompt.size();
      // The end of the prefix for prefix-LM style attention in Paligemma.
      // See Figure 2 of https://arxiv.org/abs/2407.07726.
      prefix_end = prompt_size;
      // We need to look at all the tokens for the prefix.
      runtime_config.prefill_tbatch_size = prompt_size;
    } else {
      prompt = WrapAndTokenize(model.Tokenizer(), model.ChatTemplate(),
                               model.Info(), abs_pos, prompt_string);
      prompt_size = prompt.size();
    }

    if constexpr (kVerboseLogTokens) {
      for (int i = 0; i < prompt_size; ++i) {
        fprintf(stderr, "DDD TOKEN %3d: %6d\n", i, prompt[i]);
      }
    }

    // Generate until EOS or max_generated_tokens.
    if (inference.verbosity >= 1) {
      std::cerr << "\n[ Reading prompt ] " << std::flush;
    }
    model.Generate(runtime_config, prompt, abs_pos, prefix_end, kv_cache,
                   timing_info);
    std::cout << "\n\n";

    // Prepare for the next turn. Works only for PaliGemma.
    if (!inference.multiturn ||
        model.Info().wrapping == PromptWrapping::PALIGEMMA) {
      abs_pos = 0;  // Start a new turn at position 0.
      InitGenerator(inference, gen);
    } else {
      // The last token was either EOS, then it should be ignored because it is
      // never part of the dialog, see Table 5 in the Gemma-2 paper:
      // https://arxiv.org/pdf/2408.00118
      // Or we have hit max_generated_tokens, then the last token will be lost.
      // (We could store it in stream_token, and then prepend to the next turn,
      // but it's not worth the complexity, as multi-turn with max_generated is
      // not a common use case.)
      // In either case, we need to rewind abs_pos by one.
      HWY_ASSERT(abs_pos > 0);
      abs_pos--;
    }
  }
}

void Run(ThreadingArgs& threading, LoaderArgs& loader,
         InferenceArgs& inference) {
  PROFILER_ZONE("Run.misc");

  // Note that num_threads is an upper bound; we also limit to the number of
  // detected and enabled cores.
  MatMulEnv env(MakeMatMulEnv(threading));
  if (inference.verbosity >= 2) env.print_best = true;
  Gemma model = CreateGemma(loader, env);
  KVCache kv_cache =
      KVCache::Create(model.GetModelConfig(), inference.prefill_tbatch_size);

  if (inference.verbosity >= 1) {
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
    ShowConfig(threading, loader, inference);
    std::cout << "\n" << instructions << "\n";
  }

  ReplGemma(threading, inference, model, kv_cache);
}

}  // namespace gcpp

int main(int argc, char** argv) {
  {
    PROFILER_ZONE("Startup.misc");

    gcpp::ThreadingArgs threading(argc, argv);
    gcpp::LoaderArgs loader(argc, argv);
    gcpp::InferenceArgs inference(argc, argv);

    if (gcpp::HasHelp(argc, argv)) {
      std::cerr << gcpp::kAsciiArtBanner;
      gcpp::ShowHelp(threading, loader, inference);
      return 0;
    }

    if (const char* error = loader.Validate()) {
      std::cerr << gcpp::kAsciiArtBanner;
      gcpp::ShowHelp(threading, loader, inference);
      HWY_ABORT("\nInvalid args: %s", error);
    }

    if (const char* error = inference.Validate()) {
      std::cerr << gcpp::kAsciiArtBanner;
      gcpp::ShowHelp(threading, loader, inference);
      HWY_ABORT("\nInvalid args: %s", error);
    }

    gcpp::Run(threading, loader, inference);
  }
  PROFILER_PRINT_RESULTS();  // Must call outside the zone above.
  return 0;
}
