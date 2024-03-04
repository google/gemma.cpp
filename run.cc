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

#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <thread>  // NOLINT
#include <vector>

// copybara:import_next_line:gemma_cpp
#include "compression/compress.h"
// copybara:end
// copybara:import_next_line:gemma_cpp
#include "gemma.h"  // Gemma
// copybara:end
// copybara:import_next_line:gemma_cpp
#include "util/app.h"
// copybara:end
// copybara:import_next_line:gemma_cpp
#include "util/args.h"  // HasHelp
// copybara:end
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/highway.h"
#include "hwy/per_target.h"
#include "hwy/profiler.h"
#include "hwy/timer.h"

namespace gcpp {

static constexpr std::string_view kAsciiArtBanner =
    "  __ _  ___ _ __ ___  _ __ ___   __ _   ___ _ __  _ __\n"
    " / _` |/ _ \\ '_ ` _ \\| '_ ` _ \\ / _` | / __| '_ \\| '_ \\\n"
    "| (_| |  __/ | | | | | | | | | | (_| || (__| |_) | |_) |\n"
    " \\__, |\\___|_| |_| |_|_| |_| |_|\\__,_(_)___| .__/| .__/\n"
    "  __/ |                                    | |   | |\n"
    " |___/                                     |_|   |_|";

void ShowConfig(Gemma_args garg) {
  //for (std::string mem_str : get_struct_print(garg.load)) {//TODO(RROP) fix these
  //  std::cout << mem_str << std::endl;
  //}
  //for (std::string mem_str : get_struct_print(garg.inf)) {
  //  std::cout << mem_str << std::endl;
  //}
  //for (std::string mem_str : get_struct_print(garg.app)) {
  //  std::cout << mem_str << std::endl;
  //}
  if (garg.app.verbosity >= 2) {
    time_t now = time(nullptr);
    char* dt = ctime(&now);  // NOLINT
    std::cout << "Date & Time                   : " << dt
              << "Prefill Token Batch Size      : " << gcpp::kPrefillBatchSize
              << "\n"
              << "Hardware concurrency          : "
              << std::thread::hardware_concurrency() << std::endl
              << "Instruction set               : "
              << hwy::TargetName(hwy::DispatchedTarget()) << " ("
              << hwy::VectorBytes() * 8 << " bits)"
              << "\n"
              << "Weight Type                   : "
              << gcpp::TypeName(gcpp::WeightT()) << "\n"
              << "EmbedderInput Type            : "
              << gcpp::TypeName(gcpp::EmbedderInputT()) << "\n";
  }
}

void ShowHelp() {
  std::cerr
      << kAsciiArtBanner
      << "\n\ngemma.cpp : a lightweight, standalone C++ inference engine\n"
         "==========================================================\n\n"
         "To run gemma.cpp, you need to "
         "specify 3 required model loading arguments:\n    --tokenizer\n    "
         "--compressed_weights\n"
         "    --model.\n";
  std::cerr << "\n*Example Usage*\n\n./gemma --tokenizer tokenizer.spm "
               "--compressed_weights 2b-it-sfp.sbs --model 2b-it\n";
  std::cerr << "\n*Model Loading Arguments*\n\n";
  for (auto memstr: loader_help) {
    std::cerr << "--" << memstr.first << "\t:\t" << memstr.second
              << std::endl;
  }
  std::cerr << "\n*Inference Arguments*\n\n";
  for (auto memstr : inference_help) {
    std::cerr << "--" << memstr.first << "\t:\t" << memstr.second << std::endl;
  }
  std::cerr << "\n*Application Arguments*\n\n";
  for (auto memstr : application_help) {
    std::cerr << "--" << memstr.first << "\t:\t" << memstr.second << std::endl;
  }
  std::cerr << "\n";
}

void ReplGemma(gcpp::Gemma& model, hwy::ThreadPool& pool, hwy::ThreadPool& inner_pool,
               gcpp::gemma_consts& nconsts) {
  PROFILER_ZONE("Gen.misc");
  int abs_pos = 0;      // absolute token index over all turns
  int current_pos = 0;  // token index within the current turn
  int prompt_size{};

  std::mt19937 gen;
  if (nconsts.argbase.inf.deterministic) {
    gen.seed(42);
  } else {
    std::random_device rd;
    gen.seed(rd());
  }
  
  // callback function invoked for each generated token.
  nconsts.stream_token = [&abs_pos, &current_pos, &nconsts, &pool,&inner_pool, &gen, &prompt_size,
                       tokenizer = &model.Tokenizer()](int token, float) {
      ++abs_pos;
    ++current_pos;
    if (current_pos < prompt_size) {
      std::cerr << "." << std::flush;
    } else if (token == gcpp::EOS_ID) {
      if (!nconsts.argbase.inf.multiturn) {
        abs_pos = 0;
        if (nconsts.argbase.inf.deterministic) {
          gen.seed(42);
        }
      }
      if (nconsts.argbase.app.verbosity >= 2) {
        std::cout << "\n[ End ]" << std::endl;
      }
    } else {
      std::string token_text;
      HWY_ASSERT(tokenizer->Decode(std::vector<int>{token}, &token_text).ok());
      // +1 since position is incremented above
      if (current_pos == prompt_size + 1) {
        // first token of response
        token_text.erase(0, token_text.find_first_not_of(" \t\n"));
        if (nconsts.argbase.app.verbosity >= 1) {
          std::cout << std::endl << std::endl;
        }
      }
      // TODO(austinvhuang): is explicit space necessary?
      std::cout << token_text << std::flush;
    }
    return true;
  };
  nconsts.accept_token = [](int token) { return true; };
  while (abs_pos < nconsts.argbase.inf.max_tokens) {
    std::string prompt_string;
    std::vector<int> prompt;
    current_pos = 0;
    {
      PROFILER_ZONE("Gen.input");
      if (nconsts.argbase.app.verbosity >= 1) {
        std::cout << "> " << std::flush;
      }

      if (nconsts.argbase.app.eot_line.size() == 0) {
        std::getline(std::cin, prompt_string);
      } else {
        std::string line;
        while (std::getline(std::cin, line)) {
          if (line == nconsts.argbase.app.eot_line) {
            break;
          }
          prompt_string += line + "\n";
        }
      }
    }

    if (std::cin.fail() || prompt_string == "%q" || prompt_string == "%Q") {
      return;
    }

    if (prompt_string == "%c" || prompt_string == "%C") {
      abs_pos = 0;
      continue;
    }

    if (model.model_training == ModelTraining::GEMMA_IT) {
      // For instruction-tuned models: add control tokens.
      prompt_string = "<start_of_turn>user\n" + prompt_string +
                      "<end_of_turn>\n<start_of_turn>model\n";
      if (abs_pos > 0) {
        // Prepend "<end_of_turn>" token if this is a multi-turn dialogue
        // continuation.
        prompt_string = "<end_of_turn>\n" + prompt_string;
      }
    }

    HWY_ASSERT(model.Tokenizer().Encode(prompt_string, &prompt).ok());

    // For both pre-trained and instruction-tuned models: prepend "<bos>" token
    // if needed.
    if (abs_pos == 0) {
      prompt.insert(prompt.begin(), 2);
    }

    prompt_size = prompt.size();

    std::cerr << std::endl << "[ Reading prompt ] " << std::flush;

    const double time_start = hwy::platform::Now();
    GenerateGemma(model,nconsts, pool,inner_pool,gen, prompt, abs_pos);//(RROP)I'm confused if abspos is input here. Maybe a bug will happen here.
    const double time_end = hwy::platform::Now();
    const double tok_sec = current_pos / (time_end - time_start);
    if (nconsts.argbase.app.verbosity >= 2) {
      std::cout << current_pos << " tokens (" << abs_pos << " total tokens)"
                << std::endl
                << tok_sec << " tokens / sec" << std::endl;
    }
    std::cout << std::endl << std::endl;
  }
  std::cout
      << "max_tokens (" << nconsts.argbase.inf.max_tokens
      << ") exceeded. Use a larger value if desired using the --max_tokens "
      << "command line flag.\n";
}

void Run(Gemma_args& garg) {
  PROFILER_ZONE("Run.misc");
  hwy::ThreadPool pool(garg.app.num_threads);
  hwy::ThreadPool inner_pool(0);
  /*hwy::ThreadPool inner_pool(0);
  hwy::ThreadPool pool(garg.app.num_threads);*/
  // For many-core, pinning threads to cores helps.
  if (garg.app.num_threads > 10) {
    PinThreadToCore(garg.app.num_threads - 1);  // Main thread

    pool.Run(0, pool.NumThreads(),
             [](uint64_t /*task*/, size_t thread) { PinThreadToCore(thread); });
  }
  gcpp::Gemma model(garg, pool);
  

  if (const char* error = garg.inf.Validate()) {
    ShowHelp();
    HWY_ABORT("\nInvalid args: %s", error);
  }

  if (garg.app.verbosity >= 1) {
    static const std::string banner_ascii_art =
        "  __ _  ___ _ __ ___  _ __ ___   __ _   ___ _ __  _ __\n"
        " / _` |/ _ \\ '_ ` _ \\| '_ ` _ \\ / _` | / __| '_ \\| '_ \\\n"
        "| (_| |  __/ | | | | | | | | | | (_| || (__| |_) | |_) |\n"
        " \\__, |\\___|_| |_| |_|_| |_| |_|\\__,_(_)___| .__/| .__/\n"
        "  __/ |                                    | |   | |\n"
        " |___/                                     |_|   |_|";

    const std::string instructions =
        "*Usage*\n"
        "  Enter an instruction and press enter (%C resets conversation, "
        "%Q quits).\n" +
        (garg.inf.multiturn == 0
             ? std::string("  Since multiturn is set to 0, conversation will "
                           "automatically reset every turn.\n\n")
             : "\n") +
        "*Examples*\n"
        "  - Write an email to grandma thanking her for the cookies.\n"
        "  - What are some historical attractions to visit around "
        "Massachusetts?\n"
        "  - Compute the nth fibonacci number in javascript.\n"
        "  - Write a standup comedy bit about GPU programming.\n";

    std::cout << "\033[2J\033[1;1H"  // clear screen
              << kAsciiArtBanner << "\n\n";
    ShowConfig(garg);
    std::cout << "\n" << instructions << "\n";
  }
  gemma_consts constables;
  constables.argbase = garg;
  ReplGemma(model,pool,inner_pool,constables);
}

}  // namespace gcpp

int main(int argc, char** argv) {
  {
    PROFILER_ZONE("Startup.misc");

    gcpp::LoaderArgs loader(argc, argv);
    gcpp::InferenceArgs inference(argc, argv);
    gcpp::AppArgs app(argc, argv);

    gcpp::Gemma_args garg;
    garg.load.compressed_weights = loader.cache;
    garg.load.model = loader.ModelType();
    garg.load.modeltrain = loader.ModelTraining();
    garg.load.tokenizer = loader.tokenizer;
    garg.load.weights = loader.model;

    garg.inf.deterministic = inference.deterministic;
    garg.inf.max_gen_tokens = inference.max_generated_tokens;
    garg.inf.max_tokens = inference.max_tokens;
    garg.inf.multiturn = inference.multiturn;
    garg.inf.temperature = inference.temperature;

    garg.app.eot_line = app.eot_line;
    garg.app.num_threads = app.num_threads;
    garg.app.verbosity = app.verbosity;


    if (gcpp::HasHelp(argc, argv)) {
        
      gcpp::ShowHelp();
      return 0;
    }

    if (const char* error = loader.Validate()) {
      gcpp::ShowHelp();
      HWY_ABORT("\nInvalid args: %s", error);
    }

    gcpp::Run(garg);
  }
  PROFILER_PRINT_RESULTS();  // Must call outside the zone above.
  return 0;
}
