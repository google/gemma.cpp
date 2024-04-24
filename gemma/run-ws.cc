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
#include <sstream>
#include <string>
#include <thread>  // NOLINT
#include <vector>

// Placeholder for internal header, do not modify.
#include <ixwebsocket/IXSocket.h>
#include <ixwebsocket/IXSocketFactory.h>
#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXWebSocketServer.h>

#include "compression/compress.h"
#include "gemma/gemma.h"  // Gemma
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/highway.h"
#include "hwy/per_target.h"
#include "hwy/profiler.h"
#include "hwy/timer.h"
#include "util/app.h"
#include "util/args.h"  // HasHelp

/** test client by python
#!/usr/bin/python
#!pip install websocket-client
import websocket
ws = websocket.WebSocket()
ws.connect("ws://127.0.0.1:9999")
print(ws.recv())
ws.send("Hello, Gemma!")
while True:
    token = ws.recv()
    if token == "%EOS%":
        break
    print(token, end='')
ws.close()
*/

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

void ShowHelp(gcpp::LoaderArgs& loader, gcpp::InferenceArgs& inference,
              gcpp::AppArgs& app) {
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
  loader.Help();
  std::cerr << "\n*Inference Arguments*\n\n";
  inference.Help();
  std::cerr << "\n*Application Arguments*\n\n";
  app.Help();
  std::cerr << "\n";
}

void ReplGemma(gcpp::Gemma& model, ModelTraining training,
               gcpp::KVCache& kv_cache, hwy::ThreadPool& pool,
               hwy::ThreadPool& inner_pool, const InferenceArgs& args,
               int verbosity, const gcpp::AcceptFunc& accept_token,
               std::string& eot_line, int port) {
  PROFILER_ZONE("Gen.misc");

  size_t abs_pos = 0;  // absolute token index over all turns
  std::mt19937 gen;
  if (args.deterministic) {
    gen.seed(42);
  } else {
    std::random_device rd;
    gen.seed(rd());
  }

  ix::WebSocketServer ws(port);
  ws.setOnClientMessageCallback(
      [&](std::shared_ptr<ix::ConnectionState> connectionState,
          ix::WebSocket& webSocket, const ix::WebSocketMessagePtr& msg) {
        auto toClient = [&](std::string token_text) {
          for (auto&& client : ws.getClients()) {
            if (client.get() == &webSocket) {
              client->send(token_text, false);
            }
          }
        };

        if (msg->type == ix::WebSocketMessageType::Open) {
          if (verbosity >= 2) {
            std::cout << "New connection" << std::endl;
          }
          std::stringstream ntf;
          time_t now = time(nullptr);
          char* dt = ctime(&now);  // NOLINT
          ntf << "*Usage*\n"
                 "  - Enter an instruction and press enter (%C resets "
                 "conversation, "
                 "%Q quits).\n\n"
                 "*Examples*\n"
                 "  - Write an email to grandma thanking her for the cookies.\n"
                 "  - What are some historical attractions to visit around "
                 "Massachusetts?\n"
                 "  - Compute the nth fibonacci number in javascript.\n"
                 "  - Write a standup comedy bit about GPU programming.\n"
                 "\n";
          ntf << "*Config*\n"
              << "- Date & Time                   : " << dt
              << "- Prefill Token Batch Size      : " << gcpp::kPrefillBatchSize
              << "\n"
              << "- Hardware concurrency          : "
              << std::thread::hardware_concurrency() << "\n"
              << "- Instruction set               : "
              << hwy::TargetName(hwy::DispatchedTarget()) << " ("
              << hwy::VectorBytes() * 8 << " bits)"
              << "\n"
              << "- Compiled config               : " << CompiledConfig()
              << "\n"
              << "- Weight Type                   : "
              << gcpp::TypeName(gcpp::GemmaWeightT()) << "\n"
              << "- EmbedderInput Type            : "
              << gcpp::TypeName(gcpp::EmbedderInputT()) << "\n";
          toClient(ntf.str());
        } else if (msg->type == ix::WebSocketMessageType::Close) {
          if (verbosity >= 2) {
            std::cout << "Closed connection" << std::endl;
          }
        } else if (msg->type == ix::WebSocketMessageType::Message) {
          int current_pos = 0;  // token index within the current turn
          int prompt_size{};

          // callback function invoked for each generated token.
          auto stream_token = [&abs_pos, &current_pos, &args, &gen,
                               &prompt_size, tokenizer = model.Tokenizer(),
                               verbosity, &ws, &webSocket,
                               &toClient](int token, float) {
            ++abs_pos;
            ++current_pos;
            // <= since position is incremented before
            if (current_pos <= prompt_size) {
              std::cerr << "." << std::flush;
              toClient(".");
            } else if (token == gcpp::EOS_ID) {
              if (!args.multiturn) {
                abs_pos = 0;
                if (args.deterministic) {
                  gen.seed(42);
                }
              }
              if (verbosity >= 2) {
                std::cout << "\n[ End ]\n";
              }
              toClient("%EOS%");
            } else {
              std::string token_text;
              HWY_ASSERT(
                  tokenizer->Decode(std::vector<int>{token}, &token_text));
              // +1 since position is incremented above
              if (current_pos == prompt_size + 1) {
                // first token of response
                token_text.erase(0, token_text.find_first_not_of(" \t\n"));
                if (verbosity >= 1) {
                  std::cout << "\n\n";
                }
              }
              // std::cout << token_text << std::flush;
              toClient(token_text);
            }
            return true;
          };

          std::string prompt_string = msg->str;
          if (abs_pos >= args.max_tokens) {
            ws.stop();
          } else if (prompt_string == "%q" || prompt_string == "%Q") {
            ws.stop();
          } else if (prompt_string == "%c" || prompt_string == "%C") {
            abs_pos = 0;
          } else {
            std::vector<int> prompt;
            current_pos = 0;

            if (training == ModelTraining::GEMMA_IT) {
              // For instruction-tuned models: add control tokens.
              prompt_string = "<start_of_turn>user\n" + prompt_string +
                              "<end_of_turn>\n<start_of_turn>model\n";
              if (abs_pos != 0) {
                // Prepend "<end_of_turn>" token if this is a multi-turn
                // dialogue continuation.
                prompt_string = "<end_of_turn>\n" + prompt_string;
              }
            }

            HWY_ASSERT(model.Tokenizer()->Encode(prompt_string, &prompt));

            // For both pre-trained and instruction-tuned models: prepend
            // "<bos>" token if needed.
            if (abs_pos == 0) {
              prompt.insert(prompt.begin(), 2);
            }

            prompt_size = prompt.size();

            std::cerr << "\n"
                      << "[ Reading prompt ] " << std::flush;

            if constexpr (kVerboseLogTokens) {
              for (int i = 0; i < static_cast<int>(prompt.size()); ++i) {
                fprintf(stderr, "DDD TOKEN %3d: %6d\n", i, prompt[i]);
              }
            }

            const double time_start = hwy::platform::Now();
            GenerateGemma(model, args.max_tokens, args.max_generated_tokens,
                          args.temperature, prompt, abs_pos, kv_cache, pool,
                          inner_pool, stream_token, accept_token, gen,
                          verbosity);
            const double time_end = hwy::platform::Now();
            const double tok_sec = current_pos / (time_end - time_start);
            if (verbosity >= 2) {
              std::cout << current_pos << " tokens (" << abs_pos
                        << " total tokens)"
                        << "\n"
                        << tok_sec << " tokens / sec"
                        << "\n";
            }
            std::cout << "\n\n";
          }
        }
      });

  ws.listenAndStart();
  std::cout << "Listening on " << port << std::endl;
  ws.wait();

  std::cout
      << "max_tokens (" << args.max_tokens
      << ") exceeded. Use a larger value if desired using the --max_tokens "
      << "command line flag.\n";
}

void Run(LoaderArgs& loader, InferenceArgs& inference, AppArgs& app) {
  PROFILER_ZONE("Run.misc");

  hwy::ThreadPool inner_pool(0);
  hwy::ThreadPool pool(app.num_threads);
  // For many-core, pinning threads to cores helps.
  if (app.num_threads > 10) {
    PinThreadToCore(app.num_threads - 1);  // Main thread

    pool.Run(0, pool.NumThreads(),
             [](uint64_t /*task*/, size_t thread) { PinThreadToCore(thread); });
  }

  gcpp::Gemma model(loader.tokenizer, loader.weights, loader.ModelType(), pool);

  auto kv_cache = CreateKVCache(loader.ModelType());

  if (const char* error = inference.Validate()) {
    ShowHelp(loader, inference, app);
    HWY_ABORT("\nInvalid args: %s", error);
  }

  ReplGemma(
      model, loader.ModelTraining(), kv_cache, pool, inner_pool, inference,
      app.verbosity,
      /*accept_token=*/[](int) { return true; }, app.eot_line, app.port);
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
      ShowHelp(loader, inference, app);
      return 0;
    }

    if (const char* error = loader.Validate()) {
      ShowHelp(loader, inference, app);
      HWY_ABORT("\nInvalid args: %s", error);
    }

    gcpp::Run(loader, inference, app);
  }
  PROFILER_PRINT_RESULTS();  // Must call outside the zone above.
  return 0;
}
