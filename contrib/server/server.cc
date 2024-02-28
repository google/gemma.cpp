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

// GRPC interface to gemma.

#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <thread>  // NOLINT
#include <vector>

// copybara:import_next_line:gemma_cpp
#include "compression/compress.h"
// copybara:import_next_line:gemma_cpp
#include "gemma.h"    // Gemma
// copybara:import_next_line:gemma_cpp
#include "util/app.h"
// copybara:import_next_line:gemma_cpp
#include "util/args.h"  // HasHelp
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/highway.h"
#include "hwy/per_target.h"
#include "hwy/profiler.h"
#include "hwy/timer.h"


#include <grpc/grpc.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>

#include "pb/llm.grpc.pb.h"

namespace gcpp {

void ShowHelp(gcpp::LoaderArgs& loader, gcpp::InferenceArgs& inference,
              gcpp::AppArgs& app) {
  fprintf(stderr,
          "\ngemma-server\n---------\n\nTo run gemma-server, you need to "
          "specify 3 required model loading arguments: --tokenizer, "
          "--compressed_weights, "
          "and --model.\n\nModel Loading Arguments\n\n");
  loader.Help();
  fprintf(stderr, "\nInference Arguments\n\n");
  inference.Help();
  fprintf(stderr, "\nApplication Arguments\n\n");
  app.Help();
  fprintf(stderr, "\n\n");
}

void ShowConfig(LoaderArgs& loader, InferenceArgs& inference, AppArgs& app) {
  loader.Print(app.verbosity);
  inference.Print(app.verbosity);
  app.Print(app.verbosity);

  if (app.verbosity >= 2) {
    time_t now = time(nullptr);
    char* dt = ctime(&now);  // NOLINT
    std::cout << "Date & Time                   : " << dt
              << "Prefill Token Batch Size      : " << gcpp::kPrefillBatchSize
              << "\n"
              << "Hardware concurrency          : "
              << std::thread::hardware_concurrency() << std::endl
              << "Instruction set               : "
              << hwy::TargetName(hwy::DispatchedTarget()) << " ("
              << hwy::VectorBytes() * 8 << " bits)" << "\n"
              << "Weight Type                   : "
              << gcpp::TypeName(gcpp::WeightT()) << "\n"
              << "EmbedderInput Type            : "
              << gcpp::TypeName(gcpp::EmbedderInputT()) << "\n";
  }
}


// LLMImpl implements the grpc service.
class LLMImpl final : public ::llm::LLM::Service {
  const InferenceArgs& _args;
  gcpp::Gemma& _model;
  hwy::ThreadPool& _pool;
  hwy::ThreadPool& _inner_pool;
  int _verbosity;
 
 public:
  explicit LLMImpl(gcpp::Gemma& model, hwy::ThreadPool& pool,
               hwy::ThreadPool& inner_pool, const InferenceArgs& args,
               int verbosity) 
               : _model(model),
                 _pool(pool),
                 _inner_pool(inner_pool),
                 _args(args),
                 _verbosity(verbosity) {
  }

  ::grpc::Status Converse(::grpc::ServerContext* context,
                          ::grpc::ServerReaderWriter< ::llm::ConverseResponse, ::llm::ConverseRequest>* stream);
};


::grpc::Status LLMImpl::Converse(::grpc::ServerContext* context,
                          ::grpc::ServerReaderWriter< ::llm::ConverseResponse, ::llm::ConverseRequest>* stream) {
  PROFILER_ZONE("Gen.misc");
  int abs_pos = 0;      // absolute token index over all turns
  int current_pos = 0;  // token index within the current turn
  int prompt_size{};

  auto args = this->_args;
  auto verbosity = this->_verbosity;
  gcpp::Gemma& model = this->_model;

  const gcpp::AcceptFunc& accept_token = [](int) { return true; };

  std::mt19937 gen;
  if (args.deterministic) {
    gen.seed(42);
  } else {
    std::random_device rd;
    gen.seed(rd());
  }

  // callback function invoked for each generated token.
  auto stream_token = [&stream, &abs_pos, &current_pos, &args, &gen, &prompt_size,
                       tokenizer = &model.Tokenizer(),
                       verbosity](int token, float) {
    ++abs_pos;
    ++current_pos;
    if (current_pos < prompt_size) {
      // ignore tokens that are part of the prompt
    } else if (token == gcpp::EOS_ID) {
      if (!args.multiturn) {
        abs_pos = 0;
        if (args.deterministic) {
          gen.seed(42);
        }
      }
      if (verbosity >= 2) {
        std::cout << "\n[ End ]" << std::endl;
      }
    } else {
      std::string token_text;
      HWY_ASSERT(tokenizer->Decode(std::vector<int>{token}, &token_text).ok());
      // +1 since position is incremented above
      if (current_pos == prompt_size + 1) {
        // first token of response
        token_text.erase(0, token_text.find_first_not_of(" \t\n"));
      }
      ::llm::ConverseResponse converse_response;
      converse_response.add_text(token_text);
      if (verbosity >= 2) {
        std::cerr << "grpc-send: " << converse_response.DebugString() << std::endl;
      }
      if (!stream->Write(converse_response)) {
        return false; // TODO(justinsb): handle stream closed / error
      }
    }
    return true;
  };

  while (abs_pos < args.max_tokens) {
    std::string prompt_string;
    std::vector<int> prompt;
    current_pos = 0;
    {
      PROFILER_ZONE("Gen.input");
      ::llm::ConverseRequest converse_request;
      if (! stream->Read(&converse_request)) {
        return ::grpc::Status::OK;
      }
      if (verbosity >= 2) {
        std::cerr << "grpc-recv: " << converse_request.DebugString() << std::endl;
      }
      // if (converse_request.text().empty()) {
      //   if (verbosity >= 1) {
      //     std::cout << std::endl;
      //   }
      //   return ::grpc::Status::OK;
      // }
      prompt_string = converse_request.text();
    }

    // Client should just grpc close the stream instead
    // if (prompt_string == "%q" || prompt_string == "%Q") {
    //   return ::grpc::Status::OK;
    // }

    if (this->_model.model_training == ModelTraining::GEMMA_IT) {
      // For instruction-tuned models: add control tokens.
      prompt_string = "<start_of_turn>user\n" + prompt_string +
                      "<end_of_turn>\n<start_of_turn>model\n";
      if (abs_pos > 0) {
        // Prepend "<end_of_turn>" token if this is a multi-turn dialogue
        // continuation.
        prompt_string = "<end_of_turn>\n" + prompt_string;
      }
    }

    HWY_ASSERT(this->_model.Tokenizer().Encode(prompt_string, &prompt).ok());

    // For both pre-trained and instruction-tuned models: prepend "<bos>" token
    // if needed.
    if (abs_pos == 0) {
      prompt.insert(prompt.begin(), 2);
    }

    prompt_size = prompt.size();

    const double time_start = hwy::platform::Now();
    GenerateGemma(model, args, prompt, abs_pos, this->_pool, this->_inner_pool, stream_token,
                  accept_token, gen, verbosity);
    const double time_end = hwy::platform::Now();
    const double tok_sec = current_pos / (time_end - time_start);
    if (verbosity >= 2) {
      std::cout << current_pos << " tokens (" << abs_pos << " total tokens)"
                << std::endl
                << tok_sec << " tokens / sec" << std::endl;
    }
    ::llm::ConverseResponse converse_response;
    converse_response.set_end_of_response(true);
    if (verbosity >= 2) {
      std::cerr << "grpc-send: " << converse_response.DebugString() << std::endl;
    }
    if (!stream->Write(converse_response)) {
      return ::grpc::Status::OK; // TODO(justinsb): which status to send?
    }
  }

  // TODO(justinsb): send error code / message?
  std::cout
      << "max_tokens (" << args.max_tokens
      << ") exceeded. Use a larger value if desired using the --max_tokens "
      << "command line flag.\n";

  return ::grpc::Status::OK;
}


void RunServer(LoaderArgs& loader, InferenceArgs& inference, AppArgs& app) {
  PROFILER_ZONE("RunServer.misc");

  hwy::ThreadPool inner_pool(0);
  hwy::ThreadPool pool(app.num_threads);
  // For many-core, pinning threads to cores helps.
  if (app.num_threads > 10) {
    PinThreadToCore(app.num_threads - 1);  // Main thread

    pool.Run(0, pool.NumThreads(),
             [](uint64_t /*task*/, size_t thread) { PinThreadToCore(thread); });
  }

  gcpp::Gemma model(loader, pool);

  if (const char* error = inference.Validate()) {
    ShowHelp(loader, inference, app);
    HWY_ABORT("\nInvalid args: %s", error);
  }

  if (app.verbosity >= 1) {
    ShowConfig(loader, inference, app);
  }

  // TODO(justinsb): make server_address configurable
  std::string server_address("0.0.0.0:50051");
  LLMImpl service(model, pool, inner_pool, inference, app.verbosity);

  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<::grpc::Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();
}

}  // namespace gcpp

int main(int argc, char** argv) {
  {
    PROFILER_ZONE("Startup.misc");

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

    gcpp::RunServer(loader, inference, app);
  }
  PROFILER_PRINT_RESULTS();  // Must call outside the zone above.
  return 0;
}
