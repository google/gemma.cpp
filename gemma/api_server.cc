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

// HTTP API server for gemma.cpp with SSE support

#include <stdio.h>
#include <signal.h>

#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <unordered_map>

// HTTP server library
#undef CPPHTTPLIB_OPENSSL_SUPPORT
#undef CPPHTTPLIB_ZLIB_SUPPORT
#include "httplib.h"

// JSON library
#include "nlohmann/json.hpp"

#include "compression/types.h"
#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "gemma/tokenizer.h"
#include "ops/matmul.h"
#include "util/args.h"
#include "hwy/base.h"
#include "hwy/profiler.h"

using json = nlohmann::json;

namespace gcpp {

static std::atomic<bool> server_running{true};

// Server state holding model and KV caches
struct ServerState {
  std::unique_ptr<Gemma> gemma;
  MatMulEnv* env;
  ThreadingContext* ctx;

  // Session-based KV cache storage
  struct Session {
    std::unique_ptr<KVCache> kv_cache;
    size_t abs_pos = 0;
    std::chrono::steady_clock::time_point last_access;
  };

  std::unordered_map<std::string, Session> sessions;
  std::mutex sessions_mutex;
  std::mutex inference_mutex;

  // Cleanup old sessions after 30 minutes of inactivity
  void CleanupOldSessions() {
    std::lock_guard<std::mutex> lock(sessions_mutex);
    auto now = std::chrono::steady_clock::now();
    for (auto it = sessions.begin(); it != sessions.end();) {
      if (now - it->second.last_access > std::chrono::minutes(30)) {
        it = sessions.erase(it);
      } else {
        ++it;
      }
    }
  }

  // Get or create session with KV cache
  Session& GetOrCreateSession(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(sessions_mutex);
    auto& session = sessions[session_id];
    if (!session.kv_cache) {
      session.kv_cache = std::make_unique<KVCache>(gemma->Config(), InferenceArgs(), env->ctx.allocator);
    }
    session.last_access = std::chrono::steady_clock::now();
    return session;
  }
};

// Generate a unique session ID
std::string GenerateSessionId() {
  static std::atomic<uint64_t> counter{0};
  std::stringstream ss;
  ss << "session_" << std::hex
     << std::chrono::steady_clock::now().time_since_epoch().count() << "_"
     << counter.fetch_add(1);
  return ss.str();
}

// Wraps messages with start_of_turn markers - handles both with and without roles
std::string WrapMessagesWithTurnMarkers(const json& contents) {
  std::string prompt;

  for (const auto& content : contents) {
    if (content.contains("parts")) {
      // Check if role is specified (public API format) or not (local format)
      std::string role = content.value("role", "");

      for (const auto& part : content["parts"]) {
        if (part.contains("text")) {
          std::string text = part["text"];

          if (role == "user") {
            prompt += "<start_of_turn>user\n" + text + "\n<start_of_turn>model\n";
          } else if (role == "model") {
            prompt += text + "\n";
          } else if (role.empty()) {
            // Local format without roles - for now, treat as user input
            prompt += "<start_of_turn>user\n" + text + "\n<start_of_turn>model\n";
          }
        }
      }
    }
  }

  return prompt;
}

// Parse generation config
RuntimeConfig ParseGenerationConfig(const json& request) {
  RuntimeConfig config;
  config.verbosity = 0;

  // Set defaults matching public API
  config.temperature = 1.0f;
  config.top_k = 1;
  config.max_generated_tokens = 8192;

  if (request.contains("generationConfig")) {
    auto& gen_config = request["generationConfig"];

    if (gen_config.contains("temperature")) {
      config.temperature = gen_config["temperature"].get<float>();
    }
    if (gen_config.contains("topK")) {
      config.top_k = gen_config["topK"].get<int>();
    }
    if (gen_config.contains("maxOutputTokens")) {
      config.max_generated_tokens = gen_config["maxOutputTokens"].get<size_t>();
    }
  }

  return config;
}

// Unified response formatter - creates consistent format regardless of request type
json CreateAPIResponse(const std::string& text, bool is_streaming_chunk = false) {
  json response = {
    {"candidates", {{
      {"content", {
        {"parts", {{{"text", text}}}},
        {"role", "model"}
      }},
      {"index", 0}
    }}},
    {"promptFeedback", {{"safetyRatings", json::array()}}}
  };

  // Only add finishReason for non-streaming chunks
  if (!is_streaming_chunk) {
    response["candidates"][0]["finishReason"] = "STOP";
  }

  return response;
}

// Handle generateContent endpoint (non-streaming)
void HandleGenerateContentNonStreaming(ServerState& state, const httplib::Request& req, httplib::Response& res) {
  try {
    json request = json::parse(req.body);

    // Get or create session
    std::string session_id = request.value("sessionId", GenerateSessionId());
    auto& session = state.GetOrCreateSession(session_id);

    // Extract prompt from API format
    std::string prompt;
    if (request.contains("contents")) {
      prompt = WrapMessagesWithTurnMarkers(request["contents"]);
    } else {
      res.status = 400;
      res.set_content(json{{"error", {{"message", "Missing 'contents' field"}}}}.dump(), "application/json");
      return;
    }

    // Lock for inference
    std::lock_guard<std::mutex> lock(state.inference_mutex);

    // Set up runtime config
    RuntimeConfig runtime_config = ParseGenerationConfig(request);

    // Collect full response
    std::string full_response;
    runtime_config.stream_token = [&full_response](int token, float) {
      // Skip EOS token
      return true;
    };

    // Tokenize prompt
    std::vector<int> tokens = WrapAndTokenize(
        state.gemma->Tokenizer(), state.gemma->ChatTemplate(),
        state.gemma->Config().wrapping, session.abs_pos, prompt);

    // Run inference with KV cache
    TimingInfo timing_info = {.verbosity = 0};
    size_t prefix_end = 0;

    // Temporarily redirect output to capture response
    std::stringstream output;
    runtime_config.stream_token = [&output, &state, &session, &tokens](int token, float) {
      // Skip prompt tokens
      if (session.abs_pos < tokens.size()) {
        session.abs_pos++;
        return true;
      }

      session.abs_pos++;

      // Check for EOS
      if (state.gemma->Config().IsEOS(token)) {
        return true;
      }

      // Decode token
      std::string token_text;
      state.gemma->Tokenizer().Decode(std::vector<int>{token}, &token_text);
      output << token_text;

      return true;
    };

    state.gemma->Generate(runtime_config, tokens, session.abs_pos, prefix_end,
                          *session.kv_cache, *state.env, timing_info);

    // Create response
    json response = CreateAPIResponse(output.str(), false);
    response["usageMetadata"] = {
      {"promptTokenCount", tokens.size()},
      {"candidatesTokenCount", session.abs_pos - tokens.size()},
      {"totalTokenCount", session.abs_pos}
    };

    res.set_content(response.dump(), "application/json");

  } catch (const json::exception& e) {
    res.status = 400;
    res.set_content(
        json{{"error",
              {{"message", std::string("JSON parsing error: ") + e.what()}}}}
            .dump(),
        "application/json");
  } catch (const std::exception& e) {
    res.status = 500;
    res.set_content(
        json{{"error", {{"message", std::string("Server error: ") + e.what()}}}}
            .dump(),
        "application/json");
  }
}

// Handle streamGenerateContent endpoint with SSE)
void HandleGenerateContentStreaming(ServerState& state, const httplib::Request& req, httplib::Response& res) {
  try {
    json request = json::parse(req.body);

    // Get or create session
    std::string session_id = request.value("sessionId", GenerateSessionId());
    auto& session = state.GetOrCreateSession(session_id);

    // Extract prompt from API format
    std::string prompt;
    if (request.contains("contents")) {
      prompt = WrapMessagesWithTurnMarkers(request["contents"]);
    } else {
      res.status = 400;
      res.set_content(json{{"error", {{"message", "Missing 'contents' field"}}}}.dump(), "application/json");
      return;
    }

    // Set up SSE headers
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");
    res.set_header("X-Session-Id", session_id);

    // Set up chunked content provider for SSE
    res.set_chunked_content_provider(
      "text/event-stream",
      [&state, request, prompt, session_id](size_t offset, httplib::DataSink& sink) {
        try {
          // Lock for inference
          std::lock_guard<std::mutex> lock(state.inference_mutex);
          auto& session = state.GetOrCreateSession(session_id);

          // Set up runtime config
          RuntimeConfig runtime_config = ParseGenerationConfig(request);

          // Tokenize prompt
          std::vector<int> tokens = WrapAndTokenize(
              state.gemma->Tokenizer(), state.gemma->ChatTemplate(),
              state.gemma->Config().wrapping, session.abs_pos, prompt);

          // Stream token callback
          std::string accumulated_text;
          auto stream_token = [&](int token, float) {
            // Skip prompt tokens
            if (session.abs_pos < tokens.size()) {
              session.abs_pos++;
              return true;
            }

            session.abs_pos++;

            // Check for EOS
            if (state.gemma->Config().IsEOS(token)) {
              return true;
            }

            // Decode token
            std::string token_text;
            state.gemma->Tokenizer().Decode(std::vector<int>{token}, &token_text);
            accumulated_text += token_text;

            // Send SSE event using unified formatter
            json event = CreateAPIResponse(token_text, true);

            std::string sse_data = "data: " + event.dump() + "\n\n";
            sink.write(sse_data.data(), sse_data.size());

            return true;
          };

          runtime_config.stream_token = stream_token;

          // Run inference with KV cache
          TimingInfo timing_info = {.verbosity = 0};
          size_t prefix_end = 0;

          state.gemma->Generate(runtime_config, tokens, session.abs_pos,
                                prefix_end, *session.kv_cache, *state.env,
                                timing_info);

          // Send final event using unified formatter
          json final_event = CreateAPIResponse("", false);
          final_event["usageMetadata"] = {
            {"promptTokenCount", tokens.size()},
            {"candidatesTokenCount", session.abs_pos - tokens.size()},
            {"totalTokenCount", session.abs_pos}
          };

          std::string final_sse = "data: " + final_event.dump() + "\n\n";
          sink.write(final_sse.data(), final_sse.size());

          // Send done event
          sink.write("data: [DONE]\n\n", 15);

          // Ensure all data is sent
          sink.done();
          return false;  // End streaming

        } catch (const std::exception& e) {
          json error_event = {{"error", {{"message", e.what()}}}};
          std::string error_sse = "data: " + error_event.dump() + "\n\n";
          sink.write(error_sse.data(), error_sse.size());
          return false;
        }
      }
    );

  } catch (const json::exception& e) {
    res.status = 400;
    res.set_content(
        json{{"error",
              {{"message", std::string("JSON parsing error: ") + e.what()}}}}
            .dump(),
        "application/json");
  }
}

// Handle models list endpoint
void HandleListModels(ServerState& state, const InferenceArgs& inference, const httplib::Request& req, httplib::Response& res) {
  json response = {
    {"models", {{
      {"name", "models/" + inference.model},
      {"version", "001"},
      {"displayName", inference.model},
      {"description", inference.model + " model running locally"},
      {"inputTokenLimit", 8192},
      {"outputTokenLimit", 8192},
      {"supportedGenerationMethods", json::array({"generateContent", "streamGenerateContent"})},
      {"temperature", 1.0},
      {"topK", 1}
    }}}
  };

  res.set_content(response.dump(), "application/json");
}

// void HandleShutdown(int signal) {
//   std::cerr << "\nShutting down server..." << std::endl;
//   server_running = false;
// }

void RunServer(const LoaderArgs& loader, const ThreadingArgs& threading,
               const InferenceArgs& inference) {
  std::cerr << "Loading model..." << std::endl;

  // Initialize model
  ThreadingContext ctx(threading);
  MatMulEnv env(ctx);

  ServerState state;
  state.gemma = std::make_unique<Gemma>(loader, inference, ctx);
  state.env = &env;
  state.ctx = &ctx;

  httplib::Server server;

  // Set up routes
  server.Get("/", [&inference](const httplib::Request&, httplib::Response& res) {
    res.set_content("API Server (gemma.cpp) - Use POST /v1beta/models/" + inference.model + ":generateContent", "text/plain");
  });

  // API endpoints
  server.Get("/v1beta/models", [&state, &inference](const httplib::Request& req, httplib::Response& res) {
    HandleListModels(state, inference, req, res);
  });

  std::string model_endpoint = "/v1beta/models/" + inference.model;
  server.Post(model_endpoint + ":generateContent", [&state](const httplib::Request& req, httplib::Response& res) {
    HandleGenerateContentNonStreaming(state, req, res);
  });

  server.Post(model_endpoint + ":streamGenerateContent", [&state](const httplib::Request& req, httplib::Response& res) {
    HandleGenerateContentStreaming(state, req, res);
  });

  // Periodic cleanup of old sessions
  std::thread cleanup_thread([&state]() {
    while (server_running) {
      std::this_thread::sleep_for(std::chrono::minutes(5));
      state.CleanupOldSessions();
    }
  });

  std::cerr << "Starting API server on port " << inference.port << std::endl;
  std::cerr << "Model loaded successfully" << std::endl;
  std::cerr << "Endpoints:" << std::endl;
  std::cerr << "  POST /v1beta/models/" << inference.model << ":generateContent" << std::endl;
  std::cerr << "  POST /v1beta/models/" << inference.model << ":streamGenerateContent (SSE)" << std::endl;
  std::cerr << "  GET  /v1beta/models" << std::endl;

  if (!server.listen("0.0.0.0", inference.port)) {
    std::cerr << "Failed to start server on port " << inference.port << std::endl;
  }

  cleanup_thread.join();
}

}  // namespace gcpp

int main(int argc, char** argv) {
  gcpp::InternalInit();

  gcpp::LoaderArgs loader(argc, argv);
  gcpp::ThreadingArgs threading(argc, argv);
  gcpp::InferenceArgs inference(argc, argv);

  if (gcpp::HasHelp(argc, argv)) {
    std::cerr << "\n\nAPI server for gemma.cpp\n";
    std::cout << "========================\n\n";
    std::cerr << "Usage: " << argv[0] << " --weights <path> --tokenizer <path> [options]\n";
    std::cerr << "\nOptions:\n";
    std::cerr << "  --port PORT      Server port (default: 8080)\n";
    std::cerr << "  --model MODEL    Model name for endpoints (default: gemma3-4b)\n";
    std::cerr << "\n";
    std::cerr << "\n*Model Loading Arguments*\n\n";
    loader.Help();
    std::cerr << "\n*Threading Arguments*\n\n";
    threading.Help();
    std::cerr << "\n*Inference Arguments*\n\n";
    inference.Help();
    std::cerr << "\n";
    return 0;
  }

  // Arguments are now handled by InferenceArgs

  // // Set up signal handler
  // signal(SIGINT, gcpp::HandleShutdown);
  // signal(SIGTERM, gcpp::HandleShutdown);

  gcpp::RunServer(loader, threading, inference);

  return 0;
}
