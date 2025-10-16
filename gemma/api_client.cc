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

// Test client for API server

#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cstring>

#include "httplib.h"
#include "nlohmann/json.hpp"
#include "gemma/gemma_args.h"

using json = nlohmann::json;

// ANSI color codes
const std::string RESET = "\033[0m";
const std::string BOLD = "\033[1m";
const std::string GREEN = "\033[32m";
const std::string BLUE = "\033[34m";
const std::string CYAN = "\033[36m";
const std::string YELLOW = "\033[33m";
const std::string RED = "\033[31m";

class APIClient {
public:
  APIClient(const std::string& host, int port, const std::string& api_key = "", const std::string& model = "gemma3-4b") 
    : host_(host), port_(port), api_key_(api_key), model_(model), use_https_(port == 443), interactive_mode_(false) {
    if (use_https_) {
      ssl_client_ = std::make_unique<httplib::SSLClient>(host, port);
      ssl_client_->set_read_timeout(60, 0);
      ssl_client_->set_write_timeout(60, 0);
      ssl_client_->enable_server_certificate_verification(false);
    } else {
      client_ = std::make_unique<httplib::Client>(host, port);
      client_->set_read_timeout(60, 0);
      client_->set_write_timeout(60, 0);
    }
  }

  // Unified request processing for both public and local APIs
  json ProcessRequest(const json& request, bool stream = true) {
    bool is_public_api = !api_key_.empty();
    
    std::string endpoint;
    if (is_public_api) {
      endpoint = stream ? "/v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse" 
                        : "/v1beta/models/gemini-2.0-flash:generateContent";
    } else {
      endpoint = stream ? "/v1beta/models/" + model_ + ":streamGenerateContent" 
                        : "/v1beta/models/" + model_ + ":generateContent";
    }
    
    // Only show verbose output in non-interactive mode
    if (!interactive_mode_) {
      std::cout << "\n" << BOLD << BLUE << "📤 POST " << endpoint << RESET << std::endl;
      std::cout << "Request: " << request.dump(2) << std::endl;
    }
    
    if (stream) {
      return ProcessStreamingRequest(request, endpoint);
    } else {
      return ProcessNonStreamingRequest(request, endpoint);
    }
  }

  void TestGenerateContent(const std::string& prompt, bool stream = true) {
    json request = CreateAPIRequest(prompt);
    json response = ProcessRequest(request, stream);
    
    if (response.contains("error")) {
      std::cerr << RED << "❌ Error: " << response["error"]["message"] << RESET << std::endl;
    }
  }

  void TestListModels() {
    std::cout << "\n" << BOLD << BLUE << "📤 GET /v1beta/models" << RESET << std::endl;
    
    httplib::Headers headers;
    if (!api_key_.empty()) {
      headers.emplace("X-goog-api-key", api_key_);
    }
    auto res = use_https_ ? ssl_client_->Get("/v1beta/models", headers) : client_->Get("/v1beta/models", headers);
    
    if (res && res->status == 200) {
      json response = json::parse(res->body);
      std::cout << GREEN << "✅ Available models:" << RESET << std::endl;
      std::cout << response.dump(2) << std::endl;
    } else {
      std::cerr << RED << "❌ Request failed" << RESET << std::endl;
    }
  }

  void InteractiveChat() {
    std::cout << "\n" << BOLD << CYAN << "💬 Interactive Chat Mode (with session)" << RESET << std::endl;
    std::cout << "Type ':gemma %q' to end.\n" << std::endl;
    
    interactive_mode_ = true;
    json messages;
    
    while (true) {
      std::cout << BOLD << BLUE << "You: " << RESET;
      std::string input;
      std::getline(std::cin, input);
      
      if (input == ":gemma %q") {
        std::cout << BOLD << YELLOW << "👋 Goodbye!" << RESET << std::endl;
        break;
      }
      
      if (input.empty()) continue;
      
      // Add user message with proper role
      json user_message = {{"parts", {{{"text", input}}}}};
      if (!api_key_.empty()) {
        user_message["role"] = "user";
      }
      messages.push_back(user_message);
      
      // Create request using unified logic
      json request = CreateAPIRequest("", messages);
      
      std::cout << BOLD << GREEN << "Assistant: " << RESET;
      
      // Use unified processing - streaming for real-time output
      json response = ProcessRequest(request, true);
      
      if (response.contains("candidates") && !response["candidates"].empty()) {
        auto& candidate = response["candidates"][0];
        if (candidate.contains("content") && candidate["content"].contains("parts")) {
          for (const auto& part : candidate["content"]["parts"]) {
            if (part.contains("text")) {
              std::string assistant_response = part["text"].get<std::string>();
              
              // For streaming, the response is already displayed in real-time
              // Just add to message history for context
              json assistant_message = {{"parts", {{{"text", assistant_response}}}}};
              if (!api_key_.empty()) {
                assistant_message["role"] = "model";
              }
              messages.push_back(assistant_message);
            }
          }
        }
      } else if (response.contains("error")) {
        std::cerr << RED << "❌ Error: " << response["error"]["message"] << RESET << std::endl;
      }
      
      std::cout << std::endl;
    }
  }

private:
  json CreateAPIRequest(const std::string& prompt, const json& messages = json::array()) {
    json request = {
      {"generationConfig", {
        {"temperature", 0.9},
        {"topK", 1},
        {"maxOutputTokens", 1024}
      }}
    };
    
    if (messages.empty()) {
      // Single prompt
      json user_message = {{"parts", {{{"text", prompt}}}}};
      if (!api_key_.empty()) {
        user_message["role"] = "user";
      }
      request["contents"] = json::array({user_message});
    } else {
      // Use provided message history
      request["contents"] = messages;
    }
    
    return request;
  }

  json ProcessNonStreamingRequest(const json& request, const std::string& endpoint) {
    httplib::Headers headers = {{"Content-Type", "application/json"}};
    if (!api_key_.empty()) {
      headers.emplace("X-goog-api-key", api_key_);
    }
    
    auto res = use_https_ ? ssl_client_->Post(endpoint, headers, request.dump(), "application/json") 
                          : client_->Post(endpoint, headers, request.dump(), "application/json");
    
    if (res && res->status == 200) {
      json response = json::parse(res->body);
      if (!interactive_mode_) {
        std::cout << "\n" << BOLD << GREEN << "📥 Response:" << RESET << std::endl;
        std::cout << response.dump(2) << std::endl;
      }
      return response;
    } else {
      json error_response = {
        {"error", {
          {"message", "Request failed"},
          {"status", res ? res->status : -1}
        }}
      };
      if (res && !res->body.empty()) {
        error_response["error"]["details"] = res->body;
      }
      std::cerr << RED << "❌ Request failed. Status: " << (res ? res->status : -1) << RESET << std::endl;
      return error_response;
    }
  }

  json ProcessStreamingRequest(const json& request, const std::string& endpoint) {
    std::string accumulated_response;
    
    // Use same SSE logic for both public and local APIs
    httplib::Request req;
    req.method = "POST";
    req.path = endpoint;
    req.set_header("Content-Type", "application/json");
    if (!api_key_.empty()) {
      req.set_header("X-goog-api-key", api_key_);
    }
    req.body = request.dump();
    
    req.content_receiver = [&accumulated_response, this](const char* data, size_t data_length, uint64_t offset, uint64_t total_length) -> bool {
        std::string chunk(data, data_length);
        std::istringstream stream(chunk);
        std::string line;
        
        while (std::getline(stream, line)) {
          if (line.substr(0, 6) == "data: ") {
            std::string event_data = line.substr(6);
            
            if (event_data == "[DONE]") {
              if (!interactive_mode_) {
                std::cout << "\n\n" << GREEN << "✅ Generation complete!" << RESET << std::endl;
              }
            } else {
              try {
                json event = json::parse(event_data);
                if (event.contains("candidates") && !event["candidates"].empty()) {
                  auto& candidate = event["candidates"][0];
                  if (candidate.contains("content") && candidate["content"].contains("parts")) {
                    for (const auto& part : candidate["content"]["parts"]) {
                      if (part.contains("text")) {
                        std::string text = part["text"].get<std::string>();
                        std::cout << text << std::flush;
                        accumulated_response += text;
                      }
                    }
                  }
                }
              } catch (const json::exception& e) {
                // Skip parse errors
              }
            }
          }
        }
        return true;
      };
    
    httplib::Response res;
    httplib::Error error;
    bool success = use_https_ ? ssl_client_->send(req, res, error) : client_->send(req, res, error);
    
    if (res.status == 200 && !accumulated_response.empty()) {
      return json{
        {"candidates", {{
          {"content", {
            {"parts", {{{"text", accumulated_response}}}}
          }}
        }}}
      };
    } else {
      json error_response = {
        {"error", {
          {"message", "Streaming request failed"},
          {"status", res.status}
        }}
      };
      if (!res.body.empty()) {
        error_response["error"]["details"] = res.body;
      }
      std::cerr << RED << "❌ Streaming request failed. Status: " << res.status << RESET << std::endl;
      return error_response;
    }
  }

private:
  std::unique_ptr<httplib::Client> client_;
  std::unique_ptr<httplib::SSLClient> ssl_client_;
  std::string host_;
  int port_;
  std::string api_key_;
  std::string model_;
  bool use_https_;
  bool interactive_mode_;
};

int main(int argc, char* argv[]) {
  gcpp::ClientArgs client_args(argc, argv);
  
  if (gcpp::HasHelp(argc, argv)) {
    std::cout << "\nAPI Client for gemma.cpp\n";
    std::cout << "========================\n\n";
    client_args.Help();
    std::cout << std::endl;
    std::cout << "Environment Variables:" << std::endl;
    std::cout << "  GOOGLE_API_KEY : Automatically use public Google API if set" << std::endl;
    return 0;
  }
  
  // Check for GOOGLE_API_KEY environment variable
  const char* env_api_key = std::getenv("GOOGLE_API_KEY");
  if (env_api_key != nullptr && strlen(env_api_key) > 0) {
    client_args.api_key = env_api_key;
    client_args.host = "generativelanguage.googleapis.com";
    client_args.port = 443;
  }
  
  // Handle API key override
  if (!client_args.api_key.empty()) {
    client_args.host = "generativelanguage.googleapis.com";
    client_args.port = 443;
  }
  
  std::cout << BOLD << YELLOW << "🚀 Testing API Server at " 
            << client_args.host << ":" << client_args.port << RESET << std::endl;
  
  try {
    APIClient client(client_args.host, client_args.port, client_args.api_key, client_args.model);
    
    if (client_args.interactive) {
      client.InteractiveChat();
    } else {
      client.TestListModels();
      client.TestGenerateContent(client_args.prompt, true);
    }
    
  } catch (const std::exception& e) {
    std::cerr << RED << "❌ Error: " << e.what() << RESET << std::endl;
    std::cerr << "Make sure the API server is running:" << std::endl;
    std::cerr << "  ./build/gemma_api_server --tokenizer <path> --weights <path>" << std::endl;
    return 1;
  }
  
  return 0;
}
