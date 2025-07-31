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

#include <Python.h>
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "evals/benchmark_helper.h"
#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "util/threading_context.h"
#include "hwy/base.h"

namespace py = pybind11;

static void RemoveTrailingZeros(std::vector<int> &vec) {
  auto it =
      std::find_if(vec.rbegin(), vec.rend(), [](int v) { return v != 0; });
  vec.erase(it.base(), vec.end());
}

// Wrapper around GemmaEnv to expose to Python.
class GemmaModel {
 public:
  GemmaModel(const gcpp::LoaderArgs& loader,
             const gcpp::ThreadingArgs& threading,
             const gcpp::InferenceArgs& inference)
      : env_(loader, threading, inference), last_prob_(0.0f) {}

  // Generates a single example, given a prompt and a callback to stream the
  // generated tokens.
  void GenerateEx(std::string prompt, gcpp::StreamFunc stream,
                  size_t max_generated_tokens, float temperature, float seed,
                  gcpp::AcceptFunc accept, bool skip_prompt) {
    env_.MutableGen().seed(seed);
    std::vector<int> prompt_tokens = env_.WrapAndTokenize(prompt);
    gcpp::RuntimeConfig& config = env_.MutableConfig();
    config.max_generated_tokens = max_generated_tokens;
    config.temperature = temperature;
    config.verbosity = 0;
    config.accept_token = accept;
    // If skip_prompt is true, we skip the prompt tokens and only stream the
    // generated tokens.
    int count_down = prompt_tokens.size();
    auto stream_with_skipping = [&stream, &count_down](int token, float score) {
      if (count_down > 0) {
        count_down--;
        return true;
      }
      return stream(token, score);
    };
    env_.QueryModel(prompt_tokens, skip_prompt ? stream_with_skipping : stream);
  }

  // Generates a single example, given a prompt, and returns the result.
  std::string Generate(std::string prompt, size_t max_generated_tokens,
                       float temperature, float seed,
                       const std::vector<std::string>& accept,
                       const std::vector<std::string>& end) {
    std::set<int> end_token_set{};
    for (const std::string& end_token : end) {
      std::vector<int> end_token_ids = env_.Tokenize(end_token);
      end_token_set.insert(end_token_ids.begin(), end_token_ids.end());
    }

    std::vector<int> predicted_token_ids;
    predicted_token_ids.reserve(max_generated_tokens);
    std::vector<int> prompt_token_ids = env_.WrapAndTokenize(prompt);
    int generated = 0;
    auto stream_token = [&generated, &prompt_token_ids, &predicted_token_ids,
                         &end_token_set, this](int token, float proba) {
      ++generated;
      if (generated > prompt_token_ids.size()) {
        predicted_token_ids.push_back(token);
        if (!end_token_set.empty()) {
          return end_token_set.find(token) == end_token_set.end();
        }
      }
      last_prob_ = proba;
      return true;
    };

    std::set<int> accept_token_set{};
    for (const std::string& accept_token : accept) {
      std::vector<int> accept_token_ids = env_.Tokenize(accept_token);
      accept_token_set.insert(accept_token_ids.begin(), accept_token_ids.end());
    }

    auto accept_token = [&predicted_token_ids, &prompt_token_ids,
                         &accept_token_set](int token, float) {
      // i.e. we have no constraints on accepted tokens
      if (accept_token_set.empty()) {
        return true;
      }

      if (predicted_token_ids.size() >= prompt_token_ids.size()) {
        return accept_token_set.find(token) != accept_token_set.end();
      } else {
        // auto-accept prompt tokens
        return true;
      }
    };

    env_.MutableGen().seed(seed);
    gcpp::RuntimeConfig& config = env_.MutableConfig();
    config.max_generated_tokens = max_generated_tokens;
    config.temperature = temperature;
    config.verbosity = 0;
    config.accept_token = accept_token;

    env_.QueryModel(prompt_token_ids, stream_token);

    if (!predicted_token_ids.empty()) {
      return env_.StringFromTokens(predicted_token_ids);
    } else {
      return "";
    }
  }

  // Generates a batch of examples, given a list of prompts, and returns the
  // results.
  std::vector<std::string> GenerateBatch(const std::vector<std::string>& inputs,
                                         size_t max_generated_tokens,
                                         float temperature, float seed,
                                         size_t top_k) {
    gcpp::RuntimeConfig& config = env_.MutableConfig();
    config.max_generated_tokens = max_generated_tokens;
    config.temperature = temperature;
    config.top_k = top_k;
    config.verbosity = 0;
    env_.MutableGen().seed(seed);

    std::vector<gcpp::QueryResult> outputs = env_.BatchQueryModel(inputs);
    std::vector<std::string> result;
    result.reserve(outputs.size());
    for (const gcpp::QueryResult& output : outputs) {
      result.push_back(output.response.substr(output.response_start_pos));
    }
    return result;
  }

  // For a PaliGemma model, sets the image to run on. Subseqent calls to
  // Generate* will use this image. Throws an error for other models.
  void SetImage(const py::array_t<float, py::array::c_style |
                                             py::array::forcecast>& image) {
    const gcpp::Gemma& gemma = *env_.GetGemma();
    const gcpp::ModelConfig& config = gemma.Config();
    if (config.wrapping != gcpp::PromptWrapping::PALIGEMMA &&
        config.wrapping != gcpp::PromptWrapping::GEMMA_VLM) {
      throw std::invalid_argument("Not a PaliGemma model.");
    }
    py::buffer_info buffer = image.request();
    if (buffer.ndim != 3 || buffer.shape[2] != 3)
      throw std::runtime_error(
          "Expected a 3D numpy array with shape (height, width, 3)");
    int height = buffer.shape[0];
    int width = buffer.shape[1];
    float* ptr = static_cast<float*>(buffer.ptr);
    gcpp::Image c_image;
    c_image.Set(height, width, ptr);
    const size_t image_size = config.vit_config.image_size;
    c_image.Resize(image_size, image_size);
    image_tokens_.reset(new gcpp::ImageTokens(
        "image_tokens",
        gcpp::Extents2D(config.vit_config.seq_len, config.model_dim),
        env_.MutableEnv().ctx.allocator, gcpp::MatPadding::kOdd));
    gcpp::RuntimeConfig runtime_config = {.gen = &env_.MutableGen(),
                                          .verbosity = 0};
    gemma.GenerateImageTokens(runtime_config, env_.MutableKVCache().SeqLen(),
                              c_image, *image_tokens_, env_.MutableEnv());
  }

  // Generates a response to the given prompt, using the last set image.
  // Uses the prompt_tokens if provided, otherwise tokenizes the prompt string.
  std::pair<std::string, std::vector<int>> GenerateWithImage(
      std::string prompt, size_t max_generated_tokens, float temperature,
      float seed, gcpp::AcceptFunc accept, std::vector<int> prompt_tokens) {
    if (!image_tokens_) throw std::invalid_argument("No image set.");
    const gcpp::Gemma& model = *env_.GetGemma();
    env_.MutableGen().seed(seed);
    gcpp::RuntimeConfig& config = env_.MutableConfig();
    config.max_generated_tokens = max_generated_tokens;
    config.temperature = temperature;
    config.verbosity = 0;
    config.accept_token = accept;
    config.image_tokens = image_tokens_.get();
    std::vector<int> tokens;
    if (!prompt_tokens.empty()) {
      if (!prompt.empty()) {
        throw std::invalid_argument(
            "Cannot pass both prompt and prompt_tokens.");
      }
      tokens = prompt_tokens;
      RemoveTrailingZeros(tokens);  // Remove padding, if any.
    } else {
      tokens = env_.WrapAndTokenize(prompt);
    }
    tokens.insert(tokens.begin(), image_tokens_->Rows(), 0);
    size_t num_tokens = tokens.size();
    size_t prefix_end = num_tokens;
    config.prefill_tbatch_size = num_tokens;
    int count_down = static_cast<int>(num_tokens);
    std::vector<int> response_tokens;
    auto stream_token = [&](int token, float) {
      if (count_down > 0) {
        count_down--;
        return true;
      }
      response_tokens.push_back(token);
      return true;
    };
    config.stream_token = stream_token;
    gcpp::TimingInfo timing_info = {.verbosity = 0};
    model.Generate(config, tokens, /*pos=*/0, prefix_end, env_.MutableKVCache(),
                   env_.MutableEnv(), timing_info);
    std::string response;
    model.Tokenizer().Decode(response_tokens, &response);
    return {response, response_tokens};
  }

  float GetLastProb() const { return last_prob_; }

  std::string Detokenize(const std::vector<int>& token_ids) const {
    return env_.StringFromTokens(token_ids);
  }

  bool ModelIsLoaded() const { return env_.GetGemma() != nullptr; }

 private:
  gcpp::GemmaEnv env_;
  std::unique_ptr<gcpp::ImageTokens> image_tokens_;
  float last_prob_;
};

PYBIND11_MODULE(gemma, mod) {
  py::class_<GemmaModel>(mod, "GemmaModel")
      .def(py::init([](const std::string& tokenizer, const std::string& weights,
                       size_t max_threads) {
             const gcpp::LoaderArgs loader(tokenizer, weights);
             gcpp::ThreadingArgs threading;
             threading.max_lps = max_threads;
             gcpp::InferenceArgs inference;
             inference.max_generated_tokens = 512;
             auto gemma =
                 std::make_unique<GemmaModel>(loader, threading, inference);
             if (!gemma->ModelIsLoaded()) {
               throw std::invalid_argument("Could not load model.");
             }
             return gemma;
           }),
           py::arg("tokenizer_path"), py::arg("weights_path"),
           py::arg("max_threads") = 0)
      .def("generate_ex", &GemmaModel::GenerateEx, py::arg("prompt"),
           py::arg("stream"), py::arg("max_generated_tokens") = 1024,
           py::arg("temperature") = 0.9, py::arg("seed") = 123456789,
           py::arg("accept") = gcpp::AcceptFunc(),
           py::arg("skip_prompt") = false)
      .def("generate", &GemmaModel::Generate, py::arg("prompt"),
           py::arg("max_generated_tokens") = 1024, py::arg("temperature") = 0.9,
           py::arg("seed") = 123456789,
           py::arg("accept") = std::vector<std::string>(),
           py::arg("end") = std::vector<std::string>())
      .def("generate_batch", &GemmaModel::GenerateBatch, py::arg("inputs"),
           py::arg("max_generated_tokens") = 1024, py::arg("temperature") = 0.9,
           py::arg("seed") = 123456789, py::arg("top_k") = 5)
      .def("set_image", &GemmaModel::SetImage, py::arg("image"))
      .def("generate_with_image", &GemmaModel::GenerateWithImage,
           py::arg("prompt") = "", py::arg("max_generated_tokens") = 1024,
           py::arg("temperature") = 0.9, py::arg("seed") = 123456789,
           py::arg("accept") = gcpp::AcceptFunc(),
           py::arg("prompt_tokens") = std::vector<int>())
      .def("get_last_prob", &GemmaModel::GetLastProb)
      .def("detokenize", &GemmaModel::Detokenize, py::arg("token_ids"));
}
