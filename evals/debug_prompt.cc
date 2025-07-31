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

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "evals/benchmark_helper.h"
#include "gemma/gemma.h"  // LayersOutputFunc
#include "io/io.h"
#include "util/args.h"
#include "hwy/base.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace gcpp {

class PromptArgs : public ArgsBase<PromptArgs> {
 public:
  PromptArgs(int argc, char* argv[]) { InitAndParse(argc, argv); }

  Path layers_output;  // optional
  std::string prompt;

  // Returns error string or nullptr if OK.
  const char* Validate() const {
    if (prompt.empty()) return "Must specify --prompt";
    return nullptr;
  }

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(layers_output, "layers_output", Path(""),
            "Path to store layers output", 2);
    visitor(prompt, "prompt", std::string(""), "Prompt to the model", 2);
  }
};

int Run(int argc, char** argv) {
  PromptArgs prompt_args(argc, argv);
  AbortIfInvalidArgs(prompt_args);

  json json_output;
  GemmaEnv env(argc, argv);
  env.MutableConfig().layers_output =
      prompt_args.layers_output.Empty()
          ? LayersOutputFunc()
          : [&json_output](size_t query_idx, size_t pos, const std::string& key,
                           int layer, const float* values, size_t values_len) {
              const std::string& debug_key =
                  layer < 0 ? key : (key + "." + std::to_string(layer));
              const std::vector<float> v{values, values + values_len};
              json& json_base = json_output[std::to_string(query_idx)];
              json_base[std::to_string(pos)][debug_key] = v;
            };

  QueryResult result = env.QueryModel(prompt_args.prompt);
  std::cout << result.response.substr(result.response_start_pos) << "\n"
            << std::flush;

  if (env.MutableConfig().layers_output) {
    std::ofstream output_f(prompt_args.layers_output.path, std::ofstream::out);
    if (!output_f) HWY_ABORT("Opening layer output file failed");
    output_f << json_output.dump();
    if (!output_f) HWY_ABORT("Writing to layer output file failed");
    output_f.close();
  }
  return 0;
}

}  // namespace gcpp

int main(int argc, char** argv) { return gcpp::Run(argc, argv); }
