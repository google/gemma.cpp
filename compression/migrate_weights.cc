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

#include <stdio.h>

#include <string>

#include "evals/benchmark_helper.h"
#include "gemma/gemma.h"
#include "util/args.h"

namespace gcpp {
namespace {

struct WriterArgs : public ArgsBase<WriterArgs> {
  // --output_weights is required.
  WriterArgs(int argc, char* argv[]) { InitAndParse(argc, argv); }

  // Returns error string or nullptr if OK.
  const char* Validate() {
    if (output_weights.path.empty()) {
      return "Missing --output_weights flag, a file for the model weights.";
    }
    return nullptr;
  }

  Path output_weights;  // weights file location

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(output_weights, "output_weights", Path(),
            "Path name of output weights (.sbs) file.\n    Required argument.");
  }
};

}  // namespace
}  // namespace gcpp

int main(int argc, char** argv) {
  // Loads a model in the multi-file format and saves it in single-file format.
  gcpp::WriterArgs args(argc, argv);
  if (const char* err = args.Validate()) {
    fprintf(stderr, "Skipping model load because: %s\n", err);
    return 1;
  }
  gcpp::GemmaEnv env(argc, argv);
  hwy::ThreadPool pool(0);
  env.GetGemma()->Save(args.output_weights, pool);
  return 0;
}
