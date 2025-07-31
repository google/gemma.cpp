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

// Loads a model and saves it in single-file format.

#include "evals/benchmark_helper.h"  // GemmaEnv
#include "gemma/gemma.h"
#include "util/args.h"

namespace gcpp {
namespace {

struct WriterArgs : public ArgsBase<WriterArgs> {
  WriterArgs(int argc, char* argv[]) { InitAndParse(argc, argv); }

  Path output_weights;

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(output_weights, "output_weights", Path(),
            "Path name of output weights (.sbs) file.\n    Required argument.");
  }
};

}  // namespace
}  // namespace gcpp

int main(int argc, char** argv) {
  gcpp::WriterArgs args(argc, argv);
  if (args.output_weights.Empty()) {
    HWY_ABORT("Missing --output_weights flag, a file for the model weights.");
  }

  gcpp::GemmaEnv env(argc, argv);
  env.GetGemma()->Save(args.output_weights, env.Env().ctx.pools);
  return 0;
}
