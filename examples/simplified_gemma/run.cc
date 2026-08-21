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

#include <stddef.h>

#include <string>

#include "third_party/gemma_cpp/examples/simplified_gemma/gemma.hpp"
#include "gemma/gemma_args.h"

int main(int argc, char** argv) {
  // Sets arguments from argc and argv. Note that you can instead pass in
  // LoaderArgs, ThreadingArgs, and InferenceArgs directly.
  gcpp::ConsumedArgs consumed(argc, argv);
  gcpp::GemmaArgs args(argc, argv, consumed);
  consumed.AbortIfUnconsumed();

  SimplifiedGemma gemma(args);
  std::string prompt = "Write a greeting to the world.";
  gemma.Generate(prompt, 256, 0.6);

  return 0;
}
