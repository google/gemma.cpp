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

// Shared between various frontends.

#ifndef THIRD_PARTY_GEMMA_CPP_UTIL_APP_H_
#define THIRD_PARTY_GEMMA_CPP_UTIL_APP_H_

#if HWY_OS_LINUX
#include <sched.h>

#include <cerrno>  // IDE does not recognize errno.h as providing errno.
#endif
#include <stddef.h>
#include <stdio.h>

#include <algorithm>  // std::clamp
#include <thread>     // NOLINT>

// copybara:import_next_line:gemma_cpp
#include "util/args.h"
// copybara:end
#include "hwy/base.h"  // HWY_ASSERT

namespace gcpp {

static inline void PinThreadToCore(size_t cpu_index) {
#if HWY_OS_LINUX
  // Forces the thread to run on the logical processor with the same number.
  cpu_set_t cset;             // bit array
  CPU_ZERO(&cset);            // clear all
  CPU_SET(cpu_index, &cset);  // set bit indicating which processor to run on.
  const int err = sched_setaffinity(0, sizeof(cset), &cset);
  if (err != 0) {
    fprintf(stderr,
            "sched_setaffinity returned %d, errno %d. Can happen if running in "
            "a container; this warning is safe to ignore.\n",
            err, errno);
  }
#else
  (void)cpu_index;
#endif
}

class AppArgs : public ArgsBase<AppArgs> {
  static constexpr size_t kDefaultNumThreads = ~size_t{0};

  void ChooseNumThreads() {
    if (num_threads == kDefaultNumThreads) {
      // This is a rough heuristic, replace with something better in the future.
      num_threads = static_cast<size_t>(std::clamp(
          static_cast<int>(std::thread::hardware_concurrency()) - 2, 1, 18));
    }
  }

 public:
  AppArgs(int argc, char* argv[]) {
    InitAndParse(argc, argv);
    ChooseNumThreads();
  }

  Path log;  // output
  int verbosity;
  size_t num_threads;
  std::string eot_line;

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(verbosity, "verbosity", 1,
            "Show verbose developer information\n   0 = only print generation "
            "output\n   1 = standard user-facing terminal ui\n   2 = show "
            "developer/debug info).\n   Default = 1.",
            2);
    visitor(num_threads, "num_threads",
            kDefaultNumThreads,  // see ChooseNumThreads
            "Number of threads to use.\n    Default = Estimate of the "
            "number of suupported concurrent threads.",
            2);
    visitor(
        eot_line, "eot_line", std::string(""),
        "End of turn line. "
        "When you specify this, the prompt will be all lines "
        "before the line where only the given string appears.\n    Default = "
        "When a newline is encountered, that signals the end of the turn.",
        2);
  }
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_APP_H_
