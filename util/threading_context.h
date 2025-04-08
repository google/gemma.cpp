// Copyright 2025 Google LLC
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

#ifndef THIRD_PARTY_GEMMA_CPP_UTIL_THREADING_CONTEXT_H_
#define THIRD_PARTY_GEMMA_CPP_UTIL_THREADING_CONTEXT_H_

// Separate component to ensure `threading.cc` does not have access to
// `ThreadingContext`, because that could deadlock.

#include <stddef.h>
#include <stdint.h>

// IWYU pragma: begin_exports
#include "util/allocator.h"
#include "util/args.h"
#include "util/basics.h"  // Tristate
#include "util/threading.h"
#include "util/topology.h"
// IWYU pragma: end_exports

namespace gcpp {

// Optional arguments for `ThreadingContext` from the command line.
class ThreadingArgs : public ArgsBase<ThreadingArgs> {
 public:
  ThreadingArgs(int argc, char* argv[]) { InitAndParse(argc, argv); }
  ThreadingArgs() { Init(); };

  // For BoundedTopology:
  size_t skip_packages;
  size_t max_packages;
  size_t skip_clusters;
  size_t max_clusters;
  size_t skip_lps;
  size_t max_lps;

  Tristate bind;

  // For NestedPools:
  size_t max_threads;  // divided among the detected clusters
  Tristate pin;        // pin threads?
  Tristate spin;       // use spin waits?

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    // These can be used to partition CPU sockets/packages and their
    // clusters/CCXs across several program instances. The default is to use
    // all available resources.
    visitor(skip_packages, "skip_packages", size_t{0},
            "Index of the first socket to use; default 0 = unlimited.", 2);
    visitor(max_packages, "max_packages", size_t{0},
            "Maximum number of sockets to use; default 0 = unlimited.", 2);
    visitor(skip_clusters, "skip_clusters", size_t{0},
            "Index of the first CCX to use; default 0 = unlimited.", 2);
    visitor(max_clusters, "max_clusters", size_t{0},
            "Maximum number of CCXs to use; default 0 = unlimited.", 2);
    // These are only used when CPU topology is unknown.
    visitor(skip_lps, "skip_lps", size_t{0},
            "Index of the first LP to use; default 0 = unlimited.", 2);
    visitor(max_lps, "max_lps", size_t{0},
            "Maximum number of LPs to use; default 0 = unlimited.", 2);

    // The exact meaning is more subtle: see the comment at NestedPools ctor.
    visitor(max_threads, "num_threads", size_t{0},
            "Maximum number of threads to use; default 0 = unlimited.", 2);
    visitor(pin, "pin", Tristate::kDefault,
            "Pin threads? -1 = auto, 0 = no, 1 = yes.", 2);
    visitor(spin, "spin", Tristate::kDefault,
            "Use spin waits? -1 = auto, 0 = no, 1 = yes.", 2);

    visitor(bind, "bind", Tristate::kDefault,
            "Bind memory to sockets? -1 = auto, 0 = no, 1 = yes.", 2);
  }
};

// Lazily-initialized singleton with support for passing in arguments from
// `ThreadingArgs` and re-initializing with different arguments.
class ThreadingContext2 {
  struct PrivateToken {};  // avoids constructing directly

 public:
  // If not called, default arguments are used when `Get` initializes the
  // singleton. Must not be called after `Get`, unless after a call to
  // `ThreadHostileInvalidate`, because otherwise initialization already
  // happened and the arguments would have no effect. Thread-safe, though this
  // is expected to be called early in the program, before threading starts.
  static void SetArgs(const ThreadingArgs& args);

  // Returns a reference to the singleton after initializing it if necessary.
  // When initializing, uses the args passed to `SetArgs`, or defaults.
  //
  // It is safe to call this concurrently with other `Get`, but not with
  // `SetArgs`, because that will warn if called after this, nor with
  // `ThreadHostileInvalidate`, because that will invalidate the reference which
  // callers of this may still be using. Such usage only occurs in tests,
  // hence we prefer not to pull `std::shared_ptr` into the interface.
  //
  // To reduce overhead, callers should cache the result and call less often.
  static ThreadingContext2& Get();

  // Invalidates the singleton before or after a call to `Get`. This allows
  // changing the arguments between tests. Callers must again call `Get`
  // afterwards to obtain an instance. WARNING: must not be called concurrently
  // with other calls to `Get` and usages of its return value.
  static void ThreadHostileInvalidate();

  explicit ThreadingContext2(PrivateToken);  // only called via `Get`.

  BoundedTopology topology;
  Allocator2 allocator;
  NestedPools pools;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_THREADING_CONTEXT_H_
