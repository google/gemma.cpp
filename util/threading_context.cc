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

#include "util/threading_context.h"

namespace gcpp {

ThreadingContext::ThreadingContext(const ThreadingArgs& args)
    : topology(BoundedSlice(args.skip_packages, args.max_packages),
               BoundedSlice(args.skip_clusters, args.max_clusters),
               BoundedSlice(args.skip_lps, args.max_lps)),
      allocator(topology, args.bind != Tristate::kFalse),
      pools(topology, allocator, args.max_threads, args.pin) {}

}  // namespace gcpp
