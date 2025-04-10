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

#include <memory>
#include <mutex>  // NOLINT

namespace gcpp {

static ThreadingArgs s_args;
// Cannot use magic static because that does not support `Invalidate`, hence
// allocate manually.
static std::unique_ptr<ThreadingContext2> s_ctx;
static std::mutex s_ctx_mutex;

/*static*/ void ThreadingContext2::SetArgs(const ThreadingArgs& args) {
  s_ctx_mutex.lock();
  HWY_ASSERT(!s_ctx);  // Ensure not already initialized, else this is too late.
  s_args = args;
  s_ctx_mutex.unlock();
}

/*static*/ bool ThreadingContext2::IsInitialized() {
  s_ctx_mutex.lock();
  const bool initialized = !!s_ctx;
  s_ctx_mutex.unlock();
  return initialized;
}

/*static*/ ThreadingContext2& ThreadingContext2::Get() {
  // We do not bother with double-checked locking because it requires an
  // atomic pointer, but we prefer to use unique_ptr for simplicity. Also,
  // callers can cache the result and call less often.
  s_ctx_mutex.lock();
  if (HWY_UNLIKELY(!s_ctx)) {
    s_ctx = std::make_unique<ThreadingContext2>(PrivateToken());
  }
  s_ctx_mutex.unlock();
  return *s_ctx;
}

/*static*/ void ThreadingContext2::ThreadHostileInvalidate() {
  // Deliberately avoid taking the lock so that tsan can warn if this is
  // called concurrently with other calls to `Get`.
  s_ctx.reset();
}

// WARNING: called with `s_ctx_mutex` held. Calling `SetArgs` or `Get` would
// deadlock.
ThreadingContext2::ThreadingContext2(ThreadingContext2::PrivateToken)
    : topology(BoundedSlice(s_args.skip_packages, s_args.max_packages),
               BoundedSlice(s_args.skip_clusters, s_args.max_clusters),
               BoundedSlice(s_args.skip_lps, s_args.max_lps)),
      allocator(topology, s_args.bind != Tristate::kFalse),
      pools(topology, allocator, s_args.max_threads, s_args.pin) {}

}  // namespace gcpp
