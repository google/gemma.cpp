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

#ifndef THIRD_PARTY_GEMMA_CPP_OPS_MATMUL_STATIC_H_
#define THIRD_PARTY_GEMMA_CPP_OPS_MATMUL_STATIC_H_

// Declares overloads of MatMulStatic for all SIMD targets and input types.

#include <stddef.h>
#include <stdint.h>

#include "ops/matmul.h"  // IWYU pragma: keep, b/420428845
#include "hwy/highway.h"

// Invokes GEMMA_X(TA, TB, TC) for all combinations of F32 or BF16.
#define GEMMA_MATMUL_FOREACH_AC(GEMMA_X, TB) \
  GEMMA_X(float, TB, float)                  \
  GEMMA_X(float, TB, BF16)                   \
  GEMMA_X(BF16, TB, float)                   \
  GEMMA_X(BF16, TB, BF16)

// Passed to GEMMA_MATMUL_FOREACH_AC; declares one overload for one target.
#define GEMMA_MATMUL_DECL_ONE(TA, TB, TC)                               \
  MMPerKey* MatMulStatic(const MatPtrT<TA>& A, const MatPtrT<TB>& B,    \
                         const float* HWY_RESTRICT add, MatMulEnv& env, \
                         MatPtrT<TC>& C, MMOptions options);

#define GEMMA_MATMUL_FOR_B(TB)                                        \
  GEMMA_MATMUL_FOREACH_AC(GEMMA_MATMUL_DECL_ONE, TB)                  \
  void TwoMatMulStatic(const MatPtrT<BF16>& A, const MatPtrT<TB>& B1, \
                       const MatPtrT<TB>& B2, MatMulEnv& env,         \
                       MatPtrT<BF16>& C, MMOptions options);

// Passed to HWY_VISIT_TARGETS; declares all overloads for all targets.
#define GEMMA_MATMUL_DECL(TARGET, NAMESPACE)                  \
  namespace NAMESPACE {                                       \
  GEMMA_MATMUL_FOR_B(BF16)                                    \
  GEMMA_MATMUL_FOR_B(float)                                   \
  GEMMA_MATMUL_FOR_B(NuqStream)                               \
  GEMMA_MATMUL_FOR_B(SfpStream)                               \
  /* NOLINTNEXTLINE(google-readability-namespace-comments) */ \
  }  // namespace NAMESPACE

namespace gcpp {

// MatMul function declarations for each SIMD target. Allows direct call from
// the per-target namespace. We may later replace this with dynamic dispatch if
// the overhead is acceptable.
HWY_VISIT_TARGETS(GEMMA_MATMUL_DECL)

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_OPS_MATMUL_STATIC_H_
