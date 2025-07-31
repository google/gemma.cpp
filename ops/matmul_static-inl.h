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

#include "hwy/detect_compiler_arch.h"  // HWY_IDE

#ifndef GEMMA_MATMUL_TB
#if HWY_IDE
// Provide a definition so the IDE does not complain.
#define GEMMA_MATMUL_TB float
#else
#error "Only include from matmul_static_*.cc, which define GEMMA_MATMUL_TB"
#endif  // HWY_IDE
#endif  // GEMMA_MATMUL_TB

// Passed to GEMMA_MATMUL_FOREACH_AC; defines one overload for one target.
#define GEMMA_MATMUL_DEFINE_ONE(TA, TB, TC)                             \
  MMPerKey* MatMulStatic(const MatPtrT<TA>& A, const MatPtrT<TB>& B,    \
                         const float* HWY_RESTRICT add, MatMulEnv& env, \
                         MatPtrT<TC>& C) {                              \
    return MatMul(A, B, add, env, C);                                   \
  }

#if defined(THIRD_PARTY_GEMMA_CPP_GEMMA_OPS_MATMUL_STATIC_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_GEMMA_OPS_MATMUL_STATIC_INL_H_
#undef THIRD_PARTY_GEMMA_CPP_GEMMA_OPS_MATMUL_STATIC_INL_H_
#else
#define THIRD_PARTY_GEMMA_CPP_GEMMA_OPS_MATMUL_STATIC_INL_H_
#endif

#include "hwy/highway.h"
// After highway.h
#include "ops/matmul-inl.h"
#include "ops/matmul_static.h"  // includes highway.h!

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// Ignore warning that we are defining a function in a header; this is only
// included from matmul_static_*.cc.
GEMMA_MATMUL_FOREACH_AC(GEMMA_MATMUL_DEFINE_ONE, GEMMA_MATMUL_TB)  // NOLINT

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_OPS_MATMUL_STATIC_INL_H_
