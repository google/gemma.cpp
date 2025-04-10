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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_COMMON_SCALAR_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_COMMON_SCALAR_H_

#include <stddef.h>

#include <complex>

#include "util/mat.h"

namespace gcpp {

template<typename T, typename U>
U DotT(const T* a, const U* b, size_t N) {
  U sum = {};
  for (size_t i = 0; i < N; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

template<>
inline std::complex<double> DotT(const float* a, const std::complex<double>* b,
                                 size_t N) {
  std::complex<double> sum = {};
  for (size_t i = 0; i < N; ++i) {
    sum += static_cast<double>(a[i]) * b[i];
  }
  return sum;
}

template<typename T>
void MulByConstT(T c, T* x, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    x[i] *= c;
  }
}

// out += c * x
template<typename T>
void MulByConstAndAddT(T c, const T* x, T* out, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    out[i] += c * x[i];
  }
}

template <typename T>
void MulByConstAndAddT(T c, const MatPtrT<T>& x, MatPtrT<T>& out) {
  for (size_t r = 0; r < x.Rows(); ++r) {
    MulByConstAndAddT(c, x.Row(r), out.Row(r), x.Cols());
  }
}

template<typename T>
void AddFromT(const T* a, T* out, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    out[i] += a[i];
  }
}

template<typename T>
T SquaredL2(const T* x, size_t N) {
  T sum = {};
  for (size_t i = 0; i < N; ++i) {
    sum += x[i] * x[i];
  }
  return sum;
}

template<typename T>
T Gelu(T x) {
  static const T kMul = 0.044715;
  static const T kSqrt2OverPi = 0.797884560804236;

  const T x3 = x * x * x;
  const T arg = kSqrt2OverPi * (kMul * x3 + x);
  const T cdf = T(0.5) * (T(1.0) + std::tanh(arg));
  return x * cdf;
}

template<typename T, typename U>
void Rope(T* x, U base, size_t N, int i) {
  const size_t N2 = N / 2;
  for (size_t dim = 0; dim < N2; ++dim) {
    const T freq_exponents = T(2 * dim) / T(N);
    const T timescale = std::pow(base, freq_exponents);
    const T theta = T(i) / timescale;
    const T cos_val = std::cos(theta);
    const T sin_val = std::sin(theta);
    const T x0 = x[dim];
    const T x1 = x[dim + N2];
    x[dim] = x0 * cos_val - x1 * sin_val;
    x[dim + N2] = x0 * sin_val + x1 * cos_val;
  }
}

template<typename T>
void Rope(T* x, size_t N, int i) {
  Rope(x, T(10000.0), N, i);
}

template<typename T>
void Rope(std::complex<T>* x, size_t N, int i) {
  Rope(x, T(10000.0), N, i);
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_COMMON_SCALAR_H_
