// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "compression/compress.h"

#include <stddef.h>
#include <stdint.h>

#include "util/mat.h"
#include "hwy/base.h"
#include "hwy/profiler.h"

namespace gcpp {

float ScaleWeights(float* HWY_RESTRICT raw, size_t num) {
  PROFILER_FUNC;

  float maxabs = 0.0;
  for (size_t i = 0; i < num; ++i) {
    maxabs = HWY_MAX(maxabs, hwy::ScalarAbs(raw[i]));
  }
  if (maxabs <= SfpStream::kMax) {
    return 1.0f;
  }
  const float scale = maxabs / SfpStream::kMax;
  const float inv_scale = static_cast<float>(1.0 / static_cast<double>(scale));
  for (size_t i = 0; i < num; ++i) {
    // Clamp because kMax may still be exceeded.
    const float magn =
        HWY_MIN(SfpStream::kMax, hwy::ScalarAbs(raw[i] * inv_scale));
    raw[i] = hwy::ScalarCopySign(magn, raw[i]);
  }
  return scale;
}

}  // namespace gcpp
