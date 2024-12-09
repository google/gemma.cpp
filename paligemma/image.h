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

#ifndef THIRD_PARTY_GEMMA_CPP_PALIGEMMA_IMAGE_H_
#define THIRD_PARTY_GEMMA_CPP_PALIGEMMA_IMAGE_H_

#include <cstddef>
#include <string>
#include <vector>

#include "hwy/aligned_allocator.h"  // Span

namespace gcpp {

// Very basic image loading and processing for PaliGemma.
class Image {
 public:
  Image() = default;
  // Reads a file in PPM format (P6, binary), normalizes to [-1, 1].
  // Returns true on success.
  bool ReadPPM(const std::string& filename);
  // Reads PPM format (P6, binary) data from a hwy::Span, normalizes to [-1, 1].
  // Returns true on success.
  bool ReadPPM(const hwy::Span<const char>& buf);
  // Sets the image content to the given data. The data is copied and normalized
  // to [-1, 1]. The data is expected to be of size width * height * 3.
  void Set(int width, int height, const float* data);
  // Resizes to width x height (nearest-neighbor for now, bilinear or antialias
  // would be better).
  void Resize(int width, int height);
  // Writes the file as plain floats in binary. Useful to e.g. load in a colab.
  bool WriteBinary(const std::string& filename) const;
  // Stores the patch for the given patch number in `patch`.
  // Patches are numbered in usual raster-order. E.g. for an image of size
  // 224 x 224, there are 16 x 16 = 256 patches.
  // `patch` should have space for at least 14 * 14 * 3 = 588 floats.
  // Requires that Normalize() has been called and that the image width and
  // height are multiples of 14.
  void GetPatch(size_t patch_num, float* patch) const;

  float *data() { return data_.data(); }
  const float *data() const { return data_.data(); }
  int width() const { return width_; }
  int height() const { return height_; }
  size_t size() const { return data_.size(); }
  operator bool() const { return data_.size() > 0; }

 private:
  int width_ = 0;
  int height_ = 0;
  std::vector<float> data_;  // r, g, b
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_PALIGEMMA_IMAGE_H_
