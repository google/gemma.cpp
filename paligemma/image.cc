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

#include "paligemma/image.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "compression/io.h"
#include "hwy/aligned_allocator.h"  // hwy::Span
#include "hwy/base.h"
#include "hwy/profiler.h"

namespace gcpp {
namespace {
// Hardcoded for PaliGemma-224 ViT input.
constexpr size_t kPatchSize = 14;
constexpr size_t kImageSize = 224;
constexpr size_t kNumPatches = kImageSize / kPatchSize;  // 16

// Returns the linearly scaled index in [0, to_size) closest to the
// value in [0, from_size).
int NearestNeighbor(int value, int from_size, int to_size) {
  float scale_factor = static_cast<float>(to_size - 1) / (from_size - 1);
  // Apply nearest neighbor rounding.
  int nn = static_cast<int>(std::round(value * scale_factor));
  // Ensure the value is within the new range.
  nn = std::clamp(nn, 0, to_size - 1);
  return nn;
}

// Returns value in [0,1] mapped linearly to [-1,1].
float StretchToSigned(float value) {
  // = out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min);
  return value * 2.0f - 1.0f;
}

bool IsLineBreak(int c) { return c == '\r' || c == '\n'; }

const char* CheckP6Format(const char* pos, const char* end) {
  constexpr const char format[] = "P6";
  for (size_t i = 0; i < sizeof(format) - 1; ++i) {
    if (pos == end || *pos != format[i]) {
      return nullptr;
    }
    ++pos;
  }
  return pos;
}

const char* SkipWhitespaceAndComments(const char* pos, const char* end) {
  while (pos < end && std::isspace(*pos)) ++pos;
  while (pos < end && *pos == '#') {  // Skip comment lines.
    while (pos < end && !IsLineBreak(*pos)) ++pos;
    while (pos < end && std::isspace(*pos)) ++pos;
  }
  return pos;
}

const char* ParseUnsigned(const char* pos, const char* end, size_t& num) {
  if (pos == end || !std::isdigit(*pos)) {
    return nullptr;
  }
  num = 0;
  for ( ; pos < end && std::isdigit(*pos); ++pos) {
    num *= 10;
    num += *pos - '0';
  }
  return pos;
}
}  // namespace

bool Image::ReadPPM(const std::string& filename) {
  Path path(filename);
  if (!path.Exists()) {
    std::cerr << filename << " does not exist\n";
    return false;
  }
  const std::string content = ReadFileToString(path);
  return ReadPPM(hwy::Span<const char>(content.data(), content.size()));
}

bool Image::ReadPPM(const hwy::Span<const char>& buf) {
  const char* pos = CheckP6Format(buf.cbegin(), buf.cend());
  if (!pos) {
    std::cerr << "We only support binary PPM (P6)\n";
    return false;
  }
  size_t width, height, max_value;
  pos = SkipWhitespaceAndComments(pos, buf.cend());
  pos = ParseUnsigned(pos, buf.cend(), width);
  if (!pos) {
    std::cerr << "Reached end before width\n";
    return false;
  }
  pos = SkipWhitespaceAndComments(pos, buf.cend());
  pos = ParseUnsigned(pos, buf.cend(), height);
  if (!pos) {
    std::cerr << "Reached end before height\n";
    return false;
  }
  pos = SkipWhitespaceAndComments(pos, buf.cend());
  pos = ParseUnsigned(pos, buf.cend(), max_value);
  if (!pos) {
    std::cerr << "Reached end before max_value\n";
    return false;
  }
  if (max_value <= 0 || max_value > 255) {
    std::cerr << "Unsupported max value " << max_value << "\n";
    return false;
  }
  // P6 requires exactly one whitespace character after the header.
  if (!std::isspace(*pos)) {
    std::cerr << "Missing whitespace after header\n";
    return false;
  }
  ++pos;
  const size_t data_size = width * height * 3;
  if (buf.cend() - pos < static_cast<ptrdiff_t>(data_size)) {
    std::cerr << "Insufficient data remaining\n";
    return false;
  }
  data_.resize(data_size);
  width_ = width;
  height_ = height;
  for (size_t i = 0; i < data_size; ++i) {
    uint8_t value = pos[i];
    data_[i] = StretchToSigned(static_cast<float>(value) / max_value);
  }
  return true;
}

void Image::Set(int width, int height, const float* data) {
  width_ = width;
  height_ = height;
  int num_elements = width * height * 3;
  data_.resize(num_elements);
  data_.assign(data, data + num_elements);
  float min_value = std::numeric_limits<float>::infinity();
  float max_value = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < num_elements; ++i) {
    if (data_[i] < min_value) min_value = data_[i];
    if (data_[i] > max_value) max_value = data_[i];
  }
  // -> out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)
  float in_range = max_value - min_value;
  if (in_range == 0.0f) in_range = 1.0f;
  float scale = 2.0f / in_range;
  for (int i = 0; i < num_elements; ++i) {
    data_[i] = (data_[i] - min_value) * scale - 1.0f;
  }
}

void Image::Resize() {
  int new_width = 224;
  int new_height = kImageSize;
  std::vector<float> new_data(new_width * new_height * 3);
  // TODO: go to bilinear interpolation, or antialias.
  // E.g. consider WeightsSymmetric3Lowpass and SlowSymmetric3 from
  // jpegxl/lib/jxl/convolve_slow.cc
  // For now, just do nearest neighbor.
  for (int i = 0; i < new_height; ++i) {
    for (int j = 0; j < new_width; ++j) {
      int old_i = NearestNeighbor(i, new_height, height_);
      int old_j = NearestNeighbor(j, new_width, width_);
      for (int k = 0; k < 3; ++k) {
        new_data[(i * new_width + j) * 3 + k] =
            data_[(old_i * width_ + old_j) * 3 + k];
      }
    }
  }
  data_ = std::move(new_data);
  height_ = new_height;
  width_ = new_width;
}

bool Image::WriteBinary(const std::string& filename) const {
  // Writes the floating point values as float32 in binary format.
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open " << filename << "\n";
    return false;
  }
  for (size_t i = 0; i < data_.size(); ++i) {
    file.write(reinterpret_cast<const char*>(&data_[i]), sizeof(float));
  }
  file.close();
  return true;
}

// Image.data() is kImageSize x kImageSize x 3, H x W x C.
// We want the N-th patch (of 256) of size kPatchSize x kPatchSize x 3.
// Patches are numbered in usual "pixel-order".
void Image::GetPatch(size_t patch_num, float* patch) const {
  PROFILER_FUNC;
  constexpr size_t kDataSize = kImageSize * kImageSize * 3;
  HWY_ASSERT(size() == kDataSize);
  constexpr size_t kPatchDataSize = kPatchSize * kPatchSize * 3;
  size_t i_offs = patch_num / kNumPatches;
  size_t j_offs = patch_num % kNumPatches;
  HWY_ASSERT(0 <= i_offs && i_offs < kNumPatches);
  HWY_ASSERT(0 <= j_offs && j_offs < kNumPatches);
  i_offs *= kPatchSize;
  j_offs *= kPatchSize;
  // This can be made faster, but let's first see whether it matters.
  const float* image_data = data();
  for (size_t i = 0; i < kPatchSize; ++i) {
    for (size_t j = 0; j < kPatchSize; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        const size_t patch_index = (i * kPatchSize + j) * 3 + k;
        HWY_ASSERT(patch_index < kPatchDataSize);
        const size_t image_index =
            ((i + i_offs) * kImageSize + (j + j_offs)) * 3 + k;
        HWY_ASSERT(image_index < kDataSize);
        patch[patch_index] = image_data[image_index];
      }
    }
  }
}

}  // namespace gcpp
