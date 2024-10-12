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

#include <stdint.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "hwy/base.h"

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

void SkipWhitespaceAndComments(std::istream& in) {
  int value = in.get();
  while (std::isspace(value)) value = in.get();
  while (value == '#') {  // Skip comment lines.
    while (!IsLineBreak(value)) value = in.get();
    while (std::isspace(value)) value = in.get();
  }
  in.unget();  // Rewind last byte.
}
}  // namespace

bool Image::ReadPPM(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open " << filename << "\n";
    return false;
  }
  if (!ReadPPM(file)) {
    return false;
  }
  if (file.get() != EOF) {
    std::cerr << "Extra data in file\n";
    return false;
  }
  file.close();
  return true;
}

bool Image::ReadPPM(std::istream& in) {
  std::string format;
  in >> format;
  if (format != "P6") {
    std::cerr << "We only support binary PPM (P6) but got: " << format << "\n";
    return false;
  }
  int width, height, max_value;
  SkipWhitespaceAndComments(in);
  in >> width;
  SkipWhitespaceAndComments(in);
  in >> height;
  SkipWhitespaceAndComments(in);
  in >> max_value;
  if (max_value <= 0 || max_value > 255) {
    std::cerr << "Unsupported max value " << max_value << "\n";
    return false;
  }
  // P6 requires exactly one whitespace character after the header.
  int value = in.get();
  if (!std::isspace(value)) {
    std::cerr << "Missing whitespace after header\n";
    return false;
  }
  width_ = width;
  height_ = height;
  int data_size = width * height * 3;
  data_.resize(data_size);
  std::vector<uint8_t> data_bytes(data_size);
  in.read(reinterpret_cast<char*>(data_bytes.data()), data_size);
  if (in.gcount() != data_size) {
    std::cerr << "Failed to read " << data_size << " bytes\n";
    return false;
  }
  for (int i = 0; i < data_size; ++i) {
    data_[i] = StretchToSigned(static_cast<float>(data_bytes[i]) / max_value);
  }
  return true;
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
  constexpr size_t kDataSize = kImageSize * kImageSize * 3;
  HWY_ASSERT(size() == kDataSize);
  constexpr size_t kPatchDataSize = kPatchSize * kPatchSize * 3;
  int i_offs = patch_num / kNumPatches;
  int j_offs = patch_num % kNumPatches;
  HWY_ASSERT(0 <= i_offs && i_offs < kNumPatches);
  HWY_ASSERT(0 <= j_offs && j_offs < kNumPatches);
  i_offs *= kPatchSize;
  j_offs *= kPatchSize;
  // This can be made faster, but let's first see whether it matters.
  const float* image_data = data();
  for (int i = 0; i < kPatchSize; ++i) {
    for (int j = 0; j < kPatchSize; ++j) {
      for (int k = 0; k < 3; ++k) {
        const int patch_index = (i * kPatchSize + j) * 3 + k;
        HWY_ASSERT(patch_index < kPatchDataSize);
        const int image_index =
            ((i + i_offs) * kImageSize + (j + j_offs)) * 3 + k;
        HWY_ASSERT(image_index < kDataSize);
        patch[patch_index] = image_data[image_index];
      }
    }
  }
}
}  // namespace gcpp
