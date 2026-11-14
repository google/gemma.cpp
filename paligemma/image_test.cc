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

#include <cstddef>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"

namespace gcpp {
namespace {

float Normalize(float value, float max_value = 255.0f) {
  return 2.0f * (value / max_value) - 1.0f;
}

TEST(ImageTest, LoadResize224GetPatch) {
  std::string path = "paligemma/testdata/image.ppm";
  Image image;
  EXPECT_EQ(image.width(), 0);
  EXPECT_EQ(image.height(), 0);
  EXPECT_EQ(image.size(), 0);
  ASSERT_TRUE(image.ReadPPM(path));
  EXPECT_EQ(image.width(), 256);
  EXPECT_EQ(image.height(), 341);
  EXPECT_EQ(image.size(), 256 * 341 * 3);
  // Spot check a few values.
  EXPECT_EQ(image.data()[0], Normalize(160));
  EXPECT_EQ(image.data()[1], Normalize(184));
  EXPECT_EQ(image.data()[2], Normalize(188));
  EXPECT_EQ(image.data()[3], Normalize(163));
  EXPECT_EQ(image.data()[4], Normalize(185));
  EXPECT_EQ(image.data()[5], Normalize(189));
  EXPECT_EQ(image.data()[30], Normalize(164));
  EXPECT_EQ(image.data()[31], Normalize(185));
  EXPECT_EQ(image.data()[32], Normalize(191));
  EXPECT_EQ(image.data()[33], Normalize(164));
  EXPECT_EQ(image.data()[34], Normalize(185));
  EXPECT_EQ(image.data()[35], Normalize(191));
  image.Resize(224, 224);
  // Check first and last pixel.
  EXPECT_EQ(image.data()[0], Normalize(160));
  EXPECT_EQ(image.data()[1], Normalize(184));
  EXPECT_EQ(image.data()[2], Normalize(188));
  EXPECT_EQ(image.data()[image.size() - 3], Normalize(90));
  EXPECT_EQ(image.data()[image.size() - 2], Normalize(132));
  EXPECT_EQ(image.data()[image.size() - 1], Normalize(122));
  // Extract two patches.
  float patch[588];
  const hwy::Divisor div_patch_dim(14);
  image.GetPatch(0, div_patch_dim, patch);
  EXPECT_EQ(patch[0], Normalize(160));
  EXPECT_EQ(patch[1], Normalize(184));
  EXPECT_EQ(patch[2], Normalize(188));
  image.GetPatch(18, div_patch_dim, patch);
  // Check the first row of the patch.
  for (size_t i = 0; i < 14 * 3; ++i) {
    EXPECT_EQ(patch[i], image.data()[(14 * 224 + 2 * 14) * 3 + i]);
  }
}

TEST(ImageTest, Non224) {
  std::vector<float> data(28 * 42 * 3);
  for (int i = 0; i < data.size(); ++i) {
    data[i] = static_cast<float>(i);
  }
  float max_value = data.back();
  Image image;
  image.Set(28, 42, data.data());
  EXPECT_EQ(image.width(), 28);
  EXPECT_EQ(image.height(), 42);
  EXPECT_EQ(image.size(), data.size());
  // Resize 28 x 42 -> 56 x 42, "double" each pixel horizontally.
  image.Resize(/*new_width=*/56, /*new_height=*/42);
  // Check a few pixels.
  EXPECT_NEAR(image.data()[0], Normalize(0.0f, max_value), 1e-6);
  EXPECT_NEAR(image.data()[1], Normalize(1.0f, max_value), 1e-6);
  EXPECT_NEAR(image.data()[2], Normalize(2.0f, max_value), 1e-6);
  EXPECT_NEAR(image.data()[3], Normalize(0.0f, max_value), 1e-6);
  EXPECT_NEAR(image.data()[4], Normalize(1.0f, max_value), 1e-6);
  EXPECT_NEAR(image.data()[5], Normalize(2.0f, max_value), 1e-6);
  EXPECT_NEAR(image.data()[6], Normalize(3.0f, max_value), 1e-6);
  EXPECT_NEAR(image.data()[image.size() - 9],
            Normalize(data.size() - 6, max_value), 1e-6);
  EXPECT_NEAR(image.data()[image.size() - 8],
            Normalize(data.size() - 5, max_value), 1e-6);
  EXPECT_NEAR(image.data()[image.size() - 7],
            Normalize(data.size() - 4, max_value), 1e-6);
  EXPECT_NEAR(image.data()[image.size() - 3],
            Normalize(data.size() - 3, max_value), 1e-6);
  EXPECT_NEAR(image.data()[image.size() - 2],
            Normalize(data.size() - 2, max_value), 1e-6);
  EXPECT_NEAR(image.data()[image.size() - 1],
            Normalize(data.size() - 1, max_value), 1e-6);
  // Extract two patches.
  const size_t kPatchValues = 14 * 14 * 3;  // = 588
  float patch[kPatchValues];
  const hwy::Divisor div_patch_dim(14);
  // Patch 0 is just the "start" of the image.
  image.GetPatch(0, div_patch_dim, patch);
  EXPECT_NEAR(patch[0], Normalize(0.0f, max_value), 1e-6);
  EXPECT_NEAR(patch[1], Normalize(1.0f, max_value), 1e-6);
  EXPECT_NEAR(patch[2], Normalize(2.0f, max_value), 1e-6);
  // The "image" has 4x3 patches, so patch 6 has coordinates (1, 2) and its
  // pixel coordinates are offset by (14, 28).
  image.GetPatch(6, div_patch_dim, patch);
  for (size_t n = 0; n < kPatchValues; ++n) {
    size_t k = n % 3;
    size_t j = ((n - k) / 3) % 14;
    size_t i = (n - k - j * 3) / (14 * 3);
    EXPECT_EQ(n, (i * 14 + j) * 3 + k);
    i += 14;
    j += 28;
    EXPECT_EQ(patch[n], image.data()[(i * 56 + j) * 3 + k]);
  }
}

bool ReadPPMFromString(Image& image, const std::string& content) {
  return image.ReadPPM(hwy::Span<const char>(content.data(), content.size()));
}

TEST(ImageTest, RejectsInvalidPPMHeaders) {
  Image image;
  // ParseUnsigned rejects > SIZE_MAX decimal values without unsigned wrap.
  EXPECT_FALSE(
      ReadPPMFromString(image, "P6\n99999999999999999999 1\n255\nXYZ"));
  // Width that previously overflowed width * 3 in 64-bit math (SIZE_MAX/3 + 1).
  EXPECT_FALSE(ReadPPMFromString(image, "P6\n6148914691236517206 1\n255\nXYZ"));
  // Zero dimensions must return false cleanly without aborting.
  EXPECT_FALSE(ReadPPMFromString(image, "P6\n0 10\n255\nXYZ"));
  EXPECT_FALSE(ReadPPMFromString(image, "P6\n10 0\n255\nXYZ"));
  // Dimensions exceeding the kMaxImageDim (65535) policy cap.
  EXPECT_FALSE(ReadPPMFromString(image, "P6\n70000 1\n255\nXYZ"));
  EXPECT_FALSE(ReadPPMFromString(image, "P6\n1 70000\n255\nXYZ"));
  // Dimensions each <= kMaxImageDim whose byte count exceeds INT32_MAX.
  EXPECT_FALSE(ReadPPMFromString(image, "P6\n40000 40000\n255\nXYZ"));
  EXPECT_FALSE(ReadPPMFromString(image, "P6\n65535 65535\n255\nXYZ"));
  // Invalid max_value (0 or > 255).
  EXPECT_FALSE(ReadPPMFromString(image, "P6\n1 1\n0\nXYZ"));
  EXPECT_FALSE(ReadPPMFromString(image, "P6\n1 1\n256\nXYZ"));
  // Header ending right after max_value without trailing whitespace (no OOB).
  EXPECT_FALSE(ReadPPMFromString(image, "P6 1 1 255"));
  EXPECT_FALSE(ReadPPMFromString(image, "P6 1 1 255X"));
  // Insufficient pixel data remaining.
  EXPECT_FALSE(ReadPPMFromString(image, "P6\n2 2\n255\n123"));
}

TEST(ImageTest, Resize1x1) {
  Image image;
  // Valid 1x1 PPM followed by Resize (exercises NearestNeighbor from_size==1).
  ASSERT_TRUE(ReadPPMFromString(image, "P6\n1 1\n255\n\x80\x40\xff"));
  EXPECT_EQ(image.width(), 1);
  EXPECT_EQ(image.height(), 1);
  image.Resize(4, 4);
  EXPECT_EQ(image.width(), 4);
  EXPECT_EQ(image.height(), 4);
  EXPECT_NEAR(image.data()[0], Normalize(0x80), 1e-6f);
  EXPECT_NEAR(image.data()[1], Normalize(0x40), 1e-6f);
  EXPECT_NEAR(image.data()[2], Normalize(0xff), 1e-6f);

  // Downsample a 2x2 image with distinct pixels to 1x2 and 1x1 (exercises
  // NearestNeighbor to_size==1 while from_size > 1).
  const float quad[2 * 2 * 3] = {
      0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
      2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f,
  };
  image.Set(2, 2, quad);
  image.Resize(1, 2);
  EXPECT_EQ(image.width(), 1);
  EXPECT_EQ(image.height(), 2);
  EXPECT_NEAR(image.data()[0], -1.0f, 1e-6f);
  EXPECT_NEAR(image.data()[3], 2.0f * (2.0f / 3.0f) - 1.0f, 1e-6f);
  image.Resize(1, 1);
  EXPECT_EQ(image.width(), 1);
  EXPECT_EQ(image.height(), 1);
  EXPECT_NEAR(image.data()[0], -1.0f, 1e-6f);
}

TEST(ImageTest, PreconditionsAbortOnEmptyImage) {
  Image empty;
  float patch[14 * 14 * 3];
  EXPECT_DEATH_IF_SUPPORTED(empty.Resize(4, 4), "width_ > 0");
  EXPECT_DEATH_IF_SUPPORTED(empty.ResizeBilinear(4, 4), "width_ > 0");
  EXPECT_DEATH_IF_SUPPORTED(empty.GetPatch(0, hwy::Divisor(14), patch),
                            "width_ > 0");

  Image image;
  ASSERT_TRUE(ReadPPMFromString(image, "P6\n1 1\n255\n\x80\x40\xff"));
  image.Resize(14, 14);
  EXPECT_DEATH_IF_SUPPORTED(image.GetPatch(1, hwy::Divisor(14), patch),
                            "patch_y <");
}

TEST(ImageTest, ResizeBilinear1x1And2x2) {
  Image image;
  ASSERT_TRUE(ReadPPMFromString(image, "P6\n1 1\n255\n\x20\x60\xa0"));
  image.ResizeBilinear(3, 3);
  EXPECT_EQ(image.width(), 3);
  EXPECT_EQ(image.height(), 3);
  for (int i = 0; i < 9; ++i) {
    EXPECT_NEAR(image.data()[i * 3 + 0], Normalize(0x20), 1e-6f);
    EXPECT_NEAR(image.data()[i * 3 + 1], Normalize(0x60), 1e-6f);
    EXPECT_NEAR(image.data()[i * 3 + 2], Normalize(0xa0), 1e-6f);
  }
  // Same-size ResizeBilinear is a no-op.
  image.ResizeBilinear(3, 3);
  EXPECT_EQ(image.width(), 3);

  const float quad[2 * 2 * 3] = {
      0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
  };
  image.Set(2, 2, quad);
  image.ResizeBilinear(4, 4);
  EXPECT_EQ(image.width(), 4);
  EXPECT_EQ(image.height(), 4);
  EXPECT_NEAR(image.data()[0], -1.0f, 1e-6f);
  EXPECT_NEAR(image.data()[(1 * 4 + 1) * 3], -0.5f, 1e-6f);
  EXPECT_NEAR(image.data()[4 * 4 * 3 - 1], 1.0f, 1e-6f);
}

}  // namespace
}  // namespace gcpp
