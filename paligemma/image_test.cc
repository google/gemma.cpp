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
  image.GetPatch(0, patch);
  EXPECT_EQ(patch[0], Normalize(160));
  EXPECT_EQ(patch[1], Normalize(184));
  EXPECT_EQ(patch[2], Normalize(188));
  image.GetPatch(18, patch);
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
  // Patch 0 is just the "start" of the image.
  image.GetPatch(0, patch);
  EXPECT_NEAR(patch[0], Normalize(0.0f, max_value), 1e-6);
  EXPECT_NEAR(patch[1], Normalize(1.0f, max_value), 1e-6);
  EXPECT_NEAR(patch[2], Normalize(2.0f, max_value), 1e-6);
  // The "image" has 4x3 patches, so patch 6 has coordinates (1, 2) and its
  // pixel coordinates are offset by (14, 28).
  image.GetPatch(6, patch);
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

}  // namespace
}  // namespace gcpp
