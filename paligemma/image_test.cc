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

#include <string>

#include "gtest/gtest.h"

namespace gcpp {
namespace {

float Normalize(int value) { return 2.0f * (value / 255.0f) - 1.0f; }

TEST(ImageTest, BasicFunctionality) {
  return;  // Need to figure out how to get the external path for the test file.
  std::string path;
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
  image.Resize();
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
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(patch[i], image.data()[(14 * 224 + 2 * 14) * 3 + i]);
  }
}

}  // namespace
}  // namespace gcpp
