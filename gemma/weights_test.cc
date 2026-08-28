// Copyright 2026 Google LLC
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

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "compression/compress-inl.h"
#include "compression/types.h"
#include "gemma/gemma_args.h"
#include "gemma/weights_internal.h"
#include "gtest/gtest.h"
#include "hwy/base.h"
#include "io/blob_store.h"
#include "io/io.h"
#include "util/basics.h"
#include "util/mat.h"
#include "util/threading_context.h"

namespace gcpp {
namespace {

class TemporaryBlob {
 public:
  TemporaryBlob() {
    const int fd = mkstemp(path_);
    HWY_ASSERT(fd >= 0);
    HWY_ASSERT(close(fd) == 0);
  }

  ~TemporaryBlob() { unlink(path_); }

  Path path() const { return Path(path_); }

 private:
  char path_[sizeof("/tmp/weights_test.sbs-XXXXXX")] =
      "/tmp/weights_test.sbs-XXXXXX";
};

static ThreadingContext MakeContext() {
  ThreadingArgs args;
  args.max_threads = 2;
  args.pin = Tristate::kFalse;
  args.bind = Tristate::kFalse;
  return ThreadingContext(args);
}

template <typename T>
static float SourceValue(const T value) {
  return hwy::ConvertScalarTo<float>(value);
}

template <typename T>
static T MakeSourceValue(const float value) {
  return hwy::ConvertScalarTo<T>(value);
}

template <typename T>
static void TestSFPConversion(const Type source_type, const float prev_scale,
                              const std::array<float, 7>& pattern) {
  ThreadingContext ctx = MakeContext();

  // Just over 4 MiB, so the conversion has a final partial chunk.
  constexpr size_t kCols = 1024;
  constexpr size_t kRows = 4 * 1024 * 1024 / (kCols * sizeof(T)) + 1;
  const size_t num = kRows * kCols;
  std::vector<T> source(num);
  for (size_t i = 0; i < num; ++i) {
    source[i] = MakeSourceValue<T>(pattern[i % pattern.size()]);
  }

  TemporaryBlob blob;
  {
    BlobWriter writer(blob.path(), ctx);
    writer.Add("embedding", source.data(), source.size() * sizeof(T));
    writer.Finalize();
  }
  BlobReader reader(blob.path());
  const BlobRange* range = reader.Find("embedding");
  ASSERT_NE(range, nullptr);

  MatPtr mat("embedding", Type::kSFP, Extents2D(kRows, kCols));
  mat.SetScale(prev_scale);
  MatOwner owner;
  owner.AllocateFor(mat, ctx.allocator, MatPadding::kOdd);

  const weights_internal::TensorToRead tensor{
      .mat = &mat,
      .range = *range,
      .padding = MatPadding::kOdd,
      .to_sfp = true,
      .prev_type = source_type,
      .prev_packed_bytes = source.size() * sizeof(T),
  };
  weights_internal::ReadAllToSFP({tensor}, reader, ctx);

  float source_maxabs = 0.0f;
  for (const T value : source) {
    source_maxabs =
        std::max(source_maxabs, std::abs(SourceValue(value) * prev_scale));
  }
  const float expected_scale =
      source_maxabs <= SfpStream::kMax ? 1.0f : source_maxabs / SfpStream::kMax;
  EXPECT_FLOAT_EQ(mat.Scale(), expected_scale);

  const MatPtrT<SfpStream> sfp(mat);
  const std::array<size_t, 8> samples = {
      0,
      1,
      kCols - 1,
      (kRows - 1) * kCols - 1,
      (kRows - 1) * kCols,
      (kRows - 1) * kCols + 1,
      num - 2,
      num - 1,
  };
  for (const size_t i : samples) {
    const size_t row = i / kCols;
    const size_t col = i % kCols;
    const float expected = SourceValue(source[i]) * prev_scale;
    const float actual = HWY_NAMESPACE::CompressTraits<SfpStream>::ToFloatSlow(
                             sfp.Row(row)[col]) *
                         mat.Scale();
    const float tolerance = std::max(1E-6f, std::abs(expected) * 0.13f);
    EXPECT_NEAR(actual, expected, tolerance) << "index " << i;
  }
}

TEST(WeightsTest, ReadBF16EmbeddingToSFPWithScaleAndPartialChunk) {
  TestSFPConversion<BF16>(Type::kBF16, 1.25f,
                          {-2.5f, -1.25f, -0.25f, 0.0f, 0.25f, 1.0f, 2.5f});
}

TEST(WeightsTest, ReadF32EmbeddingToSFPWithoutScaleAndPartialChunk) {
  TestSFPConversion<float>(Type::kF32, 1.0f,
                           {-1.75f, -1.0f, -0.125f, 0.0f, 0.125f, 1.0f, 1.75f});
}

TEST(WeightsTest, RejectsMismatchedEmbeddingBlobSize) {
  ThreadingContext ctx = MakeContext();
  TemporaryBlob blob;
  const std::array<BF16, 4> source = {
      hwy::BF16FromF32(-1.0f), hwy::BF16FromF32(0.0f), hwy::BF16FromF32(1.0f),
      hwy::BF16FromF32(2.0f)};
  {
    BlobWriter writer(blob.path(), ctx);
    writer.Add("embedding", source.data(), sizeof(source));
    writer.Finalize();
  }
  BlobReader reader(blob.path());
  const BlobRange* range = reader.Find("embedding");
  ASSERT_NE(range, nullptr);

  MatPtr mat("embedding", Type::kSFP, Extents2D(2, 2));
  const weights_internal::TensorToRead tensor{
      .mat = &mat,
      .range = *range,
      .padding = MatPadding::kOdd,
      .to_sfp = true,
      .prev_type = Type::kBF16,
      .prev_packed_bytes = sizeof(source) + sizeof(BF16),
  };

  EXPECT_DEATH(weights_internal::ReadAllToSFP({tensor}, reader, ctx),
               "tensor.range.bytes == tensor.prev_packed_bytes");
}

TEST(WeightsTest, ExplicitSFPDisablesOnlyAutomaticMapping) {
  ThreadingContext ctx = MakeContext();
  InferenceArgs inference;
  LoaderArgs loader("", "");
  loader.to_bf16 = Tristate::kFalse;

  const uint64_t file_mib = ctx.allocator.TotalMiB() / 3 + 1;
  const uint64_t file_bytes = hwy::RoundUpTo(
      file_mib << 20, static_cast<uint64_t>(ctx.allocator.BasePageBytes()));

  // Establish that the normal automatic heuristic would map this file.
  EXPECT_EQ(weights_internal::ChooseMode(file_bytes, loader, inference,
                                         ctx.allocator),
            WeightsPtrs::Mode::kMap);

  loader.sfp_embedding = Tristate::kTrue;
  EXPECT_EQ(weights_internal::ChooseMode(file_bytes, loader, inference,
                                         ctx.allocator),
            WeightsPtrs::Mode::kRead);

  // An explicit mapping request still wins and is warned about by the loader.
  loader.map = Tristate::kTrue;
  EXPECT_EQ(weights_internal::ChooseMode(file_bytes, loader, inference,
                                         ctx.allocator),
            WeightsPtrs::Mode::kMap);

  // SFP embedding conversion composes with explicit conversion of other
  // tensors to BF16.
  loader.map = Tristate::kDefault;
  loader.to_bf16 = Tristate::kTrue;
  EXPECT_EQ(weights_internal::ChooseMode(file_bytes, loader, inference,
                                         ctx.allocator),
            WeightsPtrs::Mode::kReadBF16);
}

}  // namespace
}  // namespace gcpp
