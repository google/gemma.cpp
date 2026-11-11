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

#include "gemma/weights.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "compression/compress-inl.h"
#include "compression/types.h"
#include "gemma/gemma_args.h"
#include "gemma/weights_internal.h"
#include "io/blob_store.h"
#include "io/io.h"
#include "util/basics.h"
#include "util/mat.h"
#include "util/threading_context.h"
#include "hwy/base.h"

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

TEST(WeightsTest, MapsBlobAlignedNonPageAlignedFile) {
  ThreadingContext ctx = MakeContext();
  const size_t file_bytes = ctx.allocator.BasePageBytes() + kBlobAlign;
  ASSERT_EQ(file_bytes % kBlobAlign, 0);
  ASSERT_NE(file_bytes % ctx.allocator.BasePageBytes(), 0);
  std::vector<uint8_t> contents(file_bytes);
  contents.front() = 1;
  contents.back() = 2;

  TemporaryBlob blob;
  auto file = OpenFileOrAbort(blob.path(), "w+");
  ASSERT_TRUE(file->Write(contents.data(), contents.size(), 0));
  MapPtr mapped = file->Map();
  ASSERT_NE(mapped, nullptr);
  EXPECT_EQ(mapped[0], contents.front());
  EXPECT_EQ(mapped[file_bytes - 1], contents.back());

  InferenceArgs inference;
  LoaderArgs loader("", "");
  loader.map = Tristate::kTrue;
  loader.to_bf16 = Tristate::kFalse;
  EXPECT_EQ(weights_internal::ChooseMode(file_bytes, loader, inference,
                                         ctx.allocator),
            WeightsPtrs::Mode::kMap);
}

TEST(WeightsTest, MappingRequiresBlobAlignment) {
  ThreadingContext ctx = MakeContext();
  InferenceArgs inference;
  LoaderArgs loader("", "");
  loader.to_bf16 = Tristate::kFalse;

  // Large enough for the automatic mapping heuristic to select kMap.
  const uint64_t file_mib = ctx.allocator.TotalMiB() / 3 + 1;
  const uint64_t page_bytes = ctx.allocator.BasePageBytes();
  const uint64_t page_aligned_bytes =
      hwy::RoundUpTo(file_mib << 20, page_bytes);

  for (const Tristate map : {Tristate::kDefault, Tristate::kTrue}) {
    loader.map = map;
    for (const uint64_t file_bytes :
         {page_aligned_bytes, page_aligned_bytes + kBlobAlign}) {
      EXPECT_EQ(weights_internal::ChooseMode(file_bytes, loader, inference,
                                             ctx.allocator),
                WeightsPtrs::Mode::kMap)
          << "file_bytes=" << file_bytes;
    }
    // Cover both an almost-full page (Jan's 4090-byte example) and a file
    // whose final page has plenty of space but lacks blob alignment.
    for (const uint64_t file_bytes :
         {page_bytes - 6, page_bytes + 1, page_aligned_bytes - 6,
          page_aligned_bytes + 1}) {
      EXPECT_EQ(weights_internal::ChooseMode(file_bytes, loader, inference,
                                             ctx.allocator),
                WeightsPtrs::Mode::kRead)
          << "file_bytes=" << file_bytes;
    }
  }

  loader.map = Tristate::kTrue;
  loader.to_bf16 = Tristate::kTrue;
  EXPECT_EQ(weights_internal::ChooseMode(page_aligned_bytes + 1, loader,
                                         inference, ctx.allocator),
            WeightsPtrs::Mode::kReadBF16);
}

static void WriteModelBlobWithToc(const Path& path, ThreadingContext& ctx,
                                  const std::vector<uint32_t>& toc) {
  ModelConfig config(Model::GEMMA2_2B, Type::kBF16, PromptWrapping::GEMMA_PT);
  GemmaTokenizer tokenizer(kMockTokenizer);
  const std::array<BF16, 16> tensor_data = {};
  BlobWriter writer(path, ctx);
  writer.Add("w_25", tensor_data.data(), sizeof(tensor_data));
  WriteSingleFile(config, tokenizer, toc, writer);
}

TEST(WeightsTest, SetTypeRejectsNumElementsOverflow) {
  // 524296 * 8192 = 4,295,032,832 > UINT32_MAX (b/583871800).
  MatPtr mat("emb", Type::kUnknown, Extents2D(524296, 8192));
  EXPECT_DEATH(mat.SetType(Type::kBF16), "extents\\.rows <=");

  EXPECT_DEATH(
      MatPtr("big", Type::kBF16, Extents2D(uint64_t{UINT32_MAX} + 1, 1)),
      "extents\\.rows <=");

  MatPtr non_empty("ne", Type::kBF16, Extents2D(4, 8));
  EXPECT_DEATH(non_empty.OverrideRows(0), "rows > 0");
  EXPECT_DEATH(non_empty.OverrideCols(0), "cols > 0");

  MatPtr default_mat;
  EXPECT_EQ(default_mat.GetType(), Type::kUnknown);
  EXPECT_EQ(default_mat.Stride(), 0u);

  MatPtr reshaped("rs", Type::kBF16, Extents2D(6, 4));
  reshaped.ReshapePackedRowsToCols(2);
  EXPECT_EQ(reshaped.Rows(), 3u);
  EXPECT_EQ(reshaped.Cols(), 8u);
  EXPECT_EQ(reshaped.Stride(), 8u);
  EXPECT_EQ(reshaped.PackedBytes(), 48u);

  MatPtr odd("odd", Type::kBF16, Extents2D(5, 8));
  EXPECT_DEATH(odd.ReshapePackedRowsToCols(2), "private_rows_ % factor");

  MatPtr overridden("ov", Type::kBF16, Extents2D(4, 8));
  overridden.OverrideRows(2);
  EXPECT_DEATH(overridden.ReshapePackedRowsToCols(2), "override_rows_ == 0");
}

TEST(WeightsTest, ModelStoreValidatesDeserializedTocEntries) {
  ThreadingContext ctx = MakeContext();
  MatPtr valid("w_25", Type::kBF16, Extents2D(4, 4));
  valid.SetScale(1.5f);
  const std::vector<uint32_t> base_toc = valid.Write();
  ASSERT_EQ(base_toc.size(), 10u);

  // Full TOC roundtrip and legacy TOC omitting `stride_` (missing_fields == 1).
  std::vector<uint32_t> legacy_toc = base_toc;
  legacy_toc.pop_back();
  legacy_toc[0] -= 1;
  for (const std::vector<uint32_t>& toc : {base_toc, legacy_toc}) {
    TemporaryBlob blob;
    WriteModelBlobWithToc(blob.path(), ctx, toc);
    BlobReader reader(blob.path());
    ModelStore store(reader);
    const MatPtr* loaded = store.FindMat("w_25");
    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->GetType(), Type::kBF16);
    EXPECT_EQ(loaded->Rows(), 4u);
    EXPECT_EQ(loaded->Cols(), 4u);
    EXPECT_EQ(loaded->Stride(), 4u);
    EXPECT_EQ(loaded->PackedBytes(), 32u);
    EXPECT_FLOAT_EQ(loaded->Scale(), 1.5f);
  }

  const auto expect_toc_death = [&](const std::vector<uint32_t>& bad_toc,
                                    const char* regex) {
    TemporaryBlob blob;
    WriteModelBlobWithToc(blob.path(), ctx, bad_toc);
    BlobReader reader(blob.path());
    EXPECT_DEATH({ ModelStore store(reader); }, regex);
  };

  // rows * cols = 524296 * 8192 > UINT32_MAX (b/583871800).
  std::vector<uint32_t> overflow_toc = base_toc;
  overflow_toc[6] = 524296;
  overflow_toc[7] = 8192;
  overflow_toc[9] = 8192;
  expect_toc_death(overflow_toc, "extents\\.rows <=");

  // stride < cols.
  std::vector<uint32_t> bad_stride_toc = base_toc;
  bad_stride_toc[9] = 2;
  expect_toc_death(bad_stride_toc, "stride 2 < cols 4");

  // Type::kUnknown in TOC.
  std::vector<uint32_t> unknown_type_toc = base_toc;
  unknown_type_toc[3] = static_cast<uint32_t>(Type::kUnknown);
  expect_toc_death(unknown_type_toc, "mat\\.GetType\\(\\) != Type::kUnknown");

  // Truncated TOC entry stopping right after `name_` (missing_fields == 7).
  std::vector<uint32_t> truncated_toc = {2, 1, base_toc[2]};
  expect_toc_death(truncated_toc, "result\\.missing_fields <= 1");
}

}  // namespace
}  // namespace gcpp
