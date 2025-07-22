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

#include <stdint.h>
#include <stdio.h>
#include <string.h>  // strcmp

#include <atomic>
#include <string>
#include <vector>

#include "io/blob_store.h"
#include "io/io.h"        // Path
#include "util/basics.h"  // IndexRange
#include "util/threading.h"
#include "util/threading_context.h"
#include "hwy/aligned_allocator.h"  // Span
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/timer.h"

namespace gcpp {

// Aborts if any keys differ, because then blobs are not comparable.
void CompareKeys(const BlobReader& reader1, const BlobReader& reader2) {
  if (reader1.Keys().size() != reader2.Keys().size()) {
    HWY_ABORT("#keys mismatch: %zu vs %zu\n", reader1.Keys().size(),
              reader2.Keys().size());
  }
  for (size_t i = 0; i < reader1.Keys().size(); ++i) {
    if (reader1.Keys()[i] != reader2.Keys()[i]) {
      HWY_ABORT("key %zu mismatch: %s vs %s\n", i, reader1.Keys()[i].c_str(),
                reader2.Keys()[i].c_str());
    }
  }
}

using KeyVec = std::vector<std::string>;
using RangeVec = std::vector<BlobRange>;

RangeVec AllRanges(const KeyVec& keys, const BlobReader& reader) {
  RangeVec ranges;
  ranges.reserve(keys.size());
  for (const std::string& key : keys) {
    const BlobRange* range = reader.Find(key);
    if (!range) {
      HWY_ABORT("Key %s not found, but was in KeyVec\n", key.c_str());
    }
    ranges.push_back(*range);
  }
  return ranges;
}

// Aborts if any sizes differ, because that already guarantees a mismatch.
void CompareRangeSizes(const KeyVec& keys, const RangeVec& ranges1,
                       const RangeVec& ranges2) {
  HWY_ASSERT(keys.size() == ranges1.size());
  HWY_ASSERT(keys.size() == ranges2.size());
  for (size_t i = 0; i < ranges1.size(); ++i) {
    // Tolerate differing key_idx and offset because blobs may be in different
    // order in the two files.
    if (ranges1[i].bytes != ranges2[i].bytes) {
      HWY_ABORT("range #%zu (%s) size mismatch: %zu vs %zu\n", i,
                keys[i].c_str(), ranges1[i].bytes, ranges2[i].bytes);
    }
  }
}

// Total amount to allocate for all blobs.
size_t TotalBytes(const RangeVec& ranges) {
  size_t total_bytes = 0;
  for (const BlobRange& range : ranges) {
    total_bytes += range.bytes;
  }
  return total_bytes;
}

using BytePtr = hwy::AlignedFreeUniquePtr<uint8_t[]>;
using ByteSpan = hwy::Span<uint8_t>;    // Sections within BytePtr
using BlobVec = std::vector<ByteSpan>;  // in order of keys

// Assigns pointers within the single allocation and updates `pos`.
BlobVec ReserveMemory(const RangeVec& ranges, BytePtr& all_blobs, size_t& pos) {
  BlobVec blobs;
  for (const BlobRange& range : ranges) {
    blobs.push_back(ByteSpan(all_blobs.get() + pos, range.bytes));
    pos += range.bytes;
  }
  return blobs;
}

// Reads one set of blobs in parallel (helpful if in disk cache).
// Aborts on error.
void ReadBlobs(BlobReader& reader, const RangeVec& ranges, BlobVec& blobs,
               hwy::ThreadPool& pool) {
  HWY_ASSERT(reader.Keys().size() == blobs.size());
  HWY_ASSERT(ranges.size() == blobs.size());
  pool.Run(0, blobs.size(), [&](size_t i, size_t /*thread*/) {
    HWY_ASSERT(ranges[i].bytes == blobs[i].size());
    reader.file().Read(ranges[i].offset, ranges[i].bytes, blobs[i].data());
  });
}

// Parallelizes ReadBlobs across (two) packages, if available.
void ReadBothBlobs(BlobReader& reader1, BlobReader& reader2,
                   const RangeVec& ranges1, const RangeVec& ranges2,
                   size_t total_bytes, BlobVec& blobs1, BlobVec& blobs2,
                   NestedPools& pools) {
  const double t0 = hwy::platform::Now();
  HWY_WARN("Reading %zu GiB, %zux%zu cores: ", total_bytes >> 30,
           pools.AllPackages().NumWorkers(), pools.Pool().NumWorkers());
  pools.AllPackages().Run(0, 2, [&](size_t task, size_t pkg_idx) {
    ReadBlobs(task ? reader2 : reader1, task ? ranges2 : ranges1,
              task ? blobs2 : blobs1, pools.Pool(pkg_idx));
  });
  const double t1 = hwy::platform::Now();
  HWY_WARN("%.1f GB/s\n", total_bytes / (t1 - t0) * 1E-9);
}

// Returns number of elements with a mismatch. For float and bf16 blobs, uses
// L1 and relative error, otherwise byte-wise comparison.
size_t BlobDifferences(const ByteSpan data1, const ByteSpan data2,
                       const std::string& key) {
  if (data1.size() != data2.size() || data1.size() == 0) {
    HWY_ABORT("key %s size mismatch: %zu vs %zu\n", key.c_str(), data1.size(),
              data2.size());
  }

  size_t mismatches = 0;
  const char type = key[0];
  if (type == 'F') {
    HWY_ASSERT(data1.size() % sizeof(float) == 0);
    for (size_t j = 0; j < data1.size(); j += sizeof(float)) {
      float f1, f2;
      hwy::CopyBytes(&data1[j], &f1, sizeof(f1));
      hwy::CopyBytes(&data2[j], &f2, sizeof(f2));
      const float l1 = hwy::ScalarAbs(f1 - f2);
      const float rel = hwy::ScalarAbs(f1) == 0.0f ? 0.0f : l1 / f1;
      if (l1 > 1E-3f || rel > 1E-2f) {
        HWY_WARN("key %s %5zu: L1 %.5f rel %.4f\n", key.c_str(), j, l1, rel);
        ++mismatches;
      }
    }
  } else if (type == 'B') {
    for (size_t j = 0; j < data1.size(); j += sizeof(hwy::bfloat16_t)) {
      hwy::bfloat16_t b1, b2;
      hwy::CopyBytes(&data1[j], &b1, sizeof(b1));
      hwy::CopyBytes(&data2[j], &b2, sizeof(b2));
      const float f1 = hwy::ConvertScalarTo<float>(b1);
      const float f2 = hwy::ConvertScalarTo<float>(b2);
      const float l1 = hwy::ScalarAbs(f1 - f2);
      const float rel = hwy::ScalarAbs(f1) == 0.0f ? 0.0f : l1 / f1;
      if (l1 > 1E-2f || rel > 1E-1f) {
        HWY_WARN("key %s %5zu: L1 %.5f rel %.4f\n", key.c_str(), j, l1, rel);
        ++mismatches;
      }
    }
  } else {
    for (size_t j = 0; j < data1.size(); ++j) {
      if (data1[j] != data2[j]) {
        if (mismatches == 0) {
          HWY_WARN("key %s mismatch at byte %5zu\n", key.c_str(), j);
        }
        ++mismatches;
      }
    }
  }
  return mismatches;
}

void CompareBlobs(const KeyVec& keys, BlobVec& blobs1, BlobVec& blobs2,
                  size_t total_bytes, NestedPools& pools) {
  HWY_WARN("Comparing %zu blobs in parallel: ", keys.size());
  const double t0 = hwy::platform::Now();
  std::atomic<size_t> blobs_equal{};
  std::atomic<size_t> blobs_diff{};
  const IndexRangePartition ranges = StaticPartition(
      IndexRange(0, keys.size()), pools.AllPackages().NumWorkers(), 1);
  ParallelizeOneRange(
      ranges, pools.AllPackages(),
      [&](const IndexRange& range, size_t pkg_idx) {
        pools.Pool(pkg_idx).Run(
            range.begin(), range.end(), [&](size_t i, size_t /*thread*/) {
              const size_t mismatches =
                  BlobDifferences(blobs1[i], blobs2[i], keys[i]);
              if (mismatches != 0) {
                HWY_WARN("key %s has %zu mismatches in %zu bytes!\n",
                         keys[i].c_str(), mismatches, blobs1[i].size());
                blobs_diff.fetch_add(1);
              } else {
                blobs_equal.fetch_add(1);
              }
            });
      });
  const double t1 = hwy::platform::Now();
  HWY_WARN("%.1f GB/s; total blob matches=%zu, mismatches=%zu\n",
           total_bytes / (t1 - t0) * 1E-9, blobs_equal.load(),
           blobs_diff.load());
}

// Compares two sbs files, including blob order.
void ReadAndCompareBlobs(const Path& path1, const Path& path2) {
  BlobReader reader1(path1);
  BlobReader reader2(path2);

  CompareKeys(reader1, reader2);
  const RangeVec ranges1 = AllRanges(reader1.Keys(), reader1);
  const RangeVec ranges2 = AllRanges(reader2.Keys(), reader2);
  CompareRangeSizes(reader1.Keys(), ranges1, ranges2);

  // Single allocation, avoid initializing the memory.
  const size_t total_bytes = TotalBytes(ranges1) + TotalBytes(ranges2);
  BytePtr all_blobs = hwy::AllocateAligned<uint8_t>(total_bytes);
  size_t pos = 0;
  BlobVec blobs1 = ReserveMemory(ranges1, all_blobs, pos);
  BlobVec blobs2 = ReserveMemory(ranges2, all_blobs, pos);

  ThreadingArgs args;
  ThreadingContext ctx(args);
  ReadBothBlobs(reader1, reader2, ranges1, ranges2, total_bytes, blobs1, blobs2,
                ctx.pools);

  CompareBlobs(reader1.Keys(), blobs1, blobs2, total_bytes, ctx.pools);
}

}  // namespace gcpp

int main(int argc, char** argv) {
  if (argc != 3) {
    HWY_ABORT("Usage: %s <sbs_path> <sbs_path>\n", argv[0]);
  }
  if (strcmp(argv[1], argv[2]) == 0) {
    HWY_ABORT("Filenames are the same, skipping comparison: %s\n", argv[1]);
  }
  gcpp::ReadAndCompareBlobs(gcpp::Path(argv[1]), gcpp::Path(argv[2]));
  return 0;
}
