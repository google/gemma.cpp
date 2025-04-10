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
#include <string.h>

#include <atomic>
#include <vector>

#include "compression/blob_store.h"
#include "compression/io.h"  // Path
#include "util/allocator.h"
#include "util/basics.h"  // IndexRange
#include "util/threading.h"
#include "util/threading_context.h"
#include "hwy/aligned_allocator.h"  // Span
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/timer.h"

namespace gcpp {

using KeySpan = hwy::Span<const hwy::uint128_t>;

// Returns false if any keys differ, because then blobs are not comparable.
bool CompareKeys(const BlobReader& reader1, const BlobReader& reader2) {
  KeySpan keys1 = reader1.Keys();
  KeySpan keys2 = reader2.Keys();
  if (keys1.size() != keys2.size()) {
    fprintf(stderr, "#keys mismatch: %zu vs %zu\n", keys1.size(), keys2.size());
    return false;
  }
  for (size_t i = 0; i < keys1.size(); ++i) {
    if (keys1[i] != keys2[i]) {
      fprintf(stderr, "key %zu mismatch: %s vs %s\n", i,
              StringFromKey(keys1[i]).c_str(), StringFromKey(keys2[i]).c_str());
      return false;
    }
  }

  return true;
}

// Total amount to allocate for all blobs.
size_t TotalBytes(BlobReader& reader) {
  size_t total_bytes = 0;
  for (const hwy::uint128_t key : reader.Keys()) {
    total_bytes += reader.BlobSize(key);
  }
  return total_bytes;
}

using BytePtr = hwy::AlignedFreeUniquePtr<uint8_t[]>;
using ByteSpan = hwy::Span<uint8_t>;    // Sections within BytePtr
using BlobVec = std::vector<ByteSpan>;  // in order of keys

// Allocates memory within the single allocation and updates `pos`.
BlobVec ReserveMemory(BlobReader& reader, BytePtr& all_blobs, size_t& pos) {
  BlobVec blobs;
  for (const hwy::uint128_t key : reader.Keys()) {
    const size_t bytes = reader.BlobSize(key);
    blobs.push_back(ByteSpan(all_blobs.get() + pos, bytes));
    pos += bytes;
  }
  return blobs;
}

// Reads one set of blobs in parallel (helpful if in disk cache).
void ReadBlobs(BlobReader& reader, BlobVec& blobs, hwy::ThreadPool& pool) {
  HWY_ASSERT(reader.Keys().size() == blobs.size());
  for (size_t i = 0; i < blobs.size(); ++i) {
    reader.Enqueue(reader.Keys()[i], blobs[i].data(), blobs[i].size());
  }
  const BlobError err = reader.ReadAll(pool);
  if (err != 0) {
    HWY_ABORT("Parallel read failed: %d\n", err);
  }
}

// Parallelizes ReadBlobs across (two) packages, if available.
void ReadBothBlobs(BlobReader& reader1, BlobReader& reader2, size_t total_bytes,
                   BlobVec& blobs1, BlobVec& blobs2, NestedPools& pools) {
  const double t0 = hwy::platform::Now();
  fprintf(stderr, "Reading %zu GiB, %zux%zu cores: ", total_bytes >> 30,
          pools.AllPackages().NumWorkers(), pools.Pool().NumWorkers());
  pools.AllPackages().Run(0, 2, [&](size_t task, size_t pkg_idx) {
    ReadBlobs(task ? reader2 : reader1, task ? blobs2 : blobs1,
              pools.Pool(pkg_idx));
  });
  const double t1 = hwy::platform::Now();
  fprintf(stderr, "%.1f GB/s\n", total_bytes / (t1 - t0) * 1E-9);
}

// Returns number of elements with a mismatch. For float and bf16 blobs, uses
// L1 and relative error, otherwise byte-wise comparison.
size_t BlobDifferences(const ByteSpan& data1, const ByteSpan& data2,
                       const hwy::uint128_t key) {
  if (data1.size() != data2.size() || data1.size() == 0) {
    HWY_ABORT("key %s size mismatch: %zu vs %zu\n", StringFromKey(key).c_str(),
              data1.size(), data2.size());
  }

  size_t mismatches = 0;
  char type;
  hwy::CopyBytes(&key, &type, 1);
  if (type == 'F') {
    HWY_ASSERT(data1.size() % sizeof(float) == 0);
    for (size_t j = 0; j < data1.size(); j += sizeof(float)) {
      float f1, f2;
      hwy::CopyBytes(&data1[j], &f1, sizeof(f1));
      hwy::CopyBytes(&data2[j], &f2, sizeof(f2));
      const float l1 = hwy::ScalarAbs(f1 - f2);
      const float rel = hwy::ScalarAbs(f1) == 0.0f ? 0.0f : l1 / f1;
      if (l1 > 1E-3f || rel > 1E-2f) {
        fprintf(stderr, "key %s %5zu: L1 %.5f rel %.4f\n",
                StringFromKey(key).c_str(), j, l1, rel);
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
        fprintf(stderr, "key %s %5zu: L1 %.5f rel %.4f\n",
                StringFromKey(key).c_str(), j, l1, rel);
        ++mismatches;
      }
    }
  } else {
    for (size_t j = 0; j < data1.size(); ++j) {
      if (data1[j] != data2[j]) {
        if (mismatches == 0) {
          fprintf(stderr, "key %s mismatch at byte %5zu\n",
                  StringFromKey(key).c_str(), j);
        }
        ++mismatches;
      }
    }
  }
  return mismatches;
}

void CompareBlobs(const KeySpan& keys, BlobVec& blobs1, BlobVec& blobs2,
                  size_t total_bytes, NestedPools& pools) {
  fprintf(stderr, "Comparing %zu blobs in parallel: ", keys.size());
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
                fprintf(stderr, "key %s has %zu mismatches in %zu bytes!\n",
                        StringFromKey(keys[i]).c_str(), mismatches,
                        blobs1[i].size());
                blobs_diff.fetch_add(1);
              } else {
                blobs_equal.fetch_add(1);
              }
            });
      });
  const double t1 = hwy::platform::Now();
  fprintf(stderr, "%.1f GB/s; total blob matches=%zu, mismatches=%zu\n",
          total_bytes / (t1 - t0) * 1E-9, blobs_equal.load(),
          blobs_diff.load());
}

// Compares two sbs files, including blob order.
void ReadAndCompareBlobs(const char* path1, const char* path2) {
  // Open files.
  BlobReader reader1;
  BlobReader reader2;
  const BlobError err1 = reader1.Open(Path(path1));
  const BlobError err2 = reader2.Open(Path(path2));
  if (err1 != 0 || err2 != 0) {
    HWY_ABORT("Failed to open files: %s %s: %d %d\n", path1, path2, err1, err2);
  }

  if (!CompareKeys(reader1, reader2)) return;

  // Single allocation, avoid initializing the memory.
  const size_t total_bytes = TotalBytes(reader1) + TotalBytes(reader2);
  BytePtr all_blobs = hwy::AllocateAligned<uint8_t>(total_bytes);
  size_t pos = 0;
  BlobVec blobs1 = ReserveMemory(reader1, all_blobs, pos);
  BlobVec blobs2 = ReserveMemory(reader2, all_blobs, pos);

  NestedPools& pools = ThreadingContext2::Get().pools;
  ReadBothBlobs(reader1, reader2, total_bytes, blobs1, blobs2, pools);

  CompareBlobs(reader1.Keys(), blobs1, blobs2, total_bytes, pools);
}

}  // namespace gcpp

int main(int argc, char** argv) {
  if (argc != 3) {
    HWY_ABORT("Usage: %s <sbs_path> <sbs_path>\n", argv[0]);
  }
  if (strcmp(argv[1], argv[2]) == 0) {
    HWY_ABORT("Filenames are the same, skipping comparison: %s\n", argv[1]);
  }
  gcpp::ReadAndCompareBlobs(argv[1], argv[2]);
  return 0;
}
