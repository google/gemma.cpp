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
#include <cstdio>
#include <map>
#include <vector>

#include "compression/blob_store.h"
#include "compression/io.h"
#include "util/allocator.h"
#include "util/threading.h"
#include "hwy/aligned_allocator.h"  // Span
#include "hwy/base.h"

namespace gcpp {

// Returns whether it makes sense to continue comparing.
bool CompareKeys(const BlobReader& reader1, const BlobReader& reader2) {
  hwy::Span<const hwy::uint128_t> keys1 = reader1.Keys();
  hwy::Span<const hwy::uint128_t> keys2 = reader2.Keys();
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

using BlobMap = std::map<hwy::uint128_t, std::vector<uint8_t>>;

size_t TotalBytes(hwy::Span<const hwy::uint128_t>& keys, BlobReader& reader) {
  size_t total_bytes = 0;
  for (const hwy::uint128_t key : keys) {
    total_bytes += reader.BlobSize(key);
  }
  return total_bytes;
}

void ParallelRead(BlobReader& reader, BlobMap& blobs, hwy::ThreadPool& pool) {
  hwy::Span<const hwy::uint128_t> keys = reader.Keys();
  for (const hwy::uint128_t key : keys) {
    const auto ib = blobs.insert({key, {}});
    HWY_ASSERT(ib.second);  // newly inserted, no duplicate keys
    const size_t bytes = reader.BlobSize(key);
    // TODO: AllocateAligned instead, avoids initializing the memory.
    ib.first->second.resize(bytes);
    reader.Enqueue(key, ib.first->second.data(), bytes);
  }
  const BlobError err = reader.ReadAll(pool);
  if (err != 0) {
    HWY_ABORT("Parallel read failed: %d\n", err);
  }
}

// Returns number of elements with a mismatch. For float and bf16 blobs, uses
// L1 and relative error, otherwise byte-wise comparison.
size_t BlobDifferences(BlobMap& blobs1, BlobMap& blobs2,
                       const hwy::uint128_t key) {
  std::vector<uint8_t>& data1 = blobs1[key];
  std::vector<uint8_t>& data2 = blobs2[key];
  if (data1.size() != data2.size() || data1.empty()) {
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

// Compares two sbs files, including blob order.
void CompareBlobs(const char* path1, const char* path2) {
  BlobReader reader1;
  BlobReader reader2;
  const BlobError err1 = reader1.Open(Path(path1));
  const BlobError err2 = reader2.Open(Path(path2));
  if (err1 != 0 || err2 != 0) {
    HWY_ABORT("Failed to open files: %s %s: %d %d\n", path1, path2, err1, err2);
  }
  if (!CompareKeys(reader1, reader2)) return;

  NestedPools pools(0);
  Allocator::Init(pools.Topology());
  hwy::Span<const hwy::uint128_t> keys1 = reader1.Keys();
  BlobMap blobs1, blobs2;
  fprintf(stderr, "Reading 2x %zu GiB, %zu cores...\n",
          TotalBytes(keys1, reader1) >> 30, pools.Pool().NumWorkers());
  ParallelRead(reader1, blobs1, pools.Pool());
  ParallelRead(reader2, blobs2, pools.Pool());

  fprintf(stderr, "Comparing %zu blobs in parallel...\n", keys1.size());
  std::atomic<size_t> blobs_equal{};
  std::atomic<size_t> blobs_diff{};
  pools.Pool().Run(0, keys1.size(), [&](size_t i, size_t /*thread*/) {
    const size_t mismatches = BlobDifferences(blobs1, blobs2, keys1[i]);
    if (mismatches != 0) {
      fprintf(stderr, "key %s has %zu mismatches in %zu bytes!\n",
              StringFromKey(keys1[i]).c_str(), mismatches,
              reader1.BlobSize(keys1[i]));
      blobs_diff.fetch_add(1);
    } else {
      blobs_equal.fetch_add(1);
    }
  });
  fprintf(stderr, "Total blob matches=%zu, mismatches=%zu\n",
          blobs_equal.load(), blobs_diff.load());
}

}  // namespace gcpp

int main(int argc, char** argv) {
  if (argc != 3) {
    HWY_ABORT("Usage: %s <sbs_path> <sbs_path>\n", argv[0]);
  }
  if (strcmp(argv[1], argv[2]) == 0) {
    HWY_ABORT("Filenames are the same, skipping comparison: %s\n", argv[1]);
  }
  gcpp::CompareBlobs(argv[1], argv[2]);
  return 0;
}
