#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "compression/blob_store.h"
#include "compression/io.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"

namespace gcpp {

// Compares two sbs files, ignoring the order of the blobs.
// Gives up on the first mismatch.
void CompareBlobs(const char* path1, const char* path2) {
  BlobReader reader1;
  HWY_ASSERT(reader1.Open(Path(path1)) == 0);
  BlobReader reader2;
  HWY_ASSERT(reader2.Open(Path(path2)) == 0);
  hwy::Span<const hwy::uint128_t> keys1 = reader1.Keys();
  size_t total_matches = 0;
  size_t total_fails = 0;
  for (size_t i = 0; i < keys1.size(); ++i) {
    fprintf(stderr, "key %s, blob1 size=%zu, blob2 size=%zu\n",
            StringFromKey(keys1[i]).c_str(), reader1.BlobSize(keys1[i]),
            reader2.BlobSize(keys1[i]));
    std::vector<uint8_t> data1(reader1.BlobSize(keys1[i]));
    HWY_ASSERT(reader1.ReadOne(keys1[i], data1.data(), data1.size()) == 0);
    HWY_ASSERT(reader2.BlobSize(keys1[i]) == data1.size());
    std::vector<uint8_t> data2(reader2.BlobSize(keys1[i]));
    HWY_ASSERT(reader2.ReadOne(keys1[i], data2.data(), data2.size()) == 0);
    size_t fails = 0;
    for (size_t j = 0; j < data1.size(); ++j) {
      if (data1[j] != data2[j]) {
        if (fails == 0) {
          fprintf(stderr, "key %s Mismatch at %zu\n",
                  StringFromKey(keys1[i]).c_str(), j);
        }
        ++fails;
      }
    }
    if (fails > 0) {
      fprintf(stderr, "key %s has %.2f%% Mismatch!\n",
              StringFromKey(keys1[i]).c_str(), 100.0 * fails / data1.size());
      ++total_fails;
    } else {
      fprintf(stderr, "key %s Matched!\n", StringFromKey(keys1[i]).c_str());
      ++total_matches;
    }
  }
  fprintf(stderr, "Total matches=%zu, mismatches=%zu\n", total_matches,
          total_fails);
}

}  // namespace gcpp

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <sbs_path> <sbs_path>\n", argv[0]);
    return 1;
  }
  gcpp::CompareBlobs(argv[1], argv[2]);
  return 0;
}
