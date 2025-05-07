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

#include "io/blob_store.h"

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>  // std::move
#include <vector>

#include "io/io.h"
#include "util/threading_context.h"
#include "hwy/aligned_allocator.h"  // Span
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/detect_compiler_arch.h"
#include "hwy/profiler.h"

namespace gcpp {

static_assert(HWY_IS_LITTLE_ENDIAN, "Assumes little endian");

// Each blob offset is a multiple of this, an upper bound on SVE vectors and
// usually also larger than L2 cache lines. This is useful when memory mapping
// the entire file, because offset alignment then determines the alignment of
// the blob in memory. Aligning each blob to the (largest) page size would be
// too wasteful, see `kEndAlign`.
constexpr size_t kBlobAlign = 256;  // test also hard-codes this value

// Linux mmap requires the file to be a multiple of the (base) page size, which
// can be up to 64 KiB on Arm. Apple uses 16 KiB, most others use 4 KiB.
constexpr size_t kEndAlign = 64 * 1024;

constexpr size_t kU128Bytes = sizeof(hwy::uint128_t);

// Conversion between strings (<= `kU128Bytes` chars) and the fixed-size u128
// used to store them on disk.
static hwy::uint128_t KeyFromString(const char* string) {
  size_t length = 0;
  for (size_t i = 0; string[i] != '\0'; ++i) {
    ++length;
  }
  if (length > kU128Bytes) {
    HWY_ABORT("Key %s is too long, please truncate to 16 chars.", string);
  }
  HWY_ASSERT(length != 0);

  hwy::uint128_t ret;
  hwy::ZeroBytes<sizeof(ret)>(&ret);
  hwy::CopyBytes(string, &ret, length);
  return ret;
}

static std::string StringFromKey(hwy::uint128_t key) {
  std::string name(sizeof(key) + 1, '\0');
  hwy::CopyBytes(&key, name.data(), sizeof(key));
  name.resize(name.find('\0'));
  return name;
}

namespace {
#pragma pack(push, 1)
struct Header {             // standard layout class
  uint32_t magic = 0;       // kMagic
  uint32_t num_blobs = 0;   // never zero
  uint64_t file_bytes = 0;  // must match actual size of file
};
#pragma pack(pop)
static_assert(sizeof(Header) == 16);
}  // namespace

// Little-endian on-disk representation: a fixed-size `Header`, then a padded
// variable-length 'directory' of blob keys and their offset/sizes, then the
// 'payload' of each blob's data with padding in between, followed by padding to
// `kEndAlign`. Keys are unique, opaque 128-bit keys.
//
// The file format deliberately omits a version number because it is unchanging.
// Additional data may be added only inside new blobs. Changes to the blob
// contents or type should be handled by renaming keys.
//
// This class is for internal use by `BlobReader` and `BlobWriter`. Its
// interface is more low-level: fixed-size keys instead of strings.
class BlobStore {
  static constexpr uint32_t kMagic = 0x0A534253;  // SBS\n

  // Arbitrary upper limit to avoid allocating a huge vector.
  static constexpr size_t kMaxBlobs = 64 * 1024;

  // Returns the end of the directory, including padding, which is also the
  // start of the first payload. `num_blobs` is `NumBlobs()` if the header is
  // already available, otherwise the number of blobs to be written.
  static HWY_CXX17_CONSTEXPR size_t PaddedDirEnd(size_t num_blobs) {
    HWY_ASSERT(num_blobs < kMaxBlobs);
    // Per blob, a key and offset/size.
    return RoundUpToAlign(sizeof(Header) + 2 * kU128Bytes * num_blobs);
  }

  static uint64_t PaddedPayloadBytes(size_t num_blobs,
                                     const hwy::Span<const uint8_t> blobs[]) {
    uint64_t total_payload_bytes = 0;
    for (size_t i = 0; i < num_blobs; ++i) {
      total_payload_bytes += RoundUpToAlign(blobs[i].size());
    }
    // Do not round up to `kEndAlign` because the padding also depends on the
    // directory size. Here we only count the payload.
    return total_payload_bytes;
  }

  static void EnsureUnique(hwy::Span<const hwy::uint128_t> keys) {
    std::unordered_set<std::string> key_set;
    for (const hwy::uint128_t key : keys) {
      HWY_ASSERT(key_set.insert(StringFromKey(key)).second);  // ensure inserted
    }
  }

 public:
  template <typename T>
  static T RoundUpToAlign(T size_or_offset) {
    return hwy::RoundUpTo(size_or_offset, kBlobAlign);
  }

  // Reads header/directory from file.
  explicit BlobStore(const File& file) {
    if (!file.Read(0, sizeof(header_), &header_)) {
      HWY_WARN("Failed to read BlobStore header.");
      return;
    }
    // Avoid allocating a huge vector.
    if (header_.num_blobs >= kMaxBlobs) {
      HWY_WARN("Too many blobs, likely corrupt file.");
      return;
    }

    const size_t padded_dir_end = PaddedDirEnd(NumBlobs());
    const size_t padded_dir_bytes = padded_dir_end - sizeof(header_);
    HWY_ASSERT(padded_dir_bytes % kU128Bytes == 0);
    directory_.resize(padded_dir_bytes / kU128Bytes);
    if (!file.Read(sizeof(header_), padded_dir_bytes, directory_.data())) {
      HWY_WARN("Failed to read BlobStore directory.");
      return;
    }
  }

  // Initializes header/directory for writing to disk.
  BlobStore(size_t num_blobs, const hwy::uint128_t keys[],
            const hwy::Span<const uint8_t> blobs[]) {
    HWY_ASSERT(num_blobs < kMaxBlobs);  // Ensures safe to cast to u32.
    HWY_ASSERT(keys && blobs);
    EnsureUnique(hwy::Span<const hwy::uint128_t>(keys, num_blobs));

    uint64_t offset = PaddedDirEnd(num_blobs);
    const size_t padded_dir_bytes =
        static_cast<size_t>(offset) - sizeof(header_);

    header_.magic = kMagic;
    header_.num_blobs = static_cast<uint32_t>(num_blobs);
    header_.file_bytes = hwy::RoundUpTo(
        offset + PaddedPayloadBytes(num_blobs, blobs), kEndAlign);

    HWY_ASSERT(padded_dir_bytes % kU128Bytes == 0);
    directory_.resize(padded_dir_bytes / kU128Bytes);
    hwy::CopyBytes(keys, directory_.data(), num_blobs * kU128Bytes);
    EnsureUnique(Keys());
    // `SetRange` below will fill `directory_[num_blobs, 2 * num_blobs)`.
    hwy::ZeroBytes(directory_.data() + 2 * num_blobs,
                   padded_dir_bytes - 2 * num_blobs * kU128Bytes);

    // We already zero-initialized the directory padding;
    // `BlobWriter::WriteAll` takes care of padding after each blob via an
    // additional I/O.
    for (size_t i = 0; i < num_blobs; ++i) {
      HWY_ASSERT(blobs[i].data() != nullptr);
      SetRange(i, offset, blobs[i].size());
      offset = RoundUpToAlign(offset + blobs[i].size());
    }
    // When writing new files, we always pad to `kEndAlign`.
    HWY_ASSERT(hwy::RoundUpTo(offset, kEndAlign) == header_.file_bytes);
  }

  // Must be checked by readers before other methods.
  bool IsValid(const uint64_t file_size) const {
    // Ctor failed and already printed a warning.
    if (directory_.empty()) return false;

    if (header_.magic != kMagic) {
      HWY_WARN("Given file is not a BlobStore (magic %08x).", header_.magic);
      return false;
    }
    if (header_.num_blobs == 0) {
      HWY_WARN("Invalid BlobStore (empty), likely corrupt file.");
      return false;
    }
    if (header_.file_bytes != file_size) {
      HWY_WARN("File length %zu does not match header %zu (truncated?).",
               static_cast<size_t>(file_size),
               static_cast<size_t>(header_.file_bytes));
      return false;
    }

    // Ensure blobs are back to back.
    uint64_t expected_offset = PaddedDirEnd(NumBlobs());
    for (size_t key_idx = 0; key_idx < NumBlobs(); ++key_idx) {
      uint64_t actual_offset;
      size_t bytes;
      GetRange(key_idx, actual_offset, bytes);
      if (expected_offset != actual_offset) {
        HWY_WARN("Invalid BlobStore: blob %zu at offset %zu but expected %zu.",
                 key_idx, static_cast<size_t>(actual_offset),
                 static_cast<size_t>(expected_offset));
        return false;
      }
      expected_offset = RoundUpToAlign(expected_offset + bytes);
    }
    // Previously files were not padded to `kEndAlign`, so also allow that.
    if (expected_offset != header_.file_bytes &&
        hwy::RoundUpTo(expected_offset, kEndAlign) != header_.file_bytes) {
      HWY_WARN("Invalid BlobStore: end of blobs %zu but file size %zu.",
               static_cast<size_t>(expected_offset),
               static_cast<size_t>(header_.file_bytes));
      return false;
    }

    return true;  // all OK
  }

  void EnqueueWriteForHeaderAndDirectory(std::vector<BlobIO2>& writes) const {
    const size_t key_idx = 0;  // not actually associated with a key/blob
    writes.emplace_back(
        BlobRange{.offset = 0, .bytes = sizeof(header_), .key_idx = key_idx},
        // members are const and BlobIO2 requires non-const pointers, and they
        // are not modified by file writes.
        const_cast<Header*>(&header_));
    writes.emplace_back(
        BlobRange{.offset = sizeof(header_),
                  .bytes = PaddedDirEnd(NumBlobs()) - sizeof(header_),
                  .key_idx = key_idx},
        const_cast<hwy::uint128_t*>(directory_.data()));
  }

  size_t NumBlobs() const { return static_cast<size_t>(header_.num_blobs); }

  // Not the entirety of `directory_`! The second half is offset/size.
  hwy::Span<const hwy::uint128_t> Keys() const {
    return hwy::Span<const hwy::uint128_t>(directory_.data(), NumBlobs());
  }

  // Retrieves blob's offset and size, not including padding.
  void GetRange(size_t key_idx, uint64_t& offset, size_t& bytes) const {
    HWY_ASSERT(key_idx < NumBlobs());
    const hwy::uint128_t val = directory_[NumBlobs() + key_idx];
    offset = val.lo;
    bytes = val.hi;
    HWY_ASSERT(offset % kBlobAlign == 0);
    HWY_ASSERT(bytes != 0);
    HWY_ASSERT(offset + bytes <= header_.file_bytes);
  }

 private:
  // Stores offset and range into u128 following the keys, so the directory
  // can be one array of the same type, and read/written together with keys.
  void SetRange(size_t key_idx, uint64_t offset, size_t bytes) {
    HWY_ASSERT(key_idx < NumBlobs());
    HWY_ASSERT(offset % kBlobAlign == 0);
    HWY_ASSERT(bytes != 0);
    HWY_ASSERT(offset + bytes <= header_.file_bytes);
    hwy::uint128_t& val = directory_[NumBlobs() + key_idx];
    val.lo = offset;
    val.hi = bytes;
  }

  Header header_;

  std::vector<hwy::uint128_t> directory_;  // two per blob, see `SetRange`.
};  // BlobStore

BlobReader::BlobReader(std::unique_ptr<File> file, uint64_t file_bytes,
                       const BlobStore& bs, BlobReader::Mode mode)
    : file_(std::move(file)), file_bytes_(file_bytes), mode_(mode) {
  HWY_ASSERT(file_ && file_bytes_ != 0);

  keys_.reserve(bs.NumBlobs());
  for (const hwy::uint128_t key : bs.Keys()) {
    keys_.push_back(StringFromKey(key));
  }

  ranges_.reserve(bs.NumBlobs());
  // Populate hash map for O(1) lookups.
  for (size_t key_idx = 0; key_idx < keys_.size(); ++key_idx) {
    uint64_t offset;
    size_t bytes;
    bs.GetRange(key_idx, offset, bytes);
    ranges_.emplace_back(
        BlobRange{.offset = offset, .bytes = bytes, .key_idx = key_idx});
    key_idx_for_key_[keys_[key_idx]] = key_idx;
  }

  if (mode_ == Mode::kMap) {
    const Allocator& allocator = ThreadingContext::Get().allocator;
    // Verify `kEndAlign` is an upper bound on the page size.
    if (kEndAlign % allocator.BasePageBytes() != 0) {
      HWY_ABORT("Please raise an issue about kEndAlign %zu %% page size %zu.",
                kEndAlign, allocator.BasePageBytes());
    }
    if (file_bytes_ % allocator.BasePageBytes() == 0) {
      mapped_ = file_->Map();
      if (!mapped_) {
        HWY_WARN("Failed to map file (%zu KiB), reading instead.",
                 static_cast<size_t>(file_bytes_ >> 10));
        mode_ = Mode::kRead;  // Switch to kRead and continue.
      }
    } else {
      HWY_WARN("Unable to map non-padded file (%zu, %zu), reading instead.",
               static_cast<size_t>(file_bytes_ >> 10),
               allocator.BasePageBytes());
      mode_ = Mode::kRead;  // Switch to kRead and continue.
    }
  }

  if (mode_ == Mode::kRead) {
    // Potentially one per tensor row, so preallocate many.
    requests_.reserve(2 << 20);
  }
}

void BlobReader::Enqueue(const BlobRange& range, void* data) {
  // Debug-only because there may be many I/O requests (per row).
  if constexpr (HWY_IS_DEBUG_BUILD) {
    HWY_DASSERT(!IsMapped());
    HWY_DASSERT(range.offset != 0 && range.bytes != 0 && data != nullptr);
    const BlobRange& blob_range = Range(range.key_idx);
    HWY_DASSERT(blob_range.End() <= file_bytes_);
    if (range.End() > blob_range.End()) {
      HWY_ABORT(
          "Bug: want to read %zu bytes of %s until %zu, past blob end %zu.",
          range.bytes, keys_[range.key_idx].c_str(),
          static_cast<size_t>(range.End()),
          static_cast<size_t>(blob_range.End()));
    }
  }
  requests_.emplace_back(range, data);
}

// Parallel synchronous I/O. Alternatives considered:
// - readv is limited to 0x7FFFF000 bytes on Linux (even 64-bit). Note that
//   pread calls preadv with a single iovec.
//   TODO: use preadv for per-tensor batches of sysconf(_SC_IOV_MAX) / IOV_MAX.
// - O_DIRECT seems undesirable because we do want to use the OS cache
//   between consecutive runs.
void BlobReader::ReadAll(hwy::ThreadPool& pool) const {
  PROFILER_ZONE("Startup.ReadAll");
  HWY_ASSERT(!IsMapped());
  // >5x speedup from parallel reads when cached.
  pool.Run(0, requests_.size(), [this](uint64_t i, size_t /*thread*/) {
    const BlobRange& range = requests_[i].range;
    const uint64_t end = range.End();
    const std::string& key = keys_[range.key_idx];
    const BlobRange& blob_range = Range(range.key_idx);
    HWY_ASSERT(blob_range.End() <= file_bytes_);
    if (end > blob_range.End()) {
      HWY_ABORT(
          "Bug: want to read %zu bytes of %s until %zu, past blob end %zu.",
          range.bytes, key.c_str(), static_cast<size_t>(end),
          static_cast<size_t>(blob_range.End()));
    }
    if (!file_->Read(range.offset, range.bytes, requests_[i].data)) {
      HWY_ABORT("Read failed for %s from %zu, %zu bytes to %p.", key.c_str(),
                static_cast<size_t>(range.offset), range.bytes,
                requests_[i].data);
    }
  });
}

// Decides whether to read or map the file.
static BlobReader::Mode ChooseMode(uint64_t file_mib, Tristate map) {
  const Allocator& allocator = ThreadingContext::Get().allocator;
  // User has explicitly requested a map or read via args.
  if (map == Tristate::kTrue) return BlobReader::Mode::kMap;
  if (map == Tristate::kFalse) return BlobReader::Mode::kRead;
  // Else: use heuristics to choose. Note that `FreeMiB` is generally low
  // because idle memory is used as cache, so do not use it to decide.
  const size_t total_mib = allocator.TotalMiB();
  if (file_mib > total_mib) {
    HWY_WARN("Weight file %zu MiB > detected memory %zu MiB.",
             static_cast<size_t>(file_mib), total_mib);
  }
  // Large fraction of total.
  if (file_mib >= total_mib / 3) return BlobReader::Mode::kMap;
  // Big enough that even parallel loading wouldn't be quick.
  if (file_mib > 50 * 1024) return BlobReader::Mode::kMap;
  return BlobReader::Mode::kRead;
}

std::unique_ptr<BlobReader> BlobReader::Make(const Path& blob_path,
                                             const Tristate map) {
  if (blob_path.Empty()) HWY_ABORT("No --weights specified.");
  std::unique_ptr<File> file = OpenFileOrNull(blob_path, "r");
  if (!file) HWY_ABORT("Failed to open file %s", blob_path.path.c_str());
  const uint64_t file_bytes = file->FileSize();
  if (file_bytes == 0) HWY_ABORT("Zero-sized file %s", blob_path.path.c_str());

  // Even if `kMap`, read the directory via the `kRead` mode for simplicity.
  BlobStore bs(*file);
  if (!bs.IsValid(file_bytes)) {
    return std::unique_ptr<BlobReader>();  // IsValid already printed a warning
  }

  return std::unique_ptr<BlobReader>(new BlobReader(
      std::move(file), file_bytes, bs, ChooseMode(file_bytes >> 20, map)));
}

// Split into chunks for load-balancing even if blob sizes vary.
static void EnqueueChunks(size_t key_idx, uint64_t offset, uint64_t bytes,
                          uint8_t* data, std::vector<BlobIO2>& writes) {
  constexpr size_t kChunkBytes = 4 * 1024 * 1024;
  const uint64_t end = offset + bytes;
  // Split into whole chunks and possibly one remainder.
  if (end >= kChunkBytes) {
    for (; offset <= end - kChunkBytes;
         offset += kChunkBytes, data += kChunkBytes) {
      writes.emplace_back(
          BlobRange{.offset = offset, .bytes = kChunkBytes, .key_idx = key_idx},
          data);
    }
  }
  if (offset != end) {
    writes.emplace_back(
        BlobRange{.offset = offset, .bytes = end - offset, .key_idx = key_idx},
        data);
  }
}

static void EnqueueWritesForBlobs(const BlobStore& bs,
                                  const hwy::Span<const uint8_t> blobs[],
                                  std::vector<uint8_t>& zeros,
                                  std::vector<BlobIO2>& writes) {
  // All-zero buffer used to write padding to the file without copying the
  // input blobs.
  static constexpr uint8_t kZeros[kBlobAlign] = {0};

  uint64_t file_end = 0;  // for padding
  for (size_t key_idx = 0; key_idx < bs.NumBlobs(); ++key_idx) {
    // We know the size, but `BlobStore` tells us the offset to write each blob.
    uint64_t offset;
    size_t bytes;
    bs.GetRange(key_idx, offset, bytes);
    HWY_ASSERT(offset != 0);
    HWY_ASSERT(bytes == blobs[key_idx].size());
    const uint64_t new_file_end = offset + bytes;
    HWY_ASSERT(new_file_end >= file_end);  // blobs are ordered by offset
    file_end = new_file_end;

    EnqueueChunks(key_idx, offset, bytes,
                  const_cast<uint8_t*>(blobs[key_idx].data()), writes);
    const size_t padding = BlobStore::RoundUpToAlign(bytes) - bytes;
    if (padding != 0) {
      HWY_ASSERT(padding <= kBlobAlign);
      writes.emplace_back(
          BlobRange{
              .offset = offset + bytes, .bytes = padding, .key_idx = key_idx},
          const_cast<uint8_t*>(kZeros));
    }
  }

  const size_t padding = hwy::RoundUpTo(file_end, kEndAlign) - file_end;
  if (padding != 0) {
    // Bigger than `kZeros`, better to allocate than issue multiple I/Os. Must
    // remain alive until the last I/O is done.
    zeros.resize(padding);
    writes.emplace_back(
        BlobRange{.offset = file_end, .bytes = padding, .key_idx = 0},
        zeros.data());
  }
}

void BlobWriter::Add(const std::string& key, const void* data, size_t bytes) {
  HWY_ASSERT(data != nullptr);
  HWY_ASSERT(bytes != 0);
  keys_.push_back(KeyFromString(key.c_str()));
  blobs_.emplace_back(static_cast<const uint8_t*>(data), bytes);
}

void BlobWriter::WriteAll(hwy::ThreadPool& pool, const Path& filename) {
  const size_t num_blobs = keys_.size();
  HWY_ASSERT(num_blobs != 0);
  HWY_ASSERT(num_blobs == blobs_.size());

  std::vector<BlobIO2> writes;
  writes.reserve(16384);

  const BlobStore bs(num_blobs, keys_.data(), blobs_.data());
  bs.EnqueueWriteForHeaderAndDirectory(writes);

  std::vector<uint8_t> zeros;
  EnqueueWritesForBlobs(bs, blobs_.data(), zeros, writes);

  // Create/replace existing file.
  std::unique_ptr<File> file = OpenFileOrNull(filename, "w+");
  if (!file) HWY_ABORT("Failed to open for writing %s", filename.path.c_str());

  pool.Run(0, writes.size(),
           [this, &file, &writes](uint64_t i, size_t /*thread*/) {
             const BlobRange& range = writes[i].range;

             if (!file->Write(writes[i].data, range.bytes, range.offset)) {
               const std::string& key = StringFromKey(keys_[range.key_idx]);
               HWY_ABORT("Write failed for %s from %zu, %zu bytes to %p.",
                         key.c_str(), static_cast<size_t>(range.offset),
                         range.bytes, writes[i].data);
             }
           });
}

}  // namespace gcpp
