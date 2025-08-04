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

// A write I/O request, each serviced by one thread in a pool.
struct BlobIO {
  BlobIO(BlobRange range, const void* data) : range(range), data(data) {}

  BlobRange range;
  const void* data;  // Read-only for writes.
};

// Little-endian on-disk representation:
// For V1: the file is represented as
//    Header + Directory + PadToBlobAlign + Payload + PayToEndAlign.
// For V2: the file is represented as
//   Header + PadToBlobAlign + Payload + PadToEndAlign + Directory + Header
// The Header at the beginning has num_blobs == 0; and the Header at the end has
// the correct num_blobs. This allows writing blobs without knowing the total
// number of them, nor holding all them in memory. As of 2025-07-31, we support
// reading both, but always write V2. Note that its num_blobs == 0 was
// previously disallowed. To read V2, pull the latest code from the dev branch.
//
// Actual payload is indexed by the directory with keys, offset and bytes; keys
// are unique, opaque 128-bit keys.
//
// The file format deliberately omits a version number because it is unchanging.
// Additional data may be added only inside new blobs. Changes to the blob
// contents or type should be handled by renaming keys.
//
// This class is for internal use by `BlobReader` and `BlobWriter`. Its
// interface is more low-level: fixed-size keys instead of strings.
class BlobStore {
  static constexpr uint32_t kMagic = 0x0A534253;  // SBS\n

  // Upper limit to avoid allocating a huge vector.
  static constexpr size_t kMaxBlobs = 16 * 1024;

  // Returns the size of padded header and directory, which is also the start of
  // the first payload for V1. `num_blobs` is `NumBlobs()` if the header is
  // already available, otherwise the number of blobs to be written.
  static size_t PaddedHeaderAndDirBytes(size_t num_blobs) {
    HWY_ASSERT(num_blobs < kMaxBlobs);
    // Per blob, a key and offset/size.
    return RoundUpToAlign(sizeof(Header) + 2 * kU128Bytes * num_blobs);
  }

  static uint64_t PaddedPayloadBytes(const std::vector<size_t>& blob_sizes) {
    uint64_t total_payload_bytes = 0;
    for (size_t blob_size : blob_sizes) {
      total_payload_bytes += RoundUpToAlign(blob_size);
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

  bool ParseHeaderAndDirectoryV1(const File& file) {
    is_file_v2_ = false;
    // Read header from the beginning of the file.
    if (!file.Read(0, sizeof(header_), &header_)) {
      HWY_WARN("Failed to read BlobStore header.");
      return false;
    }

    if (header_.magic != kMagic) {
      HWY_WARN("BlobStore header magic %08x does not match %08x.",
               header_.magic, kMagic);
      return false;
    }

    if (header_.num_blobs == 0) {
      // Should parse as V2.
      return false;
    }

    if (header_.num_blobs > kMaxBlobs) {
      HWY_WARN("Too many blobs, likely corrupt file.");
      return false;
    }

    directory_.resize(header_.num_blobs * 2);
    const auto directory_bytes = 2 * kU128Bytes * header_.num_blobs;
    // Read directory after the header.
    if (!file.Read(sizeof(header_), directory_bytes, directory_.data())) {
      HWY_WARN("Failed to read BlobStore directory.");
      return false;
    }
    HWY_ASSERT(IsValid(file.FileSize()));
    return true;
  }

  bool ParseHeaderAndDirectoryV2(const File& file) {
    is_file_v2_ = true;
    // Read header from the end of the file.
    size_t offset = file.FileSize() - sizeof(header_);
    if (!file.Read(offset, sizeof(header_), &header_)) {
      HWY_WARN("Failed to read BlobStore header.");
      return false;
    }

    if (header_.magic != kMagic) {
      HWY_WARN("BlobStore header magic %08x does not match %08x.",
               header_.magic, kMagic);
      return false;
    }

    if (header_.num_blobs > kMaxBlobs) {
      HWY_WARN("Too many blobs, likely corrupt file.");
      return false;
    }
    directory_.resize(header_.num_blobs * 2);
    const auto directory_bytes = 2 * kU128Bytes * header_.num_blobs;
    offset -= directory_bytes;
    // Read directory immediately before the header.
    if (!file.Read(offset, directory_bytes, directory_.data())) {
      HWY_WARN("Failed to read BlobStore directory.");
      return false;
    }
    HWY_ASSERT(IsValid(file.FileSize()));
    return true;
  }

 public:
  template <typename T>
  static T RoundUpToAlign(T size_or_offset) {
    return hwy::RoundUpTo(size_or_offset, kBlobAlign);
  }

  // Reads header/directory from file.
  explicit BlobStore(const File& file) {
    if (ParseHeaderAndDirectoryV1(file)) {
      return;
    }
    if (ParseHeaderAndDirectoryV2(file)) {
      return;
    }
    HWY_ABORT("Failed to read BlobStore header or directory.");
  }

  // Initializes header/directory for writing to disk.
  BlobStore(const std::vector<hwy::uint128_t>& keys,
            const std::vector<size_t>& blob_sizes) {
    const size_t num_blobs = keys.size();
    HWY_ASSERT(num_blobs < kMaxBlobs);  // Ensures safe to cast to u32.
    HWY_ASSERT(keys.size() == blob_sizes.size());
    EnsureUnique(hwy::Span<const hwy::uint128_t>(keys.data(), num_blobs));

    // Set header_.
    header_.magic = kMagic;
    header_.num_blobs = static_cast<uint32_t>(num_blobs);

    const size_t size_before_blobs = BytesBeforeBlobsV2().size();
    header_.file_bytes =
        hwy::RoundUpTo(size_before_blobs + PaddedPayloadBytes(blob_sizes) +
                           PaddedHeaderAndDirBytes(num_blobs),
                       kEndAlign);

    // Set first num_blobs elements of directory_ which are the keys.
    directory_.resize(num_blobs * 2);
    hwy::CopyBytes(keys.data(), directory_.data(), num_blobs * kU128Bytes);
    EnsureUnique(Keys());

    // Set the second half of directory_ which is the offsets and sizes.
    uint64_t offset = size_before_blobs;
    for (size_t i = 0; i < num_blobs; ++i) {
      SetRange(i, offset, blob_sizes[i]);
      offset = RoundUpToAlign(offset + blob_sizes[i]);
    }

    HWY_ASSERT(IsValid(FileSize()));
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
    const size_t size_before_blobs = BytesBeforeBlobs().size();
    const size_t size_after_blobs = BytesAfterBlobs().size();

    uint64_t expected_offset = size_before_blobs;
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
        expected_offset + size_after_blobs != header_.file_bytes) {
      HWY_WARN("Invalid BlobStore: end of blobs %zu but file size %zu.",
               static_cast<size_t>(expected_offset),
               static_cast<size_t>(header_.file_bytes));
      return false;
    }

    return true;  // all OK
  }

  static std::vector<uint8_t> BytesBeforeBlobsV2() {
    const Header kFakeHeaderV2 = {
        .magic = kMagic,
        .num_blobs = 0,
        .file_bytes = kEndAlign,
    };
    std::vector<uint8_t> header(PaddedHeaderAndDirBytes(0));
    hwy::CopyBytes(&kFakeHeaderV2, header.data(), sizeof(Header));
    return header;
  }

  std::vector<uint8_t> BytesBeforeBlobs() const {
    if (is_file_v2_) {
      return BytesBeforeBlobsV2();
    } else {
      const size_t padded_header_and_directory_size =
          PaddedHeaderAndDirBytes(NumBlobs());
      std::vector<uint8_t> header_and_directory(
          padded_header_and_directory_size);

      // Copy header_ at the beginning (offset 0)
      hwy::CopyBytes(&header_, header_and_directory.data(), sizeof(header_));

      // Copy directory_ immediately after the header_
      hwy::CopyBytes(directory_.data(),
                     header_and_directory.data() + sizeof(header_),
                     2 * kU128Bytes * NumBlobs());
      return header_and_directory;
    }
  }

  std::vector<uint8_t> BytesAfterBlobs() const {
    // Gets blob end.
    uint64_t offset = 0;
    size_t bytes = 0;
    GetRange(NumBlobs() - 1, offset, bytes);
    const uint64_t blob_end = RoundUpToAlign(offset + bytes);

    // For V1, just return the file paddings.
    if (!is_file_v2_) {
      return std::vector<uint8_t>(FileSize() - blob_end);
    }

    const size_t header_and_directory_with_file_padding_size =
        FileSize() - blob_end;
    std::vector<uint8_t> header_and_directory(
        header_and_directory_with_file_padding_size);

    const size_t header_size = sizeof(Header);
    const size_t directory_size = 2 * kU128Bytes * NumBlobs();

    // Copy header_ at the end.
    offset = header_and_directory_with_file_padding_size - header_size;
    hwy::CopyBytes(&header_, header_and_directory.data() + offset, header_size);

    // Copy directory_ immediately before the header_.
    offset -= directory_size;
    hwy::CopyBytes(directory_.data(), header_and_directory.data() + offset,
                   directory_size);

    return header_and_directory;
  }

  size_t FileSize() const { return header_.file_bytes; }

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

  bool is_file_v2_ = true;

  Header header_;

  std::vector<hwy::uint128_t> directory_;  // two per blob, see `SetRange`.
};  // BlobStore

BlobReader::BlobReader(const Path& blob_path) : blob_path_(blob_path) {
  PROFILER_ZONE("Startup.BlobReader");

  file_ = OpenFileOrAbort(blob_path, "r");
  file_bytes_ = file_->FileSize();
  if (file_bytes_ == 0) HWY_ABORT("Zero-sized file %s", blob_path.path.c_str());

  BlobStore bs(*file_);

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
}

// Split into chunks for load-balancing even if blob sizes vary.
static void EnqueueChunks(size_t key_idx, uint64_t offset, uint64_t bytes,
                          const uint8_t* data, std::vector<BlobIO>& writes) {
  constexpr size_t kChunkBytes = 10 * 1024 * 1024;
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

  // Write a padding if necessary.
  static constexpr uint8_t kZeros[kBlobAlign] = {0};
  const size_t padding = BlobStore::RoundUpToAlign(bytes) - bytes;
  if (padding > 0) {
    writes.emplace_back(
        BlobRange{.offset = end, .bytes = padding, .key_idx = key_idx},
        static_cast<const uint8_t*>(kZeros));
  }
}

BlobWriter::BlobWriter(const Path& filename, hwy::ThreadPool& pool)
    : file_(OpenFileOrNull(filename, "w+")), pool_(pool) {
  if (!file_) HWY_ABORT("Failed to open for writing %s", filename.path.c_str());
  // Write a placeholder header to the beginning of the file. If append-only,
  // we will later write a footer, else we will update the header.
  std::vector<uint8_t> bytes_before_blobs = BlobStore::BytesBeforeBlobsV2();
  file_->Write(bytes_before_blobs.data(), bytes_before_blobs.size(), 0);
}

void BlobWriter::Add(const std::string& key, const void* data, size_t bytes) {
  HWY_ASSERT(data != nullptr);
  HWY_ASSERT(bytes != 0);
  keys_.push_back(KeyFromString(key.c_str()));
  blob_sizes_.push_back(bytes);

  std::vector<BlobIO> writes;
  EnqueueChunks(keys_.size() - 1, file_->FileSize(), bytes,
                static_cast<const uint8_t*>(data), writes);

  hwy::ThreadPool null_pool(0);
  hwy::ThreadPool& pool_or_serial = file_->IsAppendOnly() ? null_pool : pool_;
  pool_or_serial.Run(
      0, writes.size(), [this, &writes](uint64_t i, size_t /*thread*/) {
        const BlobRange& range = writes[i].range;

        if (!file_->Write(writes[i].data, range.bytes, range.offset)) {
          const std::string& key = StringFromKey(keys_[range.key_idx]);
          HWY_ABORT("Write failed for %s from %zu, %zu bytes to %p.",
                    key.c_str(), static_cast<size_t>(range.offset), range.bytes,
                    writes[i].data);
        }
      });
}

void BlobWriter::Finalize() {
  const BlobStore bs = BlobStore(keys_, blob_sizes_);

  // Write the rest of the bytes, which contains: paddings + directory + header.
  const auto bytes_after_blobs = bs.BytesAfterBlobs();
  file_->Write(bytes_after_blobs.data(), bytes_after_blobs.size(),
               file_->FileSize());

  file_.reset();  // closes the file
}

}  // namespace gcpp
