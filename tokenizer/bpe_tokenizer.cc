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

#include "tokenizer/bpe_tokenizer.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <utility>

#include "io/fields.h"    // IFields
#include "tokenizer/tokenizer.h"
#include "util/basics.h"  // HWY_ABORT
#include "hwy/profiler.h"
namespace gcpp {

namespace {

// The whitespace replacement character "▁" (U+2581, "lower one eighth block").
constexpr const char* kSpaceRepl = "\xe2\x96\x81";

// Bit flags stored in `BpeTokenizerBlob::flags`. Currently informational
// (the corresponding behavior is fixed); reserved so readers can branch later.
constexpr uint32_t kFlagByteFallback = 1u << 0;
constexpr uint32_t kFlagSpaceReplace = 1u << 1;


class BpeTokenizer : public Tokenizer {
 public:
  BpeTokenizer() = default;

  std::string Serialize() const override;

  bool Deserialize(std::string_view data) override;

  bool Encode(std::string_view input, std::vector<int>& ids) const override;

  bool Decode(hwy::Span<const int> ids,
              std::string& detokenized) const override;

 private:
  // Result of a single vocabulary merge: where to merge and what it becomes.
  struct MergeRule {
    int rank;
    int new_id;
  };

  void LoadByteTokens();
  const std::string& IdToToken(int id) const;
  int IdByteValue(int id) const;
  int MatchAddedToken(std::string_view input, size_t i, size_t* len) const;
  void SplitOnAddedTokens(std::string_view input,
                          std::vector<int>* ids) const;
  std::string Normalize(std::string_view text) const;
  void EncodeSpan(std::string_view text, std::vector<int>* ids) const;
  void MergeSymbols(std::vector<int>* sym_id) const;

  std::unordered_map<std::string, int> vocab_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<uint64_t, MergeRule> merge_ranks_;
  std::vector<int> byte_id_;                    // byte value -> token id
  std::unordered_map<int, int> id_byte_value_;  // token id -> byte value
  std::unordered_map<std::string, int> added_tokens_;
  std::unordered_set<unsigned char> added_first_bytes_;
  std::vector<size_t> added_lengths_;  // distinct, descending
  int unk_id_ = 0;
};


// Versioned metadata serialized with `io/fields.h`.
struct BpeTokenizerBlob : public IFields {
  const char* Name() const override { return "BpeTokenizerBlob"; }

  void VisitFields(IFieldsVisitor& visitor) override {
    visitor(flags);
    visitor(unk_id);
    visitor(vocab);
    visitor(merges);
    visitor(added_ids);
  }

  uint32_t flags = 0;
  uint32_t unk_id = 0;
  std::vector<std::string> vocab;
  std::vector<uint32_t> merges;  // left, right flattened
  std::vector<uint32_t> added_ids;
};

// Number of bytes in the UTF-8 sequence starting with lead byte `c`.
size_t Utf8Len(unsigned char c) {
  if (c < 0x80) return 1;
  if ((c >> 5) == 0x06) return 2;
  if ((c >> 4) == 0x0e) return 3;
  if ((c >> 3) == 0x1e) return 4;
  return 1;  // invalid lead byte: treat as a single byte
}

uint64_t PairKey(int left, int right) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(left)) << 32) |
         static_cast<uint32_t>(right);
}

// A candidate merge popped from the priority queue: lower `rank` merges first,
// ties broken by leftmost position, matching HuggingFace's BPE merge order.
struct QueuedMerge {
  int rank;
  int left;  // symbol index of the pair's left element
  bool operator>(const QueuedMerge& other) const {
    if (rank != other.rank) return rank > other.rank;
    return left > other.left;
  }
};

// Decodes a run of byte-fallback bytes as UTF-8, appending to `out` (invalid
// sequences become U+FFFD, matching HuggingFace ByteFallback).
void FlushBytes(std::string* bytes, std::string* out) {
  if (bytes->empty()) return;
  size_t i = 0;
  const std::string& b = *bytes;
  while (i < b.size()) {
    const size_t clen = Utf8Len(static_cast<unsigned char>(b[i]));
    if (i + clen <= b.size()) {
      out->append(b, i, clen);
      i += clen;
    } else {
      out->append("\xef\xbf\xbd");  // U+FFFD
      ++i;
    }
  }
  bytes->clear();
}

// Appends `token` to `out`, replacing U+2581 with a space (HF Replace decoder).
void AppendReplaced(const std::string& token, std::string* out) {
  size_t pos = 0;
  while (pos < token.size()) {
    if (token.compare(pos, 3, kSpaceRepl) == 0) {
      out->push_back(' ');
      pos += 3;
    } else {
      out->push_back(token[pos]);
      ++pos;
    }
  }
}


// Emits the tokenizer in compact IFields binary format.
// Serialization is delegated to IFieldsVisitor which walks BpeTokenizerBlob.
//
// The logical layout (packed into u32 words) is:
//   [ flags: u32 ]
//   [ unk_id: u32 ]
//   [ vocab: vector<string> ]
//   [ merges: vector<u32> ] (flattened left_id, right_id order)
//   [ added_ids: vector<u32> ]
//
// Native-endian. Strings pad up to u32 boundaries under IFields standards.
std::string BpeTokenizer::Serialize() const {
  // Bulk: merges as (left_id, right_id) in ascending rank order.
  std::vector<std::pair<int, std::pair<int, int>>> by_rank;
  by_rank.reserve(merge_ranks_.size());
  for (const auto& [key, rule] : merge_ranks_) {
    by_rank.push_back(
        {rule.rank,
         {static_cast<int>(key >> 32), static_cast<int>(key & 0xffffffffu)}});
  }
  std::sort(by_rank.begin(), by_rank.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  // Bulk: added-token ids.
  std::vector<int> added_ids;
  added_ids.reserve(added_tokens_.size());
  for (const auto& [content, id] : added_tokens_) added_ids.push_back(id);
  std::sort(added_ids.begin(), added_ids.end());

  BpeTokenizerBlob blob;
  blob.unk_id = static_cast<uint32_t>(unk_id_);
  blob.vocab = id_to_token_;
  blob.flags = kFlagSpaceReplace;
  for (int b : byte_id_) {
    if (b >= 0) {
      blob.flags |= kFlagByteFallback;
      break;
    }
  }

  blob.merges.reserve(by_rank.size() * 2);
  for (const auto& e : by_rank) {
    blob.merges.push_back(static_cast<uint32_t>(e.second.first));
    blob.merges.push_back(static_cast<uint32_t>(e.second.second));
  }

  blob.added_ids.reserve(added_ids.size());
  for (int id : added_ids) {
    blob.added_ids.push_back(static_cast<uint32_t>(id));
  }

  std::vector<uint32_t> words = blob.Write();
  HWY_ASSERT(!words.empty());

  return std::string(reinterpret_cast<const char*>(words.data()),
                     words.size() * sizeof(uint32_t));
}

bool BpeTokenizer::Encode(std::string_view input,
                          std::vector<int>& ids) const {
  ids.clear();
  SplitOnAddedTokens(input, &ids);
  return true;
}

bool BpeTokenizer::Decode(hwy::Span<const int> ids,
                          std::string& detokenized) const {
  detokenized.clear();
  std::string pending_bytes;  // accumulates consecutive byte-fallback tokens
  for (int id : ids) {
    const int byte_value = IdByteValue(id);
    if (byte_value >= 0) {
      pending_bytes.push_back(static_cast<char>(byte_value));
      continue;
    }
    FlushBytes(&pending_bytes, &detokenized);
    AppendReplaced(IdToToken(id), &detokenized);
  }
  FlushBytes(&pending_bytes, &detokenized);
  return true;
}


// Records the id and byte value of each "<0xHH>" byte-fallback token.
void BpeTokenizer::LoadByteTokens() {
  byte_id_.assign(256, -1);
  for (int b = 0; b < 256; ++b) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
    const auto it = vocab_.find(buf);
    if (it != vocab_.end()) {
      byte_id_[b] = it->second;
      id_byte_value_[it->second] = b;
    }
  }
}

// Inverse of Serialize. Reads the IFields blob (io/fields.h) then the
// raw byte region it describes. Aborts on malformed/truncated input.
bool BpeTokenizer::Deserialize(std::string_view data) {
  if (data.size() % sizeof(uint32_t) != 0) {
    HWY_WARN("Compact BPE blob size %zu not a multiple of 4.", data.size());
    return false;
  }
  std::vector<uint32_t> words(data.size() / sizeof(uint32_t));
  if (!words.empty()) {
    std::memcpy(words.data(), data.data(), data.size());
  }

  BpeTokenizerBlob blob;
  const IFields::ReadResult res =
      blob.Read(SerializedSpan(words.data(), words.size()), 0);
  if (res.pos == 0) {
    HWY_WARN("Compact BPE blob is invalid.");
    return false;
  }

  if (res.extra_u32) {
    HWY_WARN(
        "Compact BPE blob has %u extra fields; consider updating gemma.cpp.",
        res.extra_u32);
  }
  unk_id_ = static_cast<int>(blob.unk_id);

  id_to_token_ = std::move(blob.vocab);
  const uint32_t vocab_count = static_cast<uint32_t>(id_to_token_.size());

  vocab_.clear();
  vocab_.reserve(static_cast<size_t>(vocab_count) * 2);
  for (uint32_t id = 0; id < vocab_count; ++id) {
    const std::string& tok = id_to_token_[id];
    if (!tok.empty()) vocab_[tok] = static_cast<int>(id);
  }

  const uint32_t merges_count = static_cast<uint32_t>(blob.merges.size() / 2);
  merge_ranks_.clear();
  merge_ranks_.reserve(static_cast<size_t>(merges_count) * 2);
  for (uint32_t rank = 0; rank < merges_count; ++rank) {
    const int left = static_cast<int>(blob.merges[2 * rank]);
    const int right = static_cast<int>(blob.merges[2 * rank + 1]);
    const auto mi = vocab_.find(IdToToken(left) + IdToToken(right));
    if (mi == vocab_.end()) continue;
    merge_ranks_[PairKey(left, right)] =
        MergeRule{static_cast<int>(rank), mi->second};
  }

  const uint32_t added_count = static_cast<uint32_t>(blob.added_ids.size());
  added_tokens_.clear();
  added_first_bytes_.clear();
  std::unordered_set<size_t> lengths;
  for (uint32_t i = 0; i < added_count; ++i) {
    const std::string& content =
        IdToToken(static_cast<int>(blob.added_ids[i]));
    if (content.empty()) continue;
    added_tokens_[content] = static_cast<int>(blob.added_ids[i]);
    added_first_bytes_.insert(static_cast<unsigned char>(content[0]));
    lengths.insert(content.size());
  }
  added_lengths_.assign(lengths.begin(), lengths.end());
  std::sort(added_lengths_.begin(), added_lengths_.end(), std::greater<>());

  LoadByteTokens();
  return true;
}

const std::string& BpeTokenizer::IdToToken(int id) const {
  static const std::string kEmpty;
  if (id < 0 || id >= static_cast<int>(id_to_token_.size())) return kEmpty;
  return id_to_token_[id];
}

int BpeTokenizer::IdByteValue(int id) const {
  const auto it = id_byte_value_.find(id);
  return it == id_byte_value_.end() ? -1 : it->second;
}

// Returns the matched added-token id (and advances `i`) if one starts at
// `input[i]`, otherwise -1.
int BpeTokenizer::MatchAddedToken(std::string_view input, size_t i,
                                  size_t* len) const {
  if (added_first_bytes_.find(static_cast<unsigned char>(input[i])) ==
      added_first_bytes_.end()) {
    return -1;
  }
  for (size_t l : added_lengths_) {
    if (i + l > input.size()) continue;
    const auto it = added_tokens_.find(std::string(input.substr(i, l)));
    if (it != added_tokens_.end()) {
      *len = l;
      return it->second;
    }
  }
  return -1;
}

// Carves `input` into added-token ids and normal-text spans, BPE-encoding the
// latter. Appends all resulting ids to `ids`.
void BpeTokenizer::SplitOnAddedTokens(std::string_view input,
                                      std::vector<int>* ids) const {
  std::string buffer;
  size_t i = 0;
  while (i < input.size()) {
    size_t len = 0;
    const int added = MatchAddedToken(input, i, &len);
    if (added >= 0) {
      EncodeSpan(buffer, ids);
      buffer.clear();
      ids->push_back(added);
      i += len;
    } else {
      buffer.push_back(input[i]);
      ++i;
    }
  }
  EncodeSpan(buffer, ids);
}

// Replaces ' ' with U+2581.
std::string BpeTokenizer::Normalize(std::string_view text) const {
  std::string out;
  out.reserve(text.size());
  for (char c : text) {
    if (c == ' ') {
      out += kSpaceRepl;
    } else {
      out.push_back(c);
    }
  }
  return out;
}

// Normalizes and BPE-encodes a normal-text span, appending ids.
void BpeTokenizer::EncodeSpan(std::string_view text,
                              std::vector<int>* ids) const {
  if (text.empty()) return;
  const std::string norm = Normalize(text);

  // Initial symbols: one per whole-character vocab entry, else byte-fallback.
  std::vector<int> sym_id;
  sym_id.reserve(norm.size());
  for (size_t i = 0; i < norm.size();) {
    const size_t clen = Utf8Len(static_cast<unsigned char>(norm[i]));
    const std::string ch = norm.substr(i, clen);
    const auto it = vocab_.find(std::string(ch));
    if (it != vocab_.end()) {
      sym_id.push_back(it->second);
    } else {
      for (size_t b = 0; b < clen && i + b < norm.size(); ++b) {
        const int bid = byte_id_[static_cast<unsigned char>(norm[i + b])];
        sym_id.push_back(bid >= 0 ? bid : unk_id_);
      }
    }
    i += clen;
  }

  MergeSymbols(&sym_id);
  ids->insert(ids->end(), sym_id.begin(), sym_id.end());
}

// In-place rank-priority BPE merge over a flat symbol list, using a doubly
// linked list with a lazily-cleaned min-heap of candidate merges.
void BpeTokenizer::MergeSymbols(std::vector<int>* sym_id) const {
  const int n = static_cast<int>(sym_id->size());
  if (n < 2) return;
  std::vector<int>& id = *sym_id;
  std::vector<int> prev(n), next(n);
  std::vector<char> dead(n, 0);
  for (int i = 0; i < n; ++i) {
    prev[i] = i - 1;
    next[i] = (i + 1 < n) ? i + 1 : -1;
  }

  std::priority_queue<QueuedMerge, std::vector<QueuedMerge>, std::greater<>>
      queue;
  auto push_pair = [&](int left) {
    if (left < 0 || next[left] < 0) return;
    const auto it = merge_ranks_.find(PairKey(id[left], id[next[left]]));
    if (it != merge_ranks_.end()) queue.push({it->second.rank, left});
  };
  for (int i = 0; i + 1 < n; ++i) push_pair(i);

  while (!queue.empty()) {
    const QueuedMerge top = queue.top();
    queue.pop();
    const int left = top.left;
    if (dead[left]) continue;
    const int right = next[left];
    if (right < 0 || dead[right]) continue;
    const auto it = merge_ranks_.find(PairKey(id[left], id[right]));
    if (it == merge_ranks_.end() || it->second.rank != top.rank) continue;

    id[left] = it->second.new_id;
    dead[right] = 1;
    next[left] = next[right];
    if (next[left] >= 0) prev[next[left]] = left;
    push_pair(prev[left]);
    push_pair(left);
  }

  std::vector<int> out;
  out.reserve(n);
  int first = 0;
  while (first >= 0 && dead[first]) first = next[first];
  for (int i = first; i >= 0; i = next[i]) out.push_back(id[i]);
  *sym_id = std::move(out);
}

}  // namespace

std::unique_ptr<Tokenizer> CreateBpeTokenizer(const std::string& blob) {
  PROFILER_ZONE("Startup.tokenizer");
  auto bpe = std::make_unique<BpeTokenizer>();
  if (!bpe->Deserialize(blob)) {
    HWY_ABORT("Failed to load BpeTokenizer from compact blob.");
  }
  return bpe;
}

}  // namespace gcpp
