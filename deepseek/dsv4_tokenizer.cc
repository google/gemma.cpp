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

#include "deepseek/dsv4_tokenizer.h"

#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "hwy/base.h"  // HWY_ABORT
#include "nlohmann/json.hpp"

namespace gcpp {
namespace {

using json = nlohmann::json;

struct URange {
  uint32_t lo, hi;
};
#include "deepseek/dsv4_unicode_ranges.inc"

template <size_t N>
bool InRanges(const URange (&ranges)[N], uint32_t cp) {
  size_t lo = 0, hi = N;
  while (lo < hi) {
    const size_t mid = (lo + hi) / 2;
    if (cp < ranges[mid].lo) {
      hi = mid;
    } else if (cp > ranges[mid].hi) {
      lo = mid + 1;
    } else {
      return true;
    }
  }
  return false;
}

bool IsLetter(uint32_t cp) { return InRanges(kLetterRanges, cp); }
bool IsMark(uint32_t cp) { return InRanges(kMarkRanges, cp); }
bool IsNumber(uint32_t cp) { return InRanges(kNumberRanges, cp); }
bool IsPunct(uint32_t cp) { return InRanges(kPunctRanges, cp); }
bool IsSymbol(uint32_t cp) { return InRanges(kSymbolRanges, cp); }

// Unicode White_Space property (what the regex \s matches).
bool IsWhite(uint32_t cp) {
  switch (cp) {
    case 0x09:
    case 0x0A:
    case 0x0B:
    case 0x0C:
    case 0x0D:
    case 0x20:
    case 0x85:
    case 0xA0:
    case 0x1680:
    case 0x2028:
    case 0x2029:
    case 0x202F:
    case 0x205F:
    case 0x3000:
      return true;
    default:
      return cp >= 0x2000 && cp <= 0x200A;
  }
}

bool IsNewline(uint32_t cp) { return cp == '\r' || cp == '\n'; }

// The literal ASCII punctuation class of the main pattern's first branch.
bool IsAsciiPunct(uint32_t cp) {
  return (cp >= 0x21 && cp <= 0x2F) || (cp >= 0x3A && cp <= 0x40) ||
         (cp >= 0x5B && cp <= 0x60) || (cp >= 0x7B && cp <= 0x7E);
}

bool IsAsciiAlpha(uint32_t cp) {
  return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
}

// CJK ranges isolated by the second Split pre-tokenizer: unified ideographs
// U+4E00..9FA5, hiragana U+3040..309F, katakana U+30A0..30FF.
bool IsCjk(uint32_t cp) {
  return (cp >= 0x4E00 && cp <= 0x9FA5) || (cp >= 0x3040 && cp <= 0x309F) ||
         (cp >= 0x30A0 && cp <= 0x30FF);
}

void AppendUtf8(uint32_t cp, std::string& out) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// Decodes one UTF-8 codepoint; invalid bytes decode as themselves so that
// arbitrary input cannot crash (byte-level BPE handles any byte anyway).
uint32_t DecodeUtf8(const char* s, size_t len, size_t& i) {
  const uint8_t b0 = static_cast<uint8_t>(s[i]);
  size_t need = 0;
  uint32_t cp = b0;
  if (b0 < 0x80) {
    need = 0;
  } else if ((b0 & 0xE0) == 0xC0) {
    need = 1;
    cp = b0 & 0x1F;
  } else if ((b0 & 0xF0) == 0xE0) {
    need = 2;
    cp = b0 & 0x0F;
  } else if ((b0 & 0xF8) == 0xF0) {
    need = 3;
    cp = b0 & 0x07;
  }
  for (size_t k = 1; k <= need; ++k) {
    if (i + k >= len || (static_cast<uint8_t>(s[i + k]) & 0xC0) != 0x80) {
      ++i;
      return b0;  // invalid sequence: treat lead byte as Latin-1
    }
    cp = (cp << 6) | (static_cast<uint8_t>(s[i + k]) & 0x3F);
  }
  i += need + 1;
  return cp;
}

// GPT-2 byte-level map: printable bytes map to themselves, others to
// 256 + running index, in increasing byte order.
void BuildByteToCp(uint32_t byte_to_cp[256]) {
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    const bool direct = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) ||
                        (b >= 0xAE && b <= 0xFF);
    byte_to_cp[b] =
        direct ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + n++);
  }
}

std::string ReadFileToStringOrAbort(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) HWY_ABORT("Failed to open tokenizer file %s", path.c_str());
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

}  // namespace

Dsv4Tokenizer::Dsv4Tokenizer(const std::string& tokenizer_json_path) {
  const std::string contents = ReadFileToStringOrAbort(tokenizer_json_path);
  json j = json::parse(contents, /*cb=*/nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded()) {
    HWY_ABORT("Failed to parse tokenizer JSON %s", tokenizer_json_path.c_str());
  }

  uint32_t byte_to_cp[256];
  BuildByteToCp(byte_to_cp);
  std::unordered_map<uint32_t, uint8_t> cp_to_byte;
  for (int b = 0; b < 256; ++b) {
    byte_symbol_[b].clear();
    AppendUtf8(byte_to_cp[b], byte_symbol_[b]);
    cp_to_byte[byte_to_cp[b]] = static_cast<uint8_t>(b);
  }

  const json& model = j.at("model");
  const json& vocab = model.at("vocab");
  int max_id = 0;
  vocab_.reserve(vocab.size());
  for (auto it = vocab.begin(); it != vocab.end(); ++it) {
    const int id = it.value().get<int>();
    vocab_.emplace(it.key(), id);
    max_id = std::max(max_id, id);
  }
  for (const json& a : j.at("added_tokens")) {
    max_id = std::max(max_id, a.at("id").get<int>());
  }

  id_to_bytes_.resize(static_cast<size_t>(max_id) + 1);
  is_special_.assign(static_cast<size_t>(max_id) + 1, false);
  // Un-map byte-level vocab strings to raw bytes for decoding.
  for (const auto& kv : vocab_) {
    std::string bytes;
    bytes.reserve(kv.first.size());
    size_t i = 0;
    while (i < kv.first.size()) {
      const uint32_t cp = DecodeUtf8(kv.first.data(), kv.first.size(), i);
      const auto it = cp_to_byte.find(cp);
      if (it == cp_to_byte.end()) {
        bytes.clear();  // not a byte-level string; keep raw below
        break;
      }
      bytes.push_back(static_cast<char>(it->second));
    }
    id_to_bytes_[kv.second] = bytes.empty() ? kv.first : bytes;
  }

  const json& merges = model.at("merges");
  merge_rank_.reserve(merges.size());
  int rank = 0;
  for (const json& m : merges) {
    if (m.is_string()) {
      merge_rank_.emplace(m.get<std::string>(), rank++);
    } else {  // newer serialization: ["left", "right"]
      merge_rank_.emplace(
          m.at(0).get<std::string>() + " " + m.at(1).get<std::string>(),
          rank++);
    }
  }

  for (const json& a : j.at("added_tokens")) {
    AddedToken tok;
    tok.content = a.at("content").get<std::string>();
    tok.id = a.at("id").get<int>();
    id_to_bytes_[tok.id] = tok.content;
    is_special_[tok.id] = a.at("special").get<bool>();
    if (!tok.content.empty()) {
      added_by_first_byte_[static_cast<uint8_t>(tok.content[0])].push_back(
          std::move(tok));
    }
  }
  for (auto& kv : added_by_first_byte_) {
    std::sort(kv.second.begin(), kv.second.end(),
              [](const AddedToken& a, const AddedToken& b) {
                return a.content.size() > b.content.size();
              });
  }
}

std::string Dsv4Tokenizer::WrapChat(const std::string& user_msg,
                                    bool thinking) const {
  // U+FF5C fullwidth bar and U+2581 lower block, as used by DeepSeek markers.
  static const char kBos[] =
      "<\xEF\xBD\x9C"
      "begin\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>";
  static const char kUser[] =
      "<\xEF\xBD\x9C"
      "User\xEF\xBD\x9C>";
  static const char kAssistant[] =
      "<\xEF\xBD\x9C"
      "Assistant\xEF\xBD\x9C>";
  std::string out(kBos);
  out += kUser;
  out += user_msg;
  out += kAssistant;
  out += thinking ? "<think>" : "</think>";
  return out;
}

std::vector<int> Dsv4Tokenizer::Encode(const std::string& text) const {
  std::vector<int> ids;
  size_t segment_start = 0;
  size_t i = 0;
  while (i < text.size()) {
    const auto bucket =
        added_by_first_byte_.find(static_cast<uint8_t>(text[i]));
    const AddedToken* hit = nullptr;
    if (bucket != added_by_first_byte_.end()) {
      for (const AddedToken& tok : bucket->second) {
        if (tok.content.size() <= text.size() - i &&
            text.compare(i, tok.content.size(), tok.content) == 0) {
          hit = &tok;
          break;  // longest first
        }
      }
    }
    if (hit) {
      EncodeSegment(text.data() + segment_start, i - segment_start, ids);
      ids.push_back(hit->id);
      i += hit->content.size();
      segment_start = i;
    } else {
      ++i;
    }
  }
  EncodeSegment(text.data() + segment_start, text.size() - segment_start, ids);
  return ids;
}

void Dsv4Tokenizer::EncodeSegment(const char* bytes, size_t len,
                                  std::vector<int>& ids) const {
  if (len == 0) return;
  // Decode to codepoints once; pieces are [start, end) codepoint ranges that
  // map back to byte ranges via ofs.
  std::vector<uint32_t> cps;
  std::vector<size_t> ofs;  // byte offset of each codepoint, plus sentinel
  cps.reserve(len);
  ofs.reserve(len + 1);
  {
    size_t i = 0;
    while (i < len) {
      ofs.push_back(i);
      cps.push_back(DecodeUtf8(bytes, len, i));
    }
    ofs.push_back(len);
  }
  const size_t n = cps.size();

  // Emits one final piece (after all splits) into the BPE stage.
  const auto emit = [&](size_t b, size_t e) {
    if (b < e) BpePiece(bytes + ofs[b], ofs[e] - ofs[b], ids);
  };

  // Split 3 (main GPT-2-style pattern), applied to [b, e).
  const auto split_main = [&](size_t b, size_t e) {
    // Returns match length at i, or 0. Alternatives in pattern order.
    const auto match = [&](size_t i) -> size_t {
      const uint32_t c = cps[i];
      // [ascii punct][A-Za-z]+
      if (IsAsciiPunct(c) && i + 1 < e && IsAsciiAlpha(cps[i + 1])) {
        size_t m = i + 1;
        while (m < e && IsAsciiAlpha(cps[m])) ++m;
        return m - i;
      }
      // [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
      {
        const bool lead =
            !IsNewline(c) && !IsLetter(c) && !IsPunct(c) && !IsSymbol(c);
        const size_t k = i + (lead ? 1 : 0);
        if (k < e && (IsLetter(cps[k]) || IsMark(cps[k]))) {
          size_t m = k;
          while (m < e && (IsLetter(cps[m]) || IsMark(cps[m]))) ++m;
          return m - i;
        }
        // Optional lead not taken: the char itself may open [\p{L}\p{M}]+.
        if (lead && IsMark(c)) {
          size_t m = i;
          while (m < e && (IsLetter(cps[m]) || IsMark(cps[m]))) ++m;
          return m - i;
        }
      }
      // " ?[\p{P}\p{S}]+[\r\n]*"
      {
        size_t m = i;
        if (c == ' ' && m + 1 < e &&
            (IsPunct(cps[m + 1]) || IsSymbol(cps[m + 1]))) {
          ++m;
        }
        if (m < e && (IsPunct(cps[m]) || IsSymbol(cps[m]))) {
          while (m < e && (IsPunct(cps[m]) || IsSymbol(cps[m]))) ++m;
          while (m < e && IsNewline(cps[m])) ++m;
          return m - i;
        }
      }
      // Whitespace branches: \s*[\r\n]+ | \s+(?!\S) | \s+
      if (IsWhite(c)) {
        size_t j = i;
        size_t last_nl = 0;
        bool has_nl = false;
        while (j < e && IsWhite(cps[j])) {
          if (IsNewline(cps[j])) {
            last_nl = j;
            has_nl = true;
          }
          ++j;
        }
        if (has_nl) return last_nl + 1 - i;  // \s*[\r\n]+
        if (j == e) return j - i;            // \s+(?!\S) at end
        if (j - i >= 2) return (j - 1) - i;  // \s+(?!\S), hold back one
        return 1;                            // \s+
      }
      return 0;
    };
    size_t u = b, i = b;
    while (i < e) {
      const size_t m = match(i);
      if (m > 0) {
        if (u < i) emit(u, i);
        emit(i, i + m);
        i += m;
        u = i;
      } else {
        ++i;
      }
    }
    if (u < e) emit(u, e);
  };

  // Split 2: isolate CJK runs.
  const auto split_cjk = [&](size_t b, size_t e) {
    size_t u = b, i = b;
    while (i < e) {
      if (IsCjk(cps[i])) {
        if (u < i) split_main(u, i);
        size_t m = i;
        while (m < e && IsCjk(cps[m])) ++m;
        split_main(i, m);
        i = m;
        u = i;
      } else {
        ++i;
      }
    }
    if (u < e) split_main(u, e);
  };

  // Split 1: isolate digit groups of up to three (\p{N}{1,3}).
  size_t u = 0, i = 0;
  while (i < n) {
    if (IsNumber(cps[i])) {
      if (u < i) split_cjk(u, i);
      size_t m = i;
      while (m < n && IsNumber(cps[m])) ++m;
      for (size_t g = i; g < m; g += 3) {
        split_cjk(g, std::min(g + 3, m));
      }
      i = m;
      u = i;
    } else {
      ++i;
    }
  }
  if (u < n) split_cjk(u, n);
}

void Dsv4Tokenizer::BpePiece(const char* bytes, size_t len,
                             std::vector<int>& ids) const {
  if (len == 0) return;
  // Byte-level mapping: one symbol per input byte.
  std::vector<std::string> sym;
  sym.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    sym.push_back(byte_symbol_[static_cast<uint8_t>(bytes[i])]);
  }
  // Standard rank-greedy BPE; pieces are short, so O(n^2) lookups are fine.
  std::string key;
  while (sym.size() >= 2) {
    int best_rank = std::numeric_limits<int>::max();
    size_t best_i = 0;
    for (size_t i = 0; i + 1 < sym.size(); ++i) {
      key.assign(sym[i]);
      key.push_back(' ');
      key.append(sym[i + 1]);
      const auto it = merge_rank_.find(key);
      if (it != merge_rank_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_i = i;
      }
    }
    if (best_rank == std::numeric_limits<int>::max()) break;
    sym[best_i] += sym[best_i + 1];
    sym.erase(sym.begin() + static_cast<ptrdiff_t>(best_i) + 1);
  }
  for (const std::string& s : sym) {
    const auto it = vocab_.find(s);
    if (it != vocab_.end()) {
      ids.push_back(it->second);
    } else {
      // No unk token in this vocab; skip (cannot happen for byte-level
      // symbols since all 256 single bytes are in the vocab).
      fprintf(stderr, "Dsv4Tokenizer: dropping unknown symbol\n");
    }
  }
}

void Dsv4Tokenizer::AppendDecoded(int id, std::string& out) const {
  if (id < 0 || static_cast<size_t>(id) >= id_to_bytes_.size()) return;
  if (is_special_[id]) return;
  out += id_to_bytes_[id];
}

}  // namespace gcpp
