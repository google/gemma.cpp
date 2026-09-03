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

#ifndef THIRD_PARTY_GEMMA_CPP_DEEPSEEK_DSV4_TOKENIZER_H_
#define THIRD_PARTY_GEMMA_CPP_DEEPSEEK_DSV4_TOKENIZER_H_

// HuggingFace `tokenizer.json` byte-level BPE tokenizer for DeepSeek V4.
// Implements the exact subset that model uses: no normalizer, three
// Split pre-tokenizers (digit triplets, CJK runs, GPT-2-style main pattern)
// followed by ByteLevel, added-token extraction, and a ByteLevel decoder.
// DeepSeek does not use sentencepiece, so GemmaTokenizer does not apply.

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gcpp {

class Dsv4Tokenizer {
 public:
  // Aborts on I/O or parse errors.
  explicit Dsv4Tokenizer(const std::string& tokenizer_json_path);

  // Initializes tokenizer directly from JSON content in memory.
  // 'is_content' is used to distinguish from path constructor.
  Dsv4Tokenizer(std::string_view json_content, bool is_content);

  // Extracts added tokens (chat markers etc.), pre-tokenizes and BPE-encodes
  // everything in between. Equivalent to HF encode(add_special_tokens=false).
  std::vector<int> Encode(const std::string& text) const;

  // Wraps a single user message in the DeepSeek V4 chat template:
  // <BOS><|User|>msg<|Assistant|> followed by </think> ("chat" mode) or
  // <think> (thinking mode).
  std::string WrapChat(const std::string& user_msg, bool thinking) const;

  // Appends the decoded bytes of `id` to `out`. Tokens marked special in
  // tokenizer.json (BOS/EOS/pads) are skipped, matching HF's
  // skip_special_tokens=true; other added tokens (e.g. </think>) are kept.
  void AppendDecoded(int id, std::string& out) const;

  // Decodes a sequence of token IDs, skipping special tokens.
  std::string Decode(const std::vector<int>& ids) const;

  size_t VocabSize() const { return id_to_bytes_.size(); }

 private:
  struct AddedToken {
    std::string content;
    int id;
  };

  void Init(std::string_view json_content);

  // Splits `text` (a span with no added tokens) into pre-tokenization pieces
  // and BPE-encodes each, appending ids.
  void EncodeSegment(const char* bytes, size_t len,
                     std::vector<int>& ids) const;
  void BpePiece(const char* bytes, size_t len, std::vector<int>& ids) const;

  std::unordered_map<std::string, int> vocab_;       // byte-level strings
  std::unordered_map<std::string, int> merge_rank_;  // "left right" -> rank
  std::vector<std::string> id_to_bytes_;             // raw bytes per id
  std::vector<bool> is_special_;
  // Added tokens bucketed by first byte, longest content first.
  std::unordered_map<uint8_t, std::vector<AddedToken>> added_by_first_byte_;
  std::string byte_symbol_[256];  // UTF-8 of the byte-level codepoint per byte
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_DEEPSEEK_DSV4_TOKENIZER_H_
