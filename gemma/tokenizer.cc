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

#include "gemma/tokenizer.h"

#include <stdio.h>

#include <memory>
#include <string>
#include <vector>

#include "gemma/configs.h"  // PromptWrapping
#include "hwy/base.h"         // HWY_ASSERT
#include "hwy/profiler.h"
// copybara:import_next_line:sentencepiece
#include "src/sentencepiece_processor.h"

namespace gcpp {

// Set this to true to debug tokenizer tokens.
constexpr bool kShowTokenization = false;

class GemmaTokenizer::Impl {
 public:
  Impl() = default;
  // Loads the tokenizer from a serialized proto.
  explicit Impl(const std::string& tokenizer_proto) {
    if (tokenizer_proto == kMockTokenizer) return;
    PROFILER_ZONE("Startup.tokenizer");
    spp_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
    if (!spp_->LoadFromSerializedProto(tokenizer_proto).ok()) {
      HWY_ABORT("Failed to load tokenizer from %zu byte serialized proto.",
                tokenizer_proto.size());
    }
  }

  std::string Serialize() const {
    return spp_ ? spp_->serialized_model_proto() : kMockTokenizer;
  }

  bool Encode(const std::string& input,
              std::vector<std::string>* pieces) const {
    return spp_ && spp_->Encode(input, pieces).ok();
  }

  bool Encode(const std::string& input, std::vector<int>* ids) const {
    if constexpr (kShowTokenization) {
      bool is_ok = spp_ && spp_->Encode(input, ids).ok();
      for (int i = 0; i < static_cast<int>(ids->size()); i++) {
        fprintf(stderr, "%3d: %d\n", i, (*ids)[i]);
      }
      return is_ok;
    } else {
      return spp_ && spp_->Encode(input, ids).ok();
    }
  }

  // Given a sequence of ids, decodes it into a detokenized output.
  bool Decode(const std::vector<int>& ids, std::string* detokenized) const {
    return spp_ && spp_->Decode(ids, detokenized).ok();
  }

 private:
  std::unique_ptr<sentencepiece::SentencePieceProcessor> spp_;
};

GemmaTokenizer::GemmaTokenizer(const std::string& tokenizer_proto)
    : impl_(std::make_unique<Impl>(tokenizer_proto)) {
  HWY_ASSERT(impl_);
}

// Default suffices, but they must be defined after GemmaTokenizer::Impl.
GemmaTokenizer::~GemmaTokenizer() = default;
GemmaTokenizer::GemmaTokenizer(GemmaTokenizer&& other) = default;
GemmaTokenizer& GemmaTokenizer::operator=(GemmaTokenizer&& other) = default;

std::string GemmaTokenizer::Serialize() const { return impl_->Serialize(); }

bool GemmaTokenizer::Encode(const std::string& input,
                            std::vector<std::string>* pieces) const {
  return impl_->Encode(input, pieces);
}

bool GemmaTokenizer::Encode(const std::string& input,
                            std::vector<int>* ids) const {
  return impl_->Encode(input, ids);
}

// Given a sequence of ids, decodes it into a detokenized output.
bool GemmaTokenizer::Decode(const std::vector<int>& ids,
                            std::string* detokenized) const {
  return impl_->Decode(ids, detokenized);
}

// Negligible CPU time in the ctor body.
GemmaChatTemplate::GemmaChatTemplate(const GemmaTokenizer& tokenizer,
                                     Model model) {
  sot_user_.reserve(3);
  if (!tokenizer.Encode("<start_of_turn>user\n", &sot_user_)) return;
  sot_model_.reserve(3);
  HWY_ASSERT(tokenizer.Encode("<start_of_turn>model\n", &sot_model_));
  eot_.reserve(2);
  HWY_ASSERT(tokenizer.Encode("<end_of_turn>\n", &eot_));

  HWY_ASSERT(tokenizer.Encode("\n", &pali_sep_));
  vlm_soi_.reserve(2);
  HWY_ASSERT(tokenizer.Encode("\n\n<start_of_image>", &vlm_soi_));
  vlm_eoi_.reserve(2);
  HWY_ASSERT(tokenizer.Encode("<end_of_image>\n\n", &vlm_eoi_));
}

std::vector<int> GemmaChatTemplate::Apply(size_t pos,
                                          const std::vector<int>& ids) const {
  HWY_ASSERT_M(!sot_user_.empty() && !sot_model_.empty() && !eot_.empty(),
               "GemmaChatTemplate has not been initialized.");
  std::vector<int> out;
  out.reserve(eot_.size() + sot_user_.size() + ids.size() + eot_.size() +
              sot_model_.size());

  // Start with BOS, or prepend end_of_turn if this is a continuation.
  if (pos == 0) {
    out.push_back(BOS_ID);
  } else {
    out.insert(out.cend(), eot_.cbegin(), eot_.cend());
  }
  // Start of user turn, user prompt, end of turn; then start of model turn.
  out.insert(out.cend(), sot_user_.cbegin(), sot_user_.cend());
  out.insert(out.cend(), ids.cbegin(), ids.cend());
  out.insert(out.cend(), eot_.cbegin(), eot_.cend());
  out.insert(out.cend(), sot_model_.cbegin(), sot_model_.cend());
  return out;
}

std::vector<int> GemmaChatTemplate::WrapPali(const std::vector<int>& text_part,
                                             size_t image_batch_size) const {
  HWY_ASSERT_M(!pali_sep_.empty(),
               "GemmaChatTemplate has not been initialized.");
  std::vector<int> out;
  out.reserve(image_batch_size + 1 + text_part.size() + pali_sep_.size());
  out.resize(image_batch_size, 0);
  out.push_back(BOS_ID);
  out.insert(out.cend(), text_part.cbegin(), text_part.cend());
  out.insert(out.cend(), pali_sep_.cbegin(), pali_sep_.cend());
  return out;
}

std::vector<int> GemmaChatTemplate::WrapVLM(const std::vector<int>& text_part,
                                            size_t image_batch_size) const {
  HWY_ASSERT_M(!vlm_soi_.empty() && !vlm_eoi_.empty(),
               "GemmaChatTemplate has not been initialized.");
  std::vector<int> out;
  out.reserve(text_part.size() + vlm_soi_.size() + image_batch_size +
              vlm_eoi_.size());
  out.insert(out.cend(), text_part.cbegin(), text_part.cend());
  out.insert(out.cend(), vlm_soi_.cbegin(), vlm_soi_.cend());
  out.insert(out.cend(), image_batch_size, -2);
  out.insert(out.cend(), vlm_eoi_.cbegin(), vlm_eoi_.cend());
  return out;
}

// Text
std::vector<int> WrapAndTokenize(const GemmaTokenizer& tokenizer,
                                 const GemmaChatTemplate& chat_template,
                                 const PromptWrapping wrapping, size_t pos,
                                 const std::string& prompt) {
  std::vector<int> tokens;
  HWY_ASSERT(tokenizer.Encode(prompt, &tokens));

  switch (wrapping) {
    case PromptWrapping::GEMMA_IT:
    case PromptWrapping::GEMMA_VLM:
      return chat_template.Apply(pos, tokens);
    default:
      if (pos == 0) {
        tokens.insert(tokens.cbegin(), BOS_ID);
      }
      return tokens;
  }
}

// Vision
std::vector<int> WrapAndTokenize(const GemmaTokenizer& tokenizer,
                                 const GemmaChatTemplate& chat_template,
                                 const PromptWrapping wrapping, size_t pos,
                                 const std::string& prompt,
                                 size_t image_batch_size) {
  std::vector<int> text_part;
  HWY_ASSERT(tokenizer.Encode(prompt, &text_part));
  switch (wrapping) {
    case PromptWrapping::PALIGEMMA:
      HWY_ASSERT(pos == 0);
      return chat_template.WrapPali(text_part, image_batch_size);
    case PromptWrapping::GEMMA_VLM:
      return chat_template.Apply(
          pos, chat_template.WrapVLM(text_part, image_batch_size));
    default:
      HWY_ASSERT_M(false, "Current variant does not support vision prompt.");
  }
}

}  // namespace gcpp
