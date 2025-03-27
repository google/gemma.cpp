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

#include "compression/io.h"      // Path
#include "compression/shared.h"  // PromptWrapping
#include "gemma/common.h"        // Wrap
#include "hwy/base.h"              // HWY_ASSERT
#include "hwy/profiler.h"
// copybara:import_next_line:sentencepiece
#include "src/sentencepiece_processor.h"

namespace gcpp {

// Set this to true to debug tokenizer tokens.
constexpr bool kShowTokenization = false;

class GemmaTokenizer::Impl {
 public:
  Impl() = default;
  explicit Impl(const Path& tokenizer_path) {
    PROFILER_ZONE("Startup.tokenizer");
    spp_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
    if (!spp_->Load(tokenizer_path.path).ok()) {
      HWY_ABORT("Failed to load the tokenizer file.");
    }
  }
  // Loads the tokenizer from a serialized proto.
  explicit Impl(const std::string& tokenizer_proto) {
    PROFILER_ZONE("Startup.tokenizer");
    spp_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
    if (!spp_->LoadFromSerializedProto(tokenizer_proto).ok()) {
      fprintf(stderr, "serialized proto size=%zu.\n", tokenizer_proto.size());
      HWY_ABORT("Failed to load the tokenizer from serialized proto.");
    }
  }

  std::string Serialize() const { return spp_->serialized_model_proto(); }

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

GemmaTokenizer::GemmaTokenizer(const Path& tokenizer_path) {
  impl_ = std::make_unique<Impl>(tokenizer_path);
}

// Default suffices, but they must be defined after GemmaTokenizer::Impl.
GemmaTokenizer::GemmaTokenizer() = default;
GemmaTokenizer::~GemmaTokenizer() = default;
GemmaTokenizer::GemmaTokenizer(GemmaTokenizer&& other) = default;
GemmaTokenizer& GemmaTokenizer::operator=(GemmaTokenizer&& other) = default;

std::string GemmaTokenizer::Serialize() const { return impl_->Serialize(); }

void GemmaTokenizer::Deserialize(const std::string& tokenizer_proto) {
  impl_ = std::make_unique<Impl>(tokenizer_proto);
}

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

void GemmaChatTemplate::Init(const GemmaTokenizer& tokenizer) {
  sot_user_.reserve(3);
  HWY_ASSERT(tokenizer.Encode("<start_of_turn>user\n", &sot_user_));
  sot_model_.reserve(3);
  HWY_ASSERT(tokenizer.Encode("<start_of_turn>model\n", &sot_model_));
  eot_.reserve(2);
  HWY_ASSERT(tokenizer.Encode("<end_of_turn>\n", &eot_));
}

std::vector<int> GemmaChatTemplate::Apply(size_t pos,
                                          const std::vector<int>& ids) const {
  HWY_ASSERT_M(!sot_user_.empty() && !sot_model_.empty() && !eot_.empty(),
               "GemmaChatTemplate has not been initialized.");
  std::vector<int> out;
  out.reserve(eot_.size() +
              sot_user_.size() +
              ids.size() +
              eot_.size() +
              sot_model_.size());
  if (pos > 0) {
    out.insert(out.cend(), eot_.cbegin(), eot_.cend());
  } else {
    out.push_back(BOS_ID);
  }
  out.insert(out.cend(), sot_user_.cbegin(), sot_user_.cend());
  out.insert(out.cend(), ids.cbegin(), ids.cend());
  out.insert(out.cend(), eot_.cbegin(), eot_.cend());
  out.insert(out.cend(), sot_model_.cbegin(), sot_model_.cend());
  return out;
}

std::vector<int> WrapAndTokenize(const GemmaTokenizer& tokenizer,
                                 const GemmaChatTemplate& chat_template,
                                 const ModelInfo& info, size_t pos,
                                 const std::string& prompt) {
  std::vector<int> tokens;
  HWY_ASSERT(tokenizer.Encode(prompt, &tokens));
  switch (info.wrapping) {
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

std::vector<int> WrapAndTokenize(const GemmaTokenizer& tokenizer,
                                 const GemmaChatTemplate& chat_template,
                                 const ModelInfo& info, size_t pos,
                                 const std::string& prompt,
                                 size_t image_batch_size) {
  std::vector<int> text_part;
  HWY_ASSERT(tokenizer.Encode(prompt, &text_part));
  std::vector<int> tokens;
  switch (info.wrapping) {
    case PromptWrapping::PALIGEMMA: {
      std::vector<int> sep;
      HWY_ASSERT(tokenizer.Encode("\n", &sep));
      tokens.reserve(image_batch_size + 1 + text_part.size() + sep.size());
      tokens.resize(image_batch_size, 0);
      HWY_ASSERT(pos == 0);
      tokens.push_back(BOS_ID);
      tokens.insert(tokens.cend(), text_part.cbegin(), text_part.cend());
      tokens.insert(tokens.cend(), sep.cbegin(), sep.cend());
      return tokens;
    }
    case PromptWrapping::GEMMA_VLM: {
      std::vector<int> soi;
      soi.reserve(2);
      HWY_ASSERT(tokenizer.Encode("\n\n<start_of_image>", &soi));
      std::vector<int> eoi;
      eoi.reserve(2);
      HWY_ASSERT(tokenizer.Encode("<end_of_image>\n\n", &eoi));
      tokens.reserve(text_part.size() + soi.size() + image_batch_size + eoi.size());
      tokens.insert(tokens.cend(), text_part.cbegin(), text_part.cend());
      tokens.insert(tokens.cend(), soi.cbegin(), soi.cend());
      tokens.insert(tokens.cend(), image_batch_size, -2);
      tokens.insert(tokens.cend(), eoi.cbegin(), eoi.cend());
      return chat_template.Apply(pos, tokens);
    }
    default:
      HWY_ASSERT_M(false, "Current variant does not support vision prompt.");
  }
}

}  // namespace gcpp
