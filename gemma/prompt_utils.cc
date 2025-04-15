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

#include "gemma/prompt_utils.h"

#include <string>
#include <vector>

#include "gemma/common.h"     // ModelInfo, PromptWrapping, BOS_ID
#include "gemma/tokenizer.h"  // GemmaTokenizer, GemmaChatTemplate
#include "hwy/base.h"         // HWY_ASSERT, HWY_ASSERT_M

namespace gcpp {

// Text
std::vector<int> WrapAndTokenize(const GemmaTokenizer& tokenizer,
                                 const GemmaChatTemplate& chat_template,
                                 const ModelInfo& info, size_t pos,
                                 const std::string& prompt) {
  std::vector<int> tokens;
  HWY_ASSERT(tokenizer.Encode(prompt, &tokens));

  switch (info.wrapping) {
    case PromptWrapping::GEMMA_IT:
    case PromptWrapping::GEMMA_VLM:  // VLM uses IT template for text part
      return chat_template.Apply(pos, tokens);
    default:  // Includes GEMMA_PT, PALIGEMMA (when called without image) etc.
      if (pos == 0) {
        tokens.insert(tokens.cbegin(), BOS_ID);
      }
      return tokens;
  }
}

// Vision
std::vector<int> WrapAndTokenize(const GemmaTokenizer& tokenizer,
                                 const GemmaChatTemplate& chat_template,
                                 const ModelInfo& info, size_t pos,
                                 const std::string& prompt,
                                 size_t image_batch_size) {
  std::vector<int> text_part;
  HWY_ASSERT(tokenizer.Encode(prompt, &text_part));
  switch (info.wrapping) {
    case PromptWrapping::PALIGEMMA:
      // PaliGemma expects image tokens first, then BOS, then text, then
      // separator. `pos` is typically expected to be 0 for the initial turn
      // with an image.
      HWY_ASSERT(pos == 0);
      return chat_template.WrapPali(text_part, image_batch_size);
    case PromptWrapping::GEMMA_VLM:
      // Gemma VLM (like Gemma 3) wraps the image tokens within the standard IT
      // template structure. The `Apply` method handles the turn structure
      // (BOS/EOT), and `WrapVLM` inserts the image part.
      return chat_template.Apply(
          pos, chat_template.WrapVLM(text_part, image_batch_size));
    default:
      // Should not be called for non-multimodal models with image_batch_size >
      // 0
      HWY_ASSERT_M(
          false,
          "WrapAndTokenize vision overload called for non-vision model type.");
      // Return something reasonable in case HWY_ASSERT is disabled in release
      // builds
      return {};
  }
}

}  // namespace gcpp
