// Copyright 2026 Google LLC
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

// Model Health Check: a comprehensive cross-entropy evaluation for IT models.
//
// Computes cross-entropy on diverse chat-formatted prompts with reference
// continuations. This exercises the full model pipeline (embeddings, attention,
// norms, MoE routing, KV cache) across different content types and sequence
// lengths.
//
// Unlike WikiText-2 perplexity (designed for base models), this test works
// with instruction-tuned models by wrapping all text in the model's chat
// template.
//
// What it catches:
//   - Bad norms → probability distributions drift → cross-entropy rises
//   - MoE routing bugs → wrong experts → wrong predictions on specific domains
//   - KV cache corruption → later tokens in long sequences get wrong
//   - Quantization issues → overall cross-entropy degrades
//   - Attention bugs → long-range dependencies fail

#include <stddef.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "evals/benchmark_helper.h"
#include "evals/cross_entropy.h"
#include "gemma/configs.h"
#include "gemma/gemma.h"
#include "hwy/base.h"
#include "hwy/tests/hwy_gtest.h"

namespace gcpp {
namespace {

// A test case: a user prompt + a reference continuation that any competent
// model should assign reasonable probability to.
struct HealthCheckCase {
  const char* category;
  const char* prompt;
  // Reference continuation (model response). Should be factually correct,
  // commonly-known information written in a clear, direct style.
  const char* continuation;
};

// Diverse set of prompts covering different model capabilities.
// Each continuation is ~50-200 tokens of text that exercises different
// parts of the model.
//
// IMPORTANT: Continuations should be stable facts / universal knowledge,
// not model-specific outputs. The goal is to test that the model assigns
// reasonable probability to correct, well-known information.
static const HealthCheckCase kHealthChecks[] = {
    // 1. Factual knowledge (geography, common knowledge)
    {"factual_geography",
     "What is the capital of France and what is it known for?",
     "The capital of France is Paris. Paris is the largest city in France and "
     "serves as the country's political, economic, and cultural center. It is "
     "located in northern France on the River Seine. Paris is known worldwide "
     "for landmarks such as the Eiffel Tower, the Louvre Museum, Notre-Dame "
     "Cathedral, and the Arc de Triomphe. The city is also famous for its "
     "cuisine, fashion industry, art galleries, and cafe culture. With a "
     "population of over 2 million in the city proper and over 12 million in "
     "the metropolitan area, Paris is one of the most visited cities in the "
     "world."},

    // 2. Mathematics (step-by-step arithmetic)
    {"math_arithmetic",
     "What is 247 multiplied by 38? Show your work step by step.",
     "To calculate 247 × 38, I'll break it down:\n\n"
     "247 × 38 = 247 × 30 + 247 × 8\n\n"
     "First: 247 × 30 = 7,410\n"
     "Second: 247 × 8 = 1,976\n\n"
     "Adding them together: 7,410 + 1,976 = 9,386\n\n"
     "Therefore, 247 × 38 = 9,386."},

    // 3. Code generation (standard algorithm)
    {"code_python",
     "Write a Python function to check if a string is a palindrome.",
     "```python\n"
     "def is_palindrome(s: str) -> bool:\n"
     "    \"\"\"Check if a string is a palindrome.\n\n"
     "    Args:\n"
     "        s: The input string to check.\n\n"
     "    Returns:\n"
     "        True if the string is a palindrome, False otherwise.\n"
     "    \"\"\"\n"
     "    # Remove spaces and convert to lowercase for comparison\n"
     "    cleaned = s.lower().replace(\" \", \"\")\n"
     "    return cleaned == cleaned[::-1]\n"
     "```\n\n"
     "This function works by first converting the string to lowercase and "
     "removing spaces, then comparing it with its reverse. For example, "
     "`is_palindrome(\"racecar\")` returns `True`, and "
     "`is_palindrome(\"hello\")` returns `False`."},

    // 4. Science explanation
    {"science_chemistry",
     "What is water made of and why is it important for life?",
     "Water is a chemical compound with the formula H2O, meaning each molecule "
     "consists of two hydrogen atoms bonded to one oxygen atom. These bonds "
     "are covalent, and the molecule has a bent shape that gives water its "
     "unique polar properties.\n\n"
     "Water is essential for life for several reasons:\n\n"
     "1. It serves as a universal solvent, dissolving many substances needed "
     "for biological processes.\n"
     "2. It regulates temperature in organisms and environments due to its "
     "high specific heat capacity.\n"
     "3. It participates directly in many biochemical reactions, including "
     "photosynthesis and hydrolysis.\n"
     "4. It provides structure and support in cells through turgor pressure.\n"
     "5. It transports nutrients and waste products throughout organisms."},

    // 5. History
    {"history_general",
     "Who was the first president of the United States and when did he serve?",
     "George Washington was the first president of the United States. He "
     "served two terms in office from 1789 to 1797. Before becoming "
     "president, Washington served as the commander-in-chief of the "
     "Continental Army during the American Revolutionary War, leading the "
     "colonies to independence from Great Britain. He is often referred to "
     "as the \"Father of His Country\" for his role in founding the nation.\n\n"
     "Washington was unanimously elected by the Electoral College in both "
     "1789 and 1792, making him the only president to receive 100 percent "
     "of the electoral votes. He voluntarily stepped down after two terms, "
     "establishing a precedent that was followed by subsequent presidents "
     "until Franklin D. Roosevelt."},

    // 6. Logical reasoning
    {"reasoning_logic",
     "If all dogs are mammals, and all mammals are animals, what can we "
     "conclude about dogs? Explain your reasoning.",
     "We can conclude that all dogs are animals.\n\n"
     "This follows from a logical structure called a syllogism, specifically "
     "a transitive inference:\n\n"
     "Premise 1: All dogs are mammals.\n"
     "Premise 2: All mammals are animals.\n"
     "Conclusion: Therefore, all dogs are animals.\n\n"
     "This works because the relationship \"is a\" is transitive. If A is a "
     "subset of B, and B is a subset of C, then A is a subset of C. In set "
     "theory notation: Dogs ⊂ Mammals ⊂ Animals, therefore Dogs ⊂ Animals.\n\n"
     "This is a valid deductive argument, meaning the conclusion necessarily "
     "follows from the premises. If the premises are true, the conclusion "
     "must also be true."},

    // 7. Translation (well-known phrase)
    {"language_translation",
     "Translate the following English text to Spanish: 'The weather is "
     "beautiful today. I would like to go for a walk in the park.'",
     "Here is the translation:\n\n"
     "\"El clima está hermoso hoy. Me gustaría ir a dar un paseo por el "
     "parque.\"\n\n"
     "A few notes on the translation:\n"
     "- \"El clima\" means \"the weather\"\n"
     "- \"está hermoso\" means \"is beautiful\"\n"
     "- \"hoy\" means \"today\"\n"
     "- \"Me gustaría\" means \"I would like\"\n"
     "- \"ir a dar un paseo\" means \"to go for a walk\"\n"
     "- \"por el parque\" means \"in/through the park\""},

    // 8. Technical CS explanation (exercises model's knowledge depth)
    {"technical_cs",
     "Explain what a hash table is and what its time complexity is for "
     "common operations.",
     "A hash table (also called a hash map) is a data structure that "
     "implements an associative array, mapping keys to values. It uses a "
     "hash function to compute an index into an array of buckets or slots, "
     "from which the desired value can be found.\n\n"
     "Time complexity for common operations:\n\n"
     "- **Insertion**: O(1) average case, O(n) worst case\n"
     "- **Lookup/Search**: O(1) average case, O(n) worst case\n"
     "- **Deletion**: O(1) average case, O(n) worst case\n\n"
     "The worst case of O(n) occurs when all keys hash to the same bucket, "
     "causing a collision chain. In practice, with a good hash function and "
     "appropriate load factor, hash tables provide constant-time operations "
     "on average, making them one of the most efficient data structures for "
     "key-value storage."},

    // 9. Longer multi-paragraph response (tests KV cache over longer range)
    {"long_explanation",
     "Explain the solar system. Describe each planet briefly.",
     "The solar system consists of the Sun and everything that orbits around "
     "it, including eight planets, dwarf planets, moons, asteroids, and "
     "comets. Here is a brief description of each planet, in order from the "
     "Sun:\n\n"
     "**Mercury** is the smallest planet and closest to the Sun. It has no "
     "atmosphere and experiences extreme temperature variations between day "
     "and night.\n\n"
     "**Venus** is the second planet and is similar in size to Earth. It has "
     "a thick atmosphere of carbon dioxide that creates a runaway greenhouse "
     "effect, making it the hottest planet.\n\n"
     "**Earth** is the third planet and the only one known to support life. "
     "It has liquid water on its surface and a protective atmosphere.\n\n"
     "**Mars** is the fourth planet, known as the Red Planet due to iron "
     "oxide on its surface. It has the largest volcano in the solar system, "
     "Olympus Mons.\n\n"
     "**Jupiter** is the largest planet, a gas giant with a mass more than "
     "twice that of all other planets combined. It is known for its Great "
     "Red Spot, a massive storm.\n\n"
     "**Saturn** is the sixth planet, famous for its prominent ring system "
     "made of ice and rock particles. It is the second-largest planet.\n\n"
     "**Uranus** is the seventh planet, an ice giant that rotates on its "
     "side. It has a blue-green color due to methane in its atmosphere.\n\n"
     "**Neptune** is the eighth and farthest planet. It is an ice giant with "
     "the strongest winds in the solar system, reaching speeds of over "
     "2,000 kilometers per hour."},

    // 10. Counting / structured output (exercises precise token generation)
    {"structured_list",
     "List the days of the week and the months of the year.",
     "**Days of the week:**\n"
     "1. Monday\n"
     "2. Tuesday\n"
     "3. Wednesday\n"
     "4. Thursday\n"
     "5. Friday\n"
     "6. Saturday\n"
     "7. Sunday\n\n"
     "**Months of the year:**\n"
     "1. January\n"
     "2. February\n"
     "3. March\n"
     "4. April\n"
     "5. May\n"
     "6. June\n"
     "7. July\n"
     "8. August\n"
     "9. September\n"
     "10. October\n"
     "11. November\n"
     "12. December"},
};

static constexpr size_t kNumChecks =
    sizeof(kHealthChecks) / sizeof(kHealthChecks[0]);

// Per-prompt result.
struct PromptResult {
  const char* category;
  size_t num_tokens;        // Total tokens in the sequence
  size_t prompt_tokens;     // Tokens in the prompt (chat-wrapped)
  size_t continuation_tokens;  // Tokens in the continuation
  float total_bits;         // Total cross-entropy in bits
  float bits_per_token;     // Average bits per token
};

class ModelHealthTest : public ::testing::Test {
 public:
  static void InitEnv(int argc, char** argv) {
    HWY_ASSERT(s_env == nullptr);
    ConsumedArgs consumed(argc, argv);
    GemmaArgs args(argc, argv, consumed);
    consumed.AbortIfUnconsumed();

    s_env = new GemmaEnv(args);
    const gcpp::ModelConfig& config = s_env->GetGemma()->Config();
    fprintf(stderr, "Model: %s\n", config.Specifier().c_str());
    fprintf(stderr, "Vocab size: %d\n", config.vocab_size);
    fprintf(stderr, "Max seq len: %u\n", config.max_seq_len);
  }

  static void DeleteEnv() { delete s_env; }

 protected:
  // Tokenizes a prompt+continuation pair using the model's chat template.
  // Returns the full token sequence: BOS + chat_wrap(prompt) + continuation.
  std::vector<int> TokenizeHealthCheck(const HealthCheckCase& hc,
                                       size_t* prompt_token_count) {
    // Wrap the prompt in chat template (includes BOS, turn tokens, etc.)
    std::vector<int> tokens = s_env->WrapAndTokenize(hc.prompt);
    *prompt_token_count = tokens.size();

    // Tokenize the continuation (raw, no wrapping - this is the model's
    // response).
    std::vector<int> cont_tokens = s_env->Tokenize(hc.continuation);
    tokens.insert(tokens.end(), cont_tokens.begin(), cont_tokens.end());

    return tokens;
  }

  // Computes cross-entropy for a single health check case.
  PromptResult RunHealthCheck(const HealthCheckCase& hc) {
    size_t prompt_tokens = 0;
    std::vector<int> tokens = TokenizeHealthCheck(hc, &prompt_tokens);

    const Gemma& gemma = *s_env->GetGemma();
    // Create a fresh KV cache for each prompt to ensure isolation.
    KVCache kv_cache(gemma.Config(), gemma.Inference(),
                     s_env->MutableEnv().ctx.allocator);

    float total_bits = ComputeCrossEntropy(
        gemma, tokens.size(), tokens, kv_cache, s_env->MutableEnv(),
        s_env->Verbosity());

    PromptResult result;
    result.category = hc.category;
    result.num_tokens = tokens.size();
    result.prompt_tokens = prompt_tokens;
    result.continuation_tokens = tokens.size() - prompt_tokens;
    result.total_bits = total_bits;
    result.bits_per_token = total_bits / tokens.size();
    return result;
  }

  static GemmaEnv* s_env;
};

GemmaEnv* ModelHealthTest::s_env = nullptr;

// Main health check: compute cross-entropy on all diverse prompts.
TEST_F(ModelHealthTest, DiverseCrossEntropy) {
  HWY_ASSERT(s_env != nullptr);
  HWY_ASSERT(s_env->GetGemma() != nullptr);

  const ModelConfig& config = s_env->GetGemma()->Config();

  fprintf(stderr, "\n");
  fprintf(stderr, "========================================\n");
  fprintf(stderr, "  MODEL HEALTH CHECK: %s\n", config.Specifier().c_str());
  fprintf(stderr, "========================================\n\n");

  std::vector<PromptResult> results;
  results.reserve(kNumChecks);

  float total_bits = 0.0f;
  size_t total_tokens = 0;
  float max_bits_per_token = 0.0f;
  float min_bits_per_token = 1e9f;

  for (size_t i = 0; i < kNumChecks; ++i) {
    const HealthCheckCase& hc = kHealthChecks[i];
    fprintf(stderr, "  [%zu/%zu] %-25s ", i + 1, kNumChecks, hc.category);

    PromptResult result = RunHealthCheck(hc);
    results.push_back(result);

    total_bits += result.total_bits;
    total_tokens += result.num_tokens;
    max_bits_per_token = std::max(max_bits_per_token, result.bits_per_token);
    min_bits_per_token = std::min(min_bits_per_token, result.bits_per_token);

    fprintf(stderr, "%4zu tokens  %7.2f total bits  %5.2f bits/tok\n",
            result.num_tokens, result.total_bits, result.bits_per_token);
  }

  const float avg_bits_per_token = total_bits / total_tokens;

  // Summary.
  fprintf(stderr, "\n");
  fprintf(stderr, "----------------------------------------\n");
  fprintf(stderr, "  SUMMARY\n");
  fprintf(stderr, "----------------------------------------\n");
  fprintf(stderr, "  Model:              %s\n", config.Specifier().c_str());
  fprintf(stderr, "  Total prompts:      %zu\n", kNumChecks);
  fprintf(stderr, "  Total tokens:       %zu\n", total_tokens);
  fprintf(stderr, "  Total bits:         %.2f\n", total_bits);
  fprintf(stderr, "  Avg bits/token:     %.4f\n", avg_bits_per_token);
  fprintf(stderr, "  Min bits/token:     %.4f (%s)\n", min_bits_per_token,
          results[std::distance(
                      results.begin(),
                      std::min_element(results.begin(), results.end(),
                                       [](const PromptResult& a,
                                          const PromptResult& b) {
                                         return a.bits_per_token <
                                                b.bits_per_token;
                                       }))]
              .category);
  fprintf(stderr, "  Max bits/token:     %.4f (%s)\n", max_bits_per_token,
          results[std::distance(
                      results.begin(),
                      std::max_element(results.begin(), results.end(),
                                       [](const PromptResult& a,
                                          const PromptResult& b) {
                                         return a.bits_per_token <
                                                b.bits_per_token;
                                       }))]
              .category);
  fprintf(stderr, "  Spread (max-min):   %.4f\n",
          max_bits_per_token - min_bits_per_token);
  fprintf(stderr, "----------------------------------------\n\n");

  // Sanity checks.
  //
  // These thresholds are intentionally generous - they're meant to catch
  // broken models, not to serve as quality benchmarks. A well-functioning
  // IT model should score well under these limits on common-knowledge text.
  //
  // If a model fails these checks, something is fundamentally wrong
  // (bad norms, broken attention, corrupted weights, etc.)

  // Per-prompt check: no single prompt should be wildly off.
  for (const auto& r : results) {
    // 10 bits/token means perplexity ~1024 - still very poor but not
    // catastrophically broken. A good model should be well under 5.
    EXPECT_LT(r.bits_per_token, 10.0f)
        << "Category '" << r.category
        << "' has very high cross-entropy, suggesting model issues.";
  }

  // Aggregate check.
  EXPECT_LT(avg_bits_per_token, 8.0f)
      << "Average cross-entropy is very high (" << avg_bits_per_token
      << " bits/token). This suggests the model is not functioning correctly.";

  // Spread check: if one category is wildly different from others, it
  // suggests a domain-specific issue (e.g., code handling broken).
  const float spread = max_bits_per_token - min_bits_per_token;
  EXPECT_LT(spread, 6.0f)
      << "Cross-entropy spread across categories is very large (" << spread
      << " bits), suggesting the model handles some domains much worse than "
         "others.";
}

// Deterministic generation check: verify that the model produces reasonable
// outputs for a set of simple prompts.
TEST_F(ModelHealthTest, DeterministicGeneration) {
  HWY_ASSERT(s_env != nullptr);

  s_env->SetMaxGeneratedTokens(64);
  s_env->MutableConfig().temperature = 0.0f;  // Deterministic
  s_env->MutableConfig().verbosity = 0;

  // Simple factual questions where we can check for expected substrings.
  static const struct {
    const char* prompt;
    const char* expected_substring;
    const char* description;
  } kGenerationChecks[] = {
      {"What is 2 + 2?", "4", "basic arithmetic"},
      {"What color is the sky on a clear day?", "blue", "common knowledge"},
      {"What planet is closest to the Sun?", "Mercury", "astronomy basics"},
      {"What language is spoken in Brazil?", "Portuguese", "geography/language"},
      {"What is the boiling point of water in Celsius?", "100",
       "science basics"},
      {"Complete the sequence: 2, 4, 6, 8, ...", "10", "number patterns"},
  };

  fprintf(stderr, "\n");
  fprintf(stderr, "========================================\n");
  fprintf(stderr, "  DETERMINISTIC GENERATION CHECKS\n");
  fprintf(stderr, "========================================\n\n");

  for (const auto& check : kGenerationChecks) {
    QueryResult result = s_env->QueryModel(check.prompt);
    const std::string response =
        result.response.substr(result.response_start_pos);

    fprintf(stderr, "  %-30s → ", check.description);

    bool found = response.find(check.expected_substring) != std::string::npos;
    if (found) {
      fprintf(stderr, "PASS (found '%s')\n", check.expected_substring);
    } else {
      fprintf(stderr, "FAIL (expected '%s' in: %.80s...)\n",
              check.expected_substring, response.c_str());
    }

    EXPECT_TRUE(found) << "Prompt: '" << check.prompt << "'\n"
                       << "Expected substring: '" << check.expected_substring
                       << "'\n"
                       << "Got: '" << response << "'";
  }

  fprintf(stderr, "\n");
}

}  // namespace
}  // namespace gcpp

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  gcpp::ModelHealthTest::InitEnv(argc, argv);
  int ret = RUN_ALL_TESTS();
  gcpp::ModelHealthTest::DeleteEnv();
  return ret;
}
