// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#ifndef THIRD_PARTY_GEMMA_CPP_EVALS_MODEL_COMPARISON_H_
#define THIRD_PARTY_GEMMA_CPP_EVALS_MODEL_COMPARISON_H_

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace gcpp {

// Exact, text-safe representation of serialized tokenizer bytes (no hashing).
std::string ModelComparisonHex(const std::string& bytes);

// D_KL(root || target), in nats, over the entire vocabulary. Requires finite
// logits and matching, nonempty vocabularies. Normalizes in double precision.
double FullVocabKLDivergence(const std::vector<float>& root_logits,
                             const float* target_logits, size_t size);

struct MmluAnswer {
  int token = -1;
  int label = -1;
  std::array<float, 4> logits;
  std::array<double, 4> probabilities;
  double margin = 0.0;
};

// Scores each label by its highest-logit token spelling. Probabilities are a
// softmax over those four scores, not sums of token probabilities. Ties select
// the lowest label. Each label needs at least one in-vocabulary token.
MmluAnswer ScoreMmluAnswer(const float* logits, size_t size,
                           const std::vector<std::pair<int, int>>& tokens);

struct ModelComparisonMetadata {
  uint32_t vocab_size = 0;
  uint64_t sample_count = 0;
  std::string dataset;
  std::string tokenizer_hex;
};

struct ModelComparisonRecord {
  int64_t sample_id = 0;
  int32_t expected_label = 0;
  std::vector<int> prompt;
  std::vector<float> logits;
};

// Versioned JSON Lines: one metadata header, then one record per question.
// Float32 logits round-trip exactly. Readers hold only one question at a time.
class ModelComparisonWriter {
 public:
  ModelComparisonWriter(const std::string& path,
                        const ModelComparisonMetadata& metadata);
  void Write(const ModelComparisonRecord& record);
  void Finish();

 private:
  std::ofstream stream_;
  ModelComparisonMetadata metadata_;
  uint64_t records_written_ = 0;
  bool finished_ = false;
};

class ModelComparisonReader {
 public:
  explicit ModelComparisonReader(const std::string& path);
  void Validate(const ModelComparisonMetadata& expected) const;
  ModelComparisonRecord Read();
  void Finish();

 private:
  std::ifstream stream_;
  ModelComparisonMetadata metadata_;
  uint64_t records_read_ = 0;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_EVALS_MODEL_COMPARISON_H_
