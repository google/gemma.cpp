// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#ifndef THIRD_PARTY_GEMMA_CPP_EVALS_MODEL_COMPARISON_H_
#define THIRD_PARTY_GEMMA_CPP_EVALS_MODEL_COMPARISON_H_

#include <stdint.h>

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

namespace gcpp {

uint64_t ModelComparisonFingerprint(const std::string& bytes);

// Numerically stable full-vocabulary operations. KL is directional:
// D_KL(root || target).
double FullVocabLogSumExp(const float* logits, size_t size);
double FullVocabKLDivergence(const std::vector<float>& root_logits,
                             double root_log_sum_exp,
                             const float* target_logits, size_t size);

struct ModelComparisonMetadata {
  uint32_t vocab_size = 0;
  uint64_t sample_count = 0;
  uint64_t dataset_fingerprint = 0;
  uint64_t tokenizer_fingerprint = 0;
};

struct ModelComparisonRecord {
  int64_t sample_id = 0;
  int32_t expected_label = 0;
  double log_sum_exp = 0.0;
  std::vector<float> logits;
};

// Versioned binary store for root-model logits. It is uncompressed so target
// runs can stream one question at a time without loading the whole dataset.
class ModelComparisonWriter {
 public:
  ModelComparisonWriter(const std::string& path,
                        const ModelComparisonMetadata& metadata);
  ~ModelComparisonWriter();

  void Write(int64_t sample_id, int32_t expected_label, const float* logits,
             size_t size);
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

  const ModelComparisonMetadata& Metadata() const { return metadata_; }
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
