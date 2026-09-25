// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include "evals/model_comparison.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace gcpp {
namespace {

using Json = nlohmann::json;

// Keep the normalizer relative to the maximum: adding log(sum) to a very
// large maximum would lose the normalization, even in double precision.
double MaxLogit(const float* logits, size_t size) {
  if (size == 0) throw std::invalid_argument("empty logits");
  for (size_t i = 0; i < size; ++i) {
    if (!std::isfinite(logits[i])) {
      throw std::invalid_argument("non-finite logit");
    }
  }
  return *std::max_element(logits, logits + size);
}

double LogSumRelative(const float* logits, size_t size, double maximum) {
  double sum = 0.0;
  for (size_t i = 0; i < size; ++i) {
    sum += std::exp(static_cast<double>(logits[i]) - maximum);
  }
  return std::log(sum);
}

template <typename T>
T Integer(const Json& value) {
  if (!value.is_number_integer() || value < std::numeric_limits<T>::lowest() ||
      value > std::numeric_limits<T>::max()) {
    throw std::invalid_argument("invalid integer in KL reference");
  }
  return value.get<T>();
}

Json ReadLine(std::ifstream& stream) {
  std::string line;
  if (!std::getline(stream, line)) {
    throw std::runtime_error("truncated KL reference");
  }
  return Json::parse(line);
}

void ValidateRecord(const ModelComparisonRecord& record, size_t vocab_size) {
  if (record.expected_label < 0 || record.expected_label > 3 ||
      record.prompt.empty() || record.logits.size() != vocab_size) {
    throw std::invalid_argument("invalid KL reference record");
  }
  for (int token : record.prompt) {
    if (token < 0 || static_cast<size_t>(token) >= vocab_size) {
      throw std::invalid_argument("invalid KL reference prompt token");
    }
  }
  MaxLogit(record.logits.data(), record.logits.size());
}

}  // namespace

std::string ModelComparisonHex(const std::string& bytes) {
  constexpr char digits[] = "0123456789abcdef";
  std::string text;
  text.reserve(bytes.size() * 2);
  for (unsigned char byte : bytes) {
    text.push_back(digits[byte >> 4]);
    text.push_back(digits[byte & 15]);
  }
  return text;
}

double FullVocabKLDivergence(const std::vector<float>& root_logits,
                             const float* target_logits, size_t size) {
  if (root_logits.size() != size) {
    throw std::invalid_argument("root/target vocabulary size mismatch");
  }
  const double root_max = MaxLogit(root_logits.data(), size);
  const double target_max = MaxLogit(target_logits, size);
  const double root_norm = LogSumRelative(root_logits.data(), size, root_max);
  const double target_norm = LogSumRelative(target_logits, size, target_max);
  double kl = 0.0;
  for (size_t i = 0; i < size; ++i) {
    const double p =
        (static_cast<double>(root_logits[i]) - root_max) - root_norm;
    const double q =
        (static_cast<double>(target_logits[i]) - target_max) - target_norm;
    kl += std::exp(p) * (p - q);
  }
  // The exact result is nonnegative; roundoff can produce tiny negatives.
  return std::max(0.0, kl);
}

MmluAnswer ScoreMmluAnswer(const float* logits, size_t size,
                           const std::vector<std::pair<int, int>>& tokens) {
  MaxLogit(logits, size);  // Reject invalid model outputs before reporting.
  MmluAnswer result;
  result.logits.fill(-std::numeric_limits<float>::infinity());
  std::array<int, 4> best_tokens = {-1, -1, -1, -1};
  for (const auto& [token, label] : tokens) {
    if (token < 0 || static_cast<size_t>(token) >= size || label < 0 ||
        label > 3) {
      throw std::invalid_argument("invalid MMLU answer token/label");
    }
    if (logits[token] > result.logits[label]) {
      result.logits[label] = logits[token];
      best_tokens[label] = token;
    }
  }
  for (int token : best_tokens) {
    if (token == -1) throw std::invalid_argument("missing MMLU answer label");
  }
  result.label = static_cast<int>(
      std::max_element(result.logits.begin(), result.logits.end()) -
      result.logits.begin());
  result.token = best_tokens[result.label];
  const double maximum = result.logits[result.label];
  double sum = 0.0;
  double second = -std::numeric_limits<double>::infinity();
  for (int label = 0; label < 4; ++label) {
    result.probabilities[label] =
        std::exp(static_cast<double>(result.logits[label]) - maximum);
    sum += result.probabilities[label];
    if (label != result.label)
      second = std::max(second, static_cast<double>(result.logits[label]));
  }
  for (double& probability : result.probabilities) probability /= sum;
  result.margin = maximum - second;
  return result;
}

ModelComparisonWriter::ModelComparisonWriter(
    const std::string& path, const ModelComparisonMetadata& metadata)
    : stream_(path), metadata_(metadata) {
  if (!stream_) throw std::runtime_error("cannot create KL reference: " + path);
  if (metadata.vocab_size == 0 || metadata.sample_count == 0) {
    throw std::invalid_argument("empty KL reference metadata");
  }
  const Json header = {{"format", "gemma-mmlu-reference"},
                       {"version", 1},
                       {"vocab_size", metadata.vocab_size},
                       {"sample_count", metadata.sample_count},
                       {"dataset", metadata.dataset},
                       {"tokenizer_hex", metadata.tokenizer_hex}};
  stream_ << header.dump() << '\n';
  if (!stream_) throw std::runtime_error("failed to write KL reference header");
}

void ModelComparisonWriter::Write(const ModelComparisonRecord& record) {
  if (finished_ || records_written_ >= metadata_.sample_count) {
    throw std::runtime_error("too many KL reference writes");
  }
  ValidateRecord(record, metadata_.vocab_size);
  const Json row = {{"id", record.sample_id},
                    {"expected", record.expected_label},
                    {"prompt", record.prompt},
                    {"logits", record.logits}};
  stream_ << row.dump() << '\n';
  if (!stream_) throw std::runtime_error("failed to write KL reference logits");
  ++records_written_;
}

void ModelComparisonWriter::Finish() {
  if (finished_) return;
  if (records_written_ != metadata_.sample_count) {
    throw std::runtime_error("KL reference record count mismatch");
  }
  stream_.close();
  if (!stream_) throw std::runtime_error("failed to finish KL reference");
  finished_ = true;
}

ModelComparisonReader::ModelComparisonReader(const std::string& path)
    : stream_(path) {
  if (!stream_) throw std::runtime_error("cannot open KL reference: " + path);
  const Json header = ReadLine(stream_);
  if (header.at("format") != "gemma-mmlu-reference" ||
      Integer<int>(header.at("version")) != 1) {
    throw std::runtime_error("unsupported KL reference format/version");
  }
  metadata_.vocab_size = Integer<uint32_t>(header.at("vocab_size"));
  metadata_.sample_count = Integer<uint64_t>(header.at("sample_count"));
  metadata_.dataset = header.at("dataset").get<std::string>();
  metadata_.tokenizer_hex = header.at("tokenizer_hex").get<std::string>();
  if (metadata_.vocab_size == 0 || metadata_.sample_count == 0) {
    throw std::runtime_error("empty KL reference metadata");
  }
}

void ModelComparisonReader::Validate(
    const ModelComparisonMetadata& expected) const {
  if (metadata_.vocab_size != expected.vocab_size ||
      metadata_.sample_count != expected.sample_count ||
      metadata_.dataset != expected.dataset ||
      metadata_.tokenizer_hex != expected.tokenizer_hex) {
    throw std::runtime_error(
        "KL reference metadata mismatch (vocabulary, sample count, dataset, or "
        "tokenizer)");
  }
}

ModelComparisonRecord ModelComparisonReader::Read() {
  if (records_read_ >= metadata_.sample_count) {
    throw std::runtime_error("too many KL reference reads");
  }
  const Json row = ReadLine(stream_);
  ModelComparisonRecord record;
  record.sample_id = Integer<int64_t>(row.at("id"));
  record.expected_label = Integer<int32_t>(row.at("expected"));
  if (!row.at("prompt").is_array()) {
    throw std::invalid_argument("invalid KL reference prompt");
  }
  for (const auto& token : row.at("prompt"))
    record.prompt.push_back(Integer<int>(token));
  record.logits = row.at("logits").get<std::vector<float>>();
  ValidateRecord(record, metadata_.vocab_size);
  ++records_read_;
  return record;
}

void ModelComparisonReader::Finish() {
  if (records_read_ != metadata_.sample_count) {
    throw std::runtime_error("unread KL reference records");
  }
  if (stream_.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error("trailing data in KL reference");
  }
}

}  // namespace gcpp
