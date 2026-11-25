// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include "evals/model_comparison.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace gcpp {
namespace {

constexpr char kMagic[8] = {'G', 'C', 'P', 'P', 'K', 'L', '0', '1'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kEndianMarker = 0x01020304u;

template <typename T>
void WriteValue(std::ofstream& stream, const T& value) {
  static_assert(std::is_trivially_copyable<T>::value, "binary scalar");
  stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!stream) throw std::runtime_error("failed to write KL reference file");
}

template <typename T>
T ReadValue(std::ifstream& stream) {
  static_assert(std::is_trivially_copyable<T>::value, "binary scalar");
  T value;
  stream.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!stream) throw std::runtime_error("truncated KL reference file");
  return value;
}

void RequireEqual(const char* name, uint64_t actual, uint64_t expected) {
  if (actual != expected) {
    throw std::runtime_error(std::string("KL reference ") + name +
                             " mismatch: " + std::to_string(actual) +
                             " != " + std::to_string(expected));
  }
}

}  // namespace

uint64_t ModelComparisonFingerprint(const std::string& bytes) {
  uint64_t hash = 14695981039346656037ull;
  for (const unsigned char byte : bytes) {
    hash ^= byte;
    hash *= 1099511628211ull;
  }
  return hash;
}

double FullVocabLogSumExp(const float* logits, size_t size) {
  if (size == 0) throw std::invalid_argument("empty logits");
  float max_logit = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < size; ++i) {
    max_logit = std::max(max_logit, logits[i]);
  }
  if (!std::isfinite(max_logit)) {
    throw std::invalid_argument("non-finite maximum logit");
  }

  double sum = 0.0;
  for (size_t i = 0; i < size; ++i) {
    sum += std::exp(static_cast<double>(logits[i] - max_logit));
  }
  return static_cast<double>(max_logit) + std::log(sum);
}

double FullVocabKLDivergence(const std::vector<float>& root_logits,
                             double root_log_sum_exp,
                             const float* target_logits, size_t size) {
  if (root_logits.size() != size) {
    throw std::invalid_argument("root/target vocabulary size mismatch");
  }
  const double target_log_sum_exp = FullVocabLogSumExp(target_logits, size);
  double kl = 0.0;
  for (size_t i = 0; i < size; ++i) {
    const double root_log_prob =
        static_cast<double>(root_logits[i]) - root_log_sum_exp;
    const double target_log_prob =
        static_cast<double>(target_logits[i]) - target_log_sum_exp;
    kl += std::exp(root_log_prob) * (root_log_prob - target_log_prob);
  }
  return kl < 0.0 && kl > -1E-12 ? 0.0 : kl;
}

ModelComparisonWriter::ModelComparisonWriter(
    const std::string& path, const ModelComparisonMetadata& metadata)
    : stream_(path, std::ios::binary), metadata_(metadata) {
  static_assert(sizeof(float) == 4, "reference format requires float32");
  if (!stream_) throw std::runtime_error("cannot create KL reference: " + path);
  stream_.write(kMagic, sizeof(kMagic));
  WriteValue(stream_, kVersion);
  WriteValue(stream_, kEndianMarker);
  WriteValue(stream_, metadata_.vocab_size);
  WriteValue(stream_, metadata_.sample_count);
  WriteValue(stream_, metadata_.dataset_fingerprint);
  WriteValue(stream_, metadata_.tokenizer_fingerprint);
}

ModelComparisonWriter::~ModelComparisonWriter() {
  if (!finished_) stream_.close();
}

void ModelComparisonWriter::Write(int64_t sample_id, int32_t expected_label,
                                  const float* logits, size_t size) {
  if (finished_) throw std::runtime_error("KL reference already finished");
  if (size != metadata_.vocab_size) {
    throw std::invalid_argument("logits do not match reference vocabulary");
  }
  if (records_written_ >= metadata_.sample_count) {
    throw std::runtime_error("too many KL reference records");
  }
  WriteValue(stream_, sample_id);
  WriteValue(stream_, expected_label);
  const double log_sum_exp = FullVocabLogSumExp(logits, size);
  WriteValue(stream_, log_sum_exp);
  stream_.write(reinterpret_cast<const char*>(logits),
                static_cast<std::streamsize>(size * sizeof(float)));
  if (!stream_) throw std::runtime_error("failed to write KL reference logits");
  ++records_written_;
}

void ModelComparisonWriter::Finish() {
  if (finished_) return;
  if (records_written_ != metadata_.sample_count) {
    throw std::runtime_error("KL reference record count mismatch");
  }
  stream_.flush();
  if (!stream_) throw std::runtime_error("failed to finish KL reference file");
  finished_ = true;
}

ModelComparisonReader::ModelComparisonReader(const std::string& path)
    : stream_(path, std::ios::binary) {
  if (!stream_) throw std::runtime_error("cannot open KL reference: " + path);
  char magic[sizeof(kMagic)];
  stream_.read(magic, sizeof(magic));
  if (!stream_ || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    throw std::runtime_error("invalid KL reference magic");
  }
  RequireEqual("version", ReadValue<uint32_t>(stream_), kVersion);
  RequireEqual("endianness", ReadValue<uint32_t>(stream_), kEndianMarker);
  metadata_.vocab_size = ReadValue<uint32_t>(stream_);
  metadata_.sample_count = ReadValue<uint64_t>(stream_);
  metadata_.dataset_fingerprint = ReadValue<uint64_t>(stream_);
  metadata_.tokenizer_fingerprint = ReadValue<uint64_t>(stream_);
}

void ModelComparisonReader::Validate(
    const ModelComparisonMetadata& expected) const {
  RequireEqual("vocabulary", metadata_.vocab_size, expected.vocab_size);
  RequireEqual("sample count", metadata_.sample_count, expected.sample_count);
  RequireEqual("dataset fingerprint", metadata_.dataset_fingerprint,
               expected.dataset_fingerprint);
  RequireEqual("tokenizer fingerprint", metadata_.tokenizer_fingerprint,
               expected.tokenizer_fingerprint);
}

ModelComparisonRecord ModelComparisonReader::Read() {
  if (records_read_ >= metadata_.sample_count) {
    throw std::runtime_error("too many KL reference reads");
  }
  ModelComparisonRecord record;
  record.sample_id = ReadValue<int64_t>(stream_);
  record.expected_label = ReadValue<int32_t>(stream_);
  record.log_sum_exp = ReadValue<double>(stream_);
  record.logits.resize(metadata_.vocab_size);
  stream_.read(
      reinterpret_cast<char*>(record.logits.data()),
      static_cast<std::streamsize>(record.logits.size() * sizeof(float)));
  if (!stream_) throw std::runtime_error("truncated KL reference logits");
  ++records_read_;
  return record;
}

void ModelComparisonReader::Finish() {
  if (records_read_ != metadata_.sample_count) {
    throw std::runtime_error("unread KL reference records");
  }
  if (stream_.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error("trailing bytes in KL reference file");
  }
}

}  // namespace gcpp
