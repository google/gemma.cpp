// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include "evals/model_comparison.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace gcpp {
namespace {

TEST(ModelComparisonMath, KnownDirectionalKL) {
  const std::vector<float> root = {std::log(0.25f), std::log(0.75f)};
  const std::vector<float> uniform = {0.0f, 0.0f};
  EXPECT_NEAR(FullVocabKLDivergence(root, uniform.data(), 2),
              0.25 * std::log(0.5) + 0.75 * std::log(1.5), 1E-7);
  EXPECT_NEAR(FullVocabKLDivergence(uniform, root.data(), 2),
              0.5 * std::log(2.0) + 0.5 * std::log(2.0 / 3.0), 1E-7);
}

TEST(ModelComparisonMath, IdentityAndShiftInvariance) {
  const std::vector<float> root = {1.0f, -2.0f, 4.0f};
  const std::vector<float> shifted = {1001.0f, 998.0f, 1004.0f};
  EXPECT_DOUBLE_EQ(FullVocabKLDivergence(root, root.data(), 3), 0.0);
  EXPECT_DOUBLE_EQ(FullVocabKLDivergence(root, shifted.data(), 3), 0.0);
  const float largest = std::numeric_limits<float>::max();
  const std::vector<float> extreme = {largest, -largest};
  const std::vector<float> uniform = {largest, largest};
  EXPECT_NEAR(FullVocabKLDivergence(extreme, uniform.data(), 2), std::log(2.0),
              1E-12);
  EXPECT_TRUE(std::isfinite(FullVocabKLDivergence(uniform, extreme.data(), 2)));
}

TEST(ModelComparisonMath, RejectsInvalidLogits) {
  const std::vector<float> root = {0.0f, 1.0f};
  EXPECT_THROW(FullVocabKLDivergence({}, root.data(), 0),
               std::invalid_argument);
  EXPECT_THROW(FullVocabKLDivergence(root, root.data(), 1),
               std::invalid_argument);
  for (float value : {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()}) {
    const std::vector<float> bad = {0.0f, value};
    EXPECT_THROW(FullVocabKLDivergence(root, bad.data(), 2),
                 std::invalid_argument);
    EXPECT_THROW(FullVocabKLDivergence(bad, root.data(), 2),
                 std::invalid_argument);
  }
}

TEST(ModelComparisonMath, ExactTokenizerEncoding) {
  EXPECT_EQ(ModelComparisonHex(std::string("\0\x80\xff\n", 4)), "0080ff0a");
  EXPECT_EQ(ModelComparisonHex(""), "");
}

TEST(ModelComparisonMath, AnswerSpellingsAndProbabilities) {
  const std::vector<float> logits = {0, 1, 2, 3, 4, 100};
  const auto answer = ScoreMmluAnswer(logits.data(), logits.size(),
                                      {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 0}});
  EXPECT_EQ(answer.label, 0);
  EXPECT_EQ(answer.token, 4);  // The alternate spelling of A wins.
  EXPECT_EQ(answer.logits, (std::array<float, 4>{4, 1, 2, 3}));
  EXPECT_DOUBLE_EQ(answer.margin, 1.0);
  double sum = 0.0;
  for (double probability : answer.probabilities) sum += probability;
  EXPECT_NEAR(sum, 1.0, 1E-15);
  EXPECT_NEAR(answer.probabilities[0],
              1.0 / (1 + std::exp(-1.0) + std::exp(-2.0) + std::exp(-3.0)),
              1E-15);
  // The much larger non-answer token must not participate in label scoring.
}

TEST(ModelComparisonMath, AnswerTiesAndInvalidTokens) {
  const std::vector<float> logits(4, 1.0f);
  const std::vector<std::pair<int, int>> tokens = {
      {3, 3}, {2, 2}, {1, 1}, {0, 0}};
  const auto answer = ScoreMmluAnswer(logits.data(), 4, tokens);
  EXPECT_EQ(answer.label, 0);
  EXPECT_DOUBLE_EQ(answer.margin, 0.0);
  for (double p : answer.probabilities) EXPECT_DOUBLE_EQ(p, 0.25);
  EXPECT_THROW(ScoreMmluAnswer(logits.data(), 4, {{0, 0}}),
               std::invalid_argument);
  EXPECT_THROW(ScoreMmluAnswer(logits.data(), 4, {{-1, 0}}),
               std::invalid_argument);
  EXPECT_THROW(ScoreMmluAnswer(logits.data(), 4, {{4, 0}}),
               std::invalid_argument);
  EXPECT_THROW(ScoreMmluAnswer(logits.data(), 4, {{0, 4}}),
               std::invalid_argument);
}

class ModelComparisonReference : public testing::Test {
 protected:
  void SetUp() override {
    // Atomic directory creation prevents collisions between concurrent tests.
    std::random_device random;
    for (size_t attempt = 0; attempt < 100; ++attempt) {
      directory_ = std::filesystem::temp_directory_path() /
                   ("gemma-comparison-" + std::to_string(random()));
      if (std::filesystem::create_directory(directory_)) {
        path_ = (directory_ / "reference.jsonl").string();
        return;
      }
    }
    FAIL() << "cannot create temporary test directory";
  }
  void TearDown() override { std::filesystem::remove_all(directory_); }
  void WriteReference() {
    ModelComparisonWriter writer(path_, metadata_);
    writer.Write(record_);
    writer.Finish();
  }
  std::string ReadText() {
    std::ifstream input(path_);
    return std::string(std::istreambuf_iterator<char>(input), {});
  }
  void ReplaceText(const std::string& text) { std::ofstream(path_) << text; }

  std::filesystem::path directory_;
  std::string path_;
  ModelComparisonMetadata metadata_{3, 1, "dataset\n", "0080ff"};
  ModelComparisonRecord record_{7, 2, {1, 2}, {1.25f, -2.0f, 4.0f}};
};

TEST_F(ModelComparisonReference, TextRoundTrip) {
  record_.logits = {std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::denorm_min(), -0.0f};
  WriteReference();
  EXPECT_EQ(ReadText().front(), '{');
  ModelComparisonReader reader(path_);
  reader.Validate(metadata_);
  const auto actual = reader.Read();
  EXPECT_EQ(actual.sample_id, record_.sample_id);
  EXPECT_EQ(actual.expected_label, record_.expected_label);
  EXPECT_EQ(actual.prompt, record_.prompt);
  EXPECT_EQ(actual.logits, record_.logits);
  EXPECT_TRUE(std::signbit(actual.logits.back()));
  reader.Finish();
  EXPECT_THROW(reader.Read(), std::runtime_error);
}

TEST_F(ModelComparisonReference, RejectsMismatchedMetadata) {
  WriteReference();
  ModelComparisonReader reader(path_);
  auto changed = metadata_;
  ++changed.vocab_size;
  EXPECT_THROW(reader.Validate(changed), std::runtime_error);
  changed = metadata_;
  ++changed.sample_count;
  EXPECT_THROW(reader.Validate(changed), std::runtime_error);
  changed = metadata_;
  changed.dataset += " ";
  EXPECT_THROW(reader.Validate(changed), std::runtime_error);
  changed = metadata_;
  changed.tokenizer_hex += "00";
  EXPECT_THROW(reader.Validate(changed), std::runtime_error);
}

TEST_F(ModelComparisonReference, RejectsMissingAndExtraRecords) {
  {
    ModelComparisonWriter writer(path_, metadata_);
    EXPECT_THROW(writer.Finish(), std::runtime_error);
    writer.Write(record_);
    EXPECT_THROW(writer.Write(record_), std::runtime_error);
    writer.Finish();
    writer.Finish();
    EXPECT_THROW(writer.Write(record_), std::runtime_error);
  }
  ModelComparisonReader reader(path_);
  EXPECT_THROW(reader.Finish(), std::runtime_error);
  reader.Read();
  reader.Finish();
}

TEST_F(ModelComparisonReference, RejectsTruncatedAndTrailingData) {
  WriteReference();
  const auto original = ReadText();
  ReplaceText(original.substr(0, original.find('\n') + 1));
  {
    ModelComparisonReader reader(path_);
    EXPECT_THROW(reader.Read(), std::exception);
  }
  ReplaceText(original.substr(0, original.size() - 5));
  {
    ModelComparisonReader reader(path_);
    EXPECT_THROW(reader.Read(), std::exception);
  }
  ReplaceText(original + "{}\n");
  {
    ModelComparisonReader reader(path_);
    reader.Read();
    EXPECT_THROW(reader.Finish(), std::runtime_error);
  }
  ReplaceText("{}");
  EXPECT_THROW(ModelComparisonReader reader(path_), std::exception);
}

TEST_F(ModelComparisonReference, RejectsInvalidTextFields) {
  WriteReference();
  const auto original = ReadText();
  for (const auto& [from, to] :
       std::vector<std::pair<std::string, std::string>>{
           {"\"version\":1", "\"version\":2"},
           {"\"sample_count\":1", "\"sample_count\":-1"},
           {"\"vocab_size\":3", "\"vocab_size\":3.5"}}) {
    std::string changed = original;
    changed.replace(changed.find(from), from.size(), to);
    ReplaceText(changed);
    EXPECT_THROW(ModelComparisonReader reader(path_), std::exception);
  }
  for (const auto& [from, to] :
       std::vector<std::pair<std::string, std::string>>{
           {"\"id\":7", "\"id\":7.5"},
           {"\"expected\":2", "\"expected\":4294967298"},
           {"\"prompt\":[1,2]", "\"prompt\":[1.5,2]"},
           {"\"logits\":[1.25,-2.0,4.0]", "\"logits\":[null,-2.0,4.0]"}}) {
    std::string changed = original;
    changed.replace(changed.find(from), from.size(), to);
    ReplaceText(changed);
    ModelComparisonReader reader(path_);
    EXPECT_THROW(reader.Read(), std::exception);
  }
}

TEST_F(ModelComparisonReference, RejectsInvalidRecords) {
  ModelComparisonWriter writer(path_, metadata_);
  auto bad = record_;
  bad.logits.pop_back();
  EXPECT_THROW(writer.Write(bad), std::invalid_argument);
  bad = record_;
  bad.logits[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(writer.Write(bad), std::invalid_argument);
  bad = record_;
  bad.expected_label = 4;
  EXPECT_THROW(writer.Write(bad), std::invalid_argument);
  bad = record_;
  bad.prompt = {-1};
  EXPECT_THROW(writer.Write(bad), std::invalid_argument);
  bad.prompt = {3};
  EXPECT_THROW(writer.Write(bad), std::invalid_argument);
  bad.prompt.clear();
  EXPECT_THROW(writer.Write(bad), std::invalid_argument);
}

}  // namespace
}  // namespace gcpp
