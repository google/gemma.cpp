// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include "evals/model_comparison.h"

#include <stdio.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void CheckNear(const char* name, double actual, double expected,
               double tolerance = 1E-12) {
  if (std::abs(actual - expected) > tolerance) {
    fprintf(stderr, "FAIL %s: %.17g != %.17g\n", name, actual, expected);
    ++failures;
  }
}

void TestKL() {
  const std::vector<float> root = {std::log(0.25f), std::log(0.75f)};
  const std::vector<float> same = root;
  const std::vector<float> shifted = {root[0] + 17.0f, root[1] + 17.0f};
  const std::vector<float> uniform = {0.0f, 0.0f};
  const double root_lse = gcpp::FullVocabLogSumExp(root.data(), root.size());

  CheckNear(
      "identical",
      gcpp::FullVocabKLDivergence(root, root_lse, same.data(), same.size()),
      0.0);
  CheckNear("shift invariant",
            gcpp::FullVocabKLDivergence(root, root_lse, shifted.data(),
                                        shifted.size()),
            0.0, 1E-7);
  const double expected = 0.25 * std::log(0.5) + 0.75 * std::log(1.5);
  CheckNear("known KL",
            gcpp::FullVocabKLDivergence(root, root_lse, uniform.data(),
                                        uniform.size()),
            expected, 1E-7);
}

void TestReferenceRoundTrip() {
  const char* path = "/tmp/gemma_model_comparison_test.bin";
  std::remove(path);
  const gcpp::ModelComparisonMetadata metadata = {
      /*vocab_size=*/3,
      /*sample_count=*/1,
      /*dataset_fingerprint=*/123,
      /*tokenizer_fingerprint=*/456,
  };
  const std::vector<float> logits = {1.0f, -2.0f, 4.0f};
  {
    gcpp::ModelComparisonWriter writer(path, metadata);
    writer.Write(7, 2, logits.data(), logits.size());
    writer.Finish();
  }
  {
    gcpp::ModelComparisonReader reader(path);
    reader.Validate(metadata);
    const gcpp::ModelComparisonRecord record = reader.Read();
    if (record.sample_id != 7 || record.expected_label != 2 ||
        record.logits != logits) {
      fprintf(stderr, "FAIL reference round trip\n");
      ++failures;
    }
    reader.Finish();
  }
  std::remove(path);
}

}  // namespace

int main() {
  TestKL();
  TestReferenceRoundTrip();
  if (failures != 0) {
    fprintf(stderr, "FAIL (%d failures)\n", failures);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
