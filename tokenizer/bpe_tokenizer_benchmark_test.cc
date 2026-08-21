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

#include <cstddef>
#include <string>
#include <vector>

#include "testing/base/public/benchmark.h"
#include "gemma/tokenizer.h"
#include "tokenizer/bpe_tokenizer.h"
#include "io/io.h"

namespace gcpp {
namespace {

constexpr const char* kTokenizerPacked =
    "tokenizer/testdata/tokenizer.packed";
constexpr const char* kTokenizerModel =
    "tokenizer/testdata/tokenizer.model";
constexpr const char* kCorpus =
    "testdata/frankenstein.txt";

std::string ReadFileOrAbort(const std::string& path) {
  return ReadFileToString(Path(path));
}

void BM_LoadSentencePiece(benchmark::State& state) {
  std::string blob = ReadFileOrAbort(kTokenizerModel);
  state.SetLabel("BlobSize: " + std::to_string(blob.size()) + "B");
  for (auto s : state) {
    GemmaTokenizer sp(blob);
    benchmark::DoNotOptimize(sp);
  }
}
BENCHMARK(BM_LoadSentencePiece);

void BM_LoadCompactBpe(benchmark::State& state) {
  std::string blob = ReadFileOrAbort(kTokenizerPacked);
  state.SetLabel("BlobSize: " + std::to_string(blob.size()) + "B");
  for (auto s : state) {
    GemmaTokenizer bpe(CreateBpeTokenizer(blob));
    benchmark::DoNotOptimize(bpe);
  }
}
BENCHMARK(BM_LoadCompactBpe);

void BM_EncodeSentencePiece(benchmark::State& state) {
  std::string corpus = ReadFileOrAbort(kCorpus);
  GemmaTokenizer sp(ReadFileOrAbort(kTokenizerModel));
  for (auto s : state) {
    std::vector<int> ids;
    sp.Encode(corpus, &ids);
    benchmark::DoNotOptimize(ids);
  }
  state.SetBytesProcessed(state.iterations() * corpus.size());
}
BENCHMARK(BM_EncodeSentencePiece);

void BM_EncodeCompactBpe(benchmark::State& state) {
  std::string corpus = ReadFileOrAbort(kCorpus);
  GemmaTokenizer bpe(CreateBpeTokenizer(ReadFileOrAbort(kTokenizerPacked)));
  for (auto s : state) {
    std::vector<int> ids;
    bpe.Encode(corpus, &ids);
    benchmark::DoNotOptimize(ids);
  }
  state.SetBytesProcessed(state.iterations() * corpus.size());
}
BENCHMARK(BM_EncodeCompactBpe);

void BM_DecodeSentencePiece(benchmark::State& state) {
  std::string corpus = ReadFileOrAbort(kCorpus);
  GemmaTokenizer sp(ReadFileOrAbort(kTokenizerModel));
  std::vector<int> ids;
  sp.Encode(corpus, &ids);
  size_t detokenized_size = 0;
  for (auto s : state) {
    std::string detokenized;
    sp.Decode(ids, &detokenized);
    benchmark::DoNotOptimize(detokenized);
    detokenized_size = detokenized.size();
  }
  state.SetBytesProcessed(state.iterations() * detokenized_size);
}
BENCHMARK(BM_DecodeSentencePiece);

void BM_DecodeCompactBpe(benchmark::State& state) {
  std::string corpus = ReadFileOrAbort(kCorpus);
  GemmaTokenizer bpe(CreateBpeTokenizer(ReadFileOrAbort(kTokenizerPacked)));
  std::vector<int> ids;
  bpe.Encode(corpus, &ids);
  size_t detokenized_size = 0;
  for (auto s : state) {
    std::string detokenized;
    bpe.Decode(ids, &detokenized);
    benchmark::DoNotOptimize(detokenized);
    detokenized_size = detokenized.size();
  }
  state.SetBytesProcessed(state.iterations() * detokenized_size);
}
BENCHMARK(BM_DecodeCompactBpe);

}  // namespace
}  // namespace gcpp
