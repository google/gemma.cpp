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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_QUERY_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_QUERY_H_

#include <vector>

#include "gemma/gemma_args.h"
#include "gemma/kv_cache.h"
#include "util/basics.h"
#include "hwy/aligned_allocator.h"  // Span
#include "hwy/base.h"

namespace gcpp {

struct PerQuery {
  PromptTokens prompt;

  // Position in the KV cache: initially zero for the first turn, or when
  // multi-turn is NOT desired. Incremented by prefill and `StreamAndUpdateEOS`.
  size_t mutable_pos;
  // Allows computing the last prefill token as `mutable_pos - initial_pos`,
  // which might differ from `prompt.size() - 1` for prefix-LM.
  size_t initial_pos;
  // Zero for causal attention, or the end of the prefix for prefix-LM style
  // attention in Paligemma.
  size_t prefix_end;

  KVCachePtr kv_cache;

  // Previous token generated for this query, or the last prompt token. Will be
  // fed into the next Transformer() call.
  int prev_token = 0;
};

// Array of `PerQuery`. Referenced by `QBatch` and passed to `GenerateBatch`.
struct AllQueries {
  AllQueries() = default;

  // For `GenerateSingleT`: same prompt/pos, replicated for each KV cache.
  AllQueries(const PromptTokens& prompt, size_t pos, size_t prefix_end,
             const hwy::Span<KVCachePtr>& kv_caches) {
    per_query_.reserve(kv_caches.size());
    for (size_t i = 0; i < kv_caches.size(); ++i) {
      HWY_ASSERT(kv_caches[i].SeqLen() == kv_caches[0].SeqLen());
      per_query_.push_back(PerQuery{
          .prompt = prompt,
          .mutable_pos = pos,
          .initial_pos = pos,
          .prefix_end = prefix_end,
          .kv_cache = kv_caches[i],
      });
    }
  }

  AllQueries(const PromptTokens& prompt, size_t pos, size_t prefix_end,
             const hwy::Span<KVCache>& kv_caches)
      : AllQueries(prompt, pos, prefix_end,
                   hwy::Span<KVCachePtr>(ToKVCachePtrs(kv_caches))) {}

  // Batch of queries with initial position set to zero. Causal attention
  // is requested via empty or all-zero `prefix_end`.
  AllQueries(
      const hwy::Span<const PromptTokens>& prompts,
      const hwy::Span<KVCachePtr>& kv_caches,
      const hwy::Span<const size_t>& prefix_end = hwy::Span<const size_t>()) {
    HWY_ASSERT(prompts.size() == prefix_end.size() || prefix_end.size() == 0);
    per_query_.reserve(prompts.size());
    for (size_t i = 0; i < prompts.size(); ++i) {
      HWY_ASSERT(kv_caches.size() == 0 ||
                 kv_caches[i].SeqLen() == kv_caches[0].SeqLen());
      per_query_.push_back(PerQuery{
          .prompt = prompts[i],
          .mutable_pos = 0,
          .initial_pos = 0,
          .prefix_end = prefix_end.size() == 0 ? 0 : prefix_end[i],
          .kv_cache = kv_caches.size() == 0 ? KVCachePtr() : kv_caches[i],
      });
    }
  }

  AllQueries(
      const hwy::Span<const PromptTokens>& prompts,
      const hwy::Span<KVCache>& kv_caches,
      const hwy::Span<const size_t>& prefix_end = hwy::Span<const size_t>())
      : AllQueries(prompts, hwy::Span<KVCachePtr>(ToKVCachePtrs(kv_caches)),
                   prefix_end) {}

  void Reserve(size_t size) { per_query_.reserve(size); }
  void Append(const PerQuery& query) { per_query_.push_back(query); }

  size_t NumQueries() const { return per_query_.size(); }

  PerQuery& operator[](size_t query_idx) {
    HWY_DASSERT(query_idx < NumQueries());
    return per_query_[query_idx];
  }
  const PerQuery& operator[](size_t query_idx) const {
    HWY_DASSERT(query_idx < NumQueries());
    return per_query_[query_idx];
  }

 private:
  std::vector<PerQuery> per_query_;
};

// View into AllQueries: either a batch of queries, or a single query for use
// in PrefillTBatch or GenerateSingleT. Cheap to create because it holds a
// reference to AllQueries.
class QBatch {
 public:
  QBatch(size_t start, size_t max_size, AllQueries& queries)
      : start_(start),
        max_size_(max_size),
        queries_(queries),
        size_(HWY_MIN(max_size_, queries_.NumQueries() - start_)) {
    HWY_ASSERT(max_size_ <= kMaxBatchSize);
    HWY_DASSERT(size_ != 0);
    HWY_DASSERT(start_ + size_ <= queries_.NumQueries());
    query_idx_.reserve(size_);
    for (size_t i = 0; i < size_; ++i) {
      query_idx_.push_back(start_ + i);
    }
  }

  // Returns a single-query view starting at `qi` relative to this batch.
  QBatch Single(size_t qi) const { return QBatch(QueryIdx(qi), 1, queries_); }

  // How many queries in this batch, <= `queries_.NumQueries()` and `max_size_`.
  size_t Size() const { return size_; }

  // Returns index for use with `AllQueries` and `BatchStreamToken`.
  size_t QueryIdx(size_t qi) const {
    HWY_DASSERT(qi < size_);
    return query_idx_[qi];
  }

  // Accessor functions to bridge the previous SoA and current AoS layout.
  const PromptTokens& Prompt(size_t qi) const {
    return queries_[QueryIdx(qi)].prompt;
  }
  size_t Pos(size_t qi) const { return queries_[QueryIdx(qi)].mutable_pos; }
  size_t& MutablePos(size_t qi) { return queries_[QueryIdx(qi)].mutable_pos; }
  size_t InitialPos(size_t qi) const {
    return queries_[QueryIdx(qi)].initial_pos;
  }
  size_t PrefixEnd(size_t qi) const {
    return queries_[QueryIdx(qi)].prefix_end;
  }
  KVCachePtr& KV(size_t qi) const { return queries_[QueryIdx(qi)].kv_cache; }
  int& PrevToken(size_t qi) { return queries_[QueryIdx(qi)].prev_token; }

  // let query_idx_[to] point to the from in the queries_; this is only used if
  // the slot in the QBatch is less than the number of queries.
  void Insert(size_t from, size_t to) {
    if (from == to) return;
    HWY_ASSERT(!queries_[from].kv_cache.IsEmpty());
    HWY_ASSERT(queries_[to].kv_cache.IsEmpty());
    // Conceptually, insert from.query to location to.
    query_idx_[to] = from;
  }

 protected:
  size_t start_;
  size_t max_size_;
  AllQueries& queries_;
  std::vector<size_t> query_idx_;
  size_t size_;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_QUERY_H_
