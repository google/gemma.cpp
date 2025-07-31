// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Reads/writes model metadata (all but the weights) from/to a `BlobStore`.
#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_MODEL_STORE_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_MODEL_STORE_H_

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// IWYU pragma: begin_exports
#include "gemma/configs.h"  // ModelConfig
#include "gemma/tokenizer.h"
#include "io/blob_store.h"
#include "io/io.h"        // Path
#include "util/basics.h"  // Tristate
#include "util/mat.h"     // MatPtr
// IWYU pragma: end_exports

#include "util/allocator.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

// Reads and holds the model config, tokenizer and all `MatPtr`: everything
// except the tensor data, which are read/written by `weights.cc`.
//
// As of 2025-04, the `BlobStore` format includes blobs for `ModelConfig`,
// tokenizer, and all `MatPtr` metadata. "Pre-2025" format instead stored the
// tokenizer in a separate file, encoded tensor type in a prefix of the blob
// name, and had a blob for tensor scaling factors. We still support reading
// both, but only write single-file format.
class ModelStore {
 public:
  // Reads from file(s) or aborts on error. The latter two arguments are only
  // used for pre-2025 files.
  ModelStore(BlobReader& reader, const Path& tokenizer_path = Path(),
             Tristate wrapping = Tristate::kDefault);
  ~ModelStore();

  const ModelConfig& Config() const {
    HWY_ASSERT(config_.model != Model::UNKNOWN);
    return config_;
  }

  const GemmaTokenizer& Tokenizer() const { return tokenizer_; }

  // Returns nullptr if `name` is not available for loading, otherwise the
  // metadata of that tensor.
  const MatPtr* FindMat(const char* name) const;

  // Returns false if `mat` is not available for loading, otherwise updates
  // `mat` with metadata from the file and sets `key_idx` for use by
  // `BlobReader`. Called via `ReadOrAllocate` in `weights.cc`.
  bool FindAndUpdateMatPtr(MatPtr& mat, size_t& key_idx) const;

 private:
  void AddMatPtr(const size_t key_idx, const MatPtr& mat) {
    auto pair_ib = mat_idx_for_name_.insert({mat.Name(), mat_ptrs_.size()});
    HWY_ASSERT_M(pair_ib.second, mat.Name());  // Ensure inserted/unique.
    mat_ptrs_.push_back(mat);
    key_idx_.push_back(key_idx);
  }

  bool ReadMatPtrs(BlobReader& reader);
  void CreateMatPtrs(BlobReader& reader);  // Aborts on error.

  ModelConfig config_;
  GemmaTokenizer tokenizer_;

  // All `MatPtr` present in the `BlobStore`, see `ReadMatPtrs`/`CreateMatPtrs`.
  std::vector<MatPtr> mat_ptrs_;
  // For each of `mat_ptrs_`, the index within `BlobReader::Keys()`. This is
  // not necessarily iota because some blobs are not tensors, and callers may
  // have added blobs before ours.
  std::vector<size_t> key_idx_;
  // Index within `mat_ptrs_` and `key_idx_` for each tensor name.
  std::unordered_map<std::string, size_t> mat_idx_for_name_;

  // Only used if `!ReadMatPtrs` (pre-2025 format):
  std::vector<float> scales_;
  std::unordered_set<std::string> scale_base_names_;
  mutable size_t scales_consumed_ = 0;
};

// Adds metadata blobs to `writer` and writes everything to `path`. This
// produces a single BlobStore file holding everything required for inference.
void WriteSingleFile(const ModelConfig& config, const GemmaTokenizer& tokenizer,
                     const std::vector<uint32_t>& serialized_mat_ptrs,
                     BlobWriter& writer);

}  // namespace gcpp
#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_MODEL_STORE_H_
