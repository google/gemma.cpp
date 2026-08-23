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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_WEIGHTS_INTERNAL_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_WEIGHTS_INTERNAL_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "compression/types.h"
#include "gemma/gemma_args.h"
#include "gemma/weights.h"
#include "io/blob_store.h"
#include "util/allocator.h"
#include "util/mat.h"
#include "util/threading_context.h"

namespace gcpp {
namespace weights_internal {

// Describes one tensor whose file bytes will be mapped, read, or converted.
// Kept in this internal header so loader behavior can be tested without a full
// model-sized weights file.
struct TensorToRead {
  MatPtr* mat;
  BlobRange range;
  MatPadding padding;

  // Only for kReadBF16.
  bool keep_type = false;
  // Convert to Type::kSFP while loading.
  bool to_sfp = false;
  Type prev_type;
  size_t prev_packed_bytes = 0;
};

WeightsPtrs::Mode ChooseMode(uint64_t file_bytes, const LoaderArgs& loader,
                             const InferenceArgs& inference,
                             const Allocator& allocator);

void ReadAllToSFP(const std::vector<TensorToRead>& tensors,
                  const BlobReader& reader, ThreadingContext& ctx);

}  // namespace weights_internal
}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_WEIGHTS_INTERNAL_H_
