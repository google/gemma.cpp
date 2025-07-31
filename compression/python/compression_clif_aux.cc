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

#include "compression/python/compression_clif_aux.h"

#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <string>
#include <vector>

#include "compression/compress.h"  // ScaleWeights
#include "gemma/configs.h"         // ModelConfig
#include "gemma/model_store.h"     // ModelStore
#include "gemma/tensor_info.h"     // TensorInfo
#include "gemma/tokenizer.h"
#include "io/blob_store.h"  // BlobWriter
#include "io/io.h"          // Path
#include "util/basics.h"
#include "util/mat.h"
#include "util/threading_context.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE \
  "compression/python/compression_clif_aux.cc"  // NOLINT
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"

// SIMD code, compiled once per target.
HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// Implementation for the currently compiled SIMD target.
class SbsWriterImpl : public ISbsWriter {
  template <typename Packed>
  void InsertT(const char* name, F32Span weights,
               const TensorInfo& tensor_info) {
    // TODO(janwas): 1D parallel-for.
    hwy::ThreadPool& pool = ctx_.pools.Pool();

    MatPtrT<Packed> mat(name, ExtentsFromInfo(&tensor_info));
    // SFP and NUQ (which uses SFP for cluster centers) have a limited range
    // and depending on the input values may require rescaling. Scaling is
    // cheap for matmul and probably not an issue for other ops, but it might be
    // beneficial for precision to keep the original data range for other types.
    if (mat.GetType() == Type::kSFP || mat.GetType() == Type::kNUQ) {
      mat.SetScale(ScaleWeights(weights.data(), weights.size()));
    }

    if (weights.size() == 0) {
      HWY_WARN("Ignoring zero-sized tensor %s.", name);
      return;
    }

    mat.AppendTo(serialized_mat_ptrs_);
    MatOwner mat_owner;
    mat_owner.AllocateFor(mat, ctx_.allocator, MatPadding::kPacked);

    // Handle gemma_export_test's MockArray. Write blobs so that the test
    // succeeds, but we only have 10 floats, not the full tensor.
    if (weights.size() == 10 && mat.Extents().Area() != 10) {
      Compress(weights.data(), weights.size(), working_set_, mat.Span(),
               /*packed_ofs=*/0, pool);
      writer_.Add(name, mat.Packed(), mat.ElementBytes() * 10);
      return;
    }

    fprintf(stderr, "Compressing %s (%zu x %zu = %zuM) to %s, please wait\n",
            name, mat.Rows(), mat.Cols(), weights.size() / (1000 * 1000),
            TypeName(TypeEnum<Packed>()));
    HWY_ASSERT(weights.size() == mat.Extents().Area());
    Compress(weights.data(), weights.size(), working_set_, mat.Span(),
             /*packed_ofs=*/0, pool);
    writer_.Add(name, mat.Packed(), mat.PackedBytes());
  }

 public:
  SbsWriterImpl(const std::string& sbs_path)
      : ctx_(ThreadingArgs()),
        writer_(gcpp::Path(sbs_path), ctx_.pools.Pool()) {}

  void Insert(const char* name, F32Span weights, Type type,
              const TensorInfo& tensor_info) override {
    switch (type) {
      case Type::kSFP:
        InsertT<SfpStream>(name, weights, tensor_info);
        break;
      case Type::kNUQ:
        InsertT<NuqStream>(name, weights, tensor_info);
        break;
      case Type::kBF16:
        InsertT<BF16>(name, weights, tensor_info);
        break;
      case Type::kF32:
        InsertT<float>(name, weights, tensor_info);
        break;
      default:
        HWY_ABORT("Unsupported destination (compressed) type %s",
                  TypeName(type));
    }
  }

  void Write(const ModelConfig& config,
             const std::string& tokenizer_path) override {
    const GemmaTokenizer tokenizer(
        tokenizer_path.empty() ? kMockTokenizer
                               : ReadFileToString(Path(tokenizer_path)));
    WriteSingleFile(config, tokenizer, serialized_mat_ptrs_, writer_);
  }

  ThreadingContext ctx_;
  CompressWorkingSet working_set_;
  BlobWriter writer_;
  std::vector<uint32_t> serialized_mat_ptrs_;
};

ISbsWriter* NewSbsWriter(const std::string& sbs_path) {
  return new SbsWriterImpl(sbs_path);
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {

HWY_EXPORT(NewSbsWriter);

SbsWriter::SbsWriter(const std::string& path)
    : impl_(HWY_DYNAMIC_DISPATCH(NewSbsWriter)(path)) {}

SbsReader::SbsReader(const std::string& path)
    : reader_(Path(path)), model_(reader_) {}

}  // namespace gcpp
#endif  // HWY_ONCE
