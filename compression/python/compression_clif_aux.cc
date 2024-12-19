#include "compression/python/compression_clif_aux.h"

#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "compression/compress.h"
#include "compression/shared.h"
#include "hwy/aligned_allocator.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE \
  "compression/python/compression_clif_aux.cc"  // NOLINT
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"

// Non-SIMD includes and types. Note that HWY_ONCE is only true on the last
// compile pass, whereas we want this defined in the first.
#ifndef GEMMA_ONCE
#define GEMMA_ONCE

#include "absl/types/span.h"
#include "compression/io.h"
#include "gemma/configs.h"
#include "gemma/tensor_index.h"
#include "gemma/tokenizer.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

class WriterInterface {
 public:
  virtual ~WriterInterface() = default;

  virtual void Insert(std::string name, absl::Span<const float> weights,
                      Type type, const TensorInfo& tensor_info,
                      float scale) = 0;
  virtual void InsertSfp(std::string name, absl::Span<const float> weights) = 0;
  virtual void InsertNUQ(std::string name, absl::Span<const float> weights) = 0;
  virtual void InsertBfloat16(std::string name,
                              absl::Span<const float> weights) = 0;
  virtual void InsertFloat(std::string name,
                           absl::Span<const float> weights) = 0;
  virtual void AddScales(const std::vector<float>& scales) = 0;
  virtual void AddTokenizer(const std::string& tokenizer_path) = 0;

  virtual size_t DebugNumBlobsAdded() const = 0;

  virtual int WriteWithConfig(std::string path, const ModelConfig* config) = 0;
};

}  // namespace gcpp

#endif  // GEMMA_ONCE

// SIMD code, compiled once per target.
HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

class SbsWriterImpl : public WriterInterface {
  template <typename Packed>
  void AllocateAndCompress(const std::string& name,
                           absl::Span<const float> weights) {
    const size_t num_packed = CompressedArrayElements<Packed>(weights.size());
    MatPtrT<Packed> storage(name, 1, num_packed);
    model_memory_.push_back(storage);
    model_memory_.back().Allocate();
    storage.SetPtr(model_memory_.back());
    std::string decorated_name = storage.CacheName();
    compressor_(&storage, decorated_name.c_str(), weights.data());
  }
  template <typename Packed>
  void AllocateWithShape(const std::string& name,
                         absl::Span<const float> weights,
                         const TensorInfo& tensor_info, float scale) {
    MatPtrT<Packed> storage(name, &tensor_info);
    storage.set_scale(scale);
    storage.SetNumElements(CompressedArrayElements<Packed>(weights.size()));
    model_memory_.push_back(storage);
    if (mode_ == CompressorMode::kTEST_ONLY) return;
    model_memory_.back().Allocate();
    storage.SetPtr(model_memory_.back());
    std::string decorated_name = storage.CacheName();
    compressor_(&storage, decorated_name.c_str(), weights.data());
  }

 public:
  explicit SbsWriterImpl(CompressorMode mode)
      : pool_(0), compressor_(pool_), mode_(mode) {}

  void Insert(std::string name, absl::Span<const float> weights, Type type,
              const TensorInfo& tensor_info, float scale) override {
    switch (type) {
      case Type::kSFP:
        AllocateWithShape<SfpStream>(name, weights, tensor_info, scale);
        break;
      case Type::kNUQ:
        AllocateWithShape<NuqStream>(name, weights, tensor_info, scale);
        break;
      case Type::kBF16:
        AllocateWithShape<BF16>(name, weights, tensor_info, scale);
        break;
      case Type::kF32:
        AllocateWithShape<float>(name, weights, tensor_info, scale);
        break;
      default:
        HWY_ABORT("Unsupported type");
    }
  }

  void InsertSfp(std::string name, absl::Span<const float> weights) override {
    AllocateAndCompress<SfpStream>(name, weights);
  }

  void InsertNUQ(std::string name, absl::Span<const float> weights) override {
    AllocateAndCompress<NuqStream>(name, weights);
  }

  void InsertBfloat16(std::string name,
                      absl::Span<const float> weights) override {
    AllocateAndCompress<BF16>(name, weights);
  }

  void InsertFloat(std::string name, absl::Span<const float> weights) override {
    AllocateAndCompress<float>(name, weights);
  }

  void AddScales(const std::vector<float>& scales) override {
    HWY_ASSERT(scales_.empty());
    scales_ = scales;
    compressor_.AddScales(scales_.data(), scales_.size());
  }

  void AddTokenizer(const std::string& tokenizer_path) override {
    Path path(tokenizer_path);
    GemmaTokenizer tokenizer(path);
    tokenizer_proto_ = tokenizer.Serialize();
    compressor_.AddTokenizer(tokenizer_proto_);
  }

  // Returns the number of blobs added.
  size_t DebugNumBlobsAdded() const {
    if (mode_ == CompressorMode::kTEST_ONLY) return model_memory_.size();
    return compressor_.DebugNumBlobsAdded();
  }

  int WriteWithConfig(std::string path, const ModelConfig* config) override {
    return compressor_.WriteAll(gcpp::Path(path), config);
  }

  hwy::ThreadPool pool_;
  Compressor compressor_;
  CompressWorkingSet working_set_;
  std::vector<MatStorage> model_memory_;
  std::vector<float> scales_;
  CompressorMode mode_;
  std::string tokenizer_proto_;
};

WriterInterface* NewSbsWriter(CompressorMode mode) {
  return new SbsWriterImpl(mode);
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {

HWY_EXPORT(NewSbsWriter);

SbsWriter::SbsWriter(CompressorMode mode)
    : impl_(HWY_DYNAMIC_DISPATCH(NewSbsWriter)(mode)) {}
SbsWriter::~SbsWriter() = default;

void SbsWriter::Insert(std::string name, absl::Span<const float> weights,
                       Type type, const TensorInfo& tensor_info, float scale) {
  impl_->Insert(name, weights, type, tensor_info, scale);
}
void SbsWriter::InsertSfp(std::string name, absl::Span<const float> weights) {
  impl_->InsertSfp(name, weights);
}
void SbsWriter::InsertNUQ(std::string name, absl::Span<const float> weights) {
  impl_->InsertNUQ(name, weights);
}
void SbsWriter::InsertBfloat16(std::string name,
                               absl::Span<const float> weights) {
  impl_->InsertBfloat16(name, weights);
}
void SbsWriter::InsertFloat(std::string name, absl::Span<const float> weights) {
  impl_->InsertFloat(name, weights);
}

void SbsWriter::AddScales(const std::vector<float>& scales) {
  impl_->AddScales(scales);
}

void SbsWriter::AddTokenizer(const std::string& tokenizer_path) {
  impl_->AddTokenizer(tokenizer_path);
}

size_t SbsWriter::DebugNumBlobsAdded() const {
  return impl_->DebugNumBlobsAdded();
}

int SbsWriter::WriteWithConfig(std::string path, const ModelConfig* config) {
  return impl_->WriteWithConfig(path, config);
}

}  // namespace gcpp
#endif  // HWY_ONCE
