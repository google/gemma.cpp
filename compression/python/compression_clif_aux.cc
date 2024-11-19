#include "compression/python/compression_clif_aux.h"

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
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

class WriterInterface {
 public:
  virtual ~WriterInterface() = default;

  virtual void Insert(std::string name, absl::Span<const float> weights) = 0;
  virtual void InsertNUQ(std::string name, absl::Span<const float> weights) = 0;
  virtual void InsertBfloat16(std::string name,
                              absl::Span<const float> weights) = 0;
  virtual void InsertFloat(std::string name,
                           absl::Span<const float> weights) = 0;
  virtual void AddScales(const std::vector<float>& scales) = 0;

  virtual void Write(std::string path) = 0;
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

 public:
  SbsWriterImpl() : pool_(0), compressor_(pool_) {}

  void Insert(std::string name, absl::Span<const float> weights) override {
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

  void Write(std::string path) override {
    compressor_.WriteAll(pool_, gcpp::Path(path));
  }

  hwy::ThreadPool pool_;
  Compressor compressor_;
  CompressWorkingSet working_set_;
  std::vector<MatStorage> model_memory_;
  std::vector<float> scales_;
};

WriterInterface* NewSbsWriter() { return new SbsWriterImpl(); }

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {

HWY_EXPORT(NewSbsWriter);

SbsWriter::SbsWriter() : impl_(HWY_DYNAMIC_DISPATCH(NewSbsWriter)()) {}
SbsWriter::~SbsWriter() = default;

void SbsWriter::Insert(std::string name, absl::Span<const float> weights) {
  impl_->Insert(name, weights);
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
void SbsWriter::Write(std::string path) { impl_->Write(path); }

}  // namespace gcpp
#endif  // HWY_ONCE
