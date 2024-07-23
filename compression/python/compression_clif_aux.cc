#include "compression/python/compression_clif_aux.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE \
  "compression/python/compression_clif_aux.cc"  // NOLINT
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// Must come after foreach_target.h to avoid redefinition errors.
#include "compression/compress-inl.h"
#include "hwy/highway.h"

// Non-SIMD includes and types. Note that HWY_ONCE is only true on the last
// compile pass, whereas we want this defined in the first.
#ifndef GEMMA_ONCE
#define GEMMA_ONCE

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
 public:
  SbsWriterImpl() : pool_(0), compressor_(pool_) {}

  void Insert(std::string name, absl::Span<const float> weights) override {
    const size_t out_size = CompressedArraySize<SfpStream>(weights.size());
    sfp_streams_.push_back(std::vector<SfpStream>(out_size));
    compressor_.Insert<SfpStream>(name.data(), weights.data(), weights.size(),
                                  working_set_, out_size,
                                  sfp_streams_.back().data(), 0, pool_);
  }

  void InsertNUQ(std::string name, absl::Span<const float> weights) override {
    const size_t out_size = CompressedArraySize<NuqStream>(weights.size());
    nuq_streams_.push_back(std::vector<NuqStream>(out_size));
    compressor_.Insert<NuqStream>(name.data(), weights.data(), weights.size(),
                                  working_set_, out_size,
                                  nuq_streams_.back().data(), 0, pool_);
  }

  void InsertBfloat16(std::string name,
                      absl::Span<const float> weights) override {
    const size_t out_size =
        CompressedArraySize<hwy::bfloat16_t>(weights.size());
    bf16_streams_.push_back(std::vector<hwy::bfloat16_t>(out_size));
    compressor_.Insert<hwy::bfloat16_t>(name.data(), weights.data(),
                                        weights.size(), working_set_, out_size,
                                        bf16_streams_.back().data(), 0, pool_);
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
  std::vector<std::vector<SfpStream>> sfp_streams_;
  std::vector<std::vector<NuqStream>> nuq_streams_;
  std::vector<std::vector<hwy::bfloat16_t>> bf16_streams_;
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

void SbsWriter::AddScales(const std::vector<float>& scales) {
  impl_->AddScales(scales);
}
void SbsWriter::Write(std::string path) { impl_->Write(path); }

}  // namespace gcpp
#endif  // HWY_ONCE
