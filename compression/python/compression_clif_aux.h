#ifndef THIRD_PARTY_GEMMA_CPP_COMPRESSION_PYTHON_COMPRESSION_CLIF_AUX_H_
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_PYTHON_COMPRESSION_CLIF_AUX_H_

#include <memory>
#include <string>
#include <vector>

#include "third_party/absl/types/span.h"

namespace gcpp {

class WriterInterface;

class SbsWriter {
 public:
  SbsWriter();
  ~SbsWriter();

  void Insert(std::string name, absl::Span<const float> weights);
  void InsertNUQ(std::string name, absl::Span<const float> weights);
  void InsertBfloat16(std::string name, absl::Span<const float> weights);
  void AddScales(const std::vector<float>& scales);

  void Write(std::string path);

 private:
  // Isolates Highway-dispatched types and other internals from CLIF.
  std::unique_ptr<WriterInterface> impl_;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_PYTHON_COMPRESSION_CLIF_AUX_H_
