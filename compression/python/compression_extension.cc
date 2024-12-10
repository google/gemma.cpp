#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>

#include "absl/types/span.h"
#include "compression/python/compression_clif_aux.h"
#include "compression/shared.h"

using gcpp::SbsWriter;

namespace py = pybind11;

namespace {
template <auto Func>
void wrap_span(SbsWriter& writer, std::string name, py::array_t<float> data) {
  if (data.ndim() != 1 || data.strides(0) != sizeof(float)) {
    throw std::domain_error("Input array must be 1D and densely packed.");
  }
  std::invoke(Func, writer, name, absl::MakeSpan(data.data(0), data.size()));
}
template <auto Func>
void wrap_span_typed(SbsWriter& writer, std::string name,
                     py::array_t<float> data, gcpp::Type type) {
  if (data.ndim() != 1 || data.strides(0) != sizeof(float)) {
    throw std::domain_error("Input array must be 1D and densely packed.");
  }
  std::invoke(Func, writer, name, absl::MakeSpan(data.data(0), data.size()),
              type);
}
}  // namespace

PYBIND11_MODULE(compression, m) {
  py::class_<SbsWriter>(m, "SbsWriter")
      .def(py::init<>())
      // NOTE: Individual compression backends may impose constraints on the
      // array length, such as a minimum of (say) 32 elements.
      .def("insert", wrap_span_typed<&SbsWriter::Insert>)
      .def("insert_sfp", wrap_span<&SbsWriter::InsertSfp>)
      .def("insert_nuq", wrap_span<&SbsWriter::InsertNUQ>)
      .def("insert_bf16", wrap_span<&SbsWriter::InsertBfloat16>)
      .def("insert_float", wrap_span<&SbsWriter::InsertFloat>)
      .def("add_scales", &SbsWriter::AddScales)
      .def("write", &SbsWriter::Write);
}
