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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>

#include "absl/types/span.h"
#include "compression/python/compression_clif_aux.h"
#include "compression/shared.h"

using gcpp::CompressorMode;
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
                     py::array_t<float> data, gcpp::Type type,
                     gcpp::TensorInfo tensor_info, float scale) {
  if (data.ndim() != 1 || data.strides(0) != sizeof(float)) {
    throw std::domain_error("Input array must be 1D and densely packed.");
  }
  std::invoke(Func, writer, name, absl::MakeSpan(data.data(0), data.size()),
              type, tensor_info, scale);
}
}  // namespace

PYBIND11_MODULE(compression, m) {
  py::enum_<CompressorMode>(m, "CompressorMode")
      .value("TEST_ONLY", CompressorMode::kTEST_ONLY)
      .value("NO_TOC", CompressorMode::kNO_TOC)
      .value("WITH_TOC", CompressorMode::kWITH_TOC);

  py::class_<SbsWriter>(m, "SbsWriter")
      .def(py::init<CompressorMode>())
      // NOTE: Individual compression backends may impose constraints on the
      // array length, such as a minimum of (say) 32 elements.
      .def("insert", wrap_span_typed<&SbsWriter::Insert>)
      .def("insert_sfp", wrap_span<&SbsWriter::InsertSfp>)
      .def("insert_nuq", wrap_span<&SbsWriter::InsertNUQ>)
      .def("insert_bf16", wrap_span<&SbsWriter::InsertBfloat16>)
      .def("insert_float", wrap_span<&SbsWriter::InsertFloat>)
      .def("add_scales", &SbsWriter::AddScales)
      .def("add_tokenizer", &SbsWriter::AddTokenizer)
      .def("debug_num_blobs_added", &SbsWriter::DebugNumBlobsAdded)
      .def("write", &SbsWriter::Write)
      .def("write_with_config", &SbsWriter::WriteWithConfig);
}
