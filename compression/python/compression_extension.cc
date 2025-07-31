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

#include <string>

#include "compression/python/compression_clif_aux.h"
#include "compression/types.h"  // Type
#include "gemma/tensor_info.h"
#include "util/mat.h"

using gcpp::MatPtr;
using gcpp::SbsReader;
using gcpp::SbsWriter;

namespace pybind11 {

template <auto Func>
static void CallWithF32Span(SbsWriter& writer, const char* name,
                            array_t<float> data, gcpp::Type type,
                            const gcpp::TensorInfo& tensor_info) {
  if (data.ndim() != 1 || data.strides(0) != sizeof(float)) {
    HWY_ABORT("Input array must be 1D (not %d) and contiguous floats.",
              static_cast<int>(data.ndim()));
  }
  std::invoke(Func, writer, name,
              gcpp::F32Span(data.mutable_data(0), data.size()), type,
              tensor_info);
}

PYBIND11_MODULE(compression, m) {
  class_<SbsWriter>(m, "SbsWriter")
      .def(init<std::string>())
      .def("insert", CallWithF32Span<&SbsWriter::Insert>)
      .def("write", &SbsWriter::Write, arg("config"), arg("tokenizer_path"));

  class_<MatPtr>(m, "MatPtr")
      // No init, only created within C++.
      .def_property_readonly("rows", &MatPtr::Rows, "Number of rows")
      .def_property_readonly("cols", &MatPtr::Cols, "Number of cols")
      .def_property_readonly("type", &MatPtr::GetType, "Element type")
      .def_property_readonly("scale", &MatPtr::Scale, "Scaling factor");

  class_<SbsReader>(m, "SbsReader")
      .def(init<std::string>())
      .def_property_readonly("config", &SbsReader::Config,
                             return_value_policy::reference_internal,
                             "ModelConfig")
      .def("find_mat", &SbsReader::FindMat,
           return_value_policy::reference_internal,
           "Returns MatPtr for given name.");
}

}  // namespace pybind11
