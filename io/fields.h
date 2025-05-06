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

#ifndef THIRD_PARTY_GEMMA_CPP_COMPRESSION_FIELDS_H_
#define THIRD_PARTY_GEMMA_CPP_COMPRESSION_FIELDS_H_

// Simple serialization/deserialization for user-defined classes, inspired by
// BSD-licensed code Copyright (c) the JPEG XL Project Authors:
// https://github.com/libjxl/libjxl, lib/jxl/fields.h.

// IWYU pragma: begin_exports
#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "hwy/aligned_allocator.h"  // Span
#include "hwy/base.h"
// IWYU pragma: end_exports

namespace gcpp {

// Design goals:
// - self-contained to simplify installing/building, no separate compiler (rules
//   out Protocol Buffers, FlatBuffers, Cap'n Proto, Apache Thrift).
// - simplicity: small codebase without JIT (rules out Apache Fury and bitsery).
// - old code can read new data, and new code can read old data (rules out yas
//   and msgpack). This avoids rewriting weights when we add a new field.
// - no user-specified versions (rules out cereal) nor field names (rules out
//   JSON and GGUF). These are error-prone; users should just be able to append
//   new fields.
//
// Non-goals:
// - anything better than reasonable encoding size and decode speed: we only
//   anticipate ~KiB of data, alongside ~GiB of separately compressed weights.
// - features such as maps, interfaces, random access, and optional/deleted
//   fields: not required for the intended use case of `ModelConfig`.
// - support any other languages than C++ and Python (for the exporter).

struct IFields;  // breaks circular dependency

// Visitors are internal-only, but their base class is visible to user code
// because their `IFields::VisitFields` calls `visitor.operator()`.
//
// Supported field types `T`: `uint32_t`, `int32_t`, `uint64_t`, `float`,
// `std::string`, `IFields` subclasses, `bool`, `enum`, `std::vector<T>`.
class IFieldsVisitor {
 public:
  virtual ~IFieldsVisitor();

  // Indicates whether NotifyInvalid was called for any field. Once set, this is
  // sticky for all IFields visited by this visitor.
  bool AnyInvalid() const { return any_invalid_; }

  // None of these fail directly, but they call NotifyInvalid() if any value
  // is out of range. A single generic/overloaded function is required to
  // support `std::vector<T>`.
  virtual void operator()(uint32_t& value) = 0;
  virtual void operator()(int32_t& value) = 0;
  virtual void operator()(uint64_t& value) = 0;
  virtual void operator()(float& value) = 0;
  virtual void operator()(std::string& value) = 0;
  virtual void operator()(IFields& fields) = 0;  // recurse into nested fields

  // bool and enum fields are actually stored as uint32_t.
  void operator()(bool& value) {
    if (HWY_UNLIKELY(SkipField())) return;

    uint32_t u32 = value ? 1 : 0;
    operator()(u32);
    if (HWY_UNLIKELY(u32 > 1)) {
      return NotifyInvalid("Invalid bool %u\n", u32);
    }
    value = (u32 == 1);
  }

  template <typename EnumT, hwy::EnableIf<std::is_enum_v<EnumT>>* = nullptr>
  void operator()(EnumT& value) {
    if (HWY_UNLIKELY(SkipField())) return;

    uint32_t u32 = static_cast<uint32_t>(value);
    operator()(u32);
    if (HWY_UNLIKELY(!EnumValid(static_cast<EnumT>(u32)))) {
      return NotifyInvalid("Invalid enum %u\n", u32);
    }
    value = static_cast<EnumT>(u32);
  }

  template <typename T>
  void operator()(std::vector<T>& value) {
    if (HWY_UNLIKELY(SkipField())) return;

    uint32_t num = static_cast<uint32_t>(value.size());
    operator()(num);
    if (HWY_UNLIKELY(num > 64 * 1024)) {
      return NotifyInvalid("Vector too long %u\n", num);
    }

    if (IsReading()) {
      value.resize(num);
    }
    for (size_t i = 0; i < value.size(); ++i) {
      operator()(value[i]);
    }
  }

 protected:
  // Prints a message and causes subsequent AnyInvalid() to return true.
  void NotifyInvalid(const char* fmt, ...);

  // Must check this before modifying any field, and if it returns true,
  // avoid doing so. This is important for strings and vectors in the
  // "new code, old data" test: resizing them may destroy their contents.
  virtual bool SkipField() { return AnyInvalid(); }
  // For operator()(std::vector&).
  virtual bool IsReading() const { return false; }

 private:
  bool any_invalid_ = false;
};

using SerializedSpan = hwy::Span<const uint32_t>;

// Abstract base class for user-defined serializable classes, which are
// forward- and backward compatible collection of fields (members). This means
// old code can safely read new data, and new code can still handle old data.
//
// Fields are written in the unchanging order established by the user-defined
// `VisitFields`; any new fields must be visited after all existing fields in
// the same `IFields`. We encode each into `uint32_t` storage for simplicity.
//
// HOWTO:
// - basic usage: define a struct with member variables ("fields") and their
//   initializers, e.g. `uint32_t field = 0;`. Then define a
//   `void VisitFields(IFieldsVisitor& v)` member function that calls
//   `v(field);` etc. for each field, and a `const char* Name()` function used
//   as a caption when printing.
//
// - enum fields: define `enum class EnumT` and `bool EnumValid(EnumT)`, then
//   call `v(field);` as usual. Note that `EnumT` is not extendable insofar as
//   `EnumValid` returns false for values beyond the initially known ones. You
//   can add placeholders, which requires user code to know how to handle them,
//   or later add new fields including enums to override the first enum.
struct IFields {
  virtual ~IFields();

  // User-defined caption used during Print().
  virtual const char* Name() const = 0;

  // User-defined, called by IFieldsVisitor::operator()(IFields&).
  virtual void VisitFields(IFieldsVisitor& visitor) = 0;

  // Prints name and fields to stderr.
  void Print() const;

  struct ReadResult {
    ReadResult(size_t pos) : pos(pos), missing_fields(0), extra_u32(0) {}

    // Where to resume reading in the next Read() call, or 0 if there was an
    // unrecoverable error: any field has an invalid value, or the span is
    // shorter than the data says it should be. If so, do not use the fields nor
    // continue reading.
    size_t pos;
    // From the perspective of VisitFields, how many more fields would have
    // been read beyond the stored size. If non-zero, the data is older than
    // the code, but valid, and extra_u32 should be zero.
    uint32_t missing_fields;
    // How many extra u32 are in the stored size, vs. what we actually read as
    // requested by VisitFields. If non-zero, the data is newer than the code,
    // but valid, and missing_fields should be zero.
    uint32_t extra_u32;
  };

  // Reads fields starting at `span[pos]`.
  ReadResult Read(SerializedSpan span, size_t pos);

  // Returns false if there was an unrecoverable error, typically because a
  // field has an invalid value. If so, `storage` is undefined.
  bool AppendTo(std::vector<uint32_t>& storage) const;

  // Convenience wrapper for AppendTo when we only write once.
  std::vector<uint32_t> Write() const {
    std::vector<uint32_t> storage;
    if (!AppendTo(storage)) storage.clear();
    return storage;
  }
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_COMPRESSION_FIELDS_H_
