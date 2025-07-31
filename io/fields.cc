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

#include "io/fields.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "hwy/base.h"

namespace gcpp {

IFieldsVisitor::~IFieldsVisitor() = default;

void IFieldsVisitor::NotifyInvalid(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);

  any_invalid_ = true;
}

class VisitorBase : public IFieldsVisitor {
 public:
  VisitorBase() = default;
  ~VisitorBase() override = default;

  // This is the only call site of IFields::VisitFields.
  void operator()(IFields& fields) override { fields.VisitFields(*this); }

 protected:  // Functions shared between ReadVisitor and WriteVisitor:
  void CheckF32(float value) {
    if (HWY_UNLIKELY(hwy::ScalarIsInf(value) || hwy::ScalarIsNaN(value))) {
      NotifyInvalid("Invalid float %g\n", value);
    }
  }

  // Return bool to avoid having to check AnyInvalid() after calling.
  bool CheckStringLength(uint32_t num_u32) {
    // Disallow long strings for safety, and to prevent them being used for
    // arbitrary data (we also require them to be ASCII).
    if (HWY_UNLIKELY(num_u32 > 64)) {
      NotifyInvalid("String num_u32=%u too large\n", num_u32);
      return false;
    }
    return true;
  }

  bool CheckStringU32(uint32_t u32, uint32_t i, uint32_t num_u32) {
    // Although strings are zero-padded to u32, an entire u32 should not be
    // zero, and upper bits should not be set (ASCII-only).
    if (HWY_UNLIKELY(u32 == 0 || (u32 & 0x80808080))) {
      NotifyInvalid("Invalid characters %x at %u of %u\n", u32, i, num_u32);
      return false;
    }
    return true;
  }
};

class PrintVisitor : public VisitorBase {
 public:
  void operator()(uint32_t& value) override {
    fprintf(stderr, "%sU32 %u\n", indent_.c_str(), value);
  }

  void operator()(int32_t& value) override {
    fprintf(stderr, "%sI32 %d\n", indent_.c_str(), value);
  }

  void operator()(uint64_t& value) override {
    fprintf(stderr, "%sU64 %zu\n", indent_.c_str(), static_cast<size_t>(value));
  }

  void operator()(float& value) override {
    fprintf(stderr, "%sF32 %f\n", indent_.c_str(), value);
  }

  void operator()(std::string& value) override {
    fprintf(stderr, "%sStr %s\n", indent_.c_str(), value.c_str());
  }

  void operator()(IFields& fields) override {
    fprintf(stderr, "%s%s\n", indent_.c_str(), fields.Name());
    indent_ += "  ";

    VisitorBase::operator()(fields);

    HWY_ASSERT(!indent_.empty());
    indent_.resize(indent_.size() - 2);
  }

 private:
  std::string indent_;
};

class ReadVisitor : public VisitorBase {
 public:
  ReadVisitor(const SerializedSpan span, size_t pos)
      : span_(span), result_(pos) {}
  ~ReadVisitor() {
    HWY_ASSERT(end_.empty());  // Bug if push/pop are not balanced.
  }

  // All data is read through this overload.
  void operator()(uint32_t& value) override {
    if (HWY_UNLIKELY(SkipField())) return;

    value = span_[result_.pos++];
  }

  void operator()(int32_t& value) override {
    if (HWY_UNLIKELY(SkipField())) return;

    value = static_cast<int32_t>(span_[result_.pos++]);
  }

  void operator()(uint64_t& value) override {
    if (HWY_UNLIKELY(SkipField())) return;
    uint32_t lower = static_cast<uint32_t>(value);
    operator()(lower);
    uint32_t upper = static_cast<uint32_t>(value >> 32);
    operator()(upper);
    value = lower | (static_cast<uint64_t>(upper) << 32);
  }

  void operator()(float& value) override {
    if (HWY_UNLIKELY(SkipField())) return;

    uint32_t u32 = hwy::BitCastScalar<uint32_t>(value);
    operator()(u32);
    value = hwy::BitCastScalar<float>(u32);
    CheckF32(value);
  }

  void operator()(std::string& value) override {
    if (HWY_UNLIKELY(SkipField())) return;

    uint32_t num_u32;     // not including itself because this..
    operator()(num_u32);  // increments result_.pos for the num_u32 field
    if (HWY_UNLIKELY(!CheckStringLength(num_u32))) return;

    // Ensure we have that much data.
    if (HWY_UNLIKELY(result_.pos + num_u32 > end_.back())) {
      NotifyInvalid("Invalid string: pos %zu + num_u32 %u > end %zu\n",
                    result_.pos, num_u32, span_.size());
      return;
    }

    constexpr size_t k4 = sizeof(uint32_t);
    value.resize(num_u32 * k4);
    for (uint32_t i = 0; i < num_u32; ++i) {
      uint32_t u32;
      operator()(u32);
      (void)CheckStringU32(u32, i, num_u32);
      hwy::CopyBytes(&u32, value.data() + i * k4, k4);
    }

    // Trim 0..3 trailing nulls.
    const size_t pos = value.find_last_not_of('\0');
    if (pos != std::string::npos) {
      value.resize(pos + 1);
    }
  }

  void operator()(IFields& fields) override {
    // Our SkipField requires end_ to be set before reading num_u32, which
    // determines the actual end, so use an upper bound which is tight if this
    // IFields is last one in span_.
    end_.push_back(span_.size());

    if (HWY_UNLIKELY(SkipField())) {
      end_.pop_back();  // undo `push_back` to keep the stack balanced
      return;
    }

    uint32_t num_u32;     // not including itself because this..
    operator()(num_u32);  // increments result_.pos for the num_u32 field

    // Ensure we have that much data and set end_.
    if (HWY_UNLIKELY(result_.pos + num_u32 > span_.size())) {
      NotifyInvalid("Invalid IFields: pos %zu + num_u32 %u > size %zu\n",
                    result_.pos, num_u32, span_.size());
      return;
    }
    end_.back() = result_.pos + num_u32;

    VisitorBase::operator()(fields);

    HWY_ASSERT(!end_.empty() && result_.pos <= end_.back());
    // Count extra, which indicates old code and new data.
    result_.extra_u32 += end_.back() - result_.pos;
    end_.pop_back();
  }

  // Override because ReadVisitor also does bounds checking.
  bool SkipField() override {
    // If invalid, all bets are off and we don't count missing fields.
    if (HWY_UNLIKELY(AnyInvalid())) return true;

    // Reaching the end of the stored size, or the span, is not invalid -
    // it happens when we read old data with new code.
    if (HWY_UNLIKELY(result_.pos >= end_.back())) {
      result_.missing_fields++;
      return true;
    }

    return false;
  }

  // Override so that operator()(std::vector<T>&) resizes the vector.
  bool IsReading() const override { return true; }

  IFields::ReadResult Result() {
    if (HWY_UNLIKELY(AnyInvalid())) result_.pos = 0;
    return result_;
  }

 private:
  const SerializedSpan span_;
  IFields::ReadResult result_;
  // Stack of end positions of nested IFields. Updated in operator()(IFields&),
  // but read in SkipField.
  std::vector<uint32_t> end_;
};

class WriteVisitor : public VisitorBase {
 public:
  WriteVisitor(std::vector<uint32_t>& storage) : storage_(storage) {}

  // Note: while writing, only string lengths/characters can trigger AnyInvalid,
  // so we don't have to check SkipField.

  void operator()(uint32_t& value) override { storage_.push_back(value); }

  void operator()(int32_t& value) override {
    storage_.push_back(static_cast<uint32_t>(value));
  }

  void operator()(uint64_t& value) override {
    storage_.push_back(static_cast<uint32_t>(value));
    storage_.push_back(static_cast<uint32_t>(value >> 32));
  }

  void operator()(float& value) override {
    storage_.push_back(hwy::BitCastScalar<uint32_t>(value));
    CheckF32(value);
  }

  void operator()(std::string& value) override {
    constexpr size_t k4 = sizeof(uint32_t);

    // Write length.
    uint32_t num_u32 = hwy::DivCeil(value.size(), k4);
    if (HWY_UNLIKELY(!CheckStringLength(num_u32))) return;
    operator()(num_u32);  // always valid

    // Copy whole uint32_t.
    const size_t num_whole_u32 = value.size() / k4;
    for (uint32_t i = 0; i < num_whole_u32; ++i) {
      uint32_t u32 = 0;
      hwy::CopyBytes(value.data() + i * k4, &u32, k4);
      if (HWY_UNLIKELY(!CheckStringU32(u32, i, num_u32))) return;
      storage_.push_back(u32);
    }

    // Read remaining bytes into least-significant bits of u32.
    const size_t remainder = value.size() - num_whole_u32 * k4;
    if (remainder != 0) {
      HWY_DASSERT(remainder < k4);
      uint32_t u32 = 0;
      for (size_t i = 0; i < remainder; ++i) {
        const char c = value[num_whole_u32 * k4 + i];
        const uint32_t next = static_cast<uint32_t>(static_cast<uint8_t>(c));
        u32 += next << (i * 8);
      }
      if (HWY_UNLIKELY(!CheckStringU32(u32, num_whole_u32, num_u32))) return;
      storage_.push_back(u32);
    }
  }

  void operator()(IFields& fields) override {
    const size_t pos_before_size = storage_.size();
    storage_.push_back(0);  // placeholder, updated below

    VisitorBase::operator()(fields);

    HWY_ASSERT(storage_[pos_before_size] == 0);
    // Number of u32 written, including the one storing that number.
    const uint32_t num_u32 = storage_.size() - pos_before_size;
    HWY_ASSERT(num_u32 != 0);  // at least one due to push_back above
    // Store the payload size, not including the num_u32 field itself, because
    // that is more convenient for ReadVisitor.
    storage_[pos_before_size] = num_u32 - 1;
  }

 private:
  std::vector<uint32_t>& storage_;
};

IFields::~IFields() = default;

void IFields::Print() const {
  PrintVisitor visitor;
  // VisitFields is non-const. It is safe to cast because PrintVisitor does not
  // modify the fields.
  visitor(*const_cast<IFields*>(this));
}

IFields::ReadResult IFields::Read(const SerializedSpan span, size_t pos) {
  ReadVisitor visitor(span, pos);
  visitor(*this);
  return visitor.Result();
}

bool IFields::AppendTo(std::vector<uint32_t>& storage) const {
  // VisitFields is non-const. It is safe to cast because WriteVisitor does not
  // modify the fields.
  IFields& fields = *const_cast<IFields*>(this);

  // Reduce allocations, but not in debug builds so we notice any iterator
  // invalidation bugs.
  if constexpr (!HWY_IS_DEBUG_BUILD) {
    storage.reserve(storage.size() + 256);
  }

  WriteVisitor visitor(storage);
  visitor(fields);
  return !visitor.AnyInvalid();
}

}  // namespace gcpp
