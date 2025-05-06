// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "io/fields.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <limits>
#include <type_traits>

#include "hwy/tests/hwy_gtest.h"

namespace gcpp {
namespace {

#if !HWY_TEST_STANDALONE
class FieldsTest : public testing::Test {};
#endif

void MaybePrint(const IFields& fields) {
  if (HWY_IS_DEBUG_BUILD) {
    fields.Print();
  }
}

template <typename T>
void CheckVectorEqual(const std::vector<T>& a, const std::vector<T>& b) {
  EXPECT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    if constexpr (std::is_base_of_v<IFields, T>) {
      a[i].CheckEqual(b[i]);
    } else {
      EXPECT_EQ(a[i], b[i]);
    }
  }
}

enum class Enum : uint32_t {
  k1 = 1,
  k3 = 3,
  k8 = 8,
};
HWY_MAYBE_UNUSED bool EnumValid(Enum e) {
  return e == Enum::k1 || e == Enum::k3 || e == Enum::k8;
}

// Contains all supported types except IFields and std::vector<IFields>.
struct Nested : public IFields {
  Nested() : nested_u32(0) {}  // for std::vector
  explicit Nested(uint32_t u32) : nested_u32(u32) {}

  const char* Name() const override { return "Nested"; }
  void VisitFields(IFieldsVisitor& visitor) override {
    visitor(nested_u32);
    visitor(nested_bool);
    visitor(nested_vector);
    visitor(nested_enum);
    visitor(nested_str);
    visitor(nested_f);
  }

  void CheckEqual(const Nested& n) const {
    EXPECT_EQ(nested_u32, n.nested_u32);
    EXPECT_EQ(nested_bool, n.nested_bool);
    CheckVectorEqual(nested_vector, n.nested_vector);
    EXPECT_EQ(nested_enum, n.nested_enum);
    EXPECT_EQ(nested_str, n.nested_str);
    EXPECT_EQ(nested_f, n.nested_f);
  }

  uint32_t nested_u32;  // set in ctor
  bool nested_bool = true;
  std::vector<uint32_t> nested_vector = {1, 2, 3};
  Enum nested_enum = Enum::k1;
  std::string nested_str = "nested";
  float nested_f = 1.125f;
};

// Contains all supported types.
struct OldFields : public IFields {
  const char* Name() const override { return "OldFields"; }
  void VisitFields(IFieldsVisitor& visitor) override {
    visitor(old_str);
    visitor(old_nested);
    visitor(old1);
    visitor(oldi);
    visitor(oldl);
    visitor(old_vec_str);
    visitor(old_vec_nested);
    visitor(old_f);
    visitor(old_enum);
    visitor(old_bool);
  }

  // Template allows comparing with NewFields.
  template <typename Other>
  void CheckEqual(const Other& n) const {
    EXPECT_EQ(old_str, n.old_str);
    old_nested.CheckEqual(n.old_nested);
    EXPECT_EQ(old1, n.old1);
    EXPECT_EQ(oldi, n.oldi);
    EXPECT_EQ(oldl, n.oldl);
    CheckVectorEqual(old_vec_str, n.old_vec_str);
    CheckVectorEqual(old_vec_nested, n.old_vec_nested);
    EXPECT_EQ(old_f, n.old_f);
    EXPECT_EQ(old_enum, n.old_enum);
    EXPECT_EQ(old_bool, n.old_bool);
  }

  std::string old_str = "old";
  Nested old_nested = Nested(0);
  uint32_t old1 = 1;
  int32_t oldi = -1;
  uint64_t oldl = 1234567890123456789;
  std::vector<std::string> old_vec_str = {"abc", "1234"};
  std::vector<Nested> old_vec_nested = {Nested(1), Nested(4)};
  float old_f = 1.125f;
  Enum old_enum = Enum::k1;
  bool old_bool = true;
};  // OldFields

// Simulates adding new fields of all types to an existing struct.
struct NewFields : public IFields {
  const char* Name() const override { return "NewFields"; }
  void VisitFields(IFieldsVisitor& visitor) override {
    visitor(old_str);
    visitor(old_nested);
    visitor(old1);
    visitor(oldi);
    visitor(oldl);
    visitor(old_vec_str);
    visitor(old_vec_nested);
    visitor(old_f);
    visitor(old_enum);
    visitor(old_bool);

    // Change order of field types relative to OldFields to ensure that works.
    visitor(new_nested);
    visitor(new_bool);
    visitor(new_vec_nested);
    visitor(new_f);
    visitor(new_vec_str);
    visitor(new_enum);
    visitor(new2);
    visitor(new_str);
    visitor(new_i);
    visitor(new_l);
  }

  void CheckEqual(const NewFields& n) const {
    EXPECT_EQ(old_str, n.old_str);
    old_nested.CheckEqual(n.old_nested);
    EXPECT_EQ(old1, n.old1);
    CheckVectorEqual(old_vec_str, n.old_vec_str);
    CheckVectorEqual(old_vec_nested, n.old_vec_nested);
    EXPECT_EQ(old_f, n.old_f);
    EXPECT_EQ(old_enum, n.old_enum);
    EXPECT_EQ(old_bool, n.old_bool);

    new_nested.CheckEqual(n.new_nested);
    EXPECT_EQ(new_bool, n.new_bool);
    CheckVectorEqual(new_vec_nested, n.new_vec_nested);
    EXPECT_EQ(new_f, n.new_f);
    CheckVectorEqual(new_vec_str, n.new_vec_str);
    EXPECT_EQ(new_enum, n.new_enum);
    EXPECT_EQ(new2, n.new2);
    EXPECT_EQ(new_str, n.new_str);
  }

  // Copied from OldFields to match the use case of adding new fields. If we
  // write an OldFields member, that would change the layout due to its size.
  std::string old_str = "old";
  Nested old_nested = Nested(0);
  uint32_t old1 = 1;
  int32_t oldi = -1;
  uint64_t oldl = 1234567890123456789;
  std::vector<std::string> old_vec_str = {"abc", "1234"};
  std::vector<Nested> old_vec_nested = {Nested(1), Nested(4)};
  float old_f = 1.125f;
  Enum old_enum = Enum::k1;
  bool old_bool = true;

  Nested new_nested = Nested(999);
  bool new_bool = false;
  std::vector<Nested> new_vec_nested = {Nested(2), Nested(3)};
  float new_f = -2.0f;
  std::vector<std::string> new_vec_str = {"AB", std::string(), "56789"};
  Enum new_enum = Enum::k3;
  uint32_t new2 = 2;
  std::string new_str = std::string();  // empty is allowed
  int32_t new_i = 123456789;
  uint64_t new_l = 876543210987654321;
};  // NewFields

// Changes all fields to non-default values.
NewFields ModifiedNewFields() {
  NewFields n;
  n.old_str = "old2";
  n.old_nested = Nested(5);
  n.old1 = 11;
  n.old_vec_str = {"abc2", "431", "ZZ"};
  n.old_vec_nested = {Nested(9)};
  n.old_f = -2.5f;
  n.old_enum = Enum::k3;
  n.old_bool = false;

  n.new_nested = Nested(55);
  n.new_bool = true;
  n.new_vec_nested = {Nested(3), Nested(33), Nested(333)};
  n.new_f = 4.f;
  n.new_vec_str = {"4321", "321", "21", "1"};
  n.new_enum = Enum::k8;
  n.new2 = 22;
  n.new_str = "new and even longer";
  n.new_i = 246810121;
  n.new_l = 1357913579113579135;

  return n;
}

using Span = hwy::Span<const uint32_t>;

using ReadResult = IFields::ReadResult;
void CheckConsumedAll(const ReadResult& result, size_t storage_size) {
  EXPECT_NE(0, storage_size);  // Ensure we notice failure (pos == 0).
  EXPECT_EQ(storage_size, result.pos);
  EXPECT_EQ(0, result.missing_fields);
  EXPECT_EQ(0, result.extra_u32);
}

// If we do not change any fields, Write+Read returns the defaults.
TEST(FieldsTest, TestNewMatchesDefaults) {
  NewFields new_fields;
  const std::vector<uint32_t> storage = new_fields.Write();

  const ReadResult result = new_fields.Read(Span(storage), 0);
  CheckConsumedAll(result, storage.size());

  NewFields().CheckEqual(new_fields);
}

// Change fields from default and check that Write+Read returns them again.
TEST(FieldsTest, TestRoundTrip) {
  NewFields new_fields = ModifiedNewFields();
  const std::vector<uint32_t> storage = new_fields.Write();

  NewFields copy;
  const ReadResult result = copy.Read(Span(storage), 0);
  CheckConsumedAll(result, storage.size());

  new_fields.CheckEqual(copy);
}

// Refuse to write invalid floats.
TEST(FieldsTest, TestInvalidFloat) {
  NewFields new_fields;
  new_fields.new_f = std::numeric_limits<float>::infinity();
  EXPECT_TRUE(new_fields.Write().empty());

  new_fields.new_f = std::numeric_limits<float>::quiet_NaN();
  EXPECT_TRUE(new_fields.Write().empty());
}

// Refuse to write invalid strings.
TEST(FieldsTest, TestInvalidString) {
  NewFields new_fields;
  // Four zero bytes
  new_fields.new_str.assign(4, '\0');
  EXPECT_TRUE(new_fields.Write().empty());

  // Too long
  new_fields.new_str.assign(257, 'a');
  EXPECT_TRUE(new_fields.Write().empty());

  // First byte not ASCII
  new_fields.new_str.assign("123");
  new_fields.new_str[0] = 128;
  EXPECT_TRUE(new_fields.Write().empty());

  // Upper byte in later u32 not ASCII
  new_fields.new_str.assign("ABCDEFGH");
  new_fields.new_str[7] = 255;
  EXPECT_TRUE(new_fields.Write().empty());
}

// Write two structs to the same storage.
TEST(FieldsTest, TestMultipleWrite) {
  const NewFields modified = ModifiedNewFields();
  std::vector<uint32_t> storage = modified.Write();
  const size_t modified_size = storage.size();
  const NewFields defaults;
  defaults.AppendTo(storage);

  // Start with defaults to ensure Read retrieves the modified values.
  NewFields modified_copy;
  const ReadResult result1 = modified_copy.Read(Span(storage), 0);
  CheckConsumedAll(result1, modified_size);
  modified.CheckEqual(modified_copy);

  // Start with modified values to ensure Read retrieves the defaults.
  NewFields defaults_copy = modified;
  const ReadResult result2 = defaults_copy.Read(Span(storage), result1.pos);
  CheckConsumedAll(result2, storage.size());
  defaults.CheckEqual(defaults_copy);
}

// Write old defaults, read old using new code.
TEST(FieldsTest, TestNewCodeOldData) {
  OldFields old_fields;
  const std::vector<uint32_t> storage = old_fields.Write();

  // Start with modified old values to ensure old defaults overwrite them.
  NewFields new_fields = ModifiedNewFields();
  const ReadResult result = new_fields.Read(Span(storage), 0);
  MaybePrint(new_fields);
  EXPECT_NE(0, result.pos);  // did not fail
  EXPECT_NE(0, result.missing_fields);
  EXPECT_EQ(0, result.extra_u32);
  old_fields.CheckEqual(new_fields);  // old fields are the same in both
}

// Write old defaults, ensure new defaults remain unchanged.
TEST(FieldsTest, TestNewCodeOldDataNewUnchanged) {
  OldFields old_fields;
  const std::vector<uint32_t> storage = old_fields.Write();

  NewFields new_fields;
  const ReadResult result = new_fields.Read(Span(storage), 0);
  MaybePrint(new_fields);
  EXPECT_NE(0, result.pos);  // did not fail
  EXPECT_NE(0, result.missing_fields);
  EXPECT_EQ(0, result.extra_u32);
  NewFields().CheckEqual(new_fields);  // new fields match their defaults
}

// Write new defaults, read using old code.
TEST(FieldsTest, TestOldCodeNewData) {
  NewFields new_fields;
  const std::vector<uint32_t> storage = new_fields.Write();

  OldFields old_fields;
  const ReadResult result = old_fields.Read(Span(storage), 0);
  MaybePrint(old_fields);
  EXPECT_NE(0, result.pos);  // did not fail
  EXPECT_EQ(0, result.missing_fields);
  EXPECT_NE(0, result.extra_u32);
  EXPECT_EQ(storage.size(), result.pos + result.extra_u32);

  old_fields.CheckEqual(new_fields);  // old fields are the same in both
  // (Can't check new fields because we only read OldFields)
}

}  // namespace
}  // namespace gcpp

HWY_TEST_MAIN();
