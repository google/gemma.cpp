#include "gemma/gemma_args.h"

#include <stddef.h>

#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace gcpp {

void FillPtrs(const std::vector<std::string>& args, std::vector<char*>& ptrs) {
  ptrs.reserve(args.size());
  for (const std::string& arg : args) {
    ptrs.push_back(const_cast<char*>(arg.data()));
  }
}

static void CheckAllConsumed(const std::vector<std::string>& args) {
  std::vector<char*> ptrs;
  FillPtrs(args, ptrs);
  const int argc = static_cast<int>(args.size());
  char** argv = const_cast<char**>(ptrs.data());

  ConsumedArgs consumed(argc, argv);
  GemmaArgs gemma_args(argc, argv, consumed);
  consumed.AbortIfUnconsumed();
}

static void CheckUnconsumed(const std::vector<std::string>& args,
                            size_t expected) {
  std::vector<char*> ptrs;
  FillPtrs(args, ptrs);
  const int argc = static_cast<int>(args.size());
  char** argv = const_cast<char**>(ptrs.data());

  ConsumedArgs consumed(argc, argv);
  GemmaArgs gemma_args(argc, argv, consumed);
  ASSERT_EQ(expected, consumed.FirstUnconsumed());
}

// Note: do not use --help because that is not actually consumed; it is actually
// special-cased in `HasHelp`.
TEST(GemmaArgsTest, AllConsumedArgs) {
  // Single arg
  CheckAllConsumed({"gemma", "--weights=x"});
  // Two args, one with =
  CheckAllConsumed({"gemma", "--weights=x", "--verbosity=1"});
  // Two args, one with extra value
  CheckAllConsumed({"gemma", "--weights=x", "--verbosity", "2"});
  // Two args with values
  CheckAllConsumed({"gemma", "--verbosity", "2", "--deterministic=true"});
}

TEST(GemmaArgsTest, UnconsumedArgs) {
  // Single unconsumed arg
  CheckUnconsumed({"gemma", "--UNDEFINED"}, 1);
  // Single unconsumed arg, no --
  CheckUnconsumed({"gemma", "UNDEFINED"}, 1);
  // Single unconsumed arg after valid arg
  CheckUnconsumed({"gemma", "--weights=x", "--UNDEFINED"}, 2);
  // Single unconsumed arg before valid arg
  CheckUnconsumed({"gemma", "--UNDEFINED", "--weights=x"}, 1);
  // Single unconsumed arg with = after valid arg
  CheckUnconsumed({"gemma", "--weights=x", "--UNDEFINED=1"}, 2);
  // Single unconsumed arg with = before valid arg
  CheckUnconsumed({"gemma", "--UNDEFINED=false", "--weights=x"}, 1);
  // Multiple unconsumed args
  CheckUnconsumed({"gemma", "--UNDEFINED", "--XXX"}, 1);
  // Multiple unconsumed args with valid arg between
  CheckUnconsumed({"gemma", "--UNDEFINED", "--weights=x", "--XXX"}, 1);
}

// Parses `args` (including argv[0]) into `InferenceArgs`.
static InferenceArgs ParseInferenceArgs(const std::vector<std::string>& args) {
  std::vector<char*> ptrs;
  FillPtrs(args, ptrs);
  const int argc = static_cast<int>(args.size());
  char** argv = const_cast<char**>(ptrs.data());
  ConsumedArgs consumed(argc, argv);
  InferenceArgs inference(argc, argv, consumed);
  consumed.ClearUnconsumed();
  return inference;
}

TEST(GemmaArgsTest, ParsesValidNumbers) {
  const InferenceArgs a =
      ParseInferenceArgs({"gemma", "--top_k=40", "--temperature", "0.5",
                          "--verbosity=-1", "--seq_len=3000000000"});
  EXPECT_EQ(a.top_k, size_t{40});
  EXPECT_FLOAT_EQ(a.temperature, 0.5f);
  EXPECT_EQ(a.verbosity, -1);
  EXPECT_EQ(a.seq_len, size_t{3000000000});
}

// Invalid numeric values must produce the "Invalid value" abort message rather
// than being silently accepted (negative values for unsigned args wrapped to
// SIZE_MAX, trailing garbage was ignored) or escaping as an uncaught exception.
TEST(GemmaArgsDeathTest, RejectsInvalidNumbers) {
  EXPECT_DEATH(ParseInferenceArgs({"gemma", "--top_k=-1"}),
               "Invalid value for top_k");
  EXPECT_DEATH(ParseInferenceArgs({"gemma", "--top_k", "abc"}),
               "Invalid value for top_k");
  EXPECT_DEATH(ParseInferenceArgs({"gemma", "--max_generated_tokens=10x"}),
               "Invalid value for max_generated_tokens");
  EXPECT_DEATH(ParseInferenceArgs({"gemma", "--verbosity=3000000000"}),
               "Invalid value for verbosity");
  EXPECT_DEATH(ParseInferenceArgs({"gemma", "--temperature=1.0x"}),
               "Invalid value for temperature");
  EXPECT_DEATH(ParseInferenceArgs({"gemma", "--temperature="}),
               "Invalid value for temperature");
}

}  // namespace gcpp
