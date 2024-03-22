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

// Command line arguments.

#ifndef THIRD_PARTY_GEMMA_CPP_UTIL_ARGS_H_
#define THIRD_PARTY_GEMMA_CPP_UTIL_ARGS_H_

#include <stdio.h>

#include <algorithm>  // std::transform
#include <string>

#include "hwy/base.h"  // HWY_ABORT

namespace gcpp {

// Wrapper for strings representing a path name. Differentiates vs. arbitrary
// strings and supports shortening for display purposes.
struct Path {
  Path& operator=(const char* other) {
    path = other;
    return *this;
  }

  std::string Shortened() const {
    constexpr size_t max_len = 48;
    constexpr size_t cut_point = max_len / 2 - 5;
    if (path.size() > max_len) {
      return std::string(begin(path), begin(path) + cut_point) + " ... " +
             std::string(end(path) - cut_point, end(path));
    }
    if (path.empty()) return "[no path specified]";
    return path;
  }

  std::string path;
};

// Args is a class that provides a ForEach member function which visits each of
// its member variables. ArgsBase provides functions called by Args to
// initialize values to their defaults (passed as an argument to the visitor),
// print and parse, without having to repeat the args for each usage.
template <class Args>
class ArgsBase {
  struct InitVisitor {
    template <typename T>
    void operator()(T& t, const char* /*name*/, const T& init,
                    const char* /*help*/, int /*print_verbosity*/ = 0) const {
      t = init;
    }
  };

  struct HelpVisitor {
    template <typename T>
    void operator()(T&, const char* name, T /*init*/, const char* help,
                    int /*print_verbosity*/ = 0) const {
      fprintf(stderr, "  --%s : %s\n", name, help);
    }
  };

  class PrintVisitor {
   public:
    explicit PrintVisitor(int verbosity) : verbosity_(verbosity) {}

    template <typename T>
    void operator()(const T& t, const char* name, const T& /*init*/,
                    const char* /*help*/, int print_verbosity = 0) const {
      if (verbosity_ >= print_verbosity) {
        fprintf(stderr, "%-30s: %s\n", name, std::to_string(t).c_str());
      }
    }

    void operator()(const std::string& t, const char* name,
                    const std::string& /*init*/, const char* /*help*/,
                    int print_verbosity = 0) const {
      if (verbosity_ >= print_verbosity) {
        fprintf(stderr, "%-30s: %s\n", name, t.c_str());
      }
    }
    void operator()(const Path& t, const char* name, const Path& /*init*/,
                    const char* /*help*/, int print_verbosity = 0) const {
      if (verbosity_ >= print_verbosity) {
        fprintf(stderr, "%-30s: %s\n", name, t.Shortened().c_str());
      }
    }

   private:
    int verbosity_;
  };

  // Supported types: integer, float, std::string, bool, Path. This is O(N^2):
  // for each arg, we search through argv. If there are more than a dozen args,
  // consider adding a hash-map to speed this up.
  class ParseVisitor {
   public:
    ParseVisitor(int argc, char* argv[]) : argc_(argc), argv_(argv) {}

    template <typename T>
    void operator()(T& t, const char* name, const T& /*init*/,
                    const char* /*help*/, int /*print_verbosity*/ = 0) const {
      const std::string prefixed = std::string("--") + name;
      for (int i = 1; i < argc_; ++i) {
        if (std::string(argv_[i]) == prefixed) {
          if (i + 1 >= argc_) {
            HWY_ABORT("Missing value for %s\n", name);
          }
          if (!SetValue(argv_[i + 1], t)) {
            HWY_ABORT("Invalid value for %s, got %s\n", name, argv_[i + 1]);
          }
          return;
        }
      }
    }

   private:
    // Returns false if an invalid value is detected.
    template <typename T, HWY_IF_NOT_FLOAT(T)>
    static bool SetValue(const char* string, T& t) {
      t = std::stoi(string);
      return true;
    }

    template <typename T, HWY_IF_FLOAT(T)>
    static bool SetValue(const char* string, T& t) {
      t = std::stof(string);
      return true;
    }

    static bool SetValue(const char* string, std::string& t) {
      t = string;
      return true;
    }
    static bool SetValue(const char* string, Path& t) {
      t.path = string;
      return true;
    }

    static bool SetValue(const char* string, bool& t) {
      std::string value(string);
      // Lower-case. Arg names are expected to be ASCII-only.
      std::transform(value.begin(), value.end(), value.begin(), [](char c) {
        return 'A' <= c && c <= 'Z' ? c - ('Z' - 'z') : c;
      });

      if (value == "true" || value == "on" || value == "1") {
        t = true;
        return true;
      } else if (value == "false" || value == "off" || value == "0") {
        t = false;
        return true;
      } else {
        return false;
      }
    }

    int argc_;
    char** argv_;
  };  // ParseVisitor

  template <class Visitor>
  void ForEach(Visitor& visitor) {
    static_cast<Args*>(this)->ForEach(visitor);
  }

 public:
  // WARNING: cannot call from ctor because the derived ctor has not yet run.
  void Init() {
    InitVisitor visitor;
    ForEach(visitor);
  }

  void Help() {
    HelpVisitor visitor;
    ForEach(visitor);
  }

  void Print(int verbosity = 0) {
    PrintVisitor visitor(verbosity);
    ForEach(visitor);
  }

  void Parse(int argc, char* argv[]) {
    ParseVisitor visitor(argc, argv);
    ForEach(visitor);
  }

  // For convenience, enables single-line constructor.
  void InitAndParse(int argc, char* argv[]) {
    Init();
    Parse(argc, argv);
  }
};

static inline HWY_MAYBE_UNUSED bool HasHelp(int argc, char* argv[]) {
  // TODO(austinvhuang): handle case insensitivity
  if (argc == 1) {
    // no arguments - print help
    return true;
  }
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--help") {
      return true;
    }
  }
  return false;
}

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_UTIL_ARGS_H_
