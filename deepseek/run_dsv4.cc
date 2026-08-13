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

// Generation driver for DeepSeek V4. DeepSeek uses an HF byte-level BPE
// tokenizer.json (not sentencepiece), handled by Dsv4Tokenizer.
//
// Usage (chat, streams generated text to stdout):
//   run_dsv4 --weights model.sbs --tokenizer_json tokenizer.json \
//            --prompt "Do you like cats?" [--thinking] [--raw] [--mtp] \
//            [--max_generated_tokens N] [--temperature T] [--top_k K] ...
//
// Or with pre-tokenized ids (streams "T <id>" lines, for scripting):
//   run_dsv4 --weights model.sbs --tokens 1,2,3 ...
//
// --tokenize_only prints the prompt token ids and exits (for testing).

#include <stdio.h>
#include <stdlib.h>

#include <memory>
#include <string>
#include <vector>

#include "deepseek/dsv4_tokenizer.h"
#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "gemma/kv_cache.h"
#include "ops/matmul.h"
#include "util/threading_context.h"

namespace gcpp {

static std::vector<int> ParseTokens(const std::string& csv) {
  std::vector<int> tokens;
  size_t start = 0;
  while (start < csv.size()) {
    size_t comma = csv.find(',', start);
    if (comma == std::string::npos) comma = csv.size();
    tokens.push_back(atoi(csv.substr(start, comma - start).c_str()));
    start = comma + 1;
  }
  return tokens;
}

// Returns the length of the longest prefix of `s` that ends on a UTF-8
// character boundary, so streaming output never emits partial characters.
static size_t CompleteUtf8Prefix(const std::string& s) {
  size_t i = s.size();
  size_t cont = 0;  // number of trailing continuation bytes inspected
  while (i > 0 && cont < 3) {
    const uint8_t b = static_cast<uint8_t>(s[i - 1]);
    if ((b & 0xC0) != 0x80) {  // lead or ASCII byte at i-1
      size_t expect = 1;
      if ((b & 0xE0) == 0xC0)
        expect = 2;
      else if ((b & 0xF0) == 0xE0)
        expect = 3;
      else if ((b & 0xF8) == 0xF0)
        expect = 4;
      return (cont + 1 >= expect) ? s.size() : i - 1;
    }
    ++cont;
    --i;
  }
  return s.size() - cont == 0 ? 0 : s.size();
}

int Main(int argc, char** argv) {
  // Extract our own flags before standard arg parsing.
  std::string tokens_csv;
  std::string prompt_text;
  std::string tokenizer_json;
  bool use_mtp = false;
  size_t mtp_draft_horizon = 7;
  float mtp_confidence_threshold = 0.0f;
  size_t max_generated_tokens = 0;
  bool thinking = false;
  bool raw = false;
  bool tokenize_only = false;
  std::vector<char*> passthrough;
  passthrough.push_back(argv[0]);
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--tokens" && i + 1 < argc) {
      tokens_csv = argv[++i];
    } else if (arg == "--prompt" && i + 1 < argc) {
      prompt_text = argv[++i];
    } else if (arg == "--tokenizer_json" && i + 1 < argc) {
      tokenizer_json = argv[++i];
    } else if (arg == "--mtp") {
      use_mtp = true;
    } else if (arg == "--mtp_draft_horizon" && i + 1 < argc) {
      mtp_draft_horizon = static_cast<size_t>(std::stoull(argv[++i]));
    } else if (arg == "--mtp_confidence_threshold" && i + 1 < argc) {
      mtp_confidence_threshold = std::stof(argv[++i]);
    } else if ((arg == "--max_generated_tokens" || arg == "--max_tokens") &&
               i + 1 < argc) {
      max_generated_tokens = static_cast<size_t>(std::stoull(argv[++i]));
    } else if (arg == "--thinking") {
      thinking = true;
    } else if (arg == "--raw") {
      raw = true;
    } else if (arg == "--tokenize_only") {
      tokenize_only = true;
    } else {
      passthrough.push_back(argv[i]);
    }
  }
  int pt_argc = static_cast<int>(passthrough.size());
  char** pt_argv = passthrough.data();

  if (tokens_csv.empty() && prompt_text.empty()) {
    fprintf(stderr,
            "Missing --prompt <text> (requires --tokenizer_json) or "
            "--tokens <csv of ids>\n");
    return 1;
  }

  // Tokenize (or parse) the prompt.
  std::unique_ptr<Dsv4Tokenizer> tokenizer;
  std::vector<int> prompt_vec;
  if (!prompt_text.empty()) {
    if (tokenizer_json.empty()) {
      fprintf(stderr, "--prompt requires --tokenizer_json <tokenizer.json>\n");
      return 1;
    }
    tokenizer = std::make_unique<Dsv4Tokenizer>(tokenizer_json);
    const std::string wrapped =
        raw ? prompt_text : tokenizer->WrapChat(prompt_text, thinking);
    prompt_vec = tokenizer->Encode(wrapped);
  } else {
    prompt_vec = ParseTokens(tokens_csv);
    if (!tokenizer_json.empty()) {
      tokenizer = std::make_unique<Dsv4Tokenizer>(tokenizer_json);
    }
  }
  fprintf(stderr, "Prompt: %zu tokens\n", prompt_vec.size());

  if (tokenize_only) {
    for (size_t i = 0; i < prompt_vec.size(); ++i) {
      printf(i ? ",%d" : "%d", prompt_vec[i]);
    }
    printf("\n");
    return 0;
  }

  ConsumedArgs consumed(pt_argc, pt_argv);
  GemmaArgs args(pt_argc, pt_argv, consumed);
  consumed.AbortIfUnconsumed();

  ThreadingContext ctx(args.threading);
  MatMulEnv env(ctx);
  Gemma gemma(args, ctx);
  KVCache kv_cache(gemma.Config(), args.inference, ctx.allocator);

  const ModelConfig& config = gemma.Config();
  const size_t prompt_size = prompt_vec.size();
  size_t tokens_seen = 0;
  std::string pending;  // decoded bytes not yet flushed (UTF-8 boundary)

  RuntimeConfig runtime_config = {
      .verbosity = args.inference.verbosity,
      .batch_stream_token =
          [&](size_t /*query_idx*/, size_t /*pos*/, int token, float) {
            ++tokens_seen;
            if (tokens_seen <= prompt_size) return true;  // prompt echo
            if (config.IsEOS(token)) {
              fprintf(stderr, "[EOS]\n");
              return false;
            }
            if (tokenizer) {
              tokenizer->AppendDecoded(token, pending);
              const size_t n = CompleteUtf8Prefix(pending);
              if (n > 0) {
                fwrite(pending.data(), 1, n, stdout);
                fflush(stdout);
                pending.erase(0, n);
              }
            } else {
              printf("T %d\n", token);
              fflush(stdout);
            }
            return true;
          },
      .use_spinning = args.threading.spin,
  };
  args.inference.CopyTo(runtime_config);
  if (max_generated_tokens > 0) {
    runtime_config.max_generated_tokens = max_generated_tokens;
  }
  runtime_config.use_mtp = use_mtp;
  runtime_config.mtp_draft_horizon = mtp_draft_horizon;
  runtime_config.mtp_confidence_threshold = mtp_confidence_threshold;
  if (use_mtp) fprintf(stderr, "DSpark MTP speculative decoding enabled.\n");

  TimingInfo timing_info = {.verbosity = args.inference.verbosity};
  const PromptTokens prompt(prompt_vec);
  gemma.Generate(runtime_config, prompt, /*pos=*/0, /*prefix_end=*/0, kv_cache,
                 env, timing_info);
  if (!pending.empty()) fwrite(pending.data(), 1, pending.size(), stdout);
  if (tokenizer) printf("\n");
  fflush(stdout);
  fprintf(stderr, "Done.\n");
  return 0;
}

}  // namespace gcpp

#ifdef _WIN32
#include <shellapi.h>
#include <windows.h>

// The narrow argv is in the ANSI codepage, which destroys non-ASCII prompt
// text (and the chat-marker characters). Re-derive argv as UTF-8.
int main(int, char**) {
  int argc;
  wchar_t** wargv = CommandLineToArgvW(GetCommandLineW(), &argc);
  std::vector<std::string> utf8(argc);
  std::vector<char*> argv(argc);
  for (int i = 0; i < argc; ++i) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0,
                                      nullptr, nullptr);
    utf8[i].resize(n > 0 ? n - 1 : 0);  // n includes the terminator
    if (n > 1) {
      WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, utf8[i].data(), n, nullptr,
                          nullptr);
    }
    argv[i] = utf8[i].data();
  }
  LocalFree(wargv);
  return gcpp::Main(argc, argv.data());
}
#else
int main(int argc, char** argv) { return gcpp::Main(argc, argv); }
#endif
