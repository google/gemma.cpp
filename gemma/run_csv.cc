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

// Command line text interface to gemma.

#include <stdio.h>

#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "gemma/configs.h"
#include "gemma/gemma.h"
#include "util/app.h"
#include "util/args.h"  // ArgsBase
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/highway.h"
#include "hwy/profiler.h"
#include "third_party/riegeli/bytes/file_reader.h"
#include "third_party/riegeli/bytes/file_writer.h"
#include "third_party/riegeli/csv/csv_reader.h"
#include "third_party/riegeli/csv/csv_writer.h"

namespace gcpp {

struct CsvArgs : public ArgsBase<CsvArgs> {
  CsvArgs(int argc, char* argv[]) { InitAndParse(argc, argv); }

  Path input_csv;
  Path output_csv;
  int prompt_column;

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(input_csv, "input_csv", Path(),
            "When set, prompts will be read from this CSV.");
    visitor(output_csv, "output_csv", Path("/tmp/output.csv"),
            "When --input_csv is set, prompts will be written to this CSV.");
    visitor(prompt_column, "prompt_column", 0, "Prompt column index");
  };
};

void FileGemma(gcpp::Gemma& model, InferenceArgs& inference, AppArgs& app,
               CsvArgs& csv, hwy::ThreadPool& pool, hwy::ThreadPool& inner_pool,
               const gcpp::AcceptFunc& accept_token) {
  int abs_pos = 0;      // absolute token index over all turns
  int current_pos = 0;  // token index within the current turn
  int prompt_size{};

  std::mt19937 gen;
  if (inference.deterministic) {
    gen.seed(42);
  } else {
    std::random_device rd;
    gen.seed(rd());
  }

  std::stringstream response_stream;

  // callback function invoked for each generated token.
  auto stream_token = [&inference, &abs_pos, &current_pos, &gen, &prompt_size,
                       tokenizer = &model.Tokenizer(),
                       &response_stream](int token, float) {
    ++abs_pos;
    ++current_pos;
    if (current_pos < prompt_size) {
      // pass
    } else if (token == gcpp::EOS_ID) {
      if (!inference.multiturn) {
        abs_pos = 0;
        if (inference.deterministic) {
          gen.seed(42);
        }
      }
      // end of stream
    } else {
      std::string token_text;
      HWY_ASSERT(tokenizer->Decode({token}, &token_text).ok());
      // +1 since position is incremented above
      if (current_pos == prompt_size + 1) {
        // first token of response
        token_text.erase(0, token_text.find_first_not_of(" \t\n"));
      }
      if (token_text != "\n")
        response_stream << token_text;
      else
        response_stream << "\\n";
    }
    return true;
  };

  riegeli::CsvReader csv_reader(
      riegeli::FileReader(csv.input_csv.path),
      riegeli::CsvReaderBase::Options().set_comment('#').set_recovery(
          [](absl::Status status, riegeli::CsvReaderBase& csv_reader) {
            fprintf(stderr, "Invalid entry: %s", status.message().data());
            return true;
          }));

  riegeli::CsvWriter csv_writer(
      riegeli::FileWriter(csv.output_csv.path),
      riegeli::CsvWriterBase::Options().set_header({"prompt", "response"}));

  if (!csv_reader.ok()) {
    HWY_ABORT("Invalid input CSV path %s", csv.input_csv.path.c_str());
  }

  if (!csv_writer.ok()) {
    HWY_ABORT("Invalid output CSV path %s", csv.output_csv.path.c_str());
  }

  while (abs_pos < inference.max_tokens) {
    std::string prompt_string;
    std::vector<int> prompt;
    current_pos = 0;

    std::vector<std::string> record;
    csv_reader.ReadRecord(record);

    if (record.empty()) {
      break;
    }

    prompt_string = record[csv.prompt_column];
    fprintf(stdout, "Prompt: %s\n", prompt_string.c_str());

    prompt_string =
        "<ctrl99>user\n" + prompt_string + "<ctrl100>\n<ctrl99>model\n";
    if (abs_pos > 0) {
      // multi-turn dialogue continuation.
      prompt_string = "<ctrl100>\n" + prompt_string;
    } else {
      HWY_DASSERT(abs_pos == 0);
      if (gcpp::kSystemPrompt) {
        prompt_string =
            "<ctrl99>system\nYou are a large language model built by "
            "Google.<ctrl100>\n" +
            prompt_string;
      }
    }
    HWY_ASSERT(model.Tokenizer().Encode(prompt_string, &prompt).ok());
    prompt_size = prompt.size();

    // generate prompt
    GenerateGemma(model, inference, prompt, abs_pos, pool, inner_pool,
                  stream_token, accept_token, gen, app.verbosity);

    std::string response_string = response_stream.str();
    if (!csv_writer.WriteRecord({record[csv.prompt_column], response_string})) {
      fprintf(stderr, "Failed to write CSV: %s\n",
              csv_writer.status().message().data());
    }

    response_stream.str(std::string());  // reset stream
    response_stream.clear();
    abs_pos = 0;
  }

  if (!csv_reader.Close()) {
    fprintf(stderr, "Failed to close the CSV reader\n");
  }
  if (!csv_writer.Close()) {
    fprintf(stderr, "Failed to close the CSV writer\n");
  }
}

void Run(LoaderArgs& loader, InferenceArgs& inference, AppArgs& app,
         CsvArgs& csv) {
  PROFILER_ZONE("Run.misc");

  hwy::ThreadPool inner_pool(0);
  hwy::ThreadPool pool(app.num_threads);
  // For many-core, pinning threads to cores helps.
  if (app.num_threads > 10) {
    pool.Run(0, pool.NumThreads(),
             [](uint64_t /*task*/, size_t thread) { PinThreadToCore(thread); });
  }

  gcpp::Gemma model(loader.tokenizer, loader.compressed_weights,
                    loader.ModelType(), loader.ModelTraining(), pool);

  if (csv.input_csv.path.empty()) {
    HWY_ABORT("Need to specify csv file.");
  }

  FileGemma(model, inference, app, csv, pool, inner_pool,
            [](int) { return true; });
}

}  // namespace gcpp

int main(int argc, char** argv) {
  {
    PROFILER_ZONE("Startup.misc");
    gcpp::LoaderArgs loader(argc, argv);
    gcpp::InferenceArgs inference(argc, argv);
    gcpp::AppArgs app(argc, argv);
    gcpp::CsvArgs csv(argc, argv);

    if (const char* error = loader.Validate()) {
      loader.Help();
      HWY_ABORT("Invalid args: %s", error);
    }

    gcpp::Run(loader, inference, app, csv);
  }
  PROFILER_PRINT_RESULTS();  // Must call outside the zone above.
  return 0;
}
