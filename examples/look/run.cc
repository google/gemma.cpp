#include <iostream>

// copybara:import_next_line:gemma_cpp
#include "compression/compress.h"
// copybara:end
// copybara:import_next_line:gemma_cpp
#include "gemma.h"  // Gemma
// copybara:end
// copybara:import_next_line:gemma_cpp
#include "util/app.h"
// copybara:end
// copybara:import_next_line:gemma_cpp
#include "util/args.h"  // HasHelp
// copybara:end
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/highway.h"
#include "hwy/per_target.h"
#include "hwy/profiler.h"
#include "hwy/timer.h"

int main(int argc, char** argv) {
  gcpp::LoaderArgs loader(argc, argv);
  gcpp::AppArgs app(argc, argv);
  hwy::ThreadPool pool(app.num_threads);
  gcpp::Gemma model(loader, pool);
  std::cout << "Done" << std::endl;
}
