# gemma.cpp

gemma.cpp is a lightweight, standalone C++ inference engine for the Gemma
foundation models from Google.

For additional information about Gemma, see
[ai.google.dev/gemma](https://ai.google.dev/gemma). Model weights, including gemma.cpp
specific artifacts, are [available on
kaggle](https://www.kaggle.com/models/google/gemma).

## Who is this project for?

Modern LLM inference engines are sophisticated systems, often with bespoke
capabilities extending beyond traditional neural network runtimes. With this
comes opportunities for research and innovation through co-design of high level
algorithms and low-level computation. However, there is a gap between
deployment-oriented C++ inference runtimes, which are not designed for
experimentation, and Python-centric ML research frameworks, which abstract away
low-level computation through compilation.

gemma.cpp provides a minimalist implementation of Gemma 2B and 7B models,
focusing on simplicity and directness rather than full generality. This is
inspired by vertically-integrated model implementations such as
[ggml](https://github.com/ggerganov/ggml),
[llama.c](https://github.com/karpathy/llama2.c), and
[llama.rs](https://github.com/srush/llama2.rs).

gemma.cpp targets experimentation and research use cases. It is intended to be
straightforward to embed in other projects with minimal dependencies and also
easily modifiable with a small ~2K LoC core implementation (along with ~4K LoC
of supporting utilities). We use the [Google
Highway](https://github.com/google/highway) Library to take advantage of
portable SIMD for CPU inference.

For production-oriented edge deployments we recommend standard deployment
pathways using Python frameworks like JAX, Keras, PyTorch, and Transformers
([all model variations here](https://www.kaggle.com/models/google/gemma)).

Community contributions large and small are welcome. This project follows
[Google's Open Source Community
Guidelines](https://opensource.google.com/conduct/).

*Active development is currently done on the `dev` branch. Please open pull
requests targeting `dev` branch instead of `main`, which is intended to be more
stable.*

## Quick Start

### System requirements

Before starting, you should have installed:

- [CMake](https://cmake.org/)
- [Clang C++ compiler](https://clang.llvm.org/get_started.html), supporting at
  least C++17.
- `tar` for extracting archives from Kaggle.

Building natively on Windows requires the Visual Studio 2012 Build Tools with the
optional Clang/LLVM C++ frontend (`clang-cl`). This can be installed from the
command line with
[`winget`](https://learn.microsoft.com/en-us/windows/package-manager/winget/):

```sh
winget install --id Kitware.CMake
winget install --id Microsoft.VisualStudio.2022.BuildTools --force --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools;installRecommended --add Microsoft.VisualStudio.Component.VC.Llvm.Clang --add Microsoft.VisualStudio.Component.VC.Llvm.ClangToolset"
```

### Step 1: Obtain model weights and tokenizer from Kaggle

Visit [the Gemma model page on
Kaggle](https://www.kaggle.com/models/google/gemma) and select `Model Variations
|> Gemma C++`. On this tab, the `Variation` dropdown includes the options below.
Note bfloat16 weights are higher fidelity, while 8-bit switched floating point
weights enable faster inference.

2B instruction-tuned (`it`) and pre-trained (`pt`) models:

| Model name  | Description |
| ----------- | ----------- |
| `2b-it`     | 2 billion parameter instruction-tuned model, bfloat16 |
| `2b-it-sfp` | 2 billion parameter instruction-tuned model, 8-bit switched floating point |
| `2b-pt`     | 2 billion parameter pre-trained model, bfloat16 |
| `2b-pt-sfp` | 2 billion parameter pre-trained model, 8-bit switched floating point |

7B instruction-tuned (`it`) and pre-trained (`pt`) models:

| Model name  | Description |
| ----------- | ----------- |
| `7b-it`     | 7 billion parameter instruction-tuned model, bfloat16 |
| `7b-it-sfp` | 7 billion parameter instruction-tuned model, 8-bit switched floating point |
| `7b-pt`     | 7 billion parameter pre-trained model, bfloat16 |
| `7b-pt-sfp` | 7 billion parameter pre-trained model, 8-bit switched floating point |

> [!NOTE]
> We *recommend starting with `2b-it-sfp`* to get up and running.

### Step 2: Extract Files

After filling out the consent form, the download should proceed to retrieve a
tar archive file `archive.tar.gz`. Extract files from `archive.tar.gz` (this can
take a few minutes):

```
tar -xf archive.tar.gz
```

This should produce a file containing model weights such as `2b-it-sfp.sbs` and
a tokenizer file (`tokenizer.spm`). You may want to move these files to a
convenient directory location (e.g. the `build/` directory in this repo).

### Step 3: Build

The build system uses [CMake](https://cmake.org/). To build the gemma inference
runtime, create a build directory and generate the build files using `cmake`
from the top-level project directory:

#### Unix-like Platforms

```sh
# Configure `build` directory
cmake --preset make

# Build project using make
cmake --build --preset make -j [number of parallel threads to use]
```

If the `nproc` command is available, you can use `-j $(nproc)`.

If this is successful, you should now have a `gemma` executable in the `build/` directory.

> [!NOTE]
> On Windows Subsystem for Linux (WSL) users should set the number of
> parallel threads to 1. Using a larger number may result in errors.

#### Windows

```sh
# Configure `build` directory
cmake --preset windows

# Build project using Visual Studio Build Tools
cmake --build --preset windows -j [number of parallel threads to use]
```

If this is successful, you should now have a `gemma.exe` executable in the `build/` directory.

### Step 4: Run

You can now run `gemma` from inside the `build/` directory.

`gemma` has the following required arguments:

| Argument | Description | Example value |
| -------- | ----------- | ------------- |
| `--model` | The model type. | `2b-it`, `2b-pt`, `7b-it`, `7b-pt`, ... (see above) |
| `--compressed_weights` | The compressed weights file. | `2b-it-sfp.sbs`, ... (see above) |
| `--tokenizer` | The tokenizer file. | `tokenizer.spm` |


`gemma` is invoked as:

```sh
./gemma \
--tokenizer [tokenizer file] \
--compressed_weights [compressed weights file] \
--model [2b-it or 2b-pt or 7b-it or 7b-pt or ...]
```

Example invocation for the following configuration:

- Compressed weights file `2b-it-sfp.sbs` (2B instruction-tuned model, 8-bit
  switched floating point).
- Tokenizer file `tokenizer.spm`.

```sh
./gemma \
--tokenizer tokenizer.spm \
--compressed_weights 2b-it-sfp.sbs \
--model 2b-it
```

## Usage

`gemma` has different usage modes, controlled by the verbosity flag.

All usage modes are currently interactive, triggering text generation upon
newline input.

| Verbosity       | Usage mode | Details                                       |
| --------------- | ---------- | --------------------------------------------- |
| `--verbosity 0` | Minimal | Only prints generation output. Suitable as a CLI tool. |
| `--verbosity 1` | Default | Standard user-facing terminal UI. |
| `--verbosity 2` | Detailed | Shows additional developer and debug info. |

### Interactive Terminal App

By default, verbosity is set to 1, bringing up a terminal-based interactive
interface when `gemma` is invoked:

```console
$ ./gemma [...]
  __ _  ___ _ __ ___  _ __ ___   __ _   ___ _ __  _ __
 / _` |/ _ \ '_ ` _ \| '_ ` _ \ / _` | / __| '_ \| '_ \
| (_| |  __/ | | | | | | | | | | (_| || (__| |_) | |_) |
 \__, |\___|_| |_| |_|_| |_| |_|\__,_(_)___| .__/| .__/
  __/ |                                    | |   | |
 |___/                                     |_|   |_|

tokenizer                     : tokenizer.spm
compressed_weights            : 2b-it-sfp.sbs
model                         : 2b-it
weights                       : [no path specified]
max_tokens                    : 3072
max_generated_tokens          : 2048

*Usage*
  Enter an instruction and press enter (%Q quits).

*Examples*
  - Write an email to grandma thanking her for the cookies.
  - What are some historical attractions to visit around Massachusetts?
  - Compute the nth fibonacci number in javascript.
  - Write a standup comedy bit about WebGPU programming.

> What are some outdoorsy places to visit around Boston?

[ Reading prompt ] .....................


**Boston Harbor and Islands:**

* **Boston Harbor Islands National and State Park:** Explore pristine beaches, wildlife, and maritime history.
* **Charles River Esplanade:** Enjoy scenic views of the harbor and city skyline.
* **Boston Harbor Cruise Company:** Take a relaxing harbor cruise and admire the city from a different perspective.
* **Seaport Village:** Visit a charming waterfront area with shops, restaurants, and a seaport museum.

**Forest and Nature:**

* **Forest Park:** Hike through a scenic forest with diverse wildlife.
* **Quabbin Reservoir:** Enjoy boating, fishing, and hiking in a scenic setting.
* **Mount Forest:** Explore a mountain with breathtaking views of the city and surrounding landscape.

...
```

### Usage as a Command Line Tool

For using the `gemma` executable as a command line tool, it may be useful to
create an alias for gemma.cpp with arguments fully specified:

```sh
alias gemma2b="~/gemma.cpp/build/gemma -- --tokenizer ~/gemma.cpp/build/tokenizer.spm --compressed_weights ~/gemma.cpp/build/2b-it-sfp.sbs --model 2b-it --verbosity 0"
```

Replace the above paths with your own paths to the model and tokenizer paths
from the download.

Here is an example of prompting `gemma` with a truncated input
file (using a `gemma2b` alias like defined above):

```sh
cat configs.h | tail -35 | tr '\n' ' ' | xargs -0 echo "What does this C++ code do: " | gemma2b
```

> [!NOTE]
> CLI usage of gemma.cpp is experimental and should take context length
> limitations into account.

The output of the above command should look like:

```console
$ cat configs.h | tail -35 | tr '\n' ' ' | xargs -0 echo "What does this C++ code do: " | gemma2b
[ Reading prompt ] ......................................................................................................................................................................................................................................................................................................................................................................................................................................................................................
The code defines two C++ structs, `ConfigGemma7B` and `ConfigGemma2B`, which are used for configuring a deep learning model.

**ConfigGemma7B**:

* `seq_len`: Stores the length of the sequence to be processed. It's set to 7168.
* `vocab_size`: Stores the size of the vocabulary, which is 256128.
* `n_layers`: Number of layers in the deep learning model. It's set to 28.
* `dim_model`: Dimension of the model's internal representation. It's set to 3072.
* `dim_ffw_hidden`: Dimension of the feedforward and recurrent layers' hidden representations. It's set to 16 * 3072 / 2.

**ConfigGemma2B**:

* `seq_len`: Stores the length of the sequence to be processed. It's also set to 7168.
* `vocab_size`: Size of the vocabulary, which is 256128.
* `n_layers`: Number of layers in the deep learning model. It's set to 18.
* `dim_model`: Dimension of the model's internal representation. It's set to 2048.
* `dim_ffw_hidden`: Dimension of the feedforward and recurrent layers' hidden representations. It's set to 16 * 2048 / 2.

These structs are used to configure a deep learning model with specific parameters for either Gemma7B or Gemma2B architecture.
```

### Incorporating gemma.cpp as a Library in your Project

The easiest way to incorporate gemma.cpp in your own project is to pull in
gemma.cpp and dependencies using `FetchContent`. You can add the following to your
CMakeLists.txt:

```
include(FetchContent)

FetchContent_Declare(sentencepiece GIT_REPOSITORY https://github.com/google/sentencepiece GIT_TAG 53de76561cfc149d3c01037f0595669ad32a5e7c)
FetchContent_MakeAvailable(sentencepiece)

FetchContent_Declare(gemma GIT_REPOSITORY https://github.com/google/gemma.cpp GIT_TAG origin/main)
FetchContent_MakeAvailable(gemma)

FetchContent_Declare(highway GIT_REPOSITORY https://github.com/google/highway.git GIT_TAG da250571a45826b21eebbddc1e50d0c1137dee5f)
FetchContent_MakeAvailable(highway)
```

Note for the gemma.cpp `GIT_TAG`, you may replace `origin/main` for a specific
commit hash if you would like to pin the library version.

After your executable is defined (substitute your executable name for
`[Executable Name]` below):

```
target_link_libraries([Executable Name] libgemma hwy hwy_contrib sentencepiece)
FetchContent_GetProperties(gemma)
FetchContent_GetProperties(sentencepiece)
target_include_directories([Executable Name] PRIVATE ${gemma_SOURCE_DIR})
target_include_directories([Executable Name] PRIVATE ${sentencepiece_SOURCE_DIR})
```

### Building gemma.cpp as a Library

gemma.cpp can also be used as a library dependency in your own project. The
shared library artifact can be built by modifying the make invocation to build
the `libgemma` target instead of `gemma`.

> [!NOTE]
> If you are using gemma.cpp in your own project with the `FetchContent` steps
> in the previous section, building the library is done automatically by `cmake`
> and this section can be skipped.

First, run `cmake`:

```sh
cmake -B build
```

Then, run `make` with the `libgemma` target:

```sh
cd build
make -j [number of parallel threads to use] libgemma
```

If this is successful, you should now have a `libgemma` library file in the
`build/` directory. On Unix platforms, the filename is `libgemma.a`.

## Acknowledgements and Contacts

gemma.cpp was started in fall 2023 by [Austin Huang](mailto:austinvhuang@google.com)
and [Jan Wassenberg](mailto:janwas@google.com), and subsequently released February 2024
thanks to contributions from Phil Culliton, Paul Chang, and Dan Zheng.

This is not an officially supported Google product.
