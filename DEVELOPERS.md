# Developer Notes

## Motivation: A Minimalist C++ LLM Runtime for Research and Experimentation

In the past, neural network inference has been similar to a simple, opaque,
stateless function function with a single input and output. By contrast,
foundation model runtimes are better considered as systems with multiple forms
of state, subsystems, and heterogeneous inputs and outputs. They are often
integrated with a wide variety of other systems that have their own resources
(e.g. RAG and tools) and potentially interact with an external environment. They
have become compute engines to embed proximal tasks and goals within expansively
broad, general-purpose world models.

With this in mind, we believe that developing an experimental runtime that is
flexible and approachable will allow us to explore the design space of co-design
between high level model concerns and low-level runtime computation.

## Design Priorities

Given these motivations, we propose the following priorities for
making decisions regarding the direction and design of the codebase.

**Maximize Leverage with a Narrow Scope.** We focus on direct implementations of
foundation models like Gemma. This allows us to focus effort on bottlenecks of
specific models. We are willing to trade off generality to keep implementation
code relatively simple and readable at all layers of the stack, achieve good
performance, and maintain the velocity of a small team.

**Data Oriented Design.** Follow data oriented design principles where possible
to minimize unnecessary performance pessimization. It's best to apply these
optimizations during the initial design, or when refactoring a subcomponent. The
first step is to think in terms of batches or tuples of plain old data (POD)
types: separate arrays, instead of an array of structs. The second is to
de-emphasize control flow (if statements, virtual functions and class
hierarchies). The third step is to know intrinsic properties of data and bake
that into the layout and algorithm.

**Prioritize Small Batch Latency** Since production serving solutions are
available for large-scale serving powered by accelerators and optimizing for
throughput, this project focuses on the possibilities of local, interactive use
of foundation models. Although throughput remains important, low latency and
small batch sizes are prioritized, other things being equal.

**Maintain a Portable Baseline** Our starting point is a portable CPU SIMD (via
[highway](https://github.com/google/highway)). We expect to add accelerator and
hybrid CPU/GPU support in the future, but the project should continue to allow
builds using this portable baseline. This ensures that research-oriented and
experimental runtimes and hardware platforms will have a minimum viable option
to run Gemma even if specialized production-ready deployment paths are not
available.

## Code Organization

The implementation code is roughly split into 4 layers, from high to low level:

1.  Frontends (`run.cc`) - Either interactive interfaces or automation
    orchestration that interacts. Frontend code implements a use case objective
    in terms of invocations to model inference and generation (2). Projects that
    use gemma.cpp as a library are considered alternative frontends to `run.cc`.
    We will add examples of additional frontends in the future.

2.  Models (`gemma.cc`, `gemma.h`, `configs.h`) - Implements the compute graph
    of the model including supporting functions such as loading and compressing
    weights using transformer operations provided by layer (3).

3.  Operations (`ops.h`) - A minimal set of transformer and supporting
    mathematical operations implementations using compute backends (4). This
    code should be agnostic to the specifics of the compute graph of the model
    implementation (2).

4.  Backend (`highway`) - Low-level hardware interface (SIMD in the case of
    highway) supporting the implementations in (3).

Besides these layers, supporting utilities are:

- `compression/` - model compression operations. The 8-bit switched floating
  point model conversion is here.
- `util/` - command line argument handling and any other utilities.

## Style and Formatting

A `.clang-format` configuration is provided with our defaults, please run source
files through `clang-format` (or a formatter that produces equivalent behavior)
before finalizing PR for submission.

## Compile-Time Flags (Advanced)

There are several compile-time flags to be aware of (note these may or may not
be exposed to the build system):

- `GEMMA_WEIGHT_T` : Sets the level of compression for weights (surfaced as
  WEIGHT_TYPE in CMakeLists.txt). Currently this should be set to `SfpStream`
  (default, if no flag is specified) for 8-bit SFP, or `hwy::bfloat16_t` to
  enable for higher-fidelity (but slower) bfloat16 support. This is defined in
  `gemma.h`.
- `GEMMA_MAX_SEQ_LEN` : Sets maximum sequence length to preallocate for the KV
  Cache. The default is 4096 tokens but can be overridden. This is not exposed
  through `CMakeLists.txt` yet.

In the medium term both of these will likely be deprecated in favor of handling
options at runtime - allowing for multiple weight compression schemes in a single
build and dynamically resizes the KV cache as needed.

## Using gemma.cpp as a Library (Advanced)

Unless you are doing lower level implementations or research, from an
application standpoint you can think of gemma.h and gemma.cc as the "core" of
the library.

You can regard `run.cc` as an example application that your own application is
substituting for, so the invocations into gemma.h and gemma.cc you see in
`run.cc` are probably the functions you'll be invoking. You can find examples
of the invocations to tokenizer methods and `GenerateGemma` in `run.cc`.

Keep in mind gemma.cpp is oriented at more experimental / prototype / research
applications. If you're targeting production, there's more standard paths via
jax / pytorch / keras for NN deployments.

### Gemma struct contains all the state of the inference engine - tokenizer, weights, and activations

`Gemma(...)` - constructor, creates a gemma model object, which is a wrapper
around 3 things - the tokenizer object, weights, activations, and KV Cache.

In a standard LLM chat app, you'll probably use a Gemma object directly, in
more exotic data processing or research applications, you might decompose
working with weights, kv cache and activations (e.g. you might have multiple kv
caches and activations for a single set of weights) more directly rather than
only using a Gemma object.

### Use the tokenizer in the Gemma object (or interact with the Tokenizer object directly)

You pretty much only do things with the tokenizer, call `Encode()` to go from
string prompts to token id vectors, or `Decode()` to go from token id vector
outputs from the model back to strings.

### The main entrypoint for generation is `GenerateGemma()`

Calling into `GenerateGemma` with a tokenized prompt will 1) mutate the
activation values in `model` and 2) invoke StreamFunc - a lambda callback for
each generated token.

Your application defines its own StreamFunc as a lambda callback to do
something everytime a token string is streamed from the engine (eg print to the
screen, write data to the disk, send the string to a server, etc.). You can see
in `run.cc` the StreamFunc lambda takes care of printing each token to the
screen as it arrives.

Optionally you can define accept_token as another lambda - this is mostly for
constrained decoding type of use cases where you want to force the generation
to fit a grammar. If you're not doing this, you can send an empty lambda as a
no-op which is what `run.cc` does.

### If you want to invoke the neural network forward function directly call the `Transformer()` function

For high-level applications, you might only call `GenerateGemma()` and never
interact directly with the neural network, but if you're doing something a bit
more custom you can call transformer which performs a single inference
operation on a single token and mutates the Activations and the KVCache through
the neural network computation.

### For low level operations, defining new architectures, call `ops.h` functions directly

You use `ops.h` if you're writing other NN architectures or modifying the
inference path of the Gemma model.

## Building with Bazel

The sentencepiece library we depend on requires some additional work to build
with the Bazel build system. First, it does not export its BUILD file, so we
provide `bazel/sentencepiece.bazel`. Second, it ships with a vendored subset of
the Abseil library. `bazel/com_google_sentencepiece.patch` changes the code to
support Abseil as a standalone dependency without third_party/ prefixes, similar
to the transforms we apply to Gemma via Copybara.

## Discord

We're also trying out a discord server for discussion here -
https://discord.gg/H5jCBAWxAe
