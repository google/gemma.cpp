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
