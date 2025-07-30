# Hello World Example

This is a minimal/template project for using `gemma.cpp` as a library. Instead
of an interactive interface, it sets up the model state and generates text for a
single hard coded prompt.

Build steps are similar to the main `gemma` executable. For now only
`cmake`/`make` is available for builds (PRs welcome for other build options).

First use `cmake` to configure the project, starting from the `hello_world`
example directory (`gemma.cpp/examples/hello_world`):

```sh
cmake -B build
```

This sets up a build configuration in `gemma.cpp/examples/hello_world/build`.
Note that this fetches `libgemma` from a git commit hash on github.
Alternatively if you want to build using the local version of `gemma.cpp` use:

```sh
cmake -B build -DBUILD_MODE=local
```

Make sure you delete the contents of the build directory before changing
configurations.

Then use `make` to build the project:

```sh
cd build
make hello_world
```

As with the top-level `gemma.cpp` project you can use the `make` commands `-j`
flag to use parallel threads for faster builds.

From inside the `gemma.cpp/examples/hello_world/build` directory, there should
be a `hello_world` executable. You can run it with the same 3 model arguments as
gemma.cpp specifying the tokenizer, compressed weights file, and model type, for
example:

```sh
./hello_world --tokenizer tokenizer.spm --weights 2b-it-sfp.sbs --model 2b-it
```

Should print a greeting to the terminal:

```
"Hello, world! It's a pleasure to greet you all. May your day be filled with joy, peace, and all the things that make your heart soar.
```

For a demonstration of constrained decoding, add the `--reject` flag followed by
a list of token IDs (note that it must be the last flag, since it consumes every
subsequent argument). For example, to reject variations of the word "greeting",
run:

```sh
./hello_world [...] --reject 32338 42360 78107 106837 132832 143859 154230 190205
```
