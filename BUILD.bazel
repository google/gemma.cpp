# gemma.cpp is a lightweight, standalone C++ inference engine for the Gemma
# foundation models from Google.

load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:public"],
)

license(
    name = "license",
    package_name = "gemma_cpp",
)

# Dual-licensed Apache 2 and 3-clause BSD.
licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "transformer_ops",
    hdrs = [
        "ops.h",
    ],
    deps = [
        "//compression:compress",
        # copybara:import_next_line:hwy
        "//:algo",
        # copybara:import_next_line:hwy
        "//:dot",
        # copybara:import_next_line:hwy
        "//:hwy",
        # copybara:import_next_line:hwy
        "//:math",
        # copybara:import_next_line:hwy
        "//:matvec",
        # copybara:import_next_line:hwy
        "//:profiler",
        # copybara:import_next_line:hwy
        "//:thread_pool",
        "//hwy/contrib/sort:vqsort",
    ],
)

cc_library(
    name = "args",
    hdrs = [
        "util/args.h",
    ],
    deps = [
        # copybara:import_next_line:hwy
        "//:hwy",
    ],
)

cc_library(
    name = "app",
    hdrs = [
        "util/app.h",
    ],
    deps = [
        ":args",
        # copybara:import_next_line:hwy
        "//:hwy",
    ],
)

cc_library(
    name = "gemma_lib",
    srcs = [
        "gemma.cc",
    ],
    hdrs = [
        "configs.h",
        "gemma.h",
    ],
    deps = [
        ":args",
        ":transformer_ops",
        "//base",
        "//compression:compress",
        # copybara:import_next_line:hwy
        "//:hwy",
        # copybara:import_next_line:hwy
        "//:matvec",
        # copybara:import_next_line:hwy
        "//:nanobenchmark",  # timer
        # copybara:import_next_line:hwy
        "//:profiler",
        # copybara:import_next_line:hwy
        "//:thread_pool",
        ":sentencepiece_processor",
    ],
)

cc_binary(
    name = "gemma",
    srcs = [
        "run.cc",
    ],
    deps = [
        ":app",
        ":args",
        ":gemma_lib",
        "//compression:compress",
        # copybara:import_next_line:hwy
        "//:hwy",
        # copybara:import_next_line:hwy
        "//:nanobenchmark",
        # copybara:import_next_line:hwy
        "//:profiler",
        # copybara:import_next_line:hwy
        "//:thread_pool",
    ],
)
