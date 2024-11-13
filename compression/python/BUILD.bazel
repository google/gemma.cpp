load("//devtools/clif/python:clif_build_rule.bzl", "py_clif_cc")
# [internal] load strict.bzl

package(
    default_applicable_licenses = [
        "//:license",  # Placeholder comment, do not modify
    ],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "compression_clif_aux",
    srcs = ["compression_clif_aux.cc"],
    hdrs = ["compression_clif_aux.h"],
    deps = [
        "//third_party/absl/types:span",
        "//compression:compress",
        "//compression:io",
        "@highway//:hwy",
        "@highway//:thread_pool",
    ],
)

py_clif_cc(
    name = "compression",
    srcs = ["compression.clif"],
    deps = [
        ":compression_clif_aux",
        "//third_party/absl/python/numpy:span_clif_lib",
    ],
)

# py_strict
py_test(
    name = "compression_test",
    srcs = ["compression_test.py"],
    deps = [
        ":compression",
        "//testing/pybase",
        "//third_party/py/numpy",
    ],
)
