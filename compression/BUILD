# Weight compression, I/O and analysis

package(
    default_applicable_licenses = [
        "//:license",  # Placeholder comment, do not modify
    ],
    default_visibility = [
        # Placeholder for internal visibility,
        "//:__subpackages__",  # Placeholder, do not modify
    ],
)

config_setting(
    name = "android",
    constraint_values = [
        "@platforms//os:android",
    ],
    visibility = ["//visibility:private"],
)

FILE_DEPS = select({
    "//conditions:default": [
        # Placeholder for io deps, do not remove
    ],
    ":android": [],
})

cc_library(
    name = "io",
    srcs = [
        "io.cc",
        # Placeholder for io backend, do not remove
    ],
    hdrs = ["io.h"],
    deps = [
        "@hwy//:hwy",
    ] + FILE_DEPS,
)

cc_library(
    name = "blob_store",
    srcs = ["blob_store.cc"],
    hdrs = ["blob_store.h"],
    deps = [
        ":io",
        "@hwy//:hwy",
        "@hwy//:thread_pool",
    ],
)

cc_library(
    name = "distortion",
    hdrs = ["distortion.h"],
    deps = [
        "@hwy//:hwy",
        "@hwy//:stats",
        "@hwy//hwy/contrib/sort:vqsort",
    ],
)

cc_test(
    name = "distortion_test",
    size = "small",
    srcs = ["distortion_test.cc"],
    deps = [
        ":distortion",
        ":shared",
        "@googletest//:gtest_main",
        "//:test_util",
        "@hwy//:hwy",
        "@hwy//:hwy_test_util",
        "@hwy//:nanobenchmark",  # Unpredictable1
    ],
)

cc_library(
    name = "shared",
    hdrs = ["shared.h"],
    deps = [
        "@hwy//:hwy",
    ],
)

cc_library(
    name = "sfp",
    textual_hdrs = ["sfp-inl.h"],
    deps = [
        ":shared",
        "@hwy//:hwy",
    ],
)

cc_test(
    name = "sfp_test",
    size = "small",
    srcs = ["sfp_test.cc"],
    features = ["fully_static_link"],
    linkstatic = True,
    local_defines = ["HWY_IS_TEST"],
    # for test_suite.
    tags = ["hwy_ops_test"],
    deps = [
        ":distortion",
        ":sfp",
        ":shared",
        "@googletest//:gtest_main",
        "//:ops",
        "//:test_util",
        "@hwy//:hwy",
        "@hwy//:hwy_test_util",
        "@hwy//:nanobenchmark",
    ],
)

cc_library(
    name = "nuq",
    hdrs = ["nuq.h"],
    textual_hdrs = ["nuq-inl.h"],
    deps = [
        ":sfp",
        ":shared",
        "@hwy//:hwy",
        "@hwy//hwy/contrib/sort:vqsort",
    ],
)

cc_test(
    name = "nuq_test",
    size = "small",
    timeout = "long",
    srcs = ["nuq_test.cc"],
    features = ["fully_static_link"],
    linkstatic = True,
    local_defines = ["HWY_IS_TEST"],
    # for test_suite.
    tags = ["hwy_ops_test"],
    deps = [
        ":distortion",
        ":nuq",
        ":sfp",
        ":shared",
        "@googletest//:gtest_main",
        "//:test_util",
        "@hwy//:hwy",
        "@hwy//:hwy_test_util",
        "@hwy//:nanobenchmark",
    ],
)

cc_library(
    name = "compress",
    hdrs = ["compress.h"],
    textual_hdrs = [
        "compress-inl.h",
    ],
    deps = [
        ":blob_store",
        ":distortion",
        ":io",
        ":nuq",
        ":sfp",
        ":shared",
        "//:fp_arith",
        "@hwy//:hwy",
        "@hwy//:stats",
        "@hwy//:thread_pool",
    ],
)

cc_test(
    name = "compress_test",
    size = "small",
    srcs = ["compress_test.cc"],
    features = ["fully_static_link"],
    linkstatic = True,
    local_defines = ["HWY_IS_TEST"],
    # for test_suite.
    tags = ["hwy_ops_test"],
    deps = [
        ":compress",
        ":distortion",
        "@googletest//:gtest_main",
        "//:test_util",
        "@hwy//:hwy",
        "@hwy//:hwy_test_util",
        "@hwy//:nanobenchmark",
        "@hwy//:thread_pool",
    ],
)

# For internal experimentation
cc_library(
    name = "analyze",
    textual_hdrs = ["analyze.h"],
    deps = [
        ":distortion",
        ":nuq",
        ":sfp",
        "@hwy//:hwy",
        "@hwy//:nanobenchmark",  # timer
        "@hwy//:stats",
        "@hwy//:thread_pool",
        "@hwy//hwy/contrib/sort:vqsort",
    ],
)

cc_library(
    name = "weights_raw",
    hdrs = ["weights_raw.h"],
    deps = [
        "//:allocator",
        "//:common",
        "//compression:compress",
        "@hwy//:hwy",
        "@hwy//:thread_pool",
    ],
)

cc_binary(
    name = "compress_weights",
    srcs = ["compress_weights.cc"],
    deps = [
        ":compress",
        ":shared",
        ":weights_raw",
        # Placeholder for internal dep, do not remove.,
        "//:args",
        "//:common",
        "//:gemma_lib",
        "//:weights",
        "@hwy//:hwy",
        "@hwy//:nanobenchmark",
        "@hwy//:profiler",
        "@hwy//:thread_pool",
    ],
)
