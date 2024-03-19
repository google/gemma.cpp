# Weight compression, I/O and analysis

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//learning/gemini/prod/contrib/gemini_cpp:__subpackages__",
        "//:__subpackages__",
    ],
)

cc_library(
    name = "blob_store",
    srcs = [
        "blob_store.cc",
    ],
    hdrs = [
        "blob_store.h",
    ],
    deps = [
        "@hwy//:hwy",
        "@hwy//:thread_pool",
    ],
)

cc_library(
    name = "stats",
    srcs = [
        "stats.cc",
    ],
    hdrs = [
        "distortion.h",
        "stats.h",
    ],
    deps = [
        "@hwy//:hwy",
    ],
)

cc_library(
    name = "sfp",
    hdrs = [
        "sfp.h",
    ],
    textual_hdrs = [
        "sfp-inl.h",
    ],
    deps = [
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
        ":sfp",
        ":stats",
        "@googletest//:gtest_main",
        "@hwy//:hwy",
        "@hwy//:hwy_test_util",
        "@hwy//:nanobenchmark",
        "@hwy//:thread_pool",
    ],
)

cc_library(
    name = "nuq",
    hdrs = [
        "nuq.h",
    ],
    textual_hdrs = [
        "nuq-inl.h",
    ],
    deps = [
        ":sfp",
        "@hwy//:hwy",
        "@hwy//hwy/contrib/sort:vqsort",
    ],
)

cc_test(
    name = "nuq_test",
    size = "small",
    srcs = ["nuq_test.cc"],
    features = ["fully_static_link"],
    linkstatic = True,
    local_defines = ["HWY_IS_TEST"],
    # for test_suite.
    tags = ["hwy_ops_test"],
    deps = [
        ":nuq",
        ":sfp",
        ":stats",
        "@googletest//:gtest_main",
        "@hwy//:hwy",
        "@hwy//:hwy_test_util",
        "@hwy//:nanobenchmark",
    ],
)

cc_library(
    name = "compress",
    hdrs = [
        "compress.h",
        "nuq.h",
        "sfp.h",
    ],
    textual_hdrs = [
        "compress-inl.h",
    ],
    deps = [
        ":blob_store",
        ":nuq",
        ":sfp",
        ":stats",
        "@hwy//:dot",
        "@hwy//:hwy",
        "@hwy//:thread_pool",
    ],
)

# For internal experimentation
cc_library(
    name = "analyze",
    textual_hdrs = [
        "analyze.h",
    ],
    deps = [
        ":nuq",
        ":sfp",
        ":stats",
        "@hwy//:hwy",
        "@hwy//:nanobenchmark",  # timer
        "@hwy//:thread_pool",
        "@hwy//hwy/contrib/sort:vqsort",
    ],
)
