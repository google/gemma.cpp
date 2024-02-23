# Weight compression, I/O and analysis

package(
    default_applicable_licenses = ["//third_party/gemma_cpp:license"],
    default_visibility = [
        "//learning/gemini/prod/contrib/gemini_cpp:__subpackages__",
        "//third_party/gemma_cpp:__subpackages__",
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
        # copybara:import_next_line:hwy
        "//:hwy",
        # copybara:import_next_line:hwy
        "//:thread_pool",
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
        # copybara:import_next_line:hwy
        "//:hwy",
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
        # copybara:import_next_line:hwy
        "//:hwy",
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
        "//testing/base/public:gunit_main_no_google3",
        # copybara:import_next_line:hwy
        "//:hwy",
        # copybara:import_next_line:hwy
        "//:hwy_test_util",
        # copybara:import_next_line:hwy
        "//:nanobenchmark",
        # copybara:import_next_line:hwy
        "//:thread_pool",
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
        # copybara:import_next_line:hwy
        "//:hwy",
        "//third_party/highway/hwy/contrib/sort:vqsort",
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
        "//testing/base/public:gunit_main_no_google3",
        # copybara:import_next_line:hwy
        "//:hwy",
        # copybara:import_next_line:hwy
        "//:hwy_test_util",
        # copybara:import_next_line:hwy
        "//:nanobenchmark",
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
        # copybara:import_next_line:hwy
        "//:dot",
        # copybara:import_next_line:hwy
        "//:hwy",
        # copybara:import_next_line:hwy
        "//:thread_pool",
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
        # copybara:import_next_line:hwy
        "//:hwy",
        # copybara:import_next_line:hwy
        "//:nanobenchmark",  # timer
        # copybara:import_next_line:hwy
        "//:thread_pool",
        "//third_party/highway/hwy/contrib/sort:vqsort",
    ],
)
