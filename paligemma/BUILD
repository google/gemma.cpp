package(
    default_applicable_licenses = [
        "//:license",  # Placeholder comment, do not modify
    ],
    default_visibility = [
        "//:__subpackages__",  # Placeholder, do not modify
    ],
)

cc_library(
    name = "image",
    srcs = ["image.cc"],
    hdrs = ["image.h"],
    deps = ["@hwy//:hwy"],
)

cc_test(
    name = "image_test",
    srcs = ["image_test.cc"],
    data = ["testdata/image.ppm"],
    deps = [
        ":image",
        "@googletest//:gtest_main",  # buildcleaner: keep
    ],
)

cc_test(
    name = "paligemma_test",
    srcs = ["paligemma_test.cc"],
    # Requires model files
    tags = [
        "local",
        "manual",
        "no_tap",
    ],
    deps = [
        "@googletest//:gtest_main",
        "//:benchmark_helper",
        "//:common",
        "//:gemma_lib",
        "//:tokenizer",
        "@hwy//:hwy",
        "@hwy//:hwy_test_util",
    ],
)
