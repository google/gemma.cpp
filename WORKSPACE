workspace(name = "gemma")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

maybe(
    http_archive,
    name = "rules_license",
    sha256 = "4531deccb913639c30e5c7512a054d5d875698daeb75d8cf90f284375fe7c360",
    urls = [
        "https://github.com/bazelbuild/rules_license/releases/download/0.0.7/rules_license-0.0.7.tar.gz",
    ],
)

maybe(
    http_archive,
    name = "com_google_sentencepiece",
    sha256 = "8409b0126ebd62b256c685d5757150cf7fcb2b92a2f2b98efb3f38fc36719754",
    strip_prefix = "sentencepiece-0.1.96",
    urls = ["https://github.com/google/sentencepiece/archive/refs/tags/v0.1.96.zip"],
    build_file = "@//third_party:sentencepiece.bazel",
    patches = ["@//third_party:com_google_sentencepiece.patch"],
    patch_args = ["-p1"],
)
