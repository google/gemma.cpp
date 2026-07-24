load("@bazel_skylib//rules:expand_template.bzl", "expand_template")

# oneDNN BUILD for the THREADPOOL CPU runtime (used by GEMMA_ONEDNN_MATMUL).
#
# This is a sibling of bazel/onednn.BUILD, which builds the SAME oneDNN sources
# for the SEQ runtime used by the BRGeMM ukernel backend (GEMMA_ONEDNN_BRGEMM).
# The two differ ONLY in the config header and copts:
#   * DNNL_CPU_RUNTIME / DNNL_CPU_THREADING_RUNTIME: SEQ -> THREADPOOL, so oneDNN
#     parallelizes by calling back into gemma.cpp's thread pool via the adapter
#     in ops/onednn_matmul.h instead of running single-threaded.
#   * DNNL_EXPERIMENTAL_UKERNEL: dropped (both the config #define and the copts
#     -D). The matmul-primitive path does not use the low-level ukernel API.
#   * BUILD_MATMUL / BUILD_REORDER: set to 1. BUILD_PRIMITIVE_ALL is already 1,
#     which registers every primitive, so these are technically redundant here;
#     they are pinned on explicitly because the matmul primitive AND the weights
#     reorder are load-bearing for this backend. If a future trim sets
#     BUILD_PRIMITIVE_ALL 0 without also flipping these, every DoMatMul_OneDnn
#     call would throw "unimplemented" and silently fall back to the stock path
#     with zero speedup -- the single easiest mistake to make here.

exports_files(["LICENSE"])

expand_template(
    name = "dnnl_config_h",
    out = "include/oneapi/dnnl/dnnl_config.h",
    substitutions = {
        "#cmakedefine DNNL_EXPERIMENTAL_UKERNEL": "#undef DNNL_EXPERIMENTAL_UKERNEL",
        "#cmakedefine DNNL_SAFE_RBP": "#undef DNNL_SAFE_RBP",
        "#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}": "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_THREADPOOL",
        "#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}": "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_THREADPOOL",
        "#cmakedefine DNNL_DISABLE_GPU_REF_KERNELS": "#define DNNL_DISABLE_GPU_REF_KERNELS",
        "#cmakedefine DNNL_GPU_RUNTIME DNNL_RUNTIME_${DNNL_GPU_RUNTIME}": "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_NONE",
        "#cmakedefine DNNL_GPU_VENDOR DNNL_VENDOR_${DNNL_GPU_VENDOR}": "#define DNNL_GPU_VENDOR DNNL_VENDOR_NONE",
        "#cmakedefine DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE": "#undef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE",
        "#cmakedefine DNNL_WITH_SYCL": "#undef DNNL_WITH_SYCL",
        "#cmakedefine DNNL_WITH_LEVEL_ZERO": "#undef DNNL_WITH_LEVEL_ZERO",
        "#cmakedefine DNNL_SYCL_CUDA": "#undef DNNL_SYCL_CUDA",
        "#cmakedefine DNNL_SYCL_GENERIC": "#undef DNNL_SYCL_GENERIC",
        "#cmakedefine DNNL_SYCL_HIP": "#undef DNNL_SYCL_HIP",
        "#cmakedefine DNNL_ENABLE_STACK_CHECKER": "#undef DNNL_ENABLE_STACK_CHECKER",
        "#cmakedefine ONEDNN_BUILD_GRAPH": "#define ONEDNN_BUILD_GRAPH",
        "#cmakedefine DNNL_EXPERIMENTAL_SPARSE": "#undef DNNL_EXPERIMENTAL_SPARSE",
        "#cmakedefine DNNL_EXPERIMENTAL_LOGGING": "#undef DNNL_EXPERIMENTAL_LOGGING",
        "#cmakedefine DNNL_EXPERIMENTAL_PROFILING": "#undef DNNL_EXPERIMENTAL_PROFILING",
        "#cmakedefine DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER": "#undef DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER",
        "#cmakedefine DNNL_EXPERIMENTAL": "#undef DNNL_EXPERIMENTAL",
        "#cmakedefine01 BUILD_TRAINING": "#define BUILD_TRAINING 1",
        "#cmakedefine01 BUILD_INFERENCE": "#define BUILD_INFERENCE 0",
        "#cmakedefine01 BUILD_PRIMITIVE_ALL": "#define BUILD_PRIMITIVE_ALL 1",
        "#cmakedefine01 BUILD_BATCH_NORMALIZATION": "#define BUILD_BATCH_NORMALIZATION 0",
        "#cmakedefine01 BUILD_BINARY": "#define BUILD_BINARY 0",
        "#cmakedefine01 BUILD_CONCAT": "#define BUILD_CONCAT 0",
        "#cmakedefine01 BUILD_CONVOLUTION": "#define BUILD_CONVOLUTION 0",
        "#cmakedefine01 BUILD_DECONVOLUTION": "#define BUILD_DECONVOLUTION 0",
        "#cmakedefine01 BUILD_ELTWISE": "#define BUILD_ELTWISE 0",
        "#cmakedefine01 BUILD_GEMM_KERNELS_ALL": "#define BUILD_GEMM_KERNELS_ALL 1",
        "#cmakedefine01 BUILD_GEMM_KERNELS_NONE": "#define BUILD_GEMM_KERNELS_NONE 0",
        "#cmakedefine01 BUILD_GEMM_SSE41": "#define BUILD_GEMM_SSE41 1",
        "#cmakedefine01 BUILD_GEMM_AVX2": "#define BUILD_GEMM_AVX2 1",
        "#cmakedefine01 BUILD_GEMM_AVX512": "#define BUILD_GEMM_AVX512 1",
        "#cmakedefine01 BUILD_GROUP_NORMALIZATION": "#define BUILD_GROUP_NORMALIZATION 1",
        "#cmakedefine01 BUILD_INNER_PRODUCT": "#define BUILD_INNER_PRODUCT 0",
        "#cmakedefine01 BUILD_LAYER_NORMALIZATION": "#define BUILD_LAYER_NORMALIZATION 0",
        "#cmakedefine01 BUILD_LRN": "#define BUILD_LRN 0",
        "#cmakedefine01 BUILD_MATMUL": "#define BUILD_MATMUL 1",
        "#cmakedefine01 BUILD_POOLING": "#define BUILD_POOLING 0",
        "#cmakedefine01 BUILD_PRELU": "#define BUILD_PRELU 0",
        "#cmakedefine01 BUILD_REDUCTION": "#define BUILD_REDUCTION 0",
        "#cmakedefine01 BUILD_REORDER": "#define BUILD_REORDER 1",
        "#cmakedefine01 BUILD_RESAMPLING": "#define BUILD_RESAMPLING 0",
        "#cmakedefine01 BUILD_RNN": "#define BUILD_RNN 0",
        "#cmakedefine01 BUILD_SHUFFLE": "#define BUILD_SHUFFLE 0",
        "#cmakedefine01 BUILD_SOFTMAX": "#define BUILD_SOFTMAX 0",
        "#cmakedefine01 BUILD_SUM": "#define BUILD_SUM 0",
        "#cmakedefine01 BUILD_PRIMITIVE_CPU_ISA_ALL": "#define BUILD_PRIMITIVE_CPU_ISA_ALL 1",
        "#cmakedefine01 BUILD_SSE41": "#define BUILD_SSE41 0",
        "#cmakedefine01 BUILD_AVX2": "#define BUILD_AVX2 0",
        "#cmakedefine01 BUILD_AVX512": "#define BUILD_AVX512 0",
        "#cmakedefine01 BUILD_AMX": "#define BUILD_AMX 0",
        "#cmakedefine01 BUILD_PRIMITIVE_GPU_ISA_ALL": "#define BUILD_PRIMITIVE_GPU_ISA_ALL 0",
        "#cmakedefine01 BUILD_XE2": "#define BUILD_XE2 0",
        "#cmakedefine01 BUILD_XELP": "#define BUILD_XELP 0",
        "#cmakedefine01 BUILD_XEHPG": "#define BUILD_XEHPG 0",
        "#cmakedefine01 BUILD_XEHPC": "#define BUILD_XEHPC 0",
        "#cmakedefine01 BUILD_XEHP": "#define BUILD_XEHP 0",
        "#cmakedefine01 BUILD_SDPA": "#define BUILD_SDPA 1",
        "#cmakedefine01 BUILD_XE3": "#define BUILD_XE3 0",
    },
    template = "include/oneapi/dnnl/dnnl_config.h.in",
)

expand_template(
    name = "dnnl_version_h",
    out = "include/oneapi/dnnl/dnnl_version.h",
    substitutions = {
        "@DNNL_VERSION_MAJOR@": "3",
        "@DNNL_VERSION_MINOR@": "11",
        "@DNNL_VERSION_PATCH@": "0",
    },
    template = "include/oneapi/dnnl/dnnl_version.h.in",
)

expand_template(
    name = "dnnl_version_hash_h",
    out = "include/oneapi/dnnl/dnnl_version_hash.h",
    substitutions = {
        "@DNNL_VERSION_HASH@": "fc6151651a4577beae5ffac5a4132e75d39e1409",
    },
    template = "include/oneapi/dnnl/dnnl_version_hash.h.in",
)

cc_library(
    name = "onednn_autogen",
    srcs = glob(["src/cpu/x64/gemm/**/*_kern_autogen*.cpp"]),
    copts = [
        "-O1",
        "-U_FORTIFY_SOURCE",
        "-fexceptions",
        "-UUSE_MKL",
        "-UUSE_CBLAS",
        "-DDNNL_ENABLE_MAX_CPU_ISA",
        "-DDNNL_ENABLE_ITT_TASKS",
        "-DDNNL_ENABLE_GRAPH_DUMP",
    ],
    includes = [
        "include",
        "src",
        "src/common",
        "src/cpu",
        "src/cpu/gemm",
        "src/graph",
        "third_party",
        "third_party/ittnotify",
        "third_party/xbyak",
    ],
    textual_hdrs = glob([
        "include/**/*",
        "src/common/*.hpp",
        "src/cpu/*.hpp",
        "src/cpu/**/*.hpp",
        "src/cpu/jit_utils/**/*.hpp",
        "src/graph/interface/*.hpp",
        "src/graph/backend/*.hpp",
        "src/graph/backend/dnnl/*.hpp",
        "src/graph/backend/dnnl/executables/*.hpp",
        "src/graph/backend/fake/*.hpp",
        "src/graph/backend/dnnl/passes/*.hpp",
        "src/graph/backend/dnnl/patterns/*.hpp",
        "src/graph/backend/dnnl/kernels/*.hpp",
        "src/graph/utils/*.hpp",
        "src/graph/utils/pm/*.hpp",
        "third_party/ittnotify/**/*.h",
        "third_party/spdlog/**/*.h",
        "third_party/xbyak/*.h",
    ]) + [
        ":dnnl_config_h",
        ":dnnl_version_h",
        ":dnnl_version_hash_h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "onednn",
    srcs = glob(
        [
            "src/common/*.cpp",
            "src/cpu/*.cpp",
            "src/cpu/**/*.cpp",
            "src/cpu/jit_utils/**/*.cpp",
            "src/cpu/x64/**/*.cpp",
            "src/graph/interface/*.cpp",
            "src/graph/backend/*.cpp",
            "src/graph/backend/dnnl/*.cpp",
            "src/graph/backend/dnnl/executables/*.cpp",
            "src/graph/backend/fake/*.cpp",
            "src/graph/backend/dnnl/passes/*.cpp",
            "src/graph/backend/dnnl/patterns/*.cpp",
            "src/graph/backend/dnnl/kernels/*.cpp",
            "src/graph/utils/*.cpp",
            "src/graph/utils/pm/*.cpp",
            "third_party/ittnotify/*.c",
        ],
        exclude = [
            "src/cpu/aarch64/**",
            "src/cpu/rv64/**",
            "src/cpu/ppc64/**",
            "src/cpu/s390x/**",
            "src/cpu/x64/gemm/**/*_kern_autogen.cpp",
            "src/cpu/sycl/**",
        ],
    ),
    copts = [
        "-fexceptions",
        "-UUSE_MKL",
        "-UUSE_CBLAS",
        "-DDNNL_ENABLE_MAX_CPU_ISA",
        "-DDNNL_ENABLE_ITT_TASKS",
        "-DDNNL_ENABLE_GRAPH_DUMP",
    ],
    includes = [
        "include",
        "src",
        "src/common",
        "src/cpu",
        "src/cpu/gemm",
        "src/graph",
        "third_party",
        "third_party/ittnotify",
        "third_party/xbyak",
    ],
    linkopts = [
        "-lrt",
        "-Wl,--allow-multiple-definition",
    ],
    textual_hdrs = glob([
        "include/**/*",
        "src/common/*.hpp",
        "src/cpu/*.hpp",
        "src/cpu/**/*.hpp",
        "src/cpu/jit_utils/**/*.hpp",
        "src/graph/interface/*.hpp",
        "src/graph/backend/*.hpp",
        "src/graph/backend/dnnl/*.hpp",
        "src/graph/backend/fake/*.hpp",
        "src/graph/backend/dnnl/passes/*.hpp",
        "src/graph/backend/dnnl/patterns/*.hpp",
        "src/graph/backend/dnnl/kernels/*.hpp",
        "src/graph/utils/*.hpp",
        "src/graph/utils/pm/*.hpp",
        "third_party/ittnotify/**/*.h",
        "third_party/spdlog/**/*.h",
        "third_party/xbyak/*.h",
    ]) + [
        ":dnnl_config_h",
        ":dnnl_version_h",
        ":dnnl_version_hash_h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":onednn_autogen",
    ],
)
