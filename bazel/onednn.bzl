# Shared Bazel definitions for building oneDNN from source.
#
# oneDNN's CPU runtime is a whole-library compile-time choice, and gemma.cpp has
# two oneDNN backends that need different ones:
#
#   * SEQ        -- ops/brgemm.h (GEMMA_ONEDNN_BRGEMM) drives oneDNN's low-level
#                   BRGeMM ukernel API; gemma.cpp owns all parallelism, so oneDNN
#                   itself must be single-threaded.
#   * THREADPOOL -- ops/onednn_matmul.h (GEMMA_ONEDNN_MATMUL) uses the high-level
#                   dnnl::matmul primitive, which parallelizes internally by
#                   calling back into gemma.cpp's thread pool via the adapter.
#
# One oneDNN build cannot be both, so MODULE.bazel declares two http_archives
# over the identical tarball (@onednn and @onednn_tp, sharing the download
# cache). Their BUILD files are the same except for five generated config lines
# and one copt, so both just call onednn_targets() below with a different
# cpu_runtime. Only the arm selected by //:gemma_onednn_brgemm or
# //:gemma_onednn_matmul is ever compiled.

load("@bazel_skylib//rules:expand_template.bzl", "expand_template")

_VERSION_MAJOR = "3"

_VERSION_MINOR = "11"

_VERSION_PATCH = "0"

_VERSION_HASH = "fc6151651a4577beae5ffac5a4132e75d39e1409"

_INCLUDES = [
    "include",
    "src",
    "src/common",
    "src/cpu",
    "src/cpu/gemm",
    "src/graph",
    "third_party",
    "third_party/ittnotify",
    "third_party/xbyak",
]

_BASE_COPTS = [
    "-fexceptions",
    "-UUSE_MKL",
    "-UUSE_CBLAS",
    "-DDNNL_ENABLE_MAX_CPU_ISA",
    "-DDNNL_ENABLE_ITT_TASKS",
    "-DDNNL_ENABLE_GRAPH_DUMP",
]

_TEXTUAL_HDR_PATTERNS = [
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
]

_GENERATED_HDRS = [
    ":dnnl_config_h",
    ":dnnl_version_h",
    ":dnnl_version_hash_h",
]

# Substitutions for include/oneapi/dnnl/dnnl_config.h.in that are the same for
# every runtime. The runtime-dependent ones are added by _config_substitutions.
#
# ORDER MATTERS. expand_template applies substitutions in dict order, and these
# keys are not mutually exclusive: "#cmakedefine DNNL_EXPERIMENTAL" also matches
# the start of "#cmakedefine DNNL_EXPERIMENTAL_SPARSE" and of the _UKERNEL line
# that _config_substitutions adds. A more specific key must come before any key
# it starts with, otherwise the general rule rewrites the line first and the
# specific one silently never matches. _check_substitution_order enforces this.
_COMMON_CONFIG_SUBSTITUTIONS = {
    "#cmakedefine DNNL_SAFE_RBP": "#undef DNNL_SAFE_RBP",
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
    "#cmakedefine01 BUILD_POOLING": "#define BUILD_POOLING 0",
    "#cmakedefine01 BUILD_PRELU": "#define BUILD_PRELU 0",
    "#cmakedefine01 BUILD_REDUCTION": "#define BUILD_REDUCTION 0",
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
}

def _check_substitution_order(substitutions):
    """Fails if a general key precedes a more specific one it would shadow."""
    keys = substitutions.keys()
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if keys[j].startswith(keys[i]):
                fail("dnnl_config.h substitution %r precedes %r, which it starts " % (keys[i], keys[j]) +
                     "with, so the second would never match. Move it earlier.")
    return substitutions

def _config_substitutions(cpu_runtime):
    """Returns the dnnl_config.h.in substitutions for one CPU runtime."""

    # The ukernel API is what the BRGeMM backend calls, and the matmul primitive
    # plus its weights reorder are what the primitive backend calls. Enabling
    # only what a backend needs keeps the other backend's code out of the build.
    ukernel = cpu_runtime == "SEQ"
    primitive = cpu_runtime == "THREADPOOL"

    # Runtime-dependent keys go first so that "#cmakedefine DNNL_EXPERIMENTAL"
    # in the common set cannot shadow the _UKERNEL line. See the comment there.
    substitutions = {
        "#cmakedefine DNNL_EXPERIMENTAL_UKERNEL": (
            "#define DNNL_EXPERIMENTAL_UKERNEL 1" if ukernel else "#undef DNNL_EXPERIMENTAL_UKERNEL"
        ),
        "#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}": (
            "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_" + cpu_runtime
        ),
        "#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}": (
            "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_" + cpu_runtime
        ),
        # BUILD_PRIMITIVE_ALL is 1, which already registers every primitive, so
        # these two are redundant today. They are pinned on explicitly for the
        # primitive backend because if a future trim sets BUILD_PRIMITIVE_ALL to
        # 0 without flipping them, every DoMatMul_OneDnn call throws
        # "unimplemented" and silently falls back to the stock path with zero
        # speedup -- the single easiest mistake to make here.
        "#cmakedefine01 BUILD_MATMUL": "#define BUILD_MATMUL " + ("1" if primitive else "0"),
        "#cmakedefine01 BUILD_REORDER": "#define BUILD_REORDER " + ("1" if primitive else "0"),
    }
    substitutions.update(_COMMON_CONFIG_SUBSTITUTIONS)
    return _check_substitution_order(substitutions)

def onednn_targets(cpu_runtime):
    """Declares the oneDNN targets for one CPU runtime.

    Called from bazel/onednn.BUILD (SEQ) and bazel/onednn_threadpool.BUILD
    (THREADPOOL), each of which is the build_file of an http_archive in
    MODULE.bazel. Declares :onednn (the library to depend on) and
    :onednn_autogen, plus the three generated config headers.

    Args:
      cpu_runtime: "SEQ" or "THREADPOOL"; sets DNNL_CPU_RUNTIME and
        DNNL_CPU_THREADING_RUNTIME and selects which API is enabled.
    """
    if cpu_runtime not in ("SEQ", "THREADPOOL"):
        fail("onednn_targets: cpu_runtime must be \"SEQ\" or \"THREADPOOL\", got " + repr(cpu_runtime))

    native.exports_files(["LICENSE"])

    expand_template(
        name = "dnnl_config_h",
        out = "include/oneapi/dnnl/dnnl_config.h",
        substitutions = _config_substitutions(cpu_runtime),
        template = "include/oneapi/dnnl/dnnl_config.h.in",
    )

    expand_template(
        name = "dnnl_version_h",
        out = "include/oneapi/dnnl/dnnl_version.h",
        substitutions = {
            "@DNNL_VERSION_MAJOR@": _VERSION_MAJOR,
            "@DNNL_VERSION_MINOR@": _VERSION_MINOR,
            "@DNNL_VERSION_PATCH@": _VERSION_PATCH,
        },
        template = "include/oneapi/dnnl/dnnl_version.h.in",
    )

    expand_template(
        name = "dnnl_version_hash_h",
        out = "include/oneapi/dnnl/dnnl_version_hash.h",
        substitutions = {"@DNNL_VERSION_HASH@": _VERSION_HASH},
        template = "include/oneapi/dnnl/dnnl_version_hash.h.in",
    )

    # The ukernel API needs its -D on the command line too, not just in the
    # generated config header.
    copts = _BASE_COPTS + (["-DDNNL_EXPERIMENTAL_UKERNEL"] if cpu_runtime == "SEQ" else [])

    native.cc_library(
        name = "onednn_autogen",
        srcs = native.glob(["src/cpu/x64/gemm/**/*_kern_autogen*.cpp"]),
        copts = ["-O1", "-U_FORTIFY_SOURCE"] + copts,
        includes = _INCLUDES,
        textual_hdrs = native.glob(
            _TEXTUAL_HDR_PATTERNS + ["src/graph/backend/dnnl/executables/*.hpp"],
        ) + _GENERATED_HDRS,
        visibility = ["//visibility:public"],
    )

    native.cc_library(
        name = "onednn",
        srcs = native.glob(
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
        copts = copts,
        includes = _INCLUDES,
        linkopts = [
            "-lrt",
            "-Wl,--allow-multiple-definition",
        ],
        textual_hdrs = native.glob(_TEXTUAL_HDR_PATTERNS) + _GENERATED_HDRS,
        visibility = ["//visibility:public"],
        deps = [":onednn_autogen"],
    )
