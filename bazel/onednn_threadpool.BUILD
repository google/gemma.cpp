# oneDNN built for the THREADPOOL CPU runtime, used by the matmul-primitive
# backend (GEMMA_ONEDNN_MATMUL). oneDNN parallelizes by calling back into
# gemma.cpp's thread pool via the adapter in ops/onednn_matmul.h.
#
# Same sources as the SEQ build in bazel/onednn.BUILD; see bazel/onednn.bzl for
# exactly what the runtime changes and why the two cannot share one build.

load("@gemma//bazel:onednn.bzl", "onednn_targets")

onednn_targets(cpu_runtime = "THREADPOOL")
