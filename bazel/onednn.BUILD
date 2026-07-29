# oneDNN built for the SEQ CPU runtime, used by the BRGeMM ukernel backend
# (GEMMA_ONEDNN_BRGEMM); gemma.cpp owns all parallelism, so oneDNN itself must be
# single-threaded.
#
# Same sources as the THREADPOOL build in bazel/onednn_threadpool.BUILD; see
# bazel/onednn.bzl for exactly what the runtime changes and why the two cannot
# share one build.

load("@gemma//bazel:onednn.bzl", "onednn_targets")

onednn_targets(cpu_runtime = "SEQ")
