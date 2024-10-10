# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.11)

include(FetchContent)

project(gemma)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

FetchContent_Declare(highway GIT_REPOSITORY https://github.com/google/highway.git GIT_TAG 0ca297227a373710e76dd45e0ad4d68adb6928fe EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(highway)

## Note: absl needs to be installed by sentencepiece. This will only happen if
## cmake is invoked with -DSPM_ENABLE_SHARED=OFF and -DSPM_ABSL_PROVIDER=module
FetchContent_Declare(sentencepiece GIT_REPOSITORY https://github.com/google/sentencepiece GIT_TAG 53de76561cfc149d3c01037f0595669ad32a5e7c EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(sentencepiece)

FetchContent_Declare(json GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG 9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03 EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(json)

FetchContent_Declare(benchmark GIT_REPOSITORY https://github.com/google/benchmark.git GIT_TAG v1.8.2 EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(benchmark)

set(SOURCES
  compression/blob_store.cc
  compression/blob_store.h
  compression/compress.cc
  compression/compress.h
  compression/compress-inl.h
  compression/io_win.cc
  compression/io.cc
  compression/io.h
  compression/nuq-inl.h
  compression/sfp-inl.h
  compression/shared.h
  compression/test_util-inl.h
  backprop/activations.h
  backprop/backward.cc
  backprop/backward.h
  backprop/backward-inl.h
  backprop/backward_scalar.h
  backprop/common_scalar.h
  backprop/forward.cc
  backprop/forward.h
  backprop/forward-inl.h
  backprop/forward_scalar.h
  backprop/optimizer.cc
  backprop/optimizer.h
  evals/benchmark_helper.cc
  evals/benchmark_helper.h
  evals/cross_entropy.cc
  evals/cross_entropy.h
  gemma/activations.h
  gemma/common.cc
  gemma/common.h
  gemma/configs.h
  gemma/gemma-inl.h
  gemma/gemma.cc
  gemma/gemma.h
  gemma/instantiations/27b_bf16.cc
  gemma/instantiations/27b_f32.cc
  gemma/instantiations/27b_sfp.cc
  gemma/instantiations/2b_bf16.cc
  gemma/instantiations/2b_f32.cc
  gemma/instantiations/2b_sfp.cc
  gemma/instantiations/7b_bf16.cc
  gemma/instantiations/7b_f32.cc
  gemma/instantiations/7b_sfp.cc
  gemma/instantiations/9b_bf16.cc
  gemma/instantiations/9b_f32.cc
  gemma/instantiations/9b_sfp.cc
  gemma/instantiations/gr2b_bf16.cc
  gemma/instantiations/gr2b_f32.cc
  gemma/instantiations/gr2b_sfp.cc
  gemma/instantiations/tiny_bf16.cc
  gemma/instantiations/tiny_f32.cc
  gemma/instantiations/tiny_sfp.cc
  gemma/instantiations/gemma2_2b_bf16.cc
  gemma/instantiations/gemma2_2b_f32.cc
  gemma/instantiations/gemma2_2b_sfp.cc
  gemma/instantiations/paligemma_224_bf16.cc
  gemma/instantiations/paligemma_224_f32.cc
  gemma/instantiations/paligemma_224_sfp.cc
  gemma/kv_cache.cc
  gemma/kv_cache.h
  gemma/tokenizer.cc
  gemma/tokenizer.h
  gemma/weights.cc
  gemma/weights.h
  ops/dot-inl.h
  ops/matmul-inl.h
  ops/matvec-inl.h
  ops/ops-inl.h
  ops/sum-inl.h
  paligemma/image.cc
  paligemma/image.h
  util/allocator.h
  util/app.h
  util/args.h
  util/basics.h
  util/test_util.h
  util/threading.h
  )

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

FetchContent_GetProperties(sentencepiece)

## Library Target

add_library(libgemma ${SOURCES})
set_property(TARGET libgemma PROPERTY CXX_STANDARD 17)
set_target_properties(libgemma PROPERTIES PREFIX "")
set_property(TARGET libgemma PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(libgemma PUBLIC ./)
target_link_libraries(libgemma hwy hwy_contrib sentencepiece-static)
target_include_directories(libgemma PUBLIC ${sentencepiece_SOURCE_DIR})
target_compile_definitions(libgemma PRIVATE $<$<PLATFORM_ID:Windows>:_CRT_SECURE_NO_WARNINGS NOMINMAX>)
target_compile_options(libgemma PRIVATE $<$<PLATFORM_ID:Windows>:-Wno-deprecated-declarations>)
install(TARGETS libgemma DESTINATION lib)

# Executable Target

add_executable(gemma gemma/run.cc)
target_link_libraries(gemma libgemma hwy hwy_contrib)
install(TARGETS gemma DESTINATION bin)

add_executable(single_benchmark evals/benchmark.cc)
target_link_libraries(single_benchmark libgemma hwy hwy_contrib nlohmann_json::nlohmann_json)

add_executable(benchmarks evals/benchmarks.cc)
target_link_libraries(benchmarks libgemma hwy hwy_contrib nlohmann_json::nlohmann_json benchmark)

add_executable(debug_prompt evals/debug_prompt.cc)
target_link_libraries(debug_prompt libgemma hwy hwy_contrib nlohmann_json::nlohmann_json)

## Tests
set(GEMMA_ENABLE_TESTS OFF CACHE BOOL "Enable Gemma tests")
if (GEMMA_ENABLE_TESTS)

enable_testing()
include(GoogleTest)

set(GEMMA_TEST_FILES
  backprop/backward_scalar_test.cc
  backprop/backward_test.cc
  backprop/optimize_test.cc
  compression/compress_test.cc
  compression/distortion_test.cc
  compression/nuq_test.cc
  compression/sfp_test.cc
  evals/gemma_test.cc
  ops/dot_test.cc
  ops/gemma_matvec_test.cc
  ops/matmul_test.cc
  ops/ops_test.cc
  paligemma/image_test.cc
  paligemma/paligemma_test.cc
  util/threading_test.cc
)

foreach (TESTFILE IN LISTS GEMMA_TEST_FILES)
  # The TESTNAME is the name without the extension or directory.
  get_filename_component(TESTNAME ${TESTFILE} NAME_WE)
  add_executable(${TESTNAME} ${TESTFILE})
  # Test all targets, not just the best/baseline. This changes the default
  # policy to all-attainable; note that setting -DHWY_COMPILE_* directly can
  # cause compile errors because only one may be set, and other CMakeLists.txt
  # that include us may set them.
  target_compile_options(${TESTNAME} PRIVATE -DHWY_IS_TEST=1)

  target_link_libraries(${TESTNAME} PRIVATE libgemma GTest::gtest_main hwy hwy_contrib hwy_test)

  gtest_discover_tests(${TESTNAME})
endforeach ()
endif()  # GEMMA_ENABLE_TESTS

## Tools

add_executable(compress_weights compression/compress_weights.cc)
target_link_libraries(compress_weights libgemma hwy hwy_contrib)
