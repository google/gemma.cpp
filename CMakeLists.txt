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

FetchContent_Declare(highway GIT_REPOSITORY https://github.com/google/highway.git GIT_TAG 1d16731233de45a365b43867f27d0a5f73925300 EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(highway)

## Note: absl needs to be installed by sentencepiece. This will only happen if
## cmake is invoked with -DSPM_ENABLE_SHARED=OFF and -DSPM_ABSL_PROVIDER=module
FetchContent_Declare(sentencepiece GIT_REPOSITORY https://github.com/google/sentencepiece GIT_TAG 53de76561cfc149d3c01037f0595669ad32a5e7c EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(sentencepiece)

FetchContent_Declare(json GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG 9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03 EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(json)

# Find OpenSSL for HTTPS support
find_package(OpenSSL)
if(OPENSSL_FOUND)
  message(STATUS "OpenSSL found, enabling HTTPS support")
  set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE ON)
else()
  message(STATUS "OpenSSL not found, HTTPS support disabled")
  set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF)
endif()

# HTTP library for API server
FetchContent_Declare(httplib GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git GIT_TAG v0.18.1 EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(httplib)

# Create interface target for httplib (header-only library)
add_library(httplib_interface INTERFACE)
target_include_directories(httplib_interface INTERFACE ${httplib_SOURCE_DIR})
if(OPENSSL_FOUND)
  target_link_libraries(httplib_interface INTERFACE OpenSSL::SSL OpenSSL::Crypto)
  target_compile_definitions(httplib_interface INTERFACE CPPHTTPLIB_OPENSSL_SUPPORT)
endif()

set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)

FetchContent_Declare(benchmark GIT_REPOSITORY https://github.com/google/benchmark.git GIT_TAG v1.8.2 EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(benchmark)

# Base source files
set(SOURCES
  compression/compress-inl.h
  compression/compress.cc
  compression/compress.h
  compression/nuq-inl.h
  compression/sfp-inl.h
  compression/types.h
  compression/test_util-inl.h
  evals/benchmark_helper.cc
  evals/benchmark_helper.h
  evals/cross_entropy.cc
  evals/cross_entropy.h
  gemma/activations.h
  gemma/attention.cc
  gemma/attention.h
  gemma/configs.cc
  gemma/configs.h
  gemma/gemma_args.h
  gemma/gemma-inl.h
  gemma/gemma.cc
  gemma/gemma.h
  gemma/griffin.cc
  gemma/griffin.h
  gemma/kv_cache.cc
  gemma/kv_cache.h
  gemma/model_store.cc
  gemma/model_store.h
  gemma/tensor_info.cc
  gemma/tensor_info.h
  gemma/tokenizer.cc
  gemma/tokenizer.h
  gemma/vit.cc
  gemma/vit.h
  gemma/weights.cc
  gemma/weights.h
  io/blob_store.cc
  io/blob_store.h
  io/fields.cc
  io/fields.h
  io/io_win.cc
  io/io.cc
  io/io.h
  ops/dot-inl.h
  ops/matmul_static_bf16.cc
  ops/matmul_static_f32.cc
  ops/matmul_static_nuq.cc
  ops/matmul_static_sfp.cc
  ops/matmul-inl.h
  ops/matmul.cc
  ops/matmul.h
  ops/matvec-inl.h
  ops/ops-inl.h
  ops/ops.h
  ops/sum-inl.h
  paligemma/image.cc
  paligemma/image.h
  util/allocator.cc
  util/allocator.h
  util/basics.cc
  util/basics.h
  util/mat.cc
  util/mat.h
  util/test_util.h
  util/threading_context.cc
  util/threading_context.h
  util/threading.cc
  util/threading.h
  util/topology.cc
  util/topology.h
)

# Add C API sources only when building DLL
if(BUILD_GEMMA_DLL)
  list(APPEND SOURCES
    gemma/bindings/context.h
    gemma/bindings/context.cc
    gemma/bindings/c_api.h
    gemma/bindings/c_api.cc
  )
  message(STATUS "Including C API files for DLL build")
endif()

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

# Shared library target for C# interop
if(BUILD_GEMMA_DLL)
    add_library(gemma_shared SHARED ${SOURCES})
set_property(TARGET gemma_shared PROPERTY CXX_STANDARD 17)
set_target_properties(gemma_shared PROPERTIES 
    PREFIX ""
    OUTPUT_NAME "gemma"
)
set_property(TARGET gemma_shared PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(gemma_shared PUBLIC ./)
target_link_libraries(gemma_shared PRIVATE 
    $<LINK_LIBRARY:WHOLE_ARCHIVE,hwy>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,hwy_contrib>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,sentencepiece-static>
)
target_include_directories(gemma_shared PUBLIC ${sentencepiece_SOURCE_DIR})
target_compile_definitions(gemma_shared 
    PRIVATE 
    GEMMA_EXPORTS
    $<$<PLATFORM_ID:Windows>:_CRT_SECURE_NO_WARNINGS NOMINMAX>
)
target_compile_options(gemma_shared PRIVATE $<$<PLATFORM_ID:Windows>:-Wno-deprecated-declarations>)
install(TARGETS gemma_shared DESTINATION lib)
install(FILES gemma/c_api.h DESTINATION include/gemma)
install(FILES gemma/GemmaInterop.cs DESTINATION include/gemma)
endif()

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
  compression/compress_test.cc
  compression/distortion_test.cc
  compression/nuq_test.cc
  compression/sfp_test.cc
  evals/gemma_test.cc
  gemma/tensor_info_test.cc
  io/blob_store_test.cc
  io/fields_test.cc
  ops/bench_matmul.cc
  ops/dot_test.cc
  ops/gemma_matvec_test.cc
  ops/matmul_test.cc
  ops/ops_test.cc
  paligemma/image_test.cc
  paligemma/paligemma_test.cc
  util/basics_test.cc
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

  target_link_libraries(${TESTNAME} PRIVATE libgemma GTest::Main hwy hwy_contrib hwy_test)

  gtest_discover_tests(${TESTNAME})
endforeach ()

add_executable(gemma_batch_bench evals/gemma_batch_bench.cc)
target_link_libraries(gemma_batch_bench libgemma GTest::Main hwy hwy_contrib nlohmann_json::nlohmann_json)

endif()  # GEMMA_ENABLE_TESTS

## Tools

add_executable(migrate_weights io/migrate_weights.cc)
target_link_libraries(migrate_weights libgemma hwy hwy_contrib)


# API server with SSE support
add_executable(gemma_api_server gemma/api_server.cc)
target_link_libraries(gemma_api_server libgemma hwy hwy_contrib nlohmann_json::nlohmann_json httplib_interface)

# API client for testing
add_executable(gemma_api_client gemma/api_client.cc)
target_link_libraries(gemma_api_client libgemma hwy hwy_contrib nlohmann_json::nlohmann_json httplib_interface)
