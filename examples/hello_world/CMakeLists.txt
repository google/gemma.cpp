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
project(hello_world)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Download progress reporting, shared with the top-level build. This example is
# a standalone project, so fall back to plain FetchContent if it was copied out
# of the gemma.cpp tree and the module is not next to it.
set(GEMMA_DEP_TOTAL 3)
set(_gemma_fetch_module "${CMAKE_CURRENT_LIST_DIR}/../../cmake/GemmaFetch.cmake")
if(EXISTS "${_gemma_fetch_module}")
  include("${_gemma_fetch_module}")
else()
  function(gemma_fetch name)
    FetchContent_Declare(${name} ${ARGN})
    FetchContent_MakeAvailable(${name})
    string(TOLOWER "${name}" _lc)
    set(${_lc}_SOURCE_DIR "${${_lc}_SOURCE_DIR}" PARENT_SCOPE)
  endfunction()
endif()

gemma_fetch(highway GIT_REPOSITORY https://github.com/google/highway.git GIT_TAG 9d5b12611fcfe145f988771c45e7bae9f78cb7fa)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
  target_compile_definitions(hwy PUBLIC HWY_DISABLED_TARGETS=HWY_AVX10_2)
endif()
gemma_fetch(sentencepiece GIT_REPOSITORY https://github.com/google/sentencepiece GIT_TAG 9045b2f60fa2b323dfac0eaef8fc17565036f9f9)



# Allow for both local and remote building)
option(BUILD_MODE "'local' or 'remote' git fetch for builds")
if (NOT BUILD_MODE)
  set(BUILD_MODE "remote")
endif()
if (BUILD_MODE STREQUAL "local")
  # Relative path to gemma.cpp from examples/hello_world/build/
  gemma_fetch(gemma SOURCE_DIR ../../..)
else()
  gemma_fetch(gemma GIT_REPOSITORY https://github.com/google/gemma.cpp.git GIT_TAG 4a924f179448dc83e46a2af9520c61b4ef56174c)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_executable(hello_world run.cc)
target_link_libraries(hello_world hwy hwy_contrib sentencepiece libgemma)
FetchContent_GetProperties(sentencepiece)
target_include_directories(hello_world PRIVATE ${sentencepiece_SOURCE_DIR})
target_compile_definitions(hello_world PRIVATE $<$<PLATFORM_ID:Windows>:_CRT_SECURE_NO_WARNINGS NOMINMAX>)
target_compile_options(hello_world PRIVATE $<$<PLATFORM_ID:Windows>:-Wno-deprecated-declarations>)
