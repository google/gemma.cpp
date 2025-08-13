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
project(simplified_gemma)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(highway GIT_REPOSITORY https://github.com/google/highway.git GIT_TAG 1d16731233de45a365b43867f27d0a5f73925300)
FetchContent_MakeAvailable(highway)
FetchContent_Declare(sentencepiece GIT_REPOSITORY https://github.com/google/sentencepiece GIT_TAG 53de76561cfc149d3c01037f0595669ad32a5e7c)
FetchContent_MakeAvailable(sentencepiece)



# Allow for both local and remote building)
option(BUILD_MODE "'local' or 'remote' git fetch for builds")
if (NOT BUILD_MODE)
  set(BUILD_MODE "remote")
endif()
if (BUILD_MODE STREQUAL "local")
  # Relative path to gemma.cpp from examples/simplified_gemma/build/
  FetchContent_Declare(gemma SOURCE_DIR ../../..)
else()
  FetchContent_Declare(gemma GIT_REPOSITORY https://github.com/google/gemma.cpp.git GIT_TAG a9aa63fd2ea6b786ed0706d619588bfe2d43370e)
endif()
FetchContent_MakeAvailable(gemma)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_executable(simplified_gemma run.cc)
target_link_libraries(simplified_gemma hwy hwy_contrib sentencepiece libgemma)
FetchContent_GetProperties(sentencepiece)
target_include_directories(simplified_gemma PRIVATE ${sentencepiece_SOURCE_DIR})
target_compile_definitions(simplified_gemma PRIVATE $<$<PLATFORM_ID:Windows>:_CRT_SECURE_NO_WARNINGS NOMINMAX>)
target_compile_options(simplified_gemma PRIVATE $<$<PLATFORM_ID:Windows>:-Wno-deprecated-declarations>)
