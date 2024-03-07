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
FetchContent_Declare(highway GIT_REPOSITORY https://github.com/google/highway.git GIT_TAG da250571a45826b21eebbddc1e50d0c1137dee5f)
FetchContent_MakeAvailable(highway)
FetchContent_Declare(sentencepiece GIT_REPOSITORY https://github.com/google/sentencepiece GIT_TAG 53de76561cfc149d3c01037f0595669ad32a5e7c)
FetchContent_MakeAvailable(sentencepiece)
FetchContent_Declare(gemma GIT_REPOSITORY https://github.com/google/gemma.cpp.git GIT_TAG e781007836ec034236e90cc4d313d0a8c481bce6)
FetchContent_MakeAvailable(gemma)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_executable(hello_world run.cc)
set_property(TARGET hello_world PROPERTY CXX_STANDARD 17)
target_link_libraries(hello_world hwy hwy_contrib sentencepiece libgemma)
FetchContent_GetProperties(sentencepiece)
target_include_directories(hello_world PRIVATE ${sentencepiece_SOURCE_DIR})
target_compile_definitions(hello_world PRIVATE $<$<PLATFORM_ID:Windows>:_CRT_SECURE_NO_WARNINGS NOMINMAX>)
target_compile_options(hello_world PRIVATE $<$<PLATFORM_ID:Windows>:-Wno-deprecated-declarations>)
