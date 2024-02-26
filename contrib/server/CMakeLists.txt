cmake_minimum_required(VERSION 3.15)
project(gemma-server)

include(FetchContent)

# Include abseil

set(ABSL_ENABLE_INSTALL ON)
#FetchContent_Declare(
#  absl
#  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
#  GIT_TAG        origin/master
#  OVERRIDE_FIND_PACKAGE
#)
#FetchContent_MakeAvailable(absl)

# Include gRPC

FetchContent_Declare(
  gRPC
  GIT_REPOSITORY https://github.com/grpc/grpc
  GIT_TAG        v1.62.0
  )
set(FETCHCONTENT_QUIET OFF)
FetchContent_MakeAvailable(gRPC)


# Include gemma

FetchContent_Declare(sentencepiece GIT_REPOSITORY https://github.com/google/sentencepiece GIT_TAG 53de76561cfc149d3c01037f0595669ad32a5e7c)
FetchContent_MakeAvailable(sentencepiece)

FetchContent_Declare(gemma GIT_REPOSITORY https://github.com/google/gemma.cpp GIT_TAG origin/main)
FetchContent_MakeAvailable(gemma)

FetchContent_Declare(highway GIT_REPOSITORY https://github.com/google/highway.git GIT_TAG da250571a45826b21eebbddc1e50d0c1137dee5f)
FetchContent_MakeAvailable(highway)


# Our server

add_executable(gemma-server server.cc pb/llm.grpc.pb.cc pb/llm.pb.cc)
target_link_libraries(gemma-server libgemma hwy hwy_contrib sentencepiece grpc++)
FetchContent_GetProperties(gemma)
FetchContent_GetProperties(sentencepiece)
target_include_directories(gemma-server PRIVATE ${gemma_SOURCE_DIR})
target_include_directories(gemma-server PRIVATE ${sentencepiece_SOURCE_DIR})