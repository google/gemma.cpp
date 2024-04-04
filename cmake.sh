#!/usr/bin/env bash
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

MYDIR=$(dirname $(realpath "$0"))
BUILD_DIR="${BUILD_DIR:-${MYDIR}/build}"

CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"
CMAKE_C_COMPILER="${CMAKE_C_COMPILER:-clang-14}"
CMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER:-clang++-14}"
# Convenience flag to pass both CMAKE_C_FLAGS and CMAKE_CXX_FLAGS
CMAKE_FLAGS="${CMAKE_FLAGS:-}"
CMAKE_C_FLAGS="${CMAKE_C_FLAGS:-} ${CMAKE_FLAGS}"
CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS:-} ${CMAKE_FLAGS}"
CMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS:-}"
CMAKE_MODULE_LINKER_FLAGS="${CMAKE_MODULE_LINKER_FLAGS:-}"
CMAKE_SHARED_LINKER_FLAGS="${CMAKE_SHARED_LINKER_FLAGS:-}"

# Local flags passed to sanitizers.
UBSAN_FLAGS=(
  -fsanitize=alignment
  -fsanitize=bool
  -fsanitize=bounds
  -fsanitize=builtin
  -fsanitize=enum
  -fsanitize=float-cast-overflow
  -fsanitize=float-divide-by-zero
  -fsanitize=integer-divide-by-zero
  -fsanitize=null
  -fsanitize=object-size
  -fsanitize=pointer-overflow
  -fsanitize=return
  -fsanitize=returns-nonnull-attribute
  -fsanitize=shift-base
  -fsanitize=shift-exponent
  -fsanitize=unreachable
  -fsanitize=vla-bound

  -fno-sanitize-recover=undefined
  -fsanitize-recover=alignment
)

CLANG_VERSION="${CLANG_VERSION:-}"
# Detect the clang version suffix and store it in CLANG_VERSION. For example,
# "6.0" for clang 6 or "7" for clang 7.
detect_clang_version() {
  if [[ -n "${CLANG_VERSION}" ]]; then
    return 0
  fi
  local clang_version=$("${CMAKE_C_COMPILER:-clang}" --version | head -n1)
  clang_version=${clang_version#"Debian "}
  clang_version=${clang_version#"Ubuntu "}
  local llvm_tag
  case "${clang_version}" in
    "clang version 6."*)
      CLANG_VERSION="6.0"
      ;;
    "clang version "*)
      # Any other clang version uses just the major version number.
      local suffix="${clang_version#clang version }"
      CLANG_VERSION="${suffix%%.*}"
      ;;
    "emcc"*)
      # We can't use asan or msan in the emcc case.
      ;;
    *)
      echo "Unknown clang version: ${clang_version}" >&2
      return 1
  esac
}

# Temporary files cleanup hooks.
CLEANUP_FILES=()
cleanup() {
  if [[ ${#CLEANUP_FILES[@]} -ne 0 ]]; then
    rm -fr "${CLEANUP_FILES[@]}"
  fi
}

# Executed on exit.
on_exit() {
  local retcode="$1"
  # Always cleanup the CLEANUP_FILES.
  cleanup
}

trap 'retcode=$?; { set +x; } 2>/dev/null; on_exit ${retcode}' INT TERM EXIT


# Install libc++ libraries compiled with msan in the msan_prefix for the current
# compiler version.
cmd_msan_install() {
  local tmpdir=$(mktemp -d)
  CLEANUP_FILES+=("${tmpdir}")
  # Detect the llvm to install:
  detect_clang_version
  # Allow overriding the LLVM checkout.
  local llvm_root="${LLVM_ROOT:-}"
  if [ -z "${llvm_root}" ]; then
    local llvm_tag="llvmorg-${CLANG_VERSION}.0.0"
    case "${CLANG_VERSION}" in
      "6.0")
        llvm_tag="llvmorg-6.0.1"
        ;;
      "7")
        llvm_tag="llvmorg-7.0.1"
        ;;
    esac
    local llvm_targz="${tmpdir}/${llvm_tag}.tar.gz"
    curl -L --show-error -o "${llvm_targz}" \
      "https://github.com/llvm/llvm-project/archive/${llvm_tag}.tar.gz"
    tar -C "${tmpdir}" -zxf "${llvm_targz}"
    llvm_root="${tmpdir}/llvm-project-${llvm_tag}"
  fi

  local msan_prefix="${HOME}/.msan/${CLANG_VERSION}"
  rm -rf "${msan_prefix}"

  declare -A CMAKE_EXTRAS
  CMAKE_EXTRAS[libcxx]="\
    -DLIBCXX_CXX_ABI=libstdc++ \
    -DLIBCXX_INSTALL_EXPERIMENTAL_LIBRARY=ON \
    -DLIBCXX_INCLUDE_BENCHMARKS=OFF"

  for project in libcxx; do
    local proj_build="${tmpdir}/build-${project}"
    local proj_dir="${llvm_root}/${project}"
    mkdir -p "${proj_build}"
    cmake -B"${proj_build}" -H"${proj_dir}" \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_USE_SANITIZER=Memory \
      -DLLVM_PATH="${llvm_root}/llvm" \
      -DLLVM_CONFIG_PATH="$(which llvm-config llvm-config-7 llvm-config-6.0 | \
                            head -n1)" \
      -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}" \
      -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}" \
      -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS}" \
      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
      -DCMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS}" \
      -DCMAKE_SHARED_LINKER_FLAGS="${CMAKE_SHARED_LINKER_FLAGS}" \
      -DCMAKE_INSTALL_PREFIX="${msan_prefix}" \
      ${CMAKE_EXTRAS[${project}]}
    cmake --build "${proj_build}"
    ninja -C "${proj_build}" install
  done
}

cmd_msan() {
  detect_clang_version
  local msan_prefix="${HOME}/.msan/${CLANG_VERSION}"
  if [[ ! -d "${msan_prefix}" || -e "${msan_prefix}/lib/libc++abi.a" ]]; then
    # Install msan libraries for this version if needed or if an older version
    # with libc++abi was installed.
    cmd_msan_install
  fi

  local msan_c_flags=(
    -fsanitize=memory
    -fno-omit-frame-pointer

    -g
    -DMEMORY_SANITIZER

    # Force gtest to not use the cxxbai.
    -DGTEST_HAS_CXXABI_H_=0

    -fsanitize-memory-track-origins
  )

  local msan_cxx_flags=(
    "${msan_c_flags[@]}"

    # Some C++ sources don't use the std at all, so the -stdlib=libc++ is unused
    # in those cases. Ignore the warning.
    -Wno-unused-command-line-argument
    -stdlib=libc++

    # We include the libc++ from the msan directory instead, so we don't want
    # the std includes.
    -nostdinc++
    -cxx-isystem"${msan_prefix}/include/c++/v1"
  )

  local msan_linker_flags=(
    -L"${msan_prefix}"/lib
    -Wl,-rpath -Wl,"${msan_prefix}"/lib/
  )

  CMAKE_C_FLAGS+=" ${msan_c_flags[@]} ${UBSAN_FLAGS[@]}"
  CMAKE_CXX_FLAGS+=" ${msan_cxx_flags[@]} ${UBSAN_FLAGS[@]}"
  CMAKE_EXE_LINKER_FLAGS+=" ${msan_linker_flags[@]}"
  CMAKE_MODULE_LINKER_FLAGS+=" ${msan_linker_flags[@]}"
  CMAKE_SHARED_LINKER_FLAGS+=" ${msan_linker_flags[@]}"
  cmake_configure "$@" \
    -DCMAKE_CROSSCOMPILING=1 -DRUN_HAVE_STD_REGEX=0 -DRUN_HAVE_POSIX_REGEX=0 \
    -DCMAKE_REQUIRED_LINK_OPTIONS="${msan_linker_flags[@]}"
}

cmake_configure() {
  local args=(
    -B"${BUILD_DIR}" -H"${MYDIR}"
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    -G Ninja
    -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}"
    -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}"
    -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS}"
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"
    -DCMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS}"
    -DCMAKE_MODULE_LINKER_FLAGS="${CMAKE_MODULE_LINKER_FLAGS}"
    -DCMAKE_SHARED_LINKER_FLAGS="${CMAKE_SHARED_LINKER_FLAGS}"
    -DGEMMA_ENABLE_TESTS=ON
  )

  cmake "${args[@]}" "$@"
}

cmd_opt() {
  CMAKE_BUILD_TYPE="RelWithDebInfo"
  cmake_configure "$@"
}

cmd_asan() {
  CMAKE_C_FLAGS+=" -g -DADDRESS_SANITIZER -fsanitize=address  ${UBSAN_FLAGS[@]}"
  CMAKE_CXX_FLAGS+=" -g -DADDRESS_SANITIZER -fsanitize=address \
    ${UBSAN_FLAGS[@]}"
  cmake_configure "$@"
}

cmd_tsan() {
  SANITIZER="tsan"
  local tsan_args=(
    -g
    -DTHREAD_SANITIZER
    ${UBSAN_FLAGS[@]}
    -fsanitize=thread
  )
  CMAKE_C_FLAGS+=" ${tsan_args[@]}"
  CMAKE_CXX_FLAGS+=" ${tsan_args[@]}"

  CMAKE_BUILD_TYPE="RelWithDebInfo"
  cmake_configure "$@"
}

main() {
  local cmd="${1:-}"
  if [[ -z "${cmd}" ]]; then
    cat >&2 <<EOF
Use: $0 CMD

Where cmd is one of:
 opt       Build and test a Release with symbols build.
 asan      Build and test an ASan (AddressSanitizer) build.
 msan      Build and test an MSan (MemorySanitizer) build. Needs to have msan
           c++ libs installed with msan_install first.
 msan_install Install the libc++ libraries required to build in msan mode. This
              needs to be done once.
 tsan      Build and test a TSan (ThreadSanitizer) build.

You can pass some optional environment variables as well:
 - BUILD_DIR: The output build directory (by default "$$repo/build")
 - CMAKE_FLAGS: Convenience flag to pass both CMAKE_C_FLAGS and CMAKE_CXX_FLAGS.

These optional environment variables are forwarded to the cmake call as
parameters:
 - CMAKE_BUILD_TYPE
 - CMAKE_C_FLAGS
 - CMAKE_CXX_FLAGS
 - CMAKE_C_COMPILER
 - CMAKE_CXX_COMPILER
 - CMAKE_EXE_LINKER_FLAGS
 - CMAKE_MODULE_LINKER_FLAGS
 - CMAKE_SHARED_LINKER_FLAGS

Example:
  BUILD_DIR=/tmp/build $0 opt
EOF
    exit 1
  fi

  cmd="cmd_${cmd}"
  shift
  set -x
  "${cmd}" "$@"
}

main "$@"
