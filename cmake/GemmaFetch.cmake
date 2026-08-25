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

# Progress reporting for FetchContent dependencies.
#
# A cold configure clones several repositories from GitHub -- roughly 200 MB for
# the top-level build, of which nlohmann/json alone is ~105k objects and can
# take over ten minutes on a slow link. FetchContent hides download output by
# default, so the configure step sits silent and looks like it has stalled.
#
# gemma_fetch() wraps FetchContent_Declare + FetchContent_MakeAvailable with a
# progress bar naming the dependency and its elapsed time, and turns
# FETCHCONTENT_QUIET off so git's own "Receiving objects: ..." counter shows the
# fine-grained progress that CMake cannot know about. Configure with
# -DGEMMA_QUIET_FETCH=ON to restore the old silent behavior.
#
# Set GEMMA_DEP_TOTAL to the number of gemma_fetch() calls before the first one
# so the bar knows its denominator.
#
# Deliberately not include_guard()ed: the examples build gemma.cpp as a
# subproject, and each project that includes this module gets its own counter
# reset in its own directory scope, so the inner build counts 1/5..5/5 rather
# than continuing the outer project's numbering.

include(FetchContent)

option(GEMMA_QUIET_FETCH "Hide dependency download progress during configure" OFF)
set(FETCHCONTENT_QUIET ${GEMMA_QUIET_FETCH})

if(NOT DEFINED GEMMA_DEP_TOTAL OR GEMMA_DEP_TOTAL LESS 1)
  set(GEMMA_DEP_TOTAL 1)
endif()
set(GEMMA_DEP_INDEX 0)

# gemma_fetch(<name> [<FetchContent_Declare args>...])
# Extra arguments are forwarded verbatim to FetchContent_Declare.
function(gemma_fetch name)
  math(EXPR _index "${GEMMA_DEP_INDEX} + 1")
  set(GEMMA_DEP_INDEX ${_index} PARENT_SCOPE)

  # 20-cell bar counting dependencies already available. It steps once per
  # dependency; git prints the intra-clone progress itself.
  math(EXPR _done "(${_index} - 1) * 20 / ${GEMMA_DEP_TOTAL}")
  math(EXPR _pct "(${_index} - 1) * 100 / ${GEMMA_DEP_TOTAL}")
  set(_bar "")
  set(_cell 0)
  while(_cell LESS 20)
    if(_cell LESS _done)
      string(APPEND _bar "=")
    else()
      string(APPEND _bar "-")
    endif()
    math(EXPR _cell "${_cell} + 1")
  endwhile()
  message(STATUS
    "[${_bar}] ${_pct}% | dependency ${_index}/${GEMMA_DEP_TOTAL}: ${name} -- fetching and configuring, please wait")

  string(TIMESTAMP _start "%s")
  # GIT_PROGRESS makes git report transfer progress even though its stderr is
  # not a terminal here. Only meaningful for git downloads, so guard it.
  if("GIT_REPOSITORY" IN_LIST ARGN)
    FetchContent_Declare(${name} ${ARGN} GIT_PROGRESS TRUE)
  else()
    FetchContent_Declare(${name} ${ARGN})
  endif()
  FetchContent_MakeAvailable(${name})
  string(TIMESTAMP _end "%s")
  math(EXPR _elapsed "${_end} - ${_start}")
  message(STATUS "         ${name} ready (${_elapsed}s)")

  # FetchContent sets these in the caller's scope, which here is this function.
  # Hoist them so callers still see e.g. sentencepiece_SOURCE_DIR.
  string(TOLOWER "${name}" _lc)
  set(${_lc}_SOURCE_DIR "${${_lc}_SOURCE_DIR}" PARENT_SCOPE)
  set(${_lc}_BINARY_DIR "${${_lc}_BINARY_DIR}" PARENT_SCOPE)
  set(${_lc}_POPULATED "${${_lc}_POPULATED}" PARENT_SCOPE)
endfunction()
