#!/bin/bash

# Reproduces .clang-format file.
clang-format -style="{BasedOnStyle: Google, SortIncludes: false}" -dump-config > .clang-format
