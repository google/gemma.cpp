// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from code by abetusk (BSD-2-Clause) in
// https://github.com/jakubcerveny/gilbert.

#ifndef THIRD_PARTY_GEMMA_CPP_OPS_GILBERT_H_
#define THIRD_PARTY_GEMMA_CPP_OPS_GILBERT_H_

namespace gcpp {

// Maps a 1D Hilbert curve index to 2D coordinates (x, y).
int gilbert_d2xy(int* x, int* y, int idx, int w, int h);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_OPS_GILBERT_H_