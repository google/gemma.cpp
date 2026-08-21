// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from code by abetusk (BSD-2-Clause) in
// https://github.com/jakubcerveny/gilbert.

#include "ops/gilbert.h"

#include <math.h>
#include <stdlib.h>

namespace gcpp {

namespace {

int sgn(int x) {
  if (x < 0) {
    return -1;
  }
  if (x > 0) {
    return 1;
  }
  return 0;
}

int gilbert_d2xy_r(int dst_idx, int cur_idx, int* xres, int* yres, int ax,
                   int ay, int bx, int by) {
  int nxt_idx;
  int w, h, x, y, dax, day, dbx, dby, di;
  int ax2, ay2, bx2, by2, w2, h2;

  w = abs(ax + ay);
  h = abs(bx + by);
  x = *xres;
  y = *yres;
  dax = sgn(ax);
  day = sgn(ay);
  dbx = sgn(bx);
  dby = sgn(by);
  di = dst_idx - cur_idx;

  if (h == 1) {
    *xres = x + dax * di;
    *yres = y + day * di;
    return 0;
  }
  if (w == 1) {
    *xres = x + dbx * di;
    *yres = y + dby * di;
    return 0;
  }

  ax2 = ax >> 1;
  ay2 = ay >> 1;
  bx2 = bx >> 1;
  by2 = by >> 1;
  w2 = abs(ax2 + ay2);
  h2 = abs(bx2 + by2);

  if ((2 * w) > (3 * h)) {
    if ((w2 & 1) && (w > 2)) {
      ax2 += dax;
      ay2 += day;
    }
    nxt_idx = cur_idx + abs((ax2 + ay2) * (bx + by));
    if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx)) {
      *xres = x;
      *yres = y;
      return gilbert_d2xy_r(dst_idx, cur_idx, xres, yres, ax2, ay2, bx, by);
    }
    cur_idx = nxt_idx;
    *xres = x + ax2;
    *yres = y + ay2;
    return gilbert_d2xy_r(dst_idx, cur_idx, xres, yres, ax - ax2, ay - ay2, bx,
                          by);
  }

  if ((h2 & 1) && (h > 2)) {
    bx2 += dbx;
    by2 += dby;
  }

  nxt_idx = cur_idx + abs((bx2 + by2) * (ax2 + ay2));
  if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx)) {
    *xres = x;
    *yres = y;
    return gilbert_d2xy_r(dst_idx, cur_idx, xres, yres, bx2, by2, ax2, ay2);
  }
  cur_idx = nxt_idx;

  nxt_idx = cur_idx + abs((ax + ay) * ((bx - bx2) + (by - by2)));
  if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx)) {
    *xres = x + bx2;
    *yres = y + by2;
    return gilbert_d2xy_r(dst_idx, cur_idx, xres, yres, ax, ay, bx - bx2,
                          by - by2);
  }
  cur_idx = nxt_idx;

  *xres = x + (ax - dax) + (bx2 - dbx);
  *yres = y + (ay - day) + (by2 - dby);
  return gilbert_d2xy_r(dst_idx, cur_idx, xres, yres, -bx2, -by2, -(ax - ax2),
                        -(ay - ay2));
}

}  // namespace

int gilbert_d2xy(int* x, int* y, int idx, int w, int h) {
  *x = 0;
  *y = 0;
  if (w >= h) {
    return gilbert_d2xy_r(idx, 0, x, y, w, 0, 0, h);
  }
  return gilbert_d2xy_r(idx, 0, x, y, 0, h, w, 0);
}

}  // namespace gcpp