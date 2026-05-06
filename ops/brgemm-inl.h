// Copyright 2026 DeepMind Technologies Limited.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// BRGeMM dispatch for BF16 MatMul on Intel AMX/AVX-512.

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "ops/brgemm.h"
#include "ops/matmul.h"
#include "util/mat.h"
#include "util/threading_context.h"
#include "util/zones.h"
#include "hwy/base.h"

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_BRGEMM_TOGGLE) == defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_BRGEMM_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_BRGEMM_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_BRGEMM_TOGGLE
#endif

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

#if GEMMA_ONEDNN_BRGEMM

static bool MakeBrgemm(dnnl::ukernel::brgemm& brg, int64_t m, int64_t n,
                        int64_t k, int64_t batch, int64_t lda, int64_t ldb,
                        int64_t ldc, dnnl::memory::data_type a_dt,
                        dnnl::memory::data_type b_dt,
                        dnnl::memory::data_type c_dt, bool add_C) {
  try {
    brg = dnnl::ukernel::brgemm(m, n, k, batch, lda, ldb, ldc, a_dt, b_dt,
                                 c_dt, true);
    if (!brg) {
      HWY_WARN("BRGeMM: kernel creation failed m=%lld n=%lld k=%lld.",
               static_cast<long long>(m), static_cast<long long>(n),
               static_cast<long long>(k));
      return false;
    }
    brg.set_add_C(add_C);
    if (!brg.finalize()) {
      HWY_WARN("BRGeMM: kernel finalize failed m=%lld n=%lld k=%lld.",
               static_cast<long long>(m), static_cast<long long>(n),
               static_cast<long long>(k));
      return false;
    }
    brg.generate();
    return true;
  } catch (...) {
    HWY_WARN("BRGeMM: kernel JIT exception m=%lld n=%lld k=%lld.",
             static_cast<long long>(m), static_cast<long long>(n),
             static_cast<long long>(k));
    return false;
  }
}

// JIT-compiles brgemm kernels, B-packing transforms, and offset tables for
// the given matrix dimensions and tiling config. Returns false on failure.
static HWY_NOINLINE bool InitBRGeMMKernels(
    const BRGeMMConfig& cfg, size_t M, size_t K, size_t N, size_t lda,
    size_t ldb_orig, BRGeMMKernelEntry& ke) {
  using dnnl::ukernel::brgemm;
  using dnnl::ukernel::pack_type;
  using dnnl::ukernel::transform;

  ke.K_blk = cfg.K_blk;
  ke.N_blk = cfg.N_blk;
  ke.M_blk = std::min(cfg.M_blk, M);
  ke.div_M_blk = hwy::Divisor(ke.M_blk);
  ke.div_N_blk = hwy::Divisor(ke.N_blk);
  ke.div_K_blk = hwy::Divisor(ke.K_blk);

  ke.M_tail = ke.div_M_blk.Remainder(M);
  ke.N_tail = ke.div_N_blk.Remainder(N);
  ke.K_tail = ke.div_K_blk.Remainder(K);

  // Floor division: K_tail remainder is handled by a dedicated brg_ktail
  // kernel rather than padding K, avoiding extra memory writes to zero-pad
  // A and B along the K dimension.
  ke.K_chunks = ke.div_K_blk.Divide(K);
  ke.N_full_tiles = ke.div_N_blk.Divide(N);
  ke.M_full_tiles = ke.div_M_blk.Divide(M);
  ke.N_total_tiles = ke.N_full_tiles + (ke.N_tail ? 1 : 0);
  ke.M_total_tiles = ke.M_full_tiles + (ke.M_tail ? 1 : 0);
  ke.N_padded = ke.N_total_tiles * ke.N_blk;

  if (ke.M_total_tiles == 0 || ke.N_total_tiles == 0 ||
      (ke.K_chunks == 0 && ke.K_tail == 0)) {
    return false;
  }

  ke.K_super_size = std::min(cfg.batch_size, ke.K_chunks);
  ke.K_super_blocks = (ke.K_chunks > 0) ? ke.K_chunks / ke.K_super_size : 0;
  ke.K_super_rem = (ke.K_chunks > 0) ? ke.K_chunks % ke.K_super_size : 0;
  ke.batch_full = ke.K_super_size;
  ke.batch_rem = ke.K_super_rem;

  const auto a_dt = dnnl::memory::data_type::bf16;
  const auto b_dt = dnnl::memory::data_type::bf16;
  const auto c_dt = dnnl::memory::data_type::f32;
  ke.a_dt_size = dnnl::memory::data_type_size(a_dt);
  ke.b_dt_size = dnnl::memory::data_type_size(b_dt);

  const auto pack = brgemm::get_B_pack_type(a_dt, b_dt);
  if (pack == pack_type::undef) return false;
  ke.need_pack = (pack != pack_type::no_trans);

  ke.lda = lda;
  ke.ldb_orig = ldb_orig;

  // Indexed by tail flag: [0] = full tile size, [1] = tail size (or full if
  // no tail). Separate kernels are JIT-compiled for full vs. tail tile widths
  // along both M and N dimensions.
  ke.m_sizes[0] = ke.M_blk;
  ke.m_sizes[1] = ke.M_tail ? ke.M_tail : ke.M_blk;
  ke.n_sizes[0] = ke.N_blk;
  ke.n_sizes[1] = ke.N_tail ? ke.N_tail : ke.N_blk;
  const int64_t ldb_for[2] = {static_cast<int64_t>(ke.N_blk),
                               static_cast<int64_t>(ke.N_tail ? ke.N_tail : ke.N_blk)};
  const int64_t ldc_for[2] = {static_cast<int64_t>(ke.N_blk),
                               static_cast<int64_t>(ke.N_tail ? ke.N_tail : ke.N_blk)};

  // JIT a brgemm kernel for each (mi, ni) where mi/ni indicate whether we
  // are processing the M-tail or N-tail: 0 = full block, 1 = tail block.
  // Skipped when the corresponding tail is zero (no partial tile exists).
  size_t max_sp = 0;
  for (int mi = 0; mi < 2; ++mi) {
    for (int ni = 0; ni < 2; ++ni) {
      if (mi == 1 && ke.M_tail == 0) continue;
      if (ni == 1 && ke.N_tail == 0) continue;
      if (mi == 0 && ke.M_full_tiles == 0) continue;
      if (ni == 0 && ke.N_full_tiles == 0) continue;

      const int64_t ms = static_cast<int64_t>(ke.m_sizes[mi]);
      const int64_t ns = static_cast<int64_t>(ke.n_sizes[ni]);

      if (ke.K_chunks > 0) {
        if (!MakeBrgemm(ke.brg_first_all[mi][ni], ms, ns,
                        static_cast<int64_t>(ke.K_blk),
                        static_cast<int64_t>(ke.K_super_size),
                        static_cast<int64_t>(ke.lda), ldb_for[ni],
                        ldc_for[ni], a_dt, b_dt, c_dt, false)) {
          return false;
        }
        max_sp = std::max(max_sp,
                          ke.brg_first_all[mi][ni].get_scratchpad_size());
      }
      if (ke.K_super_blocks > 1) {
        if (!MakeBrgemm(ke.brg_full[mi][ni], ms, ns,
                        static_cast<int64_t>(ke.K_blk),
                        static_cast<int64_t>(ke.batch_full),
                        static_cast<int64_t>(ke.lda), ldb_for[ni],
                        ldc_for[ni], a_dt, b_dt, c_dt, true)) {
          return false;
        }
        max_sp =
            std::max(max_sp, ke.brg_full[mi][ni].get_scratchpad_size());
      }
      if (ke.K_super_rem > 0) {
        const bool rem_is_first = (ke.K_super_blocks == 0);
        auto& target = rem_is_first ? ke.brg_first_rem[mi][ni]
                                    : ke.brg_rem[mi][ni];
        if (!MakeBrgemm(target, ms, ns, static_cast<int64_t>(ke.K_blk),
                        static_cast<int64_t>(ke.batch_rem),
                        static_cast<int64_t>(ke.lda), ldb_for[ni],
                        ldc_for[ni], a_dt, b_dt, c_dt,
                        !rem_is_first)) {
          return false;
        }
        max_sp = std::max(max_sp, target.get_scratchpad_size());
      }
      if (ke.K_tail > 0) {
        const bool add_c = (ke.K_chunks > 0);
        if (!MakeBrgemm(ke.brg_ktail[mi][ni], ms, ns,
                        static_cast<int64_t>(ke.K_tail), 1,
                        static_cast<int64_t>(ke.lda), ldb_for[ni],
                        ldc_for[ni], a_dt, b_dt, c_dt,
                        add_c)) {
          return false;
        }
        max_sp =
            std::max(max_sp, ke.brg_ktail[mi][ni].get_scratchpad_size());
      }
    }
  }
  ke.scratchpad_size = max_sp + 64;

  // Create B-packing transforms.
  if (ke.need_pack) {
    for (int ni = 0; ni < 2; ++ni) {
      if (ni == 1 && ke.N_tail == 0) continue;
      if (ni == 0 && ke.N_full_tiles == 0) continue;

      const int64_t ns = static_cast<int64_t>(ke.n_sizes[ni]);
      if (ke.K_chunks > 0) {
        const int64_t K_full =
            static_cast<int64_t>(ke.K_chunks * ke.K_blk);
        try {
          ke.pack_B[ni] = transform(K_full, ns, pack_type::trans,
                                     static_cast<int64_t>(ke.ldb_orig),
                                     ldb_for[ni], b_dt, b_dt);
          if (!ke.pack_B[ni]) return false;
          ke.pack_B[ni].generate();
          ke.blocked_B_size[ni] = static_cast<size_t>(ldb_for[ni]) *
                                  ke.K_blk * ke.b_dt_size;
        } catch (...) {
          return false;
        }
      }
      if (ke.K_tail > 0) {
        try {
          ke.pack_B_ktail[ni] = transform(
              static_cast<int64_t>(ke.K_tail), ns, pack_type::trans,
              static_cast<int64_t>(ke.ldb_orig), ldb_for[ni], b_dt, b_dt);
          if (!ke.pack_B_ktail[ni]) return false;
          ke.pack_B_ktail[ni].generate();
          ke.blocked_B_ktail_size[ni] =
              static_cast<size_t>(ldb_for[ni]) * ke.K_tail * ke.b_dt_size;
        } catch (...) {
          return false;
        }
      }
    }
  }

  // Precompute A/B offset tables for each K-super-block.
  for (int ni = 0; ni < 2; ++ni) {
    if (ni == 1 && ke.N_tail == 0) continue;
    if (ni == 0 && ke.N_full_tiles == 0) continue;
    const size_t cur_n = ke.n_sizes[ni];

    if (ke.K_chunks > 0) {
      ke.offsets_first_all[ni].resize(ke.K_super_size);
      for (size_t i = 0; i < ke.K_super_size; ++i) {
        const int64_t a_off =
            static_cast<int64_t>(i * ke.K_blk * ke.a_dt_size);
        const int64_t b_off =
            ke.need_pack
                ? static_cast<int64_t>(i * ke.blocked_B_size[ni])
                : static_cast<int64_t>(i * cur_n * ke.K_blk * ke.b_dt_size);
        ke.offsets_first_all[ni][i] = {a_off, b_off};
      }
    }

    if (ke.K_super_blocks > 1) {
      ke.offsets_full[ni].resize(ke.K_super_blocks - 1);
      for (size_t ks = 1; ks < ke.K_super_blocks; ++ks) {
        auto& tbl = ke.offsets_full[ni][ks - 1];
        tbl.resize(ke.batch_full);
        const size_t k_start = ks * ke.K_super_size;
        for (size_t i = 0; i < ke.batch_full; ++i) {
          const size_t k_idx = k_start + i;
          const int64_t a_off =
              static_cast<int64_t>(k_idx * ke.K_blk * ke.a_dt_size);
          const int64_t b_off =
              ke.need_pack
                  ? static_cast<int64_t>(k_idx * ke.blocked_B_size[ni])
                  : static_cast<int64_t>(k_idx * cur_n * ke.K_blk *
                                         ke.b_dt_size);
          tbl[i] = {a_off, b_off};
        }
      }
    }

    if (ke.K_super_rem > 0) {
      const size_t k_base = ke.K_super_blocks * ke.K_super_size;
      auto& rem_tbl = (ke.K_super_blocks == 0) ? ke.offsets_first_rem[ni]
                                                : ke.offsets_rem[ni];
      rem_tbl.resize(ke.K_super_rem);
      for (size_t i = 0; i < ke.K_super_rem; ++i) {
        const size_t k_idx = k_base + i;
        const int64_t a_off =
            static_cast<int64_t>(k_idx * ke.K_blk * ke.a_dt_size);
        const int64_t b_off =
            ke.need_pack
                ? static_cast<int64_t>(k_idx * ke.blocked_B_size[ni])
                : static_cast<int64_t>(k_idx * cur_n * ke.K_blk *
                                       ke.b_dt_size);
        rem_tbl[i] = {a_off, b_off};
      }
    }
  }

  return true;
}

template <typename TA, typename TB, typename TC>
static HWY_NOINLINE bool DoMatMul_BRGeMM(
    const MatPtrT<TA>& A, const MatPtrT<TB>& B, RowPtrs<TC> C, size_t M,
    size_t K, size_t N, float scale, const float* HWY_RESTRICT add,
    const BRGeMMConfig& cfg, ThreadingContext& ctx, size_t cluster_idx) {
  using dnnl::ukernel::brgemm;

  // Level-1 cache: kernels keyed on (M, K, N, config).
  const BRGeMMKernelKey kern_key{M, K, N, cfg.M_blk, cfg.N_blk, cfg.K_blk,
                                 cfg.batch_size};
  auto& kern_cache = GetBRGeMMKernelCache();
  auto kern_it = kern_cache.find(kern_key);

  if (kern_it == kern_cache.end()) {
    BRGeMMKernelEntry ke;
    if (!InitBRGeMMKernels(cfg, M, K, N, A.Stride(), B.Stride(), ke)) {
      return false;
    }
    kern_it = kern_cache.emplace(kern_key, std::move(ke)).first;
  }

  BRGeMMKernelEntry& ke = kern_it->second;

  // Level-2 cache: packed B keyed on (B_ptr, K, N, config).
  const uint8_t* A_base = reinterpret_cast<const uint8_t*>(A.Row(0));
  const uint8_t* B_base = reinterpret_cast<const uint8_t*>(B.Row(0));

  const BRGeMMPackedBKey pb_key{reinterpret_cast<uintptr_t>(B_base), K, N,
                                ke.K_blk, ke.N_blk};
  auto& pb_cache = GetBRGeMMPackedBCache();
  auto pb_it = pb_cache.find(pb_key);

  if (pb_it == pb_cache.end()) {
    BRGeMMPackedBEntry pe;
    pe.B_tile_offset.resize(ke.N_total_tiles, 0);
    pe.B_ktail_offset.resize(ke.N_total_tiles, 0);

    if (ke.need_pack) {
      size_t total_packed = 0;
      for (size_t nt = 0; nt < ke.N_total_tiles; ++nt) {
        const int ni = (nt < ke.N_full_tiles) ? 0 : 1;
        pe.B_tile_offset[nt] = total_packed;
        if (ke.K_chunks > 0)
          total_packed += ke.blocked_B_size[ni] * ke.K_chunks;
        pe.B_ktail_offset[nt] = total_packed;
        if (ke.K_tail > 0) total_packed += ke.blocked_B_ktail_size[ni];
      }

      pe.B_packed_buf.Resize(total_packed);
      uint8_t* B_packed = pe.B_packed_buf.data();
      if (!B_packed) {
        HWY_WARN("BRGeMM: packed B allocation failed.");
        return false;
      }

      for (size_t nt = 0; nt < ke.N_total_tiles; ++nt) {
        const int ni = (nt < ke.N_full_tiles) ? 0 : 1;
        const size_t b_row = (nt < ke.N_full_tiles)
                                 ? nt * ke.N_blk
                                 : ke.N_full_tiles * ke.N_blk;
        const uint8_t* B_in =
            B_base + b_row * ke.ldb_orig * ke.b_dt_size;

        try {
          if (ke.K_chunks > 0) {
            ke.pack_B[ni].execute(const_cast<uint8_t*>(B_in),
                                  B_packed + pe.B_tile_offset[nt]);
          }
          if (ke.K_tail > 0) {
            const uint8_t* B_in_ktail =
                B_in + ke.K_chunks * ke.K_blk * ke.b_dt_size;
            ke.pack_B_ktail[ni].execute(const_cast<uint8_t*>(B_in_ktail),
                                        B_packed + pe.B_ktail_offset[nt]);
          }
        } catch (...) {
          HWY_WARN("BRGeMM: B-packing execution failed.");
          return false;
        }
      }
    }

    pb_it = pb_cache.emplace(pb_key, std::move(pe)).first;
  }

  const BRGeMMPackedBEntry& pe = pb_it->second;
  const uint8_t* B_packed =
      ke.need_pack ? pe.B_packed_buf.data() : nullptr;

  std::vector<std::pair<dnnl::memory::dim, dnnl::memory::dim>> offsets_ktail(1);
  if (ke.K_tail > 0) offsets_ktail[0] = {0, 0};

  // Execute one (m, n) tile for a given K-super-block.
  const auto execute_tile = [&](size_t m_start, size_t n_start,
                                size_t k_super, float* temp_C,
                                uint8_t* scratch) HWY_ATTR {
    const size_t m_tile_idx = ke.div_M_blk.Divide(m_start);
    const size_t n_tile_idx = ke.div_N_blk.Divide(n_start);
    const int mi = (m_tile_idx < ke.M_full_tiles) ? 0 : 1;
    const int ni = (n_tile_idx < ke.N_full_tiles) ? 0 : 1;
    const size_t cur_m = ke.m_sizes[mi];
    const size_t cur_n = ke.n_sizes[ni];

    const size_t real_m = (m_tile_idx < ke.M_full_tiles)
                              ? m_tile_idx * ke.M_blk
                              : ke.M_full_tiles * ke.M_blk;
    const size_t real_n = (n_tile_idx < ke.N_full_tiles)
                              ? n_tile_idx * ke.N_blk
                              : ke.N_full_tiles * ke.N_blk;

    const uint8_t* A_tile =
        A_base + real_m * ke.lda * ke.a_dt_size;
    const void* B_tile =
        ke.need_pack
            ? static_cast<const void*>(B_packed +
                                       pe.B_tile_offset[n_tile_idx])
            : static_cast<const void*>(
                  B_base +
                  real_n * ke.ldb_orig * ke.b_dt_size);

    float* C_tile_ptr = temp_C;
    const size_t k_total =
        ke.K_super_blocks + (ke.K_super_rem > 0 ? 1 : 0);

    if (k_super < ke.K_super_blocks) {
      if (k_super == 0) {
        ke.brg_first_all[mi][ni].execute(A_tile, const_cast<void*>(B_tile),
                                         ke.offsets_first_all[ni], C_tile_ptr,
                                         scratch);
      } else {
        ke.brg_full[mi][ni].execute(A_tile, const_cast<void*>(B_tile),
                                    ke.offsets_full[ni][k_super - 1],
                                    C_tile_ptr, scratch);
      }
    } else if (ke.K_super_rem > 0 && k_super == ke.K_super_blocks) {
      if (ke.K_super_blocks == 0) {
        ke.brg_first_rem[mi][ni].execute(A_tile, const_cast<void*>(B_tile),
                                         ke.offsets_first_rem[ni], C_tile_ptr,
                                         scratch);
      } else {
        ke.brg_rem[mi][ni].execute(A_tile, const_cast<void*>(B_tile),
                                   ke.offsets_rem[ni], C_tile_ptr, scratch);
      }
    }

    const bool is_last = (k_total > 0) ? (k_super == k_total - 1) : true;
    if (is_last) {
      if (ke.K_tail > 0) {
        const uint8_t* A_ktail =
            A_tile + ke.K_chunks * ke.K_blk * ke.a_dt_size;
        const void* B_ktail =
            ke.need_pack
                ? static_cast<const void*>(B_packed +
                                           pe.B_ktail_offset[n_tile_idx])
                : static_cast<const void*>(
                      B_base + (real_n * ke.ldb_orig +
                                ke.K_chunks * ke.K_blk) *
                                   ke.b_dt_size);
        ke.brg_ktail[mi][ni].execute(A_ktail, const_cast<void*>(B_ktail),
                                     offsets_ktail, C_tile_ptr, scratch);
      }

      // Scale and copy temp_C to output.
      const hn::ScalableTag<float> df;
      const auto vscale = hn::Set(df, scale);
      const size_t lanes = hn::Lanes(df);
      for (size_t m = 0; m < cur_m; ++m) {
        TC* C_row = C.Row(real_m + m) + real_n;
        const float* t_row = C_tile_ptr + m * cur_n;
        const float* add_row = add ? add + real_n : nullptr;
        size_t n = 0;
        if (add_row) {
          for (; n + lanes <= cur_n; n += lanes) {
            const auto v = hn::Load(df, t_row + n);
            const auto va = hn::Load(df, add_row + n);
            const auto result = hn::MulAdd(v, vscale, va);
            if constexpr (hwy::IsSame<TC, float>()) {
              hn::Store(result, df, HWY_RCAST_ALIGNED(float*, C_row) + n);
            } else {
              const hn::Rebind<TC, decltype(df)> dc;
              hn::Store(hn::DemoteTo(dc, result), dc, C_row + n);
            }
          }
          for (; n < cur_n; ++n) {
            float val = t_row[n] * scale + add_row[n];
            C_row[n] = hwy::ConvertScalarTo<TC>(val);
          }
        } else {
          for (; n + lanes <= cur_n; n += lanes) {
            const auto v = hn::Load(df, t_row + n);
            const auto result = hn::Mul(v, vscale);
            if constexpr (hwy::IsSame<TC, float>()) {
              hn::Store(result, df, HWY_RCAST_ALIGNED(float*, C_row) + n);
            } else {
              const hn::Rebind<TC, decltype(df)> dc;
              hn::Store(hn::DemoteTo(dc, result), dc, C_row + n);
            }
          }
          for (; n < cur_n; ++n) {
            float val = t_row[n] * scale;
            C_row[n] = hwy::ConvertScalarTo<TC>(val);
          }
        }
      }
    }
  };

  // Parallel dispatch: K-super outer, N middle, M inner (keeps B in L2).
  const size_t k_total_supers =
      ke.K_super_blocks + (ke.K_super_rem > 0 ? 1 : 0);
  const size_t k_iters = (k_total_supers > 0) ? k_total_supers : size_t{1};

  const size_t num_threads = ctx.pools.MaxWorkersPerCluster();
  const size_t total_n_tiles = ke.N_total_tiles;
  const size_t total_m_tiles = ke.M_total_tiles;
  const size_t n_tasks =
      std::max(size_t{1}, std::min(total_n_tiles, num_threads));

  const hwy::pool::Caller caller =
      ctx.pool_callers.Get(Callers::kBRGeMM);

  ParallelForWithinCluster(
      n_tasks, ctx, cluster_idx, caller,
      [&](uint64_t task_idx, size_t /*worker*/) HWY_ATTR {
        const size_t tiles_per_task = total_n_tiles / n_tasks;
        const size_t extra = total_n_tiles % n_tasks;
        const size_t n_begin =
            task_idx * tiles_per_task +
            std::min(static_cast<size_t>(task_idx), extra);
        const size_t n_end =
            n_begin + tiles_per_task + (task_idx < extra ? 1 : 0);

        auto& tbufs = GetBRGeMMThreadBufs();
        tbufs.MaybeSetHwContext(ke.brg_first_all[0][0]);
        uint8_t* sp = tbufs.EnsureScratch(ke.scratchpad_size);

        const size_t n_tiles_in_range = n_end - n_begin;
        const size_t total_tc = total_m_tiles * n_tiles_in_range;
        float* tc_base = tbufs.EnsureTempC(total_tc);

        for (size_t ks = 0; ks < k_iters; ++ks) {
          size_t n_idx = 0;
          for (size_t nt = n_begin; nt < n_end; ++nt) {
            const size_t n = nt * ke.N_blk;
            for (size_t mt = 0; mt < total_m_tiles; ++mt) {
              const size_t m = mt * ke.M_blk;
              float* temp_C =
                  tc_base + (mt * n_tiles_in_range + n_idx) *
                                BRGeMMThreadBufs::kMaxTempCSize;
              execute_tile(m, n, ks, temp_C, sp);
            }
            ++n_idx;
          }
        }
      });

  dnnl::ukernel::brgemm::release_hw_context();
  auto& main_bufs = GetBRGeMMThreadBufs();
  main_bufs.hw_ctx_kernel = nullptr;
  return true;
}

#endif  // GEMMA_ONEDNN_BRGEMM

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
