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

// OneDNN BRGeMM micro-kernel integration for MatMul on Intel AMX/AVX-512.
// Enabled at compile time via GEMMA_ONEDNN_BRGEMM=1 (Bazel: --define gemma_onednn_brgemm=1).

#ifndef THIRD_PARTY_GEMMA_CPP_OPS_BRGEMM_H_
#define THIRD_PARTY_GEMMA_CPP_OPS_BRGEMM_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hwy/base.h"

#if GEMMA_ONEDNN_BRGEMM
#include <sys/mman.h>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_ukernel.hpp"
#endif  // GEMMA_ONEDNN_BRGEMM

namespace gcpp {

struct BRGeMMConfig {
  int64_t M_blk;
  int64_t N_blk;
  int64_t K_blk;
  int64_t batch_size;
  int64_t par_m;
};

#if GEMMA_ONEDNN_BRGEMM

// Generates autotuning candidates. Fixed: N_blk=32, K_blk=32 (AMX BF16).
// Tunable: M_blk in {32,64}, batch_size in {16,32,64,128,256}.
inline std::vector<BRGeMMConfig> BRGeMMCandidates(size_t M, size_t K,
                                                   size_t N) {
  std::vector<BRGeMMConfig> out;
  static constexpr int64_t kNBlk = 32;
  static constexpr int64_t kKBlk = 32;
  static constexpr int64_t kMBlkValues[] = {32, 64};
  static constexpr int64_t kBatchValues[] = {16, 32, 64, 128, 256};

  const int64_t k_chunks = static_cast<int64_t>(K) / kKBlk;
  for (int64_t mb : kMBlkValues) {
    if (mb > static_cast<int64_t>(M)) continue;
    if (kNBlk > static_cast<int64_t>(N)) continue;
    for (int64_t bs : kBatchValues) {
      const int64_t eff_bs =
          (k_chunks > 0) ? std::min(bs, k_chunks) : int64_t{1};
      bool dup = false;
      for (const auto& c : out) {
        if (c.M_blk == mb && c.batch_size == eff_bs) {
          dup = true;
          break;
        }
      }
      if (dup) continue;
      out.push_back({mb, kNBlk, kKBlk, eff_bs, /*par_m=*/1});
    }
  }
  if (out.empty()) {
    out.push_back({static_cast<int64_t>(std::min(M, size_t{32})),
                   static_cast<int64_t>(std::min(N, size_t{32})), 32, 1, 1});
  }
  return out;
}

// Hugepage-backed buffer via mmap with MADV_HUGEPAGE for packed-B matrices.
class HugePageBuffer {
 public:
  HugePageBuffer() = default;
  ~HugePageBuffer() {
    if (ptr_ && size_) munmap(ptr_, size_);
  }

  HugePageBuffer(HugePageBuffer&& o) noexcept
      : ptr_(o.ptr_), size_(o.size_) {
    o.ptr_ = nullptr;
    o.size_ = 0;
  }
  HugePageBuffer& operator=(HugePageBuffer&& o) noexcept {
    if (this != &o) {
      if (ptr_ && size_) munmap(ptr_, size_);
      ptr_ = o.ptr_;
      size_ = o.size_;
      o.ptr_ = nullptr;
      o.size_ = 0;
    }
    return *this;
  }

  HugePageBuffer(const HugePageBuffer&) = delete;
  HugePageBuffer& operator=(const HugePageBuffer&) = delete;

  void Resize(size_t n) {
    if (ptr_ && size_) munmap(ptr_, size_);
    static constexpr size_t kHugePageSize = 2u << 20;
    size_ = (n + kHugePageSize - 1) & ~(kHugePageSize - 1);
    ptr_ = static_cast<uint8_t*>(mmap(nullptr, size_, PROT_READ | PROT_WRITE,
                                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (ptr_ == MAP_FAILED) {
      ptr_ = nullptr;
      size_ = 0;
      return;
    }
    madvise(ptr_, size_, MADV_HUGEPAGE);
    for (size_t off = 0; off < size_; off += kHugePageSize) {
      static_cast<volatile uint8_t*>(ptr_)[off] = 0;
    }
  }

  uint8_t* data() { return ptr_; }
  const uint8_t* data() const { return ptr_; }
  size_t size() const { return size_; }

 private:
  uint8_t* ptr_ = nullptr;
  size_t size_ = 0;
};

// Kernel cache key: identifies a JIT-compiled kernel set.
struct BRGeMMKernelKey {
  size_t M, K, N;
  int64_t M_blk, N_blk, K_blk, batch_size;
  bool operator==(const BRGeMMKernelKey& o) const {
    return M == o.M && K == o.K && N == o.N && M_blk == o.M_blk &&
           N_blk == o.N_blk && K_blk == o.K_blk && batch_size == o.batch_size;
  }
};

struct BRGeMMKernelKeyHash {
  size_t operator()(const BRGeMMKernelKey& k) const {
    size_t h = 14695981039346656037ULL;
    h = (h ^ k.M) * 1099511628211ULL;
    h = (h ^ k.K) * 1099511628211ULL;
    h = (h ^ k.N) * 1099511628211ULL;
    h = (h ^ static_cast<size_t>(k.M_blk)) * 1099511628211ULL;
    h = (h ^ static_cast<size_t>(k.N_blk)) * 1099511628211ULL;
    h = (h ^ static_cast<size_t>(k.K_blk)) * 1099511628211ULL;
    h = (h ^ static_cast<size_t>(k.batch_size)) * 1099511628211ULL;
    return h;
  }
};

// Cached JIT-compiled kernels with precomputed tile parameters and offsets.
struct BRGeMMKernelEntry {
  int64_t M_blk, N_blk, K_blk;
  int64_t M_tail, N_tail, K_tail;
  int64_t K_chunks;
  int64_t M_full_tiles, N_full_tiles;
  int64_t M_total_tiles, N_total_tiles;
  int64_t K_super_size, K_super_blocks;
  int64_t K_super_rem;
  int64_t batch_full, batch_rem;
  int64_t m_sizes[2], n_sizes[2];
  int64_t lda;
  int64_t ldb_orig;
  bool need_pack;
  size_t a_dt_size, b_dt_size;
  size_t N_padded;

  // Kernels indexed by [m_tail_flag][n_tail_flag].
  dnnl::ukernel::brgemm brg_first_all[2][2];
  dnnl::ukernel::brgemm brg_full[2][2];
  dnnl::ukernel::brgemm brg_ktail[2][2];
  dnnl::ukernel::brgemm brg_first_rem[2][2];
  dnnl::ukernel::brgemm brg_rem[2][2];

  // B-packing transforms indexed by n_tail_flag.
  dnnl::ukernel::transform pack_B[2], pack_B_ktail[2];
  size_t blocked_B_size[2] = {0, 0};
  size_t blocked_B_ktail_size[2] = {0, 0};

  size_t scratchpad_size = 0;

  using OffsetVec =
      std::vector<std::pair<dnnl::memory::dim, dnnl::memory::dim>>;
  OffsetVec offsets_first_all[2];
  std::vector<OffsetVec> offsets_full[2];
  OffsetVec offsets_first_rem[2];
  OffsetVec offsets_rem[2];
};

// Packed-B cache key.
struct BRGeMMPackedBKey {
  uintptr_t B_ptr;
  size_t K, N;
  int64_t K_blk, N_blk;
  bool operator==(const BRGeMMPackedBKey& o) const {
    return B_ptr == o.B_ptr && K == o.K && N == o.N && K_blk == o.K_blk &&
           N_blk == o.N_blk;
  }
};

struct BRGeMMPackedBKeyHash {
  size_t operator()(const BRGeMMPackedBKey& k) const {
    size_t h = 14695981039346656037ULL;
    h = (h ^ k.B_ptr) * 1099511628211ULL;
    h = (h ^ k.K) * 1099511628211ULL;
    h = (h ^ k.N) * 1099511628211ULL;
    h = (h ^ static_cast<size_t>(k.K_blk)) * 1099511628211ULL;
    h = (h ^ static_cast<size_t>(k.N_blk)) * 1099511628211ULL;
    return h;
  }
};

struct BRGeMMPackedBEntry {
  HugePageBuffer B_packed_buf;
  std::vector<size_t> B_tile_offset;
  std::vector<size_t> B_ktail_offset;
};

// Thread-local buffers for BRGeMM parallel dispatch.
struct BRGeMMThreadBufs {
  static constexpr size_t kMaxTempCSize = 64 * 64;

  std::vector<uint8_t> scratch;
  std::vector<uint8_t> tc_storage;
  bool hw_ctx_set = false;
  const void* hw_ctx_kernel = nullptr;

  uint8_t* EnsureScratch(size_t size) {
    if (scratch.size() < size + 64) scratch.resize(size + 64);
    return scratch.data() +
           (64 - (reinterpret_cast<uintptr_t>(scratch.data()) % 64));
  }

  float* EnsureTempC(size_t n_tiles) {
    const size_t need = n_tiles * kMaxTempCSize * sizeof(float) + 64;
    if (tc_storage.size() < need) tc_storage.resize(need);
    return reinterpret_cast<float*>(
        (reinterpret_cast<uintptr_t>(tc_storage.data()) + 63) &
        ~uintptr_t{63});
  }

  void MaybeSetHwContext(const dnnl::ukernel::brgemm& brg) {
    const void* brg_ptr = &brg;
    if (!hw_ctx_set || hw_ctx_kernel != brg_ptr) {
      brg.set_hw_context();
      hw_ctx_set = true;
      hw_ctx_kernel = brg_ptr;
    }
  }
};

inline BRGeMMThreadBufs& GetBRGeMMThreadBufs() {
  static thread_local BRGeMMThreadBufs bufs;
  return bufs;
}

// Singleton caches. Thread-safety: MatMul is not called concurrently per env.
inline auto& GetBRGeMMKernelCache() {
  static std::unordered_map<BRGeMMKernelKey, BRGeMMKernelEntry,
                            BRGeMMKernelKeyHash>
      cache;
  return cache;
}

inline auto& GetBRGeMMPackedBCache() {
  static std::unordered_map<BRGeMMPackedBKey, BRGeMMPackedBEntry,
                            BRGeMMPackedBKeyHash>
      cache;
  return cache;
}

#endif  // GEMMA_ONEDNN_BRGEMM

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_OPS_BRGEMM_H_
