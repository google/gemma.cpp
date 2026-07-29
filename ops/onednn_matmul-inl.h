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

// OneDNN matmul-primitive dispatch for BF16 MatMul via the threadpool runtime.
// See ops/onednn_matmul.h for the engine, adapter, and reordered weight cache.

#include <stddef.h>
#include <stdint.h>

#include <unordered_map>
#include <vector>

#include "ops/matmul.h"
#include "ops/onednn_matmul.h"
#include "util/mat.h"
#include "util/threading_context.h"
#include "util/zones.h"
#include "hwy/base.h"

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_ONEDNN_MATMUL_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_ONEDNN_MATMUL_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_ONEDNN_MATMUL_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_ONEDNN_MATMUL_TOGGLE
#endif

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

#if GEMMA_ONEDNN_MATMUL

// Thread-local byte buffer for oneDNN's user-managed scratchpad.
inline std::vector<uint8_t>& GetOneDnnScratchpad() {
  static thread_local std::vector<uint8_t> scratch;
  return scratch;
}

// oneDNN data type for the C element type (only f32/bf16 outputs are used).
template <typename TC>
constexpr dnnl::memory::data_type OneDnnDstType() {
  if constexpr (hwy::IsSame<TC, float>()) {
    return dnnl::memory::data_type::f32;
  } else {
    static_assert(IsBF16<TC>(), "OneDnn matmul path expects f32 or bf16 C.");
    return dnnl::memory::data_type::bf16;
  }
}

// Computes C[M,N] = scale * (A[M,K] * B[N,K]^T) (+ add) using the dnnl::matmul
// primitive, parallelized via the threadpool adapter. Scale and the optional
// per-column bias are fused into the primitive, which writes final values
// Returns false (with no writes to C) on any failure and 
// allows the caller to fall back to the stock path.
template <typename TA, typename TB, typename TC>
static HWY_NOINLINE bool DoMatMul_OneDnn(const MatPtrT<TA>& A,
                                         const MatPtrT<TB>& B, RowPtrs<TC> C,
                                         size_t M, size_t K, size_t N,
                                         float scale,
                                         const float* add,
                                         ThreadingContext& ctx,
                                         size_t cluster_idx) {
  static_assert(IsBF16<TA>() && IsBF16<TB>(),
                "OneDnn matmul path expects BF16 A and B.");
  try {
    using dt = dnnl::memory::data_type;
    using dims = dnnl::memory::dims;
    dnnl::engine& engine = OneDnnEngine();

    const hwy::pool::Caller caller =
        ctx.pool_callers.Get(Callers::kOneDnnMatMul);
    HwyThreadPoolAdapter adapter(ctx, cluster_idx, caller);
    // Must precede primitive create/exec; see comment on the helper.
    SetOneDnnMaxConcurrency(adapter.get_num_threads());
    dnnl::stream stream =
        dnnl::threadpool_interop::make_stream(engine, &adapter);

    const int64_t Mi = static_cast<int64_t>(M);
    const int64_t Ki = static_cast<int64_t>(K);
    const int64_t Ni = static_cast<int64_t>(N);
    const int64_t lda = static_cast<int64_t>(A.Stride());
    const int64_t ldb = static_cast<int64_t>(B.Stride());

    // Build oneDNN primitive once
    // JIT-compiled kernel will be used from the 2nd call onward.

    // A: [M,K] bf16, actual leading dim lda (handles non-packed A directly).
    const dnnl::memory::desc src_md({Mi, Ki}, dt::bf16, dims{lda, 1});

    // C: [M,N], written directly by the primitive. RowPtrs permits arbitrary
    // per-row pointers, but oneDNN can only address a single leading dim, so
    // verify the rows are regularly strided and bail to the stock path if not.
    TC* const C0 = C.Row(0);
    const ptrdiff_t ldc = (M > 1) ? (C.Row(1) - C0) : static_cast<ptrdiff_t>(N);
    if (ldc < static_cast<ptrdiff_t>(N)) return false;
    for (size_t m = 1; m < M; ++m) {
      if (C.Row(m) != C0 + ldc * static_cast<ptrdiff_t>(m)) return false;
    }
    const dnnl::memory::desc dst_md({Mi, Ni}, OneDnnDstType<TC>(),
                                    dims{static_cast<int64_t>(ldc), 1});

    // B: logical [K,N] bf16, format_tag::any so oneDNN picks its best layout.
    const dnnl::memory::desc wei_any(
        {Ki, Ni}, dt::bf16, dnnl::memory::format_tag::any);
    // Optional per-column bias (the `add`), broadcast over rows. Empty desc =
    // no bias.
    const dnnl::memory::desc bias_md =
        add ? dnnl::memory::desc({1, Ni}, dt::f32, dims{Ni, 1})
            : dnnl::memory::desc();

    dnnl::primitive_attr attr;
    // Fuse the scalar product scale
    attr.set_scales_mask(DNNL_ARG_WEIGHTS, 0);
    // switch from default library-managed scratchpad to user-managed scratchpad
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    const dnnl::matmul::primitive_desc pd(engine, src_md, wei_any, bias_md,
                                          dst_md, attr);
    const dnnl::memory::desc weights_md = pd.weights_desc();
    dnnl::matmul prim(pd);

    // Weights cache: B reordered into the kernel's layout, keyed on the B
    // pointer alone. Reorder happens once per distinct B. The key carries no
    // shape, but oneDNN's layout can be M-dependent and running against a
    // wrongly-packed B is silently wrong, so we reuse an entry only if its
    // actual layout equals the one this primitive wants. Large shapes like
    // those seen in gemma are M-independent.
    const uintptr_t B_ptr = reinterpret_cast<uintptr_t>(B.Row(0));
    const OneDnnWeightsKey w_key{B_ptr};
    auto& w_cache = GetOneDnnWeightsCache();
    auto w_it = w_cache.find(w_key);
    const bool needs_reorder =
        w_it == w_cache.end() || w_it->second.packed.get_desc() != weights_md;
    if (needs_reorder) {
      const dnnl::memory::desc user_wei_md({Ki, Ni}, dt::bf16, dims{1, ldb});
      dnnl::memory user_wei(user_wei_md, engine,
                            const_cast<TB*>(B.Row(0)));
      OneDnnWeightsEntry we;
      we.packed = dnnl::memory(weights_md, engine);
      dnnl::reorder(user_wei, we.packed)
          .execute(stream, user_wei, we.packed);
      stream.wait();
      if (w_it == w_cache.end()) {
        w_it = w_cache.emplace(w_key, std::move(we)).first;
      } else {
        w_it->second = std::move(we);
      }
    }
    OneDnnWeightsEntry& we = w_it->second;

    dnnl::memory src_mem(src_md, engine, const_cast<TA*>(A.Row(0)));
    dnnl::memory dst_mem(dst_md, engine, C0);
    dnnl::memory scale_mem({{1}, dt::f32, dnnl::memory::format_tag::x}, engine,
                           &scale);

    // User-managed scratchpad
    const dnnl::memory::desc scratchpad_md = pd.scratchpad_desc();
    const size_t scratchpad_size = scratchpad_md.get_size();
    std::vector<uint8_t>& sp_buf = GetOneDnnScratchpad();
    if (sp_buf.size() < scratchpad_size) {
      sp_buf.resize(scratchpad_size ? scratchpad_size : 1);
    }
    dnnl::memory scratchpad_mem(scratchpad_md, engine, sp_buf.data());

    std::unordered_map<int, dnnl::memory> args{
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, we.packed},
        {DNNL_ARG_DST, dst_mem},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale_mem},
        {DNNL_ARG_SCRATCHPAD, scratchpad_mem}};
    if (add) {
      args.emplace(DNNL_ARG_BIAS,
                   dnnl::memory(bias_md, engine, const_cast<float*>(add)));
    }
    prim.execute(stream, args);
    stream.wait();
    return true;
  } catch (...) {
    return false;
  }
}

#endif  // GEMMA_ONEDNN_MATMUL

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT (include guard)
