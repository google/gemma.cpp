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

// DeepSeek V4 transformer layer, following the reference implementation in
// inference/model.py of the DeepSeek-V4-Flash release:
//  * MLA with a single shared K=V latent per token (c_kv + decoupled RoPE
//    key), sliding-window attention over raw latents plus compressed
//    entries from a learned gated compressor (CSA: top-k via the lightning
//    indexer; HCA: all sealed blocks), a learned attention sink, and a
//    grouped low-rank output projection (wo_a / wo_b).
//  * Sigmoid/sqrtsoftplus MoE with aux-loss-free routing bias or hash-based
//    routing, shared expert, SwiGLU clamp and routed weight scale.
//  * Manifold-constrained hyper-connections (mHC) with per-token dynamic
//    read/write/mixing weights via a Sinkhorn-Knopp projection.
//
// Deliberate deviations from the reference (see configs.cc):
//  * The indexer's Hadamard rotation and FP4/FP8 activation-quantization
//    simulation are skipped (orthonormal rotation cancels in the score
//    dot-product; dropping quant-sim only removes QAT noise).
//  * The MTP layer is not implemented.
//
// The attention path is deliberately simple ("correct first"): scores and
// softmax use scalar-ish loops with Highway helpers for the inner dot
// products. Projections and the MoE path use the tiled MatMul.

#include <math.h>  // expf, sqrtf
#include <stddef.h>
#include <stdint.h>

#include <algorithm>  // std::copy_n, std::partial_sort
#include <atomic>
#include <cmath>
#include <cstring>  // memcpy

#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#include "util/zones.h"
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include "gemma/activations.h"
#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "gemma/kv_cache.h"
#include "gemma/query.h"
#include "gemma/weights.h"
#include "hwy/aligned_allocator.h"  // AlignedVector
#include "hwy/contrib/sort/order.h"
#include "hwy/profiler.h"

// Compiles this file for multiple architectures via "foreach_target.h", to
// which we pass the filename via macro 'argument'.
// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "deepseek/deepseek.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"
#include "gemma/attention.h"  // includes highway.h
#include "gemma/gemma-inl.h"
#include "ops/ops-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

static constexpr size_t kDSMaxExperts = 512;
// Latent row = kv_lora_rank + rope_head_dim.
static constexpr size_t kDSMaxLatentDim = 1024;
static constexpr size_t kDSMaxRate = 256;  // max sequence pooling rate
static constexpr size_t kDSMaxHCMult = 8;  // max residual streams
static constexpr size_t kDSMaxHeadDim = 1024;
static constexpr size_t kDSMaxIndexerDim = 8192;  // heads * head_dim
// Finite stand-in for -infinity when masking / initializing softmax maxima.
// Must stay finite: when a block is entirely masked, the online softmax
// computes exp(kDSMaskLogit - kDSMaskLogit) == 1; a true -inf would give NaN.
static constexpr float kDSMaskLogit = -1e30f;

static constexpr uint32_t DSFloatToUint32Sortkey(float val) {
  uint32_t temp = hwy::BitCastScalar<uint32_t>(val);
  return temp & 0x80000000 ? ~temp : temp ^ 0x80000000;
}

// Copies a small 1D/row tensor of any weight type to f32.
static void ReadRowF32(const MatPtr& w, size_t row, float* HWY_RESTRICT out,
                       size_t n) {
  CallUpcastedActivation(&w, [&](const auto* t) {
    for (size_t i = 0; i < n; ++i) {
      out[i] = hwy::ConvertScalarTo<float>(t->Row(row)[i]);
    }
  });
}

// Returns the element offset of this layer's segment within a flat KV cache
// row. Within the segment: [latent, compressed entry, indexer entry].
static size_t LatentLayerOffset(const KVCachePtr& kv, size_t layer_idx,
                                const LayerConfig& lc) {
  if (kv.cache != nullptr && !kv.cache->layer_flat_offsets.empty()) {
    return kv.cache->layer_flat_offsets[layer_idx];
  }
  return layer_idx * lc.CacheLayerSize();
}

// In-place plain-weight RMS norm on a raw f32 segment (per-head/per-latent).
// DeepSeek checkpoints store the true norm scale, hence kPlainWeight (the
// default gemma RMSNorm* would apply 1 + w to weights exported as scale-1).
static void ScaledRMSNorm(const MatPtr& w, float* HWY_RESTRICT x, size_t n,
                          ThreadingContext& ctx, size_t worker) {
  CallUpcastedActivation(&w, [&](const auto* t) {
    RMSNormInplace</*kPlainWeight=*/true>(t->PackedScale1(), /*w_ofs=*/0, x, n,
                                          ctx, worker);
  });
}

// ------------------------------ mHC ------------------------------
// Per-token dynamic hyper-connections: mixes = hc_fn @ flatten(streams)
// * rsqrt(mean_sq), split into read weights (sigmoid), write weights
// (2*sigmoid) and a doubly-stochastic mixing matrix (softmax followed by
// Sinkhorn-Knopp row/column normalization). Matches `hc_split_sinkhorn` and
// `Block.hc_pre`/`hc_post` in the reference implementation.

// Shared mHC "read": pre[j] = sigmoid(mixes[j] * scale + base[j]) + eps, then
// x = sum_j pre[j] * stream_j. Used by HCReadDynamic (rsqrt_ms already applied
// to `mixes`) and HCHeadCollapse (rsqrt_ms folded into `scale`).
static void HCSigmoidRead(const float* HWY_RESTRICT mixes,
                          const float* HWY_RESTRICT base, float scale,
                          float eps, size_t hc_mult,
                          const float* HWY_RESTRICT s, float* HWY_RESTRICT x,
                          size_t model_dim, ThreadingContext& ctx,
                          size_t worker) {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::CappedTag<float, kDSMaxHCMult> dh;
  using VH = hn::Vec<decltype(dh)>;
  const size_t NH = hn::Lanes(dh);
  const VH vone = hn::Set(dh, 1.0f);
  HWY_ALIGN float pre[kDSMaxHCMult];
  for (size_t j = 0; j < hc_mult; j += NH) {
    const size_t n = HWY_MIN(NH, hc_mult - j);
    const VH t = hn::MulAdd(hn::LoadN(dh, mixes + j, n), hn::Set(dh, scale),
                            hn::LoadN(dh, base + j, n));
    const VH sig = hn::Div(vone, hn::Add(vone, hn::Exp(dh, hn::Neg(t))));
    hn::StoreN(hn::Add(sig, hn::Set(dh, eps)), dh, pre + j, n);
  }
  MulByConstTo(pre[0], s, x, model_dim, ctx, worker);
  for (size_t j = 1; j < hc_mult; ++j) {
    MulByConstAndAdd(pre[j], s + j * model_dim, x, model_dim);
  }
}

// Computes mixes for the given block weights, then per token: fills
// activations.x with the pre-weighted stream sum and stores the post/comb
// weights for HCWriteDynamic.
static HWY_NOINLINE void HCReadDynamic(const MatPtr& fn_w, const MatPtr& base_w,
                                       const MatPtr& scale_w,
                                       Activations& activations,
                                       MatMulEnv& env) {
  const ModelConfig& config = activations.attention.config;
  const size_t model_dim = activations.x.Cols();
  const size_t hc_mult = config.hc_mult;
  const size_t mix_hc = (2 + hc_mult) * hc_mult;
  const size_t sinkhorn_iters = config.hc_sinkhorn_iters;
  const float eps = config.hc_eps;
  HWY_DASSERT(hc_mult >= 2 && hc_mult <= kDSMaxHCMult);

  // mixes = streams @ fn_w^T. Row scaling by rsqrt happens below.
  activations.hc_mixes.OverrideCols(mix_hc);
  CallMatMul(activations.hc_streams, fn_w, /*add=*/nullptr, env,
             activations.hc_mixes);

  HWY_ALIGN float base[(2 + kDSMaxHCMult) * kDSMaxHCMult];
  HWY_ALIGN float scale[3];
  ReadRowF32(base_w, 0, base, mix_hc);
  ReadRowF32(scale_w, 0, scale, 3);

  ParallelFor(
      Parallelism::kFlat, activations.x.Rows(), env.ctx,
      /*cluster_idx=*/0, Callers::kActivationBatched,
      [&](size_t token_idx, size_t worker) HWY_ATTR {
        namespace hn = hwy::HWY_NAMESPACE;
        const float* HWY_RESTRICT s = activations.hc_streams.Row(token_idx);
        float* HWY_RESTRICT mixes = activations.hc_mixes.Row(token_idx);
        float* HWY_RESTRICT x = activations.x.Row(token_idx);
        float* HWY_RESTRICT post = activations.hc_post_w.Row(token_idx);
        float* HWY_RESTRICT comb = activations.hc_comb.Row(token_idx);

        const hn::CappedTag<float, kDSMaxHCMult> dh;
        using VH = hn::Vec<decltype(dh)>;
        const size_t NH = hn::Lanes(dh);
        const VH vone = hn::Set(dh, 1.0f);
        const VH veps = hn::Set(dh, eps);

        // rsqrt of the mean square over the flattened streams
        // (`detail::RMSNormMul` uses a 1e-6 epsilon).
        const size_t flat = hc_mult * model_dim;
        const float rsqrt_ms = detail::RMSNormMul(s, flat, env.ctx, worker);
        {
          const VH vr = hn::Set(dh, rsqrt_ms);
          for (size_t i = 0; i < mix_hc; i += NH) {
            const size_t n = HWY_MIN(NH, mix_hc - i);
            hn::StoreN(hn::Mul(hn::LoadN(dh, mixes + i, n), vr), dh, mixes + i,
                       n);
          }
        }

        // pre = sigmoid(m * scale0 + base) + eps; read: x = sum pre_i s_i.
        HCSigmoidRead(mixes, base, scale[0], eps, hc_mult, s, x, model_dim,
                      env.ctx, worker);
        // post = 2 * sigmoid(m * scale1 + base).
        for (size_t j = 0; j < hc_mult; j += NH) {
          const size_t n = HWY_MIN(NH, hc_mult - j);
          const VH t1 = hn::MulAdd(hn::LoadN(dh, mixes + hc_mult + j, n),
                                   hn::Set(dh, scale[1]),
                                   hn::LoadN(dh, base + hc_mult + j, n));
          const VH inv_exp1 = hn::Add(vone, hn::Exp(dh, hn::Neg(t1)));
          hn::StoreN(hn::Div(hn::Set(dh, 2.0f), inv_exp1), dh, post + j, n);
        }

        // comb = softmax_rows(m * scale2 + base) + eps, then Sinkhorn.
        // comb[i*hc + j]: source stream i -> destination stream j.
        HWY_ALIGN float c[kDSMaxHCMult * kDSMaxHCMult];
        {
          const VH vs2 = hn::Set(dh, scale[2]);
          const size_t total = hc_mult * hc_mult;
          for (size_t i = 0; i < total; i += NH) {
            const size_t n = HWY_MIN(NH, total - i);
            hn::StoreN(
                hn::MulAdd(hn::LoadN(dh, mixes + 2 * hc_mult + i, n), vs2,
                           hn::LoadN(dh, base + 2 * hc_mult + i, n)),
                dh, c + i, n);
          }
        }
        // Row softmax; LoadNOr pads with the mask logit so padding lanes exp
        // to zero.
        const VH vneg = hn::Set(dh, kDSMaskLogit);
        HWY_ALIGN float erow[kDSMaxHCMult];
        for (size_t i = 0; i < hc_mult; ++i) {
          float* HWY_RESTRICT row = c + i * hc_mult;
          VH vmax = vneg;
          for (size_t j = 0; j < hc_mult; j += NH) {
            vmax = hn::Max(
                vmax, hn::LoadNOr(vneg, dh, row + j, HWY_MIN(NH, hc_mult - j)));
          }
          const VH vm = hn::Set(dh, hn::ReduceMax(dh, vmax));
          float sum = 0.0f;
          for (size_t j = 0; j < hc_mult; j += NH) {
            const size_t n = HWY_MIN(NH, hc_mult - j);
            const VH e =
                hn::Exp(dh, hn::Sub(hn::LoadNOr(vneg, dh, row + j, n), vm));
            hn::StoreN(e, dh, erow + j, n);
            sum += hn::ReduceSum(dh, e);
          }
          const VH vinv = hn::Set(dh, 1.0f / sum);
          for (size_t j = 0; j < hc_mult; j += NH) {
            const size_t n = HWY_MIN(NH, hc_mult - j);
            hn::StoreN(hn::MulAdd(hn::LoadN(dh, erow + j, n), vinv, veps), dh,
                       row + j, n);
          }
        }
        // First column normalization, then (iters-1) x (row + column).
        // Columns vectorize as vertical adds over rows; LoadN zero-pads, and
        // StoreN discards the padding lanes again.
        for (size_t iter = 0;; ++iter) {
          for (size_t j0 = 0; j0 < hc_mult; j0 += NH) {  // column chunks
            const size_t n = HWY_MIN(NH, hc_mult - j0);
            VH sum = hn::Zero(dh);
            for (size_t i = 0; i < hc_mult; ++i) {
              sum = hn::Add(sum, hn::LoadN(dh, c + i * hc_mult + j0, n));
            }
            const VH vinv = hn::Div(vone, hn::Add(sum, veps));
            for (size_t i = 0; i < hc_mult; ++i) {
              hn::StoreN(hn::Mul(hn::LoadN(dh, c + i * hc_mult + j0, n), vinv),
                         dh, c + i * hc_mult + j0, n);
            }
          }
          if (iter + 1 >= sinkhorn_iters) break;
          for (size_t i = 0; i < hc_mult; ++i) {  // rows
            float* HWY_RESTRICT row = c + i * hc_mult;
            float sum = 0.0f;
            for (size_t j = 0; j < hc_mult; j += NH) {
              sum += hn::ReduceSum(
                  dh, hn::LoadN(dh, row + j, HWY_MIN(NH, hc_mult - j)));
            }
            const VH vinv = hn::Set(dh, 1.0f / (sum + eps));
            for (size_t j = 0; j < hc_mult; j += NH) {
              const size_t n = HWY_MIN(NH, hc_mult - j);
              hn::StoreN(hn::Mul(hn::LoadN(dh, row + j, n), vinv), dh, row + j,
                         n);
            }
          }
        }
        memcpy(comb, c, hc_mult * hc_mult * sizeof(float));
      });
}

// streams_new[j] = post[j] * block_out + sum_i comb[i][j] * streams_old[i],
// using the per-token post/comb stored by HCReadDynamic.
template <typename T>
static HWY_NOINLINE void HCWriteDynamic(const MatPtrT<T>& block_out,
                                        Activations& activations,
                                        MatMulEnv& env) {
  const size_t model_dim = activations.x.Cols();
  const size_t hc_mult = activations.hc_streams.Cols() / model_dim;
  HWY_DASSERT(hc_mult >= 2 && hc_mult <= kDSMaxHCMult);

  ParallelFor(
      Parallelism::kFlat, activations.x.Rows(), env.ctx,
      /*cluster_idx=*/0, Callers::kActivationBatched,
      [&](size_t token_idx, size_t worker) HWY_ATTR {
        float* HWY_RESTRICT s = activations.hc_streams.Row(token_idx);
        float* HWY_RESTRICT tmp = activations.hc_tmp.Row(token_idx);
        const float* HWY_RESTRICT post = activations.hc_post_w.Row(token_idx);
        const float* HWY_RESTRICT comb = activations.hc_comb.Row(token_idx);
        const T* HWY_RESTRICT out = block_out.Row(token_idx);
        for (size_t j = 0; j < hc_mult; ++j) {
          MulByConstTo(comb[j], s, tmp + j * model_dim, model_dim, env.ctx,
                       worker);
          for (size_t i = 1; i < hc_mult; ++i) {
            MulByConstAndAdd(comb[i * hc_mult + j], s + i * model_dim,
                             tmp + j * model_dim, model_dim);
          }
          MulByConstAndAdd(post[j], out, tmp + j * model_dim, model_dim);
        }
        memcpy(s, tmp, hc_mult * model_dim * sizeof(float));
      });
}

void DeepSeekMaybeInitHCStreams(Activations& activations, MatMulEnv& env) {
  const ModelConfig& config = activations.attention.config;
  if (config.hc_mult <= 1) return;
  const size_t model_dim = activations.x.Cols();
  const size_t hc_mult = config.hc_mult;
  ParallelFor(Parallelism::kFlat, activations.x.Rows(), env.ctx,
              /*cluster_idx=*/0, Callers::kActivationBatched,
              [&](size_t token_idx, size_t worker) HWY_ATTR {
                const float* HWY_RESTRICT x = activations.x.Row(token_idx);
                float* HWY_RESTRICT s = activations.hc_streams.Row(token_idx);
                for (size_t i = 0; i < hc_mult; ++i) {
                  memcpy(s + i * model_dim, x, model_dim * sizeof(float));
                }
              });
}

// Collapses the streams into activations.x with per-token sigmoid read
// weights computed from the given head tensors (`hc_head` in the reference
// ParallelHead; also used with the MTP block's own head tensors).
static void HCHeadCollapse(const MatPtr& fn_w, const MatPtr& base_w,
                           const MatPtr& scale_w, Activations& activations,
                           MatMulEnv& env) {
  const ModelConfig& config = activations.attention.config;
  const size_t model_dim = activations.x.Cols();
  const size_t hc_mult = config.hc_mult;

  // mixes = streams @ fn_w^T ([hc_mult, hc_mult * model_dim]).
  activations.hc_mixes.OverrideCols(hc_mult);
  CallMatMul(activations.hc_streams, fn_w, /*add=*/nullptr, env,
             activations.hc_mixes);
  HWY_ALIGN float base[kDSMaxHCMult];
  HWY_ALIGN float scale[1];
  ReadRowF32(base_w, 0, base, hc_mult);
  ReadRowF32(scale_w, 0, scale, 1);
  const float eps = config.hc_eps;

  ParallelFor(
      Parallelism::kFlat, activations.x.Rows(), env.ctx,
      /*cluster_idx=*/0, Callers::kActivationBatched,
      [&](size_t token_idx, size_t worker) HWY_ATTR {
        const float* HWY_RESTRICT s = activations.hc_streams.Row(token_idx);
        float* HWY_RESTRICT mixes = activations.hc_mixes.Row(token_idx);
        float* HWY_RESTRICT x = activations.x.Row(token_idx);
        const size_t flat = hc_mult * model_dim;
        const float rsqrt_ms = detail::RMSNormMul(s, flat, env.ctx, worker);
        // pre = sigmoid(m * rsqrt_ms * scale0 + base) + eps; x = sum pre_i s_i.
        HCSigmoidRead(mixes, base, rsqrt_ms * scale[0], eps, hc_mult, s, x,
                      model_dim, env.ctx, worker);
      });
}

// Collapses the streams into activations.x before the final norm. Falls back
// to the mean if the model has no hc_head tensors (e.g. synthetic tests).
void DeepSeekMaybeFinalizeHCStreams(const WeightsPtrs& weights,
                                    Activations& activations, MatMulEnv& env) {
  const ModelConfig& config = activations.attention.config;
  if (config.hc_mult <= 1) return;

  if (!weights.hc_head_fn.HasPtr()) {
    const size_t model_dim = activations.x.Cols();
    const size_t hc_mult = config.hc_mult;
    const float inv = 1.0f / static_cast<float>(hc_mult);
    ParallelFor(Parallelism::kFlat, activations.x.Rows(), env.ctx,
                /*cluster_idx=*/0, Callers::kActivationBatched,
                [&](size_t token_idx, size_t worker) HWY_ATTR {
                  float* HWY_RESTRICT x = activations.x.Row(token_idx);
                  const float* HWY_RESTRICT s =
                      activations.hc_streams.Row(token_idx);
                  MulByConstTo(inv, s, x, model_dim, env.ctx, worker);
                  for (size_t i = 1; i < hc_mult; ++i) {
                    MulByConstAndAdd(inv, s + i * model_dim, x, model_dim);
                  }
                });
    return;
  }

  HCHeadCollapse(weights.hc_head_fn, weights.hc_head_base,
                 weights.hc_head_scale, activations, env);
}

// ------------------------------ Compressor ------------------------------
// Learned gated pooling over `rate` consecutive tokens with a per-slot bias
// (ape) and per-dim softmax over the block axis. With overlap (rate 4), the
// first half of the projected dims belongs to the *overlapping* window (the
// previous block), the second half to the current block. Incremental state
// (kv_state/score_state) lives in KVCache::ds_state; entries seal when
// (pos+1) % rate == 0 and are written to the layer's compressed cache region
// at the sealing row.

struct DSCompressorRefs {
  float* kv_state;     // [coff*rate, coff*d]
  float* score_state;  // [coff*rate, coff*d]
};

// Processes one token through one compressor; returns true and fills
// `entry` (width d) if the block sealed.
static bool CompressorStep(const float* HWY_RESTRICT kv_row,
                           const float* HWY_RESTRICT gate_row,
                           const float* HWY_RESTRICT ape,  // [rate, coff*d]
                           DSCompressorRefs state, size_t cache_pos,
                           size_t rate, size_t coff, size_t d,
                           float* HWY_RESTRICT entry) {
  const size_t width = coff * d;
  const size_t slot = cache_pos % rate;

  if (cache_pos == 0) {
    // Start of a sequence: reset this compressor's state.
    const size_t total = coff * rate * width;
    memset(state.kv_state, 0, total * sizeof(float));  // 0.0f is all-zero
    {
      namespace hn = hwy::HWY_NAMESPACE;
      const hn::ScalableTag<float> dreset;
      const auto vneg = hn::Set(dreset, kDSMaskLogit);
      const size_t NR = hn::Lanes(dreset);
      size_t i = 0;
      for (; i + NR <= total; i += NR) {
        hn::StoreU(vneg, dreset, state.score_state + i);
      }
      for (; i < total; ++i) state.score_state[i] = kDSMaskLogit;
    }
  }

  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> df;
  const size_t NF = hn::Lanes(df);

  // Write the current token into the "current window" half (rows
  // [offset, offset+rate)); with overlap, rows [0, rate) hold the previous
  // block.
  const size_t offset = (coff == 2 ? rate : 0) + slot;
  float* HWY_RESTRICT kv_dst = state.kv_state + offset * width;
  float* HWY_RESTRICT score_dst = state.score_state + offset * width;
  memcpy(kv_dst, kv_row, width * sizeof(float));
  {
    const float* HWY_RESTRICT ape_row = ape + slot * width;
    size_t i = 0;
    for (; i + NF <= width; i += NF) {
      hn::StoreU(
          hn::Add(hn::LoadU(df, gate_row + i), hn::LoadU(df, ape_row + i)), df,
          score_dst + i);
    }
    for (; i < width; ++i) score_dst[i] = gate_row[i] + ape_row[i];
  }

  if ((cache_pos + 1) % rate != 0) return false;

  // Seal: per-dim softmax over the block axis. Candidate rows for output
  // dim k: with overlap, rows 0..rate use their first-half dims and rows
  // rate..2*rate their second-half dims; without overlap, all rows use dim k.
  // Vectorized across k (contiguous within each row for both halves).
  const size_t num_rows = coff * rate;
  const auto row_ofs = [&](size_t r, size_t k) {
    return r * width + ((coff == 2 && r >= rate) ? d + k : k);
  };
  size_t k = 0;
  for (; k + NF <= d; k += NF) {
    auto vmax = hn::Set(df, kDSMaskLogit);
    for (size_t r = 0; r < num_rows; ++r) {
      vmax = hn::Max(vmax, hn::LoadU(df, state.score_state + row_ofs(r, k)));
    }
    auto vden = hn::Zero(df);
    auto vacc = hn::Zero(df);
    for (size_t r = 0; r < num_rows; ++r) {
      const auto w = hn::Exp(
          df, hn::Sub(hn::LoadU(df, state.score_state + row_ofs(r, k)), vmax));
      vden = hn::Add(vden, w);
      vacc = hn::MulAdd(w, hn::LoadU(df, state.kv_state + row_ofs(r, k)), vacc);
    }
    hn::StoreU(hn::Div(vacc, vden), df, entry + k);
  }
  for (; k < d; ++k) {
    float max_score = kDSMaskLogit;
    for (size_t r = 0; r < num_rows; ++r) {
      max_score = HWY_MAX(max_score, state.score_state[row_ofs(r, k)]);
    }
    float denom = 0.0f;
    float acc = 0.0f;
    for (size_t r = 0; r < num_rows; ++r) {
      const float w = expf(state.score_state[row_ofs(r, k)] - max_score);
      denom += w;
      acc += w * state.kv_state[row_ofs(r, k)];
    }
    entry[k] = acc / denom;
  }

  if (coff == 2) {
    // Shift: the sealed block becomes the next block's overlap window.
    memcpy(state.kv_state, state.kv_state + rate * width,
           rate * width * sizeof(float));
    memcpy(state.score_state, state.score_state + rate * width,
           rate * width * sizeof(float));
  }
  return true;
}

// ------------------------------ MLA attention ------------------------------

// Rows per blocked online-softmax update: one vectorized Exp instead of
// per-row scalar expf. Must cover the widest SIMD target (16 f32 lanes).
static constexpr size_t kDSSoftmaxBlock = 16;

// Online-softmax accumulation state over latent entries.
struct DSOnlineSoftmax {
  void Reset(size_t dim) {
    this->dim = dim;
    max_logit = kDSMaskLogit;
    denom = 0.0f;
    for (size_t d = 0; d < dim; ++d) acc[d] = 0.0f;
  }
  void Update(float logit, const float* HWY_RESTRICT value) {
    if (logit > max_logit) {
      const float scale = expf(max_logit - logit);
      denom = denom * scale + 1.0f;
      MulByConst(scale, acc, dim);
      MulByConstAndAdd(1.0f, value, acc, dim);
      max_logit = logit;
    } else {
      const float w = expf(logit - max_logit);
      denom += w;
      MulByConstAndAdd(w, value, acc, dim);
    }
  }
  // Accumulates `n <= kDSSoftmaxBlock` rows at once: single rescale of the
  // running state and one vectorized Exp over all logits in the block.
  void UpdateBlock(const float* HWY_RESTRICT logits,
                   const float (*HWY_RESTRICT values)[kDSMaxLatentDim],
                   size_t n) {
    namespace hn = hwy::HWY_NAMESPACE;
    // Capped so wider-than-block targets stay within the arrays.
    const hn::CappedTag<float, kDSSoftmaxBlock> df;
    const size_t NF = hn::Lanes(df);
    HWY_DASSERT(n > 0 && n <= kDSSoftmaxBlock);

    float blk_max = logits[0];
    for (size_t i = 1; i < n; ++i) blk_max = HWY_MAX(blk_max, logits[i]);
    if (blk_max > max_logit) {
      const float scale = expf(max_logit - blk_max);
      denom *= scale;
      MulByConst(scale, acc, dim);
      max_logit = blk_max;
    }

    HWY_ALIGN float shifted[kDSSoftmaxBlock];
    HWY_ALIGN float w[kDSSoftmaxBlock];
    for (size_t i = 0; i < n; ++i) shifted[i] = logits[i] - max_logit;
    for (size_t i = n; i < kDSSoftmaxBlock; ++i) shifted[i] = kDSMaskLogit;
    for (size_t i = 0; i < kDSSoftmaxBlock; i += NF) {
      hn::Store(hn::Exp(df, hn::Load(df, shifted + i)), df, w + i);
    }
    for (size_t i = 0; i < n; ++i) {
      denom += w[i];
      MulByConstAndAdd(w[i], values[i], acc, dim);
    }
  }
  // Adds `exp(logit - max)` to the denominator without contributing a value
  // (the learned attention sink).
  void UpdateLogitOnly(float logit) {
    if (logit > max_logit) {
      const float scale = expf(max_logit - logit);
      denom = denom * scale + 1.0f;
      MulByConst(scale, acc, dim);
      max_logit = logit;
    } else {
      denom += expf(logit - max_logit);
    }
  }
  void Finalize() { MulByConst(1.0f / denom, acc, dim); }

  size_t dim = 0;
  float max_logit = kDSMaskLogit;
  float denom = 0.0f;
  HWY_ALIGN float acc[kDSMaxLatentDim];
};

// Runs the attention (and indexer) compressors for all tokens of the batch,
// sequentially per query, writing sealed entries to the flat KV cache.
static HWY_NOINLINE void DeepSeekRunCompressors(
    size_t num_tokens, size_t layer_idx, const LayerWeightsPtrs& layer,
    Activations& activations, QBatch& qbatch, MatMulEnv& env,
    const float* HWY_RESTRICT inv_timescale_c) {
  const LayerConfig& lc = layer.layer_config;
  const size_t d = lc.KVLatentDim();
  const size_t rate = lc.kv_compression_rate;
  const size_t coff = lc.CompressorCoff();
  const size_t rope_dim = lc.rope_head_dim;
  const bool has_indexer = lc.HasIndexer();
  const size_t idx_dim = lc.indexer_head_dim;

  // Projections for all batch tokens.
  activations.comp_kv.OverrideCols(coff * d);
  activations.comp_gate.OverrideCols(coff * d);
  CallMatMul(activations.attention.pre_att_rms_out, layer.comp_wkv,
             /*add=*/nullptr, env, activations.comp_kv);
  CallMatMul(activations.attention.pre_att_rms_out, layer.comp_wgate,
             /*add=*/nullptr, env, activations.comp_gate);
  if (has_indexer) {
    activations.idxc_kv.OverrideCols(coff * idx_dim);
    activations.idxc_gate.OverrideCols(coff * idx_dim);
    CallMatMul(activations.attention.pre_att_rms_out, layer.idxc_wkv,
               /*add=*/nullptr, env, activations.idxc_kv);
    CallMatMul(activations.attention.pre_att_rms_out, layer.idxc_wgate,
               /*add=*/nullptr, env, activations.idxc_gate);
  }

  // comp_ape rows are [rate, coff*d]; copy row by row into a flat buffer.
  hwy::AlignedVector<float> ape(rate * coff * d);
  for (size_t r = 0; r < rate; ++r) {
    ReadRowF32(layer.comp_ape, r, ape.data() + r * coff * d, coff * d);
  }
  hwy::AlignedVector<float> idx_ape;
  if (has_indexer) {
    idx_ape.resize(rate * coff * idx_dim);
    for (size_t r = 0; r < rate; ++r) {
      ReadRowF32(layer.idxc_ape, r, idx_ape.data() + r * coff * idx_dim,
                 coff * idx_dim);
    }
  }

  const hwy::Divisor div_qbatch(static_cast<uint32_t>(qbatch.Size()));
  KVCache* cache0 = qbatch.KV(0).cache;
  HWY_ASSERT(cache0 != nullptr && cache0->ds_state.Rows() > 0);

  // Parallel over queries; sequential over the query's tokens (state carries
  // across tokens).
  ParallelFor(
      Parallelism::kFlat, qbatch.Size(), env.ctx, /*cluster_idx=*/0,
      Callers::kAttComputeQKV, [&](size_t qi, size_t worker) HWY_ATTR {
        KVCache* cache = qbatch.KV(qi).cache;
        float* state_base =
            cache->ds_state.Row(0) + cache->ds_state_offsets[layer_idx];
        DSCompressorRefs comp_state{state_base,
                                    state_base + (coff * rate) * (coff * d)};
        float* idx_state_base = state_base + 2 * (coff * rate) * (coff * d);
        DSCompressorRefs idx_state{
            idx_state_base, idx_state_base + (coff * rate) * (coff * idx_dim)};

        const size_t layer_offset =
            LatentLayerOffset(qbatch.KV(qi), layer_idx, lc);
        const size_t comp_offset = layer_offset + d;  // after the latent
        const size_t idx_offset = comp_offset + d;    // after the entry

        HWY_ALIGN float entry[kDSMaxLatentDim];
        for (size_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
          const size_t task = token_idx * qbatch.Size() + qi;
          const size_t cache_pos = qbatch.Pos(qi) + token_idx;
          // Attention compressor.
          if (CompressorStep(activations.comp_kv.Row(task),
                             activations.comp_gate.Row(task), ape.data(),
                             comp_state, cache_pos, rate, coff, d, entry)) {
            ScaledRMSNorm(layer.comp_norm, entry, d, env.ctx, worker);
            const size_t block_start = cache_pos + 1 - rate;
            Rope(entry + d - rope_dim, rope_dim, inv_timescale_c,
                 static_cast<int>(block_start), env.ctx, worker);
            KV_t* HWY_RESTRICT dst =
                qbatch.KV(qi).kv_cache.Row(cache_pos) + comp_offset;
            CompressPerThread tls;
            Compress(entry, d, tls, MakeSpan(dst, d), 0);
          }
          // Indexer compressor (128-wide; Hadamard rotation and FP4 sim
          // skipped -- see file comment).
          if (has_indexer &&
              CompressorStep(activations.idxc_kv.Row(task),
                             activations.idxc_gate.Row(task), idx_ape.data(),
                             idx_state, cache_pos, rate, coff, idx_dim,
                             entry)) {
            ScaledRMSNorm(layer.idxc_norm, entry, idx_dim, env.ctx, worker);
            const size_t block_start = cache_pos + 1 - rate;
            Rope(entry + idx_dim - rope_dim, rope_dim, inv_timescale_c,
                 static_cast<int>(block_start), env.ctx, worker);
            KV_t* HWY_RESTRICT dst =
                qbatch.KV(qi).kv_cache.Row(cache_pos) + idx_offset;
            CompressPerThread tls;
            Compress(entry, idx_dim, tls, MakeSpan(dst, idx_dim), 0);
          }
          // Speculative decoding: snapshot this layer's state at the
          // committed/draft boundary so a rejected draft can be rolled back.
          if (HWY_UNLIKELY(static_cast<int>(token_idx) ==
                           activations.ds_snapshot_after) &&
              cache->ds_state_snapshot.Rows() > 0) {
            const size_t ofs = cache->ds_state_offsets[layer_idx];
            memcpy(cache->ds_state_snapshot.Row(0) + ofs,
                   cache->ds_state.Row(0) + ofs,
                   lc.DSStateSize() * sizeof(float));
          }
        }
      });
}

// Four dot products sharing one query vector: amortizes the query loads and
// defers the horizontal reductions (one per entry instead of per call).
static HWY_INLINE void Dot4(const float* HWY_RESTRICT q,
                            const float* HWY_RESTRICT k0,
                            const float* HWY_RESTRICT k1,
                            const float* HWY_RESTRICT k2,
                            const float* HWY_RESTRICT k3, size_t n,
                            float* HWY_RESTRICT out4) {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> df;
  const size_t NF = hn::Lanes(df);
  auto acc0 = hn::Zero(df);
  auto acc1 = hn::Zero(df);
  auto acc2 = hn::Zero(df);
  auto acc3 = hn::Zero(df);
  size_t i = 0;
  for (; i + NF <= n; i += NF) {
    const auto vq = hn::LoadU(df, q + i);
    acc0 = hn::MulAdd(vq, hn::LoadU(df, k0 + i), acc0);
    acc1 = hn::MulAdd(vq, hn::LoadU(df, k1 + i), acc1);
    acc2 = hn::MulAdd(vq, hn::LoadU(df, k2 + i), acc2);
    acc3 = hn::MulAdd(vq, hn::LoadU(df, k3 + i), acc3);
  }
  out4[0] = hn::ReduceSum(df, acc0);
  out4[1] = hn::ReduceSum(df, acc1);
  out4[2] = hn::ReduceSum(df, acc2);
  out4[3] = hn::ReduceSum(df, acc3);
  for (; i < n; ++i) {
    out4[0] += q[i] * k0[i];
    out4[1] += q[i] * k1[i];
    out4[2] += q[i] * k2[i];
    out4[3] += q[i] * k3[i];
  }
}

// Selects, per query token, the top-k sealed compressed entries via the
// lightning indexer. `selected[task]` receives sealed-block indices.
static HWY_NOINLINE void DeepSeekIndexerSelect(
    size_t num_tokens, size_t layer_idx, const LayerWeightsPtrs& layer,
    Activations& activations, QBatch& qbatch, MatMulEnv& env,
    const float* HWY_RESTRICT inv_timescale_c,
    hwy::AlignedVector<hwy::AlignedVector<uint32_t>>& selected) {
  const LayerConfig& lc = layer.layer_config;
  const size_t idx_heads = lc.indexer_heads;
  const size_t idx_dim = lc.indexer_head_dim;
  const size_t rate = lc.kv_compression_rate;
  const size_t rope_dim = lc.rope_head_dim;
  const size_t d = lc.KVLatentDim();

  // Queries from the normed q-LoRA latent; per-head weights from x.
  activations.idx_q.OverrideCols(idx_heads * idx_dim);
  CallMatMul(activations.mla_q_a, layer.idx_q_w, /*add=*/nullptr, env,
             activations.idx_q);
  activations.idx_weights.OverrideCols(idx_heads);
  CallMatMul(activations.attention.pre_att_rms_out, layer.idx_w_proj,
             /*add=*/nullptr, env, activations.idx_weights);

  const float w_scale = 1.0f / (sqrtf(static_cast<float>(idx_dim)) *
                                sqrtf(static_cast<float>(idx_heads)));
  const hwy::Divisor div_qbatch(static_cast<uint32_t>(qbatch.Size()));
  const size_t num_interleaved = num_tokens * qbatch.Size();

  ParallelFor(
      Parallelism::kFlat, num_interleaved, env.ctx, /*cluster_idx=*/0,
      Callers::kAttComputeQKV, [&](size_t task, size_t worker) HWY_ATTR {
        namespace hn = hwy::HWY_NAMESPACE;
        const hn::ScalableTag<float> df;
        const size_t qi = div_qbatch.Remainder(static_cast<uint32_t>(task));
        const size_t token_idx = div_qbatch.Divide(static_cast<uint32_t>(task));
        const size_t cache_pos = qbatch.Pos(qi) + token_idx;
        const size_t num_sealed = (cache_pos + 1) / rate;
        selected[task].clear();
        if (num_sealed == 0) return;

        float* HWY_RESTRICT q = activations.idx_q.Row(task);
        // RoPE the last rope_dim dims of each indexer head (in place; this
        // task owns the row).
        for (size_t h = 0; h < idx_heads; ++h) {
          Rope(q + h * idx_dim + idx_dim - rope_dim, rope_dim, inv_timescale_c,
               static_cast<int>(cache_pos), env.ctx, worker);
        }
        const float* HWY_RESTRICT w = activations.idx_weights.Row(task);

        const size_t layer_offset =
            LatentLayerOffset(qbatch.KV(qi), layer_idx, lc);
        const size_t idx_offset = layer_offset + 2 * d;

        // All sealed entries score-ranked; keep top-k. Entries are scored
        // four at a time (Dot4) to amortize query loads across entries.
        // NOTE: all heads share the same compressed key entry.
        hwy::AlignedVector<float> scores(num_sealed);
        HWY_ALIGN float k_f[4][512];
        for (size_t e0 = 0; e0 < num_sealed; e0 += 4) {
          const size_t nb = HWY_MIN(size_t{4}, num_sealed - e0);
          for (size_t b = 0; b < nb; ++b) {
            const size_t seal_row = (e0 + b + 1) * rate - 1;
            const KV_t* HWY_RESTRICT src =
                qbatch.KV(qi).kv_cache.Row(seal_row) + idx_offset;
            DecompressAndZeroPad(df, MakeSpan(src, idx_dim), 0, k_f[b],
                                 idx_dim);
          }
          for (size_t b = nb; b < 4; ++b) {  // dummies; results ignored
            memcpy(k_f[b], k_f[0], idx_dim * sizeof(float));
          }
          HWY_ALIGN float s4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
          for (size_t h = 0; h < idx_heads; ++h) {
            HWY_ALIGN float d4[4];
            Dot4(q + h * idx_dim, k_f[0], k_f[1], k_f[2], k_f[3], idx_dim, d4);
            const float wh = w[h];
            for (size_t b = 0; b < 4; ++b) {
              s4[b] += HWY_MAX(d4[b], 0.0f) * wh;
            }
          }
          for (size_t b = 0; b < nb; ++b) scores[e0 + b] = s4[b] * w_scale;
        }
        const size_t top_k = HWY_MIN(size_t{lc.indexer_top_k}, num_sealed);
        selected[task].resize(num_sealed);
        for (size_t e = 0; e < num_sealed; ++e) selected[task][e] = e;
        if (top_k < num_sealed) {
          std::partial_sort(selected[task].begin(),
                            selected[task].begin() + top_k,
                            selected[task].end(),
                            [&](uint32_t a, uint32_t b)
                                HWY_ATTR { return scores[a] > scores[b]; });
        }
        selected[task].resize(top_k);
      });
}

// Multi-head Latent Attention, DeepSeek V4 style. Reads `pre_att_rms_out`,
// writes `att_sums`.
static HWY_NOINLINE void DeepSeekAttention(size_t num_tokens, size_t layer_idx,
                                           const LayerWeightsPtrs& layer,
                                           Activations& activations,
                                           QBatch& qbatch, MatMulEnv& env) {
  GCPP_ZONE(env.ctx, hwy::Profiler::GlobalIdx(), Zones::kGenAttention);

  AttentionActivationsPtrs& att = activations.attention;
  const ModelConfig& config = att.config;
  const LayerConfig& lc = layer.layer_config;
  const size_t heads = lc.heads;
  const size_t qkv_dim = lc.qkv_dim;  // per-head query dim (= latent dim)
  const size_t rope_dim = lc.rope_head_dim;
  const size_t kv_a_dim = lc.KVLatentDim();
  HWY_DASSERT(kv_a_dim <= kDSMaxLatentDim && qkv_dim <= kDSMaxHeadDim);
  HWY_DASSERT(qkv_dim == kv_a_dim);  // V4: queries live in latent space.

  // The MTP block has layer_idx == num_layers, past the end of the window
  // table; it uses the same sliding window as the other layers.
  const size_t window = layer_idx < config.attention_window_sizes.size()
                            ? config.attention_window_sizes[layer_idx]
                        : !config.attention_window_sizes.empty()
                            ? config.attention_window_sizes[0]
                            : config.max_seq_len;
  const bool has_compressor = lc.HasCompressor();
  const bool has_indexer = lc.HasIndexer() && layer.idx_q_w.HasPtr();
  const size_t rate = lc.kv_compression_rate;

  // Raw path (sliding-window layers) uses rope_theta; compressed layers use
  // compress_rope_theta with YaRN for everything.
  const float* HWY_RESTRICT inv_ts =
      has_compressor ? activations.mla_inv_timescale_c.PackedScale1()
                     : activations.mla_inv_timescale.PackedScale1();

  const hwy::Divisor div_qbatch(static_cast<uint32_t>(qbatch.Size()));
  const size_t num_interleaved = num_tokens * qbatch.Size();

  att.q.OverrideCols(heads * qkv_dim);
  activations.mla_kv_a.OverrideCols(kv_a_dim);

  // ---- Projections (tiled MatMul).
  if (lc.q_lora_rank > 0) {
    activations.mla_q_a.OverrideCols(lc.q_lora_rank);
    CallMatMul(att.pre_att_rms_out, layer.mla_q_a, /*add=*/nullptr, env,
               activations.mla_q_a);
    RMSNormInplaceBatched</*kPlainWeight=*/true>(layer.mla_q_a_norm,
                                                 activations.mla_q_a, env.ctx);
    CallMatMul(activations.mla_q_a, layer.mla_q_b, /*add=*/nullptr, env, att.q);
  } else {
    CallMatMul(att.pre_att_rms_out, layer.mla_q_b, /*add=*/nullptr, env, att.q);
  }
  CallMatMul(att.pre_att_rms_out, layer.mla_kv_a, /*add=*/nullptr, env,
             activations.mla_kv_a);

  // ---- Normalize the full latent, RoPE the decoupled key, write to cache.
  ParallelFor(
      Parallelism::kFlat, num_interleaved, env.ctx, /*cluster_idx=*/0,
      Callers::kAttComputeQKV, [&](size_t task, size_t worker) HWY_ATTR {
        const size_t qi = div_qbatch.Remainder(static_cast<uint32_t>(task));
        const size_t token_idx = div_qbatch.Divide(static_cast<uint32_t>(task));
        const size_t cache_pos = qbatch.Pos(qi) + token_idx;
        HWY_DASSERT(cache_pos < att.SeqLen());
        float* HWY_RESTRICT kv = activations.mla_kv_a.Row(task);
        // V4: kv_norm covers the full latent (rope applied after).
        ScaledRMSNorm(layer.mla_kv_a_norm, kv, kv_a_dim, env.ctx, worker);
        Rope(kv + kv_a_dim - rope_dim, rope_dim, inv_ts,
             static_cast<int>(cache_pos), env.ctx, worker);
        KV_t* HWY_RESTRICT dst =
            qbatch.KV(qi).kv_cache.Row(cache_pos) +
            LatentLayerOffset(qbatch.KV(qi), layer_idx, lc);
        CompressPerThread tls;
        Compress(kv, kv_a_dim, tls, MakeSpan(dst, kv_a_dim), 0);
      });

  // ---- Compressors (sequential per query) and indexer selection.
  if (has_compressor) {
    DeepSeekRunCompressors(num_tokens, layer_idx, layer, activations, qbatch,
                           env, inv_ts);
  }
  hwy::AlignedVector<hwy::AlignedVector<uint32_t>> selected;
  if (has_indexer) {
    selected.resize(num_interleaved);
    DeepSeekIndexerSelect(num_tokens, layer_idx, layer, activations, qbatch,
                          env, inv_ts, selected);
  }

  // ---- Attention sink logits.
  HWY_ALIGN float sink[kDSMaxHeadDim];
  ReadRowF32(layer.attn_sink, 0, sink, heads);

  // ---- Per (token, head): scores over window + compressed entries, online
  // softmax with sink, inverse-RoPE of the output.
  const float query_scale = att.query_scale;
  ParallelFor(
      Parallelism::kFlat, num_interleaved * heads, env.ctx, /*cluster_idx=*/0,
      Callers::kAttComputeQKV, [&](size_t task, size_t worker) HWY_ATTR {
        namespace hn = hwy::HWY_NAMESPACE;
        const hn::ScalableTag<float> df;
        const size_t head = task % heads;
        const size_t interleaved_idx = task / heads;
        const size_t qi =
            div_qbatch.Remainder(static_cast<uint32_t>(interleaved_idx));
        const size_t token_idx =
            div_qbatch.Divide(static_cast<uint32_t>(interleaved_idx));
        const size_t cache_pos = qbatch.Pos(qi) + token_idx;
        const size_t end = cache_pos + 1;

        float* HWY_RESTRICT q = att.q.Row(interleaved_idx) + head * qkv_dim;
        // Per-head RMS (no scale), then RoPE the decoupled part. This task
        // owns the range.
        RMSNormNoScaleInplace(q, qkv_dim, env.ctx, worker);
        Rope(q + qkv_dim - rope_dim, rope_dim, inv_ts,
             static_cast<int>(cache_pos), env.ctx, worker);

        const size_t layer_offset =
            LatentLayerOffset(qbatch.KV(qi), layer_idx, lc);
        const size_t comp_offset = layer_offset + kv_a_dim;

        DSOnlineSoftmax softmax;
        softmax.Reset(kv_a_dim);
        // Rows are gathered into blocks so the softmax update can use one
        // vectorized Exp per block instead of a scalar expf per row.
        HWY_ALIGN float blk_vals[kDSSoftmaxBlock][kDSMaxLatentDim];
        HWY_ALIGN float blk_logits[kDSSoftmaxBlock];
        size_t blk_n = 0;
        const auto flush = [&]() HWY_ATTR {
          if (blk_n > 0) {
            softmax.UpdateBlock(blk_logits, blk_vals, blk_n);
            blk_n = 0;
          }
        };
        const auto process_row = [&](size_t row, size_t offset) HWY_ATTR {
          const KV_t* HWY_RESTRICT src =
              qbatch.KV(qi).kv_cache.Row(row) + offset;
          DecompressAndZeroPad(df, MakeSpan(src, kv_a_dim), 0, blk_vals[blk_n],
                               kv_a_dim);
          blk_logits[blk_n] = query_scale * Dot(q, blk_vals[blk_n], kv_a_dim);
          if (++blk_n == kDSSoftmaxBlock) flush();
        };

        // Sliding window of raw latents.
        const size_t win_start = end > window ? end - window : 0;
        for (size_t p = win_start; p < end; ++p) {
          process_row(p, layer_offset);
        }
        // Compressed entries: indexer selection (CSA) or all sealed (HCA).
        if (has_compressor) {
          if (has_indexer) {
            for (const uint32_t e : selected[interleaved_idx]) {
              process_row((e + 1) * rate - 1, comp_offset);
            }
          } else {
            const size_t num_sealed = end / rate;
            for (size_t e = 0; e < num_sealed; ++e) {
              process_row((e + 1) * rate - 1, comp_offset);
            }
          }
        }
        flush();
        // Learned attention sink: contributes to the denominator only. Not
        // scaled by query_scale (matches the reference kernel).
        softmax.UpdateLogitOnly(sink[head]);
        softmax.Finalize();

        // Inverse-RoPE the rope dims of the output (values carry rotation).
        Rope(softmax.acc + kv_a_dim - rope_dim, rope_dim, inv_ts,
             -static_cast<int>(cache_pos), env.ctx, worker);

        float* HWY_RESTRICT out =
            att.att_out.Row(interleaved_idx) + head * qkv_dim;
        memcpy(out, softmax.acc, kv_a_dim * sizeof(float));
      });

  // ---- Grouped low-rank output projection.
  att.att_out.OverrideCols(heads * qkv_dim);
  const size_t o_groups = lc.o_groups;
  const size_t o_lora = lc.o_lora_rank;
  const size_t group_in = heads / o_groups * qkv_dim;
  activations.mla_o_in.OverrideCols(group_in);
  activations.mla_o_group.OverrideCols(o_lora);
  activations.mla_o_mid.OverrideCols(o_groups * o_lora);
  const size_t num_rows = att.att_out.Rows();
  for (size_t g = 0; g < o_groups; ++g) {
    // Copy this group's slice of att_out.
    for (size_t r = 0; r < num_rows; ++r) {
      const float* HWY_RESTRICT src = att.att_out.Row(r) + g * group_in;
      std::copy_n(src, group_in, activations.mla_o_in.Row(r));
    }
    // Row-slice view of wo_a for this group.
    MatPtr o_a_g("o_a_g", layer.mla_o_a.GetType(), Extents2D(o_lora, group_in));
    o_a_g.SetPtr(const_cast<uint8_t*>(layer.mla_o_a.RowBytes(g * o_lora)),
                 layer.mla_o_a.Stride());
    o_a_g.SetScale(layer.mla_o_a.Scale());
    CallMatMul(activations.mla_o_in, o_a_g, /*add=*/nullptr, env,
               activations.mla_o_group);
    for (size_t r = 0; r < num_rows; ++r) {
      std::copy_n(activations.mla_o_group.Row(r), o_lora,
                  activations.mla_o_mid.Row(r) + g * o_lora);
    }
  }
  CallMatMul(activations.mla_o_mid, layer.mla_o_b, /*add=*/nullptr, env,
             att.att_sums);
}

// ------------------------------ MoE ------------------------------

// DeepSeek MoE: sigmoid or sqrt(softplus) scoring with an aux-loss-free
// routing bias (selection only) or hash-based routing, top-k normalization
// and routed weight scale.
struct DeepSeekMoE {
  static RouterScoreFunc Scoring(const LayerConfig& lc) {
    if (lc.router_score != RouterScoreFunc::kSigmoidGatingCompat) {
      return lc.router_score;
    }
    return lc.sigmoid_gating ? RouterScoreFunc::kSigmoid
                             : RouterScoreFunc::kSoftmax;
  }

  static HWY_NOINLINE void ChooseExperts(const LayerWeightsPtrs& layer,
                                         Activations& activations,
                                         std::atomic<uint32_t>* expert_sizes,
                                         MatMulEnv& env) {
    const LayerConfig& lc = layer.layer_config;
    const size_t num_experts = lc.NumExperts();
    const size_t experts_per_token = lc.NumExpertsPerDatapoint();
    const size_t num_tokens = activations.x.Rows();
    HWY_DASSERT(num_experts <= kDSMaxExperts);

    activations.router_logits.OverrideRows(num_tokens);
    // Router input is the (already RMSNorm'd) FFN input.
    CallMatMul(activations.pre_ffw_rms_out, layer.moe_router,
               /*add=*/nullptr, env, activations.router_logits);

    HWY_ALIGN float bias[kDSMaxExperts] = {};
    if (layer.moe_router_bias.HasPtr()) {
      ReadRowF32(layer.moe_router_bias, 0, bias, num_experts);
    }
    const RouterScoreFunc scoring = Scoring(lc);
    const bool hash = lc.hash_routing && layer.hash_tid2eid.HasPtr();
    const float route_scale = lc.route_scale;

    ParallelFor(
        Parallelism::kWithinCluster, num_tokens, env.ctx, /*cluster_idx=*/0,
        Callers::kMoEChooseExperts,
        [&](size_t token_idx, size_t worker) HWY_ATTR {
          GCPP_ZONE(env.ctx, worker, Zones::kGenMoEFFWExpertsChoose);
          const float* HWY_RESTRICT logits =
              activations.router_logits.Row(token_idx);

          // Gate probabilities; the bias affects only expert *selection*.
          HWY_ALIGN float probs[kDSMaxExperts];
          for (uint32_t e = 0; e < num_experts; ++e) {
            switch (scoring) {
              case RouterScoreFunc::kSigmoid:
                probs[e] = 1.0f / (1.0f + expf(-logits[e]));
                break;
              case RouterScoreFunc::kSqrtSoftplus: {
                // softplus(x) = log1p(exp(x)), stable for large x.
                const float sp =
                    logits[e] > 20.0f ? logits[e] : log1pf(expf(logits[e]));
                probs[e] = sqrtf(sp);
                break;
              }
              default:
                probs[e] = logits[e];  // softmax applied below
                break;
            }
          }
          if (scoring == RouterScoreFunc::kSoftmax) {
            float max_logit = logits[0];
            for (uint32_t e = 1; e < num_experts; ++e) {
              max_logit = HWY_MAX(max_logit, logits[e]);
            }
            float sum = 0.0f;
            for (uint32_t e = 0; e < num_experts; ++e) {
              probs[e] = expf(logits[e] - max_logit);
              sum += probs[e];
            }
            for (uint32_t e = 0; e < num_experts; ++e) probs[e] /= sum;
          }

          // Selection: hash table lookup or top-k over biased scores.
          uint32_t chosen[64];
          HWY_DASSERT(experts_per_token <= 64);
          if (hash) {
            const int32_t token_id = token_idx < activations.token_ids.size()
                                         ? activations.token_ids[token_idx]
                                         : 0;
            HWY_ALIGN float row[64];
            ReadRowF32(layer.hash_tid2eid, static_cast<size_t>(token_id), row,
                       experts_per_token);
            for (size_t i = 0; i < experts_per_token; ++i) {
              chosen[i] = static_cast<uint32_t>(row[i]);
              HWY_DASSERT(chosen[i] < num_experts);
            }
          } else {
            hwy::K32V32 sel[kDSMaxExperts];
            for (uint32_t e = 0; e < num_experts; ++e) {
              sel[e] = {.value = e,
                        .key = DSFloatToUint32Sortkey(probs[e] + bias[e])};
            }
            hwy::HWY_NAMESPACE::VQSortStatic(sel, num_experts,
                                             hwy::SortDescending());
            for (size_t i = 0; i < experts_per_token; ++i) {
              chosen[i] = sel[i].value;
            }
          }

          PerToken* per_token = activations.GetPerToken(token_idx);
          float weight_sum = 0.0f;
          for (size_t i = 0; i < experts_per_token; ++i) {
            const size_t expert_idx = chosen[i];
            HWY_DASSERT(expert_idx < num_experts);
            const uint32_t row = expert_sizes[expert_idx].fetch_add(
                1, std::memory_order_relaxed);
            per_token[i].weight = probs[expert_idx];
            weight_sum += probs[expert_idx];
            per_token[i].expert_idx = static_cast<uint16_t>(expert_idx);
            per_token[i].row_idx = static_cast<uint16_t>(row);
          }
          // Normalize the selected weights to sum to 1 (all non-softmax
          // score functions), then apply the routed scale.
          for (size_t i = 0; i < experts_per_token; ++i) {
            if (scoring != RouterScoreFunc::kSoftmax) {
              per_token[i].weight /= weight_sum;
            }
            per_token[i].weight *= route_scale;
          }
        });
  }

  // Clamps C1 (gate, max only) and C2 (up, +/-) to swiglu_limit.
  static HWY_NOINLINE void ClampSwiGLU(MatPtrT<BF16>& C1, MatPtrT<BF16>& C2,
                                       size_t rows, size_t cols, float limit) {
    for (size_t r = 0; r < rows; ++r) {
      BF16* HWY_RESTRICT gate = C1.Row(r);
      BF16* HWY_RESTRICT up = C2.Row(r);
      for (size_t c = 0; c < cols; ++c) {
        const float g = hwy::ConvertScalarTo<float>(gate[c]);
        const float u = hwy::ConvertScalarTo<float>(up[c]);
        gate[c] = hwy::ConvertScalarTo<BF16>(HWY_MIN(g, limit));
        up[c] = hwy::ConvertScalarTo<BF16>(HWY_MAX(HWY_MIN(u, limit), -limit));
      }
    }
  }

  // Computes `activations.ffw_expert_out` for one activated expert.
  static HWY_NOINLINE void ComputeExpertOutput(
      const LayerWeightsPtrs& layer, Activations& activations,
      const size_t expert_idx, const size_t expert_size,
      const uint16_t* HWY_RESTRICT expert_tokens, Parallelism parallelism,
      const size_t cluster_idx, MatMulEnv& env) {
    HWY_DASSERT(expert_idx < layer.layer_config.NumExperts());
    HWY_DASSERT(expert_size != 0);
    HWY_DASSERT(cluster_idx < activations.per_cluster.size());
    Activations::PerCluster& per_cluster = activations.per_cluster[cluster_idx];

    const size_t model_dim = activations.attention.config.model_dim;
    const size_t worker = env.ctx.Worker(cluster_idx);
    GCPP_ZONE(env.ctx, worker, Zones::kGenMoEFFWExperts);

    MatPtrT<BF16>& tmp_in = per_cluster.ffw_expert_in;
    MatPtrT<BF16>& expert_out = activations.ffw_expert_out[expert_idx];
    tmp_in.OverrideRows(expert_size);
    expert_out.OverrideRows(expert_size);

    {
      GCPP_ZONE(env.ctx, worker, Zones::kGenMoEFFWExpertsGather);
      for (size_t i = 0; i < expert_size; ++i) {
        const size_t token_idx = expert_tokens[i];
        std::copy_n(activations.pre_ffw_rms_out.Row(token_idx), model_dim,
                    tmp_in.Row(i));
      }
    }

    const ActivationType activation = layer.layer_config.activation;
    const float swiglu_limit = layer.layer_config.swiglu_limit;
    MatPtrT<BF16>& C1 = per_cluster.moe_C1;
    C1.OverrideRows(expert_size);

    const auto fused = [&](RowPtrsBF C1_rows, IndexRange range_r,
                           IndexRange range_c, StridedViewBF C2,
                           size_t fused_worker) {
      Activation(activation, C1_rows, range_r, range_c, C2, env.ctx,
                 fused_worker);
    };
    MMOptions options{.cluster_idx = static_cast<uint32_t>(cluster_idx),
                      .parallelism = parallelism};

    const size_t expert_ff_hidden_dim =
        layer.moe_gating_einsum_w1[expert_idx].Rows();

    if (GEMMA_FUSED_FFN && swiglu_limit == 0.0f) {
      options.SetFunc(fused);
      CallTwoMatMul(tmp_in, layer.moe_gating_einsum_w1[expert_idx],
                    layer.moe_gating_einsum_w2[expert_idx], env, C1, options);
      options.func = nullptr;
    } else {
      MatPtrT<BF16>& C2 = per_cluster.moe_C2;
      C2.OverrideRows(expert_size);
      CallMatMul(tmp_in, layer.moe_gating_einsum_w1[expert_idx],
                 /*add=*/nullptr, env, C1, options);
      CallMatMul(tmp_in, layer.moe_gating_einsum_w2[expert_idx],
                 /*add=*/nullptr, env, C2, options);
      if (swiglu_limit > 0.0f) {
        ClampSwiGLU(C1, C2, expert_size, expert_ff_hidden_dim, swiglu_limit);
      }
      ActivationBatched(activation, C1, &C2, env.ctx, cluster_idx,
                        options.parallelism);
    }

    // Hidden layer -> output layer, via a buffer of the expert's exact width.
    MatStorageT<BF16> C1_narrow("C1_n",
                                Extents2D(expert_size, expert_ff_hidden_dim),
                                env.ctx.allocator, MatPadding::kOdd);
    for (size_t i = 0; i < expert_size; ++i) {
      memcpy(C1_narrow.Row(i), C1.Row(i), expert_ff_hidden_dim * sizeof(BF16));
    }
    CallMatMul(C1_narrow, layer.moe_linear_w[expert_idx],
               /*add=*/nullptr, env, expert_out, options);
  }

  static HWY_NOINLINE void ComputeAllExpertOutputs(
      const LayerWeightsPtrs& layer, Activations& activations,
      const uint32_t* HWY_RESTRICT expert_sizes, MatMulEnv& env) {
    HWY_ALIGN uint32_t expert_pos[kDSMaxExperts];
    const size_t num_experts = layer.layer_config.NumExperts();
    const size_t num_tokens = activations.x.Rows();
    const size_t experts_per_token =
        layer.layer_config.NumExpertsPerDatapoint();
    const uint32_t expert_size_sum =
        ExclusivePrefixSum(expert_sizes, num_experts, expert_pos);
    HWY_DASSERT(expert_size_sum == num_tokens * experts_per_token);
    (void)expert_size_sum;
    (void)num_tokens;

    uint32_t activated_expert_idx[kDSMaxExperts];
    size_t num_activated_experts = 0;
    for (size_t token_idx = 0; token_idx < activations.x.Rows(); ++token_idx) {
      const PerToken* per_token = activations.GetPerToken(token_idx);
      for (size_t i = 0; i < experts_per_token; ++i) {
        const size_t expert_idx = per_token[i].expert_idx;
        const size_t row = per_token[i].row_idx;
        activations.expert_tokens[expert_pos[expert_idx] + row] =
            static_cast<uint16_t>(token_idx);
        activated_expert_idx[num_activated_experts] =
            static_cast<uint32_t>(expert_idx);
        num_activated_experts += (row == 0);
      }
    }

    Parallelism outer = Parallelism::kAcrossClusters;
    Parallelism inner = Parallelism::kWithinCluster;
    if (num_activated_experts < activations.per_cluster.size()) {
      outer = Parallelism::kNone;
      inner = Parallelism::kHierarchical;
    }
    ParallelFor(outer, num_activated_experts, env.ctx,
                /*cluster_idx=*/0, Callers::kMoEComputeAllExpertOutputs,
                [&](size_t expert_idx_idx, size_t cluster_idx) {
                  const size_t expert_idx =
                      activated_expert_idx[expert_idx_idx];
                  ComputeExpertOutput(
                      layer, activations, expert_idx, expert_sizes[expert_idx],
                      &activations.expert_tokens[expert_pos[expert_idx]], inner,
                      cluster_idx, env);
                });
  }

  static HWY_NOINLINE void WeightedSumOfExperts(
      const LayerWeightsPtrs& layer, Activations& activations,
      const uint32_t* HWY_RESTRICT expert_sizes, MatMulEnv& env) {
    const size_t experts_per_token =
        layer.layer_config.NumExpertsPerDatapoint();
    const size_t model_dim = activations.attention.config.model_dim;

    ParallelFor(
        Parallelism::kFlat, activations.x.Rows(), env.ctx,
        /*cluster_idx=*/0, Callers::kMoEWeightedSumOfExperts,
        [&](size_t token_idx, size_t worker) {
          GCPP_ZONE(env.ctx, worker, Zones::kGenMoEFFWWeightedSum);
          const PerToken* per_token = activations.GetPerToken(token_idx);
          const auto get_expert_row = [&](size_t i) {
            const size_t expert_idx = per_token[i].expert_idx;
            const size_t row = per_token[i].row_idx;
            HWY_DASSERT(row < expert_sizes[expert_idx]);
            return activations.ffw_expert_out[expert_idx].Row(row);
          };
          MulByConstTo(per_token[0].weight, get_expert_row(0),
                       activations.ffw_out.Row(token_idx), model_dim, env.ctx,
                       worker);
          for (size_t i = 1; i < experts_per_token; ++i) {
            MulByConstAndAdd(per_token[i].weight, get_expert_row(i),
                             activations.ffw_out.Row(token_idx), model_dim);
          }
        });
  }

  static HWY_NOINLINE void MoEFFW(const LayerWeightsPtrs& layer,
                                  Activations& activations, MatMulEnv& env) {
    GCPP_ZONE(env.ctx, /*global_idx=*/0, Zones::kGenMoEFFW);

    HWY_ALIGN std::atomic<uint32_t> atomic_expert_sizes[kDSMaxExperts] = {};
    ChooseExperts(layer, activations, atomic_expert_sizes, env);

    HWY_ALIGN uint32_t expert_sizes[kDSMaxExperts];
    const size_t num_experts = layer.layer_config.NumExperts();
    for (size_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
      expert_sizes[expert_idx] =
          atomic_expert_sizes[expert_idx].load(std::memory_order_acquire);
    }

    ComputeAllExpertOutputs(layer, activations, expert_sizes, env);
    WeightedSumOfExperts(layer, activations, expert_sizes, env);
  }
};

// Shared-expert / dense FFN with optional SwiGLU clamp (the fused FFWNoVit
// path cannot clamp). Writes ffw_out.
static HWY_NOINLINE void DeepSeekDenseFFW(const LayerWeightsPtrs& layer,
                                          Activations& activations,
                                          MatMulEnv& env) {
  const LayerConfig& lc = layer.layer_config;
  if (lc.swiglu_limit == 0.0f) {
    FFWNoVit(layer, activations, env);
    return;
  }
  const size_t rows = activations.pre_ffw_rms_out.Rows();
  CallMatMul(activations.pre_ffw_rms_out, layer.gating_einsum_w1,
             /*add=*/nullptr, env, activations.C1);
  CallMatMul(activations.pre_ffw_rms_out, layer.gating_einsum_w2,
             /*add=*/nullptr, env, activations.C2);
  DeepSeekMoE::ClampSwiGLU(activations.C1, activations.C2, rows,
                           lc.ff_hidden_dim, lc.swiglu_limit);
  ActivationBatched(lc.activation, activations.C1, &activations.C2, env.ctx,
                    /*cluster_idx=*/0, Parallelism::kFlat);
  CallMatMul(activations.C1, layer.linear_w, /*add=*/nullptr, env,
             activations.ffw_out);
}

// ------------------------------ Layer ------------------------------

void DeepSeekTransformerLayer(size_t num_tokens, size_t layer_idx,
                              const LayerWeightsPtrs& layer,
                              Activations& activations, QBatch& qbatch,
                              MatMulEnv& env) {
  GCPP_ZONE(env.ctx, hwy::Profiler::GlobalIdx(),
            Zones::kGenTotalTransformerLayer);

  const ModelConfig& config = activations.attention.config;
  const LayerConfig& lc = layer.layer_config;
  const bool has_hc = config.hc_mult > 1 && layer.hc_att_fn.HasPtr();

  // ---- Attention block.
  if (has_hc) {
    HCReadDynamic(layer.hc_att_fn, layer.hc_att_base, layer.hc_att_scale,
                  activations, env);
  }
  RMSNormBatched</*kPlainWeight=*/true>(
      activations.x, layer.pre_attention_norm_scale,
      activations.attention.pre_att_rms_out, env.ctx);

  HWY_DASSERT(lc.type == LayerAttentionType::kDeepSeekMLA);
  HWY_DASSERT(qbatch.PrefixEnd(0) == 0);  // expect causal attention
  DeepSeekAttention(num_tokens, layer_idx, layer, activations, qbatch, env);

  if (has_hc) {
    HCWriteDynamic(activations.attention.att_sums, activations, env);
  } else {
    ResidualConnection(activations.attention.att_sums, activations.x, layer,
                       /*is_attention=*/true, env.ctx);
  }

  // ---- FFN block.
  if (has_hc) {
    HCReadDynamic(layer.hc_ffw_fn, layer.hc_ffw_base, layer.hc_ffw_scale,
                  activations, env);
  }
  RMSNormBatched</*kPlainWeight=*/true>(activations.x, layer.pre_ffw_norm_scale,
                                        activations.pre_ffw_rms_out, env.ctx);

  activations.C1.OverrideCols(lc.ff_hidden_dim);
  activations.C2.OverrideCols(lc.ff_hidden_dim);

  if (lc.IsMoE()) {
    const bool has_shared =
        lc.num_shared_experts > 0 && layer.gating_einsum_w1.HasPtr();
    if (has_shared) {
      // Shared expert = the layer's dense FFN tensors. Stash its output in
      // att_sums (BF16) while the routed experts run.
      DeepSeekDenseFFW(layer, activations, env);
      for (size_t r = 0; r < activations.x.Rows(); ++r) {
        const float* HWY_RESTRICT from = activations.ffw_out.Row(r);
        BF16* HWY_RESTRICT to = activations.attention.att_sums.Row(r);
        for (size_t c = 0; c < config.model_dim; ++c) {
          to[c] = hwy::ConvertScalarTo<BF16>(from[c]);
        }
      }
    }
    DeepSeekMoE::MoEFFW(layer, activations, env);  // writes ffw_out
    if (has_shared) {
      AddFromBatched(activations.attention.att_sums, activations.ffw_out,
                     env.ctx);
    }
  } else {
    DeepSeekDenseFFW(layer, activations, env);
  }

  if (has_hc) {
    HCWriteDynamic(activations.ffw_out, activations, env);
  } else {
    ResidualConnection(activations.ffw_out, activations.x, layer,
                       /*is_attention=*/false, env.ctx);
  }
}

// Final norm before the output head with plain weights: DeepSeek checkpoints
// store the true scale, so gemma's (1 + w) RMSNormBatched must not be used.
void DeepSeekFinalNorm(const WeightsPtrs& weights, Activations& activations,
                       MatMulEnv& env) {
  RMSNormBatched</*kPlainWeight=*/true>(activations.x, weights.final_norm_scale,
                                        activations.x_bf, env.ctx);
}

// ------------------------------ MTP ------------------------------
// Runs the multi-token-prediction block (reference `MTPBlock.forward`) on the
// first `num_tokens` batch rows. Inputs per row r: activations.hc_streams
// (the main model's pre-collapse residual streams, still intact after the
// main pass and its head collapse) and `next_tokens[r]`, the committed token
// following that row's position. Rows are processed at cache positions
// qbatch.Pos(0) + r, matching the main pass; the MTP block's KV segment has
// layer index `num_layers`. If `compute_logits`, collapses with the MTP head
// tensors, applies the MTP final norm and the shared output head, writing
// activations.logits; otherwise only the MTP KV cache is updated (prefill).
void DeepSeekMTPStep(size_t num_tokens, const int* next_tokens,
                     bool compute_logits, const WeightsPtrs& weights,
                     Activations& activations, QBatch& qbatch, MatMulEnv& env) {
  HWY_ASSERT(qbatch.Size() == 1 && !weights.mtp_layers.empty());
  const ModelConfig& config = activations.attention.config;
  const LayerWeightsPtrs& mtp = weights.mtp_layers[0];
  const size_t model_dim = config.model_dim;
  const size_t hc_mult = config.hc_mult;
  const size_t mtp_layer_idx = weights.c_layers.size();
  HWY_ASSERT(hc_mult > 1);  // V4; the reference MTP head assumes streams.

  activations.SetBatchSize(num_tokens);  // OverrideRows: keeps hc_streams.

  // Stash h = streams in hc_tmp, embed the next tokens into x.
  activations.token_ids.resize(num_tokens);
  for (size_t r = 0; r < num_tokens; ++r) {
    memcpy(activations.hc_tmp.Row(r), activations.hc_streams.Row(r),
           hc_mult * model_dim * sizeof(float));
    activations.token_ids[r] = next_tokens[r];
    // Raw embedding row (DeepSeek does not scale embeddings), then enorm.
    float* HWY_RESTRICT e = activations.x.Row(r);
    ReadRowF32(weights.embedder_input_embedding,
               static_cast<size_t>(next_tokens[r]), e, model_dim);
    ScaledRMSNorm(weights.mtp_enorm, e, model_dim, env.ctx, 0);
  }

  // e_proj over all rows: [num_tokens, model_dim] -> ffw_out.
  CallMatMul(activations.x, weights.mtp_e_proj, /*add=*/nullptr, env,
             activations.ffw_out);

  // streams = h_proj(hnorm(h)) + e_proj(enorm(e)) broadcast over streams.
  for (size_t r = 0; r < num_tokens; ++r) {
    float* HWY_RESTRICT h = activations.hc_tmp.Row(r);
    for (size_t i = 0; i < hc_mult; ++i) {
      ScaledRMSNorm(weights.mtp_hnorm, h + i * model_dim, model_dim, env.ctx,
                    0);
    }
    // Per-row [hc_mult, model_dim] views; rows within a batch row are packed.
    MatPtrT<float> h_view("mtp_h", Extents2D(hc_mult, model_dim));
    h_view.SetPtr(h, model_dim);
    MatPtrT<float> s_view("mtp_s", Extents2D(hc_mult, model_dim));
    s_view.SetPtr(activations.hc_streams.Row(r), model_dim);
    CallMatMul(h_view, weights.mtp_h_proj, /*add=*/nullptr, env, s_view);
    const float* HWY_RESTRICT e = activations.ffw_out.Row(r);
    for (size_t i = 0; i < hc_mult; ++i) {
      MulByConstAndAdd(1.0f, e, activations.hc_streams.Row(r) + i * model_dim,
                       model_dim);
    }
  }

  // The MTP block is a full V4 layer (dense MLA + MoE + per-block mHC).
  DeepSeekTransformerLayer(num_tokens, mtp_layer_idx, mtp, activations, qbatch,
                           env);

  if (!compute_logits) return;
  HCHeadCollapse(weights.mtp_hc_fn, weights.mtp_hc_base, weights.mtp_hc_scale,
                 activations, env);
  RMSNormBatched</*kPlainWeight=*/true>(activations.x, weights.mtp_norm,
                                        activations.x_bf, env.ctx);
  const MatPtr& head = weights.lm_head.HasPtr()
                           ? weights.lm_head
                           : weights.embedder_input_embedding;
  CallMatMul(activations.x_bf, head, /*add=*/nullptr, env, activations.logits);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();
