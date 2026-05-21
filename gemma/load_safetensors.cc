// Copyright 2025 Google LLC
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

// Implementation of WeightsPtrs::LoadFromSafetensors.
// Loads HuggingFace safetensors weights for Gemma 4 (E2B / E4B) directly
// into gemma.cpp's weight layout, with no BlobStore conversion step.

#include "gemma/weights.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>

#include "gemma/configs.h"
#include "gemma/tensor_info.h"
#include "io/safetensors.h"
#include "util/allocator.h"
#include "util/mat.h"
#include "util/threading_context.h"
#include "compression/types.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"

namespace gcpp {

namespace {

// Returns a HF layer tensor name for layer `i`.
static inline std::string LN(const char* tail, size_t i) {
  return "model.layers." + std::to_string(i) + "." + tail;
}

// Validates that the safetensor entry has the expected element count.
static void ValidateShape(const SafetensorEntry& e, const char* hf_name,
                          size_t expected_elems) {
  const uint64_t actual = SafetensorNumElems(e.shape);
  if (actual != expected_elems) {
    HWY_ABORT("safetensors: '%s' expected %zu elems, got %zu", hf_name,
              expected_elems, static_cast<size_t>(actual));
  }
  if (e.dtype != "BF16") {
    HWY_ABORT("safetensors: '%s' dtype '%s' not supported (need BF16)",
              hf_name, e.dtype.c_str());
  }
}

// Looks up `hf_name`, validates shape, allocates `mat` as BF16/kPacked, and
// reads the contiguous BF16 bytes directly into the allocated memory.
static void AllocAndReadDirect(MatPtr& mat, std::vector<MatOwner>& owners,
                               const Allocator& alloc,
                               const SafetensorsIndex& idx,
                               const char* hf_name) {
  const SafetensorEntry* e = idx.Find(hf_name);
  if (!e) {
    HWY_ABORT("safetensors: required tensor '%s' not found", hf_name);
  }
  const size_t expected = mat.Rows() * mat.Cols();
  ValidateShape(*e, hf_name, expected);
  mat.SetType(Type::kBF16);
  owners.emplace_back();
  owners.back().AllocateFor(mat, alloc, MatPadding::kPacked);
  if (!idx.ReadTensor(*e, mat.Packed())) {
    HWY_ABORT("safetensors: failed to read '%s'", hf_name);
  }
}

// Reads two contiguous HF tensors (rows × cols each) and concatenates them
// vertically into `mat` (2*rows × cols).
static void AllocAndReadConcat2(MatPtr& mat, std::vector<MatOwner>& owners,
                                const Allocator& alloc,
                                const SafetensorsIndex& idx,
                                const char* hf_a, const char* hf_b) {
  const SafetensorEntry* ea = idx.Find(hf_a);
  const SafetensorEntry* eb = idx.Find(hf_b);
  if (!ea) HWY_ABORT("safetensors: tensor '%s' not found", hf_a);
  if (!eb) HWY_ABORT("safetensors: tensor '%s' not found", hf_b);

  const size_t rows_a = ea->num_bytes / (mat.Cols() * 2);  // 2 bytes/BF16
  const size_t rows_b = eb->num_bytes / (mat.Cols() * 2);
  if (rows_a + rows_b != mat.Rows()) {
    HWY_ABORT(
        "safetensors: concat '%s'(%zu) + '%s'(%zu) rows != expected %zu",
        hf_a, rows_a, hf_b, rows_b, mat.Rows());
  }
  if (ea->dtype != "BF16")
    HWY_ABORT("safetensors: '%s' dtype '%s' not BF16", hf_a, ea->dtype.c_str());
  if (eb->dtype != "BF16")
    HWY_ABORT("safetensors: '%s' dtype '%s' not BF16", hf_b, eb->dtype.c_str());

  mat.SetType(Type::kBF16);
  owners.emplace_back();
  owners.back().AllocateFor(mat, alloc, MatPadding::kPacked);

  uint8_t* dst = static_cast<uint8_t*>(mat.Packed());
  if (!idx.ReadTensor(*ea, dst)) HWY_ABORT("safetensors: read failed: '%s'", hf_a);
  if (!idx.ReadTensor(*eb, dst + ea->num_bytes))
    HWY_ABORT("safetensors: read failed: '%s'", hf_b);
}

// Reads three contiguous HF tensors and concatenates vertically into `mat`.
static void AllocAndReadConcat3(MatPtr& mat, std::vector<MatOwner>& owners,
                                const Allocator& alloc,
                                const SafetensorsIndex& idx,
                                const char* hf_a, const char* hf_b,
                                const char* hf_c) {
  const SafetensorEntry* ea = idx.Find(hf_a);
  const SafetensorEntry* eb = idx.Find(hf_b);
  const SafetensorEntry* ec = idx.Find(hf_c);
  if (!ea) HWY_ABORT("safetensors: tensor '%s' not found", hf_a);
  if (!eb) HWY_ABORT("safetensors: tensor '%s' not found", hf_b);
  if (!ec) HWY_ABORT("safetensors: tensor '%s' not found", hf_c);

  const uint64_t total_bytes = ea->num_bytes + eb->num_bytes + ec->num_bytes;
  const uint64_t expected_bytes = mat.Rows() * mat.Cols() * 2;
  if (total_bytes != expected_bytes) {
    HWY_ABORT(
        "safetensors: concat3 '%s'+'%s'+'%s' bytes %" PRIu64
        " != expected %" PRIu64,
        hf_a, hf_b, hf_c, total_bytes, expected_bytes);
  }
  if (ea->dtype != "BF16" || eb->dtype != "BF16" || ec->dtype != "BF16") {
    HWY_ABORT("safetensors: expected BF16 for Q/K/V projections");
  }

  mat.SetType(Type::kBF16);
  owners.emplace_back();
  owners.back().AllocateFor(mat, alloc, MatPadding::kPacked);

  uint8_t* dst = static_cast<uint8_t*>(mat.Packed());
  if (!idx.ReadTensor(*ea, dst)) HWY_ABORT("safetensors: read failed: '%s'", hf_a);
  dst += ea->num_bytes;
  if (!idx.ReadTensor(*eb, dst)) HWY_ABORT("safetensors: read failed: '%s'", hf_b);
  dst += eb->num_bytes;
  if (!idx.ReadTensor(*ec, dst)) HWY_ABORT("safetensors: read failed: '%s'", hf_c);
}

// Loads per_layer_token_embd from HF shape [L, V, D] into gemma shape
// [L*D, V] by transposing the [V, D] sub-matrix for each layer.
// HF: data[l*V*D + v*D + d]  →  gemma row (l*D+d), col v
static void LoadPerLayerEmbd(MatPtr& mat, std::vector<MatOwner>& owners,
                             const Allocator& alloc,
                             const SafetensorsIndex& idx,
                             size_t num_layers, size_t vocab_size,
                             size_t embd_dim) {
  const char* hf_name = "model.per_layer_token_embd.weight";
  const SafetensorEntry* e = idx.Find(hf_name);
  if (!e) HWY_ABORT("safetensors: '%s' not found", hf_name);
  ValidateShape(*e, hf_name, num_layers * vocab_size * embd_dim);

  // Read HF [L, V, D] into a temp buffer.
  const size_t total_elems = num_layers * vocab_size * embd_dim;
  auto tmp = hwy::AllocateAligned<uint16_t>(total_elems);  // BF16 = uint16
  if (!idx.ReadTensor(*e, tmp.get())) {
    HWY_ABORT("safetensors: read failed: '%s'", hf_name);
  }

  // Allocate gemma [L*D, V] (rows=L*D, cols=V) as packed BF16.
  mat.SetType(Type::kBF16);
  owners.emplace_back();
  owners.back().AllocateFor(mat, alloc, MatPadding::kPacked);

  // Transpose: gemma[l*D+d, v] = HF[l*V*D + v*D + d]
  uint16_t* dst = static_cast<uint16_t*>(mat.Packed());
  const uint16_t* src = tmp.get();
  for (size_t l = 0; l < num_layers; ++l) {
    for (size_t d = 0; d < embd_dim; ++d) {
      const size_t dst_row = l * embd_dim + d;
      uint16_t* dst_row_ptr = dst + dst_row * vocab_size;
      for (size_t v = 0; v < vocab_size; ++v) {
        dst_row_ptr[v] = src[l * vocab_size * embd_dim + v * embd_dim + d];
      }
    }
  }
}

}  // namespace

void WeightsPtrs::LoadFromSafetensors(const std::string& dir,
                                      std::vector<MatOwner>& mat_owners,
                                      ThreadingContext& ctx) {
  const Allocator& alloc = ctx.allocator;
  SafetensorsIndex idx(dir);
  const ModelConfig& cfg = config_;

  // ── Global tensors ────────────────────────────────────────────────────────
  AllocAndReadDirect(embedder_input_embedding, mat_owners, alloc, idx,
                     "model.embed_tokens.weight");
  AllocAndReadDirect(final_norm_scale, mat_owners, alloc, idx,
                     "model.norm.weight");

  if (cfg.per_layer_embd_dim > 0) {
    LoadPerLayerEmbd(per_layer_input_embedding, mat_owners, alloc, idx,
                     cfg.num_layers, cfg.vocab_size, cfg.per_layer_embd_dim);
  }

  // ── Per-layer tensors ─────────────────────────────────────────────────────
  for (size_t i = 0; i < cfg.num_layers; ++i) {
    const LayerConfig& lc = cfg.layer_configs[i];
    LayerWeightsPtrs& lw = *GetLayer(i);

    // Norm scales (shape: [model_dim] → rows=1, cols=model_dim).
    AllocAndReadDirect(lw.pre_attention_norm_scale, mat_owners, alloc, idx,
                       LN("input_layernorm.weight", i).c_str());
    AllocAndReadDirect(lw.post_attention_norm_scale, mat_owners, alloc, idx,
                       LN("post_attention_layernorm.weight", i).c_str());
    AllocAndReadDirect(lw.pre_ffw_norm_scale, mat_owners, alloc, idx,
                       LN("pre_feedforward_layernorm.weight", i).c_str());
    AllocAndReadDirect(lw.post_ffw_norm_scale, mat_owners, alloc, idx,
                       LN("post_feedforward_layernorm.weight", i).c_str());

    if (lc.use_qk_norm) {
      AllocAndReadDirect(lw.query_norm_scale, mat_owners, alloc, idx,
                         LN("self_attn.q_norm.weight", i).c_str());
      AllocAndReadDirect(lw.key_norm_scale, mat_owners, alloc, idx,
                         LN("self_attn.k_norm.weight", i).c_str());
    }

    // Attention: Q + K + V → qkv_einsum_w [(heads+2*kv_heads)*qkv_dim, model_dim]
    AllocAndReadConcat3(
        lw.qkv_einsum_w, mat_owners, alloc, idx,
        LN("self_attn.q_proj.weight", i).c_str(),
        LN("self_attn.k_proj.weight", i).c_str(),
        LN("self_attn.v_proj.weight", i).c_str());

    // Output projection: att_weights [model_dim, heads*qkv_dim] (direct).
    // HF o_proj.weight shape is already [model_dim, heads*qkv_dim]. ✓
    AllocAndReadDirect(lw.att_weights, mat_owners, alloc, idx,
                       LN("self_attn.o_proj.weight", i).c_str());

    // FFN: gate + up → gating_einsum_w [2*ff_hidden_dim, model_dim]
    AllocAndReadConcat2(
        lw.gating_einsum_w, mat_owners, alloc, idx,
        LN("mlp.gate_proj.weight", i).c_str(),
        LN("mlp.up_proj.weight", i).c_str());

    // FFN down: linear_w [model_dim, ff_hidden_dim] (direct).
    AllocAndReadDirect(lw.linear_w, mat_owners, alloc, idx,
                       LN("mlp.down_proj.weight", i).c_str());
  }

  // ── Fixup (splits qkv/gating, verifies att_weights) ─────────────────────
  Fixup(mat_owners, ctx);

  fprintf(stderr, "[safetensors] loaded %zu layers from %s\n",
          cfg.num_layers, dir.c_str());
}

}  // namespace gcpp
