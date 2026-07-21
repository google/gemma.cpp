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

// Converts a DeepSeek-V4-Flash safetensors checkpoint to the gemma.cpp
// single-file .sbs format.
//
// Usage:
//   convert_dsv4 --weights <dir with model-*.safetensors and
//                 model.safetensors.index.json> --output <file.sbs>
//                [--tokenizer_json <tokenizer.json>] [--fp4_high_first]
//
// Handles:
//  * FP8 (e4m3fn) weights with 128x128-block e8m0 scales -> SFP.
//  * FP4 (e2m1, packed two per byte along the last dim, stored as I8) expert
//    weights with per-32 e8m0 scales -> SFP.
//  * BF16/F32 tensors -> BF16/F32; I64 (hash routing table) -> F32.
//  * RoPE dim permutation: the reference applies rotary embeddings to
//    interleaved (even, odd) pairs, gemma.cpp to (i, i + dim/2) halves.
//    All tensors whose rows/cols/elements correspond to rope dims are
//    permuted accordingly at conversion time.
//  * The MTP block (mtp.0.*) is converted as an extra layer at index
//    `num_layers` plus model-level `mtp_*` extras (see tensor_info.cc).
//
// The tokenizer blob is written as kMockTokenizer (DeepSeek uses a HF BPE
// tokenizer.json, not sentencepiece); pass --tokenizer_json to embed the raw
// JSON as an extra "tok_json" blob for external tokenization.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <cmath>  // std::isfinite
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include "compression/compress.h"  // ScaleWeights
#include "gemma/configs.h"
#include "gemma/model_store.h"  // WriteSingleFile
#include "gemma/tensor_info.h"
#include "gemma/tokenizer.h"
#include "gemma/weights.h"
#include "io/blob_store.h"
#include "io/io.h"
#include "util/basics.h"
#include "util/mat.h"
#include "util/threading_context.h"
#include "nlohmann/json.hpp"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE \
  "deepseek/convert_dsv4.cc"   // NOLINT
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"

// SIMD section: only the compression entry point needs per-target code.
HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// Compresses `data` into a MatPtrT of the given type and writes the blob.
// Appends the serialized MatPtr fields to `serialized`. May modify `data`
// in place (SFP/NUQ rescaling).
void CompressAndWrite(const char* name, float* data, size_t rows, size_t cols,
                      Type type, BlobWriter& writer,
                      std::vector<uint32_t>& serialized, ThreadingContext& ctx,
                      CompressWorkingSet& ws) {
  const Extents2D extents(rows, cols);
  const size_t num = rows * cols;

  const auto insert = [&](auto packed_tag) {
    using Packed = decltype(packed_tag);
    MatPtrT<Packed> mat(name, extents);
    if (mat.GetType() == Type::kSFP || mat.GetType() == Type::kNUQ) {
      // SFP has limited range; rescale (in place) and store the scale.
      mat.SetScale(ScaleWeights(data, num));
    }
    mat.AppendTo(serialized);
    MatOwner owner;
    owner.AllocateFor(mat, ctx.allocator, MatPadding::kPacked);
    Compress(data, num, ws, mat.Span(), /*packed_ofs=*/0, ctx);
    writer.Add(name, mat.Packed(), mat.PackedBytes());
  };

  switch (type) {
    case Type::kSFP:
      insert(SfpStream());
      break;
    case Type::kNUQ:
      insert(NuqStream());
      break;
    case Type::kBF16:
      insert(BF16());
      break;
    case Type::kF32:
      insert(float());
      break;
    default:
      HWY_ABORT("Unsupported output type %s for %s", TypeName(type), name);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {

HWY_EXPORT(CompressAndWrite);

using nlohmann::json;

// ---------------------------------------------------------------- safetensors

struct SourceTensor {
  size_t file_idx;
  std::string dtype;
  std::vector<size_t> shape;
  uint64_t begin;  // absolute file offset
  uint64_t end;
};

class ShardedCheckpoint {
 public:
  // `tolerate_missing` skips shards that are absent or still downloading
  // (LFS pointer stubs); their tensors are then not Find()-able.
  ShardedCheckpoint(const std::string& dir, bool tolerate_missing,
                    const std::string& index_override) {
    // Enumerate shards from the index.
    const std::string index_path = index_override.empty()
                                       ? dir + "/model.safetensors.index.json"
                                       : index_override;
    const std::string index_str = ReadFileToString(Path(index_path));
    HWY_ASSERT_M(!index_str.empty(), index_path.c_str());
    json index = json::parse(index_str);
    std::unordered_map<std::string, size_t> file_indices;
    size_t missing = 0;
    for (const auto& [tensor, file] : index["weight_map"].items()) {
      const std::string file_str = file.get<std::string>();
      if (file_indices.insert({file_str, files_.size()}).second) {
        std::unique_ptr<File> f =
            OpenFileOrNull(Path(dir + "/" + file_str), "r");
        // LFS pointer stubs are tiny; treat them as missing.
        if (f && f->FileSize() < (uint64_t{1} << 20)) f.reset();
        if (!f) {
          HWY_ASSERT_M(tolerate_missing, file_str.c_str());
          ++missing;
          files_.push_back(nullptr);
          continue;
        }
        files_.push_back(std::move(f));
        ParseHeader(files_.size() - 1);
      }
    }
    fprintf(stderr, "Checkpoint: %zu shards (%zu missing), %zu tensors\n",
            files_.size(), missing, tensors_.size());
  }

  const SourceTensor* Find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
  }

  // Reads the raw bytes of a tensor.
  std::vector<uint8_t> ReadRaw(const SourceTensor& t) const {
    std::vector<uint8_t> buf(t.end - t.begin);
    HWY_ASSERT(files_[t.file_idx]->Read(t.begin, buf.size(), buf.data()));
    return buf;
  }

 private:
  void ParseHeader(size_t file_idx) {
    uint64_t header_len = 0;
    HWY_ASSERT(files_[file_idx]->Read(0, 8, &header_len));
    std::vector<char> header(header_len);
    HWY_ASSERT(files_[file_idx]->Read(8, header_len, header.data()));
    json j = json::parse(header.begin(), header.end());
    for (const auto& [name, info] : j.items()) {
      if (name == "__metadata__") continue;
      SourceTensor t;
      t.file_idx = file_idx;
      t.dtype = info["dtype"].get<std::string>();
      t.shape = info["shape"].get<std::vector<size_t>>();
      t.begin = 8 + header_len + info["data_offsets"][0].get<uint64_t>();
      t.end = 8 + header_len + info["data_offsets"][1].get<uint64_t>();
      tensors_[name] = t;
    }
  }

  std::vector<std::unique_ptr<File>> files_;
  std::unordered_map<std::string, SourceTensor> tensors_;
};

// ---------------------------------------------------------------- dequant

// FP8 e4m3fn: 1 sign, 4 exp (bias 7), 3 mantissa; S.1111.111 is NaN.
static inline float DecodeE4M3(uint8_t b) {
  const int sign = (b >> 7) ? -1 : 1;
  const int exp = (b >> 3) & 0xF;
  const int man = b & 0x7;
  if (exp == 0xF && man == 0x7) return 0.0f;  // NaN in weights -> 0
  float v;
  if (exp == 0) {
    v = ldexpf(static_cast<float>(man), -9);  // subnormal: man/8 * 2^-6
  } else {
    v = ldexpf(1.0f + static_cast<float>(man) / 8.0f, exp - 7);
  }
  return sign * v;
}

// e8m0: pure power-of-two exponent, bias 127. 0xFF is NaN.
static inline float DecodeE8M0(uint8_t b) {
  if (b == 0xFF) return 0.0f;
  return ldexpf(1.0f, static_cast<int>(b) - 127);
}

// FP4 e2m1: 1 sign, 2 exp (bias 1), 1 mantissa.
static const float kFP4Table[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                   2.0f, 3.0f, 4.0f, 6.0f};
static inline float DecodeE2M1(uint8_t nibble) {
  const float v = kFP4Table[nibble & 0x7];
  return (nibble & 0x8) ? -v : v;
}

// ---------------------------------------------------------------- transforms

// Permutes the last `tail` elements of every `seg`-wide segment along a
// dimension from interleaved rope pairs (2i, 2i+1) to gemma.cpp's split
// halves (i, i + tail/2). `perm[i]` gives the source index within the tail.
static std::vector<size_t> RopePerm(size_t tail) {
  std::vector<size_t> perm(tail);
  const size_t half = tail / 2;
  for (size_t i = 0; i < half; ++i) {
    perm[i] = 2 * i;
    perm[half + i] = 2 * i + 1;
  }
  return perm;
}

// Permutes rows: for each segment of `seg` rows, the last `tail` rows are
// reordered by RopePerm.
static void PermuteTailRows(std::vector<float>& data, size_t rows, size_t cols,
                            size_t seg, size_t tail) {
  HWY_ASSERT(rows % seg == 0);
  const std::vector<size_t> perm = RopePerm(tail);
  std::vector<float> tmp(tail * cols);
  for (size_t s = 0; s < rows; s += seg) {
    float* base = data.data() + (s + seg - tail) * cols;
    memcpy(tmp.data(), base, tail * cols * sizeof(float));
    for (size_t i = 0; i < tail; ++i) {
      memcpy(base + i * cols, tmp.data() + perm[i] * cols,
             cols * sizeof(float));
    }
  }
}

// Permutes, within each row, the last `tail` elements of every `seg`-wide
// column segment.
static void PermuteTailCols(std::vector<float>& data, size_t rows, size_t cols,
                            size_t seg, size_t tail) {
  HWY_ASSERT(cols % seg == 0);
  const std::vector<size_t> perm = RopePerm(tail);
  std::vector<float> tmp(tail);
  for (size_t r = 0; r < rows; ++r) {
    float* row = data.data() + r * cols;
    for (size_t s = 0; s < cols; s += seg) {
      float* base = row + s + seg - tail;
      memcpy(tmp.data(), base, tail * sizeof(float));
      for (size_t i = 0; i < tail; ++i) base[i] = tmp[perm[i]];
    }
  }
}

// ---------------------------------------------------------------- converter

struct ConvertArgs {
  std::string weights_dir;
  std::string output;
  std::string tokenizer_json;
  std::string index;  // override for model.safetensors.index.json
  bool fp4_high_first = false;
  // Dry run: dequantize + transform + shape-check whatever shards are
  // present, write nothing. Usable while the download is still running.
  bool verify_only = false;
  // With verify_only: check only the MTP tensors (fast pre-flight).
  bool mtp_only = false;
};

class Converter {
 public:
  Converter(const ConvertArgs& args)
      : args_(args),
        checkpoint_(args.weights_dir, /*tolerate_missing=*/args.verify_only,
                    args.index),
        ctx_(ThreadingArgs()) {
    if (!args_.verify_only) {
      writer_ = std::make_unique<BlobWriter>(Path(args_.output), ctx_);
    }
  }

  // The e8m0 block-scale companion of an FP8/FP4 weight tensor.
  struct E8M0Scale {
    const SourceTensor* tensor;
    std::vector<uint8_t> raw;
  };

  // Loads the scale tensor named like `src_name` with ".weight" replaced by
  // ".scale". Shape checks are per-caller (FP8: per-128x128-block scales,
  // FP4: per-32 along the last dim).
  E8M0Scale LoadE8M0Scale(const std::string& src_name) {
    std::string scale_name = src_name;
    const size_t pos = scale_name.rfind(".weight");
    HWY_ASSERT(pos != std::string::npos);
    scale_name.replace(pos, 7, ".scale");
    const SourceTensor* st = checkpoint_.Find(scale_name);
    HWY_ASSERT_M(st != nullptr, scale_name.c_str());
    HWY_ASSERT(st->dtype == "F8_E8M0");
    return {st, checkpoint_.ReadRaw(*st)};
  }

  // Loads a source tensor and dequantizes to f32. `expect` is the number of
  // f32 elements the caller wants (checked).
  std::vector<float> LoadF32(const std::string& src_name, size_t expect) {
    const SourceTensor* t = checkpoint_.Find(src_name);
    HWY_ASSERT_M(t != nullptr, src_name.c_str());
    const std::vector<uint8_t> raw = checkpoint_.ReadRaw(*t);
    size_t n = 1;
    for (size_t d : t->shape) n *= d;

    std::vector<float> out;
    if (t->dtype == "BF16") {
      HWY_ASSERT_M(n == expect, src_name.c_str());
      out.resize(n);
      const hwy::bfloat16_t* p =
          reinterpret_cast<const hwy::bfloat16_t*>(raw.data());
      for (size_t i = 0; i < n; ++i) out[i] = hwy::F32FromBF16(p[i]);
    } else if (t->dtype == "F32") {
      HWY_ASSERT_M(n == expect, src_name.c_str());
      out.resize(n);
      memcpy(out.data(), raw.data(), n * sizeof(float));
    } else if (t->dtype == "I64") {
      HWY_ASSERT_M(n == expect, src_name.c_str());
      out.resize(n);
      const int64_t* p = reinterpret_cast<const int64_t*>(raw.data());
      for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
    } else if (t->dtype == "F8_E4M3") {
      // 128x128 block scales in <name minus .weight>.scale, e8m0.
      HWY_ASSERT_M(n == expect, src_name.c_str());
      HWY_ASSERT(t->shape.size() == 2);
      const size_t rows = t->shape[0], cols = t->shape[1];
      const E8M0Scale scale = LoadE8M0Scale(src_name);
      const size_t scale_cols = (cols + 127) / 128;
      HWY_ASSERT(scale.tensor->shape[0] == (rows + 127) / 128 &&
                 scale.tensor->shape[1] == scale_cols);
      out.resize(n);
      for (size_t r = 0; r < rows; ++r) {
        const uint8_t* src = raw.data() + r * cols;
        const uint8_t* srow = scale.raw.data() + (r / 128) * scale_cols;
        float* dst = out.data() + r * cols;
        for (size_t c = 0; c < cols; ++c) {
          dst[c] = DecodeE4M3(src[c]) * DecodeE8M0(srow[c / 128]);
        }
      }
    } else if (t->dtype == "I8") {
      // FP4 e2m1 packed two per byte along the last dim; per-32 e8m0 scales.
      HWY_ASSERT(t->shape.size() == 2);
      const size_t rows = t->shape[0];
      const size_t cols = t->shape[1] * 2;  // logical
      HWY_ASSERT_M(rows * cols == expect, src_name.c_str());
      const E8M0Scale scale = LoadE8M0Scale(src_name);
      const size_t scale_cols = cols / 32;
      HWY_ASSERT(scale.tensor->shape[0] == rows &&
                 scale.tensor->shape[1] == scale_cols);
      out.resize(rows * cols);
      for (size_t r = 0; r < rows; ++r) {
        const uint8_t* src = raw.data() + r * (cols / 2);
        const uint8_t* srow = scale.raw.data() + r * scale_cols;
        float* dst = out.data() + r * cols;
        for (size_t c2 = 0; c2 < cols / 2; ++c2) {
          const uint8_t byte = src[c2];
          const uint8_t lo = byte & 0xF;
          const uint8_t hi = byte >> 4;
          const size_t c = 2 * c2;
          const float s0 = DecodeE8M0(srow[c / 32]);
          const float s1 = DecodeE8M0(srow[(c + 1) / 32]);
          if (args_.fp4_high_first) {
            dst[c] = DecodeE2M1(hi) * s0;
            dst[c + 1] = DecodeE2M1(lo) * s1;
          } else {
            dst[c] = DecodeE2M1(lo) * s0;
            dst[c + 1] = DecodeE2M1(hi) * s1;
          }
        }
      }
    } else {
      HWY_ABORT("Unhandled dtype %s for %s", t->dtype.c_str(),
                src_name.c_str());
    }
    return out;
  }

  // Output type by source dtype: FP8/FP4 -> SFP, BF16 -> BF16, F32/I64 -> F32.
  Type OutType(const std::string& src_name) {
    const SourceTensor* t = checkpoint_.Find(src_name);
    HWY_ASSERT_M(t != nullptr, src_name.c_str());
    if (t->dtype == "F8_E4M3" || t->dtype == "I8") return Type::kSFP;
    if (t->dtype == "BF16") return Type::kBF16;
    return Type::kF32;
  }

  void Write(const char* name, std::vector<float>& data, size_t rows,
             size_t cols, Type type) {
    if (args_.verify_only) {
      // Basic sanity: all values finite.
      for (size_t i = 0; i < data.size(); ++i) {
        if (!std::isfinite(data[i])) {
          fprintf(stderr, "NON-FINITE value in %s at %zu\n", name, i);
          ++num_bad_;
          break;
        }
      }
      ++num_written_;
      if (num_written_ % 2000 == 0) {
        fprintf(stderr, "  ... %zu tensors verified\n", num_written_);
      }
      return;
    }
    HWY_DYNAMIC_DISPATCH(CompressAndWrite)(name, data.data(), rows, cols, type,
                                           *writer_, serialized_mat_ptrs_, ctx_,
                                           working_set_);
    ++num_written_;
    if (num_written_ % 500 == 0) {
      fprintf(stderr, "  ... %zu tensors written\n", num_written_);
    }
  }

  // Parses the trailing _<layer>[_<expert>] indices off a tensor name.
  // Returns the base name.
  static std::string ParseSuffix(const std::string& name, int& layer,
                                 int& expert) {
    layer = -1;
    expert = -1;
    std::string base = name;
    // Up to two numeric suffixes.
    for (int pass = 0; pass < 2; ++pass) {
      const size_t us = base.rfind('_');
      if (us == std::string::npos || us + 1 >= base.size()) break;
      bool numeric = true;
      for (size_t i = us + 1; i < base.size(); ++i) {
        if (base[i] < '0' || base[i] > '9') {
          numeric = false;
          break;
        }
      }
      if (!numeric) break;
      const int value = atoi(base.c_str() + us + 1);
      if (pass == 0) {
        layer = value;
      } else {
        expert = layer;  // first parsed suffix was the expert
        layer = value;   // second is the layer
      }
      base = base.substr(0, us);
    }
    return base;
  }

  void Run() {
    const ModelConfig config(Model::DEEPSEEK4_FLASH, Type::kSFP,
                             PromptWrapping::GEMMA_IT);
    WeightsPtrs weights(config);
    const bool has_mtp = config.num_mtp_layers > 0;
    const LayerConfig mtp_lc =
        has_mtp ? config.MTPLayerConfig() : LayerConfig();

    weights.ForEachTensor(nullptr, nullptr, [&](const TensorArgs& targs) {
      MatPtr& mat = targs.mat;
      const std::string name = mat.Name();
      if (mat.IsEmpty()) return;
      int layer = -1, expert = -1;
      const std::string base = ParseSuffix(name, layer, expert);
      // Skipped aliases (w1/w2 are used instead) and optional tensors.
      if (base == "gating_ein" || base == "skip_scale" || base == "qkv_ein" ||
          base == "qkv1_w" || base == "qkv2_w" || base == "att_w" ||
          base == "att_ein") {
        return;
      }
      const size_t rows = mat.Rows(), cols = mat.Cols();
      const size_t num = mat.Extents().Area();
      // The MTP block is registered as layer index `num_layers`; its source
      // tensors live under "mtp.0." instead of "layers.N.".
      const bool is_mtp = layer >= static_cast<int>(config.num_layers);
      if (args_.mtp_only && !is_mtp && base.rfind("mtp_", 0) != 0) return;
      const LayerConfig* lc =
          layer < 0 ? nullptr
                    : (is_mtp ? &mtp_lc : &config.layer_configs[layer]);
      const std::string P =
          layer < 0
              ? ""
              : (is_mtp ? "mtp.0." : "layers." + std::to_string(layer) + ".");

      std::string src;
      // Transform: 0 = none, 1 = tail rows per segment, 2 = tail cols per
      // segment, 3 = flat tail elems per segment.
      int transform = 0;
      size_t seg = 0, tail = 0;

      if (base == "c_embedding") {
        src = "embed.weight";
      } else if (base == "c_final_norm") {
        src = "norm.weight";
      } else if (base == "lm_head") {
        src = "head.weight";
      } else if (base == "hc_head_fn" || base == "hc_head_base" ||
                 base == "hc_head_scale") {
        src = base;
      } else if (base == "pre_att_ns") {
        src = P + "attn_norm.weight";
      } else if (base == "pre_ff_ns") {
        src = P + "ffn_norm.weight";
      } else if (base == "mla_q_a") {
        src = P + "attn.wq_a.weight";
      } else if (base == "mla_q_a_ns") {
        src = P + "attn.q_norm.weight";
      } else if (base == "mla_q_b") {
        src = P + "attn.wq_b.weight";
        transform = 1;
        seg = lc->qkv_dim;
        tail = lc->rope_head_dim;
      } else if (base == "mla_kv_a") {
        src = P + "attn.wkv.weight";
        transform = 1;
        seg = lc->KVLatentDim();
        tail = lc->rope_head_dim;
      } else if (base == "mla_kv_a_ns") {
        src = P + "attn.kv_norm.weight";
        transform = 3;
        seg = lc->KVLatentDim();
        tail = lc->rope_head_dim;
      } else if (base == "mla_o_a") {
        src = P + "attn.wo_a.weight";
        transform = 2;
        seg = lc->qkv_dim;
        tail = lc->rope_head_dim;
      } else if (base == "mla_o_b") {
        src = P + "attn.wo_b.weight";
      } else if (base == "attn_sink") {
        src = P + "attn.attn_sink";
      } else if (base == "comp_wkv") {
        src = P + "attn.compressor.wkv.weight";
        transform = 1;
        seg = lc->KVLatentDim();
        tail = lc->rope_head_dim;
      } else if (base == "comp_wgate") {
        src = P + "attn.compressor.wgate.weight";
        transform = 1;
        seg = lc->KVLatentDim();
        tail = lc->rope_head_dim;
      } else if (base == "comp_ape") {
        src = P + "attn.compressor.ape";
        transform = 2;
        seg = lc->KVLatentDim();
        tail = lc->rope_head_dim;
      } else if (base == "comp_ns") {
        src = P + "attn.compressor.norm.weight";
        transform = 3;
        seg = lc->KVLatentDim();
        tail = lc->rope_head_dim;
      } else if (base == "idx_q_w") {
        src = P + "attn.indexer.wq_b.weight";
        transform = 1;
        seg = lc->indexer_head_dim;
        tail = lc->rope_head_dim;
      } else if (base == "idx_w_proj") {
        src = P + "attn.indexer.weights_proj.weight";
      } else if (base == "idxc_wkv") {
        src = P + "attn.indexer.compressor.wkv.weight";
        transform = 1;
        seg = lc->indexer_head_dim;
        tail = lc->rope_head_dim;
      } else if (base == "idxc_wgate") {
        src = P + "attn.indexer.compressor.wgate.weight";
        transform = 1;
        seg = lc->indexer_head_dim;
        tail = lc->rope_head_dim;
      } else if (base == "idxc_ape") {
        src = P + "attn.indexer.compressor.ape";
        transform = 2;
        seg = lc->indexer_head_dim;
        tail = lc->rope_head_dim;
      } else if (base == "idxc_ns") {
        src = P + "attn.indexer.compressor.norm.weight";
        transform = 3;
        seg = lc->indexer_head_dim;
        tail = lc->rope_head_dim;
      } else if (base == "moe_router") {
        src = P + "ffn.gate.weight";
      } else if (base == "moe_r_bias") {
        src = P + "ffn.gate.bias";
      } else if (base == "hash_tid2eid") {
        src = P + "ffn.gate.tid2eid";
      } else if (base == "gating1_w") {
        src = expert >= 0
                  ? P + "ffn.experts." + std::to_string(expert) + ".w1.weight"
                  : P + "ffn.shared_experts.w1.weight";
      } else if (base == "gating2_w") {
        src = expert >= 0
                  ? P + "ffn.experts." + std::to_string(expert) + ".w3.weight"
                  : P + "ffn.shared_experts.w3.weight";
      } else if (base == "linear_w") {
        src = expert >= 0
                  ? P + "ffn.experts." + std::to_string(expert) + ".w2.weight"
                  : P + "ffn.shared_experts.w2.weight";
      } else if (base == "hc_att_fn") {
        src = P + "hc_attn_fn";
      } else if (base == "hc_att_base") {
        src = P + "hc_attn_base";
      } else if (base == "hc_att_scale") {
        src = P + "hc_attn_scale";
      } else if (base == "hc_ffw_fn") {
        src = P + "hc_ffn_fn";
      } else if (base == "hc_ffw_base") {
        src = P + "hc_ffn_base";
      } else if (base == "hc_ffw_scale") {
        src = P + "hc_ffn_scale";
      } else if (base == "mtp_e_proj") {
        src = "mtp.0.e_proj.weight";
      } else if (base == "mtp_h_proj") {
        src = "mtp.0.h_proj.weight";
      } else if (base == "mtp_enorm") {
        src = "mtp.0.enorm.weight";
      } else if (base == "mtp_hnorm") {
        src = "mtp.0.hnorm.weight";
      } else if (base == "mtp_norm") {
        src = "mtp.0.norm.weight";
      } else if (base == "mtp_hc_fn") {
        src = "mtp.0.hc_head_fn";
      } else if (base == "mtp_hc_base") {
        src = "mtp.0.hc_head_base";
      } else if (base == "mtp_hc_scale") {
        src = "mtp.0.hc_head_scale";
      } else {
        HWY_ABORT("No mapping for tensor %s (base %s)", name.c_str(),
                  base.c_str());
      }

      if (args_.verify_only && checkpoint_.Find(src) == nullptr) {
        ++num_skipped_;  // shard not downloaded yet
        return;
      }
      std::vector<float> data = LoadF32(src, num);
      switch (transform) {
        case 1:
          PermuteTailRows(data, rows, cols, seg, tail);
          break;
        case 2:
          PermuteTailCols(data, rows, cols, seg, tail);
          break;
        case 3:
          PermuteTailCols(data, 1, num, seg, tail);
          break;
        default:
          break;
      }
      Write(name.c_str(), data, rows, cols, OutType(src));
    });

    if (args_.verify_only) {
      fprintf(stderr,
              "Verify done: %zu tensors OK, %zu skipped (missing shards), "
              "%zu with non-finite values\n",
              num_written_, num_skipped_, num_bad_);
      HWY_ASSERT(num_bad_ == 0);
      return;
    }

    // Optional raw tokenizer JSON for external tokenization.
    if (!args_.tokenizer_json.empty()) {
      const std::string tok = ReadFileToString(Path(args_.tokenizer_json));
      HWY_ASSERT(!tok.empty());
      writer_->Add("tok_json", tok.data(), tok.size());
    }

    // Config + mock tokenizer + serialized MatPtrs; finalizes the writer.
    const GemmaTokenizer tokenizer{std::string(kMockTokenizer)};
    WriteSingleFile(config, tokenizer, serialized_mat_ptrs_, *writer_);
    fprintf(stderr, "Done: %zu tensors -> %s\n", num_written_,
            args_.output.c_str());
  }

 private:
  ConvertArgs args_;
  ShardedCheckpoint checkpoint_;
  ThreadingContext ctx_;
  CompressWorkingSet working_set_;
  std::unique_ptr<BlobWriter> writer_;
  std::vector<uint32_t> serialized_mat_ptrs_;
  size_t num_written_ = 0;
  size_t num_skipped_ = 0;
  size_t num_bad_ = 0;
};

int Main(int argc, char** argv) {
  ConvertArgs args;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--weights" && i + 1 < argc) {
      args.weights_dir = argv[++i];
    } else if (a == "--output" && i + 1 < argc) {
      args.output = argv[++i];
    } else if (a == "--tokenizer_json" && i + 1 < argc) {
      args.tokenizer_json = argv[++i];
    } else if (a == "--index" && i + 1 < argc) {
      args.index = argv[++i];
    } else if (a == "--fp4_high_first") {
      args.fp4_high_first = true;
    } else if (a == "--verify_only") {
      args.verify_only = true;
    } else if (a == "--mtp_only") {
      args.mtp_only = true;
    } else {
      fprintf(stderr, "Unknown arg %s\n", a.c_str());
      return 1;
    }
  }
  if (args.weights_dir.empty() || (args.output.empty() && !args.verify_only)) {
    fprintf(stderr,
            "Usage: convert_dsv4 --weights <dir> --output <file.sbs> "
            "[--tokenizer_json <tokenizer.json>] [--fp4_high_first] "
            "[--verify_only]\n");
    return 1;
  }
  Converter(args).Run();
  return 0;
}

}  // namespace gcpp

int main(int argc, char** argv) { return gcpp::Main(argc, argv); }
#endif  // HWY_ONCE
