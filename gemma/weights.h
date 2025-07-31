// Copyright 2024 Google LLC
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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_WEIGHTS_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_WEIGHTS_H_

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "compression/types.h"
#include "gemma/configs.h"      // ModelConfig
#include "gemma/gemma_args.h"   // InferenceArgs
#include "gemma/model_store.h"  // ModelStore
#include "gemma/tensor_info.h"  // TensorInfoRegistry
#include "io/blob_store.h"      // BlobWriter
#include "util/mat.h"           // MatPtr
#include "util/threading_context.h"

namespace gcpp {

// Argument passed to the `ForEachTensor` callback.
struct TensorArgs {
  // `other_mat1` and `other_mat2` can be nullptr, or tensor(s) of the same
  // name/type from another `LayerWeightsPtrs` for iterating over tensor pairs
  // (for copying) or triples (for `AdamUpdateMV`). Set by `TENSOR_ARGS`.
  // `flags` is a combination of zero or more `Flags`.
  TensorArgs(MatPtr& mat, MatPtr* other_mat1, MatPtr* other_mat2, int flags)
      : mat(mat),
        other_mat1(other_mat1),
        other_mat2(other_mat2),
        flags(flags) {}

  MatPtr& mat;
  MatPtr* other_mat1;  // either/both can be nullptr.
  MatPtr* other_mat2;

  enum Flags {
    // Default: Read the tensor from the file and abort if it is not found.
    kMustRead = 0,

    // Not an error if the tensor is not present in the file. For example,
    // the _w1/_w2 tensors are not always present.
    kMaybeRead = 1,

    // Avoid padding tensor rows when reading. Used for some Griffin tensors
    // whose index computations do not use Row() accessors.
    kPacked = 2,
  };
  const int flags;
};

// Shorthand for creating the argument to the `ForEachTensor` callback. A macro
// seems less bad than member pointer syntax.
#define TENSOR_ARGS(mat, flag)                     \
  TensorArgs(mat, other1 ? &other1->mat : nullptr, \
             other2 ? &other2->mat : nullptr, TensorArgs::flag)

// Finds tensors by name in `TensorInfoRegistry` (constructed from
// `ModelConfig`) and constructs `MatPtr` metadata with those shapes.
class MatFinder {
 public:
  MatFinder(const std::string& suffix, const TensorInfoRegistry& tensors)
      : suffix_(suffix), tensors_(tensors) {}

  // Retrieves shape by name via `TensorInfo` from `TensorInfoRegistry`.
  MatPtr operator()(const std::string& base_name) const {
    const std::string name = std::string(base_name) + suffix_;
    return MatPtr(name.c_str(), Type::kUnknown,
                  ExtentsFromInfo(tensors_.Find(name)));
  }

 private:
  const std::string suffix_;
  const TensorInfoRegistry& tensors_;
};

// Per-layer weight metadata and pointers. The tensor data is owned by
// `MatOwner`.
struct LayerWeightsPtrs {
  // Initializes tensor metadata without allocating.
  // NOTE: do not store layer_idx, TransformerLayer and Attention may use
  // other values for purposes of the KV cache.
  LayerWeightsPtrs(size_t layer_idx, const LayerConfig& config,
                   const TensorInfoRegistry& tensors)
      : finder_(LayerSuffix(layer_idx), tensors),
        qkv_einsum_w(finder_("qkv_ein")),
        qkv_einsum_w1(finder_("qkv1_w")),
        qkv_einsum_w2(finder_("qkv2_w")),
        attention_output_biases(finder_("attn_ob")),
        griffin({.linear_x_w = finder_("gr_lin_x_w"),
                 .linear_x_biases = finder_("gr_lin_x_b"),
                 .linear_y_w = finder_("gr_lin_y_w"),
                 .linear_y_biases = finder_("gr_lin_y_b"),
                 .linear_out_w = finder_("gr_lin_out_w"),
                 .linear_out_biases = finder_("gr_lin_out_b"),
                 .conv_w = finder_("gr_conv_w"),
                 .conv_biases = finder_("gr_conv_b"),
                 .gate_w = finder_("gr_gate_w"),
                 .gate_biases = finder_("gr_gate_b"),
                 .a = finder_("gr_a")}),
        // MultiHeadDotProductAttention.
        vit({.attn_out_w = finder_("attn_out_w"),
             .attn_out_b = finder_("attn_out_b"),
             .qkv_einsum_w = finder_("qkv_ein_w"),
             .qkv_einsum_b = finder_("qkv_ein_b"),
             .linear_0_w = finder_("linear_0_w"),
             .linear_0_b = finder_("linear_0_b"),
             .linear_1_w = finder_("linear_1_w"),
             .linear_1_b = finder_("linear_1_b"),
             .layer_norm_0_bias = finder_("ln_0_bias"),
             .layer_norm_0_scale = finder_("ln_0_scale"),
             .layer_norm_1_bias = finder_("ln_1_bias"),
             .layer_norm_1_scale = finder_("ln_1_scale")}),
        gating_einsum_w(finder_("gating_ein")),
        gating_einsum_w1(finder_("gating1_w")),
        gating_einsum_w2(finder_("gating2_w")),
        linear_w(finder_("linear_w")),
        pre_attention_norm_scale(finder_("pre_att_ns")),
        pre_ffw_norm_scale(finder_("pre_ff_ns")),
        post_attention_norm_scale(finder_("post_att_ns")),
        post_ffw_norm_scale(finder_("post_ff_ns")),
        ffw_gating_biases(finder_("ffw_gat_b")),
        ffw_output_biases(finder_("ffw_out_b")),

        attn_vec_einsum_w(finder_("att_ein")),
        att_weights(finder_("att_w")),

        key_norm_scale(finder_("key_norm")),
        query_norm_scale(finder_("query_norm")),

        layer_config(config) {
  }
  ~LayerWeightsPtrs() = default;

  const MatFinder finder_;

  // Files either have qkv_einsum_w with 2 stacked matrices or separate
  // w1/w2 tensors. Fixup ensures w1/w2 are ready for use by gemma-inl.h.
  MatPtr qkv_einsum_w;
  MatPtr qkv_einsum_w1;
  MatPtr qkv_einsum_w2;
  MatPtrT<float> attention_output_biases;

  struct {
    MatPtr linear_x_w;
    MatPtrT<float> linear_x_biases;
    MatPtr linear_y_w;
    MatPtrT<float> linear_y_biases;
    MatPtr linear_out_w;
    MatPtrT<float> linear_out_biases;
    MatPtrT<float> conv_w;
    MatPtrT<float> conv_biases;
    MatPtr gate_w;
    MatPtrT<float> gate_biases;
    MatPtrT<float> a;
  } griffin;

  struct {
    // MultiHeadDotProductAttention.
    MatPtr attn_out_w;  // at least BF16.
    MatPtrT<float> attn_out_b;
    MatPtr qkv_einsum_w;  // at least BF16.
    MatPtrT<float> qkv_einsum_b;
    // MlpBlock.
    MatPtr linear_0_w;  // at least BF16.
    MatPtrT<float> linear_0_b;
    MatPtr linear_1_w;  // at least BF16.
    MatPtrT<float> linear_1_b;
    // LayerNorm.
    MatPtr layer_norm_0_bias;   // at least BF16.
    MatPtr layer_norm_0_scale;  // at least BF16.
    MatPtr layer_norm_1_bias;   // at least BF16.
    MatPtr layer_norm_1_scale;  // at least BF16.
  } vit;

  // Files either have gating_einsum_w with 2 stacked matrices or separate
  // w1/w2 tensors. `Fixup` ensures w1/w2 are ready for use by gemma-inl.h.
  MatPtr gating_einsum_w;
  MatPtr gating_einsum_w1;
  MatPtr gating_einsum_w2;
  MatPtr linear_w;
  MatPtr pre_attention_norm_scale;   // at least BF16.
  MatPtr pre_ffw_norm_scale;         // at least BF16.
  MatPtr post_attention_norm_scale;  // at least BF16.
  MatPtr post_ffw_norm_scale;        // at least BF16.

  MatPtrT<float> ffw_gating_biases;
  MatPtrT<float> ffw_output_biases;

  MatPtr attn_vec_einsum_w;  // Use att_weights instead of this.
  MatPtr att_weights;        // Use this instead of attn_vec_einsum_w.

  MatPtr key_norm_scale;    // at least BF16.
  MatPtr query_norm_scale;  // at least BF16.

  const LayerConfig& layer_config;

  // Calls `func(TensorArgs)` for each tensor which is in use for the
  // current `layer_config`. `other1` and `other2` are optional arguments so we
  // can also iterate over pairs or triples of tensors for `AdamUpdateMV`.
  // Public because also called by `WeightsPtrs`.
  template <class Func>
  void ForEachTensor(LayerWeightsPtrs* other1, LayerWeightsPtrs* other2,
                     Func func) {
    if (layer_config.type == LayerAttentionType::kVit) {
      // MHA.
      func(TENSOR_ARGS(vit.attn_out_w, kMustRead));
      func(TENSOR_ARGS(vit.attn_out_b, kMustRead));
      func(TENSOR_ARGS(vit.qkv_einsum_w, kMustRead));
      // Used as 1D MatMul bias, but has `heads + 2 * kv_heads` rows, hence
      // must not be padded.
      func(TENSOR_ARGS(vit.qkv_einsum_b, kMustRead | TensorArgs::kPacked));
      // MlpBlock.
      func(TENSOR_ARGS(vit.linear_0_w, kMustRead));
      func(TENSOR_ARGS(vit.linear_0_b, kMustRead));
      func(TENSOR_ARGS(vit.linear_1_w, kMustRead));
      func(TENSOR_ARGS(vit.linear_1_b, kMustRead));
      // LayerNorm.
      func(TENSOR_ARGS(vit.layer_norm_0_bias, kMustRead));
      func(TENSOR_ARGS(vit.layer_norm_0_scale, kMustRead));
      func(TENSOR_ARGS(vit.layer_norm_1_bias, kMustRead));
      func(TENSOR_ARGS(vit.layer_norm_1_scale, kMustRead));
      return;
    }
    if (layer_config.type == LayerAttentionType::kGemma) {
      // Either read from file, or allocated during Fixup().
      func(TENSOR_ARGS(att_weights, kMaybeRead));
      func(TENSOR_ARGS(attn_vec_einsum_w, kMaybeRead));
      func(TENSOR_ARGS(qkv_einsum_w, kMaybeRead));
      func(TENSOR_ARGS(qkv_einsum_w1, kMaybeRead));
      func(TENSOR_ARGS(qkv_einsum_w2, kMaybeRead));
    } else {
      func(TENSOR_ARGS(griffin.linear_x_w, kMustRead));
      func(TENSOR_ARGS(griffin.linear_x_biases, kMustRead));
      func(TENSOR_ARGS(griffin.linear_y_w, kMustRead));
      func(TENSOR_ARGS(griffin.linear_y_biases, kMustRead));
      func(TENSOR_ARGS(griffin.linear_out_w, kMustRead));
      func(TENSOR_ARGS(griffin.linear_out_biases, kMustRead));
      // conv_w and gate_w are not accessed via Row(), hence must not be padded.
      // Note that *biases are 1D, hence packing/padding does not matter.
      func(TENSOR_ARGS(griffin.conv_w, kMustRead | TensorArgs::kPacked));
      func(TENSOR_ARGS(griffin.conv_biases, kMustRead));
      func(TENSOR_ARGS(griffin.gate_w, kMustRead | TensorArgs::kPacked));
      func(TENSOR_ARGS(griffin.gate_biases, kMustRead));
      func(TENSOR_ARGS(griffin.a, kMustRead));
    }
    {
      func(TENSOR_ARGS(gating_einsum_w, kMaybeRead));
      func(TENSOR_ARGS(gating_einsum_w1, kMaybeRead));
      func(TENSOR_ARGS(gating_einsum_w2, kMaybeRead));
      func(TENSOR_ARGS(linear_w, kMaybeRead));
      func(TENSOR_ARGS(pre_attention_norm_scale, kMustRead));
      func(TENSOR_ARGS(pre_ffw_norm_scale, kMustRead));
    }

    if (layer_config.post_norm == PostNormType::Scale) {
      func(TENSOR_ARGS(post_attention_norm_scale, kMustRead));
      func(TENSOR_ARGS(post_ffw_norm_scale, kMustRead));
    }
    if (layer_config.use_qk_norm) {
      func(TENSOR_ARGS(key_norm_scale, kMustRead));
      func(TENSOR_ARGS(query_norm_scale, kMustRead));
    }

    if (layer_config.ff_biases) {
      func(TENSOR_ARGS(ffw_gating_biases, kMustRead));
      func(TENSOR_ARGS(ffw_output_biases, kMustRead));
    }

    if (layer_config.softmax_attn_output_biases &&
        layer_config.type == LayerAttentionType::kGemma) {
      func(TENSOR_ARGS(attention_output_biases, kMustRead));
    }
  }  // `ForEachTensor`

  // Zero-initializes all allocated tensors in the layer.
  void ZeroInit() {
    ForEachTensor(nullptr, nullptr, [](const TensorArgs& t) {
      if (!t.mat.HasPtr()) return;
      gcpp::ZeroInit(t.mat);
    });
  }

  // Must be called after reading weights via `ForEachTensor`.
  // TODO: exporters should bake this into the weights already.
  // WARNING: called from multiple threads; `mat_owners` requires a lock.
  void Fixup(std::vector<MatOwner>& mat_owners, const Allocator& allocator);

 private:
  // Copies att_weights from `attn_vec_einsum_w`.
  void InitAttWeights(std::vector<MatOwner>& mat_owners,
                      const Allocator& allocator);

  // For FFN. Fast, only updates pointers.
  void SplitW1();

  // For attention, which might not have a w2. Fast, only updates pointers.
  void SplitAttW1();
};

// Holds layer-independent weight metadata and pointers plus per-layer
// `LayerWeightsPtrs`. The tensor data is owned by `MatOwner`.
struct WeightsPtrs {
  explicit WeightsPtrs(const ModelConfig& config)
      : config_(config),
        tensors_(config_),
        finder_("", tensors_),  // no suffix because these are per-model.
        embedder_input_embedding(finder_("c_embedding")),
        final_norm_scale(finder_("c_final_norm")),
        vit_encoder_norm_bias(finder_("enc_norm_bias")),
        vit_encoder_norm_scale(finder_("enc_norm_scale")),
        vit_img_embedding_bias(finder_("img_emb_bias")),
        vit_img_embedding_kernel(finder_("img_emb_kernel")),
        vit_img_pos_embedding(finder_("img_pos_emb")),
        vit_img_head_bias(finder_("img_head_bias")),
        vit_img_head_kernel(finder_("img_head_kernel")),
        mm_embed_norm(finder_("mm_embed_norm")),
        c_layers() {
    c_layers.reserve(config_.layer_configs.size());
    for (size_t idx = 0; idx < config_.layer_configs.size(); ++idx) {
      const LayerConfig& layer_config = config_.layer_configs[idx];
      c_layers.emplace_back(idx, layer_config, tensors_);
    }
    for (size_t idx = 0; idx < config_.vit_config.layer_configs.size(); ++idx) {
      const LayerConfig& layer_config = config_.vit_config.layer_configs[idx];
      vit_layers.emplace_back(idx, layer_config, tensors_);
    }
  }

  ~WeightsPtrs() = default;

  const ModelConfig& config_;
  // Passed to finder_, hence must be initialized first.
  const TensorInfoRegistry tensors_;
  const MatFinder finder_;

  // TODO: switch to SFP?
  MatPtr embedder_input_embedding;
  MatPtr final_norm_scale;  // at least BF16.

  // Vit parts.
  MatPtr vit_encoder_norm_bias;   // at least BF16.
  MatPtr vit_encoder_norm_scale;  // at least BF16.
  MatPtrT<float> vit_img_embedding_bias;
  MatPtr vit_img_embedding_kernel;  // at least BF16.
  MatPtr vit_img_pos_embedding;     // F32?
  // The head maps from VitConfig::model_dim (Vit final layer) to
  // model_dim (LLM input).
  MatPtrT<float> vit_img_head_bias;
  MatPtr vit_img_head_kernel;  // at least BF16.

  MatPtr mm_embed_norm;  // at least BF16.

  std::vector<LayerWeightsPtrs> c_layers;
  std::vector<LayerWeightsPtrs> vit_layers;

  const LayerWeightsPtrs* GetLayer(size_t layer) const {
    return &c_layers[layer];
  }
  LayerWeightsPtrs* GetLayer(size_t layer) { return &c_layers[layer]; }
  const LayerWeightsPtrs* VitLayer(size_t layer) const {
    return &vit_layers[layer];
  }
  LayerWeightsPtrs* VitLayer(size_t layer) { return &vit_layers[layer]; }

  // Called via `CallT`. `other1` and `other2` are usually null, but can be
  // used to copy from another set of weights. Public because called by tests
  // and `WeightsOwner`.
  template <class Func>
  void ForEachTensor(WeightsPtrs* other1, WeightsPtrs* other2, Func func) {
    LayerWeightsPtrs* other_layer1 = nullptr;
    LayerWeightsPtrs* other_layer2 = nullptr;
    func(TENSOR_ARGS(embedder_input_embedding, kMustRead));
    func(TENSOR_ARGS(final_norm_scale, kMustRead));

    if (!config_.vit_config.layer_configs.empty()) {  // Vit parts.
      func(TENSOR_ARGS(vit_encoder_norm_bias, kMustRead));
      func(TENSOR_ARGS(vit_encoder_norm_scale, kMustRead));
      func(TENSOR_ARGS(vit_img_embedding_bias, kMustRead));
      func(TENSOR_ARGS(vit_img_embedding_kernel, kMustRead));
      func(TENSOR_ARGS(vit_img_pos_embedding, kMustRead));
      func(TENSOR_ARGS(vit_img_head_bias, kMustRead));
      func(TENSOR_ARGS(vit_img_head_kernel, kMustRead));

      if (config_.wrapping == PromptWrapping::GEMMA_VLM) {
        func(TENSOR_ARGS(mm_embed_norm, kMustRead));
      }
    }

    for (size_t layer_idx = 0; layer_idx < c_layers.size(); ++layer_idx) {
      if (other1) other_layer1 = other1->GetLayer(layer_idx);
      if (other2) other_layer2 = other2->GetLayer(layer_idx);
      GetLayer(layer_idx)->ForEachTensor(other_layer1, other_layer2, func);
    }

    HWY_ASSERT(config_.vit_config.layer_configs.empty() == vit_layers.empty());
    for (size_t layer_idx = 0; layer_idx < vit_layers.size(); ++layer_idx) {
      HWY_ASSERT(vit_layers[layer_idx].layer_config.type ==
                 LayerAttentionType::kVit);
      other_layer1 = other1 ? other1->VitLayer(layer_idx) : nullptr;
      other_layer2 = other2 ? other2->VitLayer(layer_idx) : nullptr;
      VitLayer(layer_idx)->ForEachTensor(other_layer1, other_layer2, func);
    }
  }  // `ForEachTensor`

  // Zero-initializes only the allocated tensors in `*this`.
  void ZeroInit();
  // Copies only the allocated tensors in `*this` from tensors in `other`.
  void CopyFrom(const WeightsPtrs& other);

  enum class Mode {
    // Parallel I/O, decompress to BF16. Best for large batch sizes.
    kReadBF16,
    // Parallel I/O, insert row-wise padding. Safe default.
    kRead,
    // Best for large weights relative to available memory, especially for
    // frequent invocations of small batches and short sequences. Adds noise to
    // performance measurements due to I/O variability.
    kMap
  };

  static const char* ToString(Mode mode) {
    switch (mode) {
      case Mode::kReadBF16:
        return "ReadBF16";
      case Mode::kRead:
        return "Read";
      case Mode::kMap:
        return "Map";
      default:
        HWY_DASSERT(false);
        return "?";
    }
  }

  // Reads tensor data from `BlobStore` or aborts on error. `map` is a user
  // override for whether to map blobs or read them. Returns the mode used.
  Mode ReadFromBlobs(const ModelStore& model, BlobReader& reader,
                     const LoaderArgs& loader, const InferenceArgs& inference,
                     std::vector<MatOwner>& mat_owners, ThreadingContext& ctx);

  // Adds one blob for each tensor's data and returns all serialized MatPtr.
  std::vector<uint32_t> AddTensorDataToWriter(BlobWriter& writer) const;

 private:
  // For reshaping file tensors to the shape expected by the code. This would
  // ideally already happen in the importer. Called by ReadFromBlobs.
  void Fixup(std::vector<MatOwner>& mat_owners, ThreadingContext& ctx);

  MapPtr mapped_;
};  // `WeightsPtrs`
#undef TENSOR_ARGS

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_WEIGHTS_H_
