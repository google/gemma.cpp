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

#include <complex>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "compression/shared.h"  // IsF32
#include "gemma/configs.h"       // ModelConfig
#include "gemma/model_store.h"   // ModelStore
#include "gemma/tensor_info.h"   // TensorInfoRegistry
#include "io/blob_store.h"       // BlobWriter
#include "util/mat.h"            // MatPtr
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

// Argument passed to the `ForEachTensor` callback.
struct TensorArgs {
  // `other_mat1` and `other_mat2` can be nullptr, or tensor(s) of the same
  // name/type from another `LayerWeightsPtrs` for iterating over tensor pairs
  // (for copying) or triples (for `AdamUpdateMV`). Set by `TENSOR_ARGS`.
  // `flags` is a combination of zero or more `Flags`.
  TensorArgs(MatPtr& mat, const MatPtr* other_mat1, const MatPtr* other_mat2,
             int flags)
      : mat(mat), other_mat1(other_mat1), other_mat2(other_mat2), flags(flags) {
    // Does not make sense to combine both flags.
    HWY_ASSERT(flags != (kMaybeRead | kOnlyAllocate));
  }

  MatPtr& mat;
  const MatPtr* other_mat1;  // either/both can be nullptr.
  const MatPtr* other_mat2;

  // TODO: freestanding enum class instead? These are mutually exclusive.
  enum Flags {
    // Read the tensor from the file and abort if it is not found.
    kMustRead = 0,
    // Not an error if the tensor is not present in the file. For example,
    // the _w1/_w2 tensors are not always present.
    kMaybeRead = 1,
    // Do not attempt to read, just allocate the tensor. Used for `Reshape`.
    kOnlyAllocate = 2,
  };
  const int flags;
};

// Shorthand for creating the argument to the `ForEachTensor` callback. A macro
// seems less bad than member pointer syntax.
#define TENSOR_ARGS(mat, flag)                     \
  TensorArgs(mat, other1 ? &other1->mat : nullptr, \
             other2 ? &other2->mat : nullptr, TensorArgs::flag)

// Per-layer weight metadata and pointers. The tensor data is owned by
// `WeightsOwner`. Note that this class could be type-erased: member functions
// do not actually use the `Weight` template argument. See `WeightsPtrs`.
// `TensorInfoRegistry` (constructed from `ModelConfig`) is the source of truth
// for all tensor shapes.
template <class Weight>
struct LayerWeightsPtrs {
  static inline std::string Concat(const char* base_name,
                                   const std::string& suffix) {
    return std::string(base_name) + suffix;
  }

  // Initializes tensor metadata without allocating.
  LayerWeightsPtrs(size_t layer_idx, const LayerConfig& config,
                   const TensorInfoRegistry& tensors)
      : suffix_(LayerSuffix(layer_idx)),
        attn_vec_einsum_w(Concat("att_ein", suffix_), tensors),
        qkv_einsum_w(Concat("qkv_ein", suffix_), tensors),
        qkv_einsum_w1(Concat("qkv1_w", suffix_), tensors),
        qkv_einsum_w2(Concat("qkv2_w", suffix_), tensors),
        attention_output_biases(Concat("attn_ob", suffix_), tensors),
        griffin(
            {.linear_x_w = {Concat("gr_lin_x_w", suffix_), tensors},
             .linear_x_biases = {Concat("gr_lin_x_b", suffix_), tensors},
             .linear_y_w = {Concat("gr_lin_y_w", suffix_), tensors},
             .linear_y_biases = {Concat("gr_lin_y_b", suffix_), tensors},
             .linear_out_w = {Concat("gr_lin_out_w", suffix_), tensors},
             .linear_out_biases = {Concat("gr_lin_out_b", suffix_), tensors},
             .conv_w = {Concat("gr_conv_w", suffix_), tensors},
             .conv_biases = {Concat("gr_conv_b", suffix_), tensors},
             .gate_w = {Concat("gr_gate_w", suffix_), tensors},
             .gate_biases = {Concat("gr_gate_b", suffix_), tensors},
             .a = {Concat("gr_a", suffix_), tensors}}),
        // MultiHeadDotProductAttention.
        vit({.attn_out_w = {Concat("attn_out_w", suffix_), tensors},
             .attn_out_b = {Concat("attn_out_b", suffix_), tensors},
             .qkv_einsum_w = {Concat("qkv_ein_w", suffix_), tensors},
             .qkv_einsum_b = {Concat("qkv_ein_b", suffix_), tensors},
             .linear_0_w = {Concat("linear_0_w", suffix_), tensors},
             .linear_0_b = {Concat("linear_0_b", suffix_), tensors},
             .linear_1_w = {Concat("linear_1_w", suffix_), tensors},
             .linear_1_b = {Concat("linear_1_b", suffix_), tensors},
             .layer_norm_0_bias = {Concat("ln_0_bias", suffix_), tensors},
             .layer_norm_0_scale = {Concat("ln_0_scale", suffix_), tensors},
             .layer_norm_1_bias = {Concat("ln_1_bias", suffix_), tensors},
             .layer_norm_1_scale = {Concat("ln_1_scale", suffix_), tensors}}),
        gating_einsum_w(Concat("gating_ein", suffix_), tensors),
        gating_einsum_w1(Concat("gating1_w", suffix_), tensors),
        gating_einsum_w2(Concat("gating2_w", suffix_), tensors),
        linear_w(Concat("linear_w", suffix_), tensors),
        pre_attention_norm_scale(Concat("pre_att_ns", suffix_), tensors),
        pre_ffw_norm_scale(Concat("pre_ff_ns", suffix_), tensors),
        post_attention_norm_scale(Concat("post_att_ns", suffix_), tensors),
        post_ffw_norm_scale(Concat("post_ff_ns", suffix_), tensors),
        ffw_gating_biases(Concat("ffw_gat_b", suffix_), tensors),
        ffw_output_biases(Concat("ffw_out_b", suffix_), tensors),
        att_weights(Concat("att_w", suffix_), tensors),
        key_norm_scale(Concat("key_norm", suffix_), tensors),
        query_norm_scale(Concat("query_norm", suffix_), tensors),
        layer_config(config) {}
  ~LayerWeightsPtrs() = default;

  const std::string suffix_;

  // If weights are f32, also f32; otherwise at least bf16. Useful for ops that
  // do not yet support smaller compressed types, or require at least bf16. When
  // weights are f32, we also want such tensors to be f32.
  // If weights are complex, this is also complex.
  using WeightF32OrBF16 =
      hwy::If<hwy::IsSame<Weight, std::complex<double>>(), std::complex<double>,
              hwy::If<hwy::IsSame<Weight, double>(), double,
                      hwy::If<IsF32<Weight>(), float, BF16>>>;

  MatPtrT<Weight> attn_vec_einsum_w;
  // qkv_einsum_w holds 2 different matrices, which may be separated out.
  // On reading, which is used depends on what is in the file.
  // At inference, the one with a non-null ptr is used.
  MatPtrT<Weight> qkv_einsum_w;
  MatPtrT<Weight> qkv_einsum_w1;
  MatPtrT<Weight> qkv_einsum_w2;
  MatPtrT<float> attention_output_biases;

  struct {
    MatPtrT<Weight> linear_x_w;
    MatPtrT<float> linear_x_biases;
    MatPtrT<Weight> linear_y_w;
    MatPtrT<float> linear_y_biases;
    MatPtrT<Weight> linear_out_w;
    MatPtrT<float> linear_out_biases;
    MatPtrT<float> conv_w;
    MatPtrT<float> conv_biases;
    MatPtrT<Weight> gate_w;
    MatPtrT<float> gate_biases;
    MatPtrT<float> a;
  } griffin;

  struct {
    // MultiHeadDotProductAttention.
    MatPtrT<WeightF32OrBF16> attn_out_w;
    MatPtrT<float> attn_out_b;
    MatPtrT<WeightF32OrBF16> qkv_einsum_w;
    MatPtrT<float> qkv_einsum_b;
    // MlpBlock.
    MatPtrT<WeightF32OrBF16> linear_0_w;
    MatPtrT<float> linear_0_b;
    MatPtrT<WeightF32OrBF16> linear_1_w;
    MatPtrT<float> linear_1_b;
    // LayerNorm.
    MatPtrT<WeightF32OrBF16> layer_norm_0_bias;
    MatPtrT<WeightF32OrBF16> layer_norm_0_scale;
    MatPtrT<WeightF32OrBF16> layer_norm_1_bias;
    MatPtrT<WeightF32OrBF16> layer_norm_1_scale;
  } vit;

  // gating_einsum_w holds 2 different matrices, which may be separated out.
  // On reading, which is used depends on what is in the file.
  // At inference, the one with a non-null ptr is used.
  MatPtrT<Weight> gating_einsum_w;
  MatPtrT<Weight> gating_einsum_w1;
  MatPtrT<Weight> gating_einsum_w2;
  MatPtrT<Weight> linear_w;
  // We don't yet have an RMSNorm that accepts all Weight.
  MatPtrT<WeightF32OrBF16> pre_attention_norm_scale;
  MatPtrT<WeightF32OrBF16> pre_ffw_norm_scale;
  MatPtrT<WeightF32OrBF16> post_attention_norm_scale;
  MatPtrT<WeightF32OrBF16> post_ffw_norm_scale;

  MatPtrT<float> ffw_gating_biases;
  MatPtrT<float> ffw_output_biases;

  MatPtrT<Weight> att_weights;  // For Reshape(); kOnlyAllocate.

  MatPtrT<WeightF32OrBF16> key_norm_scale;
  MatPtrT<WeightF32OrBF16> query_norm_scale;

  const LayerConfig& layer_config;

  // Calls `func(TensorArgs)` for each tensor which is in use for the
  // current `layer_config`. `other1` and `other2` are optional arguments so we
  // can also iterate over pairs or triples of tensors for `AdamUpdateMV`.
  // Public because also called by `WeightsPtrs`.
  template <class Func>
  void ForEachTensor(const LayerWeightsPtrs<Weight>* other1,
                     const LayerWeightsPtrs<Weight>* other2, Func func) {
    if (layer_config.type == LayerAttentionType::kVit) {
      // MHA.
      func(TENSOR_ARGS(vit.attn_out_w, kMustRead));
      func(TENSOR_ARGS(vit.attn_out_b, kMustRead));
      func(TENSOR_ARGS(vit.qkv_einsum_w, kMustRead));
      func(TENSOR_ARGS(vit.qkv_einsum_b, kMustRead));
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
      // Not read, will be filled by Reshape() from `attn_vec_einsum_w`.
      func(TENSOR_ARGS(att_weights, kOnlyAllocate));
      func(TENSOR_ARGS(attn_vec_einsum_w, kMustRead));
      func(TENSOR_ARGS(qkv_einsum_w, kMustRead));
      func(TENSOR_ARGS(qkv_einsum_w1, kMaybeRead));
      func(TENSOR_ARGS(qkv_einsum_w2, kMaybeRead));
    } else {
      func(TENSOR_ARGS(griffin.linear_x_w, kMustRead));
      func(TENSOR_ARGS(griffin.linear_x_biases, kMustRead));
      func(TENSOR_ARGS(griffin.linear_y_w, kMustRead));
      func(TENSOR_ARGS(griffin.linear_y_biases, kMustRead));
      func(TENSOR_ARGS(griffin.linear_out_w, kMustRead));
      func(TENSOR_ARGS(griffin.linear_out_biases, kMustRead));
      func(TENSOR_ARGS(griffin.conv_w, kMustRead));
      func(TENSOR_ARGS(griffin.conv_biases, kMustRead));
      func(TENSOR_ARGS(griffin.gate_w, kMustRead));
      func(TENSOR_ARGS(griffin.gate_biases, kMustRead));
      func(TENSOR_ARGS(griffin.a, kMustRead));
    }
    {
      func(TENSOR_ARGS(gating_einsum_w, kMustRead));
      func(TENSOR_ARGS(gating_einsum_w1, kMaybeRead));
      func(TENSOR_ARGS(gating_einsum_w2, kMaybeRead));
      func(TENSOR_ARGS(linear_w, kMustRead));
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

  void RandInit(float stddev, std::mt19937& gen) {
    ForEachTensor(nullptr, nullptr, [&](const TensorArgs& t) {
      if (!t.mat.HasPtr()) return;
      gcpp::RandInit(t.mat, stddev, gen);
    });
  }

  // Allocates memory for all the tensors in the layer. Note that this is slow
  // (non-parallel) and only used for a stand-alone layer.
  void AllocateForTest(MatOwners& mat_owners) {
    ForEachTensor(nullptr, nullptr, [&](const TensorArgs& t) {
      // `backprop/` does not use row accessors and hence requires kPacked.
      mat_owners.AllocateFor(t.mat, MatPadding::kPacked);
    });
  }

  // Initializes att_weights from `attn_vec_einsum_w`, hence this must be called
  // after reading weights via `ForEachTensor`.
  // TODO: update compression/convert_weights to bake this in.
  void Reshape() {
    // NUQ is handled by a specialization in weights.cc.
    HWY_ASSERT(attn_vec_einsum_w.GetType() != Type::kNUQ);

    const size_t model_dim = layer_config.model_dim;
    const size_t heads = layer_config.heads;
    const size_t qkv_dim = layer_config.qkv_dim;

    // Reshape [heads, model_dim, qkv_dim] to [model_dim, heads * qkv_dim].
    HWY_ASSERT(att_weights.HasPtr());
    HWY_ASSERT(att_weights.GetType() == attn_vec_einsum_w.GetType());
    HWY_ASSERT(att_weights.Rows() == model_dim);
    HWY_ASSERT(att_weights.Cols() == heads * qkv_dim);
    HWY_ASSERT(attn_vec_einsum_w.HasPtr());
    HWY_ASSERT(attn_vec_einsum_w.Rows() == heads * model_dim);
    HWY_ASSERT(attn_vec_einsum_w.Cols() == qkv_dim);
    const size_t T_bytes = att_weights.ElementBytes();
    for (size_t m = 0; m < model_dim; ++m) {
      uint8_t* HWY_RESTRICT out_row =
          reinterpret_cast<uint8_t*>(att_weights.Row(m));
      for (size_t h = 0; h < heads; ++h) {
        hwy::CopyBytes(attn_vec_einsum_w.Row(h * model_dim + m),
                       out_row + h * qkv_dim * T_bytes, qkv_dim * T_bytes);
      }
    }
    att_weights.SetScale(attn_vec_einsum_w.Scale());
  }
};

// Holds layer-independent weight metadata and pointers plus per-layer
// `LayerWeightsPtrs`. The tensor data is owned by `WeightsOwner`. As with
// `LayerWeightsPtrs`, this class could be type-erased: member functions do not
// actually use the `Weight` template argument. The template does allow user
// code to dispatch only once. However, most tensors are large enough that
// dispatch at each usage would be feasible.
// TODO: move `gemma-inl.h` toward dispatch at each usage.
// TODO: rename to WeightsPtrs.
template <class Weight>
struct ModelWeightsPtrs {
  using WeightT = Weight;

  explicit ModelWeightsPtrs(const ModelConfig& config)
      : tensors_(config),
        // No suffix, these are per-model.
        embedder_input_embedding("c_embedding", tensors_),
        final_norm_scale("c_final_norm", tensors_),
        vit_encoder_norm_bias("enc_norm_bias", tensors_),
        vit_encoder_norm_scale("enc_norm_scale", tensors_),
        vit_img_embedding_bias("img_emb_bias", tensors_),
        vit_img_embedding_kernel("img_emb_kernel", tensors_),
        vit_img_pos_embedding("img_pos_emb", tensors_),
        vit_img_head_bias("img_head_bias", tensors_),
        vit_img_head_kernel("img_head_kernel", tensors_),
        mm_embed_norm("mm_embed_norm", tensors_),
        weights_config(config) {
    c_layers.reserve(config.layer_configs.size());
    for (size_t idx = 0; idx < config.layer_configs.size(); ++idx) {
      const LayerConfig& layer_config = config.layer_configs[idx];
      c_layers.emplace_back(idx, layer_config, tensors_);
    }
    for (size_t idx = 0; idx < config.vit_config.layer_configs.size(); ++idx) {
      const LayerConfig& layer_config = config.vit_config.layer_configs[idx];
      vit_layers.emplace_back(idx, layer_config, tensors_);
    }
  }

  ~ModelWeightsPtrs() = default;
  // = F32 if weights are F32, else BF16.
  using WeightF32OrBF16 = typename LayerWeightsPtrs<Weight>::WeightF32OrBF16;

  // Passed to all  `MatPtrT` initializers, hence must be initialized first.
  const TensorInfoRegistry tensors_;

  // TODO: switch to SFP?
  MatPtrT<WeightF32OrBF16> embedder_input_embedding;
  MatPtrT<WeightF32OrBF16> final_norm_scale;

  // Vit parts.
  MatPtrT<WeightF32OrBF16> vit_encoder_norm_bias;
  MatPtrT<WeightF32OrBF16> vit_encoder_norm_scale;
  MatPtrT<float> vit_img_embedding_bias;
  MatPtrT<WeightF32OrBF16> vit_img_embedding_kernel;
  MatPtrT<float> vit_img_pos_embedding;
  // The head maps from VitConfig::model_dim (Vit final layer) to
  // model_dim (LLM input).
  MatPtrT<float> vit_img_head_bias;
  MatPtrT<WeightF32OrBF16> vit_img_head_kernel;

  MatPtrT<WeightF32OrBF16> mm_embed_norm;

  const ModelConfig& weights_config;

  std::vector<LayerWeightsPtrs<Weight>> c_layers;
  std::vector<LayerWeightsPtrs<Weight>> vit_layers;

  const LayerWeightsPtrs<Weight>* GetLayer(size_t layer) const {
    return &c_layers[layer];
  }
  LayerWeightsPtrs<Weight>* GetLayer(size_t layer) { return &c_layers[layer]; }
  const LayerWeightsPtrs<Weight>* VitLayer(size_t layer) const {
    return &vit_layers[layer];
  }
  LayerWeightsPtrs<Weight>* VitLayer(size_t layer) {
    return &vit_layers[layer];
  }

  // Called via `CallT`. `other1` and `other2` are usually null, but can be
  // used to copy from another set of weights. Public because called by tests
  // and `WeightsOwner`.
  template <class Func>
  void ForEachTensor(const ModelWeightsPtrs<Weight>* other1,
                     const ModelWeightsPtrs<Weight>* other2, Func func) {
    const LayerWeightsPtrs<Weight>* other_layer1 = nullptr;
    const LayerWeightsPtrs<Weight>* other_layer2 = nullptr;
    func(TENSOR_ARGS(embedder_input_embedding, kMustRead));
    func(TENSOR_ARGS(final_norm_scale, kMustRead));

    if (!weights_config.vit_config.layer_configs.empty()) {  // Vit parts.
      func(TENSOR_ARGS(vit_encoder_norm_bias, kMustRead));
      func(TENSOR_ARGS(vit_encoder_norm_scale, kMustRead));
      func(TENSOR_ARGS(vit_img_embedding_bias, kMustRead));
      func(TENSOR_ARGS(vit_img_embedding_kernel, kMustRead));
      func(TENSOR_ARGS(vit_img_pos_embedding, kMustRead));
      func(TENSOR_ARGS(vit_img_head_bias, kMustRead));
      func(TENSOR_ARGS(vit_img_head_kernel, kMustRead));

      if (weights_config.wrapping == PromptWrapping::GEMMA_VLM) {
        func(TENSOR_ARGS(mm_embed_norm, kMustRead));
      }
    }

    for (size_t layer_idx = 0; layer_idx < c_layers.size(); ++layer_idx) {
      if (other1) other_layer1 = other1->GetLayer(layer_idx);
      if (other2) other_layer2 = other2->GetLayer(layer_idx);
      GetLayer(layer_idx)->ForEachTensor(other_layer1, other_layer2, func);
    }

    HWY_ASSERT(weights_config.vit_config.layer_configs.empty() ==
               vit_layers.empty());
    for (size_t layer_idx = 0; layer_idx < vit_layers.size(); ++layer_idx) {
      HWY_ASSERT(vit_layers[layer_idx].layer_config.type ==
                 LayerAttentionType::kVit);
      other_layer1 = other1 ? other1->VitLayer(layer_idx) : nullptr;
      other_layer2 = other2 ? other2->VitLayer(layer_idx) : nullptr;
      VitLayer(layer_idx)->ForEachTensor(other_layer1, other_layer2, func);
    }
  }  // `ForEachTensor`

  // Zero-initializes only the allocated tensors in `*this`.
  void ZeroInit() {
    ForEachTensor(nullptr, nullptr, [](const TensorArgs& t) {
      if (!t.mat.HasPtr()) return;
      gcpp::ZeroInit(t.mat);
    });
  }

  void RandInit(float stddev, std::mt19937& gen) {
    ForEachTensor(nullptr, nullptr, [stddev, &gen](const TensorArgs& t) {
      if (!t.mat.HasPtr()) return;
      gcpp::RandInit(t.mat, stddev, gen);
    });
  }

  // Copies only the allocated tensors in `*this` from tensors in `other`.
  void CopyFrom(const ModelWeightsPtrs<Weight>& other) {
    ForEachTensor(&other, nullptr, [](const TensorArgs& t) {
      if (!t.mat.HasPtr()) return;
      HWY_ASSERT(t.other_mat1 && t.other_mat1->HasPtr());
      CopyMat(*t.other_mat1, t.mat);
    });
  }

  // Instead of reading, only allocates memory for all tensors. Used by
  // `optimizer.cc` via the `Gemma` constructor without weights.
  void AllocateForTest(MatOwners& mat_owners, hwy::ThreadPool& pool) {
    // First get a list of all the tensors.
    std::vector<MatPtr*> all_mat;
    all_mat.reserve(10 * c_layers.size());
    ForEachTensor(nullptr, nullptr, [&all_mat](const TensorArgs& t) {
      all_mat.push_back(&t.mat);
    });

    // `backprop/` does not use row accessors and hence requires kPacked.
    mat_owners.AllocateFor(all_mat, MatPadding::kPacked, pool);
  }

  // For reshaping file tensors to the shape expected by the code. This would
  // ideally already happen in the importer. Must be called after reading and
  // updating the attention weights.
  void Reshape(hwy::ThreadPool& pool) {
    pool.Run(0, c_layers.size(), [this](uint64_t layer, size_t /*thread*/) {
      GetLayer(layer)->Reshape();
    });

    pool.Run(0, vit_layers.size(), [this](uint64_t layer, size_t /*thread*/) {
      VitLayer(layer)->Reshape();
    });
  }
};  // `WeightsPtrs`
#undef TENSOR_ARGS

// Type-erased facade for `WeightsPtrs<T>`, stored inside the non-template
// `Gemma`. Also owns the underlying memory.
class WeightsOwner {
 public:
  // `weight_type` is obtained from `ModelConfig` in `ModelStore`.
  WeightsOwner(Type weight_type) : weight_type_(weight_type) {}

  // Reads tensor data from `BlobStore`, or for tensors marked `kOnlyAllocate`,
  // allocates memory and reshapes. Aborts on error.
  void ReadOrAllocate(const ModelStore& model, BlobReader& reader,
                      hwy::ThreadPool& pool);

  // Calls `func(std::unique_ptr<WeightsPtrs<T>>&, args)`. `func` typically
  // calls `ForEachTensor`.
  template <class Func, typename... TArgs>
  decltype(auto) CallT(const Func& func, TArgs&&... args) const {
    if (HWY_LIKELY(weight_type_ == Type::kSFP)) {
      return func(sfp_weights_, std::forward<TArgs>(args)...);
    } else if (weight_type_ == Type::kNUQ) {
      return func(nuq_weights_, std::forward<TArgs>(args)...);
    } else if (weight_type_ == Type::kF32) {
      return func(float_weights_, std::forward<TArgs>(args)...);
    } else if (weight_type_ == Type::kBF16) {
      return func(bf16_weights_, std::forward<TArgs>(args)...);
    }
    return HWY_ABORT("Unsupported weight type %s.", TypeName(weight_type_));
  }

  // For writers:

  // Adds one blob for each tensor's data and returns all serialized MatPtr.
  std::vector<uint32_t> AddTensorDataToWriter(BlobWriter& writer) const;

  // For backprop/:

  // Only allocates; must subsequently call `ZeroInit` or `RandInit`.
  void AllocateForTest(const ModelConfig& config, hwy::ThreadPool& pool);
  void ZeroInit();
  void RandInit(float stddev, std::mt19937& gen);  // F32 or F64 only.
  void LogWeightStatsF32();

  ModelWeightsPtrs<float>* GetF32() const {
    HWY_ASSERT(weight_type_ == Type::kF32);
    return float_weights_.get();
  }

  // Usually taken care of by `ReadOrAllocate`, but must also be called by
  // `optimize_test, which updates the attention weights from which this copies.
  void Reshape(hwy::ThreadPool& pool);

 private:
  Type weight_type_;

  // Allocates `*_weights_`, but not yet the tensors inside. This is split out
  // of `CallT` so that can be const.
  void AllocatePointer(const ModelConfig& config);

  // Only one is non-null, determined by `weight_type_`.
  std::unique_ptr<ModelWeightsPtrs<float>> float_weights_;
  std::unique_ptr<ModelWeightsPtrs<BF16>> bf16_weights_;
  std::unique_ptr<ModelWeightsPtrs<SfpStream>> sfp_weights_;
  std::unique_ptr<ModelWeightsPtrs<NuqStream>> nuq_weights_;

  // Owns the memory referenced by all `MatPtr`.
  MatOwners mat_owners_;
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_WEIGHTS_H_
