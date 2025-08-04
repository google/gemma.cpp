// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gemma/model_store.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <array>
#include <charconv>
#include <cstdlib>
#include <cstring>  // strcmp
#include <string>
#include <system_error>  // std::errc  // NOLINT

#include "compression/types.h"
#include "gemma/configs.h"  // ModelConfig, kMaxQKVDim
#include "gemma/tensor_info.h"
#include "gemma/tokenizer.h"
#include "io/blob_store.h"
#include "io/fields.h"
#include "io/io.h"  // Path
#include "util/basics.h"
#include "util/threading_context.h"
#include "hwy/base.h"
#include "hwy/profiler.h"

namespace gcpp {

// Single-file format contains blobs with these names:
static constexpr char kConfigName[] = "config";
static constexpr char kTokenizerName[] = "tokenizer";
static constexpr char kMatPtrsName[] = "toc";
// Pre-2025 format has one metadata blob. 'F' denoted f32.
static constexpr char kDecoratedScalesName[] = "Fscales";

static void WarnIfExtra(const IFields::ReadResult& result, const char* name) {
  // No warning if missing_fields > 0: those fields are default-initialized.
  if (result.extra_u32) {
    HWY_WARN(
        "Serialized blob %s has %u extra fields the code is not aware of. "
        "Consider updating to the latest code from GitHub.",
        name, result.extra_u32);
  }
}

// Returns the serialized tokenizer (std::string is required for proto).
// Reads it from a blob or from a separate file if pre-2025.
static std::string ReadTokenizer(BlobReader& reader,
                                 const Path& tokenizer_path) {
  PROFILER_ZONE("Startup.ReadTokenizer");

  std::string tokenizer;
  // Check prevents `CallWithSpan` from printing a warning.
  if (reader.Find(kTokenizerName)) {
    if (!reader.CallWithSpan<char>(
            kTokenizerName, [&tokenizer](const hwy::Span<const char> bytes) {
              tokenizer.assign(bytes.data(), bytes.size());
            })) {
      HWY_WARN(
          "Reading tokenizer blob failed, please raise an issue. You can "
          "instead specify a tokenizer file via --tokenizer.");
    }
  }

  // Read actual tokenizer from blob.
  if (!tokenizer.empty() && tokenizer != kMockTokenizer) {
    if (!tokenizer_path.Empty()) {
      HWY_WARN("--weights has tokenizer but overriding with %s.",
               tokenizer_path.path.c_str());
      return ReadFileToString(tokenizer_path);
    }

    return tokenizer;
  }

  // No blob but user specified path to file: read it or abort.
  if (!tokenizer_path.Empty()) {
    return ReadFileToString(tokenizer_path);
  }

  HWY_WARN(
      "BlobStore does not contain a tokenizer and no --tokenizer was "
      "specified. Tests may continue but inference will fail.");
  return kMockTokenizer;
}

using KeyVec = std::vector<std::string>;

class TypePrefix {
 public:
  static Type TypeFromChar(char c) {
    switch (c) {
      case 'F':
        return Type::kF32;
      case 'B':
        return Type::kBF16;
      case '$':
        return Type::kSFP;
      case '2':
        return Type::kNUQ;
      default:
        // The other types were not written to pre-2025 files, hence no need to
        // encode and check for them here.
        return Type::kUnknown;
    }
  }

  TypePrefix(const KeyVec& keys, const BlobReader& reader) {
    for (size_t key_idx = 0; key_idx < keys.size(); ++key_idx) {
      const std::string& key = keys[key_idx];
      const Type type = TypeFromChar(key[0]);
      const uint64_t bytes = reader.Range(key_idx).bytes;
      bytes_[static_cast<size_t>(type)] += bytes;
      blobs_[static_cast<size_t>(type)]++;
      total_bytes_ += bytes;
    }
  }

  // Returns true for pre-2025 format, which has type prefixes and thus the
  // functions below may be used.
  bool HasPrefixes() const {
    return bytes_[static_cast<size_t>(Type::kUnknown)] != total_bytes_;
  }

  // Returns the weight type deduced from the histogram of blobs per type.
  // Rationale: We expect a mix of types due to varying precision requirements
  // for each tensor. The preferred weight type might not even be the most
  // common, because we prioritize higher compression for the *large* tensors.
  // Ignore types which only have a few blobs (might be metadata), and assume
  // that there would be at least 4 of the large tensors (in particular, global
  // attention layers). Hence return the smallest type with >= 4 blobs.
  Type DeduceWeightType() const {
    size_t min_bits = ~size_t{0};
    Type weight_type = Type::kUnknown;
    for (size_t i = 0; i < kNumTypes; ++i) {
      if (blobs_[i] < 4) continue;
      const size_t bits = TypeBits(static_cast<Type>(i));
      if (bits < min_bits) {
        min_bits = bits;
        weight_type = static_cast<Type>(i);
      }
    }
    return weight_type;
  }

  // Prints statistics on the total size of tensors by type.
  void PrintTypeBytes() const {
    for (size_t type_idx = 0; type_idx < kNumTypes; ++type_idx) {
      const Type type = static_cast<Type>(type_idx);
      const uint64_t bytes = bytes_[type_idx];
      if (bytes == 0) continue;
      const double percent = 100.0 * bytes / total_bytes_;
      fprintf(stderr, "%12zu blob bytes (%5.2f%%) of %4s\n",
              static_cast<size_t>(bytes), percent, TypeName(type));
    }
  }

 private:
  uint64_t total_bytes_ = 0;
  std::array<size_t, kNumTypes> bytes_{0};
  std::array<size_t, kNumTypes> blobs_{0};
};

// Returns 0 if the blob does not seem to be a per-layer tensor, otherwise the
// layer index.
static size_t LayerIdxFromKey(const std::string& key) {
  const auto parse_num = [&key](size_t begin, size_t end) -> int {
    HWY_DASSERT(begin <= end);
    HWY_DASSERT(end <= key.size());
    int val = 0;
    auto [ptr, ec] = std::from_chars(key.data() + begin, key.data() + end, val);
    return (ec == std::errc()) ? val : -1;
  };

  const size_t suffix_pos = key.rfind('_');
  // If there is no digit after the last underscore, it is not a layer name.
  if (suffix_pos == std::string::npos) return 0;
  if (suffix_pos == key.size() - 1) return 0;

  int layer_idx = parse_num(suffix_pos + 1, key.size());

  HWY_ASSERT(layer_idx < 999);
  return layer_idx == -1 ? 0 : static_cast<size_t>(layer_idx);
}

// Returns the number of layers based on the largest blob name suffix seen.
// This works with or without type prefixes because it searches for suffixes.
static size_t DeduceNumLayers(const KeyVec& keys) {
  // Built-in self-test.
  {
    HWY_ASSERT(LayerIdxFromKey("gr_conv_w_2") == 2);   // common case
    HWY_ASSERT(LayerIdxFromKey("prefix_") == 0);       // no number
    HWY_ASSERT(LayerIdxFromKey("c_embedding") == 0);   // per-model
    HWY_ASSERT(LayerIdxFromKey("c_final_norm") == 0);  // per-model, two _
  }

  size_t max_layer_idx = 0;
  for (const std::string& key : keys) {
    max_layer_idx = HWY_MAX(max_layer_idx, LayerIdxFromKey(key));
  }
  return max_layer_idx + 1;
}

// Looks for known tensor names associated with model families.
// This works with or without type prefixes because it searches for substrings.
static int DeduceLayerTypes(const BlobReader& reader) {
  int layer_types = 0;
  for (size_t key_idx = 0; key_idx < reader.Keys().size(); ++key_idx) {
    const std::string& key = reader.Keys()[key_idx];
    if (key.find("gr_conv_w") != std::string::npos) {  // NOLINT
      return kDeducedGriffin;
    }
    if (key.find("qkv_ein_w") != std::string::npos) {  // NOLINT
      layer_types |= kDeducedViT;
    }
    if (key.find("img_pos_emb") != std::string::npos) {  // NOLINT
      // About 5.88 elements per pixel; assume at least bf16.
      if (reader.Range(key_idx).bytes > 448 * 448 * 5 * sizeof(BF16)) {
        layer_types |= kDeduced448;
      }
    }
  }
  return layer_types;
}

// `wrapping_override` is forwarded from the command line. For pre-2025 files
// without `ModelConfig`, it is the only way to force PT.
static ModelConfig ReadOrDeduceConfig(BlobReader& reader,
                                      Tristate wrapping_override) {
  const TypePrefix type_prefix(reader.Keys(), reader);
  Type deduced_weight = Type::kUnknown;
  if (type_prefix.HasPrefixes()) {
    deduced_weight = type_prefix.DeduceWeightType();
    type_prefix.PrintTypeBytes();
  }

  // Always deduce so we can verify it against the config we read.
  const size_t layers = DeduceNumLayers(reader.Keys());
  const int layer_types = DeduceLayerTypes(reader);
  const Model deduced_model =
      DeduceModel(reader.blob_path(), layers, layer_types);

  ModelConfig config;
  // Check first to prevent `CallWithSpan` from printing a warning.
  if (reader.Find(kConfigName)) {
    HWY_ASSERT(reader.CallWithSpan<uint32_t>(
        kConfigName, [&config](const SerializedSpan serialized) {
          const IFields::ReadResult result = config.Read(serialized, 0);
          WarnIfExtra(result, kConfigName);
          HWY_ASSERT_M(result.pos != 0, "Error deserializing config");
        }));

    HWY_ASSERT(config.model != Model::UNKNOWN);
    HWY_ASSERT(config.wrapping != PromptWrapping::kSentinel);
    HWY_ASSERT(config.weight != Type::kUnknown);
    for (const LayerConfig& layer_config : config.layer_configs) {
      if (static_cast<size_t>(layer_config.qkv_dim) > kMaxQKVDim) {
        HWY_ABORT("Increase kMaxQKVDim to at least %u.", layer_config.qkv_dim);
      }
    }

    // We trust the deserialized config, but checking helps to validate the
    // deduction, which we rely on below for pre-2025 files.
    if (config.model != deduced_model) {
      const std::string suffix = WrappingSuffix(config.wrapping);
      HWY_WARN("Detected model %s does not match config %s.",
               (std::string(ModelPrefix(deduced_model)) + suffix).c_str(),
               (std::string(ModelPrefix(config.model)) + suffix).c_str());
    }
    return config;
  }

  // Pre-2025 format: no config, rely on deduction plus `wrapping_override`.
  return ModelConfig(deduced_model, deduced_weight,
                     ChooseWrapping(deduced_model, wrapping_override));
}

static std::vector<float> ReadScales(BlobReader& reader,
                                     const ModelConfig& config) {
  std::vector<float> scales;
  // Check first to prevent `CallWithSpan` from printing a warning. This blob is
  // optional even in pre-2025 format; Griffin was the first to include it.
  if (reader.Find(kDecoratedScalesName)) {
    HWY_ASSERT(reader.CallWithSpan<float>(
        kDecoratedScalesName,
        [&scales](const hwy::Span<const float> scales_blob) {
          scales.assign(scales_blob.cbegin(), scales_blob.cend());
        }));
  }
  return scales;
}

// Single-file format: reads `MatPtr` from the blob; returns false if not found.
bool ModelStore::ReadMatPtrs(BlobReader& reader) {
  // Check first to prevent `CallWithSpan` from printing a warning.
  if (!reader.Find(kMatPtrsName)) return false;

  PROFILER_ZONE("Startup.ReadMatPtrs");

  // For verifying `config_.weight`.
  size_t min_bits = ~size_t{0};
  Type weight_type = Type::kUnknown;

  HWY_ASSERT(reader.CallWithSpan<uint32_t>(
      kMatPtrsName, [&, this](SerializedSpan serialized) {
        for (size_t pos = 0; pos < serialized.size();) {
          MatPtr mat;
          const IFields::ReadResult result = mat.Read(serialized, pos);
          WarnIfExtra(result, mat.Name());
          if (result.pos == 0) {
            HWY_ABORT("Deserializing MatPtr %s failed (pos %zu of %zu).",
                      mat.Name(), pos, serialized.size());
          }
          pos = result.pos + result.extra_u32;

          // Retrieve actual key index because a writer may have written other
          // blobs before the tensor data.
          const BlobRange* range = reader.Find(mat.Name());
          HWY_ASSERT(range);
          const size_t key_idx = range->key_idx;
          AddMatPtr(key_idx, mat);

          const size_t bits = TypeBits(mat.GetType());
          if (bits < min_bits) {
            min_bits = bits;
            weight_type = mat.GetType();
          }
        }
      }));

  HWY_ASSERT(weight_type != Type::kUnknown);
  HWY_ASSERT(weight_type == config_.weight);

  return true;
}

// Pre-2025 format: synthesizes `MatPtr` from the blob names if `!ReadMatPtrs`.
void ModelStore::CreateMatPtrs(BlobReader& reader) {
  const TensorInfoRegistry tensors(config_);

  const KeyVec& keys = reader.Keys();
  mat_ptrs_.reserve(keys.size());
  // `key_idx` is the blob index. It is not the same as the index of the
  // `MatPtr` in `mat_ptrs_` because not all blobs are tensors.
  for (size_t key_idx = 0; key_idx < keys.size(); ++key_idx) {
    const Type type = TypePrefix::TypeFromChar(keys[key_idx][0]);
    if (type == Type::kUnknown) continue;  // likely not a tensor

    // Strip type prefix from the key. Still includes layer suffix.
    const std::string name = keys[key_idx].substr(1);
    const TensorInfo* info = tensors.Find(name);
    if (HWY_UNLIKELY(!info)) {
      if (name == "scales") continue;  // ignore, not a tensor.
      HWY_ABORT("Unknown tensor %s.", name.c_str());
    }
    // Unable to set scale already because they are ordered according to
    // `ForEachTensor`, which we do not know here. The initial value is 1.0f
    // and we set the correct value in `FindAndUpdateMatPtr`.
    AddMatPtr(key_idx, MatPtr(name.c_str(), type, ExtentsFromInfo(info)));
  }
  HWY_ASSERT(mat_ptrs_.size() <= keys.size());
  HWY_ASSERT(mat_ptrs_.size() == key_idx_.size());
}

ModelStore::ModelStore(BlobReader& reader, const Path& tokenizer_path,
                         Tristate wrapping)
    : config_(ReadOrDeduceConfig(reader, wrapping)),
      tokenizer_(ReadTokenizer(reader, tokenizer_path)) {
  if (!ReadMatPtrs(reader)) {  // Pre-2025 format.
    CreateMatPtrs(reader);
    scales_ = ReadScales(reader, config_);
    // ModelConfig serialized a vector of strings. Unpack into a set for more
    // efficient lookup.
    for (const std::string& name : config_.scale_base_names) {
      scale_base_names_.insert(name);
    }
    // If the model has scales, the config must know about it.
    HWY_ASSERT(scales_.empty() || !scale_base_names_.empty());
  }

  HWY_ASSERT(key_idx_.size() == mat_ptrs_.size());
}

ModelStore::~ModelStore() {
  // Sanity check: ensure all scales were consumed.
  HWY_ASSERT(scales_consumed_ == scales_.size());
}

const MatPtr* ModelStore::FindMat(const char* name) const {
  auto it = mat_idx_for_name_.find(name);
  if (it == mat_idx_for_name_.end()) return nullptr;
  const size_t mat_idx = it->second;
  const MatPtr* file_mat = &mat_ptrs_[mat_idx];
  HWY_ASSERT(!strcmp(file_mat->Name(), name));
  return file_mat;
}

bool ModelStore::FindAndUpdateMatPtr(MatPtr& mat, size_t& key_idx) const {
  const MatPtr* file_mat = FindMat(mat.Name());
  if (!file_mat) return false;
  if (file_mat->Rows() != mat.Rows() || file_mat->Cols() != mat.Cols()) {
    HWY_ABORT("Tensor %s shape %zu %zu mismatches file %zu %zu.", mat.Name(),
              mat.Rows(), mat.Cols(), file_mat->Rows(), file_mat->Cols());
  }
  // `Compress()` output is always packed because it assumes a 1D array.
  HWY_ASSERT(mat.IsPacked());
  // Update fields. Name already matched, otherwise we would not find it.
  // For MatPtr tensors, the type will be `kUnknown`. If it was a `MatPtrT`,
  // ensure the type set via code matches the file.
  HWY_ASSERT_M(
      mat.GetType() == Type::kUnknown || mat.GetType() == file_mat->GetType(),
      mat.Name());
  mat.SetType(file_mat->GetType());
  if (scales_.empty()) {
    // `file_mat->Scale()` is either read from file, or we have pre-2025 format
    // without the optional scales, and it is default-initialized to 1.0f.
    mat.SetScale(file_mat->Scale());
  } else {  // Pre-2025 with scaling factors: set next if `mat` wants one.
    if (scale_base_names_.find(StripLayerSuffix(mat.Name())) !=
        scale_base_names_.end()) {
      HWY_ASSERT(scales_consumed_ < scales_.size());
      mat.SetScale(scales_[scales_consumed_++]);
    }
  }

  key_idx = key_idx_[file_mat - mat_ptrs_.data()];
  return true;
}

static void AddBlob(const char* name, const std::vector<uint32_t>& data,
                    BlobWriter& writer) {
  HWY_ASSERT(!data.empty());
  writer.Add(name, data.data(), data.size() * sizeof(data[0]));
}

void WriteSingleFile(const ModelConfig& config, const GemmaTokenizer& tokenizer,
                     const std::vector<uint32_t>& serialized_mat_ptrs,
                     BlobWriter& writer) {
  HWY_ASSERT(config.model != Model::UNKNOWN);
  HWY_ASSERT(config.weight != Type::kUnknown);
  HWY_ASSERT(config.wrapping != PromptWrapping::kSentinel);
  const std::vector<uint32_t> serialized_config = config.Write();
  AddBlob(kConfigName, serialized_config, writer);

  const std::string serialized_tokenizer = tokenizer.Serialize();
  HWY_ASSERT(!serialized_tokenizer.empty());
  writer.Add(kTokenizerName, serialized_tokenizer.data(),
             serialized_tokenizer.size());

  AddBlob(kMatPtrsName, serialized_mat_ptrs, writer);

  writer.Finalize();
}

}  // namespace gcpp
