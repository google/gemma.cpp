// Tests that batched `ComputeEmbeddingsT` matches `ComputeEmbeddingT` of each
// prompt on a small synthetic EmbeddingGemma 2 model.

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "compression/types.h"
#include "gemma/configs.h"
#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "gemma/weights.h"
#include "ops/matmul.h"
#include "util/mat.h"
#include "util/threading_context.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/embeddings_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// Defined in gemma.cc.
void ComputeEmbeddingT(const RuntimeConfig& runtime_config_in,
                       const PromptTokens& prompt, const ModelConfig& config,
                       const InferenceArgs& inference,
                       const WeightsPtrs& weights,
                       hwy::AlignedVector<float>& embedding, MatMulEnv& env);
void ComputeEmbeddingsT(const RuntimeConfig& runtime_config_in,
                        hwy::Span<const PromptTokens> prompts,
                        const ModelConfig& config,
                        const InferenceArgs& inference,
                        const WeightsPtrs& weights,
                        std::vector<hwy::AlignedVector<float>>& embeddings,
                        MatMulEnv& env, EmbeddingsWorkspace* workspace);

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {
namespace {

HWY_EXPORT(ComputeEmbeddingT);
HWY_EXPORT(ComputeEmbeddingsT);

bool IsUnused(const char* name) {
  for (const char* prefix : {"att_ein", "qkv_ein", "gating_ein"}) {
    if (strncmp(name, prefix, strlen(prefix)) == 0) return true;
  }
  return false;
}

void AllocateRandom(MatPtr& mat, uint64_t seed, ThreadingContext& ctx,
                    std::vector<MatOwner>& owners) {
  if (mat.IsEmpty()) return;
  mat.SetType(Type::kBF16);
  mat.SetScale(1.0f);
  owners.emplace_back();
  owners.back().AllocateFor(mat, ctx.allocator, MatPadding::kPacked);
  BF16* base = reinterpret_cast<BF16*>(mat.RowBytes(0));
  float mean = 0.0f;
  if (strncmp(mat.Name(), "skip_scale", 10) == 0) {
    mean = 0.95f;
  } else if (strncmp(mat.Name(), "query_norm", 10) == 0 ||
             strncmp(mat.Name(), "key_norm", 8) == 0) {
    mean = -0.75f;  // RMSNorm adds 1.0.
  }
  uint64_t s = seed * 0x9E3779B97F4A7C15ULL + 1;
  for (size_t r = 0; r < mat.Rows(); ++r) {
    BF16* row = base + r * mat.Stride();
    for (size_t c = 0; c < mat.Stride(); ++c) {
      s ^= s >> 12;
      s ^= s << 25;
      s ^= s >> 27;
      const int32_t v = static_cast<int32_t>((s * 0x2545F4914F6CDD1DULL) >> 48);
      row[c] = hwy::BF16FromF32(mean + static_cast<float>(v - 32768) *
                                           (0.02f / 32768.0f));
    }
  }
}

struct TestModel {
  explicit TestModel(ThreadingContext& ctx)
      : config(Model::EMBEDDING_GEMMA2_LM, Type::kBF16,
               PromptWrapping::GEMMA_IT) {
    config.num_layers = 6;  // 5 sliding-window + 1 global attention layer.
    config.model_dim = 256;
    config.ple_dim = 256;
    config.vocab_size = 512;
    config.layer_configs.resize(config.num_layers);
    config.attention_window_sizes.resize(config.num_layers);
    for (size_t i = 0; i < config.num_layers; ++i) {
      LayerConfig& lc = config.layer_configs[i];
      lc.model_dim = config.model_dim;
      lc.ple_dim = config.ple_dim;
      lc.ff_hidden_dim = 512;
      if (i + 1 < config.num_layers) config.attention_window_sizes[i] = 16;
    }
    weights = std::make_unique<WeightsPtrs>(config);
    uint64_t seed = 1;
    AllocateRandom(weights->embedder_input_embedding, ++seed, ctx, owners);
    AllocateRandom(weights->final_norm_scale, ++seed, ctx, owners);
    AllocateRandom(weights->ple_model_proj, ++seed, ctx, owners);
    AllocateRandom(weights->ple_proj_norm, ++seed, ctx, owners);
    AllocateRandom(weights->default_proj, ++seed, ctx, owners);
    for (size_t i = 0; i < config.num_layers; ++i) {
      LayerWeightsPtrs* layer = weights->GetLayer(i);
      layer->ForEachTensor(nullptr, nullptr, [&](const TensorArgs& t) {
        if (IsUnused(t.mat.Name())) return;
        AllocateRandom(t.mat, ++seed, ctx, owners);
      });
    }
  }

  ModelConfig config;
  std::unique_ptr<WeightsPtrs> weights;
  std::vector<MatOwner> owners;
};

double NormSquared(hwy::Span<const float> v) {
  double sum = 0.0;
  for (float x : v) sum += static_cast<double>(x) * x;
  return sum;
}

double Cosine(hwy::Span<const float> a, hwy::Span<const float> b) {
  HWY_ASSERT(a.size() == b.size());
  double ab = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    ab += static_cast<double>(a[i]) * b[i];
  }
  const double aa = NormSquared(a);
  const double bb = NormSquared(b);
  HWY_ASSERT(aa > 0.0 && bb > 0.0);
  return ab / std::sqrt(aa * bb);
}

std::vector<std::vector<int>> MakePrompts(const std::vector<size_t>& lengths,
                                          size_t vocab_size) {
  std::vector<std::vector<int>> prompts;
  for (size_t p = 0; p < lengths.size(); ++p) {
    std::vector<int> prompt(lengths[p]);
    for (size_t i = 0; i < lengths[p]; ++i) {
      prompt[i] =
          static_cast<int>((i * 131 + p * 7919 + 7) % (vocab_size - 4)) + 2;
    }
    prompts.push_back(std::move(prompt));
  }
  return prompts;
}

void CheckBatchedMatchesSingle(const std::vector<size_t>& lengths) {
  ThreadingContext ctx{ThreadingArgs()};
  MatMulEnv env(ctx);
  TestModel model(ctx);
  InferenceArgs inference;
  inference.seq_len = 2048;
  RuntimeConfig runtime_config;

  const std::vector<std::vector<int>> storage =
      MakePrompts(lengths, model.config.vocab_size);
  std::vector<PromptTokens> prompts;
  for (const std::vector<int>& p : storage) prompts.emplace_back(p);

  // Reuses one workspace: grows from 1 prompt to all, then reuses for 2.
  EmbeddingsWorkspace workspace;
  std::vector<hwy::AlignedVector<float>> batched;
  HWY_DYNAMIC_DISPATCH(ComputeEmbeddingsT)(
      runtime_config, hwy::Span<const PromptTokens>(prompts.data(), 1),
      model.config, inference, *model.weights, batched, env, &workspace);
  HWY_DYNAMIC_DISPATCH(ComputeEmbeddingsT)(
      runtime_config, hwy::Span<const PromptTokens>(prompts), model.config,
      inference, *model.weights, batched, env, &workspace);
  ASSERT_EQ(batched.size(), prompts.size());
  const Activations* acts_ptr = workspace.activations.get();
  const KVCache* kv_ptr = workspace.kv_caches.data();

  std::vector<hwy::AlignedVector<float>> sub;
  HWY_DYNAMIC_DISPATCH(ComputeEmbeddingsT)(
      runtime_config, hwy::Span<const PromptTokens>(prompts.data(), 2),
      model.config, inference, *model.weights, sub, env, &workspace);
  EXPECT_EQ(workspace.activations.get(), acts_ptr);
  EXPECT_EQ(workspace.kv_caches.data(), kv_ptr);
  EXPECT_GT(Cosine(batched[0], sub[0]), 0.9999);
  EXPECT_GT(Cosine(batched[1], sub[1]), 0.9999);

  for (size_t p = 0; p < prompts.size(); ++p) {
    hwy::AlignedVector<float> single;
    HWY_DYNAMIC_DISPATCH(ComputeEmbeddingT)(runtime_config, prompts[p],
                                            model.config, inference,
                                            *model.weights, single, env);
    EXPECT_EQ(single.size(), kEmbeddingGemma2Dim);
    EXPECT_EQ(batched[p].size(), kEmbeddingGemma2Dim);
    EXPECT_NEAR(NormSquared(single), 1.0, 1e-5);
    EXPECT_NEAR(NormSquared(batched[p]), 1.0, 1e-5);
    EXPECT_GT(Cosine(single, batched[p]), 0.9999)
        << "prompt " << p << " of length " << lengths[p];
  }
  EXPECT_LT(Cosine(batched[0], batched[1]), 0.999);
}

TEST(EmbeddingGemma2Test, BatchedMatchesSingleVariableLength) {
  CheckBatchedMatchesSingle({5, 40, 17, 64, 1, 29});
}

// More rows than `kMaxBatchSize`, hence several groups.
TEST(EmbeddingGemma2Test, BatchedSeveralGroups) {
  CheckBatchedMatchesSingle({1500, 1400, 1300, 7});
}

TEST(EmbeddingGemma2Test, MatryoshkaTruncation) {
  ThreadingContext ctx{ThreadingArgs()};
  MatMulEnv env(ctx);
  TestModel model(ctx);
  InferenceArgs inference;
  inference.seq_len = 256;

  const std::vector<std::vector<int>> storage =
      MakePrompts({19, 37}, model.config.vocab_size);
  std::vector<PromptTokens> prompts;
  for (const std::vector<int>& p : storage) prompts.emplace_back(p);

  hwy::AlignedVector<float> full;
  HWY_DYNAMIC_DISPATCH(ComputeEmbeddingT)(RuntimeConfig(), prompts[0],
                                          model.config, inference,
                                          *model.weights, full, env);
  ASSERT_EQ(full.size(), kEmbeddingGemma2Dim);

  for (size_t dim : {size_t{128}, size_t{256}, size_t{512}, size_t{1024}}) {
    const size_t expected_dim = HWY_MIN(dim, kEmbeddingGemma2Dim);
    RuntimeConfig cfg;
    cfg.embed_dim = dim;

    hwy::AlignedVector<float> single;
    HWY_DYNAMIC_DISPATCH(ComputeEmbeddingT)(
        cfg, prompts[0], model.config, inference, *model.weights, single, env);
    ASSERT_EQ(single.size(), expected_dim);
    EXPECT_NEAR(NormSquared(single), 1.0, 1e-5);
    EXPECT_GT(Cosine(single, hwy::Span<const float>(full.data(), expected_dim)),
              0.9999)
        << "dim=" << dim;

    std::vector<hwy::AlignedVector<float>> batched;
    HWY_DYNAMIC_DISPATCH(ComputeEmbeddingsT)(
        cfg, hwy::Span<const PromptTokens>(prompts), model.config, inference,
        *model.weights, batched, env, /*workspace=*/nullptr);
    ASSERT_EQ(batched[0].size(), expected_dim);
    EXPECT_NEAR(NormSquared(batched[0]), 1.0, 1e-5);
    EXPECT_GT(Cosine(single, batched[0]), 0.9999) << "dim=" << dim;
  }
}

}  // namespace
}  // namespace gcpp
#endif  // HWY_ONCE
