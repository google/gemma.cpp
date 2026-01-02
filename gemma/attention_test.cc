#include <cstddef>
#include <cstring>  // strcmp
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

#include "gtest/gtest.h"
#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#include "gemma/activations.h"
#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "gemma/kv_cache.h"
#include "gemma/weights.h"
#include "ops/matmul.h"
#include "util/mat.h"
#include "util/threading_context.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#ifndef HWY_DISABLED_TARGETS
// These tests aren't designed to suss out instruction set specific problems.
// Disable most targets to keep the tests fast and simple and not have to
// worry about tolerances on floating point results.
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/attention_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"
#include "gemma/attention.h"
#include "gemma/configs.h"
#include "util/test_util.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

void FillRandom(MatPtrT<float>& mat, uint64_t seed) {
  hwy::RandomState rng(seed);
  for (size_t r = 0; r < mat.Rows(); ++r) {
    float* row = mat.Row(r);
    for (size_t c = 0; c < mat.Cols(); ++c) {
      row[c] = static_cast<float>(RandomGaussian(rng));
    }
  }
}

void AllocateAndFillRandom(MatPtr& mat, const Allocator& allocator,
                           std::vector<MatOwner>& mat_owners, uint64_t seed) {
  if (mat.IsEmpty()) return;
  if (mat.GetType() == Type::kUnknown) {
    mat.SetType(Type::kF32);
  }
  mat_owners.emplace_back();
  mat_owners.back().AllocateFor(mat, allocator, MatPadding::kPacked);
  MatPtrT<float> mat_f32(mat);
  FillRandom(mat_f32, seed);
}

struct TestState {
  TestState() : ctx({}), env(ctx) {}
  ThreadingContext ctx;
  std::vector<MatOwner> mat_owners;
  MatMulEnv env;
};

struct TestModelState {
  TestModelState(TestState& state)
      : config(Model::GEMMA2_2B, Type::kF32, PromptWrapping::GEMMA_PT),
        tensor_info_registry(config),
        layer_config(config.layer_configs[0]),
        layer(0, layer_config, tensor_info_registry) {
    config.att_cap = 1024.0f;
    AllocateAndFillRandom(layer.qkv_einsum_w, state.ctx.allocator,
                          state.mat_owners, 42);
    AllocateAndFillRandom(layer.attn_vec_einsum_w, state.ctx.allocator,
                          state.mat_owners, 43);
    AllocateAndFillRandom(layer.gating_einsum_w, state.ctx.allocator,
                          state.mat_owners, 44);
    AllocateAndFillRandom(layer.linear_w, state.ctx.allocator, state.mat_owners,
                          45);
    layer.Fixup(state.mat_owners, state.ctx);
  }

  ModelConfig config;
  TensorInfoRegistry tensor_info_registry;
  const LayerConfig& layer_config;
  LayerWeightsPtrs layer;
};

struct TestAttentionState {
  TestAttentionState(TestState& state, TestModelState& model_state,
                     size_t num_tokens, size_t qbatch_size,
                     AttentionImpl attention_impl)
      : num_tokens(num_tokens),
        qbatch_size(qbatch_size),
        batch_size(qbatch_size * num_tokens),
        runtime_config{.attention_impl = attention_impl},
        tokens(num_tokens),
        attention_storage_(model_state.config, model_state.layer_config,
                           batch_size, num_tokens, runtime_config,
                           state.ctx.allocator, row_ptrs_),
        attention(model_state.config, num_tokens, attention_storage_) {
    for (size_t i = 0; i < qbatch_size; ++i) {
      kv_caches.emplace_back(model_state.config, inference_args,
                             state.ctx.allocator);
    }
    activations.emplace(
        runtime_config, model_state.config, runtime_config.prefill_tbatch_size,
        kv_caches[0].SeqLen(), state.env.ctx, state.env.row_ptrs);
    // Tokens don't matter, since we fill in pre_att_rms_out before calling
    // GemmaAttention.
    std::iota(tokens.begin(), tokens.end(), 1);
    for (size_t i = 0; i < qbatch_size; ++i) {
      prompts.emplace_back(tokens);
    }
    all_queries.emplace(prompts,
                        hwy::Span<KVCache>(kv_caches.data(), kv_caches.size()));
    qbatch.emplace(/*start=*/0, /*max_size=*/qbatch_size, *all_queries);
    FillRandom(attention.pre_att_rms_out, 46);
  }

  const size_t num_tokens;
  const size_t qbatch_size;
  const size_t batch_size;
  InferenceArgs inference_args;
  RuntimeConfig runtime_config;
  std::vector<KVCache> kv_caches;
  std::optional<Activations> activations;
  std::vector<int> tokens;
  std::vector<PromptTokens> prompts;
  std::optional<AllQueries> all_queries;
  std::optional<QBatch> qbatch;
  std::vector<hwy::AlignedFreeUniquePtr<uint8_t*[]>> row_ptrs_;
  AttentionActivations attention_storage_;
  AttentionActivationsPtrs attention;
};

double GetTolerance() {
  const char* target_name = hwy::TargetName(HWY_TARGET);
  if (strncmp(target_name, "AVX2", 4) == 0) {
    return 2e-2;
  } else if (strncmp(target_name, "AVX3", 4) == 0) {
    return 3e-4;
  } else if (strncmp(target_name, "NEON", 4) == 0) {
    return 5e-3;
  } else {
    return 1e-7;
  }
}

template <size_t kNumTokens, size_t kQBatchSize, size_t kDims>
void CompareAttSumsWithGolden(
    const AttentionActivationsPtrs& attention,
    const float (&golden)[kNumTokens][kQBatchSize][kDims]) {
  ASSERT_EQ(attention.att_sums.Rows(), kNumTokens * kQBatchSize);
  ASSERT_LE(kDims, attention.att_sums.Cols());

  hwy::AlignedFreeUniquePtr<float[]> actual_row =
      hwy::AllocateAligned<float>(kDims);
  for (size_t token_idx = 0; token_idx < kNumTokens; ++token_idx) {
    for (size_t qi = 0; qi < kQBatchSize; ++qi) {
      const size_t i = token_idx * kQBatchSize + qi;
      for (size_t j = 0; j < kDims; ++j) {
        actual_row[j] = hwy::F32FromBF16(attention.att_sums.Row(i)[j]);
      }
      EXPECT_TRUE(hwy::CompareArraySimilar(
          golden[token_idx][qi], actual_row.get(), kDims, GetTolerance(),
          hwy::TargetName(HWY_TARGET), __FILE__, __LINE__))
          << "att_sums mismatch for token_idx=" << token_idx << " qi=" << qi;
    }
  }
}

template <size_t kNumTokens, size_t kQBatchSize, size_t kDims>
void CompareKVCacheWithGolden(
    const ModelConfig& config, hwy::Span<KVCache> kv_caches, const size_t layer,
    const size_t kv_head,
    const float (&k_golden)[kNumTokens][kQBatchSize][kDims],
    const float (&v_golden)[kNumTokens][kQBatchSize][kDims]) {
  const size_t qbatch_size = kv_caches.size();
  ASSERT_EQ(kQBatchSize, qbatch_size);
  const size_t start_offset = 0;
  const size_t qkv_dim = config.layer_configs[0].qkv_dim;

  hwy::AlignedFreeUniquePtr<float[]> actual_k_row =
      hwy::AllocateAligned<float>(kDims);
  hwy::AlignedFreeUniquePtr<float[]> actual_v_row =
      hwy::AllocateAligned<float>(kDims);

  const size_t cache_layer_size = config.layer_configs[layer].CacheLayerSize();
  const size_t head_offset = kv_head * qkv_dim * 2;
  const size_t kv_offset = layer * cache_layer_size + head_offset;

  for (size_t token_idx = 0; token_idx < kNumTokens; ++token_idx) {
    for (size_t qi = 0; qi < kQBatchSize; ++qi) {
      const float* cache_row =
          kv_caches[qi].kv_cache.Row(start_offset + token_idx);
      for (size_t j = 0; j < kDims; ++j) {
        actual_k_row[j] = cache_row[kv_offset + j];
        actual_v_row[j] = cache_row[kv_offset + qkv_dim + j];
      }
      EXPECT_TRUE(hwy::CompareArraySimilar(
          k_golden[token_idx][qi], actual_k_row.get(), kDims, GetTolerance(),
          hwy::TargetName(HWY_TARGET), __FILE__, __LINE__))
          << "K cache mismatch for token_idx=" << token_idx << " qi=" << qi
          << " kv_head=" << kv_head;
      EXPECT_TRUE(hwy::CompareArraySimilar(
          v_golden[token_idx][qi], actual_v_row.get(), kDims, GetTolerance(),
          hwy::TargetName(HWY_TARGET), __FILE__, __LINE__))
          << "V cache mismatch for token_idx=" << token_idx << " qi=" << qi
          << " kv_head=" << kv_head;
    }
  }
}

template <size_t kNumTokens, size_t kQBatchSize, size_t kDims>
void CompareQVecsWithGolden(
    const ModelConfig& config, const AttentionActivationsPtrs& attention,
    const size_t q_head,
    const float (&q_golden)[kNumTokens][kQBatchSize][kDims]) {
  const size_t qkv_dim = config.layer_configs[0].qkv_dim;

  hwy::AlignedFreeUniquePtr<float[]> actual_q_row =
      hwy::AllocateAligned<float>(kDims);

  const size_t head_offset = q_head * qkv_dim;
  for (size_t token_idx = 0; token_idx < kNumTokens; ++token_idx) {
    for (size_t qi = 0; qi < kQBatchSize; ++qi) {
      const size_t i = token_idx * kQBatchSize + qi;
      const float* q_row = attention.q.Row(i);
      for (size_t j = 0; j < kDims; ++j) {
        actual_q_row[j] = q_row[head_offset + j];
      }
      EXPECT_TRUE(hwy::CompareArraySimilar(
          q_golden[token_idx][qi], actual_q_row.get(), kDims, GetTolerance(),
          hwy::TargetName(HWY_TARGET), __FILE__, __LINE__))
          << "Q vec mismatch for token_idx=" << token_idx << " qi=" << qi
          << " q_head=" << q_head;
    }
  }
}

constexpr size_t kNumTokens = 10;
constexpr size_t kQBatchSize = 2;

// Only compare the first 10 dimensions of the attention output to keep the
// test manageable.  It's unlikely that the first few dimensions would have
// different behavior than the later dimensions.
const size_t kDimsToCompare = 17;  // greater than AVX-512 vector of floats

// Use a Coding Agent to update the golden values from test
// failure output (which includes the new values).
// Instructions to the Coding Agent: You must never update the
// EMU128 tolerance. Always use the value from the EMU128 test to update the
// Goldens.  If necessary, add relaxed tolerance for other instruction sets.

// Layer 0
const float kGoldenAttSums[kNumTokens][kQBatchSize][kDimsToCompare] = {
    {{46.5, 56.5, 10.0625, 65.5, -2.109375, 135, 15.8125, 51, -100, 52.5,
      26.875, 63, 3.34375, -67.5, 31.125, -190, 125},
     {-30.375, -17.875, 51.75, -78, -84, 6.40625, 15.375, 70, -22.875, 20.125,
      -14.9375, -109.5, 76, 9.25, -142, 29.5, -105}},
    {{-32.75, 38.25, 78.5, 107.5, 20.25, 197, -136, 42.5, -84, 25.625, 4.96875,
      128, 27.25, -161, 19.125, -58, 97.5},
     {-18.5, -18, 135, -13.4375, -6.625, -45.75, 29.625, 93, 18.625, 75.5,
      102.5, -184, 52.75, 83.5, -71, 46.5, -52}},
    {{-16.375, -61.5, -58.25, -27.375, -28, 71, -109.5, 60.25, 3.125, -29.125,
      6.90625, 150, 144, -155, -47.25, -98.5, 3.5625},
     {-19, -16.75, 129, 0.59765625, -82, 123.5, 60.75, -36.75, -77, 26.625, 51,
      -66.5, -0.84765625, -46.5, -152, -2.9375, -81}},
    {{3.984375, 83, -41.75, 39.5, -203, 110, -76, 131, 0.4609375, -44.5, -63.75,
      -46, -22, -19.375, -16.125, -148, 20.875},
     {-47, -19.5, 58, 81.5, 21.75, -30, -118, 44.25, -149, 22.5, 188, -66.5, 33,
      10.9375, -52.5, 23.25, 75}},
    {{64, -31, -89, -92.5, -11.1875, -54.75, -302, 3.453125, -108, 39.25,
      -34.75, 18, -52, 100, -186, -75.5, 50.75},
     {7.6875, -80, -40, 32.25, -30.25, 90, -41, 44.25, -140, -2.4375, 82.5,
      39.25, 65, 47.25, -89.5, -34.25, 137}},
    {{39.75, 17.875, 115, 38.75, -44, 139, -53.25, -23.875, -13.0625, 38.5,
      32.5, 53.75, 109, 4.09375, 57.5, -20.5, 132},
     {143, 249, 5.09375, 0.83984375, 27.875, -5.84375, 30.25, -101.5, 65.5,
      13.5, 195, -10.0625, 97.5, 2.203125, -97.5, -100, -19.25}},
    {{-30.125, -169, -150, 58, -35.75, 22.75, 36.5, -32.25, -8.9375, 55.25,
      -117, 26.375, 39.5, 125, 66, 48.75, 20.75},
     {137, 5.25, 61.25, 37, -42.75, 240, 62, -164, 11.3125, 173, 174, 23.5,
      88.5, 48.5, -46.25, -36.75, 101.5}},
    {{-103, -47.5, 39, -48, -67.5, 121, -136, 99, 80, -47.5, 107.5, 48.75, 97.5,
      125, -53.5, -14.625, 262},
     {29.875, 7.34375, -36.75, -14.5, -27.5, 44.75, -67.5, -40.75, 71.5, 172,
      81, -27.25, -3.03125, 111, -167, 59, 176}},
    {{-37.25, 109.5, -26.125, -115.5, 108, 57.25, 1.3671875, 72, -122.5, 59.25,
      -52, -12.625, 43.25, 16.25, -41.75, 26.5, 70.5},
     {40.25, 53.25, -142, 78.5, 38, 4.3125, -27.75, -134, -85, 107.5, 2.5, 93.5,
      58.25, 173, -53.5, 25.125, 4.8125}},
    {{-8.4375, -35, -35.5, 131, -33.25, 106, 109.5, -92, -135, 80, 21.5,
      -17.125, 15.25, 143, -27, 103, 101},
     {-77, 40.75, -10.125, 33.25, -33, 104, -7.6875, 85.5, -40, 93, 61, 14.5625,
      8.125, -99.5, 13.6875, -11.6875, 33}},
};

// Layer 0, *K*V Head 0
const float kGoldenK[kNumTokens][kQBatchSize][kDimsToCompare] = {
    {{-4.51717567, 6.93118095, 6.48003578, 9.12825584, 2.38755274, 11.8121576,
      1.65376127, 5.04456615, -7.19549274, 2.57609844, 3.55331731, -3.48494458,
      -8.90498638, 9.66047478, -0.379868984, 6.37043715, -2.24351144},
     {0.152208567, 3.14520073, -8.35154343, 5.44226503, -6.74000502,
      -1.43484437, -4.72092056, -9.48932, -6.12409401, -1.55352509, -3.90701318,
      2.12124252, 3.93649936, -8.09877586, -3.30277514, -0.898857355,
      1.76684189}},
    {{4.378829, 5.05565643, -7.63948059, -5.74608946, 2.90109587, 0.155819178,
      4.56115055, 1.37885749, 1.48427355, -1.07145202, 2.82399392, -1.20864201,
      3.05434561, -2.65185618, -0.0731391, -8.2279253, 7.63228416},
     {-0.702698231, 1.49563932, 6.42149782, -6.68306589, 1.85317755,
      -7.70267582, 2.07357907, -7.60303402, -0.514724255, 0.308567047,
      5.99250412, -4.67359257, -3.49322176, -2.62086344, -3.18411255,
      2.04027057, -4.29057407}},
    {{-1.20844436, 4.14724302, 6.04515219, 8.7753458, -0.975198627, 0.564640105,
      5.39941597, 4.64036179, 0.366614938, 3.48258138, -0.470701456, 15.2267399,
      4.63302803, 9.12662697, -5.89148045, 2.25731587, 5.24449492},
     {4.57078934, -4.60315752, -3.3364439, 1.29875994, -3.40833569, -6.95262,
      -6.39040232, -6.60212612, 6.63269806, -0.815209687, -5.0346446,
      -4.13564968, 8.25674057, -6.0910182, -8.21130085, -8.91020393,
      10.6188011}},
    {{0.602011144, 2.22505236, 3.62411499, -4.07026958, 12.8036356, 3.76139069,
      6.99502087, 7.02500725, -2.51568675, 4.2489934, 0.00210827589,
      -1.43267739, -2.10394144, -0.0506809056, -1.54883039, 4.3740139,
      -1.61869526},
     {-6.37204599, -3.34989691, 2.10935307, 4.23634195, 5.79134035, 13.502944,
      -2.19158888, -1.55771351, -1.22244942, 3.36499929, -2.11375904,
      -4.5448761, 1.0611912, -2.47849369, -0.212709218, 0.363292456,
      7.91467094}},
    {{-8.85739231, -4.08585882, -0.618261, 6.52911091, 5.14922285, 7.6869874,
      0.750387549, -0.812200725, 2.7509625, 6.29693508, -1.77248931, 5.68896484,
      -6.9369607, -4.61359406, 0.184977874, -1.27769828, -2.1619854},
     {-8.2555, 2.84032059, -1.03791106, 2.07648611, -4.94546843, 1.76888537,
      -1.75901175, 11.2628574, 1.41086221, -3.58669901, -2.85925198, 2.29133463,
      1.55509436, -0.0553357825, -10.0363655, 1.94261, -2.95691729}},
    {{0.919141412, 1.97533965, -11.3202848, -3.3137629, -4.7161727, 5.07012081,
      1.76256621, 8.20588207, 6.05700159, -3.89765406, -1.13639557, -1.32326794,
      -3.01544905, -0.585309267, 2.60637712, 2.83708405, -3.39202118},
     {9.11918, 2.11261511, -5.87290621, 11.6033278, -4.66597795, -7.13774204,
      -9.10563755, -2.48294282, 3.35282946, -3.75122213, 0.404774547,
      -9.11625195, 4.85711479, 1.43184578, 1.47673059, -4.75093, -3.45323014}},
    {{4.17705393, -4.95192289, -10.5068378, 3.90004015, -3.51306129, 5.38068056,
      0.901511431, 11.222868, 2.67285442, 9.18779, 5.61346769, 3.06534624,
      -3.78898215, 0.767340839, 15.8207836, -4.14079094, -4.63177109},
     {3.61795235, -7.00262165, 2.08284521, -6.70515728, 1.93205631, 2.84467721,
      3.94591737, -6.18882942, -1.78465152, -9.39100933, -10.8780289,
      6.32468653, 6.53142738, -3.30765963, 2.89132166, 4.53347206, 1.89792418}},
    {{-0.361971855, -1.57735932, 5.07296801, -1.55669761, -1.44996238,
      7.29838896, 5.23075104, -0.512441278, -3.59834242, 2.38584423, 6.48518324,
      -1.48220074, -2.4264791, 10.7237988, 5.64735842, 5.6251297, -7.04244423},
     {-0.795628309, 7.30230665, -1.71035647, -16.6999454, 3.05102086,
      -4.9243927, 4.28508186, -0.694577456, 6.58464718, 4.40330124, 3.3250041,
      1.90579033, -6.29048729, 2.55308104, -4.9746747, -0.681708, -5.98152351}},
    {{2.57555652, -3.5651083, 0.784440041, -4.7043705, 2.37520599, -3.62385964,
      -3.48913693, -7.28049421, -5.48726082, 1.95519221, 7.25192928, 3.07074118,
      -11.9897156, 5.92244673, 5.07564354, 0.162699938, -6.00809956},
     {5.56260443, -5.7683115, 1.26402235, -17.507719, 4.18873024, -3.20694613,
      -4.42512083, 1.78077614, -3.25167561, 0.864362717, 0.474019766,
      -7.92327404, -2.27795148, -0.436354101, -3.15722394, 0.415780187,
      2.60931611}},
    {{-9.43858051, 0.391518891, -2.74012518, 4.9842453, 7.48263216, -16.3434925,
      -4.75156116, -1.99114823, 3.99918842, -5.95400572, 10.8700314, 1.07596064,
      0.30389142, 8.39548779, -5.11913681, 5.45641088, -5.63240337},
     {-1.22347319, 9.57339382, -1.31736016, -5.02770805, -4.81617355,
      -1.96618557, -0.456317186, 12.6451035, -1.50221801, 6.7991147,
      -5.97842169, 1.85410941, -8.44729, 0.378282309, 0.0442156792, 17.6773052,
      -7.43491}},
};

// Layer 0, K*V* Head 0
const float kGoldenV[kNumTokens][kQBatchSize][kDimsToCompare] = {
    {{2.77553034, -7.67514181, -1.60433948, 4.67795134, -1.75084186, 8.57896423,
      -1.15065813, -3.75088787, -4.7442131, -1.68890858, -10.0202332,
      -4.20167446, 9.36844635, 13.7364845, 11.5634, 2.95288706, 2.89380026},
     {-4.79950905, -1.66658688, 4.14471292, -4.95649052, -5.4200325, 3.52626801,
      -10.9432049, 0.338347554, -1.53204226, 0.473476171, -0.58271, 1.42195463,
      0.301399827, -4.40214968, -2.12298298, 9.27825642, -0.690600872}},
    {{-10.6566734, 4.12785721, 4.54053593, -1.39667869, -1.55028772, 0.20508635,
      -0.00620913506, 2.93214, -0.788117647, 2.78032446, -2.68898249, 9.5985508,
      -10.6630878, -11.9006901, 0.851743698, 0.581826329, 5.21927929},
     {-0.322291255, 2.63848567, -2.30808377, -13.0153809, 2.74378228,
      3.21460533, 0.688529968, 2.37544608, 6.06825066, 4.57566404, 1.17124248,
      -7.96587658, -2.65279341, 4.75271225, -4.09937954, -10.3570251,
      3.30500841}},
    {{-3.34342527, 6.03099537, 6.335958, 0.993818045, 0.905343294, 6.93058586,
      3.9635396, 10.8044815, 7.8620863, -10.1157322, -3.92666101, -0.183003783,
      -5.27309418, -1.45110512, -8.96734, -2.63866425, 2.19913912},
     {16.416317, -1.62025332, 2.3161006, 3.32571959, -1.79581594, -10.2925539,
      -5.86338425, -6.36642933, 9.18872166, 5.95524168, 6.38640785, 8.23832,
      -6.57342291, -14.2017632, 1.10925388, 4.27255058, -2.65661311}},
    {{6.58254147, -6.96165133, -4.97437, -2.33467388, 5.83671236, -0.794236898,
      -2.03117108, -3.93387103, -5.96872902, 5.83316422, 3.01795, -4.05260706,
      -4.39556885, 3.24399853, 10.1573639, 4.71967888, 0.274738848},
     {7.13243389, -8.04649162, 2.53055143, 2.0771277, -0.667295456, -13.0285645,
      0.960428238, -2.11983275, 8.18105602, -6.72609901, -5.46944714,
      0.204244614, 0.0900330544, 8.86620903, 4.63697529, 3.19756651,
      2.99392676}},
    {{9.52539158, -4.3840766, -6.94514465, -2.75913763, -10.8364506,
      -3.95606327, 2.43603897, -5.78482246, -0.801304817, 8.23436832,
      -7.11484337, 2.53943753, -0.652261257, 9.77392, 3.53345847, -9.62052822,
      16.0471916},
     {6.89768124, 2.36394405, -2.08569574, -0.682706833, 3.38872, -6.28313875,
      4.79594612, 4.93417454, -6.40791416, -10.7355442, -5.66094208, 2.44881392,
      1.99794042, -9.19855404, -4.02383137, -3.63013959, -5.65853405}},
    {{1.64614546, -3.93421197, -0.48935914, 5.48284435, -7.69781828, 11.8203125,
      1.81672478, -1.42535269, -5.26496315, -5.31612349, -4.19499826,
      7.06049395, 0.18029356, -0.0519902706, 10.317358, 2.19345617, 3.5296216},
     {7.52353811, 3.56836724, 0.414305687, 0.340799928, 2.44263697, 7.52111912,
      0.246491909, -11.1172791, -3.82061529, 3.24794388, 0.751524329,
      3.14019632, 6.33881855, -0.169233799, 7.82640171, 1.5389179, 8.15851307}},
    {{-2.48950672, -8.55112743, 8.04663277, -5.77116871, -0.637019753,
      -7.65882111, -7.49037457, 3.8041625, -3.57038307, 9.37715435, -6.42604256,
      1.62610793, -1.54000568, 2.52110147, 5.30775261, -4.10454893,
      -4.96251774},
     {-2.95554614, -5.18210888, 1.00015664, -4.03864431, -7.14954519,
      5.99929142, 5.86350155, 2.03810191, -4.23009968, 9.39885902, -5.68198299,
      2.72845244, 11.7133255, 0.838779449, -13.2235403, 2.94607735,
      -2.7902379}},
    {{2.86876941, -0.836064458, -0.374509573, -0.277966499, 3.20654631,
      -3.68510771, -7.76134634, 2.23905277, -8.35530376, 5.25071716,
      -1.38490796, -2.93542218, 0.509032726, -3.57361269, -2.82580233,
      -4.49954033, 2.91235542},
     {-4.37938213, 4.78577232, 2.03453469, 5.48564529, -1.05589461, -1.65940428,
      4.0130887, 5.26074123, 4.67537832, 0.791350365, 6.3880868, 2.50402451,
      7.6603322, -3.16343474, -2.71949649, 4.61576128, 1.3817997}},
    {{0.289200783, 7.06031752, -1.15099299, -5.29136801, -1.343642, -8.36283112,
      4.13158274, -1.93137062, 3.16199875, 2.21854591, 2.18270063, 0.77002573,
      6.90393353, -0.644045949, -5.62211609, -1.09085155, 1.07821059},
     {-3.04716778, -2.52233481, -5.99031925, 2.80152273, 0.340899587,
      0.667474508, -2.39674735, 8.83768654, -5.45613146, -1.55994594, -2.216362,
      1.49354, -4.27255821, -9.05310917, 5.90691471, -1.29772806, -8.50278}},
    {{-3.1383903, -7.71573353, 3.38072681, 6.07642221, -2.39587545, -7.84178352,
      -1.60108304, -8.6121521, -5.151721, 4.17612457, -2.86532378, 1.64645958,
      -0.37970829, -4.34561253, -0.454322815, 0.331385136, -5.74550819},
     {4.77026033, -5.51171303, -7.38155365, -5.38462543, 2.95842505, 5.18372536,
      0.521988213, 7.23966122, -4.90852165, 7.18465281, 2.99289083, 10.0519466,
      -2.09695673, 7.34368706, -2.40495348, 3.61603308, 0.131510735}},
};

// Layer 0, QHead 0
const float kGoldenQ[kNumTokens][kQBatchSize][kDimsToCompare] = {
    {{-0.574401975, 0.370210886, -0.426894158, -0.543187439, -0.0266762674,
      -0.177960411, -0.00839618221, 0.411925405, 0.536462784, 0.528389931,
      -0.499812007, -0.123897657, -0.0170236826, 0.266041577, -0.0781469196,
      -0.44081074, 0.185976267},
     {0.270543516, -0.109283224, -0.58602041, -0.358663559, -0.393124342,
      -0.0895933211, -0.632167816, 0.386703, 0.314152211, 0.0554139167,
      0.0241559595, -0.194484815, 0.143893063, 0.103837147, -0.384245932,
      -0.00418212265, 0.385817379}},
    {{-0.0331106335, -0.100827977, 0.322449774, 0.225943685, -0.384854138,
      -0.208085626, 0.0206767023, 0.287796348, -0.139513299, 0.255447835,
      -0.0845065042, -0.0619940236, 0.477489054, 0.517492294, -0.0172665715,
      -0.0302075297, 0.365989387},
     {-0.0266781822, -0.453293771, 0.560033202, 0.105156079, -0.35259968,
      0.711447716, -0.253611088, 0.0487165749, -0.086192511, -0.0338740349,
      -0.655441046, 0.00413730741, -0.510472536, -0.0748229772, -0.29113093,
      -0.0432077348, 0.09223634}},
    {{-0.321974993, -0.466039479, 0.207254037, -0.126807183, -0.192775592,
      -0.0953654051, 0.209789664, 0.405356169, -0.00627984107, -0.0590961352,
      0.0907663852, -0.190793216, -0.730463982, 0.340142608, -0.295675993,
      -0.165913597, -0.233714506},
     {-0.345578939, 0.394073665, 0.299743414, -0.0075177839, -0.288939595,
      0.127782941, -0.207550645, 0.0655022636, -0.705084503, -0.241842598,
      0.333820701, 0.217911497, 0.29735288, 0.0147881694, -0.152306199,
      -0.589594781, -0.373093933}},
    {{0.216089666, 0.0918798149, 0.0560657382, -0.157523662, -0.00141695142,
      0.51770103, 0.596379519, -0.271057904, 0.241035417, -0.275827706,
      0.112851456, 0.026878573, -0.579843462, -0.5116328, 0.192026839,
      0.125176072, 0.34234497},
     {-0.0744233653, 0.180814236, 0.170143247, -0.337861449, -0.175804421,
      0.213403732, -0.173699334, 0.109528325, -0.385727316, 0.109683953,
      0.475667775, 0.253016889, 0.477347463, 0.111096457, 0.394625545,
      0.0172286481, -0.357992649}},
    {{-0.350524545, -0.142550975, -0.212269634, -0.0589753427, -0.434021264,
      0.384472728, 0.445421219, -0.635599554, -0.246593416, 0.120986834,
      0.623568773, -0.161932915, -0.702406883, 0.44038102, 0.268234134,
      0.480264157, 0.103595078},
     {-0.227436215, 0.357608706, -0.25339672, -0.0683218762, -0.179259315,
      0.23657614, 0.559984326, 0.165754288, -0.0402980596, -0.101906747,
      -0.278261065, -0.16327399, 0.235923961, -0.428657919, -0.290629387,
      0.579215467, -0.0717103705}},
    {{-0.246389642, -0.266164362, -0.0967710763, -0.4011603, 0.242542207,
      0.0869855583, 0.20158039, 0.207793877, -0.0875666738, -0.242263764,
      -0.0462955758, -0.617374003, 0.454443514, 0.207072973, -0.0235372931,
      -0.0193868056, -0.660622239},
     {0.703284621, 0.0382430181, 0.43997851, -0.858277559, 0.342218578,
      0.414044619, 0.403636098, -0.579880178, -1.12243, -0.112913512,
      0.629238605, -0.0285760984, -0.152203664, -0.088969171, -0.0681343,
      0.476349175, 0.283238202}},
    {{0.138267457, 0.483219147, 0.230450034, -0.568304598, 0.204461277,
      -0.286731184, -0.416590065, -0.483460307, -0.561008453, 0.395195067,
      0.104367018, -0.196090236, -0.324770749, -0.0881370157, -0.626873195,
      0.0936089084, 0.262185335},
     {0.282603383, 0.0723766163, -0.206548154, 0.561849833, 0.482716829,
      0.135281503, -0.438841999, 0.472577304, -0.346201897, -0.0211652666,
      -0.0905084163, -0.168639392, -0.154975936, -0.303443581, -0.41771856,
      0.400717318, 0.426146686}},
    {{-0.0537007451, -0.227346331, -0.2871463, 0.247746795, -0.0975416005,
      -0.0123391449, 0.0612513907, -0.374673814, 0.283457696, 0.40945363,
      0.137944818, -0.0119741419, 0.775918365, -0.308365196, 0.230615795,
      -0.440364927, 0.218536288},
     {0.0688965544, -0.149037778, -0.246169299, 0.0599289536, -0.456733435,
      0.0808929354, 0.115154952, 0.0997388735, -0.408117741, 0.576600909,
      -0.193775773, 0.0340575948, -0.29254055, 0.695465446, 0.373336494,
      0.421431482, 0.00197479129}},
    {{0.402076721, -0.118151993, 0.542394996, 0.0382412486, -0.614983976,
      0.28617692, 0.318540633, -0.299300969, -0.177486539, 0.394140214,
      0.0644133314, -0.0321308076, 0.671587527, -0.0173831787, -0.219400048,
      -0.340277791, 0.5130288},
     {0.105372488, -0.145784974, 0.0695323348, -0.106080391, -0.755512118,
      0.975362539, -0.15056029, 0.58882606, -0.059625227, -0.810613,
      -0.321623206, 0.193939567, 0.0340242684, -0.626081824, 0.109950632,
      -0.141072854, 0.0177994221}},
    {{0.243249148, 0.0904035419, -0.472183734, -0.176162, 0.314925164,
      -0.191137731, 0.492265761, -0.0120046511, 0.824757636, 0.298175,
      0.148151726, -0.0197859108, -0.64297086, 0.432318538, -0.555079758,
      0.101636633, 0.155741245},
     {0.0523641109, 0.224086404, 0.0143201668, 0.0090854, 0.304901183,
      -0.391372293, 0.267655343, 0.117368169, 0.645064473, 0.336050332,
      -0.282133281, -0.231817603, 0.376230389, -0.575031936, -0.628365576,
      0.484799922, 0.0824087635}},
};

void RunAttentionTest(AttentionImpl attention_impl) {
  TestState state;
  TestModelState model_state(state);
  TestAttentionState attention_state(state, model_state, kNumTokens,
                                     kQBatchSize, attention_impl);

  GemmaAttention(attention_state.tokens.size(), 0, model_state.layer,
                 attention_state.attention, *attention_state.qbatch, state.env,
                 AttentionImplToFlags(attention_impl, HWY_NATIVE_DOT_BF16));

  CompareAttSumsWithGolden(attention_state.attention, kGoldenAttSums);
  CompareKVCacheWithGolden(model_state.config,
                           hwy::Span<KVCache>(attention_state.kv_caches.data(),
                                              attention_state.kv_caches.size()),
                           /*layer=*/0, /*kv_head=*/0, kGoldenK, kGoldenV);
  CompareQVecsWithGolden(model_state.config, attention_state.attention,
                         /*q_head=*/0, kGoldenQ);
}

void TestGemmaAttentionOld() { RunAttentionTest(AttentionImpl::kOld); }

void TestGemmaAttentionFlash() { RunAttentionTest(AttentionImpl::kFlash); }

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gcpp {
HWY_BEFORE_TEST(AttentionTest);
HWY_EXPORT_AND_TEST_P(AttentionTest, TestGemmaAttentionOld);
HWY_EXPORT_AND_TEST_P(AttentionTest, TestGemmaAttentionFlash);
HWY_AFTER_TEST();

}  // namespace gcpp

#endif
