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
  hwy::RandomState rng0(seed);
  for (size_t r = 0; r < mat.Rows(); ++r) {
    hwy::RandomState rng(rng0());
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
    AllocateAndFillRandom(layer.linear_w, state.ctx.allocator,
                          state.mat_owners, 45);
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
        tokens(num_tokens),
        attention_storage_(model_state.config, model_state.layer_config,
                           batch_size, num_tokens, attention_impl,
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
  if (strcmp(target_name, "EMU128") == 0) {
    return 1e-2;  // Flash and Old don't agree sometimes!
  } else if (strncmp(target_name, "AVX2", 4) == 0) {
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
    {{-107, 32.25, 70.5, -70, -130, -41.5, 142, 98.5, -7.03125, 39.75, -51.5,
      43.25, 18.125, 152, 61, 56, 27.25},
     {-132, -53.5, -48.75, -52.5, 1.015625, -24.5, 226, -53.75, -26.75,
      1.6484375, -12.75, 68, 107, 92.5, 46.75, -36.25, 118}},
    {{-41.5, 50, 3.953125, -37.75, 158, -22.25, 35, 27, -59.5, 67, -12.5625,
      -23, -9.8125, 58.25, 54, -123, -39.75},
     {55.75, -0.859375, -148, 36.5, 48.75, 33, 205, -25.375, 110.5, 63.5, 88.5,
      50.25, 106, 156, 18.125, -20.5, 70}},
    {{-21, 35, 141, -1.4609375, 14.5, -37.25, 104.5, 86, -56.5, 78, 119, 19.625,
      -19.875, 227, 58, 19, 38},
     {-71.5, -59, -116.5, 13.5625, -71, -94.5, 67.5, -54.5, -28.875, 87, 161,
      66.5, 131, 86.5, 104, -79.5, 1.8984375}},
    {{-55.5, 65, -83, 17.75, 41.25, -16.125, -175, -22.875, 62, -46.75, 182,
      16.25, 45.5, 84.5, -32.75, 40.25, 108.5},
     {-15.75, -59.75, 99.5, -43.5, 35.5, 76.5, 73, 173, -37.25, 70, -31, 103.5,
      -27.375, -9, 71, -62, -174}},
    {{-90.5, -8.25, 75.5, -117, 68, 11.4375, -90, 47.75, -48.25, 8.9375, 25.5,
      79.5, -39.25, 102, 66, -63.5, -44.5},
     {63, 80.5, -59, 81.5, -71, 190, 67.5, -46.75, 10.5625, 100, 123.5, 101.5,
      -50.75, -24.25, 80.5, 31.75, -8.3125}},
    {{-148, 20.125, 24, 110, -148, 2.5625, -117.5, 1.609375, -67, -91.5, 105,
      151, 203, -23.25, 64.5, 21.625, -51.5},
     {-49.25, 87, -97.5, 21.625, -231, 42.5, 117.5, -70.5, 4.71875, 118, 68,
      69.5, 15.4375, 88.5, -67.5, -17.625, -13.5625}},
    {{-92, 38.5, -21.875, 165, -32.25, -108.5, -143, 36.75, 49.5, 11.1875, 70,
      17.75, -16.125, 151, -191, -22.625, 49},
     {-36.75, -24.25, 42.75, -19.125, -118, -220, 169, -97, -75.5, 19.25,
      -41.25, 107, -24.75, 157, 99, 54, -129}},
    {{-35.5, -34, -34.75, -63.25, -70, -22.375, -66, 232, -74, -54, 125, -67.5,
      -109.5, 119, 101, -98, 22.5},
     {-17.875, -88, 0.8671875, 55, -42, -53.25, 114, 26.125, 87.5, 27.375,
      -27.75, 18.125, 75, 26.25, -35.75, 20, 193}},
    {{52, 46.25, 28.875, 66.5, 119, -7.59375, -40.5, 135, -6.4375, 57.75, 97.5,
      30.375, -153, -17.5, 2.359375, -82.5, -39.25},
     {8.75, 46, -66, 4.6875, -111, 196, -50, -106.5, -71.5, 43.5, 12.375,
      -50.75, 71, 52.75, -17.625, -78.5, -172}},
    {{-96, -22.5, -4.96875, -4.21875, -77, -67.5, -28, -12, 14.3125, -44, 72.5,
      -43.5, 34, 29, 67, 10.625, 40.25},
     {-110.5, -1.2734375, 101, 78.5, -116.5, -125.5, 172, 49, 1.078125, -50.25,
      -33.5, -3.59375, -19.625, -13.625, -14.875, 39, 115}},
};

// Layer 0, *K*V Head 0
const float kGoldenK[kNumTokens][kQBatchSize][kDimsToCompare] = {
    {{15.2907486, -9.24563789, -1.87377763, -1.6078732, -2.52019691, 3.78340316,
      1.56531, -0.419910669, 0.0457177162, 1.7699399, 0.973267794, -11.2898827,
      3.79524374, 3.8804853, 8.05621147, 1.64328313, -7.22062826},
     {-1.33305621, -1.20374441, 5.16571712, -0.245627165, 1.00112915,
      -3.94195318, -1.53855979, -2.24500442, 4.81447029, -8.42467785, 2.6451962,
      -5.42961216, -1.04181266, -6.57116222, -2.43039203, -9.50760841,
      3.21791911}},
    {{1.35395038, -0.375163317, 2.66030908, -3.00428605, 6.10236216,
      -10.4410543, -1.12052476, 5.85763407, -0.0452268124, -2.42987514,
      6.85442591, 1.17080283, -3.25781202, 6.65555668, -5.64402437, 4.7492609,
      9.98779583},
     {-11.0549402, -10.9070759, -9.21442795, 8.93494606, -0.663663864,
      -0.127197742, -0.418648839, -0.12933588, 10.0827341, 13.9710932,
      -7.22307491, -2.81767416, 2.61202765, -10.5902529, -1.11884749,
      -0.00246357918, 2.00061131}},
    {{-4.12993002, 3.06688476, -3.34329081, 0.188707948, 2.42000532,
      -0.339237094, 5.88325405, -2.4620254, 3.93701172, -0.949787855,
      -3.56888604, -4.52016211, -6.81539917, 3.83921003, -1.64406776,
      -4.28217793, 4.09804487},
     {9.04821014, -6.12610292, -3.91204882, 2.46237516, 2.26863813,
      -1.05252552, 0.674160719, -0.543522477, 0.315010548, -6.30216789,
      -7.87714481, 2.71428013, 6.90030003, 8.48286819, -3.15425754, 5.1051693,
      2.59031558}},
    {{3.85839581, -4.56797647, -5.07595825, -0.837815881, -3.84364843,
      -5.15372133, -0.232586145, 7.362432, 0.107376553, 2.64676356, 0.902205765,
      -7.68729115, -1.04463434, -7.04473209, -2.12464309, -2.62663937,
      2.3179245},
     {-10.2786751, -7.18292856, -1.0349617, 5.58713627, -4.24747801,
      -0.505107284, -3.58366871, -5.82409763, 1.5151974, 3.69901705,
      0.225643635, -1.91915131, -9.39223576, -2.99991035, 3.88195848,
      -0.975675821, 9.08020401}},
    {{0.713129759, 0.831702948, 4.85394859, -1.3690424, 1.06993294, 1.77343011,
      4.4732461, 2.77546239, 1.76154709, -10.2734528, 4.89345741, 1.56878746,
      0.557243943, 2.686064, -0.480260491, -1.30898976, 7.84716129},
     {-0.48303628, 1.8997345, 9.41060734, -1.07365155, 16.2980633, 0.842305303,
      1.46111321, -5.46785688, 9.73378944, 1.76110291, -0.617839932,
      -0.699874997, -6.00970268, -2.25671721, -4.34198618, 10.7963381,
      -1.31340837}},
    {{0.839338958, -0.991259813, 2.44353271, 5.51663303, -4.78505135,
      -4.73743773, -6.66635752, -12.1987858, 0.619547904, 1.12478662,
      -2.90830898, 3.32718873, -5.1365242, 0.0782394409, 6.71992254,
      -1.30097711, -10.1333361},
     {-4.03514862, -1.19420063, -0.467277795, -7.10551929, -2.79278111,
      -5.32330513, 4.69234657, 1.59959948, -10.0435543, -0.308479786,
      2.11825275, -3.33224726, 1.42422175, 10.0299196, 3.14650702, -4.50784397,
      1.13975036}},
    {{-7.77441454, 6.60742712, -3.2969532, 4.07419205, 0.553794742,
      -0.980163574, -0.80379802, 5.47732353, -2.80931783, -7.27533054,
      1.96269298, 0.103360891, 11.9011269, -1.67654371, -4.00289297,
      3.95645094, 6.72452736},
     {-2.08075809, -0.622131109, 6.95990324, -10.1613321, -6.5728159,
      -1.83433318, -7.4444685, -1.17990899, 0.949428558, -7.08294106,
      6.8835268, 0.593178153, -1.11343932, 11.1121941, -3.24285984,
      5.95768023, 1.86565471}},
    {{-3.98357534, -5.07885265, 2.99530745, 2.21132183, -5.06690884,
      7.19524574, -8.69441986, -5.43023586, 3.60415602, 6.77679777,
      7.39095974, -11.7769651, -1.51282454, 10.512928, 8.33419418,
      -4.89421844, 0.684614658},
     {3.33132195, -2.80186033, 7.80674505, -3.47060919, 1.73025632,
      -3.24225068, -5.88360023, 5.90776682, -1.00811982, 9.21799469,
      -0.796300411, -6.04880476, -2.39337349, 1.74686813, 7.84074497,
      -1.17035842, -3.03220415}},
    {{-2.54733372, 7.53344202, 4.13780975, -9.24725914, -8.49006271,
      -6.72345352, -1.11408019, -0.0324454904, -2.94914579, 3.31400394,
      -2.5422883, 4.42092514, -2.48425007, -1.06791162, 0.47528255,
      -5.99708033, -1.02899408},
     {1.68688703, 4.75695753, 5.33531904, -2.97416735, -2.4486413,
      -8.94855595, -2.54400206, -0.263463914, 7.70630169, -2.4543817,
      0.341010422, -6.5072546, 6.57980537, -8.83047295, -5.90621185,
      -1.36317229, -8.00853157}},
    {{-5.81304836, 7.35501003, -1.7505573, 4.28803205, -0.106060743,
      7.27207994, 3.63292217, 3.05916095, 2.7457571, 0.898360848, 6.84973812,
      0.0843296051, 6.84243679, 9.31108475, -0.37638694, -3.97468519,
      0.128682166},
     {-0.340807438, -3.57352829, 2.74731278, -8.07462502, -2.55854392,
      -0.0783569366, 9.2572813, -2.07895994, -1.34830523, 0.524608493, 1.701473,
      -6.40128899, -2.29863024, -0.430005044, -1.20804024, 7.26425266,
      8.14774704}},
};

// Layer 0, K*V* Head 0
const float kGoldenV[kNumTokens][kQBatchSize][kDimsToCompare] = {
    {{-6.57186985, 4.65591288, -2.99893808, -11.1538782, 0.244000077,
      -10.2325764, 0.103694201, 0.521099567, -3.99905825, -1.62405348,
      -4.68134117, 6.1718998, -1.34258807, 0.629202843, 2.19743776,
      -0.994996071, 11.087513},
     {6.30781364, -0.809881091, -2.28015828, 7.74938059, -7.73279285,
      -2.66831946, -2.19984651, 0.331729531, -2.91752172, -3.65055728,
      5.1676836, -7.56936884, 1.81388354, -4.26828051, -2.01722169,
      -0.324608445, 7.27558804}},
    {{9.26140213, -3.20177221, 0.539388776, 5.40602064, -0.743577957,
      0.394759417, 9.85691643, 2.08870316, 0.901947498, -1.50783658, 2.25597,
      -1.95216775, 0.435392141, -0.702769041, -4.18087959, -4.37605,
      7.78122902},
     {-2.62402225, 1.53574657, -8.48229218, -6.17764902, -8.80739498,
      3.71258497, -0.00548219681, 1.16554821, 4.96417856, 12.0105095,
      -9.01848125, 0.977133036, 2.64647341, 6.30225754, 1.42601275,
      -9.98334408, 0.879288554}},
    {{5.97513628, -6.88194704, -1.16571558, -4.31768417, -1.14049578,
      -2.82398677, -6.27558422, 2.18296051, 2.75785732, 5.18285942, -4.07532883,
      -7.07251263, 1.9271419, -8.29465675, -6.54444408, 7.7866087, -2.06813526},
     {-5.63859415, 5.49219513, 1.35068834, 3.48846531, 6.94235802,
      -4.82062531, 2.47111416, -3.67039084, -2.86166239, 1.72953558,
      -6.94025803, 3.77951097, 3.43053484, -0.421885848, -5.10398674,
      7.37130451, 4.5244031}},
    {{-2.85907269, -7.74109554, -2.99573851, -5.80393362, -4.41116858,
      -2.96661329, 0.529096365, 4.8533392, -0.586824358, 10.2085228,
      -7.89174175, 11.6699429, 3.2624352, -5.73311234, -7.5428834, -1.59121943,
      -1.98875427},
     {-2.01318312, 7.78195047, 4.17572403, -0.517796278, 1.73962998,
      6.3888917, -1.03050208, -4.90732288, -2.38260913, -0.94410181,
      -2.34225774, -5.66976643, 0.0630166531, 5.22525358, 9.27637863,
      -6.84555054, -3.40093827}},
    {{0.713163614, -5.89050484, -1.6664927, -1.1432848, 10.4444027, 8.94331741,
      -9.69797707, 4.2944026, -6.69290638, 4.72696638, -3.301085, 5.89265633,
      0.634907007, -6.85523701, 7.27885437, -12.9960146, 1.07775009},
     {-2.30551481, -0.188415289, 5.51380777, -2.09227371, -6.09918642,
      -2.92235994, -10.7518473, 6.63548946, 6.40411043, 0.495648265,
      -0.0361406803, -2.30997944, 2.38069057, -2.46818423, 0.144803047,
      4.35358715, -1.88418579}},
    {{0.667566538, -8.06617641, -2.83349943, -8.64362812, -5.26301479,
      -4.63245106, -3.16837788, -1.80521441, -4.16031981, -4.48559904,
      -8.40764809, 5.11661053, -3.19849682, 2.49863052, -0.809394836,
      8.11068916, -2.00028992},
     {-3.50956917, 3.46693277, 3.71822405, 6.78018379, 0.00734519958,
      7.60286093, 9.44774818, -0.519026041, -9.28906822, 7.12584591,
      8.12778854, 2.7033093, -2.83234954, -2.78084874, 3.65403628,
      0.215320587, -1.59024906}},
    {{10.8861427, 6.59166813, 9.7520752, -2.61776686, -0.63697052, 0.804175496,
      6.64749336, 2.6748116, 11.0225735, -5.11313915, -0.60951817, 1.94157505,
      0.709332824, -1.59864545, 9.76169205, -5.00956106, -1.29182816},
     {-6.48537397, -0.724315166, 3.55528641, 10.9164925, -10.999507,
      -1.26528633, -1.44988942, -5.16796589, -0.320435524, 1.20271659,
      -5.23793507, -1.60932314, 0.000490188599, -2.50121546, -5.03053236,
      7.05981207, 0.729410648}},
    {{0.193905115, -10.848567, 5.42073679, -3.42887449, -1.63425016,
      -7.5447526, -2.04208255, -3.28060675, -0.136736155, 1.27700531,
      0.377272248, -1.60267282, -5.29419708, -2.20173168, 11.6071215,
      2.40224266, -4.04324436},
     {-3.08099103, 7.33237839, -11.2906342, 0.958051205, -4.04783964,
      -1.28419411, 4.54195166, 5.41813755, -1.85887122, -5.0294466,
      -5.22293329, 6.89848137, 1.11226559, -3.14861584, -3.68246865,
      3.34404039, 2.97509623}},
    {{-0.759357333, 1.27064419, -2.41022944, -5.52269745, 2.91421509,
      -0.782507896, -0.228662491, 4.27539682, 2.97740626, 12.5008287,
      -9.4860878, 1.21384573, 9.5913868, 5.45113611, 0.403315663, -6.16194582,
      -1.2852304},
     {-0.207204342, 3.74191999, 1.23634934, 2.39491701, 2.05387831, 8.58817196,
      3.65675569, 9.16720486, -5.8212862, 3.89707994, 13.4189224, -3.09973836,
      7.5796423, -0.365473986, -1.54334283, -5.30818748, 0.602919102}},
    {{-4.85392904, -2.36758995, -8.77992058, 3.50987387, -1.12358332,
      -6.46516418, 3.44891453, -3.35269594, -6.95946836, -2.25799656,
      0.080966711, 3.76473641, -1.4134531, 3.168015, 1.69996285, -2.40649772,
      -9.11525726},
     {9.77986431, 1.73628068, -9.28857327, -0.881102562, 2.03340697,
      -2.93252277, -5.35455704, 1.34708834, -4.76539326, 1.6799016, 5.09027529,
      -4.21229887, -2.32152724, -1.53899908, 6.4186182, -0.891803145,
      6.0681715}},
};

// Layer 0, QHead 0
const float kGoldenQ[kNumTokens][kQBatchSize][kDimsToCompare] = {
    {{-0.374841154, -0.269048423, 0.324933857, 0.270255983, 0.192583397,
      0.0567071736, 0.250502706, 0.625115335, -0.403177321, 0.271447271,
      0.286808699, -0.0656447411, 0.276836812, 0.0164474752, 0.315540373,
      0.265531778, 0.143433452},
     {0.303192079, -0.0379101634, 0.154115498, -0.00872713327, -0.103512973,
      -0.0887796879, -0.216018289, 0.607339799, 0.055648379, -0.191132426,
      -0.319971651, -0.208316207, -0.264384329, -0.299360216, 0.0837299377,
      -0.283533514, -0.501275897}},
    {{-0.114549503, -0.118767068, -0.456864387, 0.144393563, 0.0955479592,
      -0.133590534, 0.444972277, 0.114303589, -0.0884202197, -0.0573218763,
      -0.0792874247, 0.403315246, -0.278178513, -0.00494343042, -0.257657051,
      0.030698413, -0.0186916813},
     {-0.373288035, -0.215933442, -0.201702699, -0.114249617, -0.52541703,
      0.275511354, 0.335507631, 0.62828052, 0.248843148, 0.513091445,
      -0.0282848328, -0.248418555, -0.522639215, -0.0390388519, 0.192302689,
      -0.449831903, -0.179292724}},
    {{0.142575517, -0.237895951, 0.146644697, -0.503801346, -0.523338497,
      0.0719232783, 0.0608261451, 0.151101857, 0.02000916, -0.725266218,
      0.163600311, -0.02573248, 0.293753356, -0.450484604, 0.20146054,
      0.110477969, 0.354954362},
     {-0.239320278, 0.526096821, -0.286867231, -0.443862438, 0.735460579,
      -0.245309472, 0.722944438, 0.0783652365, 0.21042797, 0.569268048,
      -0.0406528264, 0.0399431735, -0.305004865, -0.137150392, -0.130049363,
      0.330584168, 0.0668990687}},
    {{-0.194874108, -0.205414161, -0.220138401, 0.0517282933, -0.161865696,
      -0.233355582, -0.144200221, 0.535177469, 0.219330966, 0.217425376,
      -0.13133359, 0.195236742, 0.257307261, 0.279794693, 0.384352505,
      0.174138933, 0.0952773392},
     {0.122517705, -0.532220542, 0.231840312, 0.421907842, -0.693262935,
      0.379204452, 0.904855072, -0.238233089, -0.0102335168, -0.385086507,
      0.0983751193, -0.0335776061, 0.00405130535, 0.363216281, -0.131849915,
      0.0302671418, -0.00287117064}},
    {{-0.136619762, -0.916439533, -0.250397354, -0.0263281856, -0.607887447,
      -0.12422359, 0.0350730009, 0.0140353218, -0.156378835, 0.979060471,
      0.0746487826, -0.223096639, 0.0214309599, -0.226047188, 0.0714672953,
      -0.405700892, -0.132313401},
     {0.439182878, 0.084455654, -0.776320815, -0.592856288, 0.365012228,
      0.185673609, -0.24275738, 0.275207847, -0.746165574, -0.256350815,
      -0.481744856, 0.524834514, 0.152572945, 0.405694962, -0.279294074,
      -0.619180143, 0.16503042}},
    {{0.307029665, -0.258573472, -0.497068763, 0.133658186, 0.112126596,
      -0.13778466, -0.469314516, -0.144993082, 0.341157258, -0.223292619,
      0.338864386, -0.165094376, 0.317748159, 0.131249368, -0.310955763,
      -0.141406, -0.618950605},
     {-0.405226409, -0.289102376, 0.0477564782, -0.149198949, -0.424721092,
      -0.113134548, -0.0732265264, -0.341526538, 0.124277025, -0.260352641,
      -0.0306069255, 0.385291427, -0.279991835, 0.135148734, 0.251948118,
      0.0279652774, -0.0242935997}},
    {{0.123339117, 0.112210952, -0.423181385, 0.112272829, -0.279016107,
      0.307293028, 0.613147676, 0.00073248148, 0.819842041, 0.0347603858,
      -0.0396398082, 0.074497737, -0.0331122801, -0.205312088, 0.954650819,
      -0.284037501, -0.17986232},
     {0.127260983, -0.184656262, -0.257579148, 0.214763999, 0.4361099,
      -0.0158195253, 0.0339632668, -0.133950815, 0.204951435, -0.247553974,
      0.739190161, -0.0878294855, -0.127532601, -0.549639583, 0.254371703,
      0.0851583332, 0.307077497}},
    {{-0.0720033944, -0.230760068, 0.204314083, -0.346839815, -0.0487727225,
      0.151570067, 0.710862041, -0.4089351, 0.300317228, 0.571746171,
      -0.546940625, 0.0928032696, 0.0187496543, 0.29309383, -0.322793603,
      -0.186359257, -0.550192237},
     {-0.333711773, 0.250101328, -0.538163781, -0.436006278, 0.247505322,
      0.279933214, -0.259696215, 0.0872357413, 0.333090097, 0.950338364,
      -0.110226423, -0.253991336, -0.194895253, 0.336680681, 0.175827622,
      0.184941083, 0.565679312}},
    {{0.492006898, 0.106031463, -0.0973178521, -0.214457124, -0.0938223451,
      0.202232271, 0.293491513, -0.319558859, 0.0366688259, -0.044666674,
      -0.523907304, 0.401466191, 0.0948085636, -0.665217042, 0.0531942286,
      -0.707738578, -0.155400679},
     {-0.309382081, 0.238702834, -0.154397696, 0.153635919, 0.0586032122,
      -0.356307834, -0.242223755, 0.211881027, 0.686982214, 0.361260235,
      -0.487024903, -0.181656718, -0.104096822, -0.0305453707, 0.331899464,
      0.0255006049, -0.826909781}},
    {{0.0855419636, -0.325473666, -0.378067434, 0.599543989, -0.115204476,
      -0.479211658, -0.0426419526, 0.0785699934, -0.409276605, 0.028221447,
      0.0391969681, 0.428700686, -0.132882744, -0.173993275, 0.697183192,
      0.160488009, 0.611800015},
     {0.177823097, 0.604698062, 0.917836607, 0.250253111, -0.775083899,
      0.308443069, 0.194380283, -0.572413027, -0.286389142, -0.382753521,
      0.0876774341, 0.0594621263, -0.192462415, -0.0088978298, -0.449309558,
      0.139618352, 0.164170146}},
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
