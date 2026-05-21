// Standalone test for Gemma 4 model configs.
// Validates layer counts, qkv_dim per layer, KV cache layout, and attention
// window sizes against the values observed in the GGUF metadata.
#include <cassert>
#include <cstdio>
#include <vector>

#include "gemma/configs.h"
#include "compression/types.h"

namespace gcpp {

static void PrintFail(const char* msg) {
  fprintf(stderr, "FAIL: %s\n", msg);
}

static bool TestGemma4E2B() {
  fprintf(stderr, "\n=== Gemma 4 E2B (nano) ===\n");
  ModelConfig cfg(Model::GEMMA4_E2B, Type::kSFP, ChooseWrapping(Model::GEMMA4_E2B));

  bool ok = true;

  // Basic dimensions (from GGUF: block_count=35, embedding_length=1536)
  if (cfg.num_layers != 35) { PrintFail("E2B num_layers != 35"); ok = false; }
  if (cfg.model_dim != 1536) { PrintFail("E2B model_dim != 1536"); ok = false; }
  if (cfg.vocab_size != 262144) { PrintFail("E2B vocab_size != 262144"); ok = false; }
  if (cfg.per_layer_embd_dim != 256) { PrintFail("E2B per_layer_embd_dim != 256"); ok = false; }
  fprintf(stderr, "  layers=%u  model_dim=%u  vocab=%u  per_layer_embd=%u\n",
          cfg.num_layers, cfg.model_dim, cfg.vocab_size, cfg.per_layer_embd_dim);

  // Pattern TTTTF x7: SWA layers at indices NOT in {4,9,14,19,24,29,34}
  // Full-att (global) layers: indices where i%5==4
  size_t swa_count = 0, full_count = 0;
  for (size_t i = 0; i < cfg.num_layers; ++i) {
    bool is_global = cfg.IsGlobalLayer(i);
    if (is_global) {
      ++full_count;
      if (cfg.layer_configs[i].qkv_dim != 512) {
        PrintFail("E2B full-att layer qkv_dim != 512"); ok = false;
      }
    } else {
      ++swa_count;
      if (cfg.layer_configs[i].qkv_dim != 256) {
        PrintFail("E2B SWA layer qkv_dim != 256"); ok = false;
      }
    }
  }
  fprintf(stderr, "  SWA layers=%zu  full-att layers=%zu\n", swa_count, full_count);
  if (swa_count != 28) { PrintFail("E2B SWA count != 28"); ok = false; }
  if (full_count != 7) { PrintFail("E2B full-att count != 7"); ok = false; }

  // FFN: first 15 layers = 6144, last 20 = 12288
  for (size_t i = 0; i < 15; ++i) {
    if (cfg.layer_configs[i].ff_hidden_dim != 6144) {
      fprintf(stderr, "  FAIL: E2B layer %zu ff_hidden_dim=%u (want 6144)\n",
              i, cfg.layer_configs[i].ff_hidden_dim);
      ok = false;
    }
  }
  for (size_t i = 15; i < 35; ++i) {
    if (cfg.layer_configs[i].ff_hidden_dim != 12288) {
      fprintf(stderr, "  FAIL: E2B layer %zu ff_hidden_dim=%u (want 12288)\n",
              i, cfg.layer_configs[i].ff_hidden_dim);
      ok = false;
    }
  }
  fprintf(stderr, "  FFN: layers 0-14=%u, layers 15-34=%u\n",
          cfg.layer_configs[0].ff_hidden_dim, cfg.layer_configs[15].ff_hidden_dim);

  // KV cache: verify KVCacheLayerOffset is non-uniform (SWA < full-att layers)
  size_t offset_layer4 = cfg.KVCacheLayerOffset(4);
  size_t size_layer4 = cfg.layer_configs[4].CacheLayerSize();
  // Layers 0-3 are SWA (qkv_dim=256, kv_heads=1 → size=512 each)
  // So offset for layer 4 should be 4*512=2048, not 4*1024=4096
  fprintf(stderr, "  KV offset[4]=%zu  layer[4].CacheLayerSize=%zu\n",
          offset_layer4, size_layer4);
  if (offset_layer4 != 4 * 512u) {
    fprintf(stderr, "  FAIL: E2B KVCacheLayerOffset(4)=%zu (want %u)\n",
            offset_layer4, 4*512u);
    ok = false;
  }
  if (size_layer4 != 1024) {
    fprintf(stderr, "  FAIL: E2B layer[4].CacheLayerSize=%zu (want 1024)\n", size_layer4);
    ok = false;
  }

  // Total KV cache cols
  size_t kv_cols = cfg.KVCacheCols();
  // 28 SWA * 512 + 7 full-att * 1024 = 14336 + 7168 = 21504
  fprintf(stderr, "  KVCacheCols=%zu (expected 21504)\n", kv_cols);
  if (kv_cols != 21504) { PrintFail("E2B KVCacheCols wrong"); ok = false; }

  fprintf(stderr, "  E2B: %s\n", ok ? "PASS" : "FAIL");
  return ok;
}

static bool TestGemma4E4B() {
  fprintf(stderr, "\n=== Gemma 4 E4B (turbo) ===\n");
  ModelConfig cfg(Model::GEMMA4_E4B, Type::kSFP, ChooseWrapping(Model::GEMMA4_E4B));

  bool ok = true;

  // Basic dimensions (from GGUF: block_count=42, embedding_length=2560)
  if (cfg.num_layers != 42) { PrintFail("E4B num_layers != 42"); ok = false; }
  if (cfg.model_dim != 2560) { PrintFail("E4B model_dim != 2560"); ok = false; }
  if (cfg.vocab_size != 262144) { PrintFail("E4B vocab_size != 262144"); ok = false; }
  fprintf(stderr, "  layers=%u  model_dim=%u  vocab=%u\n",
          cfg.num_layers, cfg.model_dim, cfg.vocab_size);

  // Pattern TTTTTF x7: full-att at indices 5,11,17,23,29,35,41
  size_t swa_count = 0, full_count = 0;
  for (size_t i = 0; i < cfg.num_layers; ++i) {
    bool is_global = cfg.IsGlobalLayer(i);
    if (is_global) {
      ++full_count;
      if (cfg.layer_configs[i].qkv_dim != 512) {
        PrintFail("E4B full-att layer qkv_dim != 512"); ok = false;
      }
    } else {
      ++swa_count;
      if (cfg.layer_configs[i].qkv_dim != 256) {
        PrintFail("E4B SWA layer qkv_dim != 256"); ok = false;
      }
    }
  }
  fprintf(stderr, "  SWA layers=%zu  full-att layers=%zu\n", swa_count, full_count);
  if (swa_count != 35) { PrintFail("E4B SWA count != 35"); ok = false; }
  if (full_count != 7) { PrintFail("E4B full-att count != 7"); ok = false; }

  // FFN: uniform 10240
  for (size_t i = 0; i < 42; ++i) {
    if (cfg.layer_configs[i].ff_hidden_dim != 10240) {
      fprintf(stderr, "  FAIL: E4B layer %zu ff_hidden_dim=%u (want 10240)\n",
              i, cfg.layer_configs[i].ff_hidden_dim);
      ok = false;
      break;
    }
  }
  fprintf(stderr, "  FFN (uniform): %u\n", cfg.layer_configs[0].ff_hidden_dim);

  // KV cache: E4B has kv_heads=2
  // SWA: kv_heads=2, qkv_dim=256 → CacheLayerSize = 2*256*2 = 1024
  // Full-att: kv_heads=2, qkv_dim=512 → CacheLayerSize = 2*512*2 = 2048
  size_t offset_layer5 = cfg.KVCacheLayerOffset(5);
  size_t size_layer5 = cfg.layer_configs[5].CacheLayerSize();
  // Layers 0-4 are SWA → offset = 5*1024 = 5120
  fprintf(stderr, "  KV offset[5]=%zu  layer[5].CacheLayerSize=%zu\n",
          offset_layer5, size_layer5);
  if (offset_layer5 != 5 * 1024u) {
    fprintf(stderr, "  FAIL: E4B KVCacheLayerOffset(5)=%zu (want %u)\n",
            offset_layer5, 5*1024u);
    ok = false;
  }
  if (size_layer5 != 2048) {
    fprintf(stderr, "  FAIL: E4B layer[5].CacheLayerSize=%zu (want 2048)\n", size_layer5);
    ok = false;
  }

  // Total KV: 35 SWA * 1024 + 7 full-att * 2048 = 35840 + 14336 = 50176
  size_t kv_cols = cfg.KVCacheCols();
  fprintf(stderr, "  KVCacheCols=%zu (expected 50176)\n", kv_cols);
  if (kv_cols != 50176) { PrintFail("E4B KVCacheCols wrong"); ok = false; }

  fprintf(stderr, "  E4B: %s\n", ok ? "PASS" : "FAIL");
  return ok;
}

static bool TestSerializeRoundTrip(Model model) {
  ModelConfig cfg(model, Type::kSFP, ChooseWrapping(model));
  const std::vector<uint32_t> serialized = cfg.Write();
  ModelConfig deserialized;
  const IFields::ReadResult result =
      deserialized.Read(hwy::Span<const uint32_t>(serialized), /*pos=*/0);
  bool ok = true;
  if (result.pos != serialized.size()) { PrintFail("serialized size mismatch"); ok = false; }
  if (!deserialized.TestEqual(cfg, /*print=*/true)) { PrintFail("TestEqual failed"); ok = false; }
  if (deserialized.model != model) { PrintFail("model mismatch after deserialize"); ok = false; }
  return ok;
}

}  // namespace gcpp

int main() {
  bool all_ok = true;
  all_ok &= gcpp::TestGemma4E2B();
  all_ok &= gcpp::TestGemma4E4B();

  fprintf(stderr, "\n=== Serialize round-trip ===\n");
  bool rt_e2b = gcpp::TestSerializeRoundTrip(gcpp::Model::GEMMA4_E2B);
  bool rt_e4b = gcpp::TestSerializeRoundTrip(gcpp::Model::GEMMA4_E4B);
  fprintf(stderr, "  E2B round-trip: %s\n", rt_e2b ? "PASS" : "FAIL");
  fprintf(stderr, "  E4B round-trip: %s\n", rt_e4b ? "PASS" : "FAIL");
  all_ok &= rt_e2b && rt_e4b;

  fprintf(stderr, "\n=== Overall: %s ===\n\n", all_ok ? "PASS" : "FAIL");
  return all_ok ? 0 : 1;
}
