// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

// Experimental offline calibration for W8A8. Capture and refinement are both
// opt-in; normal inference does not retain calibration activations or matrices.
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(THIRD_PARTY_GEMMA_CPP_MATMUL_I8_CALIBRATION_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_MATMUL_I8_CALIBRATION_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_MATMUL_I8_CALIBRATION_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_MATMUL_I8_CALIBRATION_TOGGLE
#endif

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// Percent-escape every byte except ASCII letters, digits, '_' and '-'.
// Model tensor names already use these safe characters and are unique.
static inline std::string MMI8CalibrationName(const char* name) {
  static constexpr char hex[] = "0123456789ABCDEF";
  std::string result;
  for (const unsigned char* p = reinterpret_cast<const unsigned char*>(name);
       *p != 0; ++p) {
    if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
        (*p >= '0' && *p <= '9') || *p == '_' || *p == '-') {
      result += static_cast<char>(*p);
    } else {
      result += '%';
      result += hex[*p >> 4];
      result += hex[*p & 15];
    }
  }
  return result;
}

static inline size_t MMI8CalibrationSize(const char* name, size_t fallback,
                                         size_t maximum) {
  const char* value = getenv(name);
  if (value == nullptr || *value == '\0' || *value == '-') return fallback;
  char* end = nullptr;
  const unsigned long long parsed = strtoull(value, &end, 10);
  if (end == value || *end != '\0') return fallback;
  return static_cast<size_t>(HWY_MIN(parsed,
                                     static_cast<unsigned long long>(maximum)));
}

static inline bool MMI8CalibrationCaptureEnabled() {
  static const bool enabled = []() {
    const char* path = getenv("GEMMA_MM_I8_CALIBRATION_CAPTURE");
    return path != nullptr && *path != '\0';
  }();
  return enabled;
}

// Raw files are little-endian F32 [rows,K], with dimensions and transform
// settings in a JSON sidecar. Each call samples evenly spaced finite rows.
class MMI8CalibrationCapture {
 public:
  static MMI8CalibrationCapture& Get() {
    static MMI8CalibrationCapture capture;
    return capture;
  }

  template <typename TA>
  void Capture(const MatPtrT<TA>& A, const MatPtr& B) {
    if (directory_.empty() || max_rows_ == 0 || !B.HasPtr() ||
        A.Cols() != B.Cols() || B.Rows() % kNR != 0 || A.Cols() == 0 ||
        A.Cols() % MMI8RotateBlockSize() != 0)
      return;
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string name = MMI8CalibrationName(B.Name());
    auto& entry = entries_[name];
    if (entry.k != 0 && entry.k != A.Cols())
      HWY_ABORT("Calibration tensor %s changed shape", B.Name());
    entry.k = A.Cols();
    if (entry.rows >= max_rows_ || total_bytes_ >= kMaxBytes) return;
    const size_t k = A.Cols();
    const size_t row_bytes = k * sizeof(float);
    if (row_bytes > kMaxBytes - total_bytes_) return;
    const size_t count = HWY_MIN(HWY_MIN(A.Rows(), rows_per_call_),
                                 max_rows_ - entry.rows);
    hwy::AlignedVector<float> row(k);
    std::vector<float> captured;
    captured.reserve(HWY_MIN(count, (kMaxBytes - total_bytes_) / row_bytes) * k);
    for (size_t r = 0; r < count; ++r) {
      if ((captured.size() + k) * sizeof(float) > kMaxBytes - total_bytes_)
        break;
      const size_t source_row = count == 1 ? A.Rows() - 1
                                          : r * (A.Rows() - 1) / (count - 1);
      MMI8PrepareInputRow(A.Row(source_row), k, nullptr, true, row.data());
      MMI8Rotate(row.data(), k);
      bool finite = true;
      for (size_t c = 0; c < k; ++c) {
        row[c] *= A.Scale();
        finite &= std::isfinite(row[c]);
      }
      if (!finite) {
        ++entry.skipped;
        continue;
      }
      captured.insert(captured.end(), row.begin(), row.end());
    }
    if (captured.empty()) return;
    const std::string path = directory_ + "/" + name + ".f32";
    if (entry.rows == 0 && std::filesystem::exists(path))
      HWY_ABORT("Calibration output already exists: %s; use a fresh directory",
                path.c_str());
    FILE* file = fopen(path.c_str(), entry.rows == 0 ? "wb" : "ab");
    if (file == nullptr) HWY_ABORT("Cannot open calibration output %s", path.c_str());
    const size_t written = fwrite(captured.data(), sizeof(float), captured.size(), file);
    const int close_result = fclose(file);
    if (written != captured.size() || close_result != 0)
      HWY_ABORT("Cannot write calibration output %s", path.c_str());
    entry.rows += captured.size() / k;
    total_bytes_ += captured.size() * sizeof(float);
    const std::string metadata = directory_ + "/" + name + ".json";
    file = fopen(metadata.c_str(), "w");
    if (file == nullptr) HWY_ABORT("Cannot open calibration metadata %s", metadata.c_str());
    const int result = fprintf(
        file,
        "{\"name\":\"%s\",\"K\":%zu,\"rows\":%zu,\"dtype\":\"float32_le\","
        "\"rotation_block\":%zu,\"hash_bits\":%zu,\"source_type\":\"%s\","
        "\"bf16_rounded_input\":true,\"activation_scale_applied\":true,"
        "\"skipped_nonfinite\":%zu}\n",
        name.c_str(), k, entry.rows, MMI8RotateBlockSize(), MMI8HashBits(),
        TypeName<TA>(), entry.skipped);
    const int metadata_close = fclose(file);
    if (result < 0 || metadata_close != 0)
      HWY_ABORT("Cannot write calibration metadata %s", metadata.c_str());
  }

 private:
  MMI8CalibrationCapture() {
    const char* path = getenv("GEMMA_MM_I8_CALIBRATION_CAPTURE");
    if (path == nullptr || *path == '\0') return;
    directory_ = path;
    max_rows_ = MMI8CalibrationSize("GEMMA_MM_I8_CALIBRATION_SAMPLES", 2048, 8192);
    rows_per_call_ = MMI8CalibrationSize("GEMMA_MM_I8_CALIBRATION_ROWS_PER_CALL", 32, 8192);
    const uint16_t endian = 1;
    if (*reinterpret_cast<const unsigned char*>(&endian) != 1)
      HWY_ABORT("Calibration capture requires a little-endian host");
    std::error_code error;
    std::filesystem::create_directories(directory_, error);
    if (error) HWY_ABORT("Cannot create calibration directory %s", directory_.c_str());
  }

  struct Entry {
    size_t k = 0;
    size_t rows = 0;
    size_t skipped = 0;
  };
  static constexpr size_t kMaxBytes = size_t{1} << 30;
  std::string directory_;
  size_t max_rows_ = 0;
  size_t rows_per_call_ = 32;
  size_t total_bytes_ = 0;
  std::mutex mutex_;
  std::unordered_map<std::string, Entry> entries_;
};

// Offline calibration interchange. All header integers and float payloads are
// little-endian. Export omits B.Scale(); import scales are multiplied by it in
// the model packer. Version 02 adds K input-scale floats after the fixed header
// so a transformed weight basis cannot be imported into an incompatible model.
// Files must come from the matching original checkpoint.
class MMI8WeightIO {
 public:
  MMI8WeightIO(const MatPtr& B, size_t block_size, bool selected,
               const float* expected_input_scale = nullptr)
      : n_(B.Rows()), k_(B.Cols()), block_(block_size) {
    if (!selected) return;
    const char* input = getenv("GEMMA_MM_I8_IMPORT_DIR");
    const char* output = getenv("GEMMA_MM_I8_EXPORT_DIR");
    if ((input == nullptr || *input == '\0') &&
        (output == nullptr || *output == '\0')) return;
    const uint16_t endian = 1;
    if (*reinterpret_cast<const unsigned char*>(&endian) != 1)
      HWY_ABORT("Weight calibration interchange requires little endian");
    // Bound products before computing payload sizes or seeking. In particular,
    // a malformed dimension must not wrap into a plausible small file size.
    const uint64_t max_offset = static_cast<uint64_t>(LONG_MAX);
    if (n_ == 0 || k_ == 0 || uint64_t{k_} > max_offset / sizeof(float) ||
        uint64_t{n_} > max_offset / k_)
      HWY_ABORT("Invalid weight interchange dimensions for %s", B.Name());
    const uint64_t q_bytes = uint64_t{n_} * k_;
    const uint64_t basis_bytes =
        expected_input_scale == nullptr ? 0 : uint64_t{k_} * sizeof(float);
    if (expected_input_scale != nullptr) {
      for (size_t c = 0; c < k_; ++c)
        if (!(expected_input_scale[c] > 0.0f) ||
            !std::isfinite(expected_input_scale[c]))
          HWY_ABORT("Invalid input scale for weight interchange: %s", B.Name());
    }
    const std::string name = MMI8CalibrationName(B.Name());
    if (input != nullptr && *input != '\0') {
      const std::string path = std::string(input) + "/" + name + ".wq";
      import_ = fopen(path.c_str(), "rb");
      if (import_ == nullptr && errno != ENOENT)
        HWY_ABORT("Cannot open calibrated weights %s", path.c_str());
      if (import_ != nullptr) {
        unsigned char header[48];
        if (fread(header, 1, sizeof(header), import_) != sizeof(header))
          HWY_ABORT("Invalid calibrated weight header %s", path.c_str());
        const bool has_basis = memcmp(header, "MMI8WQ02", 8) == 0;
        if (!has_basis && memcmp(header, "MMI8WQ01", 8) != 0)
          HWY_ABORT("Invalid calibrated weight header %s", path.c_str());
        if (has_basis != (expected_input_scale != nullptr))
          HWY_ABORT("Calibrated weight input basis mismatch for %s", B.Name());
        const auto read64 = [&](size_t offset) {
          uint64_t value = 0;
          for (size_t i = 0; i < 8; ++i)
            value |= uint64_t{header[offset + i]} << (8 * i);
          return value;
        };
        if (read64(8) != n_ || read64(16) != k_ ||
            read64(24) != block_ || read64(32) != MMI8RotateBlockSize() ||
            read64(40) != MMI8HashBits() ||
            (block_ != 32 && block_ != 64 && block_ != 128) || k_ % block_ != 0)
          HWY_ABORT("Calibrated weight settings mismatch for %s", B.Name());
        const uint64_t scale_count = q_bytes / block_;
        if (basis_bytes > max_offset - sizeof(header))
          HWY_ABORT("Invalid calibrated weight basis size %s", path.c_str());
        const uint64_t data_offset = sizeof(header) + basis_bytes;
        if (q_bytes > max_offset - data_offset ||
            scale_count > (uint64_t{1} << 28) ||
            scale_count >
                (max_offset - data_offset - q_bytes) / sizeof(float))
          HWY_ABORT("Invalid calibrated weight payload size %s", path.c_str());
        const uint64_t scale_offset = data_offset + q_bytes;
        std::error_code error;
        const uint64_t file_size = std::filesystem::file_size(path, error);
        if (error || file_size != scale_offset + scale_count * sizeof(float))
          HWY_ABORT("Invalid calibrated weight payload size %s", path.c_str());
        if (has_basis) {
          std::vector<float> input_scale(k_);
          if (fread(input_scale.data(), sizeof(float), k_, import_) != k_)
            HWY_ABORT("Cannot read calibrated weight basis %s", path.c_str());
          for (float scale : input_scale)
            if (!(scale > 0.0f) || !std::isfinite(scale))
              HWY_ABORT("Invalid calibrated weight input scale %s", path.c_str());
          if (memcmp(input_scale.data(), expected_input_scale,
                     static_cast<size_t>(basis_bytes)) != 0)
            HWY_ABORT("Calibrated weight input scale mismatch for %s", B.Name());
        }
        scales_.resize(static_cast<size_t>(scale_count));
        if (fseek(import_, static_cast<long>(scale_offset), SEEK_SET) != 0 ||
            fread(scales_.data(), sizeof(float), scales_.size(), import_) != scales_.size() ||
            fseek(import_, static_cast<long>(data_offset), SEEK_SET) != 0)
          HWY_ABORT("Cannot read calibrated weight scales %s", path.c_str());
        for (float scale : scales_)
          if (!(scale > 0.0f) || !std::isfinite(scale))
            HWY_ABORT("Invalid calibrated weight scale %s", path.c_str());
      }
    }
    if (output != nullptr && *output != '\0') {
      constexpr uint64_t header_bytes = 40;
      if (basis_bytes > max_offset - header_bytes ||
          q_bytes > (max_offset - header_bytes - basis_bytes) / sizeof(float))
        HWY_ABORT("Invalid weight export payload size for %s", B.Name());
      std::error_code error;
      std::filesystem::create_directories(output, error);
      if (error) HWY_ABORT("Cannot create weight export directory %s", output);
      const std::string path = std::string(output) + "/" + name + ".f32";
      if (std::filesystem::exists(path))
        HWY_ABORT("Weight export already exists: %s; use a fresh directory", path.c_str());
      export_ = fopen(path.c_str(), "wb");
      if (export_ == nullptr) HWY_ABORT("Cannot open weight export %s", path.c_str());
      const uint64_t header[4] = {n_, k_, MMI8RotateBlockSize(), MMI8HashBits()};
      const char* magic = expected_input_scale == nullptr ? "W8RAW001" : "W8RAW002";
      if (fwrite(magic, 1, 8, export_) != 8 ||
          fwrite(header, 1, sizeof(header), export_) != sizeof(header) ||
          (expected_input_scale != nullptr &&
           fwrite(expected_input_scale, sizeof(float), k_, export_) != k_))
        HWY_ABORT("Cannot write weight export header %s", path.c_str());
    }
  }

  MMI8WeightIO(const MMI8WeightIO&) = delete;
  MMI8WeightIO& operator=(const MMI8WeightIO&) = delete;
  ~MMI8WeightIO() {
    if (import_ != nullptr && (imported_rows_ != n_ || fclose(import_) != 0))
      HWY_ABORT("Incomplete calibrated weight import");
    if (export_ != nullptr && (exported_rows_ != n_ || fclose(export_) != 0))
      HWY_ABORT("Incomplete rotated weight export");
  }

  bool Importing() const { return import_ != nullptr; }
  bool Exporting() const { return export_ != nullptr; }

  bool ImportRow(size_t row, MMI8BT* out) {
    if (!Importing()) return false;
    if (row != imported_rows_ || fread(out, 1, k_, import_) != k_)
      HWY_ABORT("Cannot read calibrated weight row %zu", row);
    for (size_t c = 0; c < k_; ++c) {
      const int q = reinterpret_cast<const int8_t*>(out)[c];
      if (q == -128) HWY_ABORT("Calibrated weight q must be in [-127,127]");
      out[c] = static_cast<MMI8BT>(q + (GEMMA_MM_I8_BIASED_B ? 128 : 0));
    }
    ++imported_rows_;
    return true;
  }

  float Scale(size_t row, size_t group) const {
    HWY_DASSERT(Importing() && row < n_ && group < k_ / block_);
    return scales_[group * n_ + row];
  }

  void ExportRow(size_t row, const float* weights) {
    if (!Exporting()) return;
    if (row != exported_rows_) HWY_ABORT("Out-of-order weight export");
    for (size_t c = 0; c < k_; ++c)
      if (!std::isfinite(weights[c])) HWY_ABORT("Nonfinite rotated weight export");
    if (fwrite(weights, sizeof(float), k_, export_) != k_)
      HWY_ABORT("Cannot write rotated weight row %zu", row);
    ++exported_rows_;
  }

 private:
  const size_t n_;
  const size_t k_;
  const size_t block_;
  FILE* import_ = nullptr;
  FILE* export_ = nullptr;
  size_t imported_rows_ = 0;
  size_t exported_rows_ = 0;
  std::vector<float> scales_;
};

// Mean-only correction is O(K) per packed row. The sidecar is magic MMI8MU01,
// little-endian uint64 K/group, then float32 muX[K], followed by muXhat[K].
class MMI8MeanCalibration {
 public:
  MMI8MeanCalibration(const MatPtr& B, size_t block_size, bool selected) {
    const char* directory = getenv("GEMMA_MM_I8_CALIBRATION_DIR");
    if (!selected || !MMI8Flag("GEMMA_MM_I8_BIAS_CORRECTION") ||
        block_size == 0 || directory == nullptr || *directory == '\0')
      return;
    const std::string path = std::string(directory) + "/" +
                             MMI8CalibrationName(B.Name()) + ".mean";
    FILE* file = fopen(path.c_str(), "rb");
    if (file == nullptr) {
      if (errno == ENOENT) return;
      HWY_ABORT("Cannot open mean calibration %s", path.c_str());
    }
    unsigned char header[24];
    if (fread(header, 1, sizeof(header), file) != sizeof(header) ||
        memcmp(header, "MMI8MU01", 8) != 0)
      HWY_ABORT("Invalid mean calibration header %s", path.c_str());
    const auto read64 = [&](size_t offset) {
      uint64_t value = 0;
      for (size_t i = 0; i < 8; ++i)
        value |= uint64_t{header[offset + i]} << (8 * i);
      return value;
    };
    const uint64_t k = read64(8), group = read64(16);
    if (k != B.Cols() || group != block_size ||
        (group != 32 && group != 64 && group != 128) || k % group != 0)
      HWY_ABORT("Mean calibration dimensions do not match %s", B.Name());
    const uint16_t endian = 1;
    if (*reinterpret_cast<const unsigned char*>(&endian) != 1)
      HWY_ABORT("Mean calibration requires a little-endian host");
    std::error_code error;
    const auto file_size = std::filesystem::file_size(path, error);
    if (error || k > (uint64_t{1} << 27) ||
        file_size != sizeof(header) + 2 * k * sizeof(float))
      HWY_ABORT("Invalid mean calibration payload size %s", path.c_str());
    means_.resize(static_cast<size_t>(2 * k));
    const size_t loaded =
        fread(means_.data(), sizeof(float), means_.size(), file);
    const int close_result = fclose(file);
    if (loaded != means_.size() || close_result != 0)
      HWY_ABORT("Cannot read mean calibration payload %s", path.c_str());
    for (float value : means_)
      if (!std::isfinite(value))
        HWY_ABORT("Nonfinite mean calibration %s", path.c_str());
    k_ = static_cast<size_t>(k);
    block_ = block_size;
  }

  bool Enabled() const { return !means_.empty(); }

  double Correction(const float* weights, size_t begin, const MMI8BT* bytes,
                    float scale) const {
    if (!Enabled()) return 0.0;
    double correction = 0.0;
    for (size_t i = 0; i < block_; ++i) {
      const int q = static_cast<int>(bytes[i]) -
                    (GEMMA_MM_I8_BIASED_B ? 128 : 0);
      correction += double(means_[begin + i]) * weights[i] -
                    double(means_[k_ + begin + i]) * q * scale;
    }
    return correction;
  }

 private:
  size_t k_ = 0;
  size_t block_ = 0;
  std::vector<float> means_;
};

// A calibration file contains magic MMI8HG01, little-endian uint64 K/group/
// samples, followed by row-major F32 H and G for each group. H = Xhat^T Xhat,
// G = Xhat^T X. Offline generation may add the same ridge prior to H and G.
// The object lives only while one tensor is packed, bounding retained memory.
class MMI8WeightCalibration {
 public:
  MMI8WeightCalibration(const MatPtr& B, size_t block_size, bool selected) {
    const char* directory = getenv("GEMMA_MM_I8_CALIBRATION_DIR");
    if (!selected || block_size == 0 || directory == nullptr || *directory == '\0')
      return;
    const std::string path = std::string(directory) + "/" +
                             MMI8CalibrationName(B.Name()) + ".hg";
    FILE* file = fopen(path.c_str(), "rb");
    if (file == nullptr) {
      if (errno == ENOENT) return;
      HWY_ABORT("Cannot open weight calibration %s", path.c_str());
    }
    unsigned char header[32];
    if (fread(header, 1, sizeof(header), file) != sizeof(header) ||
        memcmp(header, "MMI8HG01", 8) != 0)
      HWY_ABORT("Invalid calibration header %s", path.c_str());
    const auto read64 = [&](size_t offset) {
      uint64_t value = 0;
      for (size_t i = 0; i < 8; ++i) value |= uint64_t{header[offset + i]} << (8 * i);
      return value;
    };
    const uint64_t k = read64(8), group = read64(16), samples = read64(24);
    if (k != B.Cols() || group != block_size || samples == 0 ||
        (group != 32 && group != 64 && group != 128) || k % group != 0)
      HWY_ABORT("Calibration dimensions do not match %s", B.Name());
    const uint16_t endian = 1;
    if (*reinterpret_cast<const unsigned char*>(&endian) != 1)
      HWY_ABORT("Calibration refinement requires a little-endian host");
    const uint64_t count = 2 * k * group;
    std::error_code error;
    const auto file_size = std::filesystem::file_size(path, error);
    if (error || count > (uint64_t{1} << 28) || file_size != sizeof(header) + count * sizeof(float))
      HWY_ABORT("Invalid calibration payload size %s", path.c_str());
    matrices_.resize(static_cast<size_t>(count));
    const size_t loaded = fread(matrices_.data(), sizeof(float), matrices_.size(), file);
    const int close_result = fclose(file);
    if (loaded != matrices_.size() || close_result != 0)
      HWY_ABORT("Cannot read calibration payload %s", path.c_str());
    for (float value : matrices_)
      if (!std::isfinite(value)) HWY_ABORT("Nonfinite calibration matrix %s", path.c_str());
    block_ = block_size;
    // Make the Hessian explicitly symmetric for the quadratic coordinate updates.
    for (size_t c = 0; c < B.Cols(); c += block_) {
      float* h = matrices_.data() + 2 * c * block_;
      for (size_t i = 0; i < block_; ++i) {
        if (h[i * block_ + i] < 0.0f)
          HWY_ABORT("Negative calibration diagonal %s", path.c_str());
        for (size_t j = 0; j < i; ++j) {
          const float average = static_cast<float>(
              (double(h[i * block_ + j]) + h[j * block_ + i]) * 0.5);
          h[i * block_ + j] = h[j * block_ + i] = average;
        }
      }
    }
    sweeps_ = MMI8CalibrationSize("GEMMA_MM_I8_CALIBRATION_SWEEPS", 1, 8);
  }

  float Refine(const float* weights, size_t begin, MMI8BT* bytes, float initial_scale) const {
    if (matrices_.empty()) return initial_scale;
    const float* h = matrices_.data() + 2 * begin * block_;
    const float* g = h + block_ * block_;
    std::array<int, 128> q{}, best_q{};
    std::array<double, 128> target{}, gradient{}, hq{};
    for (size_t i = 0; i < block_; ++i) {
      q[i] = static_cast<int>(bytes[i]) - (GEMMA_MM_I8_BIASED_B ? 128 : 0);
      best_q[i] = q[i];
      for (size_t j = 0; j < block_; ++j)
        target[i] += double(g[i * block_ + j]) * weights[j];
    }
    // H*q changes only when the integer vector changes, not when its scale
    // changes. Keep the original reduction order while avoiding repeated GEMV.
    const auto refresh_hq = [&]() {
      for (size_t i = 0; i < block_; ++i) {
        hq[i] = 0.0;
        for (size_t j = 0; j < block_; ++j)
          hq[i] += double(h[i * block_ + j]) * q[j];
      }
    };
    const auto objective = [&](float scale) {
      double value = 0.0;
      for (size_t i = 0; i < block_; ++i)
        value += double(scale) * q[i] * (double(scale) * hq[i] - 2.0 * target[i]);
      return value;
    };
    float scale = initial_scale, best_scale = initial_scale;
    refresh_hq();
    double best_objective = objective(scale);
    if (!std::isfinite(best_objective)) return initial_scale;
    for (size_t pass = 0; pass <= sweeps_; ++pass) {
      double numerator = 0.0, denominator = 0.0;
      for (size_t i = 0; i < block_; ++i) {
        numerator += q[i] * target[i];
        denominator += q[i] * hq[i];
      }
      if (denominator > 0.0) {
        const float candidate = static_cast<float>(numerator / denominator);
        if (candidate > 0.0f && std::isfinite(candidate)) scale = candidate;
      }
      const double loss = objective(scale);
      if (std::isfinite(loss) && loss < best_objective) {
        best_objective = loss;
        best_scale = scale;
        best_q = q;
      }
      if (pass == sweeps_ || !(scale > 0.0f) || !std::isfinite(scale)) break;
      for (size_t i = 0; i < block_; ++i)
        gradient[i] = double(scale) * hq[i] - target[i];
      for (size_t j = 0; j < block_; ++j) {
        const double diagonal = h[j * block_ + j];
        if (!(diagonal > 0.0)) continue;
        const double desired = q[j] - gradient[j] / (double(scale) * diagonal);
        if (!std::isfinite(desired)) continue;
        const int rounded = static_cast<int>(std::lround(HWY_MIN(127.0, HWY_MAX(-127.0, desired))));
        const int change = rounded - q[j];
        const double step = double(scale) * change;
        const double delta = 2.0 * step * gradient[j] + step * step * diagonal;
        if (change == 0 || !(delta < 0.0)) continue;
        q[j] = rounded;
        for (size_t i = 0; i < block_; ++i)
          gradient[i] += step * h[i * block_ + j];
      }
      refresh_hq();
    }
    for (size_t i = 0; i < block_; ++i)
      bytes[i] = static_cast<MMI8BT>(best_q[i] + (GEMMA_MM_I8_BIASED_B ? 128 : 0));
    return best_scale;
  }

 private:
  size_t block_ = 0;
  size_t sweeps_ = 0;
  std::vector<float> matrices_;
};

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_MATMUL_I8_CALIBRATION_TOGGLE
