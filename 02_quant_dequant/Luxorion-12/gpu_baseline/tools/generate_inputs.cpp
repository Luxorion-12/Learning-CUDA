/*
 * 生成GPU性能baseline使用的固定输入。生成时间不计入性能测量；三个文件均
 * 使用20字节tensor header和FP32行主序data，并在写出后重新读取逐元素校验。
 */
#include "io.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::int64_t kRows = 1024;
constexpr std::int64_t kCols = 1024;
constexpr std::uint32_t kSeed = 1234;
constexpr float kUniformLimit = 6000.0F;
constexpr float kNormalMean = 0.0F;
constexpr float kNormalStddev = 20.0F;
constexpr std::size_t kOutlierInterval = 1000;
constexpr float kOutlierMagnitude = 6000.0F;

std::string filename(const std::string& distribution) {
  return distribution + "_fp32_1024x1024_seed1234.tensor.bin";
}

void write_and_verify(const std::filesystem::path& path,
                      const std::vector<float>& values) {
  const luxorion::TensorData tensor{
      kRows, kCols, luxorion::TensorDType::FP32, values};
  luxorion::write_tensor_file(path.string(), tensor);
  const auto restored = luxorion::read_tensor_file(path.string());
  if (restored.num_rows != kRows || restored.num_cols != kCols
      || restored.dtype != luxorion::TensorDType::FP32
      || restored.values != values) {
    throw std::runtime_error("generated tensor verification failed: "
                             + path.string());
  }
}

void write_manifest(const std::filesystem::path& directory) {
  std::ofstream file(directory / "manifest.json", std::ios::trunc);
  if (!file) throw std::runtime_error("cannot create benchmark manifest");
  file << "{\n"
       << "  \"schema_version\": 1,\n"
       << "  \"tensor_header_bytes\": 20,\n"
       << "  \"dtype\": \"fp32\",\n"
       << "  \"endianness\": \"little\",\n"
       << "  \"row_major\": true,\n"
       << "  \"block_layout\": \"continuous_flattened\",\n"
       << "  \"num_rows\": " << kRows << ",\n"
       << "  \"num_cols\": " << kCols << ",\n"
       << "  \"element_count\": " << (kRows * kCols) << ",\n"
       << "  \"seed\": " << kSeed << ",\n"
       << "  \"datasets\": [\n"
       << "    {\"file\": \"" << filename("uniform")
       << "\", \"distribution\": \"uniform\", \"min\": -6000.0, \"max\": 6000.0},\n"
       << "    {\"file\": \"" << filename("normal")
       << "\", \"distribution\": \"normal\", \"mean\": 0.0, \"stddev\": 20.0},\n"
       << "    {\"file\": \"" << filename("outliers")
       << "\", \"distribution\": \"normal_with_outliers\", \"mean\": 0.0, "
          "\"stddev\": 20.0, \"outlier_interval\": 1000, "
          "\"outlier_magnitude\": 6000.0}\n"
       << "  ]\n"
       << "}\n";
  if (!file) throw std::runtime_error("failed to write benchmark manifest");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const std::filesystem::path output = argc > 1
        ? std::filesystem::path(argv[1])
        : std::filesystem::path("input/benchmark");
    std::filesystem::create_directories(output);
    const std::size_t count = static_cast<std::size_t>(kRows * kCols);

    std::mt19937 uniform_engine(kSeed);
    std::uniform_real_distribution<float> uniform_distribution(
        -kUniformLimit, kUniformLimit);
    std::vector<float> uniform(count);
    for (float& value : uniform) value = uniform_distribution(uniform_engine);

    std::mt19937 normal_engine(kSeed);
    std::normal_distribution<float> normal_distribution(
        kNormalMean, kNormalStddev);
    std::vector<float> normal(count);
    for (float& value : normal) value = normal_distribution(normal_engine);

    std::vector<float> outliers = normal;
    for (std::size_t i = 0; i < outliers.size(); i += kOutlierInterval) {
      outliers[i] = (i / kOutlierInterval) % 2 == 0
          ? kOutlierMagnitude : -kOutlierMagnitude;
    }

    write_and_verify(output / filename("uniform"), uniform);
    write_and_verify(output / filename("normal"), normal);
    write_and_verify(output / filename("outliers"), outliers);
    write_manifest(output);
    std::cout << "Generated and verified 3 benchmark tensors in "
              << output.string() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Benchmark input generation failed: " << error.what() << '\n';
    return 1;
  }
}
