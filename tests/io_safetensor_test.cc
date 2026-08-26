#include "core/device.h"
#include "core/tensor.h"
#include "io/safetensor_loader.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace hybridai {
namespace {

std::filesystem::path MakeFixture() {
    const nlohmann::json header = {
        {"x", {{"dtype", "F32"}, {"shape", {2, 2}},
                {"data_offsets", {0, 16}}}},
        {"__metadata__", {{"format", "pt"}}}
    };
    const std::string header_text = header.dump();
    const auto path = std::filesystem::temp_directory_path() /
                      "hybridai_safetensor_test.safetensors";
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    const uint64_t size = header_text.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    file.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    const std::array<float, 4> values{1.0f, 2.0f, 3.0f, 4.0f};
    file.write(reinterpret_cast<const char*>(values.data()), sizeof(values));
    return path;
}

TEST(SafeTensorLoaderTest, ReadsHeaderAndTensorData) {
    const auto path = MakeFixture();
    io::SafeTensorLoader loader;
    ASSERT_TRUE(loader.open(path.string()).ok());
    ASSERT_EQ(loader.tensor_names().size(), 1u);

    const auto* info = loader.tensor_info("x");
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->dtype, DType::FP32);
    EXPECT_EQ(info->shape, Shape({2, 2}));

    Tensor tensor;
    ASSERT_TRUE(loader.load("x", Device::Cpu(), &tensor).ok());
    ASSERT_NE(tensor.data(), nullptr);
    const float* values = static_cast<const float*>(tensor.data());
    EXPECT_FLOAT_EQ(values[0], 1.0f);
    EXPECT_FLOAT_EQ(values[3], 4.0f);
    std::filesystem::remove(path);
}

} // namespace
} // namespace hybridai