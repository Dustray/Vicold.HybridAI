#include "backends/backend_registry.h"
#include "core/device.h"
#include "core/dtype.h"
#include "core/tensor.h"
#include "ops/fp8_dequant.h"

#include <gtest/gtest.h>

#include <array>
#include <cstring>

namespace hybridai {
namespace {

TEST(FP8DequantizerTest, DecodesE4M3AndAppliesBlockScale) {
    InitializeBuiltinBackends();
    auto backend = BackendRegistry::instance().create_backend(Device::Cpu());
    ASSERT_NE(backend, nullptr);

    const std::array<uint8_t, 2> encoded{0x38, 0x40}; // 1.0 and 2.0
    auto input_buffer = backend->create_buffer(encoded.size(), MemoryType::Host);
    ASSERT_NE(input_buffer, nullptr);
    std::memcpy(input_buffer->data(), encoded.data(), encoded.size());
    Tensor input(Shape{2}, DType::FP8_E4M3, Device::Cpu(), input_buffer);

    const std::array<float, 1> scales{2.0f};
    auto scale_buffer = backend->create_buffer(sizeof(scales), MemoryType::Host);
    ASSERT_NE(scale_buffer, nullptr);
    std::memcpy(scale_buffer->data(), scales.data(), sizeof(scales));
    Tensor scale(Shape{1}, DType::FP32, Device::Cpu(), scale_buffer);

    Tensor output;
    ASSERT_TRUE(ops::FP8Dequantizer::dequantize(input, &output, &scale, 128).ok());
    const float* values = static_cast<const float*>(output.data());
    EXPECT_NEAR(values[0], 2.0f, 1e-6f);
    EXPECT_NEAR(values[1], 4.0f, 1e-6f);
}

TEST(FP8DequantizerTest, DequantizesTwoDimensionalBlocks) {
    InitializeBuiltinBackends();
    auto backend = BackendRegistry::instance().create_backend(Device::Cpu());
    ASSERT_NE(backend, nullptr);

    const std::array<uint8_t, 12> encoded{
        0x38, 0x38, 0x38, 0x38, 0x40, 0x40,
        0x40, 0x40, 0x38, 0x38, 0x40, 0x40};
    auto input_buffer = backend->create_buffer(encoded.size(), MemoryType::Host);
    ASSERT_NE(input_buffer, nullptr);
    std::memcpy(input_buffer->data(), encoded.data(), encoded.size());
    Tensor input(Shape{3, 4}, DType::FP8_E4M3, Device::Cpu(), input_buffer);

    const std::array<float, 4> scales{1.0f, 2.0f, 3.0f, 4.0f};
    auto scale_buffer = backend->create_buffer(sizeof(scales), MemoryType::Host);
    ASSERT_NE(scale_buffer, nullptr);
    std::memcpy(scale_buffer->data(), scales.data(), sizeof(scales));
    Tensor scale(Shape{2, 2}, DType::FP32, Device::Cpu(), scale_buffer);

    Tensor output;
    ASSERT_TRUE(ops::FP8Dequantizer::dequantize_2d(
                    input, scale, &output, 2, 2)
                    .ok());
    const float* values = static_cast<const float*>(output.data());
    EXPECT_FLOAT_EQ(values[0], 1.0f);
    EXPECT_FLOAT_EQ(values[1], 1.0f);
    EXPECT_FLOAT_EQ(values[2], 2.0f);
    EXPECT_FLOAT_EQ(values[3], 2.0f);
    EXPECT_FLOAT_EQ(values[4], 2.0f);
    EXPECT_FLOAT_EQ(values[6], 4.0f);
    EXPECT_FLOAT_EQ(values[8], 3.0f);
    EXPECT_FLOAT_EQ(values[10], 8.0f);
}

} // namespace
} // namespace hybridai