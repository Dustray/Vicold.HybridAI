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

} // namespace
} // namespace hybridai