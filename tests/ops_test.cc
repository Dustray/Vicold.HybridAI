#include "backends/backend_registry.h"
#include "core/device.h"
#include "core/dtype.h"
#include "core/shape.h"
#include "core/tensor.h"
#include "ops/linear.h"
#include "ops/registry.h"

#include <gtest/gtest.h>

#include <cstring>

namespace hybridai {
namespace {

TEST(OpsRegistryTest, CountKernelsAfterInit) {
    InitializeBuiltinBackends();
    auto backend = BackendRegistry::instance().create_backend(Device::Cpu());
    ASSERT_NE(backend, nullptr);
    backend->register_kernels();
    EXPECT_GT(ops::KernelRegistry::instance().kernel_count(), 0u);
}

TEST(LinearTest, ValidateMismatchK) {
    InitializeBuiltinBackends();

    Device cpu = Device::Cpu();
    Tensor input(Shape{2, 16}, DType::FP32, cpu, nullptr);
    Tensor weight(Shape{32, 8}, DType::FP32, cpu, nullptr);

    Status status = ops::Linear::validate(input, weight, true, nullptr);
    EXPECT_FALSE(status.ok());
}

TEST(LinearTest, OutputShape) {
    Shape out;
    Status status = ops::Linear::compute_output_shape(
        Shape{2, 16}, Shape{32, 16}, true, &out);
    EXPECT_TRUE(status.ok());
    EXPECT_EQ(out.ndim(), 2u);
    EXPECT_EQ(out.dim(0), 2);
    EXPECT_EQ(out.dim(1), 32);
}

TEST(LinearTest, ForwardOnCpu) {
    InitializeBuiltinBackends();
    auto backend = BackendRegistry::instance().create_backend(Device::Cpu());
    ASSERT_NE(backend, nullptr);

    // input [M=2, K=3]
    std::vector<float> input_data = {1, 2, 3, 4, 5, 6};
    auto input_buf = backend->create_buffer(input_data.size() * sizeof(float),
                                            MemoryType::Host);
    ASSERT_NE(input_buf, nullptr);
    std::memcpy(input_buf->data(), input_data.data(),
                  input_data.size() * sizeof(float));
    Tensor input(Shape{2, 3}, DType::FP32, Device::Cpu(), input_buf);

    // weight [N=4, K=3] stored transposed; Linear treats it as [N, K] -> transpose
    std::vector<float> weight_data = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        1, 1, 1,
    };
    auto weight_buf = backend->create_buffer(weight_data.size() * sizeof(float),
                                             MemoryType::Host);
    ASSERT_NE(weight_buf, nullptr);
    std::memcpy(weight_buf->data(), weight_data.data(),
                  weight_data.size() * sizeof(float));
    Tensor weight(Shape{4, 3}, DType::FP32, Device::Cpu(), weight_buf);

    Tensor output = ops::Linear::forward(input, weight, true, nullptr);
    ASSERT_NE(output.data(), nullptr);
    EXPECT_EQ(output.shape().dim(0), 2);
    EXPECT_EQ(output.shape().dim(1), 4);

    const float* out_ptr = static_cast<const float*>(output.data());
    // Row 0 = [1,2,3] * I + [6] => [1,2,3,6]
    EXPECT_FLOAT_EQ(out_ptr[0], 1.0f);
    EXPECT_FLOAT_EQ(out_ptr[1], 2.0f);
    EXPECT_FLOAT_EQ(out_ptr[2], 3.0f);
    EXPECT_FLOAT_EQ(out_ptr[3], 6.0f);
    // Row 1 = [4,5,6] => [4,5,6,15]
    EXPECT_FLOAT_EQ(out_ptr[4], 4.0f);
    EXPECT_FLOAT_EQ(out_ptr[5], 5.0f);
    EXPECT_FLOAT_EQ(out_ptr[6], 6.0f);
    EXPECT_FLOAT_EQ(out_ptr[7], 15.0f);
}

TEST(LinearTest, ForwardAddsBiasAcrossRowsOnCpu) {
    InitializeBuiltinBackends();
    auto backend = BackendRegistry::instance().create_backend(Device::Cpu());
    ASSERT_NE(backend, nullptr);

    const std::vector<float> input_data = {1, 2, 3, 4, 5, 6};
    const std::vector<float> weight_data = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        1, 1, 1,
    };
    const std::vector<float> bias_data = {0.5f, -1.0f, 2.0f, 10.0f};

    auto make_tensor = [&](const Shape& shape,
                           const std::vector<float>& values) {
        auto buffer = backend->create_buffer(values.size() * sizeof(float),
                                             MemoryType::Host);
        std::memcpy(buffer->data(), values.data(),
                    values.size() * sizeof(float));
        return Tensor(shape, DType::FP32, Device::Cpu(), std::move(buffer));
    };

    Tensor input = make_tensor(Shape{2, 3}, input_data);
    Tensor weight = make_tensor(Shape{4, 3}, weight_data);
    Tensor bias = make_tensor(Shape{4}, bias_data);
    Tensor output = ops::Linear::forward(input, weight, true, &bias);

    ASSERT_NE(output.data(), nullptr);
    const float* values = static_cast<const float*>(output.data());
    const std::vector<float> expected = {
        1.5f, 1.0f, 5.0f, 16.0f,
        4.5f, 4.0f, 8.0f, 25.0f,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(values[i], expected[i]);
    }
}

} // namespace
} // namespace hybridai
