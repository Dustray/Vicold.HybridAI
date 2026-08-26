#include "backends/backend_registry.h"
#include "core/device.h"
#include "core/dtype.h"
#include "core/shape.h"
#include "core/tensor.h"
#include "ops/elementwise.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>

namespace hybridai {
namespace {

class ElementwiseTest : public ::testing::Test {
protected:
    void SetUp() override {
        InitializeBuiltinBackends();
        backend_ = BackendRegistry::instance().create_backend(Device::Cpu());
        ASSERT_NE(backend_, nullptr);
    }

    Tensor make_tensor(const std::vector<float>& data,
                       const Shape& shape) {
        auto buf = backend_->create_buffer(data.size() * sizeof(float),
                                           MemoryType::Host);
        std::memcpy(buf->data(), data.data(), data.size() * sizeof(float));
        return Tensor(shape, DType::FP32, Device::Cpu(), buf);
    }

    std::unique_ptr<Backend> backend_;
};

TEST_F(ElementwiseTest, ReLU) {
    Tensor input = make_tensor({-1.0f, 0.0f, 2.0f, -0.5f}, Shape{2, 2});
    Tensor output = ops::Elementwise::relu(input);
    ASSERT_NE(output.data(), nullptr);
    const float* out = static_cast<const float*>(output.data());
    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_FLOAT_EQ(out[1], 0.0f);
    EXPECT_FLOAT_EQ(out[2], 2.0f);
    EXPECT_FLOAT_EQ(out[3], 0.0f);
}

TEST_F(ElementwiseTest, SiLU) {
    Tensor input = make_tensor({0.0f}, Shape{1});
    Tensor output = ops::Elementwise::silu(input);
    ASSERT_NE(output.data(), nullptr);
    const float* out = static_cast<const float*>(output.data());
    EXPECT_FLOAT_EQ(out[0], 0.0f * (1.0f / (1.0f + std::exp(0.0f))));
}

TEST_F(ElementwiseTest, GELUApproximation) {
    Tensor input = make_tensor({0.0f, 1.0f, -1.0f, 2.0f}, Shape{4});
    Tensor output = ops::Elementwise::gelu(input);
    ASSERT_NE(output.data(), nullptr);
    const float* out = static_cast<const float*>(output.data());
    // Compute reference values directly with the same tanh approximation.
    auto gelu_ref = [](float x) {
        return 0.5f * x *
               (1.0f +
                std::tanh(0.7978845608f *
                            (x + 0.0447149983f * x * x * x)));
    };
    EXPECT_NEAR(out[0], gelu_ref(0.0f), 1e-5f);
    EXPECT_NEAR(out[1], gelu_ref(1.0f), 1e-5f);
    EXPECT_NEAR(out[2], gelu_ref(-1.0f), 1e-5f);
    EXPECT_NEAR(out[3], gelu_ref(2.0f), 1e-5f);
}

} // namespace
} // namespace hybridai
