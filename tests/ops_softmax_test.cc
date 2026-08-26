#include "backends/backend_registry.h"
#include "core/device.h"
#include "core/dtype.h"
#include "core/shape.h"
#include "core/tensor.h"
#include "ops/softmax.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>

namespace hybridai {
namespace {

class SoftmaxTest : public ::testing::Test {
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
        if (buf == nullptr) {
            return Tensor();
        }
        std::memcpy(buf->data(), data.data(), data.size() * sizeof(float));
        return Tensor(shape, DType::FP32, Device::Cpu(), buf);
    }

    std::unique_ptr<Backend> backend_;
};

TEST_F(SoftmaxTest, SumsToOne) {
    Tensor input = make_tensor({1.0f, 2.0f, 3.0f}, Shape{3});
    Tensor output = ops::Softmax::forward(input);
    ASSERT_NE(output.data(), nullptr);

    const float* out = static_cast<const float*>(output.data());
    float sum = 0.0f;
    for (int i = 0; i < 3; ++i) {
        sum += out[i];
        EXPECT_GE(out[i], 0.0f);
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST_F(SoftmaxTest, BatchedRows) {
    Tensor input = make_tensor({0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f},
                               Shape{2, 3});
    Tensor output = ops::Softmax::forward(input);
    ASSERT_NE(output.data(), nullptr);

    const float* out = static_cast<const float*>(output.data());
    // Row 0: uniform => [1/3, 1/3, 1/3]
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(out[i], 1.0f / 3.0f, 1e-5f);
    }
    // Row 1: softmax of [1,2,3] => increasing.
    EXPECT_LT(out[3], out[4]);
    EXPECT_LT(out[4], out[5]);
    EXPECT_NEAR(out[3] + out[4] + out[5], 1.0f, 1e-5f);
}

} // namespace
} // namespace hybridai
