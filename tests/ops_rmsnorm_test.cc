#include "backends/backend_registry.h"
#include "core/device.h"
#include "core/dtype.h"
#include "core/shape.h"
#include "core/tensor.h"
#include "ops/rmsnorm.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>

namespace hybridai {
namespace {

class RMSNormTest : public ::testing::Test {
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

TEST_F(RMSNormTest, Basic) {
    // Single row [1, 1, 1, 1], weight [1, 1, 1, 1].
    // rms = sqrt((1+1+1+1)/4 + eps) = sqrt(1 + eps) ~= 1
    // output ~= [1,1,1,1]
    Tensor input = make_tensor({1.0f, 1.0f, 1.0f, 1.0f}, Shape{1, 4});
    Tensor weight = make_tensor({1.0f, 1.0f, 1.0f, 1.0f}, Shape{4});

    Tensor output = ops::RMSNorm::forward(input, weight, 1e-6f);
    ASSERT_NE(output.data(), nullptr);

    const float* out = static_cast<const float*>(output.data());
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(out[i], 1.0f, 1e-5f);
    }
}

TEST_F(RMSNormTest, ScaledWeight) {
    // input [1, 2, 3], weight [2, 2, 2]
    Tensor input = make_tensor({1.0f, 2.0f, 3.0f}, Shape{1, 3});
    Tensor weight = make_tensor({2.0f, 2.0f, 2.0f}, Shape{3});

    Tensor output = ops::RMSNorm::forward(input, weight, 1e-6f);
    ASSERT_NE(output.data(), nullptr);

    const float* out = static_cast<const float*>(output.data());
    float mean_sq = (1.0f + 4.0f + 9.0f) / 3.0f;
    float inv_rms = 1.0f / std::sqrt(mean_sq + 1e-6f);
    EXPECT_NEAR(out[0], 1.0f * inv_rms * 2.0f, 1e-5f);
    EXPECT_NEAR(out[1], 2.0f * inv_rms * 2.0f, 1e-5f);
    EXPECT_NEAR(out[2], 3.0f * inv_rms * 2.0f, 1e-5f);
}

TEST_F(RMSNormTest, BatchedRows) {
    Tensor input = make_tensor({1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f},
                               Shape{3, 2});
    Tensor weight = make_tensor({1.0f, 1.0f}, Shape{2});

    Tensor output = ops::RMSNorm::forward(input, weight, 1e-6f);
    ASSERT_NE(output.data(), nullptr);

    const float* out = static_cast<const float*>(output.data());
    for (int r = 0; r < 3; ++r) {
        float x = static_cast<float>(r + 1);
        float rms = std::sqrt((x * x + x * x) / 2.0f + 1e-6f);
        EXPECT_NEAR(out[r * 2 + 0], x / rms, 1e-5f);
        EXPECT_NEAR(out[r * 2 + 1], x / rms, 1e-5f);
    }
}

} // namespace
} // namespace hybridai
