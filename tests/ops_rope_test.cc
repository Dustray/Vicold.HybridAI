#include "backends/backend_registry.h"
#include "core/device.h"
#include "core/dtype.h"
#include "core/shape.h"
#include "core/tensor.h"
#include "ops/rope.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>

namespace hybridai {
namespace {

class RoPETest : public ::testing::Test {
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

TEST_F(RoPETest, SingleHeadIdentityAtZero) {
    // [seq_len=1, num_heads=1, head_dim=4]. At pos=0 rotation is identity.
    Tensor input = make_tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{1, 1, 4});
    Tensor output = ops::RoPE::forward(input, /*head_dim=*/4);
    ASSERT_NE(output.data(), nullptr);

    const float* out = static_cast<const float*>(output.data());
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
    EXPECT_FLOAT_EQ(out[3], 4.0f);
}

TEST_F(RoPETest, SinglePairRotation) {
    // [seq_len=2, num_heads=1, head_dim=2]. Pair [x0=1, x1=0] at pos=1
    // rotated by theta=1 (base=1 => inv_freq=1).
    Tensor input = make_tensor({1.0f, 0.0f, 1.0f, 0.0f}, Shape{2, 1, 2});
    Tensor output = ops::RoPE::forward(input, /*head_dim=*/2,
                                         /*base=*/1.0f);
    ASSERT_NE(output.data(), nullptr);

    const float* out = static_cast<const float*>(output.data());
    // Position 0: identity.
    EXPECT_NEAR(out[0], 1.0f, 1e-5f);
    EXPECT_NEAR(out[1], 0.0f, 1e-5f);
    // Position 1: x0*cos - x1*sin = cos(1), x0*sin + x1*cos = sin(1).
    EXPECT_NEAR(out[2], std::cos(1.0f), 1e-5f);
    EXPECT_NEAR(out[3], std::sin(1.0f), 1e-5f);
}

TEST_F(RoPETest, TwoPositionsTwoHeads) {
    // [seq_len=2, num_heads=2, head_dim=4].
    std::vector<float> data = {
        // pos 0, head 0
        1.0f, 0.0f, 0.0f, 0.0f,
        // pos 0, head 1
        1.0f, 0.0f, 0.0f, 0.0f,
        // pos 1, head 0
        1.0f, 0.0f, 0.0f, 0.0f,
        // pos 1, head 1
        1.0f, 0.0f, 0.0f, 0.0f,
    };
    Tensor input = make_tensor(data, Shape{2, 2, 4});
    Tensor output = ops::RoPE::forward(input, /*head_dim=*/4);
    ASSERT_NE(output.data(), nullptr);

    const float* out = static_cast<const float*>(output.data());
    // Position 0: identity for both heads.
    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(out[i], data[i]);
    }
    // Position 1, both heads: [1,0,0,0] rotated; only first pair affected.
    // inv_freq[0] = base^(-0/4) = 1, theta = seq_pos * inv_freq = 1.
    for (int head = 0; head < 2; ++head) {
        int offset = 8 + head * 4;
        EXPECT_NEAR(out[offset + 0], std::cos(1.0f), 1e-5f);
        EXPECT_NEAR(out[offset + 1], std::sin(1.0f), 1e-5f);
        EXPECT_NEAR(out[offset + 2], 0.0f, 1e-5f);
        EXPECT_NEAR(out[offset + 3], 0.0f, 1e-5f);
    }
}

} // namespace
} // namespace hybridai
