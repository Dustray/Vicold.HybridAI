#include "backends/backend_registry.h"
#include "core/device.h"
#include "core/dtype.h"
#include "core/shape.h"
#include "core/tensor.h"
#include "ops/delta_net.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace hybridai {
namespace {

class DeltaNetTest : public ::testing::Test {
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

    Tensor identity_weight(int64_t out_features, int64_t in_features) {
        std::vector<float> data(out_features * in_features, 0.0f);
        int64_t min_dim = std::min(out_features, in_features);
        for (int64_t i = 0; i < min_dim; ++i) {
            data[i * in_features + i] = 1.0f;
        }
        return make_tensor(data, Shape{out_features, in_features});
    }

    std::unique_ptr<Backend> backend_;
};

TEST_F(DeltaNetTest, OutputShapeAndNonZero) {
    const int64_t hidden_size = 6;
    const int64_t num_qk_heads = 2;
    const int64_t num_v_heads = 4;
    const int64_t head_dim = 3;

    Tensor input = make_tensor(std::vector<float>(2 * hidden_size, 0.5f),
                               Shape{2, hidden_size});
    Tensor wq = identity_weight(num_qk_heads * head_dim, hidden_size);
    Tensor wk = identity_weight(num_qk_heads * head_dim, hidden_size);
    Tensor wv = identity_weight(num_v_heads * head_dim, hidden_size);
    Tensor wo = identity_weight(hidden_size, num_v_heads * head_dim);

    Tensor output = ops::GatedDeltaNet::forward(
        input, wq, wk, wv, wo,
        num_qk_heads, num_v_heads, head_dim);
    ASSERT_NE(output.data(), nullptr);
    EXPECT_EQ(output.shape(), input.shape());

    const float* out = static_cast<const float*>(output.data());
    bool has_nonzero = false;
    for (int i = 0; i < output.numel(); ++i) {
        if (std::abs(out[i]) > 1e-5f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST_F(DeltaNetTest, ValidationRejectsMismatchedWo) {
    const int64_t hidden_size = 4;
    const int64_t num_qk_heads = 2;
    const int64_t num_v_heads = 2;
    const int64_t head_dim = 2;

    Tensor input = make_tensor(std::vector<float>(2 * hidden_size, 1.0f),
                               Shape{2, hidden_size});
    Tensor wq = identity_weight(num_qk_heads * head_dim, hidden_size);
    Tensor wk = identity_weight(num_qk_heads * head_dim, hidden_size);
    Tensor wv = identity_weight(num_v_heads * head_dim, hidden_size);
    // wo input dim must equal num_v_heads * head_dim (= 4).
    Tensor wo = identity_weight(hidden_size,
                                num_v_heads * head_dim + 1);

    Status status = ops::GatedDeltaNet::validate(
        input, wq, wk, wv, wo,
        num_qk_heads, num_v_heads, head_dim);
    EXPECT_FALSE(status.ok());
}

TEST(DeltaNetCacheTest, ResetReleasesAllRequestState) {
    InitializeBuiltinBackends();
    auto backend =
        BackendRegistry::instance().create_backend(Device::Cpu());
    ASSERT_NE(backend, nullptr);

    auto conv_buffer = backend->create_buffer(4 * sizeof(float),
                                               MemoryType::Host);
    auto recurrent_buffer = backend->create_buffer(6 * sizeof(float),
                                                    MemoryType::Host);
    ASSERT_NE(conv_buffer, nullptr);
    ASSERT_NE(recurrent_buffer, nullptr);

    ops::DeltaNetCache cache;
    cache.conv_state = Tensor(Shape{2, 2}, DType::FP32, Device::Cpu(),
                              std::move(conv_buffer));
    cache.recurrent_state = Tensor(Shape{1, 2, 3}, DType::FP32,
                                   Device::Cpu(),
                                   std::move(recurrent_buffer));
    cache.length = 17;
    ASSERT_TRUE(cache.initialized());

    cache.reset();

    EXPECT_FALSE(cache.initialized());
    EXPECT_EQ(cache.conv_state.buffer(), nullptr);
    EXPECT_EQ(cache.recurrent_state.buffer(), nullptr);
    EXPECT_EQ(cache.length, 0);
}

TEST(DeltaNetCacheTest, CloneCopiesBothStateBuffers) {
    InitializeBuiltinBackends();
    auto backend =
        BackendRegistry::instance().create_backend(Device::Cpu());
    ASSERT_NE(backend, nullptr);
    auto conv_buffer = backend->create_buffer(4 * sizeof(float), MemoryType::Host);
    auto recurrent_buffer = backend->create_buffer(6 * sizeof(float), MemoryType::Host);
    ASSERT_NE(conv_buffer, nullptr);
    ASSERT_NE(recurrent_buffer, nullptr);
    const float conv_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float recurrent_data[] = {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    std::memcpy(conv_buffer->data(), conv_data, sizeof(conv_data));
    std::memcpy(recurrent_buffer->data(), recurrent_data,
                sizeof(recurrent_data));
    ops::DeltaNetCache source;
    source.conv_state = Tensor(Shape{2, 2}, DType::FP32, Device::Cpu(),
                               std::move(conv_buffer));
    source.recurrent_state = Tensor(Shape{1, 2, 3}, DType::FP32,
                                     Device::Cpu(), std::move(recurrent_buffer));
    source.length = 5;
    ops::DeltaNetCache clone;
    ASSERT_TRUE(source.clone(&clone).ok());
    EXPECT_EQ(clone.length, source.length);
    EXPECT_NE(clone.conv_state.buffer().get(), source.conv_state.buffer().get());
    EXPECT_NE(clone.recurrent_state.buffer().get(),
              source.recurrent_state.buffer().get());
    EXPECT_EQ(std::memcmp(clone.conv_state.data(), conv_data, sizeof(conv_data)),
              0);
    EXPECT_EQ(std::memcmp(clone.recurrent_state.data(), recurrent_data,
                          sizeof(recurrent_data)), 0);

    static_cast<float*>(clone.conv_state.data())[0] = 99.0f;
    static_cast<float*>(clone.recurrent_state.data())[0] = 88.0f;
    EXPECT_FLOAT_EQ(static_cast<const float*>(source.conv_state.data())[0], 1.0f);
    EXPECT_FLOAT_EQ(static_cast<const float*>(source.recurrent_state.data())[0],
                    5.0f);
}

TEST(DeltaNetCacheTest, CheckpointAndRestoreRoundTripCompleteState) {
    InitializeBuiltinBackends();
    auto backend =
        BackendRegistry::instance().create_backend(Device::Cpu());
    ASSERT_NE(backend, nullptr);

    auto conv_buffer = backend->create_buffer(2 * sizeof(float), MemoryType::Host);
    auto recurrent_buffer = backend->create_buffer(2 * sizeof(float),
                                                    MemoryType::Host);
    ASSERT_NE(conv_buffer, nullptr);
    ASSERT_NE(recurrent_buffer, nullptr);
    const float conv_data[] = {1.0f, 2.0f};
    const float recurrent_data[] = {3.0f, 4.0f};
    std::memcpy(conv_buffer->data(), conv_data, sizeof(conv_data));
    std::memcpy(recurrent_buffer->data(), recurrent_data,
                sizeof(recurrent_data));

    ops::DeltaNetCache original;
    original.conv_state = Tensor(Shape{1, 2}, DType::FP32, Device::Cpu(),
                                  std::move(conv_buffer));
    original.recurrent_state = Tensor(Shape{1, 2}, DType::FP32,
                                       Device::Cpu(),
                                       std::move(recurrent_buffer));
    original.length = 4;

    ops::DeltaNetCache checkpoint;
    ASSERT_TRUE(original.checkpoint(&checkpoint).ok());
    EXPECT_EQ(checkpoint.length, 4);
    EXPECT_NE(checkpoint.conv_state.buffer().get(),
              original.conv_state.buffer().get());
    EXPECT_NE(checkpoint.recurrent_state.buffer().get(),
              original.recurrent_state.buffer().get());

    original.length = 9;
    static_cast<float*>(original.conv_state.data())[0] = 9.0f;
    static_cast<float*>(original.recurrent_state.data())[0] = 8.0f;
    ASSERT_TRUE(original.restore(&checkpoint).ok());

    EXPECT_EQ(original.length, 4);
    EXPECT_FLOAT_EQ(static_cast<const float*>(original.conv_state.data())[0],
                    1.0f);
    EXPECT_FLOAT_EQ(
        static_cast<const float*>(original.recurrent_state.data())[0], 3.0f);
}

TEST(DeltaNetCacheTest, RestoreRejectsNullOrIncompatibleCheckpoint) {
    ops::DeltaNetCache original;
    EXPECT_FALSE(original.restore(nullptr).ok());

    InitializeBuiltinBackends();
    auto backend =
        BackendRegistry::instance().create_backend(Device::Cpu());
    ASSERT_NE(backend, nullptr);
    auto original_conv = backend->create_buffer(2 * sizeof(float),
                                                 MemoryType::Host);
    auto original_recurrent = backend->create_buffer(2 * sizeof(float),
                                                      MemoryType::Host);
    auto checkpoint_conv = backend->create_buffer(4 * sizeof(float),
                                                  MemoryType::Host);
    auto checkpoint_recurrent = backend->create_buffer(4 * sizeof(float),
                                                        MemoryType::Host);
    ASSERT_NE(original_conv, nullptr);
    ASSERT_NE(original_recurrent, nullptr);
    ASSERT_NE(checkpoint_conv, nullptr);
    ASSERT_NE(checkpoint_recurrent, nullptr);
    original.conv_state = Tensor(Shape{1, 2}, DType::FP32, Device::Cpu(),
                                 std::move(original_conv));
    original.recurrent_state = Tensor(Shape{1, 2}, DType::FP32,
                                      Device::Cpu(),
                                      std::move(original_recurrent));
    ops::DeltaNetCache incompatible;
    incompatible.conv_state = Tensor(Shape{2, 2}, DType::FP32,
                                     Device::Cpu(), std::move(checkpoint_conv));
    incompatible.recurrent_state = Tensor(
        Shape{2, 2}, DType::FP32, Device::Cpu(), std::move(checkpoint_recurrent));
    incompatible.length = 3;
    EXPECT_FALSE(original.restore(&incompatible).ok());
    EXPECT_EQ(original.length, 0);
}

TEST(DeltaNetCacheTest, RestoreCanInitializeAnEmptyCache) {
    InitializeBuiltinBackends();
    auto backend =
        BackendRegistry::instance().create_backend(Device::Cpu());
    ASSERT_NE(backend, nullptr);
    auto conv_buffer = backend->create_buffer(2 * sizeof(float), MemoryType::Host);
    auto recurrent_buffer = backend->create_buffer(2 * sizeof(float),
                                                    MemoryType::Host);
    ASSERT_NE(conv_buffer, nullptr);
    ASSERT_NE(recurrent_buffer, nullptr);
    ops::DeltaNetCache checkpoint;
    checkpoint.conv_state = Tensor(Shape{1, 2}, DType::FP32, Device::Cpu(),
                                   std::move(conv_buffer));
    checkpoint.recurrent_state = Tensor(Shape{1, 2}, DType::FP32,
                                        Device::Cpu(),
                                        std::move(recurrent_buffer));
    checkpoint.length = 2;

    ops::DeltaNetCache empty;
    ASSERT_TRUE(empty.restore(&checkpoint).ok());
    EXPECT_TRUE(empty.initialized());
    EXPECT_EQ(empty.length, 2);
}

TEST(DeltaNetCacheTest, SwapCommitsConvAndRecurrentStateTogether) {
    ops::DeltaNetCache committed;
    ops::DeltaNetCache scratch;
    committed.length = 3;
    scratch.length = 7;

    committed.swap(&scratch);

    EXPECT_EQ(committed.length, 7);
    EXPECT_EQ(scratch.length, 3);
}

} // namespace
} // namespace hybridai
